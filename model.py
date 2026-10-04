"""
nanoPOPIXA v2 — Architecture moderne (Claude-inspired, reverse-engineered)
RMSNorm · SwiGLU · RoPE · KV-Cache · Nucleus sampling (top-p)

Améliorations vs v1 (nanoGPT-style) :
  - RMSNorm   : normalisation sans biais, plus rapide que LayerNorm
  - SwiGLU    : activation gate (Claude, LLaMA, PaLM) → meilleur gradient flow
  - RoPE      : encodage rotatif des positions → meilleure généralisation sur la longueur
  - KV-Cache  : cache clés/valeurs en inférence → décodage O(1) par token
  - top-p     : nucleus sampling → diversité mieux contrôlée
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class POPIXAConfig:
    block_size: int = 1024       # longueur maximale de séquence
    vocab_size: int = 65         # taille du vocabulaire
    n_layer:    int = 6          # nombre de blocs Transformer
    n_head:     int = 6          # nombre de têtes d'attention
    n_embd:     int = 384        # dimension des embeddings
    dropout:    float = 0.1      # taux de dropout
    bias:       bool = False     # conservé pour compatibilité checkpoints v1 (ignoré)
    rope_base:  int = 10_000     # base fréquentielle RoPE (10k standard, 500k LongRoPE)


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm — Root Mean Square Normalization
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Normalisation par RMS — utilisée dans Claude, LLaMA, Mistral.
    Plus simple que LayerNorm : pas de biais, pas de centrage, seulement la mise à l'échelle.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calcul en float32 : en fp16, x² déborde dès |x| > 256 → vecteur entier mis à 0
        xf  = x.float()
        rms = xf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (xf * rms).to(x.dtype) * self.weight


# ─────────────────────────────────────────────────────────────────────────────
# RoPE — Rotary Position Embeddings
# ─────────────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    Encodage rotatif des positions (Su et al., 2021).
    Avantages vs embeddings appris :
      - 0 paramètre supplémentaire
      - généralise mieux hors de la fenêtre d'entraînement
      - encode les distances relatives directement dans le produit Q·K
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10_000):
        super().__init__()
        # Fréquences inverses : θ_i = 1 / base^(2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)           # (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)          # (seq_len, dim)
        # Shape (1, 1, seq_len, dim) pour broadcaster sur (B, n_head, T, head_dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, seq_len: int, offset: int = 0):
        """Retourne (cos, sin) pour les positions [offset, offset+seq_len)."""
        return (
            self.cos_cached[:, :, offset:offset + seq_len, :],
            self.sin_cached[:, :, offset:offset + seq_len, :],
        )


def _rotate_half(x):
    """Rotation de 90° dans l'espace complexe : (x1, x2) → (-x2, x1)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    """Applique RoPE sur Q et K (cos/sin castés au dtype de Q → compatible bf16/fp16)."""
    cos, sin = cos.to(q.dtype), sin.to(q.dtype)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


# ─────────────────────────────────────────────────────────────────────────────
# Attention causale avec RoPE + KV-Cache
# ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention causale avec :
      - RoPE pour l'encodage des positions
      - KV-Cache pour l'inférence incrémentale (O(1) par token au lieu de O(T²))
      - Flash Attention quand disponible (PyTorch ≥ 2.0)
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert (config.n_embd // config.n_head) % 2 == 0, "RoPE requiert un head_dim pair"

        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Projections Q, K, V fusionnées + projection de sortie (sans biais)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd,     bias=False)

        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            # Fallback : masque causal pré-alloué
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                      .view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x, rotary_emb: RotaryEmbedding, past_kv=None):
        """
        Args:
            x          : (B, T, C) — tokens actuels
            rotary_emb : module RoPE partagé
            past_kv    : (K_cache, V_cache) ou None — KV cache des tokens précédents
        Returns:
            output     : (B, T, C)
            present_kv : (K, V) mis à jour pour ce bloc
        """
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # RoPE — offset = longueur du cache existant
        offset = past_kv[0].size(2) if past_kv is not None else 0
        cos, sin = rotary_emb(T, offset=offset)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Concaténation avec le cache (inférence incrémentale)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present_kv = (k, v)

        kv_len   = k.size(2)
        past_len = kv_len - T
        # Trois cas :
        #   - prefill sans cache (past_len == 0)  → masque causal standard
        #   - décodage d'un token (T == 1)        → le token voit tout le cache, pas de masque
        #   - prefill par morceaux sur un cache   → masque causal DÉCALÉ : la requête i
        #     (position absolue past_len+i) voit les clés 0..past_len+i
        #     (session restaurée, vérification du speculative decoding)

        if self.flash:
            attn_mask = None
            if T > 1 and past_len > 0:
                attn_mask = torch.ones(T, kv_len, dtype=torch.bool, device=x.device).tril(diagonal=past_len)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=(T > 1 and past_len == 0),
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if T > 1:
                att = att.masked_fill(
                    self.bias[:, :, past_len:kv_len, :kv_len] == 0, float("-inf")
                )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y   = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y)), present_kv


# ─────────────────────────────────────────────────────────────────────────────
# SwiGLU — Swish-Gated Linear Unit
# ─────────────────────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """
    MLP avec gate SwiGLU — utilisé dans Claude, LLaMA, PaLM.
    output = down( silu(gate(x)) ⊙ up(x) )

    Dimension cachée = 8/3 × n_embd (arrondie à 64)
    → même nombre de paramètres qu'un MLP GELU 4× mais meilleures performances.
    """

    def __init__(self, config):
        super().__init__()
        hidden = int(8 / 3 * config.n_embd)
        hidden = ((hidden + 63) // 64) * 64  # arrondi efficace (multiple de 64)

        self.gate    = nn.Linear(config.n_embd, hidden, bias=False)
        self.up      = nn.Linear(config.n_embd, hidden, bias=False)
        self.down    = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Silu (= Swish) : x * σ(x) — différentiable et sans saturation
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


# ─────────────────────────────────────────────────────────────────────────────
# Bloc Transformer
# ─────────────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    """Bloc Transformer : RMSNorm → Attention → RMSNorm → SwiGLU (+ résiduels)."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp  = SwiGLU(config)

    def forward(self, x, rotary_emb: RotaryEmbedding, past_kv=None):
        attn_out, present_kv = self.attn(self.ln_1(x), rotary_emb, past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv


# ─────────────────────────────────────────────────────────────────────────────
# KV-Cache annoté + décodeur incrémental
# ─────────────────────────────────────────────────────────────────────────────

def checkpoint_v1_error(state_dict) -> str:
    """
    Détecte un checkpoint v1 (nanoGPT-style : wpe + LayerNorm + GELU), incompatible v2.
    Retourne un message d'erreur lisible, ou None si le checkpoint est bien v2.
    """
    keys = set(state_dict.keys())
    v1_markers = ("transformer.wpe.weight", "transformer.h.0.mlp.c_fc.weight", "transformer.ln_f.bias")
    if any(k in keys for k in v1_markers) or "transformer.h.0.mlp.gate.weight" not in keys:
        return ("Checkpoint v1 (LayerNorm + wpe + GELU) incompatible avec l'architecture v2 "
                "(RMSNorm · SwiGLU · RoPE) — réentraîne le modèle : popixa train --data_dir data/")
    return None


def _cache_len(past_kvs) -> int:
    """Nombre de positions déjà présentes dans un KV-cache (0 si absent)."""
    return past_kvs[0][0].size(2) if past_kvs else 0


class KVCache(list):
    """
    KV-cache d'inférence : liste de (K, V) par couche, de forme (B, n_head, T, head_dim).

    Sous-classe de `list` → reste compatible avec tout code qui attend une simple
    liste de tuples (forward(past_kvs=...), save_session, CI).

    token_ids : ids exacts des tokens couverts par le cache (ligne 0 du batch), dans
                l'ordre — ou None s'ils sont inconnus (cache fourni sans ids).
                Invariant : len(token_ids) == seq_len.
    """

    def __init__(self, kvs=(), token_ids=None):
        super().__init__(kvs)
        self.token_ids = token_ids

    @property
    def seq_len(self) -> int:
        return _cache_len(self)


class _Decoder:
    """
    Décodage incrémental avec KV-cache — logique commune à tous les générateurs.

    Invariants :
      - idx      : contexte connu (B, T) — ids du cache initial (si connus) + entrée
                   + tokens générés
      - past_kvs : KV-cache couvrant idx, sauf les `pending` derniers tokens
                   (générés mais pas encore passés dans le modèle)
      - ctx_ids  : ids couverts par le cache (None si le préfixe est inconnu)

    Fenêtre glissante : quand le cache atteint block_size, on refait un prefill sur les
    `keep` derniers tokens connus (RoPE repart de 0) au lieu de planter.
    """

    def __init__(self, model, idx, initial_past_kvs=None, slide_keep: float = 0.75):
        self.model      = model
        self.block_size = model.config.block_size
        self.keep       = max(1, int(self.block_size * slide_keep))
        self.pending    = 0
        self.past_kvs   = None
        self.ctx_ids    = None

        if idx.size(1) == 0:
            known = getattr(initial_past_kvs, "token_ids", None)
            if known:
                # Entrée vide sur un cache connu : on rejoue le dernier token du cache
                # pour obtenir les logits du token suivant
                n = len(known) - 1
                initial_past_kvs = KVCache(
                    [(k[:, :, :n], v[:, :, :n]) for k, v in initial_past_kvs], list(known[:n])
                )
                idx = torch.tensor([[known[-1]]], dtype=torch.long, device=idx.device)
            else:
                # Génération libre : token 0 comme amorce (comme train.py)
                initial_past_kvs = None
                idx = torch.zeros((idx.size(0), 1), dtype=torch.long, device=idx.device)

        self.logits = self._prefill(idx, initial_past_kvs)

    @property
    def cache_len(self) -> int:
        return _cache_len(self.past_kvs)

    def _prefill(self, idx, past_kvs):
        # Ids du cache initial connus (KVCache) → ils font partie du contexte connu :
        # la fenêtre glissante, la repetition penalty et les drafts n-grammes les voient.
        prefix = getattr(past_kvs, "token_ids", None) if past_kvs is not None else None
        if prefix is not None and (idx.size(0) != 1 or len(prefix) != _cache_len(past_kvs)):
            prefix = None
        if prefix:
            pre = torch.tensor([list(prefix)], dtype=torch.long, device=idx.device)
            self.idx = torch.cat((pre, idx), dim=1)
        else:
            self.idx = idx

        if past_kvs is not None and _cache_len(past_kvs) + idx.size(1) <= self.block_size:
            logits, self.past_kvs = self.model(idx, past_kvs=past_kvs)
            self.ctx_ids = self.idx[0].tolist() if prefix is not None else None
            return logits

        # Pas de cache, ou cache + entrée > block_size → prefill sur le contexte connu,
        # tronqué aux block_size derniers tokens (ou `keep` en cas de débordement, pour
        # laisser de la place à la génération)
        full = self.idx
        ctx  = full[:, -self.block_size:] if full.size(1) <= self.block_size else full[:, -self.keep:]
        logits, self.past_kvs = self.model(ctx)
        self.ctx_ids = ctx[0].tolist()
        return logits

    def push(self, tok) -> None:
        """Ajoute un token (int ou tenseur (B, 1)) au contexte, en attente d'être passé au modèle."""
        if not torch.is_tensor(tok):
            tok = torch.full((self.idx.size(0), 1), int(tok), dtype=torch.long, device=self.idx.device)
        self.idx = torch.cat((self.idx, tok), dim=1)
        self.pending += 1

    def advance(self):
        """Passe les tokens en attente dans le modèle → logits du prochain token."""
        if self.pending == 0:
            return self.logits
        if self.cache_len + self.pending > self.block_size:
            # Fenêtre pleine → prefill glissant sur les derniers tokens connus
            ctx = self.idx[:, -self.keep:]
            self.logits, self.past_kvs = self.model(ctx)
            self.ctx_ids = ctx[0].tolist()
        else:
            new = self.idx[:, -self.pending:]
            self.logits, self.past_kvs = self.model(new, past_kvs=self.past_kvs)
            if self.ctx_ids is not None:
                self.ctx_ids.extend(new[0].tolist())
        self.pending = 0
        return self.logits

    def commit(self, base: int, kvs, drafts: list, n_accepted: int, next_tok: int) -> None:
        """
        Speculative decoding : garde dans le cache vérifié `kvs` les `n_accepted` premiers
        drafts (positions base..base+n_accepted), puis ajoute `next_tok` (en attente).
        """
        n = base + n_accepted
        self.past_kvs = [(k[:, :, :n], v[:, :, :n]) for k, v in kvs]
        if n_accepted:
            acc = torch.tensor([drafts[:n_accepted]], dtype=torch.long, device=self.idx.device)
            self.idx = torch.cat((self.idx, acc), dim=1)
            if self.ctx_ids is not None:
                self.ctx_ids.extend(drafts[:n_accepted])
        self.push(next_tok)

    def kv_cache(self) -> KVCache:
        """KV-cache final couvrant TOUT le contexte (les tokens en attente sont d'abord passés)."""
        self.advance()
        return KVCache(self.past_kvs, list(self.ctx_ids) if self.ctx_ids is not None else None)


def _store_cache(dec, cache_ref) -> None:
    """Remplit cache_ref avec le KV-cache final — y compris si le générateur est interrompu."""
    if cache_ref is None or dec is None:
        return
    cache_ref.clear()
    try:
        cache_ref.append(dec.kv_cache())
    except Exception:
        pass  # cache inutilisable → cache_ref reste vide, l'appelant reconstruit le contexte


# ─────────────────────────────────────────────────────────────────────────────
# nanoPOPIXA v2
# ─────────────────────────────────────────────────────────────────────────────

class nanoPOPIXA(nn.Module):
    """
    nanoPOPIXA v2 — Architecture :
        Token Embedding → N × [RMSNorm → Attention(RoPE) → RMSNorm → SwiGLU] → RMSNorm → LM Head

    Vs v1 (nanoGPT-style) :
      ✗ wpe (positional embeddings appris)  → ✓ RoPE (0 paramètre, meilleure généralisation)
      ✗ LayerNorm avec biais               → ✓ RMSNorm (plus rapide)
      ✗ GELU MLP                           → ✓ SwiGLU (meilleur gradient flow)
      ✗ Recompute complet à chaque token   → ✓ KV-Cache (décodage O(1))
      ✗ top-k uniquement                   → ✓ top-p nucleus sampling

    Note : incompatible avec les checkpoints v1 (architecture différente — réentraîner).
    """

    def __init__(self, config: POPIXAConfig):
        super().__init__()
        self.config = config

        head_dim   = config.n_embd // config.n_head
        rope_base  = getattr(config, "rope_base", 10_000)  # compat checkpoints v1

        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len=config.block_size, base=rope_base)

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying : embedding et lm_head partagent les mêmes poids
        self.transformer.wte.weight = self.lm_head.weight

        # Initialisation des poids
        self.apply(self._init_weights)
        # Mise à l'échelle des projections résiduelles (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("down.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        flash = "Flash Attention ✓" if hasattr(F, "scaled_dot_product_attention") else "Attention manuelle"
        print(f"nanoPOPIXA v2 — {self.get_num_params()/1e6:.2f}M params hors embeddings "
              f"({self.get_num_params(False)/1e6:.2f}M au total) | RMSNorm · SwiGLU · RoPE | {flash}")

    # ── Utilitaires ──────────────────────────────────────────────────────────

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Nombre de paramètres. non_embedding=True exclut la matrice wte (partagée avec
        lm_head) — convention des lois d'échelle ; non_embedding=False → total réel.
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wte.weight.numel()
        return n

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, idx, targets=None, past_kvs=None, return_all_logits=False):
        """
        Mode entraînement (targets != None) :
            Retourne (logits, loss) — compatible train.py.

        Mode inférence (targets == None) :
            Retourne (logits, present_kvs) — utilisé par generate/generate_stream.
            logits : (B, 1, vocab_size) — seulement le dernier token.

        return_all_logits=True :
            Retourne (logits, present_kvs) avec logits (B, T, vocab_size).
            Utilisé par speculative decoding pour vérifier tous les tokens draft en une passe.
        """
        B, T = idx.size()
        past_len = _cache_len(past_kvs)
        assert T > 0, "Séquence vide"
        assert past_len + T <= self.config.block_size, (
            f"Séquence trop longue ({past_len} en cache + {T} > block_size {self.config.block_size})"
        )

        x = self.transformer.drop(self.transformer.wte(idx))

        present_kvs = []
        for i, block in enumerate(self.transformer.h):
            past_kv = past_kvs[i] if past_kvs else None
            x, present_kv = block(x, self.rotary_emb, past_kv=past_kv)
            present_kvs.append(present_kv)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # Entraînement — loss sur toute la séquence
            logits = self.lm_head(x)
            loss   = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            return logits, loss

        if return_all_logits:
            # Speculative decoding — tous les logits (B, T, vocab_size)
            return self.lm_head(x), present_kvs

        # Inférence standard — seulement le dernier token, on retourne le cache
        logits = self.lm_head(x[:, [-1], :])
        return logits, present_kvs

    # ── Sampling ─────────────────────────────────────────────────────────────

    def _apply_sampling(self, logits, temperature, top_k, top_p, repetition_penalty, idx,
                        logit_bias=None, mask=None):
        """
        Transforme les logits du dernier token en distribution de probabilités, dans l'ordre :
          1. Repetition penalty — divise si logit > 0, multiplie si logit ≤ 0
          2. Logit bias         — forçage / interdiction de tokens {id: biais}
          3. Masque             — tokens autorisés par une grammaire (structured outputs)
          4. Temperature        — temperature ≤ 0 → greedy (distribution one-hot sur l'argmax)
          5. Top-k
          6. Top-p (nucleus sampling)
        Ne modifie jamais le tenseur `logits` de l'appelant. Retourne (B, vocab_size).
        """
        if repetition_penalty is None or repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty doit être > 0 (reçu {repetition_penalty})")
        logits = logits[:, -1, :].float().clone()  # (B, vocab_size)

        # Repetition penalty — pénalise les tokens déjà présents dans le contexte (par ligne)
        if repetition_penalty != 1.0 and idx is not None and idx.numel() > 0:
            score = torch.gather(logits, 1, idx)
            score = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            logits.scatter_(1, idx, score)

        # Logit bias — appliqué AVANT top-k pour pouvoir forcer un token hors du top-k
        if logit_bias:
            ids = [t for t in logit_bias if 0 <= t < logits.size(-1)]
            if ids:
                bias = torch.tensor([float(logit_bias[t]) for t in ids], device=logits.device)
                logits[:, ids] += bias

        # Masque grammatical — les tokens interdits ne peuvent jamais être tirés
        if mask is not None:
            logits = logits.masked_fill(~mask.to(logits.device), float("-inf"))

        # Greedy — temperature ≤ 0 (évite la division par zéro)
        if temperature is None or temperature <= 0:
            probs = torch.zeros_like(logits)
            return probs.scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)

        logits = logits / temperature

        # Top-k — ne garde que les k meilleurs logits
        if top_k is not None and 0 < top_k < logits.size(-1):
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")

        # Top-p — nucleus sampling : garde le noyau minimal qui couvre p% de la proba
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Retire les tokens dont la proba cumulée dépasse top_p
            # (décalé d'un cran pour toujours garder au moins un token)
            to_remove = torch.zeros_like(sorted_logits, dtype=torch.bool)
            to_remove[:, 1:] = cum_probs[:, :-1] > top_p
            sorted_logits[to_remove] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)

        return F.softmax(logits, dim=-1)

    # ── Diminishing returns ──────────────────────────────────────────────────

    @staticmethod
    def _is_repetitive(tokens: list, window: int = 40, threshold: float = 0.28) -> bool:
        """
        Détecte si la génération tourne en rond (inspiré de tokenBudget.ts).
        Calcule la diversité des bigrammes sur les derniers `window` tokens.
        Retourne True si diversity < threshold (génération répétitive).

        Claude utilise DIMINISHING_THRESHOLD=500 tokens delta sur 3 itérations.
        Ici on adapte en diversité de bigrammes — plus granulaire pour les petits modèles.
        """
        if len(tokens) < window:
            return False
        w = tokens[-window:]
        bigrams   = [(w[i], w[i + 1]) for i in range(len(w) - 1)]
        diversity = len(set(bigrams)) / len(bigrams)
        return diversity < threshold

    @staticmethod
    def _has_diminishing_returns(
        tokens: list,
        window: int = 40,
        n_checks: int = 3,
        threshold: float = 0.28,
    ) -> bool:
        """
        Vérifie si N fenêtres consécutives montrent toutes une diversité faible.
        Fidèle à tokenBudget.ts : DIMINISHING_THRESHOLD sur 3 itérations consécutives.

        Différence avec _is_repetitive :
          _is_repetitive  → détecte UNE fenêtre répétitive (arrêt immédiat)
          _has_diminishing_returns → requiert N fenêtres consécutives (plus conservateur)

        Paramètres :
          n_checks  : nombre de fenêtres consécutives à vérifier (défaut 3, comme Claude)
          window    : taille de chaque fenêtre en tokens
          threshold : seuil de diversité bigrammes (0.28 = 28%)
        """
        if len(tokens) < window * n_checks:
            return False
        for i in range(n_checks):
            start = len(tokens) - window * (i + 1)
            end   = len(tokens) - window * i if i > 0 else len(tokens)
            w = tokens[start:end]
            if len(w) < 2:
                return False
            bigrams = [(w[j], w[j + 1]) for j in range(len(w) - 1)]
            if len(set(bigrams)) / len(bigrams) >= threshold:
                return False  # une fenêtre diverse → pas encore diminishing
        return True  # toutes les n_checks fenêtres sont répétitives

    # Politiques d'arrêt anticipé :
    #   "repetitive"  → _is_repetitive          (1 fenêtre — arrêt immédiat)
    #   "diminishing" → _has_diminishing_returns (3 fenêtres consécutives — conservateur)
    #   None / "off"  → jamais
    STOP_POLICIES = ("repetitive", "diminishing", "off")

    @staticmethod
    def _resolve_stop_policy(stop_policy=None, stop_on_repetition: bool = False):
        """Normalise la politique d'arrêt (compat : stop_on_repetition=True → 'repetitive')."""
        if stop_policy is None:
            return "repetitive" if stop_on_repetition else None
        if stop_policy == "off":
            return None
        if stop_policy not in ("repetitive", "diminishing"):
            raise ValueError(f"stop_policy inconnue : {stop_policy!r} (repetitive|diminishing|off)")
        return stop_policy

    @classmethod
    def _should_stop(cls, tokens: list, stop_policy) -> bool:
        if stop_policy == "repetitive":
            return cls._is_repetitive(tokens)
        if stop_policy == "diminishing":
            return cls._has_diminishing_returns(tokens)
        return False

    # ── Boucle de décodage commune ───────────────────────────────────────────

    def _sample_stream(self, dec, n_tokens, temperature, top_k, top_p, repetition_penalty,
                       stop_policy=None, logit_bias=None, history=None):
        """
        Échantillonne jusqu'à n_tokens tokens via le décodeur `dec`.
        Yield le tenseur (B, 1) de chaque token. S'arrête si la politique d'arrêt se
        déclenche (le token déclencheur est écarté : ni yieldé, ni ajouté au contexte).
        history : liste partagée des tokens déjà émis (pour la politique d'arrêt).
        """
        generated = [] if history is None else history
        for _ in range(max(0, n_tokens)):
            logits   = dec.advance()
            probs    = self._apply_sampling(logits, temperature, top_k, top_p, repetition_penalty,
                                            dec.idx, logit_bias=logit_bias)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated.append(idx_next[0, 0].item())
            if self._should_stop(generated, stop_policy):
                return  # Diminishing returns détecté → fin de la phase
            dec.push(idx_next)
            yield idx_next

    # ── Génération standard ──────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 repetition_penalty=1.0, top_p=None, stop_on_repetition=False,
                 stop_policy=None, logit_bias=None):
        """
        Génération avec KV-Cache :
          1. Prefill  — traite tout le prompt en une passe, construit le cache
          2. Decode   — génère un token à la fois en O(1) grâce au cache
        Fenêtre glissante automatique au-delà de block_size.

        stop_on_repetition : arrêt anticipé si la génération diverge (= stop_policy="repetitive")
        stop_policy        : "repetitive" | "diminishing" | "off"
        logit_bias         : {token_id: biais} ajouté aux logits avant top-k
        Retourne idx (B, T + n_générés).
        """
        policy = self._resolve_stop_policy(stop_policy, stop_on_repetition)
        dec = _Decoder(self, idx)
        for _ in self._sample_stream(dec, max_new_tokens, temperature, top_k, top_p,
                                     repetition_penalty, policy, logit_bias):
            pass
        return dec.idx

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                        repetition_penalty=1.0, top_p=None, stop_on_repetition=False,
                        initial_past_kvs=None, cache_ref=None,
                        stop_policy=None, logit_bias=None):
        """
        Streaming token par token avec KV-Cache.
        Yield chaque token généré (int).

        initial_past_kvs : KV-cache existant (session persistante) — si fourni,
                           seul `idx` (nouveaux tokens) est traité en prefill,
                           le reste est récupéré depuis le cache. O(N_new) au lieu de O(N_total).
                           Si cache + idx dépasse block_size, le cache est abandonné.
        cache_ref        : liste mutable — contiendra [KVCache] couvrant tout le contexte
                           (entrée + tokens yieldés) à la fin du générateur, y compris en
                           cas d'interruption (close / exception). KVCache.token_ids donne
                           les ids exacts couverts par le cache.
        stop_on_repetition / stop_policy : arrêt automatique si la génération diverge.
        """
        policy = self._resolve_stop_policy(stop_policy, stop_on_repetition)
        dec = None
        try:
            dec = _Decoder(self, idx, initial_past_kvs)
            for t in self._sample_stream(dec, max_new_tokens, temperature, top_k, top_p,
                                         repetition_penalty, policy, logit_bias):
                yield t[0, 0].item()
        finally:
            # Stocker le cache final pour persistance inter-sessions
            _store_cache(dec, cache_ref)

    # ── Thinking blocks (Claude-inspired) ───────────────────────────────────

    @staticmethod
    def adaptive_think_budget(prompt_len: int) -> int:
        """
        Budget de thinking adaptatif selon la complexité estimée du prompt.
        Inspiré de l'adaptive thinking de Claude 4.6+ (pas de budget fixe — le modèle
        décide lui-même). Ici on estime via la longueur du prompt.
        """
        if prompt_len < 50:
            return 50
        elif prompt_len < 150:
            return 150
        elif prompt_len < 400:
            return 300
        else:
            return 500

    @torch.no_grad()
    def generate_stream_with_thinking(
        self, idx,
        think_budget: int = 150,
        response_budget: int = 300,
        temperature: float = 0.8,
        top_k: int = None,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        initial_past_kvs=None,
        cache_ref=None,
        think_stop_policy: str = "diminishing",
        stop_policy: str = "repetitive",
        logit_bias=None,
    ):
        """
        Génération deux phases inspirée de Claude's extended thinking.

        Phase 1 — Think (temperature=1 forcée, comme Claude) :
            Le modèle génère un raisonnement interne libre.
            Arrêt anticipé via think_stop_policy (défaut "diminishing" : 3 fenêtres
            répétitives consécutives — le thinking a le droit d'explorer plus longtemps).

        Phase 2 — Response (temperature normale) :
            Le modèle génère la réponse finale en ayant "vu" son propre raisonnement.
            Arrêt anticipé via stop_policy (défaut "repetitive").

        Yield : tuples (phase, token_int)
            phase = 'think' | 'response'
        """
        think_policy = self._resolve_stop_policy(think_stop_policy)
        resp_policy  = self._resolve_stop_policy(stop_policy)
        dec = None
        try:
            # Prefill — avec cache existant si session restaurée
            dec = _Decoder(self, idx, initial_past_kvs)

            # Phase 1 : Thinking — temperature=1 (contrainte API Claude, claude.ts:1598)
            for t in self._sample_stream(dec, think_budget, 1.0, top_k, top_p, repetition_penalty,
                                         think_policy, logit_bias):
                yield ("think", t[0, 0].item())

            # Phase 2 : Response — temperature normale
            for t in self._sample_stream(dec, response_budget, temperature, top_k, top_p,
                                         repetition_penalty, resp_policy, logit_bias):
                yield ("response", t[0, 0].item())
        finally:
            _store_cache(dec, cache_ref)

    # ── Interleaved Thinking (Claude-inspired) ───────────────────────────────

    @torch.no_grad()
    def generate_stream_with_interleaved_thinking(
        self, idx,
        response_budget: int = 300,
        think_per_interleave: int = 20,
        interleave_every: int = 50,
        temperature: float = 0.8,
        top_k: int = None,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        initial_past_kvs=None,
        cache_ref=None,
        stop_policy: str = "repetitive",
        logit_bias=None,
    ):
        """
        Thinking intercalé — inspiré du beta `interleaved-thinking-2025-05-14`.

        Alterne génération normale et mini-pauses de réflexion :
          1. Génère `interleave_every` tokens de réponse
          2. Pause : génère `think_per_interleave` tokens de thinking (temp=1)
          3. Reprend la réponse — en boucle jusqu'à response_budget

        Avantage vs thinking pur : la réflexion est distribuée tout au long
        de la réponse, permettant des corrections en cours de route.
        La politique d'arrêt ne regarde que les tokens de réponse.

        Yield : tuples (phase, token_int)  où phase = 'think' | 'response'
        """
        policy = self._resolve_stop_policy(stop_policy)
        every  = max(1, interleave_every)
        dec = None
        try:
            dec = _Decoder(self, idx, initial_past_kvs)
            generated_resp = []
            produced = 0
            while produced < response_budget:
                # ── Segment de réponse ──────────────────────────────────────
                n = min(every, response_budget - produced)
                count = 0
                for t in self._sample_stream(dec, n, temperature, top_k, top_p, repetition_penalty,
                                             policy, logit_bias, history=generated_resp):
                    count += 1
                    yield ("response", t[0, 0].item())
                produced += count
                if count < n or produced >= response_budget:
                    break  # arrêt anticipé ou budget atteint

                # ── Mini-pause thinking (temperature=1) ─────────────────────
                for t in self._sample_stream(dec, think_per_interleave, 1.0, top_k, top_p,
                                             repetition_penalty, None, logit_bias):
                    yield ("think", t[0, 0].item())
        finally:
            _store_cache(dec, cache_ref)

    # ── Speculative Decoding (fast mode) ─────────────────────────────────────

    @staticmethod
    def _draft_ngram(ids: list, n_draft: int, max_ngram: int = 3) -> list:
        """
        Prompt lookup decoding : cherche l'occurrence la plus récente du suffixe courant
        (n-gramme de max_ngram → 1 tokens) dans le contexte et propose les tokens qui la
        suivaient. Draft gratuit (aucune passe du modèle) — efficace sur texte/code répétitif.
        """
        L = len(ids)
        for n in range(min(max_ngram, L - 1), 0, -1):
            suffix = ids[L - n:]
            for start in range(L - n - 1, -1, -1):
                if ids[start:start + n] == suffix:
                    cont = ids[start + n:start + n + n_draft]
                    if cont:
                        return cont
        return []

    def _draft_self(self, dec, k, top_k, top_p, repetition_penalty, logit_bias,
                    draft_temperature: float = 0.05):
        """
        Draft avec le MÊME modèle à température quasi nulle (quasi greedy).
        Retourne (drafts, distributions q) — q sert au test accept/reject.
        Le cache de travail est jeté : seul le cache vérifié est conservé.
        """
        drafts, qs = [], []
        logits, kvs, ctx = dec.logits, dec.past_kvs, dec.idx
        for i in range(k):
            q = self._apply_sampling(logits, draft_temperature, top_k, top_p, repetition_penalty,
                                     ctx, logit_bias=logit_bias)[0]
            d = torch.multinomial(q, num_samples=1).item()
            drafts.append(d)
            qs.append(q)
            if i < k - 1:
                t = torch.tensor([[d]], dtype=torch.long, device=ctx.device)
                logits, kvs = self(t, past_kvs=kvs)
                ctx = torch.cat((ctx, t), dim=1)
        return drafts, qs

    @torch.no_grad()
    def speculative_generate_stream(
        self, idx,
        max_new_tokens: int = 200,
        n_draft: int = 4,
        temperature: float = 0.8,
        top_k: int = None,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        initial_past_kvs=None,
        cache_ref=None,
        draft: str = "ngram",
        stop_policy=None,
        logit_bias=None,
    ):
        """
        Speculative decoding — inspiré du beta `fast-mode-2026-02-01`.

        Principe :
          1. Draft  — propose k tokens :
               · "ngram" (défaut) : prompt lookup — recopie la suite d'un n-gramme déjà vu
                 dans le contexte. Gratuit → vrai gain de vitesse sur texte répétitif.
               · "self" : le MÊME modèle à temp≈0 (pédagogique ; pas plus rapide sans
                 modèle draft plus léger).
          2. Verify — UNE passe du modèle sur les k drafts (masque causal décalé) →
             distributions cibles p_0..p_k pour chaque position.
          3. Accept/Reject (Leviathan et al., 2023) — draft d_i accepté avec proba
             min(1, p_i(d_i) / q_i(d_i)) ; sinon on ré-échantillonne depuis
             max(0, p_i − q_i) normalisé et on s'arrête. Distribution de sortie
             IDENTIQUE à l'échantillonnage normal.
          4. Bonus — si les k drafts sont acceptés, un token de plus gratuit (p_k).
          Le KV-cache est tronqué aux tokens acceptés (les drafts rejetés en sortent).

        Yield : int (token généré)
        """
        if draft not in ("ngram", "self"):
            raise ValueError(f"draft inconnu : {draft!r} (ngram|self)")
        policy = self._resolve_stop_policy(stop_policy)
        dec = None
        try:
            dec = _Decoder(self, idx, initial_past_kvs)
            device = dec.idx.device

            def target(step_logits, ctx):
                return self._apply_sampling(step_logits, temperature, top_k, top_p,
                                            repetition_penalty, ctx, logit_bias=logit_bias)[0]

            history  = []
            produced = 0
            while produced < max_new_tokens:
                logits = dec.advance()  # plus aucun token en attente
                room   = self.config.block_size - dec.cache_len
                k      = min(n_draft, max_new_tokens - produced - 1, room)

                # ── Phase Draft ───────────────────────────────────────────────
                drafts, q_probs = [], None
                if k > 0:
                    if draft == "self":
                        drafts, q_probs = self._draft_self(dec, k, top_k, top_p,
                                                           repetition_penalty, logit_bias)
                    else:
                        drafts = self._draft_ngram(dec.idx[0].tolist(), k)

                if not drafts:
                    # Aucun draft → pas de décodage classique
                    p   = target(logits, dec.idx)
                    tok = torch.multinomial(p, num_samples=1).item()
                    burst = [tok]
                    dec.push(tok)
                else:
                    # ── Phase Verify : une passe sur tous les drafts ──────────
                    base = dec.cache_len
                    d_tensor = torch.tensor([drafts], dtype=torch.long, device=device)
                    v_logits, v_kvs = self(d_tensor, past_kvs=dec.past_kvs, return_all_logits=True)

                    # ── Accept/Reject ─────────────────────────────────────────
                    ctx   = dec.idx
                    burst = []
                    n_acc = 0
                    for i, d in enumerate(drafts):
                        # p_0 = logits courants ; p_i = sortie du verifier à la position i-1
                        p   = target(logits if i == 0 else v_logits[:, i - 1:i, :], ctx)
                        q_d = q_probs[i][d].item() if q_probs is not None else 1.0
                        if torch.rand(1).item() < min(1.0, p[d].item() / max(q_d, 1e-12)):
                            burst.append(d)
                            n_acc += 1
                            ctx = torch.cat((ctx, d_tensor[:, i:i + 1]), dim=1)
                            continue
                        # Rejet — ré-échantillonne depuis max(0, p − q) normalisé
                        if q_probs is not None:
                            residual = torch.clamp(p - q_probs[i], min=0)
                        else:
                            residual = p.clone()
                            residual[d] = 0.0  # q one-hot (draft n-gramme)
                        s = residual.sum()
                        dist = residual / s if s > 0 else p
                        burst.append(torch.multinomial(dist, num_samples=1).item())
                        break
                    else:
                        # ── Bonus token : tous les drafts acceptés ───────────
                        p = target(v_logits[:, -1:, :], ctx)
                        burst.append(torch.multinomial(p, num_samples=1).item())

                    # Cache = contexte + drafts acceptés ; dernier token en attente
                    dec.commit(base, v_kvs, drafts, n_acc, burst[-1])

                for tok in burst:
                    yield tok
                produced += len(burst)
                history.extend(burst)
                if self._should_stop(history, policy):
                    break
        finally:
            _store_cache(dec, cache_ref)

    # ── Structured outputs (JSON / JSON Schema) ──────────────────────────────

    @torch.no_grad()
    def generate_structured(
        self, idx, constraint,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = 1.0,
        initial_past_kvs=None,
        cache_ref=None,
        force_complete: bool = True,
    ):
        """
        Structured outputs — génération contrainte par une grammaire JSON / JSON Schema
        (inspiré du beta `structured-outputs-2025-12-15`).

        constraint : structured.TokenConstraint — avancé en place, token par token.

        Échantillonnage par rejet : on tire d'abord dans la distribution normale ; si le
        token viole la grammaire, on retire avec le masque des tokens autorisés (exact
        sans top-k/top-p). Si le JSON est déjà complet et que le modèle sort de la
        grammaire → fin naturelle.

        force_complete : à chaque pas, la complétion la plus courte doit tenir dans le
        budget restant — un token qui la rendrait trop longue est banni puis re-tiré, et
        quand le budget est juste suffisant on ferme le JSON. Sortie toujours complète
        si max_new_tokens >= len(constraint.completion_tokens()) au départ.

        Yield : int (token généré). À la fin, constraint.is_complete() indique si le JSON
        est complet.
        """
        closing_cache = {}

        def closing_for(c):
            if c.state not in closing_cache:
                closing_cache[c.state] = c.completion_tokens()
            return closing_cache[c.state]

        dec = None
        try:
            dec = _Decoder(self, idx, initial_past_kvs)

            def emit(tokens):
                nonlocal produced
                for t in tokens:
                    constraint.advance(t)
                    dec.push(t)
                    produced += 1
                    yield t

            produced = 0
            while produced < max_new_tokens and not constraint.is_terminal():
                remaining = max_new_tokens - produced
                # Invariant (force_complete) : la fermeture la plus courte tient toujours
                # dans le budget restant → plus de marge du tout : on ferme maintenant
                current = closing_for(constraint) if force_complete else None
                if current is not None and len(current) >= remaining:
                    yield from emit(current[:remaining])
                    break

                logits = dec.advance()
                probs  = self._apply_sampling(logits, temperature, top_k, top_p,
                                              repetition_penalty, dec.idx)
                tok = torch.multinomial(probs, num_samples=1)[0, 0].item()
                if not constraint.is_allowed(tok):
                    if constraint.is_complete():
                        break  # JSON complet et le modèle veut s'arrêter
                    tok = None

                # Tirage masqué si besoin ; un token qui rendrait la fermeture impossible
                # avec le budget restant est banni puis on re-tire (jamais de JSON tronqué)
                banned = []
                for _ in range(16):
                    if tok is None:
                        mask = constraint.allowed_mask().clone()
                        if banned:
                            mask[banned] = False
                        if not bool(mask.any()):
                            break  # plus aucun token possible
                        probs = self._apply_sampling(logits, temperature, top_k, top_p,
                                                     repetition_penalty, dec.idx, mask=mask)
                        tok = torch.multinomial(probs, num_samples=1)[0, 0].item()
                    if current is None:
                        break  # fermeture inexprimable avec ce vocabulaire : meilleur effort
                    nxt = constraint.clone()
                    nxt.advance(tok)
                    after = closing_for(nxt)
                    if after is not None and len(after) <= remaining - 1:
                        break  # token abordable
                    banned.append(tok)
                    tok = None

                if tok is None:
                    if current is not None:
                        yield from emit(current[:remaining])
                    break
                yield from emit([tok])
        finally:
            _store_cache(dec, cache_ref)
