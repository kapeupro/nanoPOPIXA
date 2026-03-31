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
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


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
    """Applique RoPE sur Q et K."""
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

        kv_len    = k.size(2)
        is_causal = (T == kv_len)  # True au prefill, False en décodage

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if is_causal:
                att = att.masked_fill(
                    self.bias[:, :, :T, :kv_len] == 0, float("-inf")
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
        print(f"nanoPOPIXA v2 — {self.get_num_params()/1e6:.2f}M params | RMSNorm · SwiGLU · RoPE | {flash}")

    # ── Utilitaires ──────────────────────────────────────────────────────────

    def get_num_params(self, non_embedding: bool = True) -> int:
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

    def forward(self, idx, targets=None, past_kvs=None):
        """
        Mode entraînement (targets != None) :
            Retourne (logits, loss) — compatible train.py.

        Mode inférence (targets == None) :
            Retourne (logits, present_kvs) — utilisé par generate/generate_stream.
            logits : (B, 1, vocab_size) — seulement le dernier token.
        """
        B, T = idx.size()
        assert T <= self.config.block_size, f"Séquence trop longue ({T} > {self.config.block_size})"

        x = self.transformer.drop(self.transformer.wte(idx))

        present_kvs = []
        for i, block in enumerate(self.transformer.h):
            past_kv = past_kvs[i] if past_kvs is not None else None
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

        # Inférence — seulement le dernier token, on retourne le cache
        logits = self.lm_head(x[:, [-1], :])
        return logits, present_kvs

    # ── Sampling ─────────────────────────────────────────────────────────────

    def _apply_sampling(self, logits, temperature, top_k, top_p, repetition_penalty, idx):
        """
        Applique dans l'ordre :
          1. Temperature scaling
          2. Repetition penalty
          3. Top-k
          4. Top-p (nucleus sampling)
        Retourne les probabilités finales.
        """
        logits = logits[:, -1, :] / temperature  # (B, vocab_size)

        # Repetition penalty — pénalise les tokens déjà générés
        if repetition_penalty != 1.0:
            for token_id in set(idx[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # Top-k — ne garde que les k meilleurs logits
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
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

    # ── Génération ───────────────────────────────────────────────────────────

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

    # ── Génération standard ──────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 repetition_penalty=1.0, top_p=None, stop_on_repetition=False):
        """
        Génération avec KV-Cache :
          1. Prefill  — traite tout le prompt en une passe, construit le cache
          2. Decode   — génère un token à la fois en O(1) grâce au cache

        stop_on_repetition : arrêt anticipé si la génération diverge (diminishing returns)
        """
        # Prefill
        logits, past_kvs = self(idx)
        probs    = self._apply_sampling(logits, temperature, top_k, top_p, repetition_penalty, idx)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx      = torch.cat((idx, idx_next), dim=1)
        generated = [idx_next[0, 0].item()]

        # Decode
        for _ in range(max_new_tokens - 1):
            logits, past_kvs = self(idx[:, [-1]], past_kvs=past_kvs)
            probs    = self._apply_sampling(logits, temperature, top_k, top_p, repetition_penalty, idx)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat((idx, idx_next), dim=1)
            tok = idx_next[0, 0].item()
            generated.append(tok)
            if stop_on_repetition and self._is_repetitive(generated):
                break

        return idx

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                        repetition_penalty=1.0, top_p=None, stop_on_repetition=False):
        """
        Streaming token par token avec KV-Cache.
        Yield chaque token généré (int).
        stop_on_repetition : s'arrête automatiquement si la génération diverge.
        """
        # Prefill
        logits, past_kvs = self(idx)
        probs    = self._apply_sampling(logits, temperature, top_k, top_p, repetition_penalty, idx)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx      = torch.cat((idx, idx_next), dim=1)
        tok = idx_next[0, 0].item()
        generated = [tok]
        yield tok

        # Decode
        for _ in range(max_new_tokens - 1):
            logits, past_kvs = self(idx[:, [-1]], past_kvs=past_kvs)
            probs    = self._apply_sampling(logits, temperature, top_k, top_p, repetition_penalty, idx)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat((idx, idx_next), dim=1)
            tok = idx_next[0, 0].item()
            generated.append(tok)
            if stop_on_repetition and self._is_repetitive(generated):
                break
            yield tok

    # ── Thinking blocks (Claude-inspired) ───────────────────────────────────

    @torch.no_grad()
    def generate_stream_with_thinking(
        self, idx,
        think_budget: int = 150,
        response_budget: int = 300,
        temperature: float = 0.8,
        top_k: int = None,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ):
        """
        Génération deux phases inspirée de Claude's extended thinking.

        Phase 1 — Think (temperature=1 forcée, comme Claude) :
            Le modèle génère un raisonnement interne libre.
            Arrêt anticipé si répétitif (diminishing returns).

        Phase 2 — Response (temperature normale) :
            Le modèle génère la réponse finale en ayant "vu" son propre raisonnement.
            Arrêt anticipé si répétitif.

        Yield : tuples (phase, token_int)
            phase = 'think' | 'response'
        """
        # ── Prefill ──────────────────────────────────────────────────────────
        logits, past_kvs = self(idx)

        # ── Phase 1 : Thinking — temperature=1 (contrainte API Claude) ──────
        generated_think = []
        probs    = self._apply_sampling(logits, 1.0, top_k, top_p, repetition_penalty, idx)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx      = torch.cat((idx, idx_next), dim=1)
        tok = idx_next[0, 0].item()
        generated_think.append(tok)
        yield ("think", tok)

        for _ in range(think_budget - 1):
            logits, past_kvs = self(idx[:, [-1]], past_kvs=past_kvs)
            probs    = self._apply_sampling(logits, 1.0, top_k, top_p, repetition_penalty, idx)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat((idx, idx_next), dim=1)
            tok = idx_next[0, 0].item()
            generated_think.append(tok)
            yield ("think", tok)
            if self._is_repetitive(generated_think):
                break  # Diminishing returns détecté → fin du thinking

        # ── Phase 2 : Response — temperature normale ──────────────────────────
        generated_resp = []
        logits, past_kvs = self(idx[:, [-1]], past_kvs=past_kvs)
        probs    = self._apply_sampling(logits, temperature, top_k, top_p, repetition_penalty, idx)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx      = torch.cat((idx, idx_next), dim=1)
        tok = idx_next[0, 0].item()
        generated_resp.append(tok)
        yield ("response", tok)

        for _ in range(response_budget - 1):
            logits, past_kvs = self(idx[:, [-1]], past_kvs=past_kvs)
            probs    = self._apply_sampling(logits, temperature, top_k, top_p, repetition_penalty, idx)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat((idx, idx_next), dim=1)
            tok = idx_next[0, 0].item()
            generated_resp.append(tok)
            yield ("response", tok)
            if self._is_repetitive(generated_resp):
                break  # Diminishing returns → fin de la réponse
