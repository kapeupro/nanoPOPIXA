"""
nanoPOPIXA v2 — Architecture moderne (Claude-inspired, reverse-engineered)
RMSNorm · SwiGLU · RoPE · attention causale + KV-Cache (vs v1 : docstring de nanoPOPIXA)

Ce module ne contient que l'ARCHITECTURE : config, couches, forward. La génération
(sampling top-k/top-p, generate*, thinking, speculative decoding, structured outputs)
est dans generation.py, dont nanoPOPIXA hérite via GenerationMixin (découpage 2.3).
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

# Ré-export : `from model import KVCache, _Decoder…` (code d'avant la 2.3) reste valable
from generation import GenerationMixin, KVCache, _Decoder, _cache_len, _store_cache  # noqa: F401


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
    dropout:    float = 0.1      # dropout des embeddings, résiduels et MLP
    bias:       bool = False     # conservé pour compatibilité checkpoints v1 (ignoré)
    rope_base:  int = 10_000     # base fréquentielle RoPE (10k standard, 500k LongRoPE)
    attn_dropout: float = 0.0   # dropout DANS l'attention (0 : noyau SDPA rapide ; LLaMA)


# Presets d'architecture partagés par train.py et popixa bench
# (params hors embeddings : nano ~0.9M, small ~10M, medium ~85M)
SIZE_PRESETS = {
    "nano":   dict(block_size=512,  n_layer=4,  n_head=4,  n_embd=128),
    "small":  dict(block_size=1024, n_layer=6,  n_head=6,  n_embd=384),
    "medium": dict(block_size=1024, n_layer=12, n_head=12, n_embd=768),
}


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
        # Dropout DANS l'attention (≠ résiduel) : dropout_p > 0 fait retomber SDPA sur le noyau
        # « math » lent sur CPU/MPS (2.2 : entraînement ≈ 2,2× plus lent). getattr : configs < 2.3
        self.attn_dropout_p = getattr(config, "attn_dropout", 0.0)

        # Projections Q, K, V fusionnées + projection de sortie (sans biais)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd,     bias=False)

        self.attn_dropout  = nn.Dropout(self.attn_dropout_p)
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
                dropout_p=self.attn_dropout_p if self.training else 0.0,
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
# nanoPOPIXA v2 (+ détection des checkpoints v1 incompatibles)
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


class nanoPOPIXA(GenerationMixin, nn.Module):
    """
    nanoPOPIXA v2 — Architecture :
        Token Embedding → N × [RMSNorm → Attention(RoPE) → RMSNorm → SwiGLU] → RMSNorm → LM Head

    Vs v1 (nanoGPT-style) :
      ✗ wpe (positional embeddings appris)  → ✓ RoPE (0 paramètre, meilleure généralisation)
      ✗ LayerNorm avec biais               → ✓ RMSNorm (plus rapide)
      ✗ GELU MLP                           → ✓ SwiGLU (meilleur gradient flow)
      ✗ Recompute complet à chaque token   → ✓ KV-Cache (décodage O(1))
      ✗ top-k uniquement                   → ✓ top-p nucleus sampling (generation.py)

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
