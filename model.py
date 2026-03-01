"""
nanoPOPIXA - Un LLM minimaliste from scratch
Inspiré de nanoGPT (Karpathy), renommé et personnalisé par Dimitri
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class POPIXAConfig:
    block_size: int = 256       # longueur maximale de séquence
    vocab_size: int = 65        # taille du vocabulaire (caractères par défaut)
    n_layer: int = 6            # nombre de couches Transformer
    n_head: int = 6             # nombre de têtes d'attention
    n_embd: int = 384           # dimension des embeddings
    dropout: float = 0.2        # taux de dropout
    bias: bool = True           # biais dans les couches Linear et LayerNorm


class LayerNorm(nn.Module):
    """LayerNorm avec biais optionnel (PyTorch ne supporte pas bias=False nativement)"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention causale (masquée)
    Le cœur du Transformer — chaque token ne peut voir que les tokens précédents
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # projections clé, requête, valeur pour toutes les têtes
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # projection de sortie
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Flash Attention (PyTorch >= 2.0) — O(N) mémoire au lieu de O(N²)
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            # masque causal pour l'attention manuelle (fallback)
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = x.size()  # batch, séquence, embedding

        # calcul Q, K, V pour toutes les têtes
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            # Flash Attention — noyau CUDA fusionné, masque causal intégré
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Attention manuelle avec masque causal (fallback PyTorch < 2.0)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Feed-Forward Network — traitement non-linéaire après l'attention"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Un bloc Transformer = Attention + MLP avec connexions résiduelles"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # attention + résiduel
        x = x + self.mlp(self.ln_2(x))    # MLP + résiduel
        return x


class nanoPOPIXA(nn.Module):
    """
    nanoPOPIXA — Le modèle principal
    Architecture GPT-style : Embeddings → N×Transformer Block → LM Head
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),      # token embeddings
                wpe=nn.Embedding(config.block_size, config.n_embd),      # position embeddings
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying — partage des poids embeddings/lm_head (optimisation classique)
        self.transformer.wte.weight = self.lm_head.weight

        # initialisation des poids
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        flash_status = "Flash Attention ✓" if self.transformer.h[0].attn.flash else "Attention manuelle"
        print(f"nanoPOPIXA initialisé — {self.get_num_params()/1e6:.2f}M paramètres | {flash_status}")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Séquence trop longue ({T} > {self.config.block_size})"

        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # embeddings tokens + positions
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # passage dans les N blocs Transformer
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # mode entraînement — calcul de la loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # mode inférence — seulement le dernier token
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def _apply_sampling(self, logits, temperature, top_k, repetition_penalty, idx):
        """Applique temperature, repetition penalty et top-k sur les logits."""
        logits = logits[:, -1, :] / temperature

        # Repetition penalty — pénalise les tokens déjà générés
        if repetition_penalty != 1.0:
            for token_id in set(idx[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 repetition_penalty=1.0):
        """Génération classique (tout d'un coup)."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            probs     = self._apply_sampling(logits, temperature, top_k, repetition_penalty, idx)
            idx_next  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                        repetition_penalty=1.0):
        """Génération en streaming — yield chaque nouveau token."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            probs     = self._apply_sampling(logits, temperature, top_k, repetition_penalty, idx)
            idx_next  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat((idx, idx_next), dim=1)
            yield idx_next[0, 0].item()
