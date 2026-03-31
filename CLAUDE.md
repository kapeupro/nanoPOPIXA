# nanoPOPIXA — Instructions Claude Code

## Contexte du projet

nanoPOPIXA est un LLM minimaliste from scratch en Python/PyTorch, architecture v2 inspirée de Claude par reverse engineering du source `@anthropic-ai/claude-code` v2.1.88.

## Architecture actuelle (v2)

```
Token Embedding (wte, weight-tied avec lm_head)
  → N × Block [ RMSNorm → CausalSelfAttention(RoPE) → RMSNorm → SwiGLU ]
  → RMSNorm final → LM Head
```

- **RMSNorm** — pas de biais, `eps=1e-6`
- **RoPE** — `head_dim = n_embd // n_head`, `rope_base=10000`
- **SwiGLU** — `hidden = round_up(8/3 * n_embd, 64)`, pas de biais
- **KV-Cache** — `past_kvs: list[tuple[Tensor, Tensor]]` par couche
- **Flash Attention** — `F.scaled_dot_product_attention` si PyTorch ≥ 2.0

## Fichiers clés

| Fichier | Rôle |
|---|---|
| `model.py` | Architecture complète + génération (generate, generate_stream, thinking) |
| `train.py` | Boucle d'entraînement, LR scheduling cosine+warmup |
| `chat.py` | CLI interactif, effort levels, thinking blocks, KV-cache persistant |
| `session_cache.py` | Sérialisation/restauration du KV-cache entre sessions |
| `data_prep.py` | Téléchargement + tokenisation (tiktoken BPE ou char-level) |
| `popixa_cli.py` | Point d'entrée `popixa` avec shell interactif |

## Conventions de code

- Pas de biais (`bias=False`) sur toutes les Linear dans v2
- `POPIXAConfig` est un `@dataclass` — ajouter `rope_base: int = 10_000` si besoin
- `forward(idx, targets=None, past_kvs=None)` :
  - Avec `targets` → retourne `(logits, loss)` pour l'entraînement
  - Sans `targets` → retourne `(logits, present_kvs)` pour l'inférence
- Les checkpoints v1 (LayerNorm + wpe + GELU) sont incompatibles avec v2 → réentraîner
- `generate_stream` accepte `initial_past_kvs` et `cache_ref` pour la persistance de session

## Patterns extraits par reverse engineering (claude-code v2.1.88)

### Sampling
- **Temperature = 1 forcée quand thinking actif** (`src/services/api/claude.ts:1598`)
- **Nucleus sampling top-p** avant softmax, après top-k
- **Repetition penalty** : divise si logit > 0, multiplie si logit ≤ 0

### Gestion du contexte
- **Auto-compact à 90%** de la fenêtre, garde 50% récent (`autoCompact.ts:62-65`)
- **Warning à 80%** (`WARNING_BUFFER`)
- **Diminishing returns** : `COMPLETION_THRESHOLD=0.9`, `DIMINISHING_THRESHOLD=500` tokens
- Traduit en nanoPOPIXA : diversité bigrammes < 0.28 sur 40 tokens → arrêt

### Thinking blocks
- Phase 1 à `temperature=1` (contrainte API, `claude.ts:1598-1602`)
- Phase 2 à temperature normale
- Arrêt anticipé si répétitif dans chaque phase

### KV-Cache persistant
- Inspiré de `prompt-caching-scope-2026-01-05` et `context-management-2025-06-27`
- Fingerprint checkpoint = `md5(path:size:mtime)[:16]`
- Invalider si : modèle changé, overflow block_size, auto-compact, `/reset`

### Effort levels (inspiré `effort.ts`)
```python
EFFORT_PRESETS = {
    "low":    dict(temperature=1.0, top_k=20,  top_p=0.85, max_tokens=100),
    "medium": dict(temperature=0.8, top_k=40,  top_p=0.90, max_tokens=200),
    "high":   dict(temperature=0.7, top_k=50,  top_p=0.95, max_tokens=400),
    "max":    dict(temperature=1.0, top_k=None, top_p=0.95, max_tokens=600),
}
```

## Ce qui reste à implémenter (backlog)

1. **KV-cache persistant pour le mode thinking** — actuellement invalidé après chaque /think
2. **Interleaved thinking** (`interleaved-thinking-2025-05-14`) — raisonnement intercalé entre phrases
3. **LongRoPE** — `rope_base=500_000` pour fenêtre ~10× (`context-1m-2025-08-07`)
4. **Structured outputs** — contraintes logits pour JSON valide (`structured-outputs-2025-12-15`)
5. **Fast mode** — sous-modèle draft + re-scoring (`fast-mode-2026-02-01`)
6. **Adaptive thinking** — budget dynamique selon complexité du prompt

## Tests rapides

```bash
# Vérifier le modèle
python3 -c "from model import nanoPOPIXA, POPIXAConfig; m = nanoPOPIXA(POPIXAConfig(vocab_size=100)); import torch; print(m(torch.zeros(1,10,dtype=torch.long), torch.zeros(1,10,dtype=torch.long)))"

# Lancer le chat
popixa chat
# ou
python chat.py

# Entraîner (nano = rapide pour tester)
popixa train --size nano --data_dir data/
```
