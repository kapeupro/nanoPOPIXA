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

## Patterns extraits — vague 2 (exploration complète)

### Depuis `src/services/compact/autoCompact.ts`
- `MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3` → circuit breaker : désactive l'auto-compact après 3 échecs
- `POST_COMPACT_MAX_FILES_TO_RESTORE = 5` · `POST_COMPACT_TOKEN_BUDGET = 50_000`

### Depuis `src/constants/toolLimits.ts`
- `BYTES_PER_TOKEN = 4` → estimation conservative chars→tokens pour `/ctx` et seuils de compaction

### Depuis `src/constants/betas.ts` (betas supplémentaires)
- `redact-thinking-2026-02-12` → supprime tokens thinking du contexte (implémenté : `/redactthink`)
- `task-budgets-2026-03-13` → budget total tokens sur une tâche multi-tours (implémenté : `/taskbudget`)
- `token-efficient-tools-2026-03-28` → FC v3 token-efficient (non applicable sans outils)
- `advisor-tool-2026-03-01` → outil advisor (non applicable)

### Depuis `src/query/tokenBudget.ts`
- Delta tracking sur 3 fenêtres : `_has_diminishing_returns(tokens, window=40, n_checks=3, threshold=0.28)`
  Complète `_is_repetitive` (1 fenêtre) avec une version multi-fenêtre plus conservative

### Depuis `src/utils/context.ts`
- `CAPPED_DEFAULT_MAX_TOKENS = 8_000` (p99 = 4 911 tokens) — cap par défaut optimisé
- `ESCALATED_MAX_TOKENS = 64_000` — escalade si dépassement du p99

### Depuis `src/utils/fastMode.ts`
- Fast mode : états cooldown (rate_limit | overloaded), `PREFETCH_MIN_INTERVAL_MS = 30_000`
- Dans nanoPOPIXA : fast mode = speculative decoding, pas de cooldown API

## Ce qui reste à implémenter (backlog)

1. **Structured outputs** — contraintes logits pour JSON valide (`structured-outputs-2025-12-15`)
   logit_bias déjà dans `_apply_sampling`, il manque le parser de schéma JSON
2. **`_has_diminishing_returns`** dans les générateurs — actuellement seul `_is_repetitive` est appelé

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
