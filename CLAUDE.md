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
- **KV-Cache** — `past_kvs: list[tuple[Tensor, Tensor]]` par couche ; les générateurs renvoient un
  `KVCache` (sous-classe de `list`) dont `token_ids` = ids exacts couverts par le cache
- **Flash Attention** — `F.scaled_dot_product_attention` si PyTorch ≥ 2.0 ; prefill par morceaux
  sur un cache existant → masque causal **décalé** (`tril(diagonal=past_len)`)
- **Dropout** — `dropout` (embeddings, résiduels, MLP) ; `attn_dropout = 0.0` par défaut DANS
  l'attention : `dropout_p > 0` force le noyau SDPA « math » lent sur CPU/MPS (≈ 2,2× en entraînement)

## Fichiers clés

| Fichier | Rôle |
|---|---|
| `model.py` | Architecture : `POPIXAConfig`, `SIZE_PRESETS`, couches, `nanoPOPIXA` (init, forward) ; ré-exporte `KVCache`, `_Decoder`… |
| `generation.py` | Génération : `GenerationMixin` hérité par `nanoPOPIXA` (sampling, generate, generate_stream, thinking, speculative, structured), `KVCache`, `_Decoder` — n'importe pas `model` |
| `train.py` | Boucle d'entraînement : cosine+warmup, memmap, autocast (`--dtype`), best.pt, `--patience`, `--run` |
| `checkpoint.py` | Format de checkpoint 2 (types simples) : `save_checkpoint` atomique, `load_checkpoint` sûr (`weights_only=True`) |
| `chat.py` | CLI interactif, effort levels, thinking blocks, KV-cache persistant |
| `session_cache.py` | Sérialisation/restauration du KV-cache entre sessions (format v3 : ids exacts + historique) |
| `structured.py` | Structured outputs : matcher JSON Schema octet par octet, masques de tokens, complétion la plus courte |
| `popixa_eval.py` | `popixa eval` (bpb, paires minimales FR, échantillons → eval.json) + `popixa bench` |
| `evals/` | `fr_paires.jsonl` (généré par `build_paires.py` depuis `paires/*.py`), `prompts_fr.txt`, `BASELINES.md` |
| `tests/` | Suite pytest (modèle, session, chat, structured, eval, checkpoint, train) — lancée par la CI |
| `data_prep.py` | Téléchargement + tokenisation (tiktoken BPE ou char-level) |
| `popixa_cli.py` | Point d'entrée `popixa` avec shell interactif |

## Conventions de code

- Pas de biais (`bias=False`) sur toutes les Linear dans v2
- `POPIXAConfig` est un `@dataclass` — ajouter `rope_base: int = 10_000` si besoin ; picklé dans les
  checkpoints sous `model.POPIXAConfig` (ne pas le déplacer) → nouveau champ à la FIN, avec défaut,
  lu via `getattr(config, "champ", défaut)` (configs anciens sans l'attribut)
- Presets d'architecture : `model.SIZE_PRESETS` (nano/small/medium), partagés par train.py et bench ;
  `popixa train` = `small` par défaut, `--seed` (défaut 1337) → poids identiques bit à bit sur CPU
- `forward(idx, targets=None, past_kvs=None)` :
  - Avec `targets` → retourne `(logits, loss)` pour l'entraînement
  - Sans `targets` → retourne `(logits, present_kvs)` pour l'inférence
- Les checkpoints v1 (LayerNorm + wpe + GELU) sont incompatibles avec v2 → réentraîner
  (`checkpoint_v1_error(state_dict)` donne un message propre dans chat / gen / train --resume)
- `generate_stream` accepte `initial_past_kvs` et `cache_ref` pour la persistance de session
- Tous les générateurs passent par `_Decoder` (generation.py) — invariants :
  - `dec.idx` = contexte connu (ids du cache initial s'ils sont connus + entrée + tokens générés)
  - `dec.past_kvs` couvre `dec.idx` sauf les `pending` derniers tokens
  - au-delà de `block_size` : prefill glissant sur les 75% derniers tokens (jamais de crash)
  - `cache_ref` est rempli dans un `finally` → aussi en cas d'interruption / `gen.close()`,
    et le cache final couvre l'entrée + TOUS les tokens yieldés
- `_apply_sampling` : penalty → logit_bias → masque → temperature (≤ 0 = greedy) → top-k (≤ 0 = off) → top-p ;
  ne modifie jamais les logits de l'appelant
- Politiques d'arrêt : `stop_policy="repetitive" | "diminishing" | "off"` (`_should_stop`)
- Tests : `python -m pytest -q tests` (rapides, CPU, modèles minuscules)

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

## Implémenté — vague 3

- **Structured outputs** (`structured-outputs-2025-12-15`) — `structured.py` + `generate_structured`
  - Matcher octet par octet (états immuables/hashables, frozenset de piles) → masque de tokens
    via un trie du vocabulaire (mis en cache par état)
  - Échantillonnage par rejet : tirage normal, masque seulement si le token viole la grammaire
  - `force_complete` : ferme le JSON avec `completion_tokens()` quand le budget s'épuise
  - Sous-ensemble JSON Schema : types (+ unions), properties/required (ordre de déclaration,
    pas de propriétés additionnelles), enum/const, items/prefixItems/minItems/maxItems,
    minLength/maxLength, anyOf/oneOf, allOf à 1 élément, `$ref` locaux récursifs
  - Mots-clés non appliqués (pattern, format, minimum…) → `ignored_keywords`, signalés dans le chat
  - CLI : `/json [schéma]` dans le chat, `popixa gen --json / --schema`
- **`_has_diminishing_returns`** branché : `think_stop_policy="diminishing"` (phase thinking),
  `/stop repetitive|diminishing|off` pour les réponses
- **Speculative decoding corrigé** — verify en une passe avec masque décalé, cache tronqué aux
  drafts acceptés, drafts n-grammes (prompt lookup) par défaut ; greedy spéculatif == greedy normal

## Implémenté — v2.2 « Mètre-étalon »

- **`popixa eval`** — résultats déterministes (pas de date/durée dans eval.json) :
  - `bpb` = Σ NLL / (ln 2 × octets UTF-8 des tokens prédits) — fenêtres de block_size sans
    chevauchement, octets exacts par token (`token_nbytes`) → comparable char vs BPE
  - `paires` = % log P(bonne) > log P(fautive) (sommes BLiMP, amorce « \n »), comparé à
    `baseline_longueur` (préférer la phrase la plus courte) ; un modèle non entraîné ≈ baseline
  - `samples` = 20 amorces × 128 tokens, graine `seed + i`, `stop_policy="off"`, distinct-2 + drapeaux
    répétitifs (None si trop court pour le détecteur), amorces altérées par le tokenizer signalées
  - empreinte = sha256 des poids (`weights_fingerprint`), pas du fichier ; IC 95 % (Wilson) sur les paires
- **`popixa prep`** coupe train/val à 90 % du TEXTE avant tokenisation → même `val.bin` (en texte)
  pour char et BPE ; `popixa eval --max_bytes N` = même extrait pour tous les tokenizers
- **Paires** : modifier `evals/paires/<phénomène>.py` puis `python evals/build_paires.py`
  (la CI vérifie avec `--check`) ; une seule différence par paire, longueurs équilibrées
- **`popixa bench`** — tokens/s train (fwd+bwd+AdamW, dropout de train.py) et génération greedy,
  mémoire (nature indiquée : pic CUDA / driver MPS / RSS processus), TFLOPS ≈ (6N + 12·L·T·d) × tokens/s
- **`train.py --resume`** sans `--size` → réglages enregistrés dans `ckpt["train"]` (size, batch,
  max_iters, lr, warmup), sinon taille déduite de l'architecture (`_size_of`) ; garde MPS testée sur
  l'architecture effective

## Implémenté — v2.3 « Socle »

- **Checkpoints format 2** (`checkpoint.py`) : `{"format": 2, "model", "config": dict, "iter", "tokenizer",
  "vocab"?, "optimizer"?, "train", "best_val", "best_iter", …}` — JAMAIS d'objet picklé ; charger avec
  `load_checkpoint` (jamais `torch.load` direct) ; anciens checkpoints : seule `POPIXAConfig` autorisée
- **train.py** :
  - données `np.memmap` (uint16) → `get_batch` empile des tranches ; mode `input.txt` en numpy en RAM
  - AdamW : 2 groupes (dim ≥ 2 → weight_decay 0.1 ; RMSNorm → 0), betas (0.9, 0.999) — 0.95 mesuré
    moins bon (ablations dans evals/BASELINES.md) ; réglages experts `--lr --weight_decay --beta2
    --dropout --attn_dropout` (`_hp` : option > réglage du run repris > défaut)
  - `estimate_loss` : `_eval_gen` re-seedé (`seed + 1`) à chaque évaluation → mêmes batchs ; les batchs
    d'entraînement ont leur propre générateur (`_batch_gen`, reprise : `seed + iter`)
  - `best.pt` (sans optimizer) à chaque amélioration de la val loss ; `--patience N` → arrêt anticipé,
    `stopped_early` dans checkpoint.pt (`--resume` ne repart pas sauf `--patience` explicite)
  - `--dtype auto` : bfloat16 (sinon float16 + GradScaler) sur CUDA, float32 ailleurs ; autocast partagé
    entraînement / éval
  - `--run NOM` ≡ `--out_dir runs/NOM` ; `train.log` et `run.json` DANS le dossier ; `--eval` → eval.json
  - réglages repris par `--resume` : `ckpt["train"]` (size, seed, eval_interval, eval_iters, batch,
    max_iters, lr, warmup, patience, grad_accum, weight_decay, beta2, dropout, attn_dropout) — PAS le
    dtype (propre à la machine : `--dtype auto` à chaque lancement) ; `--weight_decay` / `--beta2` à la
    reprise sont réappliqués aux groupes de l'optimiseur (load_state_dict remet les anciens)
  - best.pt jamais écrit à l'évaluation initiale ; à la reprise, vérifié contre (best_iter, best_val)
    du checkpoint (restauré si le meilleur point est le checkpoint, sinon réécrit plus tard) ; un
    best.pt d'un autre run est supprimé à la 1re sauvegarde (copie dans best.prev.pt)
  - nouvel entraînement : checkpoint.pt / best.pt COPIÉS en *.prev.pt ; rien d'autre ne change dans le
    dossier avant la 1re sauvegarde (`_commit_launch` : train.log, run.json « en cours », eval.json →
    *.prev) → interrompu avant, le dossier reste celui du run précédent ; run.json « interrompu » /
    « erreur » si la boucle ou la sauvegarde finale échoue après engagement
- **CLI** : `popixa runs`, `popixa monitor --run`, `popixa bench --dtype`

## Ce qui reste à implémenter (backlog)

1. **Structured outputs + thinking** — phase de réflexion libre puis réponse JSON contrainte
2. **Mots-clés JSON Schema restants** — `pattern` (regex → automate), `minimum`/`maximum`, `format`
3. **LongRoPE réel** — étendre la table RoPE au-delà de `block_size` (aujourd'hui `--longrope`
   ne change que `rope_base`, la fenêtre reste `block_size`)

## Tests rapides

```bash
# Suite complète (≈ 8 min sur CPU 4 cœurs)
python -m pytest -q tests

# Vérifier le modèle
python3 -c "from model import nanoPOPIXA, POPIXAConfig; m = nanoPOPIXA(POPIXAConfig(vocab_size=100)); import torch; print(m(torch.zeros(1,10,dtype=torch.long), torch.zeros(1,10,dtype=torch.long)))"

# Lancer le chat
popixa chat
# ou
python chat.py

# Entraîner (nano = rapide pour tester)
popixa train --size nano --data_dir data/
```
