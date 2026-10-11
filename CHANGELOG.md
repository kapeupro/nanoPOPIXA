# Changelog

Toutes les versions notables de nanoPOPIXA. Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/).

## [2.3.0] — « Socle »

Un entraînement sérieux avant d'attaquer un vrai corpus français : plus rapide, plus sobre en
mémoire, reproductible et sûr. Chaque changement est mesuré avec `popixa bench` / `popixa eval`
(voir `evals/BASELINES.md`).

### Ajouté
- **Expériences nommées** : `popixa train --run NOM` écrit tout dans `runs/NOM/` (`checkpoint.pt`,
  `best.pt`, `train.log`, `run.json`) ; `--eval` lance `popixa eval` sur `best.pt` en fin
  d'entraînement ; `popixa runs` compare les expériences ; `popixa monitor --run NOM`.
- **`best.pt`** (poids à la meilleure val loss) et **arrêt anticipé** `--patience N`.
- **Précision mixte** `--dtype auto|float32|bfloat16|float16` (auto : bfloat16 sur GPU CUDA) ;
  `popixa bench --dtype` pour mesurer avant d'adopter une précision réduite.
- `--grad_accum N`, `--eval_interval`, `--eval_iters` ; réglages experts `--lr`, `--weight_decay`,
  `--beta2`, `--dropout`, `--attn_dropout` (ablations, repris par `--resume`).
- **`checkpoint.py`** : format de checkpoint 2, chargé avec `weights_only=True`.

### Modifié
- **Dropout d'attention supprimé** (`attn_dropout = 0`, le dropout résiduel reste à 0.1) : sur
  CPU / Mac, un dropout dans `scaled_dot_product_attention` forçait l'implémentation lente.
- **Données en memmap** : `train.bin` n'est plus chargé en RAM.
- **Weight decay 0.1 sur les seules matrices** (il s'appliquait aussi aux poids des RMSNorm).
  betas AdamW inchangés (0.9, 0.999) : 0.95, essayé, coûtait ≈ 0.08 bit par octet (mesuré).
- **Validation sur des batchs fixes** : les val loss successives sont comparables entre elles.
- `train.log` est écrit dans le dossier de l'expérience (`out-nanopopixa/train.log` par défaut) ;
  `popixa monitor` suit par défaut le `train.log` le plus récent.
- `--resume` reprend aussi la graine et les batchs d'évaluation du run (val loss comparables),
  et applique à l'optimiseur les `--weight_decay` / `--beta2` demandés.
- `popixa train` transmet enfin les options valant 0 (`--seed 0`, `--dropout 0`, `--patience 0`…).
- `--dtype auto` choisit float16 + GradScaler sur les GPU antérieurs à Ampere (bfloat16 y est émulé).
- **`model.py` découpé** : l'architecture reste dans `model.py`, toute la génération passe dans
  `generation.py` (API inchangée : `model.generate_stream(...)` etc.).

### Sécurité
- Les checkpoints sont chargés en mode sûr : un fichier piégé est refusé au lieu d'exécuter du
  code. Les checkpoints 2.0 – 2.2 restent lisibles (seule la classe `POPIXAConfig` est autorisée
  en plus des types simples) et sont réécrits au format 2 à la reprise. Avec PyTorch < 2.4, qui ne
  sait pas n'autoriser que cette classe, il faut l'accord explicite `POPIXA_TRUST_CHECKPOINTS=1` :
  jamais de repli silencieux sur un chargement complet.

## [2.2.0] — « Mètre-étalon »

On mesure avant de changer : chaque version suivante devra prouver, chiffres à l'appui,
qu'elle fait mieux que celle-ci.

### Ajouté
- **`popixa eval`** — évaluation déterministe d'un checkpoint, résultats dans `eval.json`
  (sans date ni durée, empreinte calculée sur les poids : mêmes poids → même fichier) :
  - `bpb` : bits par octet sur `val.bin`, comparable entre tokenizers (caractère, BPE) ;
    `--max_bytes` évalue le même extrait de texte (à un token près) quel que soit le tokenizer ;
  - `paires` : 344 paires minimales françaises (accord sujet-verbe, accord nominal, participe
    passé, élision, prépositions), score avec intervalle de confiance à 95 %, comparé à une
    baseline « phrase la plus courte » ;
  - `samples` : 20 amorces françaises à graine fixe, 128 tokens → `samples.md`, distinct-2,
    taux de sorties répétitives, amorces altérées par le tokenizer signalées.
- **`evals/`** — jeu de paires minimales `fr_paires.jsonl`, reconstruit et vérifié par
  `evals/build_paires.py` (`--check` en CI), amorces `prompts_fr.txt`, chiffres de
  référence `BASELINES.md`.
- **`popixa bench`** — tokens/s en entraînement (avec le dropout de train.py, `--dropout`) et en
  génération, mémoire, TFLOPS effectifs, temps estimé pour 1 milliard de tokens.
- **`train.py --seed`** (défaut 1337) — initialisation, dropout et tirage des batchs
  reproductibles : même graine → mêmes poids, bit à bit (CPU).
- **`popixa --version`**.

### Modifié
- `popixa train` entraîne un modèle **`small`** par défaut (au lieu de `medium`, qui
  dépassait la mémoire MPS des Mac Apple Silicon). Le checkpoint enregistre ses réglages
  d'entraînement : `--resume` (sans `--size`, ou avec la même taille) reprend le run tel quel
  (batch, LR, `max_iters`) ; un
  checkpoint 2.1 reprend le preset de sa taille (un `medium` lancé en 2.1 reste `medium`).
- `popixa prep` coupe train/val à 90 % du **texte** (avant tokenisation, au lieu de 90 % des
  tokens) : `val.bin` contient le même texte quel que soit le tokenizer, les bpb sont comparables.
  Données caractère inchangées ; données BPE à préparer de nouveau pour comparer.
- README : les durées d'entraînement annoncées (« nano : 10-15 min sur CPU ») étaient fausses ;
  remplacées par la mesure (`popixa bench`, `evals/BASELINES.md`).
- Les presets d'architecture sont partagés (`model.SIZE_PRESETS`) entre `train.py` et
  `popixa bench`.

### Mise à jour depuis 2.1
- Installation éditable : après `git pull`, relance `pip install -e .` pour que la commande
  `popixa` connaisse les nouveaux modules (`popixa update` fait `git pull` + `pip install -e .`).

## [2.1.0] — « Point zéro »

### Ajouté
- **Structured outputs** (JSON Schema) : `generate_structured`, `/json` dans le chat,
  `popixa gen --json / --schema` ; le JSON produit est toujours complet et valide.
- Suite de tests pytest (≈ 670 tests) lancée par la CI sur Python 3.9 et 3.11.

### Corrigé
- Génération : masque causal décalé pour le prefill sur cache, fenêtre glissante au-delà de
  `block_size` (plus de plantage), speculative decoding exact (même distribution que
  l'échantillonnage normal).
- Chat et sessions : KV-cache persistant avec les ids exacts des tokens, auto-compact en
  vrais tokens, sauvegarde même en cas d'interruption.
- Entraînement, préparation des données, CLI et monitor : codes de sortie, reprise,
  sauvegarde finale, divergence visible.

## [2.0.0]

- Architecture v2 : RMSNorm · RoPE · SwiGLU · KV-cache · Flash Attention · thinking blocks,
  effort levels, speculative decoding, auto-compact.
