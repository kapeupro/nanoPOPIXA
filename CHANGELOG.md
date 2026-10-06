# Changelog

Toutes les versions notables de nanoPOPIXA. Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/).

## [2.2.0] — « Mètre-étalon »

On mesure avant de changer : chaque version suivante devra prouver, chiffres à l'appui,
qu'elle fait mieux que celle-ci.

### Ajouté
- **`popixa eval`** — évaluation déterministe d'un checkpoint, résultats dans `eval.json`
  (sans date ni durée : même checkpoint → même fichier) :
  - `bpb` : bits par octet sur `val.bin`, comparable entre tokenizers (caractère, BPE) ;
  - `paires` : paires minimales françaises (accord sujet-verbe, accord nominal, participe
    passé, élision, prépositions), score comparé à une baseline « phrase la plus courte » ;
  - `samples` : 20 amorces françaises à graine fixe → `samples.md`, distinct-2 et taux de
    sorties répétitives.
- **`evals/`** — jeu de paires minimales `fr_paires.jsonl`, reconstruit et vérifié par
  `evals/build_paires.py` (`--check` en CI), amorces `prompts_fr.txt`, chiffres de
  référence `BASELINES.md`.
- **`popixa bench`** — tokens/s en entraînement et en génération (normale et speculative),
  mémoire pic, TFLOPS effectifs, temps estimé pour 1 milliard de tokens.
- **`train.py --seed`** (défaut 1337) — initialisation, dropout et tirage des batchs
  reproductibles : même graine → mêmes poids, bit à bit (CPU).
- **`popixa --version`**.

### Modifié
- `popixa train` entraîne un modèle **`small`** par défaut (au lieu de `medium`, qui
  dépassait la mémoire MPS des Mac Apple Silicon).
- Les presets d'architecture sont partagés (`model.SIZE_PRESETS`) entre `train.py` et
  `popixa bench`.

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
