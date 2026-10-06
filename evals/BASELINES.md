# Baselines — nanoPOPIXA 2.2 « Mètre-étalon »

Chiffres de référence pour juger les versions suivantes : une amélioration doit se voir ici,
au-delà du bruit de mesure (intervalle de confiance des paires ≈ ± 5 points).

## Évaluation (`popixa eval`)

Corpus : Victor Hugo, *Les Misérables* tome I (`popixa prep --dataset hugo --char`),
677 686 caractères, tokenizer **caractère** (vocabulaire 115), split `val` = 67 769 tokens.
Échantillons : 20 amorces × 128 tokens, graine 1337, temperature 0.8, top_k 40 ; aucune sortie
répétitive ni à rendements décroissants pour les deux modèles.

| Modèle | bpb ↓ | loss | Paires ↑ [IC 95 %] | Baseline longueur | distinct-2 |
|---|---|---|---|---|---|
| `nano` aléatoire (non entraîné) | 6.766 | 4.784 | 53.2 % [47.9–58.5] | 53.0 % | 0.93 (bruit) |
| `nano` · 1 000 itérations · batch 16 | **2.900** | 2.050 | **61.5 %** [56.2–66.6] | 53.0 % | 0.66 |

Paires par phénomène (338 paires couvertes ; 6 contiennent « œ », absent du vocabulaire de Hugo) :

| Modèle | accord sujet-verbe | accord nominal | participe passé | élision | prépositions |
|---|---|---|---|---|---|
| aléatoire | 50 % | 50 % | 55 % | 55 % | 56 % |
| `nano` 1 000 it. | 45 % | **72 %** | 61 % | 63 % | 66 % |

Lecture :
- Le modèle aléatoire tombe sur la **baseline longueur** (préférer la phrase la plus courte) : le jeu
  de paires n'est pas gagnable sans apprendre la langue.
- Le `nano` entraîné la dépasse de 8,5 points (≈ 3 écarts-types) : les paires mesurent bien quelque chose.
- L'**accord sujet-verbe** reste sous le hasard : un petit modèle caractère ne relie pas encore
  le sujet au verbe (accords à distance, attracteurs) — premier objectif mesurable pour 2.3+.
- Les échantillons (`samples.md`) ont la texture du français (« il regardait », « qu'il ») mais
  pas encore de mots fiables : à comparer à chaque version.

Reproduire :

```bash
popixa prep --dataset hugo --char --data_dir data_hugo_char
popixa train --size nano --data_dir data_hugo_char --max_iters 1000 --batch_size 16   # --seed 1337 par défaut
popixa eval --data_dir data_hugo_char --out eval.json
```

Entraînement : ≈ 26 min sur CPU 4 cœurs (Xeon 2,1 GHz, 3 threads), courbe de `train.log` :

| itération | 0 | 200 | 400 | 600 | 800 | 1 000 |
|---|---|---|---|---|---|---|
| val loss | 4.79 | 2.48 | 2.27 | 2.13 | 2.08 | 2.03 |

## Débit (`popixa bench`)

Machine : conteneur cloud, Intel Xeon 2,1 GHz, 4 cœurs, sans GPU, PyTorch 2.x CPU, 4 threads.

| Preset | Vocab | Batch | Entraînement (dropout 0.1) | Sans dropout | Génération (greedy) |
|---|---|---|---|---|---|
| `nano` (0.9M) | 115 | 16 × 512 | 6 540 tok/s | 14 720 tok/s | 610 tok/s |
| `small` (30M avec embeddings) | 50 257 | 4 × 1024 | 515 tok/s | 1 150 tok/s | 156 tok/s |

À retenir :
- Sur CPU, le **dropout d'attention** fait passer `scaled_dot_product_attention` sur
  l'implémentation lente : l'entraînement est ≈ 2,2× plus lent qu'avec `dropout=0`.
- Le preset `small` par défaut (16 × 1024 × 10 000 itérations ≈ 164M tokens) demanderait ≈ 88 h sur
  ce CPU : sur CPU, entraîner `nano` ; `small` est fait pour Apple Silicon (MPS) ou un GPU.
- Lance `popixa bench` sur TA machine pour recaler ces estimations.
