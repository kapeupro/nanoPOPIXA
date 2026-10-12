# Baselines — nanoPOPIXA

Chiffres de référence pour juger les versions suivantes : une amélioration doit se voir ici,
au-delà du bruit de mesure (intervalle de confiance des paires ≈ ± 5 points ; d'une graine à
l'autre, ≈ 0,01 bit par octet).

## 2.3 « Socle » — référence actuelle

Même protocole qu'en 2.2 (`nano` caractère, Hugo, 1 000 itérations, batch 16, graine 1337) avec
l'entraînement 2.3 (pas de dropout dans l'attention, weight decay 0.1 sur les matrices, batchs de
validation fixes, memmap). CPU 4 cœurs (Xeon 2,1 GHz), 3 threads.

| Version | bpb ↓ | Paires ↑ [IC 95 %] | val loss | Durée d'entraînement |
|---|---|---|---|---|
| 2.2 | 2.900 | 61.5 % [56.2–66.6] | 2.03 (val bruitée) | 26 min |
| **2.3** | **2.881** | **62.1 %** [56.9–67.1] | **1.981** | **10 min** |

Paires par phénomène (2.3) : accord sujet-verbe 47 % · accord nominal 72 % · participe passé 58 % ·
élision 63 % · prépositions 71 %. distinct-2 des échantillons : 0.66, aucune sortie répétitive.

Courbe de val loss (batchs fixes : chaque point est comparable au précédent, la courbe ne
« saute » plus comme en 2.2) :

| itération | 0 | 200 | 400 | 600 | 800 | 1 000 |
|---|---|---|---|---|---|---|
| val loss | 4.79 | 2.50 | 2.15 | 2.04 | 2.01 | 1.98 |

### Ablations : d'où vient chaque gain ?

Chaque ligne ne change qu'un réglage (`popixa train ... --attn_dropout / --weight_decay / --beta2`) :

| Run | dropout d'attention | weight decay | beta2 | bpb ↓ | Paires | Durée |
|---|---|---|---|---|---|---|
| config 2.2 rejouée en 2.3 | 0.1 | 0.01 | 0.999 | 2.893 | 58.0 % | 31 min |
| sans dropout d'attention | 0 | 0.01 | 0.999 | 2.880 | 61.8 % | 10 min |
| **+ weight decay 0.1 (défaut 2.3)** | 0 | 0.1 | 0.999 | **2.881** | **62.1 %** | 10 min |
| + beta2 0.95 (essayé, rejeté) | 0 | 0.1 | 0.95 | 2.966 | 56.8 % | 9 min |
| idem, graine 7 (bruit) | 0 | 0.1 | 0.95 | 2.953 | 58.0 % | 10 min |

Lecture :
- **Supprimer le dropout d'attention** : 3× plus rapide, et même un peu meilleur (2.880 vs 2.893).
- **Weight decay 0.1 sur les matrices** : neutre ici (2.880 vs 2.881), gardé (réglage standard,
  utile sur des corpus plus grands où le sur-apprentissage compte).
- **beta2 = 0.95**, courant pour les grands LLM, coûte ≈ 0.08 bit par octet sur ce petit modèle —
  6 à 8 fois le bruit d'une graine à l'autre : rejeté, beta2 reste 0.999.
- La config 2.2 rejouée avec le code 2.3 retombe sur les chiffres 2.2 (2.893 vs 2.900) : le reste
  de la refonte (memmap, batchs de validation fixes, checkpoints) ne change pas la qualité.

Reproduire :

```bash
popixa prep --dataset hugo --char --data_dir data_hugo_char
popixa train --size nano --data_dir data_hugo_char --max_iters 1000 --batch_size 16 --run hugo-nano --eval
popixa runs                                   # compare les expériences (runs/*/run.json + eval.json)
```

## 2.2 « Mètre-étalon » — référence historique

### Évaluation (`popixa eval`)

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

> Modèles entraînés en 2.2, avec le dropout 0.1 aussi DANS l'attention. Depuis 2.3
> (`attn_dropout = 0` par défaut), la même commande donne d'autres poids (autre régularisation,
> autre tirage aléatoire) et va plus vite : ces chiffres restent la référence 2.2.

Entraînement (2.2) : ≈ 26 min sur CPU 4 cœurs (Xeon 2,1 GHz, 3 threads), courbe de `train.log` :

| itération | 0 | 200 | 400 | 600 | 800 | 1 000 |
|---|---|---|---|---|---|---|
| val loss | 4.79 | 2.48 | 2.27 | 2.13 | 2.08 | 2.03 |

## Débit (`popixa bench`, remesuré en 2.3)

Machine : conteneur cloud, Intel Xeon 2,1 GHz, 4 cœurs, sans GPU, PyTorch 2.14 CPU, 4 threads.
Remesuré en 2.3 (moyenne de 2 mesures ; le bruit de la machine atteint ± 10 %).

| Preset | Vocab | Batch | Entraînement 2.3 (dropout 0.1, `attn_dropout` 0) | Sans dropout | 2.2 : dropout 0.1 aussi dans l'attention | Génération (greedy) |
|---|---|---|---|---|---|---|
| `nano` (0.9M) | 115 | 16 × 512 | 18 600 tok/s | 22 400 tok/s | 5 370 tok/s | 670 tok/s |
| `small` (30M avec embeddings) | 50 257 | 4 × 1024 | 915 tok/s | 905 tok/s | 535 tok/s | 140 tok/s |

À retenir :
- Depuis 2.3, `attn_dropout = 0.0` par défaut (comme LLaMA) : le `dropout` de train.py (0.1) ne
  touche plus que les embeddings et les résiduels, et `scaled_dot_product_attention` garde son
  noyau rapide. Ce dropout coûte ≈ 17 % sur `nano`, rien de mesurable sur `small`.
- Historique 2.2 : un `dropout_p > 0` DANS l'attention fait retomber SDPA sur l'implémentation
  « math » lente (sur CPU, probablement aussi sur MPS), qui matérialise la matrice d'attention.
  Mesuré en 2.2 : entraînement ≈ 2,2× plus lent ; remesuré en 2.3 : ≈ 3,5× (`nano`) et ≈ 1,7×
  (`small`, où la tête LM de 50 257 tokens pèse davantage), avec un pic de mémoire (RSS)
  de 1,8 Go au lieu de 0,9 Go (`nano`) et de 6,9 Go au lieu de 5,0 Go (`small`).
  `POPIXAConfig(attn_dropout=0.1)` rétablit ce comportement (pas d'option dans `popixa bench`).
- Le preset `small` par défaut (16 × 1024 × 10 000 itérations ≈ 164M tokens) demanderait ≈ 50 h sur
  ce CPU (≈ 88 h en 2.2) : sur CPU, entraîner `nano` ; `small` est fait pour Apple Silicon (MPS) ou un GPU.
- Lance `popixa bench` sur TA machine pour recaler ces estimations
  (`popixa bench --size nano --vocab 115 --batch 16`, `popixa bench --size small` ; `--dropout 0` pour
  la colonne « Sans dropout »).
