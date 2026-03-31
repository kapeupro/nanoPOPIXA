# nanoPOPIXA

> Un LLM minimaliste from scratch — inspiré de [nanoGPT](https://github.com/karpathy/nanoGPT), personnalisé et étendu par Dimitri.
> Architecture v2 : **RMSNorm · SwiGLU · RoPE · KV-Cache · Thinking blocks · Nucleus sampling**

---

## Qu'est-ce que c'est ?

nanoPOPIXA est un Transformer GPT-style entraînable sur n'importe quel texte, entièrement en Python/PyTorch. Le code est volontairement court et lisible : chaque fichier fait une chose, chaque paramètre est expliqué.

**Conçu pour apprendre** : tu peux lire l'intégralité du code source en une heure et comprendre comment un LLM fonctionne de A à Z.

---

## Architecture v2

```
Input tokens
     │
     ▼
Token Embedding (wte)          ← pas de wpe : RoPE encode les positions
     │
     ▼
 ┌─────────────────────────────────────────┐
 │  Transformer Block  x  N               │
 │  ┌──────────────────────────────────┐  │
 │  │  RMSNorm                         │  │  ← plus rapide que LayerNorm
 │  │  CausalSelfAttention + RoPE      │  │  ← Flash Attention + KV-Cache
 │  │  + residual                      │  │
 │  ├──────────────────────────────────┤  │
 │  │  RMSNorm                         │  │
 │  │  SwiGLU  (8/3x expansion)        │  │  ← gate mechanism (Claude, LLaMA)
 │  │  + residual                      │  │
 │  └──────────────────────────────────┘  │
 └─────────────────────────────────────────┘
     │
     ▼
RMSNorm final
     │
     ▼
LM Head (weight tying avec wte)
     │
     ▼
Logits → top-k / top-p / temperature → token suivant
```

### Comparaison v1 → v2

| Composant | v1 (nanoGPT-style) | v2 (Claude-inspired) | Gain |
|---|---|---|---|
| Positional encoding | `wpe` appris | **RoPE** (0 param) | Meilleure généralisation longueur |
| Normalisation | `LayerNorm` avec biais | **RMSNorm** sans biais | Plus rapide |
| Activation MLP | `GELU` (4×) | **SwiGLU** (8/3×) | Meilleur gradient flow |
| Inférence | Recalcul complet O(T²) | **KV-Cache** O(1)/token | 2-5× plus rapide |
| Sampling | top-k uniquement | top-k + **top-p nucleus** | Meilleure qualité |
| Génération | Simple | **Thinking blocks** | Raisonnement interne |

### Presets de taille

| Taille  | Params | Blocs | Têtes | Embd | Block size |
|---------|--------|-------|-------|------|------------|
| nano    | ~2M    | 4     | 4     | 128  | 512        |
| small   | ~10M   | 6     | 6     | 384  | 1024       |
| medium  | ~85M   | 12    | 12    | 768  | 1024       |

---

## Installation

```bash
git clone https://github.com/ton-user/nanopopixa
cd nanopopixa
pip install -e .
```

Dépendances : `torch >= 2.0`, `numpy`, `tiktoken`

---

## Démarrage rapide

```bash
# 1. Préparer un dataset
popixa prep --dataset shakespeare

# 2. Entraîner
popixa train --data_dir data/ --size small

# 3. Discuter avec le modèle
popixa chat
```

---

## CLI complète

```
popixa                     → shell interactif
popixa chat                → chat avec le modèle entraîné
popixa train               → lancer l'entraînement
popixa prep                → télécharger et préparer un dataset
popixa gen                 → générer du texte (non-interactif)
popixa monitor             → dashboard live de la loss
popixa scrape              → crawler web → corpus
popixa collect SOURCE_DIR  → assembler du code source en corpus
```

### Options entraînement

```bash
popixa train --size nano --data_dir data/       # modèle léger, rapide
popixa train --size medium --resume             # reprendre depuis checkpoint
```

### Options chat

```bash
popixa chat --effort medium                     # preset tout-en-un
popixa chat --temp 0.7 --tokens 300 --top_p 0.9
popixa chat --top_p 0.95 --penalty 1.3
```

### Commandes in-chat

#### Paramètres de génération

```
/temp 0.7          → changer la température
/tokens 400        → nb de tokens à générer
/topp 0.9          → nucleus sampling (top-p)
/penalty 1.3       → repetition penalty (1.0 = désactivé)
```

#### Modes (nouveauté v2)

```
/think             → activer/désactiver le mode Thinking blocks
                     Phase 1 : raisonnement interne  (temperature=1, affiché en ocre)
                     Phase 2 : réponse finale        (temperature normale, en violet)

/thinkbudget 200   → tokens alloués à la phase de réflexion (défaut : 150)

/effort low        → preset : temp=1.0  top_k=20  top_p=0.85  tokens=100
/effort medium     → preset : temp=0.8  top_k=40  top_p=0.90  tokens=200
/effort high       → preset : temp=0.7  top_k=50  top_p=0.95  tokens=400
/effort max        → preset : temp=1.0  top_k=off top_p=0.95  tokens=600
```

#### Contexte & outils

```
/ctx               → afficher l'état du contexte (barre de progression)
/reset             → remettre le contexte à zéro
/libre             → génération libre sans prompt
/save conv.txt     → sauvegarder la conversation
```

---

## Fonctionnalités

### Architecture modèle
- **RMSNorm** — normalisation sans biais, plus rapide que LayerNorm
- **RoPE** (Rotary Position Embeddings) — encodage rotatif, 0 paramètre supplémentaire
- **SwiGLU** — activation gate (Claude, LLaMA, PaLM) avec meilleur gradient flow
- **Flash Attention** (PyTorch 2.0+) — O(N) mémoire
- **KV-Cache** — décodage incrémental O(1) par token en inférence
- **Weight tying** — embedding et lm_head partagent les mêmes poids

### Sampling & génération
- **Nucleus sampling (top-p)** — diversité mieux contrôlée que top-k seul
- **Repetition penalty** — éviter les boucles de tokens
- **Thinking blocks** — raisonnement interne deux phases (temperature=1 puis normale)
- **Diminishing returns** — arrêt automatique si la génération tourne en rond
- **Effort levels** — presets low/medium/high/max ajustant tous les paramètres

### Entraînement
- **Cosine LR scheduling** avec warmup linéaire
- **Gradient clipping** (1.0)
- **Gradient accumulation** — simuler de grands batches
- **Checkpoint resume** — reprendre un entraînement interrompu
- **Streaming token par token** en chat

### Gestion du contexte
- **Auto-compact** — compaction automatique à 90% de la fenêtre (garde 50% récent)
- **Warning visuel** à 80% avec barre de progression colorée
- **Indicateur contexte** dans le prompt `[ctx 83%]`
- **Tokenisation BPE** (tiktoken gpt2, 50 257 tokens) ou caractère

---

## Datasets disponibles

| Nom           | Description                          |
|---------------|--------------------------------------|
| `shakespeare` | Œuvres complètes (défaut)            |
| `hugo`        | Victor Hugo — Les Misérables         |
| `moliere`     | Molière — pièces complètes           |
| `bible`       | Bible (King James Version)           |
| `linux`       | Code source noyau Linux              |
| `javascript`  | lodash + jquery + vue + react        |

---

## Structure du projet

```
nanopopixa/
├── model.py       # Transformer v2 : RMSNorm · SwiGLU · RoPE · KV-Cache · Thinking
├── train.py       # Boucle d'entraînement + LR scheduling
├── chat.py        # REPL interactif : thinking blocks · effort levels · auto-compact
├── data_prep.py   # Téléchargement et tokenisation des datasets
├── monitor.py     # Dashboard terminal (courbe loss en braille)
├── scrape.py      # Crawler web → corpus texte
├── splash.py      # Globe 3D + logo animé
├── popixa_cli.py  # Point d'entrée CLI unifié
└── setup.py       # Package installable
```

---

## Inspirations & reverse engineering

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy (base)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- [RoFormer](https://arxiv.org/abs/2104.09864) — Su et al. (RoPE)
- [LLaMA](https://arxiv.org/abs/2302.13971) — Meta AI (RMSNorm + SwiGLU)
- [GLU Variants](https://arxiv.org/abs/2002.05202) — Noam Shazeer (SwiGLU)
- Claude Code source — reverse engineering des mécanismes thinking, effort, token budget, auto-compact

---

## Licence

MIT — voir [LICENSE](LICENSE)
