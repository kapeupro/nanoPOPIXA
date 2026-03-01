# nanoPOPIXA

> Un LLM minimaliste from scratch — inspiré de [nanoGPT](https://github.com/karpathy/nanoGPT), personnalisé et étendu par Dimitri.

---

## Qu'est-ce que c'est ?

nanoPOPIXA est un Transformer GPT-style entraînable sur n'importe quel texte, entièrement en Python/PyTorch. Le code est volontairement court et lisible : chaque fichier fait une chose, chaque paramètre est expliqué.

**Conçu pour apprendre** : tu peux lire l'intégralité du code source en une heure et comprendre comment un LLM fonctionne de A à Z.

---

## Architecture

```
Input tokens
     │
     ▼
Token Embedding (wte) + Position Embedding (wpe)
     │
     ▼
 ┌─────────────────────────────────┐
 │  Transformer Block  x  N       │
 │  ┌──────────────────────────┐  │
 │  │  LayerNorm               │  │
 │  │  CausalSelfAttention     │  │  <- Flash Attention (PyTorch 2.0+)
 │  │  + residual              │  │
 │  ├──────────────────────────┤  │
 │  │  LayerNorm               │  │
 │  │  MLP (4x expansion GELU) │  │
 │  │  + residual              │  │
 │  └──────────────────────────┘  │
 └─────────────────────────────────┘
     │
     ▼
LayerNorm final
     │
     ▼
LM Head (Linear, weight tying avec wte)
     │
     ▼
Logits -> softmax -> token suivant
```

| Taille  | Params | Blocs | Tetes | Embd | Batch |
|---------|--------|-------|-------|------|-------|
| nano    | ~2M    | 4     | 4     | 128  | 32    |
| small   | ~10M   | 6     | 6     | 384  | 16    |
| medium  | ~85M   | 12    | 12    | 768  | 8     |

---

## Installation

```bash
git clone https://github.com/ton-user/nanopopixa
cd nanopopixa
pip install -e .
```

Dependances : `torch >= 2.0`, `numpy`, `tiktoken`

---

## Demarrage rapide

```bash
# 1. Preparer un dataset
popixa prep --dataset shakespeare

# 2. Entrainer
popixa train --data_dir data/ --size small

# 3. Discuter avec le modele
popixa chat
```

---

## CLI complete

```
popixa                     -> shell interactif
popixa chat                -> chat avec le modele entraine
popixa train               -> lancer l'entrainement
popixa prep                -> telecharger et preparer un dataset
popixa gen                 -> generer du texte (non-interactif)
popixa monitor             -> dashboard live de la loss
popixa scrape              -> crawler web -> corpus
popixa collect SOURCE_DIR  -> assembler du code source en corpus
```

### Options utiles

```bash
# Entrainement
popixa train --size nano --data_dir data/          # modele leger, rapide
popixa train --size medium --resume                # reprendre depuis checkpoint

# Chat
popixa chat --temp 0.7 --tokens 300 --penalty 1.3

# Generation
popixa gen --prompt "Il etait une fois" --tokens 500 --temp 0.9

# Donnees
popixa prep --dataset hugo              # Victor Hugo (Gutenberg)
popixa prep --dataset javascript        # code JS (lodash, jquery, vue, react)
popixa prep --char                      # tokenisation caractere au lieu de BPE
```

### Commandes in-chat

```
/temp 0.7      -> changer la temperature
/tokens 400    -> nb de tokens a generer
/penalty 1.3   -> repetition penalty (1.0 = desactive)
/reset         -> remettre le contexte a zero
/libre         -> generation libre sans prompt
/save conv.txt -> sauvegarder la conversation
```

---

## Datasets disponibles

| Nom           | Description                          |
|---------------|--------------------------------------|
| `shakespeare` | Oeuvres completes (defaut)           |
| `hugo`        | Victor Hugo -- Les Miserables        |
| `moliere`     | Moliere -- pieces completes          |
| `bible`       | Bible (King James Version)           |
| `linux`       | Code source noyau Linux              |
| `javascript`  | lodash + jquery + vue + react        |

---

## Fonctionnalites

- **Flash Attention** (PyTorch 2.0+) -- `O(N)` memoire
- **Weight tying** -- embedding et lm_head partagent les memes poids
- **Cosine LR scheduling** avec warmup lineaire
- **Gradient clipping** (1.0)
- **Gradient accumulation** -- simuler de grands batches
- **Checkpoint resume** -- reprendre un entrainement interrompu
- **Repetition penalty** -- eviter les boucles de tokens
- **Streaming token par token** en chat
- **Tokenisation BPE** (tiktoken gpt2, 50257 tokens) ou **caractere**
- **Monitor live** -- courbe de loss en temps reel dans le terminal

---

## Structure du projet

```
nanopopixa/
├── model.py       # Transformer GPT-style (CausalSelfAttention, MLP, Block)
├── train.py       # Boucle d'entrainement + LR scheduling
├── chat.py        # REPL interactif streaming
├── data_prep.py   # Telechargement et tokenisation des datasets
├── monitor.py     # Dashboard terminal (courbe loss en braille)
├── scrape.py      # Crawler web -> corpus texte
├── splash.py      # Globe 3D + logo anime
├── popixa_cli.py  # Point d'entree CLI unifie
└── setup.py       # Package installable
```

---

## Inspirations

- [nanoGPT](https://github.com/karpathy/nanoGPT) -- Andrej Karpathy
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) -- Vaswani et al.
- [GPT-2](https://openai.com/research/better-language-models) -- OpenAI

---

## Licence

MIT -- voir [LICENSE](LICENSE)
# nanoPOPIXA
