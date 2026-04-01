"""
nanoPOPIXA — Script d'entraînement
Supporte : gradient clipping, cosine LR scheduling, données binaires pré-traitées
Usage :
    python train.py                         # texte brut (input.txt)
    python train.py --data_dir data/        # données pré-traitées par data_prep.py
"""

import os
import math
import time
import pickle
import argparse

import numpy as np
import torch

try:
    import tiktoken as _tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False

from model import nanoPOPIXA, POPIXAConfig


# ─────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",  default=None,        help="Dossier data/ créé par data_prep.py")
parser.add_argument("--input",     default="input.txt", help="Fichier texte brut (si pas de --data_dir)")
parser.add_argument("--resume",    action="store_true", help="Reprendre depuis le dernier checkpoint")
parser.add_argument("--size",      default="medium",    choices=["nano","small","medium"],
                    help="Taille du modèle : nano (~2M), small (~10M), medium (~85M)")
parser.add_argument("--longrope",  action="store_true",
                    help="LongRoPE : rope_base=500_000 → fenêtre contexte ~10× (beta context-1m)")
args = parser.parse_args()


# ─────────────────────────────────────────
# Hyperparamètres
# ─────────────────────────────────────────

out_dir = "out-nanopopixa"

# ── Presets de taille ─────────────────────────────────────────────────────────
_SIZES = {
    #          block  layer head  embd  batch  max_iters  lr      warmup
    "nano":   (512,   4,    4,    128,  32,    5_000,     3e-4,   100),   # RoPE → block_size 256→512
    "small":  (1024,  6,    6,    384,  16,    10_000,    3e-4,   200),   # RoPE → block_size 512→1024
    "medium": (1024,  12,   12,   768,  8,     10_000,    5e-4,   200),
}
(block_size, n_layer, n_head, n_embd,
 batch_size, max_iters, learning_rate, warmup_iters) = _SIZES[args.size]

dropout    = 0.1
min_lr     = learning_rate / 10
lr_decay_iters  = max_iters
grad_clip       = 1.0

# Évaluation — toutes les 100 iters, moyenne sur 10 batches
eval_interval   = 100
eval_iters      = 10

# Gradient accumulation — simule un batch effectif plus grand
gradient_accumulation_steps = 1   # augmenter si OOM (ex: 4 → batch effectif ×4)

# Détection de l'appareil (Ajout du support Mac MPS)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"🚀 Device détecté : {device.upper()}")

# Guard OOM — medium sur MPS dépasse facilement les 20 GB
# (85M params × batch 8 × block 1024 × bfloat16 ≈ 20+ GB activations)
if device == "mps" and args.size == "medium":
    print("⚠️  Attention : --size medium peut provoquer un OOM sur Apple Silicon (>20 GB MPS).")
    print("   Recommandation : utilise --size small (~10M params, ~3 GB) ou --size nano (~2M params).")
    print("   Tu peux aussi réduire le batch en éditant batch_size dans train.py.")
    print("   Pour forcer quand même : relance avec PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 popixa train ...")
    import sys as _sys
    _sys.exit(1)


# ─────────────────────────────────────────
# LR Scheduling — cosine avec warmup linéaire
# ─────────────────────────────────────────

def get_lr(it):
    # 1. Warmup linéaire
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2. Après la fenêtre de decay : LR minimum
    if it > lr_decay_iters:
        return min_lr
    # 3. Cosine decay entre warmup et lr_decay_iters
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ─────────────────────────────────────────
# Chargement des données
# ─────────────────────────────────────────

meta = {}   # initialisé vide — peuplé selon le mode de données

if args.data_dir and os.path.exists(os.path.join(args.data_dir, "train.bin")):
    # Données binaires pré-traitées (data_prep.py)
    meta_path = os.path.join(args.data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]
    
    # Gestion du tokenizer (tiktoken vs character-based)
    if meta.get("tokenizer") == "tiktoken_gpt2":
        if not _HAS_TIKTOKEN:
            raise ImportError("tiktoken requis : pip install tiktoken")
        enc    = _tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode_ordinary(s)
        decode = lambda l: enc.decode(l)
        print("Tokenizer BPE (tiktoken gpt2) chargé.")
    else:
        stoi   = meta["stoi"]
        itos   = meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])
        print("Tokenizer Caractère chargé.")

    train_data = np.fromfile(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16)
    val_data   = np.fromfile(os.path.join(args.data_dir, "val.bin"),   dtype=np.uint16)
    train_data = torch.from_numpy(train_data.astype(np.int64))
    val_data   = torch.from_numpy(val_data.astype(np.int64))
    print(f"Données binaires chargées depuis {args.data_dir}/")

else:
    # Mode texte brut (input.txt) — tokenisation caractère
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
    chars      = sorted(set(text))
    vocab_size = len(chars)
    stoi       = {c: i for i, c in enumerate(chars)}
    itos       = {i: c for i, c in enumerate(chars)}
    encode     = lambda s: [stoi[c] for c in s]
    decode     = lambda l: "".join([itos[i] for i in l])
    data       = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n          = int(0.9 * len(data))
    train_data = data[:n]
    val_data   = data[n:]
    meta       = {"tokenizer": "char", "stoi": stoi, "itos": itos}
    print(f"Texte brut chargé : {len(text):,} caractères | vocab {vocab_size}")

print(f"Vocabulaire : {vocab_size} tokens | Train : {len(train_data):,} | Val : {len(val_data):,}")


def get_batch(split):
    d  = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i     : i + block_size    ] for i in ix])
    y  = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    model.train(False)
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train(True)
    return out


# ─────────────────────────────────────────
# Initialisation du modèle
# ─────────────────────────────────────────

rope_base = 500_000 if args.longrope else 10_000
if args.longrope:
    print("🔭 LongRoPE activé — rope_base=500_000 (contexte ~10× étendu)")

config = POPIXAConfig(
    block_size=block_size,
    vocab_size=vocab_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    rope_base=rope_base,
)

model     = nanoPOPIXA(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

os.makedirs(out_dir, exist_ok=True)

# ── Reprise depuis checkpoint ──────────────────────────────────────────────────
iter_start = 0
if args.resume:
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        iter_start = ckpt.get("iter", 0) + 1
        print(f"✅ Reprise depuis iter {iter_start}")
    else:
        print("⚠️  Aucun checkpoint trouvé — démarrage depuis 0")


# ─────────────────────────────────────────
# Boucle d'entraînement
# ─────────────────────────────────────────

print(f"\n🚀 Démarrage entraînement nanoPOPIXA [{args.size}]...\n")
t0     = time.time()
t_last = t0

# Log propre à chaque run (sauf reprise)
if not args.resume:
    with open("train.log", "w", encoding="utf-8") as f:
        f.write(f"# max_iters={max_iters} eval_interval={eval_interval}"
                f" batch_size={batch_size} block_size={block_size}\n")

for iter in range(iter_start, max_iters):

    # LR scheduling — cosine avec warmup
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Évaluation périodique
    if iter % eval_interval == 0:
        losses  = estimate_loss(model)
        now     = time.time()
        dt      = now - t_last   # temps écoulé depuis la dernière éval
        t_last  = now
        log_line = (
            f"iter {iter:5d} | "
            f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
            f"lr {lr:.2e} | {dt:.1f}s"
        )
        # Barre de progression ASCII
        pct    = iter / max_iters
        filled = int(30 * pct)
        bar    = "█" * filled + "░" * (30 - filled)
        print(f"[{bar}] {pct:5.1%}  {log_line}")
        with open("train.log", "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
        
        # Sauvegarde checkpoint (+ optimizer pour --resume)
        checkpoint = {
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config":    config,
            "iter":      iter,
            "tokenizer": meta.get("tokenizer", "char"),
        }
        if meta.get("tokenizer") != "tiktoken_gpt2":
            checkpoint["vocab"] = {"stoi": stoi, "itos": itos}
        torch.save(checkpoint, os.path.join(out_dir, "checkpoint.pt"))

    # ── Forward + backward avec gradient accumulation ─────────────────────────
    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(gradient_accumulation_steps):
        X, Y        = get_batch("train")
        _, loss     = model(X, Y)
        loss        = loss / gradient_accumulation_steps
        loss.backward()

    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

print(f"\n✅ Entraînement terminé en {time.time() - t0:.1f}s")


# ─────────────────────────────────────────
# Génération rapide post-entraînement
# ─────────────────────────────────────────

print("\n📝 Exemple de génération :\n")
model.train(False)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
output  = decode(model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40, top_p=0.9)[0].tolist())
print(output)
