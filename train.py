"""
nanoPOPIXA — Script d'entraînement
Supporte : gradient clipping, cosine LR scheduling, données binaires pré-traitées
Usage :
    python train.py                         # texte brut (input.txt)
    python train.py --data_dir data/        # données pré-traitées par data_prep.py
"""

import os
import sys
import math
import time
import pickle
import random
import argparse

import numpy as np
import torch

try:
    import tiktoken as _tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False

from model import nanoPOPIXA, POPIXAConfig, SIZE_PRESETS, checkpoint_v1_error


# ─────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",  default=None,        help="Dossier data/ créé par data_prep.py")
parser.add_argument("--input",     default="input.txt", help="Fichier texte brut (si pas de --data_dir)")
parser.add_argument("--resume",    action="store_true", help="Reprendre depuis le dernier checkpoint")
parser.add_argument("--size",      default="small",     choices=["nano","small","medium"],
                    help="Taille du modèle (params hors embeddings) : nano (~0.9M), small (~10M), medium (~85M)")
parser.add_argument("--longrope",  action="store_true",
                    help="LongRoPE : rope_base=500_000 — rotations plus lentes, meilleure base pour "
                         "étendre le contexte plus tard (la fenêtre reste block_size)")
parser.add_argument("--max_iters",  type=int, default=None, help="Surcharge du nb d'itérations du preset")
parser.add_argument("--batch_size", type=int, default=None, help="Surcharge de la taille de batch du preset")
parser.add_argument("--seed",       type=int, default=1337,
                    help="Graine : initialisation, dropout et tirage des batchs reproductibles")
args = parser.parse_args()

# ── Reproductibilité ──────────────────────────────────────────────────────────
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# ─────────────────────────────────────────
# Hyperparamètres
# ─────────────────────────────────────────

out_dir = "out-nanopopixa"

# ── Presets de taille ─────────────────────────────────────────────────────────
# Architecture : model.SIZE_PRESETS (partagée avec popixa bench) ; ici l'entraînement
_TRAIN_PRESETS = {
    #          batch  max_iters  lr      warmup
    "nano":   (32,    5_000,     3e-4,   100),
    "small":  (16,    10_000,    3e-4,   200),
    "medium": (8,     10_000,    5e-4,   200),
}
_arch = SIZE_PRESETS[args.size]
block_size, n_layer, n_head, n_embd = (_arch["block_size"], _arch["n_layer"],
                                       _arch["n_head"], _arch["n_embd"])
batch_size, max_iters, learning_rate, warmup_iters = _TRAIN_PRESETS[args.size]
if args.max_iters is not None:
    max_iters    = max(1, args.max_iters)
    warmup_iters = min(warmup_iters, max(1, max_iters // 10))
if args.batch_size is not None:
    batch_size = max(1, args.batch_size)

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
if device == "mps" and args.size == "medium" and "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
    print("⚠️  Attention : --size medium peut provoquer un OOM sur Apple Silicon (>20 GB MPS).")
    print("   Recommandation : utilise --size small (~10M params, ~3 GB) ou --size nano (~0.9M params).")
    print("   Tu peux aussi réduire le batch en éditant batch_size dans train.py.")
    print("   Pour forcer quand même : relance avec PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 popixa train ...")
    import sys as _sys
    _sys.exit(1)


rope_base = 500_000 if args.longrope else 10_000

# ── Reprise : l'architecture vient du checkpoint, pas de --size ───────────────
# (sinon load_state_dict plante si --size diffère, ou les tables RoPE du checkpoint
#  écrasent silencieusement un --longrope différent)
ckpt_path   = os.path.join(out_dir, "checkpoint.pt")
resume_ckpt = None
if args.resume:
    if os.path.exists(ckpt_path):
        resume_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        v1_error = checkpoint_v1_error(resume_ckpt.get("model", {}))
        if v1_error:
            print(f"❌ {v1_error}")
            sys.exit(1)
        ck = resume_ckpt["config"]
        ck_rope = getattr(ck, "rope_base", 10_000)
        if (ck.block_size, ck.n_layer, ck.n_head, ck.n_embd) != (block_size, n_layer, n_head, n_embd):
            print(f"⚠️  Architecture du checkpoint (block {ck.block_size}, {ck.n_layer} couches, "
                  f"{ck.n_head} têtes, embd {ck.n_embd}) ≠ --size {args.size} → on garde le checkpoint")
        if ck_rope != rope_base:
            print(f"⚠️  rope_base du checkpoint ({ck_rope}) ≠ demandé ({rope_base}) → on garde le checkpoint")
        block_size, n_layer, n_head, n_embd = ck.block_size, ck.n_layer, ck.n_head, ck.n_embd
        rope_base = ck_rope
        # Même source de données que l'entraînement initial si --data_dir est omis
        if args.data_dir is None and resume_ckpt.get("data_dir"):
            args.data_dir = resume_ckpt["data_dir"]
            print(f"📂 Données du checkpoint : {args.data_dir}")
    else:
        print("⚠️  Aucun checkpoint trouvé — démarrage depuis 0")


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
    if lr_decay_iters <= warmup_iters:
        return learning_rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ─────────────────────────────────────────
# Chargement des données
# ─────────────────────────────────────────

meta = {}   # initialisé vide — peuplé selon le mode de données

if args.data_dir:
    _missing = [f for f in ("train.bin", "val.bin", "meta.pkl")
                if not os.path.exists(os.path.join(args.data_dir, f))]
    if _missing:
        print(f"❌ {args.data_dir.rstrip('/')}/ incomplet — fichier(s) manquant(s) : {', '.join(_missing)}")
        print("   Prépare d'abord les données : popixa prep --dataset shakespeare --data_dir "
              f"{args.data_dir}")
        sys.exit(1)

if args.data_dir:
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
    if not os.path.exists(args.input):
        print(f"❌ {args.input} introuvable — utilise --data_dir data/ (après popixa prep) "
              "ou fournis un fichier texte avec --input")
        sys.exit(1)
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


for _name, _d in (("train", train_data), ("val", val_data)):
    if len(_d) <= block_size:
        print(f"❌ Split {_name} trop court ({len(_d)} tokens) pour block_size={block_size} "
              f"— utilise plus de données ou un --size plus petit")
        sys.exit(1)


# Générateur dédié au tirage des batchs : indépendant des autres usages de l'aléatoire
# (dropout, init) → mêmes batchs pour une même graine. Reprise : graine + iter de départ.
_batch_gen = torch.Generator().manual_seed(args.seed)


def get_batch(split):
    d  = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,), generator=_batch_gen)
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

if rope_base != 10_000:
    print(f"🔭 LongRoPE — rope_base={rope_base:,} (fenêtre de contexte : {block_size} tokens)")

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
# "iter" = nombre d'itérations d'entraînement DÉJÀ effectuées au moment de la sauvegarde
iter_start = 0
resume_ckpt_loaded = resume_ckpt is not None
if resume_ckpt is not None:
    if resume_ckpt["config"].vocab_size != vocab_size:
        print(f"❌ Vocabulaire du checkpoint ({resume_ckpt['config'].vocab_size}) ≠ données "
              f"({vocab_size}) — reprise impossible sur un autre tokenizer/dataset")
        sys.exit(1)
    model.load_state_dict(resume_ckpt["model"])
    if "optimizer" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer"])
    iter_start = resume_ckpt.get("iter", 0)
    _batch_gen.manual_seed(args.seed + iter_start)
    if iter_start >= max_iters:
        print(f"✅ Entraînement déjà terminé ({iter_start}/{max_iters} itérations) — rien à faire")
    else:
        print(f"✅ Reprise depuis iter {iter_start}")
    del resume_ckpt
elif os.path.exists(ckpt_path):
    # Nouvel entraînement : on ne détruit jamais silencieusement un modèle existant
    import shutil
    backup = os.path.join(out_dir, "checkpoint.prev.pt")
    shutil.copy2(ckpt_path, backup)
    print(f"💾 Checkpoint existant sauvegardé → {backup}  (--resume pour reprendre au lieu de repartir de 0)")


def save_checkpoint(it: int) -> None:
    """Sauvegarde modèle + optimizer (pour --resume). it = itérations effectuées."""
    checkpoint = {
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config":    config,
        "iter":      it,
        "tokenizer": meta.get("tokenizer", "char"),
        "data_dir":  args.data_dir,
    }
    if meta.get("tokenizer") != "tiktoken_gpt2":
        checkpoint["vocab"] = {"stoi": stoi, "itos": itos}
    # Écriture atomique : un Ctrl+C pendant la sauvegarde ne corrompt pas le checkpoint
    tmp_path = ckpt_path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, ckpt_path)


# ─────────────────────────────────────────
# Boucle d'entraînement
# ─────────────────────────────────────────

print(f"\n🚀 Démarrage entraînement nanoPOPIXA [{args.size}]...\n")
t0     = time.time()
t_last = t0

# Log propre à chaque run ; en reprise on AJOUTE un en-tête (le monitor garde le dernier :
# max_iters / batch peuvent avoir changé)
with open("train.log", "a" if resume_ckpt_loaded else "w", encoding="utf-8") as f:
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
        
        # Sauvegarde checkpoint (+ optimizer pour --resume) — inutile avant le 1er pas
        if iter > iter_start:
            save_checkpoint(iter)

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

# Évaluation + sauvegarde finales — sinon les itérations après la dernière évaluation
# seraient perdues (rien à faire si aucune itération n'a tourné : le checkpoint, et donc
# les sessions de chat, restent valides)
if iter_start < max_iters:
    losses   = estimate_loss(model)
    log_line = (f"iter {max_iters:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"lr {get_lr(max_iters):.2e} | {time.time() - t_last:.1f}s")
    print(f"[{'█' * 30}] 100.0%  {log_line}")
    with open("train.log", "a", encoding="utf-8") as f:
        f.write(log_line + "\n")
    save_checkpoint(max_iters)
print(f"\n✅ Entraînement terminé en {time.time() - t0:.1f}s — checkpoint : {ckpt_path}")


# ─────────────────────────────────────────
# Génération rapide post-entraînement
# ─────────────────────────────────────────

print("\n📝 Exemple de génération :\n")
model.train(False)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
output  = decode(model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40, top_p=0.9)[0].tolist())
print(output)
