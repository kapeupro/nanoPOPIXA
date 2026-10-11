"""
nanoPOPIXA — Script d'entraînement
Supporte : gradient clipping, cosine LR scheduling avec warmup, données binaires lues sans tout
charger en mémoire (memmap), précision mixte (--dtype), weight decay sur les seules matrices,
meilleur checkpoint (best.pt) et arrêt anticipé (--patience), un dossier par expérience (--run).
Usage :
    python train.py                         # texte brut (input.txt)
    python train.py --data_dir data/        # données pré-traitées par data_prep.py
    python train.py --data_dir data/ --run hugo-nano --size nano --eval
"""

import os
import sys
import json
import math
import time
import pickle
import random
import argparse
import contextlib

import numpy as np
import torch

try:
    import tiktoken as _tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False

from model import nanoPOPIXA, POPIXAConfig, SIZE_PRESETS, checkpoint_v1_error
from checkpoint import load_checkpoint, save_checkpoint as _write_checkpoint


# ─────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",  default=None,        help="Dossier data/ créé par data_prep.py")
parser.add_argument("--input",     default="input.txt", help="Fichier texte brut (si pas de --data_dir)")
parser.add_argument("--resume",    action="store_true", help="Reprendre depuis le dernier checkpoint")
parser.add_argument("--size",      default=None,        choices=["nano","small","medium"],
                    help="Taille du modèle (params hors embeddings) : nano (~0.9M), small (~10M, défaut), "
                         "medium (~85M) — en reprise, celle du checkpoint par défaut")
parser.add_argument("--longrope",  action="store_true",
                    help="LongRoPE : rope_base=500_000 — rotations plus lentes, meilleure base pour "
                         "étendre le contexte plus tard (la fenêtre reste block_size)")
parser.add_argument("--max_iters",  type=int, default=None, help="Surcharge du nb d'itérations du preset")
parser.add_argument("--batch_size", type=int, default=None, help="Surcharge de la taille de batch du preset")
parser.add_argument("--seed",       type=int, default=None,
                    help="Graine : initialisation, dropout et tirage des batchs reproductibles "
                         "(défaut 1337 ; en reprise, celle du run)")
parser.add_argument("--out_dir",    default="out-nanopopixa",
                    help="Dossier de l'expérience : checkpoint.pt, best.pt, train.log, run.json")
parser.add_argument("--run",        default=None,
                    help="Nom d'expérience : équivaut à --out_dir runs/NOM")
parser.add_argument("--patience",   type=int, default=None,
                    help="Arrêt anticipé après N évaluations sans amélioration de la val loss "
                         "(défaut : désactivé ; en reprise, celle du run)")
parser.add_argument("--dtype",      default="auto", choices=["auto", "float32", "bfloat16", "float16"],
                    help="Précision : auto = bfloat16 (ou float16) sur GPU CUDA, float32 ailleurs ; "
                         "bfloat16 possible sur Mac (MPS) et CPU — à mesurer avec popixa bench --dtype")
parser.add_argument("--grad_accum",    type=int, default=None,
                    help="Accumulation de gradient : batch effectif = batch_size × N, mémoire d'un seul batch")
parser.add_argument("--eval_interval", type=int, default=None,
                    help="Évaluation (et sauvegarde) toutes les N itérations (défaut 100)")
parser.add_argument("--eval_iters",    type=int, default=None, help="Batchs par évaluation de la loss (défaut 10)")
# Réglages experts (expériences, ablations) — par défaut ceux du preset ; en reprise, ceux du run
parser.add_argument("--lr",           type=float, default=None, help="Learning rate maximal (défaut : preset)")
parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay des matrices (défaut 0.1)")
parser.add_argument("--beta2",        type=float, default=None, help="beta2 d'AdamW (défaut 0.999)")
parser.add_argument("--dropout",      type=float, default=None,
                    help="Dropout embeddings / résiduels / MLP (défaut 0.1)")
parser.add_argument("--attn_dropout", type=float, default=None,
                    help="Dropout DANS l'attention (défaut 0 : > 0 rend SDPA lent sur CPU / Mac)")
parser.add_argument("--eval",       action="store_true",
                    help="En fin d'entraînement : popixa eval sur best.pt → eval.json + samples.md du run")
args = parser.parse_args()
if args.run:
    if os.path.basename(os.path.normpath(args.run)) != args.run or args.run in (".", ".."):
        parser.error(f"--run : un simple nom attendu (reçu {args.run!r})")
    args.out_dir = os.path.join("runs", args.run)
if args.patience is not None and args.patience < 0:
    parser.error("--patience doit être ≥ 0 (0 = désactivé)")
if any(v is not None and v < 1 for v in (args.eval_interval, args.eval_iters)):
    parser.error("--eval_interval et --eval_iters doivent être ≥ 1")
if args.grad_accum is not None and args.grad_accum < 1:
    parser.error("--grad_accum doit être ≥ 1")
for _opt in ("dropout", "attn_dropout"):
    if getattr(args, _opt) is not None and not 0.0 <= getattr(args, _opt) < 1.0:
        parser.error(f"--{_opt} doit être dans [0, 1)")
if args.beta2 is not None and not 0.0 < args.beta2 < 1.0:
    parser.error("--beta2 doit être dans ]0, 1[")
if (args.lr is not None and args.lr <= 0) or (args.weight_decay is not None and args.weight_decay < 0):
    parser.error("--lr doit être > 0 et --weight_decay ≥ 0")

# ─────────────────────────────────────────
# Hyperparamètres
# ─────────────────────────────────────────

out_dir   = args.out_dir
ckpt_path = os.path.join(out_dir, "checkpoint.pt")      # dernier état (reprise)
best_path = os.path.join(out_dir, "best.pt")            # meilleure val loss (chat / eval)
log_path  = os.path.join(out_dir, "train.log")
run_path  = os.path.join(out_dir, "run.json")

# ── Lancements transactionnels ────────────────────────────────────────────────
# Un NOUVEL entraînement dans un dossier occupé met d'abord tout le run précédent de côté
# (prev.new/) ; à sa 1re sauvegarde, prev.new/ devient prev/ (engagé). Interrompu ou planté
# avant, le run précédent est remis en place à l'identique (même après un arrêt brutal :
# rétabli au lancement suivant).
RUN_FILES   = ("checkpoint.pt", "best.pt", "train.log", "run.json", "eval.json", "samples.md")
prev_dir    = os.path.join(out_dir, "prev")
staging_dir = os.path.join(out_dir, "prev.new")


def _rollback_launch() -> bool:
    """Remet en place le run mis de côté dans prev.new/ ; True s'il y en avait un."""
    if not os.path.isdir(staging_dir):
        return False
    for name in RUN_FILES:
        cur = os.path.join(out_dir, name)
        if os.path.exists(cur):
            os.remove(cur)                 # fichiers du lancement avorté
    for name in os.listdir(staging_dir):
        os.replace(os.path.join(staging_dir, name), os.path.join(out_dir, name))
    os.rmdir(staging_dir)
    return True


if _rollback_launch():
    print(f"♻️  Lancement précédent arrêté avant sa 1re sauvegarde : run d'avant restauré dans {out_dir}/")

resume_ckpt = None
if args.resume:
    if os.path.exists(ckpt_path):
        try:
            resume_ckpt = load_checkpoint(ckpt_path, map_location="cpu")
        except ValueError as e:
            print(f"❌ {e}")
            sys.exit(1)
        v1_error = checkpoint_v1_error(resume_ckpt.get("model", {}))
        if v1_error:
            print(f"❌ {v1_error}")
            sys.exit(1)
    else:
        print("⚠️  Aucun checkpoint trouvé — démarrage depuis 0")

# Identité du run (graine, batchs d'évaluation) : --option, sinon celle du run repris, sinon défaut.
# Reprendre avec une autre graine ou d'autres batchs de validation rendrait les val loss (best.pt,
# arrêt anticipé) incomparables avec celles déjà mesurées.
_run_saved    = (resume_ckpt.get("train") or {}) if resume_ckpt is not None else {}
seed          = args.seed if args.seed is not None else _run_saved.get("seed", 1337)
eval_interval = args.eval_interval or _run_saved.get("eval_interval", 100)
eval_iters    = args.eval_iters or _run_saved.get("eval_iters", 10)

# ── Reproductibilité ──────────────────────────────────────────────────────────
random.seed(seed)
np.random.seed(seed % 2**32)             # numpy n'accepte que [0, 2**32)
torch.manual_seed(seed)


def _size_of(n_layer_: int, n_head_: int, n_embd_: int):
    """Nom du preset correspondant à une architecture (None si aucun)."""
    for name, a in SIZE_PRESETS.items():
        if (a["n_layer"], a["n_head"], a["n_embd"]) == (n_layer_, n_head_, n_embd_):
            return name
    return None


# Réglages d'entraînement enregistrés par le dernier lancement (checkpoints ≥ 2.2)
_saved = (resume_ckpt.get("train") or {}) if resume_ckpt is not None else {}
if _saved.get("size") not in SIZE_PRESETS:
    _saved = {}

# Taille effective : --size, sinon celle enregistrée, sinon celle de l'architecture du checkpoint
# (un run medium lancé avec l'ancien défaut garde ses hyperparamètres medium), sinon small
size = args.size
if size is None:
    if resume_ckpt is not None:
        _ck = resume_ckpt["config"]
        size = _saved.get("size") or _size_of(_ck.n_layer, _ck.n_head, _ck.n_embd) or "small"
    else:
        size = "small"

# ── Presets de taille ─────────────────────────────────────────────────────────
# Architecture : model.SIZE_PRESETS (partagée avec popixa bench) ; ici l'entraînement
_TRAIN_PRESETS = {
    #          batch  max_iters  lr      warmup
    "nano":   (32,    5_000,     3e-4,   100),
    "small":  (16,    10_000,    3e-4,   200),
    "medium": (8,     10_000,    5e-4,   200),
}
_arch = SIZE_PRESETS[size]
block_size, n_layer, n_head, n_embd = (_arch["block_size"], _arch["n_layer"],
                                       _arch["n_head"], _arch["n_embd"])
batch_size, max_iters, learning_rate, warmup_iters = _TRAIN_PRESETS[size]
# Reprise sans --size (ou avec la taille enregistrée) : réglages du dernier lancement, ses
# --max_iters / --batch_size compris, plutôt que ceux du preset. Un --size différent applique
# le preset de cette taille ; checkpoints antérieurs à 2.2 : preset de leur taille
if _saved.get("size") != size:
    _saved = {}
if _saved:
    batch_size    = _saved.get("batch_size", batch_size)
    max_iters     = _saved.get("max_iters", max_iters)
    learning_rate = _saved.get("learning_rate", learning_rate)
    warmup_iters  = _saved.get("warmup_iters", warmup_iters)
# Arrêt anticipé : --patience, sinon celle du run repris, sinon désactivé (0)
patience = args.patience if args.patience is not None else (_saved.get("patience") or 0)
if args.max_iters is not None:
    max_iters    = max(1, args.max_iters)
    warmup_iters = min(_TRAIN_PRESETS[size][3], max(1, max_iters // 10))
if args.batch_size is not None:
    batch_size = max(1, args.batch_size)

if args.lr is not None:
    learning_rate = args.lr

# Régularisation : --option, sinon réglage du run repris, sinon défaut
def _hp(name, default):
    value = getattr(args, name)
    return value if value is not None else _saved.get(name, default) if _saved else default


dropout      = _hp("dropout", 0.1)       # embeddings, résiduels, MLP
attn_dropout = _hp("attn_dropout", 0.0)  # DANS l'attention : 0 garde le noyau SDPA rapide (CPU / Mac)
min_lr     = learning_rate / 10
lr_decay_iters  = max_iters
grad_clip       = 1.0
weight_decay    = _hp("weight_decay", 0.1)   # sur les matrices seulement (pas les RMSNorm)
# beta2 = 0.999 (défaut PyTorch) : mesuré en 2.3 (evals/BASELINES.md), 0.95 — courant pour les grands
# LLM — coûte ≈ 0.08 bit par octet sur nano / Hugo (6 à 8 fois le bruit d'une graine à l'autre)
betas           = (0.9, _hp("beta2", 0.999))

# Évaluation : toutes les eval_interval iters, moyenne sur eval_iters batchs fixes (voir plus haut)

# Gradient accumulation — simule un batch effectif plus grand (--grad_accum, ex. 4 si OOM avec un
# batch réduit) ; en reprise, celle du run
gradient_accumulation_steps = args.grad_accum or (_saved.get("grad_accum") if _saved else None) or 1

# Détection de l'appareil (Ajout du support Mac MPS)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# ── Précision ─────────────────────────────────────────────────────────────────
# auto : bfloat16 sur GPU CUDA récent (float16 + GradScaler sinon), float32 ailleurs.
# Sur Mac (MPS) et CPU, bfloat16 est possible mais pas toujours plus rapide : à mesurer
# avec popixa bench --dtype bfloat16 avant de l'adopter.
dtype = args.dtype
if dtype == "auto":
    if device == "cuda":
        # bfloat16 natif à partir d'Ampere (capacité 8) ; avant, PyTorch l'ÉMULE (lent) → float16
        dtype = "bfloat16" if torch.cuda.get_device_capability()[0] >= 8 else "float16"
    else:
        dtype = "float32"
if dtype == "float16" and device != "cuda":
    print("❌ --dtype float16 n'est géré que sur GPU CUDA (GradScaler) — utilise bfloat16 ou float32")
    sys.exit(1)
_ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
autocast = contextlib.nullcontext()
if dtype != "float32":
    try:
        autocast = torch.autocast(device_type=device, dtype=_ptdtype)
        with autocast:
            _probe = torch.ones(8, 8, device=device) @ torch.ones(8, 8, device=device)
        # Un autocast non géré peut se désactiver avec un simple avertissement : on vérifie
        if _probe.dtype != _ptdtype:
            raise RuntimeError(f"autocast inactif (calcul resté en {_probe.dtype})")
    except Exception as e:   # précision non gérée par cet appareil / cette version de PyTorch
        print(f"❌ --dtype {dtype} indisponible sur {device} : {str(e).splitlines()[0]}")
        sys.exit(1)
_GradScaler = getattr(torch.amp, "GradScaler", None)
scaler = (_GradScaler("cuda", enabled=(dtype == "float16")) if _GradScaler is not None
          else torch.cuda.amp.GradScaler(enabled=(dtype == "float16")))

print(f"🚀 Device détecté : {device.upper()} · précision {dtype}")

rope_base = 500_000 if args.longrope else 10_000

# ── Reprise : l'architecture vient du checkpoint, pas de --size ───────────────
# (sinon load_state_dict plante si --size diffère, ou les tables RoPE du checkpoint
#  écrasent silencieusement un --longrope différent)
if resume_ckpt is not None:
    ck = resume_ckpt["config"]
    ck_rope = getattr(ck, "rope_base", 10_000)
    if (ck.block_size, ck.n_layer, ck.n_head, ck.n_embd) != (block_size, n_layer, n_head, n_embd):
        demande = (f"--size {size}" if args.size else
                   f"réglages enregistrés ({size})" if _saved else f"aucun preset (défaut {size})")
        print(f"⚠️  Architecture du checkpoint (block {ck.block_size}, {ck.n_layer} couches, "
              f"{ck.n_head} têtes, embd {ck.n_embd}) ≠ {demande} → on garde le checkpoint "
              f"(hyperparamètres d'entraînement de {size})")
    if ck_rope != rope_base:
        print(f"⚠️  rope_base du checkpoint ({ck_rope}) ≠ demandé ({rope_base}) → on garde le checkpoint")
    block_size, n_layer, n_head, n_embd = ck.block_size, ck.n_layer, ck.n_head, ck.n_embd
    rope_base = ck_rope
    # Même source de données que l'entraînement initial si --data_dir est omis
    if args.data_dir is None and resume_ckpt.get("data_dir"):
        args.data_dir = resume_ckpt["data_dir"]
        print(f"📂 Données du checkpoint : {args.data_dir}")

# Guard OOM — medium sur MPS dépasse facilement les 20 GB
# (85M params × batch 8 × block 1024 × bfloat16 ≈ 20+ GB activations).
# Testée sur l'architecture EFFECTIVE (un checkpoint medium repris compte aussi)
if (device == "mps" and _size_of(n_layer, n_head, n_embd) == "medium"
        and "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ):
    print("⚠️  Attention : un modèle medium peut provoquer un OOM sur Apple Silicon (>20 GB MPS).")
    if resume_ckpt is not None:
        print("   Le checkpoint repris est medium : --size ne change pas son architecture. Recommandation :")
        print("   nouvel entraînement (sans --resume) avec --size small (~3 GB) ou --size nano.")
    else:
        print("   Recommandation : utilise --size small (~10M params, ~3 GB) ou --size nano (~0.9M params).")
    print("   Pour forcer quand même (de préférence avec un batch réduit, la garde ne regarde pas le batch) :")
    print("   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 popixa train ... --batch_size 2")
    sys.exit(1)


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

    # memmap : les fichiers restent sur disque, seuls les morceaux tirés sont lus →
    # un corpus de plusieurs Go s'entraîne sans le charger en RAM
    def _open_bin(name):
        path = os.path.join(args.data_dir, name)
        # np.memmap refuse un fichier vide → tableau vide (« split trop court » plus bas)
        return (np.memmap(path, dtype=np.uint16, mode="r") if os.path.getsize(path) > 0
                else np.zeros(0, dtype=np.uint16))
    train_data = _open_bin("train.bin")
    val_data   = _open_bin("val.bin")
    print(f"Données binaires ouvertes (memmap) depuis {args.data_dir}/")

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
    data       = np.array([stoi[c] for c in text], dtype=np.int64)
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


# Générateurs dédiés : batchs d'entraînement (reprise : graine + iter de départ) et batchs
# d'évaluation, re-tirés À L'IDENTIQUE à chaque évaluation → les val loss successives sont
# comparables entre elles (best.pt, arrêt anticipé) au lieu de dépendre du tirage
_batch_gen = torch.Generator().manual_seed(seed)
_eval_gen  = torch.Generator()


def get_batch(split, gen=None):
    d  = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,), generator=gen or _batch_gen).tolist()
    x  = torch.from_numpy(np.stack([d[i     : i + block_size    ].astype(np.int64) for i in ix]))
    y  = torch.from_numpy(np.stack([d[i + 1 : i + block_size + 1].astype(np.int64) for i in ix]))
    if device == "cuda":
        return (x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True))
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    model.train(False)
    out = {}
    for split in ["train", "val"]:
        _eval_gen.manual_seed(seed + 1)          # mêmes batchs à chaque évaluation
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, _eval_gen)
            with autocast:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = float(losses.mean())
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
    attn_dropout=attn_dropout,
)

model = nanoPOPIXA(config).to(device)

# Weight decay sur les matrices (Linear, embedding partagé avec lm_head) uniquement :
# appliqué aux poids des RMSNorm, il les tire vers 0 et dégrade la normalisation
_decay    = [p for p in model.parameters() if p.dim() >= 2]
_no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW([{"params": _decay, "weight_decay": weight_decay},
                               {"params": _no_decay, "weight_decay": 0.0}],
                              lr=learning_rate, betas=betas)

os.makedirs(out_dir, exist_ok=True)

# ── Reprise depuis checkpoint ──────────────────────────────────────────────────
# "iter" = nombre d'itérations d'entraînement DÉJÀ effectuées au moment de la sauvegarde
iter_start = 0
resume_ckpt_loaded = resume_ckpt is not None
_restore_best = False
if resume_ckpt is not None:
    if resume_ckpt["config"].vocab_size != vocab_size:
        print(f"❌ Vocabulaire du checkpoint ({resume_ckpt['config'].vocab_size}) ≠ données "
              f"({vocab_size}) — reprise impossible sur un autre tokenizer/dataset")
        sys.exit(1)
    model.load_state_dict(resume_ckpt["model"])
    if "optimizer" in resume_ckpt:
        try:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        except ValueError:
            # Checkpoint < 2.3 : un seul groupe de paramètres (weight decay partout)
            print("⚠️  État de l'optimiseur d'une version précédente : moments AdamW réinitialisés")
        # load_state_dict remet les hyperparamètres ENREGISTRÉS : on applique ceux de ce lancement
        # (--weight_decay / --beta2 à la reprise, ou réglages du run) — la LR est fixée à chaque pas
        for group, wd in zip(optimizer.param_groups, (weight_decay, 0.0)):
            group["weight_decay"], group["betas"] = wd, betas
    iter_start = resume_ckpt.get("iter", 0)
    best_val   = resume_ckpt.get("best_val", float("inf"))
    best_iter  = resume_ckpt.get("best_iter")
    # best.pt doit venir de CE run : sinon (absent, ou laissé par un autre entraînement interrompu
    # dans le même dossier) on repart de zéro pour que la prochaine évaluation le réécrive
    _ckpt_best_iter = best_iter
    _nothing_to_do = (bool(resume_ckpt.get("stopped_early")) and args.patience is None) or iter_start >= max_iters
    if best_iter is not None:
        try:
            _best = load_checkpoint(best_path, map_location="cpu")
            _same = (_best.get("iter") == best_iter
                     and abs(_best.get("val_loss", float("nan")) - best_val) < 1e-9)
        except (FileNotFoundError, ValueError):
            _same = False
        if not _same and best_iter == iter_start:
            _restore_best = True          # meilleur point = ce checkpoint : mêmes poids, on le réécrit
        elif not _same:
            print(f"⚠️  {best_path} absent ou d'un autre entraînement"
                  + (" (impossible à reconstruire : son meilleur point n'est pas le dernier checkpoint)"
                     if _nothing_to_do else " : il sera réécrit à la prochaine évaluation"))
            best_val, best_iter = float("inf"), None
    bad_evals  = resume_ckpt.get("bad_evals", 0) if args.patience is None else 0
    _batch_gen.manual_seed(seed + iter_start)
    if resume_ckpt.get("stopped_early") and args.patience is None:
        print(f"✅ Entraînement arrêté plus tôt (iter {iter_start}, meilleure val loss à l'iter "
              f"{_ckpt_best_iter}) — rien à faire ; --patience N (ou 0) pour continuer")
        iter_start = max_iters
    elif iter_start >= max_iters:
        print(f"✅ Entraînement déjà terminé ({iter_start}/{max_iters} itérations) — rien à faire")
    else:
        print(f"✅ Reprise depuis iter {iter_start}")
    del resume_ckpt
else:
    best_val, best_iter, bad_evals = float("inf"), None, 0
    # Nouvel entraînement : on ne détruit jamais silencieusement un run existant (voir plus haut)
    _old = [n for n in RUN_FILES if os.path.exists(os.path.join(out_dir, n))]
    if _old:
        os.makedirs(staging_dir)
        for name in _old:
            os.replace(os.path.join(out_dir, name), os.path.join(staging_dir, name))
        print(f"📦 Run précédent mis de côté → {prev_dir}/ ({', '.join(_old)}) — restauré si ce lancement "
              "s'arrête avant sa 1re sauvegarde ; --resume pour reprendre au lieu de repartir de 0")


def _payload(it: int) -> dict:
    """Contenu commun à checkpoint.pt et best.pt (format 2 : types simples, chargement sûr)."""
    payload = {
        "model":     model.state_dict(),
        "config":    config,
        "iter":      it,
        "tokenizer": meta.get("tokenizer", "char"),
        "data_dir":  args.data_dir,
        "best_val":  best_val,
        "best_iter": best_iter,
        # Réglages d'entraînement : --resume sans options les reprend tels quels
        "train":     {"size": size, "seed": seed, "eval_interval": eval_interval, "eval_iters": eval_iters,
                      "batch_size": batch_size, "max_iters": max_iters,
                      "learning_rate": learning_rate, "warmup_iters": warmup_iters,
                      "patience": patience, "dtype": dtype, "grad_accum": gradient_accumulation_steps,
                      "weight_decay": weight_decay, "beta2": betas[1], "dropout": dropout,
                      "attn_dropout": attn_dropout},
    }
    if meta.get("tokenizer") != "tiktoken_gpt2":
        payload["vocab"] = {"stoi": stoi, "itos": itos}
    return payload


def save_checkpoint(it: int, stopped_early: bool = False) -> None:
    """checkpoint.pt : modèle + optimizer (pour --resume). it = itérations effectuées."""
    _commit_launch()
    if best_iter is None and os.path.exists(best_path):
        os.remove(best_path)        # best.pt d'un autre run (reprise) : jamais mélangé avec celui-ci
    payload = _payload(it)
    payload.update(optimizer=optimizer.state_dict(), bad_evals=bad_evals, stopped_early=stopped_early)
    _write_checkpoint(ckpt_path, payload)


def save_best(it: int, val: float) -> None:
    """best.pt : poids à la meilleure val loss (sans optimizer : plus léger, pour chat / eval)."""
    _commit_launch()
    payload = _payload(it)
    payload["val_loss"] = val
    _write_checkpoint(best_path, payload)


def write_run_json(**results) -> None:
    """run.json : réglages et résultats de l'expérience (comparés par popixa runs)."""
    info = {
        "version": _popixa_version(), "out_dir": out_dir, "size": size, "device": device, "dtype": dtype,
        "seed": seed, "data_dir": args.data_dir, "input": None if args.data_dir else args.input,
        "tokenizer": meta.get("tokenizer", "char"), "vocab_size": vocab_size,
        "params": model.get_num_params(False), "block_size": block_size, "n_layer": n_layer,
        "n_head": n_head, "n_embd": n_embd, "batch_size": batch_size, "max_iters": max_iters,
        "learning_rate": learning_rate, "warmup_iters": warmup_iters, "weight_decay": weight_decay,
        "dropout": dropout, "attn_dropout": attn_dropout, "beta2": betas[1], "patience": patience,
        "grad_accum": gradient_accumulation_steps,
    }
    info.update(results)
    tmp = run_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, run_path)


def _popixa_version() -> str:
    try:
        from popixa_cli import popixa_version
        return popixa_version()
    except Exception:
        return "dev"


# ─────────────────────────────────────────
# Boucle d'entraînement
# ─────────────────────────────────────────

print(f"\n🚀 Démarrage entraînement nanoPOPIXA [{size}] → {out_dir}/\n")
t0     = time.time()
t_last = t0

# Log propre à chaque run (écrit dès le lancement : popixa monitor le suit tout de suite) ;
# en reprise on AJOUTE un en-tête (le monitor garde le dernier : max_iters / batch peuvent changer)
_committed = resume_ckpt_loaded or not os.path.isdir(staging_dir)


def _commit_launch() -> None:
    """1re sauvegarde d'un nouveau run : le run précédent mis de côté devient prev/."""
    global _committed
    if _committed:
        return
    _committed = True
    import shutil
    if os.path.isdir(prev_dir):
        shutil.rmtree(prev_dir)
    os.replace(staging_dir, prev_dir)


if iter_start < max_iters:
    if resume_ckpt_loaded:
        # Les poids vont changer : l'évaluation d'un lancement précédent ne les décrit plus
        for _old_eval in ("eval.json", "samples.md"):
            _path = os.path.join(out_dir, _old_eval)
            if os.path.exists(_path):
                _stem, _ext = os.path.splitext(_old_eval)
                os.replace(_path, os.path.join(out_dir, f"{_stem}.prev{_ext}"))
    with open(log_path, "a" if resume_ckpt_loaded else "w", encoding="utf-8") as f:
        f.write(f"# max_iters={max_iters} eval_interval={eval_interval}"
                f" batch_size={batch_size * gradient_accumulation_steps} block_size={block_size}\n")
    write_run_json(status="en cours")
if _restore_best:
    # Mêmes poids que checkpoint.pt : écriture directe, sans engager de lancement
    _write_checkpoint(best_path, dict(_payload(best_iter), val_loss=best_val))
    print(f"♻️  {best_path} restauré depuis le checkpoint (meilleur point = dernier checkpoint)")


def log_eval(it: int, losses: dict, lr: float, dt: float, pct: float) -> None:
    log_line = (f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"lr {lr:.2e} | {dt:.1f}s")
    filled = int(30 * pct)
    print(f"[{'█' * filled + '░' * (30 - filled)}] {pct:5.1%}  {log_line}"
          + ("  ★ best.pt" if best_iter == it else ""))
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")


def track_best(it: int, val: float) -> tuple:
    """Met à jour le meilleur point ; retourne (amélioré, arrêt anticipé). L'appelant écrit
    checkpoint.pt PUIS best.pt : interrompu entre les deux, la reprise restaure best.pt depuis
    le checkpoint (meilleur point = dernier checkpoint)."""
    global best_val, best_iter, bad_evals
    if val < best_val:
        best_val, best_iter, bad_evals = val, it, 0
        return True, False
    bad_evals += 1
    return False, patience > 0 and bad_evals >= patience


stopped_early = False
it = iter_start
try:
    for it in range(iter_start, max_iters):

        # LR scheduling — cosine avec warmup
        lr = get_lr(it)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Évaluation périodique (pas au tout premier pas d'une reprise : déjà évalué)
        if it % eval_interval == 0 and not (resume_ckpt_loaded and it == iter_start):
            losses = estimate_loss(model)
            now, dt = time.time(), time.time() - t_last
            t_last = now
            # Le modèle initial n'est jamais « meilleur » ; sauvegardes inutiles avant le 1er pas
            improved, stop = track_best(it, losses["val"]) if it > iter_start else (False, False)
            log_eval(it, losses, lr, dt, it / max_iters)
            if it > iter_start:
                save_checkpoint(it, stopped_early=stop)
                if improved:
                    save_best(it, losses["val"])
            if stop:
                stopped_early = True
                print(f"\n⏹  Arrêt anticipé : {patience} évaluations sans amélioration de la val loss "
                      f"(meilleure : {best_val:.4f} à l'iter {best_iter})")
                break

        # ── Forward + backward avec gradient accumulation ─────────────────────────
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(gradient_accumulation_steps):
            X, Y = get_batch("train")
            with autocast:
                _, loss = model(X, Y)
            scaler.scale(loss / gradient_accumulation_steps).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

    # Évaluation + sauvegarde finales — sinon les itérations après la dernière évaluation
    # seraient perdues (rien à faire si aucune itération n'a tourné : le checkpoint, et donc
    # les sessions de chat, restent valides)
    if iter_start < max_iters and not stopped_early:
        losses = estimate_loss(model)
        improved, _ = track_best(max_iters, losses["val"])
        log_eval(max_iters, losses, get_lr(max_iters), time.time() - t_last, 1.0)
        save_checkpoint(max_iters)
        if improved:
            save_best(max_iters, losses["val"])
except BaseException as e:   # Ctrl+C, manque de mémoire, NaN… : run.json ne reste pas « en cours »
    if not _committed and _rollback_launch():
        print(f"\n⏹  Arrêt avant la première sauvegarde : le run précédent est remis en place dans {out_dir}/")
    else:
        write_run_json(status="interrompu" if isinstance(e, KeyboardInterrupt) else f"erreur ({type(e).__name__})",
                       last_iter=it, best_iter=best_iter,
                       best_val=None if math.isinf(best_val) else round(best_val, 6),
                       duration_s=round(time.time() - t0, 1))
    raise

if iter_start < max_iters:
    duration = time.time() - t0
    write_run_json(status="arrêt anticipé" if stopped_early else "terminé",
                   last_iter=it if stopped_early else max_iters, best_iter=best_iter,
                   best_val=None if math.isinf(best_val) else round(best_val, 6),
                   duration_s=round(duration, 1))   # durée de CE lancement (une reprise repart de 0)
    print(f"\n✅ Entraînement terminé en {duration:.1f}s — checkpoint : {ckpt_path} · "
          + (f"meilleur (val {best_val:.4f}, iter {best_iter}) : {best_path}" if best_iter is not None
             else "aucun meilleur modèle (val loss jamais finie : entraînement divergé ?)"))

    # ── Évaluation de fin d'expérience (popixa eval) ─────────────────────────────
    if args.eval:
        from popixa_eval import run_eval
        tasks = ("bpb", "paires", "samples") if args.data_dir else ("paires", "samples")
        target = best_path if best_iter is not None else ckpt_path      # best.pt de CE lancement
        print(f"\n📏 popixa eval sur {target} ({', '.join(tasks)})…")
        try:
            results, md = run_eval(target, data_dir=args.data_dir, tasks=tasks, device=device,
                                   log=lambda m: print(m))
            with open(os.path.join(out_dir, "eval.json"), "w", encoding="utf-8") as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            with open(os.path.join(out_dir, "samples.md"), "w", encoding="utf-8") as f:
                f.write(md)
            print(f"   → {out_dir}/eval.json · {out_dir}/samples.md")
        except (ValueError, OSError, SystemExit) as e:   # SystemExit : chat.load_model
            print(f"⚠️  popixa eval impossible : {e}")


# ─────────────────────────────────────────
# Génération rapide post-entraînement
# ─────────────────────────────────────────

if iter_start < max_iters:
    print("\n📝 Exemple de génération :\n")
    model.train(False)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    output  = decode(model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40,
                                    top_p=0.9)[0].tolist())
    print(output)
