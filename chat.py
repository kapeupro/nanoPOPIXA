"""
nanoPOPIXA — Chat interactif CLI
Streaming token par token · historique cumulatif · couleurs truecolor
Usage : python chat.py   ou   popixa chat
"""

import sys
import os
from datetime import datetime
import argparse
import torch

try:
    import readline  # flèches directionnelles + historique des commandes
except ImportError:
    pass

from model import nanoPOPIXA

# ─── Couleurs ────────────────────────────────────────────────────────────────
R   = "\033[0m"
B   = "\033[1m"
DIM = "\033[2m"

def fg(r, g, b): return f"\033[38;2;{r};{g};{b}m"

USER_C  = fg(0,   220, 255)   # cyan   — [Toi]
MODEL_C = fg(180,  80, 255)   # violet — [nanoPOPIXA]
INFO_C  = fg(90,   90, 120)   # gris   — infos / séparateurs
CMD_C   = fg(255, 180,   0)   # jaune  — retour des commandes
ERR_C   = fg(255,  80,  80)   # rouge  — erreurs

SEP = INFO_C + "─" * 52 + R


# ─── Chargement du modèle ────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: str):
    try:
        # PyTorch 2.6+ nécessite weights_only=False pour les objets personnalisés
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(ERR_C + f"\n  ✗ Checkpoint introuvable : {checkpoint_path}" + R)
        print(INFO_C + "  Entraîne d'abord le modèle :\n"
              "    popixa prep && popixa train --data_dir data/" + R)
        sys.exit(1)

    model = nanoPOPIXA(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.train(False)

    if ckpt.get("tokenizer") == "tiktoken_gpt2":
        try:
            import tiktoken
        except ImportError:
            print(ERR_C + "  ✗ tiktoken requis : pip install tiktoken" + R)
            sys.exit(1)
        enc    = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode_ordinary(s)
        decode = lambda l: enc.decode(l)
    elif "vocab" in ckpt:
        stoi   = ckpt["vocab"]["stoi"]
        itos   = ckpt["vocab"]["itos"]
        encode = lambda s: [stoi.get(c, 0) for c in s]
        decode = lambda l: "".join(itos.get(i, "") for i in l)
    else:
        print(ERR_C + "  ✗ Checkpoint sans vocabulaire (réentraîne le modèle)" + R)
        sys.exit(1)

    return model, encode, decode, ckpt


# ─── Génération streaming ────────────────────────────────────────────────────
def stream(model, encode, decode, context_str: str,
           max_tokens: int, temperature: float, top_k: int,
           repetition_penalty: float, device: str, top_p=None) -> str:
    """Envoie le contexte au modèle et affiche les tokens en temps réel."""
    ctx = torch.tensor(encode(context_str), dtype=torch.long, device=device).unsqueeze(0)

    sys.stdout.write("\n" + MODEL_C + B + "[nanoPOPIXA]" + R + " " + MODEL_C)
    sys.stdout.flush()

    tokens = []
    try:
        for tok in model.generate_stream(ctx, max_tokens, temperature, top_k, repetition_penalty, top_p):
            ch = decode([tok])
            tokens.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()
    except KeyboardInterrupt:
        sys.stdout.write(INFO_C + "\n\n  [Interruption génération]" + R)

    sys.stdout.write(R + "\n")
    sys.stdout.flush()
    return "".join(tokens)


# ─── Sauvegarde conversation ──────────────────────────────────────────────────
def save_conversation(turns: list, filename: str) -> None:
    """Écrit la conversation dans un fichier .txt lisible."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    sep = "═" * 48
    lines = [f"nanoPOPIXA — Conversation du {now}", sep, ""]

    for role, text in turns:
        label = "[Toi]" if role == "user" else "[nanoPOPIXA]"
        lines.append(f"{label} {text.strip()}")
        lines.append("")

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(CMD_C + f"  → sauvegardé : {filename}" + R)
    except OSError as e:
        print(ERR_C + f"  ✗ Erreur d'écriture : {e}" + R)


# ─── Boucle principale ───────────────────────────────────────────────────────
def run_chat(checkpoint_path: str, max_tokens: int, temperature: float, top_k: int,
             repetition_penalty: float = 1.0, top_p: float = None):
    # Détection Mac MPS
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    model, encode, decode, ckpt = load_model(checkpoint_path, device)

    n_params  = model.get_num_params() / 1e6
    iter_num  = ckpt.get("iter", "?")
    blk_size  = model.config.block_size

    # ── En-tête ───────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print(INFO_C + f"  nanoPOPIXA  ·  {n_params:.2f}M params  ·  iter {iter_num}  ·  {device}" + R)
    print(SEP)
    print(INFO_C + "  Commandes :" + R)
    print(CMD_C  + "    /temp 0.5" + INFO_C + "    → changer la température   (défaut 0.8)" + R)
    print(CMD_C  + "    /top_p 0.9" + INFO_C + "   → nucleus sampling          (défaut off)" + R)
    print(CMD_C  + "    /penalty 1.2" + INFO_C + "  → pénalité de répétition  (défaut 1.0)" + R)
    print(CMD_C  + "    /tokens 300" + INFO_C + "   → nb de tokens générés    (défaut 200)" + R)
    print(CMD_C  + "    /reset" + INFO_C + "     → remettre le contexte à zéro" + R)
    print(CMD_C  + "    /libre" + INFO_C + "     → génération sans prompt" + R)
    print(CMD_C  + "    /save [fichier]" + INFO_C + " → sauvegarder la conversation (.txt)" + R)
    print(INFO_C + "  Ctrl+C → quitter" + R)
    print(SEP)

    # Historique cumulatif — le modèle "voit" toute la conv précédente
    history = ""
    turns   = []   # liste de (role, content) pour /save

    while True:
        # ── Saisie utilisateur ────────────────────────────────────────────────
        try:
            sys.stdout.write("\n" + USER_C + B + "[Toi] " + R + USER_C)
            sys.stdout.flush()
            prompt = input().strip()
            sys.stdout.write(R)
            sys.stdout.flush()
        except (KeyboardInterrupt, EOFError):
            print(R + "\n\n" + INFO_C + "  À bientôt !" + R + "\n")
            break

        if not prompt:
            continue

        # ── Commandes ─────────────────────────────────────────────────────────
        if prompt.startswith("/temp "):
            try:
                temperature = float(prompt.split()[1])
                print(CMD_C + f"  → température : {temperature}" + R)
            except (IndexError, ValueError):
                print(ERR_C + "  usage : /temp 0.8" + R)
            continue

        if prompt.startswith("/top_p "):
            try:
                val = prompt.split()[1]
                top_p = None if val in ("off", "0") else float(val)
                print(CMD_C + f"  → top_p : {top_p}" + R)
            except (IndexError, ValueError):
                print(ERR_C + "  usage : /top_p 0.9  ou  /top_p off" + R)
            continue

        if prompt.startswith("/penalty "):
            try:
                repetition_penalty = float(prompt.split()[1])
                print(CMD_C + f"  → pénalité : {repetition_penalty}" + R)
            except (IndexError, ValueError):
                print(ERR_C + "  usage : /penalty 1.2" + R)
            continue

        if prompt.startswith("/tokens "):
            try:
                max_tokens = int(prompt.split()[1])
                print(CMD_C + f"  → max tokens : {max_tokens}" + R)
            except (IndexError, ValueError):
                print(ERR_C + "  usage : /tokens 300" + R)
            continue

        if prompt == "/reset":
            history = ""
            print(CMD_C + "  → contexte remis à zéro" + R)
            continue

        if prompt == "/libre":
            history = ""
            stream(model, encode, decode, "", max_tokens, temperature, top_k,
                   repetition_penalty, device, top_p)
            continue

        if prompt.startswith("/save"):
            parts    = prompt.split(maxsplit=1)
            filename = (
                parts[1]
                if len(parts) > 1
                else f"conv_{datetime.now().strftime('%Y-%m-%d_%Hh%M')}.txt"
            )
            if not turns:
                print(ERR_C + "  ✗ Aucune conversation à sauvegarder" + R)
            else:
                save_conversation(turns, filename)
            continue

        # ── Génération ────────────────────────────────────────────────────────
        # Contexte = tout ce qui a été dit avant + le nouveau prompt
        context_str = history + prompt
        response    = stream(model, encode, decode, context_str,
                             max_tokens, temperature, top_k,
                             repetition_penalty, device, top_p)

        # Mettre à jour l'historique (contexte glissant)
        history = context_str + response
        turns.append(("user",  prompt))
        turns.append(("model", response))
        # Tronquer si on dépasse block_size (en estimant ~1 char ≈ 1 token)
        if len(history) > blk_size:
            history = history[-blk_size:]


# ─── Point d'entrée ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Chat interactif nanoPOPIXA")
    parser.add_argument("--checkpoint",  default="out-nanopopixa/checkpoint.pt")
    parser.add_argument("--max_tokens",  type=int,   default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k",       type=int,   default=40)
    parser.add_argument("--penalty",     type=float, default=1.0)
    args = parser.parse_args()

    run_chat(args.checkpoint, args.max_tokens, args.temperature, args.top_k, args.penalty)


if __name__ == "__main__":
    main()
