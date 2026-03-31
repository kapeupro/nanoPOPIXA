"""
nanoPOPIXA — Chat interactif CLI v2
Streaming · Thinking blocks · Effort levels · Diminishing returns · Auto-compact
Usage : python chat.py   ou   popixa chat
"""

import sys
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

USER_C  = fg(0,   220, 255)   # cyan    — [Toi]
MODEL_C = fg(180,  80, 255)   # violet  — [nanoPOPIXA]
THINK_C = fg(160, 140,  40)   # ocre    — [Thinking] (interne, tamisé)
INFO_C  = fg(90,   90, 120)   # gris    — infos / séparateurs
CMD_C   = fg(255, 180,   0)   # jaune   — retour des commandes
ERR_C   = fg(255,  80,  80)   # rouge   — erreurs
WARN_C  = fg(255, 140,   0)   # orange  — warnings contexte

SEP = INFO_C + "─" * 52 + R

# ─── Seuils contexte (inspiré autoCompact.ts) ────────────────────────────────
# Claude : WARNING_BUFFER=20000, AUTOCOMPACT_BUFFER=13000, threshold=90%
CTX_WARNING_PCT = 0.80   # 80% → avertissement
CTX_COMPACT_PCT = 0.90   # 90% → compaction automatique (garde 50% récent)

# ─── Effort presets (inspiré effort.ts) ──────────────────────────────────────
# Claude définit low/medium/high/max avec des budgets tokens et températures
EFFORT_PRESETS = {
    "low":    dict(temperature=1.0, top_k=20,  top_p=0.85, max_tokens=100),
    "medium": dict(temperature=0.8, top_k=40,  top_p=0.90, max_tokens=200),
    "high":   dict(temperature=0.7, top_k=50,  top_p=0.95, max_tokens=400),
    "max":    dict(temperature=1.0, top_k=None, top_p=0.95, max_tokens=600),
}


# ─── Chargement du modèle ────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: str):
    try:
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


# ─── Génération streaming standard ───────────────────────────────────────────
def stream(model, encode, decode, context_str: str,
           max_tokens: int, temperature: float, top_k: int, device: str,
           repetition_penalty: float = 1.0, top_p: float = None,
           stop_on_repetition: bool = True) -> str:
    """Génère en streaming avec arrêt automatique si répétitif."""
    ctx = torch.tensor(encode(context_str), dtype=torch.long, device=device).unsqueeze(0)

    sys.stdout.write("\n" + MODEL_C + B + "[nanoPOPIXA]" + R + " " + MODEL_C)
    sys.stdout.flush()

    tokens = []
    try:
        for tok in model.generate_stream(
            ctx, max_tokens, temperature, top_k,
            repetition_penalty, top_p, stop_on_repetition
        ):
            ch = decode([tok])
            tokens.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()
    except KeyboardInterrupt:
        sys.stdout.write(INFO_C + "\n\n  [Interruption]" + R)

    sys.stdout.write(R + "\n")
    sys.stdout.flush()
    return "".join(tokens)


# ─── Génération avec Thinking blocks ─────────────────────────────────────────
def stream_think(model, encode, decode, context_str: str,
                 device: str, think_budget: int, response_budget: int,
                 temperature: float, top_k: int, top_p: float,
                 repetition_penalty: float) -> str:
    """
    Génération deux phases (Claude-inspired) :
      Phase 1 — Thinking interne  (affiché en ocre, temperature=1)
      Phase 2 — Réponse finale    (affichée en violet, temperature normale)
    """
    ctx = torch.tensor(encode(context_str), dtype=torch.long, device=device).unsqueeze(0)

    # ── Affichage thinking ────────────────────────────────────────────────────
    sys.stdout.write(
        "\n" + THINK_C + DIM + B + "[Thinking]" + R + " " + THINK_C + DIM
    )
    sys.stdout.flush()

    think_chars  = []
    resp_chars   = []
    in_response  = False

    try:
        for phase, tok in model.generate_stream_with_thinking(
            ctx,
            think_budget=think_budget,
            response_budget=response_budget,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ):
            ch = decode([tok])

            if phase == "think":
                think_chars.append(ch)
                sys.stdout.write(ch)
                sys.stdout.flush()

            elif phase == "response":
                if not in_response:
                    # Transition : ferme le bloc thinking, ouvre la réponse
                    sys.stdout.write(
                        R + "\n\n" + MODEL_C + B + "[nanoPOPIXA]" + R + " " + MODEL_C
                    )
                    sys.stdout.flush()
                    in_response = True
                resp_chars.append(ch)
                sys.stdout.write(ch)
                sys.stdout.flush()

    except KeyboardInterrupt:
        sys.stdout.write(INFO_C + "\n\n  [Interruption]" + R)

    sys.stdout.write(R + "\n")
    sys.stdout.flush()
    return "".join(resp_chars)


# ─── Gestion contexte (inspiré autoCompact.ts) ───────────────────────────────
def update_context(history: str, new_content: str, blk_size: int) -> tuple[str, bool]:
    """
    Ajoute new_content à l'historique.
    Retourne (history_updated, was_compacted).
    Compacte automatiquement à 90% (garde les 50% les plus récents).
    """
    history += new_content
    ctx_pct  = len(history) / blk_size

    if ctx_pct >= CTX_COMPACT_PCT:
        keep    = int(blk_size * 0.50)
        history = history[-keep:]
        return history, True

    return history, False


def context_warning(history: str, blk_size: int):
    """Affiche un warning si le contexte dépasse 80%."""
    pct = len(history) / blk_size
    if pct >= CTX_WARNING_PCT:
        bar_len = 20
        filled  = int(bar_len * pct)
        bar     = "█" * filled + "░" * (bar_len - filled)
        print(WARN_C + f"  ⚠ Contexte [{bar}] {pct*100:.0f}%"
              + INFO_C + " — compaction auto à 90%" + R)


# ─── Sauvegarde conversation ──────────────────────────────────────────────────
def save_conversation(turns: list, filename: str) -> None:
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
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model, encode, decode, ckpt = load_model(checkpoint_path, device)

    n_params     = model.get_num_params() / 1e6
    iter_num     = ckpt.get("iter", "?")
    blk_size     = model.config.block_size
    think_budget = 150   # tokens de réflexion interne (phase thinking)

    # ── En-tête ───────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print(INFO_C + f"  nanoPOPIXA v2  ·  {n_params:.2f}M params  ·  iter {iter_num}  ·  {device}" + R)
    print(SEP)
    print(INFO_C + "  Paramètres :" + R)
    print(CMD_C  + "    /temp 0.5"      + INFO_C + "          → température             (défaut 0.8)"  + R)
    print(CMD_C  + "    /tokens 300"    + INFO_C + "        → tokens réponse            (défaut 200)"  + R)
    print(CMD_C  + "    /topp 0.9"      + INFO_C + "         → nucleus sampling top-p   (défaut off)"  + R)
    print(CMD_C  + "    /penalty 1.3"   + INFO_C + "       → repetition penalty         (défaut 1.0)"  + R)
    print(INFO_C + "  Modes :" + R)
    print(CMD_C  + "    /think"         + INFO_C + "           → mode raisonnement interne (thinking)" + R)
    print(CMD_C  + "    /thinkbudget 200" + INFO_C + "   → tokens alloués au thinking   (défaut 150)"  + R)
    print(CMD_C  + "    /effort low|medium|high|max" + INFO_C + " → preset tout-en-un"                 + R)
    print(INFO_C + "  Contexte & outils :" + R)
    print(CMD_C  + "    /reset"         + INFO_C + "           → remettre le contexte à zéro"          + R)
    print(CMD_C  + "    /ctx"           + INFO_C + "             → afficher l'état du contexte"         + R)
    print(CMD_C  + "    /libre"         + INFO_C + "           → génération sans prompt"                + R)
    print(CMD_C  + "    /save [fichier]"+ INFO_C + "    → sauvegarder la conversation (.txt)"           + R)
    print(INFO_C + "  Ctrl+C → quitter" + R)
    print(SEP)

    history     = ""
    turns       = []
    think_mode  = False   # activé par /think

    while True:
        # ── Indicateur contexte dans le prompt ───────────────────────────────
        ctx_pct   = len(history) / blk_size if history else 0
        ctx_label = (
            WARN_C + f"[ctx {ctx_pct*100:.0f}%] " + R
            if ctx_pct >= CTX_WARNING_PCT
            else ""
        )
        think_label = THINK_C + "[think] " + R if think_mode else ""

        # ── Saisie utilisateur ────────────────────────────────────────────────
        try:
            sys.stdout.write(
                "\n" + ctx_label + think_label + USER_C + B + "[Toi] " + R + USER_C
            )
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

        # /temp
        if prompt.startswith("/temp "):
            try:
                temperature = float(prompt.split()[1])
                print(CMD_C + f"  → température : {temperature}" + R)
            except (IndexError, ValueError):
                print(ERR_C + "  usage : /temp 0.8" + R)
            continue

        # /tokens
        if prompt.startswith("/tokens "):
            try:
                max_tokens = int(prompt.split()[1])
                print(CMD_C + f"  → max tokens : {max_tokens}" + R)
            except (IndexError, ValueError):
                print(ERR_C + "  usage : /tokens 300" + R)
            continue

        # /topp
        if prompt.startswith("/topp "):
            try:
                top_p = float(prompt.split()[1])
                print(CMD_C + f"  → top-p : {top_p}" + R)
            except (IndexError, ValueError):
                print(ERR_C + "  usage : /topp 0.9" + R)
            continue

        # /penalty
        if prompt.startswith("/penalty "):
            try:
                repetition_penalty = float(prompt.split()[1])
                print(CMD_C + f"  → repetition penalty : {repetition_penalty}" + R)
            except (IndexError, ValueError):
                print(ERR_C + "  usage : /penalty 1.3" + R)
            continue

        # /effort — preset tout-en-un (inspiré effort.ts de Claude Code)
        if prompt.startswith("/effort"):
            parts = prompt.split()
            level = parts[1].lower() if len(parts) > 1 else ""
            if level not in EFFORT_PRESETS:
                print(ERR_C + "  usage : /effort low|medium|high|max" + R)
                print(INFO_C + "  Presets :" + R)
                for k, v in EFFORT_PRESETS.items():
                    print(CMD_C + f"    {k:8s}" + INFO_C
                          + f"  temp={v['temperature']}  top_k={v['top_k']}  "
                          + f"top_p={v['top_p']}  tokens={v['max_tokens']}" + R)
            else:
                p           = EFFORT_PRESETS[level]
                temperature = p["temperature"]
                top_k       = p["top_k"]
                top_p       = p["top_p"]
                max_tokens  = p["max_tokens"]
                print(CMD_C + f"  → effort [{level}]"
                      + INFO_C + f"  temp={temperature}  top_k={top_k}  "
                      + f"top_p={top_p}  tokens={max_tokens}" + R)
            continue

        # /think — bascule le mode thinking
        if prompt == "/think":
            think_mode = not think_mode
            if think_mode:
                print(THINK_C + f"  → Thinking activé  (budget={think_budget} tokens)" + R)
                print(INFO_C  + "  Phase 1 : raisonnement interne  temperature=1 (comme Claude)" + R)
                print(INFO_C  + "  Phase 2 : réponse finale        temperature normale" + R)
            else:
                print(CMD_C + "  → Thinking désactivé" + R)
            continue

        # /thinkbudget — ajuster le budget de thinking
        if prompt.startswith("/thinkbudget "):
            try:
                think_budget = int(prompt.split()[1])
                print(THINK_C + f"  → think budget : {think_budget} tokens" + R)
            except (IndexError, ValueError):
                print(ERR_C + "  usage : /thinkbudget 200" + R)
            continue

        # /ctx — afficher l'état du contexte
        if prompt == "/ctx":
            pct    = len(history) / blk_size * 100
            used   = len(history)
            bar_l  = 30
            filled = int(bar_l * pct / 100)
            bar    = "█" * filled + "░" * (bar_l - filled)
            color  = WARN_C if pct >= CTX_WARNING_PCT * 100 else CMD_C
            print(color + f"  Contexte [{bar}] {pct:.1f}%  ({used}/{blk_size} chars)" + R)
            print(INFO_C + f"  Warning à {CTX_WARNING_PCT*100:.0f}%  ·  Auto-compact à {CTX_COMPACT_PCT*100:.0f}%" + R)
            continue

        # /reset
        if prompt == "/reset":
            history = ""
            print(CMD_C + "  → contexte remis à zéro" + R)
            continue

        # /libre
        if prompt == "/libre":
            history = ""
            stream(model, encode, decode, "", max_tokens, temperature, top_k, device,
                   repetition_penalty, top_p)
            continue

        # /save
        if prompt.startswith("/save"):
            parts    = prompt.split(maxsplit=1)
            filename = (
                parts[1] if len(parts) > 1
                else f"conv_{datetime.now().strftime('%Y-%m-%d_%Hh%M')}.txt"
            )
            if not turns:
                print(ERR_C + "  ✗ Aucune conversation à sauvegarder" + R)
            else:
                save_conversation(turns, filename)
            continue

        # ── Warning contexte avant génération ────────────────────────────────
        context_warning(history, blk_size)

        # ── Génération ────────────────────────────────────────────────────────
        context_str = history + prompt

        if think_mode:
            # Mode thinking : deux phases (raisonnement + réponse)
            response = stream_think(
                model, encode, decode, context_str, device,
                think_budget=think_budget,
                response_budget=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p if top_p is not None else 0.9,
                repetition_penalty=repetition_penalty,
            )
        else:
            # Mode standard avec diminishing returns
            response = stream(
                model, encode, decode, context_str,
                max_tokens, temperature, top_k, device,
                repetition_penalty, top_p,
                stop_on_repetition=True,
            )

        # ── Mise à jour contexte avec auto-compact (inspiré autoCompact.ts) ──
        new_content = prompt + response
        history, compacted = update_context(history, new_content, blk_size)
        if compacted:
            print(INFO_C + "  [Auto-compact : contexte réduit à 50% — anciennes turns supprimées]" + R)

        turns.append(("user",  prompt))
        turns.append(("model", response))


# ─── Point d'entrée ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Chat interactif nanoPOPIXA v2")
    parser.add_argument("--checkpoint",  default="out-nanopopixa/checkpoint.pt")
    parser.add_argument("--max_tokens",  type=int,   default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k",       type=int,   default=40)
    parser.add_argument("--top_p",       type=float, default=None,
                        help="Nucleus sampling (ex: 0.9)")
    parser.add_argument("--effort",      default=None,
                        choices=list(EFFORT_PRESETS.keys()),
                        help="Preset effort : low|medium|high|max")
    args = parser.parse_args()

    # Effort preset écrase les params individuels si fourni
    max_tokens  = args.max_tokens
    temperature = args.temperature
    top_k       = args.top_k
    top_p       = args.top_p

    if args.effort:
        p           = EFFORT_PRESETS[args.effort]
        temperature = p["temperature"]
        top_k       = p["top_k"]
        top_p       = p["top_p"]
        max_tokens  = p["max_tokens"]
        print(INFO_C + f"  Effort preset : {args.effort}" + R)

    run_chat(args.checkpoint, max_tokens, temperature, top_k, top_p=top_p)


if __name__ == "__main__":
    main()
