"""
nanoPOPIXA — Point d'entrée CLI principal
Accessible via la commande `popixa` après `pip install -e .`
"""

import sys
import shlex
import argparse

# ─── Couleurs ─────────────────────────────────────────────────────────────────
R    = "\033[0m"
B    = "\033[1m"
def fg(r, g, b): return f"\033[38;2;{r};{g};{b}m"

PROMPT_C = fg(180, 80,  255)   # violet  — "popixa"
SEP_C    = fg(120, 120, 180)   # gris-bleu — "›"
INFO_C   = fg(160, 160, 200)   # gris clair
CMD_C    = fg(255, 180,   0)   # jaune
ERR_C    = fg(255,  80,  80)   # rouge


HELP = """
╔══════════════════════════════════════════╗
║         nanoPOPIXA  —  CLI  v1.0        ║
╚══════════════════════════════════════════╝

  prep    [--dataset NAME] [--data_dir DIR] [--char]
      → Préparer un dataset (shakespeare, linux, hugo, javascript…)
        Défaut : tiktoken BPE | --char pour tokenisation caractère

  train   [--data_dir DIR] [--size nano|small|medium] [--resume]
      → Entraîner le modèle  (nano ~2M, small ~10M, medium ~85M)

  chat    [--checkpoint PATH] [--temp FLOAT] [--tokens INT]
      → Chat interactif avec le modèle

  monitor [--log train.log] [--refresh 1.0]
      → Dashboard live de la courbe de loss

  gen     [--prompt TEXTE] [--tokens INT] [--temp FLOAT]
      → Générer du texte (mode non-interactif)

  scrape  --url URL [--max_pages N] [--output fichier.txt]
      → Crawler web → corpus d'entraînement

  collect SOURCE_DIR [--data_dir DIR] [--extensions .py,.js,…]
      → Assembler du code source local en corpus

  help  →  cette aide      exit  →  quitter

Exemples :
  prep --dataset shakespeare
  train --data_dir data/ --size small
  train --data_dir data/ --resume
  chat
  scrape --url https://fr.wikipedia.org/wiki/Python --max_pages 20
"""


def cmd_chat(args):
    from chat import run_chat
    top_p = args.top_p if args.top_p > 0 else None
    run_chat(args.checkpoint, args.tokens, args.temp, args.top_k, args.penalty, top_p)


def cmd_train(args):
    import sys as _sys
    import runpy, os
    _sys.argv = ["train.py", "--size", args.size]
    if args.data_dir:
        _sys.argv += ["--data_dir", args.data_dir]
    if args.input:
        _sys.argv += ["--input", args.input]
    if args.resume:
        _sys.argv += ["--resume"]
    if args.amp:
        _sys.argv += ["--amp"]
    train_path = os.path.join(os.path.dirname(__file__), "train.py")
    runpy.run_path(train_path, run_name="__main__")


def cmd_prep(args):
    from data_prep import prepare
    prepare(args.dataset, args.data_dir, use_tiktoken=not args.char)


def cmd_scrape(args):
    from scrape import scrape_recursive
    scrape_recursive(args.url, args.max_pages, args.output)


def cmd_collect(args):
    from data_prep import collect_code
    exts = tuple(e.strip() for e in args.extensions.split(","))
    collect_code(args.source_dir, args.data_dir, exts)


def cmd_monitor(args):
    from monitor import run_monitor
    run_monitor(args.log, args.refresh)


def cmd_gen(args):
    import torch
    from model import nanoPOPIXA

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model  = nanoPOPIXA(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.train(False)

    if ckpt.get("tokenizer") == "tiktoken_gpt2":
        import tiktoken as _tt
        enc    = _tt.get_encoding("gpt2")
        encode = lambda s: enc.encode_ordinary(s)
        decode = lambda l: enc.decode(l)
    elif "vocab" in ckpt:
        stoi   = ckpt["vocab"]["stoi"]
        itos   = ckpt["vocab"]["itos"]
        encode = lambda s: [stoi.get(c, 0) for c in s]
        decode = lambda l: "".join([itos[i] for i in l])
    else:
        print("Erreur : checkpoint sans vocabulaire (réentraîne le modèle)")
        return

    if args.prompt:
        ctx = torch.tensor(encode(args.prompt), dtype=torch.long, device=device).unsqueeze(0)
    else:
        ctx = torch.zeros((1, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        top_p = args.top_p if args.top_p > 0 else None
        out = model.generate(ctx, max_new_tokens=args.tokens, temperature=args.temp,
                             top_k=args.top_k, repetition_penalty=args.penalty, top_p=top_p)

    text = decode(out[0].tolist())
    print(text[len(args.prompt):] if args.prompt else text)


def _build_parser() -> argparse.ArgumentParser:
    """Construit et retourne le parser argparse principal (réutilisable)."""
    parser = argparse.ArgumentParser(prog="popixa", add_help=False)
    sub    = parser.add_subparsers(dest="command")

    # ── chat ──────────────────────────────────────────────────────────
    p_chat = sub.add_parser("chat")
    p_chat.add_argument("--checkpoint", default="out-nanopopixa/checkpoint.pt")
    p_chat.add_argument("--temp",       type=float, default=0.8)
    p_chat.add_argument("--tokens",     type=int,   default=200)
    p_chat.add_argument("--top_k",      type=int,   default=40)
    p_chat.add_argument("--penalty",    type=float, default=1.0)
    p_chat.add_argument("--top_p",      type=float, default=0.0,
                        help="Nucleus sampling 0..1 (0 = désactivé)")

    # ── train ─────────────────────────────────────────────────────────
    p_train = sub.add_parser("train")
    p_train.add_argument("--data_dir", default=None)
    p_train.add_argument("--input",    default="input.txt")
    p_train.add_argument("--size",     default="medium",
                         choices=["nano", "small", "medium"])
    p_train.add_argument("--resume",   action="store_true")
    p_train.add_argument("--amp",      action="store_true",
                         help="Mixed precision bfloat16 (CUDA uniquement)")

    # ── prep ──────────────────────────────────────────────────────────
    p_prep = sub.add_parser("prep")
    p_prep.add_argument("--dataset",  default="shakespeare")
    p_prep.add_argument("--data_dir", default="data")
    p_prep.add_argument("--char",     action="store_true",
                        help="Tokenisation caractère (défaut : tiktoken BPE)")

    # ── collect ───────────────────────────────────────────────────────
    p_col = sub.add_parser("collect")
    p_col.add_argument("source_dir")
    p_col.add_argument("--data_dir",    default="data")
    p_col.add_argument("--extensions",  default=".py,.js,.ts,.c,.h,.md")

    # ── monitor ───────────────────────────────────────────────────────
    p_mon = sub.add_parser("monitor")
    p_mon.add_argument("--log",     default="train.log")
    p_mon.add_argument("--refresh", type=float, default=1.0)

    # ── scrape ────────────────────────────────────────────────────────
    p_scr = sub.add_parser("scrape")
    p_scr.add_argument("--url",       required=False, default="",
                       help="URL de départ (ex: https://fr.wikipedia.org/wiki/...)")
    p_scr.add_argument("--max_pages", type=int, default=10)
    p_scr.add_argument("--output",    default="web_fr.txt")

    # ── gen ───────────────────────────────────────────────────────────
    p_gen = sub.add_parser("gen")
    p_gen.add_argument("--checkpoint", default="out-nanopopixa/checkpoint.pt")
    p_gen.add_argument("--prompt",     default="")
    p_gen.add_argument("--temp",       type=float, default=0.8)
    p_gen.add_argument("--tokens",     type=int,   default=300)
    p_gen.add_argument("--top_k",      type=int,   default=40)
    p_gen.add_argument("--penalty",    type=float, default=1.0)
    p_gen.add_argument("--top_p",      type=float, default=0.0,
                       help="Nucleus sampling 0..1 (0 = désactivé)")

    return parser


_DISPATCH = {
    "chat":    cmd_chat,
    "train":   cmd_train,
    "prep":    cmd_prep,
    "collect": cmd_collect,
    "monitor": cmd_monitor,
    "scrape":  cmd_scrape,
    "gen":     cmd_gen,
}


def _run_command(tokens: list[str]) -> None:
    """Parse et exécute une liste de tokens (ex. ["chat", "--temp", "0.9"])."""
    parser = _build_parser()
    try:
        args = parser.parse_args(tokens)
    except SystemExit:
        return
    if args.command not in _DISPATCH:
        print(HELP)
        return
    _DISPATCH[args.command](args)


# ─── Shell interactif ─────────────────────────────────────────────────────────
def run_shell() -> None:
    """REPL interactif — lancé quand `popixa` est appelé sans argument."""
    try:
        import readline  # historique des commandes avec flèches ↑↓
    except ImportError:
        pass

    print(HELP)
    print(INFO_C
          + "  Commandes : " + CMD_C
          + "chat  train  prep  monitor  gen  scrape  collect  help  exit" + R)

    prompt = (PROMPT_C + B + "popixa" + R + " " + SEP_C + "›" + R + " ")

    while True:
        try:
            sys.stdout.write("\n" + prompt)
            sys.stdout.flush()
            line = input().strip()
        except (KeyboardInterrupt, EOFError):
            print(R + "\n\n" + INFO_C + "  À bientôt !" + R)
            break

        if not line:
            continue

        # Découpe en tokens — fallback sur split() si apostrophe française etc.
        try:
            tokens = shlex.split(line)
        except ValueError:
            tokens = line.split()

        cmd = tokens[0]

        if cmd in ("exit", "quit", "q"):
            print(INFO_C + "  À bientôt !" + R)
            break

        if cmd in ("help", "h", "?"):
            print(HELP)
            continue

        if cmd not in _DISPATCH:
            print(ERR_C + f"  ✗ '{cmd}' n'est pas une commande." + R + "  "
                  + INFO_C + "→ tape " + CMD_C + "chat" + INFO_C
                  + " pour discuter avec le modèle, ou "
                  + CMD_C + "help" + INFO_C + " pour la liste." + R)
            continue

        _run_command(tokens)


# ─── Point d'entrée ───────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        from splash import splash
        splash()
        run_shell()
        return

    if sys.argv[1] == "chat":
        from splash import splash
        splash()

    _run_command(sys.argv[1:])


if __name__ == "__main__":
    main()
