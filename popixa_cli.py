"""
nanoPOPIXA — Point d'entrée CLI principal
Accessible via la commande `popixa` après `pip install -e .`
"""

import os
import re
import sys
import shlex
import argparse

# Installation éditable antérieure + git pull : les modules ajoutés depuis l'installation
# (ex. popixa_eval en 2.2) restent importables depuis le dossier du dépôt
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

# ─── Couleurs ─────────────────────────────────────────────────────────────────
R    = "\033[0m"
B    = "\033[1m"
def fg(r, g, b): return f"\033[38;2;{r};{g};{b}m"

PROMPT_C = fg(180, 80,  255)   # violet  — "popixa"
SEP_C    = fg(120, 120, 180)   # gris-bleu — "›"
INFO_C   = fg(160, 160, 200)   # gris clair
CMD_C    = fg(255, 180,   0)   # jaune
ERR_C    = fg(255,  80,  80)   # rouge


def popixa_version() -> str:
    """Version du code exécuté : pyproject.toml du dépôt, sinon paquet installé, sinon « dev »."""
    try:
        with open(os.path.join(_HERE, "pyproject.toml"), encoding="utf-8") as f:
            m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.MULTILINE)
        if m:
            return m.group(1)
    except OSError:
        pass
    try:
        from importlib.metadata import version
        return version("nanopopixa")
    except Exception:
        return "dev"


HELP = """
╔══════════════════════════════════════════╗
║         nanoPOPIXA  —  CLI  v1.0        ║
╚══════════════════════════════════════════╝

  prep    [--dataset NAME] [--data_dir DIR] [--char]
      → Préparer un dataset (shakespeare, linux, hugo, javascript…)
        Défaut : tiktoken BPE | --char pour tokenisation caractère

  train   [--data_dir DIR] [--size nano|small|medium] [--resume] [--longrope]
      → Entraîner le modèle  (nano ~0.9M, small ~10M, medium ~85M params hors embeddings)

  chat    [--checkpoint PATH] [--temp FLOAT] [--tokens INT]
      → Chat interactif avec le modèle

  monitor [--log train.log] [--refresh 1.0]
      → Dashboard live de la courbe de loss

  gen     [--prompt TEXTE] [--tokens INT] [--temp FLOAT] [--top_p FLOAT]
          [--json] [--schema FICHIER|JSON]
      → Générer du texte (mode non-interactif) — --json/--schema : sortie JSON garantie

  eval    [--checkpoint PATH] [--data_dir DIR] [--tasks bpb,paires,samples] [--out eval.json]
      → Mesurer la qualité : bits/octet, paires minimales FR, échantillons (déterministe)

  bench   [--size nano|small|medium] [--seconds 15]
      → Débit réel de ta machine : tokens/s entraînement et génération, mémoire, TFLOPS

  scrape  --url URL [--max_pages N] [--output fichier.txt]
      → Crawler web → corpus d'entraînement

  collect SOURCE_DIR [--data_dir DIR] [--extensions .py,.js,…]
      → Assembler du code source local en corpus

  update  → mettre à jour depuis GitHub (git pull + pip install)
  help  →  cette aide      exit  →  quitter      version  →  version installée

Exemples :
  prep --dataset shakespeare
  train --data_dir data/ --size small
  train --data_dir data/ --resume
  chat
  gen --prompt '{"nom": ' --schema schema.json
  eval --data_dir data/ --out eval.json
  bench --size small
  scrape --url https://fr.wikipedia.org/wiki/Python --max_pages 20
"""


def _positive_float(value: str) -> float:
    """Type argparse : flottant strictement positif (ex. repetition penalty)."""
    f = float(value)
    if f <= 0:
        raise argparse.ArgumentTypeError(f"doit être > 0 (reçu {value})")
    return f


def _positive_int(value: str) -> int:
    """Type argparse : entier strictement positif."""
    i = int(value)
    if i <= 0:
        raise argparse.ArgumentTypeError(f"doit être > 0 (reçu {value})")
    return i


def cmd_update(args):
    """Met à jour nanoPOPIXA depuis GitHub et réinstalle le package."""
    import subprocess, sys, os

    repo_dir = os.path.dirname(os.path.abspath(__file__))

    print(INFO_C + "  Vérification de la mise à jour…" + R)

    # Vérifier qu'on est dans un dépôt git
    result = subprocess.run(["git", "rev-parse", "--git-dir"],
                            cwd=repo_dir, capture_output=True)
    if result.returncode != 0:
        print(ERR_C + "  ✗ Dossier non reconnu comme dépôt git." + R)
        print(INFO_C + "  Clone d'abord : git clone https://github.com/kapeupro/nanoPOPIXA" + R)
        return

    # Version actuelle
    before = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            cwd=repo_dir, capture_output=True, text=True).stdout.strip()

    # git pull
    print(INFO_C + "  git pull…" + R)
    pull = subprocess.run(["git", "pull"], cwd=repo_dir, capture_output=True, text=True)
    if pull.returncode != 0:
        print(ERR_C + f"  ✗ git pull échoué :\n{pull.stderr}" + R)
        return

    after = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=repo_dir, capture_output=True, text=True).stdout.strip()

    if before == after:
        print(CMD_C + "  ✓ Déjà à jour." + R + INFO_C + f"  ({after})" + R)
        return

    # Afficher le changelog
    log = subprocess.run(
        ["git", "log", "--oneline", f"{before}..{after}"],
        cwd=repo_dir, capture_output=True, text=True
    ).stdout.strip()
    print(CMD_C + f"  ✓ Mis à jour {before} → {after}" + R)
    if log:
        print(INFO_C + "  Nouveautés :" + R)
        for line in log.splitlines():
            print(CMD_C + f"    {line}" + R)

    # pip install -e .
    print(INFO_C + "\n  Réinstallation du package…" + R)
    pip = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet", "--no-deps"],
        cwd=repo_dir, capture_output=True, text=True
    )
    if pip.returncode != 0:
        print(ERR_C + f"  ✗ pip install échoué :\n{pip.stderr}" + R)
        return

    print(CMD_C + "  ✓ nanoPOPIXA mis à jour. Relance popixa chat pour profiter des nouveautés." + R)


def cmd_chat(args):
    from chat import run_chat, EFFORT_PRESETS
    max_tokens  = args.tokens
    temperature = args.temp
    top_k       = args.top_k
    top_p       = getattr(args, "top_p", None)
    penalty     = args.penalty
    if getattr(args, "effort", None):
        p           = EFFORT_PRESETS[args.effort]
        temperature = p["temperature"]
        top_k       = p["top_k"]
        top_p       = p["top_p"]
        max_tokens  = p["max_tokens"]
    run_chat(args.checkpoint, max_tokens, temperature, top_k,
             repetition_penalty=penalty, top_p=top_p)


def cmd_train(args):
    import sys as _sys
    import runpy, os
    _sys.argv = ["train.py"] + (["--size", args.size] if args.size else [])
    if args.data_dir:
        _sys.argv += ["--data_dir", args.data_dir]
    if args.input:
        _sys.argv += ["--input", args.input]
    if args.resume:
        _sys.argv += ["--resume"]
    if args.longrope:
        _sys.argv += ["--longrope"]
    if args.max_iters is not None:
        _sys.argv += ["--max_iters", str(args.max_iters)]
    if args.batch_size is not None:
        _sys.argv += ["--batch_size", str(args.batch_size)]
    if args.seed is not None:
        _sys.argv += ["--seed", str(args.seed)]
    train_path = os.path.join(os.path.dirname(__file__), "train.py")
    runpy.run_path(train_path, run_name="__main__")


def cmd_prep(args):
    from data_prep import prepare
    prepare(args.dataset, args.data_dir, use_tiktoken=not args.char)


def cmd_scrape(args):
    if not args.url:
        print(ERR_C + "  ✗ usage : scrape --url URL [--max_pages N] [--output fichier.txt]" + R)
        return
    try:
        from scrape import scrape_recursive
    except ImportError as e:
        print(ERR_C + f"  ✗ Module manquant ({e.name}) : pip install requests beautifulsoup4" + R)
        return
    scrape_recursive(args.url, args.max_pages, args.output)


def cmd_collect(args):
    from data_prep import collect_code
    exts = tuple(e.strip() for e in args.extensions.split(","))
    collect_code(args.source_dir, args.data_dir, exts)


def cmd_monitor(args):
    from monitor import run_monitor
    run_monitor(args.log, args.refresh)


def cmd_gen(args):
    import contextlib
    import torch
    from chat import load_model, load_token_bytes

    def fail(msg: str) -> None:
        print(ERR_C + f"  ✗ {msg}" + R, file=sys.stderr)
        sys.exit(1)

    # Schéma validé AVANT de charger le modèle (erreur immédiate, code de sortie 1)
    structured_mode = args.json or args.schema is not None
    schema = None
    if structured_mode:
        import structured
        try:
            schema = structured.load_schema(args.schema) if args.schema is not None else None
            structured.JSONSchemaMatcher(schema)
        except (ValueError, OSError) as e:
            fail(f"Structured outputs : {e}")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Logs de chargement sur stderr : stdout ne contient que le texte généré
    # (exploitable tel quel, ex. popixa gen --json > sortie.json)
    with contextlib.redirect_stdout(sys.stderr):
        model, encode, decode, ckpt = load_model(args.checkpoint, device)

    ids = encode(args.prompt) if args.prompt else []
    ctx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # vide → amorce token 0

    if structured_mode:
        constraint = structured.json_constraint(
            load_token_bytes(ckpt, model.config.vocab_size), schema
        )
        closing = constraint.completion_tokens()
        if closing is None:
            fail("le vocabulaire du modèle ne permet pas d'écrire un JSON conforme à ce schéma")
        if len(closing) > args.tokens:
            fail(f"--tokens {args.tokens} insuffisant : il faut au moins {len(closing)} tokens "
                 "pour un JSON complet")
        tokens = list(model.generate_structured(
            ctx, constraint, max_new_tokens=args.tokens, temperature=args.temp,
            top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.penalty,
        ))
        if not constraint.is_complete():
            # Jamais de JSON tronqué sur stdout (ex. popixa gen --json > sortie.json)
            print(decode(tokens), file=sys.stderr)
            fail("JSON incomplet — augmente --tokens")
        print(decode(tokens))
        return

    tokens = list(model.generate_stream(
        ctx, args.tokens, args.temp, args.top_k, args.penalty, args.top_p,
    ))
    print(decode(tokens))


def cmd_eval(args):
    import json
    from popixa_eval import run_eval
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if "bpb" in tasks and not args.data_dir:
        tasks.remove("bpb")
        print(INFO_C + "  (bpb ignorée : pas de --data_dir)" + R, file=sys.stderr)
    if not tasks:
        print(ERR_C + "  ✗ aucune tâche à évaluer (bpb demande --data_dir)" + R, file=sys.stderr)
        sys.exit(1)
    # Dossiers de sortie vérifiés AVANT l'évaluation (sinon les résultats sont perdus à la fin)
    samples_out = args.samples_out if "samples" in tasks else None
    for opt, path in (("--out", args.out), ("--samples_out", samples_out)):
        if not path:
            continue
        d = os.path.dirname(os.path.abspath(path))
        is_dir = os.path.isdir(path) or not os.path.basename(path)       # « resultats/ » aussi
        if is_dir or not os.path.isdir(d):
            why = "désigne un dossier" if is_dir else f"dossier {d} introuvable"
            print(ERR_C + f"  ✗ {opt} {path} : {why}" + R, file=sys.stderr)
            sys.exit(1)
    if args.out and samples_out and os.path.realpath(args.out) == os.path.realpath(samples_out):
        print(ERR_C + f"  ✗ --out et --samples_out désignent le même fichier ({args.out})" + R, file=sys.stderr)
        sys.exit(1)
    print(INFO_C + f"  Évaluation de {args.checkpoint}…" + R, file=sys.stderr)
    try:
        results, md = run_eval(
            args.checkpoint, data_dir=args.data_dir, split=args.split, tasks=tasks,
            max_bytes=args.max_bytes,
            **({"pairs_path": args.pairs} if args.pairs else {}),
            **({"prompts_path": args.prompts} if args.prompts else {}),
            samples_tokens=args.samples_tokens, seed=args.seed,
            log=lambda m: print(CMD_C + m + R, file=sys.stderr),
        )
    except (ValueError, OSError) as e:
        print(ERR_C + f"  ✗ {e}" + R, file=sys.stderr)
        sys.exit(1)
    text = json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True)
    ok = True

    def write(path, content, fallback, stream):
        """Écrit un fichier de sortie ; en cas d'échec, le contenu part sur `stream` (jamais perdu)."""
        nonlocal ok
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(INFO_C + f"  → {path}" + R, file=sys.stderr)
        except OSError as e:
            ok = False
            where = "la sortie standard" if stream is sys.stdout else "la sortie d'erreur"
            print(ERR_C + f"  ✗ écriture de {path} impossible ({e}) : {fallback} sur {where}" + R,
                  file=sys.stderr)
            print(content, file=stream)

    # stdout ne porte que le JSON des résultats ; le markdown de secours va sur stderr
    if args.out:
        write(args.out, text + "\n", "résultats JSON", sys.stdout)
    else:
        print(text)
    if md is not None and args.samples_out:
        write(args.samples_out, md, "échantillons", sys.stderr)
    if not ok:
        sys.exit(1)


def cmd_bench(args):
    import json
    from popixa_eval import run_bench
    if not 0.0 <= args.dropout < 1.0:
        print(ERR_C + f"  ✗ --dropout doit être dans [0, 1) (reçu {args.dropout})" + R, file=sys.stderr)
        sys.exit(1)
    print(INFO_C + f"  Benchmark preset {args.size} (entraînement ≈ {args.seconds / 2:.0f} s — au moins "
          f"2 pas, plus long sur CPU pour small/medium —, puis génération)…" + R, file=sys.stderr)
    r = run_bench(size=args.size, vocab_size=args.vocab, batch_size=args.batch,
                  seconds=args.seconds, dropout=args.dropout)
    if args.json:
        print(json.dumps(r, indent=2, sort_keys=True))
        return
    print(CMD_C + f"  {r['size']} · {r['params'] / 1e6:.1f}M params · vocab {r['vocab_size']} · "
          f"{r['device']} · torch {r['torch']}" + R)
    print(f"  Entraînement : {r['train_tok_s']:,.0f} tokens/s  (batch {r['batch_size']} × "
          f"{r['block_size']}, dropout {r['dropout']}, {r['train_steps']} pas) · "
          f"{r['train_tflops']:.3f} TFLOPS effectifs")
    print(f"  Génération   : {r['gen_tok_s']:,.0f} tokens/s (greedy, KV-cache)")
    print(f"  Mémoire      : {r['peak_memory_mb']:,.0f} Mo ({r['memory_kind']})")
    hours = 1e9 / max(r["train_tok_s"], 1e-9) / 3600
    print(INFO_C + f"  ≈ {hours:,.1f} h pour 1 milliard de tokens d'entraînement à ce débit" + R)


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
    p_chat.add_argument("--top_p",      type=float, default=None)
    p_chat.add_argument("--penalty",    type=_positive_float, default=1.0)
    p_chat.add_argument("--effort",     default=None,
                        choices=["low", "medium", "high", "max"])

    # ── train ─────────────────────────────────────────────────────────
    p_train = sub.add_parser("train")
    p_train.add_argument("--data_dir", default=None)
    p_train.add_argument("--input",    default="input.txt")
    p_train.add_argument("--size",     default=None,       # small, ou la taille du checkpoint repris
                         choices=["nano", "small", "medium"])
    p_train.add_argument("--resume",   action="store_true")
    p_train.add_argument("--longrope", action="store_true")
    p_train.add_argument("--max_iters",  type=int, default=None)
    p_train.add_argument("--batch_size", type=int, default=None)
    p_train.add_argument("--seed",       type=int, default=None)

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
    p_mon.add_argument("--refresh", type=_positive_float, default=1.0)

    # ── scrape ────────────────────────────────────────────────────────
    p_scr = sub.add_parser("scrape")
    p_scr.add_argument("--url",       required=False, default="",
                       help="URL de départ (ex: https://fr.wikipedia.org/wiki/...)")
    p_scr.add_argument("--max_pages", type=int, default=10)
    p_scr.add_argument("--output",    default="web_fr.txt")

    # ── update ────────────────────────────────────────────────────────
    sub.add_parser("update")

    # ── gen ───────────────────────────────────────────────────────────
    p_gen = sub.add_parser("gen")
    p_gen.add_argument("--checkpoint", default="out-nanopopixa/checkpoint.pt")
    p_gen.add_argument("--prompt",     default="")
    p_gen.add_argument("--temp",       type=float, default=0.8)
    p_gen.add_argument("--tokens",     type=int,   default=300)
    p_gen.add_argument("--top_k",      type=int,   default=40)
    p_gen.add_argument("--top_p",      type=float, default=None)
    p_gen.add_argument("--penalty",    type=_positive_float, default=1.0)
    p_gen.add_argument("--json",       action="store_true",
                       help="Structured outputs : sortie JSON valide garantie")
    p_gen.add_argument("--schema",     default=None,
                       help="Schéma JSON (fichier .json ou JSON inline) — implique --json")

    # ── eval ──────────────────────────────────────────────────────────
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--checkpoint", "--ckpt", default="out-nanopopixa/checkpoint.pt")
    p_eval.add_argument("--data_dir",   default=None,
                        help="Dossier avec val.bin + meta.pkl (requis pour la tâche bpb)")
    p_eval.add_argument("--split",      default="val", choices=["val", "train"])
    p_eval.add_argument("--tasks",      default="bpb,paires,samples",
                        help="Liste parmi bpb,paires,samples (bpb ignorée sans --data_dir)")
    p_eval.add_argument("--max_bytes",  type=_positive_int, default=None,
                        help="bpb sur les N premiers octets de texte seulement (même extrait, à un token "
                             "près, quel que soit le tokenizer : bpb comparables)")
    p_eval.add_argument("--out",        default=None, help="Fichier JSON (sinon stdout)")
    p_eval.add_argument("--samples_out", default="samples.md")
    p_eval.add_argument("--samples_tokens", type=_positive_int, default=128,
                        help="tokens par échantillon (≥ 120 pour mesurer les rendements décroissants)")
    p_eval.add_argument("--seed",       type=int, default=1337)
    p_eval.add_argument("--pairs",      default=None, help="Fichier de paires (JSONL)")
    p_eval.add_argument("--prompts",    default=None, help="Fichier d'amorces")

    # ── bench ─────────────────────────────────────────────────────────
    p_bench = sub.add_parser("bench")
    p_bench.add_argument("--size",    default="small", choices=["nano", "small", "medium"])
    p_bench.add_argument("--vocab",   type=_positive_int, default=50257)
    p_bench.add_argument("--batch",   type=_positive_int, default=4)
    p_bench.add_argument("--seconds", type=_positive_float, default=15.0)
    p_bench.add_argument("--dropout", type=float, default=0.1,
                         help="dropout pendant la mesure (0.1 = train.py ; 0 = sans dropout)")
    p_bench.add_argument("--json",    action="store_true")

    return parser


_DISPATCH = {
    "chat":    cmd_chat,
    "train":   cmd_train,
    "prep":    cmd_prep,
    "collect": cmd_collect,
    "monitor": cmd_monitor,
    "scrape":  cmd_scrape,
    "gen":     cmd_gen,
    "eval":    cmd_eval,
    "bench":   cmd_bench,
    "update":  cmd_update,
}


def _run_command(tokens: list[str]) -> None:
    """
    Parse et exécute une liste de tokens (ex. ["chat", "--temp", "0.9"]).
    Les erreurs d'arguments lèvent SystemExit(2) (code de sortie correct en mode direct ;
    le shell interactif l'intercepte).
    """
    parser = _build_parser()
    args = parser.parse_args(tokens)
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
          + "chat  train  prep  eval  bench  monitor  gen  scrape  collect  version  help  exit" + R)

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

        # Découpe en tokens — une apostrophe française non fermée (aujourd'hui, l'IA…)
        # est échappée puis on réessaie, sans jamais laisser de guillemets parasites
        try:
            tokens = shlex.split(line)
        except ValueError:
            try:
                tokens = shlex.split(re.sub(r"(\w)'(\w)", r"\1\\'\2", line))
            except ValueError:
                print(ERR_C + "  ✗ Guillemet non fermé — entoure le texte de guillemets doubles" + R)
                continue

        cmd = tokens[0]

        if cmd in ("exit", "quit", "q"):
            print(INFO_C + "  À bientôt !" + R)
            break

        if cmd in ("help", "h", "?"):
            print(HELP)
            continue

        if cmd in ("version", "--version", "-V"):
            print(f"nanoPOPIXA {popixa_version()}")
            continue

        if cmd not in _DISPATCH:
            print(ERR_C + f"  ✗ '{cmd}' n'est pas une commande." + R + "  "
                  + INFO_C + "→ tape " + CMD_C + "chat" + INFO_C
                  + " pour discuter avec le modèle, ou "
                  + CMD_C + "help" + INFO_C + " pour la liste." + R)
            continue

        try:
            _run_command(tokens)
        except SystemExit:
            # Une commande qui abandonne (ex. chat sans checkpoint) ne ferme pas le shell
            pass
        except KeyboardInterrupt:
            # Ctrl+C arrête la commande en cours (train, gen, monitor…), pas le shell
            print(R + "\n" + INFO_C + "  [Interrompu]" + R)
        except Exception as e:
            print(R + ERR_C + f"  ✗ {cmd} : {type(e).__name__}: {e}" + R)


# ─── Point d'entrée ───────────────────────────────────────────────────────────
def main():
    argv = sys.argv[1:]
    if not argv:
        from splash import splash
        splash()
        run_shell()
        return

    if argv[0] in ("-h", "--help", "help", "h", "?"):
        print(HELP)
        return
    if argv[0] in ("exit", "quit", "q"):
        return
    if argv[0] in ("--version", "-V", "version"):
        print(f"nanoPOPIXA {popixa_version()}")
        return

    if argv[0] == "chat":
        from splash import splash
        splash()

    try:
        _run_command(argv)
    except KeyboardInterrupt:
        print(R + "\n" + INFO_C + "  [Interrompu]" + R)
        sys.exit(130)


if __name__ == "__main__":
    main()
