"""
nanoPOPIXA — Évaluation et benchmark (version 2.2 « Mètre-étalon »)

On mesure AVANT de changer : chaque version doit pouvoir dire, chiffres à l'appui,
si elle fait mieux que la précédente.

popixa eval — trois tâches déterministes, résultats dans eval.json :
  bpb     : passe complète sur val.bin en fenêtres de block_size sans chevauchement.
            Bits par octet = Σ NLL / (ln 2 × octets UTF-8 du texte prédit) : seule métrique
            comparable entre tokenizers (caractère, GPT-2, futur tokenizer français).
  paires  : paires minimales françaises (evals/fr_paires.jsonl) — la phrase correcte
            a-t-elle une log-vraisemblance plus forte que la phrase fautive ?
            (accord sujet-verbe, accord nominal, participe passé, élision, prépositions)
  samples : 20 amorces françaises (evals/prompts_fr.txt) à graine fixe → samples.md,
            avec distinct-2 et taux de sorties répétitives (_is_repetitive /
            _has_diminishing_returns).

popixa bench — débit d'entraînement et de génération, mémoire pic, TFLOPS effectifs,
pour recaler les estimations de compute sur TA machine.
"""

import os
import sys
import json
import math
import time
import pickle

import numpy as np
import torch
import torch.nn.functional as F

ROOT       = os.path.dirname(os.path.abspath(__file__))
EVALS_DIR  = os.path.join(ROOT, "evals")
PAIRS_PATH = os.path.join(EVALS_DIR, "fr_paires.jsonl")
PROMPTS_PATH = os.path.join(EVALS_DIR, "prompts_fr.txt")

TASKS = ("bpb", "paires", "samples")


# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────

def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def popixa_version() -> str:
    try:
        from importlib.metadata import version
        return version("nanopopixa")
    except Exception:
        return "dev"


def token_nbytes(ckpt: dict, vocab_size: int) -> np.ndarray:
    """Nombre d'octets UTF-8 de chaque token (0 pour les tokens spéciaux / inconnus)."""
    from chat import load_token_bytes
    return np.array([len(b) if b else 0 for b in load_token_bytes(ckpt, vocab_size)], dtype=np.int64)


def load_split(data_dir: str, split: str, ckpt: dict) -> np.ndarray:
    """Charge <data_dir>/<split>.bin (uint16, en memmap) et vérifie le tokenizer du checkpoint."""
    path = os.path.join(data_dir, f"{split}.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} introuvable (popixa prep --data_dir {data_dir})")
    meta_path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        ck_tok = ckpt.get("tokenizer", "char")
        if meta.get("tokenizer", "char") != ck_tok:
            raise ValueError(f"tokenizer des données ({meta.get('tokenizer')}) ≠ checkpoint ({ck_tok})")
        if ck_tok != "tiktoken_gpt2" and "vocab" in ckpt and meta.get("stoi") != ckpt["vocab"]["stoi"]:
            raise ValueError("vocabulaire caractère des données ≠ checkpoint")
    return np.memmap(path, dtype=np.uint16, mode="r")


# ─────────────────────────────────────────────────────────────────────────────
# Tâche bpb — bits par octet
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_bpb(model, data: np.ndarray, nbytes: np.ndarray, max_tokens: int = None,
             batch_size: int = 8, device: str = "cpu") -> dict:
    """
    Passe complète et déterministe sur `data` : fenêtres consécutives de block_size tokens,
    sans chevauchement (la dernière peut être plus courte). Chaque token prédit compte une
    fois ; le dénominateur en octets est la somme exacte des octets des tokens prédits
    (indépendant des frontières de fenêtres et du tokenizer).
    """
    block = model.config.block_size
    n_pred = len(data) - 1
    if max_tokens is not None:
        n_pred = min(n_pred, max_tokens)
    if n_pred <= 0:
        raise ValueError("pas assez de tokens pour évaluer")

    n_full, rest = divmod(n_pred, block)
    # Fenêtres pleines regroupées en batch, puis la fenêtre partielle finale seule
    groups = [[(w * block, block) for w in range(g, min(g + batch_size, n_full))]
              for g in range(0, n_full, batch_size)]
    if rest:
        groups.append([(n_full * block, rest)])

    total_nll, total_tokens, total_bytes = 0.0, 0, 0
    model.train(False)
    for group in groups:
        length = group[0][1]
        x = torch.from_numpy(np.stack([data[s:s + length].astype(np.int64) for s, _ in group])).to(device)
        y_np = np.stack([data[s + 1:s + length + 1].astype(np.int64) for s, _ in group])
        logits, _ = model(x, return_all_logits=True)
        nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                              torch.from_numpy(y_np).to(device).reshape(-1), reduction="sum")
        total_nll    += float(nll)
        total_tokens += y_np.size
        total_bytes  += int(nbytes[y_np].sum())

    loss = total_nll / total_tokens
    return {
        "tokens":     total_tokens,
        "bytes":      total_bytes,
        "block_size": block,
        "windows":    n_full + (1 if rest else 0),
        "loss":       round(loss, 6),
        "ppl":        round(math.exp(min(loss, 50.0)), 4),
        "bpb":        round(total_nll / (math.log(2) * max(1, total_bytes)), 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tâche paires — paires minimales françaises
# ─────────────────────────────────────────────────────────────────────────────

def load_pairs(path: str = PAIRS_PATH) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _encodable(text: str, encode, ckpt: dict) -> bool:
    """Vocabulaire caractère : chaque caractère doit exister (encode remplace sinon par 0)."""
    if ckpt.get("tokenizer") == "tiktoken_gpt2" or "vocab" not in ckpt:
        return True
    stoi = ckpt["vocab"]["stoi"]
    return all(c in stoi for c in text)


@torch.no_grad()
def sequence_logprob(model, prefix_ids: list, ids: list, device: str = "cpu") -> float:
    """log P(ids | prefix) = Σ log P(token_t | prefix + tokens < t)."""
    full = (list(prefix_ids) + list(ids))[-(model.config.block_size + 1):]
    n_target = min(len(ids), len(full) - 1)
    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    logits, _ = model(x, return_all_logits=True)
    logp = F.log_softmax(logits[0].float(), dim=-1)
    targets = torch.tensor(full[1:], dtype=torch.long, device=device)
    picked = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    return float(picked[-n_target:].sum())


def eval_pairs(model, encode, ckpt: dict, pairs: list, device: str = "cpu") -> dict:
    """
    Score = % de paires où log P(bonne) > log P(mauvaise) (sommes de log-probabilités,
    méthode BLiMP), avec la même amorce « \\n » (début de ligne) pour les deux phrases.
    baseline_longueur : score d'un « modèle » qui préfère toujours la phrase la plus courte
    en tokens — un vrai modèle doit faire nettement mieux.
    """
    model.train(False)
    prefix = encode("\n") or [0]
    per = {}
    n_ok = n_total = n_skip = 0
    shorter = 0.0
    for p in pairs:
        if not (_encodable(p["good"], encode, ckpt) and _encodable(p["bad"], encode, ckpt)):
            n_skip += 1
            continue
        g_ids, b_ids = encode(p["good"]), encode(p["bad"])
        ok = sequence_logprob(model, prefix, g_ids, device) > sequence_logprob(model, prefix, b_ids, device)
        stats = per.setdefault(p["phenomene"], [0, 0])
        stats[0] += int(ok)
        stats[1] += 1
        n_ok += int(ok)
        n_total += 1
        shorter += 1.0 if len(g_ids) < len(b_ids) else (0.5 if len(g_ids) == len(b_ids) else 0.0)
    return {
        "n":                 n_total,
        "non_couvertes":     n_skip,
        "accuracy":          round(n_ok / n_total, 4) if n_total else None,
        "baseline_longueur": round(shorter / n_total, 4) if n_total else None,
        "par_phenomene":     {k: {"accuracy": round(v[0] / v[1], 4), "n": v[1]}
                              for k, v in sorted(per.items())},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tâche samples — échantillons à graine fixe
# ─────────────────────────────────────────────────────────────────────────────

def load_prompts(path: str = PROMPTS_PATH) -> list:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip() and not line.startswith("#")]


def distinct2(tokens: list) -> float:
    bigrams = list(zip(tokens, tokens[1:]))
    return len(set(bigrams)) / len(bigrams) if bigrams else 0.0


def eval_samples(model, encode, decode, prompts: list, n_tokens: int = 100, seed: int = 1337,
                 temperature: float = 0.8, top_k: int = 40, device: str = "cpu") -> tuple:
    """Génère une continuation par amorce (graine fixe par amorce, aucun arrêt anticipé)."""
    from model import nanoPOPIXA
    model.train(False)
    rows = []
    for i, prompt in enumerate(prompts):
        torch.manual_seed(seed + i)
        ctx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        toks = list(model.generate_stream(ctx, n_tokens, temperature, top_k, stop_policy="off"))
        rows.append({
            "prompt":      prompt,
            "text":        decode(toks),
            "distinct2":   round(distinct2(toks), 4),
            "repetitive":  bool(nanoPOPIXA._is_repetitive(toks)),
            "diminishing": bool(nanoPOPIXA._has_diminishing_returns(toks)),
        })
    n = len(rows)
    summary = {
        "n":                n,
        "tokens":           n_tokens,
        "seed":             seed,
        "temperature":      temperature,
        "top_k":            top_k,
        "distinct2":        round(sum(r["distinct2"] for r in rows) / n, 4) if n else None,
        "taux_repetitif":   round(sum(r["repetitive"] for r in rows) / n, 4) if n else None,
        "taux_diminishing": round(sum(r["diminishing"] for r in rows) / n, 4) if n else None,
    }
    return summary, rows


def samples_markdown(rows: list, summary: dict, checkpoint: str) -> str:
    lines = [
        "# nanoPOPIXA — échantillons d'évaluation", "",
        f"- checkpoint : `{checkpoint}`",
        f"- {summary['n']} amorces · {summary['tokens']} tokens · graine {summary['seed']} · "
        f"temperature {summary['temperature']} · top_k {summary['top_k']}",
        f"- distinct-2 moyen : {summary['distinct2']} · répétitifs : {summary['taux_repetitif']:.0%}"
        f" · diminishing returns : {summary['taux_diminishing']:.0%}", "",
    ]
    for i, r in enumerate(rows, 1):
        flags = " ".join(f for f, on in (("⟳ répétitif", r["repetitive"]),
                                         ("⇣ diminishing", r["diminishing"])) if on)
        lines += [f"## {i}. {r['prompt']}", "",
                  f"distinct-2 {r['distinct2']}" + (f" · {flags}" if flags else ""), "",
                  "```text", r["prompt"] + r["text"], "```", ""]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(checkpoint: str, data_dir: str = None, split: str = "val", tasks=TASKS,
             max_tokens: int = None, pairs_path: str = PAIRS_PATH, prompts_path: str = PROMPTS_PATH,
             samples_tokens: int = 100, seed: int = 1337, device: str = None,
             log=print) -> tuple:
    """
    Évalue `checkpoint` ; retourne (résultats dict, markdown des échantillons ou None).
    Les résultats ne contiennent ni date ni durée : deux exécutions donnent un JSON identique.
    """
    import contextlib
    from chat import load_model
    from session_cache import checkpoint_fingerprint

    device = device or _device()
    with contextlib.redirect_stdout(sys.stderr):
        model, encode, decode, ckpt = load_model(checkpoint, device)

    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        raise ValueError(f"tâche(s) inconnue(s) : {', '.join(unknown)} ({', '.join(TASKS)})")

    results = {
        "popixa_version": popixa_version(),
        "checkpoint": {
            "path":        os.path.basename(checkpoint),
            "fingerprint": checkpoint_fingerprint(checkpoint),
            "iter":        ckpt.get("iter"),
            "tokenizer":   ckpt.get("tokenizer", "char"),
            "params":      model.get_num_params(False),
            "params_hors_embeddings": model.get_num_params(),
            "block_size":  model.config.block_size,
        },
        "tasks": {},
    }
    md = None

    if "bpb" in tasks:
        if not data_dir:
            raise ValueError("la tâche bpb demande --data_dir (val.bin)")
        t0 = time.time()
        data = load_split(data_dir, split, ckpt)
        res = eval_bpb(model, data, token_nbytes(ckpt, model.config.vocab_size),
                       max_tokens=max_tokens, device=device)
        res["split"] = split
        results["tasks"]["bpb"] = res
        log(f"  bpb      {res['bpb']:.4f}  · loss {res['loss']:.4f} · ppl {res['ppl']:.2f}"
            f" · {res['tokens']:,} tokens ({time.time() - t0:.1f}s)")

    if "paires" in tasks:
        t0 = time.time()
        res = eval_pairs(model, encode, ckpt, load_pairs(pairs_path), device)
        results["tasks"]["paires"] = res
        detail = "  ".join(f"{k} {v['accuracy']:.0%}" for k, v in res["par_phenomene"].items())
        acc = f"{res['accuracy']:.1%}" if res["accuracy"] is not None else "—"
        base = f"{res['baseline_longueur']:.1%}" if res["baseline_longueur"] is not None else "—"
        log(f"  paires   {acc}  (baseline longueur {base}) · {res['n']} paires"
            + (f", {res['non_couvertes']} hors vocabulaire" if res["non_couvertes"] else "")
            + f" ({time.time() - t0:.1f}s)")
        if detail:
            log(f"           {detail}")

    if "samples" in tasks:
        t0 = time.time()
        summary, rows = eval_samples(model, encode, decode, load_prompts(prompts_path),
                                     n_tokens=samples_tokens, seed=seed, device=device)
        results["tasks"]["samples"] = summary
        md = samples_markdown(rows, summary, os.path.basename(checkpoint))
        log(f"  samples  distinct-2 {summary['distinct2']} · répétitifs {summary['taux_repetitif']:.0%}"
            f" · {summary['n']} amorces ({time.time() - t0:.1f}s)")

    return results, md


# ─────────────────────────────────────────────────────────────────────────────
# popixa bench
# ─────────────────────────────────────────────────────────────────────────────

def _peak_memory(device: str) -> int:
    """Mémoire pic en octets (GPU si CUDA/MPS, sinon RSS du processus)."""
    if device == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "driver_allocated_memory"):
        return int(torch.mps.driver_allocated_memory())
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss if sys.platform == "darwin" else rss * 1024)   # octets sur macOS, Ko sur Linux
    except ImportError:
        return 0


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def run_bench(size: str = "nano", vocab_size: int = 50257, batch_size: int = 4,
              seconds: float = 15.0, gen_tokens: int = 128, device: str = None, seed: int = 0) -> dict:
    """
    Mesure le débit réel de la machine pour un preset (poids aléatoires, données aléatoires) :
      - entraînement : tokens/s (forward + backward + pas AdamW), batch × block_size tokens par pas
      - génération   : tokens/s en décodage normal et en speculative decoding (drafts n-grammes)
      - mémoire pic, TFLOPS effectifs ≈ (6·N + 12·L·T·d) × tokens/s  (N = paramètres)
    """
    from model import nanoPOPIXA, POPIXAConfig, SIZE_PRESETS

    device = device or _device()
    torch.manual_seed(seed)
    arch = SIZE_PRESETS[size]
    cfg = POPIXAConfig(vocab_size=vocab_size, dropout=0.0, **arch)
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):
        model = nanoPOPIXA(cfg).to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    T = cfg.block_size
    n_params = model.get_num_params(False)
    flops_per_token = 6 * n_params + 12 * cfg.n_layer * T * cfg.n_embd

    # ── Entraînement ──────────────────────────────────────────────────────────
    model.train(True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(vocab_size, (batch_size, T), device=device)
    y = torch.randint(vocab_size, (batch_size, T), device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()

    step()                      # échauffement (allocations, kernels)
    _sync(device)
    steps, t0 = 0, time.perf_counter()
    while steps < 2 or (time.perf_counter() - t0 < seconds / 2 and steps < 200):
        step()
        steps += 1
    _sync(device)
    train_dt = time.perf_counter() - t0
    train_tps = steps * batch_size * T / train_dt

    # ── Génération ────────────────────────────────────────────────────────────
    model.train(False)
    del opt

    def gen_rate(fn):
        _sync(device)
        t = time.perf_counter()
        n = len(list(fn()))
        _sync(device)
        return n / (time.perf_counter() - t)

    prompt = torch.randint(vocab_size, (1, 32), device=device)
    repetitive = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8] * 4], device=device)   # favorise les drafts
    n_gen = min(gen_tokens, T - 40)
    gen_tps = gen_rate(lambda: model.generate_stream(prompt, n_gen, temperature=0))
    spec_tps = gen_rate(lambda: model.speculative_generate_stream(repetitive, n_gen, temperature=0))

    return {
        "size":            size,
        "device":          device,
        "vocab_size":      vocab_size,
        "params":          n_params,
        "batch_size":      batch_size,
        "block_size":      T,
        "train_steps":     steps,
        "train_tok_s":     round(train_tps, 1),
        "train_tflops":    round(train_tps * flops_per_token / 1e12, 4),
        "gen_tok_s":       round(gen_tps, 1),
        "gen_spec_tok_s":  round(spec_tps, 1),
        "peak_memory_mb":  round(_peak_memory(device) / 2 ** 20, 1),
        "torch":           torch.__version__,
    }
