"""Tests de popixa eval / popixa bench (popixa_eval.py) et du jeu de paires minimales."""

import json
import math
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import CHARS, make_model
import popixa_eval as pe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="1")

TEXT = "\nLe chat a mangé la souris. Les enfants jouent dans le jardin, près de l'école ! " * 12


def _char_codec():
    stoi = {c: i for i, c in enumerate(CHARS)}
    return stoi, {i: c for c, i in stoi.items()}


def _data_dir(tmp_path, text=TEXT, name="data"):
    stoi, itos = _char_codec()
    d = tmp_path / name
    d.mkdir()
    np.array([stoi[c] for c in text], dtype=np.uint16).tofile(str(d / "val.bin"))
    with open(d / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": len(CHARS), "tokenizer": "char", "stoi": stoi, "itos": itos}, f)
    return str(d)


def _pairs_file(tmp_path):
    path = tmp_path / "paires.jsonl"
    rows = [("Le chat dort.", "Le chat dorment.", "accord_sujet_verbe"),
            ("Les chats dorment.", "Les chats dort.", "accord_sujet_verbe"),
            ("Elle est partie.", "Elle est parti.", "participe_passe"),
            ("Je l'aime.", "Je le aime.", "elision"),
            ("Il parle à Ŋ.", "Il parle de Ŋ.", "prepositions")]          # Ŋ hors vocabulaire
    path.write_text("".join(json.dumps({"good": g, "bad": b, "phenomene": p, "gabarit": "t"},
                                       ensure_ascii=False) + "\n" for g, b, p in rows), encoding="utf-8")
    return str(path)


# ─── bpb ─────────────────────────────────────────────────────────────────────

def test_bpb_counts_exact_bytes_and_ignores_batching():
    stoi, itos = _char_codec()
    model = make_model(block_size=32, vocab_size=len(CHARS))
    data = np.array([stoi[c] for c in TEXT], dtype=np.uint16)
    nbytes = pe.token_nbytes({"tokenizer": "char", "vocab": {"stoi": stoi, "itos": itos}}, len(CHARS))

    r = pe.eval_bpb(model, data, nbytes, batch_size=8)
    assert r["tokens"] == len(TEXT) - 1
    assert r["bytes"] == len(TEXT[1:].encode("utf-8")) > r["tokens"]          # « é » = 2 octets
    assert r["windows"] == math.ceil((len(TEXT) - 1) / 32)
    assert r["bpb"] == pytest.approx(r["loss"] * r["tokens"] / (math.log(2) * r["bytes"]), rel=1e-5)
    # Le regroupement en batch ne change pas le résultat
    assert pe.eval_bpb(model, data, nbytes, batch_size=1) == pytest.approx(r, rel=1e-5)

    # Référence : perte moyenne recalculée fenêtre par fenêtre
    x = torch.from_numpy(data.astype(np.int64))
    nll = 0.0
    with torch.no_grad():
        for s in range(0, len(TEXT) - 1, 32):
            n = min(32, len(TEXT) - 1 - s)
            logits, _ = model(x[s:s + n][None], return_all_logits=True)
            nll += float(F.cross_entropy(logits[0], x[s + 1:s + n + 1], reduction="sum"))
    assert r["loss"] == pytest.approx(nll / r["tokens"], rel=1e-5)

    short = pe.eval_bpb(model, data, nbytes, max_tokens=50)
    assert short["tokens"] == 50 and short["windows"] == 2
    with pytest.raises(ValueError):
        pe.eval_bpb(model, data[:1], nbytes)


def test_bpb_denominator_is_tokenizer_independent():
    """Même texte, deux tokenizers : même nombre d'octets prédits (bpb comparable)."""
    tiktoken = pytest.importorskip("tiktoken")
    try:
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        pytest.skip("encodage gpt2 indisponible (hors ligne)")
    stoi, itos = _char_codec()
    char_ids = np.array([stoi[c] for c in TEXT], dtype=np.uint16)
    gpt_ids = np.array(enc.encode_ordinary(TEXT), dtype=np.uint16)
    assert gpt_ids[0] == enc.encode_ordinary("\n")[0]                     # 1er token = « \n » des deux côtés

    r_char = pe.eval_bpb(make_model(block_size=64, vocab_size=len(CHARS)), char_ids,
                         pe.token_nbytes({"tokenizer": "char", "vocab": {"stoi": stoi, "itos": itos}},
                                         len(CHARS)))
    r_gpt = pe.eval_bpb(make_model(block_size=64, vocab_size=50304, n_embd=32), gpt_ids,
                        pe.token_nbytes({"tokenizer": "tiktoken_gpt2"}, 50304))
    assert r_char["bytes"] == r_gpt["bytes"] == len(TEXT.encode("utf-8")) - 1
    assert r_gpt["tokens"] < r_char["tokens"]


def test_load_split_checks_tokenizer(tmp_path):
    stoi, itos = _char_codec()
    d = _data_dir(tmp_path)
    ok = {"tokenizer": "char", "vocab": {"stoi": stoi, "itos": itos}}
    assert len(pe.load_split(d, "val", ok)) == len(TEXT)
    with pytest.raises(ValueError):
        pe.load_split(d, "val", {"tokenizer": "tiktoken_gpt2"})
    with pytest.raises(ValueError):
        pe.load_split(d, "val", {"tokenizer": "char", "vocab": {"stoi": {"a": 0}, "itos": {0: "a"}}})
    with pytest.raises(FileNotFoundError):
        pe.load_split(d, "test", ok)


# ─── paires ──────────────────────────────────────────────────────────────────

def test_sequence_logprob_matches_manual_sum():
    model = make_model(block_size=16, vocab_size=65)
    prefix, ids = [3], [5, 7, 9, 11]
    with torch.no_grad():
        logits, _ = model(torch.tensor([[3, 5, 7, 9]]), return_all_logits=True)
    logp = F.log_softmax(logits[0], dim=-1)
    manual = sum(float(logp[i, t]) for i, t in enumerate(ids))
    assert pe.sequence_logprob(model, prefix, ids) == pytest.approx(manual, rel=1e-5)
    # Au-delà de block_size : on garde la fin, sans planter
    assert math.isfinite(pe.sequence_logprob(model, prefix, list(range(1, 40))))


def test_eval_pairs_scores_and_skips_out_of_vocabulary(tmp_path):
    stoi, itos = _char_codec()
    ckpt = {"tokenizer": "char", "vocab": {"stoi": stoi, "itos": itos}}
    model = make_model(block_size=64, vocab_size=len(CHARS))
    r = pe.eval_pairs(model, lambda s: [stoi.get(c, 0) for c in s], ckpt, pe.load_pairs(_pairs_file(tmp_path)))
    assert r["n"] == 4 and r["non_couvertes"] == 1
    assert set(r["par_phenomene"]) == {"accord_sujet_verbe", "participe_passe", "elision"}
    assert 0.0 <= r["accuracy"] <= 1.0
    # Baseline longueur : « dort » < « dorment », « dorment » > « dort », « partie » > « parti »,
    # « l'aime » < « le aime » → 2 sur 4
    assert r["baseline_longueur"] == 0.5


def test_fr_pairs_dataset_is_built_and_valid():
    sys.path.insert(0, os.path.join(ROOT, "evals"))
    import build_paires
    assert build_paires.main(["--check"]) == 0, "python evals/build_paires.py"
    pairs = pe.load_pairs()
    assert build_paires.validate(pairs) == []
    counts = {}
    for p in pairs:
        counts[p["phenomene"]] = counts.get(p["phenomene"], 0) + 1
    assert set(counts) == set(build_paires.PHENOMENES)
    assert all(n >= 50 for n in counts.values()), counts


def test_untrained_model_is_near_length_baseline():
    """Un modèle non entraîné ne « comprend » rien : il suit la longueur, pas la grammaire."""
    tiktoken = pytest.importorskip("tiktoken")
    try:
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        pytest.skip("encodage gpt2 indisponible (hors ligne)")
    model = make_model(block_size=64, vocab_size=50304, n_embd=32)
    r = pe.eval_pairs(model, enc.encode_ordinary, {"tokenizer": "tiktoken_gpt2"}, pe.load_pairs())
    assert r["non_couvertes"] == 0
    assert 0.3 <= r["baseline_longueur"] <= 0.7, r          # le jeu n'est pas trivial par la longueur
    assert abs(r["accuracy"] - r["baseline_longueur"]) <= 0.1, r


# ─── samples ─────────────────────────────────────────────────────────────────

def test_samples_are_deterministic():
    stoi, itos = _char_codec()
    model = make_model(block_size=64, vocab_size=len(CHARS))
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
    prompts = pe.load_prompts()
    assert len(prompts) == 20 and all(p and not p.startswith("#") for p in prompts)
    a = pe.eval_samples(model, encode, decode, prompts[:3], n_tokens=12, seed=7)
    b = pe.eval_samples(model, encode, decode, prompts[:3], n_tokens=12, seed=7)
    assert a == b
    summary, rows = a
    assert summary["n"] == 3 and all(len(r["text"]) == 12 for r in rows)
    assert pe.eval_samples(model, encode, decode, prompts[:3], n_tokens=12, seed=8)[1] != rows
    md = pe.samples_markdown(rows, summary, "ckpt.pt")
    assert md.count("```text") == 3 and prompts[0] in md


def test_distinct2():
    assert pe.distinct2([]) == 0.0
    assert pe.distinct2([1, 2, 3, 4]) == 1.0
    assert pe.distinct2([1, 2] * 5) == pytest.approx(2 / 9)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _popixa(*args, cwd):
    return subprocess.run([sys.executable, os.path.join(ROOT, "popixa_cli.py"), *args],
                          cwd=str(cwd), env=ENV, capture_output=True, text=True, timeout=300)


def test_cli_eval_writes_reproducible_json(tmp_path, char_checkpoint):
    ckpt = char_checkpoint(block_size=64)
    d = _data_dir(tmp_path)
    pairs = _pairs_file(tmp_path)
    args = ["eval", "--checkpoint", ckpt, "--data_dir", d, "--pairs", pairs,
            "--samples_tokens", "10", "--samples_out", "s.md"]
    r1 = _popixa(*args, "--out", "e1.json", cwd=tmp_path)
    assert r1.returncode == 0, r1.stdout + r1.stderr
    r2 = _popixa(*args, "--out", "e2.json", cwd=tmp_path)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    e1 = (tmp_path / "e1.json").read_text(encoding="utf-8")
    assert e1 == (tmp_path / "e2.json").read_text(encoding="utf-8")       # aucune date / durée
    res = json.loads(e1)
    assert set(res["tasks"]) == {"bpb", "paires", "samples"}
    assert res["checkpoint"]["tokenizer"] == "char" and res["checkpoint"]["block_size"] == 64
    assert res["tasks"]["bpb"]["tokens"] == len(TEXT) - 1
    assert (tmp_path / "s.md").read_text(encoding="utf-8").startswith("# nanoPOPIXA")

    # Sans --data_dir : bpb ignorée, JSON sur la sortie standard
    r = _popixa("eval", "--checkpoint", ckpt, "--pairs", pairs, "--tasks", "bpb,paires", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert set(json.loads(r.stdout)["tasks"]) == {"paires"} and "bpb ignorée" in r.stderr


def test_cli_eval_reports_errors(tmp_path, char_checkpoint):
    ckpt = char_checkpoint(block_size=64)
    r = _popixa("eval", "--checkpoint", ckpt, "--tasks", "paires,inconnue", cwd=tmp_path)
    assert r.returncode == 1 and "inconnue" in r.stderr
    r = _popixa("eval", "--checkpoint", ckpt, "--data_dir", str(tmp_path / "absent"), "--tasks", "bpb",
                cwd=tmp_path)
    assert r.returncode == 1 and "introuvable" in r.stderr
    r = _popixa("eval", "--checkpoint", str(tmp_path / "absent.pt"), cwd=tmp_path)
    assert r.returncode == 1


def test_cli_version():
    r = _popixa("--version", cwd=ROOT)
    assert r.returncode == 0 and r.stdout.strip() == f"nanoPOPIXA {pe.popixa_version()}"
    with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as f:
        assert f'version = "{pe.popixa_version()}"' in f.read()


# ─── bench ───────────────────────────────────────────────────────────────────

def test_bench_quick_run():
    r = pe.run_bench(size="nano", vocab_size=64, batch_size=1, seconds=0.1, gen_tokens=8, device="cpu")
    assert r["size"] == "nano" and r["block_size"] == 512 and r["train_steps"] >= 2
    for k in ("train_tok_s", "train_tflops", "gen_tok_s", "gen_spec_tok_s", "peak_memory_mb"):
        assert r[k] > 0, k
    json.dumps(r)
