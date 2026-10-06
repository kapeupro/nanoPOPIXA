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
    with pytest.raises(ValueError, match="vocabulaire"):              # données d'un autre tokenizer
        pe.eval_bpb(model, np.array([1, 2, 500, 3], dtype=np.uint16), nbytes)


def test_max_bytes_selects_same_text_for_any_tokenizer():
    """--max_bytes : même extrait de texte (en octets) que le tokenizer soit caractère ou BPE."""
    tiktoken = pytest.importorskip("tiktoken")
    try:
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        pytest.skip("encodage gpt2 indisponible (hors ligne)")
    stoi, itos = _char_codec()
    char_nb = pe.token_nbytes({"tokenizer": "char", "vocab": {"stoi": stoi, "itos": itos}}, len(CHARS))
    gpt_nb = pe.token_nbytes({"tokenizer": "tiktoken_gpt2"}, 50304)
    r_c = pe.eval_bpb(make_model(block_size=64, vocab_size=len(CHARS)),
                      np.array([stoi[c] for c in TEXT], dtype=np.uint16), char_nb, max_bytes=300)
    r_g = pe.eval_bpb(make_model(block_size=64, vocab_size=50304, n_embd=32),
                      np.array(enc.encode_ordinary(TEXT), dtype=np.uint16), gpt_nb, max_bytes=300)
    assert r_c["bytes"] == 300 and 290 <= r_g["bytes"] <= 300          # frontière de token BPE
    assert r_g["tokens"] < r_c["tokens"]


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


def test_eval_pairs_counts_the_good_sentence_as_correct():
    """Sens de la comparaison : un modèle non entraîné préfère la phrase la plus courte."""
    stoi, _ = _char_codec()
    enc = lambda s: [stoi.get(c, 0) for c in s]
    ckpt = {"tokenizer": "char", "vocab": {"stoi": stoi}}
    model = make_model(block_size=64, vocab_size=len(CHARS))
    court_bon = [{"good": "Il dort.", "bad": "Il dorment bien.", "phenomene": "x"}]
    court_faux = [{"good": "Ils dorment bien.", "bad": "Ils dort.", "phenomene": "x"}]
    assert pe.eval_pairs(model, enc, ckpt, court_bon)["accuracy"] == 1.0
    assert pe.eval_pairs(model, enc, ckpt, court_faux)["accuracy"] == 0.0
    # Égalité exacte avec le calcul direct des log-probabilités
    pairs = court_bon + court_faux + [{"good": "Je l'aime.", "bad": "Je le aime.", "phenomene": "y"}]
    prefix = enc("\n")
    expected = sum(pe.sequence_logprob(model, prefix, enc(p["good"])) >
                   pe.sequence_logprob(model, prefix, enc(p["bad"])) for p in pairs) / len(pairs)
    assert pe.eval_pairs(model, enc, ckpt, pairs)["accuracy"] == round(expected, 4)


def test_eval_pairs_scores_and_skips_out_of_vocabulary(tmp_path):
    stoi, itos = _char_codec()
    ckpt = {"tokenizer": "char", "vocab": {"stoi": stoi, "itos": itos}}
    model = make_model(block_size=64, vocab_size=len(CHARS))
    r = pe.eval_pairs(model, lambda s: [stoi.get(c, 0) for c in s], ckpt, pe.load_pairs(_pairs_file(tmp_path)))
    assert r["n"] == 4 and r["non_couvertes"] == 1
    assert set(r["par_phenomene"]) == {"accord_sujet_verbe", "participe_passe", "elision"}
    assert 0.0 <= r["accuracy"] <= 1.0
    assert r["ic95"][0] <= r["accuracy"] <= r["ic95"][1]
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

def test_pairs_and_prompts_files_are_validated(tmp_path):
    bad = tmp_path / "p.jsonl"
    bad.write_text('{"good": "A.", "bad": "B."}\n', encoding="utf-8")             # phenomene manquant
    with pytest.raises(ValueError, match=":1"):
        pe.load_pairs(str(bad))
    bad.write_text('["A.", "B."]\n', encoding="utf-8")
    with pytest.raises(ValueError):
        pe.load_pairs(str(bad))
    bad.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="aucune paire"):
        pe.load_pairs(str(bad))
    empty = tmp_path / "a.txt"
    empty.write_text("# seulement un commentaire\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="aucune amorce"):
        pe.load_prompts(str(empty))


def test_weights_fingerprint_depends_on_weights_only(tmp_path, char_checkpoint):
    """Mêmes poids → même empreinte, même copiés ailleurs ; poids différents → empreinte différente."""
    import shutil
    ckpt = char_checkpoint(block_size=64)
    copy = tmp_path / "ailleurs" / "checkpoint.pt"
    copy.parent.mkdir()
    shutil.copy(ckpt, copy)
    os.utime(copy, (1, 1))
    pairs = _pairs_file(tmp_path)
    run = lambda c: pe.run_eval(c, tasks=("paires",), pairs_path=pairs, device="cpu", log=lambda m: None)[0]
    assert run(ckpt) == run(str(copy))
    sd = torch.load(ckpt, weights_only=False)["model"]
    assert pe.weights_fingerprint(sd) == pe.weights_fingerprint({k: v.clone() for k, v in sd.items()})
    sd2 = dict(sd)
    k0 = sorted(sd2)[0]
    sd2[k0] = sd2[k0] + 1e-3
    assert pe.weights_fingerprint(sd) != pe.weights_fingerprint(sd2)


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
    # 12 tokens : trop court pour les détecteurs → None, pas un faux 0 %
    assert summary["taux_repetitif"] is None and summary["taux_diminishing"] is None
    md = pe.samples_markdown(rows, summary, "ckpt.pt")
    assert md.count("```text") == 3 and prompts[0] in md and "répétitifs : —" in md
    long_summary, _ = pe.eval_samples(model, encode, decode, prompts[:1], n_tokens=120, seed=7)
    assert long_summary["taux_repetitif"] is not None and long_summary["taux_diminishing"] is not None


def test_samples_flag_prompts_altered_by_the_tokenizer():
    """Vocabulaire caractère sans accents : l'amorce vue par le modèle est signalée, pas cachée."""
    vocab = sorted(set("abcdefghijklmnopqrstuvwxyz ILM,.\n"))
    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for c, i in stoi.items()}
    model = make_model(block_size=64, vocab_size=len(vocab))
    summary, rows = pe.eval_samples(model, lambda s: [stoi.get(c, 0) for c in s],
                                    lambda ids: "".join(itos[i] for i in ids),
                                    ["Il dort", "Il était une fois"], n_tokens=5)
    assert summary["amorces_alterees"] == 1
    assert rows[0]["amorce_vue"] is None and rows[1]["amorce_vue"] == "Il \ntait une fois"
    md = pe.samples_markdown(rows, summary, "c.pt")
    assert "amorce altérée" in md and "Il \ntait une fois" in md

def test_wilson_interval():
    assert pe.wilson_ic95(0, 0) is None
    lo, hi = pe.wilson_ic95(208, 338)                    # 61.5 % sur 338 paires
    assert 0.56 < lo < 0.57 and 0.66 < hi < 0.67
    assert pe.wilson_ic95(0, 10)[0] == 0.0 and pe.wilson_ic95(10, 10)[1] == 1.0


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

    def fails(*args, msg):
        r = _popixa("eval", "--checkpoint", ckpt, *args, cwd=tmp_path)
        assert r.returncode == 1, r.stderr
        assert "Traceback" not in r.stderr and "✗" in r.stderr and msg in r.stderr, r.stderr

    fails("--tasks", "paires,inconnue", msg="inconnue")
    fails("--data_dir", str(tmp_path / "absent"), "--tasks", "bpb", msg="introuvable")
    fails("--tasks", "bpb", msg="aucune tâche")                                   # bpb sans --data_dir
    fails("--tasks", "paires", "--out", str(tmp_path / "absent" / "e.json"), msg="--out")
    nometa = tmp_path / "nometa"
    nometa.mkdir()
    np.zeros(100, dtype=np.uint16).tofile(str(nometa / "val.bin"))
    fails("--data_dir", str(nometa), "--tasks", "bpb", msg="meta.pkl")
    bad_pairs = tmp_path / "bad.jsonl"
    bad_pairs.write_text('{"good": 1}\n', encoding="utf-8")
    fails("--tasks", "paires", "--pairs", str(bad_pairs), msg="paire invalide")
    r = _popixa("eval", "--checkpoint", str(tmp_path / "absent.pt"), cwd=tmp_path)
    assert r.returncode == 1 and "Traceback" not in r.stderr
    r = _popixa("eval", "--checkpoint", ckpt, "--samples_tokens", "0", cwd=tmp_path)
    assert r.returncode == 2 and "doit être > 0" in r.stderr                       # argparse


def test_cli_version():
    import popixa_cli
    assert pe.popixa_version() == popixa_cli.popixa_version()
    r = _popixa("--version", cwd=ROOT)
    assert r.returncode == 0 and r.stdout.strip() == f"nanoPOPIXA {pe.popixa_version()}"
    with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as f:
        assert f'version = "{pe.popixa_version()}"' in f.read()


# ─── bench ───────────────────────────────────────────────────────────────────

def test_bench_quick_run():
    r = pe.run_bench(size="nano", vocab_size=64, batch_size=1, seconds=0.1, gen_tokens=8, device="cpu")
    assert r["size"] == "nano" and r["block_size"] == 512 and r["train_steps"] >= 2
    for k in ("train_tok_s", "train_tflops", "gen_tok_s", "peak_memory_mb"):
        assert r[k] > 0, k
    assert r["dropout"] == 0.1 and r["memory_kind"]
    json.dumps(r)


def test_cli_bench_validates_arguments():
    for args in (("--batch", "0"), ("--vocab", "-3")):
        r = _popixa("bench", *args, cwd=ROOT)
        assert r.returncode == 2 and "doit être > 0" in r.stderr
    r = _popixa("bench", "--dropout", "1.5", cwd=ROOT)
    assert r.returncode == 1 and "Traceback" not in r.stderr
