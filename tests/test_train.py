"""Test de bout en bout de train.py : entraînement court, sauvegarde finale, reprise, chat."""

import json
import os
import random
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 1 thread : évite la sur-souscription CPU (plusieurs process torch en parallèle)
ENV = dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="1")


def _train(tmp_path, *extra, size="nano"):
    size_args = ["--size", size] if size else []
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "train.py"), *size_args, "--batch_size", "2", *extra],
        cwd=str(tmp_path), env=ENV, capture_output=True, text=True, timeout=600,
    )


def test_train_save_resume_and_chat(tmp_path):
    random.seed(0)
    words = "le chat dort sur le canapé pendant que la pluie tombe".split()
    (tmp_path / "input.txt").write_text(" ".join(random.choice(words) for _ in range(2500)),
                                        encoding="utf-8")

    r = _train(tmp_path, "--max_iters", "3")
    assert r.returncode == 0, r.stdout + r.stderr
    ckpt_path = tmp_path / "out-nanopopixa" / "checkpoint.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert ckpt["iter"] == 3 and "vocab" in ckpt          # sauvegarde finale

    # Reprise d'un entraînement terminé : rien à faire, checkpoint intact
    mtime = os.path.getmtime(ckpt_path)
    r = _train(tmp_path, "--max_iters", "3", "--resume")
    assert r.returncode == 0 and "déjà terminé" in r.stdout
    assert os.path.getmtime(ckpt_path) == mtime

    # Reprise avec plus d'itérations et un --size différent : l'architecture du checkpoint gagne
    r = _train(tmp_path, "--max_iters", "5", "--resume", size="small")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "on garde le checkpoint" in r.stdout
    assert torch.load(str(ckpt_path), map_location="cpu", weights_only=False)["iter"] == 5

    # Nouvel entraînement : l'ancien modèle est sauvegardé, jamais écrasé en silence
    r = _train(tmp_path, "--max_iters", "2")
    assert r.returncode == 0 and (tmp_path / "out-nanopopixa" / "checkpoint.prev.pt").exists()

    # Le checkpoint produit est utilisable par le chat
    sys.path.insert(0, ROOT)
    import chat
    model, encode, decode, _ = chat.load_model(str(ckpt_path), "cpu")
    out = model.generate(torch.tensor([encode("le chat")]), 10)
    assert len(decode(out[0].tolist())) >= len("le chat")


def test_train_reports_bad_data_dir(tmp_path):
    r = _train(tmp_path, "--data_dir", "inexistant", "--max_iters", "1")
    assert r.returncode == 1 and "manquant" in r.stdout


def test_resume_rewrites_log_header(tmp_path):
    (tmp_path / "input.txt").write_text("abcdefgh " * 900, encoding="utf-8")
    assert _train(tmp_path, "--max_iters", "2").returncode == 0
    assert _train(tmp_path, "--max_iters", "4", "--resume").returncode == 0
    log = tmp_path / "out-nanopopixa" / "train.log"                    # le log vit dans le run
    headers = [l for l in log.read_text().splitlines() if l.startswith("#")]
    assert headers[-1].startswith("# max_iters=4")


def test_seed_makes_training_reproducible(tmp_path):
    text = "le chat dort sur le canapé pendant que la pluie tombe " * 120
    weights = {}
    for name, seed in (("a", "7"), ("b", "7"), ("c", "8")):
        d = tmp_path / name
        d.mkdir()
        (d / "input.txt").write_text(text, encoding="utf-8")
        r = _train(d, "--max_iters", "2", "--seed", seed)
        assert r.returncode == 0, r.stdout + r.stderr
        weights[name] = torch.load(str(d / "out-nanopopixa" / "checkpoint.pt"), map_location="cpu",
                                   weights_only=False)["model"]
    same = lambda x, y: all(torch.equal(weights[x][k], weights[y][k]) for k in weights[x])
    assert same("a", "b")                       # même graine → mêmes poids, bit à bit
    assert not same("a", "c")


def test_resume_without_size_keeps_checkpoint_size(tmp_path):
    """--resume sans --size : taille et réglages du run initial, pas le défaut small."""
    (tmp_path / "input.txt").write_text("abcdefgh " * 900, encoding="utf-8")
    assert _train(tmp_path, "--max_iters", "2").returncode == 0                  # nano
    ckpt = torch.load(str(tmp_path / "out-nanopopixa" / "checkpoint.pt"), weights_only=False)
    assert ckpt["train"]["size"] == "nano" and ckpt["train"]["max_iters"] == 2
    # Sans --max_iters : le run initial (2 itérations) est déjà terminé, pas 5 000 du preset
    for size in (None, "nano"):                     # sans --size, ou avec la taille enregistrée
        r = _train(tmp_path, "--resume", size=size)
        assert r.returncode == 0 and "déjà terminé (2/2" in r.stdout, r.stdout + r.stderr
    r = _train(tmp_path, "--max_iters", "3", "--resume", size=None)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[nano]" in r.stdout and "≠" not in r.stdout


def test_seed_out_of_numpy_range_is_accepted(tmp_path):
    (tmp_path / "input.txt").write_text("abcdefgh " * 900, encoding="utf-8")
    r = _train(tmp_path, "--max_iters", "1", "--seed", str(2 ** 33))
    assert r.returncode == 0, r.stdout + r.stderr


# ─── 2.3 « Socle » ───────────────────────────────────────────────────────────

def _words(n=3000, seed=0):
    random.seed(seed)
    w = "le chat dort sur le canapé pendant que la pluie tombe".split()
    return " ".join(random.choice(w) for _ in range(n))


def test_run_dir_best_checkpoint_safe_format_and_eval(tmp_path):
    """--run : tout dans runs/NOM ; best.pt et checkpoint.pt au format 2 (chargement sûr)."""
    (tmp_path / "input.txt").write_text(_words(), encoding="utf-8")
    r = _train(tmp_path, "--max_iters", "4", "--eval_interval", "2", "--run", "essai", "--eval")
    assert r.returncode == 0, r.stdout + r.stderr
    run = tmp_path / "runs" / "essai"
    assert {"checkpoint.pt", "best.pt", "train.log", "run.json", "eval.json", "samples.md"} <= \
        {p.name for p in run.iterdir()}
    for name in ("checkpoint.pt", "best.pt"):
        ck = torch.load(str(run / name), map_location="cpu", weights_only=True)   # aucun objet picklé
        assert ck["format"] == 2 and isinstance(ck["config"], dict)
    best = torch.load(str(run / "best.pt"), weights_only=True)
    assert "optimizer" not in best and best["iter"] == best["best_iter"]
    info = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert info["status"] == "terminé" and info["best_val"] == pytest.approx(best["val_loss"], abs=1e-5)
    assert set(json.loads((run / "eval.json").read_text(encoding="utf-8"))["tasks"]) == {"paires", "samples"}
    assert not (tmp_path / "out-nanopopixa").exists() and not (tmp_path / "train.log").exists()
    # Le chat charge best.pt (format 2)
    sys.path.insert(0, ROOT)
    import chat
    model, encode, decode, _ = chat.load_model(str(run / "best.pt"), "cpu")
    assert len(decode(model.generate(torch.tensor([encode("le")]), 5)[0].tolist())) >= 2


def test_weight_decay_only_on_matrices(tmp_path):
    (tmp_path / "input.txt").write_text(_words(), encoding="utf-8")
    assert _train(tmp_path, "--max_iters", "1").returncode == 0
    ck = torch.load(str(tmp_path / "out-nanopopixa" / "checkpoint.pt"), weights_only=True)
    groups = ck["optimizer"]["param_groups"]
    assert [g["weight_decay"] for g in groups] == [0.1, 0.0]
    shapes = {k: v.dim() for k, v in ck["model"].items() if "rotary" not in k}   # sans les tampons RoPE
    n_1d = sum(1 for k, d in shapes.items() if d < 2)
    assert len(groups[1]["params"]) == n_1d and n_1d > 0                 # poids des RMSNorm
    assert all(d >= 2 for k, d in shapes.items() if "norm" not in k and "ln_" not in k)


def test_early_stopping_and_resume(tmp_path):
    """La val loss se dégrade (val ≠ train) → arrêt anticipé ; --resume ne repart pas."""
    text = "ab" * 9000 + "xy" * 1100                  # les 10 % de validation : un autre motif
    (tmp_path / "input.txt").write_text(text, encoding="utf-8")
    r = _train(tmp_path, "--max_iters", "400", "--eval_interval", "5", "--patience", "2",
               "--eval_iters", "2", size="nano")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Arrêt anticipé" in r.stdout
    info = json.loads((tmp_path / "out-nanopopixa" / "run.json").read_text(encoding="utf-8"))
    assert info["status"] == "arrêt anticipé" and info["last_iter"] < 400
    r = _train(tmp_path, "--resume", size=None)
    assert r.returncode == 0 and "arrêté plus tôt" in r.stdout, r.stdout + r.stderr


def test_legacy_checkpoint_resumes(tmp_path):
    """Checkpoint 2.2 (config picklée, optimiseur à un seul groupe) : reprise possible."""
    (tmp_path / "input.txt").write_text(_words(), encoding="utf-8")
    assert _train(tmp_path, "--max_iters", "2").returncode == 0
    path = tmp_path / "out-nanopopixa" / "checkpoint.pt"
    ck = torch.load(str(path), weights_only=True)
    sys.path.insert(0, ROOT)
    from model import POPIXAConfig, nanoPOPIXA
    cfg = POPIXAConfig(**ck["config"])
    m = nanoPOPIXA(cfg)
    m.load_state_dict(ck["model"])
    legacy = {"model": ck["model"], "config": cfg, "iter": 2, "tokenizer": "char", "vocab": ck["vocab"],
              "optimizer": torch.optim.AdamW(m.parameters()).state_dict(), "data_dir": None}
    torch.save(legacy, str(path))
    r = _train(tmp_path, "--max_iters", "3", "--resume")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "moments AdamW réinitialisés" in r.stdout
    assert torch.load(str(path), weights_only=True)["format"] == 2      # réécrit au format 2


def test_memmap_keeps_ram_low(tmp_path):
    """Un gros train.bin n'est pas chargé en RAM (memmap) : pic mémoire quasi identique."""
    import pickle
    import numpy as np
    stoi = {c: i for i, c in enumerate("abcdefgh ")}

    def prep(name, n_tokens):
        d = tmp_path / name
        d.mkdir()
        rng = np.random.default_rng(0)
        rng.integers(0, len(stoi), n_tokens, dtype=np.uint16).tofile(str(d / "train.bin"))
        rng.integers(0, len(stoi), 5_000, dtype=np.uint16).tofile(str(d / "val.bin"))
        with open(d / "meta.pkl", "wb") as f:
            pickle.dump({"vocab_size": len(stoi), "tokenizer": "char", "stoi": stoi,
                         "itos": {i: c for c, i in stoi.items()}}, f)
        return str(d)

    def peak_kb(data_dir):
        r = subprocess.run(
            [sys.executable, "-c",
             "import runpy, sys, resource; sys.argv = sys.argv[1:]; "
             "runpy.run_path(sys.argv[0], run_name='__main__'); "
             "rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; "
             "print('MAXRSS', rss // 1024 if sys.platform == 'darwin' else rss)",   # octets sur macOS
             os.path.join(ROOT, "train.py"), "--size", "nano", "--batch_size", "2", "--max_iters", "1",
             "--data_dir", data_dir, "--out_dir", str(tmp_path / ("o" + os.path.basename(data_dir)))],
            cwd=str(tmp_path), env=ENV, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stdout + r.stderr
        return int(r.stdout.split("MAXRSS")[-1].split()[0])

    small, big = prep("petit", 20_000), prep("gros", 100_000_000)       # 200 Mo sur disque
    # Sans memmap : + 200 Mo (uint16) + 800 Mo (copie int64) ; avec : quelques pages lues
    assert peak_kb(big) - peak_kb(small) < 150_000, "train.bin chargé en mémoire ?"


def test_train_rejects_bad_options(tmp_path):
    (tmp_path / "input.txt").write_text(_words(), encoding="utf-8")
    r = _train(tmp_path, "--max_iters", "1", "--run", "../evasion")
    assert r.returncode == 2 and "simple nom" in r.stderr
    if not torch.cuda.is_available():                                           # float16 : CUDA seulement
        r = _train(tmp_path, "--max_iters", "1", "--dtype", "float16")
        assert r.returncode == 1 and "float16" in r.stdout
    r = _train(tmp_path, "--max_iters", "1", "--dtype", "bfloat16")                # autocast CPU
    assert r.returncode == 0, r.stdout + r.stderr


def _val_losses(log_path):
    out = {}
    for line in open(log_path, encoding="utf-8"):
        if line.startswith("iter"):
            parts = [p.split() for p in line.split("|")]
            out[int(parts[0][1])] = float(parts[2][1])
    return out


def test_best_pt_holds_the_best_validation_weights(tmp_path):
    """best.pt = poids à la PLUS BASSE val loss du train.log (jamais le modèle initial)."""
    text = "ab" * 9000 + "xy" * 1100                  # la val se dégrade : le meilleur point est tôt
    (tmp_path / "input.txt").write_text(text, encoding="utf-8")
    r = _train(tmp_path, "--max_iters", "30", "--eval_interval", "5", "--eval_iters", "2")
    assert r.returncode == 0, r.stdout + r.stderr
    run = tmp_path / "out-nanopopixa"
    vals = {i: v for i, v in _val_losses(run / "train.log").items() if i > 0}
    best = torch.load(str(run / "best.pt"), weights_only=True)
    best_it = min(vals, key=vals.get)
    assert best["iter"] == best_it and best["val_loss"] == pytest.approx(vals[best_it], abs=1e-4)
    assert best["iter"] < 30                                       # pas simplement le dernier


def test_resume_keeps_run_identity_and_applies_new_optimizer_settings(tmp_path):
    (tmp_path / "input.txt").write_text(_words(), encoding="utf-8")
    assert _train(tmp_path, "--max_iters", "2", "--seed", "5", "--eval_iters", "3",
                  "--eval").returncode == 0
    run = tmp_path / "out-nanopopixa"
    assert (run / "eval.json").exists()
    r = _train(tmp_path, "--max_iters", "4", "--resume", "--weight_decay", "0.05", "--beta2", "0.98")
    assert r.returncode == 0, r.stdout + r.stderr
    info = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert info["seed"] == 5 and info["weight_decay"] == 0.05            # graine du run, wd demandé
    ck = torch.load(str(run / "checkpoint.pt"), weights_only=True)
    assert ck["train"]["eval_iters"] == 3
    groups = ck["optimizer"]["param_groups"]
    assert [g["weight_decay"] for g in groups] == [0.05, 0.0] and tuple(groups[0]["betas"]) == (0.9, 0.98)
    # L'ancienne évaluation décrit d'autres poids : mise de côté
    assert not (run / "eval.json").exists() and (run / "eval.prev.json").exists()


def test_resume_rewrites_a_missing_or_foreign_best_pt(tmp_path):
    (tmp_path / "input.txt").write_text(_words(), encoding="utf-8")
    assert _train(tmp_path, "--max_iters", "2", "--eval_interval", "1").returncode == 0
    run = tmp_path / "out-nanopopixa"
    (run / "best.pt").unlink()
    r = _train(tmp_path, "--max_iters", "4", "--eval_interval", "1", "--resume")
    assert r.returncode == 0 and "il sera réécrit" in r.stdout, r.stdout + r.stderr
    best = torch.load(str(run / "best.pt"), weights_only=True)
    assert best["iter"] in (3, 4)                                   # réécrit par CE run


def test_interrupted_run_is_marked_in_run_json(tmp_path):
    import signal
    import time
    (tmp_path / "input.txt").write_text(_words(), encoding="utf-8")
    p = subprocess.Popen([sys.executable, os.path.join(ROOT, "train.py"), "--size", "nano", "--batch_size", "2",
                          "--max_iters", "100000", "--run", "coupe"],
                         cwd=str(tmp_path), env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 300
    for line in p.stdout:                                           # attendre que la boucle tourne
        if "iter     0" in line or time.time() > deadline:
            break
    time.sleep(1)
    p.send_signal(signal.SIGINT)
    p.communicate(timeout=120)
    info = json.loads((tmp_path / "runs" / "coupe" / "run.json").read_text(encoding="utf-8"))
    assert info["status"] == "interrompu" and info["last_iter"] >= 0
