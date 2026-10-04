"""Test de bout en bout de train.py : entraînement court, sauvegarde finale, reprise, chat."""

import os
import random
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 1 thread : évite la sur-souscription CPU (plusieurs process torch en parallèle)
ENV = dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="1")


def _train(tmp_path, *extra, size="nano"):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "train.py"), "--size", size, "--batch_size", "2", *extra],
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
    headers = [l for l in (tmp_path / "train.log").read_text().splitlines() if l.startswith("#")]
    assert headers[-1].startswith("# max_iters=4")
