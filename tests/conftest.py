"""Configuration pytest — rend les modules du dépôt importables et fournit des fixtures communes."""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model import nanoPOPIXA, POPIXAConfig  # noqa: E402


def make_model(block_size=32, vocab_size=65, n_layer=2, n_head=2, n_embd=64, seed=0):
    """Petit modèle aléatoire, en mode inférence (dropout désactivé)."""
    torch.manual_seed(seed)
    cfg = POPIXAConfig(block_size=block_size, vocab_size=vocab_size, n_layer=n_layer,
                       n_head=n_head, n_embd=n_embd, dropout=0.0)
    model = nanoPOPIXA(cfg)
    model.train(False)
    return model


@pytest.fixture
def model():
    return make_model()


CHARS = sorted(set(
    "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ".,;:!?'\"{}[]-_\néèàç\\/"
))


@pytest.fixture
def char_checkpoint(tmp_path):
    """Checkpoint char-level minimal (format train.py) écrit dans tmp_path."""
    def _make(block_size=128, seed=0):
        stoi = {c: i for i, c in enumerate(CHARS)}
        itos = {i: c for c, i in stoi.items()}
        m = make_model(block_size=block_size, vocab_size=len(CHARS), n_embd=32, seed=seed)
        path = tmp_path / "out-nanopopixa" / "checkpoint.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": m.state_dict(), "config": m.config, "iter": 0,
                    "tokenizer": "char", "vocab": {"stoi": stoi, "itos": itos}}, str(path))
        return str(path)
    return _make
