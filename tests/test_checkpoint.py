"""Tests du format de checkpoint 2 (checkpoint.py) : chargement sûr, compatibilité, erreurs."""

import os

import pytest
import torch

from conftest import make_model
from checkpoint import FORMAT, config_from, config_to_dict, load_checkpoint, save_checkpoint
from model import POPIXAConfig


def test_format2_roundtrip_is_loadable_with_weights_only(tmp_path):
    m = make_model()
    opt = torch.optim.AdamW(m.parameters())
    path = str(tmp_path / "c.pt")
    save_checkpoint(path, {"model": m.state_dict(), "config": m.config, "iter": 7, "optimizer": opt.state_dict(),
                           "vocab": {"stoi": {"a": 0}, "itos": {0: "a"}}, "train": {"size": "nano"}})
    assert not os.path.exists(path + ".tmp")
    raw = torch.load(path, weights_only=True)                 # aucun objet picklé
    assert raw["format"] == FORMAT and isinstance(raw["config"], dict)
    ck = load_checkpoint(path)
    assert ck["config"] == m.config and ck["iter"] == 7 and ck["vocab"]["itos"][0] == "a"
    m2 = make_model(seed=1)
    m2.load_state_dict(ck["model"])
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))


def test_legacy_checkpoint_with_pickled_config_still_loads(tmp_path, monkeypatch):
    m = make_model()
    path = str(tmp_path / "old.pt")
    torch.save({"model": m.state_dict(), "config": m.config, "iter": 3}, path)   # format 2.0 – 2.2
    if not hasattr(torch.serialization, "add_safe_globals"):                     # PyTorch < 2.4
        monkeypatch.setenv("POPIXA_TRUST_CHECKPOINTS", "1")
    ck = load_checkpoint(path)
    assert isinstance(ck["config"], POPIXAConfig) and ck["config"] == m.config and ck["iter"] == 3


def test_malicious_checkpoint_is_refused_without_running_code(tmp_path):
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (open, (str(marker), "w"))

    path = str(tmp_path / "evil.pt")
    torch.save({"model": {}, "x": Evil()}, path)
    with pytest.raises(ValueError, match="refusé"):
        load_checkpoint(path)
    assert not marker.exists()                                 # le code n'a jamais tourné


def _truncated_checkpoint(tmp_path):
    good = tmp_path / "good.pt"
    save_checkpoint(str(good), {"model": make_model().state_dict(), "iter": 1})
    data = good.read_bytes()
    return data[:len(data) // 2]


@pytest.mark.parametrize("content,match", [(b"pas un checkpoint", "illisible"), (b"", "vide"),
                                           ("tronque", "corrompu ou tronqué")])
def test_corrupt_or_empty_files_give_clear_errors(tmp_path, content, match):
    path = tmp_path / "bad.pt"
    path.write_bytes(_truncated_checkpoint(tmp_path) if content == "tronque" else content)
    with pytest.raises(ValueError, match=match):
        load_checkpoint(str(path))
    with pytest.raises(FileNotFoundError):
        load_checkpoint(str(tmp_path / "absent.pt"))


def test_config_conversions_tolerate_versions():
    cfg = POPIXAConfig(vocab_size=50, n_layer=2, n_head=2, n_embd=32, block_size=16)
    d = config_to_dict(cfg)
    assert config_from(d) == cfg and config_from(cfg) is cfg and config_from(None) is None
    d_new = dict(d, champ_futur=123)                           # checkpoint d'une version plus récente
    assert config_from(d_new) == cfg
    d_old = {k: v for k, v in d.items() if k != "rope_base"}   # checkpoint plus ancien
    assert config_from(d_old).rope_base == POPIXAConfig.rope_base


def test_old_pytorch_never_falls_back_to_unsafe_load_silently(tmp_path, monkeypatch):
    """PyTorch < 2.4 (pas de safe_globals) : un ancien checkpoint demande un accord explicite,
    un fichier piégé reste refusé même avec cet accord s'il ne ressemble pas à un checkpoint."""
    for name in ("safe_globals", "add_safe_globals"):
        monkeypatch.delattr(torch.serialization, name, raising=False)
    monkeypatch.delenv("POPIXA_TRUST_CHECKPOINTS", raising=False)
    m = make_model()
    old = str(tmp_path / "old.pt")
    torch.save({"model": m.state_dict(), "config": m.config, "iter": 1}, old)
    with pytest.raises(ValueError, match="POPIXA_TRUST_CHECKPOINTS"):
        load_checkpoint(old)

    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (open, (str(marker), "w"))

    evil = str(tmp_path / "evil.pt")
    torch.save({"model": {}, "x": Evil()}, evil)
    monkeypatch.setenv("POPIXA_TRUST_CHECKPOINTS", "1")
    with pytest.raises(ValueError, match="refusé"):
        load_checkpoint(evil)
    assert not marker.exists()
    logs = []
    ck = load_checkpoint(old, log=logs.append)                 # accord explicite : chargé, signalé
    assert ck["config"] == m.config and any("SANS protection" in l for l in logs)


def test_directory_and_newer_format(tmp_path):
    with pytest.raises(ValueError, match="dossier"):
        load_checkpoint(str(tmp_path))
    m = make_model()
    path = str(tmp_path / "futur.pt")
    torch.save({"format": FORMAT + 1, "model": m.state_dict(), "config": dict(config_to_dict(m.config), x=1)},
               path)
    logs = []
    assert load_checkpoint(path, log=logs.append)["config"] == m.config
    assert any("plus récente" in l for l in logs)


@pytest.mark.parametrize("error", [OSError(28, "No space left on device"), KeyboardInterrupt()])
def test_failed_save_keeps_the_previous_file_and_leaves_no_tmp(tmp_path, monkeypatch, error):
    """Écriture interrompue (disque plein, Ctrl+C) : l'ancien checkpoint reste intact, pas de .tmp partiel."""
    import checkpoint
    path = str(tmp_path / "checkpoint.pt")
    save_checkpoint(path, {"model": {}, "iter": 1})
    before = open(path, "rb").read()

    def partial_save(obj, f):
        with open(f, "wb") as out:
            out.write(b"partiel")
        raise error
    monkeypatch.setattr(checkpoint.torch, "save", partial_save)
    with pytest.raises(type(error)):
        save_checkpoint(path, {"model": {}, "iter": 2})
    assert open(path, "rb").read() == before and not os.path.exists(path + ".tmp")
