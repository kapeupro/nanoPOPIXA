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


def test_legacy_checkpoint_with_pickled_config_still_loads(tmp_path):
    m = make_model()
    path = str(tmp_path / "old.pt")
    torch.save({"model": m.state_dict(), "config": m.config, "iter": 3}, path)   # format 2.0 – 2.2
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


@pytest.mark.parametrize("content,match", [(b"pas un checkpoint", "illisible"), (b"", "vide")])
def test_corrupt_or_empty_files_give_clear_errors(tmp_path, content, match):
    path = tmp_path / "bad.pt"
    path.write_bytes(content)
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
