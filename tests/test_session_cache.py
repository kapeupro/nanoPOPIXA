"""Tests du KV-cache persistant entre sessions (session_cache.py)."""

import os
import time

import torch

from conftest import make_model
from model import KVCache
from session_cache import clear_session, load_session, save_session, session_info


def _session(model, tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"dummy": True}, str(ckpt))
    cache_ref = []
    toks = list(model.generate_stream(torch.tensor([[1, 2, 3]]), 4, cache_ref=cache_ref))
    return str(ckpt), cache_ref[0], [1, 2, 3] + toks


def test_roundtrip_v3_with_exact_ids_and_history(model, tmp_path):
    ckpt, kv, ids = _session(model, tmp_path)
    path = str(tmp_path / "s.cache")
    save_session(path, kv, kv.token_ids, ckpt, history="bonjour")
    assert not os.path.exists(path + ".tmp")
    loaded, loaded_ids, history = load_session(path, ckpt, "cpu", with_history=True)
    assert isinstance(loaded, KVCache)
    assert loaded.token_ids == ids == loaded_ids and loaded.seq_len == len(ids)
    assert history == "bonjour"
    for (k, v), (rk, rv) in zip(loaded, kv):
        assert torch.equal(k, rk) and torch.equal(v, rv)
    info = session_info(path, ckpt)
    assert info["valid"] and info["n_tokens"] == len(ids)
    assert clear_session(path) and not os.path.exists(path)


def test_inconsistent_ids_are_flagged(model, tmp_path):
    ckpt, kv, _ = _session(model, tmp_path)
    path = str(tmp_path / "s.cache")
    save_session(path, kv, [0, 1, 2], ckpt)               # longueur ≠ cache
    loaded, ids = load_session(path, ckpt, "cpu")
    assert loaded is not None and ids == [0, 1, 2]
    assert loaded.token_ids is None                        # l'appelant doit reconstruire


def test_v2_payload_is_readable_but_not_trusted(model, tmp_path):
    ckpt, kv, ids = _session(model, tmp_path)
    path = str(tmp_path / "s.cache")
    from session_cache import _checkpoint_fingerprint
    torch.save({"version": 2, "ckpt_fp": _checkpoint_fingerprint(ckpt), "token_ids": ids,
                "past_kvs": [(k, v) for k, v in kv]}, path)
    loaded, loaded_ids = load_session(path, ckpt, "cpu")
    assert loaded is not None and loaded.token_ids is None and loaded_ids == ids
    assert session_info(path, ckpt)["valid"]


def test_invalidation_cases(model, tmp_path):
    ckpt, kv, _ = _session(model, tmp_path)
    path = str(tmp_path / "s.cache")
    assert load_session(path, ckpt, "cpu") == (None, [])                     # absent
    save_session(path, kv, kv.token_ids, ckpt)
    time.sleep(0.01)
    torch.save({"dummy": False, "more": 1}, ckpt)                            # modèle changé
    assert load_session(path, ckpt, "cpu") == (None, [])
    assert not session_info(path, ckpt)["valid"]
    with open(path, "wb") as f:
        f.write(b"corrompu")                                                 # fichier illisible
    assert load_session(path, ckpt, "cpu", with_history=True) == (None, [], None)


def test_saved_tensors_are_compact(tmp_path):
    """Une vue tronquée du cache ne doit pas sérialiser tout le stockage sous-jacent."""
    big = torch.zeros(1, 2, 1000, 16)
    kv = [(big[:, :, :3], big[:, :, :3])]
    ckpt = tmp_path / "ckpt.pt"
    torch.save({}, str(ckpt))
    path = str(tmp_path / "s.cache")
    save_session(path, kv, [1, 2, 3], str(ckpt))
    assert os.path.getsize(path) < 20_000
