"""Tests du chat interactif : streaming UTF-8, gestion du contexte, session de bout en bout."""

import builtins
import re

import pytest
import torch

import chat
from conftest import CHARS
from session_cache import load_session

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def run_script(monkeypatch, capsys, checkpoint, lines, max_tokens=20):
    """Pilote run_chat avec une saisie scriptée ; retourne la sortie sans couleurs."""
    it = iter(lines)

    def fake_input(*_):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(builtins, "input", fake_input)
    chat.run_chat(checkpoint, max_tokens, 0.8, 40)
    return ANSI.sub("", capsys.readouterr().out)


# ── Streaming ────────────────────────────────────────────────────────────────

def test_text_stream_never_splits_multibyte_characters():
    tiktoken = pytest.importorskip("tiktoken")
    enc = tiktoken.get_encoding("gpt2")
    text = "Ça marche 🙂 très bien — ünïcödé ✓"
    ids = enc.encode_ordinary(text)
    stream = chat._TextStream(enc.decode)
    pieces = [stream.push(t) for t in ids] + [stream.flush()]
    assert all("�" not in p for p in pieces)
    assert "".join(pieces) == text == stream.text


# ── Gestion du contexte ──────────────────────────────────────────────────────

def test_update_context_counts_real_tokens():
    encode = lambda s: list(s.encode("utf-8"))          # 1 octet = 1 token (char-level)
    decode = lambda ids: bytes(ids).decode("utf-8", errors="ignore")
    history, compacted = chat.update_context("a" * 50, "b" * 45, 100, encode, decode)
    assert compacted and len(history) == 50 and history.endswith("b" * 45)
    history, compacted = chat.update_context("a" * 10, "b" * 10, 100, encode, decode)
    assert not compacted and history == "a" * 10 + "b" * 10
    # remplissage réel fourni (KV-cache avec thinking) prioritaire sur l'historique
    _, compacted = chat.update_context("a", "b", 100, encode, decode, used_tokens=95)
    assert compacted


# ── Session de bout en bout ──────────────────────────────────────────────────

SCRIPT = [
    "bonjour", "/ctx", "/think", "une question", "/think", "/interleaved", "encore",
    "/fast", "vite", "/draft self", "vite encore", "/fast", "/temp 0", "greedy",
    "/stop diminishing", "/stop bogus", "/penalty 0", "/taskbudget 30", "abc", "abc", "abc",
    "/taskbudget off", "/libre", "/redactthink", "/think", "pense", "/think",
    "fin", "/save conv_test.txt", "/cache", "/ctx",
]


def test_chat_end_to_end_and_session_restore(monkeypatch, capsys, tmp_path, char_checkpoint):
    monkeypatch.chdir(tmp_path)
    ckpt = char_checkpoint(block_size=256)
    out = run_script(monkeypatch, capsys, ckpt, SCRIPT)
    assert "Erreur" not in out and "Traceback" not in out
    assert "Task budget épuisé" in out
    assert "usage : /stop" in out and "usage : /penalty" in out
    assert (tmp_path / "conv_test.txt").exists()

    # La session sauvegardée est cohérente : ids exacts == longueur du cache
    past, ids, history = load_session(chat.CACHE_PATH, ckpt, "cpu", with_history=True)
    assert past is not None and past.token_ids == ids and past.seq_len == len(ids)
    assert history

    # Redémarrage : la session est restaurée et réutilisée
    out = run_script(monkeypatch, capsys, ckpt, ["/ctx", "suite"])
    assert f"Session restaurée — {len(ids)} tokens" in out
    assert "KV-cache" in out
    past2, ids2 = load_session(chat.CACHE_PATH, ckpt, "cpu")
    assert past2.token_ids == ids2 and past2.seq_len == len(ids2)


def test_chat_survives_tiny_window_and_reset(monkeypatch, capsys, tmp_path, char_checkpoint):
    monkeypatch.chdir(tmp_path)
    ckpt = char_checkpoint(block_size=32)
    lines = ["bonjour"] * 6 + ["/think", "x" * 50, "/reset", "/ctx"]
    out = run_script(monkeypatch, capsys, ckpt, lines, max_tokens=30)
    assert "circuit breaker" in out
    assert "Traceback" not in out and "Erreur" not in out
    assert "(0/32 tokens" in out


def test_generation_error_does_not_end_session(monkeypatch, capsys, tmp_path, char_checkpoint):
    monkeypatch.chdir(tmp_path)
    ckpt = char_checkpoint()
    calls = {"n": 0}
    real = chat.nanoPOPIXA.generate_stream

    def flaky(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("panne simulée")
        return real(self, *a, **k)

    monkeypatch.setattr(chat.nanoPOPIXA, "generate_stream", flaky)
    out = run_script(monkeypatch, capsys, ckpt, ["un", "deux"])
    assert "panne simulée" in out
    assert out.count("[nanoPOPIXA]") >= 2


def test_v1_checkpoint_gives_clean_message(tmp_path, capsys):
    path = tmp_path / "v1.pt"
    torch.save({"model": {"transformer.wpe.weight": torch.zeros(1)}, "config": None}, str(path))
    with pytest.raises(SystemExit):
        chat.load_model(str(path), "cpu")
    assert "v1" in ANSI.sub("", capsys.readouterr().out)


def test_commands_never_leak_into_the_conversation(monkeypatch, capsys, tmp_path, char_checkpoint):
    monkeypatch.chdir(tmp_path)
    ckpt = char_checkpoint()
    out = run_script(monkeypatch, capsys, ckpt,
                     ["/temp", "/tokens", "/bogus", "/help", "/think", "/fast", "bonjour"])
    assert "usage : /temp" in out and "usage : /tokens" in out
    assert "Commande inconnue : /bogus" in out
    assert out.count("/clearcache") >= 2                       # aide affichée 2×
    assert "[fast]" in out and "[think]" not in out.split("Fast mode activé")[-1]
    _, _, history = load_session(chat.CACHE_PATH, ckpt, "cpu", with_history=True)
    assert history.startswith("bonjour")


def test_cache_from_old_weights_is_not_reused_after_retraining(monkeypatch, capsys, tmp_path,
                                                               char_checkpoint):
    """Le checkpoint est réécrit pendant le chat → le cache sauvegardé ne doit pas être repris."""
    monkeypatch.chdir(tmp_path)
    ckpt = char_checkpoint(seed=0)
    real_load = chat.load_model

    def load_then_retrain(path, device):
        loaded = real_load(path, device)
        char_checkpoint(seed=1)                                  # nouvel entraînement
        return loaded

    monkeypatch.setattr(chat, "load_model", load_then_retrain)
    run_script(monkeypatch, capsys, ckpt, ["bonjour"])
    monkeypatch.setattr(chat, "load_model", real_load)
    out = run_script(monkeypatch, capsys, ckpt, ["/ctx"])
    assert "Session restaurée" not in out


def test_json_mode_with_inline_schema(monkeypatch, capsys, tmp_path, char_checkpoint):
    monkeypatch.chdir(tmp_path)
    ckpt = char_checkpoint(block_size=256)
    schema = '{"type":"object","properties":{"nom":{"type":"string","maxLength":8},"age":{"type":"integer"}},"required":["nom","age"]}'
    out = run_script(monkeypatch, capsys, ckpt,
                     [f"/json {schema}", "fiche", "/json {pas du json", "/json off", "bonjour"],
                     max_tokens=60)
    assert "Structured outputs activés" in out and "[json:schema]" in out
    assert "✓ schéma respecté" in out
    assert "Structured outputs :" in out          # schéma invalide → erreur, pas de crash
    assert "Structured outputs désactivés" in out
    assert "Traceback" not in out


def test_redact_thinking_keeps_no_thinking_tokens(monkeypatch, capsys, tmp_path, char_checkpoint):
    monkeypatch.chdir(tmp_path)
    ckpt = char_checkpoint(block_size=512)
    run_script(monkeypatch, capsys, ckpt, ["/redactthink", "/think", "bonjour"], max_tokens=20)
    past, ids, history = load_session(chat.CACHE_PATH, ckpt, "cpu", with_history=True)
    assert past is not None and past.seq_len == len(ids)
    # le cache persistant couvre exactement l'historique texte (prompt + réponse, sans thinking)
    assert "".join(CHARS[i] for i in ids) == history and history.startswith("bonjour")
