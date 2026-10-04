"""Tests de la CLI `popixa` et du monitor : codes de sortie, shell, gen --json, rendu."""

import builtins
import json
import os
import re
import subprocess
import sys

import pytest

import monitor
import popixa_cli

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="1")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
SCHEMA = '{"type":"object","properties":{"nom":{"type":"string"}},"required":["nom"]}'


def popixa(*args):
    code = "import sys, popixa_cli; sys.argv = ['popixa'] + sys.argv[1:]; popixa_cli.main()"
    return subprocess.run([sys.executable, "-c", code, *args], env=ENV, capture_output=True,
                          text=True, timeout=300)


def test_exit_codes_and_help():
    assert popixa("gen", "--tokens", "abc").returncode == 2
    assert popixa("commande-inconnue").returncode == 2
    r = popixa("--help")
    assert r.returncode == 0 and "nanoPOPIXA" in r.stdout and len(r.stdout) < 5000   # ni splash ni REPL
    assert popixa("help").returncode == 0


def test_gen_json_never_writes_truncated_json(char_checkpoint):
    ckpt = char_checkpoint()
    r = popixa("gen", "--checkpoint", ckpt, "--schema", SCHEMA, "--tokens", "3")
    assert r.returncode == 1 and r.stdout == "" and "insuffisant" in r.stderr
    r = popixa("gen", "--checkpoint", ckpt, "--schema", "{pas du json")
    assert r.returncode == 1 and r.stdout == ""
    r = popixa("gen", "--checkpoint", ckpt, "--schema", SCHEMA, "--tokens", "60")
    assert r.returncode == 0 and "nom" in json.loads(r.stdout)


def test_shell_survives_errors(monkeypatch, capsys, tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_text("pas un checkpoint")
    lines = iter(["monitor --refresh -1", f"gen --checkpoint {bad}",
                  "gen --checkpoint inexistant.pt", "gen --prompt aujourd'hui --checkpoint x.pt",
                  "help", "exit"])
    monkeypatch.setattr(builtins, "input", lambda *_: next(lines))
    popixa_cli.run_shell()
    out = ANSI.sub("", capsys.readouterr().out)
    assert "À bientôt" in out
    assert "gen : UnpicklingError" in out
    assert "Guillemet non fermé" not in out


def test_monitor_shows_divergence_and_real_speed(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "# max_iters=300 eval_interval=100 batch_size=2 block_size=8\n"
        "iter     0 | train 3.0000 | val 3.0000 | lr 0.00e+00 | 0.5s\n"
        "iter   100 | train 2.0000 | val 2.0000 | lr 1.00e-03 | 10.0s\n"
        "# max_iters=400 eval_interval=100 batch_size=2 block_size=8\n"
        "iter   100 | train 2.0000 | val 2.0000 | lr 1.00e-03 | 2.0s\n"
        "iter   200 | train nan | val inf | lr 8.00e-04 | 10.0s\n", encoding="utf-8")
    entries, max_iters, *_ = monitor.parse_log(str(log))
    assert len(entries) == 4 and max_iters == 400
    for w in (90, 40, 8):
        lines = monitor.render_dashboard(entries, w=w, max_iters=max_iters, eval_interval=100,
                                         batch_size=2, block_size=8)
        assert {len(ANSI.sub("", l)) for l in lines} == {w}
    text = ANSI.sub("", "\n".join(monitor.render_dashboard(entries, w=90, max_iters=max_iters,
                                                           eval_interval=100)))
    assert "train nan" in text and "val inf" in text
    assert "10.00 it/s" in text                      # 200 itérations / 20 s (paires réelles)
    assert monitor.parse_log(str(tmp_path))[0] == []  # un dossier n'est pas un log
    with pytest.raises(SystemExit):
        popixa_cli._build_parser().parse_args(["monitor", "--refresh", "0"])
