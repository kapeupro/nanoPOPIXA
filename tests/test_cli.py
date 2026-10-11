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
    cap = capsys.readouterr()
    out = ANSI.sub("", cap.out + cap.err)
    assert "À bientôt" in out
    assert "illisible" in out and "Traceback" not in out        # chargement sûr : message clair
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


def test_runs_lists_experiments_and_monitor_finds_run_log(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "runs" / "essai"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"size": "nano", "max_iters": 1000, "best_val": 2.0123,
                                              "status": "terminé"}), encoding="utf-8")
    (run / "eval.json").write_text(json.dumps({"tasks": {"bpb": {"bpb": 2.9}, "paires": {"accuracy": 0.615}}}),
                                   encoding="utf-8")
    (tmp_path / "runs" / "cassé").mkdir()
    (tmp_path / "runs" / "cassé" / "run.json").write_text("{pas du json", encoding="utf-8")
    popixa_cli._run_command(["runs"])
    out = ANSI.sub("", capsys.readouterr().out)
    assert "runs/essai" in out and "2.0123" in out and "2.900" in out and "61.5%" in out
    assert popixa_cli._default_log("essai") == os.path.join("runs", "essai", "train.log")
    assert popixa_cli._default_log() == os.path.join("out-nanopopixa", "train.log")
    (tmp_path / "train.log").write_text("", encoding="utf-8")         # log d'avant 2.3
    assert popixa_cli._default_log() == "train.log"


def test_train_relays_zero_valued_options(monkeypatch):
    """--seed 0, --dropout 0, --patience 0… sont des valeurs, pas des options absentes (0 == False)."""
    import runpy
    seen = {}
    monkeypatch.setattr(runpy, "run_path", lambda *a, **k: seen.setdefault("argv", list(sys.argv)))
    monkeypatch.setattr(sys, "argv", ["popixa"])
    popixa_cli._run_command(["train", "--seed", "0", "--dropout", "0", "--patience", "0",
                             "--weight_decay", "0", "--attn_dropout", "0.0", "--eval"])
    argv = seen["argv"]
    for opt, val in (("--seed", "0"), ("--dropout", "0.0"), ("--patience", "0"),
                     ("--weight_decay", "0.0"), ("--attn_dropout", "0.0")):
        assert argv[argv.index(opt) + 1] == val, argv
    assert "--eval" in argv and "--resume" not in argv and "--size" not in argv
