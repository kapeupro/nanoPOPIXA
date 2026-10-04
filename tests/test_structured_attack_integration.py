"""
Attaque — intégration de bout en bout avec le dépôt nanoPOPIXA.

structured.py (copie de travail) est placé EN PREMIER sur sys.path, le dépôt ensuite :
model.nanoPOPIXA.generate_structured, chat.run_chat (/json) et `popixa gen --json/--schema`
utilisent donc la version testée du module.

Invariants vérifiés :
  - aucune exception ;
  - nombre de tokens yieldés ≤ max_new_tokens ;
  - constraint.is_complete() ⇒ la sortie décodée passe json.loads et validate_instance == [] ;
  - force_complete=True et budget ≥ len(completion_tokens()) depuis l'état initial
    ⇒ sortie TOUJOURS complète ;
  - cache_ref[0].token_ids == (prompt ou amorce 0) + tokens yieldés (suffixe exact en cas de
    fenêtre glissante) et seq_len == len(token_ids).
"""

import builtins
import json
import os
import random
import re
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SCRATCH = REPO   # structured.py vit à la racine du dépôt

if REPO not in sys.path:
    sys.path.insert(0, REPO)

import structured  # noqa: E402
from model import KVCache, POPIXAConfig, nanoPOPIXA  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")

CHARS = sorted(set(
    "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ".,;:!?'\"{}[]-_\néèàç\\/"
))
CHAR_ITOS = {i: c for i, c in enumerate(CHARS)}
CHAR_TB = structured.token_bytes_from_itos(CHAR_ITOS, len(CHARS))

# Vocabulaire char-level de tinyshakespeare (dataset par défaut de `popixa prep --char`)
SHAKESPEARE = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

RECORDS = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {k: {"type": "string"} for k in ("nom", "prenom", "ville", "pays", "email")},
        "required": ["nom", "prenom", "ville", "pays", "email"],
    },
}

SCHEMAS = [
    None,
    {},
    {"type": "null"},
    {"type": "boolean"},
    {"type": "integer"},
    {"type": "number"},
    {"type": ["string", "null"], "maxLength": 3},
    {"type": "string", "minLength": 2, "maxLength": 5},
    {"enum": ["rouge", "vert", 3, None, True]},
    {"anyOf": [{"type": "integer"}, {"type": "array", "items": {"type": "boolean"}, "maxItems": 2}]},
    {"oneOf": [{"const": {"a": [1, "é"]}}, {"enum": ["€uro", "naïve", 1.5]}]},
    {"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}],
     "items": {"type": "null"}, "maxItems": 4},
    {"type": "array", "items": {"type": "number"}, "minItems": 1, "maxItems": 4},
    {"type": "object", "additionalProperties": False},
    {"type": "object", "additionalProperties": {"type": "integer"}},
    {"type": "object", "properties": {"nom": {"type": "string", "maxLength": 8},
                                      "age": {"type": "integer"}}, "required": ["nom", "age"]},
    {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
     "required": ["y"], "additionalProperties": False},
    {"$defs": {"n": {"type": "object", "properties": {
        "v": {"type": "integer"},
        "kids": {"type": "array", "items": {"$ref": "#/$defs/n"}, "maxItems": 2}},
        "required": ["v"]}}, "$ref": "#/$defs/n"},
    {"type": "string", "enum": ["a\"b", "c\\d", "\u0001", "😀"]},
    {"type": "object", "properties": {"d": {"type": "object", "properties": {
        "e": {"type": "array", "items": {"type": "string", "maxLength": 2},
              "minItems": 2, "maxItems": 3}}, "required": ["e"]}}, "required": ["d"]},
]


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def make_model(vocab_size, block_size=128, seed=0):
    torch.manual_seed(seed)
    cfg = POPIXAConfig(block_size=block_size, vocab_size=vocab_size, n_layer=2, n_head=2,
                       n_embd=32, dropout=0.0)
    m = nanoPOPIXA(cfg)
    m.train(False)
    return m


def save_char_ckpt(path, chars=CHARS, block_size=128, seed=0):
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    m = make_model(len(chars), block_size=block_size, seed=seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": m.state_dict(), "config": m.config, "iter": 0,
                "tokenizer": "char", "vocab": {"stoi": stoi, "itos": itos}}, path)
    return path


def save_gpt2_ckpt(path, block_size=64, seed=0):
    m = make_model(50257, block_size=block_size, seed=seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": m.state_dict(), "config": m.config, "iter": 0,
                "tokenizer": "tiktoken_gpt2"}, path)
    return path


def generate(model, tb, schema, budget, prompt=(), seed=0, force_complete=True,
             initial_past_kvs=None, **kw):
    c = structured.json_constraint(tb, schema)
    torch.manual_seed(seed)
    idx = torch.tensor([list(prompt)], dtype=torch.long)
    cache = []
    toks = list(model.generate_structured(
        idx, c, max_new_tokens=budget, cache_ref=cache, force_complete=force_complete,
        initial_past_kvs=initial_past_kvs, **kw))
    return c, toks, cache


def check_output(c, toks, decode, schema, budget, first, label):
    assert len(toks) <= budget, f"{label}: {len(toks)} tokens > budget {budget}"
    if c.is_complete():
        text = decode(toks)
        assert text.encode("utf-8") == c.generated, label
        inst = json.loads(text)
        assert structured.validate_instance(inst, schema) == [], (label, text)
    elif first is not None:
        assert budget < len(first), (
            f"{label}: JSON incomplet alors que budget {budget} ≥ fermeture minimale "
            f"{len(first)} : {c.generated[-80:]!r}")


def check_cache(cache, prefix, toks, block_size, label):
    assert len(cache) == 1, label
    kv = cache[0]
    assert isinstance(kv, KVCache) and kv.token_ids is not None, label
    exp = list(prefix) + list(toks)
    n = len(kv.token_ids)
    assert kv.seq_len == n <= block_size, (label, kv.seq_len, n)
    assert kv.token_ids == exp[-n:], label
    if len(exp) <= block_size:
        assert kv.token_ids == exp, label


def char_decode(toks):
    return "".join(CHARS[t] for t in toks)


def first_closing(tb, schema):
    return structured.json_constraint(tb, schema).completion_tokens()


@pytest.fixture(scope="module")
def char_model():
    return make_model(len(CHARS), block_size=128)


@pytest.fixture(scope="module")
def char_model_small():
    return make_model(len(CHARS), block_size=16, seed=1)


@pytest.fixture(scope="module")
def gpt2():
    tiktoken = pytest.importorskip("tiktoken")
    enc = tiktoken.get_encoding("gpt2")
    return enc, structured.token_bytes_from_tiktoken(enc)


@pytest.fixture(scope="module")
def gpt2_model():
    return make_model(50257, block_size=32)


# ─────────────────────────────────────────────────────────────────────────────
# Fuzz generate_structured — invariants (régressions, doivent passer)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("si", range(len(SCHEMAS)))
def test_fuzz_char_vocab_invariants(char_model, char_model_small, si):
    schema = SCHEMAS[si]
    first = first_closing(CHAR_TB, schema)
    assert first is not None
    for seed in range(14):
        rng = random.Random(seed * 7919 + si)
        model = char_model_small if seed % 3 == 0 else char_model   # 1/3 : débordement block_size
        budget = rng.choice([1, 2, 3, 5, 8, 13, 21, 40, 80, 150, 200])
        kw = dict(temperature=rng.choice([0, 0.7, 1.5]), top_k=rng.choice([None, 3, 40]),
                  top_p=rng.choice([None, 0.5, 0.95]),
                  repetition_penalty=rng.choice([1.0, 1.3, 0.8]))
        prompt = [rng.randrange(len(CHARS)) for _ in range(rng.choice([0, 1, 5, 30]))]
        label = f"schema#{si} seed={seed} budget={budget} {kw} prompt={len(prompt)}"
        c, toks, cache = generate(model, CHAR_TB, schema, budget, prompt, seed, **kw)
        check_output(c, toks, char_decode, schema, budget, first, label)
        check_cache(cache, prompt or [0], toks, model.config.block_size, label)


@pytest.mark.parametrize("si", [0, 4, 6, 8, 10, 11, 15, 17, 18])
def test_fuzz_gpt2_vocab_invariants(gpt2, gpt2_model, si):
    enc, tb = gpt2
    schema = SCHEMAS[si]
    first = first_closing(tb, schema)
    assert first is not None
    for seed in range(3):
        rng = random.Random(seed * 31 + si)
        budget = rng.choice([1, 2, 4, 9, 17, 30, 45])        # 45 + prompt > block_size 32
        kw = dict(temperature=rng.choice([0, 0.7, 1.5]), top_k=rng.choice([None, 40]),
                  top_p=rng.choice([None, 0.9]), repetition_penalty=rng.choice([1.0, 1.3]))
        prompt = enc.encode_ordinary(rng.choice(["", "Réponds en JSON : ", "x"]))
        label = f"gpt2 schema#{si} seed={seed} budget={budget} {kw}"
        c, toks, cache = generate(gpt2_model, tb, schema, budget, prompt, seed, **kw)
        check_output(c, toks, enc.decode, schema, budget, first, label)
        check_cache(cache, prompt or [0], toks, gpt2_model.config.block_size, label)


@pytest.mark.parametrize("si", range(len(SCHEMAS)))
def test_force_complete_with_exact_minimal_budget(char_model, si):
    """Budget == longueur de la fermeture minimale : la sortie est toujours complète."""
    schema = SCHEMAS[si]
    first = first_closing(CHAR_TB, schema)
    for seed in range(4):
        for temp in (0, 1.5):
            c, toks, _ = generate(char_model, CHAR_TB, schema, len(first), [5], seed,
                                  temperature=temp)
            check_output(c, toks, char_decode, schema, len(first), first, f"#{si} s={seed}")
            assert c.is_complete()


def test_force_complete_false_never_exceeds_budget_and_stays_sound(char_model):
    for si, schema in enumerate(SCHEMAS):
        for seed in range(3):
            c, toks, _ = generate(char_model, CHAR_TB, schema, 12, [1, 2], seed,
                                  force_complete=False, temperature=1.5)
            check_output(c, toks, char_decode, schema, 12, None, f"#{si}")


def test_zero_and_negative_budget(char_model):
    for budget in (0, -3):
        c, toks, cache = generate(char_model, CHAR_TB, RECORDS, budget, [1], 0)
        assert toks == [] and not c.is_complete()
        check_cache(cache, [1], [], 128, "budget<=0")


# ─────────────────────────────────────────────────────────────────────────────
# KV-cache persistant + fenêtre glissante (régressions)
# ─────────────────────────────────────────────────────────────────────────────

def test_cache_ref_chain_across_turns_with_overflow():
    model = make_model(len(CHARS), block_size=16)
    schema = {"type": "array", "items": {"type": "string"}}
    c1, t1, cache1 = generate(model, CHAR_TB, schema, 40, [1, 2, 3], 0, temperature=1.0)
    check_cache(cache1, [1, 2, 3], t1, 16, "tour 1")
    kv1 = cache1[0]
    # tour 2 : nouveaux ids sur le cache du tour 1 (débordement → glissement)
    c2, t2, cache2 = generate(model, CHAR_TB, schema, 40, [4, 5], 1, temperature=1.0,
                              initial_past_kvs=kv1)
    check_cache(cache2, kv1.token_ids + [4, 5], t2, 16, "tour 2")
    assert c2.is_complete()
    # tour 3 : entrée vide sur cache connu → le dernier token du cache est rejoué
    c3, t3, cache3 = generate(model, CHAR_TB, schema, 10, [], 2, temperature=1.0,
                              initial_past_kvs=cache2[0])
    check_cache(cache3, cache2[0].token_ids, t3, 16, "tour 3")


def test_cache_ref_after_early_close():
    model = make_model(len(CHARS), block_size=32)
    c = structured.json_constraint(CHAR_TB, {"type": "array", "items": {"type": "string"}})
    cache = []
    torch.manual_seed(0)
    gen = model.generate_structured(torch.tensor([[7, 8]]), c, max_new_tokens=50,
                                    cache_ref=cache, temperature=1.0)
    got = [next(gen), next(gen), next(gen)]
    gen.close()
    check_cache(cache, [7, 8], got, 32, "close")
    assert c.generated == char_decode(got).encode()
    assert not cache[0][0][0].requires_grad


@pytest.mark.parametrize("block_size", [1, 2, 3, 5])
def test_tiny_block_sizes(block_size):
    model = make_model(len(CHARS), block_size=block_size)
    for seed in range(4):
        prompt = list(range(1, 8))
        c, toks, cache = generate(model, CHAR_TB, SCHEMAS[15], 60, prompt, seed, temperature=1.0)
        assert c.is_complete(), c.generated
        check_output(c, toks, char_decode, SCHEMAS[15], 60, None, "tiny")
        check_cache(cache, prompt, toks, block_size, "tiny")


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT model.py — force_complete n'entre dans sa fenêtre de contrôle qu'à
# `remaining <= 32 + 2*len(fermeture initiale)` : si la fermeture a grandi au-delà
# (propriété optionnelle longue, élément de tableau à plusieurs champs requis…),
# la fermeture est tronquée → JSON incomplet malgré un budget largement suffisant.
# ─────────────────────────────────────────────────────────────────────────────

def test_force_complete_optional_long_string_char(char_model):
    schema = {"type": "object", "properties": {"a": {"type": "string", "minLength": 60}}}
    first = first_closing(CHAR_TB, schema)
    assert len(first) == 2
    incomplete = []
    for seed in range(6):
        c, toks, _ = generate(char_model, CHAR_TB, schema, 50, [STOI_A], seed, temperature=1.0)
        if not c.is_complete():
            incomplete.append((seed, c.generated[-40:]))
    assert not incomplete, f"budget 50 ≥ fermeture minimale 2, sorties incomplètes : {incomplete}"


def test_force_complete_array_of_records_char(char_model):
    first = first_closing(CHAR_TB, RECORDS)
    assert len(first) == 2
    incomplete = []
    for seed in range(6):
        for budget in (60, 100):
            c, toks, _ = generate(char_model, CHAR_TB, RECORDS, budget, [STOI_A], seed,
                                  temperature=1.0)
            if not c.is_complete():
                incomplete.append((seed, budget, c.generated[-50:]))
    assert not incomplete, f"fermeture minimale 2 tokens, sorties incomplètes : {incomplete}"


def test_force_complete_optional_long_string_gpt2(gpt2):
    enc, tb = gpt2
    model = make_model(50257, block_size=128)
    schema = {"type": "object", "properties": {"a": {"type": "string", "minLength": 400}}}
    assert len(first_closing(tb, schema)) == 2
    incomplete = []
    for seed in range(3):
        c, toks, _ = generate(model, tb, schema, 60, [11], seed, temperature=1.0)
        if not c.is_complete():
            incomplete.append((seed, c.generated[:40]))
    assert not incomplete, f"budget 60 ≥ fermeture minimale 2, sorties incomplètes : {incomplete}"


STOI_A = CHARS.index("a")


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT structured.py — completion_tokens() ne tokenise QUE la complétion octets la
# plus courte : si un de ses caractères manque au vocabulaire (ex. '0' absent du vocab
# char-level tinyshakespeare), elle renvoie None alors que d'autres complétions existent
# ("3", "true"…) → `popixa gen --json` refuse le modèle. (generate_structured continue
# désormais en meilleur effort quand la fermeture est None : régression ci-dessous.)
# ─────────────────────────────────────────────────────────────────────────────

def test_completion_tokens_vocab_without_zero_digit():
    itos = {i: c for i, c in enumerate(SHAKESPEARE)}
    tb = structured.token_bytes_from_itos(itos, len(SHAKESPEARE))
    for schema in (None, {"type": "integer"}, {"type": "number"},
                   {"anyOf": [{"type": "integer"}, {"type": "null"}]}):
        c = structured.json_constraint(tb, schema)
        # une instance valide est écrivable avec ce vocabulaire…
        c.advance(SHAKESPEARE.index("3"))
        assert c.is_complete()
        c.reset()
        # … donc la fermeture ne doit pas être « impossible »
        closing = c.completion_tokens()
        assert closing is not None, f"completion_tokens() None pour {schema} alors que '3' est valide"
        for t in closing:
            c.advance(t)
        assert c.is_complete()


def test_generate_structured_vocab_without_zero_digit():
    chars = [ch for ch in CHARS if ch != "0"]
    itos = {i: c for i, c in enumerate(chars)}
    tb = structured.token_bytes_from_itos(itos, len(chars))
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    model = make_model(len(chars), block_size=64)
    outs = []
    for seed in range(4):
        c, toks, _ = generate(model, tb, schema, 30, [1], seed, temperature=1.0)
        outs.append((c.is_complete(), c.generated))
    # '{"n":1}' est écrivable en 7 tokens ; budget 30
    assert all(ok for ok, _ in outs), f"sorties : {outs}"


# ─────────────────────────────────────────────────────────────────────────────
# chat.run_chat — /json (harnais de tests/test_chat.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_script(monkeypatch, capsys, checkpoint, lines, max_tokens=20, seed=0):
    import chat
    it = iter(lines)

    def fake_input(*_):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(builtins, "input", fake_input)
    torch.manual_seed(seed)
    chat.run_chat(checkpoint, max_tokens, 0.8, 40)
    return ANSI.sub("", capsys.readouterr().out)


def test_chat_json_multi_turn_session_char(monkeypatch, capsys, tmp_path):
    import chat
    from session_cache import load_session
    monkeypatch.chdir(tmp_path)
    ckpt = save_char_ckpt(str(tmp_path / "out-nanopopixa" / "checkpoint.pt"), block_size=512)
    schema = json.dumps(SCHEMAS[15])
    lines = [f"/json {schema}", "fiche", "encore", "/temp 0", "greedy", "/temp 1.5", "chaud",
             "/json", "/json", "libre", "/tokens 2", "court", "/tokens 40", "/json off", "bonjour"]
    out = run_script(monkeypatch, capsys, ckpt, lines, max_tokens=60)
    assert "Traceback" not in out and "Erreur" not in out
    assert out.count("✓ schéma respecté") == 4
    assert "hors schéma" not in out and "JSON invalide" not in out
    assert "Structured outputs désactivés" in out
    past, ids, history = load_session(chat.CACHE_PATH, ckpt, "cpu", with_history=True)
    assert past is not None and past.token_ids == ids and past.seq_len == len(ids)
    assert history.startswith("fiche")


def test_chat_json_gpt2(monkeypatch, capsys, tmp_path, gpt2):
    monkeypatch.chdir(tmp_path)
    ckpt = save_gpt2_ckpt(str(tmp_path / "out-nanopopixa" / "checkpoint.pt"), block_size=256)
    schema = json.dumps(SCHEMAS[10])
    out = run_script(monkeypatch, capsys, ckpt, [f"/json {schema}", "un", "deux"], max_tokens=40)
    assert "Traceback" not in out and "Erreur" not in out
    assert out.count("✓ schéma respecté") == 2


def test_chat_json_records_complete_with_ample_budget(monkeypatch, capsys, tmp_path):
    """DÉFAUT model.py (fenêtre force_complete) vu depuis le chat : /tokens 60 pour une
    fermeture minimale de 2 tokens → « JSON incomplet »."""
    monkeypatch.chdir(tmp_path)
    ckpt = save_char_ckpt(str(tmp_path / "out-nanopopixa" / "checkpoint.pt"), block_size=512)
    lines = [f"/json {json.dumps(RECORDS)}", "/temp 1", "liste", "liste", "liste", "liste"]
    out = run_script(monkeypatch, capsys, ckpt, lines, max_tokens=60)
    assert "Traceback" not in out
    assert "JSON incomplet" not in out, out[out.find("[json"):][-1500:]


@pytest.mark.parametrize("schema", [
    "[" * 3000 + "]" * 3000,                                       # RecursionError (json.loads)
    '{"properties":{"a":' * 600 + "{}" + "}}" * 600,               # RecursionError (json.loads)
], ids=["deep-array", "deep-properties"])
def test_chat_json_deeply_nested_schema_does_not_crash(monkeypatch, capsys, tmp_path, schema):
    """DÉFAUT structured.py : load_schema laisse passer RecursionError (pas SchemaError) →
    le /json de chat.py (qui attrape ValueError) fait planter toute la session."""
    monkeypatch.chdir(tmp_path)
    ckpt = save_char_ckpt(str(tmp_path / "out-nanopopixa" / "checkpoint.pt"))
    try:
        out = run_script(monkeypatch, capsys, ckpt, [f"/json {schema}", "bonjour"])
    except Exception as e:     # noqa: BLE001 — c'est précisément le défaut
        pytest.fail(f"/json a fait planter le chat : {type(e).__name__}")
    assert "Structured outputs :" in out


@pytest.mark.parametrize("schema", [
    '{"type":"string","minLength":1e20}',                          # OverflowError
    '{"type":"string","minLength":1e12}',                          # MemoryError
], ids=["overflow", "memory"])
def test_chat_json_huge_length_bound_does_not_crash(monkeypatch, capsys, tmp_path, schema):
    """DÉFAUT structured.py : le témoin minimal est matérialisé (b'a' * minLength) →
    OverflowError / MemoryError hors SchemaError → crash du chat."""
    monkeypatch.chdir(tmp_path)
    ckpt = save_char_ckpt(str(tmp_path / "out-nanopopixa" / "checkpoint.pt"))
    try:
        out = run_script(monkeypatch, capsys, ckpt, [f"/json {schema}", "bonjour"])
    except Exception as e:     # noqa: BLE001
        pytest.fail(f"/json a fait planter le chat : {type(e).__name__}")
    assert "Structured outputs :" in out


# ─────────────────────────────────────────────────────────────────────────────
# popixa gen --json / --schema (sous-processus)
# ─────────────────────────────────────────────────────────────────────────────

def popixa_gen(tmp_path, *args, timeout=300, seed=None):
    """`popixa gen ...` en sous-processus ; seed → torch.manual_seed avant main() (reproductible)."""
    env = dict(os.environ, PYTHONPATH=SCRATCH + os.pathsep + REPO)
    if seed is None:
        cmd = [sys.executable, os.path.join(REPO, "popixa_cli.py"), "gen", *args]
    else:
        boot = ("import sys, torch; torch.manual_seed(int(sys.argv[1])); "
                "sys.argv = ['popixa'] + sys.argv[2:]; import popixa_cli; popixa_cli.main()")
        cmd = [sys.executable, "-c", boot, str(seed), "gen", *args]
    return subprocess.run(cmd, cwd=str(tmp_path), env=env, capture_output=True, text=True,
                          timeout=timeout)


def test_cli_gen_json_and_schema_char(tmp_path):
    ckpt = save_char_ckpt(str(tmp_path / "ck" / "char.pt"), block_size=128)
    schema = SCHEMAS[15]
    sfile = tmp_path / "schema.json"
    sfile.write_text(json.dumps(schema), encoding="utf-8")
    first = first_closing(CHAR_TB, schema)
    runs = [
        (["--json", "--tokens", "40", "--temp", "1.0"], None),
        (["--schema", str(sfile), "--tokens", str(len(first)), "--temp", "0"], schema),
        (["--schema", json.dumps(schema), "--prompt", '{"nom": ', "--tokens", "80",
          "--temp", "1.5", "--top_p", "0.9", "--penalty", "1.3"], schema),
    ]
    for args, sch in runs:
        r = popixa_gen(tmp_path, "--checkpoint", ckpt, *args)
        assert r.returncode == 0, (args, r.stderr[-500:])
        assert "Traceback" not in r.stderr
        inst = json.loads(r.stdout)
        assert structured.validate_instance(inst, sch) == []


def test_cli_gen_schema_gpt2(tmp_path, gpt2):
    ckpt = save_gpt2_ckpt(str(tmp_path / "ck" / "gpt2.pt"), block_size=64)
    schema = SCHEMAS[10]
    r = popixa_gen(tmp_path, "--checkpoint", ckpt, "--schema", json.dumps(schema),
                   "--tokens", "30", "--temp", "1.0", "--prompt", "JSON:")
    assert r.returncode == 0, r.stderr[-500:]
    assert structured.validate_instance(json.loads(r.stdout), schema) == []


def test_cli_gen_records_schema_ample_budget(tmp_path):
    """DÉFAUT model.py (fenêtre force_complete) : le CLI valide --tokens 60 ≥ 2 puis
    échoue avec « JSON incomplet — augmente --tokens »."""
    ckpt = save_char_ckpt(str(tmp_path / "ck" / "char.pt"), block_size=256)
    failures = []
    for seed in range(4):
        r = popixa_gen(tmp_path, "--checkpoint", ckpt, "--schema", json.dumps(RECORDS),
                       "--tokens", "60", "--temp", "1.0", "--top_k", "0", seed=seed)
        if r.returncode != 0:
            failures.append((seed, ANSI.sub("", r.stderr)[-200:]))
            continue
        assert structured.validate_instance(json.loads(r.stdout), RECORDS) == []
    assert not failures, failures


def test_cli_gen_json_shakespeare_char_vocab(tmp_path):
    """DÉFAUT structured.py (completion_tokens) : `popixa gen --json` refuse un modèle
    char-level tinyshakespeare (« le vocabulaire ne permet pas d'écrire un JSON »)
    alors que 3, true, false, null sont écrivables."""
    ckpt = save_char_ckpt(str(tmp_path / "ck" / "shak.pt"), chars=list(SHAKESPEARE),
                          block_size=64)
    r = popixa_gen(tmp_path, "--checkpoint", ckpt, "--json", "--tokens", "20", "--temp", "1.0")
    assert r.returncode == 0, ANSI.sub("", r.stderr)[-300:]
    assert structured.validate_instance(json.loads(r.stdout), None) == []


def test_cli_gen_deep_schema_clean_error(tmp_path):
    """DÉFAUT structured.py : RecursionError non convertie en SchemaError → traceback."""
    ckpt = save_char_ckpt(str(tmp_path / "ck" / "char.pt"))
    r = popixa_gen(tmp_path, "--checkpoint", ckpt, "--schema", "[" * 3000 + "]" * 3000,
                   "--tokens", "10")
    assert r.returncode == 1
    assert "Traceback" not in r.stderr, r.stderr[-300:]
    assert "Structured outputs" in ANSI.sub("", r.stderr)
