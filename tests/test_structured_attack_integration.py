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

_CLI_BOOT = (
    # Tour 3 — correctif du harnais : `python <dépôt>/popixa_cli.py` met le
    # dossier du SCRIPT (le dépôt) en sys.path[0], AVANT PYTHONPATH → c'était le
    # structured.py du dépôt qui était testé. On passe toujours par `-c` (cwd neutre),
    # le scratch est forcé en tête et l'origine du module est vérifiée.
    "import sys; sys.path.insert(0, sys.argv[1]); import structured, os; "
    "assert os.path.dirname(os.path.abspath(structured.__file__)) == sys.argv[1], "
    "structured.__file__; import torch; seed = sys.argv[2]; "
    "seed != '-' and torch.manual_seed(int(seed)); "
    "sys.argv = ['popixa'] + sys.argv[3:]; import popixa_cli; popixa_cli.main()"
)


def popixa_gen(tmp_path, *args, timeout=300, seed=None):
    """`popixa gen ...` en sous-processus (structured.py du scratch garanti) ;
    seed → torch.manual_seed avant main() (reproductible)."""
    env = dict(os.environ, PYTHONPATH=SCRATCH + os.pathsep + REPO)
    cmd = [sys.executable, "-c", _CLI_BOOT, SCRATCH, "-" if seed is None else str(seed),
           "gen", *args]
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


# ═════════════════════════════════════════════════════════════════════════════
# TOUR 2 — fuzz à grande échelle (régressions), modèles biaisés vers les chemins
# délicats de la grammaire, et défauts d'intégration restants
# ═════════════════════════════════════════════════════════════════════════════

R2_EXTRA = [
    RECORDS,
    {"type": "object", "properties": {
        "a": {"type": "string", "minLength": 30},
        "b": {"type": "array", "items": {"type": "integer"}, "minItems": 3}}, "required": ["b"]},
    {"type": "array", "items": {"anyOf": [
        {"type": "string"}, {"type": "number"},
        {"type": "object", "properties": {"k": {"type": "boolean"}}}]}},
    {"type": "number", "enum": [1.5, -2, 300]},
    {"type": "string", "maxLength": 0},
    {"type": "array", "items": {"type": "array", "items": {
        "type": "array", "items": {"type": "null"}, "minItems": 2}, "minItems": 2}, "minItems": 2},
    {"allOf": [{"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}],
     "properties": {"b": {"type": "string"}}},
    {"$defs": {"p": {"type": "object", "properties": {"x": {"type": "number"},
                                                      "y": {"type": "number"}},
                     "required": ["x"]}}, "$ref": "#/$defs/p", "required": ["y"]},
    {"anyOf": [{"type": "string", "maxLength": 2}, {"type": "integer"}, {"type": "null"}],
     "enum": ["ab", 3, None, "abc", 4.5]},
    {"type": "array", "prefixItems": [{"const": "x"}, {"type": "boolean"}], "minItems": 3,
     "maxItems": 5, "items": {"type": "number"}},
    {"type": "array", "items": [{"type": "string"}], "additionalItems": {"type": "integer"},
     "minItems": 2},
    {"type": "object", "required": ["z", "a"], "properties": {"a": {"type": "string"}},
     "additionalProperties": {"type": "boolean"}},
    {"enum": [{"a": [1, {"b": None}]}, [], {}, "", 0, -0.5, 1e5, True]},
    {"const": "é\"\\/\n€😀"},
    {"oneOf": [
        {"type": "object", "properties": {"kind": {"const": "a"}, "v": {"type": "integer"}},
         "required": ["kind", "v"]},
        {"type": "object", "properties": {"kind": {"const": "b"}, "w": {"type": "string"}},
         "required": ["kind"]}]},
    {"$defs": {"t": {"anyOf": [{"type": "integer"},
                               {"type": "array", "items": {"$ref": "#/$defs/t"}, "maxItems": 3}]}},
     "$ref": "#/$defs/t"},
]
R2_SCHEMAS = SCHEMAS + R2_EXTRA


def _expected_prefix(prompt, init):
    """Ids que cache_ref[0].token_ids doit contenir avant les tokens yieldés."""
    if init is not None:
        return list(init.token_ids) + list(prompt)
    return list(prompt) or [0]


@pytest.fixture(scope="module")
def r2_char_models():
    return {bs: make_model(len(CHARS), block_size=bs, seed=bs) for bs in (8, 24, 128, 512)}


@pytest.mark.parametrize("chunk", range(6))
def test_r2_fuzz_char_hundreds_of_seeds(r2_char_models, chunk):
    """600 tirages : schémas variés, budgets 1..200, T 0/0.7/1.5, top-k/top-p/pénalité,
    prompts 0..200 tokens, KV-cache initial (30 %), débordement de block_size."""
    for seed in range(chunk * 100, chunk * 100 + 100):
        rng = random.Random(seed)
        schema = R2_SCHEMAS[rng.randrange(len(R2_SCHEMAS))]
        bs = rng.choice([8, 24, 128, 512])
        model = r2_char_models[bs]
        budget = rng.choice([1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 80, 120, 200])
        kw = dict(temperature=rng.choice([0, 0.7, 1.5]), top_k=rng.choice([None, 1, 3, 40]),
                  top_p=rng.choice([None, 0.3, 0.9]),
                  repetition_penalty=rng.choice([1.0, 1.3, 0.7, 2.0]))
        prompt = [rng.randrange(len(CHARS)) for _ in range(rng.choice([0, 1, 5, 30, 200]))]
        init = None
        if rng.random() < 0.3:
            ids = [rng.randrange(len(CHARS)) for _ in range(rng.choice([1, 5, bs]))][-bs:]
            with torch.no_grad():
                _, kvs = model(torch.tensor([ids]))
            init = KVCache(kvs, ids)
        first = first_closing(CHAR_TB, schema)
        label = f"seed={seed} schema={json.dumps(schema)[:60]} bs={bs} budget={budget} {kw}"
        c, toks, cache = generate(model, CHAR_TB, schema, budget, prompt, seed,
                                  initial_past_kvs=init, **kw)
        check_output(c, toks, char_decode, schema, budget, first, label)
        check_cache(cache, _expected_prefix(prompt, init), toks, bs, label)


@pytest.fixture(scope="module")
def r2_gpt2_models():
    return {bs: make_model(50257, block_size=bs, seed=bs) for bs in (16, 128)}


@pytest.mark.parametrize("chunk", range(3))
def test_r2_fuzz_gpt2_hundreds_of_seeds(gpt2, r2_gpt2_models, chunk):
    """Vocabulaire gpt2 (50257) : 150 tirages, prompts, KV-cache initial, débordement."""
    enc, tb = gpt2
    models = r2_gpt2_models
    for seed in range(chunk * 50, chunk * 50 + 50):
        rng = random.Random(10_000 + seed)
        schema = R2_SCHEMAS[rng.randrange(len(R2_SCHEMAS))]
        bs = rng.choice([16, 128])
        model = models[bs]
        budget = rng.choice([1, 2, 3, 4, 6, 10, 20, 40, 80, 200])
        kw = dict(temperature=rng.choice([0, 0.7, 1.5]), top_k=rng.choice([None, 1, 40]),
                  top_p=rng.choice([None, 0.9]), repetition_penalty=rng.choice([1.0, 1.3]))
        prompt = enc.encode_ordinary(rng.choice(["", "JSON:", "Réponds en JSON : {", "x" * 40]))
        init = None
        if rng.random() < 0.3:
            ids = enc.encode_ordinary("Contexte précédent : ")[-bs:]
            with torch.no_grad():
                _, kvs = model(torch.tensor([ids]))
            init = KVCache(kvs, ids)
        first = first_closing(tb, schema)
        label = f"gpt2 seed={seed} schema={json.dumps(schema)[:60]} bs={bs} budget={budget} {kw}"
        c, toks, cache = generate(model, tb, schema, budget, prompt, seed,
                                  initial_past_kvs=init, **kw)
        check_output(c, toks, enc.decode, schema, budget, first, label)
        check_cache(cache, _expected_prefix(prompt, init), toks, bs, label)



class _BiasedPOPIXA(nanoPOPIXA):
    """Modèle aléatoire + biais de logits fixe : pousse l'échantillonnage vers des chemins
    rares de la grammaire (échappements, \\uXXXX, exposants, UTF-8 multi-octets…)."""
    bias = None

    def forward(self, idx, targets=None, past_kvs=None):
        logits, kv = super().forward(idx, targets, past_kvs)
        if self.bias is not None and targets is None:
            logits = logits + self.bias
        return logits, kv


R2_BIAS_CHARS = sorted(set("abcdeflnrstuxyzABEFNU0123456789.+-eE{}[]\",: \n\t\\/é€😀\x7f"))
R2_BIAS_TB = structured.token_bytes_from_itos(dict(enumerate(R2_BIAS_CHARS)),
                                              len(R2_BIAS_CHARS))


def test_r2_biased_char_model_rare_grammar_paths():
    torch.manual_seed(0)
    model = _BiasedPOPIXA(POPIXAConfig(block_size=48, vocab_size=len(R2_BIAS_CHARS), n_layer=2,
                                       n_head=2, n_embd=32, dropout=0.0))
    model.train(False)
    dec = lambda t: "".join(R2_BIAS_CHARS[i] for i in t)     # noqa: E731
    n_complete = 0
    for seed in range(400):
        rng = random.Random(seed)
        bias = torch.zeros(len(R2_BIAS_CHARS))
        for i in rng.sample(range(len(R2_BIAS_CHARS)), rng.randint(1, 10)):
            bias[i] = rng.uniform(2, 9)
        model.bias = bias
        schema = R2_SCHEMAS[rng.randrange(len(R2_SCHEMAS))]
        budget = rng.choice([4, 8, 15, 30, 60, 120])
        fc = rng.random() < 0.8
        kw = dict(temperature=rng.choice([0, 0.7, 1.5]), top_k=rng.choice([None, 3]),
                  top_p=rng.choice([None, 0.9]), repetition_penalty=rng.choice([1.0, 1.3]))
        c = structured.json_constraint(R2_BIAS_TB, schema, max_whitespace=rng.choice([0, 2, 4]))
        first = c.completion_tokens()
        torch.manual_seed(seed)
        toks = list(model.generate_structured(torch.tensor([[1]]), c, max_new_tokens=budget,
                                              force_complete=fc, **kw))
        label = f"seed={seed} schema={json.dumps(schema)[:60]} fc={fc} {kw}"
        check_output(c, toks, dec, schema, budget, first if fc else None, label)
        n_complete += c.is_complete()
    assert n_complete > 250


def test_r2_biased_gpt2_model_partial_utf8_and_escapes(gpt2):
    enc, tb = gpt2
    groups = {
        "hi": [i for i, b in enumerate(tb) if b and any(x >= 0x80 for x in b)],
        "bs": [i for i, b in enumerate(tb) if b and b"\\" in b],
        "q": [i for i, b in enumerate(tb) if b and b'"' in b],
        "dig": [i for i, b in enumerate(tb) if b and b.strip(b" ").isdigit()],
        "punct": [i for i, b in enumerate(tb) if b and all(chr(x) in '{}[],:" \n.-+eE' for x in b)],
    }
    schemas = [None, {"type": "string", "maxLength": 3},
               {"type": "string", "minLength": 2, "maxLength": 4},
               {"type": "object", "additionalProperties": {"type": "string", "maxLength": 2}},
               {"type": "array", "items": {"type": "number"}, "maxItems": 4},
               {"enum": ["é", "€", "😀", "a/b", "\u007f", "x\"y"]},
               {"type": "object", "properties": {"é": {"type": "string", "maxLength": 3},
                                                 "😀": {"type": "integer"}}, "required": ["😀"]}]
    torch.manual_seed(0)
    model = _BiasedPOPIXA(POPIXAConfig(block_size=64, vocab_size=50257, n_layer=2, n_head=2,
                                       n_embd=32, dropout=0.0))
    model.train(False)
    for seed in range(120):
        rng = random.Random(seed)
        bias = torch.zeros(50257)
        for g in rng.sample(sorted(groups), rng.randint(1, 3)):
            bias[groups[g]] = rng.uniform(3, 9)
        model.bias = bias
        schema = schemas[rng.randrange(len(schemas))]
        budget = rng.choice([3, 6, 12, 25, 50])
        kw = dict(temperature=rng.choice([0, 0.7, 1.5]), top_k=rng.choice([None, 40]),
                  top_p=rng.choice([None, 0.9]), repetition_penalty=rng.choice([1.0, 1.3]))
        first = first_closing(tb, schema)
        c, toks, cache = generate(model, tb, schema, budget, [11], seed, **kw)
        check_output(c, toks, enc.decode, schema, budget, first, f"seed={seed} {kw}")
        check_cache(cache, [11], toks, 64, f"seed={seed}")


def test_r2_empty_input_over_known_caches():
    """Entrée vide sur un cache de 1 token / un cache plein : le dernier token est rejoué ;
    les ids du cache final restent exacts."""
    model = make_model(len(CHARS), block_size=16)
    with torch.no_grad():
        _, kv1 = model(torch.tensor([[5]]))
        full = list(range(16))
        _, kvf = model(torch.tensor([full]))
    for init in (KVCache(kv1, [5]), KVCache(kvf, full)):
        for seed in range(5):
            c, toks, cache = generate(model, CHAR_TB, SCHEMAS[15], 30, [], seed,
                                      temperature=1.0, initial_past_kvs=init)
            assert c.is_complete()
            check_output(c, toks, char_decode, SCHEMAS[15], 30, None, "vide")
            check_cache(cache, list(init.token_ids), toks, 16, "vide")


def test_r2_chat_json_gpt2_multi_turn_window_overflow(monkeypatch, capsys, tmp_path, gpt2):
    """gpt2, block_size 64 : plusieurs tours JSON qui font glisser la fenêtre, un tour libre,
    puis retour au JSON — jamais d'erreur, session persistée cohérente."""
    import chat
    from session_cache import load_session
    monkeypatch.chdir(tmp_path)
    ckpt = save_gpt2_ckpt(str(tmp_path / "out-nanopopixa" / "checkpoint.pt"), block_size=64)
    schema = json.dumps(SCHEMAS[15])
    lines = [f"/json {schema}", "/temp 1", "fiche un", "fiche deux", "fiche trois",
             "/json off", "bonjour", f"/json {schema}", "/tokens 4", "court", "/tokens 50",
             "dernière"]
    out = run_script(monkeypatch, capsys, ckpt, lines, max_tokens=50)
    assert "Traceback" not in out and "Erreur" not in out
    # fermeture minimale gpt2 = 7 tokens → seul le tour « /tokens 4 » est incomplet
    assert len(first_closing(gpt2[1], SCHEMAS[15])) == 7
    assert out.count("✓ schéma respecté") == 4, out[-1500:]
    assert out.count("JSON incomplet") == 1
    assert "Auto-compact" in out          # block_size 64 : la compaction auto s'est déclenchée
    # (l'auto-compact efface la session sur disque ; si une session existe, elle est exacte)
    past, ids, _ = load_session(chat.CACHE_PATH, ckpt, "cpu", with_history=True)
    if past is not None:
        assert past.token_ids == ids and past.seq_len == len(ids) <= 64


def test_r2_cli_gen_gpt2_window_overflow(tmp_path, gpt2):
    """`popixa gen --schema` gpt2 avec --tokens > block_size : JSON complet et conforme."""
    ckpt = save_gpt2_ckpt(str(tmp_path / "ck" / "gpt2.pt"), block_size=32)
    schema = {"type": "array", "items": {"type": "string", "maxLength": 12}, "minItems": 3}
    r = popixa_gen(tmp_path, "--checkpoint", ckpt, "--schema", json.dumps(schema),
                   "--tokens", "120", "--temp", "1.0", "--prompt", "Liste : " * 6, seed=3)
    assert r.returncode == 0, ANSI.sub("", r.stderr)[-400:]
    assert structured.validate_instance(json.loads(r.stdout), schema) == []


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT chat.py — report_json n'attrape que ValueError autour de json.loads : un JSON
# généré COMPLET et valide mais imbriqué sur ≥ 1000 niveaux (json.loads de CPython lève
# RecursionError) fait remonter l'exception jusqu'au `except Exception` du tour →
# « ✗ Erreur de génération : RecursionError », cache de session jeté, tour absent de
# l'historique. (validate_instance, lui, est déjà protégé contre RecursionError.)
# ─────────────────────────────────────────────────────────────────────────────

def _deep_complete_constraint(depth):
    c = structured.json_constraint(CHAR_TB, None)
    for _ in range(depth):
        c.advance(CHARS.index("["))
    for t in c.completion_tokens():
        c.advance(t)
    assert c.is_complete()
    return c


def test_r2_report_json_deep_document_minimal(capsys):
    import chat
    c = _deep_complete_constraint(1200)
    assert structured.validate_instance(json.loads("[" * 50 + "]" * 50), None) == []
    try:
        chat.report_json(c, None)
    except RecursionError:
        pytest.fail("report_json laisse fuir RecursionError (json.loads) sur un JSON valide "
                    "imbriqué sur 1200 niveaux")
    out = ANSI.sub("", capsys.readouterr().out)
    assert "JSON incomplet" not in out


def test_r2_chat_json_deep_document_reported_as_generation_error(monkeypatch, capsys, tmp_path):
    import model as model_mod
    monkeypatch.chdir(tmp_path)
    ckpt = save_char_ckpt(str(tmp_path / "out-nanopopixa" / "checkpoint.pt"), block_size=512)
    orig = model_mod.nanoPOPIXA.forward
    lb = CHARS.index("[")

    def biased(self, idx, targets=None, past_kvs=None):
        logits, kv = orig(self, idx, targets, past_kvs)
        if targets is None:
            logits = logits.clone()
            logits[..., lb] += 50.0       # le modèle « veut » imbriquer
        return logits, kv

    monkeypatch.setattr(model_mod.nanoPOPIXA, "forward", biased)
    out = run_script(monkeypatch, capsys, ckpt, ["/json", "/tokens 2100", "profond"])
    assert "Erreur de génération" not in out, \
        out[out.find("Erreur"):][:200]
    assert "✓ JSON valide" in out


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT chat.py — /json active un schéma que le vocabulaire du modèle ne sait pas écrire
# (completion_tokens() is None, ex. vocabulaire char tinyshakespeare sans '{' '[' '"')
# sans le signaler ; chaque tour affiche ensuite « JSON incomplet (budget de tokens
# épuisé — augmente /tokens) », ce qui est faux : aucun budget n'y changera rien.
# `popixa gen` fait, lui, la vérification (« le vocabulaire du modèle ne permet pas… »).
# ─────────────────────────────────────────────────────────────────────────────

def test_r2_chat_json_inexpressible_schema_blames_budget(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    ckpt = save_char_ckpt(str(tmp_path / "out-nanopopixa" / "checkpoint.pt"),
                          chars=list(SHAKESPEARE), block_size=128)
    itos = dict(enumerate(SHAKESPEARE))
    tb = structured.token_bytes_from_itos(itos, len(SHAKESPEARE))
    assert structured.json_constraint(tb, {"type": "object"}).completion_tokens() is None
    out = run_script(monkeypatch, capsys, ckpt,
                     ['/json {"type": "object"}', "/tokens 400", "bonjour"], max_tokens=60)
    assert "Traceback" not in out
    assert "budget de tokens épuisé" not in out, out[out.find("[json"):][-600:]


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT structured.py — load_schema lit les fichiers en 'utf-8' strict : un schéma .json
# enregistré avec BOM (Bloc-notes Windows, PowerShell 5 `Out-File -Encoding utf8`…) est
# refusé (« Unexpected UTF-8 BOM ») → `popixa gen --schema fiche.json` et `/json fiche.json`
# échouent sur un schéma parfaitement valide.
# ─────────────────────────────────────────────────────────────────────────────

def test_r2_load_schema_utf8_bom_file(tmp_path):
    p = tmp_path / "fiche.json"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps(SCHEMAS[15]).encode("utf-8"))
    assert structured.load_schema(str(p)) == SCHEMAS[15]


def test_r2_cli_gen_schema_file_with_bom(tmp_path):
    ckpt = save_char_ckpt(str(tmp_path / "ck" / "char.pt"), block_size=128)
    p = tmp_path / "fiche.json"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps(SCHEMAS[15]).encode("utf-8"))
    r = popixa_gen(tmp_path, "--checkpoint", ckpt, "--schema", str(p), "--tokens", "60",
                   "--temp", "1.0")
    assert r.returncode == 0, ANSI.sub("", r.stderr)[-300:]
    assert structured.validate_instance(json.loads(r.stdout), SCHEMAS[15]) == []


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT model.py — generate_structured suppose constraint.vocab_size == vocab_size du
# modèle : avec un modèle au vocabulaire paddé (ex. 50304 à la nanoGPT, ou n'importe quel
# vocab_size > len(token_bytes)) le 1er re-tirage masqué plante en RuntimeError
# (« size of tensor a … must match ») après avoir déjà yieldé des tokens. Les ids hors du
# vocabulaire de la contrainte devraient simplement être interdits (load_token_bytes de
# chat.py fait ce padding, mais l'API publique ne le garantit pas).
# ─────────────────────────────────────────────────────────────────────────────

def test_r2_generate_structured_padded_model_vocab():
    # modèle paddé de 3 ids ; contrainte construite sur le vrai vocabulaire (84 tokens)
    model = make_model(len(CHARS) + 3, block_size=64)
    failures = []
    for seed in range(4):
        c = structured.json_constraint(CHAR_TB, RECORDS)
        torch.manual_seed(seed)
        try:
            toks = list(model.generate_structured(torch.tensor([[1]]), c, max_new_tokens=40,
                                                  temperature=1.0))
        except RuntimeError as e:
            failures.append((seed, str(e)[:90]))
            continue
        assert c.is_complete()
        assert structured.validate_instance(json.loads(char_decode(toks)), RECORDS) == []
    assert not failures, failures


# ═════════════════════════════════════════════════════════════════════════════
# TOUR 3 — schémas aléatoires (sous-ensemble documenté) via generate_structured,
# vocabulaires multi-caractères, CONTENU du KV-cache (pas seulement les ids),
# exactitude de l'échantillonnage par rejet, fuzz chat / CLI (harnais CLI corrigé :
# cf. _CLI_BOOT), défauts restants.
# ═════════════════════════════════════════════════════════════════════════════

R3_KEYS = ["a", "nom", "é", "x y", "k\"q", "b\\c", "😀", "", "1", "id"]
R3_STRS = ["", "a", "é", "€uro", "x\"y", "c\\d", "\n", "😀", "/", "\u007f", "\u0001", "abc"]


def r3_schema(rng, depth=0):
    """Schéma aléatoire du sous-ensemble documenté (peut être insatisfiable → SchemaError)."""
    if depth > 3 or rng.random() < 0.25:
        return rng.choice([
            {}, {"type": "null"}, {"type": "boolean"}, {"type": "integer"}, {"type": "number"},
            {"type": "string"}, {"type": "string", "maxLength": rng.randint(0, 4)},
            {"type": "string", "minLength": rng.randint(0, 6), "maxLength": rng.randint(6, 9)},
            {"enum": rng.sample(R3_STRS + [0, 1, -1.5, 2e3, True, False, None, [], {}, [1, "a"],
                                           {"k": None}], rng.randint(1, 5))},
            {"const": rng.choice(R3_STRS + [0, 12, -0.25, True, None,
                                            {"a": [1, {"b": "é"}]}, []])},
            {"type": rng.sample(["string", "null", "integer", "boolean", "number", "array",
                                 "object"], rng.randint(1, 3))},
            {"maxLength": 2}, {"minItems": 1}, {"required": ["q"]},
        ])
    k = rng.randrange(8)
    if k == 0:
        props = {key: r3_schema(rng, depth + 1) for key in rng.sample(R3_KEYS, rng.randint(0, 4))}
        s = {"type": "object", "properties": props}
        if props and rng.random() < 0.7:
            s["required"] = rng.sample(sorted(props), rng.randint(0, len(props)))
        if rng.random() < 0.3:
            s.setdefault("required", []).append(rng.choice(["zz", "é2"]))
        if rng.random() < 0.3:
            s["additionalProperties"] = rng.choice([False, True, {"type": "integer"}])
        return s
    if k == 1:
        s = {"type": "object"}
        if rng.random() < 0.7:
            s["additionalProperties"] = rng.choice([False, r3_schema(rng, depth + 1)])
        return s
    if k == 2:
        s = {"type": "array", "items": r3_schema(rng, depth + 1)}
        if rng.random() < 0.5:
            s["minItems"] = rng.randint(0, 3)
        if rng.random() < 0.5:
            s["maxItems"] = s.get("minItems", 0) + rng.randint(0, 3)
        return s
    if k == 3:
        s = {"type": "array",
             "prefixItems": [r3_schema(rng, depth + 1) for _ in range(rng.randint(1, 3))]}
        if rng.random() < 0.5:
            s["items"] = rng.choice([False, r3_schema(rng, depth + 1)])
        if rng.random() < 0.5:
            s["minItems"] = rng.randint(0, 4)
        return s
    if k == 4:
        return {rng.choice(["anyOf", "oneOf"]):
                [r3_schema(rng, depth + 1) for _ in range(rng.randint(1, 3))]}
    if k == 5:
        return {"allOf": [r3_schema(rng, depth + 1)]}
    if k == 6:
        return {"$defs": {"t": r3_schema(rng, depth + 1)}, "$ref": "#/$defs/t"}
    return {"$defs": {"n": {"anyOf": [{"type": "integer"}, {"type": "array",
                                                            "items": {"$ref": "#/$defs/n"},
                                                            "maxItems": 2}]}},
            "type": "object", "properties": {"r": {"$ref": "#/$defs/n"},
                                             "s": r3_schema(rng, depth + 1)},
            "required": ["r"]}


def r3_valid_schema(rng):
    s = r3_schema(rng)
    try:
        structured.JSONSchemaMatcher(s)
    except structured.SchemaError:
        return None
    return s


def _r3_reject_constant(name):
    raise ValueError(f"constante non JSON (RFC 8259) : {name}")


def r3_check(c, toks, decode, schema, budget, first, label):
    """check_output + sortie strictement RFC 8259 (ni NaN ni Infinity)."""
    check_output(c, toks, decode, schema, budget, first, label)
    if c.is_complete():
        json.loads(decode(toks), parse_constant=_r3_reject_constant)


R3_CHARS = sorted(set("abcdefiklmnoqrstuxyzABEFNU0123456789.+-eE{}[]\",: \n\t\\/é€😀\x7f\x01"))
R3_TB = structured.token_bytes_from_itos(dict(enumerate(R3_CHARS)), len(R3_CHARS))


def r3_char_decode(toks):
    return "".join(R3_CHARS[t] for t in toks)


@pytest.fixture(scope="module")
def r3_biased_char():
    torch.manual_seed(0)
    m = _BiasedPOPIXA(POPIXAConfig(block_size=48, vocab_size=len(R3_CHARS), n_layer=2,
                                   n_head=2, n_embd=32, dropout=0.0))
    m.train(False)
    return m


@pytest.mark.parametrize("chunk", range(4))
def test_r3_fuzz_random_schemas_biased_char(r3_biased_char, chunk):
    """1000 schémas aléatoires (≈ 700 compilables) × modèle biaisé (logits de quelques
    caractères dopés) : max_whitespace 0/1/4, budgets 1..100 dont == fermeture minimale,
    force_complete 85 %, T 0/0.7/1.5, top-k/top-p/pénalité."""
    model = r3_biased_char
    V = len(R3_CHARS)
    n_schemas = n_complete = 0
    for seed in range(chunk * 250, chunk * 250 + 250):
        rng = random.Random(seed)
        schema = r3_valid_schema(rng)
        if schema is None:
            continue
        c = structured.json_constraint(R3_TB, schema, max_whitespace=rng.choice([0, 1, 4]))
        first = c.completion_tokens()
        bias = torch.zeros(V)
        for i in rng.sample(range(V), rng.randint(1, 12)):
            bias[i] = rng.uniform(1, 8)
        model.bias = bias
        budget = rng.choice([1, 3, 6, 12, 25, 50, 100])
        if first is not None and rng.random() < 0.3:
            budget = len(first)
        fc = rng.random() < 0.85
        kw = dict(temperature=rng.choice([0, 0.7, 1.5]), top_k=rng.choice([None, 3, 40]),
                  top_p=rng.choice([None, 0.9]), repetition_penalty=rng.choice([1.0, 1.3]))
        torch.manual_seed(seed)
        toks = list(model.generate_structured(torch.tensor([[1]]), c, max_new_tokens=budget,
                                              force_complete=fc, **kw))
        label = f"seed={seed} schema={json.dumps(schema)[:90]} budget={budget} fc={fc} {kw}"
        r3_check(c, toks, r3_char_decode, schema, budget, first if fc else None, label)
        n_schemas += 1
        n_complete += c.is_complete()
    assert n_schemas > 120 and n_complete > n_schemas // 2, (n_schemas, n_complete)


@pytest.fixture(scope="module")
def r3_biased_gpt2():
    torch.manual_seed(0)
    m = _BiasedPOPIXA(POPIXAConfig(block_size=48, vocab_size=50257, n_layer=2, n_head=2,
                                   n_embd=32, dropout=0.0))
    m.train(False)
    return m


@pytest.mark.parametrize("chunk", range(2))
def test_r3_fuzz_random_schemas_biased_gpt2(gpt2, r3_biased_gpt2, chunk):
    """Vocabulaire gpt2 : 300 schémas aléatoires, ~3000 tokens dopés par tirage
    (tokens multi-octets, UTF-8 partiel, échappements, espaces multiples)."""
    enc, tb = gpt2
    model = r3_biased_gpt2
    n_schemas = 0
    for seed in range(chunk * 150, chunk * 150 + 150):
        rng = random.Random(70_000 + seed)
        schema = r3_valid_schema(rng)
        if schema is None:
            continue
        c = structured.json_constraint(tb, schema)
        first = c.completion_tokens()
        bias = torch.zeros(50257)
        bias[rng.sample(range(50257), 3000)] = torch.rand(3000) * 7 + 1
        model.bias = bias
        budget = rng.choice([1, 3, 6, 12, 25, 50])
        if first is not None and rng.random() < 0.3:
            budget = len(first)
        kw = dict(temperature=rng.choice([0, 0.7, 1.5]), top_k=rng.choice([None, 40]),
                  top_p=rng.choice([None, 0.9]), repetition_penalty=rng.choice([1.0, 1.3]))
        prompt = enc.encode_ordinary(rng.choice(["", "JSON : ", "x" * 60]))
        torch.manual_seed(seed)
        cache = []
        toks = list(model.generate_structured(torch.tensor([prompt or [0]][:1]), c,
                                              max_new_tokens=budget, cache_ref=cache, **kw))
        label = f"gpt2 seed={seed} schema={json.dumps(schema)[:90]} budget={budget} {kw}"
        r3_check(c, toks, enc.decode, schema, budget, first, label)
        check_cache(cache, prompt or [0], toks, 48, label)
        n_schemas += 1
    assert n_schemas > 70


R3_CORPUS = ['{"nom": "Dupont", "age": 42, "ok": true}', '[1, 2.5, -3e2, null, false]',
             '{"a": [{"b": "é"}], "c": {}}', '"x\\"y\\\\z\\u00e9\\n"', '{"kind": "a", "v": 7}',
             '[[null, null], [null, null]]', '{"x": 1.5, "y": -2}', '"rouge"', '"€uro"',
             '"naïve"']
R3_ATOMS = list('{}[],:" \n-0123456789.eE+truefalsn\\abcdxyzé')


def r3_random_itos(rng):
    """Vocabulaire « BPE » aléatoire : ~80 % des caractères seuls + fragments de JSON
    (2 à 6 caractères) ; parfois deux ids aux octets identiques."""
    toks = {ch for ch in R3_ATOMS if rng.random() < 0.8}
    for _ in range(rng.randint(5, 80)):
        s = rng.choice(R3_CORPUS)
        i = rng.randrange(len(s))
        toks.add(s[i:min(len(s), i + rng.randint(2, 6))])
    toks = sorted(toks)
    rng.shuffle(toks)
    if rng.random() < 0.3:
        toks.append(toks[0])
    return dict(enumerate(toks))


@pytest.mark.parametrize("chunk", range(3))
def test_r3_fuzz_multichar_vocabularies(chunk):
    """Vocabulaires multi-caractères aléatoires (segmentation avec retour arrière, A* quand
    un caractère manque, doublons d'octets) : invariants de generate_structured + cache."""
    for seed in range(chunk * 200, chunk * 200 + 200):
        rng = random.Random(90_000 + seed)
        itos = r3_random_itos(rng)
        V = len(itos)
        tb = structured.token_bytes_from_itos(itos, V)
        bs = rng.choice([16, 64])
        model = make_model(V, block_size=bs, seed=seed % 5)
        schema = (R2_SCHEMAS[rng.randrange(len(R2_SCHEMAS))] if rng.random() < 0.6
                  else r3_valid_schema(rng))
        ws = rng.choice([0, 1, 4])
        first = structured.json_constraint(tb, schema, max_whitespace=ws).completion_tokens()
        budget = rng.choice([1, 2, 3, 5, 8, 13, 30, 60, 120])
        if first is not None and rng.random() < 0.4:
            budget = len(first) + rng.choice([0, 0, 1])
        kw = dict(temperature=rng.choice([0, 0.7, 1.5]), top_k=rng.choice([None, 1, 5]),
                  top_p=rng.choice([None, 0.5, 0.95]),
                  repetition_penalty=rng.choice([1.0, 1.3, 0.7]))
        prompt = [rng.randrange(V) for _ in range(rng.choice([0, 1, 10, 70]))]
        fc = rng.random() < 0.85
        c = structured.json_constraint(tb, schema, max_whitespace=ws)
        torch.manual_seed(seed)
        cache = []
        toks = list(model.generate_structured(torch.tensor([prompt or [0]][:1]), c,
                                              max_new_tokens=budget, cache_ref=cache,
                                              force_complete=fc, **kw))
        dec = lambda t: "".join(itos[i] for i in t)     # noqa: E731
        label = f"seed={seed} V={V} schema={json.dumps(schema)[:70]} budget={budget} fc={fc}"
        r3_check(c, toks, dec, schema, budget, first if fc else None, label)
        check_cache(cache, prompt or [0], toks, bs, label)


def _r3_kv_matches_fresh_prefill(model, kv):
    """Le KV-cache final doit être EXACTEMENT celui d'un prefill frais sur ses token_ids
    (positions RoPE 0..n-1) — pas seulement porter les bons ids."""
    with torch.no_grad():
        _, ref = model(torch.tensor([kv.token_ids]))
    worst = 0.0
    for (k, v), (rk, rv) in zip(kv, ref):
        assert k.shape == rk.shape, (tuple(k.shape), tuple(rk.shape))
        worst = max(worst, (k - rk).abs().max().item(), (v - rv).abs().max().item())
    return worst


@pytest.mark.parametrize("chunk", range(2))
def test_r3_kv_cache_content_multi_turn_chains(chunk):
    """Chaînes de 2 à 5 tours (structuré 70 % / generate_stream 30 %) où cache_ref[0] du
    tour précédent devient initial_past_kvs ; entrées vides (rejeu du dernier token),
    fermetures anticipées (gen.close()), débordements de block_size 8..64."""
    V = len(CHARS)
    for seed in range(chunk * 120, chunk * 120 + 120):
        rng = random.Random(40_000 + seed)
        bs = rng.choice([8, 12, 24, 64])
        model = make_model(V, block_size=bs, seed=seed % 7)
        kv = None
        for turn in range(rng.randint(2, 5)):
            prompt = [rng.randrange(V) for _ in range(rng.choice([0, 1, 3, 20]))]
            if kv is None and not prompt:
                prompt = [rng.randrange(V)]
            budget = rng.choice([1, 4, 10, 30, 70])
            kw = dict(temperature=rng.choice([0, 1.0, 1.5]), top_k=rng.choice([None, 5]),
                      top_p=rng.choice([None, 0.9]))
            cache = []
            idx = torch.tensor([prompt], dtype=torch.long)
            torch.manual_seed(seed * 10 + turn)
            if rng.random() < 0.7:
                schema = r3_valid_schema(rng)
                c = structured.json_constraint(CHAR_TB, schema)
                gen = model.generate_structured(idx, c, max_new_tokens=budget, cache_ref=cache,
                                                initial_past_kvs=kv,
                                                force_complete=rng.random() < 0.8, **kw)
            else:
                gen = model.generate_stream(idx, budget, kw["temperature"], kw["top_k"], 1.0,
                                            kw["top_p"], False, initial_past_kvs=kv,
                                            cache_ref=cache)
            toks = []
            stop_at = rng.choice([None, None, None, 1, 3])
            for t in gen:
                toks.append(t)
                if stop_at is not None and len(toks) >= stop_at:
                    gen.close()
                    break
            label = f"seed={seed} tour={turn} bs={bs}"
            prefix = (list(kv.token_ids) + prompt) if kv is not None else prompt
            check_cache(cache, prefix, toks, bs, label)
            worst = _r3_kv_matches_fresh_prefill(model, cache[0])
            assert worst < 1e-4, (label, worst)
            kv = cache[0]


# ─────────────────────────────────────────────────────────────────────────────
# Échantillonnage par rejet : la docstring de generate_structured promet un tirage
# EXACT (distribution masquée renormalisée) sans top-k/top-p — vérifié statistiquement.
# ─────────────────────────────────────────────────────────────────────────────

class _ScaledPOPIXA(nanoPOPIXA):
    """Logits ×6 : distributions piquées (sinon un modèle aléatoire est quasi uniforme)."""

    def forward(self, idx, targets=None, past_kvs=None):
        logits, kv = super().forward(idx, targets, past_kvs)
        return (logits * 6 if targets is None else logits), kv


@pytest.mark.parametrize("schema,temp,penalty", [
    (None, 1.0, 1.0),
    ({"type": "object", "properties": {"a": {"type": "integer"}}}, 1.3, 1.0),
    ({"enum": ["rouge", "vert", "bleu", 3, None]}, 0.7, 1.4),
], ids=["any", "object", "enum-penalty"])
def test_r3_rejection_sampling_first_token_is_exact(schema, temp, penalty):
    torch.manual_seed(3)
    model = _ScaledPOPIXA(POPIXAConfig(block_size=32, vocab_size=len(CHARS), n_layer=2,
                                       n_head=2, n_embd=32, dropout=0.0))
    model.train(False)
    prompt = [CHARS.index(ch) for ch in "fiche "]
    c = structured.json_constraint(CHAR_TB, schema)
    with torch.no_grad():
        logits, _ = model(torch.tensor([prompt]))
    expected = model._apply_sampling(logits, temp, None, None, penalty, torch.tensor([prompt]),
                                     mask=c.allowed_mask())[0]
    N = 3000
    counts = torch.zeros(len(CHARS))
    for s in range(N):
        c.reset()
        torch.manual_seed(s)
        gen = model.generate_structured(torch.tensor([prompt]), c, max_new_tokens=60,
                                        temperature=temp, repetition_penalty=penalty)
        counts[next(gen)] += 1
        gen.close()
    tv = 0.5 * (counts / N - expected).abs().sum().item()
    assert tv < 0.06, f"distance en variation totale {tv:.3f} (attendu ≈ 0)"
    # greedy : argmax des logits autorisés
    c.reset()
    gen = model.generate_structured(torch.tensor([prompt]), c, max_new_tokens=60,
                                    temperature=0, repetition_penalty=penalty)
    assert next(gen) == int(expected.argmax())
    gen.close()


# ─────────────────────────────────────────────────────────────────────────────
# Régressions : nombres non finis dans le schéma (json.loads accepte NaN / Infinity /
# 1e400) — jamais émis : la sortie reste du JSON RFC 8259 strict.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("src", ['{"enum": [NaN, 1]}', '{"type":"number","enum":[1e400, 2]}',
                                 '{"anyOf": [{"const": -Infinity}, {"type": "boolean"}]}',
                                 '{"default": NaN, "type": "integer"}'])
def test_r3_nonfinite_schema_numbers_never_generated(src, char_model):
    schema = structured.load_schema(src)
    for seed in range(4):
        c, toks, _ = generate(char_model, CHAR_TB, schema, 30, [1], seed, temperature=1.5)
        assert c.is_complete()
        r3_check(c, toks, char_decode, schema, 30, None, src)


# ─────────────────────────────────────────────────────────────────────────────
# Harnais CLI : le sous-processus importe bien le structured.py du scratch
# ─────────────────────────────────────────────────────────────────────────────

def test_r3_cli_harness_uses_scratch_structured(tmp_path):
    r = popixa_gen(tmp_path, "--help")
    assert r.returncode == 0 and "Traceback" not in r.stderr, r.stderr[-400:]
    assert "--schema" in r.stdout


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT structured.py — load_schema choisit fichier / JSON inline avec os.path.isfile :
# un chemin qui EXISTE mais n'est pas un fichier ordinaire (tube nommé, substitution de
# processus bash `--schema <(jq … schema.json)`, `cat schema.json | popixa gen --schema
# /dev/stdin`) est traité comme du JSON inline → « fichier introuvable et JSON inline
# invalide », code de sortie 1, alors que le schéma est lisible.
# ─────────────────────────────────────────────────────────────────────────────

def test_r3_load_schema_from_pipe_path():
    r, w = os.pipe()
    os.write(w, json.dumps(SCHEMAS[15]).encode("utf-8"))
    os.close(w)
    try:
        assert structured.load_schema(f"/dev/fd/{r}") == SCHEMAS[15]
    finally:
        os.close(r)


def test_r3_load_schema_from_named_fifo(tmp_path):
    import threading
    fifo = str(tmp_path / "schema.fifo")
    os.mkfifo(fifo)

    def writer():
        with open(fifo, "w", encoding="utf-8") as f:
            f.write(json.dumps(SCHEMAS[15]))

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        assert structured.load_schema(fifo) == SCHEMAS[15]
    finally:
        if t.is_alive():                 # débloque l'écrivain si le tube n'a pas été lu
            with open(fifo, "r", encoding="utf-8") as f:
                f.read()
        t.join(timeout=5)


@pytest.mark.parametrize("how", ["process-substitution", "stdin-pipe"])
def test_r3_cli_gen_schema_from_pipe(tmp_path, how):
    ckpt = save_char_ckpt(str(tmp_path / "ck" / "char.pt"), block_size=128)
    env = dict(os.environ, PYTHONPATH=SCRATCH + os.pathsep + REPO, PY=sys.executable,
               BOOT=_CLI_BOOT, SCR=SCRATCH, CK=ckpt, SCH=json.dumps(SCHEMAS[15]))
    gen = '"$PY" -c "$BOOT" "$SCR" 0 gen --checkpoint "$CK" --tokens 60 --temp 1.0'
    script = (f'{gen} --schema <(printf %s "$SCH")' if how == "process-substitution"
              else f'printf %s "$SCH" | {gen} --schema /dev/stdin')
    r = subprocess.run(["bash", "-c", script], cwd=str(tmp_path), env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, ANSI.sub("", r.stderr)[-300:]
    assert structured.validate_instance(json.loads(r.stdout), SCHEMAS[15]) == []


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT chat.py — report_json attribue TOUT JSON incomplet au budget /tokens
# (« budget de tokens épuisé — augmente /tokens ») : faux quand le tour a été
# interrompu (Ctrl+C) ou quand c'est /taskbudget qui a plafonné le budget du tour
# (resp_budget = min(/tokens, reste du task budget)) — augmenter /tokens n'y change rien.
# ─────────────────────────────────────────────────────────────────────────────

R3_FICHE = {"type": "object", "properties": {"nom": {"type": "string", "minLength": 10},
                                             "age": {"type": "integer"}},
            "required": ["nom", "age"]}


def _json_turn_report(out):
    i = out.find("{json}")
    assert i >= 0, out[-800:]
    return out[i:i + 600]


def test_r3_chat_json_interrupted_turn_not_blamed_on_tokens(monkeypatch, capsys, tmp_path):
    import model as model_mod
    monkeypatch.chdir(tmp_path)
    ckpt = save_char_ckpt(str(tmp_path / "out-nanopopixa" / "checkpoint.pt"), block_size=256)
    orig = model_mod.nanoPOPIXA.forward
    calls = {"n": 0}

    def ctrl_c(self, idx, targets=None, past_kvs=None):
        if targets is None:
            calls["n"] += 1
            if calls["n"] == 6:          # Ctrl+C pendant la génération du JSON
                raise KeyboardInterrupt
        return orig(self, idx, targets, past_kvs)

    monkeypatch.setattr(model_mod.nanoPOPIXA, "forward", ctrl_c)
    out = run_script(monkeypatch, capsys, ckpt,
                     ["/json " + json.dumps(R3_FICHE), "/temp 1", "fiche"], max_tokens=200)
    report = _json_turn_report(out)
    assert "[Interruption]" in report
    assert "augmente /tokens" not in report, report


def test_r3_chat_json_taskbudget_cap_not_blamed_on_tokens(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    ckpt = save_char_ckpt(str(tmp_path / "out-nanopopixa" / "checkpoint.pt"), block_size=256)
    closing = len(first_closing(CHAR_TB, R3_FICHE))
    # /tokens 200 ≥ fermeture minimale (28), mais task budget 25 − 15 tokens de prompt
    # → 10 tokens pour le JSON : c'est /taskbudget, pas /tokens, qui tronque
    assert 25 - len("fiche du client") < closing <= 200
    out = run_script(monkeypatch, capsys, ckpt,
                     ["/json " + json.dumps(R3_FICHE), "/taskbudget 25", "/tokens 200",
                      "fiche du client"], max_tokens=200)
    report = _json_turn_report(out)
    assert "augmente /tokens" not in report, report


# ─────────────────────────────────────────────────────────────────────────────
# chat.run_chat — régressions de bout en bout
# ─────────────────────────────────────────────────────────────────────────────

def test_r3_chat_json_session_restored_across_runs(monkeypatch, capsys, tmp_path):
    """Deux lancements du chat : tours JSON, sortie, session restaurée depuis le disque,
    nouveaux tours JSON sur le KV-cache restauré (fenêtre 64 → glissement)."""
    import chat
    from session_cache import load_session
    monkeypatch.chdir(tmp_path)
    ckpt = save_char_ckpt(str(tmp_path / "out-nanopopixa" / "checkpoint.pt"), block_size=64)
    schema = json.dumps(SCHEMAS[16])
    out1 = run_script(monkeypatch, capsys, ckpt, [f"/json {schema}", "/temp 1", "un", "deux"],
                      max_tokens=30)
    assert "Traceback" not in out1 and "Erreur" not in out1
    assert out1.count("✓ schéma respecté") == 2, out1[-800:]
    out2 = run_script(monkeypatch, capsys, ckpt, [f"/json {schema}", "/temp 1", "trois",
                                                  "quatre", "/json off", "libre"],
                      max_tokens=30, seed=1)
    assert "Traceback" not in out2 and "Erreur" not in out2
    assert out2.count("✓ schéma respecté") == 2, out2[-800:]
    past, ids, _ = load_session(chat.CACHE_PATH, ckpt, "cpu", with_history=True)
    if past is not None:
        assert past.token_ids == ids and past.seq_len == len(ids) <= 64


def test_r3_chat_json_padded_gpt2_vocab(monkeypatch, capsys, tmp_path, gpt2):
    """Checkpoint gpt2 au vocabulaire paddé (50304, à la nanoGPT) : load_token_bytes
    complète par None → masque aligné sur le modèle, ids de padding jamais émis."""
    monkeypatch.chdir(tmp_path)
    m = make_model(50304, block_size=128)
    path = tmp_path / "out-nanopopixa" / "checkpoint.pt"
    os.makedirs(path.parent, exist_ok=True)
    torch.save({"model": m.state_dict(), "config": m.config, "iter": 0,
                "tokenizer": "tiktoken_gpt2"}, str(path))
    out = run_script(monkeypatch, capsys, str(path),
                     [f"/json {json.dumps(SCHEMAS[15])}", "/temp 1.5", "un", "deux", "trois"],
                     max_tokens=40)
    assert "Traceback" not in out and "Erreur" not in out
    assert out.count("✓ schéma respecté") == 3, out[-800:]


def _r3_chat_lines(rng):
    lines = []
    for _ in range(rng.randint(4, 14)):
        r = rng.random()
        if r < 0.25:
            s = r3_valid_schema(rng)
            lines.append("/json" if s is None
                         else "/json " + json.dumps(s, ensure_ascii=rng.random() < 0.5))
        elif r < 0.3:
            lines.append(rng.choice(["/json", "/json off", "/json {bad", "/json false",
                                     "/json []"]))
        elif r < 0.45:
            lines.append("/tokens " + str(rng.choice([0, 1, 3, 10, 40, 80, 150])))
        elif r < 0.5:
            lines.append("/temp " + str(rng.choice([0, 0.5, 1.0, 1.5])))
        elif r < 0.55:
            lines.append("/topp " + str(rng.choice([0.5, 0.9, 1.0])))
        elif r < 0.58:
            lines.append("/penalty " + str(rng.choice([1.0, 1.3, 0.8])))
        elif r < 0.62:
            lines.append("/effort " + rng.choice(["low", "medium", "high", "max"]))
        elif r < 0.66:
            lines.append(rng.choice(["/reset", "/clearcache", "/think", "/fast",
                                     "/interleaved", "/ctx"]))
        else:
            lines.append(rng.choice(["fiche client", "bonjour", "donne un JSON",
                                     "x" * rng.randint(1, 80), "é à ç"]))
    return lines


@pytest.mark.parametrize("vocab", ["char", "gpt2"])
def test_r3_chat_fuzz_random_command_sequences(monkeypatch, capsys, tmp_path, vocab):
    """Sessions de chat aléatoires : /json <schéma aléatoire> | off | invalide, /tokens 0..150,
    /temp, /topp, /penalty, /effort, /reset, /clearcache, /think, /fast… Chaque tour JSON
    est vérifié via un wrapper de chat.stream_json : complet ⇒ json.loads + validate == [] ;
    budget ≥ fermeture minimale ⇒ complet ; rapports « ✓ / incomplet » cohérents ;
    session persistée exacte."""
    import chat
    from session_cache import load_session
    if vocab == "gpt2":
        pytest.importorskip("tiktoken")
    records = []
    orig_stream_json = chat.stream_json

    def recording_stream_json(model, encode, decode, context_str, device, constraint,
                              max_tokens, *a, **kw):
        probe = constraint.clone()
        probe.reset()
        first = probe.completion_tokens()
        text = orig_stream_json(model, encode, decode, context_str, device, constraint,
                                max_tokens, *a, **kw)
        records.append(dict(budget=max_tokens, first=None if first is None else len(first),
                            complete=constraint.is_complete(), gen=constraint.generated,
                            schema=constraint.matcher.schema))
        return text

    monkeypatch.setattr(chat, "stream_json", recording_stream_json)
    n_sessions = 24 if vocab == "char" else 6
    for seed in range(n_sessions):
        rng = random.Random(55_000 + seed)
        d = tmp_path / str(seed)
        os.makedirs(d / "out-nanopopixa")
        monkeypatch.chdir(d)
        bs = rng.choice([32, 64, 256])
        ck = str(d / "out-nanopopixa" / "checkpoint.pt")
        if vocab == "char":
            save_char_ckpt(ck, block_size=bs, seed=seed)
        else:
            m = make_model(50257, block_size=bs, seed=seed)
            torch.save({"model": m.state_dict(), "config": m.config, "iter": 0,
                        "tokenizer": "tiktoken_gpt2"}, ck)
        lines = _r3_chat_lines(rng)
        records.clear()
        out = run_script(monkeypatch, capsys, ck, lines, max_tokens=rng.choice([20, 60, 120]),
                         seed=seed)
        label = f"session {seed} : {lines}"
        assert "Traceback" not in out and "Erreur de génération" not in out, label
        for rec in records:
            if rec["complete"]:
                inst = json.loads(rec["gen"].decode("utf-8"))
                assert structured.validate_instance(inst, rec["schema"]) == [], (label, rec)
            elif rec["first"] is not None:
                assert rec["budget"] < rec["first"], (label, rec)
        assert out.count("✓ ") + out.count("JSON incomplet") == len(records), label
        assert out.count("✗ JSON hors schéma") + out.count("  ✗ JSON invalide :") == 0, label
        past, ids, _ = load_session(chat.CACHE_PATH, ck, "cpu", with_history=True)
        if past is not None:
            assert past.token_ids == ids and past.seq_len == len(ids) <= bs, label


def test_r3_cli_gen_random_schemas(tmp_path, gpt2):
    """`popixa gen --schema` (fichier ou inline) sur schémas aléatoires, vocabulaires char et
    gpt2, prompts variés, --tokens autour de la fermeture minimale : code 0 + JSON conforme
    quand --tokens ≥ fermeture, sinon code 1 sans rien sur stdout."""
    enc, gtb = gpt2
    ck = {"char": save_char_ckpt(str(tmp_path / "ck" / "char.pt"), block_size=64),
          "gpt2": save_gpt2_ckpt(str(tmp_path / "ck" / "gpt2.pt"), block_size=64)}
    tbs = {"char": CHAR_TB, "gpt2": gtb}
    done = 0
    for seed in range(40):
        rng = random.Random(33_000 + seed)
        schema = r3_valid_schema(rng)
        if schema is None:
            continue
        voc = rng.choice(["char", "gpt2"])
        first = first_closing(tbs[voc], schema)
        tokens = rng.choice([len(first) - 1, len(first), len(first) + 3, 40, 100])
        args = ["--checkpoint", ck[voc], "--tokens", str(tokens),
                "--temp", str(rng.choice([0, 0.7, 1.5])), "--top_k", str(rng.choice([0, 1, 40])),
                "--penalty", str(rng.choice([1.0, 1.3]))]
        if rng.random() < 0.5:
            args += ["--top_p", str(rng.choice([0.5, 0.95]))]
        if rng.random() < 0.6:
            args += ["--prompt", rng.choice(["Fiche : ", '{"nom": ', "x" * 100, "é"])]
        if rng.random() < 0.5:
            p = tmp_path / f"s{seed}.json"
            p.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
            args += ["--schema", str(p)]
        else:
            args += ["--schema", json.dumps(schema)]
        r = popixa_gen(tmp_path, *args, seed=seed)
        label = (seed, voc, tokens, len(first), json.dumps(schema)[:80])
        assert "Traceback" not in r.stderr, (label, r.stderr[-400:])
        if tokens >= len(first):
            assert r.returncode == 0, (label, ANSI.sub("", r.stderr)[-200:])
            assert structured.validate_instance(json.loads(r.stdout), schema) == [], label
        else:
            assert r.returncode == 1 and not r.stdout.strip(), label
        done += 1
        if done >= 14:
            break
    assert done >= 10


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT chat.py — checkpoint gpt2 au vocabulaire paddé (vocab_size 50304 > 50257) :
# load_token_bytes gère ce padding pour /json (ids ≥ 50257 → None, jamais autorisés),
# mais le decode des tours NON structurés est `enc.decode(l)` brut : dès que le modèle
# échantillonne un id de padding (≈ 0,1 % des tokens pour un modèle non entraîné,
# trouvé par le fuzz de sessions ci-dessus), tiktoken lève
# « KeyError: Invalid token for decoding: 5026x » → « ✗ Erreur de génération », tour
# perdu, cache de session jeté. (Ici le modèle est poussé vers un id de padding pour
# rendre le scénario déterministe.)
# ─────────────────────────────────────────────────────────────────────────────

def test_r3_chat_plain_turn_padded_gpt2_vocab(monkeypatch, capsys, tmp_path, gpt2):
    import model as model_mod
    monkeypatch.chdir(tmp_path)
    m = make_model(50304, block_size=128)
    path = tmp_path / "out-nanopopixa" / "checkpoint.pt"
    os.makedirs(path.parent, exist_ok=True)
    torch.save({"model": m.state_dict(), "config": m.config, "iter": 0,
                "tokenizer": "tiktoken_gpt2"}, str(path))
    orig = model_mod.nanoPOPIXA.forward

    def leaning_to_padding(self, idx, targets=None, past_kvs=None):
        logits, kv = orig(self, idx, targets, past_kvs)
        if targets is None:
            logits = logits.clone()
            logits[..., 50260] += 4.0
        return logits, kv

    monkeypatch.setattr(model_mod.nanoPOPIXA, "forward", leaning_to_padding)
    out = run_script(monkeypatch, capsys, str(path), ["/temp 1", "bonjour"], max_tokens=40)
    assert "Erreur de génération" not in out, out[out.find("Erreur"):][:160]
    # le même modèle en mode /json n'émet jamais d'id de padding (masque aligné)
    out = run_script(monkeypatch, capsys, str(path),
                     [f"/json {json.dumps(SCHEMAS[15])}", "/temp 1", "fiche"], max_tokens=40)
    assert "Erreur de génération" not in out and "✓ schéma respecté" in out
