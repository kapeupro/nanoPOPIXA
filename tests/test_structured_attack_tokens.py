"""
Tests adverses — niveau TOKENS (TokenConstraint) et performance.

  1. Vocabulaires : tokens spéciaux exclus (gpt2, cl100k, o200k), vocabulaire paddé.
  2. Masque exact :
       - vs un ORACLE INDÉPENDANT (parseur JSON de préfixes écrit ici) pour le JSON libre,
         sur des états atteints par marches aléatoires et par tokenisation gpt2 réelle
         (tokens à cheval sur la structure : '":"', '"],"', '":-', ' "\\'…) ;
       - vs force brute (is_allowed sur tout le vocabulaire) pour des schémas riches ;
       - cohérence : tout token autorisé est un préfixe viable de JSON libre, et la
         complétion la plus courte derrière lui produit une instance valide.
  3. Tokens contenant des séquences UTF-8 partielles (chaînes libres, maxLength, constantes).
  4. Caches : pas de masque périmé après advance, clones indépendants, tenseur renvoyé
     non partagé avec le cache, éviction forcée (petits plafonds) toujours exacte.
  5. completion_tokens : octets == shortest_completion, tokens autorisés un à un,
     contrat « None si impossible avec ce vocabulaire ».
  6. Mémoire : génération aléatoire de 10k tokens (plafonds des caches) ; imbrication
     profonde (10k tokens '[') mesurée dans un sous-processus.
  7. Chronométrage : construction du trie, masque neuf dans une chaîne, masque en cache,
     is_allowed.
"""

import json
import os
import random
import re
import subprocess
import sys
import textwrap
import time

import pytest
import torch

import structured as S
from structured import (
    JSONSchemaMatcher, TokenConstraint, json_constraint, token_bytes_from_itos,
    token_bytes_from_tiktoken, validate_instance,
)

try:
    import tiktoken
except ImportError:     # pragma: no cover
    tiktoken = None


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))


# ─────────────────────────────────────────────────────────────────────────────
# Oracle indépendant : « data est-il un préfixe d'un document JSON accepté ? »
# (même sous-ensemble que le module : ≤ max_ws octets d'espace par interstice et
#  avant la racine, rien après la racine, ≤ D chiffres par série, UTF-8 strict)
# ─────────────────────────────────────────────────────────────────────────────

class _Incomplete(Exception):
    pass


class _Invalid(Exception):
    pass


class _Prefix:
    WS = b' \t\n\r'

    def __init__(self, data, max_ws, digits):
        self.d, self.i, self.max_ws, self.D = data, 0, max_ws, digits

    def peek(self):
        if self.i >= len(self.d):
            raise _Incomplete
        return self.d[self.i]

    def ws(self):
        n = 0
        while self.i < len(self.d) and self.d[self.i] in self.WS:
            self.i += 1
            n += 1
            if n > self.max_ws:
                raise _Invalid
        self.peek()

    def expect(self, b):
        if self.peek() != b:
            raise _Invalid
        self.i += 1

    def value(self):
        b = self.peek()
        if b == 0x7B:
            return self.obj()
        if b == 0x5B:
            return self.arr()
        if b == 0x22:
            return self.string()
        if b == 0x2D or 0x30 <= b <= 0x39:
            return self.number()
        for lit in (b'true', b'false', b'null'):
            if b == lit[0]:
                for x in lit:
                    self.expect(x)
                return
        raise _Invalid

    def obj(self):
        self.expect(0x7B)
        self.ws()
        if self.peek() == 0x7D:
            self.i += 1
            return
        while True:
            if self.peek() != 0x22:
                raise _Invalid
            self.string()
            self.ws()
            self.expect(0x3A)
            self.ws()
            self.value()
            self.ws()
            b = self.peek()
            self.i += 1
            if b == 0x2C:
                self.ws()
                continue
            if b == 0x7D:
                return
            raise _Invalid

    def arr(self):
        self.expect(0x5B)
        self.ws()
        if self.peek() == 0x5D:
            self.i += 1
            return
        while True:
            self.value()
            self.ws()
            b = self.peek()
            self.i += 1
            if b == 0x2C:
                self.ws()
                continue
            if b == 0x5D:
                return
            raise _Invalid

    def string(self):
        self.expect(0x22)
        while True:
            b = self.peek()
            self.i += 1
            if b == 0x22:
                return
            if b == 0x5C:
                e = self.peek()
                self.i += 1
                if e == 0x75:
                    for _ in range(4):
                        if self.peek() not in b'0123456789abcdefABCDEF':
                            raise _Invalid
                        self.i += 1
                elif e not in b'"\\/bfnrt':
                    raise _Invalid
                continue
            if b < 0x20:
                raise _Invalid
            if b < 0x80:
                continue
            if 0xC2 <= b <= 0xDF:
                conts = [(0x80, 0xBF)]
            elif b == 0xE0:
                conts = [(0xA0, 0xBF), (0x80, 0xBF)]
            elif 0xE1 <= b <= 0xEC or b in (0xEE, 0xEF):
                conts = [(0x80, 0xBF)] * 2
            elif b == 0xED:
                conts = [(0x80, 0x9F), (0x80, 0xBF)]
            elif b == 0xF0:
                conts = [(0x90, 0xBF)] + [(0x80, 0xBF)] * 2
            elif 0xF1 <= b <= 0xF3:
                conts = [(0x80, 0xBF)] * 3
            elif b == 0xF4:
                conts = [(0x80, 0x8F)] + [(0x80, 0xBF)] * 2
            else:
                raise _Invalid
            for lo, hi in conts:
                if not lo <= self.peek() <= hi:
                    raise _Invalid
                self.i += 1

    def digits(self):
        n = 0
        while self.i < len(self.d) and 0x30 <= self.d[self.i] <= 0x39:
            self.i += 1
            n += 1
            if n > self.D:
                raise _Invalid
        if n == 0:
            self.peek()
            raise _Invalid
        self.peek()     # fin des données au milieu d'un nombre → viable

    def number(self):
        if self.peek() == 0x2D:
            self.i += 1
        b = self.peek()
        if b == 0x30:
            self.i += 1
            self.peek()
        elif 0x31 <= b <= 0x39:
            self.digits()
        else:
            raise _Invalid
        if self.peek() == 0x2E:
            self.i += 1
            self.digits()
        if self.peek() in b'eE':
            self.i += 1
            if self.peek() in b'+-':
                self.i += 1
            self.digits()


def viable(data: bytes, max_ws: int = 4, digits: int = 20) -> bool:
    p = _Prefix(data, max_ws, digits)
    try:
        p.ws()
        p.value()
    except _Incomplete:
        return True
    except _Invalid:
        return False
    return p.i == len(data)


def test_oracle_sanity():
    assert viable(b'') and viable(b'    ') and not viable(b'     ')
    assert viable(b'{"a": [1, 2') and viable(b'[1,') and not viable(b'[1,]')
    assert viable(b'12') and not viable(b'12 ') and not viable(b'{} ') and not viable(b'01')
    assert viable(b'"\xe2\x80') and not viable(b'"\xe2a') and not viable(b'"\xed\xa0')
    assert viable(b'"\\ud83d') and not viable(b'"\\x') and not viable(b'"\x01')
    assert viable(b'-') and not viable(b'-a') and viable(b'1e+') and not viable(b'1e+]')
    assert viable(b'[' + b'1' * 20) and not viable(b'[' + b'1' * 21)
    assert viable(b'{"a"     :') is False and viable(b'{"a"    :') is True


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulaires
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _torch_single_thread():
    """nonzero()/sum() sur 50k booléens : le pool OpenMP s'effondre sous contention CPU."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(n)


@pytest.fixture(scope="module")
def gpt2():
    if tiktoken is None:
        pytest.skip("tiktoken indisponible")
    enc = tiktoken.get_encoding("gpt2")
    return enc, token_bytes_from_tiktoken(enc)


def _char_vocab(extra=''):
    chars = sorted(set([chr(i) for i in range(32, 127)] + list('\n\t\ré—中😀' + extra)))
    return chars, token_bytes_from_itos({i: ch for i, ch in enumerate(chars)}, len(chars) + 2)


_LETTERS = re.compile(rb'[A-Za-z]')


def _structural_ids(tb):
    """Tokens sans lettre ASCII (ponctuation, chiffres, espaces, octets non ASCII)."""
    return [i for i, t in enumerate(tb) if t is not None and not _LETTERS.search(t)]


@pytest.mark.parametrize("name", ["gpt2", "cl100k_base", "o200k_base"])
def test_special_and_unassigned_tokens_are_none(name):
    if tiktoken is None:
        pytest.skip("tiktoken indisponible")
    try:
        enc = tiktoken.get_encoding(name)
    except Exception as e:     # pragma: no cover — encodage absent du cache
        pytest.skip(f"{name} indisponible : {e}")
    tb = token_bytes_from_tiktoken(enc)
    assert len(tb) == enc.n_vocab
    special = {enc.encode_single_token(s) for s in enc.special_tokens_set}
    expected_none = set(special)
    for i in range(enc.n_vocab):
        if i in special:
            continue
        try:
            enc.decode_single_token_bytes(i)
        except (KeyError, ValueError):
            expected_none.add(i)
    assert {i for i, t in enumerate(tb) if t is None} == expected_none
    c = json_constraint(tb, {"type": "string"})
    c.state = c.matcher.advance_bytes(c.matcher.initial_state, b'"<|')
    mask = c.allowed_mask()
    assert int(mask.sum()) > 0.9 * enc.n_vocab
    assert not any(bool(mask[i]) for i in special)
    assert not any(c.is_allowed(i) for i in special)


def test_gpt2_eot_and_padding_never_allowed(gpt2):
    enc, tb = gpt2
    eot = enc.eot_token
    padded = list(tb) + [None] * (50304 - len(tb))        # comme chat.load_token_bytes
    c = json_constraint(padded, {"type": "object", "properties": {"t": {"type": "string"}},
                                 "required": ["t"]})
    m = c.matcher
    for prefix in (b'', b'{', b'{"t":"', b'{"t":"abc\\', b'{"t":"x"'):
        c.state = m.advance_bytes(m.initial_state, prefix)
        mask = c.allowed_mask()
        assert mask.shape == (50304,)
        assert not bool(mask[eot]) and not bool(mask[len(tb):].any())
        assert not c.is_allowed(eot) and not c.is_allowed(50303) and not c.is_allowed(50304)
        toks = c.completion_tokens()
        assert toks is not None and eot not in toks and all(t < len(tb) for t in toks)
    with pytest.raises(ValueError):
        c.advance(eot)
    with pytest.raises(ValueError):
        c.advance(50300)


# ─────────────────────────────────────────────────────────────────────────────
# Masque exact vs oracle indépendant (JSON libre, vocabulaire gpt2)
# ─────────────────────────────────────────────────────────────────────────────

def _rand_text(rng):
    pool = ['a', 'é', '😀', '\\', '"', '\n', '\t', '/', '中', ' ', 'x', '—', '\x7f', '\x01']
    return ''.join(rng.choice(pool) for _ in range(rng.randint(0, 6)))


def _rand_value(rng, d=0):
    r = rng.random()
    if d > 3 or r < 0.3:
        return rng.choice([0, -1, 12.5, 1e-7, -3e21, True, False, None, 123456789,
                           _rand_text(rng)])
    if r < 0.65:
        return [_rand_value(rng, d + 1) for _ in range(rng.randint(0, 4))]
    return {_rand_text(rng): _rand_value(rng, d + 1) for _ in range(rng.randint(0, 4))}


def _serialize(v, rng):
    k = rng.randrange(4)
    if k == 0:
        return json.dumps(v)
    if k == 1:
        return json.dumps(v, ensure_ascii=False, separators=(',', ':'))
    if k == 2:
        return json.dumps(v, indent=1)
    return json.dumps(v, ensure_ascii=False, separators=(' ,', ' : '))


def _check_against_oracle(c, tb, ids, max_ws):
    prefix = c.generated
    mask = c.allowed_mask().tolist()
    bad = [(prefix, tb[t]) for t in ids
           if tb[t] is not None and viable(prefix + tb[t], max_ws) != mask[t]]
    assert not bad, bad[:5]


def test_mask_vs_independent_oracle_random_walks_gpt2(gpt2):
    enc, tb = gpt2
    struct = _structural_ids(tb)
    c = json_constraint(tb, None)
    rng = random.Random(3)
    for _ in range(20):
        c.reset()
        for _ in range(rng.randint(0, 25)):
            ids = c.allowed_mask().nonzero().flatten().tolist()
            if not ids:
                break
            pool = [i for i in ids if not _LETTERS.search(tb[i])] if rng.random() < 0.7 else ids
            c.advance(rng.choice(pool or ids))
        _check_against_oracle(c, tb, struct + rng.sample(range(len(tb)), 300), 4)


def test_mask_vs_independent_oracle_real_tokenizations_gpt2(gpt2):
    """Préfixes de documents réels tokenisés par gpt2 (tokens à cheval sur la structure)."""
    enc, tb = gpt2
    struct = _structural_ids(tb)
    c = json_constraint(tb, None, max_whitespace=12)
    rng = random.Random(11)
    states = 0
    for _ in range(110):
        s = _serialize(_rand_value(rng), rng)
        toks = enc.encode(s, disallowed_special=())
        assert b''.join(tb[t] for t in toks) == s.encode('utf-8')
        c.reset()
        cut = rng.randint(0, len(toks))
        for t in toks[:cut]:
            assert c.allowed_mask()[t] and c.is_allowed(t), (s, c.generated, tb[t])
            c.advance(t)
        if cut == len(toks):
            assert c.is_complete() and json.loads(c.generated) == json.loads(s)
            continue
        _check_against_oracle(c, tb, struct + rng.sample(range(len(tb)), 150), 12)
        states += 1
    assert states > 45


# ─────────────────────────────────────────────────────────────────────────────
# Masque exact vs force brute (schémas riches, vocabulaire gpt2)
# ─────────────────────────────────────────────────────────────────────────────

RICH = {
    "type": "object",
    "properties": {
        "nom": {"type": "string", "minLength": 1, "maxLength": 6},
        "prénom": {"type": "string"},
        "âge": {"type": "integer"},
        "tags": {"type": "array", "items": {"enum": ["rouge", "vert é", "😀", "—x", 'q"b\\']},
                 "maxItems": 3},
        "pos": {"type": "array", "prefixItems": [{"type": "number"}, {"type": "number"}],
                "items": False, "minItems": 2},
        "kind": {"anyOf": [{"const": "circle"}, {"const": "circus"}, {"type": "null"}]},
        "meta": {"type": "object", "additionalProperties": {"type": "boolean"}},
    },
    "required": ["nom", "âge", "pos"],
    "additionalProperties": False,
}


def _gen_rich(rng):
    inst = {"nom": rng.choice(["Zoé", "a", "😀😀", "q\"\\é", "abcdef", "——"])}
    if rng.random() < 0.5:
        inst["prénom"] = _rand_text(rng)
    inst["âge"] = rng.choice([0, -7, 123456789012])
    if rng.random() < 0.6:
        inst["tags"] = rng.sample(["rouge", "vert é", "😀", "—x", 'q"b\\'], rng.randint(0, 3))
    inst["pos"] = [rng.choice([0, -1.5, 3e-8, 12]), rng.choice([7, 0.25, -2e+30])]
    if rng.random() < 0.5:
        inst["kind"] = rng.choice(["circle", "circus", None])
    if rng.random() < 0.5:
        inst["meta"] = {_rand_text(rng): rng.random() < 0.5 for _ in range(rng.randint(0, 3))}
    return inst


def test_rich_schema_real_tokenizations_always_allowed_gpt2(gpt2):
    enc, tb = gpt2
    c = json_constraint(tb, RICH)
    rng = random.Random(21)
    for _ in range(60):
        inst = _gen_rich(rng)
        for s in (json.dumps(inst), json.dumps(inst, ensure_ascii=False),
                  json.dumps(inst, ensure_ascii=False, separators=(',', ':'))):
            c.reset()
            for t in enc.encode(s):
                assert c.is_allowed(t), (s, c.generated, tb[t])
                c.advance(t)
            assert c.is_complete() and c.is_terminal(), s
            assert validate_instance(json.loads(c.generated), RICH) == []


def test_rich_schema_mask_brute_force_and_soundness_gpt2(gpt2):
    enc, tb = gpt2
    c = json_constraint(tb, RICH)
    m = c.matcher
    rng = random.Random(8)
    struct = _structural_ids(tb)
    checked = 0
    for walk in range(14):
        c.reset()
        steps = rng.randint(0, 30)
        for _ in range(steps):
            mask = c.allowed_mask()
            ids = mask.nonzero().flatten().tolist()
            if not ids:
                break
            pool = [i for i in ids if not _LETTERS.search(tb[i])] if rng.random() < 0.6 else ids
            c.advance(rng.choice(pool or ids))
        mask = c.allowed_mask().tolist()
        if walk % 2 == 0:       # force brute sur tout le vocabulaire (≈ 0.2 s)
            assert mask == [c.is_allowed(t) for t in range(len(tb))], c.generated
        prefix = c.generated
        for t in struct:
            if mask[t]:
                assert viable(prefix + tb[t]), (prefix, tb[t])
        allowed = [t for t in struct + rng.sample(range(len(tb)), 400) if mask[t]]
        for t in rng.sample(allowed, min(25, len(allowed))):
            st = m.advance_bytes(c.state, tb[t])
            comp = m.shortest_completion(st)
            doc = prefix + tb[t] + comp
            assert validate_instance(json.loads(doc.decode('utf-8')), RICH) == [], doc
            checked += 1
    assert checked > 100


# ─────────────────────────────────────────────────────────────────────────────
# Tokens portant des séquences UTF-8 partielles
# ─────────────────────────────────────────────────────────────────────────────

def _tid(tb, b):
    return tb.index(b)


def test_partial_utf8_tokens_free_string_gpt2(gpt2):
    enc, tb = gpt2
    c = json_constraint(tb, {"type": "string"})
    partial = 0
    for text in ('😀😀', '中文é', '—x', 'a😀b'):
        c.reset()
        toks = enc.encode(json.dumps(text, ensure_ascii=False))
        partial += sum(_bad_utf8(tb[t]) for t in toks)      # tokens UTF-8 partiels
        for t in toks:
            prefix = c.generated
            mask = c.allowed_mask().tolist()
            assert mask[t]
            # dans un caractère inachevé : seuls les octets de continuation passent
            ids = [i for i in range(len(tb)) if tb[i] is not None and mask[i]]
            for i in ids[:: max(1, len(ids) // 500)]:
                assert viable(prefix + tb[i]), (prefix, tb[i])
            c.advance(t)
        assert c.is_complete() and json.loads(c.generated) == text
    assert partial >= 4
    # octet de tête puis ASCII / continuation hors plage → refusés
    c.reset()
    c.advance(_tid(tb, b'"'))
    c.advance(_tid(tb, b'\xe2\x80'))
    assert not c.is_allowed(_tid(tb, b'a')) and not c.is_allowed(_tid(tb, b'"'))
    assert c.is_allowed(_tid(tb, b'\x94')) and c.is_allowed(_tid(tb, b'\x80'))
    c.reset()
    c.advance(_tid(tb, b'"'))
    c.advance(_tid(tb, b'\xed'))                                 # ED A0.. = surrogate
    assert not c.is_allowed(_tid(tb, b'\xa0')) and c.is_allowed(_tid(tb, b'\x9f'))


def _bad_utf8(b):
    try:
        b.decode('utf-8')
        return False
    except UnicodeDecodeError:
        return True


def test_partial_utf8_tokens_maxlength_boundary_gpt2(gpt2):
    enc, tb = gpt2
    c = json_constraint(tb, {"type": "string", "maxLength": 1})
    q, emo1, emo2 = _tid(tb, b'"'), _tid(tb, b'\xf0\x9f\x98'), _tid(tb, b'\x80')
    c.advance(q)
    c.advance(emo1)
    ids = c.allowed_mask().nonzero().flatten().tolist()
    assert emo2 in ids and all(0x80 <= tb[i][0] <= 0xBF for i in ids)
    c.advance(emo2)
    # un point de code atteint : seul '"' (exactement, rien après la racine)
    ids = c.allowed_mask().nonzero().flatten().tolist()
    assert ids == [q], [tb[i] for i in ids]
    c.advance(q)
    assert c.is_terminal() and json.loads(c.generated) == '😀'
    # paire de substitution échappée = 1 point de code, découpée par gpt2
    c.reset()
    for t in enc.encode('"\\ud83d\\ude00"'):
        assert c.is_allowed(t), (c.generated, tb[t])
        c.advance(t)
    assert c.is_terminal()
    c.reset()
    toks = enc.encode('"\\ud83d\\ude00x"')
    for t in toks:
        if not c.is_allowed(t):
            break
        c.advance(t)
    assert not c.is_complete() and b'x' not in c.generated


def test_partial_utf8_tokens_constant_strings_gpt2(gpt2):
    enc, tb = gpt2
    schema = {"enum": ["—x", "😀", "中文é"]}
    c = json_constraint(tb, schema)
    for v in schema["enum"]:
        upper_hex = re.sub(r'\\u([0-9a-f]{4})', lambda g: '\\u' + g.group(1).upper(),
                           json.dumps(v))
        assert upper_hex != json.dumps(v) or v == '—x'      # \u2014 : pas de a-f
        for s in (json.dumps(v), json.dumps(v, ensure_ascii=False), upper_hex):
            c.reset()
            for t in enc.encode(s):
                assert c.is_allowed(t), (s, c.generated, tb[t])
                c.advance(t)
            assert c.is_terminal() and json.loads(c.generated) == v
    # préfixe partiel d'un caractère constant puis mauvaise continuation
    c.reset()
    c.advance(_tid(tb, b'"'))
    c.advance(_tid(tb, b'\xe2\x80'))
    assert c.is_allowed(_tid(tb, b'\x94'))                 # '—' U+2014
    assert not c.is_allowed(_tid(tb, b'\x93'))             # '–' U+2013 : hors enum
    mask = c.allowed_mask()
    assert [tb[i] for i in mask.nonzero().flatten().tolist()] and all(
        tb[i][:1] == b'\x94' for i in mask.nonzero().flatten().tolist())


# ─────────────────────────────────────────────────────────────────────────────
# Caches : masques périmés, clones, tenseur renvoyé, éviction
# ─────────────────────────────────────────────────────────────────────────────

def test_no_stale_mask_after_advance_and_clone_independence(gpt2):
    enc, tb = gpt2
    c = json_constraint(tb, RICH)
    toks = enc.encode('{"nom":"Zoé","âge":12,"pos":[1,2.5]}')
    masks = []
    for t in toks:
        masks.append((c.state, c.allowed_mask().clone()))
        c.advance(t)
    # chaque masque mis en cache correspond toujours à son état
    probe = c.clone()
    for st, mk in masks:
        probe.state = st
        assert torch.equal(probe.allowed_mask(), mk)
    # clone : avance indépendante, cache partagé mais indexé par état
    c.reset()
    for t in toks[:5]:
        c.advance(t)
    before_state, before_gen = c.state, c.generated
    before = c.allowed_mask().clone()
    d = c.clone()
    nxt = toks[5]
    d.advance(nxt)
    assert c.state == before_state and c.generated == before_gen
    assert torch.equal(c.allowed_mask(), before)
    assert d.generated == before_gen + tb[nxt]
    assert d.allowed_mask().tolist() == [d.is_allowed(i) for i in range(len(tb))]
    d.reset()
    assert c.generated == before_gen and torch.equal(c.allowed_mask(), before)
    # muter le tenseur renvoyé ne corrompt pas le cache
    m1 = c.allowed_mask()
    m1.fill_(False)
    assert torch.equal(c.allowed_mask(), before)
    # apply_to_logits : lot (B, V), autres dtypes
    logits = torch.randn(3, len(tb), dtype=torch.float16)
    out = c.apply_to_logits(logits)
    assert torch.isinf(out[:, ~before]).all() and torch.equal(out[:, before], logits[:, before])


def test_forced_cache_eviction_stays_exact(gpt2):
    enc, tb = gpt2
    ref = json_constraint(tb, RICH)
    c = json_constraint(tb, RICH)
    c.matcher._max_states = 5           # vide _rows / _interned en plein DFS
    c._shared.max_masks = 2
    rng = random.Random(4)
    for _ in range(10):
        c.reset()
        for _ in range(rng.randint(0, 12)):
            m1 = c.allowed_mask()
            ref.state = c.state
            assert torch.equal(m1, ref.allowed_mask()), c.generated
            assert c.is_complete() == ref.is_complete()
            assert c.is_terminal() == ref.is_terminal()
            ids = m1.nonzero().flatten().tolist()
            if not ids:
                break
            c.advance(rng.choice(ids))
        ref.state = c.state
        assert c.completion_tokens() == ref.completion_tokens()
        assert len(c._shared.masks) <= 2


# ─────────────────────────────────────────────────────────────────────────────
# completion_tokens
# ─────────────────────────────────────────────────────────────────────────────

def test_completion_tokens_gpt2_random_states(gpt2):
    enc, tb = gpt2
    rng = random.Random(17)
    for schema in (None, RICH, {"type": "array", "items": {"type": "string", "minLength": 40},
                                "minItems": 3}):
        c = json_constraint(tb, schema)
        m = c.matcher
        for _ in range(25):
            c.reset()
            for _ in range(rng.randint(0, 20)):
                ids = c.allowed_mask().nonzero().flatten().tolist()
                if not ids:
                    break
                pool = [i for i in ids if not _LETTERS.search(tb[i])] if rng.random() < .6 else ids
                c.advance(rng.choice(pool or ids))
            st, gen = c.state, c.generated
            toks = c.completion_tokens()
            assert c.state == st and c.generated == gen          # pas d'effet de bord
            comp = m.shortest_completion(st)
            assert toks is not None and b''.join(tb[t] for t in toks) == comp
            assert len(toks) <= max(1, len(comp))
            for t in toks:
                assert c.allowed_mask()[t] and c.is_allowed(t)
                c.advance(t)
            assert c.is_complete()
            if schema is not None:
                assert validate_instance(json.loads(c.generated), schema) == []
            else:
                json.loads(c.generated)


def test_completion_tokens_char_vocab_missing_digit_zero():
    """
    Vocabulaire char-level (data_prep : sorted(set(texte))) sans '0' : la plus courte
    complétion d'un entier est b'0', intokenisable, mais '1'…'9' existent. Le contrat
    (« None si impossible avec ce vocabulaire ») est violé : completion_tokens() → None
    alors que '{"n":1}' est atteignable — generate_structured(force_complete) ne peut
    plus fermer le JSON (mesuré : budget 10 → b'  {"n"\\n  \\n', incomplet, alors que
    ':1}' tenait dans le budget).
    """
    chars = sorted(set('{}[]":, 123456789-.abcdefghijklmnopqrstuvwxyz'))
    tb = token_bytes_from_itos(dict(enumerate(chars)), len(chars))
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    c = json_constraint(tb, schema)
    probe = c.clone()
    for ch in '{"n":7}':
        probe.advance(chars.index(ch))
    assert probe.is_complete()                       # une complétion existe bien
    toks = c.completion_tokens()
    assert toks is not None
    for t in toks:
        c.advance(t)
    assert c.is_complete()


def test_completion_tokens_long_completion_single_char_vocab():
    """
    max_expansions (10 000) compte CHAQUE token posé, même sans aucun retour arrière :
    avec un vocabulaire char-level, une complétion de plus de 10 000 octets (chaîne
    minLength 9 999, tableau minItems 5 001…) renvoie None alors qu'elle se tokenise
    trivialement caractère par caractère.
    """
    chars, tb = _char_vocab()
    for schema in ({"type": "string", "minLength": 9999},
                   {"type": "array", "items": {"type": "integer"}, "minItems": 5001}):
        c = json_constraint(tb, schema)
        comp = c.matcher.shortest_completion(c.state)
        assert comp is not None and len(comp) > 10_000
        toks = c.completion_tokens()
        assert toks is not None, (schema, len(comp))
        assert b''.join(tb[t] for t in toks) == comp


# ─────────────────────────────────────────────────────────────────────────────
# Mémoire
# ─────────────────────────────────────────────────────────────────────────────

def _rss():
    try:
        with open('/proc/self/statm') as f:
            return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError):     # pragma: no cover — hors Linux
        return 0


def test_long_random_generation_caches_bounded_gpt2(gpt2):
    """10 000 tokens aléatoires (JSON libre) : caches plafonnés, mémoire stable."""
    enc, tb = gpt2
    c = json_constraint(tb, None)
    m, sh = c.matcher, c._shared
    rng = random.Random(1)
    base = _rss()
    n = resets = 0
    t0 = time.perf_counter()
    while n < 10_000:
        if c.is_terminal():
            c.reset()
            resets += 1
        ids = c.allowed_mask().nonzero().flatten()
        assert len(ids)
        c.advance(int(ids[rng.randrange(len(ids))]))
        n += 1
    elapsed = time.perf_counter() - t0
    assert resets > 10
    assert len(sh.masks) <= sh.max_masks and len(m._rows) <= m._max_states
    assert len(m._allowed_cache) <= m._max_states
    assert len(m._completion_cache) <= m._max_states
    assert _rss() - base < 300e6
    assert elapsed < 60, elapsed


def test_long_tracked_string_generation_caches_bounded_char_vocab():
    """Chaîne suivie (maxLength) : chaque token crée un état neuf → caches plafonnés."""
    chars, tb = _char_vocab()
    c = json_constraint(tb, {"type": "object",
                             "properties": {"t": {"type": "string", "maxLength": 100_000}},
                             "required": ["t"]})
    m, sh = c.matcher, c._shared
    m._max_states = 2000
    sh.max_masks = 64
    for ch in '{"t":"':
        c.advance(chars.index(ch))
    rng = random.Random(2)
    for _ in range(10_000):
        ids = c.allowed_mask().nonzero().flatten().tolist()
        ids = [i for i in ids if tb[i] not in (b'"', b'\\')]
        c.advance(rng.choice(ids))
        assert len(sh.masks) <= 64 and len(m._rows) <= 2000
        assert len(m._interned) <= 2000 * 4 and len(m._accept_cache) <= 2000 * 4
    toks = c.completion_tokens()
    for t in toks:
        c.advance(t)
    assert c.is_terminal() and len(json.loads(c.generated)["t"]) == 10_000


_DEEP_SCRIPT = textwrap.dedent('''
    import gc, os, resource, sys, time
    sys.path.insert(0, {root!r})
    import structured as S, tiktoken
    enc = tiktoken.get_encoding("gpt2")
    tb = S.token_bytes_from_tiktoken(enc)
    c = S.json_constraint(tb, None)
    c.allowed_mask()
    gc.collect()
    with open("/proc/self/statm") as f:      # RSS courant (pas le pic de l'initialisation)
        base = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    t0 = time.perf_counter()
    lb = tb.index(b"[")
    for k in range({n}):            # petit modèle dégénéré qui boucle sur '['
        assert c.is_allowed(lb)
        c.advance(lb)
        c.is_terminal()
        if {every} and k % {every} == 0:
            c.allowed_mask()         # rééchantillonnage par rejet de generate_structured
    toks = c.completion_tokens()     # force_complete de generate_structured
    assert toks is not None
    for t in toks:
        c.advance(t)
    assert c.is_complete()
    # VmHWM (pic RSS de CE processus) : ru_maxrss hérite du pic du pytest parent après fork
    with open("/proc/self/status") as f:
        peak = next(int(l.split()[1]) * 1024 for l in f if l.startswith("VmHWM:"))
    print(peak - base, time.perf_counter() - t0)
''')


@pytest.mark.parametrize("n,every", [(10_000, 0), (2_000, 1)])
def test_deep_nesting_generation_memory_bounded(n, every):
    """
    Une configuration est une pile COMPLÈTE (tuple de frames) recopiée à chaque octet :
    un état à la profondeur d coûte O(d) octets, et les caches (_rows, _interned,
    _accept_cache, _completion_cache — plafonnés en NOMBRE d'états, pas en octets) les
    gardent tous. Mesuré : 10 000 tokens '[' + fermeture ≈ 470 Mo ; 2 000 tokens '['
    avec un masque par pas (rééchantillonnage) ≈ 590 Mo. Les plafonds affichés
    (50 000 états, 1 024 masques ≈ 150 Mo avec gpt2) ne bornent donc pas la mémoire.
    """
    if tiktoken is None:
        pytest.skip("tiktoken indisponible")
    script = _DEEP_SCRIPT.format(root=ROOT, n=n, every=every)
    out = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    grown, elapsed = map(float, out.stdout.split())
    assert grown < 256e6, f"croissance mémoire {grown / 1e6:.0f} Mo ({n} tokens '[', masque/{every})"


# ─────────────────────────────────────────────────────────────────────────────
# Chronométrage
# ─────────────────────────────────────────────────────────────────────────────


def _enum_words(enc, n):
    words = sorted({enc.decode([i]).strip() for i in range(256, 30000)} - {''})
    words = [w for w in words if '"' not in w and '\\' not in w
             and all(ord(ch) >= 32 for ch in w)]
    return random.Random(0).sample(words, n)


def test_timing_trie_build_gpt2(gpt2):
    enc, tb = gpt2
    fresh = list(tb)                       # liste neuve → pas de trie en cache
    t0 = time.perf_counter()
    TokenConstraint(JSONSchemaMatcher(None), fresh)
    assert time.perf_counter() - t0 < 3.0
    t0 = time.perf_counter()
    TokenConstraint(JSONSchemaMatcher(None), fresh)          # trie réutilisé
    assert time.perf_counter() - t0 < 0.5


@pytest.mark.parametrize("case", ["free", "maxlen", "minlen", "escape", "utf8", "enum2000",
                                  "keys500", "deep_any"])
def test_timing_masks_and_is_allowed_gpt2(gpt2, case):
    enc, tb = gpt2
    schema, prefix = {
        "free": ({"type": "string"}, b'"Bonj'),
        "maxlen": ({"type": "object", "properties": {"t": {"type": "string", "maxLength": 5000}}},
                   b'{"t":"Bonj'),
        "minlen": ({"type": "string", "minLength": 5000}, b'"Bonj'),
        "escape": ({"type": "string", "maxLength": 50}, b'"ab\\ud83d\\'),
        "utf8": ({"type": "string", "maxLength": 50}, b'"ab\xf0\x9f'),
        "enum2000": ({"enum": _enum_words(enc, 2000)}, b'"'),
        "keys500": ({"type": "object", "properties": {f"k{w}": {"type": "integer"}
                                                       for w in _enum_words(enc, 500)}},
                    b'{"'),
        "deep_any": (None, b'[{"a":' * 300 + b'"x'),
    }[case]
    m = JSONSchemaMatcher(schema)
    c = TokenConstraint(m, tb)
    c.state = m.advance_bytes(m.initial_state, prefix)
    assert c.state is not None

    # is_allowed sur une contrainte neuve (aucun masque précalculé)
    rng = random.Random(0)
    ids = [rng.randrange(len(tb)) for _ in range(5000)]
    t0 = time.perf_counter()
    res = [c.is_allowed(i) for i in ids]
    t_allowed = (time.perf_counter() - t0) / len(ids)

    t0 = time.perf_counter()
    mask = c.allowed_mask()
    t_fresh = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(20):
        c.allowed_mask()
    t_cached = (time.perf_counter() - t0) / 20

    assert [bool(mask[i]) for i in ids] == res
    assert t_fresh < 2.0, t_fresh
    assert t_cached < 0.005, t_cached
    assert t_allowed < 0.0002, t_allowed


@pytest.mark.parametrize("case", ["enum8000", "props5000"])
def test_timing_constraint_build_large_schema_gpt2(gpt2, case):
    """
    Construction d'une contrainte (/json, gen --schema) pour un grand schéma, trie gpt2
    déjà en cache : la compilation est QUADRATIQUE —
      · _compile_enum déduplique par `any(_json_eq(v, w) for w in kept)` → n²/2 appels
        (8 000 valeurs ≈ 32 M appels ≈ 10 s) ;
      · _obj_tables construit cand[p] / AC[p] en O(P²) pour P propriétés optionnelles,
        et _finalize le rappelle à chaque tour du point fixe (5 000 propriétés ≈ 11 s).
    """
    enc, tb = gpt2
    if case == "enum8000":
        schema = {"enum": [f"ville_{i:05d}" for i in range(8000)]}
        probe, ok = b'"ville_04217"', True
    else:
        schema = {"type": "object", "properties": {f"k{i}": {"type": "integer"}
                                                   for i in range(5000)}}
        probe, ok = b'{"k17":1,"k4999":2}', True
    t0 = time.perf_counter()
    c = json_constraint(tb, schema)
    elapsed = time.perf_counter() - t0
    m = c.matcher
    st = m.advance_bytes(m.initial_state, probe)
    assert m.is_accepting(st) == ok
    assert elapsed < 3.0, f"{case} : construction {elapsed:.1f} s"


# ═════════════════════════════════════════════════════════════════════════════
# Tour 2
# ═════════════════════════════════════════════════════════════════════════════

R2_SCHEMAS = [
    None,
    {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 4},
    {"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string", "maxLength": 3},
                                      {"type": "boolean"}], "items": False},
    {"type": "object", "properties": {
        "a": {"type": "integer"},
        "b": {"type": ["string", "null"], "minLength": 2, "maxLength": 4},
        "c": {"type": "array", "items": {"type": "object", "properties": {"x": {"type": "number"}},
                                         "required": ["x"]}}},
     "required": ["a", "c"]},
    {"anyOf": [{"type": "integer"}, {"type": "string", "maxLength": 2},
               {"type": "array", "items": {"type": "integer"}, "maxItems": 2}]},
    {"type": "object", "additionalProperties": {"type": "array",
                                                "items": {"type": "string", "minLength": 1}}},
    {"enum": [1, 12, 1.5, "a", "ab", [1, 2], {"k": "v"}, None, True]},
    {"type": "string", "minLength": 2, "maxLength": 3},
    {"$defs": {"n": {"type": "object", "properties": {
        "v": {"type": "integer"},
        "kids": {"type": "array", "items": {"$ref": "#/$defs/n"}, "maxItems": 2}},
        "required": ["v"]}}, "$ref": "#/$defs/n"},
    {"type": "integer"},
    {"type": "number"},
    {"type": "object", "properties": {"é": {"const": "😀"}, "\"q": {"type": "string", "maxLength": 1}},
     "required": ["é"]},
]


def _r2_walk_and_check(schema, tbv, rng, walks, steps, brute_every):
    """
    Marches aléatoires (tokens surtout structurels) ; à chaque arrêt :
      - masque == force brute calculée par un matcher NEUF (caches indépendants) ;
      - vivacité : masque non vide OU JSON complet ; is_terminal ⇔ complet et masque vide ;
      - completion_tokens : tokens autorisés un à un, instance finale valide.
    """
    c = json_constraint(tbv, schema)
    checked = 0
    for w in range(walks):
        c.reset()
        for _ in range(rng.randint(0, steps)):
            ids = c.allowed_mask().nonzero().flatten().tolist()
            if not ids:
                assert c.is_complete() and c.is_terminal(), (schema, c.generated)
                break
            assert not c.is_terminal(), (schema, c.generated)
            pool = ([i for i in ids if not _LETTERS.search(tbv[i])]
                    if rng.random() < 0.75 else ids)
            c.advance(rng.choice(pool or ids))
        mask = c.allowed_mask().tolist()
        if w % brute_every == 0:
            fresh = JSONSchemaMatcher(schema)
            st = fresh.advance_bytes(fresh.initial_state, c.generated)
            assert st is not None
            bad = [(i, tbv[i]) for i in range(len(tbv))
                   if (tbv[i] is not None and fresh.advance_bytes(st, tbv[i]) is not None)
                   != mask[i]]
            assert not bad, (schema, c.generated, bad[:5])
            checked += 1
        assert any(mask) or c.is_complete(), (schema, c.generated)
        assert c.is_terminal() == (c.is_complete() and not any(mask)), (schema, c.generated)
        toks = c.completion_tokens()
        assert toks is not None, (schema, c.generated)
        d = c.clone()
        for t in toks:
            assert d.is_allowed(t)
            d.advance(t)
        assert d.is_complete(), (schema, c.generated, toks)
        inst = json.loads(d.generated)
        if schema is not None:
            assert validate_instance(inst, schema) == [], (schema, d.generated)
    return checked


@pytest.mark.parametrize("si", range(len(R2_SCHEMAS)))
def test_r2_mask_vs_fresh_matcher_and_liveness(gpt2, si):
    enc, tb = gpt2
    schema = R2_SCHEMAS[si]
    rng = random.Random(100 + si)
    assert _r2_walk_and_check(schema, tb, rng, walks=10, steps=30, brute_every=5) == 2
    chars, ctb = _char_vocab()
    assert _r2_walk_and_check(schema, ctb, rng, walks=25, steps=60, brute_every=1) == 25


def _utf8_prefix_ok(tail: bytes) -> bool:
    """`tail` (1-3 octets) est-il le début strict d'un caractère UTF-8 valide ?"""
    for cp in list(range(0x80, 0xD800)) + list(range(0xE000, 0x110000, 7)) + [0x10FFFF]:
        e = chr(cp).encode('utf-8')
        if len(e) > len(tail) and e.startswith(tail):
            return True
    return False


def _string_body_viable(body: bytes, mn: int, mx):
    """
    Oracle indépendant (sans échappement) : `body` = octets après le '"' ouvrant d'une
    chaîne racine {minLength mn, maxLength mx}. None si hors du domaine de l'oracle.
    """
    if b'\\' in body:
        return None
    q = body.find(b'"')
    content, rest = (body, b'') if q < 0 else (body[:q], body[q + 1:])
    if q >= 0 and rest:
        return False                                    # rien après la racine
    pend = 0
    try:
        s = content.decode('utf-8')
    except UnicodeDecodeError:
        if q >= 0:
            return False
        for k in (1, 2, 3):
            try:
                s = content[:-k].decode('utf-8')
            except UnicodeDecodeError:
                continue
            if not _utf8_prefix_ok(content[-k:]):
                return False
            pend = 1
            break
        else:
            return False
    if any(ord(ch) < 0x20 for ch in s):
        return False
    n = len(s) + pend
    if mx is not None and n > mx:
        return False
    return n >= mn if q >= 0 else True


def test_r2_utf8_delimiter_crossing_tokens_vs_codepoint_oracle_gpt2(gpt2):
    """
    Tokens gpt2 mêlant un caractère non ASCII et un délimiteur JSON ('…"', '"—', '—"',
    '…]', '®,', '™:'…) + un échantillon de tokens UTF-8 partiels, aux bornes
    minLength / maxLength d'une chaîne racine : masque et is_allowed == oracle.
    """
    enc, tb = gpt2
    mix = [i for i, t in enumerate(tb) if t and any(b >= 0x80 for b in t)
           and any(ch in t for ch in b'"{}[],:')]
    assert len(mix) >= 10
    partial = [i for i, t in enumerate(tb) if t and any(b >= 0x80 for b in t)][::9]
    probe = sorted(set(mix + partial))
    n = 0
    for mn, mx in [(0, None), (0, 0), (0, 1), (1, 1), (2, 2), (0, 2), (1, 3), (3, None)]:
        schema = {"type": "string", "minLength": mn}
        if mx is not None:
            schema["maxLength"] = mx
        c = json_constraint(tb, schema)
        m = c.matcher
        for pre in (b'"', b'"a', b'"ab', b'"\xe2\x80', b'"a\xe2\x80', b'"\xe2', b'"\xc2'):
            st = m.advance_bytes(m.initial_state, pre)
            if st is None:
                continue
            c.state = st
            mask = c.allowed_mask().tolist()
            for t in probe:
                o = _string_body_viable(pre[1:] + tb[t], mn, mx)
                if o is None:
                    continue
                n += 1
                assert mask[t] == o == c.is_allowed(t), (schema, pre, tb[t], o, mask[t])
    assert n > 3000


def test_r2_duplicate_bytes_and_bytearray_vocab():
    """Plusieurs ids pour les mêmes octets (trie.multi) et éléments bytearray."""
    chars = list('{}[]":,0123 abc') + ['{', '"a', '"a', ']', 'é', 'é', '\x00', '"}']
    tb = token_bytes_from_itos(dict(enumerate(chars)), len(chars) + 5)
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    for vocab in (tb, [bytearray(t) if t else None for t in tb]):
        c = json_constraint(vocab, schema)
        for step in (None, '{', '"a', '"', ':', '"', 'é'):
            if step is not None:
                c.advance(chars.index(step))
            mask = c.allowed_mask().tolist()
            assert mask == [c.is_allowed(i) for i in range(len(vocab))], c.generated
            d = c.clone()
            for t in d.completion_tokens():
                d.advance(t)
            assert d.is_terminal() and json.loads(d.generated)["a"] in ("", "é")
        dup = [i for i, ch in enumerate(chars) if ch == 'é']
        assert all(mask[i] for i in dup)                 # les deux ids de 'é'
        assert not any(mask[len(chars):])                # padding


def test_r2_mask_tensor_outlives_constraint_and_cache(gpt2):
    import gc
    enc, tb = gpt2
    c = json_constraint(tb, None)
    c.advance(tb.index(b'{'))
    mk = c.allowed_mask()
    ref = mk.tolist()
    c._shared.masks.clear()
    del c
    gc.collect()
    junk = [bytearray(b'\x01' * len(tb)) for _ in range(100)]
    assert mk.tolist() == ref and 0 < int(mk.sum()) < 200
    del junk


@pytest.mark.parametrize("case", ["oneOf40", "arrUnion39", "props5000", "enum5000"])
def test_r2_timing_many_configurations_gpt2(gpt2, case):
    """
    États à NOMBREUSES configurations : chaîne suivie partagée par 40 variantes oneOf,
    union de 39 tableaux de chaînes bornées, 5 000 clés / valeurs d'enum en concurrence.
    """
    enc, tb = gpt2
    variants = {"oneOf": [{"type": "object", "properties": {
        "title": {"type": "string", "maxLength": 200}, f"extra{i}": {"type": "integer"}},
        "required": ["title", f"extra{i}"]} for i in range(40)]}
    schema, prefix = {
        "oneOf40": (variants, b'{"title":"Bonj'),
        "arrUnion39": ({"anyOf": [{"type": "array", "items": {"type": "string", "maxLength": k}}
                                  for k in range(1, 40)]}, b'["'),
        "props5000": ({"type": "object", "properties": {f"k{i}": {"type": "integer"}
                                                         for i in range(5000)}}, b'{"'),
        "enum5000": ({"enum": [f"ville_{i:05d}" for i in range(5000)]}, b'"ville_0'),
    }[case]
    m = JSONSchemaMatcher(schema)
    c = TokenConstraint(m, tb)
    c.state = m.advance_bytes(m.initial_state, prefix)
    # enum5000 est compilé en UNE configuration (arbre des valeurs) depuis le tour 2 :
    # la précondition « nombreuses configurations » ne vaut que pour les autres cas
    assert c.state is not None and (case == "enum5000" or len(c.state) >= 39)
    rng = random.Random(1)
    ids = [rng.randrange(len(tb)) for _ in range(3000)]
    t0 = time.perf_counter()
    res = [c.is_allowed(i) for i in ids]
    t_allowed = (time.perf_counter() - t0) / len(ids)
    t0 = time.perf_counter()
    mask = c.allowed_mask()
    t_fresh = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(20):
        c.allowed_mask()
    t_cached = (time.perf_counter() - t0) / 20
    assert [bool(mask[i]) for i in ids] == res
    assert t_fresh < 2.0, t_fresh
    assert t_cached < 0.005, t_cached
    assert t_allowed < 0.0002, t_allowed


# ── completion_tokens : nombre de TOKENS de la fermeture ─────────────────────

@pytest.mark.parametrize("n", [200, 2000])
def test_r2_completion_tokens_token_count_vs_valid_closing_gpt2(gpt2, n):
    """
    completion_tokens() tokenise la plus courte complétion en OCTETS, remplie de 'a'
    pour minLength. Avec gpt2, la plus longue suite de 'a' en un token fait 4 octets,
    alors que '=' * 64, '-' * 64, '_' * 64 ou '.' * 64 sont des tokens : la fermeture
    renvoyée coûte jusqu'à 15× plus de tokens qu'une fermeture valide évidente
    (mesuré : minLength 2000 → 502 tokens contre 34 ; minLength 200 → ~57 contre ~11).

    Or le budget de generate_structured est en TOKENS : « sortie toujours complète si
    max_new_tokens >= len(completion_tokens()) », force_complete bannit tout token dont
    la fermeture ne tient pas, et `popixa gen --schema` refuse de tourner avec
    « --tokens N insuffisant : il faut au moins len(completion_tokens()) tokens ».
    Avec --tokens 100 et minLength 200, le JSON est déclaré impossible alors qu'une
    sortie complète tient en ~11 tokens.
    """
    enc, tb = gpt2
    schema = {"type": "object", "properties": {"résumé": {"type": "string", "minLength": n}},
              "required": ["résumé"]}
    c = json_constraint(tb, schema)
    alt = enc.encode('{"résumé":"' + '=' * n + '"}')
    probe = c.clone()
    for t in alt:
        assert probe.is_allowed(t), (probe.generated, tb[t])
        probe.advance(t)
    assert probe.is_terminal()
    assert validate_instance(json.loads(probe.generated), schema) == []
    toks = c.completion_tokens()
    assert toks is not None
    assert len(toks) <= 2 * len(alt), (
        f"fermeture de {len(toks)} tokens alors qu'une fermeture valide en "
        f"{len(alt)} tokens existe")
    # même constat en cours de génération (le modèle a écrit quelques mots)
    for t in enc.encode('{"résumé":"Bonjour à tous'):
        c.advance(t)
    toks = c.completion_tokens()
    alt = enc.encode('=' * (n - len('Bonjour à tous')) + '"}')
    assert len(toks) <= 2 * len(alt), (len(toks), len(alt))


def test_r2_completion_tokens_greedy_segmentation_is_token_optimal_gpt2(gpt2):
    """
    Pour les MÊMES octets (plus courte complétion), la segmentation « plus long préfixe
    d'abord » a autant de tokens que l'optimum (programmation dynamique) sur des états
    aléatoires de schémas variés (régression).
    """
    enc, tb = gpt2
    tokset = {}
    for i, t in enumerate(tb):
        if t is not None:
            tokset.setdefault(t, i)
    maxlen = max(len(t) for t in tokset)

    def dp(data):
        best = [0] + [None] * len(data)
        for i in range(len(data)):
            if best[i] is None:
                continue
            for L in range(1, min(maxlen, len(data) - i) + 1):
                if data[i:i + L] in tokset and (best[i + L] is None or best[i] + 1 < best[i + L]):
                    best[i + L] = best[i] + 1
        return best[-1]

    rng = random.Random(5)
    schemas = [None, RICH, R2_SCHEMAS[3], R2_SCHEMAS[8],
               {"type": "array", "items": {"type": "array", "items": {"type": "integer"},
                                           "minItems": 1}, "minItems": 3}]
    n = 0
    for schema in schemas:
        c = json_constraint(tb, schema)
        for _ in range(25):
            c.reset()
            for _ in range(rng.randint(0, 25)):
                ids = c.allowed_mask().nonzero().flatten().tolist()
                if not ids:
                    break
                pool = [i for i in ids if not _LETTERS.search(tb[i])] if rng.random() < .7 else ids
                c.advance(rng.choice(pool or ids))
            comp = c.matcher.shortest_completion(c.state)
            toks = c.completion_tokens()
            assert b''.join(tb[t] for t in toks) == comp
            assert len(toks) == dp(comp), (c.generated, comp)
            n += 1
    assert n == 125


# ── generate_structured (model.py) : coût de la fermeture à chaque pas ──────

_GEN_SCALING_SCRIPT = textwrap.dedent('''
    import gc, sys, time, tracemalloc
    sys.path.insert(0, {scratch!r})
    sys.path.append({repo!r})
    import structured as S, torch
    assert S.__file__.startswith({scratch!r})
    from model import nanoPOPIXA, POPIXAConfig
    torch.manual_seed(0)
    torch.set_num_threads(1)
    chars = sorted(set(chr(i) for i in range(32, 127)))
    tb = S.token_bytes_from_itos(dict(enumerate(chars)), len(chars))
    LB = chars.index("[")
    m = nanoPOPIXA(POPIXAConfig(vocab_size=len(chars), block_size=32, n_layer=1, n_head=1,
                                n_embd=8, dropout=0.0)).eval()
    orig = m._apply_sampling

    def degenerate(logits, temperature, top_k, top_p, rp, idx, logit_bias=None, mask=None):
        # petit modèle char-level dégénéré qui boucle sur '[' tant que c'est permis
        if mask is None or bool(mask.reshape(-1)[LB]):
            p = torch.zeros(1, len(chars))
            p[0, LB] = 1.0
            return p
        return orig(logits, temperature, top_k, top_p, rp, idx, logit_bias=logit_bias, mask=mask)

    m._apply_sampling = degenerate
    for n in {sizes}:
        c = S.json_constraint(tb, None)
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        toks = list(m.generate_structured(torch.zeros(1, 1, dtype=torch.long), c,
                                          max_new_tokens=n))
        dt = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert c.is_complete() and len(toks) == n
        print(n, dt, peak, flush=True)
''')


def test_r2_generate_structured_closing_cost_scales_linearly():
    """
    generate_structured (force_complete) appelle completion_tokens() pour CHAQUE token
    candidat et garde toutes les listes dans `closing_cache` jusqu'à la fin : à la
    profondeur d, la fermeture fait d tokens, donc Θ(n²) en temps et en mémoire pour un
    modèle qui imbrique (mesuré, vocabulaire char-level, modèle minuscule :
    1 000 tokens → 1,5 s / 2,2 Mo ; 3 000 → 15,9 s / 13,7 Mo ; 4 000 → 25,7 s / 22,8 Mo ;
    10 000 → 197,6 s / 119,6 Mo). Profil à 2 400 tokens : 79 % du temps dans
    closing_for → completion_tokens, 720 601 ids de tokens retenus dans closing_cache
    (Σ des longueurs de fermeture). Tripler le budget devrait tripler le coût, pas le
    décupler.
    """
    script = _GEN_SCALING_SCRIPT.format(scratch=ROOT, repo=ROOT,
                                        sizes=(800, 2400))
    out = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                         timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    rows = [tuple(map(float, line.split())) for line in out.stdout.splitlines()
            if line and line[0].isdigit()]
    (n1, t1, p1), (n2, t2, p2) = rows
    mem_ratio, time_ratio = p2 / p1, t2 / t1
    assert mem_ratio < 4.5 and time_ratio < 6, (
        f"×{n2 / n1:.0f} tokens → mémoire ×{mem_ratio:.1f} ({p1 / 1e6:.1f} → {p2 / 1e6:.1f} Mo), "
        f"temps ×{time_ratio:.1f} ({t1:.1f} → {t2:.1f} s)")


# ═════════════════════════════════════════════════════════════════════════════
# Tour 3
# ═════════════════════════════════════════════════════════════════════════════

# ── Unions récursives ambiguës : explosion du nombre de configurations ───────

R3_THREAD = {"$defs": {"c": {"anyOf": [
    {"type": "object", "properties": {"replies": {"type": "array", "items": {"$ref": "#/$defs/c"}},
                                      "text": {"type": "string"}},
     "required": ["replies", "text"]},
    {"type": "object", "properties": {"replies": {"type": "array", "items": {"$ref": "#/$defs/c"}},
                                      "image": {"type": "string"}},
     "required": ["replies", "image"]}]}},
    "$ref": "#/$defs/c"}

R3_TREE = {"$defs": {"node": {"oneOf": [
    {"type": "object", "properties": {"children": {"type": "array",
                                                   "items": {"$ref": "#/$defs/node"}},
                                      "leaf": {"type": "boolean"}}},
    {"type": "object", "properties": {"children": {"type": "array",
                                                   "items": {"$ref": "#/$defs/node"}},
                                      "name": {"type": "string"}}}]}},
    "$ref": "#/$defs/node"}


@pytest.mark.parametrize("case", ["thread_anyOf", "tree_oneOf"])
def test_r3_recursive_union_configuration_explosion_gpt2(gpt2, case):
    """
    Schéma récursif dont les DEUX alternatives d'une union commencent par la même
    propriété récursive (fil de commentaires : {replies, text} | {replies, image} ;
    arbre : {children, leaf?} | {children, name?}). Tant que le modèle imbrique, aucune
    alternative n'est éliminée à aucun niveau : chaque niveau DOUBLE le nombre de
    configurations (piles distinctes) de l'état → 2^profondeur.

    Mesuré (gpt2, 4 tokens par niveau : '{"' 'repl' 'ies' '":[') :
      profondeur 10 → 1 024 configurations ; 14 → 16 384 (56 tokens seulement) :
      is_allowed d'un token sur l'état neuf ≈ 90 ms (cible 0,2 ms), masque neuf ≈ 5,4 s
      (cible 2 s), RSS +210 Mo ; profondeur 17 → > 9 s par niveau ; 20 → ~1 M
      configurations, ≈ 8 s PAR OCTET (advance_bytes). Un petit modèle qui boucle sur
      l'imbrication bloque generate_structured (temps et mémoire non bornés).
    """
    enc, tb = gpt2
    schema, unit = {"thread_anyOf": (R3_THREAD, '{"replies":['),
                    "tree_oneOf": (R3_TREE, '{"children":[')}[case]
    c = json_constraint(tb, schema)
    toks = enc.encode(unit)
    base = _rss()
    worst = 0.0
    for _ in range(14):
        for t in toks:
            t0 = time.perf_counter()
            ok = c.is_allowed(t)            # rejet de generate_structured : état NEUF
            worst = max(worst, time.perf_counter() - t0)
            assert ok, (c.generated, tb[t])
            c.advance(t)
    t0 = time.perf_counter()
    mask = c.allowed_mask()
    t_fresh = time.perf_counter() - t0
    grown = _rss() - base
    allowed = mask.nonzero().flatten().tolist()
    assert allowed and all(viable(c.generated + tb[i]) for i in allowed)
    closing = c.completion_tokens()
    assert closing is not None
    d = c.clone()
    for t in closing:
        d.advance(t)
    assert d.is_terminal()
    assert validate_instance(json.loads(d.generated), schema) == []
    assert t_fresh < 2.0 and worst < 0.01 and grown < 100e6, (
        f"profondeur 14 : {len(c.state)} configurations, masque neuf {t_fresh:.1f} s, "
        f"is_allowed {worst * 1000:.0f} ms, RSS +{grown / 1e6:.0f} Mo")


# ── Grand enum à préfixe commun : coût par token linéaire en nb de valeurs ───

def test_r3_large_enum_shared_prefix_per_token_cost_gpt2(gpt2):
    """
    Enum de 8 000 chaînes à long préfixe commun (identifiants, URL…) : chaque valeur
    est une configuration _F_CSTR séparée, avancée octet par octet → chaque octet du
    préfixe commun reconstruit un frozenset de 8 000 piles. Générer 5 valeurs (95
    tokens gpt2, un is_allowed + advance par pas comme le rejet de
    generate_structured) : mesuré 7,3 s, jusqu'à 880 ms pour UN token, contre 0,05 s
    pour les mêmes octets avec un enum de 200 valeurs (×40 valeurs → ×140 temps).
    is_allowed moyen sur l'état '"https://api.example.com/v1/resources/' : 0,36 ms
    (cible 0,2 ms).
    """
    enc, tb = gpt2
    vals = [f"https://api.example.com/v1/resources/{i:05d}/details" for i in range(8000)]
    schema = {"type": "array", "items": {"enum": vals}, "maxItems": 5}
    c = json_constraint(tb, schema)
    doc = json.dumps(vals[17:22])
    per = []
    t_all = time.perf_counter()
    for t in enc.encode(doc):
        t0 = time.perf_counter()
        assert c.is_allowed(t), (c.generated, tb[t])
        c.advance(t)
        per.append(time.perf_counter() - t0)
    total = time.perf_counter() - t_all
    assert c.is_terminal() and json.loads(c.generated) == vals[17:22]
    assert max(per) < 0.1 and total < 2.0, (
        f"{len(per)} tokens : {total:.1f} s au total, pire token {max(per) * 1000:.0f} ms")


# ── Masque exact : variantes max_whitespace / max_number_digits, structures croisées ─

R3_SCHEMAS = [
    {"type": "array", "items": {"enum": [1, 12, 123, -1, 1.5, 1e5, 0, -0.0]}},
    {"type": "object", "properties": {"a": {"type": "integer"}, "ab": {"type": "integer"},
                                      "abc": {"type": "string"}}, "required": ["abc"]},
    {"anyOf": [{"type": "object", "properties": {"k": {"const": 1}}, "required": ["k"]},
               {"type": "object", "properties": {"k": {"const": 12}}, "required": ["k"]},
               {"type": "object", "properties": {"kk": {"type": "array", "items": {"type": "null"}}},
                "required": ["kk"]}]},
    {"type": "array", "items": {"type": "array", "items": {"type": "array", "items": {"type": "integer"},
                                                           "maxItems": 2}, "maxItems": 2},
     "maxItems": 3},
    {"const": {"a": [1, {"b": "é\"\\"}], "c": None}},
    {"type": "object", "additionalProperties": {"type": "object",
                                                "additionalProperties": {"type": "integer"}}},
    {"type": "array", "items": {"type": "string", "maxLength": 2}, "minItems": 1, "maxItems": 3},
    {"type": "object", "properties": {"éé": {"type": "boolean"}, "\\": {"type": "null"},
                                      "/": {"type": "number"}}, "required": ["/"]},
    {"type": ["integer", "string", "null"], "maxLength": 3},
    {"enum": ["a", "ab", "abc", "b\"", "—", "\U0001F600x"]},
]


@pytest.mark.parametrize("si", range(len(R3_SCHEMAS)))
def test_r3_mask_vs_fresh_matcher_ws_and_digit_limits_gpt2(gpt2, si):
    """
    Régression : masque == force brute d'un matcher NEUF pour max_whitespace 0 / 1 / 4 et
    max_number_digits 2 / 20 (tokens gpt2 à cheval : '":[', '1,', '0]', '"}', ' "'…),
    puis completion_tokens → instance valide.
    """
    enc, tb = gpt2
    schema = R3_SCHEMAS[si]
    rng = random.Random(300 + si)
    for mws, dig in [(4, 20), (0, 20), (1, 2)]:
        m = JSONSchemaMatcher(schema, max_whitespace=mws, max_number_digits=dig, filler='-')
        c = TokenConstraint(m, tb)
        for _ in range(3):
            c.reset()
            for _ in range(rng.randint(0, 25)):
                ids = c.allowed_mask().nonzero().flatten().tolist()
                if not ids:
                    break
                pool = [i for i in ids if not _LETTERS.search(tb[i])] if rng.random() < .75 else ids
                c.advance(rng.choice(pool or ids))
            mask = c.allowed_mask().tolist()
            fresh = JSONSchemaMatcher(schema, max_whitespace=mws, max_number_digits=dig)
            st = fresh.advance_bytes(fresh.initial_state, c.generated)
            assert st is not None
            bad = [(i, tb[i]) for i in range(len(tb))
                   if (tb[i] is not None and fresh.advance_bytes(st, tb[i]) is not None) != mask[i]]
            assert not bad, (schema, mws, dig, c.generated, bad[:5])
            toks = c.completion_tokens()
            assert toks is not None, (schema, c.generated)
            d = c.clone()
            for t in toks:
                assert d.is_allowed(t)
                d.advance(t)
            assert d.is_complete()
            assert validate_instance(json.loads(d.generated), schema) == [], d.generated


# ── completion_tokens : recherche A* avec des vocabulaires char-level lacunaires ─

def _r3_vocab(chars):
    chars = sorted(set(chars))
    return chars, token_bytes_from_itos(dict(enumerate(chars)), len(chars))


_R3_ASCII = [chr(i) for i in range(32, 127)] + ['\n']


@pytest.mark.parametrize("case", ["key_e_acute", "key_no_lower_hex", "items3000_no_zero",
                                  "minlen_then_const", "pending_u_only_zero", "enum_no_one",
                                  "enum_none", "deep_no_bracket"])
def test_r3_completion_tokens_search_char_vocab_missing_chars(case):
    """
    Régression : la plus courte complétion en octets est inécrivable (caractère absent du
    vocabulaire) → la recherche A* trouve une AUTRE fermeture valide ('é' → \\u00E9 en
    majuscules si a-f manquent, '0' → '1' sur 3 000 éléments, \\u0 → \\u0000…) ou
    renvoie None quand aucune n'existe ('1' absent pour enum [1], ']' absent).
    """
    E_KEY = {"type": "object", "properties": {"é": {"type": "integer"}}, "required": ["é"]}
    spec = {
        "key_e_acute": (_R3_ASCII, E_KEY, '', True),
        "key_no_lower_hex": ([x for x in _R3_ASCII if x not in 'abcdef'], E_KEY, '', True),
        "items3000_no_zero": ([x for x in _R3_ASCII if x != '0'],
                              {"type": "array", "items": {"type": "integer"}, "minItems": 3000},
                              '[1,2', True),
        "minlen_then_const": (_R3_ASCII, {"type": "object", "properties": {
            "s": {"type": "string", "minLength": 2000}, "k": {"const": "é"}},
            "required": ["s", "k"]}, '{"s":"x', True),
        "pending_u_only_zero": ([x for x in _R3_ASCII if x not in '123456789abcdefABCDEF'],
                                {"type": "string"}, '"\\u0', True),
        "enum_no_one": ([x for x in _R3_ASCII if x != '1'], {"enum": [1, 2]}, '', True),
        "enum_none": ([x for x in _R3_ASCII if x != '1'], {"enum": [1]}, '', False),
        "deep_no_bracket": ([x for x in _R3_ASCII if x != ']'], None, '[' * 50, False),
    }[case]
    chars, schema, prefix, possible = spec[0], spec[1], spec[2], spec[3]
    chars, tb = _r3_vocab(chars)
    c = json_constraint(tb, schema)
    for ch in prefix:
        c.advance(chars.index(ch))
    t0 = time.perf_counter()
    toks = c.completion_tokens()
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, elapsed
    if not possible:
        assert toks is None
        return
    assert toks is not None
    d = c.clone()
    for t in toks:
        assert d.is_allowed(t)
        d.advance(t)
    assert d.is_complete()
    inst = json.loads(d.generated)
    if schema is not None:
        assert validate_instance(inst, schema) == [], d.generated


def test_r3_shortest_completion_never_uses_bfs_fallback_char_vocab():
    """
    Régression : sur des marches aléatoires (paires de substitution en cours, UTF-8
    partiel impossible en char-level mais échappements partiels oui, prefixItems +
    minItems, chaînes constantes à échappements), la complétion analytique est
    toujours vérifiée — le BFS de secours (borné à 512 octets) ne sert jamais.
    """
    chars, tb = _char_vocab()
    schemas = [
        {"type": "string", "minLength": 3, "maxLength": 3},
        {"type": "array", "prefixItems": [{"type": "string", "minLength": 2},
                                          {"enum": ["é", "\\u"]}],
         "minItems": 4, "items": {"type": "integer"}},
        {"enum": ["ab", "a\"", "\U0001F600", "éé", "x\\y", "\x01"]},
        {"anyOf": [{"type": "string", "maxLength": 1}, {"type": "string", "minLength": 3}]},
        {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 2},
         "minItems": 3},
        {"$defs": {"n": {"type": "array", "items": {"$ref": "#/$defs/n"}, "minItems": 1,
                         "maxItems": 1},
                   "m": {"anyOf": [{"$ref": "#/$defs/n"}, {"type": "null"}]}},
         "type": "array", "items": {"$ref": "#/$defs/m"}},
    ]
    rng = random.Random(9)
    for schema in schemas:
        c = json_constraint(tb, schema)
        m = c.matcher
        for _ in range(120):
            c.reset()
            for _ in range(rng.randint(0, 30)):
                ids = c.allowed_mask().nonzero().flatten().tolist()
                if not ids:
                    break
                c.advance(rng.choice(ids))
            assert m.shortest_completion(c.state) is not None, (schema, c.generated)
        assert m._fallbacks == 0, schema
