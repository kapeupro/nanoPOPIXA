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
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
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
