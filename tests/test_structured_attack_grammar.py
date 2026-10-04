"""
Tests adverses — solidité de la grammaire JSON (structured.py) face à json.loads.

  · Différentiel exhaustif (chaînes courtes) et aléatoire (générées + mutées) :
    l'automate accepte un document  ⇔  json.loads l'accepte ET les limites
    documentées sont respectées (espaces ≤ max_whitespace par interstice, pas
    d'espace final, ≤ max_number_digits chiffres par série, pas de NaN/Infinity).
  · Viabilité : tout préfixe accepté se complète (shortest_completion) en un
    document valide ; aucun état mort.
  · Fermeture par préfixe des sorties json.dumps (compact / séparateurs par défaut,
    ensure_ascii=True/False).
  · Chaînes suivies (minLength / maxLength) : comptage des points de code avec
    échappements et paires de substitution, comparé à len(json.loads(...)).
  · Nombres : -0, 1e5, 1E+5, 0.0, 01, -, 1., limites de chiffres, contextes.
  · Chaînes constantes : règle « littéral seulement » exacte.
  · Plafonds d'espaces à chaque interstice, imbrication profonde.
"""

import itertools
import json
import random
import re

import pytest

import structured as S
from structured import JSONSchemaMatcher, json_constraint, validate_instance


# ─────────────────────────────────────────────────────────────────────────────
# Référence
# ─────────────────────────────────────────────────────────────────────────────

_WS = b' \t\n\r'


def _no_constant(name):
    raise ValueError(name)          # NaN / Infinity / -Infinity : pas du JSON strict


def ref_value(data: bytes):
    """(True, valeur) si `data` est un document JSON strict (UTF-8), sinon (False, None)."""
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        return False, None
    try:
        return True, json.loads(text, parse_constant=_no_constant)
    except (ValueError, RecursionError):
        return False, None


def lex_ok(data: bytes, max_ws=4, max_digits=20) -> bool:
    """Limites lexicales DOCUMENTÉES de l'automate (document supposé JSON valide)."""
    n = len(data)
    if n and data[-1] in _WS:
        return False                # pas d'espace final
    i = 0
    while i < n:
        c = data[i]
        if c == 0x22:
            i += 1
            while data[i] != 0x22:
                i += 2 if data[i] == 0x5C else 1
            i += 1
        elif c in _WS:
            j = i
            while j < n and data[j] in _WS:
                j += 1
            if j - i > max_ws:
                return False
            i = j
        elif 0x30 <= c <= 0x39:
            j = i
            while j < n and 0x30 <= data[j] <= 0x39:
                j += 1
            if j - i > max_digits:
                return False
            i = j
        else:
            i += 1
    return True


def ref_ok(data: bytes, max_ws=4, max_digits=20) -> bool:
    ok, _ = ref_value(data)
    return ok and lex_ok(data, max_ws, max_digits)


def run(m, data):
    st = m.initial_state
    for b in data:
        st = m.advance(st, b)
        if st is None:
            return None
    return st


def accepted(m, data) -> bool:
    st = run(m, data)
    return st is not None and m.is_accepting(st)


_ANY = JSONSchemaMatcher(None)


# ─────────────────────────────────────────────────────────────────────────────
# Générateur de documents JSON « tordus » (tous échappements, nombres, espaces)
# ─────────────────────────────────────────────────────────────────────────────

def rand_char(rng):
    r = rng.random()
    if r < 0.4:
        return chr(rng.randint(0x20, 0x7E))
    if r < 0.5:
        return chr(rng.randint(0, 0x1F))
    if r < 0.6:
        return chr(rng.randint(0x80, 0x7FF))
    if r < 0.7:
        return chr(rng.choice([0x7F, 0x2028, 0xFEFF, 0xFFFF, 0xE000, 0xD7FF]))
    if r < 0.8:
        return chr(rng.randint(0x10000, 0x10FFFF))
    if r < 0.9:
        return chr(rng.randint(0xD800, 0xDFFF))       # surrogate isolé
    return rng.choice('"\\/\b\f\n\r\t')


def enc_char(ch, rng):
    """Un encodage JSON valide quelconque du caractère (brut, court, \\u casse mixte)."""
    cp = ord(ch)
    opts = []
    if ch in '"\\':
        opts.append('\\' + ch)
    elif ch == '/':
        opts += ['/', '\\/']
    elif ch in '\b\f\n\r\t':
        opts.append({'\b': '\\b', '\f': '\\f', '\n': '\\n', '\r': '\\r', '\t': '\\t'}[ch])
    elif cp >= 0x20 and not 0xD800 <= cp <= 0xDFFF:
        opts.append(ch)
    if cp < 0x10000:
        h = '%04x' % cp
        opts.append('\\u' + ''.join(c.upper() if rng.random() < 0.5 else c for c in h))
    else:
        v = cp - 0x10000
        opts.append('\\u%04x\\u%04X' % (0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF)))
    return rng.choice(opts)


def rand_num(rng):
    s = rng.choice(['', '-'])
    s += rng.choice(['0', str(rng.randint(1, 9)) +
                     ''.join(rng.choice('0123456789') for _ in range(rng.randint(0, 19)))])
    if rng.random() < 0.4:
        s += '.' + ''.join(rng.choice('0123456789') for _ in range(rng.randint(1, 20)))
    if rng.random() < 0.4:
        s += (rng.choice('eE') + rng.choice(['', '+', '-']) +
              ''.join(rng.choice('0123456789') for _ in range(rng.randint(1, 3))))
    return s


def rand_ws(rng, mx=4):
    if rng.random() < 0.6:
        return ''
    return ''.join(rng.choice(' \t\n\r') for _ in range(rng.randint(0, mx)))


def rand_str(rng, n):
    return '"' + ''.join(enc_char(rand_char(rng), rng) for _ in range(n)) + '"'


def rand_doc(rng, depth=0):
    r = rng.random()
    if depth > 6 or r < 0.35:
        k = rng.randrange(5)
        if k == 0:
            return rand_num(rng)
        if k == 1:
            return rand_str(rng, rng.randint(0, 6))
        return rng.choice(['true', 'false', 'null'])
    sep = lambda: rand_ws(rng) + ',' + rand_ws(rng)     # noqa: E731
    n = rng.randint(0, 4)
    if r < 0.7:
        if not n:
            return '[' + rand_ws(rng) + ']'
        return ('[' + rand_ws(rng) + sep().join(rand_doc(rng, depth + 1) for _ in range(n))
                + rand_ws(rng) + ']')
    if not n:
        return '{' + rand_ws(rng) + '}'
    items = [rand_str(rng, rng.randint(0, 4)) + rand_ws(rng) + ':' + rand_ws(rng)
             + rand_doc(rng, depth + 1) for _ in range(n)]
    return '{' + rand_ws(rng) + sep().join(items) + rand_ws(rng) + '}'


def to_bytes(s: str) -> bytes:
    return s.encode('utf-8', 'surrogatepass')   # les surrogates n'apparaissent qu'échappés


_MUT_BYTES = (b'[]{},:"\\u0123456789abcdefABCDEF-+.eE ntrulsf\t\n'
              b'\x00\x1f\x7f\x80\xbf\xc2\xe0\xed\xf0\xf4\xff')


def mutate(rng, data: bytes) -> bytes:
    dd = bytearray(data)
    for _ in range(rng.randint(1, 3)):
        op = rng.randrange(3)
        pos = rng.randrange(len(dd) + 1)
        byte = rng.choice(_MUT_BYTES)
        if op == 0:
            dd.insert(pos, byte)
        elif pos < len(dd):
            if op == 1:
                del dd[pos]
            else:
                dd[pos] = byte
    return bytes(dd)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Grammaire libre (schema=None) : différentiel avec json.loads
# ─────────────────────────────────────────────────────────────────────────────

_SHORT_ALPHABET = [b'[', b']', b'{', b'}', b',', b':', b'"', b'\\', b'u', b'0', b'1', b'-',
                   b'.', b'e', b'E', b'+', b' ', b'n', b't', b'a', b'd', b'\xc3', b'\xa9',
                   b'\x7f', b'\x1f']


def test_free_grammar_exhaustive_short_strings_match_json_loads():
    """Toutes les chaînes de ≤ 4 symboles (≈ 400 000) : acceptation ⇔ référence."""
    bad = []
    for n in range(5):
        for combo in itertools.product(_SHORT_ALPHABET, repeat=n):
            d = b''.join(combo)
            if accepted(_ANY, d) != ref_ok(d):
                bad.append(d)
    assert not bad, bad[:10]


@pytest.mark.parametrize("seed", [7, 8])
def test_free_grammar_random_and_mutated_docs_match_json_loads(seed):
    rng = random.Random(seed)
    bad = []
    for _ in range(1200):
        d = to_bytes(rand_ws(rng) + rand_doc(rng))
        if accepted(_ANY, d) != ref_ok(d):
            bad.append(('gen', d))
        for _ in range(8):
            dd = mutate(rng, d)
            if accepted(_ANY, dd) != ref_ok(dd):
                bad.append(('mut', dd))
    assert not bad, bad[:10]


def test_every_accepted_prefix_is_viable():
    """Aucun état mort : chaque préfixe non-None se complète en JSON valide."""
    rng = random.Random(11)
    m = _ANY
    for _ in range(250):
        d = bytearray(to_bytes(rand_ws(rng) + rand_doc(rng)))
        for _ in range(rng.randint(0, 2)):
            d.insert(rng.randrange(len(d) + 1), rng.choice(b'[]{},:"\\u0-.e \xc3\xa9'))
        st = m.initial_state
        for i, b in enumerate(d):
            st = m.advance(st, b)
            if st is None:
                break
            assert m.is_accepting(st) or m.can_continue(st), bytes(d[:i + 1])
            c = m.shortest_completion(st)
            assert c is not None, bytes(d[:i + 1])
            assert ref_ok(bytes(d[:i + 1]) + c), bytes(d[:i + 1]) + c
    assert m._fallbacks == 0


@pytest.mark.parametrize("seed", [1, 2])
def test_json_dumps_output_prefix_closed(seed):
    """Sorties json.dumps (compactes / séparateurs par défaut, ensure_ascii True/False)."""
    rng = random.Random(seed)
    for _ in range(300):
        v = json.loads(to_bytes(rand_doc(rng)).decode('utf-8'))
        for sep in ((',', ':'), None):
            for ea in (True, False):
                try:
                    data = json.dumps(v, ensure_ascii=ea, separators=sep).encode('utf-8')
                except UnicodeEncodeError:
                    continue            # surrogate isolé non encodable en UTF-8 brut
                if not ref_ok(data):
                    continue            # Infinity (1e999) ou > max_number_digits (documenté)
                st = _ANY.initial_state
                for i, b in enumerate(data):
                    st = _ANY.advance(st, b)
                    assert st is not None, (data, i)
                assert _ANY.is_accepting(st), data


def test_leading_whitespace_and_root_scalars():
    for mw in (0, 1, 4):
        m = JSONSchemaMatcher(None, max_whitespace=mw)
        for root in (b'0', b'-0', b'1e5', b'1E+5', b'0.0', b'"x"', b'true', b'null', b'[]', b'{}'):
            for k in range(6):
                for w in (b' ', b'\n', b'\t', b'\r'):
                    assert accepted(m, w * k + root) == (k <= mw), (mw, k, root)
        assert not accepted(m, b'\xef\xbb\xbf{}')            # BOM
        assert not accepted(m, b'\xc2\xa0{}')                # espace insécable
        assert not accepted(m, b'')
    for bad in (b'01', b'-', b'1.', b'.5', b'+1', b'1e', b'1e+', b'--1', b'0x1', b'1_0',
                b'NaN', b'Infinity', b'-Infinity', b'nul', b'True', b"'a'", b'"a', b'[1 2]'):
        assert not accepted(_ANY, bad), bad


def test_deep_nesting():
    for depth in (300, 1500):
        assert accepted(_ANY, b'[' * depth + b']' * depth)
        assert accepted(_ANY, b'{"a":' * depth + b'-0.5e+3' + b'}' * depth)
        assert not accepted(_ANY, b'[' * depth + b']' * (depth - 1))
        assert run(_ANY, b'[' * depth + b']' * (depth + 1)) is None
        st = run(_ANY, b'{"a":[' * depth + b'"\\ud83d')
        c = _ANY.shortest_completion(st)
        full = b'{"a":[' * depth + b'"\\ud83d' + c
        assert accepted(_ANY, full) and lex_ok(full)
        assert len(c) == len(b'"') + 2 * depth
    tree = {"type": "object",
            "properties": {"v": {"type": "integer"}, "c": {"type": "array", "items": {"$ref": "#"}}},
            "required": ["v"]}
    mt = JSONSchemaMatcher(tree)
    assert accepted(mt, b'{"v":1,"c":[' * 400 + b'{"v":2}' + b']}' * 400)
    assert not accepted(mt, b'{"v":1,"c":[' * 400 + b'{"w":2}' + b']}' * 400)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Chaînes suivies : minLength / maxLength vs len(json.loads(...))
# ─────────────────────────────────────────────────────────────────────────────

_STR_PIECES = ['a', 'é', '😀', '\\ud83d', '\\uD83D', '\\ude00', '\\uDE00', '\\udbff',
               '\\udfff', '\\ud800', '\\u0041', '\\n', '\\/', '\\ue000', '\\udBfF']


@pytest.mark.parametrize("mn,mx", [(0, 0), (0, 1), (1, 1), (0, 2), (2, 2), (1, 3), (2, None),
                                   (3, None), (1, None)])
def test_tracked_string_length_matches_python(mn, mx):
    sch = {"type": "string", "minLength": mn}
    if mx is not None:
        sch["maxLength"] = mx
    m = JSONSchemaMatcher(sch)
    bad = []
    for n in range(4):
        for combo in itertools.product(_STR_PIECES, repeat=n):
            d = to_bytes('"' + ''.join(combo) + '"')
            ok, v = ref_value(d)
            exp = ok and mn <= len(v) and (mx is None or len(v) <= mx)
            if accepted(m, d) != exp:
                bad.append(d)
    assert not bad, bad[:10]


def test_tracked_string_prefixes_viable():
    rng = random.Random(4)
    for sch in ({"type": "string", "maxLength": 2}, {"type": "string", "minLength": 3},
                {"type": "string", "minLength": 1, "maxLength": 1}):
        m = JSONSchemaMatcher(sch)
        for _ in range(300):
            d = to_bytes('"' + ''.join(rng.choice(_STR_PIECES) for _ in range(3)))
            st = m.initial_state
            for i, b in enumerate(d):
                st = m.advance(st, b)
                if st is None:
                    break
                c = m.shortest_completion(st)
                full = d[:i + 1] + c
                ok, v = ref_value(full)
                assert ok and validate_instance(v, sch) == [], (sch, full)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Nombres
# ─────────────────────────────────────────────────────────────────────────────

_INT_RE = re.compile(r'-?(0|[1-9][0-9]*)\Z')


@pytest.mark.parametrize("integer", [False, True])
@pytest.mark.parametrize("ctx", ['{}', '[{}]', '[{},1]', '{{"a":{}}}', '{{"a":{},"b":1}}'])
def test_number_grammar_differential(integer, ctx):
    num_s = {"type": "integer" if integer else "number"}
    if ctx == '{}':
        sch = num_s
    elif ctx.startswith('['):
        sch = {"type": "array", "items": num_s}
    else:
        sch = {"type": "object", "properties": {"a": num_s, "b": {"type": "integer"}},
               "required": ["a"]}
    for D in (1, 20):
        m = JSONSchemaMatcher(sch, max_number_digits=D)
        bad = []
        for n in range(1, 5):
            for combo in itertools.product('-+.eE019 ', repeat=n):
                num = ''.join(combo)
                d = ctx.format(num).encode()
                ok, v = ref_value(d)
                exp = ok and lex_ok(d, 4, D) and validate_instance(v, sch) == []
                if exp and integer and num.strip():
                    exp = bool(_INT_RE.match(num.strip()))
                if accepted(m, d) != exp:
                    bad.append(d)
        assert not bad, (D, bad[:10])


def test_number_digit_caps_per_run():
    m = JSONSchemaMatcher({"type": "number"}, max_number_digits=3)
    for good in (b'999', b'-999', b'0.999', b'999.999e999', b'1E+999', b'-0.000'):
        assert accepted(m, good), good
    for bad in (b'1000', b'0.1234', b'1e1234', b'-1234.5'):
        assert not accepted(m, bad), bad
    st = run(m, b'123')
    assert m.is_accepting(st) and m.can_continue(st)            # '.', 'e' encore permis
    assert m.allowed_bytes(st) == tuple(sorted(b'.eE'))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Chaînes constantes : règle « littéral seulement »
# ─────────────────────────────────────────────────────────────────────────────

def _all_encodings(ch):
    """(encodage, canonique selon le docstring) pour un caractère."""
    cp = ord(ch)
    out = []
    short = {'"': '\\"', '\\': '\\\\', '/': '\\/', '\b': '\\b', '\f': '\\f', '\n': '\\n',
             '\r': '\\r', '\t': '\\t'}
    if ch in short:
        out.append((short[ch], ch != '/'))
    if cp >= 0x20 and not 0xD800 <= cp <= 0xDFFF and ch not in '"\\':
        out.append((ch, True))
    if cp < 0x10000:
        h = '%04x' % cp
        canon = not 0x20 <= cp <= 0x7E and ch not in '\b\f\n\r\t'
        out += [('\\u' + h, canon), ('\\u' + h.upper(), canon)]
    else:
        v = cp - 0x10000
        hi, lo = 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF)
        out += [('\\u%04x\\u%04x' % (hi, lo), True), ('\\u%04X\\u%04X' % (hi, lo), True)]
    return out


@pytest.mark.parametrize("ch", ['a', '/', '"', '\\', '\n', '\b', '\x00', '\x1f', '\x7f', 'é',
                                '€', '😀', '\ud800', '\udfff', ' ', ' ', '~', 'A', 'F',
                                'u'])
def test_constant_string_encoding_rule_exact(ch):
    s = 'x' + ch + 'y'
    cases = [
        ({"const": s}, lambda e: '"' + e + '"'),
        ({"type": "object", "properties": {s: {"type": "null"}}, "required": [s]},
         lambda e: '{"' + e + '":null}'),
        ({"enum": [{s: 1}, s]}, lambda e: '{"' + e + '":1}'),
    ]
    for sch, wrap in cases:
        m = JSONSchemaMatcher(sch)
        for enc, canon in _all_encodings(ch):
            d = to_bytes(wrap('x' + enc + 'y'))
            a = accepted(m, d)
            assert a == canon, (sch, d)
            if a:
                ok, v = ref_value(d)
                assert ok and validate_instance(v, sch) == [], (sch, d)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Plafonds d'espaces à chaque interstice
# ─────────────────────────────────────────────────────────────────────────────

RICH = {
    "type": "object",
    "properties": {
        "k": {"const": {"a": [1, -2.5, None, True, "s", {}], "b": {"c": []}}},
        "e": {"enum": [[1, 2], {"x": "y"}, 7, "z"]},
        "f": {"type": "object",
              "additionalProperties": {"type": "array", "items": {"type": "integer"}}},
        "t": {"type": "array", "prefixItems": [{"type": "string"}, {"type": "number"}],
              "items": {"type": "boolean"}, "minItems": 2},
        "n": {"type": "null"},
        "o": {"type": "integer"},
    },
    "required": ["k", "e", "f", "t", "n"],
}
RICH_INST = {"k": {"a": [1, -2.5, None, True, "s", {}], "b": {"c": []}}, "e": {"x": "y"},
             "f": {"p": [1, 2], "q": []}, "t": ["s", 1.5e3, True], "n": None, "o": -0}


def _tokens(text):
    toks, i = [], 0
    while i < len(text):
        c = text[i]
        if c == '"':
            j = i + 1
            while text[j] != '"':
                j += 2 if text[j] == '\\' else 1
            toks.append(text[i:j + 1])
            i = j + 1
        elif c in '{}[],:':
            toks.append(c)
            i += 1
        else:
            j = i
            while j < len(text) and text[j] not in '{}[],:"':
                j += 1
            toks.append(text[i:j])
            i = j
    return toks


@pytest.mark.parametrize("mw", [0, 1, 4])
def test_whitespace_cap_every_gap(mw):
    m = JSONSchemaMatcher(RICH, max_whitespace=mw)
    toks = _tokens(json.dumps(RICH_INST, separators=(',', ':')))
    for g in range(len(toks) + 1):
        for k in range(6):
            for w in (' ', '\n', '\t\r'):
                d = (''.join(toks[:g]) + (w * 6)[:k] + ''.join(toks[g:])).encode()
                exp = k == 0 or (g < len(toks) and k <= mw)     # jamais d'espace final
                assert accepted(m, d) == exp, (mw, g, k, d)


def test_rich_schema_json_dumps_variants():
    m = JSONSchemaMatcher(RICH)
    for kw in (dict(), dict(ensure_ascii=False), dict(separators=(',', ':')),
               dict(indent=1), dict(indent='\t')):
        assert accepted(m, json.dumps(RICH_INST, **kw).encode()), kw
    assert not accepted(m, json.dumps(RICH_INST, indent=2).encode())     # retrait 6 > 4
    assert accepted(JSONSchemaMatcher(RICH, max_whitespace=16),
                    json.dumps(RICH_INST, indent=2).encode())


# ─────────────────────────────────────────────────────────────────────────────
# 6. Défauts
# ─────────────────────────────────────────────────────────────────────────────

_PAIR_UNITS = '\ud83d' + '\ude00'      # 2 unités de code Python (≠ '😀', 1 caractère)


@pytest.mark.parametrize("schema,doc", [
    ({"const": _PAIR_UNITS}, b'"\\ud83d\\ude00"'),
    ({"enum": ["a", _PAIR_UNITS]}, b'"\\ud83d\\ude00"'),
    ({"type": "object", "properties": {_PAIR_UNITS: {"type": "null"}},
      "required": [_PAIR_UNITS]}, b'{"\\ud83d\\ude00":null}'),
])
def test_constant_with_surrogate_code_unit_pair_is_sound(schema, doc):
    """
    Une chaîne constante contenant une paire haut+bas sous forme de DEUX unités de
    code Python est encodée '\\ud83d\\ude00', que json.loads recombine en '😀'
    (un seul caractère) : le document accepté n'est plus valide pour le schéma.
    Tout document accepté doit être décodable ET valide.
    """
    m = JSONSchemaMatcher(schema)
    if accepted(m, doc):
        assert validate_instance(json.loads(doc), schema) == [], doc
    w = m.shortest_completion(m.initial_state)
    assert validate_instance(json.loads(w), schema) == [], w


def test_completion_tokens_uses_alternative_when_vocab_lacks_shortest_bytes():
    """
    completion_tokens() ne cherche à tokeniser QUE la plus courte complétion en octets ;
    si le vocabulaire n'a pas exactement ces octets (ex. 'é' brut, 'a' de remplissage)
    il renvoie None alors qu'une autre complétion est atteignable avec ce vocabulaire
    (docstring : « None si impossible avec ce vocabulaire »).
    """
    chars = list('{}[]:,"\\ 0123456789abcdefnulrstuopmABCDEF-+.')
    tb = [c.encode() for c in chars]
    sch = {"type": "object", "properties": {"prénom": {"type": "string"}},
           "required": ["prénom"]}
    tc = json_constraint(tb, sch)
    # la complétion existe bien avec ce vocabulaire : é au lieu de 'é' brut
    probe = tc.clone()
    for ch in '{"pr\\u00e9nom":""}':
        probe.advance(chars.index(ch))
    assert probe.is_complete()
    toks = tc.completion_tokens()
    assert toks is not None
    c = tc.clone()
    for t in toks:
        c.advance(t)
    assert c.is_complete()

    tb2 = [c.encode() for c in '"bcd']
    tc2 = json_constraint(tb2, {"type": "string", "minLength": 2})     # '"bb"' possible
    assert tc2.completion_tokens() is not None
