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


# ─────────────────────────────────────────────────────────────────────────────
# 7. Vague 2 — différentiels élargis (régressions)
# ─────────────────────────────────────────────────────────────────────────────

import sys      # noqa: E402

from structured import SchemaError      # noqa: E402


def test_utf8_inside_free_strings_exhaustive():
    """Tout contenu de 1 ou 2 octets quelconques, et les octets de tête 3/4 octets avec
    continuations aux bornes : acceptation ⇔ UTF-8 valide + json.loads (contrôles < 0x20,
    surlongs, surrogates ED A0.., > U+10FFFF, continuations orphelines)."""
    bad = []
    for n in (1, 2):
        for combo in itertools.product(range(256), repeat=n):
            d = b'"' + bytes(combo) + b'"'
            if accepted(_ANY, d) != ref_ok(d):
                bad.append(d)
    edge = (0x7F, 0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF, 0xC0)
    for a in range(0xE0, 0xF8):
        for b in edge:
            for c in edge:
                tails = (b'',) if a < 0xF0 else tuple(bytes((e,)) for e in edge)
                for t in tails:
                    d = b'"' + bytes((a, b, c)) + t + b'"'
                    if accepted(_ANY, d) != ref_ok(d):
                        bad.append(d)
    assert not bad, bad[:10]


def test_structural_exhaustive_tight_whitespace():
    """Toutes les suites de ≤ 5 symboles structurels (avec espaces) pour max_whitespace
    0 et 1 : acceptation ⇔ json.loads + plafond d'espaces documenté."""
    alpha = [b'[', b']', b'{', b'}', b',', b':', b'"a"', b' ', b'1', b'\t', b'-', b'0']
    for mw in (0, 1):
        m = JSONSchemaMatcher(None, max_whitespace=mw)
        bad = []
        stack = [(m.initial_state, b'', 0)]
        while stack:
            st, d, depth = stack.pop()
            for s in alpha:
                nd = d + s
                ns = m.advance_bytes(st, s) if st is not None else None
                if (ns is not None and m.is_accepting(ns)) != ref_ok(nd, mw):
                    bad.append(nd)
                if depth < 4:
                    stack.append((ns, nd, depth + 1))
        assert not bad, (mw, bad[:10])


_SUITE = [
    # JSONTestSuite (y_ / n_ / i_) — verdict calculé par la référence, pas codé en dur
    b'[1E22]', b'[1E-2]', b'[1e+2]', b'[-0]', b'[-0.0]', b'[0e1]', b'[0E+0]', b'[123e65]',
    b'[-123.123e-123]', b'[1.0e+]', b'[1.0e-]', b'[1.0e]', b'[-]', b'[.123]', b'[0x1]',
    b'[1.]', b'[012]', b'[-01]', b'[+1]', b'[1e]', b'[1e+]', b'[- 1]', b'[-1.0.]', b'[0.e1]',
    b'[2.e+3]', b'[1.2a-3]', b'[1ea]', b'[1+2]', b'[Infinity]', b'[-Infinity]', b'[NaN]',
    b'["\\u00A"]', b'["\\x00"]', b'["\\a"]', b'["\\uqqqq"]', b'["\t"]', b'["a\x00"]',
    b'["\\uD800\\uD800\\n"]', b'{"\\uDFAA":0}', b'["\\uFFFF"]', b'["\\u0061\\u30af\\u30EA"]',
    b'["\\uDBFF\\uDFFF"]', b'["\\ud834\\udd1e"]', b'["\\/"]', b'["\xed\xa0\x80"]',
    b'["\xc0\xaf"]', b'["\xf4\x90\x80\x80"]', b'["\xe0\x80\xaf"]', b'["\xff"]', b'["\x81"]',
    b'["\xef\xbf\xbf"]', b'["\xf4\x8f\xbf\xbf"]', b'["\xe2\x80\xa8"]', b'[1,]', b'{"a":1,}',
    b'[,1]', b'{"a" "b"}', b'{,}', b'{"a":1 "b":2}', b'[1]x', b'[1]]', b'[1,,2]', b'[\n]',
    b'{"a":"b","a":"c"}', b'{"a":"b"}#', b'[""', b'["a",', b'{"a"', b'{"a":', b'{:1}',
    b'{1:1}', b'{null:1}', b'[true', b'[tru]', b'[nul]', b'[True]', b'["\\"]', b'"\\u"',
    b'[\x0c]', b'[\xc2\xa0]', b'\xef\xbb\xbf[]', b'[]\x00', b'[\x00]', b'["\x7f"]',
    b' [] ', b'  []', b'[[]   ]', b'{"":0}', b'{"a":[]}', b'[{}]', b'[{"a":{"b":[]}}]',
]


def test_json_test_suite_style_cases():
    bad = [d for d in _SUITE if accepted(_ANY, d) != ref_ok(d)]
    assert not bad, bad


# ── Schémas aléatoires (fusions, $ref, enum + frères, types déduits, draft-04) ──

_RCH = ['a', 'b', '/', '"', '\\', '\n', '\x00', '\x7f', 'é', '€', '😀', '\ud800', '\udfff', ' ',
        'u', 'A', ' ']


def _rstr(rng, n=None):
    n = rng.randint(0, 3) if n is None else n
    return ''.join(rng.choice(_RCH) for _ in range(n))


def _rval(rng, d=0):
    r = rng.random()
    if d > 2 or r < 0.5:
        return rng.choice([None, True, False, 0, 1, -1, 1.5, -0.0, 10, 1e20, 123456, 0.1,
                           _rstr(rng), _rstr(rng)])
    if r < 0.75:
        return [_rval(rng, d + 1) for _ in range(rng.randint(0, 3))]
    return {_rstr(rng, rng.randint(0, 2)): _rval(rng, d + 1) for _ in range(rng.randint(0, 3))}


def _rsch(rng, d=0, defs=()):
    r = rng.random() * (0.5 if d > 3 else 1.0)
    if r < 0.06:
        return {}
    if r < 0.12:
        return rng.choice([{"type": "integer"}, {"type": "number"}, {"type": "boolean"},
                           {"type": "null"}, {"type": ["integer", "number"]}, {"minimum": 0}])
    if r < 0.2:
        s = {"minLength": rng.randint(0, 2)}
        if rng.random() < 0.6:
            s["type"] = "string"
        if rng.random() < 0.6:
            s["maxLength"] = rng.randint(0, 3)
        return s
    if r < 0.3:
        s = {"enum": [_rval(rng) for _ in range(rng.randint(1, 5))]}
        k = rng.random()
        if k < 0.2:
            s["type"] = rng.choice(["string", "integer", "number", "array", "object",
                                    ["string", "null"]])
        elif k < 0.3:
            s["minLength"] = rng.randint(0, 2)
        elif k < 0.4:
            s["maxItems"] = rng.randint(0, 2)
        elif k < 0.5:
            s["properties"] = {"": {"type": "integer"}}
        elif k < 0.6:
            s["const"] = rng.choice(s["enum"])
        return s
    if r < 0.36:
        s = {"const": _rval(rng)}
        if rng.random() < 0.3:
            s["type"] = rng.choice(["string", "integer", "number", "array", "object"])
        return s
    if r < 0.55:
        props = {_rstr(rng, rng.randint(0, 2)): _rsch(rng, d + 1, defs)
                 for _ in range(rng.randint(0, 3))}
        s = {"properties": props, "required": [n for n in props if rng.random() < 0.5]}
        if rng.random() < 0.7:
            s["type"] = "object"
        if rng.random() < 0.2:
            s["required"].append("zz")
        if rng.random() < 0.3:
            s["additionalProperties"] = (_rsch(rng, d + 1, defs) if rng.random() < 0.6
                                         else False)
        if rng.random() < 0.15:
            del s["properties"]
        return s
    if r < 0.68:
        s = {"type": "array"} if rng.random() < 0.6 else {}
        k = rng.random()
        if k < 0.25:
            s["items"] = [_rsch(rng, d + 1, defs) for _ in range(rng.randint(1, 2))]
            if rng.random() < 0.5:
                s["additionalItems"] = _rsch(rng, d + 1, defs) if rng.random() < 0.7 else False
        elif k < 0.5:
            s["prefixItems"] = [_rsch(rng, d + 1, defs) for _ in range(rng.randint(1, 2))]
            if rng.random() < 0.5:
                s["items"] = _rsch(rng, d + 1, defs) if rng.random() < 0.7 else False
        else:
            s["items"] = _rsch(rng, d + 1, defs)
        if rng.random() < 0.4:
            s["minItems"] = rng.randint(0, 2)
        if rng.random() < 0.4:
            s["maxItems"] = rng.randint(0, 3)
        return s
    if r < 0.78:
        s = {rng.choice(["anyOf", "oneOf"]): [_rsch(rng, d + 1, defs)
                                              for _ in range(rng.randint(1, 3))]}
        k = rng.random()
        if k < 0.2:
            s["type"] = rng.choice(["object", "string", "array", "integer", ["object", "array"]])
        elif k < 0.3:
            s["required"] = ["a"]
        elif k < 0.4:
            s["minLength"] = 1
        elif k < 0.5:
            s["enum"] = [_rval(rng) for _ in range(3)]
        elif k < 0.55:
            s["properties"] = {"a": {"type": "string"}}
        return s
    if r < 0.86:
        s = {"allOf": [_rsch(rng, d + 1, defs)]}
        k = rng.random()
        if k < 0.3:
            s["type"] = rng.choice(["object", "string", "array", "number"])
        elif k < 0.5:
            s["required"] = ["a"]
        elif k < 0.6:
            s["maxLength"] = 2
        return s
    if defs:
        s = {"$ref": "#/$defs/" + rng.choice(defs)}
        k = rng.random()
        if k < 0.2:
            s["type"] = rng.choice(["object", "string", "array", "integer"])
        elif k < 0.3:
            s["minItems"] = 1
        elif k < 0.4:
            s["description"] = "x"
        return s
    return {"type": rng.sample(["null", "boolean", "integer", "number", "string", "array",
                                "object"], 2)}


def _rroot(rng):
    names = tuple(f"d{i}" for i in range(rng.randint(0, 3)))
    s = dict(_rsch(rng, 0, names))
    if names:
        s["$defs"] = {n: _rsch(rng, 1, names) for n in names}
    return s


def _walk(m, rng, maxlen=70):
    """Marche aléatoire sur les octets autorisés (biaisée vers l'ASCII)."""
    st, out = m.initial_state, bytearray()
    while len(out) < maxlen:
        al = m.allowed_bytes(st)
        if not al or (m.is_accepting(st) and rng.random() < 0.3) or rng.random() < 0.05:
            break
        asc = [b for b in al if b < 0x80]
        b = rng.choice(asc if asc and rng.random() < 0.85 else al)
        st = m.advance(st, b)
        out.append(b)
    return bytes(out), st


def _schemas(seed, n):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        sch = _rroot(rng)
        try:
            out.append((sch, JSONSchemaMatcher(sch, max_whitespace=rng.choice([0, 2, 4]))))
        except SchemaError:
            pass
    return rng, out


@pytest.mark.parametrize("seed", [101, 202, 303])
def test_random_schema_walks_complete_to_valid_documents(seed):
    """Soundness : tout préfixe atteint par une marche sur les octets autorisés se
    complète (shortest_completion) en un document que l'automate accepte, que json.loads
    décode, et que validate_instance valide ; aucun BFS de secours."""
    rng, ms = _schemas(seed, 120)
    for sch, m in ms:
        for _ in range(12):
            pre, st = _walk(m, rng)
            c = m.shortest_completion(st)
            assert c is not None, (sch, pre)
            full = pre + c
            ok, v = ref_value(full)
            assert ok and validate_instance(v, sch) == [], (sch, full)
            assert accepted(m, full), (sch, full)
        assert m._fallbacks == 0, sch


def _finite(v):
    if isinstance(v, float):
        return v == v and v not in (float('inf'), float('-inf'))
    if isinstance(v, list):
        return all(_finite(x) for x in v)
    if isinstance(v, dict):
        return all(_finite(x) for x in v.values())
    return True


@pytest.mark.parametrize("seed", [11, 12])
def test_reserialized_outputs_are_prefix_closed(seed):
    """Complétude : un document produit par l'automate, relu par json.loads puis réécrit
    par json.dumps (compact, séparateurs par défaut, ensure_ascii True/False, indent=1)
    est accepté, chaque préfixe restant vivant."""
    rng = random.Random(seed)
    n = 0
    while n < 120:
        sch = _rroot(rng)
        try:
            m = JSONSchemaMatcher(sch, max_whitespace=64)
        except SchemaError:
            continue
        n += 1
        for _ in range(8):
            pre, st = _walk(m, rng)
            ok, v = ref_value(pre + m.shortest_completion(st))
            if not ok or not _finite(v):
                continue
            for kw in (dict(separators=(',', ':')), dict(), dict(ensure_ascii=False),
                       dict(indent=1), dict(indent='\t', ensure_ascii=False)):
                try:
                    data = json.dumps(v, **kw).encode('utf-8')
                except UnicodeEncodeError:
                    continue            # surrogate isolé : pas d'UTF-8 brut possible
                s = m.initial_state
                for i, b in enumerate(data):
                    s = m.advance(s, b)
                    assert s is not None, (sch, data, i)
                assert m.is_accepting(s), (sch, data)


def test_shortest_completion_is_exactly_minimal():
    """shortest_completion == plus courte complétion trouvée par BFS exhaustif, et
    _completion_len (sans matérialiser) == len(shortest_completion)."""
    rng, ms = _schemas(77, 60)
    for sch, m in ms:
        for _ in range(4):
            pre, st = _walk(m, rng, maxlen=30)
            c = m.shortest_completion(st)
            assert m._completion_len(st) == len(c), (sch, pre)
            b = m._bfs_completion(st, max_depth=len(c), max_nodes=6_000)
            assert b is None or len(b) >= len(c), (sch, pre, c, b)


def test_cache_eviction_never_changes_the_language():
    """Caches vidés en permanence (_max_states minuscule) : mêmes octets autorisés,
    même acceptation, même complétion qu'un automate aux caches intacts."""
    rng = random.Random(5)
    n = 0
    while n < 100:
        sch = _rroot(rng)
        try:
            m = JSONSchemaMatcher(sch, max_whitespace=2)
            ref_m = JSONSchemaMatcher(sch, max_whitespace=2)
        except SchemaError:
            continue
        n += 1
        m._max_states = 3
        m._max_completion_bytes = 50
        for _ in range(5):
            pre, st_ref = _walk(ref_m, rng, maxlen=50)
            s, r = m.initial_state, ref_m.initial_state
            for b in pre:
                s, r = m.advance(s, b), ref_m.advance(r, b)
                assert s is not None
                assert m.allowed_bytes(s) == ref_m.allowed_bytes(r), (sch, pre)
            assert m.is_accepting(s) == ref_m.is_accepting(st_ref)
            assert m.shortest_completion(s) == ref_m.shortest_completion(st_ref), (sch, pre)


def test_tracked_strings_four_surrogate_pieces():
    """minLength / maxLength : 4 morceaux pris parmi hauts / bas isolés, paires (casse
    mixte), astral brut, \\n, \\u0041 — longueur comparée à len(json.loads(...))."""
    pieces = ['\\ud83d', '\\ude00', '\\uD800', '\\uDFFF', 'a', '\\n', '😀', '\\u0041',
              '\\uDbFf']
    for mn, mx in [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (0, 4), (4, None),
                   (2, None)]:
        sch = {"type": "string", "minLength": mn}
        if mx is not None:
            sch["maxLength"] = mx
        m = JSONSchemaMatcher(sch)
        bad = []
        for n in range(5):
            for combo in itertools.product(pieces, repeat=n):
                d = ('"' + ''.join(combo) + '"').encode()
                ok, v = ref_value(d)
                exp = ok and mn <= len(v) and (mx is None or len(v) <= mx)
                if accepted(m, d) != exp:
                    bad.append(d)
        assert not bad, (sch, bad[:10])


def _every_encoding(ch):
    """Tous les encodages JSON valides d'un caractère (canoniques ou non)."""
    cp = ord(ch)
    out = []
    short = {'"': '\\"', '\\': '\\\\', '/': '\\/', '\b': '\\b', '\f': '\\f', '\n': '\\n',
             '\r': '\\r', '\t': '\\t'}
    if ch in short:
        out.append(short[ch])
    if cp >= 0x20 and not 0xD800 <= cp <= 0xDFFF and ch not in '"\\':
        out.append(ch)
    if cp < 0x10000:
        out += ['\\u%04x' % cp, '\\u%04X' % cp]
    else:
        v = cp - 0x10000
        hi, lo = 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF)
        out += ['\\u%04x\\u%04x' % (hi, lo), '\\u%04X\\u%04x' % (hi, lo)]
    return out


def test_multichar_constants_sound_and_json_dumps_accepted():
    """Chaînes constantes de 1 à 3 caractères (surrogates isolés adjacents, astral, '/',
    DEL, contrôle, '"') : tout encodage accepté se relit en la constante (valeur ET clé
    de propriété), et json.dumps(ensure_ascii=True/False) est toujours accepté."""
    chars = ['\ud83d', '\ude00', 'x', '😀', '/', 'é', '\x7f', '\x01', '"']
    bad = []
    for n in (1, 2, 3):
        for cs in itertools.product(chars, repeat=n):
            s = ''.join(cs)
            sch = {"const": s}
            sch2 = {"type": "object", "properties": {s: {"type": "null"}}, "required": [s]}
            m, m2 = JSONSchemaMatcher(sch), JSONSchemaMatcher(sch2)
            for parts in itertools.product(*(_every_encoding(c) for c in cs)):
                body = ''.join(parts)
                for mm, sc, doc in ((m, sch, '"' + body + '"'),
                                    (m2, sch2, '{"' + body + '":null}')):
                    d = to_bytes(doc)
                    if accepted(mm, d):
                        ok, v = ref_value(d)
                        if not ok or validate_instance(v, sc):
                            bad.append(('unsound', s, d))
            for ea in (True, False):
                try:
                    d = json.dumps(s, ensure_ascii=ea).encode('utf-8')
                except UnicodeEncodeError:
                    continue
                if not accepted(m, d):
                    bad.append(('rejected', s, d))
    assert not bad, bad[:10]


# ─────────────────────────────────────────────────────────────────────────────
# 8. Vague 2 — défauts
# ─────────────────────────────────────────────────────────────────────────────

def test_cli_main_survives_deeply_nested_accepted_document(tmp_path, capsys):
    """
    L'automate accepte une imbrication illimitée (voulu, cf. test_deep_nesting), mais
    `python3 structured.py SCHEMA DOC` relit ensuite le document accepté avec json.loads,
    qui lève RecursionError dès ~1000 niveaux : traceback non rattrapée au lieu d'un
    verdict (code de sortie 0 / 1).
    """
    doc = tmp_path / "deep.json"
    doc.write_bytes(b'[' * 1000 + b']' * 1000)
    assert accepted(_ANY, doc.read_bytes())         # l'automate, lui, l'accepte
    rc = S._main(['{}', str(doc)])
    assert rc in (0, 1)


def test_number_digit_cap_above_python_int_limit_stays_sound():
    """
    max_number_digits > sys.get_int_max_str_digits() (4300 par défaut, CPython ≥ 3.11 /
    correctifs 3.9.14+) : l'automate accepte un entier que json.loads REFUSE (ValueError
    « Exceeds the limit (4300 digits) for integer string conversion »). Un document
    accepté doit toujours être décodable (refuser le réglage, ou plafonner les chiffres
    de la partie entière quand il n'y a ni fraction ni exposant).
    """
    lim = getattr(sys, 'get_int_max_str_digits', lambda: 0)()
    if not lim:
        pytest.skip("interpréteur sans limite de conversion int ↔ str")
    try:
        m = JSONSchemaMatcher({"type": "integer"}, max_number_digits=lim + 1)
    except ValueError:
        return                                      # refus explicite du réglage : correct
    for d in (b'1' * (lim + 1), b'[' + b'9' * (lim + 1) + b']', b'-' + b'1' * (lim + 1)):
        if accepted(m, d):
            ok, _ = ref_value(d)
            assert ok, d[:20] + b'...'


def test_huge_integer_const_is_schema_error_not_bare_value_error():
    """
    {"const": 10**N} (schéma Python, N > limite int ↔ str) : la compilation appelle
    json.dumps(v), qui lève un ValueError brut de CPython (« Exceeds the limit… ») au lieu
    de SchemaError (« schéma invalide, non supporté »). Un appelant de l'API Python qui ne
    rattrape que SchemaError plante (load_schema n'est pas concerné : json.loads refuse
    déjà un tel entier dans le texte du schéma). Si le schéma compile, l'instance minimale
    doit être décodable par json.loads.
    """
    lim = getattr(sys, 'get_int_max_str_digits', lambda: 0)()
    if not lim:
        pytest.skip("interpréteur sans limite de conversion int ↔ str")
    for sch in ({"const": 10 ** lim}, {"enum": [1, -(10 ** lim)]},
                {"type": "integer", "enum": [10 ** lim]}):
        try:
            m = JSONSchemaMatcher(sch)
        except SchemaError:
            continue
        w = m.shortest_completion(m.initial_state)
        assert ref_value(w)[0]


# ─────────────────────────────────────────────────────────────────────────────
# 9. Vague 3 — oracles indépendants, espace d'états, instances générées
# ─────────────────────────────────────────────────────────────────────────────

from collections import deque      # noqa: E402


@pytest.mark.parametrize("mw,D", [(0, 1), (0, 3), (2, 2), (7, 20)])
def test_free_grammar_whitespace_and_digit_caps_matrix(mw, D):
    """Grammaire libre pour plusieurs couples (max_whitespace, max_number_digits) :
    acceptation ⇔ json.loads + plafonds documentés, sur documents générés ET mutés ;
    chaque préfixe vivant se complète en un document accepté (aucun BFS de secours)."""
    rng = random.Random(1000 * mw + D)
    m = JSONSchemaMatcher(None, max_whitespace=mw, max_number_digits=D)
    bad = []
    for _ in range(150):
        d = to_bytes(rand_ws(rng, mw + 1) + rand_doc(rng))
        for dd in [d] + [mutate(rng, d) for _ in range(5)]:
            if accepted(m, dd) != ref_ok(dd, mw, D):
                bad.append(dd)
        st = m.initial_state
        for i, b in enumerate(d):
            st = m.advance(st, b)
            if st is None:
                break
            c = m.shortest_completion(st)
            full = d[:i + 1] + (c or b'')
            if c is None or not ref_ok(full, mw, D) or not accepted(m, full):
                bad.append(('prefix', d[:i + 1], c))
                break
    assert not bad, bad[:10]
    assert m._fallbacks == 0


@pytest.mark.parametrize("seed", [21, 22])
def test_schema_mutations_never_accept_invalid_documents(seed):
    """Soundness sous mutation : documents valides (marche + complétion) mutés de 1 à 3
    octets ; tout document que l'automate accepte se relit (json.loads), respecte le
    plafond d'espaces et est valide pour le schéma."""
    rng = random.Random(seed)
    bad, n = [], 0
    while n < 250:
        sch = _rroot(rng)
        mw = rng.choice([0, 1, 4])
        try:
            m = JSONSchemaMatcher(sch, max_whitespace=mw)
        except SchemaError:
            continue
        n += 1
        for _ in range(4):
            pre, st = _walk(m, rng, maxlen=60)
            full = pre + m.shortest_completion(st)
            for dd in [full] + [mutate(rng, full) for _ in range(25)]:
                if accepted(m, dd):
                    ok, v = ref_value(dd)
                    if not ok or not lex_ok(dd, mw, 10 ** 6) or validate_instance(v, sch):
                        bad.append((sch, dd))
    assert not bad, bad[:5]


def _to_2020(s):
    """Schéma de test → draft 2020-12 équivalent pour l'oracle jsonschema : oneOf → anyOf
    (exclusivité non imposée, documenté), `items` liste → prefixItems, chaînes constantes
    sous forme canonique (_norm_str : paire de substitution en 2 unités = caractère astral,
    documenté)."""
    if isinstance(s, list):
        return [_to_2020(x) for x in s]
    if not isinstance(s, dict):
        return s
    out = {('anyOf' if k == 'oneOf' else k): v for k, v in s.items()}
    if isinstance(out.get('items'), list):
        out['prefixItems'] = out['items']
        if 'additionalItems' in out:
            out['items'] = out.pop('additionalItems')
        else:
            del out['items']
    res = {}
    for k, v in out.items():
        if k in ('const', 'enum'):
            res[k] = _norm_val(v)
        elif k == 'properties' and isinstance(v, dict):
            res[k] = {S._norm_str(n): _to_2020(x) for n, x in v.items()}
        elif k == 'required' and isinstance(v, list):
            res[k] = [S._norm_str(n) for n in v]
        else:
            res[k] = _to_2020(v)
    return res


def _norm_val(v):
    if isinstance(v, str):
        return S._norm_str(v)
    if isinstance(v, list):
        return [_norm_val(x) for x in v]
    if isinstance(v, dict):
        return {S._norm_str(k): _norm_val(x) for k, x in v.items()}
    return v


@pytest.mark.parametrize("seed", [31, 32])
def test_accepted_documents_valid_for_independent_jsonschema_oracle(seed):
    """Oracle INDÉPENDANT (bibliothèque jsonschema, draft 2020-12) : tout document accepté
    — marche + complétion, puis mutations — est valide pour le schéma (hors mots-clés non
    appliqués). validate_instance partage les conventions du compilateur ; jsonschema non."""
    jsonschema = pytest.importorskip("jsonschema")
    rng = random.Random(seed)
    bad, n = [], 0
    while n < 300:
        sch = _rroot(rng)
        try:
            m = JSONSchemaMatcher(sch, max_whitespace=rng.choice([0, 4]))
        except SchemaError:
            continue
        if m.ignored_keywords:
            continue
        n += 1
        val = jsonschema.Draft202012Validator(_to_2020(sch))
        for _ in range(5):
            pre, st = _walk(m, rng, maxlen=60)
            full = pre + m.shortest_completion(st)
            for dd in [full] + [mutate(rng, full) for _ in range(8)]:
                if not accepted(m, dd):
                    continue
                ok, v = ref_value(dd)
                assert ok, (sch, dd)
                try:
                    good = val.is_valid(v)
                except RecursionError:
                    continue        # $ref circulaire sans consommation : limite de l'oracle
                if not good:
                    bad.append((sch, dd))
    assert not bad, bad[:5]


# ── Instances générées depuis le schéma (ordre de déclaration), json.dumps ──────

_G3_CH = ['a', 'z', '/', '"', '\\', '\n', '\b', '\x00', '\x1f', '\x7f', 'é', '€', '😀', '\U0010ffff',
          '\ud800', '\udbff', '\udc00', '\udfff', ' ', ' ', '﻿', '￿', 'u', 'F', '\t',
          '\x80', 'ࠀ', '퟿', '']


def _g3_str(rng, n):
    """Chaîne de EXACTEMENT n points de code (forme canonique : paires recombinées)."""
    out = ''
    while len(out) < n:     # +1 caractère par tour (une paire recombinée ne l'allonge pas)
        out = S._norm_str(out + rng.choice(_G3_CH))
    return out


def _g3_num(rng, integer):
    if integer or rng.random() < 0.3:
        k = rng.randint(1, 20)
        v = rng.randint(10 ** (k - 1) if k > 1 else 0, 10 ** k - 1)
        return -v if rng.random() < 0.3 else v
    return rng.choice([0.0, -0.0, 1.5, 1e-7, 1e20, 1e-300, 5e-324, 1.7976931348623157e308,
                       0.1 + 0.2, 123456.789, -2.5e-5, 1e16, 1e15 + 0.5, float(10 ** 15)])


def _g3_val(rng, d=0):
    r = rng.random()
    if d > 2 or r < 0.5:
        return rng.choice([None, True, False, 0, -1, 1.5, -0.0, 1e20,
                           _g3_str(rng, rng.randint(0, 3)), _g3_num(rng, False)])
    if r < 0.75:
        return [_g3_val(rng, d + 1) for _ in range(rng.randint(0, 3))]
    return {_g3_str(rng, rng.randint(0, 2)): _g3_val(rng, d + 1) for _ in range(rng.randint(0, 3))}


def _g3_sch(rng, d=0, defs=()):
    """Schémas SANS fusion (le générateur d'instances ci-dessous les suit exactement)."""
    r = rng.random() * (0.45 if d > 3 else 1.0)
    if r < 0.05:
        return {}
    if r < 0.15:
        return {"type": rng.choice(["integer", "number", "boolean", "null"])}
    if r < 0.25:
        s = {"type": "string"}
        if rng.random() < 0.6:
            s["minLength"] = rng.randint(0, 3)
        if rng.random() < 0.6:
            s["maxLength"] = s.get("minLength", 0) + rng.randint(0, 3)
        return s
    if r < 0.33:
        return {"enum": [_g3_val(rng) for _ in range(rng.randint(1, 4))]}
    if r < 0.38:
        return {"const": _g3_val(rng)}
    if r < 0.55:
        props = {_g3_str(rng, rng.randint(0, 3)): _g3_sch(rng, d + 1, defs)
                 for _ in range(rng.randint(0, 4))}
        s = {"type": "object", "properties": props,
             "required": [k for k in props if rng.random() < 0.5]}
        if rng.random() < 0.2:
            s["required"].append(_g3_str(rng, 2))
            if rng.random() < 0.5:
                s["additionalProperties"] = _g3_sch(rng, d + 1, defs)
        return s
    if r < 0.62:
        return {"type": "object", "additionalProperties": _g3_sch(rng, d + 1, defs)}
    if r < 0.78:
        s = {"type": "array"}
        if rng.random() < 0.4:
            s["prefixItems"] = [_g3_sch(rng, d + 1, defs) for _ in range(rng.randint(1, 3))]
            if rng.random() < 0.4:
                s["items"] = False if rng.random() < 0.5 else _g3_sch(rng, d + 1, defs)
        else:
            s["items"] = _g3_sch(rng, d + 1, defs)
        if rng.random() < 0.4:
            s["minItems"] = rng.randint(0, 2)
        if rng.random() < 0.4:
            s["maxItems"] = s.get("minItems", 0) + rng.randint(0, 3)
        return s
    if r < 0.9:
        return {"anyOf": [_g3_sch(rng, d + 1, defs) for _ in range(rng.randint(1, 3))]}
    if defs:
        return {"$ref": "#/$defs/" + rng.choice(defs)}
    return {"type": rng.sample(["null", "boolean", "integer", "number", "string"], 2)}


def _g3_inst(root, s, rng, d=0):
    """Instance valide de `s` : propriétés dans l'ordre de déclaration, aucune additionnelle
    quand `properties` est présent, membre d'enum / const sous SA forme."""
    if d > 12:
        raise RecursionError
    if s is None or s is True or not (set(s) - {'$defs'}):
        return _g3_val(rng)
    if '$ref' in s:
        return _g3_inst(root, S._resolve_ref(root, s['$ref']), rng, d + 1)
    if 'anyOf' in s:
        alts = list(s['anyOf'])
        rng.shuffle(alts)
        for a in alts:
            try:
                return _g3_inst(root, a, rng, d + 1)
            except (RecursionError, ValueError):
                continue
        raise ValueError
    if 'enum' in s:
        return rng.choice(s['enum'])
    if 'const' in s:
        return s['const']
    t = s['type']
    if isinstance(t, list):
        t = rng.choice(t)
    if t in ('null', 'boolean'):
        return None if t == 'null' else rng.random() < 0.5
    if t in ('integer', 'number'):
        return _g3_num(rng, t == 'integer')
    if t == 'string':
        mn = s.get('minLength', 0)
        return _g3_str(rng, rng.randint(mn, s.get('maxLength', mn + 4)))
    if t == 'object':
        addl = s.get('additionalProperties', True)
        if 'properties' not in s and 'required' not in s:
            return {_g3_str(rng, rng.randint(0, 3)): _g3_inst(root, addl, rng, d + 1)
                    for _ in range(rng.randint(0, 3))}
        props, req, out = s.get('properties', {}), s.get('required', []), {}
        for k, sub in props.items():
            if k in req or rng.random() < 0.5:
                out[k] = _g3_inst(root, sub, rng, d + 1)
        for k in req:
            if k not in props:
                out[k] = _g3_inst(root, addl, rng, d + 1)
        return out
    pre, items = s.get('prefixItems', []), s.get('items', True)
    mn, mx = s.get('minItems', 0), s.get('maxItems', s.get('minItems', 0) + 3)
    if items is False:
        mx = min(mx, len(pre))
    if mn > mx:
        raise ValueError
    return [_g3_inst(root, pre[i] if i < len(pre) else items, rng, d + 1)
            for i in range(rng.randint(mn, mx))]


@pytest.mark.parametrize("seed", [41, 42])
def test_generated_instances_json_dumps_prefix_closed(seed):
    """Complétude indépendante de l'automate : instances construites DIRECTEMENT depuis le
    schéma (chaînes à surrogates isolés / astrales / contrôles, flottants extrêmes, entiers
    ≤ 20 chiffres, membres d'enum objets / tableaux), sérialisées par json.dumps (compact,
    défaut, ensure_ascii=False, indent=2, indent='\\t') : chaque préfixe reste vivant et le
    document complet est accepté."""
    rng = random.Random(seed)
    bad, checked, n = [], 0, 0
    while n < 500:
        names = tuple(f"d{i}" for i in range(rng.randint(0, 2)))
        sch = dict(_g3_sch(rng, 0, names))
        if names:
            sch["$defs"] = {k: _g3_sch(rng, 1, names) for k in names}
        try:
            m = JSONSchemaMatcher(sch, max_whitespace=64)
        except SchemaError:
            continue
        n += 1
        for _ in range(8):
            try:
                v = _g3_inst(sch, sch, rng)
            except (RecursionError, ValueError):
                continue
            assert validate_instance(v, sch) == [], (sch, v)
            for kw in (dict(separators=(',', ':')), dict(), dict(ensure_ascii=False),
                       dict(indent=2), dict(indent='\t', ensure_ascii=False)):
                try:
                    data = json.dumps(v, **kw).encode('utf-8')
                except UnicodeEncodeError:
                    continue            # surrogate isolé : pas d'UTF-8 brut possible
                checked += 1
                st = m.initial_state
                for i, b in enumerate(data):
                    st = m.advance(st, b)
                    if st is None:
                        bad.append((sch, data, i))
                        break
                else:
                    if not m.is_accepting(st):
                        bad.append((sch, data, 'incomplete'))
    assert checked > 5000
    assert not bad, bad[:5]


# ── Exploration exhaustive (BFS) de l'espace d'états ─────────────────────────

_BFS_SCHEMAS = [
    None,
    {"type": "string", "minLength": 2, "maxLength": 2},
    {"type": "string", "maxLength": 1},
    {"type": "string", "minLength": 3},
    {"enum": ["\ud83d", "😀", "😀x", "é\x7f/", "\"\\"]},
    {"const": {"\ud800": ["\udfff", 1e-7, -0.0], "a/b": {"\x00": None}}},
    {"type": "array", "prefixItems": [{"type": "integer"}, {"const": 10}],
     "items": {"type": "number"}, "minItems": 1, "maxItems": 3},
    {"type": "object", "properties": {"a": {"type": "integer"}, "ab": {"enum": [1, 12, 1.5]},
                                      "b": {"type": "string", "maxLength": 1}},
     "required": ["b"]},
    {"anyOf": [{"type": "integer"}, {"enum": [1.5, 10, -0.0, 1e+20]},
               {"type": "string", "minLength": 1}]},
    {"type": "object",
     "additionalProperties": {"type": "array", "items": {"type": "integer"}, "maxItems": 2}},
    {"$defs": {"t": {"type": "array", "items": {"$ref": "#/$defs/t"}, "maxItems": 2}},
     "$ref": "#/$defs/t"},
]


@pytest.mark.parametrize("idx", range(len(_BFS_SCHEMAS)))
def test_state_space_bfs_no_dead_state_and_exact_api(idx):
    """BFS sur TOUS les octets (0..255) depuis l'état initial : chaque état atteint a une
    complétion acceptée, décodable et valide ; un état acceptant est un document valide ;
    can_continue ⇔ allowed_bytes non vide ; is_accepting ⇔ complétion vide ;
    allowed_bytes == {b : advance(b) non None}. Les plafonds de chiffres ne s'appliquent
    qu'à la grammaire number / integer (pas aux littéraux d'enum / const)."""
    sch = _BFS_SCHEMAS[idx]
    for mw, D in ((0, 2), (1, 1), (2, 20)):
        m = JSONSchemaMatcher(sch, max_whitespace=mw, max_number_digits=D)
        seen = {m.initial_state: b''}
        q = deque([m.initial_state])
        bad = []
        while q and len(seen) < 20000:
            st = q.popleft()
            pre = seen[st]
            c = m.shortest_completion(st)
            if c is None:
                bad.append(('dead', pre))
                continue
            for doc in ((pre + c,) + ((pre,) if m.is_accepting(st) else ())):
                ok, v = ref_value(doc)
                if (not ok or not lex_ok(doc, mw, 10 ** 6) or not accepted(m, doc)
                        or (sch is not None and validate_instance(v, sch))):
                    bad.append(('invalid', pre, doc))
            al = m.allowed_bytes(st)
            if m.can_continue(st) != bool(al) or m.is_accepting(st) != (c == b''):
                bad.append(('api', pre))
            if al != tuple(b for b in range(256) if m.advance(st, b) is not None):
                bad.append(('allowed', pre))
            if len(pre) >= 14:
                continue
            for b in al:
                ns = m.advance(st, b)
                if ns not in seen:
                    seen[ns] = pre + bytes((b,))
                    q.append(ns)
        assert not bad, (sch, mw, D, bad[:5])
        assert m._fallbacks == 0


def _g3_all_encodings(ch):
    cp = ord(ch)
    out = []
    short = {'"': '\\"', '\\': '\\\\', '/': '\\/', '\b': '\\b', '\f': '\\f', '\n': '\\n',
             '\r': '\\r', '\t': '\\t'}
    if ch in short:
        out.append(short[ch])
    if cp >= 0x20 and not 0xD800 <= cp <= 0xDFFF and ch not in '"\\':
        out.append(ch)
    if cp < 0x10000:
        h = '%04x' % cp
        out += ['\\u' + x for x in {h, h.upper(), h.capitalize()}]
    else:
        v = cp - 0x10000
        hi, lo = 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF)
        out += ['\\u%04x\\u%04x' % (hi, lo), '\\u%04X\\u%04X' % (hi, lo)]
    return out


def _g3_canonical(ch, e):
    """Règle « littéral seulement » du docstring (Limites connues)."""
    cp = ord(ch)
    if ch in '"\\':
        return e == '\\' + ch
    if ch in '\b\f\n\r\t':
        return e == {'\b': '\\b', '\f': '\\f', '\n': '\\n', '\r': '\\r', '\t': '\\t'}[ch]
    if cp < 0x20 or 0xD800 <= cp <= 0xDFFF:
        return e.lower() == '\\u%04x' % cp
    if cp < 0x7F:
        return e == ch
    return True


def test_constant_rule_every_ascii_code_point_and_unicode_boundaries():
    """Les 128 points de code ASCII + bornes Unicode (U+0080, U+07FF/U+0800, U+D7FF,
    surrogates aux bornes, U+E000, U+FFFF, U+10000, U+10FFFF) dans une valeur const ET
    dans un nom requis : ensemble exact des encodages acceptés selon la règle, relecture
    valide, et json.dumps(ensure_ascii=True/False) toujours accepté."""
    chars = [chr(c) for c in range(0x80)] + [
        '\x80', '\xa0', '\xff', '߿', 'ࠀ', '퟿', '\ud800', '\udbff', '\udc00',
        '\udfff', '', '�', '￿', '\U00010000', '\U0010ffff', ' ']
    bad = []
    for ch in chars:
        for sch, wrap, inst in (({"const": "q" + ch}, lambda b: '"q' + b + '"', "q" + ch),
                                ({"required": [ch + "q"]}, lambda b: '{"' + b + 'q":0}',
                                 {ch + "q": 0})):
            m = JSONSchemaMatcher(sch)
            for e in _g3_all_encodings(ch):
                d = to_bytes(wrap(e))
                a = accepted(m, d)
                if a != _g3_canonical(ch, e):
                    bad.append((repr(ch), e, a))
                if a:
                    ok, v = ref_value(d)
                    if not ok or validate_instance(v, sch):
                        bad.append(('unsound', repr(ch), e))
            for ea in (True, False):
                try:
                    d = json.dumps(inst, ensure_ascii=ea, separators=(',', ':')).encode('utf-8')
                except UnicodeEncodeError:
                    continue
                if not accepted(m, d):
                    bad.append(('dumps-rejected', repr(ch), ea, d))
    assert not bad, bad[:10]


def test_very_deep_nesting_is_linear_and_closes():
    """100 000 niveaux : avance, complétion exacte (un ']' / '}' par niveau), acceptation,
    sans RecursionError (piles chaînées, comparaisons et fins itératives)."""
    depth = 100_000
    m = JSONSchemaMatcher(None)
    st = m.advance_bytes(m.initial_state, b'[{"k":' * (depth // 2))
    c = m.shortest_completion(st)
    assert c == b'0' + b'}]' * (depth // 2)
    assert m.is_accepting(m.advance_bytes(st, c))
    assert m.advance_bytes(st, b'0' + b'}]' * (depth // 2) + b']') is None
    assert m._fallbacks == 0
