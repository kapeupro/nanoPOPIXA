"""
Tests — structured outputs (décodage contraint JSON / JSON Schema).

  1. Acceptation   : instances valides aléatoires (graine fixe), 4 sérialisations
  2. Rejet         : documents invalides écrits à la main
     2 bis.          chaînes constantes (clés déclarées, enum / const) : encodage canonique
                     (ASCII brut seulement, "\\u006Eom" refusé ; non-ASCII brut ou \\uXXXX)
  3. Marche aléatoire (test clé) : préfixe aléatoire autorisé + shortest_completion
                     → json.loads OK et validate_instance == []
  4. Masque exact  : allowed_mask() == [is_allowed(t) for t]
  5. Complétion    : minimalité vs BFS brute-force ; completion_tokens() → complet
  6. Performance   : vocabulaire gpt2 (tiktoken)
  7. load_schema / SchemaError
"""

import copy
import json
import random
import time

import pytest
import torch

import structured as S
from structured import (
    JSONSchemaMatcher, SchemaError, TokenConstraint, json_constraint, load_schema,
    token_bytes_from_itos, token_bytes_from_tiktoken, validate_instance,
)

try:
    import tiktoken
except ImportError:     # pragma: no cover
    tiktoken = None


# ─────────────────────────────────────────────────────────────────────────────
# Schémas de test
# ─────────────────────────────────────────────────────────────────────────────

PERSON = {
    "title": "Personne",
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 20},
        "age": {"type": "integer", "minimum": 0},
        "email": {"type": "string", "format": "email"},
        "score": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "string", "maxLength": 8}, "maxItems": 3},
        "address": {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"},
                "zip": {"type": ["string", "null"]},
            },
            "required": ["street", "city"],
            "additionalProperties": False,
        },
        "active": {"type": "boolean"},
    },
    "required": ["name", "age", "active"],
    "additionalProperties": False,
}

ARRAY_MINMAX = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 4}

ENUM_MIXED = {"enum": [
    "rouge", "vert é", "😀", "", "quote\"back\\slash/\n\t", 42, -3.5, 1e21, 0,
    True, False, None, ["a", 1, None, []], {"k": "v", "n": [1, 2], "é": {}},
]}

UNION = {"type": ["string", "integer", "null"], "maxLength": 3}

SHAPES = {
    "description": "oneOf traité comme anyOf",
    "oneOf": [
        {"type": "object",
         "properties": {"kind": {"const": "circle"}, "r": {"type": "number"}},
         "required": ["kind", "r"], "additionalProperties": False},
        {"type": "object",
         "properties": {"kind": {"const": "rect"}, "w": {"type": "number"},
                        "h": {"type": "number"}, "label": {"type": "string"}},
         "required": ["kind", "w", "h"], "additionalProperties": False},
    ],
}

TREE = {
    "type": "object",
    "properties": {
        "value": {"type": "integer"},
        "children": {"type": "array", "items": {"$ref": "#"}, "maxItems": 3},
    },
    "required": ["value", "children"],
}

EXPR = {
    "$defs": {
        "expr": {"anyOf": [{"type": "number"}, {"$ref": "#/$defs/op"}]},
        "op": {
            "type": "object",
            "properties": {
                "op": {"enum": ["+", "-", "*"]},
                "args": {"type": "array", "items": {"$ref": "#/$defs/expr"},
                         "minItems": 1, "maxItems": 3},
            },
            "required": ["op", "args"],
        },
    },
    "$ref": "#/$defs/op",
}

STRINGS = {
    "type": "object",
    "properties": {
        "short": {"type": "string", "maxLength": 2},
        "exact": {"type": "string", "minLength": 3, "maxLength": 3},
        "long": {"type": "string", "minLength": 5},
        "any": {"type": "string"},
    },
    "required": ["short", "exact", "long", "any"],
}

OPTIONAL = {
    "type": "object",
    "properties": {
        "a": {"type": "integer"}, "b": {"type": "string"}, "c": {"type": "boolean"},
        "d": {"type": "null"}, "e": {"enum": ["x", "y"]},
    },
    "required": ["c"],
}

FREE = {"type": "object", "additionalProperties": {"type": "integer"}}

ANY = {}

TUPLE = {
    "type": "array",
    "prefixItems": [{"type": "string"}, {"type": "integer"}],
    "items": {"type": "boolean"},
    "minItems": 1, "maxItems": 5,
}

UNICODE_KEYS = {
    "type": "object",
    "properties": {
        "prénom": {"type": "string"},
        "âge": {"type": "integer"},
        "clé/\"x\"\\": {"type": "boolean"},
        "emoji😀": {"type": "null"},
    },
    "required": ["prénom", "âge", "clé/\"x\"\\", "emoji😀"],
}

ROOT_STRING = {"type": "string", "minLength": 2, "maxLength": 4}
ROOT_NUMBER = {"type": "number"}

SCHEMAS = {
    "person": PERSON, "array_minmax": ARRAY_MINMAX, "enum_mixed": ENUM_MIXED,
    "union": UNION, "shapes": SHAPES, "tree": TREE, "expr": EXPR, "strings": STRINGS,
    "optional": OPTIONAL, "free": FREE, "any": ANY, "tuple": TUPLE,
    "unicode_keys": UNICODE_KEYS, "root_string": ROOT_STRING, "root_number": ROOT_NUMBER,
}

_MATCHERS = {}


def matcher(name):
    """Matcher partagé par schéma (caches chauds entre tests)."""
    if name not in _MATCHERS:
        _MATCHERS[name] = JSONSchemaMatcher(SCHEMAS[name])
    return _MATCHERS[name]


# ─────────────────────────────────────────────────────────────────────────────
# Générateur d'instances valides (graine fixe)
# ─────────────────────────────────────────────────────────────────────────────

ALPHABET = ['a', 'b', 'Z', '0', ' ', '"', '\\', '/', '\n', '\t', '\b', '\f', '\r',
            '\x00', '\x1f', '\x7f', 'é', 'à', 'ü', '€', '中', '😀', '🎉', ' ',
            '퟿', '', '￿', '\U0010ffff']
MAX_DEPTH = 3


def gen_string(rng, mn=0, mx=None):
    hi = mx if mx is not None else mn + 6
    n = rng.randint(mn, max(mn, hi))
    return ''.join(rng.choice(ALPHABET) for _ in range(n))


def gen_number(rng, integer):
    if integer:
        return rng.choice([0, -1, 7, rng.randint(-10 ** 6, 10 ** 6), 10 ** 15, -(10 ** 12)])
    return rng.choice([0, 1.5, -0.0, -2.25e-7, 3e20, 123456789, 0.1,
                       rng.uniform(-1e6, 1e6), rng.randint(-50, 50)])


def gen_any(rng, depth):
    kinds = ['string', 'number', 'integer', 'boolean', 'null']
    if depth < MAX_DEPTH:
        kinds += ['object', 'array']
    k = rng.choice(kinds)
    if k == 'object':
        return {gen_string(rng, 0, 4): gen_any(rng, depth + 1) for _ in range(rng.randint(0, 3))}
    if k == 'array':
        return [gen_any(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    return gen_typed({}, k, rng, None, depth)


def gen(schema, rng, root, depth=0):
    if schema is True or schema is None or schema == {}:
        return gen_any(rng, depth)
    if '$ref' in schema:
        return gen(S._resolve_ref(root, schema['$ref']), rng, root, depth)
    for key in ('anyOf', 'oneOf'):
        if key in schema:
            alts = schema[key]
            return gen(alts[0] if depth >= MAX_DEPTH else rng.choice(alts), rng, root, depth)
    if 'const' in schema:
        return copy.deepcopy(schema['const'])
    if 'enum' in schema:
        return copy.deepcopy(rng.choice(schema['enum']))
    t = schema.get('type')
    if isinstance(t, list):
        t = rng.choice(t)
    if t is None:
        t = 'object' if 'properties' in schema else 'array' if 'items' in schema else None
    if t is None:
        return gen_any(rng, depth)
    return gen_typed(schema, t, rng, root, depth)


def gen_typed(schema, t, rng, root, depth):
    if t == 'string':
        return gen_string(rng, schema.get('minLength', 0), schema.get('maxLength'))
    if t in ('number', 'integer'):
        return gen_number(rng, t == 'integer')
    if t == 'boolean':
        return rng.random() < 0.5
    if t == 'null':
        return None
    if t == 'object':
        props = schema.get('properties')
        req = schema.get('required', [])
        if props is None:
            addl = schema.get('additionalProperties', True)
            if addl is False:
                return {}
            n = rng.randint(0, 3) if depth < MAX_DEPTH else 0
            return {gen_string(rng, 0, 4): gen(addl, rng, root, depth + 1) for _ in range(n)}
        out = {}
        for name, sub in props.items():     # ordre de déclaration
            if name in req or (depth < MAX_DEPTH and rng.random() < 0.6):
                out[name] = gen(sub, rng, root, depth + 1)
        return out
    if t == 'array':
        prefix = schema.get('prefixItems', [])
        items = schema.get('items', True)
        mn, mx = schema.get('minItems', 0), schema.get('maxItems')
        hi = mn if depth >= MAX_DEPTH else (mx if mx is not None else mn + 3)
        hi = min(hi, mn + 3)
        n = rng.randint(mn, max(mn, hi))
        return [gen(prefix[i] if i < len(prefix) else items, rng, root, depth + 1)
                for i in range(n)]
    raise AssertionError(t)


def serializations(inst):
    out = []
    for sep in ((',', ':'), None):
        for ea in (True, False):
            out.append(json.dumps(inst, ensure_ascii=ea, separators=sep).encode('utf-8'))
    return out


def run(m, data):
    st = m.initial_state
    for b in data:
        st = m.advance(st, b)
        if st is None:
            return None
    return st


def accepted(m, data):
    st = run(m, data)
    return st is not None and m.is_accepting(st)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Acceptation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(SCHEMAS))
def test_acceptance_random_valid_instances(name):
    schema = SCHEMAS[name]
    m = matcher(name)
    rng = random.Random(1234 + len(name))
    for _ in range(40):
        inst = gen(schema, rng, schema)
        assert validate_instance(inst, schema) == [], (inst, validate_instance(inst, schema))
        for data in serializations(inst):
            st = run(m, data)
            assert st is not None, (name, data)
            assert m.is_accepting(st), (name, data)
            if isinstance(inst, (dict, list, str)):
                assert not m.can_continue(st), (name, data)    # racine fermée → terminale


def test_acceptance_leading_whitespace_and_indent():
    m = matcher("person")
    doc = {"name": "Zoé", "age": 3, "active": True}
    assert accepted(m, b'    ' + json.dumps(doc).encode())
    assert not accepted(m, b'     ' + json.dumps(doc).encode())       # 5 > max_whitespace
    deep = JSONSchemaMatcher(PERSON, max_whitespace=16)
    inst = {"name": "a", "age": 1, "tags": ["x", "y"],
            "address": {"street": "s", "city": "c", "zip": None}, "active": False}
    assert accepted(deep, json.dumps(inst, indent=2).encode())
    assert not accepted(m, json.dumps(inst, indent=4).encode())        # retraits > 4


def test_states_are_hashable_and_canonical():
    m = JSONSchemaMatcher({"type": "string"})
    s0 = m.advance(m.initial_state, ord('"'))
    after = [m.advance_bytes(s0, x) for x in (b'a', b'Z', b' ', 'é'.encode(), '😀'.encode(),
                                               b'\\n', b'\\u00e9', b'abc')]
    assert all(isinstance(s, frozenset) for s in after)
    assert len({hash(s) for s in after}) == 1 and all(s == after[0] for s in after)
    assert after[0] == s0           # chaîne libre : l'état ne dépend pas du contenu
    assert m.advance(s0, ord('a')) is m.advance(s0, ord('a'))       # mémoïsé


def test_max_whitespace_zero():
    m = JSONSchemaMatcher(None, max_whitespace=0)
    assert accepted(m, b'{"a":[1,{}]}')
    for bad in (b' {}', b'{ }', b'[1, 2]', b'{"a" :1}'):
        assert not accepted(m, bad)
    with pytest.raises(ValueError):
        JSONSchemaMatcher(None, max_whitespace=-1)


def test_terminal_and_number_root():
    m = matcher("root_number")
    st = run(m, b'12')
    assert m.is_accepting(st) and m.can_continue(st)
    assert run(m, b'12 ') is None                # pas d'espace final
    mi = JSONSchemaMatcher({"type": "integer"}, max_number_digits=5)
    assert accepted(mi, b'12345') and not accepted(mi, b'123456')
    st = run(mi, b'12345')
    assert mi.is_accepting(st) and not mi.can_continue(st)
    st = run(mi, b'-0')
    assert mi.is_accepting(st) and not mi.can_continue(st)
    assert run(mi, b'1.0') is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rejet
# ─────────────────────────────────────────────────────────────────────────────

REJECT = [
    ("person", b'{"name":"Al","age":3,"active":true,}'),          # virgule finale
    ("person", b'{"name":"Al" "age":3,"active":true}'),           # virgule manquante
    ("person", b'{"name":5,"age":3,"active":true}'),              # mauvais type
    ("person", b'{"name":"Al","active":true}'),                    # requis manquant
    ("person", b'{"name":"Al","age":3,"active":true,"zzz":1}'),   # propriété en trop
    ("person", b'{"name":"Al","age":3,"active":true} '),          # espace final
    ("person", b'{"name":"Al","age":3,"active":true}}'),          # accolade en trop
    ("person", b'{"name":"A\nl","age":3,"active":true}'),         # saut de ligne brut
    ("person", b'{"name":"A\tl","age":3,"active":true}'),         # tabulation brute
    ("person", b'{"name":"\xc3","age":3,"active":true}'),         # UTF-8 tronqué
    ("person", b'{"name":"\xff","age":3,"active":true}'),         # octet invalide
    ("person", b'{"name":"\xc0\xaf","age":3,"active":true}'),     # UTF-8 surlong
    ("person", b'{"name":"\xed\xa0\x80","age":3,"active":true}'),  # surrogate UTF-8
    ("person", b'{"name":"\xf4\x90\x80\x80","age":3,"active":true}'),  # > U+10FFFF
    ("person", b'{"name":"A","age":012,"active":true}'),          # zéros en tête
    ("person", b'{"name":"A","age":1.5,"active":true}'),          # integer
    ("person", b"{'name':'A','age':1,'active':true}"),            # apostrophes
    ("person", b'{"name":"\\x","age":1,"active":true}'),          # échappement invalide
    ("person", b'{"name":"\\u12G4","age":1,"active":true}'),      # \u invalide
    ("person", b'{"name":"","age":1,"active":true}'),             # minLength 1
    ("person", b'{"name":"' + b'a' * 21 + b'","age":1,"active":true}'),   # maxLength 20
    ("person", b'{"name":"A","age":1,"tags":["a","b","c","d"],"active":true}'),  # maxItems
    ("person", b'{"name":"A","age":1,"tags":["123456789"],"active":true}'),      # maxLength 8
    ("person", b'{"age":1,"name":"A","active":true}'),            # ordre de déclaration
    ("person", b'{"name":"A","age":1,"active":tru}'),
    ("person", b'{"name":"A","age":1,"active":True}'),
    ("person", b'{"name":"A","age":1,"address":{"street":"s"},"active":true}'),
    ("person", b''),
    ("person", b'   '),
    ("person", b'[]'),
    ("array_minmax", b'[1]'),
    ("array_minmax", b'[1,2,3,4,5]'),
    ("array_minmax", b'[1,,2]'),
    ("array_minmax", b'[1,2,]'),
    ("array_minmax", b'[01,2]'),
    ("array_minmax", b'[1.,2]'),
    ("array_minmax", b'[.5,2]'),
    ("array_minmax", b'[1e,2]'),
    ("array_minmax", b'[+1,2]'),
    ("array_minmax", b'[1,2'),
    ("enum_mixed", b'"bleu"'),
    ("enum_mixed", b'42.0'),
    ("enum_mixed", b'1'),
    ("enum_mixed", b'"Rouge"'),
    ("enum_mixed", b'["a",1,null]'),
    ("enum_mixed", b'{"n":[1,2],"k":"v","\\u00e9":{}}'),       # ordre des clés de la constante
    ("union", b'"abcd"'),
    ("union", b'1.5'),
    ("union", b'true'),
    ("shapes", b'{"kind":"circle","w":1,"h":2}'),
    ("shapes", b'{"kind":"square","r":1}'),
    ("tree", b'{"value":1}'),
    ("tree", b'{"value":1,"children":[{"value":2}]}'),
    ("tree", b'{"value":1,"children":[{},{},{},{}]}'),
    ("expr", b'{"op":"/","args":[1]}'),
    ("expr", b'{"op":"+","args":[]}'),
    ("strings", b'{"short":"abc","exact":"abc","long":"abcde","any":""}'),
    ("strings", b'{"short":"","exact":"ab","long":"abcde","any":""}'),
    ("strings", b'{"short":"","exact":"abc","long":"abcd","any":""}'),
    ("strings", b'{"short":"\\ud83d\\ude00\\ud83d\\ude00\\ud83d\\ude00","exact":"abc","long":"abcde","any":""}'),
    ("optional", b'{"a":1,"b":"x"}'),
    ("optional", b'{"c":true,"a":1}'),
    ("optional", b'{"c":true,"e":"z"}'),
    ("free", b'{"a":"x"}'),
    ("free", b'{"a":1,}'),
    ("free", b'{"a" 1}'),
    ("any", b'     {}'),           # 5 espaces > max_whitespace
    ("any", b'{"a":1} '),
    ("any", b'nul'),
    ("any", b'[1 2]'),
    ("any", b'"\x01"'),
    ("any", b'NaN'),
    ("tuple", b'[]'),
    ("tuple", b'[1]'),
    ("tuple", b'["a","b"]'),
    ("tuple", b'["a",1,2]'),
    ("tuple", b'["a",1,true,true,true,true]'),
    ("unicode_keys", b'{"prenom":"x","\xc3\xa2ge":1,"cl\xc3\xa9/\\"x\\"\\\\":true,"emoji\xf0\x9f\x98\x80":null}'),
    ("root_string", b'"a"'),
    ("root_string", b'"abcde"'),
    ("root_string", b'"\\ud83d\\ude00"'),                       # 1 point de code
    ("root_string", b'"ab\\ud83d\\ude00\\ud83d\\ude00\\ud83d\\ude00"'),
    ("root_string", b'"ab" '),
    ("root_number", b'-'),
    ("root_number", b'1.5e+'),
    ("root_number", b'1' * 21),     # max_number_digits = 20
]


@pytest.mark.parametrize("name,data", REJECT)
def test_rejection(name, data):
    assert not accepted(matcher(name), data), data


def test_rejection_extra_cases_free_and_false():
    m = JSONSchemaMatcher({"type": "object", "additionalProperties": False})
    assert accepted(m, b'{}') and accepted(m, b'{ }')
    assert not accepted(m, b'{"a":1}')
    m = JSONSchemaMatcher({"type": "object", "properties": {}})
    assert accepted(m, b'{}') and not accepted(m, b'{"a":1}')


def test_surrogate_pairs_counted_as_one_code_point():
    m = matcher("root_string")      # 2..4 points de code
    assert accepted(m, b'"a\\ud83d\\ude00"')
    assert accepted(m, b'"\\ud83d\\ude00\\ud83d\\ude00\\ud83d\\ude00\\uD83D\\uDE00"')
    assert accepted(m, '"😀😀😀😀"'.encode())
    assert not accepted(m, '"😀😀😀😀😀"'.encode())
    assert accepted(m, b'"\\ud83d\\ud83d"')                     # 2 hauts isolés = 2
    assert accepted(m, b'"abc\\ud83d"')
    assert accepted(m, b'"abc\\ud83d\\ude00"')                 # 4 avec la paire
    assert not accepted(m, b'"abc\\ud83d\\u0041"')             # 5


# ─────────────────────────────────────────────────────────────────────────────
# 2 bis. Chaînes constantes : encodage canonique (clés déclarées, enum / const)
# ─────────────────────────────────────────────────────────────────────────────

NOM = {"type": "object",
       "properties": {"nom": {"type": "string"}, "né": {"type": "integer"},
                      "a/b": {"type": "null"}},
       "required": ["nom"]}

# Constante couvrant toutes les catégories : (canonique, variantes acceptées, refusées)
CANON_PARTS = [
    (b'q',        [],                                  [b'\\u0071', b'\\x71', b'Q']),
    (b'\\"',      [],                                  [b'\\u0022', b'"']),
    (b'\\\\',     [],                                  [b'\\u005c', b'\\u005C', b'\\']),
    (b'/',        [],                                  [b'\\/', b'\\u002f', b'\\u002F']),
    (b' ',        [],                                  [b'\\u0020']),
    (b'~',        [],                                  [b'\\u007e', b'\\u007E']),
    (b'\\n',      [],                                  [b'\\u000a', b'\\u000A', b'\n']),
    (b'\\t',      [],                                  [b'\\u0009', b'\t']),
    (b'\\u0001',  [],                                  [b'\x01', b'\\x01']),
    (b'\\u001f',  [b'\\u001F'],                        [b'\x1f']),
    (b'\x7f',     [b'\\u007f', b'\\u007F'],            []),
    ('é'.encode(), [b'\\u00e9', b'\\u00E9'],           [b'\\u00e', b'\\u0065\\u0301']),
    ('😀'.encode(), [b'\\ud83d\\ude00', b'\\uD83D\\uDE00', b'\\uD83D\\ude00'],
                                                       [b'\\ud83d', b'\\ud83d\\u0041']),
]
CANON_VALUE = 'q"\\/ ~\n\t\x01\x1f\x7fé😀'


def canon_doc(key_parts, val_parts):
    return b'{"' + b''.join(key_parts) + b'":"' + b''.join(val_parts) + b'"}'


def test_constant_keys_reject_escaped_ascii():
    """Le cas observé en génération réelle : "\\u006Eom" au lieu de "nom"."""
    m = JSONSchemaMatcher(NOM)
    assert accepted(m, b'{"nom":"x"}')
    for bad in (b'{"\\u006Eom":"x"}', b'{"\\u006eom":"x"}', b'{"n\\u006fm":"x"}',
                b'{"no\\u006D":"x"}', b'{"nom":"x","\\u006e\\u00e9":1}',
                b'{"nom":"x","a\\/b":null}', b'{"nom":"x","a\\u002fb":null}'):
        assert run(m, bad) is None, bad
        assert validate_instance(json.loads(bad), NOM) == []    # sémantique inchangée
    # Refus dès l'octet fautif : '\\' ne peut pas commencer un caractère ASCII de clé
    assert m.advance(run(m, b'{"'), ord('\\')) is None
    assert m.advance(run(m, b'{"nom":"x","n'), ord('\\')) is not None     # é → é
    # Valeur libre : grammaire JSON complète (\uXXXX d'un ASCII et \/ toujours permis)
    assert accepted(m, b'{"nom":"\\u006Eom\\/"}')


def test_constant_keys_non_ascii_escapes_accepted():
    m = JSONSchemaMatcher(NOM)
    for good in (b'{"nom":"x","n\\u00e9":1}', b'{"nom":"x","n\\u00E9":1}',
                 '{"nom":"x","né":1}'.encode(), b'{"nom":"x","a/b":null}',
                 b'{"nom":"x","n\\u00e9":1,"a/b":null}'):
        assert accepted(m, good), good
        assert validate_instance(json.loads(good), NOM) == []
    m = matcher("unicode_keys")
    assert accepted(m, b'{"pr\\u00e9nom":"x","\\u00e2ge":1,"cl\\u00E9/\\"x\\"\\\\":true,'
                       b'"emoji\\ud83d\\ude00":null}')
    assert not accepted(m, b'{"pr\\u00e9nom":"x","\\u00e2ge":1,"cl\\u00E9\\/\\"x\\"\\\\":true,'
                           b'"emoji\\ud83d\\ude00":null}')               # \/ refusé


def test_constant_strings_canonical_escapes():
    """Chaque caractère d'une constante : seules les formes canoniques sont acceptées."""
    canon = [p[0] for p in CANON_PARTS]
    assert json.loads(b'"' + b''.join(canon) + b'"') == CANON_VALUE
    schema = {"type": "object", "properties": {CANON_VALUE: {"enum": [CANON_VALUE, 1]}},
              "required": [CANON_VALUE]}
    m = JSONSchemaMatcher(schema)
    assert accepted(m, canon_doc(canon, canon))
    for i, (_, goods, bads) in enumerate(CANON_PARTS):
        for alt, ok in [(g, True) for g in goods] + [(b, False) for b in bads]:
            parts = canon[:i] + [alt] + canon[i + 1:]
            for doc in (canon_doc(parts, canon), canon_doc(canon, parts)):    # clé, valeur
                assert accepted(m, doc) is ok, (doc, ok)
                if ok:
                    assert validate_instance(json.loads(doc), schema) == []


def test_constant_strings_accept_json_dumps_all_chars():
    """json.dumps(ensure_ascii=True/False) d'une constante quelconque reste accepté."""
    chars = ''.join(chr(i) for i in range(0x80)) + ''.join(ALPHABET) + ' ﻿\x80'
    inst = {chars: chars, "k/é": [chars, "😀"]}
    schema = {"type": "object",
              "properties": {chars: {"enum": [chars, "autre"]}, "k/é": {"const": [chars, "😀"]}},
              "required": [chars, "k/é"]}
    m = JSONSchemaMatcher(schema)
    for data in serializations(inst):
        assert accepted(m, data), data
    for value in (chars, "😀", "é/x", "\x7f "):
        me = JSONSchemaMatcher({"enum": [value]})
        for ea in (True, False):
            assert accepted(me, json.dumps(value, ensure_ascii=ea).encode())
    # Forme canonique minimale = json.dumps(ensure_ascii=False) compact (UTF-8 brut)
    comp = m.shortest_completion(m.initial_state)
    assert comp == json.dumps({chars: "autre", "k/é": [chars, "😀"]}, ensure_ascii=False,
                              separators=(',', ':')).encode('utf-8')
    assert m._fallbacks == 0


def test_shortest_completion_constants_raw():
    m = matcher("unicode_keys")
    full = json.dumps({"prénom": "", "âge": 0, 'clé/"x"\\': True, "emoji😀": None},
                      ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    assert m.shortest_completion(m.initial_state) == full
    assert b'\\u' not in full and b'\\/' not in full
    # Au milieu d'un \u (non-ASCII) : on termine l'échappement, la suite est brute
    assert m.shortest_completion(run(m, b'{"pr\\u00')) == b'e9' + full[len('{"pré'.encode()):]
    assert m.shortest_completion(run(m, b'{"pr\\u00E')) == b'9' + full[len('{"pré'.encode()):]
    m = JSONSchemaMatcher(NOM)
    assert m.shortest_completion(run(m, b'{"no')) == b'm":""}'
    assert m.shortest_completion(run(m, b'{"nom":"x","')) == 'né":0}'.encode()
    assert m.shortest_completion(run(m, b'{"nom":"x","a')) == b'/b":null}'
    m = JSONSchemaMatcher({"enum": ["a/b", "x\ty\u0001"]})
    assert m.shortest_completion(run(m, b'"')) == b'a/b"'
    assert m.shortest_completion(run(m, b'"x')) == b'\\ty\\u0001"'
    assert m.shortest_completion(run(m, b'"x\\ty\\u000')) == b'1"'


def test_completion_tokens_constants_raw():
    vocab = fake_vocab()            # ids 0..255 = octets isolés
    c = TokenConstraint(matcher("unicode_keys"), vocab)
    toks = c.completion_tokens()
    for t in toks:
        c.advance(t)
    assert c.is_terminal()
    assert c.generated == c.matcher.shortest_completion(c.matcher.initial_state)
    assert b'\\u' not in c.generated
    # Préfixe avec é (é échappé, majuscule) : la complétion reste brute ensuite
    c.reset()
    for b in b'{"pr\\u00E9n':
        c.advance(b)
    toks = c.completion_tokens()
    for t in toks:
        c.advance(t)
    assert c.is_complete()
    assert b'\\u' not in c.generated[len(b'{"pr\\u00E9n'):]
    assert json.loads(c.generated)["prénom"] == ""
    # Masque : aucun token d'échappement dans une clé ASCII, é permis pour « é »
    c = json_constraint(vocab, NOM)
    c.advance(vocab.index(b'{"'))
    for tok in (b'\\u', b'\\u00E9', b'\\', b'\\"', b'\\n'):
        assert not c.is_allowed(vocab.index(tok)), tok
    assert c.is_allowed(vocab.index(b'n'))
    for b in b'nom":"x","n':
        c.advance(b)
    assert c.is_allowed(vocab.index(b'\\u00E9')) and c.is_allowed(vocab.index('é'.encode()))
    assert c.is_allowed(vocab.index(b'\\u')) and not c.is_allowed(vocab.index(b'\\n'))
    mask = c.allowed_mask()
    assert mask.tolist() == [c.is_allowed(t) for t in range(len(vocab))]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Marches aléatoires (solidité)
# ─────────────────────────────────────────────────────────────────────────────

INTERESTING = set(b'{}[]",:0123456789-.eE+tfn \\uabd8c')


def check_completion(m, schema, prefix, st):
    comp = m.shortest_completion(st)
    assert comp is not None, prefix
    full = prefix + comp
    assert accepted(m, full), full
    obj = json.loads(full.decode('utf-8'))
    errs = validate_instance(obj, schema)
    assert errs == [], (full, errs)
    return full


@pytest.mark.parametrize("name", sorted(SCHEMAS))
def test_random_walk_bytes(name):
    schema = SCHEMAS[name]
    m = matcher(name)
    rng = random.Random(42 + len(name) * 7)
    for _ in range(400):
        st, prefix = m.initial_state, bytearray()
        for _ in range(rng.randint(0, 120)):
            allowed = m.allowed_bytes(st)
            if not allowed:
                break
            pref = [b for b in allowed if b in INTERESTING]
            b = rng.choice(pref) if (pref and rng.random() < 0.6) else rng.choice(allowed)
            st = m.advance(st, b)
            prefix.append(b)
        check_completion(m, schema, bytes(prefix), st)
    assert m._fallbacks == 0     # complétion analytique toujours exacte (jamais le BFS)


@pytest.mark.parametrize("name", sorted(SCHEMAS))
def test_mutation_fuzz_soundness(name):
    """Documents valides mutés (1-3 octets) : tout ce que l'automate accepte doit être valide."""
    schema = SCHEMAS[name]
    m = matcher(name)
    rng = random.Random(2024 + len(name))
    alphabet = b'{}[]",:0123456789-.eE+ \\u\nabtrufalsn\x00\xc3\xa9\xff'
    accepted_count = 0
    for _ in range(300):
        data = bytearray(rng.choice(serializations(gen(schema, rng, schema))))
        for _ in range(rng.randint(1, 3)):
            pos = rng.randrange(len(data) + 1)
            op = rng.random()
            if op < 0.33 and data:
                data[min(pos, len(data) - 1)] = rng.choice(alphabet)
            elif op < 0.66:
                data.insert(pos, rng.choice(alphabet))
            elif data:
                del data[min(pos, len(data) - 1)]
        data = bytes(data)
        if accepted(m, data):
            accepted_count += 1
            assert validate_instance(json.loads(data.decode('utf-8')), schema) == [], data
    assert accepted_count > 0


def fake_vocab():
    toks = [bytes((i,)) for i in range(256)]
    toks += [s.encode('utf-8') if isinstance(s, str) else s for s in [
        '{"', '"}', '":', '","', '", "', '": ', '[]', '{}', '[{', '}]', '},', '],', ']}',
        'true', 'false', 'null', 'tr', 'ue', 'nu', 'll', 'fa', 'lse', '12', '-0', '0.', '.5',
        'e+', 'E-', '1e', '\\u', '\\"', '\\\\', '\\ud83d', '\\ude00', '\\u00E9', '\\n',
        'é', '€', '😀', '中', 'name', '"name"', '"name":', 'age', 'active', 'tags',
        'children', 'value', '"value": ', 'kind', 'circle', 'rect', 'op', 'args', 'short',
        'exact', 'long', 'prénom', 'âge', 'clé', 'emoji', '  ', '\n', '\n  ', ',\n',
        '"a', 'a"', 'ab', 'abc', ' "', '",', 'rouge', 'vert', '42', '-3.5',
    ]]
    toks += ['😀'.encode()[:2], '😀'.encode()[2:], 'é'.encode()[:1], None, b'']
    return toks


@pytest.mark.parametrize("name", sorted(SCHEMAS))
def test_random_walk_tokens(name):
    schema = SCHEMAS[name]
    vocab = fake_vocab()
    c = TokenConstraint(matcher(name), vocab)
    rng = random.Random(7 + len(name))
    for _ in range(80):
        c.reset()
        for _ in range(rng.randint(0, 60)):
            mask = c.allowed_mask()
            ids = mask.nonzero().flatten().tolist()
            if not ids:
                break
            tok = rng.choice(ids)
            assert c.is_allowed(tok)
            c.advance(tok)
        prefix = c.generated
        check_completion(c.matcher, schema, prefix, c.state)
        toks = c.completion_tokens()
        assert toks is not None
        for t in toks:
            c.advance(t)
        assert c.is_complete()
        assert validate_instance(json.loads(c.generated.decode('utf-8')), schema) == []
        if isinstance(json.loads(c.generated), (dict, list, str)):
            assert c.is_terminal()


def test_token_constraint_basics():
    vocab = fake_vocab()
    c = json_constraint(vocab, OPTIONAL)
    none_ids = [i for i, t in enumerate(vocab) if not t]
    assert none_ids and not any(c.is_allowed(i) for i in none_ids)
    assert not c.is_allowed(-1) and not c.is_allowed(len(vocab))
    assert not bool(c.allowed_mask()[none_ids].any())
    with pytest.raises(ValueError):
        c.advance(vocab.index(b'"}'))
    c.advance(vocab.index(b'{"'))
    d = c.clone()
    d.advance(vocab.index(b'c'))
    assert c.generated == b'{"' and d.generated == b'{"c'
    assert c.state != d.state
    c.reset()
    assert c.generated == b'' and c.state == c.matcher.initial_state
    logits = torch.zeros(2, len(vocab))
    out = c.apply_to_logits(logits)
    assert torch.isinf(out[0, vocab.index(b'"')]) and out[0, vocab.index(b'{')] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Masque exact
# ─────────────────────────────────────────────────────────────────────────────

def char_vocab():
    chars = sorted(set([chr(i) for i in range(32, 127)] + list('\n\téà€中😀')))
    itos = {i: ch for i, ch in enumerate(chars)}
    return token_bytes_from_itos(itos, len(chars) + 3)     # 3 ids de padding → None


@pytest.mark.parametrize("name", ["person", "enum_mixed", "tree", "any", "unicode_keys", "strings"])
def test_mask_exact_char_vocab(name):
    tb = char_vocab()
    assert tb[-1] is None and tb[0] == b'\t'
    c = TokenConstraint(matcher(name), tb)
    rng = random.Random(99)
    checked = 0
    for _ in range(12):
        c.reset()
        for _ in range(rng.randint(0, 40)):
            mask = c.allowed_mask()
            assert mask.dtype == torch.bool and mask.shape == (len(tb),)
            assert mask.tolist() == [c.is_allowed(t) for t in range(len(tb))]
            checked += 1
            ids = mask.nonzero().flatten().tolist()
            if not ids:
                break
            c.advance(rng.choice(ids))
    assert checked > 50


@pytest.fixture(scope="module")
def gpt2_tb():
    if tiktoken is None:
        pytest.skip("tiktoken indisponible")
    try:
        enc = tiktoken.get_encoding("gpt2")
    except Exception as e:     # pragma: no cover — pas de réseau / cache
        pytest.skip(f"encodage gpt2 indisponible : {e}")
    return enc, token_bytes_from_tiktoken(enc)


def test_token_bytes_from_tiktoken(gpt2_tb):
    enc, tb = gpt2_tb
    assert len(tb) == enc.n_vocab == 50257
    assert tb[50256] is None                        # <|endoftext|>
    assert tb[enc.encode('hello')[0]] == b'hello'
    assert all(t is None or isinstance(t, bytes) for t in tb)


def test_mask_exact_gpt2(gpt2_tb):
    enc, tb = gpt2_tb
    c = json_constraint(tb, PERSON)
    prefixes = [b'', b'{', b'{"na', b'{"name":"Zo', b'{"name":"Zo\\u00', b'{"name":"Z",',
                b'{"name":"Z","age":1', b'{"name":"Z","age":12,"active":tr']
    for p in prefixes:
        c.reset()
        c.state = c.matcher.advance_bytes(c.matcher.initial_state, p)
        assert c.state is not None
        mask = c.allowed_mask()
        assert mask.tolist() == [c.is_allowed(t) for t in range(len(tb))], p
        toks = c.completion_tokens()
        assert toks is not None
        for t in toks:
            c.advance(t)
        assert c.is_complete()


def test_constant_keys_gpt2_raw_only(gpt2_tb):
    """Vrai vocabulaire BPE : aucun token d'échappement dans une clé ASCII déclarée."""
    enc, tb = gpt2_tb
    c = json_constraint(tb, PERSON)
    m = c.matcher
    # Après la virgule : email, score, tags, address, active (optionnelles puis requise)
    for p, first in ((b'{"', b'n'), (b'{"name":"Z","', b'a'), (b'{"name":"Z","age":1,"', b'esta')):
        c.state = m.advance_bytes(m.initial_state, p)
        allowed = [tb[i] for i in c.allowed_mask().nonzero().flatten().tolist()]
        assert allowed and all(t[0] in first for t in allowed), (p, allowed)
        toks = c.completion_tokens()
        assert b''.join(tb[t] for t in toks) == m.shortest_completion(c.state)
        assert b'\\' not in b''.join(tb[t] for t in toks)
    c = json_constraint(tb, UNICODE_KEYS)
    c.state = c.matcher.advance_bytes(c.matcher.initial_state, b'{"pr')
    allowed = {tb[i] for i in c.allowed_mask().nonzero().flatten().tolist()}
    assert b'\\' in allowed and 'é'.encode() in allowed      # é brut ou é
    assert all(t[:1] in (b'\\', b'\xc3') for t in allowed), allowed


@pytest.mark.parametrize("name", ["person", "any", "unicode_keys"])
def test_random_walk_tokens_gpt2(gpt2_tb, name):
    """Marches aléatoires sur le vrai vocabulaire BPE (tokens à cheval sur la structure)."""
    enc, tb = gpt2_tb
    schema = SCHEMAS[name]
    c = TokenConstraint(matcher(name), tb)
    rng = random.Random(5)
    for _ in range(8):
        c.reset()
        for _ in range(rng.randint(1, 25)):
            ids = c.allowed_mask().nonzero().flatten().tolist()
            if not ids:
                break
            c.advance(rng.choice(ids))
        toks = c.completion_tokens()
        assert toks is not None
        for t in toks:
            c.advance(t)
        assert c.is_complete()
        assert validate_instance(json.loads(c.generated.decode('utf-8')), schema) == []


# ─────────────────────────────────────────────────────────────────────────────
# 5. Complétion minimale
# ─────────────────────────────────────────────────────────────────────────────

SMALL = [
    {"type": "boolean"},
    {"enum": ["ab", "abc", 1, [True], {"k": None}]},
    {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"enum": ["p", "q"]}},
     "required": ["y"]},
    {"type": "object", "properties": {"x": {"type": "integer"}, "zz": {"type": "string"}}},
    {"type": "array", "items": {"type": "string", "maxLength": 2}, "minItems": 1, "maxItems": 2},
    {"type": "string", "minLength": 2, "maxLength": 3},
    {"type": "number"},
    {"type": ["null", "array"], "items": {"type": "integer"}, "minItems": 2},
    {"type": "array", "prefixItems": [{"const": "é"}, {"type": "boolean"}], "minItems": 2},
    TREE,
]


def bfs_distance(m, st, limit):
    if m.is_accepting(st):
        return 0
    frontier, seen = {st}, {st}
    for d in range(1, limit + 1):
        nxt = set()
        for s in frontier:
            for b in m.allowed_bytes(s):
                ns = m.advance(s, b)
                if m.is_accepting(ns):
                    return d
                if ns not in seen:
                    seen.add(ns)
                    nxt.add(ns)
        frontier = nxt
    return None


def minimality_prefixes(m, schema, rng, n=60):
    """Préfixes : marches aléatoires courtes + documents valides tronqués près de la fin."""
    for i in range(n):
        if i % 2 == 0:
            st, prefix = m.initial_state, bytearray()
            for _ in range(rng.randint(0, 14)):
                allowed = m.allowed_bytes(st)
                if not allowed:
                    break
                pref = [b for b in allowed if b in INTERESTING]
                b = rng.choice(pref) if (pref and rng.random() < 0.7) else rng.choice(allowed)
                st = m.advance(st, b)
                prefix.append(b)
            yield bytes(prefix), st
        else:
            data = rng.choice(serializations(gen(schema, rng, schema)))
            cut = max(0, len(data) - rng.randint(0, 12))
            yield data[:cut], run(m, data[:cut])


@pytest.mark.parametrize("idx", range(len(SMALL)))
def test_shortest_completion_minimal(idx):
    schema = SMALL[idx]
    m = JSONSchemaMatcher(schema)
    rng = random.Random(idx)
    checked = 0
    for prefix, st in minimality_prefixes(m, schema, rng):
        assert st is not None, prefix
        comp = m.shortest_completion(st)
        assert comp is not None
        assert m.is_accepting(m.advance_bytes(st, comp))
        if len(comp) <= 10:
            assert bfs_distance(m, st, len(comp)) == len(comp), (prefix, comp)
            checked += 1
    assert checked >= 20
    assert m._fallbacks == 0


def test_shortest_completion_examples():
    m = JSONSchemaMatcher(PERSON)
    assert m.shortest_completion(m.initial_state) == b'{"name":"a","age":0,"active":true}'
    st = run(m, b'{"name":"Zo')
    assert m.shortest_completion(st) == b'","age":0,"active":true}'
    m = JSONSchemaMatcher(ROOT_STRING)
    assert m.shortest_completion(run(m, b'"\\ud8')) == b'00a"'     # haut isolé = 1 point
    assert m.shortest_completion(run(m, b'"ab\\ud83d\\ude00\\ud83d')) == b'"'
    assert m.shortest_completion(run(m, b'"ab\\ud83d\\ude00\\ud83d\\')) == b'udc00"'
    m = JSONSchemaMatcher(None)
    assert m.shortest_completion(run(m, b'[1, {"a": [tr')) == b'ue]}]'
    assert m.shortest_completion(run(m, b'-')) == b'0'
    assert m.shortest_completion(run(m, b'12')) == b''


# ─────────────────────────────────────────────────────────────────────────────
# 6. Performance (gpt2)
# ─────────────────────────────────────────────────────────────────────────────

def test_performance_gpt2(gpt2_tb):
    enc, tb = gpt2_tb
    schema = {"type": "object", "properties": {"text": {"type": "string"}, "n": {"type": "integer"}},
              "required": ["text", "n"]}
    fresh = list(tb)                 # liste neuve → pas de trie en cache
    m = JSONSchemaMatcher(schema)
    t0 = time.perf_counter()
    c = TokenConstraint(m, fresh)
    t_build = time.perf_counter() - t0
    assert t_build < 3.0, t_build

    c.state = m.advance_bytes(m.initial_state, b'{"text":"Bonj')
    t0 = time.perf_counter()
    mask = c.allowed_mask()
    t_mask = time.perf_counter() - t0
    assert t_mask < 2.0, t_mask
    assert int(mask.sum()) > 45_000      # presque tout est permis dans une chaîne libre

    t0 = time.perf_counter()
    for _ in range(20):
        c.allowed_mask()
    t_cached = (time.perf_counter() - t0) / 20
    assert t_cached < 0.005, t_cached

    rng = random.Random(0)
    ids = [rng.randrange(len(tb)) for _ in range(5000)]
    c2 = json_constraint(fresh, schema)
    c2.state = m.advance_bytes(c2.matcher.initial_state, b'{"text":"x","n":1')
    t0 = time.perf_counter()
    for i in ids:
        c2.is_allowed(i)
    t_allowed = (time.perf_counter() - t0) / len(ids)
    assert t_allowed < 0.0002, t_allowed


# ─────────────────────────────────────────────────────────────────────────────
# 7. load_schema / SchemaError / mots-clés ignorés / validation
# ─────────────────────────────────────────────────────────────────────────────

def test_load_schema_file_and_inline(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(PERSON, ensure_ascii=False), encoding="utf-8")
    assert load_schema(str(p)) == PERSON
    assert load_schema(json.dumps(TREE)) == TREE
    assert load_schema('true') == {}
    for bad in ['false', '{"allOf": [{"type": "string"}, {"maxLength": 2}]}',
                '{"$ref": "#/$defs/nope"}', '{"type": "strin"}', '{not json', '[1, 2]',
                str(tmp_path / "absent.json"), '',
                '{"type": "object", "properties": {"a": {"$ref": "#"}}, "required": ["a"]}']:
        with pytest.raises(SchemaError):
            load_schema(bad)


def test_schema_errors():
    with pytest.raises(SchemaError):
        JSONSchemaMatcher(False)
    with pytest.raises(SchemaError):
        JSONSchemaMatcher({"allOf": []})
    with pytest.raises(SchemaError):
        JSONSchemaMatcher({"allOf": [{}, {}]})
    with pytest.raises(SchemaError):
        JSONSchemaMatcher({"$ref": "#/definitions/missing"})
    with pytest.raises(SchemaError):
        JSONSchemaMatcher({"$ref": "http://example.com/s.json"})
    with pytest.raises(SchemaError):
        JSONSchemaMatcher({"$defs": {"a": {"$ref": "#/$defs/b"}, "b": {"$ref": "#/$defs/a"}},
                           "$ref": "#/$defs/a"})          # cycle pur → insatisfiable
    with pytest.raises(SchemaError):
        JSONSchemaMatcher({"type": "string", "minLength": 3, "maxLength": 2})
    assert issubclass(SchemaError, ValueError)


def test_refs_definitions_and_merges():
    m = JSONSchemaMatcher({"definitions": {"pos": {"type": "integer"}},
                           "type": "array", "items": {"$ref": "#/definitions/pos"}})
    assert accepted(m, b'[1,2]') and not accepted(m, b'["a"]')
    m = JSONSchemaMatcher({"properties": {"a": {"type": "string"}, "b": {"$ref": "#/properties/a"}}})
    assert accepted(m, b'{"a":"x","b":"y"}') and not accepted(m, b'{"b":1}')
    merged = {"allOf": [{"$ref": "#/$defs/base"}], "properties": {"extra": {"type": "integer"}},
              "required": ["extra"],
              "$defs": {"base": {"type": "object", "properties": {"id": {"type": "string"}},
                                 "required": ["id"]}}}
    m = JSONSchemaMatcher(merged)
    doc = m.shortest_completion(m.initial_state)
    assert validate_instance(json.loads(doc), merged) == []
    assert accepted(m, b'{"extra":1,"id":"x"}') and not accepted(m, b'{"extra":1}')
    m = JSONSchemaMatcher({"$ref": "#/$defs/s", "maxLength": 2,
                           "$defs": {"s": {"type": "string", "minLength": 1}}})
    assert accepted(m, b'"ab"') and not accepted(m, b'"abc"') and not accepted(m, b'""')
    m = JSONSchemaMatcher({"type": "string", "anyOf": [{"maxLength": 1}, {"minLength": 4}]})
    assert accepted(m, b'"a"') and accepted(m, b'"abcd"') and not accepted(m, b'"ab"')
    m = JSONSchemaMatcher({"type": "string", "enum": ["a", 1, "b"]})
    assert accepted(m, b'"b"') and not accepted(m, b'1')


def test_ignored_keywords_not_enforced():
    schema = {"type": "object",
              "properties": {"mail": {"type": "string", "format": "email", "pattern": "^x"},
                             "n": {"type": "integer", "minimum": 10, "multipleOf": 3},
                             "l": {"type": "array", "uniqueItems": True}},
              "required": ["mail", "n", "l"], "not": {"required": ["zz"]},
              "$defs": {"unused": {"description": "annotation"}}}
    m = JSONSchemaMatcher(schema)
    assert {"format", "pattern", "minimum", "multipleOf", "uniqueItems", "not"} <= m.ignored_keywords
    assert "description" not in m.ignored_keywords and "title" not in m.ignored_keywords
    doc = b'{"mail":"nope","n":1,"l":[1,1]}'
    assert accepted(m, doc)
    assert validate_instance(json.loads(doc), schema) == []
    assert JSONSchemaMatcher(PERSON).ignored_keywords == {"minimum", "format"}


def test_validate_instance():
    assert validate_instance({"name": "a", "age": 1, "active": True}, PERSON) == []
    errs = validate_instance({"name": "", "age": "x", "zz": 1}, PERSON)
    assert len(errs) >= 4 and all(isinstance(e, str) for e in errs)
    assert validate_instance(True, {"enum": [1]}) != []           # bool ≠ nombre
    assert validate_instance(1.0, {"enum": [1]}) == []
    assert validate_instance(2.0, {"type": "integer"}) == []
    assert validate_instance({"a": [1, {"b": None}]}, None) == []
    assert validate_instance("😀", {"type": "string", "maxLength": 1}) == []
    assert validate_instance([1, "x"], {"oneOf": [{"type": "array"}, {"type": "array"}]}) == []
    assert validate_instance(1, False) != []
    assert validate_instance({"value": 1, "children": [{"value": 2, "children": []}]}, TREE) == []
    assert validate_instance({"value": 1, "children": [{"value": "x", "children": []}]}, TREE) != []


def test_token_bytes_from_itos():
    tb = token_bytes_from_itos({0: 'a', 1: 'é', 3: '😀', 4: ''}, 6)
    assert tb == [b'a', 'é'.encode(), None, '😀'.encode(), None, None]
