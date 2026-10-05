"""
Tests adverses — sémantique JSON Schema (round 1).

Propriétés vérifiées pour chaque mot-clé supporté :
  · soundness   : tout document accepté par l'automate est valide pour validate_instance
  · complétude  : les documents valides écrits dans la forme documentée sont acceptés
  · complétion  : shortest_completion() depuis n'importe quel préfixe → document valide
  · rapport     : ignored_keywords liste toute assertion non appliquée
Les tests qui échouent révèlent un défaut réel (aucun xfail / skip).
"""

import itertools
import json
import random
import sys

import pytest

from structured import JSONSchemaMatcher, SchemaError, validate_instance


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def _acc(m, data: bytes) -> bool:
    st = m.advance_bytes(m.initial_state, data)
    return st is not None and m.is_accepting(st)


def _valid(doc: bytes, schema) -> bool:
    return validate_instance(json.loads(doc.decode('utf-8')), schema) == []


def _assert_sound(schema, docs):
    """Soit le schéma est refusé (SchemaError), soit accepté ⇒ valide pour chaque doc."""
    try:
        m = JSONSchemaMatcher(schema)
    except SchemaError:
        return None
    for d in docs:
        if _acc(m, d):
            assert _valid(d, schema), (d, validate_instance(json.loads(d), schema))
    c = m.shortest_completion(m.initial_state)
    assert c is not None and _valid(c, schema), (c, validate_instance(json.loads(c), schema))
    return m


def _all_prefix_completions_valid(m, schema, docs):
    """Chaque préfixe de chaque document (accepté par l'automate) se complète en doc valide."""
    seen = set()
    for doc in docs:
        st = m.initial_state
        for i in range(len(doc) + 1):
            if i:
                st = m.advance(st, doc[i - 1])
                if st is None:
                    break
            if st in seen:
                continue
            seen.add(st)
            c = m.shortest_completion(st)
            assert c is not None, (doc[:i],)
            full = doc[:i] + c
            assert _acc(m, full), full
            assert _valid(full, schema), (full, validate_instance(json.loads(full), schema))


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT A — fusion de schémas : `==` Python confond true/1 et false/0
#   merge() court-circuite avec `va == vb` (et `ia != ib` pour items) ; en Python
#   True == 1, [False, True] == [0, 1], {"const": 0} == {"const": False}. L'intersection
#   (vide en JSON Schema) garde donc la contrainte d'UN seul côté → l'automate accepte
#   (et shortest_completion produit) un document que validate_instance rejette.
# ─────────────────────────────────────────────────────────────────────────────

_BOOLNUM_DOCS = [b'true', b'false', b'0', b'1']


def test_merge_enum_bool_vs_number_allof_ref():
    schema = {"$defs": {"bit": {"enum": [0, 1]}},
              "allOf": [{"$ref": "#/$defs/bit"}],
              "enum": [False, True]}
    _assert_sound(schema, _BOOLNUM_DOCS)


def test_merge_const_true_vs_const_1():
    schema = {"const": True, "allOf": [{"const": 1}]}
    _assert_sound(schema, _BOOLNUM_DOCS)


def test_merge_const_false_vs_const_0_via_ref():
    schema = {"$ref": "#/$defs/z", "const": False, "$defs": {"z": {"const": 0}}}
    _assert_sound(schema, _BOOLNUM_DOCS)


def test_merge_nested_properties_bool_vs_number():
    schema = {"$ref": "#/$defs/x",
              "properties": {"a": {"const": False}},
              "$defs": {"x": {"type": "object", "properties": {"a": {"const": 0}},
                              "required": ["a"]}}}
    _assert_sound(schema, [b'{"a":0}', b'{"a":false}'])


def test_merge_items_bool_vs_number():
    schema = {"type": "array", "items": {"const": 1}, "allOf": [{"items": {"const": True}}]}
    _assert_sound(schema, [b'[]', b'[1]', b'[true]'])


def test_merge_anyof_siblings_bool_vs_number():
    schema = {"enum": [1, 2], "anyOf": [{"enum": [True, 2]}]}
    _assert_sound(schema, [b'1', b'2', b'true'])


def test_merge_genuinely_equal_values_still_merge():
    """Régression : des valeurs JSON réellement égales (1 et 1.0) restent compatibles."""
    m = JSONSchemaMatcher({"const": 1, "allOf": [{"const": 1.0}]})
    assert _acc(m, b'1')
    m = JSONSchemaMatcher({"enum": ["a", 2], "allOf": [{"enum": ["a", 2]}]})
    assert _acc(m, b'"a"') and _acc(m, b'2') and not _acc(m, b'3')


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT B — ignored_keywords ne voit pas les cibles de $ref hors conteneurs standard
#   Le docstring annonce "$ref → pointeur JSON quelconque dans la racine" et
#   "assertions non appliquées … listées dans ignored_keywords". Une cible rangée sous
#   une clé non standard (ex. OpenAPI components/schemas) est compilée, mais ses
#   assertions non appliquées ne sont pas rapportées.
# ─────────────────────────────────────────────────────────────────────────────

def test_ignored_keywords_reported_through_ref_to_custom_container():
    schema = {"$ref": "#/components/schemas/Mail",
              "components": {"schemas": {"Mail": {"type": "string", "format": "email",
                                                  "pattern": "^[^@]+@[^@]+$"}}}}
    m = JSONSchemaMatcher(schema)
    assert _acc(m, b'"pas un mail"')            # non appliqué (documenté)…
    assert {"format", "pattern"} <= m.ignored_keywords   # …mais doit être rapporté


def test_ignored_keywords_reported_through_nested_ref_chain():
    schema = {"type": "object",
              "properties": {"n": {"$ref": "#/x-models/Count"}},
              "required": ["n"],
              "x-models": {"Count": {"type": "integer", "minimum": 1, "maximum": 9}}}
    m = JSONSchemaMatcher(schema)
    assert _acc(m, b'{"n":42}')
    assert {"minimum", "maximum"} <= m.ignored_keywords


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT D — $dynamicRef / $recursiveRef ignorés silencieusement
#   Applicateurs JSON Schema (2020-12 / 2019-09) ni appliqués, ni refusés (SchemaError),
#   ni rapportés : la propriété devient « any » sans avertissement.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw", ["$dynamicRef", "$recursiveRef"])
def test_dynamic_ref_not_silently_ignored(kw):
    schema = {"type": "object", "properties": {"a": {kw: "#/$defs/int"}}, "required": ["a"],
              "$defs": {"int": {"type": "integer"}}}
    try:
        m = JSONSchemaMatcher(schema)
    except SchemaError:
        return                                   # refus explicite : acceptable
    enforced = not _acc(m, b'{"a":"texte"}')
    assert enforced or kw in m.ignored_keywords, m.ignored_keywords


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT C — validate_instance échoue sur des documents profonds acceptés
#   _MAX_VALIDATE_DEPTH (400) compte aussi les sauts $ref / anyOf : ~99 niveaux suffisent
#   à lever RecursionError (non rattrapée), ~134 niveaux de TREE donnent l'erreur
#   factice « récursion trop profonde ». L'automate, lui, accepte sans limite → la
#   promesse « une sortie générée est donc toujours valide » est rompue (chat.py
#   report_json appelle validate_instance sans try).
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_instance_deep_list_through_anyof_ref():
    schema = {"$defs": {"l": {"anyOf": [{"type": "null"},
                                        {"type": "array",
                                         "items": {"anyOf": [{"$ref": "#/$defs/l"}]}}]}},
              "$ref": "#/$defs/l"}
    m = JSONSchemaMatcher(schema)
    d = 120
    doc = b'[' * d + b'null' + b']' * d          # 248 octets
    assert _acc(m, doc)
    assert validate_instance(json.loads(doc), schema) == []


def test_validate_instance_deep_tree():
    tree = {"type": "object",
            "properties": {"value": {"type": "integer"},
                           "children": {"type": "array", "items": {"$ref": "#"},
                                        "maxItems": 3}},
            "required": ["value", "children"]}
    inner = {"value": 0, "children": []}
    for _ in range(150):
        inner = {"value": 1, "children": [inner]}
    doc = json.dumps(inner, separators=(',', ':')).encode()
    m = JSONSchemaMatcher(tree)
    assert _acc(m, doc)
    assert validate_instance(json.loads(doc), tree) == []


def test_validate_instance_moderate_depth_ok():
    """Régression : profondeur raisonnable (30) validée correctement."""
    schema = {"type": "array", "items": {"$ref": "#"}}
    doc = b'[' * 30 + b']' * 30
    assert _acc(JSONSchemaMatcher(schema), doc)
    assert validate_instance(json.loads(doc), schema) == []


# ─────────────────────────────────────────────────────────────────────────────
# Régressions (passent) — enum / const
# ─────────────────────────────────────────────────────────────────────────────

ENUM_VALUES = [None, True, False, 0, 1, -1, 1.5, -2.25e-7, 1e21, -0.0, "", "x", "é😀",
               [], [None, [True]], {}, {"a": [1, {"b": None}], "c": "d"}]


@pytest.mark.parametrize("v", ENUM_VALUES, ids=[repr(v)[:20] for v in ENUM_VALUES])
def test_const_every_json_kind(v):
    schema = {"const": v}
    m = JSONSchemaMatcher(schema)
    for ea in (True, False):
        for sep in ((',', ':'), None):
            doc = json.dumps(v, ensure_ascii=ea, separators=sep).encode('utf-8')
            assert _acc(m, doc), doc
    c = m.shortest_completion(m.initial_state)
    assert _valid(c, schema)
    for other in ENUM_VALUES:
        doc = json.dumps(other, separators=(',', ':')).encode('utf-8')
        if _acc(m, doc):
            assert _valid(doc, schema), doc


def test_enum_bool_number_null_distinct():
    m = JSONSchemaMatcher({"enum": [1, 0, None]})
    assert _acc(m, b'1') and _acc(m, b'0') and _acc(m, b'null')
    assert not _acc(m, b'true') and not _acc(m, b'false')
    m = JSONSchemaMatcher({"enum": [True, False]})
    assert not _acc(m, b'1') and not _acc(m, b'0') and not _acc(m, b'null')
    m = JSONSchemaMatcher({"enum": [12, 1]})
    assert _acc(m, b'1') and _acc(m, b'12') and not _acc(m, b'121') and not _acc(m, b'2')
    m = JSONSchemaMatcher({"type": "array", "items": {"enum": [1, 12]}})
    assert _acc(m, b'[1,12,1]') and not _acc(m, b'[13]') and not _acc(m, b'[1.0]')


def test_enum_filtered_by_sibling_type_and_lengths():
    schema = {"type": ["string", "null"], "maxLength": 2, "enum": ["ab", "abc", None, 1, "é😀"]}
    m = JSONSchemaMatcher(schema)
    assert _acc(m, b'"ab"') and _acc(m, b'null') and _acc(m, '"é😀"'.encode())
    assert not _acc(m, b'"abc"') and not _acc(m, b'1')
    with pytest.raises(SchemaError):
        JSONSchemaMatcher({"type": "integer", "enum": [1.5, "1", True]})


def test_enum_object_rejects_missing_and_extra_keys():
    schema = {"enum": [{"a": 1, "b": [2]}]}
    m = JSONSchemaMatcher(schema)
    assert _acc(m, b'{"a":1,"b":[2]}') and _acc(m, b'{ "a" : 1 , "b" : [ 2 ] }')
    for bad in [b'{"a":1}', b'{"a":1,"b":[2],"c":3}', b'{"a":1,"b":[2,2]}', b'{"a":1,"b":[]}',
                b'{"a":true,"b":[2]}']:
        assert not _acc(m, bad), bad


# ─────────────────────────────────────────────────────────────────────────────
# Régressions — propriétés requises / optionnelles (ordre de déclaration)
# ─────────────────────────────────────────────────────────────────────────────

def test_required_optional_exhaustive_in_declaration_order():
    schema = {"type": "object",
              "properties": {"a": {"type": "integer"}, "b": {"type": "string"},
                             "c": {"type": "null"}, "d": {"type": "boolean"}},
              "required": ["b", "d"]}
    m = JSONSchemaMatcher(schema)
    vals = {"a": 1, "b": "x", "c": None, "d": True}
    names = list(vals)
    for r in range(len(names) + 1):
        for subset in itertools.combinations(names, r):
            for perm in itertools.permutations(subset):
                inst = {k: vals[k] for k in perm}
                doc = json.dumps(inst, separators=(',', ':')).encode()
                ok = validate_instance(inst, schema) == []
                ordered = list(perm) == [k for k in names if k in perm]
                assert _acc(m, doc) == (ok and ordered), (doc, ok, ordered)


def test_required_not_in_properties_uses_additional_schema():
    schema = {"type": "object", "properties": {"a": {"type": "string"}},
              "required": ["z"], "additionalProperties": {"type": "integer"}}
    m = JSONSchemaMatcher(schema)
    assert _acc(m, b'{"z":1}') and _acc(m, b'{"a":"s","z":-3}')
    assert not _acc(m, b'{"z":"s"}') and not _acc(m, b'{"a":"s"}') and not _acc(m, b'{}')
    _all_prefix_completions_valid(m, schema, [b'{"a":"s","z":-3}', b'{"z":1}'])
    with pytest.raises(SchemaError):
        JSONSchemaMatcher({"type": "object", "properties": {"a": {}}, "required": ["z"],
                           "additionalProperties": False})


def test_optional_unsatisfiable_property_is_skipped():
    schema = {"type": "object", "properties": {"never": False, "ok": {"type": "integer"}},
              "required": ["ok"]}
    m = JSONSchemaMatcher(schema)
    assert _acc(m, b'{"ok":1}') and not _acc(m, b'{"never":1,"ok":1}')


# ─────────────────────────────────────────────────────────────────────────────
# Régressions — objets libres avec additionalProperties
# ─────────────────────────────────────────────────────────────────────────────

def test_free_object_additional_schema():
    schema = {"type": "object",
              "additionalProperties": {"type": "array", "items": {"type": "integer"},
                                       "maxItems": 2}}
    m = JSONSchemaMatcher(schema)
    good = [b'{}', b'{"":[]}', b'{"k":[1,2],"\\u00e9":[3]}', b'{"a\\"b":[0]}']
    for d in good:
        assert _acc(m, d) and _valid(d, schema), d
    for d in [b'{"k":[1,2,3]}', b'{"k":1}', b'{"k":["x"]}', b'{k:[]}', b'{"k":[],}']:
        assert not _acc(m, d), d
    _all_prefix_completions_valid(m, schema, good)
    m = JSONSchemaMatcher({"type": "object", "additionalProperties": False})
    assert _acc(m, b'{}') and _acc(m, b'{ }') and not _acc(m, b'{"a":1}')


# ─────────────────────────────────────────────────────────────────────────────
# Régressions — tableaux : minItems / maxItems (0, égaux), prefixItems
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mn,mx", [(0, 0), (1, 1), (2, 2), (0, 1), (0, None), (3, None)])
def test_array_bounds_exhaustive(mn, mx):
    schema = {"type": "array", "items": {"type": "integer"}, "minItems": mn}
    if mx is not None:
        schema["maxItems"] = mx
    m = JSONSchemaMatcher(schema)
    for n in range(6):
        doc = json.dumps(list(range(n))).encode()
        assert _acc(m, doc) == (n >= mn and (mx is None or n <= mx)), doc
    c = m.shortest_completion(m.initial_state)
    assert _valid(c, schema) and len(json.loads(c)) == mn


def test_prefix_items_variants():
    s1 = {"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}],
          "items": False}
    m = JSONSchemaMatcher(s1)
    assert _acc(m, b'[]') and _acc(m, b'["a"]') and _acc(m, b'["a",1]')
    assert not _acc(m, b'["a",1,2]') and not _acc(m, b'[1]')
    s2 = {"type": "array", "items": [{"type": "null"}], "additionalItems": {"type": "boolean"},
          "minItems": 2, "maxItems": 3}
    m = JSONSchemaMatcher(s2)
    assert _acc(m, b'[null,true]') and _acc(m, b'[null,true,false]')
    assert not _acc(m, b'[null]') and not _acc(m, b'[null,1]') and not _acc(m, b'[true,true]')
    assert _valid(m.shortest_completion(m.initial_state), s2)
    s3 = {"type": "array", "prefixItems": [{"const": "x"}, False], "minItems": 1}
    m = JSONSchemaMatcher(s3)
    assert _acc(m, b'["x"]') and not _acc(m, b'["x",1]')
    with pytest.raises(SchemaError):
        JSONSchemaMatcher({"type": "array", "prefixItems": [{}, False], "minItems": 2})
    s4 = {"type": "array", "prefixItems": [{"type": "integer"}] * 3, "maxItems": 2}
    m = JSONSchemaMatcher(s4)
    assert _acc(m, b'[1,2]') and not _acc(m, b'[1,2,3]')


# ─────────────────────────────────────────────────────────────────────────────
# Régressions — chaînes : minLength / maxLength (échappements, multioctets, paires)
# ─────────────────────────────────────────────────────────────────────────────

_FRAGS = ['a', '\\ud83d', '\\uDE00', '\\n', 'é', '😀', '\\u0041', '\\udc00']


@pytest.mark.parametrize("mn,mx", [(0, 0), (1, 1), (2, 2), (0, 1), (2, None), (1, 3)])
def test_string_length_matches_validator_exhaustive(mn, mx):
    schema = {"type": "string", "minLength": mn}
    if mx is not None:
        schema["maxLength"] = mx
    m = JSONSchemaMatcher(schema)
    for L in range(4):
        for combo in itertools.product(_FRAGS, repeat=L):
            doc = ('"' + ''.join(combo) + '"').encode('utf-8')
            assert _acc(m, doc) == _valid(doc, schema), doc


def test_string_length_prefix_completions_valid():
    schema = {"type": "object",
              "properties": {"s": {"type": "string", "minLength": 2, "maxLength": 2}},
              "required": ["s"]}
    m = JSONSchemaMatcher(schema)
    docs = [('{"s":"' + ''.join(c) + '"}').encode('utf-8')
            for c in itertools.product(_FRAGS[:6], repeat=2)]
    _all_prefix_completions_valid(m, schema, docs)


# ─────────────────────────────────────────────────────────────────────────────
# Régressions — anyOf / oneOf / allOf / $ref
# ─────────────────────────────────────────────────────────────────────────────

def test_overlapping_anyof_object_branches():
    schema = {"oneOf": [
        {"type": "object", "properties": {"k": {"const": "a"}, "v": {"type": "integer"}},
         "required": ["k", "v"]},
        {"type": "object", "properties": {"k": {"const": "ab"}, "v": {"type": "string"}},
         "required": ["k", "v"]},
        {"type": "object", "additionalProperties": {"type": "null"}},
    ]}
    m = JSONSchemaMatcher(schema)
    good = [b'{"k":"a","v":1}', b'{"k":"ab","v":"s"}', b'{"k":null}', b'{}']
    for d in good:
        assert _acc(m, d) and _valid(d, schema), d
    for d in [b'{"k":"a","v":"s"}', b'{"k":"ab","v":1}', b'{"k":"b","v":1}']:
        assert not _acc(m, d), d
    _all_prefix_completions_valid(m, schema, good)


def test_anyof_with_required_siblings():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
              "anyOf": [{"required": ["a"]}, {"required": ["b"]}]}
    m = JSONSchemaMatcher(schema)
    assert _acc(m, b'{"a":1}') and _acc(m, b'{"b":1}') and _acc(m, b'{"a":1,"b":2}')
    assert not _acc(m, b'{}')
    assert _valid(m.shortest_completion(m.initial_state), schema)


def test_ref_recursion_defs_and_definitions():
    schema = {"definitions": {"leaf": {"type": "integer"}},
              "$defs": {"node": {"type": "object",
                                 "properties": {"v": {"$ref": "#/definitions/leaf"},
                                                "kids": {"type": "array",
                                                         "items": {"$ref": "#/$defs/node"},
                                                         "maxItems": 2}},
                                 "required": ["v"]}},
              "$ref": "#/$defs/node"}
    m = JSONSchemaMatcher(schema)
    good = [b'{"v":1}', b'{"v":1,"kids":[]}', b'{"v":1,"kids":[{"v":2,"kids":[{"v":3}]}]}']
    for d in good:
        assert _acc(m, d) and _valid(d, schema), d
    for d in [b'{"v":"x"}', b'{"kids":[]}', b'{"v":1,"kids":[{"v":2},{"v":3},{"v":4}]}']:
        assert not _acc(m, d), d
    _all_prefix_completions_valid(m, schema, good)


def test_allof_single_with_siblings():
    schema = {"allOf": [{"type": "string", "minLength": 2}], "maxLength": 3}
    m = JSONSchemaMatcher(schema)
    assert _acc(m, b'"ab"') and _acc(m, b'"abc"')
    assert not _acc(m, b'"a"') and not _acc(m, b'"abcd"') and not _acc(m, b'1')


def test_ignored_keywords_in_nested_positions():
    schema = {"type": "object",
              "properties": {"a": {"type": "array", "items": {"type": "string", "format": "uri"},
                                   "uniqueItems": True}},
              "$defs": {"x": {"anyOf": [{"type": "number", "multipleOf": 2}]}},
              "additionalProperties": {"propertyNames": {"pattern": "^x"}},
              "description": "annotation", "examples": [{"pattern": "pas un mot-clé"}]}
    m = JSONSchemaMatcher(schema)
    assert m.ignored_keywords == {"format", "uniqueItems", "multipleOf", "propertyNames",
                                  "pattern"}


# ─────────────────────────────────────────────────────────────────────────────
# Fuzz différentiel (graine fixe) : schémas aléatoires du sous-ensemble supporté
# ─────────────────────────────────────────────────────────────────────────────

_CONSTS = [None, True, False, 0, 1, 2, -1, 1.5, "", "a", "ab", "é", "😀", 'q"\\', [], [1, "a"],
           {}, {"k": None}]


def _rschema(rng, depth, defs):
    r = rng.random() if depth <= 2 else rng.random() * 0.5
    if r < 0.08:
        return {"enum": [rng.choice(_CONSTS) for _ in range(rng.randint(1, 3))]}
    if r < 0.12:
        return {"const": rng.choice(_CONSTS)}
    if r < 0.24:
        t = rng.choice(["string", "integer", "number", "boolean", "null"])
        s = {"type": t}
        if t == "string" and rng.random() < 0.6:
            s["minLength"] = rng.randint(0, 2)
            if rng.random() < 0.5:
                s["maxLength"] = s["minLength"] + rng.randint(0, 2)
        return s
    if r < 0.30:
        return {"type": rng.sample(["string", "integer", "null", "array", "object"], 2),
                "maxLength": rng.randint(0, 2)}
    if r < 0.45:
        props = {n: _rschema(rng, depth + 1, defs) for n in rng.sample(["a", "b", "c"], rng.randint(0, 3))}
        s = {"type": "object", "properties": props}
        req = [n for n in props if rng.random() < 0.5]
        if req:
            s["required"] = req
        return s
    if r < 0.52:
        return {"type": "object", "additionalProperties": rng.choice([False, _rschema(rng, depth + 1, defs)])}
    if r < 0.65:
        s = {"type": "array", "items": _rschema(rng, depth + 1, defs)}
        if rng.random() < 0.3:
            s["prefixItems"] = [_rschema(rng, depth + 1, defs)]
        mn = rng.randint(0, 2)
        s["minItems"] = mn
        if rng.random() < 0.5:
            s["maxItems"] = mn + rng.randint(0, 1)
        return s
    if r < 0.78:
        s = {rng.choice(["anyOf", "oneOf"]): [_rschema(rng, depth + 1, defs)
                                              for _ in range(rng.randint(1, 3))]}
        if rng.random() < 0.3:
            s["required"] = ["a"]
        return s
    if r < 0.86:
        return {"allOf": [_rschema(rng, depth + 1, defs)],
                "type": rng.choice(["string", "object", "array", "integer"])}
    if r < 0.94 and defs:
        return {"$ref": "#/$defs/" + rng.choice(defs)}
    return {}


def _rinst(rng, depth=0):
    k = rng.random() if depth <= 2 else rng.random() * 0.6
    if k < 0.45:
        return rng.choice(_CONSTS)
    if k < 0.7:
        return [_rinst(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    return {rng.choice(["a", "b", "c", "z"]): _rinst(rng, depth + 1) for _ in range(rng.randint(0, 3))}


def test_differential_fuzz_matcher_vs_validator():
    built = 0
    for seed in range(250):
        rng = random.Random(seed)
        defs = {"d0": _rschema(rng, 1, [])}
        root = dict(_rschema(rng, 0, ["d0"]))
        root["$defs"] = defs
        try:
            m = JSONSchemaMatcher(root)
        except SchemaError:
            continue
        built += 1
        c = m.shortest_completion(m.initial_state)
        assert c is not None and _valid(c, root), (seed, root, c)
        for _ in range(6):                       # marche aléatoire + complétion
            st, data = m.initial_state, bytearray()
            for _ in range(rng.randint(0, 30)):
                al = m.allowed_bytes(st)
                if not al:
                    break
                b = rng.choice(al)
                data.append(b)
                st = m.advance(st, b)
            cc = m.shortest_completion(st)
            assert cc is not None, (seed, bytes(data))
            assert _valid(bytes(data) + cc, root), (seed, root, bytes(data) + cc)
        for _ in range(25):                      # accepté ⇒ valide
            doc = json.dumps(_rinst(rng), separators=(',', ':')).encode()
            if _acc(m, doc):
                assert _valid(doc, root), (seed, root, doc)
    assert built > 150


# ═════════════════════════════════════════════════════════════════════════════
# ROUND 2 — sémantique JSON Schema : valeurs de mots-clés, fusions, enum, oracles
# ═════════════════════════════════════════════════════════════════════════════

from structured import load_schema   # noqa: E402  (ajout round 2)


def _validate_or_raise(doc: bytes, schema):
    """validate_instance sur un document accepté — toute exception est un désaccord."""
    return validate_instance(json.loads(doc.decode('utf-8')), schema)


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT E — mot-clé explicitement `null` : le compilateur le traite comme ABSENT,
#   validate_instance l'utilise tel quel → TypeError sur CHAQUE document accepté.
#   `_nonneg(S, 'maxLength', None)` confond la valeur par défaut None et un `null`
#   explicite (`if v is default`) ; `type: null` passe par `_infer_types`. load_schema
#   accepte donc ces schémas, l'automate génère, puis validate_instance (report_json de
#   chat.py) lève TypeError — alors que minLength: null / minItems: null → SchemaError.
# ─────────────────────────────────────────────────────────────────────────────

_NULL_KEYWORD_CASES = [
    ({"type": "string", "maxLength": None}, b'"abc"'),
    ({"maxLength": None}, b'"x"'),
    ({"type": "array", "maxItems": None}, b'[1,2]'),
    ({"type": "array", "items": {"type": "string", "maxLength": None}}, b'["abc"]'),
    ({"type": None}, b'"x"'),
    ({"type": None, "properties": {"a": {"type": "integer"}}}, b'{"a":1}'),
]


@pytest.mark.parametrize("schema,doc", _NULL_KEYWORD_CASES,
                         ids=[json.dumps(s)[:40] for s, _ in _NULL_KEYWORD_CASES])
def test_null_keyword_value_matcher_and_validator_agree(schema, doc):
    try:
        loaded = load_schema(json.dumps(schema))
        m = JSONSchemaMatcher(loaded)
    except SchemaError:
        return                                   # refus explicite : acceptable
    for d in (m.shortest_completion(m.initial_state), doc):
        if _acc(m, d):
            assert _validate_or_raise(d, loaded) == [], d


def test_null_min_bounds_already_rejected():
    """Régression : la variante min* de la même faute est bien refusée."""
    for s in ({"type": "string", "minLength": None}, {"type": "array", "minItems": None}):
        with pytest.raises(SchemaError):
            load_schema(json.dumps(s))


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT F — nom requis hors `properties` + `additionalProperties` qui n'est pas un
#   schéma ("x", 0, [1]…) : _compile_object fait `addl if isinstance(addl, dict) else
#   (addl is not False)` → valeur LIBRE ; validate_instance, lui, valide la propriété
#   contre "x" → SchemaError levée sur le document que l'automate vient de produire.
#   (Sans `required`, le même additionalProperties lève SchemaError à la compilation.)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("addl", ["x", 0, [1], 1.5], ids=repr)
def test_required_outside_properties_non_schema_additional(addl):
    schema = {"type": "object", "properties": {"b": {}}, "required": ["a"],
              "additionalProperties": addl}
    with pytest.raises(SchemaError):             # cohérence : forme libre refusée
        JSONSchemaMatcher({"type": "object", "additionalProperties": addl})
    try:
        m = JSONSchemaMatcher(schema)
    except SchemaError:
        return                                   # refus explicite : acceptable
    c = m.shortest_completion(m.initial_state)
    assert _validate_or_raise(c, schema) == [], c


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT G — valeurs de mots-clés malformées dans les chemins de FUSION / filtrage
#   d'enum : TypeError / AttributeError au lieu de SchemaError (contrat de load_schema,
#   « Schéma JSON invalide… » → SchemaError, sous-classe de ValueError). chat.py `/json`
#   n'attrape que ValueError/ImportError/OSError → la session de chat plante ;
#   popixa gen --schema affiche une trace au lieu d'un message propre. Les mêmes
#   valeurs SANS fusion sont correctement refusées (SchemaError) ; un type inconnu
#   fusionné est même accepté silencieusement.
# ─────────────────────────────────────────────────────────────────────────────

_MALFORMED = [
    {"minLength": 2, "allOf": [{"minLength": "3"}]},
    {"$ref": "#/$defs/s", "minLength": "2", "$defs": {"s": {"type": "string", "minLength": 1}}},
    {"maxItems": 2, "anyOf": [{"maxItems": [1]}]},
    {"required": ["a"], "allOf": [{"required": 5}]},
    {"required": ["a"], "allOf": [{"required": [["x"]]}]},
    {"properties": {"a": {}}, "anyOf": [{"properties": 5}]},
    {"additionalProperties": False, "allOf": [{"properties": [[1]]}]},
    {"type": "string", "allOf": [{"type": {"a": 1}}]},
    {"type": ["string", "foo"], "allOf": [{"type": "string"}]},
    {"enum": ["a"], "minLength": "x"},
    {"enum": [{"a": 1}], "required": 5},
]


@pytest.mark.parametrize("schema", _MALFORMED, ids=[json.dumps(s)[:48] for s in _MALFORMED])
def test_malformed_keyword_in_merge_raises_schema_error(schema):
    with pytest.raises(SchemaError):
        load_schema(json.dumps(schema))


def test_malformed_keyword_without_merge_is_schema_error():
    """Régression : hors fusion, ces mêmes valeurs sont bien des SchemaError."""
    for s in ({"type": "string", "minLength": "3"}, {"type": ["string", "foo"]},
              {"type": "object", "required": 5}, {"type": "object", "properties": 5},
              {"type": {"a": 1}}):
        with pytest.raises(SchemaError):
            load_schema(json.dumps(s))


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT H — enum : la déduplication (_json_key) jette la FORME des membres égaux
#   au sens JSON. Le docstring promet « chaque valeur est compilée en grammaire exacte »
#   et « les nombres doivent apparaître sous la forme json.dumps(v) (1 ≠ 1.0) » : avec
#   enum [1, 1.0], json.dumps(1.0) == "1.0" est pourtant refusé (seul le 1er membre
#   survit) ; idem [0, -0.0], [[1], [1.0]] et deux objets égaux écrits dans un ordre de
#   clés différent (chacun dans son propre ordre de déclaration).
# ─────────────────────────────────────────────────────────────────────────────

_DEDUP_ENUMS = [[1, 1.0], [1.0, 1], [0, -0.0], [[1], [1.0]],
                [{"a": 1, "b": 2}, {"b": 2, "a": 1}]]


@pytest.mark.parametrize("values", _DEDUP_ENUMS, ids=[json.dumps(v) for v in _DEDUP_ENUMS])
def test_enum_every_member_accepted_in_its_json_dumps_form(values):
    schema = {"enum": values}
    m = JSONSchemaMatcher(schema)
    for v in values:
        for sep in ((',', ':'), None):
            doc = json.dumps(v, separators=sep).encode()
            assert _valid(doc, schema)
            assert _acc(m, doc), doc


def test_enum_dedup_regression_distinct_values_kept():
    """Régression : bool ≠ nombre, chaînes égales en JSON fusionnées sans perte."""
    m = JSONSchemaMatcher({"enum": [1, True, 0, False, "😀", "😀"]})
    for d in (b'1', b'true', b'0', b'false', '"😀"'.encode(), b'"\\ud83d\\ude00"'):
        assert _acc(m, d), d
    assert not _acc(m, b'1.0') and not _acc(m, b'null')


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT I — `id` (identifiant draft-04, équivalent de `$id`) traité comme une
#   ASSERTION inconnue : deux sous-schémas fusionnés (allOf unique / $ref + frères)
#   portant chacun leur `id` → SchemaError « fusion non supportée pour « id » ». Un
#   schéma draft-04 valide (definitions + allOf + required, tous supportés) est refusé
#   alors que `$id` dans la même position est ignoré comme annotation.
# ─────────────────────────────────────────────────────────────────────────────

def test_draft04_id_is_annotation_in_merges():
    base = {"type": "object", "properties": {"n": {"type": "integer"}}}
    for idkey in ("$id", "id"):
        schema = {idkey: "http://example.com/root.json#",
                  "allOf": [{"$ref": "#/definitions/item"}], "required": ["n"],
                  "definitions": {"item": dict(base, **{idkey: "#item"})}}
        m = JSONSchemaMatcher(schema)
        assert _acc(m, b'{"n":1}') and not _acc(m, b'{}') and not _acc(m, b'{"n":"x"}'), idkey


# ─────────────────────────────────────────────────────────────────────────────
# Régressions round 2 (passent)
# ─────────────────────────────────────────────────────────────────────────────

def test_prefix_items_then_items_with_min_items_exhaustive():
    schema = {"type": "array", "prefixItems": [{"type": "string"}, {"type": "null"}],
              "items": {"type": "integer"}, "minItems": 3, "maxItems": 4}
    m = JSONSchemaMatcher(schema)
    pool = ['"s"', 'null', '1']
    for n in range(6):
        for combo in itertools.product(pool, repeat=n):
            doc = ('[' + ','.join(combo) + ']').encode()
            assert _acc(m, doc) == _valid(doc, schema), doc
    _all_prefix_completions_valid(m, schema, [b'["s",null,1]', b'["s",null,1,2]'])


@pytest.mark.parametrize("ref,good,bad", [
    ("#/definitions/a~1b", b'1', b'"s"'), ("#/definitions/a~0b", b'"s"', b'1'),
    ("#/definitions/a%20b", b'null', b'1'), ("#/definitions/%C3%A9", b'true', b'1'),
    ("#/definitions/0", b'"zero"', b'0')])
def test_ref_pointer_escapes_matcher_and_validator(ref, good, bad):
    schema = {"definitions": {"a/b": {"type": "integer"}, "a~b": {"type": "string"},
                              "a b": {"type": "null"}, "é": {"type": "boolean"},
                              "0": {"const": "zero"}},
              "allOf": [{"$ref": ref}]}
    m = JSONSchemaMatcher(schema)
    assert _acc(m, good) and _valid(good, schema)
    assert not _acc(m, bad) and not _valid(bad, schema)


def test_enum_filtered_through_every_merge_path():
    s_def = {"type": "string", "maxLength": 1}
    for schema in ({"$ref": "#/$defs/s", "enum": ["a", "bb", 1, None], "$defs": {"s": s_def}},
                   {"enum": ["a", "bb", 1, None], "allOf": [{"$ref": "#/$defs/s"}], "$defs": {"s": s_def}},
                   {"enum": ["a", "bb", 1, None], "anyOf": [{"$ref": "#/$defs/s"}], "$defs": {"s": s_def}},
                   {"$ref": "#/$defs/e", "maxLength": 1, "type": "string",
                    "$defs": {"e": {"enum": ["a", "bb", 1, None]}}}):
        m = JSONSchemaMatcher(schema)
        assert _acc(m, b'"a"')
        for bad in (b'"bb"', b'1', b'null'):
            assert not _acc(m, bad), (schema, bad)


# Générateur round 2 : insiste sur les FUSIONS ($ref / allOf / anyOf + frères), les
# required hors properties, additionalProperties schéma, items liste + additionalItems.
_C2 = [None, True, False, 0, 1, 2, -1, 1.0, 1.5, "", "a", "ab", "é", "😀", 'q"\\', "/", [],
       [1, "a"], {}, {"k": None}, {"a": 1}, {"a": 1, "b": "x"}, [True], 0.0]
_N2 = ["a", "b", "c", "k"]


def _rs2(rng, d, defs):
    if d > 4:
        return rng.choice([{"type": "integer"}, {"type": "string", "maxLength": 2},
                           {"const": rng.choice(_C2)}, {}])
    r = rng.random() if d <= 2 else rng.random() * 0.55
    if r < 0.06:
        return {"enum": [rng.choice(_C2) for _ in range(rng.randint(0, 4))]}
    if r < 0.09:
        return {"const": rng.choice(_C2)}
    if r < 0.18:
        s = {"type": rng.choice(["string", "integer", "number", "boolean", "null",
                                 ["string", "null"], ["integer", "string"],
                                 ["number", "integer"], ["array", "object"]])}
        if rng.random() < 0.4:
            s["minLength"] = rng.randint(0, 2)
        if rng.random() < 0.4:
            s["maxLength"] = rng.randint(0, 3)
        if rng.random() < 0.2:
            s["enum"] = [rng.choice(_C2) for _ in range(rng.randint(1, 4))]
        return s
    if r < 0.32:
        s = {"properties": {n: _rs2(rng, d + 1, defs)
                            for n in rng.sample(_N2, rng.randint(0, 3))}}
        if rng.random() < 0.7:
            s["type"] = "object"
        req = [n for n in _N2 if rng.random() < 0.3]
        if req:
            s["required"] = req
        if rng.random() < 0.35:
            s["additionalProperties"] = rng.choice([False, True, _rs2(rng, d + 1, defs)])
        return s
    if r < 0.38:
        s = {"type": "object",
             "additionalProperties": rng.choice([False, True, _rs2(rng, d + 1, defs)])}
        if rng.random() < 0.3:
            s["required"] = rng.sample(_N2, 1)
        return s
    if r < 0.50:
        s = {"type": "array"}
        k = rng.random()
        if k < 0.4:
            s["items"] = _rs2(rng, d + 1, defs)
        elif k < 0.6:
            s["prefixItems"] = [_rs2(rng, d + 1, defs) for _ in range(rng.randint(0, 2))]
            if rng.random() < 0.5:
                s["items"] = rng.choice([False, _rs2(rng, d + 1, defs)])
        elif k < 0.8:
            s["items"] = [_rs2(rng, d + 1, defs) for _ in range(rng.randint(0, 2))]
            if rng.random() < 0.6:
                s["additionalItems"] = rng.choice([False, _rs2(rng, d + 1, defs)])
        if rng.random() < 0.5:
            s["minItems"] = rng.randint(0, 2)
        if rng.random() < 0.5:
            s["maxItems"] = rng.randint(0, 3)
        return s
    if r < 0.66:
        s = {rng.choice(["anyOf", "oneOf"]): [_rs2(rng, d + 1, defs)
                                              for _ in range(rng.randint(1, 3))]}
        k = rng.random()
        if k < 0.25:
            s["required"] = rng.sample(_N2, 1)
        elif k < 0.4:
            s["type"] = rng.choice(["object", "string", "array", "integer", "number"])
        elif k < 0.5:
            s["properties"] = {rng.choice(_N2): _rs2(rng, d + 1, defs)}
        elif k < 0.6:
            s["enum"] = [rng.choice(_C2) for _ in range(3)]
        elif k < 0.7:
            s["maxLength"] = 1
        return s
    if r < 0.76:
        s = {"allOf": [_rs2(rng, d + 1, defs)]}
        k = rng.random()
        if k < 0.3:
            s["type"] = rng.choice(["string", "object", "array", "integer", "number", "boolean"])
        elif k < 0.5:
            s["properties"] = {rng.choice(_N2): _rs2(rng, d + 1, defs)}
        elif k < 0.6:
            s["required"] = rng.sample(_N2, 1)
        elif k < 0.7:
            s["const"] = rng.choice(_C2)
        elif k < 0.8:
            s["minItems"] = 1
        return s
    if r < 0.9 and defs:
        s = {"$ref": "#/$defs/" + rng.choice(defs)}
        k = rng.random()
        if k < 0.2:
            s["type"] = rng.choice(["string", "object", "array", "integer"])
        elif k < 0.35:
            s["required"] = rng.sample(_N2, 1)
        elif k < 0.45:
            s["enum"] = [rng.choice(_C2) for _ in range(3)]
        elif k < 0.55:
            s["properties"] = {rng.choice(_N2): _rs2(rng, d + 1, defs)}
        return s
    return rng.choice([{}, True, {"not": {}}, {"minimum": 3}])


def _root2(seed):
    rng = random.Random(seed)
    defs = {"d0": _rs2(rng, 1, ["d0", "d1"]), "d1": _rs2(rng, 1, ["d0"])}
    root = _rs2(rng, 0, ["d0", "d1"])
    root = dict(root) if isinstance(root, dict) else {"allOf": [root]}
    root["$defs"] = defs
    return rng, root


def _walk(rng, m, n):
    st, data = m.initial_state, bytearray()
    for _ in range(n):
        if m.is_accepting(st) and rng.random() < 0.1:
            break
        al = m.allowed_bytes(st)
        if not al:
            break
        pref = [b for b in al if b in b'{}[],:"0123456789-tfn \\u']
        b = rng.choice(pref if pref and rng.random() < 0.85 else al)
        data.append(b)
        st = m.advance(st, b)
    return st, bytes(data)


def test_differential_fuzz_merge_paths():
    built = 0
    for seed in range(400):
        rng, root = _root2(seed)
        try:
            m = JSONSchemaMatcher(root)
        except SchemaError:
            continue
        built += 1
        c = m.shortest_completion(m.initial_state)
        assert c is not None and _valid(c, root), (seed, root, c)
        for _ in range(10):
            st, data = _walk(rng, m, rng.randint(0, 60))
            cc = m.shortest_completion(st)
            assert cc is not None, (seed, data)
            assert _valid(data + cc, root), (seed, root, data + cc)
    assert built > 250


def test_shortest_completion_is_minimal_vs_bfs_oracle():
    """shortest_completion (analytique) == plus court chemin BFS exact, en octets."""
    checked = 0
    for seed in range(60):
        rng, root = _root2(seed)
        try:
            m = JSONSchemaMatcher(root)
        except SchemaError:
            continue
        for _ in range(3):
            st, data = _walk(rng, m, rng.randint(0, 30))
            c = m.shortest_completion(st)
            bf = m._bfs_completion(st, max_depth=48, max_nodes=8000)
            if bf is None:
                continue
            checked += 1
            assert c is not None and len(c) == len(bf), (seed, root, data, c, bf)
    assert checked > 60


# ═════════════════════════════════════════════════════════════════════════════
# ROUND 3 — fusions (mots-clés inconnus, récursion, const ∩ const), valeurs de
#   mots-clés malformées hors du type compilé, oracles de complétude
# ═════════════════════════════════════════════════════════════════════════════

def _build_ok(schema):
    """Le schéma doit compiler ; son instance minimale doit être valide."""
    m = JSONSchemaMatcher(schema)
    c = m.shortest_completion(m.initial_state)
    assert c is not None and _valid(c, schema), (c, schema)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT J — mots-clés INCONNUS (OpenAPI `example`, `nullable`, `discriminator`,
#   extensions `x-*`) traités comme des assertions dans merge() : deux valeurs
#   différentes → SchemaError « fusion de schémas non supportée pour « example » ».
#   Or ni l'automate ni validate_instance ne les appliquent (un schéma isolé qui les
#   porte compile sans broncher ; JSON Schema 2020-12 §6.5 : un mot-clé inconnu se
#   traite comme une annotation). Composition OpenAPI typique (allOf + $ref vers un
#   composant qui a son propre `example`) → load_schema / `popixa gen --schema` refusent.
# ─────────────────────────────────────────────────────────────────────────────

_PET = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"],
        "example": {"name": "Rex"}, "x-tags": ["pets"], "nullable": False,
        "discriminator": {"propertyName": "name"}}


@pytest.mark.parametrize("schema", [
    {"components": {"schemas": {"Pet": _PET}},
     "allOf": [{"$ref": "#/components/schemas/Pet"}],
     "properties": {"age": {"type": "integer"}}, "example": {"name": "Rex", "age": 3}},
    {"allOf": [{"type": "string", "x-order": 1}], "x-order": 2},
    {"$defs": {"a": {"type": "string", "nullable": True}}, "$ref": "#/$defs/a", "nullable": False},
    {"$defs": {"a": {"type": "string", "example": "x"}}, "$ref": "#/$defs/a", "example": "y"},
    {"anyOf": [{"type": "string", "example": "x"}, {"type": "null"}], "example": "y"},
    {"components": {"schemas": {"Pet": _PET}},
     "oneOf": [{"$ref": "#/components/schemas/Pet"}], "discriminator": {"propertyName": "kind"}},
], ids=["openapi-allOf-example", "x-vendor", "nullable", "ref-example", "anyOf-example",
        "oneOf-discriminator"])
def test_unknown_keywords_do_not_block_merges(schema):
    m = _build_ok(schema)
    # mêmes documents qu'avec le schéma débarrassé de ses mots-clés inconnus
    probes = [b'""', b'"y"', b'null', b'{"name":""}', b'{"name":"a","age":3}', b'1']
    for d in probes:
        if _acc(m, d):
            assert _valid(d, schema), d


def test_unknown_keyword_without_merge_regression():
    """Sans fusion, un mot-clé inconnu est (déjà) ignoré : la référence du défaut J."""
    for schema in ({"type": "string", "x-order": 1}, dict(_PET), {"type": "string", "example": "y"}):
        m = _build_ok(schema)
        assert not m.ignored_keywords


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT K — fusion RÉCURSIVE : « $ref → Récursion supportée ; $ref + frères = fusion
#   quand elle est exprimable » et properties / required / type sont fusionnables.
#   Mais dès qu'une propriété surchargée pointe (directement) vers le schéma en cours
#   de fusion, merge()/deref() recopient la cible dans un dict NEUF à chaque niveau
#   (aucun mémo des fusions) → chaîne infinie de fusions → SchemaError « fusion de
#   schémas trop profonde » / « fusion récursive non supportée » sur des schémas de
#   cinq lignes, satisfiables, sans imbrication profonde (sous-type qui resserre sa
#   propriété récursive : liste chaînée typée, catégorie / parent…). Le même schéma
#   avec une propriété surchargée NON récursive compile.
# ─────────────────────────────────────────────────────────────────────────────

_NODE = {"type": "object", "properties": {"v": {"type": "integer"}, "next": {"$ref": "#/$defs/N"}},
         "required": ["v"]}

_RECURSIVE_MERGES = {
    "ref-override-self": {"$defs": {"N": _NODE,
                                    "M": {"$ref": "#/$defs/N",
                                          "properties": {"next": {"$ref": "#/$defs/M"}}}},
                          "$ref": "#/$defs/M"},
    "allOf-extends-parent": {"$defs": {
        "Named": {"type": "object", "properties": {"name": {"type": "string"},
                                                   "parent": {"type": "object"}},
                  "required": ["name"]},
        "Category": {"allOf": [{"$ref": "#/$defs/Named"}],
                     "properties": {"parent": {"$ref": "#/$defs/Category"}}}},
        "$ref": "#/$defs/Category"},
    "allOf-annotation-only-base": {"$defs": {
        "L": {"allOf": [{"type": "object", "properties": {"next": {"description": "lien"}}}],
              "properties": {"next": {"$ref": "#/$defs/L"}}}},
        "$ref": "#/$defs/L"},
}


@pytest.mark.parametrize("name", list(_RECURSIVE_MERGES))
def test_recursive_merge_is_supported(name):
    schema = _RECURSIVE_MERGES[name]
    m = _build_ok(schema)
    good = {"ref-override-self": [b'{"v":1}', b'{"v":1,"next":{"v":2,"next":{"v":3}}}'],
            "allOf-extends-parent": [b'{"name":"a"}', b'{"name":"a","parent":{"name":"b"}}'],
            "allOf-annotation-only-base": [b'{}', b'{"next":{"next":{}}}']}[name]
    bad = {"ref-override-self": [b'{"v":1,"next":{}}', b'{"v":1,"next":{"v":"x"}}'],
           "allOf-extends-parent": [b'{"name":"a","parent":{}}'],
           "allOf-annotation-only-base": [b'{"next":1}']}[name]
    for d in good:
        assert _valid(d, schema) and _acc(m, d), d
    for d in bad:
        assert not _valid(d, schema) and not _acc(m, d), d
    _all_prefix_completions_valid(m, schema, good)


def test_non_recursive_override_regression():
    """Même forme sans récursion : compile déjà (isole la cause du défaut K)."""
    schema = {"$defs": {"N": {"type": "object",
                              "properties": {"v": {"type": "integer"}, "next": {"type": "object"}},
                              "required": ["v"]},
                        "Leaf": {"type": "object", "properties": {"w": {"type": "string"}}}},
              "$ref": "#/$defs/N", "properties": {"next": {"$ref": "#/$defs/Leaf"}}}
    m = _build_ok(schema)
    assert _acc(m, b'{"v":1,"next":{"w":"x"}}')
    assert not _acc(m, b'{"v":1,"next":{"w":1}}')


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT L — const ∩ const : merge() garde la forme d'UN seul côté (`_json_eq` →
#   continue), alors que enum ∩ enum garde les formes des deux côtés (« enum [1] ∩
#   enum [1.0] accepte '1' et '1.0', comme validate_instance ») et que le docstring
#   promet « chaque membre listé garde SA forme ». {"allOf":[{"const":1}],"const":1.0}
#   refuse donc json.dumps(1) == "1" (valide) ; idem l'ordre de clés d'un const objet.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", [
    (1, 1.0), (1.0, 1), (0, -0.0),
    ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
    ([1, {"x": 1, "y": 2}], [1.0, {"y": 2, "x": 1}]),
])
def test_const_intersection_keeps_both_forms(a, b):
    for schema in ({"allOf": [{"const": a}], "const": b},
                   {"$defs": {"c": {"const": a}}, "$ref": "#/$defs/c", "const": b},
                   {"anyOf": [{"const": a}], "const": b}):
        m = _build_ok(schema)
        for v in (a, b):
            d = json.dumps(v, separators=(',', ':')).encode()
            assert _valid(d, schema)
            assert _acc(m, d), (schema, d)


def test_enum_or_mixed_intersection_keeps_both_forms_regression():
    for schema in ({"allOf": [{"enum": [1]}], "enum": [1.0]},
                   {"allOf": [{"enum": [1]}], "const": 1.0},
                   {"allOf": [{"const": 1}], "enum": [1.0]}):
        m = _build_ok(schema)
        assert _acc(m, b'1') and _acc(m, b'1.0'), schema


# ─────────────────────────────────────────────────────────────────────────────
# DÉFAUT M — valeurs de mots-clés malformées ACCEPTÉES dès que le type compilé ne lit
#   pas le mot-clé. Le docstring : « minLength / maxLength / minItems / maxItems non
#   entiers ≥ 0, required qui n'est pas une liste de chaînes, properties qui n'est pas
#   un objet, additionalProperties qui n'est pas un schéma… → SchemaError, avec ou sans
#   fusion (… enum + frères) ». `_compile_types` ne lit minLength que pour 'string',
#   `_compile_enum` ne vérifie les frères que pour le type de chaque valeur → load_schema
#   laisse passer la faute de frappe, et validate_instance lève SchemaError (au lieu
#   d'une liste d'erreurs) sur toute instance du type concerné.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("schema", [
    {"type": "integer", "minLength": "x"},
    {"type": "integer", "maxLength": -1},
    {"type": "string", "minItems": "x"},
    {"type": "string", "items": 5},
    {"type": "string", "properties": [1]},
    {"type": "string", "required": "a"},
    {"type": "string", "additionalProperties": 5},
    {"type": "string", "prefixItems": 5},
    {"enum": [1], "minLength": -3},
    {"enum": ["a"], "properties": 5},
    {"type": "null", "maxItems": "q"},
], ids=lambda s: json.dumps(s, sort_keys=True))
def test_malformed_keyword_rejected_even_if_type_does_not_use_it(schema):
    with pytest.raises(SchemaError):
        JSONSchemaMatcher(schema)


def test_malformed_keyword_on_used_type_regression():
    for schema in ({"type": "string", "minLength": "x"}, {"type": "array", "items": 5},
                   {"type": "object", "required": "a"}, {"enum": ["a"], "minLength": -3}):
        with pytest.raises(SchemaError):
            JSONSchemaMatcher(schema)


# ─────────────────────────────────────────────────────────────────────────────
# Régressions round 3 (passent) — oracles exhaustifs
# ─────────────────────────────────────────────────────────────────────────────

_R3_ALPHA = b'{}[],:"01-.e5tfnrulsabck\\'


def _accepted_docs(m, maxlen, alpha=_R3_ALPHA, limit=60000):
    """Tous les documents acceptés de longueur ≤ maxlen sur l'alphabet (DFS borné)."""
    out, stack, seen = [], [(m.initial_state, b'')], 0
    while stack and seen < limit:
        st, d = stack.pop()
        seen += 1
        if m.is_accepting(st):
            out.append(d)
        if len(d) < maxlen:
            for b in m.allowed_bytes(st):
                if b in alpha:
                    stack.append((m.advance(st, b), d + bytes((b,))))
    return out


def test_exhaustive_short_documents_sound_on_merge_fuzz():
    """Tous les documents courts acceptés (pas seulement des marches aléatoires) sont
    valides, pour les schémas du générateur de fusions du round 2."""
    checked = 0
    for seed in range(150):
        _, root = _root2(seed)
        try:
            m = JSONSchemaMatcher(root, max_whitespace=0)
        except SchemaError:
            continue
        for d in _accepted_docs(m, 7):
            assert _valid(d, root), (seed, root, d)
            checked += 1
    assert checked > 1000


_R3_LEAVES = [None, True, False, 0, 1, -3, 2.5, -0.0, 1e16, 1e-7, "", "a", "é", "😀", "\x00",
              "\x7f", 'q"\\/', " ", [], {}, [1, "x"], {"k": None, "j": [1.5]}]


def _r3_schema(rng, d, defs):
    r = rng.random() if d < 3 else rng.random() * 0.45
    if r < 0.1:
        return {"enum": rng.sample(_R3_LEAVES, rng.randint(1, 3))}
    if r < 0.15:
        return {"const": rng.choice(_R3_LEAVES)}
    if r < 0.3:
        t = rng.choice(["string", "integer", "number", "boolean", "null"])
        s = {"type": t}
        if t == "string" and rng.random() < 0.5:
            s["minLength"] = rng.randint(0, 2)
            s["maxLength"] = s["minLength"] + rng.randint(0, 3)
        return s
    if r < 0.45:
        names = rng.sample(["a", "b", "é", "c/d", 'q"', ""], rng.randint(0, 3))
        s = {"type": "object", "properties": {n: _r3_schema(rng, d + 1, defs) for n in names}}
        req = [n for n in names if rng.random() < 0.5]
        if rng.random() < 0.2:
            req.append("zz")
            s["additionalProperties"] = _r3_schema(rng, d + 1, defs)
        if req:
            s["required"] = req
        return s
    if r < 0.5:
        return {"type": "object", "additionalProperties": _r3_schema(rng, d + 1, defs)}
    if r < 0.65:
        s = {"type": "array"}
        if rng.random() < 0.4:
            s["prefixItems"] = [_r3_schema(rng, d + 1, defs) for _ in range(rng.randint(1, 2))]
            s["items"] = rng.choice([False, _r3_schema(rng, d + 1, defs)])
        else:
            s["items"] = _r3_schema(rng, d + 1, defs)
        s["minItems"] = rng.randint(0, 2)
        if rng.random() < 0.5:
            s["maxItems"] = s["minItems"] + rng.randint(0, 2)
        return s
    if r < 0.78:
        return {rng.choice(["anyOf", "oneOf"]): [_r3_schema(rng, d + 1, defs)
                                                 for _ in range(rng.randint(1, 3))]}
    if r < 0.85:
        return {"allOf": [_r3_schema(rng, d + 1, defs)]}
    if r < 0.93 and defs:
        return {"$ref": "#/$defs/" + rng.choice(defs)}
    return {}


class _NoInst(Exception):
    pass


def _r3_instance(rng, s, root, depth=0):
    """Instance valide tirée DU schéma + sérialisation dans la forme documentée
    (ordre de déclaration, json.dumps pour les feuilles, ensure_ascii au hasard)."""
    if depth > 6:
        raise _NoInst
    asc = rng.random() < 0.5
    if s is True or s is None or s == {}:
        v = rng.choice(_R3_LEAVES)
        return json.dumps(v, ensure_ascii=asc).encode()
    if "$ref" in s:
        return _r3_instance(rng, root["$defs"][s["$ref"].rsplit("/", 1)[1]], root, depth + 1)
    if "allOf" in s:
        return _r3_instance(rng, s["allOf"][0], root, depth + 1)
    for k in ("anyOf", "oneOf"):
        if k in s:
            alts = list(s[k])
            rng.shuffle(alts)
            for a in alts:
                try:
                    return _r3_instance(rng, a, root, depth + 1)
                except _NoInst:
                    pass
            raise _NoInst
    if "enum" in s or "const" in s:
        v = rng.choice(s["enum"] if "enum" in s else [s["const"]])
        return json.dumps(v, ensure_ascii=asc,
                          separators=rng.choice([(',', ':'), (', ', ': ')])).encode()
    t = s.get("type")
    if t == "string":
        n = rng.randint(s.get("minLength", 0), s.get("maxLength", 6))
        v = "".join(rng.choice(["a", "é", "😀", "\n", '"', "\\", "/"]) for _ in range(n))
        return json.dumps(v, ensure_ascii=asc).encode()
    if t in ("integer", "number", "boolean", "null"):
        v = {"integer": [0, -1, 123456], "number": [0, -1.5, 1e-5, 10],
             "boolean": [True, False], "null": [None]}[t]
        return json.dumps(rng.choice(v)).encode()
    if t == "object":
        props, req = s.get("properties"), s.get("required", [])
        ap = s.get("additionalProperties", True)
        if props is None and not req:
            if ap is False:
                return b"{}"
            parts = [json.dumps(k).encode() + b":" + _r3_instance(rng, ap, root, depth + 1)
                     for k in rng.sample(["k", "é", "x y"], rng.randint(0, 2))]
            return b"{" + b",".join(parts) + b"}"
        props = props or {}
        parts = []
        for n in list(props) + [r for r in req if r not in props]:
            if n in req or rng.random() < 0.5:
                try:
                    b = _r3_instance(rng, props.get(n, ap), root, depth + 1)
                except _NoInst:
                    if n in req:
                        raise
                    continue
                parts.append(json.dumps(n, ensure_ascii=asc).encode() + b":" + b)
        return b"{" + b",".join(parts) + b"}"
    if t == "array":
        pre, it = s.get("prefixItems", []), s.get("items", True)
        mn = s.get("minItems", 0)
        n = rng.randint(mn, s.get("maxItems", mn + 3))
        if it is False:
            n = min(n, len(pre))
        if n < mn:
            raise _NoInst
        return b"[" + b",".join(_r3_instance(rng, pre[i] if i < len(pre) else it, root, depth + 1)
                                for i in range(n)) + b"]"
    raise _NoInst


def test_completeness_schema_directed_instances_in_documented_form():
    """Complétude : toute instance VALIDE écrite dans la forme documentée (ordre de
    déclaration, json.dumps, ensure_ascii True/False, séparateurs compacts ou par
    défaut pour enum / const) est acceptée par l'automate."""
    built = tried = 0
    for seed in range(600):
        rng = random.Random(seed)
        root = dict(_r3_schema(rng, 0, ["d0"]))
        root["$defs"] = {"d0": _r3_schema(rng, 1, ["d0"])}
        try:
            m = JSONSchemaMatcher(root)
        except SchemaError:
            continue
        built += 1
        for _ in range(12):
            try:
                doc = _r3_instance(rng, root, root)
            except _NoInst:
                continue
            if not _valid(doc, root):
                continue
            tried += 1
            assert _acc(m, doc), (seed, root, doc)
    assert built > 500 and tried > 3000


def test_no_completion_fallback_on_merge_fuzz():
    """La complétion analytique est toujours vérifiée sans le BFS de secours (borné à
    512 octets : un repli sur une longue complétion rendrait None)."""
    for seed in range(250):
        rng, root = _root2(seed)
        try:
            m = JSONSchemaMatcher(root)
        except SchemaError:
            continue
        for _ in range(8):
            st, data = _walk(rng, m, rng.randint(0, 80))
            assert m.shortest_completion(st) is not None, (seed, data)
        assert m._fallbacks == 0, (seed, root)
