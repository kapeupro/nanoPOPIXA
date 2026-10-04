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
