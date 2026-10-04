"""
nanoPOPIXA — Structured outputs : décodage contraint JSON / JSON Schema
Inspiré du beta Anthropic `structured-outputs-2025-12-15`.

Principe
--------
  1. Le schéma est compilé en une petite grammaire (graphe de nœuds, références
     `$ref` résolues *par référence* → les schémas récursifs ne bouclent jamais).
  2. Un automate à pile NON déterministe lit la sortie octet par octet. Un état
     est un `frozenset` de configurations ; une configuration est une pile
     immuable (tuple de frames). Les états sont donc hashables et canoniques
     (ex. dans une chaîne libre, l'état après n'importe quel octet ordinaire est
     identique) → toutes les transitions sont mémoïsées (dict état → 256 cases).
  3. `TokenConstraint` parcourt un trie d'octets du vocabulaire en propageant les
     états de l'automate (élagage dès qu'un octet est refusé) → masque booléen des
     tokens autorisés, mis en cache par état. `completion_tokens()` ferme le JSON
     au plus court quand le budget de tokens s'épuise.

Sous-ensemble JSON Schema supporté
----------------------------------
  - None / True / {}            → n'importe quelle valeur JSON. False → SchemaError.
  - type                        → 'string' | 'number' | 'integer' | 'boolean' | 'null'
                                  | 'object' | 'array', ou une liste (union).
                                  Sans `type`, le type est déduit des mots-clés présents
                                  (properties → object, items → array, maxLength → string…),
                                  sinon toutes les valeurs sont permises.
  - enum / const                → comparés structurellement (égalité JSON, bool ≠ nombre) :
                                  chaque valeur est compilée en grammaire exacte, donc les
                                  formes `ensure_ascii=True/False`, séparateurs compacts ou
                                  par défaut sont toutes acceptées. Les nombres doivent
                                  apparaître sous la forme `json.dumps(v)` (1 ≠ 1.0).
                                  Les valeurs incompatibles avec les mots-clés frères
                                  (ex. type) sont retirées.
  - object + properties         → propriétés émises dans l'ORDRE DE DÉCLARATION ; celles de
                                  `required` sont obligatoires, les autres optionnelles.
                                  AUCUNE propriété additionnelle n'est jamais générée quand
                                  `properties` (ou `required`) est présent ; un nom requis
                                  absent de `properties` est ajouté en fin de liste avec le
                                  schéma `additionalProperties` (any par défaut).
  - object sans properties      → objet libre : clés = chaînes JSON quelconques, valeurs
                                  conformes à `additionalProperties` (any par défaut) ;
                                  `additionalProperties: false` → uniquement `{}`.
  - array                       → items (schéma unique), prefixItems (ou `items` liste +
                                  `additionalItems`, style draft-04), minItems, maxItems.
  - string                      → minLength / maxLength en points de code Unicode (une
                                  séquence d'échappement compte pour un point de code ; une
                                  paire de substitution \\uD83D\\uDE00 compte pour un).
                                  Chaînes libres : grammaire JSON complète (les constantes
                                  ont un encodage canonique, cf. Limites connues) :
                                  \\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX,
                                  octets de contrôle < 0x20 bruts refusés, UTF-8 validé
                                  (octets de tête / continuation, surrogates et surlongs).
  - number                      → -?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?
    integer                     → -?(0|[1-9][0-9]*)
                                  au plus `max_number_digits` chiffres par série de chiffres.
  - boolean / null              → littéraux.
  - anyOf / oneOf               → union (oneOf est traité comme anyOf : l'exclusivité n'est
                                  ni imposée ni vérifiée). Les mots-clés frères sont fusionnés
                                  dans chaque alternative.
  - allOf                       → exactement UN sous-schéma (fusionné avec les frères),
                                  sinon SchemaError.
  - $ref                        → pointeurs locaux : '#', '#/$defs/<nom>',
                                  '#/definitions/<nom>', pointeur JSON quelconque dans la
                                  racine. Récursion supportée ; `$ref` + mots-clés frères =
                                  fusion (intersection) quand elle est exprimable.
  - Annotations (title, description, default, examples, $schema, $id, $comment,
    deprecated, readOnly, writeOnly…) ignorées silencieusement.
  - Assertions NON appliquées (pattern, format, minimum, maximum, exclusiveMinimum,
    exclusiveMaximum, multipleOf, uniqueItems, minProperties, maxProperties,
    propertyNames, patternProperties, dependentRequired, if/then/else, not, contains…)
    → listées dans `JSONSchemaMatcher.ignored_keywords`. `validate_instance` ne les
    applique pas non plus (une sortie générée est donc toujours valide).
  - Espaces : JSON (espace, \\n, \\r, \\t) uniquement autour des caractères structurels
    (après '{' '[' ',' ':' et avant '}' ']' ',' ':') et AVANT la valeur racine, au plus
    `max_whitespace` octets consécutifs par interstice. Pas d'espace final : une racine
    objet / tableau / chaîne fermée est immédiatement terminale.
  - Un schéma sans aucune instance finie (ex. récursion obligatoire infinie) → SchemaError.

Limites connues
---------------
  - Chaînes CONSTANTES (noms de propriétés déclarés dans properties / required, valeurs
    chaînes d'enum / const) : encodage canonique par caractère, pour qu'un modèle imparfait
    ne puisse pas écrire une clé illisible comme "\\u006Eom" au lieu de "nom" :
      · ASCII imprimable (0x20..0x7E) hors '"' et '\\' → forme brute UNIQUEMENT
        (ni \\uXXXX, ni \\/ pour '/') ;
      · '"' et '\\'                       → \\" et \\\\ uniquement ;
      · contrôle (< 0x20)                 → échappement court s'il existe
                                            (\\b \\f \\n \\r \\t), sinon \\u00XX ;
      · DEL (0x7F) et non-ASCII           → UTF-8 brut OU \\uXXXX (paire de substitution
                                            pour les caractères astraux).
    Les sorties json.dumps(..., ensure_ascii=True) et ensure_ascii=False restent donc
    acceptées. Les chaînes LIBRES (type string, clés d'objets libres) gardent la grammaire
    JSON complète ; `validate_instance` (sémantique, pas syntaxe) n'est pas concerné.
  - Les échappements \\uXXXX sont acceptés en hexadécimal minuscule ou majuscule.
  - L'ordre des propriétés est imposé (ordre de déclaration) ; `validate_instance`, lui,
    accepte n'importe quel ordre et les propriétés additionnelles si le schéma les permet.
  - Fusions de schémas (allOf / $ref + frères / anyOf + frères) limitées aux cas
    exprimables sans perte (type, required, properties, bornes min/max, enum) ; les
    autres combinaisons lèvent SchemaError plutôt que de générer une sortie invalide.
"""

from __future__ import annotations

import json
import os
from collections import deque
from urllib.parse import unquote

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Erreurs & mots-clés
# ─────────────────────────────────────────────────────────────────────────────

class SchemaError(ValueError):
    """Schéma JSON invalide, non supporté ou insatisfiable."""


# Annotations — ignorées silencieusement (n'affectent pas la validité)
_ANNOTATIONS = frozenset({
    'title', 'description', 'default', 'examples', '$schema', '$id', '$comment',
    'deprecated', 'readOnly', 'writeOnly', '$defs', 'definitions', '$anchor',
    '$vocabulary', 'contentMediaType', 'contentEncoding', 'contentSchema',
})

# Assertions reconnues mais NON appliquées — rapportées dans `ignored_keywords`
_UNENFORCED = frozenset({
    'pattern', 'format', 'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum',
    'multipleOf', 'uniqueItems', 'minProperties', 'maxProperties', 'propertyNames',
    'patternProperties', 'dependentRequired', 'dependentSchemas', 'dependencies',
    'if', 'then', 'else', 'not', 'contains', 'minContains', 'maxContains',
    'unevaluatedProperties', 'unevaluatedItems',
})

_TYPES = ('null', 'boolean', 'object', 'array', 'number', 'integer', 'string')

# Mots-clés qui contiennent des sous-schémas (pour le scan des mots-clés ignorés)
_SUB_MAPS = ('properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas')
_SUB_ONE = ('additionalProperties', 'items', 'additionalItems', 'contains', 'propertyNames',
            'not', 'if', 'then', 'else', 'unevaluatedProperties', 'unevaluatedItems')
_SUB_LISTS = ('allOf', 'anyOf', 'oneOf', 'prefixItems', 'items')

_MAX_COMPILE_DEPTH = 100    # imbrication maximale à la compilation (fusions récursives)
_MAX_MERGE_DEPTH = 32       # chaînes $ref / fusions imbriquées
_MAX_VALIDATE_DEPTH = 400   # profondeur de validation (instances + $ref)


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulaire → octets par token
# ─────────────────────────────────────────────────────────────────────────────

def token_bytes_from_tiktoken(enc) -> list:
    """
    Octets de chaque token d'un encodeur tiktoken (longueur = enc.n_vocab).
    None pour les tokens spéciaux (<|endoftext|>…) et les ids non attribués.
    """
    special = set()
    for name in getattr(enc, 'special_tokens_set', ()) or ():
        try:
            special.add(enc.encode_single_token(name))
        except (KeyError, ValueError):
            pass
    out = []
    for i in range(enc.n_vocab):
        if i in special:
            out.append(None)
            continue
        try:
            b = enc.decode_single_token_bytes(i)
        except (KeyError, ValueError):
            b = None
        out.append(b if b else None)
    return out


def token_bytes_from_itos(itos: dict, vocab_size: int) -> list:
    """
    Vocabulaire char-level (data_prep.py) : itos[i].encode('utf-8').
    Ids absents (padding du modèle) ou chaînes non encodables → None.
    """
    out = []
    for i in range(vocab_size):
        try:
            s = itos[i]
        except (KeyError, IndexError):
            out.append(None)
            continue
        try:
            b = s.encode('utf-8') if isinstance(s, str) else bytes(s)
        except (UnicodeEncodeError, TypeError, ValueError):
            b = None
        out.append(b if b else None)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Chargement de schéma
# ─────────────────────────────────────────────────────────────────────────────

def load_schema(source: str) -> dict:
    """
    Charge un schéma depuis un chemin de fichier .json OU une chaîne JSON inline.
    Le schéma est compilé à blanc pour détecter tôt les erreurs (SchemaError).
    `true` → {} ; `false`, JSON invalide, fichier illisible → SchemaError.
    """
    if not isinstance(source, str) or not source.strip():
        raise SchemaError("schéma vide : chemin de fichier ou JSON inline attendu")
    text = source
    path = os.path.expanduser(source.strip())
    if os.path.isfile(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        except (OSError, UnicodeDecodeError) as e:
            raise SchemaError(f"lecture du schéma impossible ({path}) : {e}") from e
    try:
        schema = json.loads(text)
    except ValueError as e:
        if text is source and not source.lstrip().startswith(('{', '[', 't', 'f', 'n')):
            raise SchemaError(f"fichier introuvable et JSON inline invalide : {source!r}") from e
        raise SchemaError(f"JSON invalide dans le schéma : {e}") from e
    if schema is True:
        schema = {}
    if schema is False:
        raise SchemaError("schéma `false` : aucune instance possible")
    if not isinstance(schema, dict):
        raise SchemaError(f"un schéma doit être un objet JSON, obtenu {type(schema).__name__}")
    JSONSchemaMatcher(schema)   # validation à blanc (lève SchemaError)
    return schema


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires JSON
# ─────────────────────────────────────────────────────────────────────────────

def _json_eq(a, b) -> bool:
    """Égalité JSON : bool ≠ nombre, 1 == 1.0, comparaison profonde."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str) and a == b
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_eq(a[k], b[k]) for k in a)
    return False


def _is_json_value(v) -> bool:
    """Valeur sérialisable en JSON strict (pas de NaN/Infinity, clés chaînes)."""
    if v is None or isinstance(v, (bool, int, str)):
        return True
    if isinstance(v, float):
        return v == v and v not in (float('inf'), float('-inf'))
    if isinstance(v, list):
        return all(_is_json_value(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _is_json_value(x) for k, x in v.items())
    return False


def _json_type(v) -> str:
    if v is None:
        return 'null'
    if isinstance(v, bool):
        return 'boolean'
    if isinstance(v, int):
        return 'integer'
    if isinstance(v, float):
        return 'number'
    if isinstance(v, str):
        return 'string'
    if isinstance(v, list):
        return 'array'
    if isinstance(v, dict):
        return 'object'
    return type(v).__name__


def _type_ok(v, t) -> bool:
    if t == 'null':
        return v is None
    if t == 'boolean':
        return isinstance(v, bool)
    if t == 'number':
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == 'integer':
        if isinstance(v, bool):
            return False
        return isinstance(v, int) or (isinstance(v, float) and v.is_integer())
    if t == 'string':
        return isinstance(v, str)
    if t == 'array':
        return isinstance(v, list)
    if t == 'object':
        return isinstance(v, dict)
    return False


def _resolve_ref(root, ref):
    """Résout un $ref local ('#', '#/$defs/x', pointeur JSON). SchemaError sinon."""
    if not isinstance(ref, str) or not ref.startswith('#'):
        raise SchemaError(f"$ref non local non supporté : {ref!r}")
    frag = ref[1:]
    if frag == '':
        return root
    if not frag.startswith('/'):
        raise SchemaError(f"$ref par ancre non supporté : {ref!r}")
    cur = root
    for tok in frag[1:].split('/'):
        tok = unquote(tok).replace('~1', '/').replace('~0', '~')
        if isinstance(cur, dict) and tok in cur:
            cur = cur[tok]
        elif isinstance(cur, list) and tok.isdigit() and int(tok) < len(cur):
            cur = cur[int(tok)]
        else:
            raise SchemaError(f"$ref introuvable : {ref!r}")
    return cur


def _has_assertions(schema: dict) -> bool:
    return any(k not in _ANNOTATIONS for k in schema)


def _strip(schema: dict, keys) -> dict:
    return {k: v for k, v in schema.items() if k not in keys}


def _nonneg(schema: dict, key: str, default):
    v = schema.get(key, default)
    if v is default:
        return v
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise SchemaError(f"{key} doit être un entier ≥ 0, obtenu {v!r}")
    return v


def _scan_ignored(schema) -> set:
    """Assertions non appliquées présentes n'importe où dans le document de schéma."""
    found, seen, stack = set(), set(), [schema]
    while stack:
        s = stack.pop()
        if not isinstance(s, dict) or id(s) in seen:
            continue
        seen.add(id(s))
        for k, v in s.items():
            if k in _UNENFORCED:
                found.add(k)
            if k in _SUB_MAPS and isinstance(v, dict):
                stack.extend(v.values())
            elif k in _SUB_ONE or k in _SUB_LISTS:
                if isinstance(v, list):
                    stack.extend(v)
                else:
                    stack.append(v)
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Validation d'instance (indépendante de l'automate)
# ─────────────────────────────────────────────────────────────────────────────

class _Validator:
    """Validateur direct sur le schéma brut — même sous-ensemble que l'automate."""

    def __init__(self, root):
        self.root = root

    def errors(self, inst, sch, path, depth) -> list:
        errs = []
        self.run(inst, sch, path, errs, depth)
        return errs

    def run(self, inst, sch, path, errs, depth):
        if depth > _MAX_VALIDATE_DEPTH:
            errs.append(f"{path} : récursion trop profonde")
            return
        if sch is None or sch is True:
            return
        if sch is False:
            errs.append(f"{path} : aucune valeur autorisée (schéma false)")
            return
        if not isinstance(sch, dict):
            raise SchemaError(f"sous-schéma invalide en {path} : {sch!r}")

        if '$ref' in sch:
            self.run(inst, _resolve_ref(self.root, sch['$ref']), path, errs, depth + 1)
        for sub in sch.get('allOf', ()) or ():
            self.run(inst, sub, path, errs, depth + 1)
        for key in ('anyOf', 'oneOf'):
            if key in sch:
                alts = sch[key] or []
                if not any(not self.errors(inst, a, path, depth + 1) for a in alts):
                    errs.append(f"{path} : aucune alternative de {key} ne correspond")
        if 'enum' in sch and not any(_json_eq(inst, v) for v in sch['enum']):
            errs.append(f"{path} : valeur hors enum")
        if 'const' in sch and not _json_eq(inst, sch['const']):
            errs.append(f"{path} : valeur différente de const")
        if 'type' in sch:
            types = sch['type'] if isinstance(sch['type'], list) else [sch['type']]
            if not any(_type_ok(inst, t) for t in types):
                errs.append(f"{path} : type attendu {'|'.join(types)}, obtenu {_json_type(inst)}")

        if isinstance(inst, str):
            n = len(inst)
            if 'minLength' in sch and n < sch['minLength']:
                errs.append(f"{path} : chaîne trop courte ({n} < {sch['minLength']})")
            if 'maxLength' in sch and n > sch['maxLength']:
                errs.append(f"{path} : chaîne trop longue ({n} > {sch['maxLength']})")
        elif isinstance(inst, dict):
            props = sch.get('properties') or {}
            for name, sub in props.items():
                if name in inst:
                    self.run(inst[name], sub, f"{path}.{name}", errs, depth + 1)
            for name in sch.get('required', ()) or ():
                if name not in inst:
                    errs.append(f"{path} : propriété requise manquante « {name} »")
            addl = sch.get('additionalProperties', True)
            if addl is not True and addl is not None:
                for name, v in inst.items():
                    if name in props:
                        continue
                    if addl is False:
                        errs.append(f"{path} : propriété non autorisée « {name} »")
                    else:
                        self.run(v, addl, f"{path}.{name}", errs, depth + 1)
        elif isinstance(inst, list):
            items = sch.get('items', True)
            prefix = sch.get('prefixItems') or []
            if isinstance(items, list):
                prefix, items = items, sch.get('additionalItems', True)
            for i, v in enumerate(inst):
                sub = prefix[i] if i < len(prefix) else items
                if sub is not True and sub is not None:
                    self.run(v, sub, f"{path}[{i}]", errs, depth + 1)
            if 'minItems' in sch and len(inst) < sch['minItems']:
                errs.append(f"{path} : trop peu d'éléments ({len(inst)} < {sch['minItems']})")
            if 'maxItems' in sch and len(inst) > sch['maxItems']:
                errs.append(f"{path} : trop d'éléments ({len(inst)} > {sch['maxItems']})")


def validate_instance(instance, schema) -> list:
    """
    Valide une instance (déjà décodée par json.loads) pour le sous-ensemble supporté.
    Retourne la liste des erreurs (vide = valide). Les assertions non appliquées
    (pattern, format, minimum…) sont ignorées, comme dans l'automate.
    """
    if schema is False:
        return ["$ : aucune valeur autorisée (schéma false)"]
    return _Validator(schema).errors(instance, schema, '$', 0)


# ─────────────────────────────────────────────────────────────────────────────
# Compilation du schéma → grammaire (graphe de nœuds)
# ─────────────────────────────────────────────────────────────────────────────

# Nœuds :  (_N_UNION, enfants) · (_N_STR, minLen, maxLen) · (_N_NUM, entier)
#          (_N_LIT, octets) · (_N_CSTR, sid) · (_N_OBJ, props) · (_N_FOBJ, valeur)
#          (_N_ARR, prefix, items, minItems, maxItems)     (-1 = aucun nœud)
_N_UNION, _N_STR, _N_NUM, _N_LIT, _N_CSTR, _N_OBJ, _N_FOBJ, _N_ARR = range(8)

_LIT_NULL, _LIT_TRUE, _LIT_FALSE = (_N_LIT, b'null'), (_N_LIT, b'true'), (_N_LIT, b'false')


class _Compiler:
    """Compile un schéma en nœuds. Mémo par identité de dict → récursion sans boucle."""

    def __init__(self, root):
        self.root = root
        self.nodes = []
        self.memo = {}       # id(dict de schéma) → nid
        self.hcons = {}      # nœud (tuple) → nid  (partage des nœuds identiques)
        self.keep = []       # dicts fusionnés gardés en vie (ids stables)
        self.strings = []    # sid → chaîne constante
        self.sids = {}
        self.depth = 0
        self._any = None

    # ── Allocation ───────────────────────────────────────────────────────────

    def new(self, node=None) -> int:
        self.nodes.append(node)
        return len(self.nodes) - 1

    def cons(self, node) -> int:
        nid = self.hcons.get(node)
        if nid is None:
            nid = self.new(node)
            self.hcons[node] = nid
        return nid

    def sid(self, s: str) -> int:
        i = self.sids.get(s)
        if i is None:
            i = len(self.strings)
            self.strings.append(s)
            self.sids[s] = i
        return i

    def never(self) -> int:
        return self.cons((_N_UNION, ()))

    def any(self) -> int:
        if self._any is None:
            u = self._any = self.new()
            self.nodes[u] = (_N_UNION, (
                self.cons((_N_STR, 0, None)), self.cons((_N_NUM, 0)),
                self.cons((_N_FOBJ, u)), self.cons((_N_ARR, (), u, 0, None)),
                self.cons(_LIT_NULL), self.cons(_LIT_TRUE), self.cons(_LIT_FALSE),
            ))
        return self._any

    # ── Compilation ──────────────────────────────────────────────────────────

    def compile(self, schema) -> int:
        if schema is None or schema is True:
            return self.any()
        if schema is False:
            return self.never()
        if not isinstance(schema, dict):
            raise SchemaError(f"sous-schéma invalide : {schema!r}")
        nid = self.memo.get(id(schema))
        if nid is not None:
            return nid
        self.depth += 1
        if self.depth > _MAX_COMPILE_DEPTH:
            raise SchemaError("schéma trop profond ou fusion récursive non supportée")
        try:
            nid = self.new()
            self.memo[id(schema)] = nid
            self.nodes[nid] = self._compile_dict(schema)
        finally:
            self.depth -= 1
        return nid

    def compile_new(self, schema) -> int:
        self.keep.append(schema)
        return self.compile(schema)

    def _compile_dict(self, S: dict):
        if '$ref' in S:
            target = _resolve_ref(self.root, S['$ref'])
            rest = _strip(S, ('$ref',))
            if not _has_assertions(rest):
                return (_N_UNION, (self.compile(target),))
            return (_N_UNION, (self.compile_new(self.merge(target, rest)),))

        if 'allOf' in S:
            subs = S['allOf']
            if not isinstance(subs, list) or len(subs) != 1:
                raise SchemaError("allOf : exactement un sous-schéma est supporté")
            rest = _strip(S, ('allOf',))
            if not _has_assertions(rest):
                return (_N_UNION, (self.compile(subs[0]),))
            return (_N_UNION, (self.compile_new(self.merge(rest, subs[0])),))

        for key in ('anyOf', 'oneOf'):
            if key in S:
                alts = S[key]
                if not isinstance(alts, list) or not alts:
                    raise SchemaError(f"{key} doit être une liste non vide")
                rest = _strip(S, (key,))
                if _has_assertions(rest):
                    kids = [self.compile_new(self.merge(rest, a)) for a in alts]
                else:
                    kids = [self.compile(a) for a in alts]
                return (_N_UNION, tuple(kids))

        if 'enum' in S or 'const' in S:
            return self._compile_enum(S)
        return self._compile_types(S)

    def _compile_enum(self, S: dict):
        vals = None
        if 'enum' in S:
            if not isinstance(S['enum'], list):
                raise SchemaError("enum doit être une liste")
            vals = list(S['enum'])
        if 'const' in S:
            c = S['const']
            vals = [c] if vals is None else [v for v in vals if _json_eq(v, c)]
        rest = _strip(S, ('enum', 'const'))
        kept, kids = [], []
        for v in vals:
            if not _is_json_value(v) or any(_json_eq(v, w) for w in kept):
                continue
            if _Validator(self.root).errors(v, rest, '$', 0):
                continue    # incompatible avec les mots-clés frères (type…)
            kept.append(v)
            kids.append(self.const(v))
        return (_N_UNION, tuple(kids))

    def const(self, v) -> int:
        """Valeur constante → grammaire exacte (espaces libres, échappements canoniques)."""
        if v is None:
            return self.cons(_LIT_NULL)
        if v is True:
            return self.cons(_LIT_TRUE)
        if v is False:
            return self.cons(_LIT_FALSE)
        if isinstance(v, (int, float)):
            return self.cons((_N_LIT, json.dumps(v).encode('ascii')))
        if isinstance(v, str):
            return self.cons((_N_CSTR, self.sid(v)))
        if isinstance(v, list):
            return self.cons((_N_ARR, tuple(self.const(x) for x in v), -1, len(v), len(v)))
        return self.cons((_N_OBJ, tuple((k, self.sid(k), self.const(x), True)
                                        for k, x in v.items())))

    def _compile_types(self, S: dict):
        t = S.get('type')
        if t is None:
            types = self._infer_types(S)
        elif isinstance(t, str):
            types = [t]
        elif isinstance(t, list):
            types = list(t)
        else:
            raise SchemaError(f"type invalide : {t!r}")
        for x in types:
            if x not in _TYPES:
                raise SchemaError(f"type inconnu : {x!r}")
        if 'number' in types:
            types = [x for x in types if x != 'integer']
        kids = []
        for x in dict.fromkeys(types):
            if x == 'null':
                kids.append(self.cons(_LIT_NULL))
            elif x == 'boolean':
                kids += [self.cons(_LIT_TRUE), self.cons(_LIT_FALSE)]
            elif x == 'number':
                kids.append(self.cons((_N_NUM, 0)))
            elif x == 'integer':
                kids.append(self.cons((_N_NUM, 1)))
            elif x == 'string':
                kids.append(self.cons((_N_STR, _nonneg(S, 'minLength', 0),
                                       _nonneg(S, 'maxLength', None))))
            elif x == 'object':
                kids.append(self._compile_object(S))
            else:
                kids.append(self._compile_array(S))
        return (_N_UNION, tuple(kids))

    @staticmethod
    def _infer_types(S: dict) -> list:
        types = []
        if any(k in S for k in ('properties', 'required', 'additionalProperties')):
            types.append('object')
        if any(k in S for k in ('items', 'prefixItems', 'additionalItems', 'minItems', 'maxItems')):
            types.append('array')
        if any(k in S for k in ('minLength', 'maxLength', 'pattern')):
            types.append('string')
        if any(k in S for k in ('minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum',
                                'multipleOf')):
            types.append('number')
        return types or list(_TYPES)

    def _compile_object(self, S: dict) -> int:
        props = S.get('properties')
        req = S.get('required', [])
        addl = S.get('additionalProperties', True)
        if props is not None and not isinstance(props, dict):
            raise SchemaError("properties doit être un objet")
        if not isinstance(req, list) or not all(isinstance(r, str) for r in req):
            raise SchemaError("required doit être une liste de chaînes")
        if addl is None:
            addl = True
        if props is None and not req:
            # Objet libre : clés quelconques, valeurs selon additionalProperties
            if addl is False:
                return self.cons((_N_FOBJ, -1))
            return self.cons((_N_FOBJ, self.compile(addl)))
        # `properties` présent (même vide) : jamais de propriété additionnelle générée
        props = props or {}
        reqset = set(req)
        plist = [(name, self.sid(name), self.compile(sub), name in reqset)
                 for name, sub in props.items()]
        for name in dict.fromkeys(req):
            if name not in props:
                sub = addl if isinstance(addl, dict) else (addl is not False)
                plist.append((name, self.sid(name), self.compile(sub), True))
        return self.new((_N_OBJ, tuple(plist)))

    def _compile_array(self, S: dict) -> int:
        items = S.get('items', True)
        prefix = S.get('prefixItems')
        if isinstance(items, list):
            prefix, items = items, S.get('additionalItems', True)
        if prefix is not None and not isinstance(prefix, list):
            raise SchemaError("prefixItems doit être une liste")
        pre = tuple(self.compile(x) for x in (prefix or ()))
        it = -1 if items is False else self.compile(items)
        mn = _nonneg(S, 'minItems', 0)
        mx = _nonneg(S, 'maxItems', None)
        return self.cons((_N_ARR, pre, it, mn, mx))

    # ── Fusion de schémas (intersection exprimable) ──────────────────────────

    def deref(self, s, depth=0):
        """Remplace un $ref de tête par sa cible (fusionnée avec ses mots-clés frères)."""
        while isinstance(s, dict) and '$ref' in s:
            if depth > _MAX_MERGE_DEPTH:
                raise SchemaError("chaîne de $ref trop longue ou cyclique")
            target = _resolve_ref(self.root, s['$ref'])
            rest = _strip(s, ('$ref',))
            s = self.merge(target, rest, depth + 1) if _has_assertions(rest) else target
            depth += 1
        return s

    def merge(self, a, b, depth=0):
        if depth > _MAX_MERGE_DEPTH:
            raise SchemaError("fusion de schémas trop profonde")
        a, b = self.deref(a, depth), self.deref(b, depth)
        if a is None or a is True:
            return b
        if b is None or b is True:
            return a
        if a is False or b is False:
            return False
        if not isinstance(a, dict) or not isinstance(b, dict):
            raise SchemaError("sous-schéma invalide dans une fusion")
        for x, y in ((a, b), (b, a)):
            # additionalProperties d'un côté + propriétés inconnues de l'autre → non exprimable
            ap = x.get('additionalProperties', True)
            if ap is not True and ap is not None:
                if not set(y.get('properties', {})) <= set(x.get('properties', {})):
                    raise SchemaError("fusion non supportée : additionalProperties "
                                      "et propriétés d'un autre sous-schéma")
        ia = {k: a[k] for k in ('items', 'prefixItems', 'additionalItems') if k in a}
        ib = {k: b[k] for k in ('items', 'prefixItems', 'additionalItems') if k in b}
        if ia and ib and ia != ib:
            raise SchemaError("fusion non supportée : items / prefixItems différents")

        out = dict(a)
        for k, vb in b.items():
            if k not in out:
                out[k] = vb
                continue
            va = out[k]
            if va == vb or k in _ANNOTATIONS or k in _UNENFORCED:
                continue
            if k == 'type':
                ta = set(va if isinstance(va, list) else [va])
                tb = set(vb if isinstance(vb, list) else [vb])
                for t1, t2 in ((ta, tb), (tb, ta)):
                    if 'number' in t1 and 'integer' in t2:
                        t1.add('integer')
                out[k] = [t for t in _TYPES if t in ta and t in tb]
            elif k == 'required':
                out[k] = list(dict.fromkeys(list(va) + list(vb)))
            elif k == 'properties':
                merged = dict(va)
                for name, sub in vb.items():
                    merged[name] = self.merge(va[name], sub, depth + 1) if name in va else sub
                out[k] = merged
            elif k in ('minLength', 'minItems'):
                out[k] = max(va, vb)
            elif k in ('maxLength', 'maxItems'):
                out[k] = min(va, vb)
            elif k == 'enum':
                out[k] = [v for v in va if any(_json_eq(v, w) for w in vb)]
            elif k == 'const':
                if not _json_eq(va, vb):
                    out['enum'] = []
            else:
                raise SchemaError(f"fusion de schémas non supportée pour « {k} »")
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Automate : frames, phases, tables d'octets
# ─────────────────────────────────────────────────────────────────────────────

# Frames (éléments de pile) :
#   (_F_ROOT, ws)                                   avant la valeur racine
#   (_F_STR, minLen, maxLen, compte, sous_état, haut_en_attente)
#   (_F_CSTR, sid, index_caractère, octets_partiels)  chaîne constante (clé, enum)
#   (_F_NUM, entier, phase, chiffres)
#   (_F_LIT, octets, index)                         true / false / null / nombre constant
#   (_F_OBJ, nid, phase, position, ws)              objet à propriétés ordonnées
#   (_F_FOBJ, nid, phase, ws)                       objet libre
#   (_F_ARR, nid, phase, compte, ws)
_F_ROOT, _F_STR, _F_CSTR, _F_NUM, _F_LIT, _F_OBJ, _F_FOBJ, _F_ARR = range(8)

_O_OPEN, _O_KEY, _O_AKEY, _O_ACOLON, _O_VAL, _O_AVAL, _O_ACOMMA = range(7)
_A_OPEN, _A_VAL, _A_AVAL, _A_ACOMMA = range(4)

_P_MINUS, _P_ZERO, _P_INT, _P_DOT, _P_FRAC, _P_E, _P_ESIGN, _P_EXP = range(8)
_P_COMPLETE = (False, True, True, False, True, False, False, True)
_P_FINISH = (b'0', b'', b'', b'0', b'', b'0', b'0', b'')

# Sous-états de chaîne : échappements (\, \u + 0..3 chiffres hexa, variantes H/L pour
# suivre les paires de substitution) et continuations UTF-8.
(_S_NORM, _S_BS, _S_BSL, _S_U0, _S_U0L, _S_U1D, _S_U1L, _S_U1X,
 _S_U2H, _S_U2L, _S_U2X, _S_U3H, _S_U3L, _S_U3X,
 _S_C1, _S_C2, _S_C3, _S_E0, _S_ED, _S_F0, _S_F4) = range(21)

# Octets minimaux pour terminer le caractère en cours, par sous-état
_S_FINISH = {
    _S_NORM: b'', _S_BS: b'n', _S_BSL: b'udc00', _S_U0: b'0000', _S_U0L: b'dc00',
    _S_U1D: b'000', _S_U1L: b'c00', _S_U1X: b'000', _S_U2H: b'00', _S_U2L: b'00',
    _S_U2X: b'00', _S_U3H: b'0', _S_U3L: b'0', _S_U3X: b'0',
    _S_C1: b'\x80', _S_C2: b'\x80\x80', _S_C3: b'\x80\x80\x80', _S_E0: b'\xa0\x80',
    _S_ED: b'\x80\x80', _S_F0: b'\x90\x80\x80', _S_F4: b'\x80\x80\x80',
}
# Sous-états où terminer le caractère forme une paire de substitution (compte − 1)
_S_PAIRING = frozenset({_S_BSL, _S_U0L, _S_U1L, _S_U2L, _S_U3L})

# Continuations UTF-8 : sous-état → (min, max, sous-état suivant)
_UTF8_CONT = {
    _S_C1: (0x80, 0xBF, _S_NORM), _S_C2: (0x80, 0xBF, _S_C1), _S_C3: (0x80, 0xBF, _S_C2),
    _S_E0: (0xA0, 0xBF, _S_C1), _S_ED: (0x80, 0x9F, _S_C1),
    _S_F0: (0x90, 0xBF, _S_C2), _S_F4: (0x80, 0x8F, _S_C2),
}


def _lead_table():
    """Sous-état après un octet de début de caractère (None = interdit)."""
    t = [None] * 256
    for b in range(0x20, 0x80):
        t[b] = _S_NORM
    for b in range(0xC2, 0xE0):
        t[b] = _S_C1
    t[0xE0] = _S_E0
    for b in list(range(0xE1, 0xED)) + [0xEE, 0xEF]:
        t[b] = _S_C2
    t[0xED] = _S_ED
    t[0xF0] = _S_F0
    for b in range(0xF1, 0xF4):
        t[b] = _S_C3
    t[0xF4] = _S_F4
    return tuple(t)


_LEAD = _lead_table()
_IS_WS = tuple(b in b' \n\r\t' for b in range(256))
_IS_HEX = tuple(b in b'0123456789abcdefABCDEF' for b in range(256))
_SHORT_ESC = frozenset(b'"\\/bfnrt')
_BYTE = tuple(bytes((i,)) for i in range(256))
_FREE_STR = (_F_STR, 0, None, 0, _S_NORM, 0)
_UNSET = object()


# Échappements courts admis dans une chaîne constante ('/' en est exclu : brut seulement)
_CONST_SHORT_ESC = {'"': b'\\"', '\\': b'\\\\', '\b': b'\\b', '\f': b'\\f',
                    '\n': b'\\n', '\r': b'\\r', '\t': b'\\t'}


def _char_encodings(ch: str) -> tuple:
    """
    Encodages JSON acceptés pour un caractère d'une chaîne constante (nom de propriété
    déclaré, valeur chaîne d'enum / const), le plus court en premier. L'hexadécimal est
    stocké en minuscule (`_step_cstr` normalise la casse à la lecture).

      '"' et '\\'                     → échappement court uniquement (\\" \\\\)
      contrôle < 0x20                → échappement court s'il existe (\\b \\f \\n \\r \\t),
                                       sinon \\u00XX
      ASCII imprimable 0x20..0x7E    → forme brute UNIQUEMENT (ni \\uXXXX, ni \\/)
      DEL 0x7F et non-ASCII          → UTF-8 brut OU \\uXXXX (paire de substitution pour
                                       les astraux) : json.dumps(ensure_ascii=True/False)
      surrogate isolé                → \\uXXXX uniquement (aucun UTF-8 valide)

    Un modèle imparfait ne peut donc plus écrire une clé illisible comme "\\u006Eom".
    """
    short = _CONST_SHORT_ESC.get(ch)
    if short is not None:
        return (short,)
    cp = ord(ch)
    if cp < 0x20 or 0xD800 <= cp <= 0xDFFF:
        return (b'\\u%04x' % cp,)
    if cp < 0x7F:
        return (_BYTE[cp],)
    if cp < 0x10000:
        esc = b'\\u%04x' % cp
    else:
        v = cp - 0x10000
        esc = b'\\u%04x\\u%04x' % (0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF))
    return (ch.encode('utf-8'), esc)     # brut (1-4 octets) plus court que \uXXXX (≥ 6)


def _minb(*opts):
    """Plus courte option (puis ordre lexicographique) parmi les non-None."""
    best = None
    for o in opts:
        if o is not None and (best is None or (len(o), o) < (len(best), best)):
            best = o
    return best


# ─────────────────────────────────────────────────────────────────────────────
# JSONSchemaMatcher — automate octet par octet
# ─────────────────────────────────────────────────────────────────────────────

class JSONSchemaMatcher:
    """
    Byte-level incremental matcher. States are IMMUTABLE and HASHABLE (so they can be
    memoized/cached).

    Automate à pile non déterministe : un état = frozenset de configurations (piles
    immuables). Les états sont internés et toutes les transitions mémoïsées.
    """

    def __init__(self, schema=None, *, max_whitespace: int = 4, max_number_digits: int = 20):
        if schema is False:
            raise SchemaError("schéma `false` : aucune instance possible")
        if schema is not None and not isinstance(schema, (dict, bool)):
            raise SchemaError(f"un schéma doit être un objet JSON, obtenu {type(schema).__name__}")
        if int(max_whitespace) < 0:
            raise ValueError("max_whitespace doit être ≥ 0")
        if int(max_number_digits) < 1:
            raise ValueError("max_number_digits doit être ≥ 1")
        self.schema = schema
        self.max_whitespace = int(max_whitespace)
        self.max_number_digits = int(max_number_digits)
        self.ignored_keywords = _scan_ignored(schema)

        comp = _Compiler(schema)
        try:
            root = comp.compile(schema)
        except RecursionError as e:
            raise SchemaError("schéma trop profond") from e
        self._root = root
        self._nodes = comp.nodes
        self._build_strings(comp.strings)
        self._finalize()
        if self._W[root] is None:
            raise SchemaError("schéma insatisfiable : aucune instance JSON finie")

        # Caches (bornés : vidés au-delà de _max_states états)
        self._max_states = 50_000
        self._rows = {}           # état → [transition par octet] (_UNSET = non calculée)
        self._interned = {}       # état → état (objet canonique)
        self._start_memo = {}     # (nid, octet) → frames de départ
        self._accept_cache = {}
        self._allowed_cache = {}
        self._completion_cache = {}
        self._fallbacks = 0       # nb de complétions passées par le BFS de secours
        init = frozenset({((_F_ROOT, 0),)})
        self._initial = self._interned.setdefault(init, init)

    # ── Préparation ──────────────────────────────────────────────────────────

    def _build_strings(self, strings):
        """Tables des chaînes constantes : préfixes d'encodage, encodages, minimaux."""
        self._cstr_tab, self._cstr_encs, self._cstr_min, self._cstr_lit = [], [], [], []
        for s in strings:
            tabs, encs, mins = [], [], []
            for ch in s:
                e = _char_encodings(ch)
                d = {}
                for enc in e:
                    for k in range(1, len(enc)):
                        d.setdefault(enc[:k], False)
                    d[enc] = True
                tabs.append(d)
                encs.append(e)
                mins.append(e[0])
            self._cstr_tab.append(tuple(tabs))
            self._cstr_encs.append(tuple(encs))
            self._cstr_min.append(tuple(mins))
            self._cstr_lit.append(b'"' + b''.join(mins) + b'"')

    def _finalize(self):
        """Alternatives, point fixe des témoins minimaux (productivité), tables finales."""
        nodes = self._nodes
        N = len(nodes)

        # 1) Alternatives non-union atteignables par les unions (cycles tolérés)
        alts = []
        for n in range(N):
            if nodes[n][0] != _N_UNION:
                alts.append((n,))
                continue
            seen, out, stack = {n}, set(), [n]
            while stack:
                for c in nodes[stack.pop()][1]:
                    if c in seen:
                        continue
                    seen.add(c)
                    if nodes[c][0] == _N_UNION:
                        stack.append(c)
                    else:
                        out.add(c)
            alts.append(tuple(sorted(out)))

        # 2) Point fixe : plus court témoin (octets) de chaque nœud ; None = improductif
        W = [None] * N

        def wv(n):
            return _minb(*(W[a] for a in alts[n]))

        changed = True
        while changed:
            changed = False
            for n in range(N):
                if nodes[n][0] == _N_UNION:
                    continue
                w = self._witness(nodes[n], wv)
                if w is not None and (W[n] is None or (len(w), w) < (len(W[n]), W[n])):
                    W[n] = w
                    changed = True

        self._W = [wv(n) for n in range(N)]
        self._palts = [tuple(a for a in alts[n] if W[a] is not None) for n in range(N)]

        # 3) Tables par nœud (utilisées par les transitions et les complétions)
        self._info = [None] * N
        for n in range(N):
            node = nodes[n]
            k = node[0]
            if k == _N_OBJ:
                props = node[1]
                cand, cc, entry, AV, AC = self._obj_tables(props, wv)
                self._info[n] = (tuple(p[1] for p in props), tuple(p[2] for p in props),
                                 cand, cc, AV, AC)
            elif k == _N_FOBJ:
                v = node[1]
                if v >= 0 and self._W[v] is None:
                    v = -1
                self._info[n] = (v, self._W[v] if v >= 0 else None)
            elif k == _N_ARR:
                _, prefix, items, mn, mx = node
                pre = []
                for p in prefix:
                    if self._W[p] is None:
                        break
                    pre.append(p)
                it = items
                if len(pre) < len(prefix) or (it >= 0 and self._W[it] is None):
                    it = -1
                effmax = len(pre) if it < 0 else None
                if mx is not None:
                    effmax = mx if effmax is None else min(effmax, mx)
                cap = effmax if effmax is not None else max(mn, len(pre))
                self._info[n] = (tuple(pre), it, mn, effmax, cap)

    def _obj_tables(self, props, wv):
        """
        Programmation dynamique sur les propriétés ordonnées :
          cand[p]  : propriétés pouvant venir ensuite (on saute les optionnelles)
          cc[p]    : '}' permis (plus aucune requise à partir de p)
          AV[p]    : plus courte fin après une valeur, p propriétés traitées
          AC[p]    : plus courte fin après une virgule
        """
        P = len(props)
        entry = []
        for (_, sid, v, _) in props:
            w = wv(v)
            entry.append(None if w is None else self._cstr_lit[sid] + b':' + w)
        cc = [True] * (P + 1)
        for p in range(P - 1, -1, -1):
            cc[p] = cc[p + 1] and not props[p][3]
        cand = []
        for p in range(P + 1):
            c = []
            for j in range(p, P):
                if entry[j] is not None:
                    c.append(j)
                if props[j][3]:
                    break
            cand.append(tuple(c))
        AV = [None] * (P + 1)
        AC = [None] * (P + 1)
        AV[P] = b'}'
        for p in range(P - 1, -1, -1):
            AC[p] = _minb(*(entry[j] + AV[j + 1] for j in cand[p] if AV[j + 1] is not None))
            AV[p] = _minb(b'}' if cc[p] else None, b',' + AC[p] if AC[p] is not None else None)
        return tuple(cand), tuple(cc), entry, tuple(AV), tuple(AC)

    def _witness(self, node, wv):
        """Plus courte valeur JSON (octets) d'un nœud non-union, ou None."""
        k = node[0]
        if k == _N_STR:
            _, mn, mx = node
            return None if (mx is not None and mn > mx) else b'"' + b'a' * mn + b'"'
        if k == _N_NUM:
            return b'0'
        if k == _N_LIT:
            return node[1]
        if k == _N_CSTR:
            return self._cstr_lit[node[1]]
        if k == _N_FOBJ:
            return b'{}'
        if k == _N_OBJ:
            cand, cc, entry, AV, AC = self._obj_tables(node[1], wv)
            fin = _minb(b'}' if cc[0] else None, AC[0])
            return None if fin is None else b'{' + fin
        # _N_ARR
        _, prefix, items, mn, mx = node
        if mx is not None and mn > mx:
            return None
        parts = []
        for i in range(mn):
            it = prefix[i] if i < len(prefix) else items
            w = wv(it) if it >= 0 else None
            if w is None:
                return None
            parts.append(w)
        return b'[' + b','.join(parts) + b']'

    # ── Interface publique ───────────────────────────────────────────────────

    @property
    def initial_state(self):
        return self._initial

    def advance(self, state, byte: int):
        """Nouvel état après `byte`, ou None si l'octet est interdit (mémoïsé)."""
        if state is None:
            return None
        row = self._rows.get(state)
        if row is None:
            row = self._new_row(state)
        if not 0 <= byte < 256:
            raise ValueError(f"octet hors bornes : {byte!r}")
        r = row[byte]
        if r is _UNSET:
            r = row[byte] = self._compute(state, byte)
        return r

    def advance_bytes(self, state, data: bytes):
        for b in data:
            state = self.advance(state, b)
            if state is None:
                return None
        return state

    def is_accepting(self, state) -> bool:
        """Une valeur JSON racine complète a été produite."""
        if state is None:
            return False
        r = self._accept_cache.get(state)
        if r is None:
            r = self._accept_cache[state] = any(self._cfg_accepting(c) for c in state)
        return r

    def can_continue(self, state) -> bool:
        """Au moins un octet supplémentaire est acceptable."""
        return bool(self.allowed_bytes(state))

    def allowed_bytes(self, state) -> tuple:
        """Octets autorisés depuis `state` (tuple trié, mis en cache)."""
        if state is None:
            return ()
        r = self._allowed_cache.get(state)
        if r is None:
            r = tuple(b for b in range(256) if self.advance(state, b) is not None)
            if len(self._allowed_cache) >= self._max_states:
                self._allowed_cache.clear()
            self._allowed_cache[state] = r
        return r

    def shortest_completion(self, state):
        """
        Plus courte suite d'octets menant à un état acceptant (b'' si déjà acceptant),
        None si impossible. Calcul exact par décomposition de la pile (la complétion
        d'une configuration = fin du frame du sommet + fin de chaque parent, chacune
        minimale par programmation dynamique), vérifié ; BFS borné en secours.
        """
        if state is None:
            return None
        if state in self._completion_cache:
            return self._completion_cache[state]
        best = _minb(*(self._cfg_completion(c) for c in state))
        if best is None or not self.is_accepting(self.advance_bytes(state, best)):
            self._fallbacks += 1
            best = self._bfs_completion(state)
        if len(self._completion_cache) >= self._max_states:
            self._completion_cache.clear()
        self._completion_cache[state] = best
        return best

    # ── Transitions ──────────────────────────────────────────────────────────

    def _new_row(self, state):
        if len(self._rows) >= self._max_states:
            self._rows.clear()
            self._interned.clear()
            self._accept_cache.clear()
        row = [_UNSET] * 256
        self._rows[state] = row
        return row

    def _compute(self, state, b):
        out = set()
        step = self._step
        for cfg in state:
            if cfg:     # () = racine terminée : plus aucun octet (pas d'espace final)
                step(cfg, b, out)
        if not out:
            return None
        fs = frozenset(out)
        return self._interned.setdefault(fs, fs)

    def _start(self, n, b):
        """Frames créés quand l'octet `b` débute une valeur du nœud n (None = valeur finie)."""
        key = (n, b)
        r = self._start_memo.get(key)
        if r is not None:
            return r
        res = []
        nodes = self._nodes
        for a in self._palts[n]:
            node = nodes[a]
            k = node[0]
            if k == _N_STR:
                if b == 0x22:
                    res.append((_F_STR, node[1], node[2], 0, _S_NORM, 0))
            elif k == _N_NUM:
                if b == 0x2D:
                    res.append((_F_NUM, node[1], _P_MINUS, 0))
                elif b == 0x30:
                    res.append((_F_NUM, node[1], _P_ZERO, 1))
                elif 0x31 <= b <= 0x39:
                    res.append((_F_NUM, node[1], _P_INT, 1))
            elif k == _N_LIT:
                lit = node[1]
                if b == lit[0]:
                    res.append(None if len(lit) == 1 else (_F_LIT, lit, 1))
            elif k == _N_CSTR:
                if b == 0x22:
                    res.append((_F_CSTR, node[1], 0, b''))
            elif k == _N_OBJ:
                if b == 0x7B:
                    res.append((_F_OBJ, a, _O_OPEN, 0, 0))
            elif k == _N_FOBJ:
                if b == 0x7B:
                    res.append((_F_FOBJ, a, _O_OPEN, 0))
            elif k == _N_ARR:
                if b == 0x5B:
                    res.append((_F_ARR, a, _A_OPEN, 0, 0))
        r = self._start_memo[key] = tuple(dict.fromkeys(res))
        return r

    def _pop(self, rest):
        """Le frame du sommet vient de se terminer : le parent passe à la phase suivante."""
        if not rest:
            return ()
        par = rest[-1]
        k = par[0]
        if k == _F_OBJ:
            if par[2] == _O_KEY:
                np = (_F_OBJ, par[1], _O_AKEY, par[3], 0)
            else:
                np = (_F_OBJ, par[1], _O_AVAL, par[3] + 1, 0)
        elif k == _F_FOBJ:
            np = (_F_FOBJ, par[1], _O_AKEY if par[2] == _O_KEY else _O_AVAL, 0)
        else:   # _F_ARR
            c = par[3] + 1
            cap = self._info[par[1]][4]
            np = (_F_ARR, par[1], _A_AVAL, c if c < cap else cap, 0)
        return rest[:-1] + (np,)

    def _step(self, cfg, b, out):
        """Avance une configuration d'un octet ; ajoute les configurations résultantes à out."""
        fr = cfg[-1]
        k = fr[0]
        if k == _F_STR:
            self._step_str(cfg, fr, b, out)
        elif k == _F_OBJ:
            self._step_obj(cfg, fr, b, out)
        elif k == _F_CSTR:
            self._step_cstr(cfg, fr, b, out)
        elif k == _F_NUM:
            self._step_num(cfg, fr, b, out)
        elif k == _F_LIT:
            lit, i = fr[1], fr[2]
            if b == lit[i]:
                if i + 1 == len(lit):
                    out.add(self._pop(cfg[:-1]))
                else:
                    out.add(cfg[:-1] + ((_F_LIT, lit, i + 1),))
        elif k == _F_ARR:
            self._step_arr(cfg, fr, b, out)
        elif k == _F_FOBJ:
            self._step_fobj(cfg, fr, b, out)
        else:   # _F_ROOT
            if _IS_WS[b]:
                if fr[1] < self.max_whitespace:
                    out.add(((_F_ROOT, fr[1] + 1),))
                return
            for f in self._start(self._root, b):
                out.add(() if f is None else (f,))

    def _step_str(self, cfg, fr, b, out):
        _, mn, mx, cnt, sub, pend = fr
        tracked = mn > 0 or mx is not None
        if sub == _S_NORM:
            if b == 0x22:
                if cnt >= mn:
                    out.add(self._pop(cfg[:-1]))
                return
            if b == 0x5C:
                if not tracked:
                    nf = (_F_STR, 0, None, 0, _S_BS, 0)
                elif mx is not None and cnt >= mx:
                    # maxLength atteint : seul le bas d'une paire de substitution est permis
                    if not (pend and cnt == mx):
                        return
                    nf = (_F_STR, mn, mx, cnt + 1, _S_BSL, 1)
                else:
                    nc = cnt + 1
                    if mx is None and nc > mn + 1:
                        nc = mn + 1
                    nf = (_F_STR, mn, mx, nc, _S_BS, pend)
            else:
                nsub = _LEAD[b]
                if nsub is None:
                    return
                if not tracked:
                    if nsub == _S_NORM:
                        out.add(cfg)    # état canonique : identique quel que soit l'octet
                        return
                    nf = (_F_STR, 0, None, 0, nsub, 0)
                else:
                    if mx is not None and cnt >= mx:
                        return
                    nc = cnt + 1
                    if mx is None and nc > mn + 1:
                        nc = mn + 1
                    nf = (_F_STR, mn, mx, nc, nsub, 0)
            out.add(cfg[:-1] + (nf,))
            return

        cont = _UTF8_CONT.get(sub)
        if cont is not None:
            if cont[0] <= b <= cont[1]:
                out.add(cfg[:-1] + ((_F_STR, mn, mx, cnt, cont[2], 0),))
            return
        if sub == _S_BS:
            if b in _SHORT_ESC:
                nf = (_F_STR, mn, mx, cnt, _S_NORM, 0)
            elif b == 0x75:
                nf = (_F_STR, mn, mx, cnt, _S_U0, pend)
            else:
                return
            out.add(cfg[:-1] + (nf,))
            return
        if sub == _S_BSL:
            if b == 0x75:
                out.add(cfg[:-1] + ((_F_STR, mn, mx, cnt, _S_U0L, pend),))
            return
        if not _IS_HEX[b]:
            return
        low = b | 0x20      # minuscule pour a-f (les chiffres sont inchangés)
        if sub == _S_U0:
            ns = _S_U1D if (tracked and low == 0x64) else _S_U1X
        elif sub == _S_U0L:
            if low != 0x64:
                return
            ns = _S_U1L
        elif sub == _S_U1D:
            ns = _S_U2H if low in b'89ab' else (_S_U2L if low in b'cdef' else _S_U2X)
        elif sub == _S_U1L:
            if low not in b'cdef':
                return
            ns = _S_U2L
        elif sub == _S_U1X:
            ns = _S_U2X
        elif sub == _S_U2H:
            ns = _S_U3H
        elif sub == _S_U2L:
            ns = _S_U3L
        elif sub == _S_U2X:
            ns = _S_U3X
        elif sub == _S_U3H:
            out.add(cfg[:-1] + ((_F_STR, mn, mx, cnt, _S_NORM, 1),))
            return
        elif sub == _S_U3L:
            out.add(cfg[:-1] + ((_F_STR, mn, mx, cnt - 1 if pend else cnt, _S_NORM, 0),))
            return
        else:   # _S_U3X
            out.add(cfg[:-1] + ((_F_STR, mn, mx, cnt, _S_NORM, 0),))
            return
        out.add(cfg[:-1] + ((_F_STR, mn, mx, cnt, ns, pend),))

    def _step_cstr(self, cfg, fr, b, out):
        _, sid, i, part = fr
        tab = self._cstr_tab[sid]
        if i == len(tab):
            if b == 0x22 and not part:
                out.add(self._pop(cfg[:-1]))
            return
        if 0x41 <= b <= 0x46 and part[:2] == b'\\u':
            b |= 0x20       # hexadécimal insensible à la casse (forme canonique minuscule)
        np = part + _BYTE[b]
        full = tab[i].get(np)
        if full is None:
            return
        nf = (_F_CSTR, sid, i + 1, b'') if full else (_F_CSTR, sid, i, np)
        out.add(cfg[:-1] + (nf,))

    def _step_num(self, cfg, fr, b, out):
        _, integer, ph, d = fr
        D = self.max_number_digits
        isdig = 0x30 <= b <= 0x39
        nf = None
        if ph == _P_MINUS:
            if b == 0x30:
                nf = (_F_NUM, integer, _P_ZERO, 1)
            elif isdig:
                nf = (_F_NUM, integer, _P_INT, 1)
        elif ph == _P_ZERO or ph == _P_INT:
            if ph == _P_INT and isdig:
                if d < D:
                    nf = (_F_NUM, integer, _P_INT, d + 1)
            elif not integer:
                if b == 0x2E:
                    nf = (_F_NUM, integer, _P_DOT, 0)
                elif b == 0x65 or b == 0x45:
                    nf = (_F_NUM, integer, _P_E, 0)
        elif ph == _P_DOT:
            if isdig:
                nf = (_F_NUM, integer, _P_FRAC, 1)
        elif ph == _P_FRAC:
            if isdig:
                if d < D:
                    nf = (_F_NUM, integer, _P_FRAC, d + 1)
            elif b == 0x65 or b == 0x45:
                nf = (_F_NUM, integer, _P_E, 0)
        elif ph == _P_E:
            if b == 0x2B or b == 0x2D:
                nf = (_F_NUM, integer, _P_ESIGN, 0)
            elif isdig:
                nf = (_F_NUM, integer, _P_EXP, 1)
        elif ph == _P_ESIGN:
            if isdig:
                nf = (_F_NUM, integer, _P_EXP, 1)
        elif ph == _P_EXP:
            if isdig and d < D:
                nf = (_F_NUM, integer, _P_EXP, d + 1)
        if nf is not None:
            out.add(cfg[:-1] + (nf,))
        elif _P_COMPLETE[ph] and len(cfg) > 1:
            # Fin implicite du nombre : l'octet est passé au parent
            self._step(self._pop(cfg[:-1]), b, out)

    def _step_obj(self, cfg, fr, b, out):
        _, n, ph, p, ws = fr
        rest = cfg[:-1]
        if _IS_WS[b]:
            if ws < self.max_whitespace:
                out.add(rest + ((_F_OBJ, n, ph, p, ws + 1),))
            return
        sids, vals, cand, cc, _, _ = self._info[n]
        if ph == _O_OPEN or ph == _O_ACOMMA:
            if b == 0x22:
                for j in cand[p]:
                    out.add(rest + ((_F_OBJ, n, _O_KEY, j, 0), (_F_CSTR, sids[j], 0, b'')))
            elif b == 0x7D and ph == _O_OPEN and cc[p]:
                out.add(self._pop(rest))
        elif ph == _O_AKEY:
            if b == 0x3A:
                out.add(rest + ((_F_OBJ, n, _O_ACOLON, p, 0),))
        elif ph == _O_ACOLON:
            for f in self._start(vals[p], b):
                if f is None:
                    out.add(rest + ((_F_OBJ, n, _O_AVAL, p + 1, 0),))
                else:
                    out.add(rest + ((_F_OBJ, n, _O_VAL, p, 0), f))
        elif ph == _O_AVAL:
            if b == 0x2C:
                if cand[p]:
                    out.add(rest + ((_F_OBJ, n, _O_ACOMMA, p, 0),))
            elif b == 0x7D and cc[p]:
                out.add(self._pop(rest))

    def _step_fobj(self, cfg, fr, b, out):
        _, n, ph, ws = fr
        rest = cfg[:-1]
        if _IS_WS[b]:
            if ws < self.max_whitespace:
                out.add(rest + ((_F_FOBJ, n, ph, ws + 1),))
            return
        v = self._info[n][0]
        if ph == _O_OPEN or ph == _O_ACOMMA:
            if b == 0x22 and v >= 0:
                out.add(rest + ((_F_FOBJ, n, _O_KEY, 0), _FREE_STR))
            elif b == 0x7D and ph == _O_OPEN:
                out.add(self._pop(rest))
        elif ph == _O_AKEY:
            if b == 0x3A:
                out.add(rest + ((_F_FOBJ, n, _O_ACOLON, 0),))
        elif ph == _O_ACOLON:
            for f in self._start(v, b):
                if f is None:
                    out.add(rest + ((_F_FOBJ, n, _O_AVAL, 0),))
                else:
                    out.add(rest + ((_F_FOBJ, n, _O_VAL, 0), f))
        elif ph == _O_AVAL:
            if b == 0x2C:
                out.add(rest + ((_F_FOBJ, n, _O_ACOMMA, 0),))
            elif b == 0x7D:
                out.add(self._pop(rest))

    def _step_arr(self, cfg, fr, b, out):
        _, n, ph, c, ws = fr
        rest = cfg[:-1]
        if _IS_WS[b]:
            if ws < self.max_whitespace:
                out.add(rest + ((_F_ARR, n, ph, c, ws + 1),))
            return
        prefix, items, mn, effmax, cap = self._info[n]
        if ph == _A_AVAL:
            if b == 0x2C:
                if effmax is None or c < effmax:
                    out.add(rest + ((_F_ARR, n, _A_ACOMMA, c, 0),))
            elif b == 0x5D and c >= mn:
                out.add(self._pop(rest))
            return
        if ph == _A_OPEN and b == 0x5D:
            if mn == 0:
                out.add(self._pop(rest))
            return
        if effmax is not None and c >= effmax:
            return
        item = prefix[c] if c < len(prefix) else items
        for f in self._start(item, b):
            if f is None:
                c1 = c + 1
                out.add(rest + ((_F_ARR, n, _A_AVAL, c1 if c1 < cap else cap, 0),))
            else:
                out.add(rest + ((_F_ARR, n, _A_VAL, c, 0), f))

    # ── Acceptation & complétion ─────────────────────────────────────────────

    @staticmethod
    def _cfg_accepting(cfg) -> bool:
        if not cfg:
            return True
        return len(cfg) == 1 and cfg[0][0] == _F_NUM and _P_COMPLETE[cfg[0][2]]

    def _cfg_completion(self, cfg):
        """Plus courte complétion d'une configuration : sommet puis chaque parent."""
        if not cfg:
            return b''
        parts = [self._finish_top(cfg[-1])]
        for fr in reversed(cfg[:-1]):
            parts.append(self._finish_parent(fr))
        if any(p is None for p in parts):
            return None
        return b''.join(parts)

    def _arr_after(self, n, c):
        """Plus courte fin d'un tableau après c éléments (AV) — ',' + item… + ']'."""
        prefix, items, mn, _, _ = self._info[n]
        W = self._W
        return b''.join(b',' + W[prefix[i] if i < len(prefix) else items]
                        for i in range(c, mn)) + b']'

    def _arr_item_then_after(self, n, c):
        prefix, items, _, _, cap = self._info[n]
        item = prefix[c] if c < len(prefix) else items
        return self._W[item] + self._arr_after(n, min(c + 1, cap))

    def _finish_top(self, fr):
        k = fr[0]
        W = self._W
        if k == _F_ROOT:
            return W[self._root]
        if k == _F_STR:
            _, mn, mx, cnt, sub, pend = fr
            if (mn > 0 or mx is not None) and pend and sub in _S_PAIRING:
                cnt -= 1
            return _S_FINISH[sub] + b'a' * max(0, mn - cnt) + b'"'
        if k == _F_CSTR:
            _, sid, i, part = fr
            mins = self._cstr_min[sid]
            if i == len(mins):
                return b'"'
            cur = _minb(*(e[len(part):] for e in self._cstr_encs[sid][i] if e.startswith(part)))
            return cur + b''.join(mins[i + 1:]) + b'"'
        if k == _F_NUM:
            return _P_FINISH[fr[2]]
        if k == _F_LIT:
            return fr[1][fr[2]:]
        if k == _F_OBJ:
            _, n, ph, p, _ = fr
            sids, vals, cand, cc, AV, AC = self._info[n]
            if ph == _O_OPEN:
                return _minb(b'}' if cc[0] else None, AC[0])
            if ph == _O_AKEY:
                return b':' + W[vals[p]] + AV[p + 1]
            if ph == _O_ACOLON:
                return W[vals[p]] + AV[p + 1]
            if ph == _O_AVAL:
                return AV[p]
            return AC[p]
        if k == _F_FOBJ:
            ph = fr[2]
            wv = self._info[fr[1]][1]
            if ph == _O_OPEN or ph == _O_AVAL:
                return b'}'
            if ph == _O_AKEY:
                return b':' + wv + b'}'
            if ph == _O_ACOLON:
                return wv + b'}'
            return b'"":' + wv + b'}'
        # _F_ARR
        _, n, ph, c, _ = fr
        if ph == _A_OPEN:
            return b']' if self._info[n][2] == 0 else self._arr_item_then_after(n, 0)
        if ph == _A_AVAL:
            return self._arr_after(n, c)
        return self._arr_item_then_after(n, c)

    def _finish_parent(self, fr):
        """Fin minimale d'un frame parent une fois son enfant (clé ou valeur) terminé."""
        k = fr[0]
        if k == _F_OBJ:
            _, n, ph, p, _ = fr
            _, vals, _, _, AV, _ = self._info[n]
            if ph == _O_KEY:
                return b':' + self._W[vals[p]] + AV[p + 1]
            return AV[p + 1]
        if k == _F_FOBJ:
            if fr[2] == _O_KEY:
                return b':' + self._info[fr[1]][1] + b'}'
            return b'}'
        _, n, _, c, _ = fr
        return self._arr_after(n, min(c + 1, self._info[n][4]))

    def _bfs_completion(self, state, max_depth=512, max_nodes=200_000):
        """BFS de secours sur les états (borné) — ne devrait jamais servir."""
        if self.is_accepting(state):
            return b''
        prev = {state: None}
        depth = {state: 0}
        q = deque([state])
        while q:
            s = q.popleft()
            if depth[s] >= max_depth:
                continue
            for b in self.allowed_bytes(s):
                ns = self.advance(s, b)
                if ns in prev:
                    continue
                prev[ns] = (s, b)
                depth[ns] = depth[s] + 1
                if self.is_accepting(ns):
                    out = bytearray()
                    while prev[ns] is not None:
                        ns, bb = prev[ns]
                        out.append(bb)
                    return bytes(reversed(out))
                if len(prev) > max_nodes:
                    return None
                q.append(ns)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulaire → trie d'octets (partagé entre contraintes)
# ─────────────────────────────────────────────────────────────────────────────

class _VocabTrie:
    """Trie des octets des tokens : kids[nœud] = {octet: nœud}, ends[nœud] = [ids]."""

    def __init__(self, token_bytes):
        self.token_bytes = [tb if tb else None for tb in token_bytes]
        kids = [{}]
        ends = [None]
        for tid, tb in enumerate(self.token_bytes):
            if tb is None:
                continue
            node = 0
            for byte in tb:
                d = kids[node]
                nxt = d.get(byte)
                if nxt is None:
                    nxt = d[byte] = len(kids)
                    kids.append({})
                    ends.append(None)
                node = nxt
            if ends[node] is None:
                ends[node] = [tid]
            else:
                ends[node].append(tid)
        self.kids = kids
        self.ends = ends


_TRIE_CACHE = {}   # id(liste) → (copie de la liste, trie)   — réutilisation entre /json


def _get_trie(token_bytes) -> _VocabTrie:
    hit = _TRIE_CACHE.get(id(token_bytes))
    if hit is not None and hit[0] == token_bytes:
        return hit[1]
    trie = _VocabTrie(token_bytes)
    if len(_TRIE_CACHE) >= 4:
        _TRIE_CACHE.clear()
    _TRIE_CACHE[id(token_bytes)] = (list(token_bytes), trie)
    return trie


class _Shared:
    """Partagé par une contrainte et ses clones : trie + cache des masques par état."""

    def __init__(self, trie):
        self.trie = trie
        self.masks = {}         # état → bytes (1 octet 0/1 par token)
        self.max_masks = 1024


# ─────────────────────────────────────────────────────────────────────────────
# TokenConstraint — curseur au niveau tokens
# ─────────────────────────────────────────────────────────────────────────────

class TokenConstraint:
    """Mutable cursor over a matcher + a vocabulary given as token bytes."""

    def __init__(self, matcher: JSONSchemaMatcher, token_bytes: list):
        self.matcher = matcher
        self.vocab_size = len(token_bytes)
        self._shared = _Shared(_get_trie(token_bytes))
        self.reset()

    # ── État ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.state = self.matcher.initial_state
        self._buf = bytearray()

    @property
    def generated(self) -> bytes:
        """Octets acceptés depuis le dernier reset()."""
        return bytes(self._buf)

    def clone(self) -> 'TokenConstraint':
        """Copie légère : partage matcher, trie et caches."""
        c = object.__new__(TokenConstraint)
        c.matcher = self.matcher
        c.vocab_size = self.vocab_size
        c._shared = self._shared
        c.state = self.state
        c._buf = bytearray(self._buf)
        return c

    # ── Tokens autorisés ─────────────────────────────────────────────────────

    def _walk(self, state, tb):
        adv = self.matcher.advance
        for b in tb:
            state = adv(state, b)
            if state is None:
                return None
        return state

    def is_allowed(self, token_id: int) -> bool:
        if not 0 <= token_id < self.vocab_size:
            return False
        tb = self._shared.trie.token_bytes[token_id]
        if tb is None:
            return False
        return self._walk(self.state, tb) is not None

    def allowed_mask(self):
        """
        BoolTensor (vocab_size,) CPU des tokens autorisés — DFS sur le trie, cache par état.
        Le cache stocke des `bytes` immuables (0/1) ; chaque appel renvoie un tenseur neuf
        (torch.frombuffer sur une copie : quelques µs, sans pool de threads).
        """
        sh = self._shared
        raw = sh.masks.get(self.state)
        if raw is None:
            buf = bytearray(self.vocab_size)
            for i in self._allowed_ids(self.state):
                buf[i] = 1
            raw = bytes(buf)
            if len(sh.masks) >= sh.max_masks:
                sh.masks.clear()
            sh.masks[self.state] = raw
        if not raw:
            return torch.zeros(0, dtype=torch.bool)
        return torch.frombuffer(bytearray(raw), dtype=torch.bool)

    def _allowed_ids(self, state) -> list:
        """DFS (nœud du trie, état de l'automate) avec élagage sur octet refusé."""
        mt = self.matcher
        rows = mt._rows
        compute = mt._compute
        new_row = mt._new_row
        unset = _UNSET
        kids = self._shared.trie.kids
        ends = self._shared.trie.ends
        ids = []
        stack = [(0, state)]
        pop, push = stack.pop, stack.append
        while stack:
            node, st = pop()
            row = rows.get(st)
            if row is None:
                row = new_row(st)
            for b, child in kids[node].items():
                ns = row[b]
                if ns is unset:
                    ns = row[b] = compute(st, b)
                if ns is None:
                    continue
                e = ends[child]
                if e is not None:
                    ids.extend(e)
                if kids[child]:
                    push((child, ns))
        return ids

    def apply_to_logits(self, logits):
        """Masque des logits (..., vocab_size) : tokens interdits → -inf."""
        mask = self.allowed_mask().to(logits.device)
        return logits.masked_fill(~mask, float('-inf'))

    # ── Avance & fin ─────────────────────────────────────────────────────────

    def advance(self, token_id: int) -> None:
        tb = (self._shared.trie.token_bytes[token_id]
              if 0 <= token_id < self.vocab_size else None)
        ns = self._walk(self.state, tb) if tb is not None else None
        if ns is None:
            raise ValueError(f"token {token_id} non autorisé par la contrainte JSON")
        self.state = ns
        self._buf += tb

    def is_complete(self) -> bool:
        return self.matcher.is_accepting(self.state)

    def is_terminal(self) -> bool:
        """JSON complet et plus aucun octet possible → la génération doit s'arrêter."""
        return self.is_complete() and not self.matcher.can_continue(self.state)

    def completion_tokens(self, max_expansions: int = 10_000):
        """
        Tokens menant à la plus courte complétion acceptante (plus long préfixe d'abord,
        retour arrière si une impasse se présente). None si impossible avec ce vocabulaire.
        """
        mt = self.matcher
        c = mt.shortest_completion(self.state)
        if c is None:
            return None
        kids, ends = self._shared.trie.kids, self._shared.trie.ends
        tbytes = self._shared.trie.token_bytes
        L = len(c)

        def candidates(pos):
            found, node = [], 0
            for i in range(pos, L):
                node = kids[node].get(c[i])
                if node is None:
                    break
                if ends[node] is not None:
                    found.extend(ends[node])
            found.reverse()     # plus long d'abord
            return found

        tokens = []
        stack = [(0, self.state, candidates(0))]
        expansions = 0
        while stack:
            pos, st, cands = stack[-1]
            if pos == L:
                return tokens if mt.is_accepting(st) else None
            if not cands:
                stack.pop()
                if tokens:
                    tokens.pop()
                continue
            expansions += 1
            if expansions > max_expansions:
                return None
            tid = cands.pop(0)
            ns = self._walk(st, tbytes[tid])     # == is_allowed depuis st
            if ns is None:
                continue
            tokens.append(tid)
            npos = pos + len(tbytes[tid])
            stack.append((npos, ns, candidates(npos)))
        return None


def json_constraint(token_bytes: list, schema=None, *, max_whitespace: int = 4) -> TokenConstraint:
    """Contrainte prête à l'emploi : JSON valide (schema=None) ou conforme au schéma."""
    return TokenConstraint(JSONSchemaMatcher(schema, max_whitespace=max_whitespace), token_bytes)


# ─────────────────────────────────────────────────────────────────────────────
# CLI minimal : python3 structured.py SCHEMA [DOCUMENT]
# ─────────────────────────────────────────────────────────────────────────────

def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Vérifie un document JSON avec l'automate "
                                             "structured outputs de nanoPOPIXA.")
    ap.add_argument('schema', help="chemin .json ou schéma JSON inline")
    ap.add_argument('document', nargs='?', help="fichier JSON à vérifier (optionnel)")
    args = ap.parse_args(argv)
    try:
        schema = load_schema(args.schema)
        m = JSONSchemaMatcher(schema)
    except SchemaError as e:
        print(f"✗ schéma : {e}")
        return 2
    if m.ignored_keywords:
        print("⚠ mots-clés non appliqués : " + ", ".join(sorted(m.ignored_keywords)))
    print("instance minimale : " + m.shortest_completion(m.initial_state).decode('utf-8'))
    if not args.document:
        return 0
    with open(args.document, 'rb') as f:
        data = f.read()
    st = m.initial_state
    for i, b in enumerate(data):
        st = m.advance(st, b)
        if st is None:
            print(f"✗ octet refusé à la position {i} : {data[max(0, i - 20):i + 1]!r}")
            return 1
    if not m.is_accepting(st):
        print("✗ document incomplet")
        return 1
    errs = validate_instance(json.loads(data.decode('utf-8')), schema)
    print("✓ document accepté" if not errs else "✗ " + "; ".join(errs))
    return 0 if not errs else 1


if __name__ == '__main__':
    raise SystemExit(_main())
