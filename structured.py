"""
nanoPOPIXA — Structured outputs : décodage contraint JSON / JSON Schema
Inspiré du beta Anthropic `structured-outputs-2025-12-15`.

Principe
--------
  1. Le schéma est compilé en une petite grammaire (graphe de nœuds, références
     `$ref` résolues *par référence* → les schémas récursifs ne bouclent jamais).
  2. Un automate à pile NON déterministe lit la sortie octet par octet. Un état
     est un `frozenset` de configurations ; une configuration est une pile
     immuable chaînée (sommet → reste, queue partagée, hash mis en cache : un état
     coûte O(1) en mémoire quelle que soit la profondeur d'imbrication). Les états
     sont donc hashables et canoniques (ex. dans une chaîne libre, l'état après
     n'importe quel octet ordinaire est identique) → toutes les transitions sont
     mémoïsées (dict état → {octet: état}, caches plafonnés en nombre d'états).
  3. `TokenConstraint` parcourt un trie d'octets du vocabulaire en propageant les
     états de l'automate (élagage dès qu'un octet est refusé) → masque booléen des
     tokens autorisés, mis en cache par état. `completion_tokens()` ferme le JSON
     au plus court quand le budget de tokens s'épuise (tokenisation de la plus courte
     complétion en octets ; si le vocabulaire ne sait pas l'écrire, recherche A* d'une
     autre complétion écrivable avec ses tokens).
     Remplissage des chaînes minLength : `filler` du matcher ('a' par défaut).
     `json_constraint` choisit l'octet dont les répétitions coûtent le MOINS de tokens
     dans le vocabulaire (gpt2 : '-' * 64 = 1 token, contre 4 octets au plus pour 'a') :
     la fermeture, budgétée en tokens par les appelants, reste courte en tokens.

Sous-ensemble JSON Schema supporté
----------------------------------
  - None / True / {}            → n'importe quelle valeur JSON. False → SchemaError.
  - type                        → 'string' | 'number' | 'integer' | 'boolean' | 'null'
                                  | 'object' | 'array', ou une liste (union).
                                  Sans `type`, le type est déduit des mots-clés présents
                                  (properties → object, items → array, maxLength → string…),
                                  sinon toutes les valeurs sont permises.
  - enum / const                → comparés structurellement (égalité JSON, bool ≠ nombre ;
                                  deux chaînes sont égales si leur texte JSON l'est : une
                                  paire haut+bas en deux unités de code Python vaut le
                                  caractère astral correspondant, comme pour json.loads) :
                                  chaque valeur est compilée en grammaire exacte, donc les
                                  formes `ensure_ascii=True/False`, séparateurs compacts ou
                                  par défaut sont toutes acceptées. Les nombres doivent
                                  apparaître sous la forme `json.dumps(v)` (1 ≠ 1.0) ;
                                  chaque membre listé garde SA forme, même s'il est égal
                                  en JSON à un autre (enum [1, 1.0] → '1' et '1.0' ;
                                  objets : ordre de déclaration de chaque membre).
                                  Les valeurs incompatibles avec les mots-clés frères
                                  (ex. type) sont retirées. Un entier trop long pour
                                  json.dumps / json.loads (limite int ↔ str de Python,
                                  4300 chiffres par défaut) → SchemaError.
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
                                  au plus `max_number_digits` chiffres par série de chiffres
                                  (réglage > sys.get_int_max_str_digits() → ValueError :
                                  json.loads refuserait les entiers acceptés).
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
  - Annotations (title, description, default, examples, $schema, $id, id (draft-04),
    $comment, deprecated, readOnly, writeOnly, $anchor, $dynamicAnchor…) ignorées
    silencieusement (y compris dans les fusions).
  - Valeurs de mots-clés malformées (type inconnu ou `null`, minLength / maxLength /
    minItems / maxItems non entiers ≥ 0 ou `null`, required qui n'est pas une liste de
    chaînes, properties qui n'est pas un objet, additionalProperties qui n'est pas un
    schéma…) → SchemaError, avec ou sans fusion (allOf / $ref / anyOf / enum + frères) ;
    validate_instance lève aussi SchemaError (jamais TypeError) sur ces valeurs.
  - Assertions NON appliquées (pattern, format, minimum, maximum, exclusiveMinimum,
    exclusiveMaximum, multipleOf, uniqueItems, minProperties, maxProperties,
    propertyNames, patternProperties, dependentRequired, if/then/else, not, contains,
    $dynamicRef, $recursiveRef…) → listées dans `JSONSchemaMatcher.ignored_keywords`
    (y compris dans les cibles de `$ref`). `validate_instance` ne les applique pas non
    plus (une sortie générée est donc toujours valide, quelle que soit sa profondeur).
  - Espaces : JSON (espace, \\n, \\r, \\t) uniquement autour des caractères structurels
    (après '{' '[' ',' ':' et avant '}' ']' ',' ':') et AVANT la valeur racine, au plus
    `max_whitespace` octets consécutifs par interstice. Pas d'espace final : une racine
    objet / tableau / chaîne fermée est immédiatement terminale.
  - Un schéma sans aucune instance finie (ex. récursion obligatoire infinie) → SchemaError.
  - Taille : un sous-schéma dont l'instance minimale dépasse 1 Mio (minLength / minItems
    démesurés), un objet dont les tables de complétion dépasseraient 64 Mio (plusieurs
    milliers de propriétés REQUISES), un document de schéma trop imbriqué → SchemaError
    (jamais RecursionError / OverflowError / MemoryError).

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

import heapq
import json
import os
import re
import sys
from array import array
from bisect import bisect_left, bisect_right
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
    'title', 'description', 'default', 'examples', '$schema', '$id', 'id', '$comment',
    'deprecated', 'readOnly', 'writeOnly', '$defs', 'definitions', '$anchor',
    '$dynamicAnchor', '$recursiveAnchor',
    '$vocabulary', 'contentMediaType', 'contentEncoding', 'contentSchema',
})

# Assertions reconnues mais NON appliquées — rapportées dans `ignored_keywords`
_UNENFORCED = frozenset({
    'pattern', 'format', 'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum',
    'multipleOf', 'uniqueItems', 'minProperties', 'maxProperties', 'propertyNames',
    'patternProperties', 'dependentRequired', 'dependentSchemas', 'dependencies',
    'if', 'then', 'else', 'not', 'contains', 'minContains', 'maxContains',
    'unevaluatedProperties', 'unevaluatedItems',
    # Applicateurs à portée dynamique : ni résolus ni appliqués (la valeur devient libre)
    '$dynamicRef', '$recursiveRef',
})

# Mots-clés APPLIQUÉS par l'automate (et par validate_instance). Tout autre mot-clé
# (annotation, extension `x-*`, OpenAPI `example` / `nullable` / `discriminator`…) est
# traité comme une annotation (JSON Schema 2020-12 §6.5), y compris dans les fusions.
_APPLIED = frozenset({
    'type', 'enum', 'const', 'properties', 'required', 'additionalProperties',
    'items', 'prefixItems', 'additionalItems', 'minLength', 'maxLength', 'minItems',
    'maxItems', 'anyOf', 'oneOf', 'allOf', '$ref',
})

_ITEMS_KEYS = ('items', 'prefixItems', 'additionalItems')

_TYPES = ('null', 'boolean', 'object', 'array', 'number', 'integer', 'string')

# Mots-clés qui contiennent des sous-schémas (pour le scan des mots-clés ignorés)
_SUB_MAPS = ('properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas')
_SUB_ONE = ('additionalProperties', 'items', 'additionalItems', 'contains', 'propertyNames',
            'not', 'if', 'then', 'else', 'unevaluatedProperties', 'unevaluatedItems')
_SUB_LISTS = ('allOf', 'anyOf', 'oneOf', 'prefixItems', 'items')

_MAX_COMPILE_DEPTH = 100    # imbrication maximale à la compilation
_MAX_MERGE_DEPTH = 64       # chaînes $ref / allOf + frères imbriquées (sans descente)
_MAX_MERGES = 1 << 15       # schémas fusionnés distincts (conjonctions d'atomes)
_MAX_VALIDATE_HOPS = 400    # sauts $ref / allOf / anyOf successifs SANS descendre dans
                            # l'instance (la profondeur du document, elle, est illimitée)
_MAX_WITNESS = 1 << 20      # taille maximale (octets) de l'instance minimale d'un nœud
_MAX_TABLES = 1 << 26       # octets cumulés des tables de complétion des objets


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
    `true` → {} ; `false`, JSON invalide, fichier illisible, imbrication excessive,
    bornes démesurées → SchemaError (sous-classe de ValueError, jamais RecursionError).
    Un BOM UTF-8 en tête (Bloc-notes Windows, PowerShell 5…) est ignoré (RFC 8259 §8.1).
    Tout chemin EXISTANT qui n'est pas un dossier est lu comme un fichier, y compris un
    tube : tube nommé (FIFO), substitution de processus bash (`--schema <(jq …)` →
    /dev/fd/63), entrée standard (`cat schema.json | popixa gen --schema /dev/stdin`).
    """
    if not isinstance(source, str) or not source.strip():
        raise SchemaError("schéma vide : chemin de fichier ou JSON inline attendu")
    text = source
    inline = True
    path = os.path.expanduser(source.strip())
    if os.path.isdir(path):
        raise SchemaError(f"le schéma doit être un fichier, pas un dossier : {_brief(path)}")
    if os.path.exists(path):        # fichier ordinaire, tube, /dev/stdin… (lu une seule fois)
        inline = False
        try:
            with open(path, 'r', encoding='utf-8-sig') as f:      # BOM éventuel retiré
                text = f.read()
        except (OSError, UnicodeDecodeError) as e:
            raise SchemaError(f"lecture du schéma impossible ({path}) : {e}") from e
    elif text.startswith('\ufeff'):
        text = text[1:]
    try:
        schema = json.loads(text)
    except RecursionError as e:
        raise SchemaError("schéma trop profond (imbrication JSON excessive)") from e
    except ValueError as e:
        if inline and not text.lstrip().startswith(('{', '[', 't', 'f', 'n')):
            raise SchemaError(f"fichier introuvable et JSON inline invalide : {_brief(source)}") from e
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

_SURROGATE = re.compile('[\ud800-\udfff]')


def _norm_str(s: str) -> str:
    """
    Forme JSON canonique d'une chaîne Python : une paire haut+bas écrite en DEUX unités
    de code ('\\ud83d' + '\\ude00') devient le caractère astral ('😀'), exactement comme
    json.loads la relit (json.dumps les écrit identiquement). Surrogates isolés conservés.
    """
    if _SURROGATE.search(s) is None:
        return s
    return s.encode('utf-16-le', 'surrogatepass').decode('utf-16-le', 'surrogatepass')


def _norm_keys(d: dict) -> dict:
    """Clés normalisées (_norm_str) ; le dict lui-même si aucune clé n'est concernée."""
    if not any(_SURROGATE.search(k) for k in d if isinstance(k, str)):
        return d
    return {(_norm_str(k) if isinstance(k, str) else k): v for k, v in d.items()}


def _json_eq(a, b) -> bool:
    """Égalité JSON : bool ≠ nombre, 1 == 1.0, chaînes normalisées, comparaison profonde."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, str) or isinstance(b, str):
        return (isinstance(a, str) and isinstance(b, str)
                and (a == b or _norm_str(a) == _norm_str(b)))
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        a, b = _norm_keys(a), _norm_keys(b)
        return a.keys() == b.keys() and all(_json_eq(a[k], b[k]) for k in a)
    return False


def _json_key(v):
    """
    Clé hashable cohérente avec _json_eq pour une valeur JSON (_is_json_value) :
    _json_eq(a, b) ⇔ _json_key(a) == _json_key(b). Déduplication / intersection en O(n).
    """
    if v is None:
        return ('n',)
    if isinstance(v, bool):
        return ('b', v)
    if isinstance(v, (int, float)):
        return ('x', v)          # 1 et 1.0 : même clé (égaux et même hash)
    if isinstance(v, str):
        return ('s', _norm_str(v))
    if isinstance(v, list):
        return ('l', tuple(_json_key(x) for x in v))
    return ('d', frozenset((_norm_str(k), _json_key(x)) for k, x in v.items()))


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
        raise SchemaError(f"$ref non local non supporté : {_brief(ref)}")
    frag = ref[1:]
    if frag == '':
        return root
    if not frag.startswith('/'):
        raise SchemaError(f"$ref par ancre non supporté : {_brief(ref)}")
    cur = root
    for tok in frag[1:].split('/'):
        tok = unquote(tok).replace('~1', '/').replace('~0', '~')
        if isinstance(cur, dict) and tok in cur:
            cur = cur[tok]
        elif isinstance(cur, list) and tok.isdigit() and int(tok) < len(cur):
            cur = cur[int(tok)]
        else:
            raise SchemaError(f"$ref introuvable : {_brief(ref)}")
    return cur


def _has_assertions(schema: dict) -> bool:
    """Le schéma porte au moins un mot-clé appliqué (ou une assertion non appliquée, qui
    sert à déduire le type) ; annotations et mots-clés inconnus ne comptent pas."""
    return any(k in _APPLIED or k in _UNENFORCED for k in schema)


def _strip(schema: dict, keys) -> dict:
    return {k: v for k, v in schema.items() if k not in keys}


def _brief(v, limit: int = 80) -> str:
    """repr() court et sûr pour les messages d'erreur (entier démesuré : repr lève ValueError)."""
    try:
        r = repr(v)
    except ValueError:      # > sys.get_int_max_str_digits() chiffres
        return f"<{type(v).__name__} démesuré>"
    return r if len(r) <= limit else r[:limit - 1] + '…'


def _nonneg_value(key: str, v) -> int:
    """Valeur de minLength / maxLength / minItems / maxItems : entier ≥ 0 (2.0 admis)."""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise SchemaError(f"{key} doit être un entier ≥ 0, obtenu {_brief(v)}")
    return v


def _nonneg(schema: dict, key: str, default):
    """Borne `key` du schéma ; `default` si ABSENTE (un `null` explicite → SchemaError)."""
    if key not in schema:
        return default
    return _nonneg_value(key, schema[key])


def _type_list(t) -> list:
    """Valeur de `type` → liste de types connus ; null, objet, type inconnu → SchemaError."""
    if isinstance(t, str):
        types = [t]
    elif isinstance(t, list):
        types = t
    else:
        raise SchemaError(f"type invalide : {_brief(t)}")
    for x in types:
        if not isinstance(x, str) or x not in _TYPES:
            raise SchemaError(f"type inconnu : {_brief(x)}")
    return types


def _props_value(v) -> dict:
    """Valeur de `properties` : objet (null = absent → {}), sinon SchemaError."""
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise SchemaError(f"properties doit être un objet, obtenu {_brief(v)}")
    return v


def _required_value(v) -> list:
    """Valeur de `required` : liste de chaînes, sinon SchemaError."""
    if not isinstance(v, list) or not all(isinstance(r, str) for r in v):
        raise SchemaError(f"required doit être une liste de chaînes, obtenu {_brief(v)}")
    return v


def _check_subschema(key: str, v):
    """Valeur qui doit être un schéma (objet, booléen ou null = any), sinon SchemaError."""
    if v is not None and not isinstance(v, (bool, dict)):
        raise SchemaError(f"{key} doit être un schéma (objet ou booléen), obtenu {_brief(v)}")
    return v


def _check_keywords(S: dict) -> None:
    """
    Valeurs des mots-clés appliqués, vérifiées pour TOUT sous-schéma compilé, que le type
    compilé les lise ou non ({"type": "integer", "minLength": "x"} → SchemaError, comme
    {"type": "string", "minLength": "x"}) : même lecture que le compilateur et
    validate_instance (_nonneg_value, _type_list, _required_value, _props_value…).
    """
    for k in ('minLength', 'maxLength', 'minItems', 'maxItems'):
        if k in S:
            _nonneg_value(k, S[k])
    if 'type' in S:
        _type_list(S['type'])
    if 'required' in S:
        _required_value(S['required'])
    if 'properties' in S:
        for name, sub in _props_value(S['properties']).items():
            _check_subschema(f"properties[{_brief(name)}]", sub)
    for k in ('additionalProperties', 'additionalItems'):
        if k in S:
            _check_subschema(k, S[k])
    if 'items' in S:
        items = S['items']
        for sub in (items if isinstance(items, list) else (items,)):
            _check_subschema('items', sub)
    if 'prefixItems' in S:
        prefix = S['prefixItems']
        if prefix is not None and not isinstance(prefix, list):
            raise SchemaError(f"prefixItems doit être une liste, obtenu {_brief(prefix)}")
        for sub in prefix or ():
            _check_subschema('prefixItems', sub)
    if 'enum' in S and not isinstance(S['enum'], list):
        raise SchemaError(f"enum doit être une liste, obtenu {_brief(S['enum'])}")
    for k in ('anyOf', 'oneOf'):
        if k in S and (not isinstance(S[k], list) or not S[k]):
            raise SchemaError(f"{k} doit être une liste non vide")
    if 'allOf' in S and (not isinstance(S['allOf'], list) or len(S['allOf']) != 1):
        raise SchemaError("allOf : exactement un sous-schéma est supporté")


def _form_key(v):
    """
    Clé de la GRAMMAIRE que _Compiler.const compile pour une valeur JSON : deux valeurs
    égales en JSON mais écrites différemment par json.dumps restent distinctes (1 / 1.0,
    0 / -0.0, ordre des clés d'un objet) ; chaînes sous forme canonique (_norm_str),
    exactement comme les constantes compilées.
    """
    if v is None:
        return ('n',)
    if isinstance(v, bool):
        return ('b', v)
    if isinstance(v, int):
        return ('i', int(v))
    if isinstance(v, float):
        return ('f', float.__repr__(v))     # == json.dumps(v) pour un flottant fini
    if isinstance(v, str):
        return ('s', _norm_str(v))
    if isinstance(v, list):
        return ('l', tuple(_form_key(x) for x in v))
    return ('d', tuple((_norm_str(k), _form_key(x)) for k, x in v.items()))


def _scan_ignored(schema) -> set:
    """
    Assertions non appliquées présentes dans le document de schéma : sous-schémas
    standard ET cibles de chaque `$ref` (pointeur quelconque, ex. #/components/schemas/X).
    """
    found, seen, stack = set(), set(), [schema]
    while stack:
        s = stack.pop()
        if not isinstance(s, dict) or id(s) in seen:
            continue
        seen.add(id(s))
        for k, v in s.items():
            if k in _UNENFORCED:
                found.add(k)
            if k == '$ref':
                try:
                    stack.append(_resolve_ref(schema, v))
                except SchemaError:
                    pass        # la compilation signalera le $ref invalide
            elif k in _SUB_MAPS and isinstance(v, dict):
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
    """
    Validateur direct sur le schéma brut — même sous-ensemble que l'automate.

    ITÉRATIF (trampoline de générateurs : chaque sous-validation est demandée par
    `yield (instance, schéma, chemin, chaîne)` et reçoit sa liste d'erreurs) → aucune
    limite de profondeur de document : tout ce que l'automate accepte est validable.
    `chaîne` = ids des schémas traversés par $ref / allOf / anyOf / oneOf SANS descendre
    dans l'instance : un retour sur l'un d'eux est une référence circulaire (bouclerait
    à l'infini) → erreur, comme au-delà de _MAX_VALIDATE_HOPS sauts consécutifs.
    """

    _ROOT_CHAIN = ()

    def __init__(self, root):
        self.root = root

    def errors(self, inst, sch, path='$', depth=0) -> list:
        stack = [self._check(inst, sch, path, self._ROOT_CHAIN)]
        sent = None
        while True:
            try:
                req = stack[-1].send(sent)
            except StopIteration as stop:
                stack.pop()
                if not stack:
                    return stop.value
                sent = stop.value
                continue
            stack.append(self._check(*req))
            sent = None

    def _check(self, inst, sch, path, chain):
        errs = []
        if sch is None or sch is True:
            return errs
        if sch is False:
            errs.append(f"{path} : aucune valeur autorisée (schéma false)")
            return errs
        if not isinstance(sch, dict):
            raise SchemaError(f"sous-schéma invalide en {path} : {_brief(sch)}")
        if id(sch) in chain or len(chain) > _MAX_VALIDATE_HOPS:
            errs.append(f"{path} : récursion $ref sans fin")
            return errs
        hop = chain + (id(sch),)

        # Valeurs de mots-clés malformées → SchemaError (jamais TypeError / AttributeError),
        # avec la même lecture que le compilateur (cf. _nonneg, _type_list…)
        if '$ref' in sch:
            errs += yield (inst, _resolve_ref(self.root, sch['$ref']), path, hop)
        if 'allOf' in sch:
            subs = sch['allOf']
            if not isinstance(subs, list):
                raise SchemaError(f"allOf doit être une liste en {path}")
            for sub in subs:
                errs += yield (inst, sub, path, hop)
        for key in ('anyOf', 'oneOf'):
            if key in sch:
                alts = sch[key]
                if not isinstance(alts, list):
                    raise SchemaError(f"{key} doit être une liste en {path}")
                ok = False
                for alt in alts:
                    if not (yield (inst, alt, path, hop)):
                        ok = True
                        break
                if not ok:
                    errs.append(f"{path} : aucune alternative de {key} ne correspond")
        if 'enum' in sch:
            if not isinstance(sch['enum'], list):
                raise SchemaError(f"enum doit être une liste en {path}")
            if not any(_json_eq(inst, v) for v in sch['enum']):
                errs.append(f"{path} : valeur hors enum")
        if 'const' in sch and not _json_eq(inst, sch['const']):
            errs.append(f"{path} : valeur différente de const")
        if 'type' in sch:
            types = _type_list(sch['type'])
            if not any(_type_ok(inst, t) for t in types):
                errs.append(f"{path} : type attendu {'|'.join(types)}, obtenu {_json_type(inst)}")

        top = self._ROOT_CHAIN      # descente dans l'instance : la chaîne repart de zéro
        if isinstance(inst, str):
            n = len(_norm_str(inst))    # points de code (paire de substitution = 1)
            mn = _nonneg(sch, 'minLength', 0)
            mx = _nonneg(sch, 'maxLength', None)
            if n < mn:
                errs.append(f"{path} : chaîne trop courte ({n} < {_brief(mn)})")
            if mx is not None and n > mx:
                errs.append(f"{path} : chaîne trop longue ({n} > {_brief(mx)})")
        elif isinstance(inst, dict):
            keys = _norm_keys(inst)
            props = _norm_keys(_props_value(sch.get('properties')))
            for name, sub in props.items():
                if name in keys:
                    errs += yield (keys[name], sub, f"{path}.{name}", top)
            for name in (_required_value(sch['required']) if 'required' in sch else ()):
                if _norm_str(name) not in keys:
                    errs.append(f"{path} : propriété requise manquante « {name} »")
            addl = _check_subschema('additionalProperties', sch.get('additionalProperties', True))
            if addl is not True and addl is not None:
                for name, v in keys.items():
                    if name in props:
                        continue
                    if addl is False:
                        errs.append(f"{path} : propriété non autorisée « {name} »")
                    else:
                        errs += yield (v, addl, f"{path}.{name}", top)
        elif isinstance(inst, list):
            items = sch.get('items', True)
            prefix = sch.get('prefixItems')
            if isinstance(items, list):
                prefix, items = items, sch.get('additionalItems', True)
            if prefix is None:
                prefix = []
            elif not isinstance(prefix, list):
                raise SchemaError(f"prefixItems doit être une liste en {path}")
            for i, v in enumerate(inst):
                sub = prefix[i] if i < len(prefix) else items
                if sub is not True and sub is not None:
                    errs += yield (v, sub, f"{path}[{i}]", top)
            mn = _nonneg(sch, 'minItems', 0)
            mx = _nonneg(sch, 'maxItems', None)
            if len(inst) < mn:
                errs.append(f"{path} : trop peu d'éléments ({len(inst)} < {_brief(mn)})")
            if mx is not None and len(inst) > mx:
                errs.append(f"{path} : trop d'éléments ({len(inst)} > {_brief(mx)})")
        return errs


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
    """
    Compile un schéma en nœuds. Mémo par identité de dict → récursion sans boucle.

    Fusions ($ref / allOf + frères, anyOf / oneOf + frères) : un schéma fusionné est la
    CONJONCTION d'« atomes » — dicts du document (ou copies mémoïsées privées d'un
    mot-clé) sans $ref ni allOf, développés par `conj`. Chaque conjonction (tuple
    d'atomes, ordre de première apparition, doublons retirés) a UN dict fusionné
    canonique (`merged`) : surcharger une propriété par une référence au schéma en cours
    de fusion (liste chaînée typée, catégorie / parent…) retombe sur la même conjonction
    → même dict → le mémo de `compile` referme le cycle. Les propriétés communes à
    plusieurs atomes deviennent elles-mêmes des conjonctions (construites par une file,
    sans récursion).
    """

    def __init__(self, root):
        self.root = root
        self.nodes = []
        self.memo = {}       # id(dict de schéma) → nid
        self.hcons = {}      # nœud (tuple) → nid  (partage des nœuds identiques)
        self.keep = []       # dicts gardés en vie (ids stables pour les mémos)
        self.strings = []    # sid → chaîne constante
        self.sids = {}
        self.depth = 0
        self._any = None
        self._strips = {}    # (id(dict), mot-clé) → copie privée du mot-clé
        self._conjs = {}     # id(dict) → atomes (tuple) ; None = false
        self._busy = set()   # conj en cours (chaîne $ref / allOf cyclique)
        self._merged = {}    # tuple(id des atomes) → dict fusionné
        self._mkeys = {}     # id(dict fusionné ou dérivé) → atomes
        self._todo = []      # dicts fusionnés à remplir
        self._draining = False

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
        """Indice d'une chaîne constante (forme JSON canonique, cf. _norm_str)."""
        s = _norm_str(s)
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
            raise SchemaError(f"sous-schéma invalide : {_brief(schema)}")
        nid = self.memo.get(id(schema))
        if nid is not None:
            return nid
        self.depth += 1
        if self.depth > _MAX_COMPILE_DEPTH:
            raise SchemaError("schéma trop profond (imbrication de sous-schémas excessive)")
        try:
            nid = self.new()
            self.memo[id(schema)] = nid
            self.keep.append(schema)
            self.nodes[nid] = self._compile_dict(schema)
        finally:
            self.depth -= 1
        return nid

    def _compile_dict(self, S: dict):
        _check_keywords(S)      # valeurs malformées → SchemaError, quel que soit le type
        if '$ref' in S:
            target = _resolve_ref(self.root, S['$ref'])
            if not _has_assertions(self.strip(S, '$ref')):
                return (_N_UNION, (self.compile(target),))
            return (_N_UNION, (self.compile(self.merged(self.conj(S))),))

        if 'allOf' in S:
            if not _has_assertions(self.strip(S, 'allOf')):
                return (_N_UNION, (self.compile(S['allOf'][0]),))
            return (_N_UNION, (self.compile(self.merged(self.conj(S))),))

        for key in ('anyOf', 'oneOf'):
            if key in S:
                alts = S[key]
                rest = self.strip(S, key)
                if _has_assertions(rest):
                    rc = self.conj(rest)
                    kids = []
                    for a in alts:
                        ca = self.conj(a)
                        both = None if rc is None or ca is None else _cat_atoms(rc, ca)
                        kids.append(self.compile(self.merged(both)))
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
            if vals is None:
                vals = [c]
            else:   # membres égaux à const (chacun sous SA forme) + la forme de const
                vals = [v for v in vals if _json_eq(v, c)]
                if vals:
                    vals.append(c)
        rest = _strip(S, ('enum', 'const'))
        if 'type' in rest:
            _type_list(rest['type'])        # type invalide → SchemaError même sans valeur
        # Les mots-clés frères sont vérifiés par _Validator (SchemaError si malformés)
        check = _Validator(self.root) if _has_assertions(rest) else None
        seen, kids = set(), []
        for v in vals:
            if not _is_json_value(v):
                continue
            # Déduplication par GRAMMAIRE (forme json.dumps), pas par égalité JSON : enum
            # [1, 1.0] garde '1' et '1.0', [0, -0.0] garde '-0.0', deux objets égaux écrits
            # dans un ordre de clés différent gardent chacun leur ordre de déclaration.
            key = _form_key(v)
            if key in seen:
                continue
            seen.add(key)
            if check is not None and check.errors(v, rest):
                continue    # incompatible avec les mots-clés frères (type…)
            kids.append(self.const(v))
        return (_N_UNION, tuple(dict.fromkeys(kids)))

    def const(self, v) -> int:
        """Valeur constante → grammaire exacte (espaces libres, échappements canoniques)."""
        if v is None:
            return self.cons(_LIT_NULL)
        if v is True:
            return self.cons(_LIT_TRUE)
        if v is False:
            return self.cons(_LIT_FALSE)
        if isinstance(v, (int, float)):
            try:
                lit = json.dumps(v)
            except ValueError as e:     # entier > sys.get_int_max_str_digits() chiffres
                raise SchemaError("entier constant trop long pour JSON (limite de conversion "
                                  "int ↔ str de Python) : json.loads ne pourrait pas le "
                                  "relire") from e
            return self.cons((_N_LIT, lit.encode('ascii')))
        if isinstance(v, str):
            return self.cons((_N_CSTR, self.sid(v)))
        if isinstance(v, list):
            return self.cons((_N_ARR, tuple(self.const(x) for x in v), -1, len(v), len(v)))
        return self.cons((_N_OBJ, tuple((k, self.sid(k), self.const(x), True)
                                        for k, x in v.items())))

    def _compile_types(self, S: dict):
        if 'type' in S:     # `type: null` explicite → SchemaError (pas « absent »)
            types = list(_type_list(S['type']))
        else:
            types = self._infer_types(S)
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
        if props is not None:
            _props_value(props)
        req = _required_value(S['required']) if 'required' in S else []
        # additionalProperties doit être un schéma (objet / booléen / null), même quand il
        # ne sert qu'aux noms requis absents de `properties` (cohérence avec l'objet libre
        # et avec validate_instance, qui le lit tel quel)
        addl = _check_subschema('additionalProperties', S.get('additionalProperties', True))
        if addl is None:
            addl = True
        if props is None and not req:
            # Objet libre : clés quelconques, valeurs selon additionalProperties
            if addl is False:
                return self.cons((_N_FOBJ, -1))
            return self.cons((_N_FOBJ, self.compile(addl)))
        # `properties` présent (même vide) : jamais de propriété additionnelle générée.
        # Noms comparés sous forme JSON canonique (_norm_str), comme validate_instance.
        props = props or {}
        reqset = {_norm_str(r) for r in req}
        plist, names = [], set()
        for name, sub in props.items():
            if not isinstance(name, str):
                raise SchemaError(f"nom de propriété invalide : {_brief(name)}")
            key = _norm_str(name)
            if key in names:
                raise SchemaError(f"propriété en double (paire de substitution) : {_brief(name)}")
            names.add(key)
            plist.append((key, self.sid(key), self.compile(sub), key in reqset))
        for name in dict.fromkeys(_norm_str(r) for r in req):
            if name not in names:
                plist.append((name, self.sid(name), self.compile(addl), True))
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

    def strip(self, S: dict, key: str) -> dict:
        """Copie de S sans `key`, mémoïsée (identité stable pour les mémos). Si S est un
        dict fusionné, la copie est la conjonction de ses atomes privés de `key`."""
        mk = (id(S), key)
        r = self._strips.get(mk)
        if r is None:
            r = _strip(S, (key,))
            self._strips[mk] = r
            self.keep.append(S)
            K = self._mkeys.get(id(S))
            if K is not None:
                atoms = []
                for a in K:
                    if key in a:
                        a = self.strip(a, key)
                        if not _has_assertions(a):
                            continue
                    atoms.append(a)
                self._mkeys[id(r)] = tuple(atoms)
        return r

    def conj(self, s, depth: int = 0):
        """
        Atomes de la conjonction équivalente à `s` (tuple ; () = tout ; None = rien) :
        $ref → atomes de la cible puis des frères ; allOf → frères puis sous-schéma ;
        autre dict → lui-même s'il porte une assertion. Ne descend jamais dans les
        propriétés : seule une chaîne $ref / allOf cyclique (sans descente dans
        l'instance, donc sans instance finie) est refusée.
        """
        if s is None or s is True:
            return ()
        if s is False:
            return None
        if not isinstance(s, dict):
            raise SchemaError(f"sous-schéma invalide dans une fusion : {_brief(s)}")
        r = self._mkeys.get(id(s), _UNSET)
        if r is _UNSET:
            r = self._conjs.get(id(s), _UNSET)
        if r is not _UNSET:
            return r
        if id(s) in self._busy or depth > _MAX_MERGE_DEPTH:
            raise SchemaError("chaîne de $ref / allOf cyclique ou trop longue dans une fusion")
        _check_keywords(s)
        self._busy.add(id(s))
        try:
            if '$ref' in s:
                target = _resolve_ref(self.root, s['$ref'])
                parts = (self.conj(target, depth + 1), self.conj(self.strip(s, '$ref'), depth + 1))
            elif 'allOf' in s:
                parts = (self.conj(self.strip(s, 'allOf'), depth + 1),
                         self.conj(s['allOf'][0], depth + 1))
            else:
                parts = ((s,) if _has_assertions(s) else (),)
        finally:
            self._busy.discard(id(s))
        r = None if any(p is None for p in parts) else _cat_atoms(*parts)
        self._conjs[id(s)] = r
        self.keep.append(s)
        return r

    def merged(self, atoms):
        """Schéma (dict fusionné canonique, atome seul, True ou False) d'une conjonction."""
        if atoms is None:
            return False
        if not atoms:
            return True
        if len(atoms) == 1:
            return atoms[0]
        key = tuple(id(a) for a in atoms)
        D = self._merged.get(key)
        if D is None:
            if len(self._merged) >= _MAX_MERGES:
                raise SchemaError("schéma trop complexe : fusions de schémas trop nombreuses")
            D = {}
            self._merged[key] = D
            self._mkeys[id(D)] = atoms
            self._todo.append((D, atoms))
            if not self._draining:          # file (pas de récursion) : propriétés imbriquées
                self._draining = True
                try:
                    while self._todo:
                        self._fold(*self._todo.pop())
                finally:
                    self._draining = False
        return D

    def _conj_all(self, subs):
        parts = [self.conj(s) for s in subs]
        return None if any(p is None for p in parts) else _cat_atoms(*parts)

    def _fold(self, out: dict, atoms) -> None:
        """
        Remplit `out` avec l'intersection des atomes (dans l'ordre). Les valeurs combinées
        sont vérifiées (SchemaError, jamais TypeError) ; celles d'un seul atome sont
        recopiées puis vérifiées à la compilation du schéma fusionné. Annotations,
        assertions non appliquées et mots-clés inconnus : première valeur conservée.
        """
        props = {}              # nom → sous-schémas des atomes qui le déclarent
        order = []              # ordre d'émission des propriétés fusionnées
        has_props = False       # au moins un atome porte `properties` (même null)
        names = set()           # noms déclarés par les atomes déjà fusionnés
        forms = []              # autres formes de `const` égales en JSON
        for b in atoms:
            bprops = _props_value(b.get('properties'))
            # additionalProperties d'un côté + propriétés inconnues de l'autre → non exprimable
            ap_a = _check_subschema('additionalProperties', out.get('additionalProperties', True))
            ap_b = _check_subschema('additionalProperties', b.get('additionalProperties', True))
            if ((ap_a is not True and ap_a is not None and not set(bprops) <= names)
                    or (ap_b is not True and ap_b is not None and not names <= set(bprops))):
                raise SchemaError("fusion non supportée : additionalProperties "
                                  "et propriétés d'un autre sous-schéma")
            ia = {k: out[k] for k in _ITEMS_KEYS if k in out}
            ib = {k: b[k] for k in _ITEMS_KEYS if k in b}
            if ia and ib and not _json_eq(ia, ib):      # pas `!=` : True == 1 en Python
                raise SchemaError("fusion non supportée : items / prefixItems différents")
            for k, vb in b.items():
                if k == 'properties':
                    has_props = True
                    order = _merge_order(order, props, bprops)
                    for name, sub in bprops.items():
                        props.setdefault(name, []).append(sub)
                    continue
                if k not in out:
                    out[k] = vb
                    continue
                if k not in _APPLIED:
                    continue        # annotation / assertion non appliquée / mot-clé inconnu
                va = out[k]
                if k == 'const':
                    if not _json_eq(va, vb):
                        out['enum'] = []
                    elif _form_key(va) != _form_key(vb):
                        forms.append(vb)    # const ∩ const : les deux formes restent permises
                    continue
                if k != 'enum' and _json_eq(va, vb):
                    continue    # égalité JSON (bool ≠ nombre), pas `==` Python
                if k == 'type':
                    ta, tb = set(_type_list(va)), set(_type_list(vb))
                    for t1, t2 in ((ta, tb), (tb, ta)):
                        if 'number' in t1 and 'integer' in t2:
                            t1.add('integer')
                    out[k] = [t for t in _TYPES if t in ta and t in tb]
                elif k == 'required':
                    out[k] = list(dict.fromkeys(_required_value(va) + _required_value(vb)))
                elif k in ('minLength', 'minItems'):
                    out[k] = max(_nonneg_value(k, va), _nonneg_value(k, vb))
                elif k in ('maxLength', 'maxItems'):
                    out[k] = min(_nonneg_value(k, va), _nonneg_value(k, vb))
                elif k == 'enum':
                    if not isinstance(va, list) or not isinstance(vb, list):
                        raise SchemaError("enum doit être une liste")
                    # Intersection au sens JSON ; chaque membre retenu garde SA forme (des deux
                    # côtés : enum [1] ∩ enum [1.0] accepte '1' et '1.0', comme validate_instance)
                    va = [v for v in va if _is_json_value(v)]
                    vb = [w for w in vb if _is_json_value(w)]
                    ka = {_json_key(v) for v in va}
                    kb = {_json_key(w) for w in vb}
                    both = ([v for v in va if _json_key(v) in kb]
                            + [w for w in vb if _json_key(w) in ka])
                    seen, res = set(), []
                    for v in both:
                        fk = _form_key(v)
                        if fk not in seen:
                            seen.add(fk)
                            res.append(v)
                    out[k] = res
                else:
                    raise SchemaError(f"fusion de schémas non supportée pour « {k} »")
            names |= set(bprops)
        if has_props:
            # Propriété déclarée par un seul atome : son sous-schéma tel quel ; par plusieurs :
            # conjonction (dict fusionné canonique, éventuellement en cours de construction)
            out['properties'] = {name: props[name][0] if len(props[name]) == 1
                                 else self.merged(self._conj_all(props[name]))
                                 for name in order}
        if forms:
            # const ∩ const égaux en JSON : chaque forme reste écrivable (comme enum ∩ enum),
            # sauf si un enum (intersecté) exclut déjà la valeur
            c = out['const']
            if 'enum' not in out:
                out['enum'] = [c] + forms
            elif any(_is_json_value(v) and _json_eq(v, c) for v in out['enum']):
                out['enum'] = list(out['enum']) + forms


def _merge_order(order: list, placed, names) -> list:
    """
    Ordre des propriétés d'une fusion : celles d'un nouvel atome déjà placées gardent leur
    place ; chaque nouvelle est insérée juste avant la propriété suivante de CE MÊME atome
    déjà placée (sinon à la fin). L'ordre de déclaration de chaque atome est ainsi
    respecté quand c'est possible : Category = allOf [Named {name, parent}] + {parent}
    → name, parent ; allOf [{id}] + {extra} → extra, id (frères d'abord).
    """
    before, anchor = {}, None
    for name in reversed(list(names)):
        if name in placed:
            anchor = name
        else:
            before.setdefault(anchor, []).append(name)
    if not before:
        return order
    new = []
    for name in order:
        if name in before:
            new.extend(reversed(before[name]))
        new.append(name)
    new.extend(reversed(before.get(None, ())))
    return new


def _cat_atoms(*parts) -> tuple:
    """Concaténation de tuples d'atomes, doublons (identité) retirés, ordre conservé."""
    seen, out = set(), []
    for p in parts:
        for a in p:
            if id(a) not in seen:
                seen.add(id(a))
                out.append(a)
    return tuple(out)


# ─────────────────────────────────────────────────────────────────────────────
# Automate : frames, phases, tables d'octets
# ─────────────────────────────────────────────────────────────────────────────

# Frames (éléments de pile) :
#   (_F_ROOT, ws)                                   avant la valeur racine
#   (_F_STR, minLen, maxLen, compte, sous_état, haut_en_attente)
#   (_F_CSTR, sid, index_caractère, octets_partiels)  chaîne constante (clé, enum d'UNE
#                                                   chaîne)
#   (_F_NUM, entier, phase, chiffres)
#   (_F_LIT, octets, index)                         true / false / null / nombre constant
#   (_F_OBJ, nid, phase, position, ws)              objet à propriétés ordonnées
#   (_F_FOBJ, nid, phase, ws)                       objet libre
#   (_F_ARR, nid, phase, compte, ws)
#   (_F_CSET, cid, lo, hi, profondeur, octets_partiels)
#                                                   ENSEMBLE de chaînes constantes (enum de
#                                                   plusieurs chaînes) : trie implicite sur
#                                                   les chaînes triées, [lo, hi) = chaînes
#                                                   qui partagent les `profondeur` premiers
#                                                   caractères déjà lus
_F_ROOT, _F_STR, _F_CSTR, _F_NUM, _F_LIT, _F_OBJ, _F_FOBJ, _F_ARR, _F_CSET = range(9)

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
_CHR = tuple(chr(i) for i in range(256))
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


def _menc(s: str) -> bytes:
    """Encodage le plus court d'une chaîne constante (sans guillemets) : _char_encodings[0]
    de chaque caractère (ASCII imprimable hors '"' et '\\' : la chaîne elle-même)."""
    if s.isascii() and s.isprintable() and '"' not in s and '\\' not in s:
        return s.encode('ascii')
    return b''.join(_char_encodings(ch)[0] for ch in s)


def _is_plain(ch: str) -> bool:
    """Caractère dont le SEUL encodage dans une chaîne constante est son octet brut."""
    return ' ' <= ch <= '~' and ch != '"' and ch != '\\'


def _lb(strs, lo, hi, d, ch):
    """Premier indice de [lo, hi) dont le caractère d est ≥ ch (chaînes triées, len > d)."""
    while lo < hi:
        m = (lo + hi) >> 1
        if strs[m][d] < ch:
            lo = m + 1
        else:
            hi = m
    return lo


def _ub(strs, lo, hi, d, ch):
    """Premier indice de [lo, hi) dont le caractère d est > ch (chaînes triées, len > d)."""
    while lo < hi:
        m = (lo + hi) >> 1
        if strs[m][d] <= ch:
            lo = m + 1
        else:
            hi = m
    return lo


def _join(*parts):
    """Concaténation, None si une partie manque (impasse)."""
    return None if None in parts else b''.join(parts)


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

class _Stk:
    """
    Nœud de pile immuable CHAÎNÉE : `top` = frame, `up` = reste de la pile — None (pile
    vide), un _Stk, ou un _Fork (plusieurs restes possibles : pile en graphe). Une
    CONFIGURATION de l'automate est un _Stk dont `up` n'est jamais un _Fork (sommet +
    parent unique ; une configuration None = racine terminée) ; les embranchements
    n'apparaissent que plus bas. La queue est partagée entre configurations : empiler,
    dépiler ou remplacer le sommet coûte O(1) en temps et en mémoire quelle que soit la
    profondeur. Hash calculé une fois ; égalité structurelle ITÉRATIVE (_graph_eq, aucune
    récursion) avec raccourci d'identité.
    `tl` : cache de la longueur de fin des frames parents (cf. _tail_len), -2 = inconnu.
    """

    __slots__ = ('top', 'up', 'h', 'tl')

    def __init__(self, top, up):
        self.top = top
        self.up = up
        self.h = hash((top, 0 if up is None else up.h))
        self.tl = -2

    def __hash__(self):
        return self.h

    def __eq__(self, other):
        if type(other) is not _Stk:
            return NotImplemented
        return _graph_eq(self, other)

    def __repr__(self):
        frames, s = [], self
        while s is not None:
            if type(s) is _Fork:
                frames.append(f"<{len(s.items)} restes>")
                break
            frames.append(s.top)
            s = s.up
        return f"_Stk{tuple(reversed(frames))!r}"


class _Fork:
    """
    Embranchement d'une pile en graphe : ensemble (≥ 2) de restes possibles — _Stk ou
    None (fond de pile). Deux configurations de même sommet et de même frame parent sont
    fusionnées en une seule dont le parent pointe vers l'union de leurs restes : une union
    récursive ambiguë ({replies:[#], text} | {replies:[#], image}…) garde ainsi un nombre
    de configurations borné par la grammaire, au lieu de 2^profondeur piles distinctes.
    `tl` / `best` : longueur et reste de la plus courte fin (cf. _tail_len).
    """

    __slots__ = ('items', 'h', 'tl', 'best')

    def __init__(self, items: frozenset):
        self.items = items
        self.h = hash((_FORK_TAG, items))
        self.tl = -2
        self.best = None

    def __hash__(self):
        return self.h

    def __eq__(self, other):
        if type(other) is not _Fork:
            return NotImplemented
        return _graph_eq(self, other)


_FORK_TAG = 0x51F0
_NONE_H = hash(None)


def _graph_eq(a, b) -> bool:
    """Égalité structurelle de deux piles en graphe, itérative (file de paires)."""
    work = [(a, b)]
    while work:
        a, b = work.pop()
        while a is not b:
            if a is None or b is None:
                return False
            t = type(a)
            if t is not type(b) or a.h != b.h:
                return False
            if t is _Stk:
                if a.top != b.top:
                    return False
                a, b = a.up, b.up
                continue
            ia, ib = a.items, b.items       # _Fork : appariement par hash
            if len(ia) != len(ib):
                return False
            byh = {}
            for y in ib:
                hy = _NONE_H if y is None else y.h
                if hy in byh:
                    return ia == ib     # collision de hash interne (rarissime) : récursif
                byh[hy] = y
            for x in ia:
                y = byh.get(_NONE_H if x is None else x.h, _UNSET)
                if y is _UNSET:
                    return False
                if x is not y:
                    work.append((x, y))
            break
    return True


def _rest_of(items, forks=None):
    """Reste unique (_Stk / None) ou _Fork pour un ensemble de restes (sans _Fork) ;
    `forks` : mémo local frozenset → _Fork (embranchements identiques partagés)."""
    if len(items) == 1:
        for x in items:
            return x
    fs = frozenset(items)
    if forks is None:
        return _Fork(fs)
    f = forks.get(fs)
    if f is None:
        f = forks[fs] = _Fork(fs)
    return f


def _merge_cfgs(out: set) -> set:
    """
    Forme normale d'un état : les configurations de même sommet ET de même frame parent
    sont fusionnées en une seule, dont le parent pointe vers l'union de leurs restes
    (_Fork). Le langage reconnu est inchangé (mêmes piles représentées) ; le nombre de
    configurations reste borné par la grammaire, même quand une union récursive ambiguë
    garde ses alternatives ouvertes à chaque niveau (sinon 2^profondeur). Des
    configurations de même sommet mais de parents différents (40 variantes oneOf qui
    partagent une chaîne…) restent distinctes.
    """
    tops = {}
    dup = False
    for c in out:
        if c is not None:
            t = c.top
            g = tops.get(t)
            if g is None:
                tops[t] = [c]
            else:
                g.append(c)
                dup = True
    if not dup:
        return out
    res = set()
    forks = {}
    for t, cs in tops.items():
        if len(cs) == 1:
            res.add(cs[0])
            continue
        byp = {}
        for c in cs:
            p = c.up
            if p is None:   # fond de pile : rien à fusionner ((t, None) est unique)
                res.add(c)
            else:
                byp.setdefault(p.top, []).append(c)
        for pt, grp in byp.items():
            if len(grp) == 1:
                res.add(grp[0])
                continue
            items = set()
            for c in grp:
                u = c.up.up
                if type(u) is _Fork:
                    items |= u.items
                else:
                    items.add(u)
            res.add(_Stk(t, _Stk(pt, _rest_of(items, forks))))
    if None in out:
        res.add(None)
    return res


class JSONSchemaMatcher:
    """
    Byte-level incremental matcher. States are IMMUTABLE and HASHABLE (so they can be
    memoized/cached).

    Automate à pile non déterministe : un état = frozenset de configurations (piles
    immuables chaînées, cf. _Stk). Les états sont internés et toutes les transitions
    mémoïsées ; les caches sont plafonnés (nombre d'états, et octets pour les complétions).
    """

    def __init__(self, schema=None, *, max_whitespace: int = 4, max_number_digits: int = 20,
                 filler: str = 'a'):
        """
        filler : caractère ASCII imprimable (ni '"' ni '\\') qui remplit les chaînes
                 minLength dans les complétions (shortest_completion / completion_tokens) ;
                 json_constraint le choisit selon le vocabulaire (cf. _VocabTrie.filler).
        """
        if schema is False:
            raise SchemaError("schéma `false` : aucune instance possible")
        if schema is not None and not isinstance(schema, (dict, bool)):
            raise SchemaError(f"un schéma doit être un objet JSON, obtenu {type(schema).__name__}")
        if int(max_whitespace) < 0:
            raise ValueError("max_whitespace doit être ≥ 0")
        if int(max_number_digits) < 1:
            raise ValueError("max_number_digits doit être ≥ 1")
        # Un entier de plus de sys.get_int_max_str_digits() chiffres (4300 par défaut,
        # CPython ≥ 3.11 / 3.10.7 / 3.9.14) est refusé par json.loads : un document accepté
        # doit toujours être décodable → réglage refusé plutôt qu'automate trop permissif
        lim = getattr(sys, 'get_int_max_str_digits', lambda: 0)()
        if lim and int(max_number_digits) > lim:
            raise ValueError(f"max_number_digits ({int(max_number_digits)}) > limite de "
                             f"conversion int ↔ str de Python ({lim}) : json.loads refuserait "
                             "les entiers acceptés")
        try:
            fb = filler.encode('ascii') if isinstance(filler, str) else bytes(filler)
        except (UnicodeEncodeError, TypeError, ValueError):
            fb = b''
        if len(fb) != 1 or not 0x20 <= fb[0] <= 0x7E or fb in (b'"', b'\\'):
            raise ValueError("filler doit être un caractère ASCII imprimable autre que '\"' "
                             f"et '\\\\', obtenu {_brief(filler)}")
        self.schema = schema
        self.max_whitespace = int(max_whitespace)
        self.max_number_digits = int(max_number_digits)
        self.filler = fb.decode('ascii')
        self._fill = fb

        try:
            self.ignored_keywords = _scan_ignored(schema)
            comp = _Compiler(schema)
            root = comp.compile(schema)
            self._root = root
            self._nodes = comp.nodes
            self._build_strings(comp.strings)
            self._finalize()
        except RecursionError as e:
            raise SchemaError("schéma trop profond") from e
        except (MemoryError, OverflowError) as e:
            raise SchemaError(f"schéma trop gros ({type(e).__name__})") from e
        if self._W[root] is None:
            raise SchemaError("schéma insatisfiable : aucune instance JSON finie")

        # Caches bornés : vidés au-delà de _max_states états. Un état coûte O(1) octets
        # quelle que soit la profondeur (piles chaînées) et sa ligne de transitions est
        # creuse (~0,7 Ko en tout par état) → ~15 Mo au pire ; un masque frais crée au plus
        # quelques milliers d'états. Les complétions sont en plus bornées en octets cumulés.
        self._max_states = 20_000
        self._max_completion_bytes = 1 << 23
        self._completion_bytes = 0
        self._rows = {}           # état → {octet: état suivant ou None} (calculées seulement)
        self._interned = {}       # état → état (objet canonique)
        self._start_memo = {}     # (nid, octet) → frames de départ
        self._accept_cache = {}
        self._allowed_cache = {}
        self._completion_cache = {}
        self._parent_memo = {}    # frame parent → fin minimale (octets), borné
        self._parent_bytes = 0
        self._fallbacks = 0       # nb de complétions passées par le BFS de secours
        init = frozenset({_Stk((_F_ROOT, 0), None)})
        self._initial = self._interned.setdefault(init, init)

    # ── Préparation ──────────────────────────────────────────────────────────

    def _build_strings(self, strings):
        """
        Chaînes constantes : forme minimale (témoins) calculée d'emblée ; tables par
        caractère (préfixes d'encodage, encodages, minimaux) construites à la première
        lecture (_cstr) — les chaînes d'un enum de plusieurs chaînes, lues par un frame
        _F_CSET, n'en ont jamais besoin (8 000 URL : ~120 Mo et 3 s économisés).
        """
        self._strings = strings
        self._cstr_lit = [b'"' + _menc(s) + b'"' for s in strings]
        self._cstr_tabs = [None] * len(strings)
        self._csets = []          # cid → (chaînes triées, sids alignés)
        self._cset_ids = {}       # tuple(sids triés) → cid
        self._cset_tabs = {}      # (cid, lo, hi, profondeur) → table des encodages partiels
        self._cset_best = {}      # (cid, lo, hi) → indice de la chaîne la plus courte

    def _cstr(self, sid):
        """(tables de préfixes, encodages, encodages minimaux) par caractère de la chaîne."""
        t = self._cstr_tabs[sid]
        if t is None:
            tabs, encs, mins = [], [], []
            for ch in self._strings[sid]:
                e = _char_encodings(ch)
                d = {}
                for enc in e:
                    for k in range(1, len(enc)):
                        d.setdefault(enc[:k], False)
                    d[enc] = True
                tabs.append(d)
                encs.append(e)
                mins.append(e[0])
            t = self._cstr_tabs[sid] = (tuple(tabs), tuple(encs), tuple(mins))
        return t

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

        # 2) Point fixe : plus court témoin (octets) de chaque nœud ; None = improductif.
        #    Parcours à rebours : les enfants sont en général alloués après leur parent,
        #    donc convergence en ~2 tours. Un témoin > _MAX_WITNESS n'est jamais construit.
        W = [None] * N
        too_long = set()        # nœuds dont un témoin candidat dépassait _MAX_WITNESS

        def wv(n):
            return _minb(*(W[a] for a in alts[n]))

        changed = True
        while changed:
            changed = False
            for n in range(N - 1, -1, -1):
                if nodes[n][0] == _N_UNION:
                    continue
                self._too_long = False
                w = self._witness(nodes[n], wv)
                if self._too_long:
                    too_long.add(n)
                if w is not None and (W[n] is None or (len(w), w) < (len(W[n]), W[n])):
                    W[n] = w
                    changed = True
        self._too_long = False

        if any(W[n] is None for n in too_long):
            # Ce nœud n'a de témoin que plus long que la limite (minLength / minItems
            # démesurés) : refus explicite plutôt qu'une branche silencieusement retirée
            raise SchemaError(f"instance minimale trop longue (> {_MAX_WITNESS} octets) : "
                              "bornes minLength / minItems démesurées")
        self._W = [wv(n) for n in range(N)]
        self._palts = [tuple(a for a in alts[n] if W[a] is not None) for n in range(N)]

        # 3) Tables par nœud (utilisées par les transitions et les complétions)
        self._info = [None] * N
        for n in range(N):
            node = nodes[n]
            k = node[0]
            if k == _N_OBJ:
                props = node[1]
                cand, cc, entry, AV, AC, prod = self._obj_tables(props, wv)
                self._info[n] = (tuple(p[1] for p in props), tuple(p[2] for p in props),
                                 cand, cc, AV, AC, prod)
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

    def _cat(self, *parts):
        """Concaténation bornée : None si une partie manque ou si > _MAX_WITNESS octets."""
        if any(p is None for p in parts):
            return None
        if sum(len(p) for p in parts) > _MAX_WITNESS:
            self._too_long = True
            return None
        return b''.join(parts)

    def _obj_tables(self, props, wv):
        """
        Programmation dynamique sur les propriétés ordonnées, en O(P) :
          cand[p]  : (lo, hi) → prod[lo:hi] = propriétés pouvant venir ensuite (on saute
                     les optionnelles, jusqu'à la première requise incluse)
          cc[p]    : '}' permis (plus aucune requise à partir de p)
          AV[p]    : plus courte fin après une valeur, p propriétés traitées
          AC[p]    : plus courte fin après une virgule
                     AC[p] = min(entry[p] + AV[p+1], AC[p+1] si p optionnelle)
          prod     : indices des propriétés productives (témoin fini), triés
        """
        P = len(props)
        entry = []
        for (_, sid, v, _) in props:
            entry.append(self._cat(self._cstr_lit[sid], b':', wv(v)))
        cc = [True] * (P + 1)
        for p in range(P - 1, -1, -1):
            cc[p] = cc[p + 1] and not props[p][3]
        prod = tuple(j for j in range(P) if entry[j] is not None)
        cand = [None] * (P + 1)
        cand[P] = (len(prod), len(prod))
        last = P - 1                     # première requise ≥ p (sinon la dernière)
        for p in range(P - 1, -1, -1):
            if props[p][3]:
                last = p
            cand[p] = (bisect_left(prod, p), bisect_right(prod, last))
        AV = [None] * (P + 1)
        AC = [None] * (P + 1)
        AV[P] = b'}'
        total = 0       # une longue suite de propriétés REQUISES donne des fins en O(P²)
        for p in range(P - 1, -1, -1):
            AC[p] = _minb(self._cat(entry[p], AV[p + 1]) if entry[p] is not None else None,
                          AC[p + 1] if not props[p][3] else None)
            AV[p] = _minb(b'}' if cc[p] else None,
                          self._cat(b',', AC[p]) if AC[p] is not None else None)
            total += len(AV[p] or b'') + len(AC[p] or b'')
            if total > _MAX_TABLES:
                raise SchemaError("schéma trop gros : tables de complétion d'un objet "
                                  f"> {_MAX_TABLES} octets (propriétés requises trop nombreuses)")
        return tuple(cand), tuple(cc), entry, tuple(AV), tuple(AC), prod

    def _witness(self, node, wv):
        """Plus courte valeur JSON (octets) d'un nœud non-union, ou None."""
        k = node[0]
        if k == _N_STR:
            _, mn, mx = node
            if mx is not None and mn > mx:
                return None
            if mn + 2 > _MAX_WITNESS:
                self._too_long = True
                return None
            return b'"' + self._fill * mn + b'"'
        if k == _N_NUM:
            return b'0'
        if k == _N_LIT:
            return node[1]
        if k == _N_CSTR:
            return self._cstr_lit[node[1]]
        if k == _N_FOBJ:
            return b'{}'
        if k == _N_OBJ:
            cand, cc, entry, AV, AC, prod = self._obj_tables(node[1], wv)
            fin = _minb(b'}' if cc[0] else None, AC[0])
            return None if fin is None else b'{' + fin
        # _N_ARR — longueur calculée AVANT de matérialiser (minItems démesuré)
        _, prefix, items, mn, mx = node
        if mx is not None and mn > mx:
            return None
        parts = []
        for i in range(min(mn, len(prefix))):
            w = wv(prefix[i])
            if w is None:
                return None
            parts.append(w)
        extra = mn - len(parts)
        size = 2 + sum(len(w) + 1 for w in parts)
        if extra > 0:
            w = wv(items) if items >= 0 else None
            if w is None:
                return None
            size += extra * (len(w) + 1)
            if size > _MAX_WITNESS + 1:
                self._too_long = True
                return None
            parts.extend([w] * extra)
        if size > _MAX_WITNESS + 1:
            self._too_long = True
            return None
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
        r = row.get(byte, _UNSET)
        if r is _UNSET:
            r = row[byte] = self._compute(state, byte)
        return r

    def advance_bytes(self, state, data: bytes):
        """État après `data` (None dès qu'un octet est refusé). Boucle en ligne pour les
        octets (pas d'appel de méthode par octet : longues complétions vérifiées)."""
        if not isinstance(data, (bytes, bytearray)):
            for b in data:
                state = self.advance(state, b)
                if state is None:
                    return None
            return state
        rows = self._rows           # vidé EN PLACE par _new_row : la référence reste valide
        for b in data:
            if state is None:
                return None
            row = rows.get(state)
            if row is None:
                row = self._new_row(state)
            r = row.get(b, _UNSET)
            if r is _UNSET:
                r = row[b] = self._compute(state, b)
            state = r
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
        """Au moins un octet supplémentaire est acceptable (s'arrête au premier trouvé)."""
        if state is None:
            return False
        r = self._allowed_cache.get(state)
        if r is not None:
            return bool(r)
        return any(self.advance(state, b) is not None for b in range(256))

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
        cache = self._completion_cache
        best = cache.get(state, _UNSET)
        if best is not _UNSET:
            return best
        best = _minb(*(self._cfg_completion(c) for c in state))
        if best is None or not self.is_accepting(self.advance_bytes(state, best)):
            self._fallbacks += 1
            best = self._bfs_completion(state)
        size = 64 + (len(best) if best is not None else 0)
        if (len(cache) >= self._max_states
                or self._completion_bytes + size > self._max_completion_bytes):
            cache.clear()       # plafond en nombre d'états ET en octets cumulés
            self._completion_bytes = 0
        cache[state] = best
        self._completion_bytes += size
        return best

    # ── Transitions ──────────────────────────────────────────────────────────

    def _new_row(self, state):
        """Ligne de transitions CREUSE (dict octet → état) : seuls les octets essayés
        sont stockés (~200 o par état au lieu de 2 Ko pour 256 cases)."""
        if len(self._rows) >= self._max_states:
            self._rows.clear()
            self._interned.clear()
            self._accept_cache.clear()
        row = {}
        self._rows[state] = row
        return row

    def _compute(self, state, b):
        out = set()
        step = self._step
        for cfg in state:
            if cfg is not None:     # None = racine terminée : plus aucun octet
                step(cfg, b, out)   # (pas d'espace final)
        if not out:
            return None
        if len(out) > 1:
            out = _merge_cfgs(out)
        fs = frozenset(out)
        interned = self._interned
        if len(interned) >= 4 * self._max_states:
            interned.clear()
        return interned.setdefault(fs, fs)

    def _start(self, n, b):
        """Frames créés quand l'octet `b` débute une valeur du nœud n (None = valeur finie).
        Plusieurs chaînes constantes (enum) → UN frame _F_CSET (trie implicite)."""
        key = (n, b)
        r = self._start_memo.get(key)
        if r is not None:
            return r
        res = []
        cstr = []
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
                    cstr.append(node[1])
            elif k == _N_OBJ:
                if b == 0x7B:
                    res.append((_F_OBJ, a, _O_OPEN, 0, 0))
            elif k == _N_FOBJ:
                if b == 0x7B:
                    res.append((_F_FOBJ, a, _O_OPEN, 0))
            elif k == _N_ARR:
                if b == 0x5B:
                    res.append((_F_ARR, a, _A_OPEN, 0, 0))
        cstr = sorted(set(cstr))
        if len(cstr) == 1:
            res.append((_F_CSTR, cstr[0], 0, b''))
        elif cstr:
            res.append(self._cset_frame(tuple(cstr)))
        r = self._start_memo[key] = tuple(dict.fromkeys(res))
        return r

    def _cset_frame(self, sids):
        """Frame initial d'un ensemble de chaînes constantes (un cid par ensemble de sids)."""
        cid = self._cset_ids.get(sids)
        if cid is None:
            pairs = sorted((self._strings[s], s) for s in sids)     # ordre des points de code
            cid = self._cset_ids[sids] = len(self._csets)
            self._csets.append((tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)))
        return (_F_CSET, cid, 0, len(sids), 0, b'')

    def _pop_into(self, rest, out):
        """
        Le frame du sommet vient de se terminer : le parent (sommet de `rest`) passe à la
        phase suivante. `rest` None → la valeur racine est terminée (configuration None).
        Si le parent a plusieurs restes possibles (_Fork), une configuration par reste.
        """
        if rest is None:
            out.add(None)
            return
        par = rest.top
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
        up = rest.up
        if type(up) is _Fork:
            for u in up.items:
                out.add(_Stk(np, u))
        else:
            out.add(_Stk(np, up))

    def _step(self, cfg, b, out):
        """Avance une configuration d'un octet ; ajoute les configurations résultantes à out."""
        fr = cfg.top
        k = fr[0]
        if k == _F_STR:
            self._step_str(cfg, fr, b, out)
        elif k == _F_OBJ:
            self._step_obj(cfg, fr, b, out)
        elif k == _F_CSTR:
            self._step_cstr(cfg, fr, b, out)
        elif k == _F_CSET:
            self._step_cset(cfg, fr, b, out)
        elif k == _F_NUM:
            self._step_num(cfg, fr, b, out)
        elif k == _F_LIT:
            lit, i = fr[1], fr[2]
            if b == lit[i]:
                if i + 1 == len(lit):
                    self._pop_into(cfg.up, out)
                else:
                    out.add(_Stk((_F_LIT, lit, i + 1), cfg.up))
        elif k == _F_ARR:
            self._step_arr(cfg, fr, b, out)
        elif k == _F_FOBJ:
            self._step_fobj(cfg, fr, b, out)
        else:   # _F_ROOT
            if _IS_WS[b]:
                if fr[1] < self.max_whitespace:
                    out.add(_Stk((_F_ROOT, fr[1] + 1), None))
                return
            for f in self._start(self._root, b):
                out.add(None if f is None else _Stk(f, None))

    def _step_str(self, cfg, fr, b, out):
        _, mn, mx, cnt, sub, pend = fr
        tracked = mn > 0 or mx is not None
        if sub == _S_NORM:
            if b == 0x22:
                if cnt >= mn:
                    self._pop_into(cfg.up, out)
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
            out.add(_Stk(nf, cfg.up))
            return

        cont = _UTF8_CONT.get(sub)
        if cont is not None:
            if cont[0] <= b <= cont[1]:
                out.add(_Stk((_F_STR, mn, mx, cnt, cont[2], 0), cfg.up))
            return
        if sub == _S_BS:
            if b in _SHORT_ESC:
                nf = (_F_STR, mn, mx, cnt, _S_NORM, 0)
            elif b == 0x75:
                nf = (_F_STR, mn, mx, cnt, _S_U0, pend)
            else:
                return
            out.add(_Stk(nf, cfg.up))
            return
        if sub == _S_BSL:
            if b == 0x75:
                out.add(_Stk((_F_STR, mn, mx, cnt, _S_U0L, pend), cfg.up))
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
            out.add(_Stk((_F_STR, mn, mx, cnt, _S_NORM, 1), cfg.up))
            return
        elif sub == _S_U3L:
            out.add(_Stk((_F_STR, mn, mx, cnt - 1 if pend else cnt, _S_NORM, 0), cfg.up))
            return
        else:   # _S_U3X
            out.add(_Stk((_F_STR, mn, mx, cnt, _S_NORM, 0), cfg.up))
            return
        out.add(_Stk((_F_STR, mn, mx, cnt, ns, pend), cfg.up))

    def _step_cstr(self, cfg, fr, b, out):
        _, sid, i, part = fr
        tab = self._cstr(sid)[0]
        if i == len(tab):
            if b == 0x22 and not part:
                self._pop_into(cfg.up, out)
            return
        if 0x41 <= b <= 0x46 and part[:2] == b'\\u':
            b |= 0x20       # hexadécimal insensible à la casse (forme canonique minuscule)
        np = part + _BYTE[b]
        full = tab[i].get(np)
        if full is None:
            return
        nf = (_F_CSTR, sid, i + 1, b'') if full else (_F_CSTR, sid, i, np)
        out.add(_Stk(nf, cfg.up))

    # ── Ensembles de chaînes constantes (enum) : trie implicite ──────────────
    #   Les chaînes de l'ensemble sont triées par points de code : celles qui partagent
    #   les d premiers caractères forment un intervalle [lo, hi), dans lequel le caractère
    #   d est croissant (la chaîne de longueur d, s'il y en a une, en tête). Un octet
    #   ASCII imprimable (hors '"' et '\\') ne peut écrire que lui-même → recherche
    #   dichotomique du sous-intervalle ; les autres octets (échappements, UTF-8, DEL)
    #   passent par une table des encodages des enfants « non bruts », construite à la
    #   demande. Mêmes encodages que _F_CSTR (_char_encodings), une SEULE configuration
    #   quel que soit le nombre de chaînes (8 000 URL à préfixe commun : coût par octet
    #   en O(log n) au lieu de O(n)).

    def _step_cset(self, cfg, fr, b, out):
        _, cid, lo, hi, d, part = fr
        strs = self._csets[cid][0]
        if not part:
            if b == 0x22:
                if len(strs[lo]) == d:
                    self._pop_into(cfg.up, out)
                return
            if 0x20 <= b <= 0x7E and b != 0x5C:
                if len(strs[lo]) == d:
                    lo += 1
                ch = _CHR[b]
                if hi - lo == 1:
                    if strs[lo][d] != ch:
                        return
                    nlo, nhi = lo, hi
                else:
                    nlo = _lb(strs, lo, hi, d, ch)
                    nhi = _ub(strs, nlo, hi, d, ch)
                    if nlo == nhi:
                        return
                out.add(_Stk((_F_CSET, cid, nlo, nhi, d + 1, b''), cfg.up))
                return
            if b < 0x20:
                return      # octet de contrôle brut : jamais permis dans une chaîne
        elif 0x41 <= b <= 0x46 and part[:2] == b'\\u':
            b |= 0x20       # hexadécimal insensible à la casse (forme canonique minuscule)
        np = part + _BYTE[b]
        e = self._cset_table(cid, lo, hi, d).get(np)
        if e is None:
            return
        nlo, nhi, longer = e
        if nlo >= 0:        # encodage complet d'un caractère
            out.add(_Stk((_F_CSET, cid, nlo, nhi, d + 1, b''), cfg.up))
        if longer:          # préfixe d'un encodage plus long (ex. \ud83d → 😀)
            out.add(_Stk((_F_CSET, cid, lo, hi, d, np), cfg.up))

    def _cset_kids(self, cid, lo, hi, d):
        """Caractères d suivants « non bruts » (contrôle, '"', '\\', DEL et non-ASCII) de
        l'intervalle : (caractère, lo, hi) de chaque sous-intervalle."""
        strs = self._csets[cid][0]
        if lo < hi and len(strs[lo]) == d:
            lo += 1
        segs = ((lo, _lb(strs, lo, hi, d, ' ')),
                (_lb(strs, lo, hi, d, '"'), _lb(strs, lo, hi, d, '#')),
                (_lb(strs, lo, hi, d, '\\'), _lb(strs, lo, hi, d, ']')),
                (_lb(strs, lo, hi, d, '\x7f'), hi))
        for a, z in segs:
            while a < z:
                ch = strs[a][d]
                j = _ub(strs, a, z, d, ch)
                yield ch, a, j
                a = j

    def _cset_table(self, cid, lo, hi, d):
        """Préfixe d'encodage → (lo, hi du caractère s'il est complet sinon -1, -1,
        préfixe d'un encodage plus long ?) pour les enfants non bruts de [lo, hi)."""
        key = (cid, lo, hi, d)
        tab = self._cset_tabs.get(key)
        if tab is None:
            tab = {}
            for ch, i, j in self._cset_kids(cid, lo, hi, d):
                for enc in _char_encodings(ch):
                    for k in range(1, len(enc)):
                        p = enc[:k]
                        old = tab.get(p)
                        tab[p] = (-1, -1, True) if old is None else (old[0], old[1], True)
                    old = tab.get(enc)
                    tab[enc] = (i, j, False) if old is None else (i, j, old[2])
            if len(self._cset_tabs) >= 4096:
                self._cset_tabs.clear()
            self._cset_tabs[key] = tab
        return tab

    def _cset_argbest(self, cid, lo, hi):
        """Indice de la chaîne de [lo, hi) à l'encodage minimal le plus court (puis le plus
        petit) : toutes partagent le préfixe, donc leur suffixe minimal l'est aussi."""
        key = (cid, lo, hi)
        r = self._cset_best.get(key)
        if r is None:
            sids = self._csets[cid][1]
            lit = self._cstr_lit
            r = min(range(lo, hi), key=lambda i: (len(lit[sids[i]]), lit[sids[i]]))
            if len(self._cset_best) >= 4096:
                self._cset_best.clear()
            self._cset_best[key] = r
        return r

    def _cset_finish(self, fr):
        """Plus courte fin (octets, '"' compris) d'un frame _F_CSET."""
        _, cid, lo, hi, d, part = fr
        strs = self._csets[cid][0]
        if not part:
            return _menc(strs[self._cset_argbest(cid, lo, hi)][d:]) + b'"'
        best = None
        for ch, i, j in self._cset_kids(cid, lo, hi, d):
            for enc in _char_encodings(ch):
                if len(enc) > len(part) and enc.startswith(part):
                    rest = _menc(strs[self._cset_argbest(cid, i, j)][d + 1:])
                    best = _minb(best, enc[len(part):] + rest + b'"')
        return best

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
            out.add(_Stk(nf, cfg.up))
        elif _P_COMPLETE[ph] and cfg.up is not None:
            # Fin implicite du nombre : l'octet est passé au parent (à chacun de ses restes)
            parents = set()
            self._pop_into(cfg.up, parents)
            for p in parents:
                self._step(p, b, out)

    def _step_obj(self, cfg, fr, b, out):
        _, n, ph, p, ws = fr
        rest = cfg.up
        if _IS_WS[b]:
            if ws < self.max_whitespace:
                out.add(_Stk((_F_OBJ, n, ph, p, ws + 1), rest))
            return
        info = self._info[n]        # (sids, vals, cand, cc, AV, AC, prod)
        if ph == _O_OPEN or ph == _O_ACOMMA:
            if b == 0x22:
                sids = info[0]
                lo, hi = info[2][p]
                for j in info[6][lo:hi]:
                    out.add(_Stk((_F_CSTR, sids[j], 0, b''),
                                 _Stk((_F_OBJ, n, _O_KEY, j, 0), rest)))
            elif b == 0x7D and ph == _O_OPEN and info[3][p]:
                self._pop_into(rest, out)
        elif ph == _O_AKEY:
            if b == 0x3A:
                out.add(_Stk((_F_OBJ, n, _O_ACOLON, p, 0), rest))
        elif ph == _O_ACOLON:
            par = None
            for f in self._start(info[1][p], b):
                if f is None:
                    out.add(_Stk((_F_OBJ, n, _O_AVAL, p + 1, 0), rest))
                else:
                    if par is None:
                        par = _Stk((_F_OBJ, n, _O_VAL, p, 0), rest)
                    out.add(_Stk(f, par))
        elif ph == _O_AVAL:
            if b == 0x2C:
                lo, hi = info[2][p]
                if lo < hi:
                    out.add(_Stk((_F_OBJ, n, _O_ACOMMA, p, 0), rest))
            elif b == 0x7D and info[3][p]:
                self._pop_into(rest, out)

    def _step_fobj(self, cfg, fr, b, out):
        _, n, ph, ws = fr
        rest = cfg.up
        if _IS_WS[b]:
            if ws < self.max_whitespace:
                out.add(_Stk((_F_FOBJ, n, ph, ws + 1), rest))
            return
        v = self._info[n][0]
        if ph == _O_OPEN or ph == _O_ACOMMA:
            if b == 0x22 and v >= 0:
                out.add(_Stk(_FREE_STR, _Stk((_F_FOBJ, n, _O_KEY, 0), rest)))
            elif b == 0x7D and ph == _O_OPEN:
                self._pop_into(rest, out)
        elif ph == _O_AKEY:
            if b == 0x3A:
                out.add(_Stk((_F_FOBJ, n, _O_ACOLON, 0), rest))
        elif ph == _O_ACOLON:
            par = None
            for f in self._start(v, b):
                if f is None:
                    out.add(_Stk((_F_FOBJ, n, _O_AVAL, 0), rest))
                else:
                    if par is None:
                        par = _Stk((_F_FOBJ, n, _O_VAL, 0), rest)
                    out.add(_Stk(f, par))
        elif ph == _O_AVAL:
            if b == 0x2C:
                out.add(_Stk((_F_FOBJ, n, _O_ACOMMA, 0), rest))
            elif b == 0x7D:
                self._pop_into(rest, out)

    def _step_arr(self, cfg, fr, b, out):
        _, n, ph, c, ws = fr
        rest = cfg.up
        if _IS_WS[b]:
            if ws < self.max_whitespace:
                out.add(_Stk((_F_ARR, n, ph, c, ws + 1), rest))
            return
        prefix, items, mn, effmax, cap = self._info[n]
        if ph == _A_AVAL:
            if b == 0x2C:
                if effmax is None or c < effmax:
                    out.add(_Stk((_F_ARR, n, _A_ACOMMA, c, 0), rest))
            elif b == 0x5D and c >= mn:
                self._pop_into(rest, out)
            return
        if ph == _A_OPEN and b == 0x5D:
            if mn == 0:
                self._pop_into(rest, out)
            return
        if effmax is not None and c >= effmax:
            return
        item = prefix[c] if c < len(prefix) else items
        par = None
        for f in self._start(item, b):
            if f is None:
                c1 = c + 1
                out.add(_Stk((_F_ARR, n, _A_AVAL, c1 if c1 < cap else cap, 0), rest))
            else:
                if par is None:
                    par = _Stk((_F_ARR, n, _A_VAL, c, 0), rest)
                out.add(_Stk(f, par))

    # ── Acceptation & complétion ─────────────────────────────────────────────

    @staticmethod
    def _cfg_accepting(cfg) -> bool:
        if cfg is None:
            return True
        fr = cfg.top
        return cfg.up is None and fr[0] == _F_NUM and _P_COMPLETE[fr[2]]

    def _cfg_completion(self, cfg):
        """Plus courte complétion d'une configuration : sommet puis chaque parent (à chaque
        embranchement, le reste à la fin la plus courte — cf. _tail_len)."""
        if cfg is None:
            return b''
        top = self._finish_top(cfg.top)
        if top is None or self._tail_len(cfg.up) < 0:
            return None
        parts = [top]
        memo = self._parent_memo
        s = cfg.up
        while s is not None:
            if type(s) is _Fork:
                s = s.best
                continue
            part = memo.get(s.top, _UNSET)
            if part is _UNSET:
                part = self._finish_parent(s.top)
            parts.append(part)
            s = s.up
        return b''.join(parts)

    def _tail_len(self, s):
        """
        Longueur de la fin des frames parents s, s.up, … (-1 = impasse). Mise en cache
        dans chaque nœud (queue partagée) → O(1) amorti, sans récursion. Un _Fork retient
        le reste à la fin la plus courte (`best`), à égalité la plus petite en octets
        (comparaison paresseuse, _tail_cmp) : la complétion ne dépend ni de l'ordre
        d'itération des ensembles ni de l'histoire des caches.
        """
        if s is None:
            return 0
        if s.tl != -2:
            return s.tl
        stack = [s]
        while stack:
            x = stack[-1]
            if x.tl != -2:
                stack.pop()
                continue
            if type(x) is _Stk:
                u = x.up
                if u is not None and u.tl == -2:
                    stack.append(u)
                    continue
                acc = 0 if u is None else u.tl
                if acc >= 0:
                    part = self._finish_parent(x.top)
                    acc = -1 if part is None else acc + len(part)
                x.tl = acc
            else:
                todo = [u for u in x.items if u is not None and u.tl == -2]
                if todo:
                    stack.extend(todo)
                    continue
                best, bl = None, -1
                for u in x.items:
                    n = 0 if u is None else u.tl
                    if n >= 0 and (bl < 0 or n < bl
                                   or (n == bl and self._tail_cmp(u, best) < 0)):
                        best, bl = u, n
                x.best = best
                x.tl = bl
            stack.pop()
        return s.tl

    def _tail_cmp(self, u, v) -> int:
        """Comparaison lexicographique des fins (calculées, même longueur) de u et v."""
        bu = bv = b''
        while True:
            if u is v and not bu and not bv:
                return 0
            while not bu and u is not None:
                if type(u) is _Fork:
                    u = u.best
                    continue
                bu = self._finish_parent(u.top)
                u = u.up
            while not bv and v is not None:
                if type(v) is _Fork:
                    v = v.best
                    continue
                bv = self._finish_parent(v.top)
                v = v.up
            if not bu or not bv:
                return (1 if bu else 0) - (1 if bv else 0)
            n = min(len(bu), len(bv))
            if bu[:n] != bv[:n]:
                return -1 if bu[:n] < bv[:n] else 1
            bu, bv = bu[n:], bv[n:]

    def _completion_len(self, state):
        """len(plus courte complétion analytique) sans la matérialiser ; None = impasse."""
        best = None
        for cfg in state:
            n = self._cfg_completion_len(cfg)
            if n is not None and (best is None or n < best):
                best = n
        return best

    def _cfg_completion_len(self, cfg):
        """len(_cfg_completion(cfg)) en O(1) amorti (fins des parents mises en cache)."""
        if cfg is None:
            return 0
        fr = cfg.top
        if fr[0] == _F_STR:
            _, mn, mx, cnt, sub, pend = fr
            if (mn > 0 or mx is not None) and pend and sub in _S_PAIRING:
                cnt -= 1
            top = len(_S_FINISH[sub]) + max(0, mn - cnt) + 1
        else:
            w = self._finish_top(fr)
            if w is None:
                return None
            top = len(w)
        tail = self._tail_len(cfg.up)
        return None if tail < 0 else top + tail

    @staticmethod
    def _frame_closable(fr, present) -> bool:
        """
        Condition NÉCESSAIRE pour terminer ce frame avec un vocabulaire dont `present[b]`
        dit si l'octet b figure dans au moins un token : ']' pour un tableau, '}' pour
        un objet, '"' pour une chaîne, la fin exacte d'un littéral, un chiffre pour un
        nombre inachevé.
        """
        k = fr[0]
        if k == _F_ARR:
            return bool(present[0x5D])
        if k == _F_OBJ or k == _F_FOBJ:
            return bool(present[0x7D])
        if k == _F_STR or k == _F_CSTR or k == _F_CSET:
            return bool(present[0x22])
        if k == _F_LIT:
            return all(present[b] for b in fr[1][fr[2]:])
        if k == _F_NUM:
            return _P_COMPLETE[fr[2]] or any(present[b] for b in b'0123456789')
        return True

    def _arr_after(self, n, c):
        """Plus courte fin d'un tableau après c éléments (AV) — ',' + item… + ']'."""
        prefix, items, mn, _, _ = self._info[n]
        W = self._W
        lp = len(prefix)
        out = b''.join(b',' + W[prefix[i]] for i in range(c, min(mn, lp)))
        k = mn - max(c, lp)
        if k > 0:
            out += (b',' + W[items]) * k
        return out + b']'

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
            return _S_FINISH[sub] + self._fill * max(0, mn - cnt) + b'"'
        if k == _F_CSTR:
            _, sid, i, part = fr
            _, encs, mins = self._cstr(sid)
            if i == len(mins):
                return b'"'
            cur = _minb(*(e[len(part):] for e in encs[i] if e.startswith(part)))
            return cur + b''.join(mins[i + 1:]) + b'"'
        if k == _F_CSET:
            return self._cset_finish(fr)
        if k == _F_NUM:
            return _P_FINISH[fr[2]]
        if k == _F_LIT:
            return fr[1][fr[2]:]
        if k == _F_OBJ:
            _, n, ph, p, _ = fr
            _, vals, _, cc, AV, AC, _ = self._info[n]
            if ph == _O_OPEN:
                return _minb(b'}' if cc[0] else None, AC[0])
            if ph == _O_AKEY:
                return _join(b':', W[vals[p]], AV[p + 1])
            if ph == _O_ACOLON:
                return _join(W[vals[p]], AV[p + 1])
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
        """Fin minimale d'un frame parent une fois son enfant (clé ou valeur) terminé
        (mémoïsée par frame : une pile profonde répète les mêmes frames parents)."""
        memo = self._parent_memo
        r = memo.get(fr, _UNSET)
        if r is _UNSET:
            r = self._finish_parent_raw(fr)
            size = 64 + (len(r) if r is not None else 0)
            if len(memo) >= self._max_states or self._parent_bytes + size > (1 << 22):
                memo.clear()        # plafond en nombre ET en octets (fins de minItems…)
                self._parent_bytes = 0
            memo[fr] = r
            self._parent_bytes += size
        return r

    def _finish_parent_raw(self, fr):
        k = fr[0]
        if k == _F_OBJ:
            _, n, ph, p, _ = fr
            _, vals, _, _, AV, _, _ = self._info[n]
            if ph == _O_KEY:
                return _join(b':', self._W[vals[p]], AV[p + 1])
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
    """
    Trie des octets des tokens, COMPACT (~9 octets par nœud au lieu d'un dict par nœud :
    o200k ≈ 4 Mo au lieu de ≈ 120 Mo, gpt2 ≈ 1 Mo au lieu de ≈ 28 Mo) et plus rapide à
    parcourir. Nœuds numérotés en largeur (BFS) ; les enfants d'un nœud sont contigus et
    triés par octet :
      first[x] … first[x+1]-1 : enfants de x                       (array 'i', N+1)
      lab[y]                  : octet de l'arête menant à y        (bytes, N)
      endtok[y]               : plus petit token se terminant en y, -1 sinon (array 'i')
      multi[y]                : autres tokens de mêmes octets, croissants (rare)
    Construit directement depuis les tokens triés (préfixe commun avec le précédent),
    sans trie intermédiaire en dicts (pic mémoire réduit d'autant).
    """

    def __init__(self, token_bytes):
        self.token_bytes = [tb if tb else None for tb in token_bytes]
        items = sorted((bytes(tb), tid) for tid, tb in enumerate(self.token_bytes)
                       if tb is not None)
        # 1) Nœuds en ordre préfixe (= ordre lexicographique des tokens triés)
        plab = bytearray(1)                 # octet d'entrée (racine : 0)
        pdepth = array('i', [0])
        nkids = array('i', [0])
        pend = array('i', [-1])
        pmulti = {}
        path = [0]                          # nœuds du token courant, par profondeur
        prev = b''
        for tb, tid in items:
            m = min(len(prev), len(tb))
            L = 0
            while L < m and prev[L] == tb[L]:
                L += 1
            del path[L + 1:]
            for d in range(L, len(tb)):
                idx = len(plab)
                plab.append(tb[d])
                pdepth.append(d + 1)
                nkids.append(0)
                pend.append(-1)
                nkids[path[-1]] += 1
                path.append(idx)
            node = path[-1]
            if pend[node] < 0:
                pend[node] = tid
            else:
                pmulti.setdefault(node, []).append(tid)
            prev = tb
        del items, path
        # 2) Renumérotation en largeur : tri par (profondeur, ordre préfixe) — les enfants
        #    d'un nœud deviennent contigus, dans l'ordre de leurs parents
        N = len(plab)
        start = [0] * (max(pdepth) + 2)
        for d in pdepth:
            start[d + 1] += 1
        for d in range(1, len(start)):
            start[d] += start[d - 1]
        pos = array('i', bytes(4 * N))
        for i in range(N):
            d = pdepth[i]
            pos[i] = start[d]
            start[d] += 1
        lab = bytearray(N)
        endtok = array('i', bytes(4 * N))
        nk = array('i', bytes(4 * N))
        for i in range(N):
            j = pos[i]
            lab[j] = plab[i]
            endtok[j] = pend[i]
            nk[j] = nkids[i]
        first = array('i', bytes(4 * (N + 1)))
        ptr = 1
        for x in range(N):
            first[x] = ptr
            ptr += nk[x]
        first[N] = ptr
        self.first = first
        self.lab = bytes(lab)
        present = bytearray(256)        # octets figurant dans au moins un token
        for b in set(self.lab[1:]):
            present[b] = 1
        self.present = bytes(present)
        self.endtok = endtok
        self.multi = {pos[i]: ids for i, ids in pmulti.items()}
        self._filler = None
        # Longueur maximale d'un token ; token de chaque octet seul (-1 : absent). Un
        # vocabulaire char-level (maxlen 1) se tokenise par simple correspondance.
        self.maxlen = max(pdepth)
        single = [-1] * 256
        for child in range(first[0], first[1]):
            dup = self.multi.get(child)
            single[self.lab[child]] = dup[-1] if dup else endtok[child]
        self.single = single

    def filler(self) -> str:
        """
        Caractère de remplissage (chaînes minLength) le moins coûteux en TOKENS : celui dont
        les répétitions se découpent (plus long token d'abord, comme _tokenize) en le moins
        de tokens — gpt2 : '-' * 64 est un seul token, alors que 'a' * 4 est le plus long
        token de 'a'. Critères : coût de 256 répétitions, puis coût cumulé des longueurs
        1..64, puis 'a' (comportement par défaut), puis le plus petit octet. Le caractère
        doit exister seul dans le vocabulaire. Calculé une fois par vocabulaire.
        """
        if self._filler is None:
            def greedy(runs, n):
                count = 0
                while n:
                    n -= runs[bisect_right(runs, n) - 1]
                    count += 1
                return count

            best, choice = None, 'a'
            for b in range(0x20, 0x7F):
                if b == 0x22 or b == 0x5C:
                    continue
                runs, node, k = [], 0, 0      # k tel que bytes([b]) * k est un token
                while True:
                    node = self.child(node, b)
                    if node < 0:
                        break
                    k += 1
                    if self.endtok[node] >= 0:
                        runs.append(k)
                if not runs or runs[0] != 1:
                    continue
                cost = (greedy(runs, 256), sum(greedy(runs, n) for n in range(1, 65)),
                        b != 0x61, b)
                if best is None or cost < best:
                    best, choice = cost, chr(b)
            self._filler = choice
        return self._filler

    def child(self, node: int, byte: int) -> int:
        """Enfant de `node` par l'octet `byte`, -1 s'il n'existe pas (dichotomie)."""
        lo, hi = self.first[node], self.first[node + 1]
        j = bisect_left(self.lab, byte, lo, hi)
        return j if j < hi and self.lab[j] == byte else -1

    def ends(self, node: int) -> list:
        """Tokens se terminant au nœud `node` (croissants)."""
        e = self.endtok[node]
        if e < 0:
            return []
        return [e] + self.multi.get(node, [])


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


# Masques compactés : 8 tokens par octet, bit de poids fort d'abord (le token 8·i + j
# est le bit 7 − j de l'octet i) → conversion en C par un entier en base 2.
_TO_BITCHARS = bytes.maketrans(b'\x00\x01', b'01')
_FROM_BITCHARS = bytes.maketrans(b'01', b'\x00\x01')


def _pack_bits(flags) -> bytes:
    """Octets 0/1 (longueur multiple de 8) → 1 bit par octet d'entrée."""
    return int(bytes(flags).translate(_TO_BITCHARS), 2).to_bytes(len(flags) // 8, 'big')


def _unpack_bits(raw: bytes, n: int) -> bytearray:
    """Inverse de _pack_bits, tronqué à n octets 0/1 (bytearray neuf, modifiable)."""
    bits = bytearray(format(int.from_bytes(raw, 'big'), '0%db' % (8 * len(raw)))
                     .encode('ascii').translate(_FROM_BITCHARS))
    del bits[n:]
    return bits


class _Shared:
    """Partagé par une contrainte et ses clones : trie + cache des masques par état."""

    def __init__(self, trie):
        self.trie = trie
        self.masks = {}         # état → bytes (1 bit par token, cf. _PACK)
        # ≤ 1024 masques et ≤ ~32 Mo (gpt2, o200k : 1024 ; vocabulaires géants : moins)
        nbytes = (len(trie.token_bytes) + 7) // 8
        self.max_masks = max(16, min(1024, (32 << 20) // max(1, nbytes)))


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
        Le cache stocke des `bytes` immuables COMPACTÉS (1 bit par token : gpt2 ≈ 6 Ko par
        masque au lieu de 50 Ko) ; chaque appel renvoie un tenseur neuf (dépliage en C via
        int/format/translate ≈ 0,1 ms, sans opération torch parallélisée ni pool de threads).
        """
        V = self.vocab_size
        if not V:
            return torch.zeros(0, dtype=torch.bool)
        sh = self._shared
        raw = sh.masks.get(self.state)
        if raw is None:
            buf = bytearray(V + (-V) % 8)
            for i in self._allowed_ids(self.state):
                buf[i] = 1
            raw = _pack_bits(buf)
            if len(sh.masks) >= sh.max_masks:
                sh.masks.clear()
            sh.masks[self.state] = raw
        return torch.frombuffer(_unpack_bits(raw, V), dtype=torch.bool)

    def _allowed_ids(self, state) -> list:
        """DFS (nœud du trie, état de l'automate) avec élagage sur octet refusé."""
        mt = self.matcher
        rows = mt._rows
        compute = mt._compute
        new_row = mt._new_row
        unset = _UNSET
        trie = self._shared.trie
        first, lab, endtok, multi = trie.first, trie.lab, trie.endtok, trie.multi
        ids = []
        append = ids.append
        stack = [(0, state)]
        pop, push = stack.pop, stack.append
        while stack:
            node, st = pop()
            row = rows.get(st)
            if row is None:
                row = new_row(st)
            for child in range(first[node], first[node + 1]):
                b = lab[child]
                ns = row.get(b, unset)
                if ns is unset:
                    ns = row[b] = compute(st, b)
                if ns is None:
                    continue
                e = endtok[child]
                if e >= 0:
                    append(e)
                    if multi and child in multi:
                        ids.extend(multi[child])
                if first[child] < first[child + 1]:
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
        Tokens fermant le JSON au plus court : d'abord une tokenisation de la plus courte
        complétion en octets (plus long préfixe d'abord, retour arrière si une impasse se
        présente) ; si le vocabulaire ne sait pas écrire ces octets précis (ex. '0' ou 'é'
        absents d'un vocabulaire char-level), recherche A* d'une AUTRE complétion
        écrivable avec ses tokens (la plus courte en octets). None si impossible avec ce
        vocabulaire (ou recherche épuisée). `max_expansions` borne les retours arrière et
        la recherche (pas la longueur de la complétion).
        Les chaînes minLength sont remplies avec `matcher.filler` : via json_constraint,
        le caractère le moins coûteux en tokens pour ce vocabulaire (gpt2 : une chaîne
        minLength 200 se ferme en ~10 tokens au lieu de ~56 avec 'a').
        """
        c = self.matcher.shortest_completion(self.state)
        if c is None:
            return None
        toks = self._tokenize(c, max_expansions)
        if toks is None:
            toks = self._search_completion(max_expansions)
        return toks

    def _tokenize(self, c: bytes, max_expansions: int):
        """
        Segmentation exacte de `c` en tokens (DFS, plus long préfixe d'abord, retour arrière
        sur impasse). `c` vient de shortest_completion, qui l'a VÉRIFIÉE octet par octet
        depuis l'état courant : toute segmentation produit la même suite d'octets, donc est
        acceptée → aucune marche de l'automate ici (coût linéaire, constante réduite).
        """
        trie = self._shared.trie
        if trie.maxlen == 1:            # char-level : segmentation unique, octet par octet
            single = trie.single
            toks = [single[b] for b in c]
            return None if -1 in toks else toks
        tbytes = trie.token_bytes
        first, lab, endtok, multi = trie.first, trie.lab, trie.endtok, trie.multi
        L = len(c)

        # Passe gloutonne sans allocation (cas courant : aucune impasse) — même résultat
        # que le DFS ci-dessous, qui ne revient en arrière QUE sur une impasse
        tokens, pos = [], 0
        while pos < L:
            node, best, blen = 0, -1, 0
            for i in range(pos, L):
                b = c[i]
                lo, hi = first[node], first[node + 1]
                node = bisect_left(lab, b, lo, hi)
                if node >= hi or lab[node] != b:
                    break
                if endtok[node] >= 0:
                    best, blen = node, i + 1 - pos
            if best < 0:
                break           # impasse → DFS avec retour arrière
            dup = multi.get(best) if multi else None
            tokens.append(dup[-1] if dup else endtok[best])
            pos += blen
        else:
            return tokens

        def candidates(pos):
            found, node = [], 0
            for i in range(pos, L):
                b = c[i]
                lo, hi = first[node], first[node + 1]
                node = bisect_left(lab, b, lo, hi)
                if node >= hi or lab[node] != b:
                    break
                e = endtok[node]
                if e >= 0:
                    found.append(e)
                    if multi and node in multi:
                        found.extend(multi[node])
            return found        # du plus court au plus long : pop() → plus long d'abord

        tokens = []
        stack = [(0, candidates(0))]
        failures = 0            # seuls les échecs comptent (pas les tokens posés)
        while stack:
            pos, cands = stack[-1]
            if pos == L:
                return tokens
            if not cands:
                stack.pop()
                if tokens:
                    tokens.pop()
                failures += 1
                if failures > max_expansions:
                    return None
                continue
            tid = cands.pop()
            tokens.append(tid)
            npos = pos + len(tbytes[tid])
            stack.append((npos, candidates(npos)))
        return None

    def _search_completion(self, max_expansions: int):
        """
        A* sur le produit (état de l'automate, nœud du trie) : coût = octets écrits,
        heuristique = longueur de la plus courte complétion en octets (minorant exact,
        donc admissible) → la plus courte complétion ÉCRIVABLE avec le vocabulaire.
        Nœud du trie 0 = frontière de token ; but = état acceptant à une frontière.
        Élagage : une configuration dont un frame ne peut plus être fermé avec les octets
        du vocabulaire (ex. ']' absent) est abandonnée → impasse détectée sans recherche.
        """
        mt = self.matcher
        trie = self._shared.trie
        first, lab, endtok, present = trie.first, trie.lab, trie.endtok, trie.present
        closable = {}                   # nœud de pile → fermable (mémo, piles partagées)

        def cfg_ok(cfg):
            # Fermable si au moins une des piles représentées l'est (embranchement : un
            # reste fermable suffit). Parcours postfixe itératif, mémo par nœud partagé.
            stack = [cfg]
            while stack:
                x = stack[-1]
                if x in closable:
                    stack.pop()
                    continue
                if type(x) is _Stk:
                    u = x.up
                    if u is not None and u not in closable:
                        stack.append(u)
                        continue
                    closable[x] = ((u is None or closable[u])
                                   and mt._frame_closable(x.top, present))
                else:
                    todo = [u for u in x.items if u is not None and u not in closable]
                    if todo:
                        stack.extend(todo)
                        continue
                    closable[x] = any(u is None or closable[u] for u in x.items)
                stack.pop()
            return closable[cfg]

        def h(state):
            best = None
            for cfg in state:
                if cfg is not None and not cfg_ok(cfg):
                    continue
                n = mt._cfg_completion_len(cfg)
                if n is not None and (best is None or n < best):
                    best = n
            return best

        h0 = h(self.state)
        if h0 is None:
            return None
        budget = max_expansions // 4 + 4 * (h0 + 1)
        start = (self.state, 0)
        best_g = {start: 0}
        parent = {start: None}          # clé → (clé précédente, token fermé ou -1)
        heap = [(h0, 0, 0, start)]       # (f, -g, ordre, clé) : à f égal, le plus avancé
        order = 0
        pops = 0
        hcache = {}
        while heap:
            f, ng, _, key = heapq.heappop(heap)
            g = -ng
            if best_g.get(key, g) < g:
                continue                 # entrée périmée
            st, node = key
            if node == 0 and mt.is_accepting(st):
                tokens = []
                while parent[key] is not None:
                    key, tid = parent[key]
                    if tid >= 0:
                        tokens.append(tid)
                tokens.reverse()
                return tokens
            pops += 1
            if pops > budget:
                return None
            g2 = g + 1
            for child in range(first[node], first[node + 1]):
                ns = mt.advance(st, lab[child])
                if ns is None:
                    continue
                hn = hcache.get(ns, _UNSET)
                if hn is _UNSET:
                    hn = hcache[ns] = h(ns)
                if hn is None:
                    continue
                succ = []
                if endtok[child] >= 0:
                    succ.append(((ns, 0), endtok[child]))
                if first[child] < first[child + 1]:
                    succ.append(((ns, child), -1))
                for k2, tid in succ:
                    if g2 < best_g.get(k2, g2 + 1):
                        best_g[k2] = g2
                        parent[k2] = (key, tid)
                        order += 1
                        heapq.heappush(heap, (g2 + hn, -g2, order, k2))
        return None


def json_constraint(token_bytes: list, schema=None, *, max_whitespace: int = 4) -> TokenConstraint:
    """
    Contrainte prête à l'emploi : JSON valide (schema=None) ou conforme au schéma.
    Le remplissage des chaînes minLength est le caractère le moins coûteux en tokens pour
    ce vocabulaire (_VocabTrie.filler) : completion_tokens() reste court en TOKENS, l'unité
    des budgets de generate_structured / `popixa gen --tokens`.
    """
    filler = _get_trie(token_bytes).filler()        # trie mis en cache, réutilisé ci-dessous
    return TokenConstraint(JSONSchemaMatcher(schema, max_whitespace=max_whitespace,
                                             filler=filler), token_bytes)


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
    try:
        with open(args.document, 'rb') as f:
            data = f.read()
    except OSError as e:
        print(f"✗ document illisible : {e}")
        return 2
    st = m.initial_state
    for i, b in enumerate(data):
        st = m.advance(st, b)
        if st is None:
            print(f"✗ octet refusé à la position {i} : {data[max(0, i - 20):i + 1]!r}")
            return 1
    if not m.is_accepting(st):
        print("✗ document incomplet")
        return 1
    # Contre-vérification indépendante (json.loads + validate_instance). L'automate accepte
    # une imbrication illimitée, json.loads (récursif) non : au-delà, verdict de l'automate.
    try:
        errs = validate_instance(json.loads(data.decode('utf-8')), schema)
    except RecursionError:
        print("✓ document accepté par l'automate (trop imbriqué pour json.loads : "
              "contre-vérification non effectuée)")
        return 0
    except SchemaError as e:
        print(f"✗ schéma : {e}")
        return 2
    except ValueError as e:     # JSONDecodeError, UnicodeDecodeError, limite int ↔ str
        print(f"✗ json.loads refuse le document accepté par l'automate : {e}")
        return 1
    print("✓ document accepté" if not errs else "✗ " + "; ".join(errs))
    return 0 if not errs else 1


if __name__ == '__main__':
    raise SystemExit(_main())
