"""
nanoPOPIXA — KV-Cache persistant entre sessions
Inspiré du prompt caching de Claude Code (prompt-caching-scope, TTL 5min/1h)

Principe : au lieu de retraiter tout l'historique à chaque démarrage,
on sérialise les tenseurs K/V sur disque et on les restaure directement.

Avantage : démarrage instantané même après un redémarrage — le modèle
"se souvient" sans avoir relu un seul token.
"""

import os
import hashlib
import torch

# Version du format sur disque :
#   2 — token_ids approximatifs (ré-encodés depuis le texte)
#   3 — token_ids exacts couverts par le KV-cache + historique texte (sans thinking)
SESSION_VERSION   = 3
_READABLE_VERSIONS = (2, 3)


# ── Helpers ──────────────────────────────────────────────────────────────────

def checkpoint_fingerprint(checkpoint_path: str) -> str:
    """
    Empreinte du checkpoint pour invalider le cache si le modèle change.
    Utilise taille + mtime (rapide, pas de lecture du fichier).
    Inspiré de la logique de cache-busting de Claude (prompt cache TTL).
    """
    try:
        s = os.stat(checkpoint_path)
        raw = f"{os.path.abspath(checkpoint_path)}:{s.st_size}:{s.st_mtime}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]
    except OSError:
        return "unknown"


_checkpoint_fingerprint = checkpoint_fingerprint   # alias historique


# ── Save ─────────────────────────────────────────────────────────────────────

def save_session(
    cache_path: str,
    past_kvs: list,
    token_ids: list,
    checkpoint_path: str,
    history: str = None,
    fingerprint: str = None,
) -> None:
    """
    Sérialise le KV-cache et les token IDs sur disque.

    past_kvs   : liste de (K, V) tenseurs — un par couche Transformer
    token_ids  : liste d'entiers — les tokens couverts par le cache, dans l'ordre
    checkpoint_path : chemin du checkpoint pour fingerprinting
    history    : historique texte de la conversation (sans thinking) — optionnel
    fingerprint : empreinte du checkpoint calculée AU CHARGEMENT du modèle (recommandé) —
                  sinon le fichier est re-stat-é maintenant : si un entraînement l'a
                  réécrit entre-temps, un cache calculé avec les anciens poids serait
                  estampillé avec la nouvelle empreinte.

    Écriture atomique (fichier temporaire + os.replace) : un crash pendant la
    sauvegarde ne laisse jamais un cache corrompu.
    """
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    payload = {
        "version":    SESSION_VERSION,
        "ckpt_fp":    fingerprint or checkpoint_fingerprint(checkpoint_path),
        "token_ids":  list(token_ids),
        # Déplacer sur CPU avant de sauvegarder (portable MPS → CPU → CUDA) ;
        # clone() compacte les vues tronquées (sinon tout le stockage serait écrit)
        "past_kvs":   [(k.detach().cpu().clone(), v.detach().cpu().clone()) for k, v in past_kvs],
        "history":    history,
    }
    tmp_path = cache_path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)


# ── Load ─────────────────────────────────────────────────────────────────────

def load_session(
    cache_path: str,
    checkpoint_path: str,
    device: str,
    with_history: bool = False,
    fingerprint: str = None,
) -> tuple:
    """
    Restaure le KV-cache depuis le disque.

    Retourne (past_kvs, token_ids) si valide,
    sinon     (None, [])           si cache absent ou invalide.
    with_history=True → (past_kvs, token_ids, history) ; history vaut None si absent.
    fingerprint       : empreinte du checkpoint réellement chargé (sinon celle du fichier).

    past_kvs est un model.KVCache (liste de (K, V)) dont `token_ids` vaut les ids
    du cache s'ils sont cohérents avec sa longueur (format v3), sinon None :
    l'appelant doit alors reconstruire le contexte plutôt que réutiliser le cache.

    Invalidations :
      - Fichier absent
      - Version incompatible
      - Fingerprint du checkpoint différent (modèle changé)
      - Erreur de lecture ou contenu malformé
    """
    invalid = (None, [], None) if with_history else (None, [])

    if not os.path.exists(cache_path):
        return invalid

    # Si le checkpoint lui-même n'existe pas, invalider immédiatement
    if not os.path.exists(checkpoint_path):
        return invalid

    try:
        # weights_only=True : le cache ne contient que tenseurs / ints / str → aucun
        # code arbitraire exécuté si un fichier piégé traîne dans le dossier courant
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            return invalid

        # Vérification version
        if payload.get("version") not in _READABLE_VERSIONS:
            return invalid

        # Vérification modèle (si le checkpoint a changé, le cache est invalide)
        if payload.get("ckpt_fp") != (fingerprint or checkpoint_fingerprint(checkpoint_path)):
            return invalid

        from model import KVCache
        kvs       = [(k.to(device), v.to(device)) for k, v in payload["past_kvs"]]
        token_ids = list(payload["token_ids"])
    except Exception:
        return invalid

    seq_len  = kvs[0][0].size(2) if kvs else 0
    exact    = payload.get("version") == SESSION_VERSION and seq_len == len(token_ids)
    past_kvs = KVCache(kvs, list(token_ids) if exact else None)
    if with_history:
        return past_kvs, token_ids, payload.get("history")
    return past_kvs, token_ids


# ── Clear ─────────────────────────────────────────────────────────────────────

def clear_session(cache_path: str) -> bool:
    """Supprime le cache de session. Retourne True si supprimé (jamais d'exception)."""
    try:
        os.remove(cache_path)
        return True
    except OSError:
        return False


# ── Info ──────────────────────────────────────────────────────────────────────

def session_info(cache_path: str, checkpoint_path: str, fingerprint: str = None) -> dict:
    """Retourne des métadonnées sur le cache (taille, nb tokens, validité)."""
    if not os.path.exists(cache_path):
        return {"exists": False}

    size_kb = os.path.getsize(cache_path) / 1024
    mtime   = os.path.getmtime(cache_path)

    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        valid   = (
            payload.get("version") in _READABLE_VERSIONS
            and payload.get("ckpt_fp") == (fingerprint or checkpoint_fingerprint(checkpoint_path))
        )
        n_tokens = len(payload.get("token_ids", []))
    except Exception:
        valid    = False
        n_tokens = 0

    return {
        "exists":   True,
        "valid":    valid,
        "size_kb":  size_kb,
        "n_tokens": n_tokens,
        "mtime":    mtime,
    }
