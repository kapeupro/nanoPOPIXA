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


# ── Helpers ──────────────────────────────────────────────────────────────────

def _checkpoint_fingerprint(checkpoint_path: str) -> str:
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


# ── Save ─────────────────────────────────────────────────────────────────────

def save_session(
    cache_path: str,
    past_kvs: list,
    token_ids: list,
    checkpoint_path: str,
) -> None:
    """
    Sérialise le KV-cache et les token IDs sur disque.

    past_kvs   : liste de (K, V) tenseurs — un par couche Transformer
    token_ids  : liste d'entiers — tous les tokens traités depuis le début
    checkpoint_path : chemin du checkpoint pour fingerprinting
    """
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    payload = {
        "version":    2,
        "ckpt_fp":    _checkpoint_fingerprint(checkpoint_path),
        "token_ids":  token_ids,
        # Déplacer sur CPU avant de sauvegarder (portable MPS → CPU → CUDA)
        "past_kvs":   [(k.cpu(), v.cpu()) for k, v in past_kvs],
    }
    torch.save(payload, cache_path)


# ── Load ─────────────────────────────────────────────────────────────────────

def load_session(
    cache_path: str,
    checkpoint_path: str,
    device: str,
) -> tuple:
    """
    Restaure le KV-cache depuis le disque.

    Retourne (past_kvs, token_ids) si valide,
    sinon     (None, [])           si cache absent ou invalide.

    Invalidations :
      - Fichier absent
      - Version incompatible
      - Fingerprint du checkpoint différent (modèle changé)
      - Erreur de lecture
    """
    if not os.path.exists(cache_path):
        return None, []

    # Si le checkpoint lui-même n'existe pas, invalider immédiatement
    if not os.path.exists(checkpoint_path):
        return None, []

    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    except Exception:
        return None, []

    # Vérification version
    if payload.get("version") != 2:
        return None, []

    # Vérification modèle (si le checkpoint a changé, le cache est invalide)
    if payload.get("ckpt_fp") != _checkpoint_fingerprint(checkpoint_path):
        return None, []

    past_kvs  = [(k.to(device), v.to(device)) for k, v in payload["past_kvs"]]
    token_ids = payload["token_ids"]
    return past_kvs, token_ids


# ── Clear ─────────────────────────────────────────────────────────────────────

def clear_session(cache_path: str) -> bool:
    """Supprime le cache de session. Retourne True si supprimé."""
    if os.path.exists(cache_path):
        os.remove(cache_path)
        return True
    return False


# ── Info ──────────────────────────────────────────────────────────────────────

def session_info(cache_path: str, checkpoint_path: str) -> dict:
    """Retourne des métadonnées sur le cache (taille, nb tokens, validité)."""
    if not os.path.exists(cache_path):
        return {"exists": False}

    size_kb = os.path.getsize(cache_path) / 1024
    mtime   = os.path.getmtime(cache_path)

    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        valid   = (
            payload.get("version") == 2
            and payload.get("ckpt_fp") == _checkpoint_fingerprint(checkpoint_path)
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
