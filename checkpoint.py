"""
nanoPOPIXA — Format des checkpoints (version 2.3 « Socle »)

Format 2 : un dict de types simples uniquement (tenseurs, nombres, chaînes, listes, dicts),
chargé avec torch.load(weights_only=True) → ouvrir un checkpoint téléchargé ne peut pas
exécuter de code (un .pt « classique » est un pickle : il peut lancer n'importe quoi).

    {"format": 2, "model": state_dict, "config": {champs de POPIXAConfig}, "iter": int,
     "tokenizer": "char" | "tiktoken_gpt2", "vocab": {"stoi", "itos"} (caractère),
     "optimizer": état AdamW (reprise), "train": réglages du run, "data_dir": str, ...}

Anciens checkpoints (2.0 – 2.2) : la config est un objet POPIXAConfig picklé. On les charge
en n'autorisant QUE cette classe (torch.serialization.safe_globals, PyTorch ≥ 2.4) ; avec un
PyTorch plus ancien, chargement complet signalé — à ne faire que sur ses propres fichiers.
La prochaine sauvegarde (popixa train --resume) les réécrit au format 2.
"""

import os
import pickle
import dataclasses

import torch

from model import POPIXAConfig

FORMAT = 2


def config_to_dict(config) -> dict:
    """POPIXAConfig (ou dict) → dict de types simples."""
    if config is None or isinstance(config, dict):
        return config
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    return dict(vars(config))


def config_from(obj):
    """dict (format 2) ou POPIXAConfig (ancien format) → POPIXAConfig. Les champs inconnus
    (checkpoint d'une version plus récente) sont ignorés, les champs absents prennent leur
    valeur par défaut (checkpoint plus ancien)."""
    if obj is None or isinstance(obj, POPIXAConfig):
        return obj
    if not isinstance(obj, dict):
        obj = vars(obj)
    names = {f.name for f in dataclasses.fields(POPIXAConfig)}
    return POPIXAConfig(**{k: v for k, v in obj.items() if k in names})


def save_checkpoint(path: str, ckpt: dict) -> None:
    """Écrit `ckpt` au format 2, de façon atomique (un Ctrl+C pendant l'écriture ne corrompt
    pas le checkpoint précédent)."""
    payload = dict(ckpt)
    payload["format"] = FORMAT
    if "config" in payload:
        payload["config"] = config_to_dict(payload["config"])
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _load_legacy(path: str, map_location):
    """Ancien format : seul POPIXAConfig est autorisé en plus des types simples."""
    ser = torch.serialization
    if hasattr(ser, "safe_globals"):                          # PyTorch ≥ 2.5
        with ser.safe_globals([POPIXAConfig]):
            return torch.load(path, map_location=map_location, weights_only=True), True
    if hasattr(ser, "add_safe_globals"):                      # PyTorch 2.4
        ser.add_safe_globals([POPIXAConfig])
        return torch.load(path, map_location=map_location, weights_only=True), True
    return torch.load(path, map_location=map_location, weights_only=False), False


def load_checkpoint(path: str, map_location="cpu", log=print) -> dict:
    """
    Charge un checkpoint en mode sûr ; « config » est rendue sous forme de POPIXAConfig.
    FileNotFoundError si absent ; ValueError (message clair) si illisible ou non autorisé.
    """
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except FileNotFoundError:
        raise
    except (pickle.UnpicklingError, RuntimeError) as first:
        # Ancien format (POPIXAConfig picklé) — ou fichier corrompu : on retente prudemment
        try:
            ckpt, safe = _load_legacy(path, map_location)
        except Exception as second:
            if "global" in str(second).lower():          # classe / fonction non autorisée
                raise ValueError(f"checkpoint {path} refusé par le chargement sûr : il contient des "
                                 "objets autres que des tenseurs et des types simples (fichier d'un "
                                 "autre programme, ou piégé)") from None
            raise ValueError(f"checkpoint {path} illisible : fichier corrompu ou qui n'est pas "
                             "un checkpoint PyTorch") from None
        if not safe:
            log(f"⚠️  {path} : ancien format chargé sans protection (PyTorch < 2.4) — "
                "à ne faire que sur tes propres fichiers")
    except EOFError:
        raise ValueError(f"checkpoint {path} vide ou tronqué") from None
    if not isinstance(ckpt, dict):
        raise ValueError(f"checkpoint {path} : contenu inattendu ({type(ckpt).__name__})")
    if ckpt.get("config") is not None:
        ckpt["config"] = config_from(ckpt["config"])
    return ckpt
