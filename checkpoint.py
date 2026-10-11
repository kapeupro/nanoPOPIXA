"""
nanoPOPIXA — Format des checkpoints (version 2.3 « Socle »)

Format 2 : un dict de types simples uniquement (tenseurs, nombres, chaînes, listes, dicts),
chargé avec torch.load(weights_only=True) → ouvrir un checkpoint téléchargé ne peut pas
exécuter de code (un .pt « classique » est un pickle : il peut lancer n'importe quoi).

    {"format": 2, "model": state_dict, "config": {champs de POPIXAConfig}, "iter": int,
     "tokenizer": "char" | "tiktoken_gpt2", "vocab": {"stoi", "itos"} (caractère),
     "optimizer": état AdamW (reprise), "train": réglages du run, "data_dir": str, ...}

Anciens checkpoints (2.0 – 2.2) : la config est un objet POPIXAConfig picklé. On les charge
en n'autorisant QUE cette classe (torch.serialization.safe_globals, PyTorch ≥ 2.4).
Avec un PyTorch plus ancien (< 2.4), ils ne sont chargés qu'avec l'accord explicite
POPIXA_TRUST_CHECKPOINTS=1 (chargement complet : à ne faire que sur ses propres fichiers).
La prochaine sauvegarde (popixa train --resume) les réécrit au format 2.
"""

import os
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


TRUST_ENV = "POPIXA_TRUST_CHECKPOINTS"     # =1 : autorise le chargement NON sûr d'un ancien checkpoint


class _NeedsTrust(Exception):
    """Ancien format illisible en mode sûr avec ce PyTorch (< 2.4)."""


def _load_legacy(path: str, map_location):
    """Ancien format : seule POPIXAConfig est autorisée en plus des types simples. Retourne
    (checkpoint, chargé_en_mode_sûr)."""
    ser = torch.serialization
    if hasattr(ser, "safe_globals"):                          # PyTorch ≥ 2.5
        with ser.safe_globals([POPIXAConfig]):
            return torch.load(path, map_location=map_location, weights_only=True), True
    if hasattr(ser, "add_safe_globals"):                      # PyTorch 2.4 (liste globale)
        ser.add_safe_globals([POPIXAConfig])
        return torch.load(path, map_location=map_location, weights_only=True), True
    # PyTorch < 2.4 : impossible d'autoriser POPIXAConfig seule → jamais de repli silencieux
    # sur un chargement complet (un fichier piégé exécuterait son code) : il faut l'accord explicite
    if os.environ.get(TRUST_ENV) == "1":
        return torch.load(path, map_location=map_location, weights_only=False), False
    raise _NeedsTrust()


def _refused(path: str, err: Exception) -> ValueError:
    if "global" in str(err).lower() or "class" in str(err).lower():   # objet non autorisé
        return ValueError(f"checkpoint {path} refusé par le chargement sûr : il contient des objets "
                          "autres que des tenseurs et des types simples (fichier d'un autre "
                          "programme, ou piégé)")
    return ValueError(f"checkpoint {path} illisible : fichier corrompu ou qui n'est pas un "
                      "checkpoint PyTorch")


def load_checkpoint(path: str, map_location="cpu", log=print) -> dict:
    """
    Charge un checkpoint en mode sûr ; « config » est rendue sous forme de POPIXAConfig.
    FileNotFoundError si absent ; ValueError (message clair) si illisible ou non autorisé.
    """
    if os.path.isdir(path):
        raise ValueError(f"{path} est un dossier, pas un checkpoint")
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except FileNotFoundError:
        raise
    except EOFError:
        raise ValueError(f"checkpoint {path} vide ou tronqué") from None
    except PermissionError as e:
        raise ValueError(f"checkpoint {path} illisible : {e.strerror or e}") from None
    except OSError:                                   # EINVAL… : archive tronquée ou abîmée
        raise ValueError(f"checkpoint {path} illisible : fichier corrompu ou tronqué") from None
    except Exception as first:
        # Ancien format (POPIXAConfig picklée) — ou fichier corrompu / piégé
        if "POPIXAConfig" not in str(first):
            raise _refused(path, first) from None
        try:
            ckpt, safe = _load_legacy(path, map_location)
        except _NeedsTrust:
            raise ValueError(
                f"checkpoint {path} : ancien format (≤ 2.2), lisible en mode sûr avec PyTorch ≥ 2.4 "
                f"seulement — mets PyTorch à jour ou, si ce fichier vient de toi, relance avec "
                f"{TRUST_ENV}=1 (il sera réécrit au format 2 à la prochaine reprise)") from None
        except EOFError:
            raise ValueError(f"checkpoint {path} vide ou tronqué") from None
        except Exception as second:
            raise _refused(path, second) from None
        if not safe:
            log(f"⚠️  {path} : ancien format chargé SANS protection ({TRUST_ENV}=1, PyTorch < 2.4)")
    if not isinstance(ckpt, dict):
        raise ValueError(f"checkpoint {path} : contenu inattendu ({type(ckpt).__name__})")
    if isinstance(ckpt.get("format"), int) and ckpt["format"] > FORMAT:
        log(f"⚠️  {path} : format {ckpt['format']} (version plus récente de nanoPOPIXA) — "
            "les champs inconnus sont ignorés")
    if ckpt.get("config") is not None:
        ckpt["config"] = config_from(ckpt["config"])
    return ckpt
