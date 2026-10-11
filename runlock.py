"""
nanoPOPIXA — Verrou d'un dossier d'expérience (2.3 « Socle »)

Un seul entraînement à la fois par dossier (out-nanopopixa/, runs/NOM/) : train.py tient un verrou
exclusif sur DOSSIER/.train.lock pendant toute sa vie (libéré par le système à la fin du processus,
même tué brutalement). popixa monitor s'en sert pour savoir si le run qu'il suit tourne encore.
"""

import os
import time
import errno

LOCK_NAME = ".train.lock"

# Verrou déjà tenu par un autre processus (selon le système)
_BUSY = {errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK, errno.EDEADLK}

if os.name == "nt":
    import msvcrt

    def _lock(f, shared: bool) -> None:          # Windows : pas de verrou partagé, exclusif court
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(f) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock(f, shared: bool) -> None:
        fcntl.flock(f.fileno(), (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)

    def _unlock(f) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def acquire(out_dir: str, wait: float = 1.0):
    """
    Verrou exclusif sur out_dir (qui doit exister). Retourne le fichier ouvert, à garder jusqu'à
    la fin du processus, ou None si un autre entraînement le tient. Le court délai `wait` couvre
    le test de popixa monitor (is_active), qui prend le verrou une fraction de milliseconde.
    Système de fichiers sans verrous (certains montages réseau) : pas de protection plutôt qu'un refus.
    """
    f = open(os.path.join(out_dir, LOCK_NAME), "a+")
    deadline = time.monotonic() + wait
    while True:
        try:
            _lock(f, shared=False)
            return f
        except OSError as e:
            if e.errno not in _BUSY:
                return f
            if time.monotonic() >= deadline:
                f.close()
                return None
            time.sleep(0.05)


def is_active(out_dir: str):
    """True : un entraînement tient le verrou de out_dir ; False : verrou libre (run terminé ou
    tué) ; None : on ne sait pas (pas de fichier de verrou — run < 2.3 —, verrous non gérés)."""
    try:
        f = open(os.path.join(out_dir, LOCK_NAME), "r")
    except OSError:
        return None
    with f:
        try:
            _lock(f, shared=True)
        except OSError as e:
            return True if e.errno in _BUSY else None
        _unlock(f)
        return False
