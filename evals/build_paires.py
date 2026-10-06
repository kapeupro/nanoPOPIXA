"""
Construit evals/fr_paires.jsonl à partir des générateurs de evals/paires/*.py.

Chaque module expose PHENOMENE et generate() -> [{"good", "bad", "phenomene", "gabarit"}].
Le fichier produit est déterministe (même contenu à chaque exécution) et validé :
schéma, unicité, phrases distinctes, ponctuation, apostrophes droites.

Usage : python evals/build_paires.py [--check]
  --check : vérifie que evals/fr_paires.jsonl est à jour (utilisé par les tests / la CI)
"""

import os
import sys
import json
import importlib.util

HERE      = os.path.dirname(os.path.abspath(__file__))
PAIRS_DIR = os.path.join(HERE, "paires")
OUT_PATH  = os.path.join(HERE, "fr_paires.jsonl")

# Ordre d'apparition dans le fichier (et dans les rapports)
PHENOMENES = ("accord_sujet_verbe", "accord_nominal", "participe_passe", "elision", "prepositions")


def _load(name: str):
    path = os.path.join(PAIRS_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"paires_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate(pairs: list) -> list:
    """Retourne la liste des problèmes (vide = valide)."""
    errors = []
    seen_good, seen_bad = set(), set()
    for i, p in enumerate(pairs):
        where = f"paire {i} ({p.get('phenomene')}/{p.get('gabarit')})"
        if set(p) != {"good", "bad", "phenomene", "gabarit"}:
            errors.append(f"{where} : clés {sorted(p)}")
            continue
        g, b = p["good"], p["bad"]
        if g == b:
            errors.append(f"{where} : phrases identiques")
        for label, s in (("good", g), ("bad", b)):
            if not s or s != s.strip() or not s[0].isupper() or s[-1] not in ".?!":
                errors.append(f"{where} : {label} mal formée : {s!r}")
            if "’" in s or "  " in s:
                errors.append(f"{where} : {label} apostrophe typographique ou double espace : {s!r}")
        if g in seen_good or b in seen_bad:
            errors.append(f"{where} : doublon")
        seen_good.add(g)
        seen_bad.add(b)
    return errors


def build() -> list:
    pairs = []
    for name in PHENOMENES:
        mod = _load(name)
        assert mod.PHENOMENE == name, f"{name}.py : PHENOMENE = {mod.PHENOMENE!r}"
        for p in mod.generate():
            pairs.append({"good": p["good"], "bad": p["bad"], "phenomene": p["phenomene"],
                          "gabarit": p["gabarit"]})
    return pairs


def render(pairs: list) -> str:
    return "".join(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n" for p in pairs)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    pairs = build()
    errors = validate(pairs)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    text = render(pairs)
    if "--check" in argv:
        if not os.path.exists(OUT_PATH):
            print("evals/fr_paires.jsonl absent : python evals/build_paires.py", file=sys.stderr)
            return 1
        with open(OUT_PATH, encoding="utf-8") as f:
            if f.read() != text:
                print("evals/fr_paires.jsonl n'est pas à jour : python evals/build_paires.py",
                      file=sys.stderr)
                return 1
        return 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    counts = {}
    for p in pairs:
        counts[p["phenomene"]] = counts.get(p["phenomene"], 0) + 1
    print(f"{len(pairs)} paires → {OUT_PATH}")
    for k in PHENOMENES:
        print(f"  {k:20s} {counts.get(k, 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
