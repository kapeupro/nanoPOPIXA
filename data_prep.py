"""
nanoPOPIXA — Préparation des données
Supporte tiktoken BPE (GPT-2, ~50k tokens) et tokenisation caractère.
Usage :
    python data_prep.py --dataset shakespeare
    python data_prep.py --dataset mon_fichier.txt
    python data_prep.py --dataset javascript
"""

import os
import pickle
import argparse
import urllib.request

import numpy as np


DATASETS = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "moliere":     "https://www.gutenberg.org/cache/epub/4106/pg4106.txt",
    "hugo":        "https://www.gutenberg.org/cache/epub/1952/pg1952.txt",
    "bible":       "https://raw.githubusercontent.com/mxw/grmr/master/src/finley/data/bible.txt",
    "linux":       "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/linux/input.txt",
    "javascript": [
        "https://unpkg.com/lodash@4.17.21/lodash.js",
        "https://unpkg.com/jquery@3.7.1/dist/jquery.js",
        "https://unpkg.com/vue@3/dist/vue.global.js",
        "https://unpkg.com/react@18/umd/react.development.js",
        "https://unpkg.com/react-dom@18/umd/react-dom.development.js",
    ],
}


# ─── Téléchargement ───────────────────────────────────────────────────────────
def _download(url: str, filepath: str) -> None:
    print(f"  Téléchargement : {url}")
    urllib.request.urlretrieve(url, filepath)
    size_mb = os.path.getsize(filepath) / 1e6
    print(f"  -> {filepath} ({size_mb:.1f} Mo)")


def _download_multi(urls: list, filepath: str) -> None:
    import tempfile
    parts = []
    for url in urls:
        name = url.rstrip("/").split("/")[-1]
        print(f"  Téléchargement : {url}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".js") as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            with open(tmp.name, "r", encoding="utf-8", errors="replace") as f:
                parts.append(f"// -- {name} --\n" + f.read())
            os.unlink(tmp.name)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts))
    size_mb = os.path.getsize(filepath) / 1e6
    print(f"  -> {filepath} ({size_mb:.1f} Mo, {len(urls)} fichiers)")


# ─── Tokenisation ─────────────────────────────────────────────────────────────
def _tokenize_and_save(text: str, data_dir: str, use_tiktoken: bool = True) -> None:
    os.makedirs(data_dir, exist_ok=True)

    if use_tiktoken:
        try:
            import tiktoken
        except ImportError:
            raise ImportError("tiktoken requis pour BPE : pip install tiktoken")
        enc    = tiktoken.get_encoding("gpt2")
        tokens = np.array(enc.encode_ordinary(text), dtype=np.uint16)
        n      = len(tokens)
        tokens[:int(n * 0.9)].tofile(os.path.join(data_dir, "train.bin"))
        tokens[int(n * 0.9):].tofile(os.path.join(data_dir, "val.bin"))
        meta = {"vocab_size": enc.n_vocab, "tokenizer": "tiktoken_gpt2"}
        print(f"  Tokenizer : BPE tiktoken-gpt2 | {n:,} tokens | vocab {enc.n_vocab}")
    else:
        chars      = sorted(set(text))
        stoi       = {c: i for i, c in enumerate(chars)}
        itos       = {i: c for i, c in enumerate(chars)}
        data       = np.array([stoi[c] for c in text], dtype=np.uint16)
        n          = len(data)
        data[:int(n * 0.9)].tofile(os.path.join(data_dir, "train.bin"))
        data[int(n * 0.9):].tofile(os.path.join(data_dir, "val.bin"))
        meta = {"vocab_size": len(chars), "tokenizer": "char",
                "stoi": stoi, "itos": itos}
        print(f"  Tokenizer : caractère | {n:,} chars | vocab {len(chars)}")

    with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"  -> train.bin + val.bin + meta.pkl dans {data_dir}/")


# ─── API publique ─────────────────────────────────────────────────────────────
def prepare(dataset: str = "shakespeare", data_dir: str = "data",
            use_tiktoken: bool = True) -> None:
    """Télécharge et tokenise un dataset."""
    os.makedirs(data_dir, exist_ok=True)
    raw_path = os.path.join(data_dir, "raw.txt")

    if dataset in DATASETS:
        urls = DATASETS[dataset]
        if isinstance(urls, list):
            _download_multi(urls, raw_path)
        else:
            _download(urls, raw_path)
    elif os.path.isfile(dataset):
        raw_path = dataset
        print(f"  Fichier local : {dataset}")
    else:
        print(f"  ✗ Dataset inconnu : '{dataset}'")
        print(f"  Disponibles : {', '.join(DATASETS)}")
        return

    with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    print(f"  Texte brut : {len(text):,} caractères")
    _tokenize_and_save(text, data_dir, use_tiktoken=use_tiktoken)


def collect_code(source_dir: str, data_dir: str = "data",
                 extensions: tuple = (".py", ".js", ".ts", ".c", ".h", ".md"),
                 use_tiktoken: bool = True) -> None:
    """Rassemble du code source local en corpus et le tokenise."""
    parts = []
    found = 0
    for root, _, files in os.walk(source_dir):
        for name in sorted(files):
            if any(name.endswith(ext) for ext in extensions):
                path = os.path.join(root, name)
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    parts.append(f"// -- {path} --\n{content}")
                    found += 1
                except OSError:
                    pass

    if not parts:
        print(f"  ✗ Aucun fichier trouvé dans {source_dir} avec {extensions}")
        return

    text = "\n\n".join(parts)
    print(f"  {found} fichiers collectés | {len(text):,} caractères")
    _tokenize_and_save(text, data_dir, use_tiktoken=use_tiktoken)


# ─── Point d'entrée direct ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Préparation données nanoPOPIXA")
    parser.add_argument("--dataset",  default="shakespeare")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--char",     action="store_true",
                        help="Tokenisation caractère (défaut : tiktoken BPE)")
    args = parser.parse_args()
    prepare(args.dataset, args.data_dir, use_tiktoken=not args.char)


if __name__ == "__main__":
    main()
