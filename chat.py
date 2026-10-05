"""
nanoPOPIXA — Chat interactif CLI v2
Streaming · Thinking blocks · Effort levels · Diminishing returns · Auto-compact · Structured outputs
Usage : python chat.py   ou   popixa chat
"""

import sys
import json
from datetime import datetime
import argparse
import torch

try:
    import readline  # flèches directionnelles + historique des commandes
except ImportError:
    pass

from model import nanoPOPIXA, KVCache, checkpoint_v1_error
from session_cache import (save_session, load_session, clear_session, session_info,
                           checkpoint_fingerprint)

# ─── Couleurs ────────────────────────────────────────────────────────────────
R   = "\033[0m"
B   = "\033[1m"
DIM = "\033[2m"

def fg(r, g, b): return f"\033[38;2;{r};{g};{b}m"

USER_C  = fg(0,   220, 255)   # cyan    — [Toi]
MODEL_C = fg(180,  80, 255)   # violet  — [nanoPOPIXA]
THINK_C = fg(160, 140,  40)   # ocre    — [Thinking] (interne, tamisé)
INFO_C  = fg(90,   90, 120)   # gris    — infos / séparateurs
CMD_C   = fg(255, 180,   0)   # jaune   — retour des commandes
ERR_C   = fg(255,  80,  80)   # rouge   — erreurs
WARN_C  = fg(255, 140,   0)   # orange  — warnings contexte
CACHE_C = fg(80,  200, 120)   # vert    — session restaurée
JSON_C  = fg(80,  180, 255)   # bleu    — structured outputs

SEP = INFO_C + "─" * 52 + R

CACHE_PATH = "out-nanopopixa/session.cache"   # fichier KV-cache persistant

# ─── Seuils contexte (inspiré autoCompact.ts) ────────────────────────────────
# Claude : WARNING_BUFFER=20000, AUTOCOMPACT_BUFFER=13000, threshold=90%
CTX_WARNING_PCT = 0.80   # 80% → avertissement
CTX_COMPACT_PCT = 0.90   # 90% → compaction automatique (garde 50% récent)
MAX_COMPACT_FAILURES = 3 # circuit breaker — désactive l'auto-compact après 3 compactions consécutives

# ─── Estimation tokens (inspiré toolLimits.ts) ───────────────────────────────
# Claude : BYTES_PER_TOKEN = 4 (estimation conservative)
# Fallback quand aucun tokenizer n'est disponible — le chat compte les VRAIS tokens
# (encode) : en tokenisation caractère, 1 octet ≈ 1 token et l'estimation /4 sous-estimerait ×4.
BYTES_PER_TOKEN = 4

# ─── Effort presets (inspiré effort.ts) ──────────────────────────────────────
# Claude définit low/medium/high/max avec des budgets tokens et températures
EFFORT_PRESETS = {
    "low":    dict(temperature=1.0, top_k=20,  top_p=0.85, max_tokens=100),
    "medium": dict(temperature=0.8, top_k=40,  top_p=0.90, max_tokens=200),
    "high":   dict(temperature=0.7, top_k=50,  top_p=0.95, max_tokens=400),
    "max":    dict(temperature=1.0, top_k=None, top_p=0.95, max_tokens=600),
}


# ─── Chargement du modèle ────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: str):
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(ERR_C + f"\n  ✗ Checkpoint introuvable : {checkpoint_path}" + R)
        print(INFO_C + "  Entraîne d'abord le modèle :\n"
              "    popixa prep && popixa train --data_dir data/" + R)
        sys.exit(1)

    v1_error = checkpoint_v1_error(ckpt.get("model", {}))
    if v1_error:
        print(ERR_C + f"\n  ✗ {v1_error}" + R)
        sys.exit(1)

    model = nanoPOPIXA(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.train(False)

    if ckpt.get("tokenizer") == "tiktoken_gpt2":
        try:
            import tiktoken
        except ImportError:
            print(ERR_C + "  ✗ tiktoken requis : pip install tiktoken" + R)
            sys.exit(1)
        enc    = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode_ordinary(s)
        # Vocabulaire du modèle éventuellement rembourré (ex. 50304) : les ids de padding,
        # inconnus de tiktoken, sont ignorés au décodage au lieu de faire planter le tour
        decode = lambda l: enc.decode([i for i in l if 0 <= i < enc.n_vocab])
    elif "vocab" in ckpt:
        stoi   = ckpt["vocab"]["stoi"]
        itos   = ckpt["vocab"]["itos"]
        encode = lambda s: [stoi.get(c, 0) for c in s]
        decode = lambda l: "".join(itos.get(i, "") for i in l)
    else:
        print(ERR_C + "  ✗ Checkpoint sans vocabulaire (réentraîne le modèle)" + R)
        sys.exit(1)

    return model, encode, decode, ckpt


def load_token_bytes(ckpt: dict, vocab_size: int) -> list:
    """
    Octets de chaque token du vocabulaire (structured outputs) — None si inutilisable.
    Aligné sur vocab_size (padding éventuel du modèle → None).
    """
    import structured
    if ckpt.get("tokenizer") == "tiktoken_gpt2":
        import tiktoken
        tb = structured.token_bytes_from_tiktoken(tiktoken.get_encoding("gpt2"))
    else:
        tb = structured.token_bytes_from_itos(ckpt["vocab"]["itos"], vocab_size)
    tb = list(tb[:vocab_size])
    return tb + [None] * (vocab_size - len(tb))


# ─── Décodage incrémental (UTF-8 sûr) ────────────────────────────────────────
class _TextStream:
    """
    Décode un flux de tokens en texte sans casser les caractères multi-octets :
    un token BPE peut ne contenir qu'une partie d'un caractère UTF-8 (é, emoji…).
    On décode la séquence entière et on retient la fin tant qu'elle se termine
    par un caractère de remplacement (octets incomplets).
    """

    def __init__(self, decode):
        self.decode  = decode
        self.ids     = []
        self.emitted = 0   # nb de caractères déjà rendus

    def push(self, tok: int) -> str:
        self.ids.append(tok)
        text = self.decode(self.ids)
        if text.endswith("�"):
            return ""   # caractère incomplet — attendre le token suivant
        out = text[self.emitted:]
        self.emitted = len(text)
        return out

    def flush(self) -> str:
        text = self.decode(self.ids)
        out  = text[self.emitted:]
        self.emitted = len(text)
        return out

    @property
    def text(self) -> str:
        return self.decode(self.ids)


def _write(s: str) -> None:
    if s:
        sys.stdout.write(s)
        sys.stdout.flush()


def _drive(gen, decode, style: str = "plain", stats: dict = None) -> str:
    """
    Consomme un générateur de tokens (int) ou de (phase, token) en streamant le texte.

    style : "plain"       → tout est réponse
            "think"       → thinking affiché en ocre, puis transition vers la réponse
            "interleaved" → pauses de thinking signalées par [·] (texte masqué)
    Le générateur est TOUJOURS fermé (même sur Ctrl+C) → son KV-cache final est
    stocké dans cache_ref et reste cohérent avec ce qui a été généré.
    Retourne le texte de la réponse (sans thinking).
    """
    resp  = _TextStream(decode)
    think = _TextStream(decode)
    prev  = None
    n_resp = n_think = 0
    interrupted = False
    try:
        for item in gen:
            phase, tok = item if isinstance(item, tuple) else ("response", item)
            if phase == "think":
                n_think += 1
                if style == "interleaved":
                    if prev != "think":
                        _write(R + THINK_C + DIM + " [·] " + R + THINK_C + DIM)
                    think.push(tok)
                else:
                    _write(think.push(tok))
            else:
                n_resp += 1
                if style == "think" and prev != "response":
                    # Transition : ferme le bloc thinking, ouvre la réponse
                    _write(think.flush())
                    _write(R + "\n\n" + MODEL_C + B + "[nanoPOPIXA]" + R + " " + MODEL_C)
                elif style == "interleaved" and prev == "think":
                    _write(R + MODEL_C + " ")
                _write(resp.push(tok))
            prev = phase
    except KeyboardInterrupt:
        interrupted = True
    finally:
        gen.close()

    _write(think.flush() if prev == "think" and style != "interleaved" else "")
    _write(resp.flush())
    if interrupted:
        _write(INFO_C + "\n\n  [Interruption]" + R)
    _write(R + "\n")

    if stats is not None:
        stats.update(generated=n_resp + n_think, think=n_think, response=n_resp,
                     interrupted=interrupted)
    return resp.text


@torch.no_grad()
def _prefill_cache(model, ids: list, device: str):
    """Construit un KVCache couvrant exactement `ids` (tronqués à block_size) — None si vide."""
    ids = list(ids)[-model.config.block_size:]
    if not ids:
        return None
    _, kvs = model(torch.tensor([ids], dtype=torch.long, device=device))
    return KVCache(kvs, ids)


def _context_tensor(encode, context, device: str):
    """Texte ou liste d'ids → tenseur (1, T) — T peut être 0 (le modèle amorce seul)."""
    ids = encode(context) if isinstance(context, str) else list(context)
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


# ─── Génération streaming standard ───────────────────────────────────────────
def stream(model, encode, decode, context_str,
           max_tokens: int, temperature: float, top_k: int, device: str,
           repetition_penalty: float = 1.0, top_p: float = None,
           stop_on_repetition: bool = True,
           initial_past_kvs=None, cache_ref=None,
           stop_policy: str = None, stats: dict = None) -> str:
    """
    Génère en streaming avec arrêt automatique si répétitif.

    context_str      : texte ou liste d'ids à traiter en prefill.
    initial_past_kvs : cache KV existant (session persistante) — si fourni,
                       context_str doit contenir UNIQUEMENT les nouveaux tokens,
                       pas l'historique déjà caché.
    cache_ref        : liste mutable remplie avec [KVCache final] après génération.
    stop_policy      : "repetitive" | "diminishing" | "off" (prioritaire sur stop_on_repetition)
    stats            : dict optionnel rempli avec les compteurs de tokens.
    """
    ctx = _context_tensor(encode, context_str, device)
    _write("\n" + MODEL_C + B + "[nanoPOPIXA]" + R + " " + MODEL_C)
    gen = model.generate_stream(
        ctx, max_tokens, temperature, top_k,
        repetition_penalty, top_p, stop_on_repetition,
        initial_past_kvs=initial_past_kvs, cache_ref=cache_ref,
        stop_policy=stop_policy,
    )
    return _drive(gen, decode, "plain", stats)


# ─── Génération avec Thinking blocks ─────────────────────────────────────────
def stream_think(model, encode, decode, context_str,
                 device: str, think_budget: int, response_budget: int,
                 temperature: float, top_k: int, top_p: float,
                 repetition_penalty: float,
                 initial_past_kvs=None, cache_ref=None,
                 stop_policy: str = "repetitive", stats: dict = None) -> str:
    """
    Génération deux phases (Claude-inspired) :
      Phase 1 — Thinking interne  (affiché en ocre, temperature=1, arrêt "diminishing")
      Phase 2 — Réponse finale    (affichée en violet, temperature normale)
    Retourne la réponse seule (le thinking n'entre pas dans l'historique texte).
    """
    ctx = _context_tensor(encode, context_str, device)
    _write("\n" + THINK_C + DIM + B + "[Thinking]" + R + " " + THINK_C + DIM)
    gen = model.generate_stream_with_thinking(
        ctx,
        think_budget=think_budget,
        response_budget=response_budget,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        initial_past_kvs=initial_past_kvs,
        cache_ref=cache_ref,
        stop_policy=stop_policy,
    )
    return _drive(gen, decode, "think", stats)


# ─── Génération avec Interleaved Thinking ────────────────────────────────────
def stream_interleaved(model, encode, decode, context_str,
                       device: str, response_budget: int,
                       think_per_interleave: int, interleave_every: int,
                       temperature: float, top_k: int, top_p: float,
                       repetition_penalty: float,
                       initial_past_kvs=None, cache_ref=None,
                       stop_policy: str = "repetitive", stats: dict = None) -> str:
    """
    Thinking intercalé — mini-pauses de réflexion toutes les N tokens de réponse.
    Inspiré du beta `interleaved-thinking-2025-05-14`.
    """
    ctx = _context_tensor(encode, context_str, device)
    _write("\n" + MODEL_C + B + "[nanoPOPIXA]" + R + " " + MODEL_C)
    gen = model.generate_stream_with_interleaved_thinking(
        ctx,
        response_budget=response_budget,
        think_per_interleave=think_per_interleave,
        interleave_every=interleave_every,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        initial_past_kvs=initial_past_kvs,
        cache_ref=cache_ref,
        stop_policy=stop_policy,
    )
    return _drive(gen, decode, "interleaved", stats)


# ─── Génération speculative (fast mode) ──────────────────────────────────────
def stream_fast(model, encode, decode, context_str,
                device: str, max_tokens: int, n_draft: int,
                temperature: float, top_k: int, top_p: float,
                repetition_penalty: float,
                initial_past_kvs=None, cache_ref=None,
                draft: str = "ngram", stop_policy: str = None, stats: dict = None) -> str:
    """
    Speculative decoding — inspiré du beta `fast-mode-2026-02-01`.
    Drafts n-grammes (prompt lookup) vérifiés en une passe : même distribution de
    sortie que l'échantillonnage normal, plusieurs tokens par passe sur texte répétitif.
    """
    ctx = _context_tensor(encode, context_str, device)
    _write("\n" + MODEL_C + B + "[nanoPOPIXA]" + R + fg(80, 200, 120) + " ⚡ " + R + MODEL_C)
    gen = model.speculative_generate_stream(
        ctx, max_tokens, n_draft,
        temperature, top_k, top_p, repetition_penalty,
        initial_past_kvs=initial_past_kvs, cache_ref=cache_ref,
        draft=draft, stop_policy=stop_policy,
    )
    return _drive(gen, decode, "plain", stats)


# ─── Génération structurée (JSON / JSON Schema) ──────────────────────────────
def stream_json(model, encode, decode, context_str,
                device: str, constraint, max_tokens: int,
                temperature: float, top_k: int, top_p: float,
                repetition_penalty: float,
                initial_past_kvs=None, cache_ref=None, stats: dict = None) -> str:
    """
    Structured outputs — inspiré du beta `structured-outputs-2025-12-15`.
    Chaque token est contraint par la grammaire JSON (et le schéma s'il y en a un) ;
    le JSON est fermé automatiquement si le budget de tokens s'épuise.
    """
    ctx = _context_tensor(encode, context_str, device)
    _write("\n" + MODEL_C + B + "[nanoPOPIXA]" + R + JSON_C + " {json} " + R + MODEL_C)
    constraint.reset()
    gen = model.generate_structured(
        ctx, constraint,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        initial_past_kvs=initial_past_kvs,
        cache_ref=cache_ref,
    )
    text = _drive(gen, decode, "plain", stats)
    if stats is not None:
        stats["json_complete"] = constraint.is_complete()
    return text


def report_json(constraint, schema, interrupted: bool = False, task_capped: bool = False) -> None:
    """
    Affiche le statut du JSON généré (complet + valide vis-à-vis du schéma ?).
    interrupted : Ctrl+C pendant le tour · task_capped : budget du tour réduit par /taskbudget
    """
    import structured
    if not constraint.is_complete():
        if interrupted:
            reason = "génération interrompue"
        elif task_capped:
            reason = "task budget presque épuisé — /taskbudget N pour l'augmenter"
        else:
            reason = "budget de tokens épuisé — augmente /tokens"
        print(WARN_C + "  ⚠ JSON incomplet" + INFO_C + f"  ({reason})" + R)
        return
    try:
        instance = json.loads(constraint.generated.decode("utf-8"))
    except RecursionError:
        # json.loads plafonne vers ~1000 niveaux ; le décodage contraint garantit déjà le JSON
        detail = "grammaire garantie" if schema is None else "schéma garanti par le décodage"
        print(JSON_C + "  ✓ JSON valide" + INFO_C
              + f"  (trop profond pour être relu par json.loads — {detail})" + R)
        return
    except ValueError as e:
        print(ERR_C + f"  ✗ JSON invalide : {e}" + R)
        return
    errors = structured.validate_instance(instance, schema)
    if errors:
        print(ERR_C + "  ✗ JSON hors schéma : " + "; ".join(errors[:3]) + R)
    else:
        label = "schéma respecté" if schema is not None else "JSON valide"
        print(JSON_C + f"  ✓ {label}" + R)


# ─── Gestion contexte (inspiré autoCompact.ts) ───────────────────────────────
def _estimate_tokens(text: str) -> int:
    """Estime le nombre de tokens depuis une chaîne (BYTES_PER_TOKEN=4, d'après toolLimits.ts)."""
    return max(1, len(text.encode("utf-8")) // BYTES_PER_TOKEN)


def _count_tokens(text: str, encode=None) -> int:
    """Nombre de tokens réel si un tokenizer est fourni, sinon estimation BYTES_PER_TOKEN."""
    if not text:
        return 0
    return len(encode(text)) if encode is not None else _estimate_tokens(text)


def _tail_tokens(text: str, n_tokens: int, encode=None, decode=None) -> str:
    """Garde les n_tokens derniers tokens de `text` (approximation en octets sans tokenizer)."""
    if n_tokens <= 0:
        return ""
    if encode is not None and decode is not None:
        ids = encode(text)
        return text if len(ids) <= n_tokens else decode(ids[-n_tokens:])
    data = text.encode("utf-8")[-n_tokens * BYTES_PER_TOKEN:]
    return data.decode("utf-8", errors="ignore")


def update_context(history: str, new_content: str, blk_size: int,
                   encode=None, decode=None, used_tokens: int = None) -> tuple:
    """
    Ajoute new_content à l'historique.
    Retourne (history_updated, was_compacted).
    Compacte automatiquement à 90% de la fenêtre (garde les 50% de tokens les plus récents).

    used_tokens : remplissage réel de la fenêtre si connu (ex. longueur du KV-cache,
                  thinking inclus) — sinon on compte les tokens de l'historique.
    """
    history += new_content
    used     = max(used_tokens or 0, _count_tokens(history, encode))

    if used / blk_size >= CTX_COMPACT_PCT:
        return _tail_tokens(history, blk_size // 2, encode, decode), True

    return history, False


def context_warning(used_tokens: int, blk_size: int):
    """Affiche un warning si le contexte dépasse 80% de la fenêtre."""
    pct = used_tokens / blk_size
    if pct >= CTX_WARNING_PCT:
        bar_len = 20
        filled  = int(bar_len * min(pct, 1.0))
        bar     = "█" * filled + "░" * (bar_len - filled)
        print(WARN_C + f"  ⚠ Contexte [{bar}] {pct*100:.0f}%"
              + INFO_C + " — compaction auto à 90%" + R)


# ─── Sauvegarde conversation ──────────────────────────────────────────────────
def save_conversation(turns: list, filename: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    sep = "═" * 48
    lines = [f"nanoPOPIXA — Conversation du {now}", sep, ""]
    for role, text in turns:
        label = "[Toi]" if role == "user" else "[nanoPOPIXA]"
        lines.append(f"{label} {text.strip()}")
        lines.append("")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(CMD_C + f"  → sauvegardé : {filename}" + R)
    except OSError as e:
        print(ERR_C + f"  ✗ Erreur d'écriture : {e}" + R)


# ─── Boucle principale ───────────────────────────────────────────────────────
def _print_help():
    """Liste des commandes in-chat (affichée au démarrage et via /help)."""
    print(INFO_C + "  Paramètres :" + R)
    print(CMD_C  + "    /temp 0.5"      + INFO_C + "          → température  (0 = greedy)  (défaut 0.8)" + R)
    print(CMD_C  + "    /tokens 300"    + INFO_C + "        → tokens réponse            (défaut 200)"  + R)
    print(CMD_C  + "    /topp 0.9"      + INFO_C + "         → nucleus sampling top-p   (défaut off)"  + R)
    print(CMD_C  + "    /penalty 1.3"   + INFO_C + "       → repetition penalty         (défaut 1.0)"  + R)
    print(CMD_C  + "    /stop diminishing" + INFO_C + "  → arrêt anticipé : repetitive|diminishing|off" + R)
    print(INFO_C + "  Modes :" + R)
    print(CMD_C  + "    /think"         + INFO_C + "           → mode raisonnement interne (thinking)" + R)
    print(CMD_C  + "    /thinkbudget 200" + INFO_C + "   → tokens alloués au thinking   (défaut 150)"  + R)
    print(CMD_C  + "    /adaptive"     + INFO_C + "         → budget thinking adaptatif selon prompt"  + R)
    print(CMD_C  + "    /interleaved"  + INFO_C + "       → thinking intercalé toutes les N tokens"    + R)
    print(CMD_C  + "    /fast"         + INFO_C + "             → speculative decoding ⚡ (fast mode)" + R)
    print(CMD_C  + "    /draft ngram|self" + INFO_C + "  → source des drafts du fast mode"             + R)
    print(CMD_C  + "    /json [schéma]" + INFO_C + "     → structured outputs : JSON (schéma optionnel)" + R)
    print(CMD_C  + "    /redactthink"  + INFO_C + "       → exclure tokens thinking du contexte"       + R)
    print(CMD_C  + "    /effort low|medium|high|max" + INFO_C + " → preset tout-en-un"                 + R)
    print(INFO_C + "  Budget & contexte :" + R)
    print(CMD_C  + "    /taskbudget 2000" + INFO_C + "  → budget tokens sur la tâche (multi-tours)"   + R)
    print(CMD_C  + "    /taskbudget off"  + INFO_C + "   → désactiver le task budget"                  + R)
    print(INFO_C + "  Contexte & outils :" + R)
    print(CMD_C  + "    /reset"         + INFO_C + "           → remettre le contexte à zéro"          + R)
    print(CMD_C  + "    /ctx"           + INFO_C + "             → afficher l'état du contexte"         + R)
    print(CMD_C  + "    /cache"         + INFO_C + "           → infos sur la session persistante"       + R)
    print(CMD_C  + "    /clearcache"    + INFO_C + "       → effacer la mémoire inter-sessions (KV-cache)" + R)
    print(CMD_C  + "    /libre"         + INFO_C + "           → génération sans prompt"                + R)
    print(CMD_C  + "    /save [fichier]"+ INFO_C + "    → sauvegarder la conversation (.txt)"           + R)
    print(CMD_C  + "    /help"          + INFO_C + "            → cette aide"                           + R)
    print(INFO_C + "  Ctrl+C → quitter" + R)


def run_chat(checkpoint_path: str, max_tokens: int, temperature: float, top_k: int,
             repetition_penalty: float = 1.0, top_p: float = None):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Empreinte prise AVANT le chargement : si un entraînement réécrit le checkpoint
    # pendant le chat, le cache (calculé avec les poids chargés) ne sera pas réutilisé
    ckpt_fp = checkpoint_fingerprint(checkpoint_path)
    model, encode, decode, ckpt = load_model(checkpoint_path, device)

    n_params     = model.get_num_params(False) / 1e6
    iter_num     = ckpt.get("iter", "?")
    blk_size     = model.config.block_size
    think_budget = 150   # tokens de réflexion interne (phase thinking)

    def n_tokens(text: str) -> int:
        return _count_tokens(text, encode)

    # ── Restauration de session (KV-cache persistant) ─────────────────────────
    # Inspiré du prompt caching de Claude Code (prompt-caching-scope beta)
    # Le cache KV encode tous les tokens précédents — O(0) au redémarrage.
    # Invariant : session_token_ids == ids exacts couverts par session_past_kvs.
    session_past_kvs = None
    session_token_ids: list = []
    restored_history = None

    past_kvs_loaded, token_ids_loaded, restored_history = load_session(
        CACHE_PATH, checkpoint_path, device, with_history=True, fingerprint=ckpt_fp
    )
    if (past_kvs_loaded is not None
            and past_kvs_loaded.token_ids is not None
            and past_kvs_loaded.seq_len <= blk_size):
        session_past_kvs  = past_kvs_loaded
        session_token_ids = list(past_kvs_loaded.token_ids)

    # ── En-tête ───────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print(INFO_C + f"  nanoPOPIXA v2  ·  {n_params:.2f}M params  ·  iter {iter_num}  ·  {device}" + R)
    if session_past_kvs is not None:
        print(CACHE_C + f"  Session restaurée — {len(session_token_ids)} tokens en mémoire (KV-cache)" + R)
    elif token_ids_loaded:
        print(CACHE_C + "  Historique restauré — KV-cache reconstruit au prochain message" + R)
    print(SEP)
    _print_help()
    print(SEP)

    # Historique texte (sans thinking) — restauré depuis la session si disponible
    if restored_history is not None:
        history = restored_history
    else:
        history = decode(token_ids_loaded) if token_ids_loaded else ""
    turns      = []
    think_mode       = False   # activé par /think
    adaptive_mode    = False   # budget de thinking adaptatif
    fast_mode        = False   # speculative decoding
    interleaved_mode = False   # thinking intercalé
    redact_thinking  = False   # supprime tokens thinking du contexte (redact-thinking-2026-02-12)
    n_draft          = 4       # tokens draft en fast mode
    draft_mode       = "ngram" # source des drafts : ngram (prompt lookup) | self
    think_per_inter  = 20      # tokens thinking par pause (interleaved)
    interleave_every = 50      # tokens réponse entre deux pauses (interleaved)
    stop_policy      = "repetitive"  # arrêt anticipé des réponses (tokenBudget.ts)

    # ── Structured outputs (structured-outputs-2025-12-15) ───────────────────
    json_mode        = False
    json_schema      = None    # schéma JSON actif (None = n'importe quel JSON)
    json_constraint  = None    # structured.TokenConstraint (trie du vocabulaire réutilisé)
    token_bytes      = None    # octets de chaque token — calculés à la 1re activation

    # ── Circuit breaker auto-compact (MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES=3) ─
    compact_failures = 0       # nb de compactions consécutives
    compact_disabled = False   # circuit breaker déclenché → fenêtre glissante seule

    # ── Task budget (task-budgets-2026-03-13) ────────────────────────────────
    task_budget      = None    # budget total tokens pour la tâche en cours (None = illimité)
    task_tokens_used = 0       # tokens consommés depuis le début de la tâche

    def context_used() -> int:
        """Tokens que le modèle verra au prochain tour (cache, thinking inclus, ou historique)."""
        if session_past_kvs is not None:
            return len(session_token_ids)
        return min(n_tokens(history), blk_size)

    while True:
        # ── Indicateur contexte dans le prompt ───────────────────────────────
        ctx_pct   = context_used() / blk_size
        ctx_label = (
            WARN_C + f"[ctx {ctx_pct*100:.0f}%] " + R
            if ctx_pct >= CTX_WARNING_PCT
            else ""
        )
        # Task budget label
        if task_budget is not None:
            tb_pct = task_tokens_used / task_budget
            if tb_pct >= 0.8:
                ctx_label += fg(255, 140, 0) + f"[task {tb_pct*100:.0f}%] " + R
        mode_parts = []
        if json_mode:
            mode_parts.append(JSON_C + ("[json:schema]" if json_schema is not None else "[json]") + R)
        elif think_mode:
            mode_parts.append(THINK_C + ("[think:adaptive]" if adaptive_mode else "[think]") + R)
        elif fast_mode:
            mode_parts.append(fg(80, 200, 120) + "[fast]" + R)
        elif interleaved_mode:
            mode_parts.append(THINK_C + "[interleaved]" + R)
        think_label = " ".join(mode_parts) + " " if mode_parts else ""

        # ── Saisie utilisateur ────────────────────────────────────────────────
        try:
            sys.stdout.write(
                "\n" + ctx_label + think_label + USER_C + B + "[Toi] " + R + USER_C
            )
            sys.stdout.flush()
            prompt = input().strip()
            sys.stdout.write(R)
            sys.stdout.flush()
        except (KeyboardInterrupt, EOFError):
            print(R + "\n\n" + INFO_C + "  À bientôt !" + R + "\n")
            break

        if not prompt:
            continue

        # ── Commandes ─────────────────────────────────────────────────────────
        # On compare le MOT de commande (pas un préfixe) : "/temp" sans argument
        # affiche l'usage au lieu de partir au modèle comme un message.
        cmd = prompt.split()[0].lower() if prompt.startswith("/") else ""

        if cmd == "/help":
            _print_help()
            continue

        # /temp
        if cmd == "/temp":
            try:
                temperature = float(prompt.split()[1])
                label = " (greedy)" if temperature <= 0 else ""
                print(CMD_C + f"  → température : {temperature}{label}" + R)
            except (IndexError, ValueError):
                print(ERR_C + "  usage : /temp 0.8" + R)
            continue

        # /tokens
        if cmd == "/tokens":
            try:
                max_tokens = int(prompt.split()[1])
                print(CMD_C + f"  → max tokens : {max_tokens}" + R)
            except (IndexError, ValueError):
                print(ERR_C + "  usage : /tokens 300" + R)
            continue

        # /topp
        if cmd == "/topp":
            try:
                top_p = float(prompt.split()[1])
                print(CMD_C + f"  → top-p : {top_p}" + R)
            except (IndexError, ValueError):
                print(ERR_C + "  usage : /topp 0.9" + R)
            continue

        # /penalty
        if cmd == "/penalty":
            try:
                value = float(prompt.split()[1])
                if value <= 0:
                    raise ValueError
                repetition_penalty = value
                print(CMD_C + f"  → repetition penalty : {repetition_penalty}" + R)
            except (IndexError, ValueError):
                print(ERR_C + "  usage : /penalty 1.3   (> 0 ; 1.0 = désactivé)" + R)
            continue

        # /stop — politique d'arrêt anticipé des réponses (tokenBudget.ts)
        if cmd == "/stop":
            parts  = prompt.split()
            policy = parts[1].lower() if len(parts) > 1 else ""
            if policy not in nanoPOPIXA.STOP_POLICIES:
                print(ERR_C + "  usage : /stop repetitive|diminishing|off" + R)
                print(INFO_C + "  repetitive  → 1 fenêtre de 40 tokens répétitive suffit (défaut)" + R)
                print(INFO_C + "  diminishing → 3 fenêtres consécutives (DIMINISHING_THRESHOLD ×3)" + R)
            else:
                stop_policy = policy
                print(CMD_C + f"  → arrêt anticipé : {stop_policy}" + R)
            continue

        # /effort — preset tout-en-un (inspiré effort.ts de Claude Code)
        if cmd == "/effort":
            parts = prompt.split()
            level = parts[1].lower() if len(parts) > 1 else ""
            if level not in EFFORT_PRESETS:
                print(ERR_C + "  usage : /effort low|medium|high|max" + R)
                print(INFO_C + "  Presets :" + R)
                for k, v in EFFORT_PRESETS.items():
                    print(CMD_C + f"    {k:8s}" + INFO_C
                          + f"  temp={v['temperature']}  top_k={v['top_k']}  "
                          + f"top_p={v['top_p']}  tokens={v['max_tokens']}" + R)
            else:
                p           = EFFORT_PRESETS[level]
                temperature = p["temperature"]
                top_k       = p["top_k"]
                top_p       = p["top_p"]
                max_tokens  = p["max_tokens"]
                print(CMD_C + f"  → effort [{level}]"
                      + INFO_C + f"  temp={temperature}  top_k={top_k}  "
                      + f"top_p={top_p}  tokens={max_tokens}" + R)
            continue

        # /think — bascule le mode thinking
        if prompt == "/think":
            think_mode = not think_mode
            if think_mode:
                fast_mode = interleaved_mode = False   # mutuellement exclusifs
                print(THINK_C + f"  → Thinking activé  (budget={think_budget} tokens)" + R)
                print(INFO_C  + "  Phase 1 : raisonnement interne  temperature=1 (comme Claude)" + R)
                print(INFO_C  + "  Phase 2 : réponse finale        temperature normale" + R)
            else:
                print(CMD_C + "  → Thinking désactivé" + R)
            continue

        # /thinkbudget — ajuster le budget de thinking
        if cmd == "/thinkbudget":
            try:
                think_budget = int(prompt.split()[1])
                print(THINK_C + f"  → think budget : {think_budget} tokens" + R)
            except (IndexError, ValueError):
                print(ERR_C + "  usage : /thinkbudget 200" + R)
            continue

        # /adaptive — budget de thinking adaptatif selon complexité du prompt
        if prompt == "/adaptive":
            adaptive_mode = not adaptive_mode
            if adaptive_mode:
                print(THINK_C + "  → Adaptive thinking activé"
                      + INFO_C + "  (budget auto selon longueur du prompt)" + R)
                print(INFO_C   + "  Court (<50 tok) → 50  ·  Moyen → 150/300  ·  Long → 500" + R)
                if not think_mode:
                    print(INFO_C + "  (effectif en mode /think)" + R)
            else:
                print(CMD_C + "  → Adaptive thinking désactivé" + R)
            continue

        # /fast — speculative decoding (fast mode)
        if prompt == "/fast":
            fast_mode = not fast_mode
            if fast_mode:
                think_mode = interleaved_mode = False   # mutuellement exclusifs
            if fast_mode:
                print(fg(80, 200, 120) + f"  → Fast mode activé ⚡  (draft N={n_draft}, {draft_mode})" + R)
                print(INFO_C + f"  Speculative decoding : propose {n_draft} tokens draft,"
                      + " vérifie en une passe (distribution inchangée)." + R)
            else:
                print(CMD_C + "  → Fast mode désactivé" + R)
            continue

        # /draft — source des drafts du fast mode
        if cmd == "/draft":
            parts = prompt.split()
            mode  = parts[1].lower() if len(parts) > 1 else ""
            if mode not in ("ngram", "self"):
                print(ERR_C + "  usage : /draft ngram|self" + R)
                print(INFO_C + "  ngram → prompt lookup : recopie une suite déjà vue (gratuit, rapide)" + R)
                print(INFO_C + "  self  → même modèle à temp≈0 (pédagogique, pas plus rapide)" + R)
            else:
                draft_mode = mode
                print(CMD_C + f"  → drafts : {draft_mode}" + R)
            continue

        # /interleaved — thinking intercalé
        if prompt == "/interleaved":
            interleaved_mode = not interleaved_mode
            if interleaved_mode:
                think_mode = fast_mode = False   # mutuellement exclusifs
            if interleaved_mode:
                print(THINK_C + "  → Interleaved thinking activé"
                      + INFO_C + f"  (pause {think_per_inter} tok every {interleave_every} tok)" + R)
            else:
                print(CMD_C + "  → Interleaved thinking désactivé" + R)
            continue

        # /json — structured outputs (structured-outputs-2025-12-15)
        if cmd == "/json":
            arg = prompt[len("/json"):].strip()
            if arg == "off" or (not arg and json_mode):
                json_mode = False
                print(CMD_C + "  → Structured outputs désactivés" + R)
                continue
            try:
                import structured
                schema = structured.load_schema(arg) if arg else None
                if token_bytes is None:
                    print(INFO_C + "  Indexation du vocabulaire…" + R)
                    token_bytes = load_token_bytes(ckpt, model.config.vocab_size)
                candidate = structured.json_constraint(token_bytes, schema)
                if candidate.completion_tokens() is None:
                    raise ValueError("le vocabulaire du modèle ne permet pas d'écrire un JSON "
                                     "conforme à ce schéma")
                json_constraint = candidate
                json_schema     = schema
                json_mode       = True
            except (ImportError, ValueError, OSError) as e:
                print(ERR_C + f"  ✗ Structured outputs : {e}" + R)
                continue
            label = "JSON conforme au schéma" if schema is not None else "JSON valide"
            print(JSON_C + f"  → Structured outputs activés — {label}" + R)
            print(INFO_C + "  Prioritaire sur think/fast/interleaved · /json off pour désactiver" + R)
            ignored = sorted(getattr(json_constraint.matcher, "ignored_keywords", ()) or ())
            if ignored:
                print(WARN_C + "  ⚠ Mots-clés non appliqués : " + ", ".join(ignored) + R)
            continue

        # /redactthink — supprime les tokens thinking du contexte (redact-thinking-2026-02-12)
        if prompt == "/redactthink":
            redact_thinking = not redact_thinking
            if redact_thinking:
                print(THINK_C + "  → Redact thinking activé"
                      + INFO_C + "  (tokens de réflexion exclus du contexte — fenêtre préservée)" + R)
            else:
                print(CMD_C + "  → Redact thinking désactivé" + R)
            continue

        # /taskbudget — budget total tokens sur la tâche (task-budgets-2026-03-13)
        if cmd == "/taskbudget":
            parts = prompt.split()
            if len(parts) < 2 or parts[1] == "off":
                task_budget      = None
                task_tokens_used = 0
                print(CMD_C + "  → Task budget désactivé" + R)
            else:
                try:
                    task_budget = int(parts[1])
                    if task_budget <= 0:
                        raise ValueError
                    task_tokens_used = 0
                    print(fg(255, 140, 0) + f"  → Task budget : {task_budget} tokens"
                          + INFO_C + "  (réinitialisé)" + R)
                except ValueError:
                    task_budget = None
                    print(ERR_C + "  usage : /taskbudget 2000   ou   /taskbudget off" + R)
            continue

        # /ctx — afficher l'état du contexte
        if prompt == "/ctx":
            used    = context_used()
            pct     = used / blk_size * 100
            bar_l   = 30
            filled  = int(bar_l * min(pct, 100) / 100)
            bar     = "█" * filled + "░" * (bar_l - filled)
            color   = WARN_C if pct >= CTX_WARNING_PCT * 100 else CMD_C
            source  = "KV-cache" if session_past_kvs is not None else "historique"
            print(color + f"  Contexte [{bar}] {pct:.1f}%  ({used}/{blk_size} tokens · {source})" + R)
            print(INFO_C + f"  Warning à {CTX_WARNING_PCT*100:.0f}%"
                  + f"  ·  Auto-compact à {CTX_COMPACT_PCT*100:.0f}%"
                  + (f"  ·  circuit breaker {compact_failures}/{MAX_COMPACT_FAILURES}" if compact_failures else "")
                  + ("  ·  auto-compact désactivé (fenêtre glissante)" if compact_disabled else "")
                  + R)
            if task_budget is not None:
                tb_pct = task_tokens_used / task_budget * 100
                tb_bar = "█" * int(bar_l * min(tb_pct,100)/100) + "░" * (bar_l - int(bar_l * min(tb_pct,100)/100))
                tc = WARN_C if tb_pct >= 80 else CMD_C
                print(tc + f"  Task budget [{tb_bar}] {tb_pct:.1f}%  ({task_tokens_used}/{task_budget} tokens)" + R)
            continue

        # /cache — infos session persistante
        if prompt == "/cache":
            info = session_info(CACHE_PATH, checkpoint_path, fingerprint=ckpt_fp)
            if not info["exists"]:
                print(INFO_C + "  Aucune session persistante sauvegardée." + R)
            else:
                age = datetime.now().timestamp() - info["mtime"]
                age_str = f"{age/60:.0f}min" if age < 3600 else f"{age/3600:.1f}h"
                status  = CACHE_C + "valide" if info["valid"] else ERR_C + "invalide (modèle changé)"
                print(CACHE_C + f"  Session cache : {info['n_tokens']} tokens · "
                      + f"{info['size_kb']:.1f} KB · {age_str} ago · {status}" + R)
            continue

        # /clearcache — effacer la mémoire persistante
        if prompt == "/clearcache":
            erased            = clear_session(CACHE_PATH)
            had_cache         = session_past_kvs is not None
            session_past_kvs  = None
            session_token_ids = []
            compact_failures  = 0
            compact_disabled  = False
            if erased or had_cache:
                print(CACHE_C + "  → KV-cache et session persistante effacés." + R)
                print(INFO_C + "  La conversation en cours reste en mémoire et sera de nouveau"
                      " sauvegardée au prochain message — /reset pour tout effacer." + R)
            else:
                print(INFO_C + "  Aucune session à effacer." + R)
            continue

        # /reset
        if prompt == "/reset":
            history           = ""
            session_past_kvs  = None
            session_token_ids = []
            compact_failures  = 0
            compact_disabled  = False
            clear_session(CACHE_PATH)
            print(CMD_C + "  → contexte et session remis à zéro" + R)
            continue

        # /libre — échantillon libre, sans toucher à la conversation
        if prompt == "/libre":
            stream(model, encode, decode, "", max_tokens, temperature, top_k, device,
                   repetition_penalty, top_p, stop_policy=stop_policy)
            continue

        # /save
        if cmd == "/save":
            parts    = prompt.split(maxsplit=1)
            filename = (
                parts[1] if len(parts) > 1
                else f"conv_{datetime.now().strftime('%Y-%m-%d_%Hh%M')}.txt"
            )
            if not turns:
                print(ERR_C + "  ✗ Aucune conversation à sauvegarder" + R)
            else:
                save_conversation(turns, filename)
            continue

        if cmd:
            print(ERR_C + f"  ✗ Commande inconnue : {cmd}" + INFO_C + "  — /help pour la liste" + R)
            continue

        # ── Budgets du tour ───────────────────────────────────────────────────
        prompt_ids = encode(prompt)
        active_think_budget = (
            nanoPOPIXA.adaptive_think_budget(len(prompt_ids))
            if adaptive_mode else think_budget
        )
        use_think    = think_mode and not json_mode
        use_inter    = interleaved_mode and not json_mode and not use_think
        use_fast     = fast_mode and not json_mode and not use_think and not use_inter
        resp_budget  = max_tokens
        think_tokens = active_think_budget if use_think else 0

        # Task budget : on ne génère jamais au-delà du budget restant
        if task_budget is not None:
            remaining = task_budget - task_tokens_used - len(prompt_ids)
            if remaining <= 0:
                print(ERR_C + f"  ✗ Task budget épuisé ({task_tokens_used}/{task_budget} tokens)"
                      + INFO_C + "  — /taskbudget N pour redéfinir" + R)
                continue
            if use_think:
                think_tokens = min(think_tokens, remaining // 2)
            resp_budget = min(resp_budget, remaining - think_tokens)

        # ── Contexte d'entrée ─────────────────────────────────────────────────
        # Avec KV-cache persistant : on ne passe que les NOUVEAUX tokens au modèle.
        # Le cache contient déjà tous les K/V de l'historique → O(N_new) au lieu de O(N_total).
        # Si la fenêtre déborde, le modèle fait glisser son contexte (ids exacts du cache).
        # Sans cache (premier démarrage, /reset, compaction) : historique complet,
        # tronqué pour laisser de la place à la génération.
        if session_past_kvs is not None:
            gen_input = prompt_ids
            init_kvs  = session_past_kvs
        else:
            reserve   = min(think_tokens + resp_budget, blk_size // 2)
            gen_input = encode(history + prompt)[-max(1, blk_size - reserve):]
            init_kvs  = None

        context_warning(len(session_token_ids) + len(prompt_ids) if init_kvs is not None
                        else len(gen_input), blk_size)

        # ── Génération ────────────────────────────────────────────────────────
        cache_ref = []
        stats     = {}
        sampling  = dict(temperature=temperature, top_k=top_k,
                         top_p=top_p if top_p is not None else 0.9,
                         repetition_penalty=repetition_penalty,
                         initial_past_kvs=init_kvs, cache_ref=cache_ref, stats=stats)

        try:
            if json_mode:
                response = stream_json(
                    model, encode, decode, gen_input, device, json_constraint, resp_budget,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    initial_past_kvs=init_kvs, cache_ref=cache_ref, stats=stats,
                )
                report_json(json_constraint, json_schema, interrupted=stats.get("interrupted", False),
                        task_capped=resp_budget < max_tokens)

            elif use_think:
                response = stream_think(
                    model, encode, decode, gen_input, device,
                    think_budget=think_tokens, response_budget=resp_budget,
                    stop_policy=stop_policy, **sampling,
                )

            elif use_inter:
                response = stream_interleaved(
                    model, encode, decode, gen_input, device,
                    response_budget=resp_budget,
                    think_per_interleave=think_per_inter,
                    interleave_every=interleave_every,
                    stop_policy=stop_policy, **sampling,
                )

            elif use_fast:
                response = stream_fast(
                    model, encode, decode, gen_input, device,
                    max_tokens=resp_budget, n_draft=n_draft,
                    draft=draft_mode, stop_policy=stop_policy, **sampling,
                )

            else:
                # Mode standard — KV-cache persistant actif
                response = stream(
                    model, encode, decode, gen_input,
                    resp_budget, temperature, top_k, device,
                    repetition_penalty, top_p,
                    initial_past_kvs=init_kvs,
                    cache_ref=cache_ref,
                    stop_policy=stop_policy,
                    stats=stats,
                )
        except Exception as e:
            # Une erreur pendant un tour ne doit jamais terminer la session
            print(R + ERR_C + f"\n  ✗ Erreur de génération : {type(e).__name__}: {e}" + R)
            session_past_kvs  = None
            session_token_ids = []
            continue

        # ── Mise à jour de la session : ids exacts du KV-cache ───────────────
        kv = cache_ref[0] if cache_ref else None
        if kv is not None and kv.token_ids is not None:
            if init_kvs is not None and kv.token_ids[:len(session_token_ids)] != session_token_ids:
                print(INFO_C + "  [Fenêtre glissante — le début de la conversation sort du contexte]" + R)
            session_past_kvs  = kv
            session_token_ids = list(kv.token_ids)
        else:
            session_past_kvs  = None   # cache inutilisable → reconstruit au prochain tour
            session_token_ids = []

        # ── Mise à jour contexte string avec auto-compact + circuit breaker ─────
        # Le thinking n'entre jamais dans l'historique texte (stream_think retourne resp seul)
        new_content = prompt + response
        used_now    = len(session_token_ids) if session_past_kvs is not None else None

        if compact_disabled:
            # Circuit breaker déclenché — plus de compaction : fenêtre glissante seule
            # (l'historique texte est borné à la fenêtre, le cache glisse dans le modèle)
            history = _tail_tokens(history + new_content, blk_size, encode, decode)
        else:
            history, compacted = update_context(history, new_content, blk_size,
                                                encode, decode, used_tokens=used_now)
            if compacted:
                compact_failures += 1
                session_past_kvs  = None
                session_token_ids = []
                clear_session(CACHE_PATH)
                msg = "  [Auto-compact : contexte réduit à 50% · session cache réinitialisé]"
                if compact_failures >= MAX_COMPACT_FAILURES:
                    compact_disabled = True
                    msg += (f" — circuit breaker déclenché après {MAX_COMPACT_FAILURES}"
                            " compactions consécutives (fenêtre glissante seule)")
                print(INFO_C + msg + R)
            else:
                compact_failures = 0  # reset si pas de compaction ce tour

        # Redact thinking : le cache contient les tokens de réflexion → on le reconstruit
        # tout de suite depuis l'historique texte (sans thinking) : contexte et session
        # persistante ne gardent jamais la réflexion
        if (use_think or use_inter) and redact_thinking and session_past_kvs is not None:
            session_past_kvs  = _prefill_cache(model, encode(history), device)
            session_token_ids = list(session_past_kvs.token_ids) if session_past_kvs is not None else []

        # ── Sauvegarde du cache sur disque ────────────────────────────────────
        if session_past_kvs is not None:
            try:
                save_session(CACHE_PATH, session_past_kvs, session_token_ids,
                             checkpoint_path, history=history, fingerprint=ckpt_fp)
            except (OSError, RuntimeError) as e:
                print(ERR_C + f"  ✗ Sauvegarde de session impossible : {e}" + R)

        # ── Task budget : tracking tokens consommés (task-budgets-2026-03-13) ─
        turn_tokens = len(prompt_ids) + stats.get("generated", n_tokens(response))
        if task_budget is not None:
            task_tokens_used += turn_tokens
            remaining = task_budget - task_tokens_used
            if task_tokens_used >= task_budget:
                print(fg(255, 80, 80) + f"  ✗ Task budget épuisé ({task_tokens_used}/{task_budget} tokens)"
                      + INFO_C + "  — /taskbudget N pour redéfinir" + R)
            elif remaining < task_budget * 0.20:
                print(fg(255, 140, 0) + f"  ⚠ Task budget à {task_tokens_used/task_budget*100:.0f}%"
                      + INFO_C + f"  ({remaining} tokens restants)" + R)

        turns.append(("user",  prompt))
        turns.append(("model", response))


# ─── Point d'entrée ───────────────────────────────────────────────────────────
def _print_launch_banner():
    """Banner affiché au lancement du CLI chat."""
    c1 = fg(200,  20, 255)
    c2 = fg(100, 100, 255)
    c3 = fg(0,   220, 255)
    print()
    print(c1 + B + "  ███╗   ██╗ █████╗ ███╗   ██╗ ██████╗ " + R)
    print(c2 + B + "  ████╗  ██║██╔══██╗████╗  ██║██╔═══██╗" + R)
    print(c3 + B + "  ██╔██╗ ██║███████║██╔██╗ ██║██║   ██║" + R)
    print(c2 + B + "  ██║╚██╗██║██╔══██║██║╚██╗██║██║   ██║" + R)
    print(c1 + B + "  ██║ ╚████║██║  ██║██║ ╚████║╚██████╔╝" + R)
    print(fg(90, 90, 120) + "  ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ " + R)
    print()
    print(fg(180, 80, 255) + B + "  P O P I X A" + R
          + fg(90, 90, 120) + "  ·  Chat CLI v2  ·  démarrage..." + R)
    print()


def _positive_float(value: str) -> float:
    f = float(value)
    if f <= 0:
        raise argparse.ArgumentTypeError(f"doit être > 0 (reçu {value})")
    return f


def main():
    parser = argparse.ArgumentParser(description="Chat interactif nanoPOPIXA v2")
    parser.add_argument("--checkpoint",  default="out-nanopopixa/checkpoint.pt")
    parser.add_argument("--max_tokens",  "--tokens", type=int,   default=200)
    parser.add_argument("--temperature", "--temp",   type=float, default=0.8)
    parser.add_argument("--top_k",       type=int,   default=40)
    parser.add_argument("--top_p",       type=float, default=None,
                        help="Nucleus sampling (ex: 0.9)")
    parser.add_argument("--penalty",     type=_positive_float, default=1.0,
                        help="Repetition penalty (1.0 = désactivé)")
    parser.add_argument("--effort",      default=None,
                        choices=list(EFFORT_PRESETS.keys()),
                        help="Preset effort : low|medium|high|max")
    args = parser.parse_args()

    # Effort preset écrase les params individuels si fourni
    max_tokens  = args.max_tokens
    temperature = args.temperature
    top_k       = args.top_k
    top_p       = args.top_p

    if args.effort:
        p           = EFFORT_PRESETS[args.effort]
        temperature = p["temperature"]
        top_k       = p["top_k"]
        top_p       = p["top_p"]
        max_tokens  = p["max_tokens"]

    _print_launch_banner()
    run_chat(args.checkpoint, max_tokens, temperature, top_k,
             repetition_penalty=args.penalty, top_p=top_p)


if __name__ == "__main__":
    main()
