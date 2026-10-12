"""
nanoPOPIXA — génération : KV-cache d'inférence, décodeur incrémental, échantillonnage

Découpage de la 2.3 « Socle » : model.py ne garde que l'architecture (POPIXAConfig,
RMSNorm, RoPE, attention, SwiGLU, Block, nanoPOPIXA.forward) ; tout le code qui
GÉNÈRE des tokens vit ici, copié tel quel depuis model.py :

  - _cache_len, KVCache        : KV-cache annoté (ids exacts des tokens couverts)
  - _Decoder, _store_cache     : décodage incrémental commun à tous les générateurs
                                 (fenêtre glissante, cache_ref rempli dans un finally)
  - GenerationMixin            : les méthodes de génération, héritées par le modèle via
                                 `class nanoPOPIXA(GenerationMixin, nn.Module)` —
                                 _apply_sampling, politiques d'arrêt, generate,
                                 generate_stream, thinking (normal / intercalé),
                                 speculative decoding, structured outputs

Le mixin ne suppose du modèle que `self.config.block_size` et l'appel
`self(idx, past_kvs=..., return_all_logits=...)` → (logits, present_kvs).

Ce module n'importe PAS model (model importe generation → import circulaire) ;
model.py ré-exporte GenerationMixin, KVCache, _Decoder, _cache_len et _store_cache :
`from model import KVCache` reste valable, et nanoPOPIXA hérite des méthodes
(`modele.generate(...)`, `nanoPOPIXA._apply_sampling`, `nanoPOPIXA._draft_ngram`…).
"""

import torch
from torch.nn import functional as F


# ─────────────────────────────────────────────────────────────────────────────
# KV-Cache annoté + décodeur incrémental
# ─────────────────────────────────────────────────────────────────────────────

def _cache_len(past_kvs) -> int:
    """Nombre de positions déjà présentes dans un KV-cache (0 si absent)."""
    return past_kvs[0][0].size(2) if past_kvs else 0


class KVCache(list):
    """
    KV-cache d'inférence : liste de (K, V) par couche, de forme (B, n_head, T, head_dim).

    Sous-classe de `list` → reste compatible avec tout code qui attend une simple
    liste de tuples (forward(past_kvs=...), save_session, CI).

    token_ids : ids exacts des tokens couverts par le cache (ligne 0 du batch), dans
                l'ordre — ou None s'ils sont inconnus (cache fourni sans ids).
                Invariant : len(token_ids) == seq_len.
    """

    def __init__(self, kvs=(), token_ids=None):
        super().__init__(kvs)
        self.token_ids = token_ids

    @property
    def seq_len(self) -> int:
        return _cache_len(self)


class _Decoder:
    """
    Décodage incrémental avec KV-cache — logique commune à tous les générateurs.

    Invariants :
      - idx      : contexte connu (B, T) — ids du cache initial (si connus) + entrée
                   + tokens générés
      - past_kvs : KV-cache couvrant idx, sauf les `pending` derniers tokens
                   (générés mais pas encore passés dans le modèle)
      - ctx_ids  : ids couverts par le cache (None si le préfixe est inconnu)

    Fenêtre glissante : quand le cache atteint block_size, on refait un prefill sur les
    `keep` derniers tokens connus (RoPE repart de 0) au lieu de planter.
    """

    def __init__(self, model, idx, initial_past_kvs=None, slide_keep: float = 0.75):
        self.model      = model
        self.block_size = model.config.block_size
        self.keep       = max(1, int(self.block_size * slide_keep))
        self.pending    = 0
        self.past_kvs   = None
        self.ctx_ids    = None

        if idx.size(1) == 0:
            known = getattr(initial_past_kvs, "token_ids", None)
            if known:
                # Entrée vide sur un cache connu : on rejoue le dernier token du cache
                # pour obtenir les logits du token suivant
                n = len(known) - 1
                initial_past_kvs = KVCache(
                    [(k[:, :, :n], v[:, :, :n]) for k, v in initial_past_kvs], list(known[:n])
                )
                idx = torch.tensor([[known[-1]]], dtype=torch.long, device=idx.device)
            else:
                # Génération libre : token 0 comme amorce (comme train.py)
                initial_past_kvs = None
                idx = torch.zeros((idx.size(0), 1), dtype=torch.long, device=idx.device)

        self.logits = self._prefill(idx, initial_past_kvs)

    @property
    def cache_len(self) -> int:
        return _cache_len(self.past_kvs)

    def _prefill(self, idx, past_kvs):
        # Ids du cache initial connus (KVCache) → ils font partie du contexte connu :
        # la fenêtre glissante, la repetition penalty et les drafts n-grammes les voient.
        prefix = getattr(past_kvs, "token_ids", None) if past_kvs is not None else None
        if prefix is not None and (idx.size(0) != 1 or len(prefix) != _cache_len(past_kvs)):
            prefix = None
        if prefix:
            pre = torch.tensor([list(prefix)], dtype=torch.long, device=idx.device)
            self.idx = torch.cat((pre, idx), dim=1)
        else:
            self.idx = idx

        if past_kvs is not None and _cache_len(past_kvs) + idx.size(1) <= self.block_size:
            logits, self.past_kvs = self.model(idx, past_kvs=past_kvs)
            self.ctx_ids = self.idx[0].tolist() if prefix is not None else None
            return logits

        # Pas de cache, ou cache + entrée > block_size → prefill sur le contexte connu,
        # tronqué aux block_size derniers tokens (ou `keep` en cas de débordement, pour
        # laisser de la place à la génération)
        full = self.idx
        ctx  = full[:, -self.block_size:] if full.size(1) <= self.block_size else full[:, -self.keep:]
        logits, self.past_kvs = self.model(ctx)
        self.ctx_ids = ctx[0].tolist()
        return logits

    def push(self, tok) -> None:
        """Ajoute un token (int ou tenseur (B, 1)) au contexte, en attente d'être passé au modèle."""
        if not torch.is_tensor(tok):
            tok = torch.full((self.idx.size(0), 1), int(tok), dtype=torch.long, device=self.idx.device)
        self.idx = torch.cat((self.idx, tok), dim=1)
        self.pending += 1

    def advance(self):
        """Passe les tokens en attente dans le modèle → logits du prochain token."""
        if self.pending == 0:
            return self.logits
        if self.cache_len + self.pending > self.block_size:
            # Fenêtre pleine → prefill glissant sur les derniers tokens connus
            ctx = self.idx[:, -self.keep:]
            self.logits, self.past_kvs = self.model(ctx)
            self.ctx_ids = ctx[0].tolist()
        else:
            new = self.idx[:, -self.pending:]
            self.logits, self.past_kvs = self.model(new, past_kvs=self.past_kvs)
            if self.ctx_ids is not None:
                self.ctx_ids.extend(new[0].tolist())
        self.pending = 0
        return self.logits

    def commit(self, base: int, kvs, drafts: list, n_accepted: int, next_tok: int) -> None:
        """
        Speculative decoding : garde dans le cache vérifié `kvs` les `n_accepted` premiers
        drafts (positions base..base+n_accepted), puis ajoute `next_tok` (en attente).
        """
        n = base + n_accepted
        self.past_kvs = [(k[:, :, :n], v[:, :, :n]) for k, v in kvs]
        if n_accepted:
            acc = torch.tensor([drafts[:n_accepted]], dtype=torch.long, device=self.idx.device)
            self.idx = torch.cat((self.idx, acc), dim=1)
            if self.ctx_ids is not None:
                self.ctx_ids.extend(drafts[:n_accepted])
        self.push(next_tok)

    def kv_cache(self) -> KVCache:
        """KV-cache final couvrant TOUT le contexte (les tokens en attente sont d'abord passés)."""
        self.advance()
        return KVCache(self.past_kvs, list(self.ctx_ids) if self.ctx_ids is not None else None)


def _store_cache(dec, cache_ref) -> None:
    """Remplit cache_ref avec le KV-cache final — y compris si le générateur est interrompu."""
    if cache_ref is None or dec is None:
        return
    cache_ref.clear()
    try:
        cache_ref.append(dec.kv_cache())
    except Exception:
        pass  # cache inutilisable → cache_ref reste vide, l'appelant reconstruit le contexte


# ─────────────────────────────────────────────────────────────────────────────
# Génération — méthodes héritées par nanoPOPIXA
# ─────────────────────────────────────────────────────────────────────────────

class GenerationMixin:
    """
    Méthodes de génération de nanoPOPIXA (sampling, arrêt anticipé, streaming, thinking,
    speculative decoding, structured outputs). Tous les générateurs passent par _Decoder.
    """

    # ── Sampling ─────────────────────────────────────────────────────────────

    def _apply_sampling(self, logits, temperature, top_k, top_p, repetition_penalty, idx,
                        logit_bias=None, mask=None):
        """
        Transforme les logits du dernier token en distribution de probabilités, dans l'ordre :
          1. Repetition penalty — divise si logit > 0, multiplie si logit ≤ 0
          2. Logit bias         — forçage / interdiction de tokens {id: biais}
          3. Masque             — tokens autorisés par une grammaire (structured outputs)
          4. Temperature        — temperature ≤ 0 → greedy (distribution one-hot sur l'argmax)
          5. Top-k
          6. Top-p (nucleus sampling)
        Ne modifie jamais le tenseur `logits` de l'appelant. Retourne (B, vocab_size).
        """
        if repetition_penalty is None or repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty doit être > 0 (reçu {repetition_penalty})")
        logits = logits[:, -1, :].float().clone()  # (B, vocab_size)

        # Repetition penalty — pénalise les tokens déjà présents dans le contexte (par ligne)
        if repetition_penalty != 1.0 and idx is not None and idx.numel() > 0:
            score = torch.gather(logits, 1, idx)
            score = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            logits.scatter_(1, idx, score)

        # Logit bias — appliqué AVANT top-k pour pouvoir forcer un token hors du top-k
        if logit_bias:
            ids = [t for t in logit_bias if 0 <= t < logits.size(-1)]
            if ids:
                bias = torch.tensor([float(logit_bias[t]) for t in ids], device=logits.device)
                logits[:, ids] += bias

        # Masque grammatical — les tokens interdits ne peuvent jamais être tirés
        if mask is not None:
            logits = logits.masked_fill(~mask.to(logits.device), float("-inf"))

        # Greedy — temperature ≤ 0 (évite la division par zéro)
        if temperature is None or temperature <= 0:
            probs = torch.zeros_like(logits)
            return probs.scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)

        logits = logits / temperature

        # Top-k — ne garde que les k meilleurs logits
        if top_k is not None and 0 < top_k < logits.size(-1):
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")

        # Top-p — nucleus sampling : garde le noyau minimal qui couvre p% de la proba
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Retire les tokens dont la proba cumulée dépasse top_p
            # (décalé d'un cran pour toujours garder au moins un token)
            to_remove = torch.zeros_like(sorted_logits, dtype=torch.bool)
            to_remove[:, 1:] = cum_probs[:, :-1] > top_p
            sorted_logits[to_remove] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)

        return F.softmax(logits, dim=-1)

    # ── Diminishing returns ──────────────────────────────────────────────────

    @staticmethod
    def _is_repetitive(tokens: list, window: int = 40, threshold: float = 0.28) -> bool:
        """
        Détecte si la génération tourne en rond (inspiré de tokenBudget.ts).
        Calcule la diversité des bigrammes sur les derniers `window` tokens.
        Retourne True si diversity < threshold (génération répétitive).

        Claude utilise DIMINISHING_THRESHOLD=500 tokens delta sur 3 itérations.
        Ici on adapte en diversité de bigrammes — plus granulaire pour les petits modèles.
        """
        if len(tokens) < window:
            return False
        w = tokens[-window:]
        bigrams   = [(w[i], w[i + 1]) for i in range(len(w) - 1)]
        diversity = len(set(bigrams)) / len(bigrams)
        return diversity < threshold

    @staticmethod
    def _has_diminishing_returns(
        tokens: list,
        window: int = 40,
        n_checks: int = 3,
        threshold: float = 0.28,
    ) -> bool:
        """
        Vérifie si N fenêtres consécutives montrent toutes une diversité faible.
        Fidèle à tokenBudget.ts : DIMINISHING_THRESHOLD sur 3 itérations consécutives.

        Différence avec _is_repetitive :
          _is_repetitive  → détecte UNE fenêtre répétitive (arrêt immédiat)
          _has_diminishing_returns → requiert N fenêtres consécutives (plus conservateur)

        Paramètres :
          n_checks  : nombre de fenêtres consécutives à vérifier (défaut 3, comme Claude)
          window    : taille de chaque fenêtre en tokens
          threshold : seuil de diversité bigrammes (0.28 = 28%)
        """
        if len(tokens) < window * n_checks:
            return False
        for i in range(n_checks):
            start = len(tokens) - window * (i + 1)
            end   = len(tokens) - window * i if i > 0 else len(tokens)
            w = tokens[start:end]
            if len(w) < 2:
                return False
            bigrams = [(w[j], w[j + 1]) for j in range(len(w) - 1)]
            if len(set(bigrams)) / len(bigrams) >= threshold:
                return False  # une fenêtre diverse → pas encore diminishing
        return True  # toutes les n_checks fenêtres sont répétitives

    # Politiques d'arrêt anticipé :
    #   "repetitive"  → _is_repetitive          (1 fenêtre — arrêt immédiat)
    #   "diminishing" → _has_diminishing_returns (3 fenêtres consécutives — conservateur)
    #   None / "off"  → jamais
    STOP_POLICIES = ("repetitive", "diminishing", "off")

    @staticmethod
    def _resolve_stop_policy(stop_policy=None, stop_on_repetition: bool = False):
        """Normalise la politique d'arrêt (compat : stop_on_repetition=True → 'repetitive')."""
        if stop_policy is None:
            return "repetitive" if stop_on_repetition else None
        if stop_policy == "off":
            return None
        if stop_policy not in ("repetitive", "diminishing"):
            raise ValueError(f"stop_policy inconnue : {stop_policy!r} (repetitive|diminishing|off)")
        return stop_policy

    @classmethod
    def _should_stop(cls, tokens: list, stop_policy) -> bool:
        if stop_policy == "repetitive":
            return cls._is_repetitive(tokens)
        if stop_policy == "diminishing":
            return cls._has_diminishing_returns(tokens)
        return False

    # ── Boucle de décodage commune ───────────────────────────────────────────

    def _sample_stream(self, dec, n_tokens, temperature, top_k, top_p, repetition_penalty,
                       stop_policy=None, logit_bias=None, history=None):
        """
        Échantillonne jusqu'à n_tokens tokens via le décodeur `dec`.
        Yield le tenseur (B, 1) de chaque token. S'arrête si la politique d'arrêt se
        déclenche (le token déclencheur est écarté : ni yieldé, ni ajouté au contexte).
        history : liste partagée des tokens déjà émis (pour la politique d'arrêt).
        """
        generated = [] if history is None else history
        for _ in range(max(0, n_tokens)):
            logits   = dec.advance()
            probs    = self._apply_sampling(logits, temperature, top_k, top_p, repetition_penalty,
                                            dec.idx, logit_bias=logit_bias)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated.append(idx_next[0, 0].item())
            if self._should_stop(generated, stop_policy):
                return  # Diminishing returns détecté → fin de la phase
            dec.push(idx_next)
            yield idx_next

    # ── Génération standard ──────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 repetition_penalty=1.0, top_p=None, stop_on_repetition=False,
                 stop_policy=None, logit_bias=None):
        """
        Génération avec KV-Cache :
          1. Prefill  — traite tout le prompt en une passe, construit le cache
          2. Decode   — génère un token à la fois en O(1) grâce au cache
        Fenêtre glissante automatique au-delà de block_size.

        stop_on_repetition : arrêt anticipé si la génération diverge (= stop_policy="repetitive")
        stop_policy        : "repetitive" | "diminishing" | "off"
        logit_bias         : {token_id: biais} ajouté aux logits avant top-k
        Retourne idx (B, T + n_générés).
        """
        policy = self._resolve_stop_policy(stop_policy, stop_on_repetition)
        dec = _Decoder(self, idx)
        for _ in self._sample_stream(dec, max_new_tokens, temperature, top_k, top_p,
                                     repetition_penalty, policy, logit_bias):
            pass
        return dec.idx

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                        repetition_penalty=1.0, top_p=None, stop_on_repetition=False,
                        initial_past_kvs=None, cache_ref=None,
                        stop_policy=None, logit_bias=None):
        """
        Streaming token par token avec KV-Cache.
        Yield chaque token généré (int).

        initial_past_kvs : KV-cache existant (session persistante) — si fourni,
                           seul `idx` (nouveaux tokens) est traité en prefill,
                           le reste est récupéré depuis le cache. O(N_new) au lieu de O(N_total).
                           Si cache + idx dépasse block_size, le cache est abandonné.
        cache_ref        : liste mutable — contiendra [KVCache] couvrant tout le contexte
                           (entrée + tokens yieldés) à la fin du générateur, y compris en
                           cas d'interruption (close / exception). KVCache.token_ids donne
                           les ids exacts couverts par le cache.
        stop_on_repetition / stop_policy : arrêt automatique si la génération diverge.
        """
        policy = self._resolve_stop_policy(stop_policy, stop_on_repetition)
        dec = None
        try:
            dec = _Decoder(self, idx, initial_past_kvs)
            for t in self._sample_stream(dec, max_new_tokens, temperature, top_k, top_p,
                                         repetition_penalty, policy, logit_bias):
                yield t[0, 0].item()
        finally:
            # Stocker le cache final pour persistance inter-sessions
            _store_cache(dec, cache_ref)

    # ── Thinking blocks (Claude-inspired) ───────────────────────────────────

    @staticmethod
    def adaptive_think_budget(prompt_len: int) -> int:
        """
        Budget de thinking adaptatif selon la complexité estimée du prompt.
        Inspiré de l'adaptive thinking de Claude 4.6+ (pas de budget fixe — le modèle
        décide lui-même). Ici on estime via la longueur du prompt.
        """
        if prompt_len < 50:
            return 50
        elif prompt_len < 150:
            return 150
        elif prompt_len < 400:
            return 300
        else:
            return 500

    @torch.no_grad()
    def generate_stream_with_thinking(
        self, idx,
        think_budget: int = 150,
        response_budget: int = 300,
        temperature: float = 0.8,
        top_k: int = None,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        initial_past_kvs=None,
        cache_ref=None,
        think_stop_policy: str = "diminishing",
        stop_policy: str = "repetitive",
        logit_bias=None,
    ):
        """
        Génération deux phases inspirée de Claude's extended thinking.

        Phase 1 — Think (temperature=1 forcée, comme Claude) :
            Le modèle génère un raisonnement interne libre.
            Arrêt anticipé via think_stop_policy (défaut "diminishing" : 3 fenêtres
            répétitives consécutives — le thinking a le droit d'explorer plus longtemps).

        Phase 2 — Response (temperature normale) :
            Le modèle génère la réponse finale en ayant "vu" son propre raisonnement.
            Arrêt anticipé via stop_policy (défaut "repetitive").

        Yield : tuples (phase, token_int)
            phase = 'think' | 'response'
        """
        think_policy = self._resolve_stop_policy(think_stop_policy)
        resp_policy  = self._resolve_stop_policy(stop_policy)
        dec = None
        try:
            # Prefill — avec cache existant si session restaurée
            dec = _Decoder(self, idx, initial_past_kvs)

            # Phase 1 : Thinking — temperature=1 (contrainte API Claude, claude.ts:1598)
            for t in self._sample_stream(dec, think_budget, 1.0, top_k, top_p, repetition_penalty,
                                         think_policy, logit_bias):
                yield ("think", t[0, 0].item())

            # Phase 2 : Response — temperature normale
            for t in self._sample_stream(dec, response_budget, temperature, top_k, top_p,
                                         repetition_penalty, resp_policy, logit_bias):
                yield ("response", t[0, 0].item())
        finally:
            _store_cache(dec, cache_ref)

    # ── Interleaved Thinking (Claude-inspired) ───────────────────────────────

    @torch.no_grad()
    def generate_stream_with_interleaved_thinking(
        self, idx,
        response_budget: int = 300,
        think_per_interleave: int = 20,
        interleave_every: int = 50,
        temperature: float = 0.8,
        top_k: int = None,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        initial_past_kvs=None,
        cache_ref=None,
        stop_policy: str = "repetitive",
        logit_bias=None,
    ):
        """
        Thinking intercalé — inspiré du beta `interleaved-thinking-2025-05-14`.

        Alterne génération normale et mini-pauses de réflexion :
          1. Génère `interleave_every` tokens de réponse
          2. Pause : génère `think_per_interleave` tokens de thinking (temp=1)
          3. Reprend la réponse — en boucle jusqu'à response_budget

        Avantage vs thinking pur : la réflexion est distribuée tout au long
        de la réponse, permettant des corrections en cours de route.
        La politique d'arrêt ne regarde que les tokens de réponse.

        Yield : tuples (phase, token_int)  où phase = 'think' | 'response'
        """
        policy = self._resolve_stop_policy(stop_policy)
        every  = max(1, interleave_every)
        dec = None
        try:
            dec = _Decoder(self, idx, initial_past_kvs)
            generated_resp = []
            produced = 0
            while produced < response_budget:
                # ── Segment de réponse ──────────────────────────────────────
                n = min(every, response_budget - produced)
                count = 0
                for t in self._sample_stream(dec, n, temperature, top_k, top_p, repetition_penalty,
                                             policy, logit_bias, history=generated_resp):
                    count += 1
                    yield ("response", t[0, 0].item())
                produced += count
                if count < n or produced >= response_budget:
                    break  # arrêt anticipé ou budget atteint

                # ── Mini-pause thinking (temperature=1) ─────────────────────
                for t in self._sample_stream(dec, think_per_interleave, 1.0, top_k, top_p,
                                             repetition_penalty, None, logit_bias):
                    yield ("think", t[0, 0].item())
        finally:
            _store_cache(dec, cache_ref)

    # ── Speculative Decoding (fast mode) ─────────────────────────────────────

    @staticmethod
    def _draft_ngram(ids: list, n_draft: int, max_ngram: int = 3) -> list:
        """
        Prompt lookup decoding : cherche l'occurrence la plus récente du suffixe courant
        (n-gramme de max_ngram → 1 tokens) dans le contexte et propose les tokens qui la
        suivaient. Draft gratuit (aucune passe du modèle) — efficace sur texte/code répétitif.
        """
        L = len(ids)
        for n in range(min(max_ngram, L - 1), 0, -1):
            suffix = ids[L - n:]
            for start in range(L - n - 1, -1, -1):
                if ids[start:start + n] == suffix:
                    cont = ids[start + n:start + n + n_draft]
                    if cont:
                        return cont
        return []

    def _draft_self(self, dec, k, top_k, top_p, repetition_penalty, logit_bias,
                    draft_temperature: float = 0.05):
        """
        Draft avec le MÊME modèle à température quasi nulle (quasi greedy).
        Retourne (drafts, distributions q) — q sert au test accept/reject.
        Le cache de travail est jeté : seul le cache vérifié est conservé.
        """
        drafts, qs = [], []
        logits, kvs, ctx = dec.logits, dec.past_kvs, dec.idx
        for i in range(k):
            q = self._apply_sampling(logits, draft_temperature, top_k, top_p, repetition_penalty,
                                     ctx, logit_bias=logit_bias)[0]
            d = torch.multinomial(q, num_samples=1).item()
            drafts.append(d)
            qs.append(q)
            if i < k - 1:
                t = torch.tensor([[d]], dtype=torch.long, device=ctx.device)
                logits, kvs = self(t, past_kvs=kvs)
                ctx = torch.cat((ctx, t), dim=1)
        return drafts, qs

    @torch.no_grad()
    def speculative_generate_stream(
        self, idx,
        max_new_tokens: int = 200,
        n_draft: int = 4,
        temperature: float = 0.8,
        top_k: int = None,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        initial_past_kvs=None,
        cache_ref=None,
        draft: str = "ngram",
        stop_policy=None,
        logit_bias=None,
    ):
        """
        Speculative decoding — inspiré du beta `fast-mode-2026-02-01`.

        Principe :
          1. Draft  — propose k tokens :
               · "ngram" (défaut) : prompt lookup — recopie la suite d'un n-gramme déjà vu
                 dans le contexte. Gratuit → vrai gain de vitesse sur texte répétitif.
               · "self" : le MÊME modèle à temp≈0 (pédagogique ; pas plus rapide sans
                 modèle draft plus léger).
          2. Verify — UNE passe du modèle sur les k drafts (masque causal décalé) →
             distributions cibles p_0..p_k pour chaque position.
          3. Accept/Reject (Leviathan et al., 2023) — draft d_i accepté avec proba
             min(1, p_i(d_i) / q_i(d_i)) ; sinon on ré-échantillonne depuis
             max(0, p_i − q_i) normalisé et on s'arrête. Distribution de sortie
             IDENTIQUE à l'échantillonnage normal.
          4. Bonus — si les k drafts sont acceptés, un token de plus gratuit (p_k).
          Le KV-cache est tronqué aux tokens acceptés (les drafts rejetés en sortent).

        Yield : int (token généré)
        """
        if draft not in ("ngram", "self"):
            raise ValueError(f"draft inconnu : {draft!r} (ngram|self)")
        policy = self._resolve_stop_policy(stop_policy)
        dec = None
        try:
            dec = _Decoder(self, idx, initial_past_kvs)
            device = dec.idx.device

            def target(step_logits, ctx):
                return self._apply_sampling(step_logits, temperature, top_k, top_p,
                                            repetition_penalty, ctx, logit_bias=logit_bias)[0]

            history  = []
            produced = 0
            while produced < max_new_tokens:
                logits = dec.advance()  # plus aucun token en attente
                room   = self.config.block_size - dec.cache_len
                k      = min(n_draft, max_new_tokens - produced - 1, room)

                # ── Phase Draft ───────────────────────────────────────────────
                drafts, q_probs = [], None
                if k > 0:
                    if draft == "self":
                        drafts, q_probs = self._draft_self(dec, k, top_k, top_p,
                                                           repetition_penalty, logit_bias)
                    else:
                        drafts = self._draft_ngram(dec.idx[0].tolist(), k)

                if not drafts:
                    # Aucun draft → pas de décodage classique
                    p   = target(logits, dec.idx)
                    tok = torch.multinomial(p, num_samples=1).item()
                    burst = [tok]
                    dec.push(tok)
                else:
                    # ── Phase Verify : une passe sur tous les drafts ──────────
                    base = dec.cache_len
                    d_tensor = torch.tensor([drafts], dtype=torch.long, device=device)
                    v_logits, v_kvs = self(d_tensor, past_kvs=dec.past_kvs, return_all_logits=True)

                    # ── Accept/Reject ─────────────────────────────────────────
                    ctx   = dec.idx
                    burst = []
                    n_acc = 0
                    for i, d in enumerate(drafts):
                        # p_0 = logits courants ; p_i = sortie du verifier à la position i-1
                        p   = target(logits if i == 0 else v_logits[:, i - 1:i, :], ctx)
                        q_d = q_probs[i][d].item() if q_probs is not None else 1.0
                        if torch.rand(1).item() < min(1.0, p[d].item() / max(q_d, 1e-12)):
                            burst.append(d)
                            n_acc += 1
                            ctx = torch.cat((ctx, d_tensor[:, i:i + 1]), dim=1)
                            continue
                        # Rejet — ré-échantillonne depuis max(0, p − q) normalisé
                        if q_probs is not None:
                            residual = torch.clamp(p - q_probs[i], min=0)
                        else:
                            residual = p.clone()
                            residual[d] = 0.0  # q one-hot (draft n-gramme)
                        s = residual.sum()
                        dist = residual / s if s > 0 else p
                        burst.append(torch.multinomial(dist, num_samples=1).item())
                        break
                    else:
                        # ── Bonus token : tous les drafts acceptés ───────────
                        p = target(v_logits[:, -1:, :], ctx)
                        burst.append(torch.multinomial(p, num_samples=1).item())

                    # Cache = contexte + drafts acceptés ; dernier token en attente
                    dec.commit(base, v_kvs, drafts, n_acc, burst[-1])

                for tok in burst:
                    yield tok
                produced += len(burst)
                history.extend(burst)
                if self._should_stop(history, policy):
                    break
        finally:
            _store_cache(dec, cache_ref)

    # ── Structured outputs (JSON / JSON Schema) ──────────────────────────────

    @torch.no_grad()
    def generate_structured(
        self, idx, constraint,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = 1.0,
        initial_past_kvs=None,
        cache_ref=None,
        force_complete: bool = True,
    ):
        """
        Structured outputs — génération contrainte par une grammaire JSON / JSON Schema
        (inspiré du beta `structured-outputs-2025-12-15`).

        constraint : structured.TokenConstraint — avancé en place, token par token.

        Échantillonnage par rejet : on tire d'abord dans la distribution normale ; si le
        token viole la grammaire, on retire avec le masque des tokens autorisés (exact
        sans top-k/top-p). Si le JSON est déjà complet et que le modèle sort de la
        grammaire → fin naturelle.

        force_complete : à chaque pas, la complétion la plus courte doit tenir dans le
        budget restant — un token qui la rendrait trop longue est banni puis re-tiré, et
        quand le budget est juste suffisant on ferme le JSON. Sortie toujours complète
        si max_new_tokens >= len(constraint.completion_tokens()) au départ.

        Yield : int (token généré). À la fin, constraint.is_complete() indique si le JSON
        est complet.
        """
        # Seule la LONGUEUR de la fermeture sert au contrôle de budget : on ne garde que des
        # entiers, dans un cache borné (sinon mémoire Θ(n²) quand le modèle imbrique sans fin)
        closing_len_cache = {}

        def closing_len(c):
            if c.state not in closing_len_cache:
                if len(closing_len_cache) > 512:
                    closing_len_cache.clear()
                toks = c.completion_tokens()
                closing_len_cache[c.state] = None if toks is None else len(toks)
            return closing_len_cache[c.state]

        def pad_mask(mask, size):
            # Vocabulaire du modèle rembourré (ex. 50304) : les ids hors contrainte sont interdits
            if mask.numel() >= size:
                return mask[:size]
            return torch.cat((mask, torch.zeros(size - mask.numel(), dtype=torch.bool)))

        dec = None
        try:
            dec = _Decoder(self, idx, initial_past_kvs)

            def emit(tokens):
                nonlocal produced
                for t in tokens:
                    constraint.advance(t)
                    dec.push(t)
                    produced += 1
                    yield t

            produced = 0
            while produced < max_new_tokens and not constraint.is_terminal():
                remaining = max_new_tokens - produced
                # Invariant (force_complete) : la fermeture la plus courte tient toujours
                # dans le budget restant → plus de marge du tout : on ferme maintenant
                current = closing_len(constraint) if force_complete else None
                if current is not None and current >= remaining:
                    yield from emit((constraint.completion_tokens() or [])[:remaining])
                    break

                logits = dec.advance()
                probs  = self._apply_sampling(logits, temperature, top_k, top_p,
                                              repetition_penalty, dec.idx)
                tok = torch.multinomial(probs, num_samples=1)[0, 0].item()
                if not constraint.is_allowed(tok):
                    if constraint.is_complete():
                        break  # JSON complet et le modèle veut s'arrêter
                    tok = None

                # Tirage masqué si besoin ; un token qui rendrait la fermeture impossible
                # avec le budget restant est banni puis on re-tire (jamais de JSON tronqué)
                banned = []
                for _ in range(16):
                    if tok is None:
                        mask = pad_mask(constraint.allowed_mask(), logits.size(-1)).clone()
                        if banned:
                            mask[banned] = False
                        if not bool(mask.any()):
                            break  # plus aucun token possible
                        probs = self._apply_sampling(logits, temperature, top_k, top_p,
                                                     repetition_penalty, dec.idx, mask=mask)
                        tok = torch.multinomial(probs, num_samples=1)[0, 0].item()
                    if current is None:
                        break  # fermeture inexprimable avec ce vocabulaire : meilleur effort
                    nxt = constraint.clone()
                    nxt.advance(tok)
                    after = closing_len(nxt)
                    if after is not None and after <= remaining - 1:
                        break  # token abordable
                    banned.append(tok)
                    tok = None

                if tok is None:
                    if current is not None:
                        yield from emit((constraint.completion_tokens() or [])[:remaining])
                    break
                yield from emit([tok])
        finally:
            _store_cache(dec, cache_ref)
