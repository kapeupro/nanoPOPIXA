"""Tests du modèle : attention + KV-cache, fenêtre glissante, sampling, générateurs, speculative decoding."""

import collections
import os
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import model as model_mod
from conftest import ROOT, make_model
from model import KVCache, POPIXAConfig, RMSNorm, checkpoint_v1_error, nanoPOPIXA


def _set_flash(model, flash: bool):
    for blk in model.transformer.h:
        blk.attn.flash = flash
        if not flash and not hasattr(blk.attn, "bias"):
            n = model.config.block_size
            blk.attn.register_buffer("bias", torch.tril(torch.ones(n, n)).view(1, 1, n, n))


# ── Forward / attention ──────────────────────────────────────────────────────

def test_forward_training_contract(model):
    idx = torch.randint(0, 65, (2, 8))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 8, 65)
    assert loss.ndim == 0 and torch.isfinite(loss)


@pytest.mark.parametrize("flash", [True, False])
def test_chunked_prefill_matches_full_forward(model, flash):
    """Prefill par morceaux sur un cache (session restaurée, verify spéculatif) == forward complet."""
    _set_flash(model, flash)
    x = torch.randint(0, 65, (1, 12))
    with torch.no_grad():
        full, _ = model(x, return_all_logits=True)
        _, kv = model(x[:, :5])
        chunk, _ = model(x[:, 5:], past_kvs=kv, return_all_logits=True)
    assert torch.allclose(full[:, 5:], chunk, atol=1e-5)


def test_forward_rejects_overflow_with_clear_message(model):
    with torch.no_grad():
        _, kv = model(torch.zeros((1, 30), dtype=torch.long))
        with pytest.raises(AssertionError, match="block_size"):
            model(torch.zeros((1, 5), dtype=torch.long), past_kvs=kv)


def test_rmsnorm_fp16_does_not_collapse():
    x = torch.ones(1, 1, 8, dtype=torch.half)
    x[0, 0, 0] = 300
    out = RMSNorm(8).half()(x)
    assert torch.isfinite(out).all() and out.abs().sum() > 0


def test_odd_head_dim_is_rejected():
    from model import POPIXAConfig
    with pytest.raises(AssertionError, match="head_dim"):
        nanoPOPIXA(POPIXAConfig(vocab_size=10, n_embd=66, n_head=2, block_size=8))


def test_checkpoint_v1_detection(model):
    assert checkpoint_v1_error(model.state_dict()) is None
    v1 = {"transformer.wpe.weight": torch.zeros(1), "transformer.h.0.mlp.c_fc.weight": torch.zeros(1)}
    assert "v1" in checkpoint_v1_error(v1)


# ── Sampling ─────────────────────────────────────────────────────────────────

def test_apply_sampling_does_not_mutate_and_handles_edge_cases(model):
    logits = torch.randn(1, 1, 65)
    before = logits.clone()
    idx = torch.randint(0, 65, (1, 10))
    probs = model._apply_sampling(logits, 0.8, 0, 0.9, 1.3, idx)          # top_k=0 → désactivé
    assert torch.equal(logits, before)
    assert torch.isclose(probs.sum(), torch.tensor(1.0))
    greedy = model._apply_sampling(logits, 0, None, None, 1.0, idx)       # temperature 0 → greedy
    assert greedy[0].max() == 1.0 and greedy[0].argmax() == logits[0, -1].argmax()
    with pytest.raises(ValueError):
        model._apply_sampling(logits, 1.0, None, None, 0.0, idx)


def test_repetition_penalty_is_per_row(model):
    logits = torch.randn(1, 1, 65).repeat(3, 1, 1)
    idx = torch.randint(0, 65, (1, 6)).repeat(3, 1)
    probs = model._apply_sampling(logits, 1.0, None, None, 1.5, idx)
    assert torch.allclose(probs[0], probs[1]) and torch.allclose(probs[0], probs[2])


def test_logit_bias_can_force_token_outside_top_k(model):
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), 10, top_k=1, logit_bias={7: 1000.0})
    assert out[0, 1:].tolist() == [7] * 10


# ── Générateurs + KV-cache ───────────────────────────────────────────────────

def test_generate_survives_block_size_overflow(model):
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)
    assert out.shape == (1, 101)
    long_prompt = torch.randint(0, 65, (1, 50))                # prompt > block_size
    assert model.generate(long_prompt, 5).shape == (1, 55)


def test_empty_prompt_and_zero_budgets(model):
    empty = torch.zeros((1, 0), dtype=torch.long)
    assert len(list(model.generate_stream(empty, 5))) == 5
    assert list(model.generate_stream(torch.zeros((1, 1), dtype=torch.long), 0)) == []
    phases = list(model.generate_stream_with_thinking(torch.zeros((1, 1), dtype=torch.long),
                                                     think_budget=0, response_budget=3))
    assert [p for p, _ in phases] == ["response"] * 3


def test_stream_cache_covers_input_and_all_yielded_tokens(model):
    idx = torch.tensor([[1, 2, 3, 4]])
    cache_ref = []
    toks = list(model.generate_stream(idx, 6, temperature=0, cache_ref=cache_ref))
    kv = cache_ref[0]
    assert isinstance(kv, KVCache) and isinstance(kv, list)
    assert kv.token_ids == [1, 2, 3, 4] + toks and kv.seq_len == 10

    # Continuer depuis le cache == tout recalculer
    nxt = list(model.generate_stream(torch.tensor([[5, 6]]), 5, temperature=0, initial_past_kvs=kv))
    ref = model.generate(torch.tensor([[1, 2, 3, 4] + toks + [5, 6]]), 5, temperature=0)[0, -5:].tolist()
    assert nxt == ref


def test_cache_ref_filled_when_consumer_stops_early(model):
    cache_ref = []
    gen = model.generate_stream(torch.tensor([[1, 2]]), 10, temperature=0, cache_ref=cache_ref)
    got = [next(gen), next(gen)]
    gen.close()
    assert cache_ref[0].token_ids == [1, 2] + got


def test_cache_overflow_slides_with_known_prefix(model):
    cache_ref = []
    list(model.generate_stream(torch.randint(0, 65, (1, 20)), 5, cache_ref=cache_ref))
    kv = cache_ref[0]
    toks = list(model.generate_stream(torch.randint(0, 65, (1, 6)), 20, initial_past_kvs=kv,
                                      cache_ref=cache_ref))
    kv2 = cache_ref[0]
    assert len(toks) == 20
    assert kv2.seq_len == len(kv2.token_ids) <= model.config.block_size
    assert kv2.token_ids[-len(toks):] == toks


@pytest.mark.parametrize("policy,expected", [("repetitive", 39), ("diminishing", 119), ("off", 150)])
def test_stop_policies(policy, expected):
    model = make_model(block_size=256)
    toks = list(model.generate_stream(torch.zeros((1, 1), dtype=torch.long), 150,
                                      stop_policy=policy, logit_bias={5: 1000.0}))
    assert len(toks) == expected


def test_stop_policy_validation(model):
    assert nanoPOPIXA._resolve_stop_policy(None, True) == "repetitive"
    assert nanoPOPIXA._resolve_stop_policy("off", True) is None
    with pytest.raises(ValueError):
        nanoPOPIXA._resolve_stop_policy("bogus")


def test_thinking_uses_diminishing_returns_by_default():
    model = make_model(block_size=512)
    phases = list(model.generate_stream_with_thinking(
        torch.zeros((1, 1), dtype=torch.long), think_budget=200, response_budget=60,
        logit_bias={5: 1000.0}))
    kinds = [p for p, _ in phases]
    assert kinds.count("think") == 119      # 3 fenêtres répétitives (tokenBudget.ts)
    assert kinds.count("response") == 39     # 1 fenêtre répétitive


def test_interleaved_pattern(model):
    phases = list(model.generate_stream_with_interleaved_thinking(
        torch.zeros((1, 1), dtype=torch.long), response_budget=12,
        think_per_interleave=2, interleave_every=5))
    assert "".join("T" if p == "think" else "r" for p, _ in phases) == "rrrrrTTrrrrrTTrr"


# ── Speculative decoding ─────────────────────────────────────────────────────

@pytest.mark.parametrize("draft", ["ngram", "self"])
def test_speculative_greedy_matches_generate_and_cache_is_exact(model, draft):
    idx = torch.tensor([[1, 2, 3, 1, 2, 3, 1, 2]])
    ref = model.generate(idx, 20, temperature=0)[0, idx.size(1):].tolist()
    cache_ref = []
    spec = list(model.speculative_generate_stream(idx, 20, n_draft=4, temperature=0,
                                                  draft=draft, cache_ref=cache_ref))
    assert spec == ref
    kv = cache_ref[0]
    seq = idx[0].tolist() + spec
    assert kv.token_ids == seq and kv.seq_len == len(seq)
    with torch.no_grad():
        _, ref_kv = model(torch.tensor([seq]))
    for (k, v), (rk, rv) in zip(kv, ref_kv):
        assert torch.allclose(k, rk, atol=1e-5) and torch.allclose(v, rv, atol=1e-5)


def test_speculative_long_run_and_edge_cases(model):
    idx = torch.tensor([[1, 2, 3, 1, 2]])
    cache_ref = []
    toks = list(model.speculative_generate_stream(idx, 80, n_draft=4, cache_ref=cache_ref))
    assert len(toks) == 80 and cache_ref[0].seq_len <= model.config.block_size
    assert len(list(model.speculative_generate_stream(idx, 5, n_draft=0))) == 5
    assert list(model.speculative_generate_stream(idx, 0)) == []
    with pytest.raises(ValueError):
        list(model.speculative_generate_stream(idx, 5, draft="bogus"))


def test_speculative_preserves_sampling_distribution():
    """Leviathan et al. : la distribution de sortie doit être celle de l'échantillonnage normal."""
    torch.set_num_threads(1)
    model = make_model(vocab_size=4, n_layer=1, n_embd=32)
    with torch.no_grad():
        model.lm_head.weight.mul_(6)
    idx = torch.tensor([[0, 1, 2, 0, 1]])
    n = 800

    def dist(fn, seed):
        torch.manual_seed(seed)
        return collections.Counter(tuple(fn()) for _ in range(n))

    ref = dist(lambda: model.generate(idx, 3, temperature=1.0)[0, -3:].tolist(), 1)
    for k, draft in enumerate(("ngram", "self")):
        got = dist(lambda: list(model.speculative_generate_stream(
            idx, 3, n_draft=2, temperature=1.0, top_p=None, draft=draft)), 10 + k)
        keys = set(ref) | set(got)
        tv = 0.5 * sum(abs(ref[x] - got[x]) for x in keys) / n
        assert tv < 0.08, (draft, tv)


def test_draft_ngram():
    assert nanoPOPIXA._draft_ngram([1, 2, 3, 9, 1, 2, 3], 3) == [9, 1, 2]
    assert nanoPOPIXA._draft_ngram([1, 2, 3], 3) == []


class _OpenConstraint:
    """Contrainte factice : tout token est permis, jamais complète, fermeture inconnue."""

    def __init__(self, vocab_size):
        self.state, self.vocab_size = 0, vocab_size

    def is_terminal(self):
        return False

    def is_complete(self):
        return False

    def is_allowed(self, tok):
        return True

    def allowed_mask(self):
        return torch.ones(self.vocab_size, dtype=torch.bool)

    def advance(self, tok):
        self.state += 1

    def clone(self):
        c = _OpenConstraint(self.vocab_size)
        c.state = self.state
        return c

    def completion_tokens(self):
        return None


def test_structured_keeps_sampling_when_closing_is_unknown(model):
    toks = list(model.generate_structured(torch.tensor([[1]]), _OpenConstraint(65), max_new_tokens=50))
    assert len(toks) == 50     # pas d'arrêt prématuré quand la fermeture est inexprimable


# ── Découpage model.py / generation.py (2.3) ────────────────────────────────

def test_generation_split_keeps_model_api():
    import generation
    assert issubclass(nanoPOPIXA, generation.GenerationMixin)
    # Ré-exports : `from model import KVCache, _Decoder…` (chat, session_cache, tests) reste valable
    for name in ("GenerationMixin", "KVCache", "_Decoder", "_cache_len", "_store_cache"):
        assert getattr(model_mod, name) is getattr(generation, name)
    for meth in ("_apply_sampling", "_is_repetitive", "_has_diminishing_returns", "_should_stop",
                 "_sample_stream", "generate", "generate_stream", "adaptive_think_budget",
                 "generate_stream_with_thinking", "generate_stream_with_interleaved_thinking",
                 "_draft_ngram", "_draft_self", "speculative_generate_stream", "generate_structured"):
        assert getattr(nanoPOPIXA, meth).__module__ == "generation", meth
    assert nanoPOPIXA.STOP_POLICIES == ("repetitive", "diminishing", "off")
    # Les checkpoints existants contiennent un config picklé sous `model.POPIXAConfig`
    assert POPIXAConfig.__module__ == "model"
    cfg = pickle.loads(pickle.dumps(POPIXAConfig(vocab_size=10)))
    assert type(cfg) is POPIXAConfig and cfg.vocab_size == 10


def test_generation_does_not_import_model():
    """generation.py n'importe pas model au niveau module (model l'importe → import circulaire)."""
    code = "import sys, generation; assert 'model' not in sys.modules, sorted(sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=ROOT), timeout=300)
    assert r.returncode == 0, r.stderr[-800:]


# ── Dropout d'attention (2.3) ────────────────────────────────────────────────

def _tiny_config(**kw):
    return POPIXAConfig(**dict(dict(block_size=16, vocab_size=65, n_layer=2, n_head=2, n_embd=32), **kw))


def _spy_sdpa_dropout(monkeypatch):
    """Espionne F.scaled_dot_product_attention tel que model.py l'appelle → liste des dropout_p."""
    seen, real = [], model_mod.F.scaled_dot_product_attention

    def spy(*args, **kwargs):
        seen.append(kwargs.get("dropout_p", args[4] if len(args) > 4 else 0.0))
        return real(*args, **kwargs)

    monkeypatch.setattr(model_mod.F, "scaled_dot_product_attention", spy)
    return seen


def test_attn_dropout_off_by_default_keeps_fast_sdpa(monkeypatch):
    """dropout=0.1 (train.py) ne force plus le noyau SDPA « math » lent : dropout_p=0 en train."""
    cfg = _tiny_config(dropout=0.1)
    assert cfg.attn_dropout == 0.0
    m = nanoPOPIXA(cfg)
    m.train()
    assert all(blk.attn.flash for blk in m.transformer.h)
    seen = _spy_sdpa_dropout(monkeypatch)
    idx = torch.randint(0, 65, (2, 16))
    _, loss = m(idx, idx)
    loss.backward()
    assert seen == [0.0, 0.0]                                   # une fois par couche
    attn = m.transformer.h[0].attn
    assert attn.attn_dropout.p == 0.0                           # chemin manuel aussi
    # Le dropout résiduel, MLP et embeddings garde config.dropout
    assert attn.resid_dropout.p == 0.1
    assert m.transformer.h[0].mlp.dropout.p == 0.1 and m.transformer.drop.p == 0.1


def test_attn_dropout_applies_in_training_only(monkeypatch):
    m = nanoPOPIXA(_tiny_config(dropout=0.1, attn_dropout=0.2))
    assert m.transformer.h[0].attn.attn_dropout.p == 0.2        # chemin manuel : nn.Dropout(0.2)
    seen = _spy_sdpa_dropout(monkeypatch)
    idx = torch.randint(0, 65, (1, 8))
    m.train()
    m(idx, idx)
    assert seen == [0.2, 0.2]
    seen.clear()
    m.train(False)
    with torch.no_grad():
        m(idx)                                                  # inférence
        m.generate(idx, 3, temperature=0)                       # prefill + décodage KV-cache
    assert seen and all(p == 0.0 for p in seen)


@pytest.mark.parametrize("attn_dropout,stochastic", [(0.0, False), (0.5, True)])
def test_manual_attention_path_uses_attn_dropout(attn_dropout, stochastic):
    """Sans SDPA : seul nn.Dropout(attn_dropout) agit sur les poids d'attention."""
    torch.manual_seed(0)
    m = nanoPOPIXA(_tiny_config(n_layer=1, dropout=0.0, attn_dropout=attn_dropout))
    _set_flash(m, False)
    m.train()
    idx = torch.randint(0, 65, (1, 16))
    a, _ = m(idx, idx)
    b, _ = m(idx, idx)
    assert torch.equal(a, b) != stochastic


def test_config_pickled_without_attn_dropout_still_loads(tmp_path):
    """Checkpoints 2.1/2.2 : POPIXAConfig picklé sous model.POPIXAConfig, sans attn_dropout."""
    old = _tiny_config(dropout=0.1)
    del old.__dict__["attn_dropout"]                            # objet tel que picklé avant la 2.3
    path = tmp_path / "checkpoint.pt"
    torch.save({"config": old}, str(path))
    cfg = torch.load(str(path), map_location="cpu", weights_only=False)["config"]
    assert type(cfg) is POPIXAConfig and "attn_dropout" not in vars(cfg)
    m = nanoPOPIXA(cfg)
    assert m.transformer.h[0].attn.attn_dropout_p == 0.0
    assert m.transformer.h[0].attn.resid_dropout.p == 0.1
    # Config sans défaut de classe (objet quelconque) : le getattr de l'attention suffit
    ns = SimpleNamespace(**vars(cfg))
    assert not hasattr(ns, "attn_dropout")
    assert nanoPOPIXA(ns).transformer.h[0].attn.attn_dropout_p == 0.0
