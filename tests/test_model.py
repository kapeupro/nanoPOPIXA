"""Tests du modèle : attention + KV-cache, fenêtre glissante, sampling, générateurs, speculative decoding."""

import collections

import pytest
import torch

from conftest import make_model
from model import KVCache, RMSNorm, checkpoint_v1_error, nanoPOPIXA


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
