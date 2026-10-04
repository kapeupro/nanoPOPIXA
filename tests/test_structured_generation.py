"""Intégration modèle × structured outputs : la sortie est toujours un JSON complet et valide."""

import json

import pytest
import torch

import structured
from conftest import CHARS, make_model

SCHEMAS = {
    # une propriété optionnelle coûteuse ne doit jamais rendre la fermeture impossible
    "optional-long": {"type": "object", "properties": {"a": {"type": "string", "minLength": 60}}},
    "records": {"type": "array", "items": {"type": "object",
                                           "properties": {k: {"type": "string"} for k in "abcde"},
                                           "required": list("abcde")}},
    "nested": {"type": "object", "properties": {"x": {"type": "object", "properties": {
        "y": {"type": "array", "items": {"type": "string", "minLength": 5}, "minItems": 2}},
        "required": ["y"]}}},
    "any": None,
}


@pytest.fixture(scope="module")
def setup():
    tb = structured.token_bytes_from_itos(dict(enumerate(CHARS)), len(CHARS))
    return make_model(block_size=256, vocab_size=len(CHARS), n_embd=32), tb


@pytest.mark.parametrize("name", list(SCHEMAS))
def test_output_always_complete_and_valid(setup, name):
    model, tb = setup
    schema = SCHEMAS[name]
    for budget in (10, 30, 60):
        for seed in range(8):
            c = structured.json_constraint(tb, schema)
            if len(c.completion_tokens()) > budget:
                continue
            torch.manual_seed(seed)
            toks = list(model.generate_structured(torch.tensor([[1]]), c, max_new_tokens=budget,
                                                  temperature=1.0))
            assert len(toks) <= budget and c.is_complete(), (name, budget, seed)
            doc = json.loads("".join(CHARS[t] for t in toks))
            assert structured.validate_instance(doc, schema) == []


def test_structured_cache_ref_covers_prompt_and_output(setup):
    model, tb = setup
    c = structured.json_constraint(tb, SCHEMAS["records"])
    cache_ref = []
    toks = list(model.generate_structured(torch.tensor([[1, 2, 3]]), c, max_new_tokens=40,
                                          cache_ref=cache_ref))
    kv = cache_ref[0]
    assert kv.token_ids == [1, 2, 3] + toks and kv.seq_len == len(kv.token_ids)
