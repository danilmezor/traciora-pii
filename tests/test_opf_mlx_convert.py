"""An opf checkpoint converts to the mlx-embeddings privacy-filter layout."""
import json

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_embeddings.models.openai_privacy_filter")

from traciora_pii.opf_mlx_convert import (  # noqa: E402
    convert_checkpoint,
    converted_path,
    is_current,
    load_converted,
)

HIDDEN, INTER, HEADS, KV_HEADS, HEAD_DIM, EXPERTS, VOCAB = 64, 64, 2, 1, 16, 2, 128
LABELS = ["O", "S-private_person", "S-private_address"]


def _opf_weights(seed: int) -> dict:
    mx.random.seed(seed)
    bf = mx.bfloat16
    qkv_rows = (HEADS + 2 * KV_HEADS) * HEAD_DIM

    def rnd(*shape):
        return mx.random.normal(shape).astype(bf)

    weights = {
        "embedding.weight": rnd(VOCAB, HIDDEN),
        "norm.scale": mx.ones((HIDDEN,), dtype=mx.float32),
        "unembedding.weight": rnd(len(LABELS), HIDDEN),
        "block.0.attn.norm.scale": mx.ones((HIDDEN,), dtype=mx.float32),
        "block.0.attn.qkv.weight": rnd(qkv_rows, HIDDEN),
        "block.0.attn.qkv.bias": rnd(qkv_rows),
        "block.0.attn.out.weight": rnd(HIDDEN, HEADS * HEAD_DIM),
        "block.0.attn.out.bias": rnd(HIDDEN),
        "block.0.attn.sinks": mx.random.normal((HEADS,)),
        "block.0.mlp.norm.scale": mx.ones((HIDDEN,), dtype=mx.float32),
        "block.0.mlp.gate.weight": rnd(EXPERTS, HIDDEN),
        "block.0.mlp.gate.bias": rnd(EXPERTS),
        "block.0.mlp.swiglu.weight": rnd(EXPERTS, HIDDEN, 2 * INTER),
        "block.0.mlp.swiglu.bias": rnd(EXPERTS, 2 * INTER),
        "block.0.mlp.out.weight": rnd(EXPERTS, INTER, HIDDEN),
        "block.0.mlp.out.bias": rnd(EXPERTS, HIDDEN),
    }
    return weights


@pytest.fixture
def opf_checkpoint(tmp_path):
    src = tmp_path / "tiny-opf"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({
        "bidirectional_context": True,
        "bidirectional_left_context": 4,
        "bidirectional_right_context": 4,
        "encoding": "o200k_base",
        "experts_per_token": 2,
        "head_dim": HEAD_DIM,
        "hidden_size": HIDDEN,
        "initial_context_length": 4096,
        "intermediate_size": INTER,
        "max_position_embeddings": 131072,
        "ner_class_names": LABELS,
        "num_attention_heads": HEADS,
        "num_experts": EXPERTS,
        "num_hidden_layers": 1,
        "num_key_value_heads": KV_HEADS,
        "rope_ntk_alpha": 1.0,
        "rope_ntk_beta": 32.0,
        "rope_scaling_factor": 32.0,
        "rope_theta": 150000,
        "sliding_window": 9,
        "vocab_size": VOCAB,
    }))
    mx.save_safetensors(str(src / "model.safetensors"), _opf_weights(0))
    (src / "viterbi_calibration.json").write_text("{}")
    return src


def test_convert_maps_every_tensor_and_shapes(opf_checkpoint):
    dst = convert_checkpoint(opf_checkpoint, converted_path(opf_checkpoint, None), bits=None)
    assert dst.name == "tiny-opf-mlx-bf16"
    weights = mx.load(str(dst / "model.safetensors"))
    p = "model.layers.0."
    expected = {
        "model.embed_tokens.weight": (VOCAB, HIDDEN),
        "model.norm.weight": (HIDDEN,),
        "score.weight": (len(LABELS), HIDDEN),
        "score.bias": (len(LABELS),),
        p + "input_layernorm.weight": (HIDDEN,),
        p + "post_attention_layernorm.weight": (HIDDEN,),
        p + "self_attn.q_proj.weight": (HEADS * HEAD_DIM, HIDDEN),
        p + "self_attn.k_proj.weight": (KV_HEADS * HEAD_DIM, HIDDEN),
        p + "self_attn.v_proj.weight": (KV_HEADS * HEAD_DIM, HIDDEN),
        p + "self_attn.q_proj.bias": (HEADS * HEAD_DIM,),
        p + "self_attn.k_proj.bias": (KV_HEADS * HEAD_DIM,),
        p + "self_attn.v_proj.bias": (KV_HEADS * HEAD_DIM,),
        p + "self_attn.o_proj.weight": (HIDDEN, HEADS * HEAD_DIM),
        p + "self_attn.o_proj.bias": (HIDDEN,),
        p + "self_attn.sinks": (HEADS,),
        p + "mlp.router.weight": (EXPERTS, HIDDEN),
        p + "mlp.router.bias": (EXPERTS,),
        p + "mlp.experts.gate_proj.weight": (EXPERTS, INTER, HIDDEN),
        p + "mlp.experts.up_proj.weight": (EXPERTS, INTER, HIDDEN),
        p + "mlp.experts.gate_proj.bias": (EXPERTS, INTER),
        p + "mlp.experts.up_proj.bias": (EXPERTS, INTER),
        p + "mlp.experts.down_proj.weight": (EXPERTS, HIDDEN, INTER),
        p + "mlp.experts.down_proj.bias": (EXPERTS, HIDDEN),
    }
    assert {k: tuple(v.shape) for k, v in weights.items()} == expected
    assert all(v.dtype == mx.bfloat16 for v in weights.values())

    src = mx.load(str(opf_checkpoint / "model.safetensors"))
    swiglu = src["block.0.mlp.swiglu.weight"]
    assert mx.array_equal(weights[p + "mlp.experts.gate_proj.weight"], swiglu[..., :INTER].swapaxes(-1, -2))
    assert mx.array_equal(weights[p + "mlp.experts.up_proj.weight"], swiglu[..., INTER:].swapaxes(-1, -2))
    assert mx.array_equal(weights[p + "self_attn.k_proj.weight"],
                          src["block.0.attn.qkv.weight"][HEADS * HEAD_DIM:(HEADS + KV_HEADS) * HEAD_DIM])
    assert not mx.any(weights["score.bias"]).item()

    config = json.loads((dst / "config.json").read_text())
    assert config["id2label"] == {str(i): n for i, n in enumerate(LABELS)}
    assert config["sliding_window"] == 4
    assert "quantization" not in config
    assert (dst / "viterbi_calibration.json").exists()

    model = load_converted(dst)
    logits = model(mx.array([[1, 2, 3]])).logits
    assert logits.shape == (1, 3, len(LABELS))


def test_convert_is_idempotent_until_source_changes(opf_checkpoint):
    dst = converted_path(opf_checkpoint, None)
    assert not is_current(opf_checkpoint, dst, None)
    convert_checkpoint(opf_checkpoint, dst, bits=None)
    assert is_current(opf_checkpoint, dst, None)
    assert not is_current(opf_checkpoint, dst, 8)
    mx.save_safetensors(str(opf_checkpoint / "model.safetensors"), _opf_weights(1))
    assert not is_current(opf_checkpoint, dst, None)


def test_convert_8bit_writes_quantization_block(opf_checkpoint):
    dst = convert_checkpoint(opf_checkpoint, converted_path(opf_checkpoint, 8), bits=8)
    assert dst.name == "tiny-opf-mlx-8bit"
    config = json.loads((dst / "config.json").read_text())
    assert config["quantization"]["bits"] == 8
    assert config["quantization"]["group_size"] == 64
    assert config["quantization"]["mode"] == "affine"
    assert json.loads((dst / "source.json").read_text())["bits"] == 8
    weights = mx.load(str(dst / "model.safetensors"))
    assert "model.layers.0.mlp.experts.gate_proj.scales" in weights
    model = load_converted(dst)
    assert model(mx.array([[1, 2, 3]])).logits.shape == (1, 3, len(LABELS))
