"""Convert an opf privacy-filter checkpoint to the mlx-embeddings port's layout.

The converted checkpoint lives beside its source in the model cache
(``<source>-mlx-bf16`` or ``<source>-mlx-8bit``) and records the SHA-256, size and
mtime of the source weights, so a fine-tune re-trained under the same name is
converted again while an unchanged source is recognised without rehashing.
The port's config and weights are derived from the opf checkpoint alone.

Weight mapping (opf name -> port name):

* ``block.i.attn.qkv`` is split into q/k/v at rows ``nq*D`` and ``(nq+nkv)*D``.
* ``block.i.mlp.swiglu.weight`` ``(E, in, 2I)`` splits into gate (first half) and
  up (second half), each transposed to ``(E, I, in)``; ``mlp.out`` is transposed
  into ``down_proj``.
* ``attn.sinks`` are stored in log2 units by opf and become natural-log units.
* Norm scales are renamed; the score head gets a zero bias (opf's unembedding has
  none). Everything is stored in bf16.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

_SOURCE_RECORD = "source.json"
_GROUP_SIZE = 64


def converted_path(src: Path, bits: int | None) -> Path:
    """Where the converted copy of ``src`` lives for the given quantisation."""
    src = Path(src)
    return src.with_name(src.name + ("-mlx-8bit" if bits == 8 else "-mlx-bf16"))


def _source_sha256(src: Path) -> str:
    digest = hashlib.sha256()
    with (Path(src) / "model.safetensors").open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_record(src: Path, bits: int | None, sha256: str) -> dict:
    stat = (Path(src) / "model.safetensors").stat()
    return {"sha256": sha256, "bits": bits, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def is_current(src: Path, dst: Path, bits: int | None) -> bool:
    """True when ``dst`` was converted from the present ``src`` weights at ``bits``.

    The source is hashed only when its size or mtime differs from the record; a
    matching hash then refreshes the record so the next check skips the hash.
    """
    record = Path(dst) / _SOURCE_RECORD
    if not record.is_file() or not (Path(dst) / "model.safetensors").is_file():
        return False
    recorded = json.loads(record.read_text())
    if recorded.get("bits") != bits:
        return False
    stat = (Path(src) / "model.safetensors").stat()
    if recorded.get("size") == stat.st_size and recorded.get("mtime_ns") == stat.st_mtime_ns:
        return True
    sha256 = _source_sha256(src)
    if recorded.get("sha256") != sha256:
        return False
    record.write_text(json.dumps(_source_record(src, bits, sha256)))
    return True


def _port_config(opf_cfg: dict) -> dict:
    left = opf_cfg["bidirectional_left_context"]
    if left != opf_cfg["bidirectional_right_context"]:
        raise ValueError("The MLX port needs a symmetric band: "
                         f"left={left}, right={opf_cfg['bidirectional_right_context']}")
    names = list(opf_cfg["ner_class_names"])
    return {
        "model_type": "openai_privacy_filter",
        "architectures": ["OpenAIPrivacyFilterForTokenClassification"],
        "vocab_size": opf_cfg["vocab_size"],
        "hidden_size": opf_cfg["hidden_size"],
        "intermediate_size": opf_cfg["intermediate_size"],
        "num_hidden_layers": opf_cfg["num_hidden_layers"],
        "num_attention_heads": opf_cfg["num_attention_heads"],
        "num_key_value_heads": opf_cfg["num_key_value_heads"],
        "head_dim": opf_cfg["head_dim"],
        # The port masks |i - j| <= sliding_window; opf's band is left/right context.
        "sliding_window": left,
        "max_position_embeddings": opf_cfg["max_position_embeddings"],
        # opf's RMSNorm eps is a constant, not a config field.
        "rms_norm_eps": 1e-05,
        "attention_bias": True,
        "num_local_experts": opf_cfg["num_experts"],
        "num_experts_per_tok": opf_cfg["experts_per_token"],
        "tie_word_embeddings": False,
        "rope_parameters": {
            "rope_type": "yarn",
            "rope_theta": float(opf_cfg["rope_theta"]),
            "factor": float(opf_cfg["rope_scaling_factor"]),
            "beta_fast": float(opf_cfg["rope_ntk_beta"]),
            "beta_slow": float(opf_cfg["rope_ntk_alpha"]),
            "original_max_position_embeddings": opf_cfg["initial_context_length"],
            "truncate": False,
        },
        "id2label": {str(i): name for i, name in enumerate(names)},
        "label2id": {name: i for i, name in enumerate(names)},
    }


def _port_weights(weights: dict, opf_cfg: dict) -> dict:
    import mlx.core as mx

    bf = mx.bfloat16
    inter = opf_cfg["intermediate_size"]
    nq, nkv, head = (opf_cfg["num_attention_heads"], opf_cfg["num_key_value_heads"],
                     opf_cfg["head_dim"])
    cuts = (0, nq * head, (nq + nkv) * head, (nq + 2 * nkv) * head)
    unembedding = weights["unembedding.weight"]
    out = {
        "model.embed_tokens.weight": weights["embedding.weight"],
        "model.norm.weight": weights["norm.scale"],
        "score.weight": unembedding,
        "score.bias": mx.zeros((unembedding.shape[0],), dtype=bf),
    }
    for i in range(opf_cfg["num_hidden_layers"]):
        b, p = f"block.{i}.", f"model.layers.{i}."
        qkv_w, qkv_b = weights[b + "attn.qkv.weight"], weights[b + "attn.qkv.bias"]
        for j, name in enumerate(("q_proj", "k_proj", "v_proj")):
            out[p + f"self_attn.{name}.weight"] = qkv_w[cuts[j]:cuts[j + 1]]
            out[p + f"self_attn.{name}.bias"] = qkv_b[cuts[j]:cuts[j + 1]]
        out[p + "self_attn.o_proj.weight"] = weights[b + "attn.out.weight"]
        out[p + "self_attn.o_proj.bias"] = weights[b + "attn.out.bias"]
        out[p + "self_attn.sinks"] = weights[b + "attn.sinks"].astype(mx.float32) * math.log(2.0)
        out[p + "input_layernorm.weight"] = weights[b + "attn.norm.scale"]
        out[p + "post_attention_layernorm.weight"] = weights[b + "mlp.norm.scale"]
        out[p + "mlp.router.weight"] = weights[b + "mlp.gate.weight"]
        out[p + "mlp.router.bias"] = weights[b + "mlp.gate.bias"]
        swiglu_w, swiglu_b = weights[b + "mlp.swiglu.weight"], weights[b + "mlp.swiglu.bias"]
        out[p + "mlp.experts.gate_proj.weight"] = mx.contiguous(swiglu_w[..., :inter].swapaxes(-1, -2))
        out[p + "mlp.experts.up_proj.weight"] = mx.contiguous(swiglu_w[..., inter:].swapaxes(-1, -2))
        out[p + "mlp.experts.gate_proj.bias"] = swiglu_b[:, :inter]
        out[p + "mlp.experts.up_proj.bias"] = swiglu_b[:, inter:]
        out[p + "mlp.experts.down_proj.weight"] = mx.contiguous(
            weights[b + "mlp.out.weight"].swapaxes(-1, -2))
        out[p + "mlp.experts.down_proj.bias"] = weights[b + "mlp.out.bias"]
    return {key: value.astype(bf) for key, value in out.items()}


def convert_checkpoint(src: Path, dst: Path, *, bits: int | None) -> Path:
    """Write the port's ``config.json`` and ``model.safetensors`` for ``src`` into ``dst``.

    ``source.json`` is written last, so an interrupted conversion is never current.
    """
    import mlx.core as mx

    if bits not in (None, 8):
        raise ValueError(f"bits must be None (bf16) or 8, got {bits!r}")
    src, dst = Path(src), Path(dst)
    opf_cfg = json.loads((src / "config.json").read_text())
    config = _port_config(opf_cfg)
    weights = _port_weights(mx.load(str(src / "model.safetensors")), opf_cfg)
    if bits == 8:
        from mlx_embeddings.convert import quantize_model

        model = _build_model(config)
        model.load_weights(list(weights.items()))
        weights, config = quantize_model(model, config, _GROUP_SIZE, bits, mode="affine")

    dst.mkdir(parents=True, exist_ok=True)
    (dst / _SOURCE_RECORD).unlink(missing_ok=True)
    mx.save_safetensors(str(dst / "model.safetensors"), weights, metadata={"format": "mlx"})
    (dst / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    calibration = src / "viterbi_calibration.json"
    if calibration.is_file():
        shutil.copyfile(calibration, dst / calibration.name)
    (dst / _SOURCE_RECORD).write_text(json.dumps(_source_record(src, bits, _source_sha256(src))))
    return dst


def _build_model(config: dict):
    from mlx_embeddings.models.openai_privacy_filter import Model, ModelArgs

    return Model(ModelArgs.from_dict(config))


def load_converted(path: Path):
    """Build the port's model from a converted checkpoint and materialise its weights."""
    import mlx.core as mx
    import mlx.nn as nn

    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    weights = mx.load(str(path / "model.safetensors"))
    model = _build_model(config)
    quantization = config.get("quantization")
    if quantization is not None:
        def quantized(module_path, module):
            if module_path in quantization:
                return quantization[module_path]
            return hasattr(module, "to_quantized") and f"{module_path}.scales" in weights

        nn.quantize(model, group_size=quantization["group_size"], bits=quantization["bits"],
                    mode=quantization["mode"], class_predicate=quantized)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    model.eval()
    return model
