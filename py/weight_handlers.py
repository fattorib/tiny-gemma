"""Weight only Int8 quantization"""

import numpy as np
from typing import NamedTuple, TypeAlias
from dataclasses import dataclass

Array: TypeAlias = np.ndarray


def quantize_weight(weights: Array) -> tuple[Array, Array]:
    # performs int8 weight quantization
    assert weights.ndim >= 2, "Error, quantizing 1d dimensional weights"

    scales = np.max(np.abs(weights), axis=0, keepdims=True) / 127.0

    scales = scales.astype(np.float32)

    weights_i8 = np.rint(weights / scales).astype(np.int8)

    return weights_i8, scales


def dequantize(quantized: Array, scales: Array) -> Array:
    return quantized.astype(np.float32) * scales


class MLPWeights(NamedTuple):
    gate_proj: Array  # [d1, d2]
    up_proj: Array  # [d1, d2]
    down_proj: Array  # [d2, d1]

    rms_weight_post: Array  # [d1]
    rms_weight_pre: Array  # [d1]


class AttentionWeights(NamedTuple):
    q_proj: Array  # [640, 1024]
    k_proj: Array  # [640, 256]
    v_proj: Array  # [640, 256]
    o_proj: Array  # [1024, 640]

    q_rms_weight: Array  # [640]
    k_rms_weight: Array  # [640]

    rms_weight_pre: Array  # [640]
    rms_weight_post: Array  # [640]

    heads: int
    is_sliding: bool


class LayerWeights(NamedTuple):
    attn: AttentionWeights
    mlp: MLPWeights


class TransformerWeights(NamedTuple):
    wte: Array  # [v, d]

    layers: list[LayerWeights]

    rms_weight: Array  # [d]

    lm_head: Array  # [d, v]


@dataclass
class KVCache:
    keys: Array  # [b, h, sq, d]
    values: Array  # [b, h, sq, d]
    pos_ptr: int


@dataclass
class RoPE:
    cos: Array  # [l, d]
    sin: Array  # [l, d]

    cos_local: Array  # [l, d]
    sin_local: Array  # [l, d]
