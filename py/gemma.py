"""NumPy implementation of Gemma3-270M"""

import numpy as np
import math
from einops import rearrange
from dataclasses import dataclass

try:
    import torch
    from safetensors import safe_open
except ImportError:
    pass

from .weight_handlers import (
    MLPWeights,
    AttentionWeights,
    LayerWeights,
    TransformerWeights,
    KVCache,
    Array,
    RoPE,
    quantize_weight,
    dequantize,
)


def write_from_transformer_weights_quantized(weights: TransformerWeights, path: str):
    with open(path, "wb") as f:
        wte, wte_scales = quantize_weight(weights.wte)
        f.write(wte.data)
        f.write(wte_scales.data)

        f.write(weights.rms_weight.data)

        for layer in weights.layers:
            qq_proj, q_scales = quantize_weight(layer.attn.q_proj)
            qk_proj, k_scales = quantize_weight(layer.attn.k_proj)
            qv_proj, v_scales = quantize_weight(layer.attn.v_proj)
            qo_proj, o_scales = quantize_weight(layer.attn.o_proj)

            f.write(qq_proj.data)
            f.write(q_scales.data)

            f.write(qk_proj.data)
            f.write(k_scales.data)

            f.write(qv_proj.data)
            f.write(v_scales.data)

            f.write(qo_proj.data)
            f.write(o_scales.data)

            f.write(layer.attn.q_rms_weight.data)
            f.write(layer.attn.k_rms_weight.data)

            f.write(layer.attn.rms_weight_pre.data)
            f.write(layer.attn.rms_weight_post.data)

            qup_proj, up_proj_scales = quantize_weight(layer.mlp.up_proj)
            qgate_proj, gate_proj_scales = quantize_weight(layer.mlp.gate_proj)
            qdown_proj, down_proj_scales = quantize_weight(layer.mlp.down_proj)

            f.write(qup_proj.data)
            f.write(up_proj_scales.data)

            f.write(qgate_proj.data)
            f.write(gate_proj_scales.data)

            f.write(qdown_proj.data)
            f.write(down_proj_scales.data)

            f.write(layer.mlp.rms_weight_pre.data)
            f.write(layer.mlp.rms_weight_post.data)

        qlm_head, lm_head_scales = quantize_weight(weights.lm_head)
        f.write(qlm_head.data)
        f.write(lm_head_scales.data)


@dataclass
class Config:
    d_model: int
    d_intermediate: int
    n_layers: int
    n_vocab: int
    n_pos: int
    n_pos_local: int

    head_dim: int
    n_head: int
    n_kv: int

    theta: float
    theta_local: float

    window_pattern: int


CONFIG_MAP = {
    "270m": Config(640, 2048, 18, 262144, 32768, 512, 256, 4, 1, 1_000_000, 10_000, 6),
    "1B": Config(1152, 6912, 26, 262144, 32768, 512, 256, 4, 1, 1_000_000, 10_000, 6),
}


def linear(x: Array, weight: Array, bias: Array | None = None) -> Array:
    weight_dq = weight

    if bias is None:
        return x @ weight_dq
    else:
        return (x @ weight_dq) + bias


def rmsnorm(x: Array, weight: Array, eps: float = 1e-06) -> Array:
    x_norm = x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps))

    return x_norm * (1.0 + weight)


def gelu(x: Array) -> Array:
    return (
        0.5
        * x
        * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3.0))))
    )


def softmax(x: Array) -> Array:
    _max = x.max(axis=-1, keepdims=True)
    return np.exp(x - _max) / np.sum(np.exp(x - _max), axis=-1, keepdims=True)


def attention(q: Array, k: Array, v: Array, scale: float, max_len: int = 1024) -> Array:
    qs = q.shape[2]
    ks = k.shape[2]

    logits = np.einsum("bhqj,bgkj -> bhqk", q, k, optimize=True)
    logits *= scale

    if qs == 1:
        _mask = np.ones((qs, ks))
        if ks > max_len:
            fill_range = np.arange(start=0, stop=(ks - max_len))
            _mask[fill_range[None, ...]] = 0.0

    else:
        _mask = np.tril(np.ones((qs, ks)))[:qs, :ks]

    logits = np.where(_mask, logits, float("-inf"))
    scores = softmax(logits)

    return np.einsum("bhqk,bgkj -> bhqj", scores, v, optimize=True)


def mlp(x: Array, weights: MLPWeights) -> Array:
    resid = x

    x = rmsnorm(x, weights.rms_weight_pre)
    up_gate_proj = gelu(linear(x, weights.gate_proj)) * linear(x, weights.up_proj)
    down_proj = linear(up_gate_proj, weights.down_proj)

    return resid + rmsnorm(down_proj, weights.rms_weight_post)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    assert dim % 2 == 0, "Dimension must be even for RoPE"
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))  # [128]
    t = np.arange(end, dtype=np.float32)  # [32768]
    freqs = np.outer(t, freqs)  # [32768, 128]

    emb = np.concatenate((freqs, freqs), axis=-1)  # [32768, 256]
    freqs_cos = np.cos(emb)
    freqs_sin = np.sin(emb)
    return freqs_cos.astype(np.float32), freqs_sin.astype(np.float32)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return np.concatenate((-x2, x1), axis=-1)


def apply_rotary_emb(
    xq: np.ndarray,
    xk: np.ndarray,
    freqs_cos: np.ndarray,
    freqs_sin: np.ndarray,
    offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    seq_len = xq.shape[-2]

    cos = freqs_cos[offset : offset + seq_len]
    sin = freqs_sin[offset : offset + seq_len]

    cos = cos[None, None, :, :]

    sin = sin[None, None, :, :]

    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)

    return xq_out, xk_out  # [b, h, l, d], [b, g, l, d]


def attn(
    x: Array,
    weights: AttentionWeights,
    cache: KVCache,
    freqs_cos: Array,
    freqs_sin: Array,
    max_len: int,
) -> Array:
    resid = x

    x = rmsnorm(x, weights.rms_weight_pre)

    q, k, v = (
        linear(x, weights.q_proj),
        linear(x, weights.k_proj),
        linear(x, weights.v_proj),
    )

    q = rearrange(q, "b l (h d) -> b h l d", h=weights.heads)
    k = rearrange(k, "b l d -> b 1 l d")
    v = rearrange(v, "b l d -> b 1 l d")

    q = rmsnorm(q, weights.q_rms_weight)

    k = rmsnorm(k, weights.k_rms_weight)

    current_pos_offset = cache.pos_ptr

    q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin, offset=current_pos_offset)

    if cache.pos_ptr == 0:
        new_kv_len = k.shape[-2]
        cache.keys[..., :new_kv_len, :] = k
        cache.values[..., :new_kv_len, :] = v
        cache.pos_ptr += new_kv_len

    else:
        cached_keys = cache.keys[..., : cache.pos_ptr, :]  # [b, h, p, d]
        cached_values = cache.values[..., : cache.pos_ptr, :]  # [b, h, p, d]

        k = np.concatenate([cached_keys, k], axis=-2)
        v = np.concatenate([cached_values, v], axis=-2)

        # write next value to KV cache
        cache.keys[..., cache.pos_ptr, :] = k[..., -1, :]
        cache.values[..., cache.pos_ptr, :] = v[..., -1, :]
        cache.pos_ptr += 1

    scale = 1.0 / math.sqrt(q.shape[-1])

    attn_out = attention(q, k, v, scale, max_len=max_len)
    attn_out = rearrange(attn_out, "b h l d -> b l (h d)")

    out = linear(attn_out, weights.o_proj)

    return resid + rmsnorm(out, weights.rms_weight_post), cache


def forward_model(
    tokens: Array, weights: TransformerWeights, cache: list[KVCache], rope: RoPE
) -> tuple[Array, list[KVCache]]:
    assert tokens.ndim == 2
    assert tokens.shape[0] == 1
    assert len(cache) == len(weights.layers)

    hidden_embeds = weights.wte[tokens]

    hidden_embeds = hidden_embeds * (hidden_embeds.shape[-1] ** 0.5)

    hidden = hidden_embeds

    updated_cache = []

    for layer, layer_cache in zip(weights.layers, cache):
        if layer.attn.is_sliding:
            hidden, layer_cache_updated = attn(
                hidden,
                layer.attn,
                layer_cache,
                rope.cos_local,
                rope.sin_local,
                max_len=512,
            )

        else:
            hidden, layer_cache_updated = attn(
                hidden, layer.attn, layer_cache, rope.cos, rope.sin, max_len=32768
            )

        updated_cache.append(layer_cache_updated)
        hidden = mlp(hidden, layer.mlp)

    hidden = rmsnorm(hidden, weights.rms_weight)

    logits = linear(hidden, weights.lm_head, None)

    return logits, updated_cache


def cache_inititialzer(
    bs: int, groups: int, pos: int, d_model: int, layers: int
) -> list[KVCache]:
    def _init_empty(size) -> Array:
        return np.empty(size, dtype=np.float32)

    def cache_size(i: int) -> int:
        if (i % 6) == 5:
            return pos
        else:
            return 512

    kv_cache = [
        KVCache(
            _init_empty((bs, 1, cache_size(layer), d_model // groups)),
            _init_empty((bs, 1, cache_size(layer), d_model // groups)),
            0,
        )
        for layer in range(layers)
    ]

    return kv_cache


# NOTE: seems like a bug but these are the same seq for normal and sw...?
def rope_initializer(max_seq: int, theta: float, local_theta: float, dim: int) -> RoPE:
    freqs_cos, freqs_sin = precompute_freqs_cis(dim, max_seq, theta)
    freqs_cos_local, freqs_sin_local = precompute_freqs_cis(dim, max_seq, local_theta)
    return RoPE(
        cos=freqs_cos,
        sin=freqs_sin,
        cos_local=freqs_cos_local,
        sin_local=freqs_sin_local,
    )


def logit_processor(
    logits: Array, temp: float, top_k: int, do_sample: bool = True
) -> Array:
    if do_sample:
        vocab = logits.shape[-1]
        logits = logits / temp

        top_k_indices = np.argpartition(logits[:, -1, :], -top_k)[-top_k:]

        # Create a mask for top-k tokens
        mask = np.zeros(vocab, dtype=bool)
        mask[top_k_indices] = True

        # Set non-top-k logits to negative infinity
        logits[0, -1, ~mask] = -float("inf")

        logits = softmax(logits)

        tokens = np.random.choice(
            a=np.array(list(range(0, vocab))), size=1, p=logits[0, -1, :]
        )[:, None]

        return tokens

    return logits[:, -1, :].argmax(axis=-1)[:, None]

def checkpoint_initializer(
    weights_path: str, config: Config, quant_weights: bool = False
):
    """Initializes model weights from a pretrained checkpoint"""

    with (
        safe_open(weights_path, framework="pt", device="cpu") as f1,
    ):
        lm_head_pattern = "model.embed_tokens.weight"
        wte_pattern = "model.embed_tokens.weight"

        rms_norm_w_pattern = "model.norm.weight"

        layer_mlp_pattern = (
            "model.layers.{l}.mlp.{w}.weight"  # w in (down_proj, gate_proj, up_proj)
        )

        layer_norm_pre_pattern = (
            "model.layers.{l}.{w}_layernorm.weight"  # w in (pre_feedforward, input)
        )
        layer_norm_pattern = "model.layers.{l}.post_{w}_layernorm.weight"  # w in (attention, feedforward)
        layer_attn_pattern = "model.layers.{l}.self_attn.{w}.weight"  # w in (k_norm, k_proj, o_proj, q_norm, q_proj, v_proj)

        def _try_key(key: str):
            if key in f1.keys():
                return f1.get_tensor(key)

            else:
                raise KeyError

        def _convert(tensor: torch.Tensor, do_transpose: bool = True) -> Array:  # type: ignore
            t = tensor.float().numpy()
            if (t.ndim > 1) and (do_transpose):
                return np.ascontiguousarray(t.transpose())
            else:
                return t

        if quant_weights:
            # this is done to simulate W8A32 quantization in the cpp vers
            # integer gemms are painfully slow in numpy so we keep all computation in float32
            def _quant_mlp(tensor: torch.Tensor, do_transpose: bool = True):
                t = _convert(tensor, do_transpose)
                qt, scales = quantize_weight(t)
                t = dequantize(qt, scales)
                return t

        quant_mlp = _quant_mlp if quant_weights else _convert

        def _make_layer(idx: int) -> LayerWeights:
            mlp_weights = MLPWeights(
                gate_proj=quant_mlp(
                    _try_key(layer_mlp_pattern.format(l=idx, w="gate_proj"))
                ),
                up_proj=quant_mlp(
                    _try_key(layer_mlp_pattern.format(l=idx, w="up_proj"))
                ),
                down_proj=quant_mlp(
                    _try_key(layer_mlp_pattern.format(l=idx, w="down_proj"))
                ),
                rms_weight_post=_convert(
                    _try_key(layer_norm_pattern.format(l=idx, w="feedforward"))
                ),
                rms_weight_pre=_convert(
                    _try_key(layer_norm_pre_pattern.format(l=idx, w="pre_feedforward"))
                ),
            )

            attn_weights = AttentionWeights(
                is_sliding=(idx % config.window_pattern) != (config.window_pattern - 1),
                q_proj=quant_mlp(
                    _try_key(layer_attn_pattern.format(l=idx, w="q_proj"))
                ),
                k_proj=quant_mlp(
                    _try_key(layer_attn_pattern.format(l=idx, w="k_proj"))
                ),
                v_proj=quant_mlp(
                    _try_key(layer_attn_pattern.format(l=idx, w="v_proj"))
                ),
                o_proj=quant_mlp(
                    _try_key(layer_attn_pattern.format(l=idx, w="o_proj"))
                ),
                q_rms_weight=_convert(
                    _try_key(layer_attn_pattern.format(l=idx, w="q_norm"))
                ),
                k_rms_weight=_convert(
                    _try_key(layer_attn_pattern.format(l=idx, w="k_norm"))
                ),
                rms_weight_pre=_convert(
                    _try_key(layer_norm_pre_pattern.format(l=idx, w="input"))
                ),
                rms_weight_post=_convert(
                    _try_key(layer_norm_pattern.format(l=idx, w="attention"))
                ),
                heads=config.n_head,
            )

            return LayerWeights(attn=attn_weights, mlp=mlp_weights)

        weights = TransformerWeights(
            wte=_convert(_try_key(wte_pattern), do_transpose=False),
            layers=[_make_layer(i) for i in range(config.n_layers)],
            rms_weight=_convert(_try_key(rms_norm_w_pattern)),
            lm_head=quant_mlp(_try_key(lm_head_pattern)),
        )

        return weights
