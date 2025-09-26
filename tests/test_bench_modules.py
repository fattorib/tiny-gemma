import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import logging
from dataclasses import dataclass
from time import time
from pathlib import Path
import numpy as np
from typing import Callable
import math
from py.gemma import (
    attention,
    linear,
    rearrange,
    rmsnorm,
    apply_rotary_emb,
    precompute_freqs_cis,
)
from py.gemma import (
    KVCache,
)
from py.weight_handlers import quantize_weight, dequantize
from py.gemma import gelu
from py.gemma import CONFIG_MAP

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

np.random.seed(0)


BASE_CONFIG = CONFIG_MAP["270m"]


@dataclass
class CLIArgs:
    op: str
    dir: str
    batch_size: int
    sequence_length: int
    dmodel: int
    check_perf: bool


def process_path(
    path: str, op_type: str, batch_size: int, sequence_length: int, dmodel: int
) -> str:
    if not path.endswith("/"):
        path += "/"

    path_metadata = (
        path + f"{op_type}_layer_{batch_size}_{sequence_length}_{dmodel}.bin"
    )

    return path_metadata


def time_op(
    fn: Callable, warmup: int = 100, benchmark: int = 100
) -> tuple[float, float]:
    """general timing function"""

    for i in range(warmup):
        _ = fn()

    t0 = time()
    for i in range(benchmark):
        _ = fn()
    t1 = time()
    s = t1 - t0

    return s, s / benchmark


def parse() -> CLIArgs:
    parser = argparse.ArgumentParser(description="Test bench")

    parser.add_argument(
        "--op",
        type=str,
        help="Op type.",
    )
    parser.add_argument("--dir", type=str, help="Save path for array binary.")
    parser.add_argument("--B", type=int, default=1, help="Batch size (defaults to 1).")
    parser.add_argument("--S", type=int, default=512, help="Context (defaults to 768).")
    parser.add_argument(
        "--D", type=int, default=768, help="Model dimension (defaults to 768)."
    )
    parser.add_argument(
        "--check-perf",
        action="store_true",
        help="Flag to enable performance benchmarks.",
    )

    args = parser.parse_args()

    # make directory
    Path(args.dir).mkdir(parents=True, exist_ok=True)

    dir_processed = process_path(args.dir, args.op, args.B, args.S, args.D)

    return CLIArgs(
        op=args.op,
        dir=dir_processed,
        batch_size=args.B,
        sequence_length=args.S,
        dmodel=args.D,
        check_perf=args.check_perf,
    )


def benchmark_attention_prefill(args: CLIArgs):
    batch_size, sequence_length, dmodel, dir, check_perf = (
        args.batch_size,
        args.sequence_length,
        args.dmodel,
        args.dir,
        args.check_perf,
    )

    head_dimension = BASE_CONFIG.head_dim
    num_heads = BASE_CONFIG.n_head
    num_groups = BASE_CONFIG.n_kv

    scale = 1.0 / (dmodel ** (0.5))

    act = np.random.randn(batch_size, sequence_length, dmodel).astype(np.float32)
    q_proj = (
        np.random.randn(dmodel, head_dimension * num_heads).astype(np.float32) * scale
    )
    k_proj = (
        np.random.randn(dmodel, head_dimension * num_groups).astype(np.float32) * scale
    )
    v_proj = (
        np.random.randn(dmodel, head_dimension * num_groups).astype(np.float32) * scale
    )
    o_proj = (
        np.random.randn(head_dimension * num_heads, dmodel).astype(np.float32) * scale
    )

    qq_proj, q_scales = quantize_weight(q_proj)
    qk_proj, k_scales = quantize_weight(k_proj)
    qv_proj, v_scales = quantize_weight(v_proj)
    qo_proj, o_scales = quantize_weight(o_proj)

    rms_weight_pre = np.random.randn(dmodel).astype(np.float32)
    rms_weight_post = np.random.randn(dmodel).astype(np.float32)

    rms_weight_q = np.random.randn(head_dimension).astype(np.float32)
    rms_weight_k = np.random.randn(head_dimension).astype(np.float32)

    freqs_cos, freqs_sin = precompute_freqs_cis(
        head_dimension, BASE_CONFIG.n_pos, BASE_CONFIG.theta
    )

    def fn():
        resid = act

        act0 = rmsnorm(act, rms_weight_pre)

        q, k, v = (
            linear(act0, dequantize(qq_proj, q_scales), None),
            linear(act0, dequantize(qk_proj, k_scales), None),
            linear(act0, dequantize(qv_proj, v_scales), None),
        )

        q = rearrange(q, "b l (h d) -> b h l d", h=num_heads)
        k = rearrange(k, "b l d -> b 1 l d")
        v = rearrange(v, "b l d -> b 1 l d")

        q = rmsnorm(q, rms_weight_q)
        k = rmsnorm(k, rms_weight_k)

        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin, offset=0)

        attn_scale = 1.0 / math.sqrt(q.shape[-1])

        attn_out = attention(q, k, v, attn_scale)
        attn_out = rearrange(attn_out, "b h l d -> b l (h d)")

        out = linear(attn_out, dequantize(qo_proj, o_scales), None)

        return rmsnorm(out, rms_weight_post) + resid

    out = fn()

    assert out.dtype == np.float32

    with open(dir, "wb") as f:
        f.write(act.data)
        f.write(out.data)

        f.write(qq_proj.astype(np.int8).data)
        f.write(q_scales.astype(np.float32).data)

        f.write(qk_proj.astype(np.int8).data)
        f.write(k_scales.astype(np.float32).data)

        f.write(qv_proj.astype(np.int8).data)
        f.write(v_scales.astype(np.float32).data)

        f.write(qo_proj.astype(np.int8).data)
        f.write(o_scales.astype(np.float32).data)

        f.write(rms_weight_pre.data)
        f.write(rms_weight_post.data)
        f.write(rms_weight_q.data)
        f.write(rms_weight_k.data)

    logger.info(f"Wrote binary to {dir}")

    if check_perf:
        benchmark = 100
        s, s_per_it = time_op(fn, benchmark=benchmark)
        # FLOP calculation for attention prefill:
        # QKV projections: 3 * (B * S * D * D) = 3 * B * S * D^2
        # Attention computation: B * H * S^2 * D (QK matmul) + B * H * S^2 * D (softmax+V matmul)
        # Output projection: B * S * D * D = B * S * D^2
        # RMSNorm operations (3 instances): ~3 * B * S * D
        flops_per_iter = (
            3 * batch_size * sequence_length * dmodel * dmodel  # QKV projections
            + 2
            * batch_size
            * num_heads
            * sequence_length
            * sequence_length
            * head_dimension  # attention
            + batch_size * sequence_length * dmodel * dmodel  # output projection
            + 3 * batch_size * sequence_length * dmodel  # RMSNorm operations
        )
        total_flops = benchmark * flops_per_iter

        logger.info(f"Time elapsed: {s:.8f}")
        logger.info(f"GFLOP/s: {(total_flops / s) / 1e9:.8f}")


def benchmark_attention_decode(args: CLIArgs):
    batch_size, sequence_length, dmodel, dir, check_perf = (
        args.batch_size,
        args.sequence_length,
        args.dmodel,
        args.dir,
        args.check_perf,
    )

    head_dimension = BASE_CONFIG.head_dim
    num_heads = BASE_CONFIG.n_head
    num_groups = BASE_CONFIG.n_kv

    scale = 1.0 / (dmodel ** (0.5))

    act = np.random.randn(batch_size, 1, dmodel).astype(np.float32)
    q_proj = (
        np.random.randn(dmodel, head_dimension * num_heads).astype(np.float32) * scale
    )

    k_proj = (
        np.random.randn(dmodel, head_dimension * num_groups).astype(np.float32) * scale
    )
    v_proj = (
        np.random.randn(dmodel, head_dimension * num_groups).astype(np.float32) * scale
    )

    o_proj = (
        np.random.randn(head_dimension * num_heads, dmodel).astype(np.float32) * scale
    )

    qq_proj, q_scales = quantize_weight(q_proj)
    qk_proj, k_scales = quantize_weight(k_proj)
    qv_proj, v_scales = quantize_weight(v_proj)
    qo_proj, o_scales = quantize_weight(o_proj)

    rms_weight_pre = np.random.randn(dmodel).astype(np.float32)
    rms_weight_post = np.random.randn(dmodel).astype(np.float32)

    rms_weight_q = np.random.randn(head_dimension).astype(np.float32)
    rms_weight_k = np.random.randn(head_dimension).astype(np.float32)

    keys_kv = np.random.randn(
        batch_size, num_groups, sequence_length, head_dimension
    ).astype(np.float32)

    freqs_cos, freqs_sin = precompute_freqs_cis(
        head_dimension, BASE_CONFIG.n_pos, BASE_CONFIG.theta
    )

    # pre-proc on Keys in $
    keys_kv = rmsnorm(keys_kv, rms_weight_k)
    keys_kv, _ = apply_rotary_emb(keys_kv, keys_kv, freqs_cos, freqs_sin, offset=0)

    values_kv = np.random.randn(
        batch_size, num_groups, sequence_length, head_dimension
    ).astype(np.float32)

    cache = KVCache(
        np.empty(
            (batch_size, num_groups, BASE_CONFIG.n_pos, head_dimension),
            dtype=np.float32,
        ),
        np.empty(
            (batch_size, num_groups, BASE_CONFIG.n_pos, head_dimension),
            dtype=np.float32,
        ),
        0,
    )

    new_kv_len = keys_kv.shape[-2]
    cache.keys[..., :new_kv_len, :] = keys_kv
    cache.values[..., :new_kv_len, :] = values_kv
    cache.pos_ptr += new_kv_len

    def fn():
        resid = act

        act0 = rmsnorm(act, rms_weight_pre)

        q, k, v = (
            linear(act0, dequantize(qq_proj, q_scales), None),
            linear(act0, dequantize(qk_proj, k_scales), None),
            linear(act0, dequantize(qv_proj, v_scales), None),
        )

        q = rearrange(q, "b l (h d) -> b h l d", h=num_heads)
        k = rearrange(k, "b l d -> b 1 l d")
        v = rearrange(v, "b l d -> b 1 l d")

        q = rmsnorm(q, rms_weight_q)
        k = rmsnorm(k, rms_weight_k)

        current_pos_offset = cache.pos_ptr

        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin, offset=current_pos_offset)

        cached_keys = cache.keys[..., : cache.pos_ptr, :]  # [b, h, p, d]
        cached_values = cache.values[..., : cache.pos_ptr, :]  # [b, h, p, d]

        k = np.concatenate([cached_keys, k], axis=-2)
        v = np.concatenate([cached_values, v], axis=-2)

        # write next value to KV cache
        cache.keys[..., cache.pos_ptr, :] = k[..., -1, :]
        cache.values[..., cache.pos_ptr, :] = v[..., -1, :]
        cache.pos_ptr += 1

        scale = 1.0 / math.sqrt(q.shape[-1])

        attn_out = attention(q, k, v, scale)
        attn_out = rearrange(attn_out, "b h l d -> b l (h d)")

        out = linear(attn_out, dequantize(qo_proj, o_scales), None)

        return rmsnorm(out, rms_weight_post) + resid

    out = fn()

    assert out.dtype == np.float32

    with open(dir, "wb") as f:
        f.write(act.data)
        f.write(out.data)
        f.write(keys_kv.data)
        f.write(values_kv.data)

        f.write(qq_proj.astype(np.int8).data)
        f.write(q_scales.astype(np.float32).data)

        f.write(qk_proj.astype(np.int8).data)
        f.write(k_scales.astype(np.float32).data)

        f.write(qv_proj.astype(np.int8).data)
        f.write(v_scales.astype(np.float32).data)

        f.write(qo_proj.astype(np.int8).data)
        f.write(o_scales.astype(np.float32).data)

        f.write(rms_weight_pre.data)
        f.write(rms_weight_post.data)
        f.write(rms_weight_q.data)
        f.write(rms_weight_k.data)

    logger.info(f"Wrote binary to {dir}")

    if check_perf:
        benchmark = 100
        s, s_per_it = time_op(fn, benchmark=benchmark)

        # Memory bandwidth calculation for attention decode (memory bound):
        # Input activations: B * 1 * D
        # QKV weights: D * (H*D + 2*KV_H*D)
        # Cached K,V: B * KV_H * S * D (read from cache)
        # Output: B * 1 * D
        # Weight loads dominate for decode
        bytes_per_iter = (
            batch_size * 1 * dmodel * 4  # input activations
            + dmodel
            * (num_heads * head_dimension + 2 * num_groups * head_dimension)
            * 4  # QKV weights
            + batch_size
            * num_groups
            * sequence_length
            * head_dimension
            * 4
            * 2  # cached K,V
            + batch_size
            * num_heads
            * head_dimension
            * dmodel
            * 4  # output projection weight
            + batch_size * 1 * dmodel * 4  # output
        )
        total_bytes = benchmark * bytes_per_iter

        logger.info(f"Time elapsed: {s:.8f}")
        logger.info(f"Bandwidth GB/s: {(total_bytes / s) / 1e9:.8f}")


def benchmark_mlp_prefill(args: CLIArgs):
    intermediate = BASE_CONFIG.d_intermediate

    batch_size, sequence_length, dmodel, dir, check_perf = (
        args.batch_size,
        args.sequence_length,
        args.dmodel,
        args.dir,
        args.check_perf,
    )

    act = np.random.randn(batch_size, sequence_length, dmodel).astype(np.float32)

    up_proj = np.random.randn(dmodel, intermediate).astype(np.float32)
    gate_proj = np.random.randn(dmodel, intermediate).astype(np.float32)
    down_proj = np.random.randn(intermediate, dmodel).astype(np.float32)

    rms_weight_pre = np.random.randn(dmodel).astype(np.float32)
    rms_weight_post = np.random.randn(dmodel).astype(np.float32)

    qup_proj, up_scales = quantize_weight(up_proj)
    qgate_proj, gate_scales = quantize_weight(gate_proj)
    qdown_proj, down_scales = quantize_weight(down_proj)

    def fn():
        resid = act
        act0 = rmsnorm(act, rms_weight_pre)
        up_gate_proj = gelu(linear(act0, dequantize(qgate_proj, gate_scales))) * linear(
            act0, dequantize(qup_proj, up_scales)
        )
        out = linear(up_gate_proj, (dequantize(qdown_proj, down_scales)))
        out = rmsnorm(out, rms_weight_post)

        return out + resid

    out = fn()

    assert out.dtype == np.float32

    with open(dir, "wb") as f:
        f.write(act.data)
        f.write(out.data)
        f.write(qup_proj.astype(np.int8).data)
        f.write(up_scales.astype(np.float32).data)

        f.write(qgate_proj.astype(np.int8).data)
        f.write(gate_scales.astype(np.float32).data)

        f.write(qdown_proj.astype(np.int8).data)
        f.write(down_scales.astype(np.float32).data)

        f.write(rms_weight_pre.data)
        f.write(rms_weight_post.data)

    logger.info(f"Wrote binary to {dir}")

    if check_perf:
        benchmark = 100
        s, s_per_it = time_op(fn, benchmark=benchmark)
        # FLOP calculation for MLP prefill with GeGLU:
        # Gate projection: B * S * D * I
        # Up projection: B * S * D * I
        # Element-wise ops (gelu, multiply): B * S * I
        # Down projection: B * S * I * D
        # RMSNorm operations: 2 * B * S * D
        flops_per_iter = (
            batch_size * sequence_length * dmodel * intermediate  # gate proj
            + batch_size * sequence_length * dmodel * intermediate  # up proj
            + batch_size * sequence_length * intermediate  # gelu + multiply
            + batch_size * sequence_length * intermediate * dmodel  # down proj
            + 2 * batch_size * sequence_length * dmodel  # RMSNorm operations
        )
        total_flops = benchmark * flops_per_iter
        logger.info(f"Time elapsed: {s:.8f}")
        logger.info(f"GFLOP/s: {(total_flops / s) / 1e9:.8f}")


def benchmark_mlp_decode(args: CLIArgs):
    intermediate = BASE_CONFIG.d_intermediate

    batch_size, _, dmodel, dir, check_perf = (
        args.batch_size,
        args.sequence_length,
        args.dmodel,
        args.dir,
        args.check_perf,
    )

    act = np.random.randn(batch_size, 1, dmodel).astype(np.float32)

    up_proj = np.random.randn(dmodel, intermediate).astype(np.float32)
    gate_proj = np.random.randn(dmodel, intermediate).astype(np.float32)

    down_proj = np.random.randn(intermediate, dmodel).astype(np.float32)

    rms_weight_pre = np.random.randn(dmodel).astype(np.float32)
    rms_weight_post = np.random.randn(dmodel).astype(np.float32)

    qup_proj, up_scales = quantize_weight(up_proj)
    qgate_proj, gate_scales = quantize_weight(gate_proj)
    qdown_proj, down_scales = quantize_weight(down_proj)

    def fn():
        resid = act
        act0 = rmsnorm(act, rms_weight_pre)
        up_gate_proj = gelu(linear(act0, dequantize(qgate_proj, gate_scales))) * linear(
            act0, dequantize(qup_proj, up_scales)
        )
        out = linear(up_gate_proj, (dequantize(qdown_proj, down_scales)))
        out = rmsnorm(out, rms_weight_post)

        return out + resid

    out = fn()

    assert out.dtype == np.float32

    with open(dir, "wb") as f:
        f.write(act.data)
        f.write(out.data)

        f.write(qup_proj.astype(np.int8).data)
        f.write(up_scales.astype(np.float32).data)

        f.write(qgate_proj.astype(np.int8).data)
        f.write(gate_scales.astype(np.float32).data)

        f.write(qdown_proj.astype(np.int8).data)
        f.write(down_scales.astype(np.float32).data)

        f.write(rms_weight_pre.data)
        f.write(rms_weight_post.data)

    logger.info(f"Wrote binary to {dir}")

    if check_perf:
        benchmark = 100
        s, s_per_it = time_op(fn, benchmark=benchmark)
        # Memory bandwidth calculation for MLP decode (memory bound):
        # Input: B * 1 * D
        # Gate/Up weights: 2 * D * I
        # Intermediate activations: B * 1 * I (gate, up, gelu*up results)
        # Down weight: I * D
        # Output: B * 1 * D
        bytes_per_iter = (
            act.nbytes  # input
            + up_proj.nbytes
            + gate_proj.nbytes  # up and gate weights
            + 3 * batch_size * 1 * intermediate * 4  # intermediate activations
            + down_proj.nbytes  # down weight
            + out.nbytes  # output
        )
        total_bytes = benchmark * bytes_per_iter
        logger.info(f"Time elapsed: {s:.8f}")
        logger.info(f"Bandwidth GB/s: {(total_bytes / s) / 1e9:.8f}")


def benchmark_lm_head_decode(args: CLIArgs):
    from py.gemma import linear, rmsnorm
    from py.weight_handlers import quantize_weight, dequantize

    batch_size, _, dmodel, dir, check_perf = (
        args.batch_size,
        args.sequence_length,
        args.dmodel,
        args.dir,
        args.check_perf,
    )

    act = np.random.randn(batch_size, 1, dmodel).astype(np.float32)
    lm_head = np.random.randn(dmodel, BASE_CONFIG.n_vocab).astype(np.float32)

    rms_weight = np.random.randn(dmodel).astype(np.float32)

    qlm_head, scales = quantize_weight(lm_head)

    def fn():
        act0 = rmsnorm(act, rms_weight)
        x = linear(act0, dequantize(qlm_head, scales), None)
        return x[:, -1, :]

    out = fn()

    assert out.dtype == np.float32

    with open(dir, "wb") as f:
        f.write(act.data)
        f.write(np.ascontiguousarray(out.data))
        f.write(qlm_head.astype(np.int8).data)
        f.write(scales.astype(np.float32).data)

        f.write(rms_weight.data)

    logger.info(f"Wrote binary to {dir}")

    if check_perf:
        benchmark = 100
        s, s_per_it = time_op(fn, benchmark=benchmark)
        # Memory bandwidth calculation for LM head decode (memory bound):
        # Input: B * 1 * D
        # RMSNorm weight: D
        # Normalized input: B * 1 * D
        # LM head weight: D * vocab_size
        # Output: B * vocab_size
        bytes_per_iter = (
            act.nbytes  # input
            + rms_weight.nbytes  # RMSNorm weight
            + act.nbytes  # normalized activations (temp)
            + lm_head.nbytes  # LM head weight
            + out.nbytes  # output
        )
        total_bytes = benchmark * bytes_per_iter
        logger.info(f"Time elapsed: {s:.8f}")
        logger.info(f"Bandwidth GB/s: {(total_bytes / s) / 1e9:.8f}")


def benchmark_lm_head_prefill(args: CLIArgs):
    batch_size, sequence_length, dmodel, dir, check_perf = (
        args.batch_size,
        args.sequence_length,
        args.dmodel,
        args.dir,
        args.check_perf,
    )

    act = np.random.randn(batch_size, sequence_length, dmodel).astype(np.float32)
    lm_head = np.random.randn(dmodel, BASE_CONFIG.n_vocab).astype(np.float32)

    rms_weight = np.random.randn(dmodel).astype(np.float32)

    qlm_head, scales = quantize_weight(lm_head)

    def fn():
        act0 = rmsnorm(act, rms_weight)
        x = linear(act0, dequantize(qlm_head, scales), None)
        return x[:, -1, :]

    out = fn()

    assert out.dtype == np.float32

    with open(dir, "wb") as f:
        f.write(act.data)
        f.write(np.ascontiguousarray(out.data))
        f.write(qlm_head.astype(np.int8).data)
        f.write(scales.astype(np.float32).data)

        f.write(rms_weight.data)

    logger.info(f"Wrote binary to {dir}")

    if check_perf:
        benchmark = 100
        s, s_per_it = time_op(fn, benchmark=benchmark)
        # FLOP calculation for LM head prefill:
        # RMSNorm: B * S * D
        # Linear projection: B * S * D * vocab_size
        flops_per_iter = (
            batch_size * sequence_length * dmodel  # RMSNorm
            + batch_size
            * sequence_length
            * dmodel
            * BASE_CONFIG.n_vocab  # linear projection
        )
        total_flops = benchmark * flops_per_iter
        logger.info(f"Time elapsed: {s:.8f}")
        logger.info(f"GFLOP/s: {(total_flops / s) / 1e9:.8f}")


def main():
    args = parse()
    logger.info(f"Processed: {args}")

    match args.op:
        case "attention_prefill":
            benchmark_attention_prefill(args)
        case "attention_decode":
            benchmark_attention_decode(args)
        case "mlp_prefill":
            benchmark_mlp_prefill(args)
        case "mlp_decode":
            benchmark_mlp_decode(args)
        case "lm_head_prefill":
            benchmark_lm_head_prefill(args)
        case "lm_head_decode":
            benchmark_lm_head_decode(args)
        case _:
            raise NotImplementedError


if __name__ == "__main__":
    main()
