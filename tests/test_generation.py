import argparse
from dataclasses import dataclass
import logging
from py.gemma import (
    checkpoint_initializer,
    cache_inititialzer,
    forward_model,
    write_from_transformer_weights_quantized,
    rope_initializer,
    CONFIG_MAP,
)
import numpy as np
from pathlib import Path

np.random.seed(0)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

HEAD_DIM = 256
VOCAB = 262144
MAX_POS = 32768
GROUPS = 1
INTERMEDIATE = 2048


@dataclass
class CLIArgs:
    safetensors_path: str
    weights_path: str
    logits_path: str
    n_generation_steps: int


def parse() -> CLIArgs:
    parser = argparse.ArgumentParser(description="Forward pass integration test")

    parser.add_argument("--safetensors_path", type=str, help="Load path for safetensors weights")
    parser.add_argument("--save_weights_path", type=str, help="Save path for cpp weights")
    parser.add_argument(
        "--logits_path", type=str, help="Save path for tokens and logits"
    )
    parser.add_argument(
        "--n_generation_steps",
        type=int,
        help="Number of argmax decoding steps to run (includes prefill step)",
    )

    args = parser.parse_args()

    Path("tmp").mkdir(parents=True, exist_ok=True)

    return CLIArgs(
        safetensors_path=args.safetensors_path,
        weights_path=args.weights_path,
        logits_path=args.logits_path,
        n_generation_steps=args.n_generation_steps,
    )


def main(
    safetensors_path: str,
    weights_path: str,
    logits_path: str,
    n_generation_steps: int,
):
    from transformers import AutoTokenizer

    config = CONFIG_MAP["270m"]

    weights = checkpoint_initializer(safetensors_path, config, quant_weights=True)
    write_from_transformer_weights_quantized(weights, weights_path)

    logger.info(f"Weights saved to {weights_path=}")

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
    prompt = "We introduce Gemma 3, a multimodal addition to the Gemma family of lightweight open models, ranging in scale from 1 to 27 billion parameters. This version introduces vision understanding abilities, a wider coverage of languages and longer context - at least 128K tokens. We also change the architecture of the model to reduce the KV-cache memory that tends to explode with long context. This is achieved by increasing the ratio of local to global attention layers, and keeping the span on local attention short."

    tokens = np.array(tokenizer.encode(prompt))[None, :]

    print(f"Prompt length: {len(tokens[0])}")

    tokens = tokens.astype(np.int32)

    tokens_input = tokens.copy()

    cache = cache_inititialzer(
        1, config.n_kv, config.n_pos, config.head_dim, config.n_layers
    )
    rope = rope_initializer(
        config.n_pos, config.theta, config.theta_local, config.head_dim
    )

    logits = []
    tokens_argmax = []

    print(tokenizer.decode([t for t in tokens[0]]), end="", flush=True)

    # run greedy decoding for N_GENERATION steps
    for _ in range(0, n_generation_steps):
        out, cache = forward_model(tokens, weights, cache, rope)
        logits.append(out[:, -1, :])
        tokens = out[:, -1, :].argmax(axis=-1)[:, None]

        print(tokenizer.decode(tokens[0]), end="", flush=True)

        tokens_argmax.append(tokens[0])

    with open(logits_path, "wb") as f:
        f.write(tokens_input.data)
        for l in logits:  # noQA E741
            f.write(np.ascontiguousarray(l.data))
        for t in tokens_argmax:
            f.write(np.ascontiguousarray(t.astype(np.int32).data))

    return weights


if __name__ == "__main__":
    args = parse()
    logger.info(f"Processed: {args}")
    weights = main(
        args.safetensors_path,
        args.weights_path,
        args.logits_path,
        args.n_generation_steps,
    )
