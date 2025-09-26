import argparse
import logging
import struct
from dataclasses import dataclass
from typing import Literal
from sentencepiece import SentencePieceProcessor
from pathlib import Path
from .gemma import (
    checkpoint_initializer,
    write_from_transformer_weights_quantized,
    CONFIG_MAP,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CLIArgs:
    weights_in_path: str
    weights_out_path: str
    model_size: Literal["270m", "1B"]
    tok_in_path: str
    tok_out_path: str


def parse() -> CLIArgs:
    parser = argparse.ArgumentParser(description="Convert torch weights and tokenizer.")

    parser.add_argument(
        "--weights-in-path",
        type=str,
        required=True,
        help="Load path for safetensors weights.",
    )
    parser.add_argument(
        "--weights-out-path", type=str, required=True, help="Save path for cpp weights"
    )
    parser.add_argument(
        "--model-size", type=str, required=True, help="Model size to load"
    )
    parser.add_argument(
        "--tok-in-path",
        type=str,
        required=True,
        help="Path to SentencePiece model file",
    )
    parser.add_argument(
        "--tok-out-path",
        type=str,
        required=True,
        help="Output path for tokenizer binary",
    )

    args = parser.parse_args()

    return CLIArgs(
        weights_in_path=args.weights_in_path,
        weights_out_path=args.weights_out_path,
        model_size=args.model_size,
        tok_in_path=args.tok_in_path,
        tok_out_path=args.tok_out_path,
    )


def convert_weights(in_path: str, out_path: str, model_size: str):
    assert model_size in CONFIG_MAP.keys()

    cfg = CONFIG_MAP[model_size]

    weights = checkpoint_initializer(
        in_path,
        cfg,
    )

    write_from_transformer_weights_quantized(weights, out_path)

    logger.info(f"Weights saved to {out_path=}")

    return weights


def convert_tokenizer(sp_path: str, tokenizer_path: str):
    sp_model = SentencePieceProcessor(model_file=sp_path)

    n_words = sp_model.vocab_size()
    bos_id = sp_model.bos_id()
    eos_id = sp_model.eos_id()

    tokens, scores = [], []
    for i in range(n_words):
        t = sp_model.id_to_piece(i)
        s = sp_model.get_score(i)

        if i == bos_id:
            t = "<bos>"
        elif i == eos_id:
            t = "<eos>"

        t = t.replace("▁", " ")
        b = t.encode("utf-8")

        tokens.append(b)
        scores.append(s)

    max_token_length = max(len(t) for t in tokens)

    with open(tokenizer_path, "wb") as f:
        f.write(struct.pack("I", max_token_length))

        for bytes, score in zip(tokens, scores):
            f.write(struct.pack("fI", score, len(bytes)))
            f.write(bytes)

    logger.info(f"Tokenizer saved to {tokenizer_path}")


def main():
    args = parse()
    logger.info(f"Processing: {args}")

    base_dir = args.weights_out_path.split('/')[0]
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    _ = convert_weights(args.weights_in_path, args.weights_out_path, args.model_size)

    convert_tokenizer(args.tok_in_path, args.tok_out_path)

    logger.info("Conversion complete")


if __name__ == "__main__":
    # python3 convert.py --weights-in-path 'weights/model.safetensors' --weights-out-path 'weights/gemma_i8_1B.bin' --model-size '1B' --tok-in-path 'weights/tokenizer.model' --tok-out-path 'weights/tokenizer_gemma3.bin'
    main()
