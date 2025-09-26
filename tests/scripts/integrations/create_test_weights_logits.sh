#!/bin/bash
set -e
export PYTHONPATH="$PWD:$PYTHONPATH"
python3 tests/test_generation.py --safetensors_path "weights/model.safetensors" --save_weights_path "tmp/gemma_it.bin" --logits_path "tmp/gemma_it_logits.bin" --n_generation_steps 255
