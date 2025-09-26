#!/bin/bash
set -e
./build/gemma-it --n_generation_steps 255 --weights_path tmp/gemma_it.bin --logits_path tmp/gemma_it_logits.bin --prefill_size 103