#!/bin/bash
set -e

HIDDEN=640

./build/test_modules --op attention_prefill --bench 0 --batch 1 --seq 1024 --dmodel $HIDDEN
./build/test_modules --op attention_prefill --bench 0 --batch 1 --seq 281 --dmodel $HIDDEN
./build/test_modules --op attention_prefill --bench 0 --batch 1 --seq 593 --dmodel $HIDDEN
./build/test_modules --op attention_prefill --bench 0 --batch 1 --seq 480 --dmodel $HIDDEN
./build/test_modules --op attention_prefill --bench 0 --batch 1 --seq 217 --dmodel $HIDDEN
./build/test_modules --op attention_prefill --bench 0 --batch 1 --seq 57 --dmodel $HIDDEN

./build/test_modules --op mlp_prefill --bench 0 --batch 1 --seq 281 --dmodel $HIDDEN
./build/test_modules --op mlp_prefill --bench 0 --batch 1 --seq 593 --dmodel $HIDDEN
./build/test_modules --op mlp_prefill --bench 0 --batch 1 --seq 480 --dmodel $HIDDEN
./build/test_modules --op mlp_prefill --bench 0 --batch 1 --seq 217 --dmodel $HIDDEN
./build/test_modules --op mlp_prefill --bench 0 --batch 1 --seq 57 --dmodel $HIDDEN

./build/test_modules --op mlp_decode --bench 0 --batch 1 --seq 1 --dmodel $HIDDEN

./build/test_modules --op attention_decode --bench 0 --batch 1 --seq 648 --dmodel $HIDDEN
./build/test_modules --op attention_decode --bench 0 --batch 1 --seq 209 --dmodel $HIDDEN
./build/test_modules --op attention_decode --bench 0 --batch 1 --seq 830 --dmodel $HIDDEN
./build/test_modules --op attention_decode --bench 0 --batch 1 --seq 103 --dmodel $HIDDEN
./build/test_modules --op attention_decode --bench 0 --batch 1 --seq 389 --dmodel $HIDDEN
./build/test_modules --op attention_decode --bench 0 --batch 1 --seq 57 --dmodel $HIDDEN

./build/test_modules --op lm_head_prefill --bench 0 --batch 1 --seq 103 --dmodel $HIDDEN
./build/test_modules --op lm_head_prefill --bench 0 --batch 1 --seq 389 --dmodel $HIDDEN
./build/test_modules --op lm_head_prefill --bench 0 --batch 1 --seq 209 --dmodel $HIDDEN
./build/test_modules --op lm_head_prefill --bench 0 --batch 1 --seq 57 --dmodel $HIDDEN
./build/test_modules --op lm_head_decode --bench 0 --batch 1 --seq 1 --dmodel $HIDDEN