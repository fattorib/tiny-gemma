#!/bin/bash
set -e
export PYTHONPATH="$PWD:$PYTHONPATH"
HIDDEN=640

python3 tests/test_bench_modules.py --dir tmp --B 1 --S 1024 --D $HIDDEN --op attention_prefill
python3 tests/test_bench_modules.py --dir tmp --B 1 --S 281 --D $HIDDEN --op attention_prefill
python3 tests/test_bench_modules.py --dir tmp --B 1 --S 593 --D $HIDDEN --op attention_prefill
python3 tests/test_bench_modules.py --dir tmp --B 1 --S 480 --D $HIDDEN --op attention_prefill
python3 tests/test_bench_modules.py --dir tmp --B 1 --S 217 --D $HIDDEN --op attention_prefill
python3 tests/test_bench_modules.py --dir tmp --B 1 --S 57 --D $HIDDEN --op attention_prefill

python3 tests/test_bench_modules.py --dir tmp --B 1 --S 281 --D $HIDDEN --op mlp_prefill
python3 tests/test_bench_modules.py --dir tmp --B 1 --S 593 --D $HIDDEN --op mlp_prefill
python3 tests/test_bench_modules.py --dir tmp --B 1 --S 480 --D $HIDDEN --op mlp_prefill
python3 tests/test_bench_modules.py --dir tmp --B 1 --S 217 --D $HIDDEN --op mlp_prefill
python3 tests/test_bench_modules.py --dir tmp --B 1 --S 57 --D $HIDDEN --op mlp_prefill

python3 tests/test_bench_modules.py --dir tmp --B 1 --S 1 --D $HIDDEN --op mlp_decode

python3 tests/test_bench_modules.py --dir tmp --B 1 --S 648 --D $HIDDEN --op attention_decode
python3 tests/test_bench_modules.py --dir tmp --B 1 --S 209 --D $HIDDEN --op attention_decode
python3 tests/test_bench_modules.py --dir tmp --B 1 --S 830 --D $HIDDEN --op attention_decode
python3 tests/test_bench_modules.py --dir tmp --B 1 --S 103 --D $HIDDEN --op attention_decode
python3 tests/test_bench_modules.py --dir tmp --B 1 --S 389 --D $HIDDEN --op attention_decode
python3 tests/test_bench_modules.py --dir tmp --B 1 --S 57 --D $HIDDEN --op attention_decode

python3 tests/test_bench_modules.py --dir tmp --B 1 --S 103 --D $HIDDEN --op lm_head_prefill
python3 tests/test_bench_modules.py --dir tmp --B 1 --S 389 --D $HIDDEN --op lm_head_prefill
python3 tests/test_bench_modules.py --dir tmp --B 1 --S 209 --D $HIDDEN --op lm_head_prefill
python3 tests/test_bench_modules.py --dir tmp --B 1 --S 57 --D $HIDDEN --op lm_head_prefill
python3 tests/test_bench_modules.py --dir tmp --B 1 --S 1 --D $HIDDEN --op lm_head_decode
