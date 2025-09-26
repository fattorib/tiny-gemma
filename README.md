# Tiny Gemma

Fast single-threaded Gemma 3 CPU inference with no external dependencies. Supports the 270m and 1B models with int8 weight-only quantization [^1].

## Requirements

- A C++20 compatible compiler (tested on GCC 10+)
- A CPU with AVX2 support
- Access to the gemma-3 weights from huggingface: [1](https://huggingface.co/google/gemma-3-270m) [2](https://huggingface.co/google/gemma-3-270m-it) [3](https://huggingface.co/google/gemma-3-1b-pt) [4](https://huggingface.co/google/gemma-3-1b-it)

## Setup

To download and prepare the 270m instruct model: 
```bash
git clone https://github.com/fattorib/tiny-gemma.git
cd tiny-gemma

# install requirements for conversion scripts
pip install -r requirements.txt

# download weights and tokenizer
hf auth login 
hf download google/gemma-3-270m-it tokenizer.model --local-dir weights
hf download google/gemma-3-270m-it model.safetensors --local-dir weights

# convert weights and tokenizer
python3 -m py.convert --weights-in-path 'model.safetensors' --weights-out-path 'gemma_i8.bin' --model-size '270m' --tok-in-path 'tokenizer.model' --tok-out-path 'tokenizer_gemma3.bin'

# Generate text 
make gemma && ./build/gemma --weights_path weights/gemma_i8.bin --model_size "270m" --n_dec 250 --minp 0.1 --temp 0.7  --prompt "What is a transformer?" --terminate_on_eos 1 --chat_format 1
```

## Generating text

`make gemma` exposes a simple CLI which can be run with `./build/gemma` the only non-trivial sampler implemented is `min-p + temperature` sampling. To generate (at most) 250 tokens with the Gemma chat format using min-p sampling (p = 0.1), you can run:

```bash
./build/gemma --weights_path weights/gemma_i8.bin --model_size "270m" --n_dec 250 --minp 0.1 --temp 0.7  --prompt "Tell me about the byte-pair-encoding (BPE) algorithm." --terminate_on_eos 1 --chat_format 1
```

Setting `--chat_format 1` ensures that the generation will terminate if an EOS token is generated. 

To run greedy decoding, just don't specify the `minp` argument:
```bash
./build/gemma --weights_path gemma_i8_1B.bin --model_size "1B" --n_dec 250  --prompt "What is one difference between GPT2 and BERT?" --terminate_on_eos 1 --chat_format 1
```

## Benchmarks

Comparisons againt [`llama.cpp`](https://github.com/ggml-org/llama.cpp) (build 6442) [^2]. Benchmarks performed on 1024 token completions with 10 randomly sampled prompts from [`ultrachat200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), the average prompt length was 178 tokens. 

llama-cpp was run with the following settings:

```bash
-t 1 -hf *model*:Q8_0 -n 10224 -no-cnv --ignore-eos --sampling-seq k --top-k
```

`google/gemma-3-270m`:
| Engine    | Prefill (t/s) | Decode (t/s) |
|-----------|-----------------|----------------|
| llama.cpp | 126.133         | 19.21         |
| **this repo**      | 94.28           | 58.78          |


`google/gemma-3-1B`:
| Engine    | Prefill (t/s) | Decode (t/s) |
|-----------|-----------------|----------------|
| llama.cpp | 29.37           | 7.53           |
| **this repo**      | 19.87           | 16.45          |

## Passkey

You can run a [passkey test](https://github.com/jquesnelle/yarn/blob/master/eval/passkey.py) on both the C++ and Python models to test their long context abilities. To build, run `make passkey`. To run a passkey test on `gemma-270m` with `16384` characters of garbage, run:

```bash
./build/passkey --weights_path weights/gemma_i8.bin --model_size "270m" --n_garbage 16384
```

Note that this test is quite slow since you will be processing a very long prompt on a single thread!

## Tests

There are unittests and integration tests to check for correctness against a numpy implementation of the model. The unittests run in the CI, the integration tests do not. 

The unittest scripts are located under `tests/scripts/unittests`:
- To build the C++ tests, run `make unittests`. 
- To create all the reference data in Python and serialize it, run: `./tests/scripts/unittests/create_test_data.sh`. 
- You can run run `./tests/scripts/unittests/run_tests.sh` to run the tests.

An integration test for end-to-end decoding is also provided under `tests/scripts/integrations`, it requires that you have downloaded the safetensor weights and tokenizer for `gemma-270m`. The integration test runs 255 steps of greedy decoding for the same prompt and compares the output tokens and logit distributions. To build the C++ tests, run `make integrations`. To create all the reference data in Python and serialize it, run: `./tests/scripts/integrations/create_test_weights_logits.sh`. Once the data is created, you can run run `./tests/scripts/integrations/run_test.sh` to run the test.

## Limitations
This implementation prioritizes simplicity over features. Current limitations include:

- *Architecture:* Extensive use of AVX2 intrinsics means the code is x86-only
- *Context:* Only supports single-turn conversations (no multi-turn chat history)
- *Sampling:* Limited to greedy and min-p sampling (no top-k, top-p, etc.)
- *Precision:* Int8 quantization only (no fp16, fp32, or other quantization formats)

## Acknowledgements 

- [`andrewkchan/yalm`](https://github.com/andrewkchan/yalm): A simple self-contained implementation of high-performance CPU/GPU inference which I used as a reference in some parts. 
- [`llama2.c`](https://github.com/karpathy/llama2.c): Tokenizer implementation and inverse-transform sampling are based off of these implementations.
- [Advanced Matrix Multiplication Optimization on Modern Multi-Core Processors](https://salykova.github.io/matmul-cpu): Original GEMM implementations were constructed following this post. 

[^1]: Technically there isn't anything in this codebase that prevents running the larger model, but do you really want to run a ≥4B param model on a single CPU core?
[^2]: All benchmarks were performed on a Ryzen 5-5500 with 16GB of 3200 MT/s memory.