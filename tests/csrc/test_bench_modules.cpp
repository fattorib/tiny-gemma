#include <chrono>
#include <functional>

#include "argparse.hpp"
#include "constants.hpp"
#include "decode.hpp"
#include "linalg.hpp"
#include "modules.hpp"
#include "tensor.hpp"
#include "test_utils.hpp"
#include "transformer.hpp"

void test_attention_prefill(size_t batch, size_t sequence, size_t d_model, const char* file,
                            bool bench) {
	bool is_sw = false;

	config::Config cfg = config::CONFIG_MAP.at("270m");
	size_t numel = batch * sequence * d_model;

	size_t numelOQ = d_model * (cfg.head_dim) * (cfg.n_head);
	size_t numelKV = d_model * (cfg.head_dim) * (cfg.n_kv);

	size_t numelb = d_model;

	// reference solution
	tensor::AlignedArray<float> ref_out = tensor::make_unique_aligned<float>(numel);

	// input activation
	tensor::AlignedArray<float> ref_act = tensor::make_unique_aligned<float>(numel);

	modules::AttentionLayer l(
	    d_model,
	    cfg.head_dim,
	    cfg.n_head, cfg.n_kv, is_sw);

	decode::KVCache kv(batch, &cfg, is_sw);

	FILE* f = fopen(file, "rb");

	if (f != nullptr) {
		auto res = fread(ref_act.get(), 1, sizeof(float) * numel, f);
		res = fread(ref_out.get(), 1, sizeof(float) * numel, f);

		res = fread(l.Qw.data.get(), 1, sizeof(int8_t) * numelOQ, f);
		res = fread(l.Qw.scales.get(), 1, sizeof(float) * (cfg.head_dim * cfg.n_head), f);

		res = fread(l.Kw.data.get(), 1, sizeof(int8_t) * numelKV, f);
		res = fread(l.Kw.scales.get(), 1, sizeof(float) * (cfg.head_dim * cfg.n_kv), f);

		res = fread(l.Vw.data.get(), 1, sizeof(int8_t) * numelKV, f);
		res = fread(l.Vw.scales.get(), 1, sizeof(float) * (cfg.head_dim * cfg.n_kv), f);

		res = fread(l.Ow.data.get(), 1, sizeof(int8_t) * numelOQ, f);
		res = fread(l.Ow.scales.get(), 1, sizeof(float) * d_model, f);

		res = fread(l.RMSw_pre.data.get(), 1, sizeof(float) * numelb, f);
		res = fread(l.RMSw_post.data.get(), 1, sizeof(float) * numelb, f);

		res = fread(l.RMSq.data.get(), 1, sizeof(float) * cfg.head_dim, f);
		res = fread(l.RMSk.data.get(), 1, sizeof(float) * cfg.head_dim, f);

		fclose(f);
	} else {
		throw std::runtime_error(
		    "Attention Prefill: File is empty. Please generate by running `test_bench_layers.py`.");
	}

	tensor::activation act(ref_act, batch, sequence, d_model);

	decode::PrefillState p(batch, sequence, &cfg);

	decode::RoPE rope(&cfg, is_sw);

	// capture addresses from global scope
	auto fn = [&act, &l, &p, &kv, &rope]() { modules::attention(&act, &l, nullptr, &kv, &p, &rope); kv.position_ptr = 0; };
	fn();

	printf("\nCorrecntess Results (Attention Prefill):\n");
	bool pass = tests::check_error(act.data, ref_out, numel, 1.0, 0.001);

	assert(pass);

	if (bench) {
		int warmup = 100;
		int benchmark = 100;

		double s = tests::time_op(fn, warmup, benchmark);

		// FLOP calculation for attention prefill:
		// QKV projections: 3 * (B * S * D * D) = 3 * B * S * D^2
		// Attention computation: B * H * S^2 * D (QK matmul) + B * H * S^2 * D (softmax+V matmul)
		// Output projection: B * S * D * D = B * S * D^2
		// RMSNorm operations (3 instances): ~3 * B * S * D
		size_t flops_per_iter = (3 * batch * sequence * d_model * d_model                       // QKV projections
		                         + 2 * batch * cfg.n_head * sequence * sequence * cfg.head_dim  // attention
		                         + batch * sequence * d_model * d_model                         // output projection
		                         + 3 * batch * sequence * d_model                               // RMSNorm operations
		);
		double total_flops = benchmark * flops_per_iter;

		printf("Time elapsed: %fs \n", s);
		printf("GFLOP/s: %f \n\n", (total_flops / s) / 1e9);
	}
}

void test_attention_decode(size_t batch, size_t sequence, size_t d_model, const char* file,
                           bool bench) {
	bool is_sw = false;
	size_t numel = batch * 1 * d_model;

	config::Config cfg = config::CONFIG_MAP.at("270m");

	size_t numelOQ = d_model * (cfg.head_dim) * (cfg.n_head);
	size_t numelKV = d_model * (cfg.head_dim) * (cfg.n_kv);

	size_t numelKVa = sequence * (cfg.head_dim) * (cfg.n_kv);

	size_t numelb = d_model;

	// reference solution
	tensor::AlignedArray<float> ref_out = tensor::make_unique_aligned<float>(numel);

	// input activation
	tensor::AlignedArray<float> ref_act = tensor::make_unique_aligned<float>(numel);

	tensor::AlignedArray<float> k_kv = tensor::make_unique_aligned<float>(numelKVa);
	tensor::AlignedArray<float> v_kv = tensor::make_unique_aligned<float>(numelKVa);

	modules::AttentionLayer l(
	    d_model,
	    cfg.head_dim,
	    cfg.n_head, cfg.n_kv, is_sw);

	decode::KVCache kv(batch, &cfg, is_sw);

	decode::RoPE rope(&cfg, is_sw);

	FILE* f = fopen(file, "rb");

	if (f != nullptr) {
		auto res = fread(ref_act.get(), 1, sizeof(float) * numel, f);
		res = fread(ref_out.get(), 1, sizeof(float) * numel, f);
		res = fread(k_kv.get(), 1, sizeof(float) * numelKVa, f);
		res = fread(v_kv.get(), 1, sizeof(float) * numelKVa, f);

		res = fread(l.Qw.data.get(), 1, sizeof(int8_t) * numelOQ, f);
		res = fread(l.Qw.scales.get(), 1, sizeof(float) * (cfg.head_dim * cfg.n_head), f);

		res = fread(l.Kw.data.get(), 1, sizeof(int8_t) * numelKV, f);
		res = fread(l.Kw.scales.get(), 1, sizeof(float) * (cfg.head_dim * cfg.n_kv), f);

		res = fread(l.Vw.data.get(), 1, sizeof(int8_t) * numelKV, f);
		res = fread(l.Vw.scales.get(), 1, sizeof(float) * (cfg.head_dim * cfg.n_kv), f);

		res = fread(l.Ow.data.get(), 1, sizeof(int8_t) * numelOQ, f);
		res = fread(l.Ow.scales.get(), 1, sizeof(float) * d_model, f);

		res = fread(l.RMSw_pre.data.get(), 1, sizeof(float) * numelb, f);
		res = fread(l.RMSw_post.data.get(), 1, sizeof(float) * numelb, f);

		res = fread(l.RMSq.data.get(), 1, sizeof(float) * cfg.head_dim, f);
		res = fread(l.RMSk.data.get(), 1, sizeof(float) * cfg.head_dim, f);

		fclose(f);
	} else {
		throw std::runtime_error(
		    "Attention Decode: File is empty. Please generate by running `test_bench_layers.py`.");
	}

	tensor::activation act(ref_act, batch, 1, d_model);

	tensor::activation keys_kv(k_kv, batch, cfg.n_kv, sequence, cfg.head_dim);
	tensor::activation values_kv(v_kv, batch, cfg.n_kv, sequence, cfg.head_dim);

	kv.update_prefill(&keys_kv, &values_kv);

	cfg.n_layers = 1;

	decode::DecodeState d(1, &cfg);

	// capture addresses from global scope
	auto fn = [&act, &l, &d, &kv, sequence, &rope]() { modules::attention(&act, &l, &d, &kv, nullptr, &rope); kv.position_ptr = sequence; };
	fn();

	printf("\nCorrecntess Results (Attention Decode):\n");

	bool pass = tests::check_error(act.data, ref_out, numel, 1.0, 0.001);

	assert(pass);

	if (bench) {
		int warmup = 100;
		int benchmark = 100;

		double s = tests::time_op(fn, warmup, benchmark, false);

		// Memory bandwidth calculation for attention decode (memory bound):
		// Input activations: B * 1 * D
		// QKV weights: D * (H*D + 2*KV_H*D)
		// Cached K,V: B * KV_H * S * D (read from cache)
		// Output: B * 1 * D
		// Weight loads dominate for decode
		size_t bytes_per_iter = (batch * 1 * d_model * sizeof(float)                                                    // input activations
		                         + d_model * (cfg.n_head * cfg.head_dim + 2 * cfg.n_kv * cfg.head_dim) * sizeof(float)  // QKV weights
		                         + batch * cfg.n_kv * sequence * cfg.head_dim * sizeof(float) * 2                       // cached K,V
		                         + batch * cfg.n_head * cfg.head_dim * d_model * sizeof(float)                          // output projection weight
		                         + batch * 1 * d_model * sizeof(float)                                                  // output
		);
		double total_bytes = benchmark * bytes_per_iter;

		printf("Time elapsed: %fs \n", s);
		printf("Bandwidth GB/s: %f \n\n", (total_bytes / s) / 1e9);
	}
}

void test_mlp_prefill(size_t batch, size_t sequence, size_t d_model, const char* file,
                      bool bench) {
	config::Config cfg = config::CONFIG_MAP.at("270m");

	size_t numel = batch * sequence * d_model;
	size_t numelW = d_model * cfg.intermediate_size;
	size_t numelb = d_model;

	// reference solution
	tensor::AlignedArray<float> ref_out = tensor::make_unique_aligned<float>(numel);

	// input activation
	tensor::AlignedArray<float> ref_act = tensor::make_unique_aligned<float>(numel);

	modules::MLPLayer l(d_model, cfg.intermediate_size);

	FILE* f = fopen(file, "rb");

	if (f != nullptr) {
		auto res = fread(ref_act.get(), 1, sizeof(float) * numel, f);
		res = fread(ref_out.get(), 1, sizeof(float) * numel, f);

		res = fread(l.Uw.data.get(), 1, sizeof(int8_t) * numelW, f);
		res = fread(l.Uw.scales.get(), 1, sizeof(float) * cfg.intermediate_size, f);

		res = fread(l.Gw.data.get(), 1, sizeof(int8_t) * numelW, f);
		res = fread(l.Gw.scales.get(), 1, sizeof(float) * cfg.intermediate_size, f);

		res = fread(l.Dw.data.get(), 1, sizeof(int8_t) * numelW, f);
		res = fread(l.Dw.scales.get(), 1, sizeof(float) * d_model, f);

		res = fread(l.RMSw_pre.data.get(), 1, sizeof(float) * numelb, f);
		res = fread(l.RMSw_post.data.get(), 1, sizeof(float) * numelb, f);

		fclose(f);
	} else {
		throw std::runtime_error(
		    "MLP Prefill: File is empty. Please generate by running `test_bench_layers.py`.");
	}

	tensor::activation act(ref_act, batch, sequence, d_model);

	lut::GeLUTable table{-4.0, 4.0, 8192};

	decode::PrefillState p(batch, sequence, &cfg);

	auto fn = [&act, &l, &p, &table]() { modules::mlp(&act, &l, nullptr, &p, &table); };
	fn();

	printf("\nCorrecntess Results (MLP Prefill):\n");
	bool pass = tests::check_error(act.data, ref_out, numel, 10.0, 0.001);

	assert(pass);

	if (bench) {
		int warmup = 100;
		int benchmark = 100;

		double s = tests::time_op(fn, warmup, benchmark, false);

		// FLOP calculation for MLP prefill with GeGLU:
		// Gate projection: B * S * D * I
		// Up projection: B * S * D * I
		// Element-wise ops (gelu, multiply): B * S * I
		// Down projection: B * S * I * D
		// RMSNorm operations: 2 * B * S * D
		size_t flops_per_iter = (batch * sequence * d_model * cfg.intermediate_size    // gate proj
		                         + batch * sequence * d_model * cfg.intermediate_size  // up proj
		                         + batch * sequence * cfg.intermediate_size            // gelu + multiply
		                         + batch * sequence * cfg.intermediate_size * d_model  // down proj
		                         + 2 * batch * sequence * d_model                      // RMSNorm operations
		);
		double total_flops = benchmark * flops_per_iter;

		printf("Time elapsed: %fs \n", s);
		printf("GFLOP/s: %f \n\n", (total_flops / s) / 1e9);
	}
}

void test_lm_head_prefill(size_t batch, size_t sequence, size_t d_model, const char* file,
                          bool bench) {
	config::Config cfg = config::CONFIG_MAP.at("270m");
	cfg.n_layers = 1;

	size_t numelin = batch * sequence * d_model;
	size_t numelout = batch * cfg.n_vocab;
	size_t numelW = d_model * cfg.n_vocab;
	size_t numelb = d_model;

	// reference solution
	tensor::AlignedArray<float> ref_out = tensor::make_unique_aligned<float>(numelout);

	// input activation
	tensor::AlignedArray<float> ref_act = tensor::make_unique_aligned<float>(numelin);

	FILE* f = fopen(file, "rb");

	gemma::Transformer l(cfg);

	if (f != nullptr) {
		auto res = fread(ref_act.get(), 1, sizeof(float) * numelin, f);
		res = fread(ref_out.get(), 1, sizeof(float) * numelout, f);

		res = fread(l.lm_head.data.get(), 1, sizeof(int8_t) * numelW, f);
		res = fread(l.lm_head.scales.get(), 1, sizeof(float) * cfg.n_vocab, f);

		res = fread(l.RMSw.data.get(), 1, sizeof(float) * numelb, f);

		fclose(f);
	} else {
		throw std::runtime_error(
		    "LM Head Prefill: File is empty. Please generate by running `test_bench_layers.py`.");
	}

	tensor::activation act(ref_act, batch, sequence, d_model);

	tensor::activation out(batch, 1, cfg.n_vocab);
	// capture addresses from global scope
	auto fn = [&out, &act, &l]() { modules::lm_head(&out, &act, &l.lm_head, &l.RMSw); };
	fn();

	printf("\nCorrecntess Results (LM Head Prefill):\n");
	bool pass = tests::check_error(out.data, ref_out, numelout, 0.1, 0.001);

	assert(pass);

	if (bench) {
		int warmup = 100;
		int benchmark = 100;

		double s = tests::time_op(fn, warmup, benchmark, false);

		// FLOP calculation for LM head prefill:
		// RMSNorm: B * S * D
		// Linear projection: B * S * D * vocab_size
		size_t flops_per_iter = (batch * sequence * d_model                  // RMSNorm
		                         + batch * sequence * d_model * cfg.n_vocab  // linear projection
		);
		double total_flops = benchmark * flops_per_iter;

		printf("Time elapsed: %fs \n", s);
		printf("GFLOP/s: %f \n\n", (total_flops / s) / 1e9);
	}
}

void test_lm_head_decode(size_t batch, size_t sequence, size_t d_model, const char* file,
                         bool bench) {
	config::Config cfg = config::CONFIG_MAP.at("270m");
	cfg.n_layers = 1;

	size_t numelin = batch * d_model;

	size_t numelout = batch * cfg.n_vocab;
	size_t numelW = d_model * cfg.n_vocab;
	size_t numelb = d_model;

	// reference solution
	tensor::AlignedArray<float> ref_out = tensor::make_unique_aligned<float>(numelout);

	// input activation
	tensor::AlignedArray<float> ref_act = tensor::make_unique_aligned<float>(numelin);

	FILE* f = fopen(file, "rb");

	gemma::Transformer l(cfg);

	if (f != nullptr) {
		auto res = fread(ref_act.get(), 1, sizeof(float) * numelin, f);
		res = fread(ref_out.get(), 1, sizeof(float) * numelout, f);

		res = fread(l.lm_head.data.get(), 1, sizeof(int8_t) * numelW, f);
		res = fread(l.lm_head.scales.get(), 1, sizeof(float) * cfg.n_vocab, f);

		res = fread(l.RMSw.data.get(), 1, sizeof(float) * numelb, f);

		fclose(f);
	} else {
		throw std::runtime_error(
		    "LM Head Decode: File is empty. Please generate by running `test_bench_layers.py`.");
	}

	tensor::activation act(ref_act, batch, sequence, d_model);

	tensor::activation out(batch, 1, cfg.n_vocab);
	// capture addresses from global scope
	auto fn = [&out, &act, &l]() { modules::lm_head(&out, &act, &l.lm_head, &l.RMSw); };
	fn();

	printf("\nCorrecntess Results (LM Head Decode):\n");
	bool pass = tests::check_error(out.data, ref_out, numelout, 0.1, 0.001);

	assert(pass);

	if (bench) {
		int warmup = 100;
		int benchmark = 100;

		double s = tests::time_op(fn, warmup, benchmark, false);

		// Memory bandwidth calculation for LM head decode (memory bound):
		// Input: B * 1 * D
		// RMSNorm weight: D
		// Normalized input: B * 1 * D
		// LM head weight: D * vocab_size
		// Output: B * vocab_size
		size_t bytes_per_iter = (numelin * sizeof(float)     // input
		                         + numelb * sizeof(float)    // RMSNorm weight
		                         + numelin * sizeof(float)   // normalized activations (temp)
		                         + numelW * sizeof(float)    // LM head weight
		                         + numelout * sizeof(float)  // output
		);
		double total_bytes = benchmark * bytes_per_iter;

		printf("Time elapsed: %fs \n", s);
		printf("Bandwidth GB/s: %f \n\n", (total_bytes / s) / 1e9);
	}
}

void test_mlp_decode(size_t batch, size_t sequence, size_t d_model, const char* file,
                     bool bench) {
	config::Config cfg = config::CONFIG_MAP.at("270m");

	size_t numel = batch * sequence * d_model;
	size_t numelW = d_model * cfg.intermediate_size;
	size_t numelb = d_model;

	// reference solution
	tensor::AlignedArray<float> ref_out = tensor::make_unique_aligned<float>(numel);

	// input activation
	tensor::AlignedArray<float> ref_act = tensor::make_unique_aligned<float>(numel);

	FILE* f = fopen(file, "rb");

	modules::MLPLayer l(d_model, cfg.intermediate_size);

	if (f != nullptr) {
		auto res = fread(ref_act.get(), 1, sizeof(float) * numel, f);
		res = fread(ref_out.get(), 1, sizeof(float) * numel, f);

		res = fread(l.Uw.data.get(), 1, sizeof(int8_t) * numelW, f);
		res = fread(l.Uw.scales.get(), 1, sizeof(float) * cfg.intermediate_size, f);

		res = fread(l.Gw.data.get(), 1, sizeof(int8_t) * numelW, f);
		res = fread(l.Gw.scales.get(), 1, sizeof(float) * cfg.intermediate_size, f);

		res = fread(l.Dw.data.get(), 1, sizeof(int8_t) * numelW, f);
		res = fread(l.Dw.scales.get(), 1, sizeof(float) * d_model, f);

		res = fread(l.RMSw_pre.data.get(), 1, sizeof(float) * numelb, f);
		res = fread(l.RMSw_post.data.get(), 1, sizeof(float) * numelb, f);

		fclose(f);
	} else {
		throw std::runtime_error(
		    "MLP Decode: File is empty. Please generate by running `test_bench_layers.py`.");
	}

	tensor::activation act(ref_act, batch, sequence, d_model);

	lut::GeLUTable table{-4.0, 4.0, 8192};

	decode::DecodeState d(1, &cfg);

	// capture addresses from global scope
	auto fn = [&act, &l, &d, &table]() { modules::mlp(&act, &l, &d, nullptr, &table); };
	fn();

	printf("\nCorrecntess Results (MLP Decode):\n");
	bool pass = tests::check_error(act.data, ref_out, numel, 10.0, 0.001);

	assert(pass);

	if (bench) {
		int warmup = 100;
		int benchmark = 100;

		double s = tests::time_op(fn, warmup, benchmark, false);

		// Memory bandwidth calculation for MLP decode (memory bound):
		// Input: B * 1 * D
		// Gate/Up weights: 2 * D * I
		// Intermediate activations: B * 1 * I (gate, up, gelu*up results)
		// Down weight: I * D
		// Output: B * 1 * D
		size_t bytes_per_iter = (numel * sizeof(float)                                    // input
		                         + 2 * numelW * sizeof(float)                             // up and gate weights
		                         + 3 * batch * 1 * cfg.intermediate_size * sizeof(float)  // intermediate activations
		                         + numelW * sizeof(float)                                 // down weight
		                         + numel * sizeof(float)                                  // output
		);
		double total_bytes = benchmark * bytes_per_iter;

		printf("Time elapsed: %fs \n", s);
		printf("Bandwidth GB/s: %f \n\n", (total_bytes / s) / 1e9);
	}
}

int main(int argc, char** argv) {
	int bench;
	std::string op;

	int bs, seq, dmodel;

	ArgParser app(argv[0], "CLI Application for testing and benchmarking ops.");
	app.add_option("op", op, "Operaton to test/benchmark.");
	app.add_option("bench", bench, "Boolean flag to enable benchmarking of op.");
	app.add_option("batch", bs, "Batch size.");
	app.add_option("seq", seq, "Sequence length.");
	app.add_option("dmodel", dmodel, "Model dimension.");

	try {
		int rc = app.parse(argc, argv);
		if (rc != 0) return rc;  // parsing failed: help already printed
	} catch (const std::exception& ex) {
		std::cerr << "Parse error: " << ex.what() << '\n';
		return 1;
	}

	if (op == "attention_prefill") {
		std::string fname = tests::format_string("tmp/attention_prefill_layer_", bs, seq, dmodel);
		const char* file = fname.c_str();
		test_attention_prefill(bs, seq, dmodel, file, bench);
	} else if (op == "attention_decode") {
		std::string fname = tests::format_string("tmp/attention_decode_layer_", bs, seq, dmodel);
		const char* file = fname.c_str();
		test_attention_decode(bs, seq, dmodel, file, bench);
	} else if (op == "mlp_prefill") {
		std::string fname = tests::format_string("tmp/mlp_prefill_layer_", bs, seq, dmodel);
		const char* file = fname.c_str();
		test_mlp_prefill(bs, seq, dmodel, file, bench);
	} else if (op == "mlp_decode") {
		std::string fname = tests::format_string("tmp/mlp_decode_layer_", bs, seq, dmodel);
		const char* file = fname.c_str();
		test_mlp_decode(bs, seq, dmodel, file, bench);
	} else if (op == "lm_head_prefill") {
		std::string fname = tests::format_string("tmp/lm_head_prefill_layer_", bs, seq, dmodel);
		const char* file = fname.c_str();
		test_lm_head_prefill(bs, seq, dmodel, file, bench);
	} else if (op == "lm_head_decode") {
		std::string fname = tests::format_string("tmp/lm_head_decode_layer_", bs, seq, dmodel);
		const char* file = fname.c_str();
		test_lm_head_decode(bs, seq, dmodel, file, bench);
	}
}