#include <iostream>

#include "argparse.hpp"
#include "constants.hpp"
#include "decode.hpp"
#include "linalg.hpp"
#include "modules.hpp"
#include "test_utils.hpp"
#include "tokenizer.hpp"
#include "transformer.hpp"

int main(int argc, char** argv) {
	int dmodel, n_layers, n_generation_steps, prefill_size;
	std::string weights_path;
	std::string logits_path;

	ArgParser app(argv[0], "CLI Application for testing and benchmarking ops.");
	app.add_option("d_model", dmodel, "Model dimension.");
	app.add_option("n_layers", n_layers, "Numner of layers.");
	app.add_option("n_generation_steps", n_generation_steps, "Number of steps to run argmax decoding for.");
	app.add_option("weights_path", weights_path, "Weight binary path.");
	app.add_option("logits_path", logits_path, "Logit binary path.");
	app.add_option("prefill_size", prefill_size, "Number of prefill tokens.");

	try {
		int rc = app.parse(argc, argv);
		if (rc != 0) return rc;  // parsing failed: help already printed
	} catch (const std::exception& ex) {
		std::cerr << "Parse error: " << ex.what() << '\n';
		return 1;
	}

	config::Config cfg = config::CONFIG_MAP.at("270m");

	gemma::Transformer t(cfg);

	// make the LUT
	lut::GeLUTable table{-4.0, 4.0, 8192};  // note, decent generations even with few samples (~64)

	const char* weights = weights_path.c_str();
	gemma::read_weights(&t, weights);

	printf("Weights loaded!\n");

	auto tokens_data = tensor::make_unique_aligned<int>(prefill_size);
	auto logits_gens = tensor::make_unique_aligned<float>(cfg.n_vocab * n_generation_steps);
	auto logits = tensor::make_unique_aligned<float>(cfg.n_vocab);
	auto tokens_argmax = tensor::make_unique_aligned<int>(n_generation_steps);

	decode::RoPE rope(&cfg, false);
	decode::RoPE rope_sw(&cfg, true);

	decode::DecodeState d(1, &cfg);

	const char* logit_token_path = logits_path.c_str();
	FILE* f = fopen(logit_token_path, "rb");
	size_t success;
	if (f != nullptr) {
		success = fread(tokens_data.get(), 1, sizeof(int) * prefill_size, f);

		for (int g = 0; g < n_generation_steps; g++) {
			success = fread(logits_gens.get() + cfg.n_vocab * g, 1, sizeof(float) * cfg.n_vocab, f);
		}

		success = fread(tokens_argmax.get(), 1, sizeof(int) * n_generation_steps, f);

	} else {
		throw std::runtime_error("Could not open file.");
	}
	fclose(f);
	tensor::tokens tokens(tokens_data, 1, prefill_size);

	sentencepiece::Tokenizer tokenizer("tokenizer_gemma3.bin", constants::bos_id, constants::eos_id, cfg.n_vocab, false);
	std::string prompt = "We introduce Gemma 3, a multimodal addition to the Gemma family of lightweight open models, ranging in scale from 1 to 27 billion parameters. This version introduces vision understanding abilities, a wider coverage of languages and longer context - at least 128K tokens. We also change the architecture of the model to reduce the KV-cache memory that tends to explode with long context. This is achieved by increasing the ratio of local to global attention layers, and keeping the span on local attention short.";
	std::vector<int> ids = tokenizer.encode(prompt);
	int prev_token = ids[ids.size() - 1];

	bool pass;

	pass = true;
	printf("\nTest Tokenizer encode:\n");
	printf("Length of GT tokens %u | Length of tokenized %lu \n", prefill_size, ids.size());
	for (int i = 0; i < prefill_size; i++) {
		if (tokens.data[i] != ids.at(i)) {
			printf("Mismatch at %u \n", i);
			pass = false;
		}
	}
	assert(pass);

	tensor::activation o(1, 1, cfg.n_vocab);

	printf("Activations created!\n");

	gemma::forward_prefill(&o, &tokens, &t, &d, &rope, &rope_sw, &table);

	tensor::tokens tokens_decode(1, 1);
	ops::argmax_logits(&tokens_decode, &o);

	printf("\nTest prefill step 0:\n");
	for (int i = 0; i < cfg.n_vocab; i++) {
		logits[i] = logits_gens[i];
	}

	pass = tests::check_error(o.data, logits, cfg.n_vocab, 1.0, 0.1);
	assert(pass);

	printf("\nTest decode step (%u):\n", 0);
	printf("Ground Truth argmax: %u | cpp argmax: %u \n", tokens_argmax[0], tokens_decode.data[0]);

	int pos_offset = tokens.y;

	for (int step = 1; step < (n_generation_steps); step++, pos_offset++) {
		gemma::forward_decode(&o, &tokens_decode, &t, &d, &rope, &rope_sw, pos_offset, &table);
		ops::argmax_logits(&tokens_decode, &o);
		std::cout << tokenizer.decode(tokens_decode.data[0], prev_token) << std::endl;
		if (tokens_argmax[step] != tokens_decode.data[0]) {
			break;
		}

		for (int i = 0; i < cfg.n_vocab; i++) {
			logits[i] = logits_gens[i + step * cfg.n_vocab];
		}
		pass = tests::check_error(o.data, logits, cfg.n_vocab, 1.0, 0.1);
		assert(pass);
		prev_token = tokens_decode.data[0];
	}

	return 0;
}