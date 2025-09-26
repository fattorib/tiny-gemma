#include <iostream>

#include "argparse.hpp"
#include "tokenizer.hpp"
#include "transformer.hpp"

int main(int argc, char** argv) {
	std::string weights_path;
	std::string model_size;
	std::string prompt;
	int n_decode;
	float minp = 2.0;
	float temp;
	int chat_format;
	int terminate_on_eos;

	ArgParser app(argv[0], "CLI Application for Gemma 3 text generation.");

	app.add_option("weights_path", weights_path, "Load path for cpp weights.");
	app.add_option("model_size", model_size, "Model size to load.");
	app.add_option("prompt", prompt, "The prompt.");
	app.add_option("n_dec", n_decode, "Number of tokens to generate.");
	app.add_option("minp", minp, "Minp sampler, defaults to greedy sampling if `minp >= 1`.");
	app.add_option("temp", temp, "Sampler temperature.");
	app.add_option("chat_format", chat_format, "Flag to enable using chat template.");
	app.add_option("terminate_on_eos", terminate_on_eos, "Flag to enable early termination if model generates an EOS/EOT token.");

	try {
		int rc = app.parse(argc, argv);
		if (rc != 0) return rc;
	} catch (const std::exception& ex) {
		std::cerr << "Parse error: " << ex.what() << '\n';
		return 1;
	}

	gemma::Config config = gemma::CONFIG_MAP.at(model_size);

	gemma::Transformer t(config);

	gemma::read_weights(&t, weights_path.c_str());
	printf("Weights loaded!\n");

	sentencepiece::Tokenizer tokenizer("weights/tokenizer_gemma3.bin", constants::bos_id, constants::eos_id, constants::vocab, bool(chat_format));

	auto generations = gemma::generate(prompt, n_decode, &tokenizer, &t, minp, temp, bool(terminate_on_eos));

	int n_generated_tokens = std::get<0>(generations);

	int n_prompt_tokens = tokenizer.encode(prompt).size();

	float total_s = std::get<2>(generations) + std::get<3>(generations);

	float prefill_tok_s = n_prompt_tokens / std::get<2>(generations);
	float decode_tok_s = n_generated_tokens / std::get<3>(generations);

	printf("\n\nGenerating %u tokens took %f seconds. Prefill Tok/s: %f  Decode Tok/s %f ms/token: %f\n", n_generated_tokens, total_s, prefill_tok_s, decode_tok_s, 1e3 * std::get<3>(generations) / n_generated_tokens);

	return 0;
}