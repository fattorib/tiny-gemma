#include <chrono>
#include <iostream>
#include <tuple>
#include <algorithm>
#include "argparse.hpp"
#include "tokenizer.hpp"
#include "transformer.hpp"


std::tuple<std::string, std::string> create_passkey(int n_garbage){
    std::random_device rd;
	std::mt19937 gen(rd());

    std::uniform_int_distribution<int> unif(1, n_garbage);
    std::uniform_int_distribution<int> key(1, 50000);

    int n_garbage_prefix = unif(gen);
    int n_garbage_suffix = n_garbage - n_garbage_prefix;

    int passkey = key(gen);

    std::string desc = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there. ";
    std::string garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ";

    int len_garbage = garbage.length();

    std::string information_line = "\nThe pass key is " + std::to_string(passkey) + ". Remember it. " + std::to_string(passkey) + " is the pass key.\n";
    std::string final_question = "\nWhat is the pass key? The pass key is";


    int n_full_prefix = n_garbage_prefix / len_garbage;
    int n_additional_prefix = n_garbage_prefix % len_garbage;
    
    int n_full_suffix = n_garbage_suffix / len_garbage;
    int n_additional_suffix = n_garbage_prefix % len_garbage;

    for (int i = 0; i < n_full_prefix; i++){
        desc.append(garbage);
    }
    desc.append(garbage, 0, n_additional_prefix);

    desc.append(information_line);

    for (int i = 0; i < n_full_suffix; i++){
        desc.append(garbage);
    }
    desc.append(garbage, 0, n_additional_suffix);
    
    desc.append(final_question);

    std::string fpasskey = std::to_string(passkey);

    return {desc, fpasskey};
}

int main(int argc, char** argv) {
	std::string weights_path;
	std::string model_size;
    std::string tokenizer_path = "weights/tokenizer_gemma3.bin";
    int n_garbage;

    int n_decode = 50;

	ArgParser app(argv[0], "CLI Application for Gemma 3 Passkey retrieval.");

	app.add_option("weights_path", weights_path, "File path for model weights.");
	app.add_option("model_size", model_size, "Model size to load.");
	app.add_option("n_garbage", n_garbage, "Total number of garbage characters.");
    app.add_option("tokenizer_path", tokenizer_path, "File path for tokenizer.");


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

	sentencepiece::Tokenizer tokenizer(tokenizer_path, constants::bos_id, constants::eos_id, constants::vocab, true);

    auto passkey_w_prompt = create_passkey(n_garbage);

    std::string prompt = std::get<0>(passkey_w_prompt);
    std::string passkey = std::get<1>(passkey_w_prompt);

	auto t_bench_1 = std::chrono::high_resolution_clock::now();
	auto generations = gemma::generate(prompt, n_decode, &tokenizer, &t, 1.0, 0.0, true);

	auto t_bench_2 = std::chrono::high_resolution_clock::now();
	double s = (t_bench_2 - t_bench_1).count() / 1e9;

    int n_generated_tokens = std::get<0>(generations); 
    std::string generation = std::get<1>(generations); 

    auto found_index = generation.find(passkey);

    if (std::string::npos == found_index){
        printf("\nPasskey=%s not found!\n", passkey.c_str());
    }

    else{
        printf("\nPasskey=%s found!\n", passkey.c_str()
        );
    }


	printf("\n\nGenerating %u tokens took %f seconds. Tok/s: %f ms/token: %f\n", n_generated_tokens, s, n_generated_tokens / s, 1e3 * s / n_generated_tokens);

	return 0;
}