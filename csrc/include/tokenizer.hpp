#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

namespace sentencepiece {

// Tokenizer implementation modified from: https://github.com/karpathy/llama2.c
class Tokenizer {
   public:
	struct SortedToken {
		std::string token;
		int id;

		bool operator<(const SortedToken& other) const {
			return token < other.token;
		}
	};

   private:
	int bos_id;
	int eos_id;
	int vocab_size;
	int max_token_length;
	bool chat_format;

	// https://ai.google.dev/gemma/docs/core/prompt-structure
	int sot_id = 105;  // <start_of_turn>
	int eot_id = 106;  // <end_of_turn>
	int newline_id = 107;

	int user_id = 2364;   // user
	int model_id = 4368;  // model

	std::vector<std::string> vocab;
	std::vector<float> vocab_scores;
	std::vector<SortedToken> sorted_vocab;
	std::vector<std::string> byte_pieces;

	void sort_vocab() {
		sorted_vocab.clear();
		sorted_vocab.reserve(vocab_size);

		for (int i = 0; i < vocab_size; ++i) {
			sorted_vocab.push_back({vocab[i], i});
		}

		std::sort(sorted_vocab.begin(), sorted_vocab.end());
	}

	int str_lookup(const std::string& x) const {
		int s = 0;
		int e = vocab_size - 1;

		while (s <= e) {
			int m = (s + e) / 2;

			if (sorted_vocab[m].token == x) {
				return sorted_vocab[m].id;
			} else if (sorted_vocab[m].token > x) {
				e = m - 1;
			} else {
				s = m + 1;
			}
		}

		return -1;
	}

   public:
	Tokenizer(const std::string& model_path, const int bos_id, const int eos_id, const int vocab_size, bool chat_format) : chat_format(chat_format), bos_id(bos_id), eos_id(eos_id), vocab_size(vocab_size) {
		byte_pieces.resize(512);
		for (int i = 0; i < 256; ++i) {
			byte_pieces[i * 2] = std::to_string(i);
			byte_pieces[i * 2 + 1] = "\0";
		}

		std::ifstream model_file(model_path, std::ios::binary);
		if (!model_file) {
			throw std::runtime_error("Cannot open model file: " + model_path);
		}

		uint32_t max_len;
		model_file.read(reinterpret_cast<char*>(&max_len), sizeof(uint32_t));
		max_token_length = max_len;

		vocab.resize(vocab_size);
		vocab_scores.resize(vocab_size);

		for (int i = 0; i < vocab_size; ++i) {
			float score;
			uint32_t nbytes;

			model_file.read(reinterpret_cast<char*>(&score), sizeof(float));
			model_file.read(reinterpret_cast<char*>(&nbytes), sizeof(uint32_t));

			std::string token(nbytes, '\0');
			model_file.read(token.data(), nbytes);

			vocab[i] = token;
			vocab_scores[i] = score;
		}

		sort_vocab();
	}

	std::vector<int> encode(const std::string& text) {
		std::vector<int> tokens;
		tokens.push_back(bos_id);

		if (chat_format) {
			tokens.insert(tokens.end(), {sot_id, user_id, newline_id});
		}

		const auto& text_bytes = text;
		size_t i = 0;

		std::string str_buffer;

		while (i < text_bytes.length()) {
			uint8_t current_byte = static_cast<uint8_t>(text_bytes[i]);

			// 0xC0 = 11000000, 0x80 = 10000000
			// UTF-8 continuation bytes start with "10" in first two bits
			if ((current_byte & 0xC0) != 0x80) {
				str_buffer.clear();
			}

			str_buffer.push_back(text_bytes[i]);

			if (i + 1 < text_bytes.length() &&
			    (static_cast<uint8_t>(text_bytes[i + 1]) & 0xC0) == 0x80 &&
			    str_buffer.length() < 4) {
				++i;
				continue;
			}

			int token_id = str_lookup(str_buffer);

			if (token_id != -1) {
				tokens.push_back(token_id);
			} else {
				// Byte fallback: +3 offset for <unk>, <s>, </s> tokens
				for (char c : str_buffer) {
					tokens.push_back(static_cast<uint8_t>(c) + 3);
				}
			}

			++i;
		}

		// BPE merging
		while (true) {
			float best_score = -1e10f;
			int best_id = -1;
			int best_idx = -1;

			int ntokens = tokens.size();

			for (int i = 0; i < ntokens - 1; ++i) {
				std::string merged = vocab[tokens[i]] + vocab[tokens[i + 1]];
				int id = str_lookup(merged);

				if (id != -1) {
					float score = vocab_scores[id];
					if (score > best_score) {
						best_score = score;
						best_id = id;
						best_idx = i;
					}
				}
			}

			if (best_idx == -1) {
				break;
			}

			tokens[best_idx] = best_id;

			for (int i = best_idx + 1; i < ntokens - 1; ++i) {
				tokens[i] = tokens[i + 1];
			}

			tokens.pop_back();
		}

		if (chat_format) {
			tokens.insert(tokens.end(), {eot_id, newline_id, sot_id, model_id, newline_id});
		}

		return tokens;
	}

	std::string decode(int token, int prev_token) {
		std::string piece = vocab[token];

		if (prev_token == 1 && !piece.empty() && piece[0] == ' ') {
			piece = piece.substr(1);
		}

		// Byte tokens are encoded as '<0x##>'
		std::regex byte_pattern(R"(<0x[0-9A-Fa-f]{2,3}>)");
		if (std::regex_match(piece, byte_pattern)) {
			std::string hex_str = piece.substr(2, piece.length() - 3);
			int hex_val = std::stoi(hex_str, nullptr, 16);
			piece = byte_pieces[2 * hex_val];
		}

		return piece;
	}

	std::string decode(const std::vector<int>& tokens) {
		std::string result;
		int prev_token = bos_id;

		result += decode(prev_token, -1);

		for (int token : tokens) {
			if (token != bos_id && token != eos_id) {
				result += decode(token, prev_token);
			}
			prev_token = token;
		}

		return result;
	}

	int get_bos_id() const { return bos_id; }
	int get_eos_id() const { return eos_id; }
	int get_eot_id() const { return eot_id; }
	int get_vocab_size() const { return vocab_size; }
};
}  // namespace sentencepiece
