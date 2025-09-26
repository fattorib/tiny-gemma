#pragma once
#include <algorithm>
#include <cassert>
#include <chrono>
#include <map>
#include <random>

#include "config.hpp"
#include "constants.hpp"
#include "decode.hpp"
#include "modules.hpp"
#include "ops.hpp"
#include "tensor.hpp"
#include "tokenizer.hpp"

namespace gemma {

using namespace tensor;
using namespace modules;
using namespace config;

struct Transformer {
	QuantArray wte;

	FloatArray RMSw;

	std::unique_ptr<modules::LayerWeights[]> layers;

	QuantArray lm_head;

	Config cfg;

	Transformer(
	    Config cfg) : cfg(cfg) {
		wte = QuantArray(cfg.n_vocab, cfg.hidden_size, true);
		lm_head = QuantArray(cfg.hidden_size, cfg.n_vocab, true);

		RMSw = FloatArray(cfg.hidden_size);

		layers = std::make_unique<LayerWeights[]>(cfg.n_layers);

		for (int l = 0; l < cfg.n_layers; l++) {
			bool is_sw = (l % cfg.window_pattern) != (cfg.window_pattern - 1);
			layers[l] = LayerWeights(cfg.hidden_size, cfg.intermediate_size, cfg.n_head, cfg.n_kv, cfg.head_dim, is_sw);
		}
	}
};

void forward_prefill(tensor::activation *o, tensor::tokens *toks, Transformer *t, decode::DecodeState *d, decode::RoPE *r, decode::RoPE *r_sw, GeLUTable *table) {
	assert(o->ndim == 3);

	size_t prefill_seq, batch;
	batch = toks->x;
	prefill_seq = toks->y;

	decode::PrefillState pf(batch, prefill_seq, &(t->cfg));

	ops::embedding_prefill(&(pf.xact), toks, &(t->wte));

	for (int l = 0; l < (t->cfg.n_layers); l++) {
		if ((t->layers)[l].attn.is_sw) {
			modules::attention(&(pf.xact), &(t->layers)[l].attn, d, &(d->kvcache[l]), &pf, r_sw);
		}

		else {
			modules::attention(&(pf.xact), &(t->layers)[l].attn, d, &(d->kvcache[l]), &pf, r);
		}

		modules::mlp(&(pf.xact), &(t->layers)[l].mlp, d, &pf, table);
	}

	modules::lm_head(o, &(pf.xact), &(t->lm_head), &(t->RMSw));
}

void forward_decode(tensor::activation *o, tensor::tokens *toks, Transformer *t, decode::DecodeState *d, decode::RoPE *r, decode::RoPE *r_sw, size_t pos_offset, GeLUTable *table) {
	assert(o->ndim == 3);

	size_t dmodel, batch;
	batch = toks->x;

	assert(toks->y == 1);

	ops::embedding_decode(&(d->xact), toks, &(t->wte), pos_offset);

	for (int l = 0; l < (t->cfg.n_layers); l++) {
		if ((t->layers)[l].attn.is_sw) {
			modules::attention(&(d->xact), &(t->layers)[l].attn, d, &(d->kvcache[l]), nullptr, r_sw);
		}

		else {
			modules::attention(&(d->xact), &(t->layers)[l].attn, d, &(d->kvcache[l]), nullptr, r);
		}
		modules::mlp(&(d->xact), &(t->layers)[l].mlp, d, nullptr, table);
	}

	modules::lm_head(o, &(d->xact), &(t->lm_head), &(t->RMSw));
}

void read_weights(Transformer *t, const char *file) {
	FILE *f = fopen(file, "rb");

	int hidden = t->cfg.hidden_size;
	int intermediate = t->cfg.intermediate_size;
	int n_head = t->cfg.n_head;
	int n_kv = t->cfg.n_kv;
	int head_dim = t->cfg.head_dim;

	int qo_wsize = (hidden) * (n_head) * (head_dim);
	int kv_wsize = (hidden) * (n_kv) * (head_dim);

	int rms_qksize = (head_dim);
	int rms_wsize = (hidden);

	int mlp_size = (hidden) * (intermediate);

	int down_proj_bsize = (hidden);

	int wte_size = (hidden) * (t->cfg.n_vocab);

	size_t success;
	if (f != nullptr) {
		success = fread((t->wte).data.get(), 1, sizeof(int8_t) * wte_size, f);
		success = fread((t->wte).scales.get(), 1, sizeof(float) * (hidden), f);

		success = fread((t->RMSw).data.get(), 1, sizeof(float) * rms_wsize, f);

		for (int l = 0; l < (t->cfg.n_layers); l++) {
			success = fread((t->layers[l].attn.Qw).data.get(), 1, sizeof(int8_t) * qo_wsize, f);
			success = fread((t->layers[l].attn.Qw).scales.get(), 1, sizeof(float) * (n_head * head_dim), f);

			success = fread((t->layers[l].attn.Kw).data.get(), 1, sizeof(int8_t) * kv_wsize, f);
			success = fread((t->layers[l].attn.Kw).scales.get(), 1, sizeof(float) * (n_kv * head_dim), f);

			success = fread((t->layers[l].attn.Vw).data.get(), 1, sizeof(int8_t) * kv_wsize, f);
			success = fread((t->layers[l].attn.Vw).scales.get(), 1, sizeof(float) * (n_kv * head_dim), f);

			success = fread((t->layers[l].attn.Ow).data.get(), 1, sizeof(int8_t) * qo_wsize, f);
			success = fread((t->layers[l].attn.Ow).scales.get(), 1, sizeof(float) * (hidden), f);

			success = fread((t->layers[l].attn.RMSq).data.get(), 1, sizeof(float) * rms_qksize, f);
			success = fread((t->layers[l].attn.RMSk).data.get(), 1, sizeof(float) * rms_qksize, f);

			success = fread((t->layers[l].attn.RMSw_pre).data.get(), 1, sizeof(float) * rms_wsize, f);
			success = fread((t->layers[l].attn.RMSw_post).data.get(), 1, sizeof(float) * rms_wsize, f);

			success = fread((t->layers[l].mlp.Uw).data.get(), 1, sizeof(int8_t) * mlp_size, f);
			success = fread((t->layers[l].mlp.Uw).scales.get(), 1, sizeof(float) * (intermediate), f);

			success = fread((t->layers[l].mlp.Gw).data.get(), 1, sizeof(int8_t) * mlp_size, f);
			success = fread((t->layers[l].mlp.Gw).scales.get(), 1, sizeof(float) * (intermediate), f);

			success = fread((t->layers[l].mlp.Dw).data.get(), 1, sizeof(int8_t) * mlp_size, f);
			success = fread((t->layers[l].mlp.Dw).scales.get(), 1, sizeof(float) * hidden, f);

			success = fread((t->layers[l].mlp.RMSw_pre).data.get(), 1, sizeof(float) * rms_wsize, f);
			success = fread((t->layers[l].mlp.RMSw_post).data.get(), 1, sizeof(float) * rms_wsize, f);
		}

		success = fread((t->lm_head).data.get(), 1, sizeof(int8_t) * wte_size, f);
		success = fread((t->lm_head).scales.get(), 1, sizeof(float) * (t->cfg.n_vocab), f);

	} else {
		throw std::runtime_error("Could not open file.");
	}
}

std::tuple<int, std::string, float, float> generate(std::string prompt, int n_decode, sentencepiece::Tokenizer *tokenizer, Transformer *t, const float minp, const float temp, const bool terminate_on_end) {
	auto t_prefill_s = std::chrono::high_resolution_clock::now();

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> unif(0.0, 1.0);

	std::vector<int> ids = tokenizer->encode(prompt);

	size_t prompt_len = ids.size();

	std::string decoded_token;

	std::string decoded;
	decoded.reserve(n_decode);

	printf("Prompt length: %lu\n", prompt_len);

	std::cout << tokenizer->decode(ids);

	auto tokens_data = tensor::make_unique_aligned<int>(prompt_len);

	for (int t = 0; t < prompt_len; t++) {
		tokens_data[t] = ids[t];
	}

	decode::RoPE rope(&(t->cfg), false);
	decode::RoPE rope_sw(&(t->cfg), true);

	// (using a LUT is around 2-3% faster e2e)
	lut::GeLUTable table{-4.0, 4.0, 1024};

	decode::DecodeState d(1, &(t->cfg));

	tensor::tokens tokens(tokens_data, 1, prompt_len);
	tensor::activation o(1, 1, t->cfg.n_vocab);

	forward_prefill(&o, &tokens, t, &d, &rope, &rope_sw, &table);

	tensor::tokens tokens_decode(1, 1);

	if (minp < 1) {
		float randf = unif(gen);
		ops::minp_logits(&tokens_decode, &o, minp, temp, randf);
	} else {
		ops::argmax_logits(&tokens_decode, &o);
	}

	int prev_token = ids[ids.size() - 1];

	std::cout << tokenizer->decode(tokens_decode.data[0], prev_token) << std::flush;
	prev_token = tokens_decode.data[0];

	auto t_prefill_e = std::chrono::high_resolution_clock::now();

	int pos_offset = tokens.y;

	auto t_decode_s = std::chrono::high_resolution_clock::now();

	for (int step = 0; step < n_decode; step++, pos_offset++) {
		forward_decode(&o, &tokens_decode, t, &d, &rope, &rope_sw, pos_offset, &table);

		if (minp < 1) {
			float randf = unif(gen);
			ops::minp_logits(&tokens_decode, &o, minp, temp, randf);
		} else {
			ops::argmax_logits(&tokens_decode, &o);
		}

		if (terminate_on_end) {
			if (tokens_decode.data[0] == tokenizer->get_eot_id()) {
				break;
			}
		}

		decoded_token = tokenizer->decode(tokens_decode.data[0], prev_token);

		decoded.append(decoded_token);

		std::cout << decoded_token << std::flush;
		decoded_token = tokenizer->decode(tokens_decode.data[0], prev_token);

		prev_token = tokens_decode.data[0];
	}

	auto t_decode_e = std::chrono::high_resolution_clock::now();

	double s_prefill = (t_prefill_e - t_prefill_s).count() / 1e9;
	double s_decode = (t_decode_e - t_decode_s).count() / 1e9;

	return {pos_offset - int(prompt_len), decoded, s_prefill, s_decode};
}

}  // namespace gemma
