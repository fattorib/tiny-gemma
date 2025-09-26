#pragma once
#include <math.h>

#include <cassert>

#include "config.hpp"
#include "constants.hpp"
#include "simd.hpp"
#include "tensor.hpp"

namespace decode {
using namespace tensor;

struct KVCache {
	activation keys;    // [B,G,S,D]
	activation values;  // [B,G,S,D]

	int position_ptr;
	int buffer_ptr;
	int batch;
	int n_kv;
	int max_pos;
	int head_dim;
	bool is_sw;

	KVCache() = default;

	KVCache(int batch, config::Config *cfg, bool is_sw) : batch(batch), is_sw(is_sw), position_ptr(0), buffer_ptr(0) {
		n_kv = cfg->n_kv;
		max_pos = (is_sw) ? (cfg->n_pos_local) : (cfg->n_pos);
		head_dim = cfg->head_dim;
		keys = activation(batch, n_kv, max_pos, head_dim);
		values = activation(batch, n_kv, max_pos, head_dim);
	}

	// writes a single value at buffer_ptr
	void update_single(activation *key, activation *value);

	// writes a chunk of keys and values coming from the prefill step
	void update_prefill(activation *key, activation *value);

	int valid_pos();
};

int KVCache::valid_pos() {
	return (max_pos > position_ptr) ? position_ptr : max_pos;
}

void KVCache::update_single(activation *key, activation *value) {
	assert((key->x) == (keys.x));
	assert((key->z) == 1);
	assert((value->z) == 1);

	assert(n_kv == key->y);
	assert(head_dim == key->w);

	for (int b = 0; b < (key->x); b++) {
		for (int g = 0; g < n_kv; g++) {
			int offset_cache = b * (keys.x_stride()) + g * (keys.y_stride()) + ((buffer_ptr % max_pos) * (keys.z_stride()));

			int offset_act = b * (key->x_stride()) + g * (key->y_stride());

			simd::storealignedx32((keys.data).get(), (key->data).get(), head_dim, offset_cache, offset_act);
			simd::storealignedx32((values.data).get(), (value->data).get(), head_dim, offset_cache, offset_act);
		}
	}

	buffer_ptr += 1;
	position_ptr += 1;
}

void KVCache::update_prefill(activation *key, activation *value) {
	assert((key->x) == (keys.x));
	assert((value->x) == (values.x));

	assert(key->ndim == 4);
	assert(value->ndim == 4);
	assert(position_ptr == 0);
	assert(buffer_ptr == 0);

	// for sliding window layers, context can be larger than max_pos
	// we compute the starting pos to ensure we write the tail of the prompt to KV$

	int context = (key->z);
	int start_pos = (context >= max_pos) ? (context - max_pos) : 0;

	for (int b = 0; b < (key->x); b++) {
		for (int g = 0; g < n_kv; g++) {
			for (int s = start_pos, c = 0; s < context; s++, c++) {
				int offset_cache = b * (keys.x_stride()) + g * (keys.y_stride()) + c * (keys.z_stride());
				int offset_act = b * (key->x_stride()) + g * (key->y_stride()) + s * (key->z_stride());

				simd::storealignedx32((keys.data).get(), (key->data).get(), head_dim, offset_cache, offset_act);
				simd::storealignedx32((values.data).get(), (value->data).get(), head_dim, offset_cache, offset_act);
			}
		}
	}

	if (start_pos > 0) {
		buffer_ptr += max_pos;
	} else {
		buffer_ptr += context;
	}
	position_ptr += context;
}

struct RoPE {
	weight<float> freqs_cos;
	weight<float> freqs_sin;

	int dhead;
	int max_pos;

	float theta;

	RoPE() = default;

	RoPE(config::Config *cfg, bool is_sw) {
		max_pos = cfg->n_pos;
		theta = (is_sw) ? (cfg->rope_theta_local) : (cfg->rope_theta);
		dhead = cfg->head_dim;

		freqs_cos = weight<float>(max_pos, dhead);
		freqs_sin = weight<float>(max_pos, dhead);

		pre_compute_freqs();
	}

	void apply_rotary_emb(activation *xs, int offset, int qlen);

   private:
	void rotate_half(float *__restrict__ a, const float *__restrict__ b);
	void pre_compute_freqs();
};

void RoPE::apply_rotary_emb(activation *xs, int offset, int qlen) {
	int b, h, d, off_xs, off_sincos;
	assert(xs->z == qlen);

	b = xs->x;
	h = xs->y;
	d = xs->w;

	alignas(constants::alignment) float xs_slice[dhead];
	alignas(constants::alignment) float xs_slice_rot[dhead];

	alignas(constants::alignment) float cos_slice[dhead];
	alignas(constants::alignment) float sin_slice[dhead];

	for (int bs = 0; bs < b; bs++) {
		for (int hs = 0; hs < h; hs++) {
			for (int s = 0; s < qlen; s++) {
				off_xs = (bs * (xs->x_stride())) + (hs * (xs->y_stride())) + (s * (xs->z_stride()));

				off_sincos = (s + offset) * (freqs_cos.row_stride());

				// load cos, sin
				simd::loadalignedx32(cos_slice, freqs_cos.data.get(), dhead, 0, off_sincos);
				simd::loadalignedx32(sin_slice, freqs_sin.data.get(), dhead, 0, off_sincos);

				simd::loadalignedx32(xs_slice, xs->data.get(), dhead, 0, off_xs);

				rotate_half(xs_slice_rot, xs_slice);

				// NOTE: these can be optimized
				for (int d = 0; d < dhead; d++) {
					xs_slice[d] = (cos_slice[d] * xs_slice[d]) + (xs_slice_rot[d] * sin_slice[d]);
				}

				simd::storealignedx32(xs->data.get(), xs_slice, dhead, off_xs, 0);
			}
		}
	}
}

void RoPE::pre_compute_freqs() {
	int dhalf = dhead / 2;

	AlignedArray<float> freqs = make_unique_aligned<float>(dhalf);

	for (int i = 0; i < dhalf; i++) {
		float denom = pow(theta, float(2 * i) / dhead);
		freqs[i] = (1.0 / denom);
	}

	float outer;
	int stidx;

	// outer product
	for (int p = 0; p < max_pos; p++) {
		for (int d = 0; d < dhalf; d++) {
			outer = (p * freqs[d]);
			stidx = p * (freqs_cos.row_stride()) + d * (freqs_cos.col_stride());
			freqs_cos.data[stidx] = cos(outer);
			freqs_cos.data[stidx + dhalf] = cos(outer);

			freqs_sin.data[stidx] = sin(outer);
			freqs_sin.data[stidx + dhalf] = sin(outer);
		}
	}
}

void RoPE::rotate_half(float *__restrict__ a, const float *__restrict__ b) {
	// write b into a

	// NOTE: this can be optimized
	for (int i = 0; i < dhead / 2; i++) {
		a[i] = -1.0 * b[i + (dhead / 2)];
		a[i + (dhead / 2)] = b[i];
	}
}

// Holds the full decode state (excl. the actual activation)
// KVCache + decode activations
struct DecodeState {
	activation xact;  //[B, 1, D]
	activation xr;    // residual [B, 1, D]

	activation xq;  // q [B, 1, H*HD]
	activation xk;  // k [B, 1, G*HD]
	activation xv;  // v [B, 1, G*HD]
	activation xo;  // o [B, 1, H*HD]

	activation xqs;  // qs [B, H, 1, HD]
	activation xks;  // ks [B, G, 1, HD]
	activation xvs;  // vs [B, G, 1, HD]
	activation xos;  // os [B, H, 1, HD]

	activation xup;    // mlp up [B, 1, D']
	activation xgate;  // mlp gate [B, 1, D']

	int batch;
	int hidden_size;
	int intermediate_size;
	int n_layers;
	int n_head;
	int n_kv;
	int head_dim;

	std::unique_ptr<KVCache[]> kvcache;

	DecodeState(int batch, config::Config *cfg) : batch(batch) {
		hidden_size = cfg->hidden_size;
		intermediate_size = cfg->intermediate_size;
		n_layers = cfg->n_layers;
		n_head = cfg->n_head;
		n_kv = cfg->n_kv;
		head_dim = cfg->head_dim;

		int qout_dim, kvout_dim, sw_pattern;
		qout_dim = n_head * head_dim;
		kvout_dim = n_kv * head_dim;
		sw_pattern = cfg->window_pattern;

		xact = activation(batch, 1, hidden_size);
		xr = activation(batch, 1, hidden_size);

		xq = activation(batch, 1, qout_dim);
		xk = activation(batch, 1, kvout_dim);
		xv = activation(batch, 1, kvout_dim);
		xo = activation(batch, 1, qout_dim);

		xqs = activation(batch, n_head, 1, head_dim);
		xks = activation(batch, n_kv, 1, head_dim);
		xvs = activation(batch, n_kv, 1, head_dim);
		xos = activation(batch, n_head, 1, head_dim);

		xup = activation(batch, 1, intermediate_size);
		xgate = activation(batch, 1, intermediate_size);

		kvcache = std::make_unique<KVCache[]>(n_layers);
		for (int l = 0; l < n_layers; l++) {
			bool is_sw = (l % sw_pattern) != (sw_pattern - 1);
			kvcache[l] = KVCache(1, cfg, is_sw);
		}
	}

	// Zero out any activations needed for accumulators
	void zero();
};

void DecodeState::zero() {
	xq.zero();
	xk.zero();
	xv.zero();
}

struct PrefillState {
	activation xact;  //[B, S, D]
	activation xr;    // residual [B, S, D]

	// NOTE: G=1 for these models but keeping in for some flexibility

	activation xq;  // q [B, S, H*HD]
	activation xk;  // k [B, S, G*HD]
	activation xv;  // v [B, S, G*HD]
	activation xo;  // o [B, S, H*HD]

	activation xqs;  // qs [B, H, S, HD]
	activation xks;  // ks [B, G, S, HD]
	activation xvs;  // vs [B, G, S, HD]
	activation xos;  // os [B, H, S, HD]

	activation xup;    // mlp up [B, S, D']
	activation xgate;  // mlp gate [B, S, D']

	int seq;
	int batch;
	int hidden_size;
	int intermediate_size;
	int n_head;
	int n_kv;
	int head_dim;

	PrefillState(int batch, int seq, config::Config *cfg) : batch(batch) {
		hidden_size = cfg->hidden_size;
		intermediate_size = cfg->intermediate_size;
		n_head = cfg->n_head;
		n_kv = cfg->n_kv;
		head_dim = cfg->head_dim;

		int qout_dim, kvout_dim;
		qout_dim = n_head * head_dim;
		kvout_dim = n_kv * head_dim;

		xact = activation(batch, seq, hidden_size);
		xr = activation(batch, seq, hidden_size);

		xq = activation(batch, seq, qout_dim);
		xk = activation(batch, seq, kvout_dim);
		xv = activation(batch, seq, kvout_dim);
		xo = activation(batch, seq, qout_dim);

		xqs = activation(batch, n_head, seq, head_dim);
		xks = activation(batch, n_kv, seq, head_dim);
		xvs = activation(batch, n_kv, seq, head_dim);
		xos = activation(batch, n_head, seq, head_dim);

		xup = activation(batch, seq, intermediate_size);
		xgate = activation(batch, seq, intermediate_size);
	}

	// Zero out any activations needed for accumulators
	void zero();
};

void PrefillState::zero() {
	xq.zero();
	xk.zero();
	xv.zero();
}

}  // namespace decode
