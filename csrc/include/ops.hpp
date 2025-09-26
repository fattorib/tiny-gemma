#pragma once
#include <math.h>

#include <cassert>
#include <iostream>

#include "constants.hpp"
#include "decode.hpp"
#include "immintrin.h"
#include "simd.hpp"
#include "static_switch.hpp"
#include "tensor.hpp"

namespace lut {

inline float geluf(float a) {
	const float tmp0 = 0.7978845608;  // sqrt(2 / pi)
	const float tmp1 = 0.044715;
	return (0.5 * a) * (1 + std::tanh(tmp0 * (a + tmp1 * std::pow(a, 3.0))));
}

struct GeLUTable {
	float min;
	float max;
	size_t table_size;
	tensor::AlignedArray<float> table;
	float eps;

	GeLUTable(float min, float max, size_t table_size)
	    : min(min), max(max), table_size(table_size) {
		table = tensor::make_unique_aligned<float>(table_size);

		eps = (float)(max - min) / float(table_size);

		for (int i = 0; i < table_size; i++) {
			float x = float(min) + i * eps;
			table[i] = geluf(x);
		}
	}

	inline float lookup(float x) {
		int idx = int((x - min) / eps);
		if (idx < 0) {
			return 0;
		}
		if (idx >= table_size) {
			return x;
		}

		return table[idx];
	}
};

}  // namespace lut

namespace ops {

using namespace tensor;
using namespace decode;
using namespace lut;

void gelu(activation *a, GeLUTable *table) {
	int numel = a->numel();

	for (size_t idx = 0; idx < numel; idx += 8) {
		(a->data)[idx + 0] = (table->lookup((a->data)[idx + 0]));
		(a->data)[idx + 1] = (table->lookup((a->data)[idx + 1]));
		(a->data)[idx + 2] = (table->lookup((a->data)[idx + 2]));
		(a->data)[idx + 3] = (table->lookup((a->data)[idx + 3]));
		(a->data)[idx + 4] = (table->lookup((a->data)[idx + 4]));
		(a->data)[idx + 5] = (table->lookup((a->data)[idx + 5]));
		(a->data)[idx + 6] = (table->lookup((a->data)[idx + 6]));
		(a->data)[idx + 7] = (table->lookup((a->data)[idx + 7]));
	}
}

// computes a <- a + b without broadcasting
void add(activation *a, activation *b) {
	assert(a->numel() == b->numel());

	int numel = a->numel();

	__m256 tmpa[4] = {};
	__m256 tmpb[4] = {};

	constexpr int UNROLL = 4 * constants::simd;

	for (int i = 0; i < numel; i += UNROLL) {
		simd::loadalignedvx4(tmpa, (a->data).get(), i);
		simd::loadalignedvx4(tmpb, (b->data).get(), i);

		tmpa[0] = _mm256_add_ps(tmpa[0], tmpb[0]);
		tmpa[1] = _mm256_add_ps(tmpa[1], tmpb[1]);
		tmpa[2] = _mm256_add_ps(tmpa[2], tmpb[2]);
		tmpa[3] = _mm256_add_ps(tmpa[3], tmpb[3]);

		simd::storealignedvx4((a->data).get(), tmpa, i);
	}
}

// computes a <- a * b without broadcasting
void mul(activation *a, activation *b) {
	assert(a->numel() == b->numel());

	int numel = a->numel();

	__m256 tmpa[4] = {};
	__m256 tmpb[4] = {};

	constexpr int UNROLL = 4 * constants::simd;

	for (int i = 0; i < numel; i += UNROLL) {
		simd::loadalignedvx4(tmpa, (a->data).get(), i);
		simd::loadalignedvx4(tmpb, (b->data).get(), i);

		tmpa[0] = _mm256_mul_ps(tmpa[0], tmpb[0]);
		tmpa[1] = _mm256_mul_ps(tmpa[1], tmpb[1]);
		tmpa[2] = _mm256_mul_ps(tmpa[2], tmpb[2]);
		tmpa[3] = _mm256_mul_ps(tmpa[3], tmpb[3]);

		simd::storealignedvx4((a->data).get(), tmpa, i);
	}
}

// applies RMSNorm the provided activation slice (loads `dim` elements starting from `act + off`)
void rmsnorm_along_dim(float *act, float *weight_cache, int dim, int off, float eps) {
	constexpr int UNROLL = 4 * constants::simd;
	alignas(constants::alignment) float act_cache[dim];

	simd::loadalignedx32(act_cache, act, dim, 0, off);

	// compute E[x**2]
	__m256 ms0 = _mm256_setzero_ps();
	__m256 ms1 = _mm256_setzero_ps();
	__m256 ms2 = _mm256_setzero_ps();
	__m256 ms3 = _mm256_setzero_ps();

	__m256 acts[4];

	for (int d = 0; d < dim; d += UNROLL) {
		simd::loadalignedvx4(acts, act_cache, d);

		ms0 = _mm256_add_ps(ms0, _mm256_mul_ps(acts[0], acts[0]));
		ms1 = _mm256_add_ps(ms1, _mm256_mul_ps(acts[1], acts[1]));
		ms2 = _mm256_add_ps(ms2, _mm256_mul_ps(acts[2], acts[2]));
		ms3 = _mm256_add_ps(ms3, _mm256_mul_ps(acts[3], acts[3]));
	}

	float x_ms = simd::reduce_horizontalx4(ms0, ms1, ms2, ms3) / dim;

	ms0 = _mm256_broadcast_ss(&x_ms);

	const float x_irms = 1.0 / (std::sqrt(x_ms + eps));

	__m256 v_x_irms = _mm256_broadcast_ss(&x_irms);

	__m256 weight[4];

	__m256 v_one = _mm256_set1_ps(1.0);

	for (int d = 0; d < dim; d += UNROLL) {
		simd::loadalignedvx4(acts, act_cache, d);

		simd::loadalignedvx4(weight, weight_cache, d);

		acts[0] = _mm256_mul_ps(acts[0], v_x_irms);
		acts[0] = _mm256_mul_ps(acts[0], _mm256_add_ps(weight[0], v_one));

		acts[1] = _mm256_mul_ps(acts[1], v_x_irms);
		acts[1] = _mm256_mul_ps(acts[1], _mm256_add_ps(weight[1], v_one));

		acts[2] = _mm256_mul_ps(acts[2], v_x_irms);
		acts[2] = _mm256_mul_ps(acts[2], _mm256_add_ps(weight[2], v_one));

		acts[3] = _mm256_mul_ps(acts[3], v_x_irms);
		acts[3] = _mm256_mul_ps(acts[3], _mm256_add_ps(weight[3], v_one));

		simd::storealignedvx4(act, acts, off + d);
	}
}

void rmsnorm(activation *a, weight<float> *w, float eps = 1e-06) {
	int bs, sq, hs, dim, off;

	if (a->ndim == 3) {
		bs = (a->x);
		sq = (a->y);
		dim = (a->z);
		hs = 0;

	} else {
		assert(a->ndim == 4);
		bs = (a->x);
		hs = (a->y);
		sq = (a->z);
		dim = (a->w);
	}

	alignas(constants::alignment) float w_cache[dim];

	simd::loadalignedx32(w_cache, (w->data).get(), dim, 0, 0);

	if (a->ndim == 3) {
		for (int b = 0; b < bs; b++) {
			for (int s = 0; s < sq; s++) {
				off = (b * a->x_stride()) + (s * a->y_stride());
				rmsnorm_along_dim((a->data).get(), w_cache, dim, off, eps);
			}
		}
	}

	else if (a->ndim == 4) {
		for (int b = 0; b < bs; b++) {
			for (int h = 0; h < hs; h++) {
				for (int s = 0; s < sq; s++) {
					off = (b * a->x_stride()) + (h * a->y_stride()) + (s * a->z_stride());
					rmsnorm_along_dim((a->data).get(), w_cache, dim, off, eps);
				}
			}
		}
	}
}

void attention_decode(activation *o, activation *q, decode::KVCache *kv) {
	int heads, dim, bs, groups, qseq, seq, offoq, offk, offv, g;
	activation *k = &(kv->keys);
	activation *v = &(kv->values);

	bs = q->x;
	heads = q->y;
	dim = q->w;
	groups = k->y;

	qseq = q->z;

	seq = kv->valid_pos();

	assert(qseq == 1);

	assert(dim == (k->w));
	assert(dim == (v->w));

	assert(groups == (v->y));

	assert(bs == (v->x));
	assert(bs == (k->x));

	alignas(constants::alignment) float q_tile[heads][dim];
	alignas(constants::alignment) float o_tile[heads][dim];

	alignas(constants::alignment) float v_tile[dim];
	alignas(constants::alignment) float k_tile[dim];

	int ldkt = k->z_stride();
	int ldv = v->z_stride();

	const float scale = 1.0 / std::sqrt(float(dim));

	int heads_per_group = heads / groups;

	constexpr int UNROLL = 4 * constants::simd;

	int qs = 0;

	float ms[heads];
	float ls[heads];

	for (int b = 0; b < bs; b++) {
		for (int hidx = 0; hidx < heads; hidx++) {
			ms[hidx] = -INFINITY;
			ls[hidx] = 0.0;

			for (int d = 0; d < dim; d++) {
				o_tile[hidx][d] = 0.0;
			}
		}

		__m256 vscale = _mm256_set1_ps(scale);

		for (int hidx = 0; hidx < heads; hidx++) {
			offoq = b * (q->x_stride()) + hidx * (q->y_stride());
			simd::loadalignedx32(q_tile[hidx], (q->data).get(), dim, 0, offoq + (qs * (q->z_stride())));

#pragma unroll
			for (int d = 0; d < dim; d += constants::simd) {
				*(__m256 *)(&q_tile[hidx][d]) = _mm256_mul_ps(*(__m256 *)(&q_tile[hidx][d]), vscale);
			}
		}

		for (int g = 0; g < groups; g++) {
			offk = b * (k->x_stride()) + g * (k->y_stride());
			offv = b * (v->x_stride()) + g * (v->y_stride());

			for (int kidx = 0; kidx < seq; kidx++) {
				// load K_i, V_i
				simd::loadalignedx32(k_tile, (k->data).get(), dim, 0, offk + (kidx * (k->z_stride())));
				simd::loadalignedx32(v_tile, (v->data).get(), dim, 0, offv + (kidx * (v->z_stride())));

				for (int hidx = 0; hidx < heads; hidx++) {
					__m256 tmpq[4];
					__m256 tmpk[4];
					__m256 tmpacc[4] = {};

					// dot(q, K_i)
					for (int d = 0; d < dim; d += UNROLL) {
						simd::loadalignedvx4(tmpq, q_tile[hidx], d);
						simd::loadalignedvx4(tmpk, k_tile, d);
						tmpacc[0] = _mm256_fmadd_ps(tmpq[0], tmpk[0], tmpacc[0]);
						tmpacc[1] = _mm256_fmadd_ps(tmpq[1], tmpk[1], tmpacc[1]);
						tmpacc[2] = _mm256_fmadd_ps(tmpq[2], tmpk[2], tmpacc[2]);
						tmpacc[3] = _mm256_fmadd_ps(tmpq[3], tmpk[3], tmpacc[3]);
					}

					float s = simd::reduce_horizontalx4(tmpacc[0], tmpacc[1], tmpacc[2], tmpacc[3]);

					float m = std::fmaxf(ms[hidx], s);

					float corr = std::exp(ms[hidx] - m);
					float update = std::exp(s - m);

					const float l_new = (corr * ls[hidx]) + (update);
					const float il_new = 1.0 / l_new;

					__m256 vl_new_recip = _mm256_set1_ps(il_new);
					__m256 vls = _mm256_set1_ps(ls[hidx]);
					__m256 vcorr = _mm256_set1_ps(corr);
					__m256 vupdate = _mm256_set1_ps(update);
					__m256 vv_tile[4];

					__m256 o_val[4];

					for (int d = 0; d < dim; d += UNROLL) {
						simd::loadalignedvx4(o_val, &(o_tile[hidx][d]), 0);
						simd::loadalignedvx4(vv_tile, &v_tile[d], 0);

						o_val[0] = _mm256_mul_ps(vls, o_val[0]);
						o_val[0] = _mm256_mul_ps(vcorr, o_val[0]);
						o_val[0] = _mm256_fmadd_ps(vupdate, vv_tile[0], o_val[0]);
						o_val[0] = _mm256_mul_ps(o_val[0], vl_new_recip);

						o_val[1] = _mm256_mul_ps(vls, o_val[1]);
						o_val[1] = _mm256_mul_ps(vcorr, o_val[1]);
						o_val[1] = _mm256_fmadd_ps(vupdate, vv_tile[1], o_val[1]);
						o_val[1] = _mm256_mul_ps(o_val[1], vl_new_recip);

						o_val[2] = _mm256_mul_ps(vls, o_val[2]);
						o_val[2] = _mm256_mul_ps(vcorr, o_val[2]);
						o_val[2] = _mm256_fmadd_ps(vupdate, vv_tile[2], o_val[2]);
						o_val[2] = _mm256_mul_ps(o_val[2], vl_new_recip);

						o_val[3] = _mm256_mul_ps(vls, o_val[3]);
						o_val[3] = _mm256_mul_ps(vcorr, o_val[3]);
						o_val[3] = _mm256_fmadd_ps(vupdate, vv_tile[3], o_val[3]);
						o_val[3] = _mm256_mul_ps(o_val[3], vl_new_recip);

						simd::storealignedvx4(&(o_tile[hidx][d]), o_val, 0);
					}

					ms[hidx] = m;
					ls[hidx] = l_new;
				}
			}

			for (int hidx = 0; hidx < heads; hidx++) {
				offoq = b * (q->x_stride()) + hidx * (q->y_stride());
				simd::storealignedx32((o->data).get(), (o_tile[hidx]), dim, offoq, 0);
			}
		}
	}
}

// attention prefill step using the sequential alg from https://arxiv.org/abs/2112.05682
// NOTE: Actually faster than the GEMM based approach
// o,q are [B,H,S,D]
// k is [B,G,S,D] --> row major
// v is [B,G,S,D] --> row major
void attention_prefill(activation *o, activation *q, activation *k, activation *v, bool is_sw, int sw_context) {
	int bs, heads, groups, dim, qseq, seq, offoq, offk, offv, g;

	bs = q->x;
	heads = q->y;
	groups = k->y;
	dim = q->w;

	qseq = q->z;

	assert(dim == (k->w));
	assert(dim == (v->w));

	assert(qseq == (v->z));
	assert(qseq == (k->z));

	assert(groups == (v->y));
	assert(groups == 1);

	assert(bs == (v->x));
	assert(bs == (k->x));

	alignas(constants::alignment) float q_tile[dim];
	alignas(constants::alignment) float v_acc_tile[dim];
	alignas(constants::alignment) float v_tile[dim];
	alignas(constants::alignment) float k_tile[dim];

	int heads_per_group = heads / groups;

	const float scale = 1.0 / std::sqrt(float(dim));

	constexpr int UNROLL = 4 * constants::simd;

	for (int b = 0; b < bs; b++) {
		for (int h = 0; h < heads; h++) {
			// offset pointers to beginning of batch x head
			offoq = b * (q->x_stride()) + h * (q->y_stride());

			g = h / heads_per_group;

			offk = b * (k->x_stride()) + g * (k->y_stride());
			offv = b * (v->x_stride()) + g * (v->y_stride());

			for (int qs = 0; qs < qseq; qs++) {
				// load qs, this will persist in cache

				simd::loadalignedx32(q_tile, (q->data).get(), dim, 0, offoq + (qs * (q->z_stride())));

				for (int d = 0; d < dim; d += UNROLL) {
					*(__m256 *)(&v_acc_tile[d + 0]) = _mm256_setzero_ps();
					*(__m256 *)(&v_acc_tile[d + 8]) = _mm256_setzero_ps();
					*(__m256 *)(&v_acc_tile[d + 16]) = _mm256_setzero_ps();
					*(__m256 *)(&v_acc_tile[d + 24]) = _mm256_setzero_ps();
				}

				// s* and m* in paper
				float s_star = 0.0f;
				float m_star = -INFINITY;
				float m;

				int ks_swa = 0;

				if (is_sw) {
					ks_swa = (qs >= sw_context) ? qs - sw_context : 0;
				}

				for (int ks = ks_swa; ks <= qs; ks++) {
					simd::loadalignedx32(k_tile, (k->data).get(), dim, 0, offk + (ks * (k->z_stride())));
					simd::loadalignedx32(v_tile, (v->data).get(), dim, 0, offv + (ks * (v->z_stride())));

					__m256 tmpq[4];
					__m256 tmpk[4];
					__m256 tmpacc[4] = {};

					for (int d = 0; d < dim; d += UNROLL) {
						simd::loadalignedvx4(tmpq, q_tile, d);
						simd::loadalignedvx4(tmpk, k_tile, d);

						tmpacc[0] = _mm256_fmadd_ps(tmpq[0], tmpk[0], tmpacc[0]);
						tmpacc[1] = _mm256_fmadd_ps(tmpq[1], tmpk[1], tmpacc[1]);
						tmpacc[2] = _mm256_fmadd_ps(tmpq[2], tmpk[2], tmpacc[2]);
						tmpacc[3] = _mm256_fmadd_ps(tmpq[3], tmpk[3], tmpacc[3]);
					}

					float s = simd::reduce_horizontalx4(tmpacc[0], tmpacc[1], tmpacc[2], tmpacc[3]);

					s = s * scale;
					m = std::fmaxf(m_star, s);

					// float corr = std::exp(m_star - m);
					// float update = std::exp(s - m);

					float corr;
					float update;

					if (s > m_star) {
						corr = std::exp(m_star - s);
						update = 1;
					} else {
						corr = 1;
						update = exp(s - m_star);
					}

					__m256 vcorr = _mm256_broadcast_ss(&corr);
					__m256 vupdate = _mm256_broadcast_ss(&update);

					__m256 vacc[4];
					__m256 vv[4];

					for (int d = 0; d < dim; d += UNROLL) {
						simd::loadalignedvx4(vacc, v_acc_tile, d);
						simd::loadalignedvx4(vv, v_tile, d);

						vacc[0] = _mm256_fmadd_ps(vv[0], vupdate,
						                          _mm256_mul_ps(vacc[0], vcorr));
						vacc[1] = _mm256_fmadd_ps(vv[1], vupdate,
						                          _mm256_mul_ps(vacc[1], vcorr));
						vacc[2] = _mm256_fmadd_ps(vv[2], vupdate,
						                          _mm256_mul_ps(vacc[2], vcorr));
						vacc[3] = _mm256_fmadd_ps(vv[3], vupdate,
						                          _mm256_mul_ps(vacc[3], vcorr));
						simd::storealignedvx4(v_acc_tile, vacc, d);
					}

					s_star = (s_star * corr) + update;

					m_star = m;
				}

				const float i_s_star = (1.0 / s_star);

				__m256 v_i_s_star = _mm256_broadcast_ss(&i_s_star);
				__m256 vv[4];

				for (int d = 0; d < dim; d += UNROLL) {
					simd::loadalignedvx4(vv, v_acc_tile, d);

					vv[0] = _mm256_mul_ps(vv[0], v_i_s_star);
					vv[1] = _mm256_mul_ps(vv[1], v_i_s_star);
					vv[2] = _mm256_mul_ps(vv[2], v_i_s_star);
					vv[3] = _mm256_mul_ps(vv[3], v_i_s_star);

					simd::storealignedvx4(v_acc_tile, vv, d);
				}

				simd::storealignedx32((o->data).get(), v_acc_tile, dim, offoq + (qs * (q->z_stride())), 0);
			}
		}
	}
}

// Compute Token embeddings during prefill
void embedding_prefill(activation *o, tokens *t, weight<int8_t> *wte) {
	int bs = (t->x);
	int context = (t->y);

	int dims = wte->cols;

	int ldwte = wte->row_stride();

	const float scale = sqrt(dims);

	__m256 scale_v = _mm256_set1_ps(scale);

	int offo, offt, pos, token;
	for (int b = 0; b < bs; b++) {
		for (int s = 0; s < context; s++) {
			offo = b * (o->x_stride()) + s * (o->y_stride());
			offt = b * (t->x_stride()) + s;

			pos = s;
			token = (t->data)[offt];

			__m256 tmpb[4] = {};
			__m256 tmpc[4] = {};
			__m256 tmps[4] = {};

			constexpr int UNROLL = 4 * constants::simd;
			int offwte = token * ldwte;

			for (int d = 0; d < dims; d += UNROLL) {
				load_i8x32(tmpb, (wte->data).get(), offwte + d);

				simd::loadalignedvx4(tmps, (wte->scales).get(), d);

				tmpc[0] = _mm256_mul_ps(tmps[0], _mm256_mul_ps(scale_v, tmpb[0]));
				tmpc[1] = _mm256_mul_ps(tmps[1], _mm256_mul_ps(scale_v, tmpb[1]));
				tmpc[2] = _mm256_mul_ps(tmps[2], _mm256_mul_ps(scale_v, tmpb[2]));
				tmpc[3] = _mm256_mul_ps(tmps[3], _mm256_mul_ps(scale_v, tmpb[3]));

				simd::storealignedvx4((o->data).get(), tmpc, offo + d);
			}
		}
	}
}

// Compute Token + Position embeddings during decode step
void embedding_decode(activation *o, tokens *t, weight<int8_t> *wte, size_t pos_offset) {
	int bs = (t->x);

	int dims = wte->cols;

	assert((o->y) == 1);
	assert((t->y) == 1);

	int ldwte = wte->row_stride();

	int offo, offt, pos, token;

	const float scale = sqrt(dims);

	__m256 scale_v = _mm256_set1_ps(scale);

	for (int b = 0; b < bs; b++) {
		offo = b * (o->x_stride());
		offt = b * (t->x_stride());

		token = (t->data)[offt];

		__m256 tmpb[4] = {};
		__m256 tmpc[4] = {};
		__m256 tmps[4] = {};

		constexpr int UNROLL = 4 * constants::simd;
		int offwte = token * ldwte;

		for (int d = 0; d < dims; d += UNROLL) {
			load_i8x32(tmpb, (wte->data).get(), offwte + d);

			simd::loadalignedvx4(tmps, (wte->scales).get(), d);

			tmpc[0] = _mm256_mul_ps(tmps[0], _mm256_mul_ps(scale_v, tmpb[0]));
			tmpc[1] = _mm256_mul_ps(tmps[1], _mm256_mul_ps(scale_v, tmpb[1]));
			tmpc[2] = _mm256_mul_ps(tmps[2], _mm256_mul_ps(scale_v, tmpb[2]));
			tmpc[3] = _mm256_mul_ps(tmps[3], _mm256_mul_ps(scale_v, tmpb[3]));

			simd::storealignedvx4((o->data).get(), tmpc, offo + d);
		}
	}
}

void linear_prefill(activation *o, activation *a, weight<int8_t> *w) {
	size_t bs, seq, din, dout;
	bs = a->x;

	assert(bs == 1);
	assert(a->ndim == 3);
	assert(w->ndim == 2);

	seq = a->y;

	din = a->z;
	dout = o->z;

	gemm::launch_qgemm((o->data).get(), (a->data).get(), (w->data).get(), (w->scales).get(), seq, din, dout);
}

void linear_decode(activation *o, activation *a, weight<int8_t> *w) {
	size_t bs, seq, din, dout;
	bs = a->x;
	seq = a->y;

	assert(bs == 1);
	assert(seq == 1);
	assert(a->ndim == 3);
	assert(w->ndim == 2);

	din = (a->z);
	dout = (w->cols);
	assert(din == (w->rows));

	gemv::launch_qgemv((o->data).get(), (a->data).get(), (w->data).get(), (w->scales).get(), din, dout);
}

// o is [...,V]
// a is [...,d]
// w is [V, d]
void lm_head_decode(activation *o, activation *a, weight<int8_t> *w) {
	size_t bs, seq, din, dout;
	bs = a->x;
	seq = a->y;

	assert(bs == 1);
	assert(seq == 1);
	assert(a->ndim == 3);
	assert(w->ndim == 2);

	din = (a->z);
	dout = (w->cols);  // row major
	gemv::launch_qgemv_logits((o->data).get(), (a->data).get(), (w->data).get(), (w->scales).get(), din, dout);
}

//  performs spilt and rearrange: [B,L,(N*H)] -> [B, N, L, H]
void split_rearrange(activation *o, const activation *a, int nh) {
	assert(a->ndim == 3);

	int bs = a->x;
	int sq = a->y;
	int d_model = a->z;
	int hd = d_model / nh;

	int ldr, ldc;
	int str, stc;

	ldr = d_model;
	ldc = hd;

	str = sq * hd;
	stc = hd;

	int offs_ld, offs_st;

	__m256 tmp[4];
	constexpr int UNROLL = 4 * constants::simd;

	assert(bs == 1);

	for (int row = 0; row < sq; row++) {
		for (int col = 0; col < nh; col++) {
			offs_ld = row * ldr + col * ldc;
			offs_st = col * str + row * stc;

			for (int dim = 0; dim < hd; dim += UNROLL) {
				simd::loadalignedvx4(tmp, (a->data).get(), offs_ld + dim);
				simd::storealignedvx4((o->data).get(), tmp, offs_st + dim);
			}
		}
	}
}

// Performs [B, N, L, H] -> [B, L, (N*H)]
void fuse_rearrange(activation *__restrict__ o, const activation *a) {
	assert(a->ndim == 4);

	int bs = a->x;
	int nh = a->y;
	int sq = a->z;
	int hd = a->w;

	int ldr, ldc;
	int str, stc;

	ldr = sq * hd;
	ldc = hd;

	str = nh * hd;
	stc = hd;

	int offs_ld, offs_st;

	__m256 tmp[4];

	constexpr int UNROLL = 4 * constants::simd;

	for (int row = 0; row < nh; row++) {
		for (int col = 0; col < sq; col++) {
			offs_ld = row * ldr + col * ldc;
			offs_st = col * str + row * stc;

			for (int dim = 0; dim < hd; dim += UNROLL) {
				simd::loadalignedvx4(tmp, (a->data).get(), offs_ld + dim);
				simd::storealignedvx4((o->data).get(), tmp, offs_st + dim);
			}
		}
	}
}

void argmax_logits(tokens *toks, activation *o) {
	assert(toks->x == 1);
	assert(o->ndim == 3);
	assert(o->y == 1);

	float max_logits = -INFINITY;
	int argmax = -1;

	for (int i = 0; i < constants::vocab; i++) {
		if (o->data[i] > max_logits) {
			argmax = i;
			max_logits = (o->data[i]);
		}
	}

	(toks->data[0]) = int(argmax);
}

void minp_logits(tensor::tokens *toks, tensor::activation *o, float minp, float temp, float randf) {
	assert(toks->x == 1);
	assert(o->ndim == 3);
	assert(o->y == 1);

	for (int idx = 0; idx < constants::vocab; idx++) {
		o->data[idx] /= temp;
	}

	constexpr int UNROLL = 4 * constants::simd;

	float rowmax = -INFINITY;

	// compute the softmax
	__m256 rowmax_vs[4] = {_mm256_set1_ps(rowmax), _mm256_set1_ps(rowmax), _mm256_set1_ps(rowmax), _mm256_set1_ps(rowmax)};
	__m256 act_vs[4];

	for (int i = 0; i < constants::vocab; i += UNROLL) {
		simd::loadalignedvx4(act_vs, (o->data).get(), i);

		rowmax_vs[0] = _mm256_max_ps(rowmax_vs[0], act_vs[0]);
		rowmax_vs[1] = _mm256_max_ps(rowmax_vs[1], act_vs[1]);
		rowmax_vs[2] = _mm256_max_ps(rowmax_vs[2], act_vs[2]);
		rowmax_vs[3] = _mm256_max_ps(rowmax_vs[3], act_vs[3]);
	}

	rowmax = simd::max_horizontalx4(rowmax_vs[0], rowmax_vs[1], rowmax_vs[2], rowmax_vs[3]);
	float maxp = -INFINITY;

	for (int i = 0; i < constants::vocab; i++) {
		o->data[i] = std::exp((o->data[i] - rowmax));

		if ((o->data[i]) > maxp) {
			maxp = o->data[i];
		}
	}

	float denom = 0.0f;
	float cut = minp * maxp;

	__m256 cut_v = _mm256_set1_ps(cut);
	__m256 denom_vs[4] = {};
	__m256 zeros = _mm256_setzero_ps();

	// compute the denominator (\sum {x_k} with masking on token indices with values below `cut` )
	for (int i = 0; i < constants::vocab; i += UNROLL) {
		simd::loadalignedvx4(act_vs, (o->data).get(), i);

		__m256 mask0 = _mm256_cmp_ps(act_vs[0], cut_v, 1);
		act_vs[0] = _mm256_blendv_ps(act_vs[0], zeros, mask0);

		__m256 mask1 = _mm256_cmp_ps(act_vs[1], cut_v, 1);
		act_vs[1] = _mm256_blendv_ps(act_vs[1], zeros, mask1);

		__m256 mask2 = _mm256_cmp_ps(act_vs[2], cut_v, 1);
		act_vs[2] = _mm256_blendv_ps(act_vs[2], zeros, mask2);

		__m256 mask3 = _mm256_cmp_ps(act_vs[3], cut_v, 1);
		act_vs[3] = _mm256_blendv_ps(act_vs[3], zeros, mask3);

		denom_vs[0] = _mm256_add_ps(denom_vs[0], act_vs[0]);
		denom_vs[1] = _mm256_add_ps(denom_vs[1], act_vs[1]);
		denom_vs[2] = _mm256_add_ps(denom_vs[2], act_vs[2]);
		denom_vs[3] = _mm256_add_ps(denom_vs[3], act_vs[3]);
	}

	denom = simd::reduce_horizontalx4(
	    denom_vs[0],
	    denom_vs[1],
	    denom_vs[2],
	    denom_vs[3]);

	__m256 denom_v = _mm256_set1_ps(1.0 / denom);

	for (int i = 0; i < constants::vocab; i += UNROLL) {
		simd::loadalignedvx4(act_vs, (o->data).get(), i);

		act_vs[0] = _mm256_mul_ps(act_vs[0], denom_v);
		act_vs[1] = _mm256_mul_ps(act_vs[1], denom_v);
		act_vs[2] = _mm256_mul_ps(act_vs[2], denom_v);
		act_vs[3] = _mm256_mul_ps(act_vs[3], denom_v);

		simd::storealignedvx4((o->data).get(), act_vs, i);
	}

	// inverse transform sampling
	float cumprobs = 0.0;
	int index = 0;
	for (int idx = 0; idx < constants::vocab; idx++) {
		cumprobs += (o->data)[idx];
		if (cumprobs >= randf) {
			index = idx;
			break;
		}
	}

	(toks->data[0]) = index;
}

}  // namespace ops
