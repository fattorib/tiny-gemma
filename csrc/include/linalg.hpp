#pragma once
#include "cassert"
#include "constants.hpp"
#include "immintrin.h"
#include "static_switch.hpp"

// loads 16 floats from a quantized int8 span
static inline void load_i8x16(__m256 ws[2], const int8_t *__restrict__ w, size_t off) {
	__m128i lo16 = _mm_load_si128((const __m128i *)(w + off + 0));

	__m128i lo_hi8 = _mm_srli_si128(lo16, 8);

	__m256i i0 = _mm256_cvtepi8_epi32(lo16);    // bytes  0.. 7
	__m256i i1 = _mm256_cvtepi8_epi32(lo_hi8);  // bytes  8..15

	ws[0] = _mm256_cvtepi32_ps(i0);
	ws[1] = _mm256_cvtepi32_ps(i1);
}

// loads 32 floats from a quantized int8 span
static inline void load_i8x32(__m256 ws[4], const int8_t *__restrict__ w, size_t off) {
	__m128i lo16 = _mm_load_si128((const __m128i *)(w + off + 0));
	__m128i hi16 = _mm_load_si128((const __m128i *)(w + off + 16));

	__m128i lo_hi8 = _mm_srli_si128(lo16, 8);
	__m128i hi_hi8 = _mm_srli_si128(hi16, 8);

	__m256i i0 = _mm256_cvtepi8_epi32(lo16);    // bytes  0.. 7
	__m256i i1 = _mm256_cvtepi8_epi32(lo_hi8);  // bytes  8..15
	__m256i i2 = _mm256_cvtepi8_epi32(hi16);    // bytes 16..23
	__m256i i3 = _mm256_cvtepi8_epi32(hi_hi8);  // bytes 24..31

	ws[0] = _mm256_cvtepi32_ps(i0);
	ws[1] = _mm256_cvtepi32_ps(i1);
	ws[2] = _mm256_cvtepi32_ps(i2);
	ws[3] = _mm256_cvtepi32_ps(i3);
}

namespace gemv {

template <int rows, int cols>
void qgemv(float *__restrict__ o, const float *__restrict__ a, const int8_t *__restrict__ w,
           const float *__restrict__ scales) {
	constexpr int LDW = cols;

	for (int r = 0; r < rows; r += 1) {
		__m256 vx0 = _mm256_broadcast_ss(&a[r + 0]);

		const int8_t *w_row = w + r * LDW;

		for (int c = 0; c < cols; c += 32) {
			__m256 o0 = _mm256_load_ps(&o[c + 0]);
			__m256 o1 = _mm256_load_ps(&o[c + 8]);
			__m256 o2 = _mm256_load_ps(&o[c + 16]);
			__m256 o3 = _mm256_load_ps(&o[c + 24]);

			__m256 acc0 = _mm256_setzero_ps();
			__m256 acc1 = _mm256_setzero_ps();
			__m256 acc2 = _mm256_setzero_ps();
			__m256 acc3 = _mm256_setzero_ps();

			__m256 ws[4];

			load_i8x32(ws, w_row, c);

			acc0 = _mm256_fmadd_ps(vx0, ws[0], acc0);
			acc1 = _mm256_fmadd_ps(vx0, ws[1], acc1);
			acc2 = _mm256_fmadd_ps(vx0, ws[2], acc2);
			acc3 = _mm256_fmadd_ps(vx0, ws[3], acc3);

			o0 = _mm256_add_ps(o0, acc0);
			o1 = _mm256_add_ps(o1, acc1);
			o2 = _mm256_add_ps(o2, acc2);
			o3 = _mm256_add_ps(o3, acc3);

			_mm256_store_ps(&o[c + 0], o0);
			_mm256_store_ps(&o[c + 8], o1);
			_mm256_store_ps(&o[c + 16], o2);
			_mm256_store_ps(&o[c + 24], o3);
		}
	}

	for (int c = 0; c < cols; c += 32) {
		__m256 s0 = _mm256_load_ps(&scales[c + 0]);
		__m256 o0 = _mm256_load_ps(&o[c + 0]);

		__m256 s1 = _mm256_load_ps(&scales[c + 8]);
		__m256 o1 = _mm256_load_ps(&o[c + 8]);

		__m256 s2 = _mm256_load_ps(&scales[c + 16]);
		__m256 o2 = _mm256_load_ps(&o[c + 16]);

		__m256 s3 = _mm256_load_ps(&scales[c + 24]);
		__m256 o3 = _mm256_load_ps(&o[c + 24]);

		o0 = _mm256_mul_ps(o0, s0);
		_mm256_store_ps(&o[c + 0], o0);

		o1 = _mm256_mul_ps(o1, s1);
		_mm256_store_ps(&o[c + 8], o1);

		o2 = _mm256_mul_ps(o2, s2);
		_mm256_store_ps(&o[c + 16], o2);

		o3 = _mm256_mul_ps(o3, s3);
		_mm256_store_ps(&o[c + 24], o3);
	}
}

// for most shapes, the above gemv is sufficient as o fits in L1/L2. This is not the case for the 262K vocab
// and some of the 1B shapes. as such, we use a kernel which unrolls the row loop a bit
template <int rows, int cols, int row_tile = 4>
void qgemv_row_tile(float *__restrict__ o, const float *__restrict__ a, const int8_t *__restrict__ w,
                    const float *__restrict__ scales) {
	// NOTE: if LDW % 4096 or 65536, then we get cache overwrites (kernel is ~3% slower)
	constexpr int LDW = cols;
	__m256 vxs[row_tile];

	for (int r = 0; r < rows; r += row_tile) {
#pragma unroll
		for (int off = 0; off < row_tile; off++) {
			vxs[off] = _mm256_broadcast_ss(&a[r + off]);
		}

		const int8_t *w_row = w + r * LDW;

		for (int c = 0; c < cols; c += 16) {
			__m256 o0 = _mm256_load_ps(&o[c + 0]);
			__m256 o1 = _mm256_load_ps(&o[c + 8]);

			__m256 acc0 = _mm256_setzero_ps();
			__m256 acc1 = _mm256_setzero_ps();

			__m256 ws[2];

#pragma unroll
			for (int off = 0; off < row_tile; off++) {
				load_i8x16(ws, w_row, c + (off * LDW));
				acc0 = _mm256_fmadd_ps(vxs[off], ws[0], acc0);
				acc1 = _mm256_fmadd_ps(vxs[off], ws[1], acc1);
			}

			o0 = _mm256_add_ps(o0, acc0);
			o1 = _mm256_add_ps(o1, acc1);

			_mm256_store_ps(&o[c + 0], o0);
			_mm256_store_ps(&o[c + 8], o1);
		}
	}

	for (int c = 0; c < cols; c += 32) {
		__m256 s0 = _mm256_load_ps(&scales[c + 0]);
		__m256 o0 = _mm256_load_ps(&o[c + 0]);

		__m256 s1 = _mm256_load_ps(&scales[c + 8]);
		__m256 o1 = _mm256_load_ps(&o[c + 8]);

		__m256 s2 = _mm256_load_ps(&scales[c + 16]);
		__m256 o2 = _mm256_load_ps(&o[c + 16]);

		__m256 s3 = _mm256_load_ps(&scales[c + 24]);
		__m256 o3 = _mm256_load_ps(&o[c + 24]);

		o0 = _mm256_mul_ps(o0, s0);
		o1 = _mm256_mul_ps(o1, s1);
		o2 = _mm256_mul_ps(o2, s2);
		o3 = _mm256_mul_ps(o3, s3);

		_mm256_store_ps(&o[c + 0], o0);
		_mm256_store_ps(&o[c + 8], o1);
		_mm256_store_ps(&o[c + 16], o2);
		_mm256_store_ps(&o[c + 24], o3);
	}
}

void launch_qgemv(float *__restrict__ C, float *A, int8_t *B, float *scales, int inDim, int outDim) {
	if ((outDim == 6912)) {
		const int unroll = 8;

		ROW_SWITCH(inDim, [&] {
			COL_SWITCH(outDim, [&] {
				qgemv_row_tile<rowDim, colDim, unroll>(C, A, B, scales);
			});
		});

	}

	else {
		ROW_SWITCH(inDim, [&] {
			COL_SWITCH(outDim, [&] {
				qgemv<rowDim, colDim>(C, A, B, scales);
			});
		});
	}
}

void launch_qgemv_logits(float *__restrict__ C, float *A, int8_t *B, float *scales, int inDim, int outDim) {
	ROW_SWITCH(inDim, [&] {
		COL_SWITCH(outDim, [&] {
			const int unroll = 4;
			qgemv_row_tile<rowDim, colDim, unroll>(C, A, B, scales);
		});
	});
}

}  // namespace gemv

namespace gemm {

template <int tr, int tc, int ca, int cb, int k, int n>
inline void qavx16x6microkernel(float *__restrict__ C, const float *Apack, const int8_t *Bpack,
                                int br, int m, float *scales) {
	__m256 C_tile[tr][tc / constants::simd] = {};

	for (int k_outer = 0; k_outer < k; k_outer++) {
		int b_offs = (cb * k_outer);

		for (int row = 0; row < tr; row++) {
			__m256 tmpA;

			if (row >= br) {
				// mask out value
				tmpA = _mm256_set1_ps(0.0f);
			}

			else {
				tmpA = _mm256_broadcast_ss(&Apack[k_outer + k * row]);
			}

			__m256 tmpBs[2];
			load_i8x16(tmpBs, Bpack, b_offs);  // NOTE only works if tc = 16
			C_tile[row][0] = _mm256_fmadd_ps(tmpA, tmpBs[0], C_tile[row][0]);
			C_tile[row][1] = _mm256_fmadd_ps(tmpA, tmpBs[1], C_tile[row][1]);
		}
	}

	__m256 vs[2] = {_mm256_load_ps(scales), _mm256_load_ps(scales + 8)};

	for (int row = 0; row < tr; row++) {
		if (row < br) {
			for (int col = 0; col < tc; col += constants::simd) {
				C_tile[row][col / constants::simd] = _mm256_mul_ps(C_tile[row][col / constants::simd], vs[col / constants::simd]);

				_mm256_store_ps(&C[(row * n) + col],
				                C_tile[row][col / constants::simd]);
			}
		}
	}
}

// A is [..., m, k]
// B is [k, n]
// C is [..., m, n]
template <int tr, int tc, int ca, int cb, int k, int n>
void qgemmnn(float *__restrict__ C, float *A, int8_t *B, int m, float *scales) {
	int a_offs, b_offs, c_offs;

	alignas(constants::alignment) float Apack[ca * k];
	alignas(constants::alignment) int8_t Bpack[k * cb];

	for (int a_tile = 0; a_tile < m; a_tile += ca) {
		int a_cache_boundary = std::min(ca, m - a_tile);

		// load A to $
		for (int r = 0; r < ca; r++) {
			for (int c = 0; c < k; c++) {
				if (r >= a_cache_boundary) {
					Apack[r * k + c] = 0.0f;
				} else {
					Apack[r * k + c] = A[(a_tile * k) + r * k + c];
				}
			}
		}

		for (int b_tile = 0; b_tile < n; b_tile += cb) {
			// load B to $
			for (int r = 0; r < k; r++) {
				for (int c = 0; c < cb; c++) {
					Bpack[r * cb + c] = B[r * n + (b_tile + c)];
				}
			}

			for (int a_block = 0; a_block < ca; a_block += tr) {
				// compute masking condition
				int br = std::min(tr, a_cache_boundary - a_block);

				a_offs = k * a_block;

				for (int b_block = 0; b_block < cb; b_block += tc) {
					c_offs = ((a_tile + a_block) * n) + (b_block + b_tile);
					qavx16x6microkernel<tr, tc, ca, cb, k, n>(&C[c_offs], &Apack[a_offs], &Bpack[b_block], br, m, &scales[b_block + b_tile]);
				}
			}
		}
	}
}

void launch_qgemm(float *__restrict__ C, float *A, int8_t *B, float *scales, int m, int inDim, int outDim) {
	ROW_SWITCH(inDim, [&] {
		COL_SWITCH(outDim, [&] {
			const int tr = 6;    // row tile size
			const int tc = 16;   // col tile size
			const int ca = 256;  // cache blocking size for rows of A: [ca, k]
			const int cb = 64;   // cache blocking size for cols of B: [k, cb]
			qgemmnn<tr, tc, ca, cb, rowDim, colDim>(C, A, B, m, scales);
		});
	});
}

}  // namespace gemm
