/*
    Consolidates common SIMD uses:
        - Reductions
        - Multi register reductions
        - Vectorized loads and stores
*/
#pragma once
#include "constants.hpp"
#include "immintrin.h"
#include "tensor.hpp"

namespace simd {

using namespace tensor;

static inline float v_hsum(__m256 a) {
	__m256 sum_halves = _mm256_hadd_ps(a, a);
	sum_halves = _mm256_hadd_ps(sum_halves, sum_halves);
	__m128 lo = _mm256_castps256_ps128(sum_halves);
	__m128 hi = _mm256_extractf128_ps(sum_halves, 1);
	__m128 sum = _mm_add_ps(lo, hi);
	return _mm_cvtss_f32(sum);
}

// Computes a 4 way sum reduction among 4 independent accumulators
static inline float reduce_horizontalx4(__m256 a, __m256 b, __m256 c, __m256 d) {
	a = _mm256_add_ps(a, b);
	c = _mm256_add_ps(c, d);
	a = _mm256_add_ps(a, c);
	float r = v_hsum(a);
	return r;
}

static inline float max_horizontalx4(__m256 a, __m256 b, __m256 c, __m256 d) {
	float max = -INFINITY;
	a = _mm256_max_ps(a, b);
	c = _mm256_max_ps(c, d);
	a = _mm256_max_ps(a, c);
	float *a_v = (float *)&a;
	for (int i = 0; i < constants::simd; i++) {
		max = std::fmaxf(a_v[i], max);
	}

	return max;
}
static inline void loadalignedx32(float *__restrict__ arr, const float *__restrict__ a, int d, int off_st, int off_ld) {
	constexpr int UNROLL = 4 * constants::simd;

	for (int i = 0; i < d; i += UNROLL) {
		*(__m256 *)(&arr[i + off_st]) = _mm256_load_ps(&(a)[off_ld + i]);
		*(__m256 *)(&arr[i + off_st + 8]) = _mm256_load_ps(&(a)[off_ld + i + 8]);
		*(__m256 *)(&arr[i + off_st + 16]) = _mm256_load_ps(&(a)[off_ld + i + 16]);
		*(__m256 *)(&arr[i + off_st + 24]) = _mm256_load_ps(&(a)[off_ld + i + 24]);
	}
}

static inline void storealignedx32(float *__restrict__ a, const float *__restrict__ arr, int d, int off_st, int off_ld) {
	constexpr int UNROLL = 4 * constants::simd;

	for (int i = 0; i < d; i += UNROLL) {
		_mm256_store_ps(&(a)[off_st + i], *(__m256 *)(&arr[i + off_ld]));
		_mm256_store_ps(&(a)[off_st + i + 8], *(__m256 *)(&arr[i + off_ld + 8]));
		_mm256_store_ps(&(a)[off_st + i + 16], *(__m256 *)(&arr[i + off_ld + 16]));
		_mm256_store_ps(&(a)[off_st + i + 24], *(__m256 *)(&arr[i + off_ld + 24]));
	}
}

static inline void loadalignedvx4(__m256 v[4], const float *__restrict__ a, int off_ld) {
	v[0] = _mm256_load_ps(&(a[off_ld]));
	v[1] = _mm256_load_ps(&(a[off_ld + 8]));
	v[2] = _mm256_load_ps(&(a[off_ld + 16]));
	v[3] = _mm256_load_ps(&(a[off_ld + 24]));
}

static inline void storealignedvx4(float *__restrict__ arr, const __m256 v[4], int off_st) {
	_mm256_store_ps(&(arr[off_st]), v[0]);
	_mm256_store_ps(&(arr[off_st + 8]), v[1]);
	_mm256_store_ps(&(arr[off_st + 16]), v[2]);
	_mm256_store_ps(&(arr[off_st + 24]), v[3]);
}

}  // namespace simd