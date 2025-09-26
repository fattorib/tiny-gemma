#pragma once
#include <iostream>
#include <memory>
#include <new>
#include <random>
#include <stdexcept>

#include "constants.hpp"
#include "linalg.hpp"

namespace tensor {

template <typename T>
struct AlignedDeleter {
	void operator()(T *ptr) {
		::operator delete[](ptr, std::align_val_t(constants::alignment));
	}
};

// create a unique pointer of a given size with 32-byte alignment
template <typename T>
std::unique_ptr<T[], AlignedDeleter<T>> make_unique_aligned(size_t numel) {
	return std::unique_ptr<T[], AlignedDeleter<T>>(
	    new (std::align_val_t(constants::alignment)) T[numel]);
}

template <class T>
using AlignedArray = std::unique_ptr<T[], AlignedDeleter<T>>;

// constant init: x_i = c
template <typename T>
AlignedArray<T> make_unique_aligned_constant(size_t numel, T c) {
	AlignedArray<T> ptr = make_unique_aligned<T>(numel);

	// fill with c
	for (int i = 0; i < numel; i++) {
		ptr[i] = T(c);
	}

	return ptr;
}

// ranged: x_0 = 0, ..., x_n = n
template <typename T>
AlignedArray<T> make_unique_aligned_arange(size_t numel) {
	AlignedArray<T> ptr = make_unique_aligned<T>(numel);

	// fill with c
	for (int i = 0; i < numel; i++) {
		ptr[i] = T(i);
	}

	return ptr;
}

// 3D or 4D activation
struct activation {
	AlignedArray<float> data;
	size_t x;
	size_t y;
	size_t z;
	size_t w;
	size_t ndim;

	activation() = default;

	activation(size_t x, size_t y, size_t z) : x(x), y(y), z(z), w(0), ndim(3) {
		data = make_unique_aligned_constant<float>(x * y * z, 0.0);
	}

	activation(size_t x, size_t y, size_t z, size_t w)
	    : x(x), y(y), z(z), w(w), ndim(4) {
		data = make_unique_aligned_constant<float>(x * y * z * w, 0.0);
	}

	//  Constructor for 3D activation -> take ownership of data
	activation(AlignedArray<float> &act, size_t x, size_t y, size_t z)
	    : data(std::move(act)), x(x), y(y), z(z), w(0), ndim(3) {}

	// Constructor for 4D activation
	activation(AlignedArray<float> &act, size_t x, size_t y, size_t z, size_t w)
	    : data(std::move(act)), x(x), y(y), z(z), w(w), ndim(4) {}

	size_t numel() const { return (ndim < 4) ? x * y * z : x * y * z * w; }
	size_t x_stride() const { return (ndim < 4) ? y * z : y * z * w; }
	size_t y_stride() const { return (ndim < 4) ? z : z * w; }
	size_t z_stride() const { return (ndim < 4) ? 1 : w; }
	size_t w_stride() const { return (ndim < 4) ? 0 : 1; }

	// copies data from c into this
	void copy(const activation *c) {
		constexpr size_t unroll = 4 * constants::simd;
		for (int i = 0; i < this->numel(); i += unroll) {
			*(__m256 *)(&data[i + 0]) = *(__m256 *)(&c->data[i + 0]);
			*(__m256 *)(&data[i + 8]) = *(__m256 *)(&c->data[i + 8]);
			*(__m256 *)(&data[i + 16]) = *(__m256 *)(&c->data[i + 16]);
			*(__m256 *)(&data[i + 24]) = *(__m256 *)(&c->data[i + 24]);
		}
	}

	// copies the last element of activation sequence: [B,T,D] -> [B, 1, D]
	void copy_last_slice(activation *c, size_t offset) {
		constexpr size_t unroll = 4 * constants::simd;
		for (int i = 0; i < this->numel(); i += unroll) {
			*(__m256 *)(&data[i + 0]) = *(__m256 *)(&c->data[i + offset + 0]);
			*(__m256 *)(&data[i + 8]) = *(__m256 *)(&c->data[i + offset + 8]);
			*(__m256 *)(&data[i + 16]) = *(__m256 *)(&c->data[i + offset + 16]);
			*(__m256 *)(&data[i + 24]) = *(__m256 *)(&c->data[i + offset + 24]);
		}
	}

	void zero() {
		constexpr size_t unroll = 4 * constants::simd;
		for (size_t i = 0; i < (this->numel()); i += unroll) {
			*(__m256 *)(&data[i + 0]) = _mm256_setzero_ps();
			*(__m256 *)(&data[i + 8]) = _mm256_setzero_ps();
			*(__m256 *)(&data[i + 16]) = _mm256_setzero_ps();
			*(__m256 *)(&data[i + 24]) = _mm256_setzero_ps();
		}
	}
};

// 2D token struct: [BS, SEQ]
struct tokens {
	AlignedArray<int> data;
	size_t x;
	size_t y;

	tokens() = default;

	tokens(size_t x, size_t y)
	    : x(x), y(y) {
		data = make_unique_aligned_constant<int>(x * y, 0);
	}

	tokens(AlignedArray<int> &tokens, size_t x, size_t y)
	    : data(std::move(tokens)), x(x), y(y) {}

	size_t numel() const { return x * y; }
	size_t x_stride() const { return y; }
	size_t y_stride() const { return 1; }
};

template <typename T>
struct weight {
	AlignedArray<T> data;
	size_t rows;
	size_t cols;
	size_t ndim;

	AlignedArray<float> scales = nullptr;
	bool quantized = false;

	weight() = default;

	// 2D weights
	weight(size_t rows, size_t cols, bool quantized) : rows(rows), cols(cols), quantized(quantized), ndim(2) {
		data = make_unique_aligned_constant<T>(rows * cols, T(0.0));
		if (quantized) {
			scales = make_unique_aligned_constant<float>(cols, T(0.0));
		} else {
			scales = nullptr;
		}
	}

	weight(size_t rows, size_t cols) : rows(rows), cols(cols), quantized(false), ndim(2) {
		data = make_unique_aligned_constant<T>(rows * cols, T(0.0));
	}

	// 1D weights
	weight(size_t rows) : rows(rows), cols(0), ndim(1), quantized(false) {
		data = make_unique_aligned_constant<T>(rows, T(0.0));
		scales = nullptr;
	}

	// 2D and 1D weight copies
	weight(AlignedArray<T> &data, AlignedArray<float> &scales, size_t rows, size_t cols, bool quantized)
	    : data(std::move(data)), rows(rows), cols(cols), ndim(2), scales(std::move(scales)), quantized(quantized) {
	}

	weight(AlignedArray<T> &data, size_t rows)
	    : data(std::move(data)), rows(rows), cols(0), ndim(1), quantized(quantized) {
		scales = nullptr;
	}

	bool is_quantized() const { return quantized; }
	size_t numel() const { return (ndim == 1) ? rows : rows * cols; }
	size_t row_stride() const { return (ndim == 1) ? 1 : cols; }
	size_t col_stride() const {
		return (ndim == 1) ? 0 : 1;
	}  // always row-major
};

using FloatArray = weight<float>;
using QuantArray = weight<int8_t>;
using ActArray = activation;

}  // namespace tensor
