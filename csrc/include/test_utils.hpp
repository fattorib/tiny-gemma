#pragma once
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>

#include "tensor.hpp"

namespace tests {

// computes maximum absolute and relative errors

bool check_error(tensor::AlignedArray<float>& arr,
                 tensor::AlignedArray<float>& ref, size_t numel,
                 float abs_tol = 1e-3, float rel_tol = 1e-2, bool return_early = false) {
	float max_rel_error = -INFINITY;
	float max_abs_error = -INFINITY;

	float diff_norm = 0.0f;
	float norm = 0.0f;

	float rel_err;
	float abs_err;

	float a_val;
	float a_ref_val;

	bool passes = true;

	for (size_t i = 0; i < numel; i++) {
		rel_err = (ref[i] != 0.0) ? std::abs(arr[i] - ref[i]) / std::abs(ref[i]) : 0.0;  // so you don't get inf rel error
		abs_err = std::abs(arr[i] - ref[i]);

		if (abs_err > abs_tol) {
			std::cout << "ERROR: `abs_err` > " << abs_tol << std::endl;
			std::cout << i << std::endl;
			std::cout << arr[i] << " " << ref[i] << std::endl;
			std::cout << std::abs(arr[i] - ref[i]) << std::endl;
			std::cout << rel_err << std::endl;
			std::cout << std::endl;
			passes = false;
			if (return_early) {
				max_rel_error = std::fmaxf(max_rel_error, rel_err);
				max_abs_error = std::fmaxf(max_abs_error, abs_err);
				break;
			}
		}

		max_rel_error = std::fmaxf(max_rel_error, rel_err);
		max_abs_error = std::fmaxf(max_abs_error, abs_err);

		diff_norm += std::pow((arr[i] - ref[i]), 2.0);
		norm += std::pow((ref[i]), 2.0);

		// nan checks
		if (arr[i] != arr[i]) {
			std::cout << i << std::endl;
			std::cout << arr[i] << " " << ref[i] << std::endl;
			throw std::runtime_error(
			    "ERROR: NaN value encountered in output array");
			break;
		}

		if (ref[i] != ref[i]) {
			std::cout << i << std::endl;
			std::cout << arr[i] << " " << ref[i] << std::endl;
			throw std::runtime_error(
			    "ERROR: NaN value encountered in reference array");
		}
	}

	if (passes) {
		float linalg_rel_error = std::pow(diff_norm, 0.5) / std::pow(norm, 0.5);
		if (linalg_rel_error > rel_tol) {
			passes = false;
		}
		printf("Maximum relative error: (%1.8f)\n", max_rel_error);
		printf("Maximum absolute error: (%1.8f)\n", max_abs_error);
		printf("Linalg relative error: (%1.8f)\n", linalg_rel_error);
	}

	return passes;
}

inline std::vector<std::uint64_t>& flush_buffer() {
	static std::vector<std::uint64_t> buf(
	    (2ULL * 16 * 1024 * 1024) / sizeof(std::uint64_t));  // 2 × 16 MB LLC
	return buf;
}

inline void cold_flush() {
	constexpr std::size_t line = 64 / sizeof(std::uint64_t);  // 64-B cache line
	static std::mt19937_64 rng{std::random_device{}()};
	std::uniform_int_distribution<std::size_t> dist(0, line - 1);

	std::size_t start = dist(rng);
	volatile std::uint64_t* p = flush_buffer().data() + start;
	const volatile std::uint64_t* end = flush_buffer().data() + flush_buffer().size();

	for (; p < end; p += line) {
		*p = *p + 1;
	}
}

double time_op(std::function<void(void)> f, int warmup, int benchmark, const bool flush_l3 = true) {
	for (int i = 0; i < warmup; i++) {
		f();
	}

	double s = 0.0f;

	for (int i = 0; i < benchmark; i++) {
		if (flush_l3) {
			cold_flush();
		}
		auto t_bench_1 = std::chrono::high_resolution_clock::now();
		f();
		auto t_bench_2 = std::chrono::high_resolution_clock::now();
		s += (t_bench_2 - t_bench_1).count();
	}

	return s / 1e9;
}

std::string format_string(std::string base, int bs, int seq, int dmodel) {
	std::string fname = base + std::to_string(bs) + "_" + std::to_string(seq) + "_" + std::to_string(dmodel) + ".bin";
	return fname;
}

}  // namespace tests
