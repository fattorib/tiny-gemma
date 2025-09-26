// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h
// and https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/static_switch.h
#pragma once

#define ROW_SWITCH(ROW, ...)                                                               \
	[&] {                                                                                  \
		if (ROW == 640) {                                                                  \
			constexpr static int rowDim = 640;                                             \
			return __VA_ARGS__();                                                          \
		} else if (ROW == 2048) {                                                          \
			constexpr static int rowDim = 2048;                                            \
			return __VA_ARGS__();                                                          \
		} else if (ROW == 1024) {                                                          \
			constexpr static int rowDim = 1024;                                            \
			return __VA_ARGS__();                                                          \
		} else if (ROW == 256) {                                                           \
			constexpr static int rowDim = 256;                                             \
			return __VA_ARGS__();                                                          \
		} else if (ROW == 1152) {                                                          \
			constexpr static int rowDim = 1152;                                            \
			return __VA_ARGS__();                                                          \
		} else if (ROW == 6912) {                                                          \
			constexpr static int rowDim = 6912;                                            \
			return __VA_ARGS__();                                                          \
		} else {                                                                           \
			throw std::runtime_error("Unsupported ROW dimension: " + std::to_string(ROW)); \
		}                                                                                  \
	}()

#define COL_SWITCH(COL, ...)                                                               \
	[&] {                                                                                  \
		if (COL == 640) {                                                                  \
			constexpr static int colDim = 640;                                             \
			return __VA_ARGS__();                                                          \
		} else if (COL == 262144) {                                                        \
			constexpr static int colDim = 262144;                                          \
			return __VA_ARGS__();                                                          \
		} else if (COL == 2048) {                                                          \
			constexpr static int colDim = 2048;                                            \
			return __VA_ARGS__();                                                          \
		} else if (COL == 256) {                                                           \
			constexpr static int colDim = 256;                                             \
			return __VA_ARGS__();                                                          \
		} else if (COL == 1024) {                                                          \
			constexpr static int colDim = 1024;                                            \
			return __VA_ARGS__();                                                          \
		} else if (COL == 1152) {                                                          \
			constexpr static int colDim = 1152;                                            \
			return __VA_ARGS__();                                                          \
		} else if (COL == 6912) {                                                          \
			constexpr static int colDim = 6912;                                            \
			return __VA_ARGS__();                                                          \
		} else {                                                                           \
			throw std::runtime_error("Unsupported COL dimension: " + std::to_string(COL)); \
		}                                                                                  \
	}()
