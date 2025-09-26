CXX		=g++
DEBUG   =-g
FORMAT  =-style="{BasedOnStyle: Google, IndentWidth: 4, TabWidth: 4, UseTab: ForIndentation, ColumnLimit: 0}"
CIFORMAT=--dry-run -Werror
CFLAGS  =-std=c++20 -O3 -mavx2 -mfma -Werror -Wpedantic -Wuninitialized -Wno-vla -ffast-math 

build-dir:
	if [ ! -d build ]; then mkdir build; fi

gemma: build-dir
	$(CXX) $(CFLAGS) -I csrc/include  gemma.cpp -o build/gemma

passkey: build-dir
	$(CXX) $(CFLAGS) -I csrc/include  tests/passkey.cpp -o build/passkey

unittests: build-dir
	$(CXX) $(CFLAGS) -I csrc/include tests/csrc/test_bench_modules.cpp -o build/test_modules

integrations: build-dir
	$(CXX) $(CFLAGS) -I csrc/include tests/csrc/test_generation.cpp -o build/gemma-it 

format: 
	clang-format $(FORMAT) -i csrc/include/*.hpp
	clang-format $(FORMAT) -i tests/*/**.cpp
	clang-format $(FORMAT) -i gemma.cpp

format-ci:
	clang-format $(FORMAT) $(CIFORMAT) -i csrc/include/*.hpp
	clang-format $(FORMAT) $(CIFORMAT) -i tests/*/**.cpp
	clang-format $(FORMAT) $(CIFORMAT) -i gemma.cpp
