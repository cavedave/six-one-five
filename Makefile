# six-one-five — GPU search for a₁⁶+…+a₅⁶ = B⁶
# Also: record-style (6,1,7) hunt helpers (record_hunt / record_find5).

.PHONY: all v3 v3-a100 v4-a100 v616 v617 v2 test host-test \
        ribbon-test xor-test record record-host \
        fourcore fourcore-host fourcore-hunt-v3 fourcore-find-v3 \
        fourcore-find-v3-host clean

NVCC ?= nvcc
CXX ?= g++
# Default: PTX for Hopper+ (JIT on newer cards). Prefer v3-a100 / fatbin for rentals.
NVCCFLAGS = -O3 -std=c++20 -gencode arch=compute_90,code=compute_90 -lineinfo
# T0 / A100 rental (Ampere sm_80). See 615-a100-ribbon-shard-plan.md
NVCCFLAGS_A100 = -O3 -std=c++20 -gencode arch=compute_80,code=sm_80 -lineinfo \
	-Xcompiler -fopenmp
# Optional multi-arch fatbin (A100 + Hopper + Blackwell when toolkit supports 120)
NVCCFLAGS_FAT = -O3 -std=c++20 \
	-gencode arch=compute_80,code=sm_80 \
	-gencode arch=compute_90,code=sm_90 \
	-gencode arch=compute_90,code=compute_90 \
	-lineinfo -Xcompiler -fopenmp
CXXFLAGS = -O2 -std=c++20 -fopenmp
# record_find5.cu is C++17; keep NVCC at c++17 for that target
RECORD_NVCCFLAGS = -O3 -std=c++17 -gencode arch=compute_90,code=compute_90 -lineinfo -Xcompiler -fopenmp
RECORD_CXXFLAGS = -O3 -std=c++17 -fopenmp

# GMP (post-i128 four-core). Override on Linux if needed: GMP_PREFIX=/usr
GMP_PREFIX ?= $(shell brew --prefix gmp 2>/dev/null)
ifneq ($(GMP_PREFIX),)
  GMP_CFLAGS = -I$(GMP_PREFIX)/include
  GMP_LDFLAGS = -L$(GMP_PREFIX)/lib
endif
GMP_LIBS = -lgmpxx -lgmp
FOURCORE_CXXFLAGS = -O3 -std=c++17 $(GMP_CFLAGS)
# OpenMP is optional (Linux g++ typically has it; Apple clang often does not)
FOURCORE_OMP ?= $(shell echo | $(CXX) -fopenmp -x c++ -c - -o /dev/null 2>/dev/null && echo -fopenmp)
FOURCORE_CXXFLAGS += $(FOURCORE_OMP)
FOURCORE_NVCCFLAGS = -O3 -std=c++17 -gencode arch=compute_90,code=compute_90 -lineinfo \
	$(if $(FOURCORE_OMP),-Xcompiler $(FOURCORE_OMP)) $(GMP_CFLAGS)

all: v3 host-test

v3: solve_516_v3

solve_516_v3: solve_516_v3.cu mod60.hpp quad_sum.hpp k14_common.hpp
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# T0: native sm_80 binary for rented A100s (use --slots-log2 31 on 80GB if needed)
v3-a100: solve_516_v3_a100

solve_516_v3_a100: solve_516_v3.cu mod60.hpp quad_sum.hpp k14_common.hpp
	$(NVCC) $(NVCCFLAGS_A100) -o $@ $<

# M2+: xor/ribbon store fork (fails clearly until solve_516_v4.cu exists)
v4-a100:
	@if [ ! -f solve_516_v4.cu ]; then \
	  echo "solve_516_v4.cu not forked yet (plan M2). For T0 use: make v3-a100"; \
	  exit 1; \
	fi
	$(NVCC) $(NVCCFLAGS_A100) -o solve_516_v4 solve_516_v4.cu

v616: solve_616_v1

solve_616_v1: solve_616_v1.cu mod60.hpp quad_sum.hpp k14_common.hpp
	$(NVCC) $(NVCCFLAGS) -o $@ $<

v617: solve_617_v1

solve_617_v1: solve_617_v1.cu mod60.hpp quad_sum.hpp k14_common.hpp
	$(NVCC) $(NVCCFLAGS) -o $@ $<

v2: solve_516_v2

solve_516_v2: solve_516_v2.cpp mod60.hpp quad_sum.hpp k14_common.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

host-test: v3_host_logic_test

v3_host_logic_test: v3_host_logic_test.cpp
	$(CXX) -O2 -std=c++20 -o $@ $<

ribbon-test:
	$(CXX) -O2 -std=c++20 -o spike/ribbon/ribbon_filter_test spike/ribbon/ribbon_filter_test.cpp
	./spike/ribbon/ribbon_filter_test

# M1: host xor filter FN/FPR suite
xor-test:
	$(CXX) -O2 -std=c++20 -o spike/ribbon/xor_filter_test spike/ribbon/xor_filter_test.cpp
	./spike/ribbon/xor_filter_test

# ---- record-style (6,1,7) hunt (j=2 / 294-normalized) ----
record: record_hunt record_find5 record_probe

record_hunt: record_hunt.cpp
	$(CXX) $(RECORD_CXXFLAGS) -o $@ $<

record_find5: record_find5.cu
	$(NVCC) $(RECORD_NVCCFLAGS) -o $@ $<

record_probe: record_probe.cpp
	$(CXX) $(RECORD_CXXFLAGS) -o $@ $<

# Mac / no-GPU smoke for find5 host math + tiny table
record-host: record_find5.cu
	$(CXX) -O3 -std=c++17 -DHOST_ONLY -Wno-unknown-pragmas -x c++ -o record_find5_host $<

# ---- (6,1,5) four-core D=42h hunter (post-i128, GMP host math) ----
fourcore: fourcore_hunt fourcore_find4

fourcore_hunt: fourcore_hunt.cpp fourcore_gmp.hpp
	$(CXX) $(FOURCORE_CXXFLAGS) -o $@ $< $(GMP_LDFLAGS) $(GMP_LIBS)

fourcore_find4: fourcore_find4.cu fourcore_gmp.hpp
	$(NVCC) $(FOURCORE_NVCCFLAGS) -o $@ $< $(GMP_LDFLAGS) $(GMP_LIBS)

fourcore-host: fourcore_hunt fourcore_find4_host

fourcore_find4_host: fourcore_find4.cu fourcore_gmp.hpp
	$(CXX) $(FOURCORE_CXXFLAGS) -DHOST_ONLY -Wno-unknown-pragmas -x c++ \
	  -o $@ $< $(GMP_LDFLAGS) $(GMP_LIBS)

# ---- (6,1,5) multi-class post-i128 hunt (host-only Stage 1–2, *_v3) ----
fourcore-hunt-v3: fourcore_hunt_v3

fourcore_hunt_v3: fourcore_hunt_v3.cpp fourcore_classes_v3.hpp fourcore_gmp.hpp
	$(CXX) $(FOURCORE_CXXFLAGS) -o $@ $< $(GMP_LDFLAGS) $(GMP_LIBS)

fourcore-find-v3: fourcore_find_v3

fourcore_find_v3: fourcore_find_v3.cu fourcore_classes_v3.hpp fourcore_gmp.hpp
	$(NVCC) $(FOURCORE_NVCCFLAGS) -o $@ $< $(GMP_LDFLAGS) $(GMP_LIBS)

fourcore-find-v3-host: fourcore_find_v3_host

fourcore_find_v3_host: fourcore_find_v3.cu fourcore_classes_v3.hpp fourcore_gmp.hpp
	$(CXX) $(FOURCORE_CXXFLAGS) -DHOST_ONLY -Wno-unknown-pragmas -x c++ \
	  -o $@ $< $(GMP_LDFLAGS) $(GMP_LIBS)

clean:
	rm -f solve_516_v3 solve_516_v3_a100 solve_516_v4 solve_516_v2 \
	      solve_616_v1 solve_617_v1 \
	      v3_host_logic_test spike/ribbon/ribbon_filter_test spike/ribbon/xor_filter_test \
	      record_hunt record_find5 record_probe record_find5_host \
	      fourcore_hunt fourcore_find4 fourcore_find4_host \
	      fourcore_hunt_v3 fourcore_find_v3 fourcore_find_v3_host *.o
