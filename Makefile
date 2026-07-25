# six-one-five — GPU search for a₁⁶+…+a₅⁶ = B⁶
# Also: record-style (6,1,7) hunt helpers (record_hunt / record_find5).

.PHONY: all v3 v616 v617 v2 test host-test ribbon-test record record-host clean

NVCC ?= nvcc
CXX ?= g++
NVCCFLAGS = -O3 -std=c++20 -gencode arch=compute_90,code=compute_90 -lineinfo
CXXFLAGS = -O2 -std=c++20 -fopenmp
# record_find5.cu is C++17; keep NVCC at c++17 for that target
RECORD_NVCCFLAGS = -O3 -std=c++17 -gencode arch=compute_90,code=compute_90 -lineinfo -Xcompiler -fopenmp
RECORD_CXXFLAGS = -O3 -std=c++17 -fopenmp

all: v3 host-test

v3: solve_516_v3

solve_516_v3: solve_516_v3.cu mod60.hpp quad_sum.hpp k14_common.hpp
	$(NVCC) $(NVCCFLAGS) -o $@ $<

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

clean:
	rm -f solve_516_v3 solve_516_v2 solve_616_v1 solve_617_v1 \
	      v3_host_logic_test spike/ribbon/ribbon_filter_test \
	      record_hunt record_find5 record_probe record_find5_host *.o
