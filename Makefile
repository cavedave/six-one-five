# six-one-five — GPU search for a₁⁶+…+a₅⁶ = B⁶

.PHONY: all v3 v616 v2 test host-test ribbon-test clean

NVCC ?= nvcc
CXX ?= g++
NVCCFLAGS = -O3 -std=c++20 -gencode arch=compute_90,code=compute_90 -lineinfo
CXXFLAGS = -O2 -std=c++20 -fopenmp

all: v3 host-test

v3: solve_516_v3

solve_516_v3: solve_516_v3.cu mod60.hpp quad_sum.hpp k14_common.hpp
	$(NVCC) $(NVCCFLAGS) -o $@ $<

v616: solve_616_v1

solve_616_v1: solve_616_v1.cu mod60.hpp quad_sum.hpp k14_common.hpp
	$(NVCC) $(NVCCFLAGS) -o $@ $<

v2: solve_516_v2

solve_516_v2: solve_516_v2.cpp mod60.hpp quad_sum.hpp k14_common.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

host-test: v3_host_logic_test

v3_host_logic_test: v3_host_logic_test.cpp
	$(CXX) -O2 -std=c++20 -o $@ $<

ribbon-test:
	$(CXX) -O2 -std=c++20 -o spike/ribbon/ribbon_filter_test spike/ribbon/ribbon_filter_test.cpp

clean:
	rm -f solve_516_v3 solve_516_v2 solve_616_v1 v3_host_logic_test spike/ribbon/ribbon_filter_test *.o
