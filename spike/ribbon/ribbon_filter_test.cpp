// Host-only smoke test for the ribbon spike interface.
// Run: g++ -O2 -std=c++20 -o ribbon_filter_test ribbon_filter_test.cpp && ./ribbon_filter_test

#include "ribbon_filter.hpp"

#include <cstdio>

int main() {
    const std::vector<RibbonKey> keys = {1, 42, 999, 123456789ULL};
    const RibbonFilter f = build_ribbon(keys, 16);
    int pass = 0;
    for (RibbonKey k : keys)
        if (ribbon_might_contain(f, k)) ++pass;
    std::printf("ribbon_filter_test: %d/%zu keys pass (stub expects all)\n", pass, keys.size());
    return pass == (int)keys.size() ? 0 : 1;
}
