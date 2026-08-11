// Build packed xor pair store and write StoreHeader v2 + packed bytes.
// Host-only (no GPU). Needs lots of RAM for large N (~40–60 GB peak at N~70k).
//
//   make xor-build-save
//   ./xor_build_save --N 71428 --r 48 --out runs/xor_N71428_r48.bin
//
#include "fourcore_xor_store.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void usage() {
  std::fprintf(stderr,
               "usage: xor_build_save --N n [--r BITS] [--out FILE]\n"
               "  N=3000000/42 → 71428 for B through 3.0M\n"
               "  N=2800000/42 → 66666 for B through 2.8M\n");
}

int main(int argc, char** argv) {
  int N = 0, r = 48;
  std::string out = "runs/xor_packed.bin";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* {
      return (i + 1 < argc) ? argv[++i] : "";
    };
    if (a == "--N") N = std::atoi(next());
    else if (a == "--r") r = std::atoi(next());
    else if (a == "--out") out = next();
    else if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else {
      std::fprintf(stderr, "unknown %s\n", a.c_str());
      usage();
      return 1;
    }
  }
  if (N < 4 || N > kXorNSoftMax) {
    std::fprintf(stderr, "need --N in 4..%d\n", kXorNSoftMax);
    usage();
    return 1;
  }
  if (r < 8 || r > 64) {
    std::fprintf(stderr, "--r must be 8..64\n");
    return 1;
  }

  std::fprintf(stderr, "[xor_build_save] building N=%d r=%d → %s\n", N, r, out.c_str());
  XorFilter f = xor_build_pairs(N, r);
  if (!xor_save_file(out.c_str(), f)) return 1;
  std::printf("[xor_build_save] wrote %s  packed=%.2f GB  cells=%llu  r=%u  N=%u\n",
              out.c_str(), f.store_gb(), (unsigned long long)f.hdr.m_cells, f.hdr.r, f.hdr.N);
  return 0;
}
