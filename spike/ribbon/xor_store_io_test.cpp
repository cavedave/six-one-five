// Host xor save/load roundtrip (lean Step 2). No GPU.
// Build: make xor-io-test   (from repo root)
//
// Checks:
//   1) xor_build_pairs → xor_save_file → xor_load_file preserves header + FN=0
//   2) expected_N mismatch refuses the file
//   3) xor_fpr_smoke at r=48 stays under the campaign refuse threshold (16/200k)

#include "../../fourcore_xor_store.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

static int g_fails = 0;

static void expect(bool cond, const char* msg) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    ++g_fails;
  }
}

static std::string make_temp_path() {
  char tmpl[] = "/tmp/xor_store_io_XXXXXX";
  const int fd = ::mkstemp(tmpl);
  if (fd < 0) {
    std::perror("mkstemp");
    std::exit(2);
  }
  ::close(fd);
  ::unlink(tmpl);  // save will recreate; path stays unique enough for this process
  return std::string(tmpl) + ".bin";
}

static std::uint64_t pair_key(int i, int j) {
  auto p6 = [](int x) -> std::uint64_t {
    const std::uint64_t x2 = (std::uint64_t)x * (std::uint64_t)x;
    return x2 * x2 * x2;
  };
  return p6(i) + p6(j);
}

static void test_roundtrip_and_mismatch() {
  constexpr int N = 300;
  constexpr int r = 48;
  const std::string path = make_temp_path();

  XorFilter built = xor_build_pairs(N, r);
  expect(built.hdr.N == (std::uint32_t)N, "built hdr.N");
  expect(built.hdr.r == (std::uint32_t)r, "built hdr.r");
  expect(built.hdr.n_keys > 0, "built n_keys");
  expect(!built.packed.empty(), "built packed non-empty");

  expect(xor_save_file(path.c_str(), built), "xor_save_file");

  XorFilter loaded;
  expect(xor_load_file(path.c_str(), loaded, N), "xor_load_file exact N");
  expect(loaded.hdr.N == built.hdr.N, "loaded N");
  expect(loaded.hdr.r == built.hdr.r, "loaded r");
  expect(loaded.hdr.m_cells == built.hdr.m_cells, "loaded m_cells");
  expect(loaded.hdr.n_keys == built.hdr.n_keys, "loaded n_keys");
  expect(loaded.hdr.mix_seed == built.hdr.mix_seed, "loaded mix_seed");
  expect(loaded.packed.size() == built.packed.size(), "loaded packed size");
  expect(loaded.packed == built.packed, "loaded packed bytes identical");

  // FN=0 on a dense sample of real pair keys (full N=300 set is ~45k — cheap).
  std::size_t fn = 0, checked = 0;
  for (int i = 1; i <= N; ++i) {
    for (int j = i; j <= N; ++j) {
      ++checked;
      if (!xor_might_contain(loaded, pair_key(i, j))) ++fn;
    }
  }
  expect(fn == 0, "roundtrip: FN count must be 0");
  std::printf("  roundtrip N=%d r=%d keys_checked=%zu FN=%zu packed=%.3f MB\n", N, r, checked, fn,
              loaded.store_gb() * 1e3);

  XorFilter wrong_n;
  expect(!xor_load_file(path.c_str(), wrong_n, N + 1), "N mismatch must fail");
  expect(wrong_n.packed.empty(), "failed N-mismatch load leaves packed empty");

  const int smoke = xor_fpr_smoke(loaded);
  expect(smoke >= 0, "xor_fpr_smoke ran");
  expect(smoke <= 16, "FPR smoke <= 16/200k (campaign refuse threshold)");
  std::printf("  FPR smoke hits=%d/200000 (limit 16)\n", smoke);

  ::unlink(path.c_str());
}

int main() {
  std::fprintf(stderr, "xor_store_io_test: starting\n");
  test_roundtrip_and_mismatch();
  if (g_fails) {
    std::fprintf(stderr, "xor_store_io_test: %d FAIL(s)\n", g_fails);
    return 1;
  }
  std::fprintf(stderr, "xor_store_io_test: ALL PASS\n");
  return 0;
}
