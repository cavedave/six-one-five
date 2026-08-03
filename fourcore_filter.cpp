// =============================================================================
// fourcore_filter.cpp — deep "floursum" prefilter for cls1 four-core jobs
//
// Necessary condition: T = x1^6+x2^6+x3^6+x4^6 must lie in the 4-sum cone of
// sixth powers mod a 2/3/7-adic tower (16,32,27,81,49,343). Safe to drop jobs
// that fail any layer (no false rejects of real solutions).
//
// Typical keep ≈ 14% on post-wall cls1 .but streams.
//
// Usage:
//   ./fourcore_filter < in.but > out.deep.but \
//       [--write-rej out.rej.but] 2> filter.log
//   ./fourcore_filter --selftest
//
// Job line (same as fourcore_hunt / fourcore_find4):
//   B u T_lo T_hi
// =============================================================================

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using u128 = unsigned __int128;

struct ModLayer {
  u32 m = 0;
  const char* name = nullptr;
  const char* predict = nullptr;  // closed form if known, else "hand-check!"
  std::vector<u8> ok;             // ok[r] = 1 iff r is a 4-sum of sixth powers mod m
  u32 n_surv = 0;
};

static u32 mod_pow6(u32 x, u32 m) {
  u64 a = x % m;
  u64 x2 = (a * a) % m;
  u64 x4 = (x2 * x2) % m;
  return (u32)((x2 * x4) % m);
}

static void build_layer(ModLayer& L) {
  const u32 m = L.m;
  std::vector<u8> is6(m, 0);
  for (u32 x = 0; x < m; ++x) is6[mod_pow6(x, m)] = 1;
  std::vector<u32> sixths;
  for (u32 r = 0; r < m; ++r)
    if (is6[r]) sixths.push_back(r);

  // Exactly four sixth powers (0^6 allowed) via iterated sumset.
  std::vector<u8> cur(m, 0), nxt(m, 0);
  cur[0] = 1;
  for (int k = 0; k < 4; ++k) {
    std::fill(nxt.begin(), nxt.end(), 0);
    for (u32 a = 0; a < m; ++a) {
      if (!cur[a]) continue;
      for (u32 b : sixths) nxt[(a + b) % m] = 1;
    }
    cur.swap(nxt);
  }
  L.ok = std::move(cur);
  L.n_surv = 0;
  for (u32 r = 0; r < m; ++r)
    if (L.ok[r]) ++L.n_surv;
}

static std::array<ModLayer, 6> make_layers() {
  std::array<ModLayer, 6> L{{
      {16, "mod16", "9/16", {}, 0},
      {32, "mod32", "hand-check!", {}, 0},
      {27, "mod27", "13/27", {}, 0},
      {81, "mod81", "hand-check!", {}, 0},
      {49, "mod49", "hand-check!", {}, 0},
      {343, "mod343", "hand-check!", {}, 0},
  }};
  for (auto& layer : L) build_layer(layer);
  return L;
}

static u32 umod(u128 T, u32 m) {
  return (u32)(T % (u128)m);
}

static int selftest() {
  auto L = make_layers();
  const u32 expect[6] = {9, 17, 13, 37, 29, 197};
  printf("[selftest] floursum 4-sum sixth-power cones\n");
  bool ok = true;
  for (int i = 0; i < 6; ++i) {
    printf("  %s survivors=%u/%u (%.4f) predict %s\n", L[i].name, L[i].n_surv,
           L[i].m, (double)L[i].n_surv / L[i].m, L[i].predict);
    if (L[i].n_surv != expect[i]) {
      printf("FAIL expected %u\n", expect[i]);
      ok = false;
    }
  }
  // Spot: T=1^6+2^6+2^6+2^6 must survive every layer.
  u128 T = 1;
  T += 64;  // 2^6
  T += 64;
  T += 64;
  for (auto& layer : L) {
    if (!layer.ok[umod(T, layer.m)]) {
      printf("FAIL planted T rejected mod %u\n", layer.m);
      ok = false;
    }
  }
  printf(ok ? "[selftest] PASS\n" : "[selftest] FAIL\n");
  return ok ? 0 : 1;
}

static void usage() {
  fprintf(stderr,
          "usage: fourcore_filter [--selftest] [--write-rej FILE.rej.but]\n"
          "  reads .but lines (B u T_lo T_hi) from stdin\n"
          "  writes survivors to stdout; optional rejects to --write-rej\n");
}

int main(int argc, char** argv) {
  const char* rej_path = nullptr;
  bool do_self = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--selftest") do_self = true;
    else if (a == "--write-rej" && i + 1 < argc) rej_path = argv[++i];
    else if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else {
      fprintf(stderr, "unknown %s\n", a.c_str());
      usage();
      return 1;
    }
  }
  if (do_self) return selftest();

  auto layers = make_layers();
  fprintf(stderr, "[built] floursum residue tables (survivors = 4-sums of 6th powers mod m):\n");
  for (auto& L : layers) {
    fprintf(stderr, "  %-7s survivors=%3u/%-3u  (%.4f)   predict %s   c2e64=0\n",
            L.name, L.n_surv, L.m, (double)L.n_surv / L.m, L.predict);
  }

  FILE* frej = nullptr;
  if (rej_path) {
    frej = fopen(rej_path, "w");
    if (!frej) {
      fprintf(stderr, "cannot open --write-rej %s\n", rej_path);
      return 1;
    }
  }

  u64 jobs_in = 0, kept = 0;
  u64 keep_layer[6] = {0, 0, 0, 0, 0, 0};
  char line[512];
  while (fgets(line, sizeof line, stdin)) {
    if (line[0] == '#' || line[0] == '\n') continue;
    unsigned long long B = 0, u = 0, tlo = 0, thi = 0;
    if (sscanf(line, "%llu %llu %llu %llu", &B, &u, &tlo, &thi) != 4) continue;
    ++jobs_in;
    u128 T = (u128)tlo | ((u128)thi << 64);

    bool pass = true;
    bool layer_ok[6];
    for (int i = 0; i < 6; ++i) {
      layer_ok[i] = layers[i].ok[umod(T, layers[i].m)] != 0;
      if (layer_ok[i]) ++keep_layer[i];
      if (!layer_ok[i]) pass = false;
    }
    if (pass) {
      ++kept;
      fputs(line, stdout);
    } else if (frej) {
      fputs(line, frej);
    }
  }
  if (frej) fclose(frej);

  const double joint = jobs_in ? (double)kept / (double)jobs_in : 0.0;
  fprintf(stderr, "[stats] jobs_in=%llu kept=%llu joint=%.4f (per-layer survival above)\n",
          (unsigned long long)jobs_in, (unsigned long long)kept, joint);
  for (int i = 0; i < 6; ++i) {
    const double kr = jobs_in ? (double)keep_layer[i] / (double)jobs_in : 0.0;
    fprintf(stderr, "  %-7s keep=%.4f\n", layers[i].name, kr);
  }
  return 0;
}
