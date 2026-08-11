#pragma once
// Packed binary job file for fourcore hunt → find (.bbj).
// Much smaller than ASCII .but (~36 B/job vs ~50–80 B text); zstd-friendly.
//
// Layout (little-endian):
//   JobBinHeader
//   JobBinRec × header.n_jobs
//
// Extension convention: *.bbj  (auto-detected by hunt/find)

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace fc_jobbin {

constexpr std::uint32_t kMagic = 0x4a424346u;  // 'FCJB' LE
constexpr std::uint32_t kVersion = 2;

#pragma pack(push, 1)
struct JobBinHeader {
  std::uint32_t magic = kMagic;
  std::uint32_t version = kVersion;
  std::uint64_t n_jobs = 0;
  std::uint32_t rec_bytes = 36;  // sizeof(JobBinRec)
  std::uint32_t min_B = 0;
  std::uint32_t max_B = 0;
  std::uint32_t pad = 0;
};

struct JobBinRec {
  std::uint8_t cls = 0;
  std::uint8_t pad[3] = {};
  std::uint32_t B = 0;
  std::uint32_t u = 0;
  std::uint32_t free1 = 0;
  std::uint32_t free2 = 0;
  std::uint64_t T_lo = 0;
  std::uint64_t T_hi = 0;
};
#pragma pack(pop)

static_assert(sizeof(JobBinRec) == 36, "JobBinRec must be 36 bytes");

inline bool is_bbj_path(const std::string& path) {
  if (path.size() < 4) return false;
  const char* e = path.c_str() + path.size() - 4;
  return (e[0] == '.' && (e[1] == 'b' || e[1] == 'B') && (e[2] == 'b' || e[2] == 'B') &&
          (e[3] == 'j' || e[3] == 'J'));
}

inline bool header_ok(const JobBinHeader& h) {
  return h.magic == kMagic && h.version == kVersion && h.rec_bytes == sizeof(JobBinRec);
}

inline bool write_header(FILE* fp, std::uint64_t n_jobs, std::uint32_t min_B = 0,
                         std::uint32_t max_B = 0) {
  JobBinHeader h;
  h.n_jobs = n_jobs;
  h.min_B = min_B;
  h.max_B = max_B;
  return std::fwrite(&h, sizeof(h), 1, fp) == 1;
}

// Rewrite header after streaming writes (seek to start).
inline bool patch_header(FILE* fp, std::uint64_t n_jobs, std::uint32_t min_B,
                         std::uint32_t max_B) {
  if (std::fseek(fp, 0, SEEK_SET) != 0) return false;
  return write_header(fp, n_jobs, min_B, max_B);
}

inline bool write_rec(FILE* fp, std::uint8_t cls, std::uint32_t B, std::uint32_t u,
                      std::uint32_t free1, std::uint32_t free2, std::uint64_t tlo,
                      std::uint64_t thi) {
  JobBinRec r;
  r.cls = cls;
  r.B = B;
  r.u = u;
  r.free1 = free1;
  r.free2 = free2;
  r.T_lo = tlo;
  r.T_hi = thi;
  return std::fwrite(&r, sizeof(r), 1, fp) == 1;
}

inline bool read_header(FILE* fp, JobBinHeader& h) {
  if (std::fread(&h, sizeof(h), 1, fp) != 1) return false;
  return header_ok(h);
}

inline bool read_rec(FILE* fp, JobBinRec& r) {
  return std::fread(&r, sizeof(r), 1, fp) == 1;
}

}  // namespace fc_jobbin
