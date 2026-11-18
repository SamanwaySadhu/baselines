// lr_gc_agmpc_selftest -- checks that need no network (the multi-party runs are in scripts/test_e2e.sh and
// scripts/test_tamper.sh, because emp-agmpc keeps process-wide state):
//   1. GF(2^128) doubling (known answers) and the invertibility of the garbling-hash input maps 2, 4, 6 and
//      their xors with the identity (3, 5, 7): PATCHES.md P1 relies on all six having full rank.
//   2. The float32 per-iteration circuits and their segmentation: plaintext evaluation of the chained segments equals
//      reference_train bit for bit, for several dimensions and segment sizes (needs the circuit files).
//   lr_gc_agmpc_selftest [--quick] [--circuits DIR]
#include "emp-agmpc/cmpc_config.h"
#include "lr_agmpc_circuit.h"

#include <array>
#include <cinttypes>
#include <functional>
#include <string>
#include <cstdio>

using namespace lrgc;

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static uint64_t lo64(const emp::block& b) { return (uint64_t)_mm_extract_epi64(b, 0); }
static uint64_t hi64(const emp::block& b) { return (uint64_t)_mm_extract_epi64(b, 1); }

// rank over GF(2) of the 128x128 matrix of x -> f(x)
template <class F>
static int rank128(F f) {
  std::vector<std::array<uint64_t, 2>> rows;
  for (int i = 0; i < 128; ++i) {
    const emp::block e = i < 64 ? emp::makeBlock(0, uint64_t(1) << i) : emp::makeBlock(uint64_t(1) << (i - 64), 0);
    const emp::block y = f(e);
    rows.push_back({lo64(y), hi64(y)});
  }
  int r = 0;
  for (int c = 0; c < 128 && r < 128; ++c) {
    const int w = c / 64;
    const uint64_t m = uint64_t(1) << (c % 64);
    int piv = -1;
    for (int i = r; i < 128; ++i) if (rows[(size_t)i][(size_t)w] & m) { piv = i; break; }
    if (piv < 0) continue;
    std::swap(rows[(size_t)r], rows[(size_t)piv]);
    for (int i = 0; i < 128; ++i)
      if (i != r && (rows[(size_t)i][(size_t)w] & m)) { rows[(size_t)i][0] ^= rows[(size_t)r][0]; rows[(size_t)i][1] ^= rows[(size_t)r][1]; }
    ++r;
  }
  return r;
}

static void test_gf() {
  std::printf("[GF(2^128) doubling and hash input maps]\n");
  // x * 1 = x; x * 2^127 * 2 = x^128 = x^7 + x^2 + x + 1 (0x87)
  CHECK(lo64(gf128_double(emp::makeBlock(0, 1))) == 2 && hi64(gf128_double(emp::makeBlock(0, 1))) == 0, "double(1) != 2");
  const emp::block top = emp::makeBlock(uint64_t(1) << 63, 0);
  CHECK(lo64(gf128_double(top)) == 0x87 && hi64(gf128_double(top)) == 0, "double(x^127) != 0x87");
  CHECK(lo64(gf128_double(emp::makeBlock(0, uint64_t(1) << 63))) == 0 && hi64(gf128_double(emp::makeBlock(0, uint64_t(1) << 63))) == 1, "carry into the high word");
  auto d = [](const emp::block& x) { return gf128_double(x); };
  struct M { const char* name; std::function<emp::block(const emp::block&)> f; };
  const M maps[] = {
      {"2", [&](const emp::block& x) { return d(x); }},
      {"4", [&](const emp::block& x) { return d(d(x)); }},
      {"6", [&](const emp::block& x) { return d(x) ^ d(d(x)); }},
      {"3", [&](const emp::block& x) { return d(x) ^ x; }},
      {"5", [&](const emp::block& x) { return d(d(x)) ^ x; }},
      {"7", [&](const emp::block& x) { return d(x) ^ d(d(x)) ^ x; }},
  };
  for (const auto& m : maps) {
    const int r = rank128(m.f);
    CHECK(r == 128, "map %s has rank %d", m.name, r);
    std::printf("  map x -> %s x: rank %d\n", m.name, r);
  }
  // The upstream map sigma(a) ^ sigma^2(b): sigma ^ sigma^2 is the identity, the root of the attack PATCHES.md P1 fixes.
  const int r_sig = rank128([](const emp::block& x) { return emp::sigma(x) ^ emp::sigma(emp::sigma(x)) ^ x; });
  CHECK(r_sig == 0, "sigma ^ sigma^2 ^ I has rank %d (expected 0: sigma ^ sigma^2 = I)", r_sig);
  std::printf("  upstream sigma ^ sigma^2 ^ identity: rank %d (= 0, i.e. sigma ^ sigma^2 is the identity)\n", r_sig);
}

// Plaintext run of the segmented, chained circuits (loaded from `dir`); returns the final weights as float32.
static std::vector<float> run_plain(const std::string& dir, const Params& P, uint64_t N, uint64_t* ands, uint64_t* segs, std::vector<std::vector<float>>* per_iter) {
  const Dataset D = generate_client_data(P);
  const auto sh = w0_shares(P, generate_initial_weights(P));
  std::vector<uint8_t> carried;
  std::vector<float> w;
  *ands = 0;
  *segs = 0;
  for (int t = 0; t < P.iters; ++t) {
    const IterCircuit IC = load_iteration(dir, P, t == 0);
    *ands += IC.C.ands;
    std::vector<uint8_t> in = carried;
    for (int k = 0; k < kBatch; ++k) {
      CHECK(IC.start[(size_t)k + 2] == in.size(), "client %d input block starts at %u, expected %zu", k + 1, IC.start[(size_t)k + 2], in.size());
      client_bits(in, P, D, sh, k, t);
      CHECK(IC.end[(size_t)k + 2] == in.size(), "client %d input block ends at %u, expected %zu", k + 1, IC.end[(size_t)k + 2], in.size());
    }
    const auto sgs = segment(IC.C, N);
    *segs += sgs.size();
    std::unordered_map<uint32_t, uint8_t> val;
    for (size_t si = 0; si < sgs.size(); ++si) {
      const Seg& sg = sgs[si];
      CHECK((uint64_t)sg.ands <= N, "segment with %d > %" PRIu64 " ANDs", sg.ands, N);
      std::vector<uint8_t> x;
      for (uint32_t g : sg.in_g) {
        auto it = val.find(g);
        if (it != val.end()) x.push_back(it->second);
        else {
          CHECK(g < IC.C.n_in, "segment %zu reads wire %u that no earlier segment produced", si, g);
          x.push_back(in[g]);
        }
      }
      const auto y = eval_segment(sg, x);
      for (size_t i = 0; i < y.size(); ++i) val[sg.out_g[i]] = y[i];
    }
    std::vector<uint8_t> out;
    for (uint32_t g = IC.C.num_wire - IC.C.n_out; g < IC.C.num_wire; ++g) out.push_back(val.at(g));
    w.assign((size_t)P.d, 0.0f);
    for (int j = 0; j < P.d; ++j) {
      uint32_t u = 0;
      for (int b = 0; b < 32; ++b) u |= (uint32_t)out[(size_t)(32 * j + b)] << b;
      w[(size_t)j] = u2f(u);
    }
    if (per_iter) per_iter->push_back(w);
    carried = out;
  }
  return w;
}

static bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) if (f2u(a[i]) != f2u(b[i])) return false;
  return true;
}

// The circuit files come from gen/gen_circuit (emp-tool 1.0): scripts/gen_circuits.sh --d "..." writes them to `dir`.
static void test_circuits(const std::string& dir, bool quick) {
  std::printf("[float32 per-iteration circuits (%s) + segmentation vs reference_train]\n", dir.c_str());
  struct Cfg { int d, iters; uint64_t seed; };
  std::vector<Cfg> cfgs = {{5, 20, 1}, {3, 4, 2}};
  if (!quick) cfgs.push_back({12, 6, 3});
  for (const auto& c : cfgs) {
    Params P;
    P.d = c.d; P.iters = c.iters; P.seed = c.seed;
    P.finalize();
    RefTrace tr;
    const auto ref = reference_train(P, generate_client_data(P), generate_initial_weights(P), &tr);
    for (uint64_t N : {uint64_t(1) << 40, uint64_t(100000), uint64_t(9973)}) {
      if (quick && N == 9973) continue;
      uint64_t ands = 0, segs = 0;
      std::vector<std::vector<float>> per;
      std::vector<float> w;
      try { w = run_plain(dir, P, N, &ands, &segs, &per); } catch (const std::exception& e) { CHECK(false, "%s", e.what()); return; }
      bool every = per.size() == tr.w_after.size();
      for (size_t t = 0; every && t < per.size(); ++t) every = same_bits(per[t], tr.w_after[t]);
      CHECK(same_bits(w, ref) && every, "d=%d iters=%d N=%" PRIu64 ": mismatch", c.d, c.iters, N);
      std::printf("  d=%-3d iters=%-2d seg<=%-13" PRIu64 " ANDs=%-9" PRIu64 " segments=%-4" PRIu64 " %s\n", c.d, c.iters, N, ands, segs,
                  (same_bits(w, ref) && every) ? "ok (every iteration bit-exact)" : "FAIL");
    }
  }
}

int main(int argc, char** argv) {
  bool quick = false;
  std::string dir = "circuits";
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--quick") quick = true;
    else if (std::string(argv[i]) == "--circuits" && i + 1 < argc) dir = argv[++i];
  }
  test_gf();
  test_circuits(dir, quick);
  if (g_fail) { std::printf("FAILED (%d)\n", g_fail); return 1; }
  std::printf("ALL PASSED\n");
  return 0;
}
