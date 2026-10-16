// lr_gc_agmpc_selftest -- checks that need no network (the multi-party runs are in scripts/test_e2e.sh and
// scripts/test_tamper.sh, because emp-agmpc keeps process-wide state):
//   1. GF(2^128) doubling (known answers) and the invertibility of the garbling-hash input maps 2, 4, 6 and
//      their xors with the identity (3, 5, 7): PATCHES.md P1 relies on all six having full rank.
//   2. The per-iteration circuits and their segmentation: plaintext evaluation of the chained segments equals
//      reference_train bit for bit, for many configurations and segment sizes; AND counts equal gc_ag2pc's.
//   lr_gc_agmpc_selftest [--quick]
#include "emp-agmpc/cmpc_config.h"
#include "lr_agmpc_circuit.h"

#include <array>
#include <cinttypes>
#include <functional>
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

// Plaintext run of the segmented, chained circuits; returns the final weights (and the AND count).
static std::vector<int64_t> run_plain(const Params& P, uint64_t N, uint64_t* ands, uint64_t* segs, std::vector<std::vector<int64_t>>* per_iter) {
  const SigmoidTable S = make_sigmoid_table(P.f);
  const Dataset D = generate_client_data(P);
  const auto sh = w0_shares(P, generate_initial_weights(P));
  Layout prev;
  std::vector<uint8_t> carried;
  std::vector<int64_t> w;
  *ands = 0;
  *segs = 0;
  for (int t = 0; t < P.iters; ++t) {
    IterCircuit IC = build_iteration(P, S, t, prev);
    *ands += IC.C.ands;
    std::vector<uint8_t> in = carried;
    for (int k = 0; k < P.batch; ++k) {
      CHECK(IC.start[(size_t)k + 2] == in.size(), "client %d input block starts at %u, expected %zu", k + 1, IC.start[(size_t)k + 2], in.size());
      client_bits(in, P, D, sh, k, t);
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
    w = decode(IC.out_w, out);
    if (per_iter) per_iter->push_back(w);
    carried = out;
    prev = IC.out_w;
  }
  return w;
}

static void test_circuits(bool quick) {
  std::printf("[per-iteration circuits + segmentation vs reference_train]\n");
  struct Cfg { int d, iters, batch, qbits; double lr; bool bias, zero; uint64_t seed; };
  std::vector<Cfg> cfgs = {{5, 20, 4, 6, 0.5, true, false, 1}, {1, 3, 4, 6, 0.5, true, false, 2}, {2, 5, 1, 6, 0.25, true, false, 3},
                           {7, 6, 2, 6, 1.0, false, true, 4}, {3, 4, 3, 8, 0.75, true, false, 5}, {16, 4, 4, 6, 0.5, true, false, 6}};
  if (!quick) { cfgs.push_back({50, 3, 4, 6, 0.5, true, false, 7}); cfgs.push_back({5, 40, 4, 6, 1.0, true, false, 8}); cfgs.push_back({20, 5, 8, 6, 2.0, true, false, 9}); }
  for (const auto& c : cfgs) {
    Params P;
    P.d = c.d; P.iters = c.iters; P.batch = c.batch; P.qbits = c.qbits; P.lr = c.lr; P.bias = c.bias; P.zero_init = c.zero; P.seed = c.seed;
    try { P.finalize(); } catch (const std::exception& e) { std::printf("  skip (%s)\n", e.what()); continue; }
    RefTrace tr;
    const auto ref = reference_train(P, generate_client_data(P), generate_initial_weights(P), &tr);
    for (uint64_t N : {uint64_t(1) << 40, uint64_t(20000), uint64_t(997), uint64_t(53)}) {
      if (quick && N == 53 && c.d > 5) continue;
      uint64_t ands = 0, segs = 0;
      std::vector<std::vector<int64_t>> per;
      const auto w = run_plain(P, N, &ands, &segs, &per);
      bool every = per.size() == tr.w_after.size();
      for (size_t t = 0; every && t < per.size(); ++t) every = per[t] == tr.w_after[t];
      CHECK(w == ref && every, "d=%d iters=%d batch=%d N=%" PRIu64 ": mismatch", c.d, c.iters, c.batch, N);
      if (N == (uint64_t(1) << 40)) {
        BristolRec R0;
        R0.input(1);
        Builder<BristolRec> B0(R0);
        train_offline(B0, P, make_sigmoid_table(P.f), generate_client_data(P), generate_initial_weights(P));
        // per-iteration circuits drop dead gates, so they can be slightly smaller than gc_ag2pc's single circuit
        CHECK(ands <= B0.stats.ands, "AND count %" PRIu64 " > gc_ag2pc's %" PRIu64, ands, (uint64_t)B0.stats.ands);
        if (ands != B0.stats.ands) std::printf("  (gc_ag2pc's circuit: %" PRIu64 " ANDs; %" PRIu64 " dead ANDs removed here)\n", (uint64_t)B0.stats.ands, (uint64_t)B0.stats.ands - ands);
      }
      std::printf("  d=%-3d iters=%-2d batch=%d qbits=%d lr=%.2f%s%s seg<=%-13" PRIu64 " ANDs=%-8" PRIu64 " segments=%-4" PRIu64 " %s\n", c.d, c.iters,
                  c.batch, c.qbits, c.lr, c.bias ? "" : " no-bias", c.zero ? " zero-init" : "", N, ands, segs,
                  (w == ref && every) ? "ok (every iteration bit-exact)" : "FAIL");
    }
  }
}

int main(int argc, char** argv) {
  const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
  test_gf();
  test_circuits(quick);
  if (g_fail) { std::printf("FAILED (%d)\n", g_fail); return 1; }
  std::printf("ALL PASSED\n");
  return 0;
}
