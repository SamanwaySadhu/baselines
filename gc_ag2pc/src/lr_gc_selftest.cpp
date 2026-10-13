// lr_gc_selftest -- offline checks of the LR training circuit (no networking).
//
//  1. sigmoid threshold tables are consistent with the long-double reference;
//  2. every arithmetic primitive equals int64 arithmetic, exhaustively over
//     small ranges (evaluated with emp::ClearCtx);
//  3. the circuit's quantized sigmoid equals the reference for every z in range;
//  4. full training circuits equal reference_train() bit for bit, many configs;
//  5. prints AND-gate counts for the default setting (iters=20, batch=4, qbits=6).
//
// Usage: lr_gc_selftest [--quick] [--counts-only] [--d-list 5,10,100]
#include "emp-tool/ir/context/clear.h"
#include "emp-tool/ir/context/count.h"

#include "lr_circuit.h"
#include "lr_params.h"
#include "lr_reference.h"
#include "sbit.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using namespace lrgc;
using Clear = emp::ClearCtx;

static int g_fail = 0;
static uint64_t g_checks = 0;

#define CHECK(cond, ...)                                    \
  do {                                                      \
    ++g_checks;                                             \
    if (!(cond)) {                                          \
      if (g_fail < 25) {                                    \
        std::printf("  [FAIL] %s:%d: ", __FILE__, __LINE__); \
        std::printf(__VA_ARGS__);                           \
        std::printf("\n");                                  \
      }                                                     \
      ++g_fail;                                             \
    }                                                       \
  } while (0)

// A secret ClearCtx integer holding value v, declared range [lo, hi], at `width` bits.
static SInt<Clear> secret(Builder<Clear>& B, int64_t v, int64_t lo, int64_t hi, int width) {
  std::vector<uint8_t> ws;
  for (bool bit : to_bits(v, width)) ws.push_back(bit ? 1 : 0);
  return B.from_wires(ws, lo, hi);
}

// Checks the SInt invariants: value inside [lo, hi] and width == width_for(lo, hi).
static void check_int(const SInt<Clear>& r, int64_t expect, const char* what, int64_t a, int64_t b) {
  const int64_t got = clear_value(r);
  CHECK(got == expect, "%s(a=%" PRId64 ", b=%" PRId64 "): got %" PRId64 " expected %" PRId64, what, a, b, got,
        expect);
  CHECK(r.lo <= expect && expect <= r.hi, "%s(a=%" PRId64 ", b=%" PRId64 "): value %" PRId64
        " outside range [%" PRId64 ", %" PRId64 "]", what, a, b, expect, r.lo, r.hi);
  CHECK(r.width() == width_for(r.lo, r.hi), "%s: width %d != width_for(range) %d", what, r.width(),
        width_for(r.lo, r.hi));
}

static void test_arithmetic(bool quick) {
  std::printf("[2] arithmetic primitives vs int64 (exhaustive over small ranges)\n");
  struct R { int64_t lo, hi; };
  const std::vector<R> ranges = quick
      ? std::vector<R>{{-9, 7}, {0, 13}, {-12, -1}, {5, 5}}
      : std::vector<R>{{-9, 7}, {0, 13}, {-12, -1}, {5, 5}, {-1, 0}, {-33, 40}, {-64, 63}, {0, 1}, {-3, -3}};
  Clear ctx;
  Builder<Clear> B(ctx);
  for (const R& ra : ranges)
    for (const R& rb : ranges)
      for (int extra = 0; extra <= 1; ++extra) {  // also feed operands wider than needed
        const int wa = width_for(ra.lo, ra.hi) + extra, wb = width_for(rb.lo, rb.hi) + extra;
        for (int64_t a = ra.lo; a <= ra.hi; ++a)
          for (int64_t b = rb.lo; b <= rb.hi; ++b) {
            const SInt<Clear> A = secret(B, a, ra.lo, ra.hi, wa), Bv = secret(B, b, rb.lo, rb.hi, wb);
            check_int(B.add(A, Bv), a + b, "add", a, b);
            check_int(B.sub(A, Bv), a - b, "sub", a, b);
            check_int(B.mul(A, Bv), a * b, "mul", a, b);
            check_int(B.sum({A, Bv, A}), a + b + a, "sum", a, b);
            const SBit<Clear> lt = B.lt(A, Bv), ge = B.ge(A, Bv);
            const bool ltv = lt.is_const() ? lt.k == 1 : lt.w == 1;
            const bool gev = ge.is_const() ? ge.k == 1 : ge.w == 1;
            CHECK(ltv == (a < b), "lt(%" PRId64 ", %" PRId64 ")", a, b);
            CHECK(gev == (a >= b), "ge(%" PRId64 ", %" PRId64 ")", a, b);
            for (int sel = 0; sel <= 1; ++sel) {
              const SBit<Clear> s = Builder<Clear>::wire((uint8_t)sel);
              check_int(B.mux(s, A, Bv), sel ? a : b, "mux", a, b);
            }
          }
      }
  for (const R& ra : ranges)
    for (int64_t a = ra.lo; a <= ra.hi; ++a) {
      const SInt<Clear> A = secret(B, a, ra.lo, ra.hi, width_for(ra.lo, ra.hi));
      for (int s = 0; s <= 6; ++s) {
        check_int(B.shl(A, s), a * (int64_t(1) << s), "shl", a, s);
        check_int(B.ashr(A, s), a >> s, "ashr", a, s);
        check_int(B.tdiv_pow2(A, s), a / (int64_t(1) << s), "tdiv_pow2", a, s);
      }
      for (int64_t c = -70; c <= 70; ++c) check_int(B.mul_const(A, c), a * c, "mul_const", a, c);
    }
  // A public table lookup through one-hot minterms over 3 secret bits.
  const std::vector<int64_t> table{-5, 17, 3, 3, -40, 0, 63, 9};
  for (int idx = 0; idx < 8; ++idx) {
    std::vector<SBit<Clear>> mt{Builder<Clear>::cst(true)};
    for (int l = 2; l >= 0; --l) mt = B.extend_minterms(mt, Builder<Clear>::wire((uint8_t)((idx >> l) & 1)));
    check_int(B.select_public(mt, table), table[(size_t)idx], "select_public", idx, 0);
  }
}

static void test_sigmoid_tables() {
  std::printf("[1] sigmoid threshold tables (qbits 2..12)\n");
  for (int q = 2; q <= 12; ++q) {
    const int f = q - 1;
    const SigmoidTable S = make_sigmoid_table(f);
    for (int64_t v = 1; v <= S.maxq; ++v) {
      CHECK(qsigmoid_ref(S.t[(size_t)v], f) >= v, "t[%" PRId64 "] too small (qbits=%d)", v, q);
      CHECK(qsigmoid_ref(S.t[(size_t)v] - 1, f) < v, "t[%" PRId64 "] not minimal (qbits=%d)", v, q);
    }
    // Spot-check far outside the scan window too.
    for (int64_t z : {int64_t(-1) << 40, int64_t(-1) << 30, int64_t(1) << 30, int64_t(1) << 40})
      CHECK(qsigmoid_by_table(S, z) == qsigmoid_ref(z, f), "table vs ref at z=%" PRId64, z);
  }
}

static void test_sigmoid_circuit(bool quick) {
  std::printf("[3] circuit sigmoid vs reference (every z in range)\n");
  Clear ctx;
  Builder<Clear> B(ctx);
  for (int q = 2; q <= (quick ? 6 : 9); ++q) {
    const Params P = [&] { Params p; p.qbits = q; p.finalize(); return p; }();
    const SigmoidTable S = make_sigmoid_table(P.f);
    LRCircuit<Clear> lr(B, P, S);
    const int64_t span = S.t[(size_t)S.maxq] - S.t[1];
    struct R { int64_t lo, hi; };
    std::vector<R> ranges{{S.t[1] - span / 2 - 7, S.t[(size_t)S.maxq] + span / 2 + 7},  // covers all thresholds
                          {S.t[1] + span / 3, S.t[1] + span / 3 + 50},                 // narrow, interior
                          {-3, 4}};
    if (!quick) ranges.push_back({S.t[1] - 3 * span, S.t[(size_t)S.maxq] + 3 * span});
    for (const R& r : ranges) {
      const int w = width_for(r.lo, r.hi);
      for (int64_t z = r.lo; z <= r.hi; ++z) {
        const SInt<Clear> Z = secret(B, z, r.lo, r.hi, w);
        const SInt<Clear> Y = lr.qsigmoid(Z);
        CHECK(clear_value(Y) == qsigmoid_ref(z, P.f), "qsigmoid(z=%" PRId64 ", qbits=%d): got %" PRId64
              " expected %" PRId64, z, q, clear_value(Y), qsigmoid_ref(z, P.f));
      }
    }
  }
}

static void test_training(bool quick) {
  std::printf("[4] full training circuit vs reference_train (bit-exact)\n");
  struct Cfg { int d, iters, batch, qbits; double lr; bool bias, zero; };
  std::vector<Cfg> cfgs{
      {5, 20, 4, 6, 0.5, true, false},  {5, 20, 4, 6, 0.5, true, true},   {1, 5, 4, 6, 0.5, true, false},
      {2, 20, 4, 6, 0.5, true, false},  {3, 7, 4, 6, 0.5, false, false},  {17, 20, 4, 6, 0.5, true, false},
      {5, 20, 4, 6, 1.0, true, false},  {5, 20, 4, 6, 4.0, true, false},  {5, 20, 4, 6, 0.125, true, false},
      {6, 10, 1, 6, 0.25, true, false}, {6, 10, 2, 6, 0.5, true, false},  {4, 10, 8, 6, 0.5, true, false},
      {5, 20, 4, 4, 0.5, true, false},  {5, 20, 4, 8, 0.5, true, false},  {5, 12, 4, 2, 0.5, true, false},
      {5, 6, 4, 10, 0.5, true, false},  {40, 20, 4, 6, 0.5, true, false},
  };
  if (!quick) {
    cfgs.push_back({100, 20, 4, 6, 0.5, true, false});
    cfgs.push_back({33, 20, 4, 7, 2.0, true, false});
  }
  const int seeds = quick ? 2 : 6;
  for (const Cfg& c : cfgs)
    for (int s = 1; s <= seeds; ++s) {
      Params P;
      P.d = c.d; P.iters = c.iters; P.batch = c.batch; P.qbits = c.qbits; P.lr = c.lr;
      P.bias = c.bias; P.zero_init = c.zero; P.seed = (uint64_t)(1000 * s + c.d);
      P.finalize();
      const SigmoidTable S = make_sigmoid_table(P.f);
      const Dataset D = generate_client_data(P);
      const std::vector<int64_t> w0 = generate_server_weights(P);
      RefTrace tr;
      const std::vector<int64_t> ref = reference_train(P, D, w0, &tr);
      Clear ctx;
      Builder<Clear> B(ctx);
      const auto w = train_offline(B, P, S, D, w0);
      for (int j = 0; j < P.d; ++j) {
        CHECK(clear_value(w[(size_t)j]) == ref[(size_t)j], "train %s: w[%d] got %" PRId64 " expected %" PRId64,
              params_string(P).c_str(), j, clear_value(w[(size_t)j]), ref[(size_t)j]);
        CHECK(w[(size_t)j].lo <= ref[(size_t)j] && ref[(size_t)j] <= w[(size_t)j].hi, "train: range violated");
      }
    }
}

// Gate-recording context to measure how many emitted ANDs are live (reachable from
// the outputs), i.e. what the protocol garbles after its dead-code elimination.
struct TapeCtx {
  using Wire = uint32_t;
  struct G { uint32_t a, b; uint8_t op; };  // 0/1 const, 2 and, 3 xor, 4 not
  std::vector<G> g;
  Wire push(uint32_t a, uint32_t b, uint8_t op) { g.push_back({a, b, op}); return (Wire)(g.size() - 1); }
  Wire public_bit(bool v) { return push(0, 0, v ? 1 : 0); }
  Wire and_gate(Wire a, Wire b) { return push(a, b, 2); }
  Wire xor_gate(Wire a, Wire b) { return push(a, b, 3); }
  Wire not_gate(Wire a) { return push(a, 0, 4); }
};

static uint64_t live_ands(const TapeCtx& T, const std::vector<SInt<TapeCtx>>& outs) {
  std::vector<char> live(T.g.size(), 0);
  std::vector<uint32_t> st;
  for (const auto& o : outs)
    for (const auto& b : o.b)
      if (!b.is_const()) st.push_back(b.w);
  uint64_t ands = 0;
  while (!st.empty()) {
    const uint32_t w = st.back();
    st.pop_back();
    if (live[w]) continue;
    live[w] = 1;
    const auto& gt = T.g[w];
    if (gt.op == 2) ++ands;
    if (gt.op >= 2) st.push_back(gt.a);
    if (gt.op == 2 || gt.op == 3) st.push_back(gt.b);
  }
  return ands;
}

static void print_counts(const std::vector<int>& dlist) {
  std::printf("[5] gate counts (iters=20, batch=4, qbits=6, lr=0.5, bias column; ANDs are what GC pays for)\n");
  std::printf("    %6s %14s %14s %14s %12s %14s\n", "d", "AND (total)", "AND/iter", "XOR+NOT", "inputs(bits)",
              "live AND");
  for (int d : dlist) {
    Params P;
    P.d = d;
    P.finalize();
    const SigmoidTable S = make_sigmoid_table(P.f);
    const Dataset D = generate_client_data(P);
    const std::vector<int64_t> w0 = generate_server_weights(P);
    emp::CountCtx cc;
    Builder<emp::CountCtx> B(cc);
    (void)train_offline(B, P, S, D, w0);
    CHECK(B.stats.ands == cc.ands && B.stats.xors == cc.xors && B.stats.nots == cc.nots,
          "Builder stats disagree with emp::CountCtx");
    const uint64_t in_bits = (uint64_t)P.iters * P.batch * (P.d_data * P.qbits + 1) + (uint64_t)P.d * P.qbits;
    char live[32] = "-";
    if (d <= 200) {
      TapeCtx T;
      Builder<TapeCtx> BT(T);
      const auto out = train_offline(BT, P, S, D, w0);
      std::snprintf(live, sizeof live, "%" PRIu64, live_ands(T, out));
    }
    std::printf("    %6d %14" PRIu64 " %14" PRIu64 " %14" PRIu64 " %12" PRIu64 " %14s\n", d, cc.ands,
                cc.ands / (uint64_t)P.iters, cc.xors + cc.nots, in_bits, live);
  }
}

int main(int argc, char** argv) {
  bool quick = false, counts_only = false;
  std::vector<int> dlist{2, 5, 10, 20, 50, 100, 200, 500, 1000};
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--quick") quick = true;
    else if (a == "--counts-only") counts_only = true;
    else if (a == "--d-list" && i + 1 < argc) {
      dlist.clear();
      std::string s = argv[++i];
      size_t p = 0;
      while (p < s.size()) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        dlist.push_back(std::stoi(s.substr(p, q - p)));
        p = q + 1;
      }
    } else {
      std::printf("usage: %s [--quick] [--counts-only] [--d-list 5,10,100]\n", argv[0]);
      return 2;
    }
  }
  const auto t0 = std::chrono::steady_clock::now();
  try {
    if (!counts_only) {
      test_sigmoid_tables();
      test_arithmetic(quick);
      test_sigmoid_circuit(quick);
      test_training(quick);
    }
    print_counts(dlist);
  } catch (const std::exception& e) {
    std::printf("EXCEPTION: %s\n", e.what());
    return 1;
  }
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%s: %" PRIu64 " checks, %d failures (%.1f s)\n", g_fail ? "FAILED" : "ALL TESTS PASSED", g_checks,
              g_fail, s);
  return g_fail ? 1 : 0;
}
