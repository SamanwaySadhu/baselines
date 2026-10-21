// lr_gc_selftest -- offline checks of the fixed-point LR training circuit (no networking).
//
// Spec (lr_reference.h): inputs are qbits-bit quantized values (default qbits = 6:
// integers in [-32, 31] at scale 2^f, f = 5); every dot product (scale 2^(2f)) is
// truncated back to scale 2^f by an arithmetic shift right by f (floor), keeping the
// integer bits; yhat = min(floor(2^f * sigmoid(z / 2^f)), MAXQ).
//
//  1. sigmoid threshold tables agree with qsigmoid_ref() for every z of a wide window
//     around the transitions, and with the closed form t[v] = ceil(2^f ln(v/(2^f-v)));
//     the quantized sigmoid is monotone and saturating;
//  2. every Builder primitive equals int64 arithmetic, exhaustively over small ranges
//     (evaluated with emp::ClearCtx), with the range / width invariants of sbit.h;
//  3. truncation after dot products: LRCircuit::truncate(dot(a, b)) equals
//     trunc_f(dot_exact(a, b)) on random and edge vectors, including negative dot
//     products (floor != round-toward-zero) and results far outside qbits bits;
//  4. the circuit's quantized sigmoid equals qsigmoid_ref() for every z in realistic
//     secret ranges that cover both saturation regions;
//  5. full training circuits equal reference_train() bit for bit, many configs;
//  6. AND-gate counts for iters = 20, batch = 4, qbits = 6; every emitted AND is live
//     (reachable from the outputs) in many configs, so the count is what AG2PC garbles.
//
// usage: lr_gc_selftest [--quick]
#include "emp-tool/ir/context/clear.h"
#include "emp-tool/ir/context/count.h"

#include "lr_circuit.h"
#include "lr_params.h"
#include "lr_reference.h"
#include "sbit.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace lrgc;
using Clear = emp::ClearCtx;

static int g_fail = 0;
static uint64_t g_checks = 0;

#define CHECK(cond, ...)                                     \
  do {                                                       \
    ++g_checks;                                              \
    if (!(cond)) {                                           \
      if (g_fail < 25) {                                     \
        std::printf("  [FAIL] %s:%d: ", __FILE__, __LINE__); \
        std::printf(__VA_ARGS__);                            \
        std::printf("\n");                                   \
      }                                                      \
      ++g_fail;                                              \
    }                                                        \
  } while (0)

static Params make_params(int qbits) {
  Params P;
  P.qbits = qbits;
  P.finalize();
  return P;
}

// A secret ClearCtx integer holding value v, declared range [lo, hi], at `width` bits.
static SInt<Clear> secret(Builder<Clear>& B, int64_t v, int64_t lo, int64_t hi, int width) {
  std::vector<uint8_t> ws;
  for (bool bit : to_bits(v, width)) ws.push_back(bit ? 1 : 0);
  return B.from_wires(ws, lo, hi);
}

static bool bit_value(const SBit<Clear>& b) { return b.is_const() ? b.k == 1 : b.w == 1; }

// Checks the value and the SInt invariants: value inside [lo, hi], width == width_for(lo, hi).
static void check_int(const SInt<Clear>& r, int64_t expect, const char* what, int64_t a, int64_t b) {
  const int64_t got = clear_value(r);
  CHECK(got == expect, "%s(a=%" PRId64 ", b=%" PRId64 "): got %" PRId64 " expected %" PRId64, what, a, b, got,
        expect);
  CHECK(r.lo <= expect && expect <= r.hi, "%s(a=%" PRId64 ", b=%" PRId64 "): value %" PRId64
        " outside range [%" PRId64 ", %" PRId64 "]", what, a, b, expect, r.lo, r.hi);
  CHECK(r.width() == width_for(r.lo, r.hi), "%s: width %d != width_for(range) %d", what, r.width(),
        width_for(r.lo, r.hi));
}

static bool same_stats(const GateStats& x, const GateStats& y) {
  return x.ands == y.ands && x.xors == y.xors && x.nots == y.nots && x.consts == y.consts;
}

// ------------------------------------------------------------------------- 1
static void test_sigmoid_tables(bool quick) {
  const int qmax = quick ? 8 : 12;
  std::printf("[1] sigmoid threshold tables vs qsigmoid_ref (qbits 2..%d)\n", qmax);
  for (int q = 2; q <= qmax; ++q) {
    const int f = q - 1;
    const SigmoidTable S = make_sigmoid_table(f);
    CHECK((int64_t)S.t.size() == S.maxq + 1, "table size (qbits=%d)", q);
    for (int64_t v = 1; v <= S.maxq; ++v) {
      CHECK(qsigmoid_ref(S.t[(size_t)v], f) >= v, "t[%" PRId64 "] too small (qbits=%d)", v, q);
      CHECK(qsigmoid_ref(S.t[(size_t)v] - 1, f) < v, "t[%" PRId64 "] not minimal (qbits=%d)", v, q);
      if (v > 1) CHECK(S.t[(size_t)v] > S.t[(size_t)v - 1], "thresholds not strictly increasing (qbits=%d)", q);
    }
    // Every z of a window extending one transition width beyond both ends.
    const int64_t span = S.t[(size_t)S.maxq] - S.t[1];
    const int64_t zlo = S.t[1] - span - 64, zhi = S.t[(size_t)S.maxq] + span + 64;
    int64_t prev = qsigmoid_ref(zlo - 1, f);
    for (int64_t z = zlo; z <= zhi; ++z) {
      const int64_t r = qsigmoid_ref(z, f);
      CHECK(qsigmoid_by_table(S, z) == r, "table vs ref at z=%" PRId64 " (qbits=%d)", z, q);
      CHECK(r >= prev && r - prev <= 1, "qsigmoid_ref not monotone / skips a level at z=%" PRId64 " (qbits=%d)",
            z, q);
      CHECK(r >= 0 && r <= S.maxq, "qsigmoid_ref out of [0, MAXQ] at z=%" PRId64, z);
      if (z < S.t[1]) CHECK(r == 0, "not saturated at 0 below t[1]: z=%" PRId64 " (qbits=%d)", z, q);
      if (z >= S.t[(size_t)S.maxq]) CHECK(r == S.maxq, "not saturated at MAXQ: z=%" PRId64 " (qbits=%d)", z, q);
      prev = r;
    }
    CHECK(qsigmoid_ref(0, f) == (int64_t(1) << (f - 1)), "sigmoid(0) != 1/2 (qbits=%d)", q);
    // Far outside the window too (including the extremes of a 62-bit range).
    for (int64_t z : {-(int64_t(1) << 61), -(int64_t(1) << 40), -(int64_t(1) << 20), int64_t(1) << 20,
                      int64_t(1) << 40, int64_t(1) << 61}) {
      CHECK(qsigmoid_by_table(S, z) == qsigmoid_ref(z, f), "table vs ref at z=%" PRId64, z);
      CHECK(qsigmoid_ref(z, f) == (z < 0 ? 0 : S.maxq), "saturation at z=%" PRId64 " (qbits=%d)", z, q);
    }
  }
  // Independent known answers (not derived from qsigmoid_ref): floor(2^f sigmoid(z/2^f))
  // >= v  iff  z >= 2^f ln(v / (2^f - v)), so t[v] = ceil(2^f ln(v / (2^f - v))).
  for (int q = 2; q <= qmax; ++q) {
    const int f = q - 1;
    const SigmoidTable S = make_sigmoid_table(f);
    const long double s = std::ldexp(1.0L, f);
    for (int64_t v = 1; v <= S.maxq; ++v) {
      const int64_t t = (int64_t)std::ceil(s * std::log((long double)v / (s - (long double)v)));
      CHECK(S.t[(size_t)v] == t, "qbits=%d: t[%" PRId64 "] = %" PRId64 ", closed form ceil(2^f ln(v/(2^f-v))) = %"
            PRId64, q, v, S.t[(size_t)v], t);
    }
  }
  const SigmoidTable S6 = make_sigmoid_table(5);
  // Hand-checked: ceil(-32 ln 31) = ceil(-109.89) = -109, ceil(32 ln 31) = ceil(109.89) = 110.
  CHECK(S6.t[1] == -109 && S6.t[16] == 0 && S6.t[31] == 110, "qbits=6 thresholds t[1] / t[16] / t[31] = %" PRId64
        " / %" PRId64 " / %" PRId64 ", expected -109 / 0 / 110", S6.t[1], S6.t[16], S6.t[31]);
  std::printf("    qbits=6 thresholds t[1..31] (z at scale 2^5; yhat >= v iff z >= t[v]):\n     ");
  for (int64_t v = 1; v <= S6.maxq; ++v) std::printf(" %" PRId64, S6.t[(size_t)v]);
  std::printf("\n    => yhat = 0 for z < %" PRId64 " (%.3f), yhat = 31 for z >= %" PRId64 " (%.3f)\n", S6.t[1],
              S6.t[1] / 32.0, S6.t[31], S6.t[31] / 32.0);
}

// ------------------------------------------------------------------------- 2
static void test_arithmetic(bool quick) {
  std::printf("[2] Builder primitives vs int64 (exhaustive over small ranges)\n");
  struct R { int64_t lo, hi; };
  const std::vector<R> ranges = quick
      ? std::vector<R>{{-9, 7}, {0, 13}, {-12, -1}, {5, 5}, {-32, 31}}
      : std::vector<R>{{-9, 7}, {0, 13}, {-12, -1}, {5, 5}, {-32, 31}, {-1, 0}, {-33, 40}, {-64, 63}, {0, 1},
                       {-3, -3}, {-100, -60}};
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
            check_int(B.sum({A, Bv, A, Bv, Bv}), 2 * a + 3 * b, "sum5", a, b);
            CHECK(bit_value(B.lt(A, Bv)) == (a < b), "lt(%" PRId64 ", %" PRId64 ")", a, b);
            CHECK(bit_value(B.ge(A, Bv)) == (a >= b), "ge(%" PRId64 ", %" PRId64 ")", a, b);
            for (int sel = 0; sel <= 1; ++sel)
              check_int(B.mux(Builder<Clear>::wire((uint8_t)sel), A, Bv), sel ? a : b, "mux", a, b);
          }
      }
  // Shifts and constant multiplication. ashr must floor and tdiv_pow2 must round
  // toward zero; count the cases where the two differ to show both are exercised.
  uint64_t floor_ne_trunc = 0;
  for (const R& ra : ranges)
    for (int64_t a = ra.lo; a <= ra.hi; ++a) {
      const SInt<Clear> A = secret(B, a, ra.lo, ra.hi, width_for(ra.lo, ra.hi));
      for (int s = 0; s <= 7; ++s) {
        const GateStats before = B.stats;
        const SInt<Clear> L = B.shl(A, s), H = B.ashr(A, s);
        CHECK(same_stats(before, B.stats), "shl / ashr emitted gates (a=%" PRId64 ", s=%d)", a, s);
        check_int(L, a * (int64_t(1) << s), "shl", a, s);
        const int64_t fl = a >> s, tz = a / (int64_t(1) << s);
        check_int(H, fl, "ashr", a, s);
        check_int(B.tdiv_pow2(A, s), tz, "tdiv_pow2", a, s);
        if (fl != tz) ++floor_ne_trunc;
      }
      for (int64_t c = -70; c <= 70; ++c) check_int(B.mul_const(A, c), a * c, "mul_const", a, c);
    }
  CHECK(floor_ne_trunc > 0, "no case where floor and round-toward-zero differ");
  {  // the two roundings on a few negative values
    const SInt<Clear> A = secret(B, -1, -64, 63, 7), C = secret(B, -33, -64, 63, 7);
    check_int(B.ashr(A, 5), -1, "ashr", -1, 5);       // floor(-1/32)  = -1
    check_int(B.tdiv_pow2(A, 5), 0, "tdiv", -1, 5);   // trunc(-1/32)  =  0
    check_int(B.ashr(C, 5), -2, "ashr", -33, 5);      // floor(-33/32) = -2
    check_int(B.tdiv_pow2(C, 5), -1, "tdiv", -33, 5); // trunc(-33/32) = -1
  }
  // Public constants fold: no gates for constant-only arithmetic.
  {
    const GateStats before = B.stats;
    const SInt<Clear> c = B.add(B.mul(B.constant(-7), B.constant(9)), B.constant(5));
    check_int(c, -58, "const fold", -7, 9);
    CHECK(c.lo == c.hi && same_stats(before, B.stats), "constant arithmetic emitted gates");
  }
  // A public table lookup through one-hot minterms over 3 secret bits.
  const std::vector<int64_t> table{-5, 17, 3, 3, -40, 0, 63, 9};
  for (int idx = 0; idx < 8; ++idx) {
    std::vector<SBit<Clear>> mt{Builder<Clear>::cst(true)};
    for (int l = 2; l >= 0; --l) mt = B.extend_minterms(mt, Builder<Clear>::wire((uint8_t)((idx >> l) & 1)));
    check_int(B.select_public(mt, table), table[(size_t)idx], "select_public", idx, 0);
  }
  // Range overflow is an error, never a silent wraparound.
  {
    const SInt<Clear> big = secret(B, 3, -(int64_t(1) << 40), int64_t(1) << 40, 42);
    bool threw = false;
    try { (void)B.mul(big, big); } catch (const std::overflow_error&) { threw = true; }
    CHECK(threw, "mul of two 41-bit ranges did not throw");
    threw = false;
    try { (void)B.shl(big, 30); } catch (const std::overflow_error&) { threw = true; }
    CHECK(threw, "shl past 62 bits did not throw");
  }
  std::printf("    shifts: %" PRIu64 " cases where floor (ashr) != toward-zero (tdiv_pow2), all correct\n",
              floor_ne_trunc);
}

// ------------------------------------------------------------------------- 3
struct DotStats {
  uint64_t cases = 0, neg_floor_ne_trunc = 0, beyond_qbits = 0;
  int64_t zmin = 0, zmax = 0;
};

// One dot product + truncation through LRCircuit. a has declared range [alo, ahi];
// b holds quantized values, its last entry a public constant if b_last_public (the
// bias column). Returns the circuit's truncated value.
static int64_t dot_trunc_case(Builder<Clear>& B, LRCircuit<Clear>& lr, const Params& P,
                              const std::vector<int64_t>& a, int64_t alo, int64_t ahi,
                              const std::vector<int64_t>& b, bool b_last_public, DotStats& st) {
  std::vector<SInt<Clear>> A, Bv;
  for (size_t j = 0; j < a.size(); ++j) {
    A.push_back(secret(B, a[j], alo, ahi, width_for(alo, ahi)));
    if (b_last_public && j + 1 == b.size()) Bv.push_back(B.constant(b[j]));
    else Bv.push_back(secret(B, b[j], P.minq, P.maxq, P.qbits));
  }
  const int64_t full = dot_exact(a, b), expect = trunc_f(full, P.f);
  const int64_t n = (int64_t)a.size();
  const SInt<Clear> Dt = lr.dot(A, Bv);
  check_int(Dt, full, "dot (scale 2^2f)", n, full);
  const GateStats before = B.stats;
  const SInt<Clear> Z = lr.truncate(Dt);
  CHECK(same_stats(before, B.stats), "truncate emitted gates");
  check_int(Z, expect, "truncate(dot)", n, full);
  ++st.cases;
  if (full < 0 && full % (int64_t(1) << P.f) != 0) ++st.neg_floor_ne_trunc;
  if (expect < P.minq || expect > P.maxq) ++st.beyond_qbits;
  st.zmin = std::min(st.zmin, expect);
  st.zmax = std::max(st.zmax, expect);
  return clear_value(Z);
}

static void test_dot_truncation(bool quick) {
  std::printf("[3] truncation after dot products: truncate(dot(a, b)) == trunc_f(dot_exact(a, b))\n");
  Clear ctx;
  Builder<Clear> B(ctx);
  {  // worked examples at qbits = 6 (f = 5: scale 32 -> products at scale 1024)
    const Params P = make_params(6);
    const SigmoidTable S = make_sigmoid_table(P.f);
    LRCircuit<Clear> lr(B, P, S);
    struct Ex { std::vector<int64_t> a, b; };
    const std::vector<Ex> ex{
        {{-1}, {1}},
        {{-31, 1}, {1, 0}},
        {{5, -7, 3}, {-9, 4, -11}},
        {{31, -32, 17, 4}, {-3, 2, -1, 25}},
        {std::vector<int64_t>(8, 31), std::vector<int64_t>(8, 31)},
        {std::vector<int64_t>(8, -32), std::vector<int64_t>(8, -32)},
        {std::vector<int64_t>(8, -32), std::vector<int64_t>(8, 31)},
    };
    std::printf("    qbits=6 examples:  %3s %8s %12s %10s %12s %14s\n", "d", "dot", "dot/2^10", "circuit z",
                "z/2^5", "toward-zero z");
    DotStats st;
    for (const Ex& e : ex) {
      const int64_t full = dot_exact(e.a, e.b);
      const int64_t z = dot_trunc_case(B, lr, P, e.a, P.minq, P.maxq, e.b, false, st);
      std::printf("                       %3zu %8" PRId64 " %12.5f %10" PRId64 " %12.5f %14" PRId64 "%s\n",
                  e.a.size(), full, full / 1024.0, z, z / 32.0, full / 32,
                  (z < P.minq || z > P.maxq) ? "  (beyond 6 bits)" : "");
    }
    CHECK(st.neg_floor_ne_trunc >= 3 && st.beyond_qbits >= 3, "worked examples do not cover both effects");
  }
  const std::vector<int> qlist = quick ? std::vector<int>{6, 4, 8} : std::vector<int>{6, 2, 3, 4, 5, 7, 8, 10, 12};
  const std::vector<int> dlist{1, 2, 3, 4, 5, 8, 17, 64, 256};
  const int nrand = quick ? 200 : 2000;  // random vectors per (qbits, d)
  for (int q : qlist) {
    const Params P = make_params(q);
    const SigmoidTable S = make_sigmoid_table(P.f);
    LRCircuit<Clear> lr(B, P, S);
    SplitMix64 rng(0x5eed0000ULL + (uint64_t)q);
    DotStats st;
    // Value drawn from [lo, hi], biased toward the ends and toward zero.
    auto draw = [&](int64_t lo, int64_t hi) {
      switch (rng.uniform(0, 7)) {
        case 0: return lo;
        case 1: return hi;
        case 2: return std::clamp<int64_t>(rng.uniform(-1, 1), lo, hi);
        default: return rng.uniform(lo, hi);
      }
    };
    for (int d : dlist) {
      const size_t n = (size_t)d;
      std::vector<int64_t> a(n), b(n);
      // Edge vectors: all-extreme products (largest |dot|), zeros, alternating signs,
      // a lone -1 (dot = -1: floor gives -1, toward zero gives 0).
      const int64_t m = P.minq, M = P.maxq;
      const std::vector<std::pair<int64_t, int64_t>> fills{{m, m}, {m, M}, {M, m}, {M, M}, {0, 0}, {M, 0}, {1, -1}};
      for (auto [x, y] : fills) {
        a.assign(n, x);
        b.assign(n, y);
        dot_trunc_case(B, lr, P, a, P.minq, P.maxq, b, false, st);
      }
      for (size_t j = 0; j < n; ++j) { a[j] = (j % 2) ? M : m; b[j] = (j % 3) ? m : M; }
      dot_trunc_case(B, lr, P, a, P.minq, P.maxq, b, false, st);
      a.assign(n, 0);
      b.assign(n, 0);
      a[0] = -1;
      b[0] = 1;
      dot_trunc_case(B, lr, P, a, P.minq, P.maxq, b, false, st);
      // Random vectors: (i) forward pass  <w, x>  with quantized w (and the bias column
      // as a public MAXQ), (ii) weights grown by training (wider range), (iii) the
      // gradient  sum_i e_i x_ij  with e in [-MAXQ, MAXQ].
      const int64_t wlo = 10 * P.minq, whi = 8 * P.maxq;
      for (int r = 0; r < nrand; ++r) {
        const int kind = r % 3;
        const int64_t alo = kind == 0 ? P.minq : kind == 1 ? wlo : -P.maxq;
        const int64_t ahi = kind == 0 ? P.maxq : kind == 1 ? whi : P.maxq;
        const bool bias = kind != 2 && (r % 2 == 0);
        for (size_t j = 0; j < n; ++j) { a[j] = draw(alo, ahi); b[j] = draw(P.minq, P.maxq); }
        if (bias) b[n - 1] = P.maxq;
        dot_trunc_case(B, lr, P, a, alo, ahi, b, bias, st);
      }
    }
    CHECK(st.neg_floor_ne_trunc > 0, "qbits=%d: no negative dot product where floor != toward zero", q);
    CHECK(st.beyond_qbits > 0, "qbits=%d: no truncated dot product beyond qbits bits", q);
    std::printf("    qbits=%-2d: %6" PRIu64 " dot products (d in 1..256); %6" PRIu64 " negative with floor != "
                "toward-zero; %6" PRIu64 " truncated values outside [%" PRId64 ", %" PRId64 "] (kept, z in [%"
                PRId64 ", %" PRId64 "])\n",
                q, st.cases, st.neg_floor_ne_trunc, st.beyond_qbits, P.minq, P.maxq, st.zmin, st.zmax);
  }
}

// ------------------------------------------------------------------------- 4
static void test_sigmoid_circuit(bool quick) {
  std::printf("[4] circuit qsigmoid vs qsigmoid_ref (every z of each secret range)\n");
  Clear ctx;
  Builder<Clear> B(ctx);
  const std::vector<int> qlist = quick ? std::vector<int>{6, 4, 8} : std::vector<int>{6, 2, 3, 4, 5, 7, 8, 9, 10};
  for (int q : qlist) {
    const Params P = make_params(q);
    const SigmoidTable S = make_sigmoid_table(P.f);
    LRCircuit<Clear> lr(B, P, S);
    // Realistic ranges: z = truncate(<w, x>) as the circuit builds it for d = 5 and
    // d = 20 with quantized inputs, and for d = 5 with weights grown by training.
    auto z_range = [&](int d, int64_t wlo, int64_t whi) {
      std::vector<SInt<Clear>> w, x;
      for (int j = 0; j < d; ++j) {
        w.push_back(secret(B, 0, wlo, whi, width_for(wlo, whi)));
        x.push_back(j == d - 1 ? B.constant(P.maxq) : secret(B, 0, P.minq, P.maxq, P.qbits));
      }
      const SInt<Clear> z = lr.truncate(lr.dot(w, x));
      return std::pair<int64_t, int64_t>{z.lo, z.hi};
    };
    struct R { int64_t lo, hi; const char* what; };
    std::vector<R> ranges;
    auto add = [&](std::pair<int64_t, int64_t> r, const char* what) { ranges.push_back({r.first, r.second, what}); };
    add(z_range(5, P.minq, P.maxq), "z of <w,x>, d=5");
    add(z_range(20, P.minq, P.maxq), "z of <w,x>, d=20");
    add(z_range(5, 10 * P.minq, 8 * P.maxq), "z of <w,x>, d=5, grown w");
    const int64_t span = S.t[(size_t)S.maxq] - S.t[1];
    ranges.push_back({S.t[1] - span / 3, S.t[(size_t)S.maxq] + span / 3, "around the transitions"});
    ranges.push_back({S.t[1] + span / 3, S.t[1] + span / 3 + 50, "narrow, interior"});
    ranges.push_back({-3, 4, "tiny"});
    if (!quick) ranges.push_back({S.t[1] - 3 * span, S.t[(size_t)S.maxq] + 3 * span, "wide"});
    // At least one realistic z range must reach both saturation regions (with larger
    // qbits a d = 5 dot product of values in [-1, 1) cannot: |z| <= 5 < ln(MAXQ)).
    bool realistic_covers = false;
    for (const R& r : ranges) {
      CHECK(r.lo <= r.hi, "empty range");
      const bool covers = r.lo < S.t[1] && r.hi >= S.t[(size_t)S.maxq];
      if (r.what[0] == 'z') realistic_covers = realistic_covers || covers;
      const int w = width_for(r.lo, r.hi);
      uint64_t ands = 0;
      for (int64_t z = r.lo; z <= r.hi; ++z) {
        const SInt<Clear> Z = secret(B, z, r.lo, r.hi, w);
        const uint64_t a0 = B.stats.ands;
        const SInt<Clear> Y = lr.qsigmoid(Z);
        ands = B.stats.ands - a0;
        CHECK(clear_value(Y) == qsigmoid_ref(z, P.f), "qsigmoid(z=%" PRId64 ", qbits=%d): got %" PRId64
              " expected %" PRId64, z, q, clear_value(Y), qsigmoid_ref(z, P.f));
        CHECK(Y.lo == 0 && Y.hi == S.maxq && Y.width() <= P.qbits, "qsigmoid output not a qbits-bit value");
      }
      if (q == 6 || !quick)
        std::printf("    qbits=%-2d %-26s z in [%6" PRId64 ", %6" PRId64 "]%s: %6" PRId64 " values ok, %3" PRIu64
                    " ANDs per sigmoid\n",
                    q, r.what, r.lo, r.hi, covers ? " (both saturations)" : "                   ", r.hi - r.lo + 1,
                    ands);
    }
    CHECK(realistic_covers, "qbits=%d: no realistic z range covers both saturation regions", q);
  }
}

// ------------------------------------------------------------------------- 5
static void test_training(bool quick) {
  std::printf("[5] full training circuit (train_offline over ClearCtx) vs reference_train (bit-exact)\n");
  struct Cfg { int d, iters, batch, qbits; double lr; bool bias, zero; };
  std::vector<Cfg> cfgs{
      {5, 20, 4, 6, 0.5, true, false},  {5, 20, 4, 6, 0.5, true, true},   {1, 5, 4, 6, 0.5, true, false},
      {2, 20, 4, 6, 0.5, true, false},  {3, 7, 4, 6, 0.5, false, false},  {17, 20, 4, 6, 0.5, true, false},
      {5, 20, 4, 6, 1.0, true, false},  {5, 20, 4, 6, 4.0, true, false},  {5, 20, 4, 6, 0.125, true, false},
      {6, 10, 1, 6, 0.25, true, false}, {6, 10, 2, 6, 0.5, true, false},  {4, 10, 8, 6, 0.5, true, false},
      {5, 20, 4, 4, 0.5, true, false},  {5, 20, 4, 8, 0.5, true, false},  {5, 12, 4, 2, 0.5, true, false},
      {5, 6, 4, 10, 0.5, true, false},  {40, 20, 4, 6, 0.5, true, false}, {8, 20, 4, 6, 0.5, false, true},
  };
  if (!quick) {
    cfgs.push_back({100, 20, 4, 6, 0.5, true, false});
    cfgs.push_back({33, 20, 4, 7, 2.0, true, false});
    cfgs.push_back({10, 50, 4, 6, 0.5, true, false});
    cfgs.push_back({12, 20, 4, 12, 0.5, true, false});
  }
  const int seeds = quick ? 2 : 6;
  int runs = 0;
  int64_t zmin = 0, zmax = 0, wmin = 0, wmax = 0;  // over the qbits = 6 runs
  for (const Cfg& c : cfgs)
    for (int s = 1; s <= seeds; ++s) {
      Params P;
      P.d = c.d; P.iters = c.iters; P.batch = c.batch; P.qbits = c.qbits; P.lr = c.lr;
      P.bias = c.bias; P.zero_init = c.zero; P.seed = (uint64_t)(1000 * s + c.d);
      P.finalize();
      const SigmoidTable S = make_sigmoid_table(P.f);
      const Dataset D = generate_client_data(P);
      const std::vector<int64_t> w0 = generate_initial_weights(P);
      RefTrace tr;
      const std::vector<int64_t> ref = reference_train(P, D, w0, &tr);
      Clear ctx;
      Builder<Clear> B(ctx);
      const auto w = train_offline(B, P, S, D, w0);
      CHECK((int)w.size() == P.d, "train: %zu outputs, expected %d", w.size(), P.d);
      for (int j = 0; j < P.d && j < (int)w.size(); ++j) {
        CHECK(clear_value(w[(size_t)j]) == ref[(size_t)j], "train %s: w[%d] got %" PRId64 " expected %" PRId64,
              params_string(P).c_str(), j, clear_value(w[(size_t)j]), ref[(size_t)j]);
        CHECK(w[(size_t)j].lo <= ref[(size_t)j] && ref[(size_t)j] <= w[(size_t)j].hi, "train: range violated");
        if (P.qbits == 6) { wmin = std::min(wmin, ref[(size_t)j]); wmax = std::max(wmax, ref[(size_t)j]); }
      }
      if (P.qbits == 6) { zmin = std::min(zmin, tr.z_min); zmax = std::max(zmax, tr.z_max); }
      ++runs;
    }
  std::printf("    %d runs (%zu configs x %d seeds): circuit == reference_train for every weight\n", runs, cfgs.size(),
              seeds);
  std::printf("    qbits=6 runs: truncated z in [%" PRId64 ", %" PRId64 "], final w in [%" PRId64 ", %" PRId64
              "] (6-bit inputs are [-32, 31])\n", zmin, zmax, wmin, wmax);
}

// ------------------------------------------------------------------------- 6
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

static uint64_t count_ands(Params P) {
  P.finalize();
  const SigmoidTable S = make_sigmoid_table(P.f);
  emp::CountCtx cc;
  Builder<emp::CountCtx> B(cc);
  (void)train_offline(B, P, S, generate_client_data(P), generate_initial_weights(P));
  return cc.ands;
}

static void print_counts(bool quick) {
  std::printf("[6] gate counts (iters=20, batch=4, qbits=6, lr=0.5, bias column; ANDs are what GC pays for)\n");
  std::vector<int> dlist{2, 5, 10, 20, 50, 100};
  if (!quick) dlist.insert(dlist.end(), {200, 500, 1000});
  std::printf("    %6s %14s %14s %14s %12s %14s\n", "d", "AND (total)", "AND/iter", "XOR+NOT", "inputs(bits)",
              "live AND");
  for (int d : dlist) {
    Params P;
    P.d = d;
    P.finalize();
    const SigmoidTable S = make_sigmoid_table(P.f);
    const Dataset D = generate_client_data(P);
    const std::vector<int64_t> w0 = generate_initial_weights(P);
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
      const uint64_t la = live_ands(T, out);
      CHECK(la == cc.ands, "d=%d: %" PRIu64 " of %" PRIu64 " emitted ANDs are live (dead gates)", d, la, cc.ands);
      std::snprintf(live, sizeof live, "%" PRIu64, la);
    }
    std::printf("    %6d %14" PRIu64 " %14" PRIu64 " %14" PRIu64 " %12" PRIu64 " %14s\n", d, cc.ands,
                cc.ands / (uint64_t)P.iters, cc.xors + cc.nots, in_bits, live);
  }
  // Every emitted AND is live in other configurations too, so the CountCtx count that
  // lr_gc_ag2pc reports as "predicted" is exactly what the protocol garbles after its
  // dead-code elimination. Known exception, batch = 1: g_j = e_0 * x_0j is a single
  // product that is truncated right away, so its bit 0 (one AND) feeds nothing; that
  // is d_data dead ANDs per iteration, which emp-ag2pc removes.
  struct Cfg { int d, iters, batch, qbits; double lr; bool bias; };
  const std::vector<Cfg> live_cfgs{
      {3, 4, 4, 2, 0.5, true},  {3, 2, 1, 6, 0.25, true}, {6, 10, 1, 6, 0.25, true}, {5, 6, 4, 4, 0.5, true},
      {5, 6, 4, 8, 0.5, true},  {5, 6, 4, 12, 0.5, true}, {6, 5, 2, 6, 0.5, true},   {4, 5, 8, 6, 0.5, true},
      {5, 6, 4, 6, 4.0, true},  {5, 6, 4, 6, 1.0, true},  {5, 6, 4, 6, 0.125, true}, {3, 6, 4, 6, 0.5, false},
      {1, 3, 4, 6, 0.5, true}};
  for (const Cfg& c : live_cfgs) {
    Params P;
    P.d = c.d; P.iters = c.iters; P.batch = c.batch; P.qbits = c.qbits; P.lr = c.lr; P.bias = c.bias;
    P.finalize();
    const SigmoidTable S = make_sigmoid_table(P.f);
    const Dataset D = generate_client_data(P);
    const std::vector<int64_t> w0 = generate_initial_weights(P);
    TapeCtx T;
    Builder<TapeCtx> BT(T);
    const auto out = train_offline(BT, P, S, D, w0);
    const uint64_t la = live_ands(T, out);
    const uint64_t dead = P.batch == 1 ? (uint64_t)P.d_data * P.iters : 0;
    CHECK(la + dead == BT.stats.ands, "%s: %" PRIu64 " of %" PRIu64 " emitted ANDs are live (expected %" PRIu64
          " dead)", params_string(P).c_str(), la, BT.stats.ands, dead);
  }
  std::printf("    every emitted AND is live in the table above and in %zu other configs (qbits 2..12, batch 2..8, "
              "lr 0.125..4, with / without bias); batch=1: exactly d_data*iters dead ANDs\n", live_cfgs.size());
  // The circuit's shape depends only on public parameters, never on the client's data
  // or initial weights.
  for (int d : {2, 5, 10}) {
    Params P;
    P.d = d;
    const uint64_t base = count_ands(P);
    P.seed = 77;
    const uint64_t other_seed = count_ands(P);
    P.zero_init = true;
    const uint64_t zero = count_ands(P);
    CHECK(base == other_seed && base == zero, "d=%d: AND count depends on the data (%" PRIu64 " / %" PRIu64
          " / %" PRIu64 ")", d, base, other_seed, zero);
  }
}

int main(int argc, char** argv) {
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--quick")) quick = true;
    else {
      std::printf("usage: %s [--quick]\n", argv[0]);
      return 2;
    }
  }
  using clk = std::chrono::steady_clock;
  const auto t0 = clk::now();
  auto lap = [t = t0]() mutable {
    const auto now = clk::now();
    std::printf("    (%.2f s)\n", std::chrono::duration<double>(now - t).count());
    t = now;
  };
  try {
    test_sigmoid_tables(quick); lap();
    test_arithmetic(quick); lap();
    test_dot_truncation(quick); lap();
    test_sigmoid_circuit(quick); lap();
    test_training(quick); lap();
    print_counts(quick); lap();
  } catch (const std::exception& e) {
    std::printf("EXCEPTION: %s\n", e.what());
    return 1;
  }
  const double s = std::chrono::duration<double>(clk::now() - t0).count();
  std::printf("%s: %" PRIu64 " checks, %d failures (%.1f s)\n", g_fail ? "FAILED" : "ALL TESTS PASSED", g_checks,
              g_fail, s);
  return g_fail ? 1 : 0;
}
