// lr_gc_dot_sigmoid -- small test circuit: ONE vector multiplication (dot product),
// its truncation, and ONE sigmoid, built from the same LRCircuit::dot / truncate /
// qsigmoid code the training circuit uses, so their output can be checked in isolation.
//
// Inputs are qbits-bit quantized values (default qbits = 6: integers in [-32, 31]
// standing for v / 2^f, f = qbits - 1 = 5), fed as qbits secret input wires each,
// exactly like the training program:
//
//   zfull = sum_j w_j x_j          (lr.dot: exact products + adder tree, scale 2^(2f))
//   z     = zfull >> f             (lr.truncate: floor, back to scale 2^f; integer bits
//                                   are kept -- |z| may exceed the qbits range)
//   yhat  = qsigmoid(z)            (lr.qsigmoid: min(floor(2^f sigmoid(z / 2^f)), MAXQ),
//                                   by binary search over public thresholds)
//   p_j   = w_j * x_j              (extra mul gates, revealed only for the report)
//
// The client (garbler, ALICE) owns w and x; p, zfull, z and yhat are revealed to BOTH
// parties (test only). Each party checks the result against the plaintext spec of
// lr_reference.h (dot_exact, trunc_f, qsigmoid_ref, qsigmoid_by_table), shows that the
// truncation is floor and not round-toward-zero, and compares with the true sigmoid.
//
// usage:
//   lr_gc_dot_sigmoid local  [--d N] [--qbits N] [--seed N | --w a,b,.. --x a,b,..] [--trials N]
//   lr_gc_dot_sigmoid server [same input flags] [--port N] [--threads N] [--ssp N]
//   lr_gc_dot_sigmoid client [same input flags] [--host IP] [--port N] [--threads N] [--ssp N]
// --w / --x are INTEGERS in [MINQ, MAXQ] (value * 2^f). local runs the circuit over
// emp::ClearCtx (no protocol), then exhaustive / edge / --trials random vectors.
// The server only uses the input flags to check the revealed result.

#include "ag2pc_metrics.h"  // must precede every emp-ag2pc header (phase hooks)

#include "emp-ag2pc/emp-ag2pc.h"
#ifndef GCAG2PC_INPUT_PRIVACY_FIX
#error "built against an unpatched emp-ag2pc: its input phase sends private inputs in the clear; use third_party/emp-ag2pc"
#endif

#include "lr_circuit.h"
#include "lr_params.h"
#include "lr_reference.h"
#include "sbit.h"

#include "emp-tool/ir/context/clear.h"
#include "emp-tool/ir/context/count.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using lrgc::Builder;
using lrgc::LRCircuit;
using lrgc::Params;
using lrgc::SigmoidTable;
using lrgc::SInt;

namespace {

struct Options {
  std::string role;
  int d = 8;
  int qbits = 6;
  uint64_t seed = 1;
  std::vector<int64_t> w, x;  // explicit inputs (else generated from --seed)
  bool has_w = false, has_x = false;
  int trials = 2000;
  std::string host = "127.0.0.1";
  int port = 12346;
  int threads = 4;
  int ssp = 40;
};

std::vector<int64_t> parse_list(const std::string& s, const std::string& flag) {
  std::vector<int64_t> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) v.push_back((int64_t)lrgc::parse_int(tok, flag));
  return v;
}

Options parse_options(int argc, char** argv, Params& P) {
  Options O;
  if (argc < 2) throw std::invalid_argument("missing role (local, client or server)");
  O.role = argv[1];
  if (O.role != "local" && O.role != "client" && O.role != "server")
    throw std::invalid_argument("unknown role '" + O.role + "'");
  lrgc::ArgReader r{argc, argv, 2};
  for (; r.has(); ++r.i) {
    const std::string a = argv[r.i];
    if (a == "--d") O.d = (int)lrgc::parse_int(r.next_value(a), a);
    else if (a == "--qbits") O.qbits = (int)lrgc::parse_int(r.next_value(a), a);
    else if (a == "--seed") O.seed = (uint64_t)lrgc::parse_int(r.next_value(a), a);
    else if (a == "--w") { O.w = parse_list(r.next_value(a), a); O.has_w = true; }
    else if (a == "--x") { O.x = parse_list(r.next_value(a), a); O.has_x = true; }
    else if (a == "--trials") O.trials = (int)lrgc::parse_int(r.next_value(a), a);
    else if (a == "--host") O.host = r.next_value(a);
    else if (a == "--port") O.port = (int)lrgc::parse_int(r.next_value(a), a);
    else if (a == "--threads") O.threads = (int)lrgc::parse_int(r.next_value(a), a);
    else if (a == "--ssp") O.ssp = (int)lrgc::parse_int(r.next_value(a), a);
    else throw std::invalid_argument("unknown option " + a);
  }
  if (O.has_w != O.has_x) throw std::invalid_argument("pass both --w and --x, or neither");
  if ((O.has_w && O.w.empty()) || (O.has_x && O.x.empty())) throw std::invalid_argument("--w / --x list is empty");
  if (!O.w.empty()) {
    if (O.w.size() != O.x.size()) throw std::invalid_argument("--w and --x must have the same length");
    O.d = (int)O.w.size();
  }
  if (O.threads < 2) throw std::invalid_argument("--threads must be >= 2 (emp-ag2pc deadlocks with 1)");
  if (O.ssp < 40) throw std::invalid_argument("--ssp must be >= 40");
  // The circuit only uses d, qbits and the derived f / MINQ / MAXQ; the training
  // parameters keep their (valid) defaults.
  P.d = O.d;
  P.qbits = O.qbits;
  P.bias = false;
  P.finalize();
  for (const auto* v : {&O.w, &O.x})
    for (int64_t a : *v)
      if (a < P.minq || a > P.maxq)
        throw std::invalid_argument("input " + std::to_string(a) + " outside the " + std::to_string(P.qbits) +
                                    "-bit range [" + std::to_string(P.minq) + ", " + std::to_string(P.maxq) + "]");
  return O;
}

// Random quantized w, x, uniform in [MINQ, MAXQ].
void random_inputs(const Params& P, uint64_t seed, int d, std::vector<int64_t>& w, std::vector<int64_t>& x) {
  lrgc::SplitMix64 rng(seed * 0xD1B54A32D192ED03ULL + 3);
  w.clear();
  x.clear();
  for (int j = 0; j < d; ++j) w.push_back(rng.uniform(P.minq, P.maxq));
  for (int j = 0; j < d; ++j) x.push_back(rng.uniform(P.minq, P.maxq));
}

// ------------------------------------------------------------------- circuit
template <class Ctx>
struct DotSigmoid {
  using Int = SInt<Ctx>;
  std::vector<Int> prod;  // p_j = w_j * x_j (scale 2^(2f))
  Int zfull, z, yhat;

  DotSigmoid(Builder<Ctx>& B, LRCircuit<Ctx>& lr, const std::vector<Int>& W, const std::vector<Int>& X,
             bool with_products = true) {
    if (with_products)
      for (size_t j = 0; j < W.size(); ++j) prod.push_back(B.mul(W[j], X[j]));
    zfull = lr.dot(W, X);
    z = lr.truncate(zfull);
    yhat = lr.qsigmoid(z);
  }
  std::vector<const Int*> outputs() const {
    std::vector<const Int*> o;
    for (const Int& p : prod) o.push_back(&p);
    o.push_back(&zfull);
    o.push_back(&z);
    o.push_back(&yhat);
    return o;
  }
};

struct Result {
  std::vector<int64_t> prod;
  int64_t zfull = 0, z = 0, yhat = 0;
  // data-independent shape of the circuit's values (width in bits, tracked range)
  int zfull_w = 0, z_w = 0;
  int64_t zfull_lo = 0, zfull_hi = 0, z_lo = 0, z_hi = 0;
};

template <class Ctx, class Decode>
Result collect(const DotSigmoid<Ctx>& c, Decode&& value) {
  Result r;
  for (const auto& p : c.prod) r.prod.push_back(value(p));
  r.zfull = value(c.zfull);
  r.z = value(c.z);
  r.yhat = value(c.yhat);
  r.zfull_w = c.zfull.width(); r.zfull_lo = c.zfull.lo; r.zfull_hi = c.zfull.hi;
  r.z_w = c.z.width(); r.z_lo = c.z.lo; r.z_hi = c.z.hi;
  return r;
}

// Inputs for a non-protocol context: every bit is a secret wire (never folded), so the
// circuit is the same as in the protocol run.
template <class Ctx>
std::vector<SInt<Ctx>> offline_inputs(Builder<Ctx>& B, LRCircuit<Ctx>& lr, const Params& P,
                                      const std::vector<int64_t>& v) {
  std::vector<SInt<Ctx>> r;
  for (int64_t a : v) {
    std::vector<typename Ctx::Wire> ws;
    for (bool bit : lrgc::to_bits(a, P.qbits)) ws.push_back(B.ctx().public_bit(bit));
    r.push_back(lr.quantized(ws));
  }
  return r;
}

Result clear_run(const Params& P, const SigmoidTable& S, const std::vector<int64_t>& w,
                 const std::vector<int64_t>& x) {
  emp::ClearCtx cx;
  Builder<emp::ClearCtx> B(cx);
  LRCircuit<emp::ClearCtx> lr(B, P, S);
  const auto W = offline_inputs(B, lr, P, w), X = offline_inputs(B, lr, P, x);
  const DotSigmoid<emp::ClearCtx> c(B, lr, W, X);
  return collect(c, [](const SInt<emp::ClearCtx>& v) { return lrgc::clear_value(v); });
}

uint64_t and_count(const Params& P, const SigmoidTable& S, bool with_products) {
  emp::CountCtx cc;
  Builder<emp::CountCtx> B(cc);
  LRCircuit<emp::CountCtx> lr(B, P, S);
  const std::vector<int64_t> zeros((size_t)P.d, 0);
  const auto W = offline_inputs(B, lr, P, zeros), X = offline_inputs(B, lr, P, zeros);
  const DotSigmoid<emp::CountCtx> c(B, lr, W, X, with_products);
  return cc.ands;
}

// ------------------------------------------------------- plaintext checks
// yhat / 2^f is the floor quantization of the true sigmoid(z / 2^f):
// 0 <= sigmoid - yhat / 2^f < 2^-f. The one exception is the spec's clamp: where
// sigmoid rounds to exactly 1 in long double (z / 2^f > ~44), yhat = MAXQ and the
// difference is exactly 2^-f.
bool floor_quantized(const Params& P, int64_t z, int64_t yhat, bool* clamped = nullptr) {
  const long double s1 = std::ldexp(1.0L, P.f);
  const long double sig = lrgc::sigmoid_ld((long double)z / s1), y = (long double)yhat / s1;
  const bool clamp = yhat == P.maxq && sig - y == 1.0L / s1;
  if (clamped) *clamped = clamp;
  return y <= sig && (sig - y < 1.0L / s1 || clamp);
}

int failures = 0;
void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what.c_str());
  if (!ok) ++failures;
}

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, f);
  std::vsnprintf(buf, sizeof buf, f, ap);
  va_end(ap);
  return buf;
}

void report(const char* who, const Params& P, const SigmoidTable& S, const std::vector<int64_t>& w,
            const std::vector<int64_t>& x, const Result& gc) {
  const size_t d = w.size();
  const int f = P.f;
  const double s1 = std::ldexp(1.0, f), s2 = std::ldexp(1.0, 2 * f);  // 2^f, 2^(2f)

  std::printf("\n[%s] products (inputs: %d-bit integers in [%lld, %lld] = value * %g; products at scale 2^%d)\n",
              who, P.qbits, (long long)P.minq, (long long)P.maxq, s1, 2 * f);
  std::printf("  %3s | %4s %10s | %4s %10s | %6s %14s | %6s\n", "j", "w_j", "(real)", "x_j", "(real)", "exact",
              "(real)", "circuit p_j");
  bool prod_ok = gc.prod.size() == d;
  for (size_t j = 0; j < d; ++j) {
    const int64_t e = w[j] * x[j];
    const bool ok = j < gc.prod.size() && gc.prod[j] == e;
    prod_ok = prod_ok && ok;
    std::printf("  %3zu | %4lld %10.6g | %4lld %10.6g | %6lld %14.10g | %6lld %s\n", j, (long long)w[j], w[j] / s1,
                (long long)x[j], x[j] / s1, (long long)e, e / s2, (long long)(j < gc.prod.size() ? gc.prod[j] : 0),
                ok ? "= exact" : "!= exact");
  }

  const int64_t zref = lrgc::dot_exact(w, x);
  const int64_t floor_z = lrgc::trunc_f(zref, f);
  const int64_t tz_z = zref / (int64_t(1) << f);  // C++ '/' rounds toward zero
  const int64_t rem = zref - floor_z * (int64_t(1) << f);  // dropped low bits, in [0, 2^f)
  long double real_dot = 0;
  for (size_t j = 0; j < d; ++j) real_dot += ((long double)w[j] / s1) * ((long double)x[j] / s1);
  const bool out_of_range = gc.z < P.minq || gc.z > P.maxq;

  std::printf("\n[%s] dot product and truncation (f = %d)\n", who, f);
  std::printf("  zfull circuit  = %lld  -> %lld / 2^%d = %.15g   (%d-bit wire, range [%lld, %lld])\n",
              (long long)gc.zfull, (long long)gc.zfull, 2 * f, gc.zfull / s2, gc.zfull_w, (long long)gc.zfull_lo,
              (long long)gc.zfull_hi);
  std::printf("  zfull dot_exact= %lld   (real dot of the real values: %.15Lg)\n", (long long)zref, real_dot);
  std::printf("  z circuit      = zfull >> %d = %lld  -> %lld / 2^%d = %.15g   (%d-bit wire, range [%lld, %lld])\n", f,
              (long long)gc.z, (long long)gc.z, f, gc.z / s1, gc.z_w, (long long)gc.z_lo, (long long)gc.z_hi);
  std::printf("  dropped low bits: %lld / 2^%d = %.15g  (truncation error zfull/2^%d - z/2^%d, in [0, 2^-%d))\n",
              (long long)rem, 2 * f, rem / s2, 2 * f, f, f);
  std::printf("  floor(zfull / 2^%d) = %lld, round-toward-zero(zfull / 2^%d) = %lld  -> circuit z %s\n", f,
              (long long)floor_z, f, (long long)tz_z,
              floor_z == tz_z ? "(both agree here: zfull >= 0 or divisible)"
                              : (gc.z == floor_z ? "= FLOOR (not toward zero)" : "= toward zero"));
  std::printf("  z = %lld %s the %d-bit input range [%lld, %lld]%s\n", (long long)gc.z,
              out_of_range ? "EXCEEDS" : "is within", P.qbits, (long long)P.minq, (long long)P.maxq,
              out_of_range ? ": kept as is (integer bits kept, no clamping, no wraparound)" : "");

  const int64_t q_ref = lrgc::qsigmoid_ref(gc.z, f), q_tab = lrgc::qsigmoid_by_table(S, gc.z);
  const long double sig_z = lrgc::sigmoid_ld((long double)gc.z / s1);
  const long double sig_exact = lrgc::sigmoid_ld(real_dot);
  const long double yreal = (long double)gc.yhat / s1;
  std::printf("\n[%s] sigmoid (output: %d-bit, yhat / 2^%d)\n", who, P.qbits, f);
  std::printf("  yhat circuit          = %lld  -> %lld / 2^%d = %.15Lg\n", (long long)gc.yhat, (long long)gc.yhat, f,
              yreal);
  std::printf("  qsigmoid_ref(z)       = %lld   (min(floor(2^%d * sigmoid(z / 2^%d)), %lld) in long double)\n",
              (long long)q_ref, f, f, (long long)P.maxq);
  std::printf("  qsigmoid_by_table(z)  = %lld   (thresholds: t[%lld] = %s <= z < t[%lld] = %s)\n", (long long)q_tab,
              (long long)q_tab, q_tab >= 1 ? std::to_string(S.t[(size_t)q_tab]).c_str() : "-inf",
              (long long)(q_tab + 1), q_tab < S.maxq ? std::to_string(S.t[(size_t)q_tab + 1]).c_str() : "+inf");
  std::printf("  true sigmoid(z / 2^%d)        = %.17Lg   (quantization error yhat/2^%d - it = %+.6Lg)\n", f, sig_z,
              f, yreal - sig_z);
  std::printf("  true sigmoid(exact real dot)  = %.17Lg   (total error incl. truncation = %+.6Lg)\n", sig_exact,
              yreal - sig_exact);

  std::printf("\n[%s] checks\n", who);
  check(prod_ok, "every product p_j == w_j * x_j (exact, scale 2^" + std::to_string(2 * f) + ")");
  int64_t psum = 0;
  for (int64_t p : gc.prod) psum += p;
  check(gc.zfull == zref && gc.zfull == psum, "zfull == dot_exact(w, x) == sum of the revealed products");
  if (floor_z != tz_z)
    check(gc.z == floor_z && gc.z != tz_z,
          fmt("z == trunc_f(zfull, %d) = floor = %lld, NOT round-toward-zero = %lld (zfull < 0, not divisible)", f,
              (long long)floor_z, (long long)tz_z));
  else
    check(gc.z == floor_z, fmt("z == trunc_f(zfull, %d) = zfull >> %d = %lld", f, f, (long long)floor_z));
  check(gc.yhat == q_ref && gc.yhat == q_tab, "yhat == qsigmoid_ref(z) == qsigmoid_by_table(z)");
  bool clamped = false;
  const bool fq = floor_quantized(P, gc.z, gc.yhat, &clamped);
  check(fq, clamped ? fmt("yhat = MAXQ = %lld: sigmoid(z/2^%d) rounds to 1 in long double, so the spec's clamp "
                          "min(floor(2^%d * 1), MAXQ) applies (difference exactly 1/2^%d)",
                          (long long)P.maxq, f, f, f)
                    : fmt("0 <= sigmoid(z/2^%d) - yhat/2^%d < 1/2^%d (floor quantization of the true sigmoid)", f,
                          f, f));
}

// Exhaustive / edge / random vectors over ClearCtx: bit-exactness, sigmoid floor
// quantization (incl. the MAXQ clamp) and truncation statistics.
void trials(const Params& P0, const SigmoidTable& S, int n_random, int d0) {
  struct Stats {
    long n = 0, bad = 0, bad_sig = 0, clamped = 0, neg = 0, nonzero_rem = 0, floor_ne_tz = 0, out_of_range = 0, y0 = 0,
         ymax = 0;
  } st;
  int64_t z_min = INT64_MAX, z_max = INT64_MIN;
  auto run = [&](const std::vector<int64_t>& w, const std::vector<int64_t>& x) {
    Params P = P0;
    P.d = (int)w.size();
    P.finalize();
    const Result gc = clear_run(P, S, w, x);
    const int64_t zf = lrgc::dot_exact(w, x), z = lrgc::trunc_f(zf, P.f), y = lrgc::qsigmoid_ref(z, P.f);
    bool same = gc.zfull == zf && gc.z == z && gc.yhat == y && y == lrgc::qsigmoid_by_table(S, z);
    for (size_t j = 0; j < w.size(); ++j) same = same && gc.prod[j] == w[j] * x[j];
    bool cl = false;
    ++st.n;
    st.bad += !same;
    st.bad_sig += !floor_quantized(P, gc.z, gc.yhat, &cl);
    st.clamped += cl;
    st.neg += zf < 0;
    st.nonzero_rem += (zf & ((int64_t(1) << P.f) - 1)) != 0;
    st.floor_ne_tz += z != zf / (int64_t(1) << P.f);
    st.out_of_range += z < P.minq || z > P.maxq;
    st.y0 += gc.yhat == 0;
    st.ymax += gc.yhat == P.maxq;
    z_min = std::min(z_min, gc.z);
    z_max = std::max(z_max, gc.z);
  };

  // 1. exhaustive d = 1: every (w, x) pair (only for small qbits)
  long n_exh = 0;
  const int64_t nq = P0.maxq - P0.minq + 1;
  if (nq * nq <= 65536)
    for (int64_t a = P0.minq; a <= P0.maxq; ++a)
      for (int64_t b = P0.minq; b <= P0.maxq; ++b, ++n_exh) run({a}, {b});
  // 2. edge vectors: extreme products everywhere (largest |zfull|), and zeros
  long n_edge = 0;
  const std::pair<int64_t, int64_t> edges[] = {{P0.minq, P0.minq}, {P0.minq, P0.maxq}, {P0.maxq, P0.maxq}, {0, 0}};
  for (int d : {d0, 16, 64})
    for (const auto& [a, b] : edges) {
      run(std::vector<int64_t>((size_t)d, a), std::vector<int64_t>((size_t)d, b));
      ++n_edge;
    }
  // 3. random vectors of several lengths; every 3rd one has small x (z near the
  //    sigmoid's transition), every 3rd+1 one is forced negative (w <= 0 <= x).
  const int lens[] = {d0, 1, 2, 3, 5, 8, 16, 32, 64};
  std::vector<int64_t> w, x;
  for (int t = 0; t < n_random; ++t) {
    const int d = lens[t % (sizeof lens / sizeof lens[0])];
    random_inputs(P0, 1000 + (uint64_t)t, d, w, x);
    if (t % 3 == 1)
      for (int64_t& v : x) v = v % 4;
    if (t % 3 == 2)
      for (int j = 0; j < d; ++j) {
        w[(size_t)j] = -std::llabs(w[(size_t)j]);
        x[(size_t)j] = std::min<int64_t>(std::llabs(x[(size_t)j]), P0.maxq);
      }
    run(w, x);
  }

  std::printf("\n[local] %ld vectors over emp::ClearCtx: exhaustive d=1 (%ld pairs), %ld edge vectors, %d random "
              "(d cycling over %d,1,2,3,5,8,16,32,64)\n", st.n, n_exh, n_edge, n_random, d0);
  check(st.bad == 0, fmt("products, zfull, z = trunc_f(zfull) and yhat == reference in all %ld vectors (%ld mismatches)",
                         st.n, st.bad));
  std::printf("  zfull < 0 in %ld; truncation dropped a nonzero remainder in %ld; floor != toward-zero in %ld; "
              "|z| beyond [%lld, %lld] (kept, not clamped) in %ld (z seen in [%lld, %lld]); yhat = 0 in %ld, "
              "yhat = %lld in %ld\n", st.neg, st.nonzero_rem, st.floor_ne_tz, (long long)P0.minq, (long long)P0.maxq,
              st.out_of_range, (long long)z_min, (long long)z_max, st.y0, (long long)P0.maxq, st.ymax);
  check(st.bad_sig == 0, fmt("yhat is the floor quantization of the true sigmoid(z/2^%d) in all %ld vectors (%ld "
                             "violations; %ld use the clamp to MAXQ where sigmoid rounds to 1)", P0.f, st.n,
                             st.bad_sig, st.clamped));
  check(st.floor_ne_tz > 0 && st.out_of_range > 0 && st.y0 > 0 && st.ymax > 0 && st.clamped > 0,
        "coverage: floor != toward-zero, |z| out of the input range, both saturated outputs and the MAXQ clamp were "
        "all exercised");
}

// ------------------------------------------------------------- AG2PC run
using Ctx = emp::AG2PCSession::ctx_t;
using Wire = Ctx::Wire;

template <class F>
void with_capacity(size_t n, F&& f) {
  if (n <= 64) f(std::integral_constant<int, 64>{});
  else if (n <= 256) f(std::integral_constant<int, 256>{});
  else if (n <= 1024) f(std::integral_constant<int, 1024>{});
  else if (n <= 4096) f(std::integral_constant<int, 4096>{});
  else if (n <= 16384) f(std::integral_constant<int, 16384>{});
  else throw std::runtime_error("too many output bits; reduce --d");
}

// Materializes every output bit (public constants become public wires), packs the
// unique wires into one BitVec (as lr_gc_ag2pc does for the weights), checkpoints and
// reveals it to both parties, then decodes each SInt as two's complement at its width.
Result reveal_public(emp::AG2PCSession& sess, Builder<Ctx>& B, const DotSigmoid<Ctx>& c) {
  const std::vector<const SInt<Ctx>*> outs = c.outputs();
  std::vector<std::vector<Wire>> mat(outs.size());
  std::vector<Wire> uniq;
  std::unordered_map<Wire, size_t> pos;
  for (size_t k = 0; k < outs.size(); ++k)
    for (const auto& b : outs[k]->b) {
      const Wire w = B.materialize(b);
      mat[k].push_back(w);
      if (pos.emplace(w, uniq.size()).second) uniq.push_back(w);
    }
  std::vector<uint8_t> bits;
  with_capacity(uniq.size(), [&](auto cap) {
    constexpr int N = decltype(cap)::value;
    std::vector<Wire> ids = uniq;
    ids.resize(N, B.materialize(Builder<Ctx>::cst(false)));
    auto bv = std::make_unique<emp::BitVec_T<Ctx, N>>(emp::BitVec_T<Ctx, N>::from_wires(sess.ctx(), ids.data()));
    sess.checkpoint(*bv);
    auto out = sess.reveal(*bv, emp::PUBLIC);
    if (!out) throw std::runtime_error("reveal returned nothing");
    bits.assign(out->begin(), out->end());
  });
  std::unordered_map<const SInt<Ctx>*, size_t> idx;
  for (size_t k = 0; k < outs.size(); ++k) idx[outs[k]] = k;
  return collect(c, [&](const SInt<Ctx>& v) {
    const std::vector<Wire>& ws = mat[idx.at(&v)];
    const int n = (int)ws.size();
    uint64_t u = 0;
    for (int i = 0; i < n; ++i)
      if (bits[pos.at(ws[(size_t)i])]) u |= uint64_t(1) << i;
    if (n < 64 && ((u >> (n - 1)) & 1)) u |= ~((uint64_t(1) << n) - 1);
    return (int64_t)u;
  });
}

int run_2pc(const Options& O, const Params& P, const SigmoidTable& S, const std::vector<int64_t>& w,
            const std::vector<int64_t>& x) {
  if (emp::is_test_mode()) {
    std::fprintf(stderr, "error: EMP_TEST_MODE is set; unset it.\n");
    return 2;
  }
  const bool is_client = O.role == "client";
  const char* me = is_client ? "client" : "server";
  const int party = is_client ? emp::ALICE : emp::BOB;
  const uint64_t predicted = and_count(P, S, true), core = and_count(P, S, false);
  std::printf("[%s] d=%d qbits=%d (f=%d), circuit: %llu AND gates (%llu for dot + truncation + sigmoid, %llu for "
              "the separately revealed products); %s\n",
              me, P.d, P.qbits, P.f, (unsigned long long)predicted, (unsigned long long)core,
              (unsigned long long)(predicted - core),
              is_client ? ("connecting to " + O.host + ":" + std::to_string(O.port)).c_str()
                        : ("listening on port " + std::to_string(O.port)).c_str());
  std::unique_ptr<emp::NetIO> io = is_client ? emp::NetIO::connect(O.host.c_str(), O.port, /*quiet=*/true)
                                             : emp::NetIO::listen(O.port, /*quiet=*/true);
  ::ThreadPool pool(O.threads);
  auto sess = std::make_unique<emp::AG2PCSession>(io.get(), &pool, party, O.ssp);

  // The client inputs w and x as qbits secret bits each; the server feeds
  // placeholders for the same shape.
  using BitV = emp::Bit_T<Ctx>;
  Builder<Ctx> B(sess->ctx());
  LRCircuit<Ctx> lr(B, P, S);
  auto batch = sess->input_batch();
  std::vector<std::vector<Wire>> ww((size_t)P.d), xw((size_t)P.d);
  for (int j = 0; j < P.d; ++j)
    for (bool bit : lrgc::to_bits(is_client ? w[(size_t)j] : 0, P.qbits))
      ww[(size_t)j].push_back(batch.add<BitV>(emp::ALICE, bit).w);
  for (int j = 0; j < P.d; ++j)
    for (bool bit : lrgc::to_bits(is_client ? x[(size_t)j] : 0, P.qbits))
      xw[(size_t)j].push_back(batch.add<BitV>(emp::ALICE, bit).w);
  batch.finish();
  std::vector<SInt<Ctx>> W, X;
  for (int j = 0; j < P.d; ++j) W.push_back(lr.quantized(ww[(size_t)j]));
  for (int j = 0; j < P.d; ++j) X.push_back(lr.quantized(xw[(size_t)j]));

  const DotSigmoid<Ctx> c(B, lr, W, X);
  const Result gc = reveal_public(*sess, B, c);
  const uint64_t garbled = sess->num_and();
  std::printf("[%s] protocol done: %llu AND gates garbled/evaluated\n", me, (unsigned long long)garbled);
  sess.reset();

  // The server checks against the inputs given on its command line (test only).
  report(is_client ? "client, AG2PC" : "server, AG2PC", P, S, w, x, gc);
  check(garbled == predicted, fmt("AND gates garbled/evaluated (%llu) == CountCtx prediction (%llu)",
                                  (unsigned long long)garbled, (unsigned long long)predicted));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options O;
  Params P;
  try {
    O = parse_options(argc, argv, P);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\nusage: %s <local|client|server> [--d N] [--qbits N] "
                         "[--seed N | --w a,b,.. --x a,b,..] [--trials N] [--host IP] [--port N] [--threads N] "
                         "[--ssp N]\n  --w / --x: integers in [MINQ, MAXQ] = [-2^(qbits-1), 2^(qbits-1)-1]\n",
                 e.what(), argv[0]);
    return 2;
  }
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  const SigmoidTable S = lrgc::make_sigmoid_table(P.f);
  std::vector<int64_t> w = O.w, x = O.x;
  if (w.empty()) random_inputs(P, O.seed, O.d, w, x);

  int rc = 0;
  if (O.role == "local") {
    const uint64_t total = and_count(P, S, true), core = and_count(P, S, false);
    std::printf("[local] d=%d qbits=%d (f=%d), circuit: %llu AND gates (%llu for dot + truncation + sigmoid, %llu "
                "for the separately revealed products)\n",
                P.d, P.qbits, P.f, (unsigned long long)total, (unsigned long long)core,
                (unsigned long long)(total - core));
    report("local, ClearCtx", P, S, w, x, clear_run(P, S, w, x));
    if (O.trials > 0) trials(P, S, O.trials, O.d);
  } else {
    rc = run_2pc(O, P, S, w, x);
  }
  std::printf("\n%s\n", failures ? "SOME CHECKS FAILED" : "all checks passed");
  return rc ? rc : (failures ? 1 : 0);
}
