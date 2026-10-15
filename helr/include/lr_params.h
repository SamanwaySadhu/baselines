// lr_params.h -- training parameters, deterministic data generation, CLI helpers.
//
// COPY of ../cryptonn/include/lr_params.h (itself a copy of ../gc_ag2pc/include/lr_params.h),
// unchanged except for this note and the namespace (lrhe), so that the same --seed gives
// the same data and initial weights in all four baselines (GC, MD-ML, CryptoNN, HELR).
// Keep the copies in sync.
//
// Fixed-point convention (mirrors the FE protocol's QUANTIZATION_BITS code; default
// qbits = 6):
//   * a quantized value with q = qbits is a signed q-bit integer in
//     [MINQ, MAXQ] = [-2^(q-1), 2^(q-1)-1] representing value / 2^f, f = q-1
//     (qbits = 6: integers in [-32, 31] for values in [-1, 31/32], f = 5);
//   * features x and initial weights w are quantized values; the last feature is
//     the bias column, fixed to MAXQ (as in the FE code: x = MAX_X);
//   * labels are y in {0, MAXQ}.
//   * every dot product (scale 2^(2f)) is truncated back to scale 2^f by an
//     arithmetic shift right by f (floor); see lr_reference.h.
//
// The CLIENT owns every private input: the training data and the initial weights.
// (In the GC and HELR baselines the server learns only the trained weights; in CryptoNN the
// server holds the model in the clear, see lr_cryptonn.cpp.) The data and initial
// weights are generated exactly as in the MD-ML and GC baselines (../mdml,
// ../gc_ag2pc), so the same --seed gives the same inputs in every baseline.
#ifndef LRHE_PARAMS_H
#define LRHE_PARAMS_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace lrhe {

struct Params {
  int d = 5;          // number of features INCLUDING the bias column (FE: FEATURE_SIZE)
  int iters = 20;     // SGD iterations
  int batch = 4;      // datapoints per iteration (FE: BATCH_SIZE)
  int qbits = 6;      // quantization bits (FE: QUANTIZATION_BITS)
  double lr = 0.5;    // learning rate (FE: LR); batch / lr must be a power of two
  bool bias = true;   // last feature is the constant bias column
  bool zero_init = false;  // client's initial weights: 0 instead of uniform random
  uint64_t seed = 1;

  // ---- derived by finalize() ----
  int f = 0;           // fractional bits = qbits - 1
  int64_t minq = 0, maxq = 0;
  int k_shift = 0;     // K = batch / lr = 2^k_shift
  int d_data = 0;      // feature columns supplied by the client (d - 1 with bias)

  void finalize() {
    if (d < 1) throw std::invalid_argument("--d must be >= 1");
    if (iters < 1) throw std::invalid_argument("--iters must be >= 1");
    if (batch < 1) throw std::invalid_argument("--batch must be >= 1");
    if (qbits < 2 || qbits > 12) throw std::invalid_argument("--qbits must be in [2, 12]");
    if (!(lr > 0)) throw std::invalid_argument("--lr must be > 0");
    f = qbits - 1;
    minq = -(int64_t(1) << f);
    maxq = (int64_t(1) << f) - 1;
    d_data = bias ? d - 1 : d;
    // The FE update is w' = trunc((K*w - (g >> f)) / K) with K = batch / lr. The
    // circuit implements the division as an exact shift, so K must be 2^k, k >= 0.
    const double K = (double)batch / lr;
    const int k = (int)std::lround(std::log2(K));
    if (k < 0 || k > 30 || std::fabs(std::ldexp(1.0, k) - K) > 1e-9 * K)
      throw std::invalid_argument("batch / lr must be a power of two >= 1 (e.g. batch=4: lr in {4,2,1,0.5,0.25,...})");
    k_shift = k;
  }
};

// splitmix64: tiny, platform-independent PRNG so both parties (and the reference)
// regenerate identical data on any compiler / standard library.
struct SplitMix64 {
  uint64_t s;
  explicit SplitMix64(uint64_t seed) : s(seed) {}
  uint64_t next() {
    uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  int64_t uniform(int64_t lo, int64_t hi) {  // inclusive, unbiased (rejection sampling)
    const uint64_t range = (uint64_t)(hi - lo) + 1;
    const uint64_t limit = UINT64_MAX - (UINT64_MAX % range);
    uint64_t r;
    do { r = next(); } while (r >= limit);
    return lo + (int64_t)(r % range);
  }
};

// The client's private training data, uniformly random for now.
struct Dataset {
  int iters = 0, batch = 0, d = 0;
  std::vector<int64_t> x;  // [t][i][j], all d columns (bias column = MAXQ)
  std::vector<int> y;      // [t][i] label bit; the label value is y * MAXQ
  int64_t& X(int t, int i, int j) { return x[((size_t)t * batch + i) * d + j]; }
  int64_t X(int t, int i, int j) const { return x[((size_t)t * batch + i) * d + j]; }
  int& Y(int t, int i) { return y[(size_t)t * batch + i]; }
  int Y(int t, int i) const { return y[(size_t)t * batch + i]; }
};

inline Dataset generate_client_data(const Params& P) {
  Dataset D;
  D.iters = P.iters; D.batch = P.batch; D.d = P.d;
  D.x.assign((size_t)P.iters * P.batch * P.d, 0);
  D.y.assign((size_t)P.iters * P.batch, 0);
  SplitMix64 rng(P.seed * 0x2545F4914F6CDD1DULL + 1);
  for (int t = 0; t < P.iters; ++t)
    for (int i = 0; i < P.batch; ++i) {
      for (int j = 0; j < P.d; ++j)
        D.X(t, i, j) = (P.bias && j == P.d - 1) ? P.maxq : rng.uniform(P.minq, P.maxq);
      D.Y(t, i) = (int)rng.uniform(0, 1);
    }
  return D;
}

// The client's initial weights, uniform quantized values (or zeros with --zero-init).
inline std::vector<int64_t> generate_initial_weights(const Params& P) {
  std::vector<int64_t> w((size_t)P.d, 0);
  if (P.zero_init) return w;
  SplitMix64 rng(P.seed * 0x9E3779B97F4A7C15ULL + 2);
  for (int j = 0; j < P.d; ++j) w[j] = rng.uniform(P.minq, P.maxq);
  return w;
}

// Two's-complement bits (LSB first) of v at `width` bits.
inline std::vector<bool> to_bits(int64_t v, int width) {
  std::vector<bool> b((size_t)width);
  for (int i = 0; i < width; ++i) b[i] = ((uint64_t)v >> (i < 63 ? i : 63)) & 1;
  return b;
}

// ------------------------------------------------------------- CLI helpers
struct ArgReader {
  int argc;
  char** argv;
  int i;
  bool has() const { return i < argc; }
  std::string next_value(const std::string& flag) {
    if (i + 1 >= argc) throw std::invalid_argument("missing value for " + flag);
    return argv[++i];
  }
};

inline long long parse_int(const std::string& s, const std::string& flag) {
  try {
    size_t pos = 0;
    long long v = std::stoll(s, &pos, 0);
    if (pos != s.size()) throw std::invalid_argument("");
    return v;
  } catch (...) {
    throw std::invalid_argument("bad integer for " + flag + ": " + s);
  }
}
inline double parse_double(const std::string& s, const std::string& flag) {
  try {
    size_t pos = 0;
    double v = std::stod(s, &pos);
    if (pos != s.size()) throw std::invalid_argument("");
    return v;
  } catch (...) {
    throw std::invalid_argument("bad number for " + flag + ": " + s);
  }
}

// Consumes one training-parameter flag at argv[r.i]; returns false if not one.
inline bool parse_param_flag(ArgReader& r, Params& P) {
  const std::string a = r.argv[r.i];
  if (a == "--d") P.d = (int)parse_int(r.next_value(a), a);
  else if (a == "--iters") P.iters = (int)parse_int(r.next_value(a), a);
  else if (a == "--batch") P.batch = (int)parse_int(r.next_value(a), a);
  else if (a == "--qbits") P.qbits = (int)parse_int(r.next_value(a), a);
  else if (a == "--lr") P.lr = parse_double(r.next_value(a), a);
  else if (a == "--seed") P.seed = (uint64_t)parse_int(r.next_value(a), a);
  else if (a == "--no-bias") P.bias = false;
  else if (a == "--zero-init") P.zero_init = true;
  else return false;
  return true;
}

inline std::string params_string(const Params& P) {
  char buf[512];
  std::snprintf(buf, sizeof buf,
                "d=%d (%s) iters=%d batch=%d qbits=%d (values in [%lld,%lld], f=%d) lr=%g (K=batch/lr=%d) "
                "seed=%llu init=%s",
                P.d, P.bias ? "incl. bias column" : "no bias column", P.iters, P.batch, P.qbits,
                (long long)P.minq, (long long)P.maxq, P.f, P.lr, 1 << P.k_shift,
                (unsigned long long)P.seed, P.zero_init ? "zeros" : "uniform");
  return buf;
}

}  // namespace lrhe

#endif  // LRHE_PARAMS_H
