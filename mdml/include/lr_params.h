// lr_params.h -- training parameters, deterministic data generation, CLI helpers.
//
// The data is generated exactly as in baseline 1 (../gc_ag2pc/include/lr_params.h),
// so the same --seed gives the same dataset in both baselines:
//   * features x and initial weights w are quantized values with q = qbits:
//     signed integers in [MINQ, MAXQ] = [-2^(q-1), 2^(q-1)-1] representing v / 2^(q-1);
//   * the last feature is the bias column, fixed to MAXQ (FE code: x = MAX_X);
//   * labels are y in {0, 1}.
// MD-ML computes on fixed-point values in Z_{2^64} with F = --frac-bits fractional
// bits; a quantized value v becomes v * 2^(F - (q-1)) (exact), a label y becomes y * 2^F.
#ifndef LRMDML_PARAMS_H
#define LRMDML_PARAMS_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace lrmd {

struct Params {
  int d = 5;          // number of features INCLUDING the bias column
  int iters = 20;     // SGD iterations
  int batch = 4;      // datapoints per iteration
  int qbits = 6;      // quantization of the generated data (as in baseline 1)
  double lr = 0.5;    // learning rate; batch / lr must be a power of two
  int frac = 16;      // fixed-point fractional bits F (MD-ML code default 16; the paper uses 20)
  bool bias = true;   // last feature is the constant bias column
  bool zero_init = false;  // client's initial weights: 0 instead of uniform random
  uint64_t seed = 1;

  // ---- derived by finalize() ----
  int f = 0;           // fractional bits of the quantized data = qbits - 1
  int64_t minq = 0, maxq = 0;
  int k_shift = 0;     // batch / lr = 2^k_shift; the update truncates by F + k_shift

  void finalize() {
    if (d < 1) throw std::invalid_argument("--d must be >= 1");
    if (iters < 1) throw std::invalid_argument("--iters must be >= 1");
    if (batch < 1) throw std::invalid_argument("--batch must be >= 1");
    if (qbits < 2 || qbits > 12) throw std::invalid_argument("--qbits must be in [2, 12]");
    if (!(lr > 0)) throw std::invalid_argument("--lr must be > 0");
    f = qbits - 1;
    minq = -(int64_t(1) << f);
    maxq = (int64_t(1) << f) - 1;
    if (frac < f || frac > 28)
      throw std::invalid_argument("--frac-bits must be in [qbits-1, 28]");
    // w <- w - (lr/batch) * X^T e is computed as one multiply-then-truncate by
    // F + k with 2^k = batch / lr, so batch / lr must be 2^k, k >= 0 (as in baseline 1).
    const double K = (double)batch / lr;
    const int k = (int)std::lround(std::log2(K));
    if (k < 0 || k > 30 || std::fabs(std::ldexp(1.0, k) - K) > 1e-9 * K)
      throw std::invalid_argument("batch / lr must be a power of two >= 1 (e.g. batch=4: lr in {4,2,1,0.5,0.25,...})");
    k_shift = k;
  }

  // Identifies the circuit shape; the dealer writes it into the preprocessing files
  // and the parties refuse material generated for a different shape.
  uint64_t fingerprint() const {
    uint64_t h = 0x6d646d6c2d6c7231ULL;  // "mdml-lr1"
    for (uint64_t v : {(uint64_t)d, (uint64_t)iters, (uint64_t)batch, (uint64_t)frac, (uint64_t)k_shift}) {
      h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      h *= 0xbf58476d1ce4e5b9ULL;
    }
    return h;
  }
};

// splitmix64: platform-independent PRNG (identical to baseline 1).
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

// The client's private training data (quantized integers), uniformly random for now.
struct Dataset {
  int iters = 0, batch = 0, d = 0;
  std::vector<int64_t> x;  // [t][i][j], all d columns (bias column = MAXQ)
  std::vector<int> y;      // [t][i] label bit
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

// Same generator as baseline 1's generate_server_weights (here the client holds them).
inline std::vector<int64_t> generate_initial_weights(const Params& P) {
  std::vector<int64_t> w((size_t)P.d, 0);
  if (P.zero_init) return w;
  SplitMix64 rng(P.seed * 0x9E3779B97F4A7C15ULL + 2);
  for (int j = 0; j < P.d; ++j) w[j] = rng.uniform(P.minq, P.maxq);
  return w;
}

// Fixed-point encodings in Z_{2^64}.
inline uint64_t fx_from_quant(const Params& P, int64_t v) { return (uint64_t)(v * (int64_t(1) << (P.frac - P.f))); }
inline uint64_t fx_from_label(const Params& P, int y) { return (uint64_t)y << P.frac; }
inline double fx_to_double(const Params& P, uint64_t v) { return std::ldexp((double)(int64_t)v, -P.frac); }

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
  else if (a == "--frac-bits") P.frac = (int)parse_int(r.next_value(a), a);
  else if (a == "--seed") P.seed = (uint64_t)parse_int(r.next_value(a), a);
  else if (a == "--no-bias") P.bias = false;
  else if (a == "--zero-init") P.zero_init = true;
  else return false;
  return true;
}

inline const char* kParamHelp =
    "training (dealer, client and server must pass the same values):\n"
    "  --d N            features incl. the bias column (default 5)\n"
    "  --iters N        SGD iterations (default 20)\n"
    "  --batch N        datapoints per iteration (default 4)\n"
    "  --lr X           learning rate; batch/lr must be a power of two (default 0.5)\n"
    "  --frac-bits N    fixed-point fractional bits F (default 16)\n"
    "  --qbits N        quantization of the generated data, as in baseline 1 (default 6)\n"
    "  --no-bias        the last feature is a regular feature, not the bias column\n"
    "  --zero-init      client's initial weights are 0 (default: uniform random)\n"
    "  --seed N         data seed (default 1)\n";

inline std::string params_string(const Params& P) {
  char buf[512];
  std::snprintf(buf, sizeof buf,
                "d=%d (%s) iters=%d batch=%d lr=%g (update shift F+%d) F=%d fractional bits, "
                "data: qbits=%d seed=%llu init=%s",
                P.d, P.bias ? "incl. bias column" : "no bias column", P.iters, P.batch, P.lr, P.k_shift,
                P.frac, P.qbits, (unsigned long long)P.seed, P.zero_init ? "zeros" : "uniform");
  return buf;
}

}  // namespace lrmd

#endif  // LRMDML_PARAMS_H
