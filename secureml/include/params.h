// params.h -- training parameters, derived fixed-point formats, the client's random data, the
// plaintext reference (bit-exact twin of the protocol) and CLI helpers.
//
// Fixed-point format (all integers; a value v at "scale s" stands for v / 2^s):
//   features x             scale F                 (F = --frac, default 13)
//   labels y in {0,1}      scale F
//   weights w              scale S_w = 2F + k      (k = log2(batch / lr), the learning-rate shift)
//   z = sum_j x_j w_j      scale 3F + k            (exact, no truncation of the dot product)
//   sigmoid output yhat    scale F
//   error e = yhat - y     scale F
//   g_j = sum_i e_i x_ij   scale 2F
// Because S_w = 2F + k, the update w_j <- w_j - (lr/batch) g_j needs NO truncation: lr/batch = 2^-k
// and g has scale 2F, so (g / 2^k) at scale S_w is just the integer g. The only truncation in the whole
// protocol is yhat = clamp(floor((z + 1/2) / 2^(2F+k)), 0, 2^F): one exact (not probabilistic) floor
// per sample, done on the activation's masked opening. This replaces SecureML's local share
// truncation, which cannot be combined with MACs.
//
// Activation (SecureML, eq. 4):  f(z) = 0 for z < -1/2,  z + 1/2 for |z| <= 1/2,  1 for z > 1/2,
// and the backward pass uses the logistic-regression gradient (f(z) - y) x as in the paper.
//
// The client's data is generated exactly like the sibling baselines (same seed -> same quantized
// values in [-32, 31] / 32, bias column = 31/32); the values are shifted up to scale F.
#ifndef SML_PARAMS_H
#define SML_PARAMS_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace sml {

struct Params {
  int d = 5;           // features INCLUDING the bias column
  int iters = 20;      // SGD iterations
  int batch = 4;       // datapoints per iteration
  int frac = 13;       // F: fractional bits of features / labels / yhat
  int qbits = 6;       // quantization of the client's raw random values (as in the sibling baselines)
  double lr = 0.5;     // learning rate alpha; batch / lr must be a power of two
  bool bias = true;
  bool zero_init = false;
  uint64_t seed = 1;
  int ssp = 40;        // statistical security parameter (masks hide values up to 2^-ssp)

  // ---- derived by finalize() ----
  int fq = 0;                  // qbits - 1
  int64_t minq = 0, maxq = 0;
  int k_shift = 0;             // log2(batch / lr)
  int sw = 0;                  // weight scale S_w = 2F + k
  int tr = 0;                  // truncation of z + 1/2 to scale F: 2F + k  (== sw)
  int zs = 0;                  // scale of z: 3F + k
  int nbits = 0;               // N: width of the activation gadget (see online.h)
  int64_t n = 0;               // total samples = iters * batch
  int d_data = 0;

  void finalize() {
    if (d < 1) throw std::invalid_argument("--d must be >= 1");
    if (iters < 1) throw std::invalid_argument("--iters must be >= 1");
    if (batch < 1) throw std::invalid_argument("--batch must be >= 1");
    if (qbits < 2 || qbits > 12) throw std::invalid_argument("--qbits must be in [2, 12]");
    fq = qbits - 1;
    if (frac < fq || frac > 20) throw std::invalid_argument("--frac must be in [qbits-1, 20]");
    if (!(lr > 0)) throw std::invalid_argument("--lr must be > 0");
    if (ssp < 8 || ssp > 60) throw std::invalid_argument("--ssp must be in [8, 60]");
    minq = -(int64_t(1) << fq);
    maxq = (int64_t(1) << fq) - 1;
    d_data = bias ? d - 1 : d;
    const double K = (double)batch / lr;
    const int k = (int)std::lround(std::log2(K));
    if (k < 0 || k > 20 || std::fabs(std::ldexp(1.0, k) - K) > 1e-9 * K)
      throw std::invalid_argument("batch / lr must be a power of two >= 1 (e.g. batch=4: lr in {4,2,1,0.5,0.25,...})");
    k_shift = k;
    sw = 2 * frac + k;
    tr = sw;
    zs = frac + sw;
    n = (int64_t)iters * batch;
    // |w| <= 1 + iters*batch/2^k, |z| <= d*|w|max (|x| <= 1); the gadget needs |z + 1/2 - 1| < 2^(N-1-zs).
    const double wmax = 1.0 + (double)iters * batch / std::ldexp(1.0, k);
    const int ibits = (int)std::ceil(std::log2((double)d * wmax + 2.0));
    nbits = zs + ibits + 2;
    if (nbits + ssp + 2 > 125) throw std::invalid_argument("parameters too large for the 127-bit field (reduce --frac / --d / --iters)");
  }
  int mask_bits() const { return nbits + ssp; }  // secret random bits per activation
};

struct SplitMix64 {
  uint64_t s;
  explicit SplitMix64(uint64_t seed) : s(seed) {}
  uint64_t next() {
    uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  int64_t uniform(int64_t lo, int64_t hi) {  // inclusive, unbiased
    const uint64_t range = (uint64_t)(hi - lo) + 1;
    const uint64_t limit = UINT64_MAX - (UINT64_MAX % range);
    uint64_t r;
    do { r = next(); } while (r >= limit);
    return lo + (int64_t)(r % range);
  }
};

// The client's private data at scale F: x[t][i][j] (d columns, bias column = (2^fq - 1) << (F - fq)),
// labels y in {0,1}, and the initial weights at scale S_w.
struct Dataset {
  std::vector<int64_t> x;   // [(t*batch + i)*d + j], scale F
  std::vector<int> y;       // [t*batch + i]
  std::vector<int64_t> w0;  // [j], scale S_w
};

inline Dataset generate_client_data(const Params& P) {
  Dataset D;
  D.x.assign((size_t)(P.n * P.d), 0);
  D.y.assign((size_t)P.n, 0);
  D.w0.assign((size_t)P.d, 0);
  SplitMix64 rng(P.seed * 0x2545F4914F6CDD1DULL + 1);
  for (int64_t r = 0; r < P.n; ++r) {
    for (int j = 0; j < P.d; ++j) {
      const int64_t q = (P.bias && j == P.d - 1) ? P.maxq : rng.uniform(P.minq, P.maxq);
      D.x[(size_t)(r * P.d + j)] = q * (int64_t(1) << (P.frac - P.fq));
    }
    D.y[(size_t)r] = (int)rng.uniform(0, 1);
  }
  SplitMix64 rw(P.seed * 0x9E3779B97F4A7C15ULL + 2);
  for (int j = 0; j < P.d; ++j) {
    const int64_t q = P.zero_init ? 0 : rw.uniform(P.minq, P.maxq);
    D.w0[(size_t)j] = q * (int64_t(1) << (P.sw - P.fq));
  }
  return D;
}

// ---- plaintext reference: integer twin of the protocol (bit for bit) -----------------------------
using i128 = __int128;

// Returns the trained weights at scale S_w. The check on |v| mirrors the gadget's range assumption.
inline std::vector<i128> reference_train(const Params& P, const Dataset& D, bool* in_range = nullptr) {
  std::vector<i128> w(D.w0.begin(), D.w0.end());
  const i128 half = (i128)1 << (P.zs - 1);
  const i128 one_f = (i128)1 << P.frac;
  const i128 lim = (i128)1 << (P.nbits - 2);
  bool ok = true;
  for (int t = 0; t < P.iters; ++t) {
    std::vector<i128> e((size_t)P.batch);
    for (int i = 0; i < P.batch; ++i) {
      const int64_t r = (int64_t)t * P.batch + i;
      i128 z = 0;
      for (int j = 0; j < P.d; ++j) z += (i128)D.x[(size_t)(r * P.d + j)] * w[(size_t)j];
      const i128 v = z + half;
      if (v >= lim || v <= -lim) ok = false;
      i128 q = v >> P.tr;  // arithmetic shift = floor
      if (q < 0) q = 0;
      if (q > one_f) q = one_f;
      e[(size_t)i] = q - one_f * D.y[(size_t)r];
    }
    for (int j = 0; j < P.d; ++j) {
      i128 g = 0;
      for (int i = 0; i < P.batch; ++i) g += e[(size_t)i] * D.x[(size_t)(((int64_t)t * P.batch + i) * P.d + j)];
      w[(size_t)j] -= g;
    }
  }
  if (in_range) *in_range = ok;
  return w;
}

// Double-precision SGD on the same data with the same piecewise activation (distance of the fixed-point
// result to unquantized training).
inline std::vector<double> reference_train_double(const Params& P, const Dataset& D) {
  std::vector<double> w((size_t)P.d);
  for (int j = 0; j < P.d; ++j) w[(size_t)j] = std::ldexp((double)D.w0[(size_t)j], -P.sw);
  const double rate = P.lr / P.batch;
  for (int t = 0; t < P.iters; ++t) {
    std::vector<double> g((size_t)P.d, 0.0);
    for (int i = 0; i < P.batch; ++i) {
      const int64_t r = (int64_t)t * P.batch + i;
      double z = 0;
      for (int j = 0; j < P.d; ++j) z += std::ldexp((double)D.x[(size_t)(r * P.d + j)], -P.frac) * w[(size_t)j];
      double f = z < -0.5 ? 0.0 : (z > 0.5 ? 1.0 : z + 0.5);
      const double e = f - D.y[(size_t)r];
      for (int j = 0; j < P.d; ++j) g[(size_t)j] += e * std::ldexp((double)D.x[(size_t)(r * P.d + j)], -P.frac);
    }
    for (int j = 0; j < P.d; ++j) w[(size_t)j] -= rate * g[(size_t)j];
  }
  return w;
}

// ---- CLI helpers ------------------------------------------------------------------------------
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
  } catch (...) { throw std::invalid_argument("bad integer for " + flag + ": " + s); }
}
inline double parse_double(const std::string& s, const std::string& flag) {
  try {
    size_t pos = 0;
    double v = std::stod(s, &pos);
    if (pos != s.size()) throw std::invalid_argument("");
    return v;
  } catch (...) { throw std::invalid_argument("bad number for " + flag + ": " + s); }
}
inline bool parse_param_flag(ArgReader& r, Params& P) {
  const std::string a = r.argv[r.i];
  if (a == "--d") P.d = (int)parse_int(r.next_value(a), a);
  else if (a == "--iters") P.iters = (int)parse_int(r.next_value(a), a);
  else if (a == "--batch") P.batch = (int)parse_int(r.next_value(a), a);
  else if (a == "--frac") P.frac = (int)parse_int(r.next_value(a), a);
  else if (a == "--qbits") P.qbits = (int)parse_int(r.next_value(a), a);
  else if (a == "--lr") P.lr = parse_double(r.next_value(a), a);
  else if (a == "--seed") P.seed = (uint64_t)parse_int(r.next_value(a), a);
  else if (a == "--ssp") P.ssp = (int)parse_int(r.next_value(a), a);
  else if (a == "--no-bias") P.bias = false;
  else if (a == "--zero-init") P.zero_init = true;
  else return false;
  return true;
}

inline std::string params_string(const Params& P) {
  char buf[512];
  std::snprintf(buf, sizeof buf,
                "d=%d (%s) iters=%d batch=%d F=%d lr=%g (k=%d) scales: x/y/yhat 2^-%d, w 2^-%d, z 2^-%d; gadget width N=%d, mask bits=%d, seed=%llu init=%s",
                P.d, P.bias ? "incl. bias column" : "no bias column", P.iters, P.batch, P.frac, P.lr, P.k_shift,
                P.frac, P.sw, P.zs, P.nbits, P.mask_bits(), (unsigned long long)P.seed, P.zero_init ? "zeros" : "uniform");
  return buf;
}

}  // namespace sml

#endif  // SML_PARAMS_H
