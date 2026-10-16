// params.h -- training parameters, the client's random data, the plaintext references and CLI helpers.
//
// Fixed point (all values live in the ring Z_2^64; a value v at "scale s" stands for v / 2^s, negative
// numbers in two's complement):
//   features x, labels y, weights w, sigmoid output yhat, error e     scale F   (F = --frac, default 13
//                                                                               = Falcon's FLOAT_PRECISION)
//   z = X_t . w                                                         scale 2F  (exact in the ring)
//   u = trunc_F(z)                                                      scale F   (Falcon funcMatMul, fused
//                                                                               truncation)
//   g = trunc_{F+k}(X_t^T . e)                                          scale F   (lr / batch = 2^-k folded
//                                                                               into the shift, as in
//                                                                               Falcon's FCLayer)
// Activation (SecureML's piecewise sigmoid, exactly): yhat = ReLU(u + 1/2) - ReLU(u - 1/2) = clamp(u + 1/2,
// 0, 1); the backward pass uses the logistic-regression gradient (yhat - y) x.
//
// Truncation is Falcon's / ABY3's "truncation pair": the dealer gives a uniform mask R' and a sharing of
// floor(R' / 2^t); the servers open D = z - R' and add floor(D / 2^t). The result is floor(z / 2^t) or
// one less (and is wrong with probability ~|z| / 2^64). reference_train() replays this bit for bit from
// the dealt masks (the client is the dealer, so it knows every R'), reference_train_nominal() uses the
// exact floor, and reference_train_double() is double-precision SGD with the same activation.
//
// The client's data is generated exactly like the sibling baselines (same seed -> same quantized values
// in [-32, 31] / 32, bias column = 31/32); the values are shifted up to scale F, the initial weights too.
#ifndef FAL_PARAMS_H
#define FAL_PARAMS_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace fal {

struct Params {
  int d = 5;           // features INCLUDING the bias column
  int iters = 20;      // SGD iterations
  int batch = 4;       // datapoints per iteration
  int frac = 13;       // F: fractional bits of every fixed-point value
  int qbits = 6;       // quantization of the client's raw random values (as in the sibling baselines)
  double lr = 0.5;     // learning rate alpha; batch / lr must be a power of two
  bool bias = true;
  bool zero_init = false;
  uint64_t seed = 1;

  // ---- derived by finalize() ----
  int fq = 0;                  // qbits - 1
  int64_t minq = 0, maxq = 0;
  int k_shift = 0;             // log2(batch / lr)
  int sw = 0;                  // weight scale (= F: Falcon truncates after every product)
  int64_t n = 0;               // total samples = iters * batch
  int d_data = 0;

  void finalize() {
    if (d < 1 || d > 1000000) throw std::invalid_argument("--d must be in [1, 1000000]");
    if (iters < 1) throw std::invalid_argument("--iters must be >= 1");
    if (batch < 1 || batch > 1000000) throw std::invalid_argument("--batch must be in [1, 1000000]");
    if (qbits < 2 || qbits > 12) throw std::invalid_argument("--qbits must be in [2, 12]");
    fq = qbits - 1;
    if (frac < fq || frac > 24) throw std::invalid_argument("--frac must be in [qbits-1, 24]");
    if (!(lr > 0)) throw std::invalid_argument("--lr must be > 0");
    minq = -(int64_t(1) << fq);
    maxq = (int64_t(1) << fq) - 1;
    d_data = bias ? d - 1 : d;
    const double K = (double)batch / lr;
    const int k = (int)std::lround(std::log2(K));
    if (k < 0 || k > 20 || std::fabs(std::ldexp(1.0, k) - K) > 1e-9 * K)
      throw std::invalid_argument("batch / lr must be a power of two >= 1 (e.g. batch=4: lr in {4,2,1,0.5,0.25,...})");
    k_shift = k;
    sw = frac;
    n = (int64_t)iters * batch;
  }
  int shift_fwd() const { return frac; }            // truncation of X_t . w
  int shift_bwd() const { return frac + k_shift; }  // truncation of X_t^T . e (lr / batch folded in)
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
// labels y in {0,1}, and the initial weights at scale S_w = F.
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

// ---- plaintext references (ring Z_2^64, uint64_t wrap-around) -----------------------------------
// Truncation by the pair (R', floor(R'/2^t)): floor((z - R') / 2^t) + floor(R' / 2^t), both floors as
// arithmetic shifts (the port's replacement for upstream's double-based dividePlain).
inline uint64_t trunc_pair(uint64_t z, uint64_t rp, int t) {
  return (uint64_t)((int64_t)(z - rp) >> t) + (uint64_t)((int64_t)rp >> t);
}
inline uint64_t trunc_exact(uint64_t z, int t) { return (uint64_t)((int64_t)z >> t); }
// ReLU as computed by RELUPrime + SelectShares: a * (1 - MSB(a)) (a = 0 gives 0 either way)
inline uint64_t relu_ring(uint64_t a) { return (int64_t)a >= 0 ? a : 0; }

// The protocol's computation in the clear. trunc(z, which, index, shift): which = 0 forward (index
// t*batch + i), 1 backward (index t*d + j).
template <class Trunc>
std::vector<uint64_t> train_ring(const Params& P, const Dataset& D, Trunc trunc) {
  std::vector<uint64_t> w(P.d);
  for (int j = 0; j < P.d; ++j) w[(size_t)j] = (uint64_t)D.w0[(size_t)j];
  const uint64_t half = uint64_t(1) << (P.frac - 1);
  std::vector<uint64_t> e((size_t)P.batch);
  for (int t = 0; t < P.iters; ++t) {
    for (int i = 0; i < P.batch; ++i) {
      const int64_t r = (int64_t)t * P.batch + i;
      uint64_t z = 0;
      for (int j = 0; j < P.d; ++j) z += (uint64_t)D.x[(size_t)(r * P.d + j)] * w[(size_t)j];
      const uint64_t u = trunc(z, 0, (size_t)r, P.shift_fwd());
      const uint64_t yhat = relu_ring(u + half) - relu_ring(u - half);
      e[(size_t)i] = yhat - ((uint64_t)D.y[(size_t)r] << P.frac);
    }
    for (int j = 0; j < P.d; ++j) {
      uint64_t g = 0;
      for (int i = 0; i < P.batch; ++i) g += (uint64_t)D.x[(size_t)(((int64_t)t * P.batch + i) * P.d + j)] * e[(size_t)i];
      w[(size_t)j] -= trunc(g, 1, (size_t)t * P.d + j, P.shift_bwd());
    }
  }
  return w;
}

// Bit-exact twin of the protocol, given the dealt truncation masks R' (forward: iters*batch, backward:
// iters*d, in protocol order).
inline std::vector<uint64_t> reference_train(const Params& P, const Dataset& D, const std::vector<uint64_t>& rp_fwd,
                                             const std::vector<uint64_t>& rp_bwd) {
  if (rp_fwd.size() != (size_t)P.n || rp_bwd.size() != (size_t)P.iters * P.d)
    throw std::invalid_argument("reference_train: wrong number of truncation masks");
  return train_ring(P, D, [&](uint64_t z, int which, size_t idx, int t) {
    return trunc_pair(z, which == 0 ? rp_fwd[idx] : rp_bwd[idx], t);
  });
}
// The same with exact floor division (what the masks approximate to within 1 ulp per truncation).
inline std::vector<uint64_t> reference_train_nominal(const Params& P, const Dataset& D) {
  return train_ring(P, D, [](uint64_t z, int, size_t, int t) { return trunc_exact(z, t); });
}

// Double-precision SGD on the same data with the same piecewise activation.
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
      const double f = z < -0.5 ? 0.0 : (z > 0.5 ? 1.0 : z + 0.5);
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
  else if (a == "--no-bias") P.bias = false;
  else if (a == "--zero-init") P.zero_init = true;
  else return false;
  return true;
}

inline std::string params_string(const Params& P) {
  char buf[512];
  std::snprintf(buf, sizeof buf,
                "d=%d (%s) iters=%d batch=%d F=%d lr=%g (k=%d) ring Z_2^64, field Z_67; scales: x/y/w/yhat 2^-%d; "
                "truncation shifts %d (forward) / %d (gradient); seed=%llu init=%s",
                P.d, P.bias ? "incl. bias column" : "no bias column", P.iters, P.batch, P.frac, P.lr, P.k_shift, P.frac,
                P.shift_fwd(), P.shift_bwd(), (unsigned long long)P.seed, P.zero_init ? "zeros" : "uniform");
  return buf;
}

}  // namespace fal

#endif  // FAL_PARAMS_H
