// params.h -- parameters, the client's random data, the least-squares sigmoid and the plaintext reference
// of Han, Hong, Cheon, Park, "Logistic Regression on Homomorphic Encrypted Data at Scale" (IAAI-19).
//
// Algorithm (paper Sec. 3). With z_i = (2 y_i - 1) x_i, the logistic-regression gradient is
// sum_i sigma(-z_i . w) z_i. One iteration, on a mini-batch block of B samples:
//     u_i  = z_i . v                                    (v = the Nesterov "look-ahead" weights)
//     g    = (lr / B) * sum_{i in block} z_i * H(u_i)   H = degree-3 least-squares fit of sigmoid(-u) on [-R, R]
//     w'   = v + g
//     v'   = (1 - eta) * w' + eta * w                   Nesterov accelerated gradient, eta from the alpha sequence
//     w    = w'
// with alpha_0 = 0.01, alpha_{t+1} = (1 + sqrt(1 + 4 alpha_t^2)) / 2 and eta_t = (1 - alpha_t) / alpha_{t+1}
// (the authors' code). H(u) = a0 + a1 u + a3 u^3 needs one square and two multiplications, so an iteration
// consumes 3 wBits + 2 pBits of ciphertext modulus and the ciphertexts are bootstrapped every `iterPerBoot`
// iterations.
//
// Packing (paper Sec. 3.2). A block of B samples times a group of `bsz` features fills one ciphertext with
// slots = B * bsz slots: slot (bsz*k + l) = z[sample k][feature j*bsz + l] (ciphertext (t, j) of block t, feature
// group j). The d (padded to a power of two) features are split in `cnum` groups so that bootstrapping and the
// per-iteration work parallelize over cnum ciphertexts per weight vector.
#ifndef HAN_PARAMS_H
#define HAN_PARAMS_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace han {

inline int ceil_log2(long x) {
  int b = 0;
  while ((1L << b) < x) ++b;
  return b;
}

struct Params {
  // ---- training ----
  int d = 5;              // features including the bias column
  int iters = 20;         // NAG iterations
  int batch = 4;          // samples per iteration (the mini-batch block size B), a power of two
  double lr = 0.5;        // learning rate alpha; gamma = alpha / B
  bool bias = true;
  bool zero_init = false;
  int qbits = 6;          // quantization of the client's raw random values (as in the sibling baselines)
  uint64_t seed = 1;
  double range = 8.0;     // the sigmoid fit is least-squares on [-range, range]
  // ---- HE (the paper's defaults) ----
  int wBits = 40, pBits = 15, lBits = 5;
  int iterPerBoot = 3;
  int lambda = 80;        // the paper's security level (see README); --lambda 128 needs N = 2^17
  int logN_override = 0;  // 0 = derive from lambda; otherwise the given ring degree (INSECURE if below the bound)
  int threads = 0;        // 0 = all hardware threads
  int cnum_override = 0;  // 0 = automatic (see finalize)
  // ---- derived by finalize() ----
  int factorNum = 0;      // d padded to a power of two
  int cnum = 0;           // ciphertexts per weight vector (feature groups)
  int bsz = 0;            // features per ciphertext
  int slots = 0;          // batch * bsz
  int bBits = 0, sBits = 0;
  int logQ = 0, logq = 0, logQBoot = 0, logN = 0, logI = 4, logT = 3;
  int nblocks = 0;        // = iters (one block of `batch` fresh samples per iteration)
  bool insecure = false;  // ring degree below the security bound of `lambda`

  void finalize() {
    if (d < 1) throw std::invalid_argument("--d must be >= 1");
    if (iters < 1) throw std::invalid_argument("--iters must be >= 1");
    if (batch < 1 || (batch & (batch - 1))) throw std::invalid_argument("--batch must be a power of two (it is the mini-batch block size of the packing)");
    if (!(lr > 0)) throw std::invalid_argument("--lr must be > 0");
    if (qbits < 2 || qbits > 12) throw std::invalid_argument("--qbits must be in [2, 12]");
    if (iterPerBoot < 1 || iterPerBoot > 4) throw std::invalid_argument("iterPerBoot must be in [1, 4]");
    if (threads <= 0) threads = (int)std::max(2u, std::thread::hardware_concurrency());
    factorNum = 1 << ceil_log2(d);
    // Moduli, exactly as the authors: an iteration consumes 3 wBits + 2 pBits.
    logQ = wBits + lBits + iterPerBoot * (3 * wBits + 2 * pBits);
    logq = wBits + lBits;
    const long bitForBoot = 16 + logT + logI + (logI + logT + 6) * (logq + logI);
    logQBoot = (int)(logQ + bitForBoot);
    // Ring degree from HEAAN's estimate N >= logQBoot * (lambda + 110) / 3.6
    const long NBnd = (long)std::ceil(logQBoot * (lambda + 110) / 3.6);
    const int logNsec = ceil_log2(NBnd);
    logN = logN_override ? logN_override : logNsec;
    insecure = logN < logNsec;
    if (logN < 8 || logN > 18) throw std::invalid_argument("ring degree out of range");
    const long slotsMax = 1L << (logN - 1);
    // cnum = ciphertexts per weight vector = feature groups. The paper uses one per thread so that the
    // bootstrapping (2 cnum ciphertexts per round: w and v) and the per-iteration work run in parallel. Default
    // here: the smallest power of two >= threads/2 (so that 2 cnum >= threads), but at least 2 features per
    // ciphertext, and enough ciphertexts for the features to fit the slots.
    if (cnum_override > 0) {
      cnum = cnum_override;
      if ((cnum & (cnum - 1)) || cnum > factorNum) throw std::invalid_argument("--cnum must be a power of two <= the padded number of features");
    } else {
      int target = 1;
      while (target < (threads + 1) / 2) target *= 2;
      cnum = std::max(1, std::min(target, factorNum / 2));
    }
    while ((long)batch * (factorNum / cnum) > slotsMax && cnum < factorNum) cnum *= 2;
    bsz = factorNum / cnum;
    slots = batch * bsz;
    bBits = ceil_log2(bsz);
    sBits = ceil_log2(slots);
    nblocks = iters;
    if (slots > (1 << (logN - 1))) throw std::invalid_argument("batch * features-per-ciphertext exceeds the number of slots N/2: use more threads or a larger ring");
  }
  long slots_total_ciphertexts() const { return (long)nblocks * cnum + 2L * cnum; }
};

// ---- the least-squares sigmoid ---------------------------------------------------------------------
// H(u) = a0 + a1 u + a3 u^3 minimizing the integral of (H(u) - 1/(1+e^u))^2 over [-R, R]  (sigmoid(-u)).
struct Poly3 { double a0, a1, a3; };
inline Poly3 fit_sigmoid(double R) {
  // g(u) - 1/2 is odd, so a0 = 1/2 exactly; fit a1 u + a3 u^3 by the normal equations.
  const int M = 200000;
  auto g = [](double u) { return 1.0 / (1.0 + std::exp(u)) - 0.5; };
  double s1 = 0, s3 = 0;  // integral of u g, u^3 g (Simpson)
  const double h = 2 * R / M;
  for (int i = 0; i <= M; ++i) {
    const double u = -R + i * h;
    const double w = (i == 0 || i == M) ? 1 : (i % 2 ? 4 : 2);
    s1 += w * u * g(u);
    s3 += w * u * u * u * g(u);
  }
  s1 *= h / 3; s3 *= h / 3;
  const double m2 = 2 * std::pow(R, 3) / 3, m4 = 2 * std::pow(R, 5) / 5, m6 = 2 * std::pow(R, 7) / 7;
  const double det = m2 * m6 - m4 * m4;
  Poly3 p;
  p.a0 = 0.5;
  p.a1 = (s1 * m6 - s3 * m4) / det;
  p.a3 = (m2 * s3 - m4 * s1) / det;
  return p;
}

// ---- data --------------------------------------------------------------------------------------------
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

// The client's private data (the same generator and seeds as the sibling baselines, so the same --seed gives the
// same quantized values in [-1, 31/32] and the same labels): z[t*batch + i][j] = (2 y - 1) x_ij, and the initial
// weights. Everything is padded with zeros to factorNum features.
struct Dataset {
  std::vector<double> z;   // [(t*batch + i) * factorNum + j]
  std::vector<int> y;      // [t*batch + i]
  std::vector<double> w0;  // [factorNum]
  double zr(int64_t r, int j, int factorNum) const { return z[(size_t)(r * factorNum + j)]; }
};

inline Dataset generate_client_data(const Params& P) {
  Dataset D;
  const int64_t n = (int64_t)P.iters * P.batch;
  const int fq = P.qbits - 1;
  const int64_t minq = -(int64_t(1) << fq), maxq = (int64_t(1) << fq) - 1;
  const double sc = 1.0 / (double)(int64_t(1) << fq);
  D.z.assign((size_t)(n * P.factorNum), 0.0);
  D.y.assign((size_t)n, 0);
  D.w0.assign((size_t)P.factorNum, 0.0);
  SplitMix64 rng(P.seed * 0x2545F4914F6CDD1DULL + 1);
  for (int64_t r = 0; r < n; ++r) {
    std::vector<double> x((size_t)P.d);
    for (int j = 0; j < P.d; ++j) x[(size_t)j] = ((P.bias && j == P.d - 1) ? maxq : rng.uniform(minq, maxq)) * sc;
    D.y[(size_t)r] = (int)rng.uniform(0, 1);
    const double sign = D.y[(size_t)r] ? 1.0 : -1.0;
    for (int j = 0; j < P.d; ++j) D.z[(size_t)(r * P.factorNum + j)] = sign * x[(size_t)j];
  }
  SplitMix64 rw(P.seed * 0x9E3779B97F4A7C15ULL + 2);
  for (int j = 0; j < P.d; ++j) D.w0[(size_t)j] = P.zero_init ? 0.0 : (double)rw.uniform(minq, maxq) * sc;
  return D;
}

// ---- plaintext reference: the same computation in double precision ----------------------------------------
struct RefResult {
  std::vector<double> w;   // final weights (factorNum entries)
  double max_abs_u = 0;    // largest |z . v| met (the polynomial is only accurate within +-range)
};

inline std::vector<double> nag_alphas_eta(int iters) {
  std::vector<double> eta((size_t)iters);
  double a0 = 0.01, a1 = (1.0 + std::sqrt(1.0 + 4.0 * a0 * a0)) / 2.0;
  for (int t = 0; t < iters; ++t) {
    eta[(size_t)t] = (1.0 - a0) / a1;
    a0 = a1;
    a1 = (1.0 + std::sqrt(1.0 + 4.0 * a0 * a0)) / 2.0;
  }
  return eta;
}

// exact_sigmoid = false: the degree-3 polynomial of the protocol; true: the logistic function.
inline RefResult reference_train(const Params& P, const Dataset& D, const Poly3& H, bool exact_sigmoid = false) {
  const int F = P.factorNum;
  std::vector<double> w = D.w0, v = D.w0;
  const std::vector<double> eta = nag_alphas_eta(P.iters);
  const double gamma = P.lr / P.batch;
  RefResult R;
  for (int t = 0; t < P.iters; ++t) {
    std::vector<double> g((size_t)F, 0.0);
    for (int i = 0; i < P.batch; ++i) {
      const int64_t r = (int64_t)t * P.batch + i;
      double u = 0;
      for (int j = 0; j < F; ++j) u += D.z[(size_t)(r * F + j)] * v[(size_t)j];
      R.max_abs_u = std::max(R.max_abs_u, std::fabs(u));
      const double h = exact_sigmoid ? 1.0 / (1.0 + std::exp(u)) : H.a0 + H.a1 * u + H.a3 * u * u * u;
      for (int j = 0; j < F; ++j) g[(size_t)j] += gamma * h * D.z[(size_t)(r * F + j)];
    }
    for (int j = 0; j < F; ++j) {
      const double tmp1 = v[(size_t)j] + g[(size_t)j];
      v[(size_t)j] = tmp1 * (1.0 - eta[(size_t)t]) + eta[(size_t)t] * w[(size_t)j];
      w[(size_t)j] = tmp1;
    }
  }
  R.w = w;
  return R;
}

// ---- CLI helpers -----------------------------------------------------------------------------------
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
  else if (a == "--lr") P.lr = parse_double(r.next_value(a), a);
  else if (a == "--qbits") P.qbits = (int)parse_int(r.next_value(a), a);
  else if (a == "--seed") P.seed = (uint64_t)parse_int(r.next_value(a), a);
  else if (a == "--range") P.range = parse_double(r.next_value(a), a);
  else if (a == "--lambda") P.lambda = (int)parse_int(r.next_value(a), a);
  else if (a == "--logn") P.logN_override = (int)parse_int(r.next_value(a), a);
  else if (a == "--threads") P.threads = (int)parse_int(r.next_value(a), a);
  else if (a == "--cnum") P.cnum_override = (int)parse_int(r.next_value(a), a);
  else if (a == "--wbits") P.wBits = (int)parse_int(r.next_value(a), a);
  else if (a == "--pbits") P.pBits = (int)parse_int(r.next_value(a), a);
  else if (a == "--iter-per-boot") P.iterPerBoot = (int)parse_int(r.next_value(a), a);
  else if (a == "--no-bias") P.bias = false;
  else if (a == "--zero-init") P.zero_init = true;
  else return false;
  return true;
}

inline std::string params_string(const Params& P) {
  char buf[640];
  std::snprintf(buf, sizeof buf,
                "d=%d (%s; padded to %d) iters=%d batch=%d lr=%g | HE: logN=%d%s logQ=%d logQBoot=%d wBits=%d pBits=%d, bootstrap every %d iterations | packing: %d ciphertexts x %d features, %d slots",
                P.d, P.bias ? "incl. bias column" : "no bias column", P.factorNum, P.iters, P.batch, P.lr, P.logN,
                P.insecure ? " (BELOW the security bound, test only)" : "", P.logQ, P.logQBoot, P.wBits, P.pBits, P.iterPerBoot,
                P.cnum, P.bsz, P.slots);
  return buf;
}

}  // namespace han

#endif  // HAN_PARAMS_H
