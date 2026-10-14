// lr_params.h -- fixed training parameters, deterministic data generation, CLI helpers.
//
// Numbers are IEEE-754 binary32 ("float"): the training data x, the weights and the
// learning rate. Only the dimension, the number of iterations and the data seed are
// configurable; everything else is fixed:
//   * batch size  = 4 datapoints per SGD iteration
//   * learning rate = 0.01 (the update uses lr / batch = 0.0025)
//   * the last feature is the bias column, the constant 1.0f
//   * labels are y in {0, 1}
//
// The CLIENT owns every private input: the training data and the initial weights.
// The server has no private input and learns only the trained weights.
#ifndef LRGC_PARAMS_H
#define LRGC_PARAMS_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace lrgc {

constexpr int kBatch = 4;
constexpr float kLr = 0.01f;
constexpr float kLrOverBatch = kLr / (float)kBatch;  // exact power-of-two scaling of kLr

inline uint32_t f2u(float v) { uint32_t u; std::memcpy(&u, &v, 4); return u; }
inline float u2f(uint32_t u) { float v; std::memcpy(&v, &u, 4); return v; }

struct Params {
  int d = 5;        // number of features INCLUDING the bias column (>= 2)
  int iters = 20;   // SGD iterations
  uint64_t seed = 1;

  int batch() const { return kBatch; }
  int d_data() const { return d - 1; }  // feature columns supplied by the client

  void finalize() const {
    if (d < 2) throw std::invalid_argument("--d must be >= 2 (one feature + the bias column)");
    if (iters < 1) throw std::invalid_argument("--iters must be >= 1");
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
  // uniform float in [lo, hi) on a 2^-24 grid (exactly representable)
  float uniform_float(float lo, float hi) {
    const float u = (float)(next() >> 40) * (1.0f / 16777216.0f);
    return lo + (hi - lo) * u;
  }
};

// The client's private training data (random for now). x holds all d columns; the
// last one is the bias column 1.0f.
struct Dataset {
  int iters = 0, d = 0;
  std::vector<float> x;  // [t][i][j]
  std::vector<int> y;    // [t][i]  label bit (the label value is 0.0f / 1.0f)
  float& X(int t, int i, int j) { return x[((size_t)t * kBatch + i) * d + j]; }
  float X(int t, int i, int j) const { return x[((size_t)t * kBatch + i) * d + j]; }
  int& Y(int t, int i) { return y[(size_t)t * kBatch + i]; }
  int Y(int t, int i) const { return y[(size_t)t * kBatch + i]; }
};

inline Dataset generate_client_data(const Params& P) {
  Dataset D;
  D.iters = P.iters; D.d = P.d;
  D.x.assign((size_t)P.iters * kBatch * P.d, 0.0f);
  D.y.assign((size_t)P.iters * kBatch, 0);
  SplitMix64 rng(P.seed * 0x2545F4914F6CDD1DULL + 1);
  for (int t = 0; t < P.iters; ++t)
    for (int i = 0; i < kBatch; ++i) {
      for (int j = 0; j < P.d; ++j) D.X(t, i, j) = (j == P.d - 1) ? 1.0f : rng.uniform_float(-1.0f, 1.0f);
      D.Y(t, i) = (int)(rng.next() >> 63);
    }
  return D;
}

// The client's initial weights, uniform in [-0.5, 0.5).
inline std::vector<float> generate_initial_weights(const Params& P) {
  std::vector<float> w;
  SplitMix64 rw(P.seed * 0x9E3779B97F4A7C15ULL + 2);
  for (int j = 0; j < P.d; ++j) w.push_back(rw.uniform_float(-0.5f, 0.5f));
  return w;
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

// Consumes one training-parameter flag at argv[r.i]; returns false if not one.
inline bool parse_param_flag(ArgReader& r, Params& P) {
  const std::string a = r.argv[r.i];
  if (a == "--d") P.d = (int)parse_int(r.next_value(a), a);
  else if (a == "--iters") P.iters = (int)parse_int(r.next_value(a), a);
  else if (a == "--seed") P.seed = (uint64_t)parse_int(r.next_value(a), a);
  else return false;
  return true;
}

inline std::string params_string(const Params& P) {
  char buf[256];
  std::snprintf(buf, sizeof buf,
                "d=%d (incl. bias column) iters=%d batch=%d lr=%g (lr/batch=%g) float32; "
                "data and initial weights owned by the client, seed=%llu",
                P.d, P.iters, kBatch, (double)kLr, (double)kLrOverBatch, (unsigned long long)P.seed);
  return buf;
}

}  // namespace lrgc

#endif  // LRGC_PARAMS_H
