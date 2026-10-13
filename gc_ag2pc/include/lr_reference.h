// lr_reference.h -- plaintext specification of the quantized logistic-regression
// training that the garbled circuit computes, plus the sigmoid threshold table.
//
// One iteration t on its batch {(x_i, y_i)}, i < batch (all integers, f = qbits-1):
//   z_i    = sum_j w_j * x_ij                          (scale 2^(2f))
//   yhat_i = min( floor(2^f * sigmoid(z_i / 2^(2f))), MAXQ )
//   e_i    = yhat_i - MAXQ * y_i
//   g_j    = sum_i e_i * x_ij                          (scale 2^(2f))
//   w_j   <- trunc( (K*w_j - floor(g_j / 2^f)) / K ),  K = batch / lr
// This is the update of the FE code (backward_prop_kim.cpp: w_scaled = (B/LR)*w,
// output_value = sum (yhat - y) x, w' = (int)((w_scaled - (output_value >> (QB-1))) * LR / B))
// and the sigmoid it intends (non_linear_transform, with the factor 2^f outside sigmoid).
// Arithmetic is exact: no wrap-around; widths grow as needed.
#ifndef LRGC_REFERENCE_H
#define LRGC_REFERENCE_H

#include "lr_params.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lrgc {

inline long double sigmoid_ld(long double x) { return 1.0L / (1.0L + std::exp(-x)); }

// yhat = min(floor(2^f * sigmoid(z / 2^(2f))), 2^f - 1), evaluated in long double
// exactly like the FE code's non_linear_transform. The clamp only matters when
// sigmoid rounds to 1.0L (z / 2^(2f) > ~44), which keeps yhat in [0, MAXQ].
inline int64_t qsigmoid_ref(int64_t z, int f) {
  const long double in_scale = std::ldexp(1.0L, 2 * f);
  const long double out_scale = std::ldexp(1.0L, f);
  const int64_t v = (int64_t)(sigmoid_ld((long double)z / in_scale) * out_scale);
  return std::min<int64_t>(v, (int64_t(1) << f) - 1);
}

// thresholds t[v] = min{ z : qsigmoid_ref(z) >= v } for v = 1..MAXQ (t[0] unused),
// found by an exhaustive scan of the transition window (which also verifies that
// qsigmoid_ref is monotone there and constant outside it).
struct SigmoidTable {
  int f = 0;
  int64_t maxq = 0;
  std::vector<int64_t> t;
};

inline SigmoidTable make_sigmoid_table(int f) {
  SigmoidTable S;
  S.f = f;
  S.maxq = (int64_t(1) << f) - 1;
  // yhat >= 1 iff z / 2^(2f) >= -ln(MAXQ); yhat >= MAXQ iff z / 2^(2f) >= ln(MAXQ).
  const long double span = std::log((long double)S.maxq) * std::ldexp(1.0L, 2 * f);
  const int64_t zlo = (int64_t)std::floor(-span) - 8, zhi = (int64_t)std::ceil(span) + 8;
  if (qsigmoid_ref(zlo, f) != 0 || qsigmoid_ref(zhi, f) != S.maxq)
    throw std::runtime_error("sigmoid table: scan window does not cover the transitions");
  S.t.assign((size_t)S.maxq + 1, 0);
  int64_t prev = 0;
  for (int64_t z = zlo; z <= zhi; ++z) {
    const int64_t q = qsigmoid_ref(z, f);
    if (q < prev) throw std::runtime_error("sigmoid table: quantized sigmoid is not monotone");
    while (prev < q) S.t[(size_t)++prev] = z;
  }
  if (prev != S.maxq) throw std::runtime_error("sigmoid table: incomplete");
  return S;
}

// The same function via the thresholds (what the circuit evaluates).
inline int64_t qsigmoid_by_table(const SigmoidTable& S, int64_t z) {
  int64_t v = 0;
  while (v < S.maxq && z >= S.t[(size_t)v + 1]) ++v;
  return v;
}

struct RefTrace {
  std::vector<std::vector<int64_t>> w_after;  // weights after each iteration
  int64_t z_min = std::numeric_limits<int64_t>::max();
  int64_t z_max = std::numeric_limits<int64_t>::min();
};

inline std::vector<int64_t> reference_train(const Params& P, const Dataset& D,
                                            std::vector<int64_t> w, RefTrace* tr = nullptr) {
  const int64_t K = int64_t(1) << P.k_shift;
  std::vector<int64_t> e((size_t)P.batch);
  for (int t = 0; t < P.iters; ++t) {
    for (int i = 0; i < P.batch; ++i) {
      int64_t z = 0;
      for (int j = 0; j < P.d; ++j) z += w[j] * D.X(t, i, j);
      if (tr) { tr->z_min = std::min(tr->z_min, z); tr->z_max = std::max(tr->z_max, z); }
      e[i] = qsigmoid_ref(z, P.f) - P.maxq * D.Y(t, i);
    }
    for (int j = 0; j < P.d; ++j) {
      int64_t g = 0;
      for (int i = 0; i < P.batch; ++i) g += e[i] * D.X(t, i, j);
      const int64_t u = K * w[j] - (g >> P.f);  // C++20: >> is floor division
      w[j] = u / K;                             // C++ '/' truncates toward zero
    }
    if (tr) tr->w_after.push_back(w);
  }
  return w;
}

}  // namespace lrgc

#endif  // LRGC_REFERENCE_H
