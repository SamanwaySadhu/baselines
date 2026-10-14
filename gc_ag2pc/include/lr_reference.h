// lr_reference.h -- plaintext specification of the float32 logistic-regression
// training that the garbled circuit computes, bit for bit.
//
// One iteration t on its batch {(x_i, y_i)}, i < 4, all binary32 (IEEE round-to-nearest-even,
// NO fused multiply-add: this file must be compiled with -ffp-contract=off, which
// CMakeLists.txt sets, because the circuit's multiply-add is an unfused mul + add):
//   z_i    = ((w_0 x_i0 + w_1 x_i1) + ...) + w_bias        (bias feature = 1.0f)
//   yhat_i = sigmoid(z_i) = 1 / (1 + exp(-z_i))
//   e_i    = yhat_i - y_i
//   g_j    = ((e_0 x_0j + e_1 x_1j) + ...)   (g_bias = ((e_0 + e_1) + e_2) + e_3)
//   w_j   <- w_j + g_j * (-lr/batch)                         lr = 0.01, batch = 4
//
// This is exact logistic regression: the sigmoid is the true function, with exp()
// evaluated by Cody-Waite range reduction (exp(-a) = 2^-n * exp(-r), |r| <= ln2/2)
// and a degree-7 Taylor polynomial (relative error ~5e-9 < one float32 ulp), not a
// piecewise-linear / low-degree approximation of the sigmoid. sigmoid is evaluated
// on |z| (t = exp(-|z|) lies in (0, 1], so nothing overflows) and |z| is saturated at
// 87, where exp(-87) = 1.6e-38 is still a normal float32; the sigmoid saturates
// to 1 (z > 0) and to <= 1.6e-38 (z < 0), which differs from the true value by less
// than 1.6e-38.
#ifndef LRGC_REFERENCE_H
#define LRGC_REFERENCE_H

#include "lr_params.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace lrgc {

// Constants shared with the circuit.
constexpr float kLog2e = 1.44269504088896340736f;
constexpr float kLn2Hi = 6.9313812256e-01f;   // fdlibm: low 7 bits zero -> n * kLn2Hi exact for n < 128
constexpr float kLn2Lo = 9.0580006145e-06f;
constexpr float kMagic = 12582912.0f;         // 1.5 * 2^23: adding it rounds to the nearest integer
constexpr float kExpClamp = 87.0f;
constexpr int kPolyDegree = 7;
// 1/k! rounded to float32, k = 0..7
constexpr float kInvFact[kPolyDegree + 1] = {1.0f, 1.0f, 0.5f, 0.16666666666666666f, 0.041666666666666664f,
                                             0.008333333333333333f, 0.001388888888888889f,
                                             0.0001984126984126984f};

// sigmoid(z) in float32, operation for operation what the circuit does.
inline float sigmoid_f32(float z) {
  const bool neg = (f2u(z) >> 31) != 0;
  const float a = std::min(u2f(f2u(z) & 0x7FFFFFFFu), kExpClamp);
  const float u = a * kLog2e;
  const float r1 = u + kMagic;
  const float nf = r1 - kMagic;
  const uint32_t n = f2u(r1) & 0x7Fu;  // round(u) in [0, 126]
  float r = a - nf * kLn2Hi;
  r = r - nf * kLn2Lo;
  const float s = -r;
  float p = kInvFact[kPolyDegree];
  for (int k = kPolyDegree - 1; k >= 0; --k) p = p * s + kInvFact[k];  // unfused: mul, then add
  const float pw = u2f((127u - n) << 23);                               // 2^-n
  const float t = p * pw;                                               // exp(-a)
  const float den = t + 1.0f;
  const float num = neg ? t : 1.0f;
  return num / den;
}

// The true sigmoid in long double, for accuracy checks.
inline long double sigmoid_ld(long double x) { return 1.0L / (1.0L + std::exp(-x)); }

struct RefTrace {
  std::vector<std::vector<float>> w_after;  // weights after each iteration
};

inline std::vector<float> reference_train(const Params& P, const Dataset& D, std::vector<float> w,
                                          RefTrace* tr = nullptr) {
  const int dd = P.d_data();
  for (int t = 0; t < P.iters; ++t) {
    float e[kBatch];
    for (int i = 0; i < kBatch; ++i) {
      float z = w[0] * D.X(t, i, 0);
      for (int j = 1; j < dd; ++j) z = w[j] * D.X(t, i, j) + z;
      z = z + w[dd];
      e[i] = sigmoid_f32(z) - (D.Y(t, i) ? 1.0f : 0.0f);
    }
    for (int j = 0; j < P.d; ++j) {
      float g;
      if (j < dd) {
        g = e[0] * D.X(t, 0, j);
        for (int i = 1; i < kBatch; ++i) g = e[i] * D.X(t, i, j) + g;
      } else {
        g = e[0] + e[1];
        for (int i = 2; i < kBatch; ++i) g = g + e[i];
      }
      w[j] = g * (-kLrOverBatch) + w[j];
    }
    if (tr) tr->w_after.push_back(w);
  }
  return w;
}

// The same training in double precision with the true exp (for reporting how far the
// float32 result is from infinitely-precise logistic regression).
inline std::vector<double> reference_train_double(const Params& P, const Dataset& D, std::vector<double> w) {
  for (int t = 0; t < P.iters; ++t) {
    double e[kBatch];
    for (int i = 0; i < kBatch; ++i) {
      long double z = 0;
      for (int j = 0; j < P.d; ++j) z += (long double)w[j] * D.X(t, i, j);
      e[i] = (double)sigmoid_ld(z) - (D.Y(t, i) ? 1.0 : 0.0);
    }
    for (int j = 0; j < P.d; ++j) {
      double g = 0;
      for (int i = 0; i < kBatch; ++i) g += e[i] * D.X(t, i, j);
      w[j] -= (double)kLrOverBatch * g;
    }
  }
  return w;
}

}  // namespace lrgc

#endif  // LRGC_REFERENCE_H
