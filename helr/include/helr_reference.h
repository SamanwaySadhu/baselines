// helr_reference.h -- plaintext specification of what the HE circuit computes, in
// double precision, and the least-squares sigmoid polynomial it uses.
//
// HELR (Kim et al., JMIR Med Inform 2018) trains logistic regression by gradient descent
// on z_i = (2 y_i - 1) * x_i (label sign folded into the features), with the sigmoid
// replaced by a least-squares polynomial. One iteration t on its batch of B samples:
//
//   w_i  = z_i^T beta / R                         (R = 2^logR, HELR: 8; the data is encrypted as z / R)
//   H(w) = c_1 w + c_3 w^3 [+ c_5 w^5 + c_7 w^7]  (= 2 h(w), h the odd least-squares polynomial of
//                                                  sigmoid(R w) - 1/2 on [-1, 1])
//   g_i  = (1 - H(w_i)) / 2                       (~ sigmoid(-z_i^T beta); HELR's "g(-x)")
//   beta <- beta + (1/K) sum_{i in batch} g_i z_i,   K = batch / lr = 2^k_shift
//
// i.e. HELR's update beta <- beta + (alpha/n) sum_i g(-z_i^T beta) z_i with the batch size
// as n, learning rate alpha = lr, and a fresh batch each iteration (the paper uses the
// whole training set every iteration). With R = 8 the polynomial is the one of the paper
// (Fig. 3): sigmoid(x) ~ 1/2 + a1 (x/8) + a3 (x/8)^3 with a1 = 1.20096, a3 = -0.81562 (degree 3).
//
// The quantized inputs are the other baselines': features x_ij and weights w_j are integers
// in [MINQ, MAXQ] at scale 2^f, labels y in {0, 1}. Unlike the fixed-point baselines this
// is approximate arithmetic on reals (no truncations, polynomial sigmoid), so the trained
// weights are close to, not equal to, theirs.
#ifndef LRHE_HELR_REFERENCE_H
#define LRHE_HELR_REFERENCE_H

#include "lr_params.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace lrhe {

// ------------------------------------------------------------------ polynomial
// H(w) = sum_k c[k] w^k (k odd, <= deg), the least-squares fit 2 h(w) of
// 2 (sigmoid(R w) - 1/2) = tanh(R w / 2) on [-1, 1] by an odd polynomial of degree `deg`
// (minimizes the mean square error, as in the paper). Computed from the normal equations in
// long double; both parties compute it independently.
struct HPoly {
  int deg = 3;
  double R = 8;
  double c[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  double eval(double w) const {
    double w2 = w * w, acc = 0, pw = w;
    for (int k = 1; k <= deg; k += 2) { acc += c[k] * pw; pw *= w2; }
    return acc;
  }
};

inline HPoly fit_sigmoid_poly(int deg, int logR) {
  if (deg != 3 && deg != 7) throw std::invalid_argument("deg must be 3 or 7");
  HPoly H;
  H.deg = deg;
  H.R = std::ldexp(1.0, logR);
  const int m = (deg + 1) / 2;  // unknowns: coefficients of w, w^3, ..., w^deg
  // b_k = int_{-1}^{1} w^k tanh(R w / 2) dw (composite Simpson on 400000 panels); the Gram
  // matrix G_kl = int w^(k+l) dw = 2 / (k+l+1).
  const long n = 400000;
  const long double h = 2.0L / n;
  std::vector<long double> b(m, 0.0L);
  for (long i = 0; i <= n; ++i) {
    const long double w = -1.0L + i * h;
    const long double f = std::tanh((long double)H.R * w / 2.0L);
    const long double wgt = (i == 0 || i == n) ? 1.0L : (i % 2 ? 4.0L : 2.0L);
    long double pw = w;
    for (int a = 0; a < m; ++a) {
      b[a] += wgt * pw * f;
      pw *= w * w;
    }
  }
  for (int a = 0; a < m; ++a) b[a] *= h / 3.0L;
  std::vector<std::vector<long double>> G(m, std::vector<long double>(m + 1));
  for (int a = 0; a < m; ++a) {
    for (int c = 0; c < m; ++c) G[a][c] = 2.0L / ((2 * a + 1) + (2 * c + 1) + 1);
    G[a][m] = b[a];
  }
  for (int col = 0; col < m; ++col) {  // Gauss-Jordan with partial pivoting
    int piv = col;
    for (int r = col + 1; r < m; ++r)
      if (std::fabs(G[r][col]) > std::fabs(G[piv][col])) piv = r;
    std::swap(G[col], G[piv]);
    for (int r = 0; r < m; ++r) {
      if (r == col) continue;
      const long double f = G[r][col] / G[col][col];
      for (int c2 = col; c2 <= m; ++c2) G[r][c2] -= f * G[col][c2];
    }
  }
  for (int a = 0; a < m; ++a) H.c[2 * a + 1] = (double)(G[a][m] / G[a][a]);
  return H;
}

// ------------------------------------------------------------------- training
// z_ij = (2 y_i - 1) x_ij / 2^f, the features of the HE circuit (before the 1/R scaling).
inline double z_value(const Params& P, const Dataset& D, int t, int i, int j) {
  const double sign = D.Y(t, i) ? 1.0 : -1.0;
  return sign * (double)D.X(t, i, j) / std::ldexp(1.0, P.f);
}

// The circuit's function in double precision. `trace`, if given, receives max |z^T beta| / R
// over all samples and iterations (> 1 means the polynomial was evaluated outside its fit
// interval).
inline std::vector<double> poly_train(const Params& P, const Dataset& D, const std::vector<int64_t>& w0q,
                                      const HPoly& H, double* max_w = nullptr) {
  std::vector<double> beta(P.d);
  for (int j = 0; j < P.d; ++j) beta[j] = (double)w0q[j] / std::ldexp(1.0, P.f);
  const double invK = std::ldexp(1.0, -P.k_shift);
  double mx = 0;
  for (int t = 0; t < P.iters; ++t) {
    std::vector<double> grad(P.d, 0.0);
    for (int i = 0; i < P.batch; ++i) {
      double x = 0;
      for (int j = 0; j < P.d; ++j) x += z_value(P, D, t, i, j) * beta[j];
      const double w = x / H.R;
      mx = std::max(mx, std::fabs(w));
      const double g = 0.5 * (1.0 - H.eval(w));
      for (int j = 0; j < P.d; ++j) grad[j] += g * z_value(P, D, t, i, j);
    }
    for (int j = 0; j < P.d; ++j) beta[j] += invK * grad[j];
  }
  if (max_w) *max_w = mx;
  return beta;
}

// The same training with the exact sigmoid (what the polynomial approximates).
inline std::vector<double> sigmoid_train(const Params& P, const Dataset& D, const std::vector<int64_t>& w0q) {
  std::vector<double> beta(P.d);
  for (int j = 0; j < P.d; ++j) beta[j] = (double)w0q[j] / std::ldexp(1.0, P.f);
  const double invK = std::ldexp(1.0, -P.k_shift);
  for (int t = 0; t < P.iters; ++t) {
    std::vector<double> grad(P.d, 0.0);
    for (int i = 0; i < P.batch; ++i) {
      double x = 0;
      for (int j = 0; j < P.d; ++j) x += z_value(P, D, t, i, j) * beta[j];
      const double g = 1.0 / (1.0 + std::exp(x));  // sigmoid(-x)
      for (int j = 0; j < P.d; ++j) grad[j] += g * z_value(P, D, t, i, j);
    }
    for (int j = 0; j < P.d; ++j) beta[j] += invK * grad[j];
  }
  return beta;
}

// Largest |weight| the honest protocol can produce, rounded up to a power of two: each
// iteration moves a weight by at most (B / K) * max|g| * max|z| and |g| <= ~ 2 on the fit
// interval; the server rejects any output beyond this.
inline double weight_bound(const Params& P) {
  const double bound = 2.0 + (double)P.iters * P.batch * std::ldexp(1.0, -P.k_shift) * 2.0 * 2.0;
  double b = 2.0;
  while (b < bound) b *= 2.0;
  return b;
}

// -log2 of the probability that a malicious client forges the decryption MAC of one weight
// (helr_protocol.h): it must shift u1 by delta with |delta| <= 2 * weight_bound * 2^logp, and
// succeeds with probability <= 2^(v + 1 - L) for delta with 2-adic valuation v, L = out_bits.
inline double mac_forgery_bits(int out_bits, int logp, const Params& P) {
  return (double)out_bits - ((double)logp + std::log2(weight_bound(P)) + 2.0);
}

}  // namespace lrhe

#endif  // LRHE_HELR_REFERENCE_H
