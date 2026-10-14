// lr_circuit.h -- the logistic-regression training circuit, built once per backend:
// OnlineBackend (the two parties) and DealerBackend (fake offline). Both build the
// identical gate graph, so the dealer's file order matches the parties' reads.
//
// Per iteration t, on the batch (X_t: B x d, y_t: B x 1), fixed point with F bits:
//   z  = trunc_F(X_t w)                         Pi_MultTrunc        (1 round)
//   b1 = (z + 1/2 >= 0), b2 = (z - 1/2 >= 0)    one LTZ gate on 2B   (8 rounds)
//   s  = (b1 - b2) (z + 1/2) + b2               = clamp(z + 1/2, 0, 1): the piecewise-linear
//                                                sigmoid of SecureML/ABY3 (1 round)
//   e  = s - y_t
//   w  = w - trunc_{F+k}(X_t^T e)               lr/B = 2^-k folded into the truncation (1 round)
// The client inputs w_0 and all (X_t, y_t); only the server learns the final w.
#ifndef LRMDML_CIRCUIT_H
#define LRMDML_CIRCUIT_H

#include "lr_params.h"
#include "mdml_dealer.h"
#include "mdml_ext.h"

#include <cmath>
#include <memory>
#include <vector>

namespace lrmd {

constexpr size_t kClient = 0, kServer = 1;

struct OnlineBackend {
  using Wire = GatePtr;
  OnlineCtx& ctx;
  std::vector<std::shared_ptr<InputGateU64>> inputs;  // in creation order
  std::shared_ptr<OutputToGate> output;

  Wire input(size_t r, size_t c, size_t owner) {
    auto g = std::make_shared<InputGateU64>(ctx, r, c, owner);
    inputs.push_back(g);
    return g;
  }
  Wire matmul(const Wire& x, const Wire& y, int shift, Cat cat) {
    return std::make_shared<MulGate>(ctx, x, y, MulGate::MATMUL, shift, cat);
  }
  Wire elem_mul(const Wire& x, const Wire& y) {
    return std::make_shared<MulGate>(ctx, x, y, MulGate::ELEMENTWISE, 0, C_SIGMUL);
  }
  Wire geqz(const Wire& x) { return std::make_shared<GeqZeroGate>(ctx, x); }
  Wire linear(const Wire& x, const Wire& y, size_t r, size_t c, LinFn f, std::vector<u64> k = {}) {
    return std::make_shared<LinearGate>(ctx, x, y, r, c, std::move(f), std::move(k));
  }
  Wire output_to(const Wire& x, size_t owner) {
    output = std::make_shared<OutputToGate>(ctx, x, owner);
    return output;
  }
};

struct DealerBackend {
  using Wire = FakeGatePtr;
  DealerCtx& ctx;

  Wire input(size_t r, size_t c, size_t owner) {
    ctx.st.input_masks += r * c;
    return std::make_shared<md_ml::FakeInputGate<Shr, 2>>(ctx.fp, r, c, owner);
  }
  Wire matmul(const Wire& x, const Wire& y, int shift, Cat) { return std::make_shared<FakeMatMul>(ctx, x, y, shift); }
  Wire elem_mul(const Wire& x, const Wire& y) { return std::make_shared<FakeElemMul>(ctx, x, y); }
  Wire geqz(const Wire& x) { return std::make_shared<FakeGeqZero>(ctx, x); }
  Wire linear(const Wire& x, const Wire& y, size_t r, size_t c, LinFn f, std::vector<u64> = {}) {
    return std::make_shared<FakeLinear>(x, y, r, c, std::move(f));
  }
  Wire output_to(const Wire& x, size_t owner) { return std::make_shared<FakeOutputTo>(ctx, x, owner); }
};

template <class BE>
struct LrCircuit {
  typename BE::Wire w0;
  std::vector<typename BE::Wire> X, Y;  // per iteration
  typename BE::Wire out;                 // final weights, to the server
};

template <class BE>
LrCircuit<BE> build_lr(BE& be, const Params& P) {
  const size_t B = (size_t)P.batch, d = (size_t)P.d;
  const int F = P.frac;
  const u64 half = u64(1) << (F - 1);

  // public linear maps (applied to Delta, lambda shares and MAC shares alike)
  const LinFn dup = [B](const std::vector<u128>& z, const std::vector<u128>&) {  // z -> [z; z]
    std::vector<u128> o(2 * B);
    for (size_t i = 0; i < B; ++i) o[i] = o[B + i] = z[i];
    return o;
  };
  const LinFn top_minus_bottom = [B](const std::vector<u128>& b, const std::vector<u128>&) {
    std::vector<u128> o(B);
    for (size_t i = 0; i < B; ++i) o[i] = b[i] - b[B + i];
    return o;
  };
  const LinFn ident = [](const std::vector<u128>& x, const std::vector<u128>&) { return x; };
  const LinFn plus_bottom_one = [B, F](const std::vector<u128>& m, const std::vector<u128>& b) {  // m + b2 * 2^F
    std::vector<u128> o(B);
    for (size_t i = 0; i < B; ++i) o[i] = m[i] + (b[B + i] << F);
    return o;
  };
  const LinFn sub = [](const std::vector<u128>& x, const std::vector<u128>& y) {
    std::vector<u128> o(x.size());
    for (size_t i = 0; i < x.size(); ++i) o[i] = x[i] - y[i];
    return o;
  };
  const LinFn transpose = [B, d](const std::vector<u128>& x, const std::vector<u128>&) {  // B x d -> d x B
    std::vector<u128> o(B * d);
    for (size_t i = 0; i < B; ++i)
      for (size_t j = 0; j < d; ++j) o[j * B + i] = x[i * d + j];
    return o;
  };
  std::vector<u64> pm_half(2 * B);
  for (size_t i = 0; i < B; ++i) { pm_half[i] = half; pm_half[B + i] = (u64)0 - half; }
  const std::vector<u64> plus_half(B, half);

  LrCircuit<BE> C;
  C.w0 = be.input(d, 1, kClient);
  auto w = C.w0;
  for (int t = 0; t < P.iters; ++t) {
    auto X = be.input(B, d, kClient);
    auto Y = be.input(B, 1, kClient);
    C.X.push_back(X);
    C.Y.push_back(Y);
    auto z = be.matmul(X, w, F, C_FWD);
    auto u = be.linear(z, nullptr, 2 * B, 1, dup, pm_half);  // [z + 1/2; z - 1/2]
    auto bits = be.geqz(u);
    auto db = be.linear(bits, nullptr, B, 1, top_minus_bottom);
    auto zh = be.linear(z, nullptr, B, 1, ident, plus_half);
    auto m = be.elem_mul(db, zh);
    auto s = be.linear(m, bits, B, 1, plus_bottom_one);
    auto e = be.linear(s, Y, B, 1, sub);
    auto XT = be.linear(X, nullptr, d, B, transpose);
    auto g = be.matmul(XT, e, F + P.k_shift, C_BWD);
    w = be.linear(w, g, d, 1, sub);
  }
  C.out = be.output_to(w, kServer);
  return C;
}

// ------------------------------------------------------------- client inputs
struct ClientInputs {
  std::vector<u64> w0;
  std::vector<std::vector<u64>> X, Y;
};
inline ClientInputs encode_client_inputs(const Params& P, const Dataset& D, const std::vector<int64_t>& w0q) {
  ClientInputs I;
  for (int j = 0; j < P.d; ++j) I.w0.push_back(fx_from_quant(P, w0q[j]));
  for (int t = 0; t < P.iters; ++t) {
    std::vector<u64> x, y;
    for (int i = 0; i < P.batch; ++i) {
      for (int j = 0; j < P.d; ++j) x.push_back(fx_from_quant(P, D.X(t, i, j)));
      y.push_back(fx_from_label(P, D.Y(t, i)));
    }
    I.X.push_back(std::move(x));
    I.Y.push_back(std::move(y));
  }
  return I;
}

// ------------------------------------------------------------- references
// Bit-exact emulation of the protocol's arithmetic in Z_{2^64}. The only data-dependent
// effect of the masks is the +-1 of probabilistic truncation,
//   trunc(v) = ((v + lambda) mod 2^64 >> s) - (lambda >> s),
// so given the dealer's pre-truncation masks (TEST ONLY) the result is reproduced exactly.
inline std::vector<u64> emulate_protocol(const Params& P, const ClientInputs& I,
                                         const std::vector<std::vector<u64>>& trunc_lambdas) {
  const size_t B = (size_t)P.batch, d = (size_t)P.d;
  const int F = P.frac;
  const u64 half = u64(1) << (F - 1);
  if (trunc_lambdas.size() != 2 * (size_t)P.iters) throw std::runtime_error("verify data: wrong number of truncations");
  size_t ti = 0;
  auto trunc = [&](std::vector<u64> v, int s) {
    const auto& lam = trunc_lambdas[ti++];
    if (lam.size() != v.size()) throw std::runtime_error("verify data: truncation size mismatch");
    for (size_t i = 0; i < v.size(); ++i) v[i] = ((v[i] + lam[i]) >> s) - (lam[i] >> s);
    return v;
  };
  std::vector<u64> w = I.w0;
  for (int t = 0; t < P.iters; ++t) {
    const auto& X = I.X[t];
    const auto& Y = I.Y[t];
    std::vector<u64> zp(B, 0);
    for (size_t i = 0; i < B; ++i)
      for (size_t j = 0; j < d; ++j) zp[i] += X[i * d + j] * w[j];
    auto z = trunc(zp, F);
    std::vector<u64> e(B);
    for (size_t i = 0; i < B; ++i) {
      const u64 b1 = (int64_t)(z[i] + half) >= 0, b2 = (int64_t)(z[i] - half) >= 0;
      const u64 s = (b1 - b2) * (z[i] + half) + (b2 << F);
      e[i] = s - Y[i];
    }
    std::vector<u64> gp(d, 0);
    for (size_t j = 0; j < d; ++j)
      for (size_t i = 0; i < B; ++i) gp[j] += X[i * d + j] * e[i];
    auto g = trunc(gp, F + P.k_shift);
    for (size_t j = 0; j < d; ++j) w[j] -= g[j];
  }
  return w;
}

// The same training in double precision (what the fixed-point circuit approximates).
inline std::vector<double> reference_double(const Params& P, const Dataset& D, const std::vector<int64_t>& w0q) {
  const double qs = std::ldexp(1.0, -P.f), step = P.lr / P.batch;
  std::vector<double> w(P.d);
  for (int j = 0; j < P.d; ++j) w[j] = w0q[j] * qs;
  std::vector<double> e(P.batch);
  for (int t = 0; t < P.iters; ++t) {
    for (int i = 0; i < P.batch; ++i) {
      double z = 0;
      for (int j = 0; j < P.d; ++j) z += D.X(t, i, j) * qs * w[j];
      e[i] = std::min(1.0, std::max(0.0, z + 0.5)) - D.Y(t, i);
    }
    for (int j = 0; j < P.d; ++j) {
      double g = 0;
      for (int i = 0; i < P.batch; ++i) g += e[i] * D.X(t, i, j) * qs;
      w[j] -= step * g;
    }
  }
  return w;
}

}  // namespace lrmd

#endif  // LRMDML_CIRCUIT_H
