// lr_circuit.h -- the logistic-regression training circuit, written once over any
// emp-tool BooleanContext (see sbit.h). It computes exactly reference_train()
// from lr_reference.h, bit for bit.
//
// Per iteration and per datapoint i:  z_i = trunc_f(<w, x_i>)  ->  yhat_i = qsigmoid(z_i)
// ->  e_i = yhat_i - MAXQ*y_i;  per feature j:  g_j = trunc_f(sum_i e_i x_ij)  ->
// w_j = trunc((K w_j - g_j) / K).  trunc_f = arithmetic shift right by f (floor): every
// dot product is truncated from scale 2^(2f) back to the inputs' scale 2^f.
#ifndef LRGC_CIRCUIT_H
#define LRGC_CIRCUIT_H

#include "lr_params.h"
#include "lr_reference.h"
#include "sbit.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace lrgc {

template <class Ctx>
class LRCircuit {
 public:
  using Int = SInt<Ctx>;
  using Bit = SBit<Ctx>;
  using Wire = typename Ctx::Wire;

  LRCircuit(Builder<Ctx>& B, const Params& P, const SigmoidTable& S) : B_(B), P_(P), S_(S) {}

  // A secret quantized value (qbits two's-complement input wires).
  Int quantized(const std::vector<Wire>& bits) { return B_.from_wires(bits, P_.minq, P_.maxq); }

  // Label bit y -> label value y * MAXQ (MAXQ = 2^f - 1: every value bit equals y).
  Int label(const Bit& y) {
    Int r;
    r.b.assign((size_t)P_.f, y);
    r.b.push_back(Builder<Ctx>::cst(false));
    r.lo = 0;
    r.hi = P_.maxq;
    B_.fit(r);
    return r;
  }

  // Feature row of one datapoint: client columns + (optionally) the bias constant.
  std::vector<Int> feature_row(const std::vector<std::vector<Wire>>& client_cols) {
    std::vector<Int> row;
    row.reserve((size_t)P_.d);
    for (const auto& bits : client_cols) row.push_back(quantized(bits));
    if (P_.bias) row.push_back(B_.constant(P_.maxq));
    return row;
  }

  // sum_j a_j * b_j, exact (scale 2^(2f) for two scale-2^f vectors). Products are
  // exact and summed by a balanced adder tree.
  Int dot(const std::vector<Int>& a, const std::vector<Int>& b) {
    std::vector<Int> prods;
    prods.reserve(a.size());
    for (size_t j = 0; j < a.size(); ++j) prods.push_back(B_.mul(a[j], b[j]));
    return B_.sum(std::move(prods));
  }

  // Truncation after a dot product: scale 2^(2f) -> 2^f, floor(v / 2^f). An arithmetic
  // shift drops the low f bits (free: no gates); the integer bits are kept.
  Int truncate(const Int& v) { return B_.ashr(v, P_.f); }

  // yhat = qsigmoid(z) for z at scale 2^f (a truncated dot product), by binary
  // search over the sorted public thresholds: bit l of yhat (from the top) is
  // [z >= t_v] with v = (prefix << (l+1)) | (1 << l).
  // The threshold of the secret prefix is a public-table lookup on one-hot prefix
  // minterms, so a sigmoid costs f comparisons + (2^(f-1) - 2) ANDs for minterms.
  Int qsigmoid(const Int& z) {
    const int f = S_.f;
    std::vector<Bit> mt{Builder<Ctx>::cst(true)};  // minterms over the empty prefix
    std::vector<Bit> out((size_t)f);               // out[l] = bit l of yhat
    for (int l = f - 1; l >= 0; --l) {
      std::vector<int64_t> table(mt.size());
      for (size_t p = 0; p < mt.size(); ++p) {
        const int64_t v = ((int64_t)p << (l + 1)) | (int64_t(1) << l);
        // Clamp into [z.lo, z.hi + 1]: comparisons outside z's range are constant.
        table[p] = std::clamp<int64_t>(S_.t[(size_t)v], z.lo, z.hi + 1);
      }
      const Int T = B_.select_public(mt, table);
      out[(size_t)l] = B_.ge(z, T);
      if (l > 0) mt = B_.extend_minterms(mt, out[(size_t)l]);
    }
    Int r;
    r.b = out;
    r.b.push_back(Builder<Ctx>::cst(false));
    r.lo = 0;
    r.hi = S_.maxq;
    B_.fit(r);
    return r;
  }

  // One SGD step. x[i][j]: feature j of datapoint i (all d columns), y[i]: label bit.
  std::vector<Int> step(const std::vector<Int>& w, const std::vector<std::vector<Int>>& x,
                        const std::vector<Bit>& y) {
    const int nb = (int)x.size(), d = (int)w.size();
    std::vector<Int> e;
    e.reserve((size_t)nb);
    for (int i = 0; i < nb; ++i) {
      const Int z = truncate(dot(w, x[(size_t)i]));
      e.push_back(B_.sub(qsigmoid(z), label(y[(size_t)i])));
    }
    std::vector<Int> nw;
    nw.reserve((size_t)d);
    for (int j = 0; j < d; ++j) {
      Int g;  // sum_i e_i x_ij at scale 2^(2f)
      if (same_constant_column(x, j)) {
        g = B_.mul_const(B_.sum(e), x[0][(size_t)j].lo);  // bias column: c * sum_i e_i
      } else {
        std::vector<Int> col;
        col.reserve((size_t)nb);
        for (int i = 0; i < nb; ++i) col.push_back(x[(size_t)i][(size_t)j]);
        g = dot(e, col);
      }
      const Int u = B_.sub(B_.shl(w[(size_t)j], P_.k_shift), truncate(g));
      nw.push_back(B_.tdiv_pow2(u, P_.k_shift));
    }
    return nw;
  }

 private:
  Builder<Ctx>& B_;
  const Params& P_;
  const SigmoidTable& S_;

  static bool same_constant_column(const std::vector<std::vector<Int>>& x, int j) {
    for (const auto& row : x) {
      const Int& v = row[(size_t)j];
      if (v.lo != v.hi || v.lo != x[0][(size_t)j].lo) return false;
    }
    return true;
  }
};

// Whole training run over a non-protocol context (emp::ClearCtx to evaluate,
// emp::CountCtx to count gates). Every input bit becomes a secret wire (never
// folded), so the circuit structure is identical to the protocol run's.
template <class Ctx>
std::vector<SInt<Ctx>> train_offline(Builder<Ctx>& B, const Params& P, const SigmoidTable& S,
                                     const Dataset& D, const std::vector<int64_t>& w0) {
  using Wire = typename Ctx::Wire;
  LRCircuit<Ctx> lr(B, P, S);
  auto secret_bits = [&](int64_t v, int width) {
    std::vector<Wire> ws;
    for (bool bit : to_bits(v, width)) ws.push_back(B.ctx().public_bit(bit));
    return ws;
  };
  std::vector<SInt<Ctx>> w;
  for (int j = 0; j < P.d; ++j) w.push_back(lr.quantized(secret_bits(w0[(size_t)j], P.qbits)));
  for (int t = 0; t < P.iters; ++t) {
    std::vector<std::vector<SInt<Ctx>>> x;
    std::vector<SBit<Ctx>> y;
    for (int i = 0; i < P.batch; ++i) {
      std::vector<std::vector<Wire>> cols;
      for (int j = 0; j < P.d_data; ++j) cols.push_back(secret_bits(D.X(t, i, j), P.qbits));
      x.push_back(lr.feature_row(cols));
      y.push_back(Builder<Ctx>::wire(B.ctx().public_bit(D.Y(t, i) != 0)));
    }
    w = lr.step(w, x, y);
  }
  return w;
}

// Value of an SInt whose wires carry clear bits (emp::ClearCtx: Wire = uint8_t 0/1).
template <class Ctx>
int64_t clear_value(const SInt<Ctx>& a) {
  uint64_t v = 0;
  const int n = a.width();
  for (int i = 0; i < n; ++i) {
    const auto& bt = a.b[(size_t)i];
    const bool bit = bt.is_const() ? (bt.k == 1) : ((bt.w & 1) != 0);
    if (bit) v |= uint64_t(1) << i;
  }
  if (n < 64 && ((v >> (n - 1)) & 1)) v |= ~((uint64_t(1) << n) - 1);
  return (int64_t)v;
}

}  // namespace lrgc

#endif  // LRGC_CIRCUIT_H
