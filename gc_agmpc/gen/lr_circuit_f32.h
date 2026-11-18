// lr_circuit.h -- the float32 logistic-regression training circuit, written once over
// any emp-tool BooleanContext (AG2PCSession::ctx_t for the real protocol,
// emp::ClearCtx to evaluate, emp::CountCtx to count gates). It computes exactly
// reference_train() from lr_reference.h, bit for bit (see there for the formulas).
//
// Building blocks are emp-tool's verified IEEE binary32 circuits (add, sub, mul, div,
// min: bit-exact against host floats). Note emp-tool's Float_T::fma is UNFUSED
// (mul then add, two roundings); the reference does the same.
#ifndef LRGC_CIRCUIT_H
#define LRGC_CIRCUIT_H

#include "lr_params.h"
#include "lr_reference.h"

#include "emp-tool/emp-tool.h"

#include <cstdint>
#include <vector>

namespace lrgc {

template <class Ctx>
class FloatLR {
 public:
  using Fl = emp::Float_T<Ctx, 32>;
  using Wire = typename Ctx::Wire;

  FloatLR(Ctx& c, int d) : c_(c), d_(d) {}

  Fl cst(float v) { return Fl::constant(c_, v); }

  // Label bit -> 0.0f / 1.0f. The IEEE pattern of 1.0f is 0x3F800000: bits 23..29 set.
  Fl label(const Wire& y) {
    const Wire zero = c_.public_bit(false);
    Fl r(c_);
    for (int i = 0; i < 32; ++i) r.w[i] = (i >= 23 && i <= 29) ? y : zero;
    return r;
  }

  // sigmoid(z) = 1 / (1 + exp(-z)), exactly as sigmoid_f32().
  Fl sigmoid(const Fl& z) {
    const Fl a = z.abs().min(cst(kExpClamp));
    const Fl u = a * cst(kLog2e);
    const Fl r1 = u + cst(kMagic);   // low 7 mantissa bits of r1 = round(u)
    const Fl nf = r1 - cst(kMagic);
    Fl r = a - nf * cst(kLn2Hi);
    r = r - nf * cst(kLn2Lo);
    const Fl s = -r;
    Fl p = cst(kInvFact[kPolyDegree]);
    for (int k = kPolyDegree - 1; k >= 0; --k) p = p.fma(s, cst(kInvFact[k]));
    // 2^-n: sign 0, exponent field 127 - n = NOT(n) on 7 bits, mantissa 0 (free).
    Fl pw(c_);
    const Wire zero = c_.public_bit(false);
    for (int i = 0; i < 32; ++i) pw.w[i] = zero;
    for (int i = 0; i < 7; ++i) pw.w[23 + i] = c_.not_gate(r1.w[i]);
    const Fl t = p * pw;             // exp(-|z|) in (0, 1]
    const Fl den = t + cst(1.0f);
    const Fl num = cst(1.0f).select(emp::Bit_T<Ctx>(c_, z.w[31]), t);  // z < 0 ? t : 1
    return num / den;
  }

  // One SGD step. x[i][j]: feature j < d-1 of datapoint i (the bias column is the
  // implicit constant 1.0f), y[i]: label as a float.
  std::vector<Fl> step(const std::vector<Fl>& w, const std::vector<std::vector<Fl>>& x, const std::vector<Fl>& y) {
    const int dd = d_ - 1;
    std::vector<Fl> e;
    e.reserve(kBatch);
    for (int i = 0; i < kBatch; ++i) {
      Fl z = w[0] * x[i][0];
      for (int j = 1; j < dd; ++j) z = w[j].fma(x[i][j], z);
      z = z + w[dd];
      e.push_back(sigmoid(z) - y[i]);
    }
    const Fl neg_c = cst(-kLrOverBatch);
    std::vector<Fl> nw;
    nw.reserve(d_);
    for (int j = 0; j < d_; ++j) {
      Fl g = (j < dd) ? e[0] * x[0][j] : e[0] + e[1];
      for (int i = (j < dd) ? 1 : 2; i < kBatch; ++i) g = (j < dd) ? e[i].fma(x[i][j], g) : g + e[i];
      nw.push_back(g.fma(neg_c, w[j]));
    }
    return nw;
  }

 private:
  Ctx& c_;
  int d_;
};

// Float value of an Fl whose wires carry clear bits (emp::ClearCtx: Wire = uint8_t 0/1).
template <class Fl>
float clear_float(const Fl& f) {
  uint32_t u = 0;
  for (int i = 0; i < 32; ++i) u |= (uint32_t)(f.w[i] & 1) << i;
  return u2f(u);
}

// Whole run (all SGD iterations from the initial weights w0) over a non-protocol
// context. Every input bit is a plain wire (never folded), so the structure matches
// the protocol's circuit. Returns the trained weights.
template <class Ctx>
std::vector<emp::Float_T<Ctx, 32>> train_offline(Ctx& ctx, const Params& P, const Dataset& D, const std::vector<float>& w0) {
  using Fl = emp::Float_T<Ctx, 32>;
  FloatLR<Ctx> lr(ctx, P.d);
  std::vector<Fl> w;
  for (int j = 0; j < P.d; ++j) w.push_back(Fl::constant(ctx, w0[j]));
  for (int t = 0; t < P.iters; ++t) {
    std::vector<std::vector<Fl>> x;
    std::vector<Fl> y;
    for (int i = 0; i < kBatch; ++i) {
      std::vector<Fl> row;
      for (int j = 0; j < P.d_data(); ++j) row.push_back(Fl::constant(ctx, D.X(t, i, j)));
      x.push_back(row);
      y.push_back(lr.label(ctx.public_bit(D.Y(t, i) != 0)));
    }
    w = lr.step(w, x, y);
  }
  return w;
}

}  // namespace lrgc

#endif  // LRGC_CIRCUIT_H
