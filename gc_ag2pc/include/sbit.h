// sbit.h -- constant-folding bits and range-tracked two's-complement integers,
// written once over any emp-tool BooleanContext (AG2PCSession::ctx_t for the real
// protocol; emp::ClearCtx / emp::CountCtx for testing and gate counting).
//
// Every integer carries a sound value interval [lo, hi]. Each operation sizes its
// result to the minimal two's-complement width that holds the exact result, so the
// circuit never overflows and never pays AND gates for unused high bits. Bits that
// are public constants are folded here and never reach the gate context.
//
// Only the context's four BooleanContext primitives are used:
//   public_bit(bool), and_gate(a, b), xor_gate(a, b), not_gate(a).
#ifndef LRGC_SBIT_H
#define LRGC_SBIT_H

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace lrgc {

// Gates actually emitted to the context (after constant folding).
struct GateStats {
  uint64_t ands = 0, xors = 0, nots = 0, consts = 0;
};

template <class Ctx>
struct SBit {
  using Wire = typename Ctx::Wire;
  int8_t k = 0;  // 0 / 1: public constant; -1: secret wire `w`
  Wire w{};
  bool is_const() const { return k >= 0; }
};

// Two's-complement integer, LSB first, whose value is guaranteed to lie in [lo, hi].
template <class Ctx>
struct SInt {
  std::vector<SBit<Ctx>> b;
  int64_t lo = 0, hi = 0;
  int width() const { return (int)b.size(); }
};

// Smallest two's-complement width (>= 1) holding every value of [lo, hi].
inline int width_for(int64_t lo, int64_t hi) {
  if (lo > hi) throw std::logic_error("width_for: empty range");
  for (int w = 1; w <= 62; ++w) {
    const int64_t mn = -(int64_t(1) << (w - 1)), mx = (int64_t(1) << (w - 1)) - 1;
    if (lo >= mn && hi <= mx) return w;
  }
  throw std::overflow_error("width_for: value range exceeds 62 bits");
}

inline int64_t checked_add(int64_t a, int64_t b) {
  int64_t r;
  if (__builtin_add_overflow(a, b, &r)) throw std::overflow_error("range overflow (add)");
  return r;
}
inline int64_t checked_sub(int64_t a, int64_t b) {
  int64_t r;
  if (__builtin_sub_overflow(a, b, &r)) throw std::overflow_error("range overflow (sub)");
  return r;
}
inline int64_t checked_mul(int64_t a, int64_t b) {
  int64_t r;
  if (__builtin_mul_overflow(a, b, &r)) throw std::overflow_error("range overflow (mul)");
  return r;
}

template <class Ctx>
class Builder {
 public:
  using Bit = SBit<Ctx>;
  using Int = SInt<Ctx>;
  using Wire = typename Ctx::Wire;

  explicit Builder(Ctx& ctx) : ctx_(ctx) {}
  Ctx& ctx() { return ctx_; }
  GateStats stats;

  // ---------------------------------------------------------------- bits
  static Bit cst(bool v) { Bit b; b.k = v ? 1 : 0; return b; }
  static Bit wire(Wire w) { Bit b; b.k = -1; b.w = w; return b; }

  Bit NOT(const Bit& a) {
    if (a.is_const()) return cst(a.k == 0);
    ++stats.nots;
    return wire(ctx_.not_gate(a.w));
  }
  Bit XOR(const Bit& a, const Bit& b) {
    if (a.is_const() && b.is_const()) return cst(a.k != b.k);
    if (a.is_const()) return a.k ? NOT(b) : b;
    if (b.is_const()) return b.k ? NOT(a) : a;
    ++stats.xors;
    return wire(ctx_.xor_gate(a.w, b.w));
  }
  Bit AND(const Bit& a, const Bit& b) {
    if (a.is_const()) return a.k ? b : cst(false);
    if (b.is_const()) return b.k ? a : cst(false);
    ++stats.ands;
    return wire(ctx_.and_gate(a.w, b.w));
  }
  Bit OR(const Bit& a, const Bit& b) {  // a ^ b ^ (a & b): 1 AND
    if (a.is_const()) return a.k ? cst(true) : b;
    if (b.is_const()) return b.k ? cst(true) : a;
    return XOR(XOR(a, b), AND(a, b));
  }
  Bit MUX(const Bit& sel, const Bit& t, const Bit& f) {  // sel ? t : f, 1 AND
    if (sel.is_const()) return sel.k ? t : f;
    if (t.is_const() && f.is_const() && t.k == f.k) return t;
    return XOR(f, AND(sel, XOR(t, f)));
  }
  // Majority (full-adder carry), 1 AND unless folded.
  Bit MAJ(const Bit& a, const Bit& b, const Bit& c) {
    if (a.is_const() && b.is_const()) return a.k == b.k ? a : c;
    if (a.is_const() && c.is_const()) return a.k == c.k ? a : b;
    if (b.is_const() && c.is_const()) return b.k == c.k ? b : a;
    if (a.is_const()) return a.k ? OR(b, c) : AND(b, c);
    if (b.is_const()) return b.k ? OR(a, c) : AND(a, c);
    if (c.is_const()) return c.k ? OR(a, b) : AND(a, b);
    return XOR(c, AND(XOR(a, c), XOR(b, c)));
  }
  // A real wire for I/O / keep lists (materializes a public constant if needed).
  Wire materialize(const Bit& b) {
    if (!b.is_const()) return b.w;
    ++stats.consts;
    return ctx_.public_bit(b.k == 1);
  }

  // ------------------------------------------------------------ integers
  Int constant(int64_t v) {
    Int r;
    r.lo = r.hi = v;
    const int w = width_for(v, v);
    for (int i = 0; i < w; ++i) r.b.push_back(cst(((uint64_t)v >> i) & 1));
    return r;
  }

  // Secret two's-complement bits (LSB first) whose value is known to lie in [lo, hi].
  Int from_wires(const std::vector<Wire>& ws, int64_t lo, int64_t hi) {
    const int n = (int)ws.size();
    if (n < 1 || n > 62) throw std::invalid_argument("from_wires: width must be in [1, 62]");
    if (lo < -(int64_t(1) << (n - 1)) || hi > (int64_t(1) << (n - 1)) - 1)
      throw std::invalid_argument("from_wires: range not representable in the given width");
    Int r;
    for (const Wire& w : ws) r.b.push_back(wire(w));
    r.lo = lo;
    r.hi = hi;
    fit(r);
    return r;
  }

  // Bits of `a` at width w: sign-extension, or truncation (only used where the
  // arithmetic is mod 2^w and the exact result is known to fit in w bits).
  std::vector<Bit> bits_at(const Int& a, int w) const {
    std::vector<Bit> r(a.b.begin(), a.b.begin() + std::min(w, a.width()));
    while ((int)r.size() < w) r.push_back(a.b.back());
    return r;
  }

  // Restore the invariant: width == width_for(lo, hi); a known sign is a constant;
  // a single-valued range is a constant.
  void fit(Int& a) {
    if (a.lo == a.hi) { a = constant(a.lo); return; }
    const int w = width_for(a.lo, a.hi);
    if (a.width() != w) a.b = bits_at(a, w);
    if (a.lo >= 0) a.b.back() = cst(false);
    else if (a.hi < 0) a.b.back() = cst(true);
  }

  Int add(const Int& a, const Int& b) {
    Int r;
    r.lo = checked_add(a.lo, b.lo);
    r.hi = checked_add(a.hi, b.hi);
    const int w = width_for(r.lo, r.hi);
    r.b = add_bits(bits_at(a, w), bits_at(b, w), cst(false));
    fit(r);
    return r;
  }

  Int sub(const Int& a, const Int& b) {
    Int r;
    r.lo = checked_sub(a.lo, b.hi);
    r.hi = checked_sub(a.hi, b.lo);
    const int w = width_for(r.lo, r.hi);
    std::vector<Bit> nb = bits_at(b, w);
    for (Bit& x : nb) x = NOT(x);
    r.b = add_bits(bits_at(a, w), nb, cst(true));
    fit(r);
    return r;
  }

  // Balanced adder tree (narrower intermediate sums than a linear chain).
  Int sum(std::vector<Int> v) {
    if (v.empty()) return constant(0);
    while (v.size() > 1) {
      std::vector<Int> nx;
      nx.reserve((v.size() + 1) / 2);
      for (size_t i = 0; i + 1 < v.size(); i += 2) nx.push_back(add(v[i], v[i + 1]));
      if (v.size() % 2) nx.push_back(std::move(v.back()));
      v.swap(nx);
    }
    return v[0];
  }

  Int shl(const Int& a, int s) {  // a * 2^s
    if (s < 0 || s > 61) throw std::invalid_argument("shl: bad shift");
    Int r;
    r.b.assign(s, cst(false));
    r.b.insert(r.b.end(), a.b.begin(), a.b.end());
    r.lo = checked_mul(a.lo, int64_t(1) << s);
    r.hi = checked_mul(a.hi, int64_t(1) << s);
    fit(r);
    return r;
  }

  Int ashr(const Int& a, int s) {  // floor(a / 2^s): an arithmetic shift is free
    if (s < 0 || s > 61) throw std::invalid_argument("ashr: bad shift");
    Int r;
    if (s >= a.width()) r.b = {a.b.back()};
    else r.b.assign(a.b.begin() + s, a.b.end());
    r.lo = a.lo >> s;  // C++20: >> on signed values rounds toward -infinity
    r.hi = a.hi >> s;
    fit(r);
    return r;
  }

  // trunc(a / 2^s), i.e. rounding toward zero (C's integer division / (int) cast).
  Int tdiv_pow2(const Int& a, int s) {
    if (s < 0 || s > 61) throw std::invalid_argument("tdiv_pow2: bad shift");
    if (s == 0 || a.lo >= 0) return ashr(a, s);
    // Negative values: (a + 2^s - 1) >> s. The bias 2^s - 1 is added only when a < 0,
    // so its low s bits are copies of a's sign bit.
    Int bias;
    bias.b.assign(s, a.b.back());
    bias.b.push_back(cst(false));
    bias.lo = 0;
    bias.hi = (int64_t(1) << s) - 1;
    fit(bias);
    Int r = ashr(add(a, bias), s);
    r.lo = a.lo / (int64_t(1) << s);  // C++ '/' truncates toward zero; trunc is monotone
    r.hi = a.hi / (int64_t(1) << s);
    fit(r);
    return r;
  }

  // a * c for a public constant c, via canonical-signed-digit recoding (one
  // add/sub per non-zero CSD digit).
  Int mul_const(const Int& a, int64_t c) {
    if (c == 0) return constant(0);
    if (a.lo == a.hi) return constant(checked_mul(a.lo, c));
    const int64_t p1 = checked_mul(a.lo, c), p2 = checked_mul(a.hi, c);
    Int r;
    r.lo = std::min(p1, p2);
    r.hi = std::max(p1, p2);
    const int w = width_for(r.lo, r.hi);
    const std::vector<Bit> X = bits_at(a, w);
    std::vector<Bit> acc(w, cst(false));
    int64_t v = c;
    for (int k = 0; v != 0 && k < w; ++k) {
      if (v & 1) {
        const int digit = 2 - (int)(v & 3);  // +1 if v = 1 (mod 4), -1 if v = 3 (mod 4)
        accumulate(acc, X, k, digit < 0);
        v -= digit;
      }
      v >>= 1;
    }
    r.b = acc;
    fit(r);
    return r;
  }

  // Signed a * b. Schoolbook over the operand with fewer secret bits; all partial
  // sums are mod 2^w where w is the exact product width.
  Int mul(const Int& a, const Int& b) {
    if (a.lo == a.hi) return mul_const(b, a.lo);
    if (b.lo == b.hi) return mul_const(a, b.lo);
    const int64_t c[4] = {checked_mul(a.lo, b.lo), checked_mul(a.lo, b.hi),
                          checked_mul(a.hi, b.lo), checked_mul(a.hi, b.hi)};
    Int r;
    r.lo = *std::min_element(c, c + 4);
    r.hi = *std::max_element(c, c + 4);
    const int w = width_for(r.lo, r.hi);
    const int sa = secret_bits(a), sb = secret_bits(b);
    const bool a_is_m = sa < sb || (sa == sb && a.width() <= b.width());
    const Int& m = a_is_m ? a : b;  // multiplier: one row per bit
    const Int& x = a_is_m ? b : a;  // multiplicand
    const std::vector<Bit> X = bits_at(x, w);
    std::vector<Bit> acc(w, cst(false));
    const int wm = m.width();
    for (int i = 0; i < wm && i < w; ++i) {
      const Bit& mi = m.b[i];
      if (mi.is_const() && mi.k == 0) continue;
      std::vector<Bit> row(w - i);
      // Bits of X at positions >= x.width()-1 are all x's sign bit: AND it once.
      Bit sign_and;
      bool have_sign_and = false;
      for (int j = 0; j < w - i; ++j) {
        if (j >= x.width() - 1) {
          if (!have_sign_and) { sign_and = AND(X[j], mi); have_sign_and = true; }
          row[j] = sign_and;
        } else {
          row[j] = AND(X[j], mi);
        }
      }
      // m's top bit weighs -2^(wm-1) (two's complement), unless m was truncated.
      const bool negative = (i == wm - 1) && (wm <= w);
      accumulate(acc, row, i, negative);
    }
    r.b = acc;
    fit(r);
    return r;
  }

  // a < b (signed). Only the sign of the exact difference is computed.
  Bit lt(const Int& a, const Int& b) {
    if (a.hi < b.lo) return cst(true);
    if (a.lo >= b.hi) return cst(false);
    const int w = width_for(checked_sub(a.lo, b.hi), checked_sub(a.hi, b.lo));
    const std::vector<Bit> A = bits_at(a, w);
    std::vector<Bit> NB = bits_at(b, w);
    for (Bit& x : NB) x = NOT(x);
    Bit c = cst(true);
    for (int i = 0; i + 1 < w; ++i) c = MAJ(A[i], NB[i], c);
    return XOR(XOR(A[w - 1], NB[w - 1]), c);
  }
  Bit ge(const Int& a, const Int& b) { return NOT(lt(a, b)); }

  Int mux(const Bit& sel, const Int& t, const Int& f) {  // sel ? t : f
    Int r;
    r.lo = std::min(t.lo, f.lo);
    r.hi = std::max(t.hi, f.hi);
    const int w = width_for(r.lo, r.hi);
    const std::vector<Bit> T = bits_at(t, w), F = bits_at(f, w);
    r.b.resize(w);
    for (int i = 0; i < w; ++i) r.b[i] = MUX(sel, T[i], F[i]);
    fit(r);
    return r;
  }

  // One-hot minterms: given minterms over prefix bits p (index p), extend with a
  // new least-significant bit nb: index 2p + nb. Costs one AND per old minterm.
  std::vector<Bit> extend_minterms(const std::vector<Bit>& mt, const Bit& nb) {
    std::vector<Bit> r(mt.size() * 2);
    for (size_t p = 0; p < mt.size(); ++p) {
      const Bit one = AND(mt[p], nb);
      r[2 * p + 1] = one;
      r[2 * p] = XOR(mt[p], one);
    }
    return r;
  }

  // Public-table lookup at a secret index given as one-hot minterms (exactly one is
  // 1): result bit j = XOR of the minterms whose entry has bit j set. No AND gates.
  Int select_public(const std::vector<Bit>& minterms, const std::vector<int64_t>& table) {
    if (minterms.size() != table.size() || table.empty())
      throw std::invalid_argument("select_public: size mismatch");
    Int r;
    r.lo = *std::min_element(table.begin(), table.end());
    r.hi = *std::max_element(table.begin(), table.end());
    const int w = width_for(r.lo, r.hi);
    r.b.resize(w);
    for (int j = 0; j < w; ++j) {
      bool all1 = true, all0 = true;
      for (int64_t e : table) {
        const bool bit = ((uint64_t)e >> j) & 1;
        all1 = all1 && bit;
        all0 = all0 && !bit;
      }
      if (all1) { r.b[j] = cst(true); continue; }
      if (all0) { r.b[j] = cst(false); continue; }
      Bit acc = cst(false);
      for (size_t p = 0; p < table.size(); ++p)
        if (((uint64_t)table[p] >> j) & 1) acc = XOR(acc, minterms[p]);
      r.b[j] = acc;
    }
    fit(r);
    return r;
  }

  static int secret_bits(const Int& a) {
    int n = 0;
    for (const Bit& x : a.b) n += x.is_const() ? 0 : 1;
    return n;
  }

 private:
  Ctx& ctx_;

  // a + b + cin over exactly a.size() bits (mod 2^w). The carry out of the top bit
  // is never needed because every result width is chosen so the value fits.
  std::vector<Bit> add_bits(const std::vector<Bit>& a, const std::vector<Bit>& b, Bit cin) {
    const int w = (int)a.size();
    std::vector<Bit> s(w);
    Bit c = cin;
    for (int i = 0; i < w; ++i) {
      s[i] = XOR(XOR(a[i], b[i]), c);
      if (i + 1 < w) c = MAJ(a[i], b[i], c);
    }
    return s;
  }

  // acc[k..w) <- acc[k..w) +/- t[0..w-k)  (mod 2^(w-k)); acc[0..k) is untouched
  // because the shifted term is zero there.
  void accumulate(std::vector<Bit>& acc, const std::vector<Bit>& t, int k, bool subtract) {
    const int w = (int)acc.size();
    if (k >= w) return;
    std::vector<Bit> hi(acc.begin() + k, acc.end());
    std::vector<Bit> tt(t.begin(), t.begin() + (w - k));
    if (subtract)
      for (Bit& x : tt) x = NOT(x);
    const std::vector<Bit> s = add_bits(hi, tt, cst(subtract));
    std::copy(s.begin(), s.end(), acc.begin() + k);
  }
};

}  // namespace lrgc

#endif  // LRGC_SBIT_H
