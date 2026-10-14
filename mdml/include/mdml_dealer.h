// mdml_dealer.h -- trusted-dealer ("fake offline", as in MD-ML) side of the gates in
// mdml_ext.h. Each Fake*Gate writes, for both parties, exactly the preprocessing
// material its online gate reads, in the same order (text files, MD-ML's format).
// Reused from MD-ML as is: FakeParty (keys, SPDZ2k share/MAC generation, files),
// FakeInputGate, FakeMultiplyGate (matrix triples + delta values), FakeElemMultiplyGate.
#ifndef LRMDML_MDML_DEALER_H
#define LRMDML_MDML_DEALER_H

#include "mdml_ext.h"

#include "fake-offline/FakeElemMultiplyGate.h"
#include "fake-offline/FakeGate.h"
#include "fake-offline/FakeInputGate.h"
#include "fake-offline/FakeMultiplyGate.h"
#include "fake-offline/FakeParty.h"
#include "utils/rand.h"

#include <array>
#include <memory>
#include <vector>

namespace lrmd {

using FakePartyT = md_ml::FakeParty<Shr, 2>;
using FakeGateT = md_ml::FakeGate<Shr, 2>;
using FakeGatePtr = std::shared_ptr<FakeGateT>;
using Shares2 = std::array<std::vector<u128>, 2>;

// Counts of the preprocessing material (what a real offline phase would have to produce).
struct DealerStats {
  uint64_t input_masks = 0;     // [lambda] with lambda known to the input owner
  uint64_t triple_elems = 0;    // elements of matrix/element-wise triples (a, b, c)
  uint64_t trunc_pairs = 0;     // (lambda', lambda' >> shift) pairs (edaBits in the paper)
  uint64_t lambda_bits = 0;     // authenticated binary decomposition bits of lambda (LTZ input)
  uint64_t bin_triples = 0;     // authenticated binary AND triples
  uint64_t dabits = 0;          // daBits for B2A
  uint64_t output_masks = 0;    // [r] with r known to the output receiver
};

struct DealerCtx {
  FakePartyT& fp;
  u128 beta = 0;                     // binary-domain MAC key = beta_0 + beta_1
  std::array<u64, 2> beta_shr{};
  DealerStats st;
  std::vector<std::vector<u64>> trunc_lambdas;  // TEST ONLY (--verify): pre-truncation masks

  // Creates the binary MAC key and writes the file header after MD-ML's alpha share:
  // beta share, circuit fingerprint.
  DealerCtx(FakePartyT& p, uint64_t fingerprint) : fp(p) {
    for (int i = 0; i < 2; ++i) {
      beta_shr[i] = md_ml::getRand<u64>();
      beta += beta_shr[i];
    }
    for (int i = 0; i < 2; ++i) fp.ithPartyFile(i) << beta_shr[i] << '\n' << fingerprint << '\n';
  }

  // SPDZ2k binary sharing (Z_{2^65}) of x, MAC beta * x.
  void bin_share(u128 x, Shares2& v, Shares2& m) const {
    x = m65(x);
    const u128 mac = m65(beta * x);
    const u128 v0 = m65(md_ml::getRand<u128>()), m0 = m65(md_ml::getRand<u128>());
    v[0].push_back(v0); v[1].push_back(m65(x - v0));
    m[0].push_back(m0); m[1].push_back(m65(mac - m0));
  }
  // A bit with uniformly random upper s bits (masks the upper bits of opened values).
  static u128 masked_bit(unsigned b) { return m65(((u128)md_ml::getRand<u64>() << 1) | (b & 1)); }

  void write(const Shares2& s) { fp.WriteSharesToAllParites(s); }
};

// ------------------------------------------------------------- gates
// Inputs use md_ml::FakeInputGate unchanged (DealerBackend counts the masks).

// Matrix triple + delta values (MD-ML's FakeMultiplyGate), then the truncation pair.
class FakeMatMul : public md_ml::FakeMultiplyGate<Shr, 2> {
 public:
  FakeMatMul(DealerCtx& c, const FakeGatePtr& x, const FakeGatePtr& y, int shift)
      : md_ml::FakeMultiplyGate<Shr, 2>(x, y), c_(c), shift_(shift) {}
 private:
  void doRunOffline() override {
    md_ml::FakeMultiplyGate<Shr, 2>::doRunOffline();
    const size_t no = dim_row() * dim_col();
    c_.st.triple_elems += dim_row() * dim_mid() + dim_mid() * dim_col() + no;
    if (shift_ == 0) return;
    // lambda_z = lambda_z' >> shift (logical, on the k-bit value), as MD-ML's
    // FakeMultiplyTruncGate but with a run-time shift.
    c_.trunc_lambdas.push_back(lambda_clear());
    std::vector<u64> lp(no);
    for (size_t i = 0; i < no; ++i) lp[i] = lambda_clear()[i] >> shift_;
    auto sh = fake_party().GenerateAllPartiesShares(lp);
    c_.write(sh.value_shares);
    c_.write(sh.mac_shares);
    lambda_clear() = std::move(lp);
    lambda_shr() = std::move(sh.value_shares);
    lambda_shr_mac() = std::move(sh.mac_shares);
    c_.st.trunc_pairs += no;
  }
  DealerCtx& c_;
  int shift_;
};

class FakeElemMul : public md_ml::FakeElemMultiplyGate<Shr, 2> {
 public:
  FakeElemMul(DealerCtx& c, const FakeGatePtr& x, const FakeGatePtr& y)
      : md_ml::FakeElemMultiplyGate<Shr, 2>(x, y), c_(c) {}
 private:
  void doRunOffline() override {
    md_ml::FakeElemMultiplyGate<Shr, 2>::doRunOffline();
    c_.st.triple_elems += 3 * dim_row() * dim_col();
  }
  DealerCtx& c_;
};

class FakeLinear : public FakeGateT {
 public:
  FakeLinear(const FakeGatePtr& x, const FakeGatePtr& y, size_t rows, size_t cols, LinFn f)
      : FakeGateT(x, y), f_(std::move(f)) {
    set_dim_row(rows);
    set_dim_col(cols);
  }
 private:
  void doRunOffline() override {  // no material: lambda_z = f(lambda_x, lambda_y)
    static const std::vector<u128> none;
    auto widen = [](const std::vector<u64>& v) { return std::vector<u128>(v.begin(), v.end()); };
    const auto& x = input_x();
    const auto& y = input_y();
    auto lc = f_(widen(x->lambda_clear()), y ? widen(y->lambda_clear()) : none);
    lambda_clear().assign(lc.size(), 0);
    for (size_t i = 0; i < lc.size(); ++i) lambda_clear()[i] = (u64)lc[i];
    for (int p = 0; p < 2; ++p) {
      lambda_shr()[p] = f_(x->lambda_shr()[p], y ? y->lambda_shr()[p] : none);
      lambda_shr_mac()[p] = f_(x->lambda_shr_mac()[p], y ? y->lambda_shr_mac()[p] : none);
    }
  }
  LinFn f_;
};

class FakeGeqZero : public FakeGateT {
 public:
  FakeGeqZero(DealerCtx& c, const FakeGatePtr& x) : FakeGateT(x, nullptr), c_(c) {
    set_dim_row(x->dim_row());
    set_dim_col(x->dim_col());
  }
 private:
  void doRunOffline() override {
    const size_t n = dim_row() * dim_col(), na = (size_t)ltz_ands() * n;
    // [lambda_z] for the output wire
    lambda_clear().resize(n);
    std::ranges::generate(lambda_clear(), md_ml::getRand<u64>);
    auto sh = fake_party().GenerateAllPartiesShares(lambda_clear());
    lambda_shr() = sh.value_shares;
    lambda_shr_mac() = sh.mac_shares;
    c_.write(sh.value_shares);
    c_.write(sh.mac_shares);
    // binary decomposition of the input mask lambda_x
    Shares2 v, m;
    for (size_t e = 0; e < n; ++e)
      for (int i = 0; i < kLtzBits; ++i) c_.bin_share(DealerCtx::masked_bit((input_x()->lambda_clear()[e] >> i) & 1), v, m);
    c_.write(v); c_.write(m);
    // AND triples (a, b uniform in Z_{2^65}, c = a b)
    Shares2 av, am, bv, bm, cv, cm;
    for (size_t t = 0; t < na; ++t) {
      const u128 a = m65(md_ml::getRand<u128>()), b = m65(md_ml::getRand<u128>());
      c_.bin_share(a, av, am);
      c_.bin_share(b, bv, bm);
      c_.bin_share(m65(a * b), cv, cm);
    }
    c_.write(av); c_.write(am); c_.write(bv); c_.write(bm); c_.write(cv); c_.write(cm);
    // daBits: the same random bit r shared in the binary and the arithmetic domain
    Shares2 rbv, rbm;
    std::vector<u64> r(n);
    for (size_t e = 0; e < n; ++e) {
      r[e] = md_ml::getRand<u64>() & 1;
      c_.bin_share(DealerCtx::masked_bit((unsigned)r[e]), rbv, rbm);
    }
    c_.write(rbv); c_.write(rbm);
    auto ra = fake_party().GenerateAllPartiesShares(r);
    c_.write(ra.value_shares);
    c_.write(ra.mac_shares);

    c_.st.lambda_bits += n * kLtzBits;
    c_.st.bin_triples += na;
    c_.st.dabits += n;
  }
  DealerCtx& c_;
};

class FakeOutputTo : public FakeGateT {
 public:
  FakeOutputTo(DealerCtx& c, const FakeGatePtr& x, size_t owner) : FakeGateT(x, nullptr), c_(c), owner_(owner) {
    set_dim_row(x->dim_row());
    set_dim_col(x->dim_col());
  }
 private:
  void doRunOffline() override {
    const size_t n = dim_row() * dim_col();
    std::vector<u64> r(n);
    std::ranges::generate(r, md_ml::getRand<u64>);
    fake_party().WriteClearToIthParty(r, owner_);
    auto sh = fake_party().GenerateAllPartiesShares(r);
    c_.write(sh.value_shares);
    c_.write(sh.mac_shares);
    c_.st.output_masks += n;
  }
  DealerCtx& c_;
  size_t owner_;
};

}  // namespace lrmd

#endif  // LRMDML_MDML_DEALER_H
