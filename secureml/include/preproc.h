// preproc.h -- the offline (data-independent) phase, with SPDZ/MASCOT-style malicious security.
//
// Everything here depends only on the public parameters (d, iters, batch, F), not on the data. It
// produces, as authenticated shares (see mpc.h):
//
//   * client-known masks  U[r][j] (one per data entry, the SecureML "U" that masks X once for the whole
//     training), Rw[j] (masks the initial weights), Ry[r] (masks the labels);  server-known Rout[j]
//     (masks the output so only the server learns it)
//   * matrix triples, per iteration t:   Zf_t = U_t * Vf_t   (batch x d times d)  and
//                                        Zb_t = U_t^T * Vb_t (d x batch times batch)
//     with fresh random shared vectors Vf_t (masks w, forward pass) and Vb_t (masks e, backward pass)
//   * scalar Beaver triples (a, b, c = ab) for the activation gadget, and random shared bits
//
// The products U*V are the "vectorized" OT products of SecureML Sec. 4.2: U is known to the client, so
// Z = U*(V_0 + V_1) = U*V_0 (local) + U*V_1 (one OT product: client = message sender, server = chooser).
// Every value is then authenticated (MAC = alpha * value) with one OT product with the other party's
// MAC key. Malicious security needs two more things, both done here:
//   - SACRIFICE: every triple is generated together with a partner (Uh, Zh = Uh*V) that shares V; a
//     random challenge rho (coin toss) checks  rho*Z - Zh - (rho*U - Uh)*V = 0  on opened values, so a
//     party that used inconsistent inputs in an OT product is caught (except with prob ~2^-127).
//   - the MAC check on every opened value.
// Random bits are made from random shared a by opening a^2 = s and setting bit = (a/sqrt(s) + 1)/2,
// which is a bit iff the opening of s is right (it is MAC-checked).
//
// --semi-honest: the partner values, the MACs, the sacrifice and the MAC check are skipped and the OT
// extension runs in its semi-honest mode.
#ifndef SML_PREPROC_H
#define SML_PREPROC_H

#include <algorithm>
#include <vector>

#include "metrics.h"
#include "mpc.h"
#include "params.h"

namespace sml {

// ---- gadget size (shared by preprocessing and the online phase) ----------------------------------
// Borrow chain over n bits as a log-depth tree: at each level, adjacent segments (lo, hi) combine to
// (G_hi + P_hi G_lo, P_hi P_lo); the lowest segment's P is never needed.
inline int chain_mults(int n) {
  int m = n, total = 0;
  while (m > 1) {
    const int pairs = m / 2;
    total += 2 * pairs - 1;
    m = pairs + (m & 1);
  }
  return total;
}
inline int chain_levels(int n) {
  int m = n, l = 0;
  while (m > 1) { m = m / 2 + (m & 1); ++l; }
  return l;
}
// Multiplications per activation: truncation-borrow chain (T bits), two sign chains (N-1 bits each),
// two xor products, one final product.
inline int64_t mults_per_sample(const Params& P) { return chain_mults(P.tr) + 2 * chain_mults(P.nbits - 1) + 2 + 1; }

struct Pre {
  std::vector<Sh> U, Rw, Ry, Rout;      // masks (client-known: U, Rw, Ry; server-known: Rout)
  std::vector<Fp> u_plain, rw_plain, ry_plain, rout_plain;  // plain values at their owner
  std::vector<Sh> Vf, Vb, Zf, Zb;       // matrix-triple parts
  std::vector<Sh> ta, tb, tc;           // scalar triples
  std::vector<Sh> bits;                 // random shared bits
  size_t tri_next = 0, bit_next = 0;
  int64_t NB = 0, NT = 0;
};

namespace detail {

// MAC shares for values known to `owner` alone: sender = owner (messages x), chooser = the other
// party (bits of its MAC key). Owner: alpha*x + S; other: R. One OT product with J = 1, L = K.
inline std::vector<Fp> auth_owner(Mpc& M, int owner, const std::vector<Fp>& x, size_t K) {
  std::vector<Fp> m(K);
  if (K == 0) return m;
  std::vector<Fp> o(K);
  if (M.me == owner) {
    product_send(M.lk, 1, (int64_t)K, x.data(), o.data());
    for (size_t k = 0; k < K; ++k) m[k] = M.alpha * x[k] + o[k];
  } else {
    product_recv(M.lk, 1, (int64_t)K, &M.alpha, o.data());
    m = o;
  }
  return m;
}

// MAC shares for values both parties hold shares of: two OT products (cross terms alpha_0 x_1 and
// alpha_1 x_0). m_i = alpha_i x_i + (share of alpha_{1-i} x_i) + (share of alpha_i x_{1-i}).
inline std::vector<Fp> auth_shared(Mpc& M, const std::vector<Fp>& x) {
  const size_t K = x.size();
  std::vector<Fp> m(K);
  if (K == 0) return m;
  std::vector<Fp> s(K), r(K);
  if (M.me == 0) {
    product_send(M.lk, 1, (int64_t)K, x.data(), s.data());     // party 0 messages, party 1 chooses
    product_recv(M.lk, 1, (int64_t)K, &M.alpha, r.data());     // party 1 messages, party 0 chooses
  } else {
    product_recv(M.lk, 1, (int64_t)K, &M.alpha, r.data());
    product_send(M.lk, 1, (int64_t)K, x.data(), s.data());
  }
  for (size_t k = 0; k < K; ++k) m[k] = M.alpha * x[k] + s[k] + r[k];
  return m;
}

}  // namespace detail

inline void generate_preprocessing(Mpc& M, const Params& P, Pre& pre, Recorder& rec) {
  using namespace detail;
  Link& lk = M.lk;
  const int me = M.me;
  const bool mal = M.mal;
  const int64_t d = P.d, B = P.batch, T = P.iters, n = P.n;
  const int64_t NB = n * P.mask_bits();
  const int64_t NT = NB + n * mults_per_sample(P);
  pre.NB = NB;
  pre.NT = NT;
  emp::PRG& g = M.prg;
  auto rnd = [&](std::vector<Fp>& v, size_t cnt) { v.resize(cnt); for (auto& x : v) x = random_fp(g); };

  // ---- 1. random values ------------------------------------------------------------------------
  std::vector<Fp> u, uh, rw, ry, rout, vf, vb, a, b, ah, abit;
  {
    Scope s(rec, "offline/randomness");
    if (me == 0) {
      rnd(u, (size_t)(n * d)); rnd(rw, (size_t)d); rnd(ry, (size_t)n);
      if (mal) rnd(uh, (size_t)(n * d));
    } else {
      rnd(rout, (size_t)d);
    }
    rnd(vf, (size_t)(T * d)); rnd(vb, (size_t)(T * B));
    rnd(a, (size_t)NT); rnd(b, (size_t)NT); rnd(abit, (size_t)NB);
    if (mal) rnd(ah, (size_t)NT);
  }

  // ---- 2. OT products: matrix triples ------------------------------------------------------------
  // Forward: Zf_t[i] = sum_j U[t*B+i][j] * Vf_t[j].  Items (t, j) with b = my share of Vf_t[j]; the
  // client's message vector for an item is column j of U_t (and of Uh_t for the sacrifice partner).
  std::vector<Fp> zf((size_t)(T * B), Fp(0)), zfh, zb((size_t)(T * d), Fp(0)), zbh;
  if (mal) { zfh.assign((size_t)(T * B), Fp(0)); zbh.assign((size_t)(T * d), Fp(0)); }
  {
    Scope s(rec, "offline/matrix_products");
    const int64_t Lf = mal ? 2 * B : B, Jf = T * d;
    const int64_t Lb = mal ? 2 * d : d, Jb = T * B;
    if (me == 0) {
      std::vector<Fp> A((size_t)(Jf * Lf)), S((size_t)(Jf * Lf));
      for (int64_t t = 0; t < T; ++t)
        for (int64_t j = 0; j < d; ++j)
          for (int64_t i = 0; i < B; ++i) {
            A[(size_t)((t * d + j) * Lf + i)] = u[(size_t)((t * B + i) * d + j)];
            if (mal) A[(size_t)((t * d + j) * Lf + B + i)] = uh[(size_t)((t * B + i) * d + j)];
          }
      if (M.tamper.hit(Tamper::OT_INPUT, me)) A[0] += Fp(1);
      product_send(lk, Jf, Lf, A.data(), S.data());
      for (int64_t t = 0; t < T; ++t)
        for (int64_t j = 0; j < d; ++j)
          for (int64_t i = 0; i < B; ++i) {
            const Fp v0 = vf[(size_t)(t * d + j)];
            zf[(size_t)(t * B + i)] += u[(size_t)((t * B + i) * d + j)] * v0 + S[(size_t)((t * d + j) * Lf + i)];
            if (mal) zfh[(size_t)(t * B + i)] += uh[(size_t)((t * B + i) * d + j)] * v0 + S[(size_t)((t * d + j) * Lf + B + i)];
          }
      // backward
      A.assign((size_t)(Jb * Lb), Fp(0)); S.assign((size_t)(Jb * Lb), Fp(0));
      for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < B; ++i)
          for (int64_t j = 0; j < d; ++j) {
            A[(size_t)((t * B + i) * Lb + j)] = u[(size_t)((t * B + i) * d + j)];
            if (mal) A[(size_t)((t * B + i) * Lb + d + j)] = uh[(size_t)((t * B + i) * d + j)];
          }
      product_send(lk, Jb, Lb, A.data(), S.data());
      for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < B; ++i) {
          const Fp v0 = vb[(size_t)(t * B + i)];
          for (int64_t j = 0; j < d; ++j) {
            zb[(size_t)(t * d + j)] += u[(size_t)((t * B + i) * d + j)] * v0 + S[(size_t)((t * B + i) * Lb + j)];
            if (mal) zbh[(size_t)(t * d + j)] += uh[(size_t)((t * B + i) * d + j)] * v0 + S[(size_t)((t * B + i) * Lb + d + j)];
          }
        }
    } else {
      std::vector<Fp> R((size_t)(Jf * Lf));
      product_recv(lk, Jf, Lf, vf.data(), R.data());
      for (int64_t t = 0; t < T; ++t)
        for (int64_t j = 0; j < d; ++j)
          for (int64_t i = 0; i < B; ++i) {
            zf[(size_t)(t * B + i)] += R[(size_t)((t * d + j) * Lf + i)];
            if (mal) zfh[(size_t)(t * B + i)] += R[(size_t)((t * d + j) * Lf + B + i)];
          }
      R.assign((size_t)(Jb * Lb), Fp(0));
      product_recv(lk, Jb, Lb, vb.data(), R.data());
      for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < B; ++i)
          for (int64_t j = 0; j < d; ++j) {
            zb[(size_t)(t * d + j)] += R[(size_t)((t * B + i) * Lb + j)];
            if (mal) zbh[(size_t)(t * d + j)] += R[(size_t)((t * B + i) * Lb + d + j)];
          }
    }
    if (M.tamper.hit(Tamper::MATRIX_Z, me)) zf[0] += Fp(1);
  }

  // ---- 3. OT products: scalar triples c = a*b (and the sacrifice partner ch = ah*b) --------------------
  std::vector<Fp> c((size_t)NT), ch;
  {
    Scope s(rec, "offline/triple_products");
    const int64_t Lt = mal ? 2 : 1;
    std::vector<Fp> A((size_t)(NT * Lt)), O1((size_t)(NT * Lt)), O2((size_t)(NT * Lt));
    for (int64_t k = 0; k < NT; ++k) {
      A[(size_t)(k * Lt)] = a[(size_t)k];
      if (mal) A[(size_t)(k * Lt + 1)] = ah[(size_t)k];
    }
    // direction A: party 0 sends (a, ah), party 1 chooses with b. direction B: reversed.
    std::vector<Fp> bch = b;  // the choice values actually used in the OTs (test hook: a chooser that lies)
    if (me == 1 && M.tamper.hit(Tamper::OT_INPUT, me)) bch[(size_t)(NT / 3)] += Fp(1);
    if (me == 0) {
      product_send(lk, NT, Lt, A.data(), O1.data());
      product_recv(lk, NT, Lt, bch.data(), O2.data());
    } else {
      product_recv(lk, NT, Lt, bch.data(), O1.data());
      product_send(lk, NT, Lt, A.data(), O2.data());
    }
    if (mal) ch.resize((size_t)NT);
    for (int64_t k = 0; k < NT; ++k) {
      c[(size_t)k] = a[(size_t)k] * b[(size_t)k] + O1[(size_t)(k * Lt)] + O2[(size_t)(k * Lt)];
      if (mal) ch[(size_t)k] = ah[(size_t)k] * b[(size_t)k] + O1[(size_t)(k * Lt + 1)] + O2[(size_t)(k * Lt + 1)];
    }
    if (M.tamper.hit(Tamper::TRIPLE_C, me)) c[(size_t)(NT / 2)] += Fp(1);
  }

  // ---- 4. authentication ---------------------------------------------------------------------------
  // client-known: U, Uh, Rw, Ry; server-known: Rout; shared: everything else.
  std::vector<Fp> mac0, mac1, macS;
  std::vector<Fp> XS;
  if (mal) {
    Scope s(rec, "offline/authenticate");
    std::vector<Fp> X0;
    if (me == 0) {
      X0.reserve((size_t)(2 * n * d + d + n));
      X0.insert(X0.end(), u.begin(), u.end());
      X0.insert(X0.end(), uh.begin(), uh.end());
      X0.insert(X0.end(), rw.begin(), rw.end());
      X0.insert(X0.end(), ry.begin(), ry.end());
    }
    const size_t K0 = (size_t)(2 * n * d + d + n);
    mac0 = auth_owner(M, 0, X0, K0);
    mac1 = auth_owner(M, 1, rout, (size_t)d);
    XS.reserve((size_t)(T * d + T * B + 2 * T * B + 2 * T * d + 5 * NT + NB));
    for (auto* v : {&vf, &vb, &zf, &zfh, &zb, &zbh, &a, &b, &ah, &c, &ch, &abit}) XS.insert(XS.end(), v->begin(), v->end());
    macS = auth_shared(M, XS);
  }

  // ---- 5. assemble authenticated shares -------------------------------------------------------------
  std::vector<Sh> Uh, Zfh, Zbh, tah, tch;
  {
    Scope s(rec, "offline/assemble");
    auto owned = [&](const std::vector<Fp>& plain, const Fp* mac, size_t cnt, bool mine, std::vector<Sh>& out) {
      out.resize(cnt);
      for (size_t k = 0; k < cnt; ++k) out[k] = {mine ? plain[k] : Fp(0), mac ? mac[k] : Fp(0)};
    };
    const Fp* m0 = mal ? mac0.data() : nullptr;
    owned(u, m0, (size_t)(n * d), me == 0, pre.U);
    if (mal) owned(uh, m0 + n * d, (size_t)(n * d), me == 0, Uh);
    owned(rw, m0 ? m0 + 2 * n * d : nullptr, (size_t)d, me == 0, pre.Rw);
    owned(ry, m0 ? m0 + 2 * n * d + d : nullptr, (size_t)n, me == 0, pre.Ry);
    owned(rout, mal ? mac1.data() : nullptr, (size_t)d, me == 1, pre.Rout);
    pre.u_plain = u; pre.rw_plain = rw; pre.ry_plain = ry; pre.rout_plain = rout;
    size_t off = 0;
    auto shared = [&](const std::vector<Fp>& v, std::vector<Sh>& out) {
      out.resize(v.size());
      for (size_t k = 0; k < v.size(); ++k) out[k] = {v[k], mal ? macS[off + k] : Fp(0)};
      off += v.size();
    };
    shared(vf, pre.Vf); shared(vb, pre.Vb); shared(zf, pre.Zf);
    if (mal) shared(zfh, Zfh); else Zfh.clear();
    shared(zb, pre.Zb);
    if (mal) shared(zbh, Zbh);
    shared(a, pre.ta); shared(b, pre.tb);
    if (mal) shared(ah, tah);
    shared(c, pre.tc);
    if (mal) shared(ch, tch);
    std::vector<Sh> abit_sh;
    shared(abit, abit_sh);
    pre.bits = std::move(abit_sh);  // temporarily the random elements; replaced below
  }

  // ---- 6. sacrifice: every triple is checked against its partner ---------------------------------------
  if (mal) {
    Scope s(rec, "offline/sacrifice");
    const emp::block seed = M.coin();
    emp::PRG rg(&seed);
    std::vector<Fp> rho_s((size_t)NT), rho_m((size_t)T);
    for (auto& r : rho_s) r = random_fp(rg);
    for (auto& r : rho_m) r = random_fp(rg);
    // exchange 1: open sigma_k = rho_k a_k - ah_k and S = rho_t U - Uh (matrix)
    std::vector<Sh> to_open;
    to_open.reserve((size_t)(NT + n * d));
    for (int64_t k = 0; k < NT; ++k) to_open.push_back(M.sub(M.scale(pre.ta[(size_t)k], rho_s[(size_t)k]), tah[(size_t)k]));
    for (int64_t r = 0; r < n; ++r)
      for (int64_t j = 0; j < d; ++j)
        to_open.push_back(M.sub(M.scale(pre.U[(size_t)(r * d + j)], rho_m[(size_t)(r / B)]), Uh[(size_t)(r * d + j)]));
    const std::vector<Fp> opened = M.open(to_open);
    const Fp* sigma = opened.data();
    const Fp* S = opened.data() + NT;
    // exchange 2: the checks, which must open to zero
    std::vector<Sh> chk;
    chk.reserve((size_t)(NT + T * B + T * d));
    for (int64_t k = 0; k < NT; ++k)
      chk.push_back(M.sub(M.sub(M.scale(pre.tc[(size_t)k], rho_s[(size_t)k]), tch[(size_t)k]), M.scale(pre.tb[(size_t)k], sigma[k])));
    for (int64_t t = 0; t < T; ++t)
      for (int64_t i = 0; i < B; ++i) {
        Sh acc = M.sub(M.scale(pre.Zf[(size_t)(t * B + i)], rho_m[(size_t)t]), Zfh[(size_t)(t * B + i)]);
        for (int64_t j = 0; j < d; ++j) acc = M.sub(acc, M.scale(pre.Vf[(size_t)(t * d + j)], S[(t * B + i) * d + j]));
        chk.push_back(acc);
      }
    for (int64_t t = 0; t < T; ++t)
      for (int64_t j = 0; j < d; ++j) {
        Sh acc = M.sub(M.scale(pre.Zb[(size_t)(t * d + j)], rho_m[(size_t)t]), Zbh[(size_t)(t * d + j)]);
        for (int64_t i = 0; i < B; ++i) acc = M.sub(acc, M.scale(pre.Vb[(size_t)(t * B + i)], S[(t * B + i) * d + j]));
        chk.push_back(acc);
      }
    const std::vector<Fp> res = M.open(chk);
    for (const Fp& x : res)
      if (!x.is_zero()) throw AbortError("sacrifice check failed: a multiplication triple is inconsistent (a party deviated in the OT products)");
    M.mac_check();
  }

  // ---- 7. random bits: bit = (a / sqrt(a^2) + 1) / 2 ---------------------------------------------------
  {
    Scope s(rec, "offline/random_bits");
    std::vector<Sh> abit_sh = std::move(pre.bits);
    // a^2 by a Beaver multiplication with the first NB (checked) triples
    std::vector<Sh> de((size_t)(2 * NB));
    for (int64_t k = 0; k < NB; ++k) {
      de[(size_t)k] = M.sub(abit_sh[(size_t)k], pre.ta[(size_t)k]);
      de[(size_t)(NB + k)] = M.sub(abit_sh[(size_t)k], pre.tb[(size_t)k]);
    }
    const std::vector<Fp> dev = M.open(de);
    std::vector<Sh> sq((size_t)NB);
    for (int64_t k = 0; k < NB; ++k) {
      const Fp dd = dev[(size_t)k], ee = dev[(size_t)(NB + k)];
      Sh z = M.add(pre.tc[(size_t)k], M.add(M.scale(pre.tb[(size_t)k], dd), M.scale(pre.ta[(size_t)k], ee)));
      sq[(size_t)k] = M.add_const(z, dd * ee);
    }
    const std::vector<Fp> sv = M.open(sq);
    pre.bits.resize((size_t)NB);
    const Fp inv2 = inv(Fp(2));
    std::atomic<bool> bad{false};
    par_ranges(lk.nw(), NB, 512, [&](int64_t lo, int64_t hi) {
      for (int64_t k = lo; k < hi; ++k) {
        const Fp sk = sv[(size_t)k];
        if (sk.is_zero()) { bad = true; continue; }
        const Fp rinv = inv(sqrt_qr(sk));
        Sh t = M.add_const(M.scale(abit_sh[(size_t)k], rinv), Fp(1));
        pre.bits[(size_t)k] = M.scale(t, inv2);
      }
    });
    if (bad) throw AbortError("random bit generation hit a = 0 (probability 2^-127): rerun");
    pre.tri_next = (size_t)NB;
    pre.bit_next = 0;
  }
  if (mal) {
    Scope s(rec, "offline/mac_check");
    M.mac_check();
  }
}

}  // namespace sml

#endif  // SML_PREPROC_H
