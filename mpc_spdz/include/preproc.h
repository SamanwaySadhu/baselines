// preproc.h -- the offline (data-independent) phase among n = batch + 1 parties, with SPDZ/MASCOT-style
// malicious security (dishonest majority: any n-1 parties may be corrupt).
//
// Everything here depends only on the public parameters (d, iters, batch, F), not on the data. It
// produces, as authenticated shares (see mpc.h):
//
//   * owner-known masks:  U[r][j] (one per data entry; it masks X once for the whole training) and Ry[r]
//     (masks the label), both known to the client that owns row r (row r belongs to client 1 + r mod batch,
//     i.e. client k holds datapoint k of every batch); Rw_k[j] (masks client k's share of the initial
//     weights); Rout[j] (known to the server, masks the output so that only the server learns it)
//   * matrix triples, per iteration t:   Zf_t = U_t * Vf_t   (batch x d times d)  and
//                                        Zb_t = U_t^T * Vb_t (d x batch times batch)
//     with fresh random shared vectors Vf_t (masks w in the forward pass) and Vb_t (masks e in the
//     backward pass); every party contributes one random share of V
//   * scalar Beaver triples (a, b, c = ab) for the activation gadget, and random shared bits
//
// The products are OT products (see mesh.h) between every ORDERED pair of parties, as in MASCOT:
//   c = (sum_i a_i)(sum_j b_j) = sum_i a_i b_i + sum_{i != j} a_i b_j     one product per ordered pair (i, j)
//   Z_t[row of client o] = sum_j U_oj sum_p V_{p,j}: the owner o is the OT sender for EVERY other party p
//     (message = its row of U, choice bits = p's share of V), so the matrix triple costs one product per
//     (owner, other party) -- SecureML Sec. 4.2 vectorization, with a vector length of the row
// Every value is then authenticated (MAC = alpha * value, alpha = sum alpha_i): a shared x costs one
// product per ordered pair with the receiver's key alpha_j as choice, and an owner-known x costs one
// product from the owner to each other party. The product of an ordered pair is ONE call whose message
// vector concatenates everything the sender has to authenticate.
// Malicious security needs two more things, both done here:
//   - SACRIFICE: every triple is generated with a partner (Uh, Zh = Uh*V) that shares V; after a coin toss
//     for rho, rho*U - Uh is opened and rho*Z - Zh - (rho*U - Uh)*V must open to 0, for the matrix triples
//     and every scalar triple, so a party that used inconsistent inputs in an OT product is caught.
//   - the MAC check on every opened value.
// Random bits are made from random shared a by opening a^2 = s and setting bit = (a/sqrt(s) + 1)/2,
// which is a bit iff the opening of s is right (it is MAC-checked).
//
// --semi-honest: the partner values, the MACs, the sacrifice and the MAC check are skipped and the OT
// extension runs in its semi-honest mode.
#ifndef SPZ_PREPROC_H
#define SPZ_PREPROC_H

#include <algorithm>
#include <vector>

#include "metrics.h"
#include "mpc.h"
#include "params.h"

namespace spz {

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

// Row r of the data belongs to client 1 + (r mod batch) (party ids: 0 = server, 1..batch = clients).
inline int row_owner(int64_t r, int batch) { return 1 + (int)(r % batch); }

struct Pre {
  std::vector<Sh> U, Rw, Ry, Rout;  // masks: U (n*d), Rw (batch*d: client k's mask at (k-1)*d), Ry (n), Rout (d)
  std::vector<Fp> u_plain, rw_plain, ry_plain, rout_plain;  // plain values at their owner (zero elsewhere)
  std::vector<Sh> Vf, Vb, Zf, Zb;   // matrix-triple parts
  std::vector<Sh> ta, tb, tc;       // scalar triples
  std::vector<Sh> bits;             // random shared bits
  size_t tri_next = 0, bit_next = 0;
  int64_t NB = 0, NT = 0;
};

inline void generate_preprocessing(Mpc& M, const Params& P, Pre& pre, Recorder& rec) {
  Mesh& mesh = M.mesh;
  const int me = M.me, n = M.n;
  const bool mal = M.mal;
  const int64_t d = P.d, B = P.batch, T = P.iters, nrows = P.n;
  if (n != B + 1) throw std::invalid_argument("the number of parties must be batch + 1");
  const bool client = me >= 1;
  const int64_t i0 = me - 1;  // this client's row in every batch
  const int64_t NB = nrows * P.mask_bits();
  const int64_t NT = NB + nrows * mults_per_sample(P);
  pre.NB = NB;
  pre.NT = NT;
  emp::PRG& g = M.prg;
  auto rnd = [&](std::vector<Fp>& v, size_t cnt) { v.resize(cnt); for (auto& x : v) x = random_fp(g); };

  // Runs f(p) for every other party p, one thread per peer (the peers are independent channels).
  auto for_peers = [&](const std::function<void(int)>& f) {
    parallel(n - 1, [&](int idx) { f(idx < me ? idx : idx + 1); });
  };
  // On each pair the two directions run one after the other, lower id as sender first; both ends derive it.
  auto sender_of = [&](int p, int dir) { return dir == 0 ? std::min(me, p) : std::max(me, p); };

  // ---- 1. random values ------------------------------------------------------------------------
  std::vector<Fp> u((size_t)(nrows * d), Fp(0)), uh, rw, ry((size_t)nrows, Fp(0)), rout, vf, vb, a, b, ah, abit;
  {
    Scope s(rec, "offline/randomness");
    if (mal) uh.assign((size_t)(nrows * d), Fp(0));
    if (client) {
      for (int64_t t = 0; t < T; ++t) {
        const int64_t r = t * B + i0;
        for (int64_t j = 0; j < d; ++j) {
          u[(size_t)(r * d + j)] = random_fp(g);
          if (mal) uh[(size_t)(r * d + j)] = random_fp(g);
        }
        ry[(size_t)r] = random_fp(g);
      }
      rnd(rw, (size_t)d);
    } else {
      rnd(rout, (size_t)d);
    }
    rnd(vf, (size_t)(T * d)); rnd(vb, (size_t)(T * B));
    rnd(a, (size_t)NT); rnd(b, (size_t)NT); rnd(abit, (size_t)NB);
    if (mal) rnd(ah, (size_t)NT);
  }

  // ---- 2. OT products: matrix triples ------------------------------------------------------------
  // Forward: Zf_t[i] = sum_j U[t*B+i][j] * Vf_t[j] with Vf_t = sum_p Vf_{p,t}. The owner o of row i is the
  // sender for every other party p: items (t, j) with choice = p's share Vf_{p,t}[j] and message = U (and
  // the partner Uh) of o's row. Backward: Zb_t[j] = sum_i U[t*B+i][j] * Vb_t[i]: items (t) with choice =
  // p's share Vb_{p,t}[i_o] and message = o's whole row (d values; plus d for the partner).
  const int64_t Lm = mal ? 2 : 1, Lb = Lm * d;
  std::vector<Fp> zf((size_t)(T * B), Fp(0)), zfh, zb((size_t)(T * d), Fp(0)), zbh;
  if (mal) { zfh.assign((size_t)(T * B), Fp(0)); zbh.assign((size_t)(T * d), Fp(0)); }
  {
    Scope s(rec, "offline/matrix_products");
    if (client) {  // the owner's own part  U * V_me
      for (int64_t t = 0; t < T; ++t) {
        const int64_t r = t * B + i0;
        for (int64_t j = 0; j < d; ++j) {
          zf[(size_t)r] += u[(size_t)(r * d + j)] * vf[(size_t)(t * d + j)];
          zb[(size_t)(t * d + j)] += u[(size_t)(r * d + j)] * vb[(size_t)(t * B + i0)];
          if (mal) {
            zfh[(size_t)r] += uh[(size_t)(r * d + j)] * vf[(size_t)(t * d + j)];
            zbh[(size_t)(t * d + j)] += uh[(size_t)(r * d + j)] * vb[(size_t)(t * B + i0)];
          }
        }
      }
    }
    std::vector<std::vector<Fp>> pzf((size_t)n), pzfh((size_t)n), pzb((size_t)n), pzbh((size_t)n);
    for_peers([&](int p) {
      PeerLink& L = mesh.peers[(size_t)p];
      pzf[(size_t)p].assign((size_t)(T * B), Fp(0)); pzb[(size_t)p].assign((size_t)(T * d), Fp(0));
      if (mal) { pzfh[(size_t)p].assign((size_t)(T * B), Fp(0)); pzbh[(size_t)p].assign((size_t)(T * d), Fp(0)); }
      for (int dir = 0; dir < 2; ++dir) {
        const int sd = sender_of(p, dir);
        if (sd == 0) continue;  // the server owns no data rows
        const int64_t is = sd - 1;
        const int64_t Jf = T * d;
        if (me == sd) {
          std::vector<Fp> A((size_t)(Jf * Lm)), S((size_t)(Jf * Lm));
          for (int64_t t = 0; t < T; ++t)
            for (int64_t j = 0; j < d; ++j) {
              A[(size_t)((t * d + j) * Lm)] = u[(size_t)((t * B + is) * d + j)];
              if (mal) A[(size_t)((t * d + j) * Lm + 1)] = uh[(size_t)((t * B + is) * d + j)];
            }
          if (M.tamper.hit(Tamper::OT_SEND, me)) A[0] += Fp(1);
          product_send(L, Jf, Lm, A.data(), S.data());
          for (int64_t t = 0; t < T; ++t)
            for (int64_t j = 0; j < d; ++j) {
              pzf[(size_t)p][(size_t)(t * B + is)] += S[(size_t)((t * d + j) * Lm)];
              if (mal) pzfh[(size_t)p][(size_t)(t * B + is)] += S[(size_t)((t * d + j) * Lm + 1)];
            }
          std::vector<Fp> A2((size_t)(T * Lb)), S2((size_t)(T * Lb));
          for (int64_t t = 0; t < T; ++t)
            for (int64_t j = 0; j < d; ++j) {
              A2[(size_t)(t * Lb + j)] = u[(size_t)((t * B + is) * d + j)];
              if (mal) A2[(size_t)(t * Lb + d + j)] = uh[(size_t)((t * B + is) * d + j)];
            }
          product_send(L, T, Lb, A2.data(), S2.data());
          for (int64_t t = 0; t < T; ++t)
            for (int64_t j = 0; j < d; ++j) {
              pzb[(size_t)p][(size_t)(t * d + j)] += S2[(size_t)(t * Lb + j)];
              if (mal) pzbh[(size_t)p][(size_t)(t * d + j)] += S2[(size_t)(t * Lb + d + j)];
            }
        } else {
          std::vector<Fp> R((size_t)(Jf * Lm));
          product_recv(L, Jf, Lm, vf.data(), R.data());
          for (int64_t t = 0; t < T; ++t)
            for (int64_t j = 0; j < d; ++j) {
              pzf[(size_t)p][(size_t)(t * B + is)] += R[(size_t)((t * d + j) * Lm)];
              if (mal) pzfh[(size_t)p][(size_t)(t * B + is)] += R[(size_t)((t * d + j) * Lm + 1)];
            }
          std::vector<Fp> bsel((size_t)T), R2((size_t)(T * Lb));
          for (int64_t t = 0; t < T; ++t) bsel[(size_t)t] = vb[(size_t)(t * B + is)];
          product_recv(L, T, Lb, bsel.data(), R2.data());
          for (int64_t t = 0; t < T; ++t)
            for (int64_t j = 0; j < d; ++j) {
              pzb[(size_t)p][(size_t)(t * d + j)] += R2[(size_t)(t * Lb + j)];
              if (mal) pzbh[(size_t)p][(size_t)(t * d + j)] += R2[(size_t)(t * Lb + d + j)];
            }
        }
      }
    });
    for (int p = 0; p < n; ++p) {
      if (p == me) continue;
      for (size_t k = 0; k < zf.size(); ++k) zf[k] += pzf[(size_t)p][k];
      for (size_t k = 0; k < zb.size(); ++k) zb[k] += pzb[(size_t)p][k];
      if (mal) {
        for (size_t k = 0; k < zfh.size(); ++k) zfh[k] += pzfh[(size_t)p][k];
        for (size_t k = 0; k < zbh.size(); ++k) zbh[k] += pzbh[(size_t)p][k];
      }
    }
    if (M.tamper.hit(Tamper::MATRIX_Z, me)) zf[0] += Fp(1);
  }

  // ---- 3. OT products: scalar triples c = a*b (and the sacrifice partner ch = ah*b) --------------------
  // c_me = a_me b_me + sum_{p != me} ( share of a_me b_p [me sends, p chooses] + share of a_p b_me [p sends, me chooses] ).
  std::vector<Fp> c((size_t)NT), ch;
  {
    Scope s(rec, "offline/triple_products");
    const int64_t Lt = mal ? 2 : 1;
    std::vector<Fp> A((size_t)(NT * Lt));
    for (int64_t k = 0; k < NT; ++k) {
      A[(size_t)(k * Lt)] = a[(size_t)k];
      if (mal) A[(size_t)(k * Lt + 1)] = ah[(size_t)k];
    }
    std::vector<Fp> bch = b;  // the choice values actually used in the OTs (test hook: a chooser that lies)
    if (M.tamper.hit(Tamper::OT_CHOOSE, me)) bch[(size_t)(NT / 3)] += Fp(1);
    std::vector<std::vector<Fp>> O((size_t)n);
    for_peers([&](int p) {
      PeerLink& L = mesh.peers[(size_t)p];
      O[(size_t)p].assign((size_t)(NT * Lt), Fp(0));
      std::vector<Fp> tmp((size_t)(NT * Lt));
      for (int dir = 0; dir < 2; ++dir) {
        if (me == sender_of(p, dir)) product_send(L, NT, Lt, A.data(), tmp.data());
        else product_recv(L, NT, Lt, bch.data(), tmp.data());
        for (size_t k = 0; k < tmp.size(); ++k) O[(size_t)p][k] += tmp[k];
      }
    });
    if (mal) ch.resize((size_t)NT);
    for (int64_t k = 0; k < NT; ++k) {
      Fp ck = a[(size_t)k] * b[(size_t)k], chk = mal ? ah[(size_t)k] * b[(size_t)k] : Fp(0);
      for (int p = 0; p < n; ++p) {
        if (p == me) continue;
        ck += O[(size_t)p][(size_t)(k * Lt)];
        if (mal) chk += O[(size_t)p][(size_t)(k * Lt + 1)];
      }
      c[(size_t)k] = ck;
      if (mal) ch[(size_t)k] = chk;
    }
    if (M.tamper.hit(Tamper::TRIPLE_C, me)) c[(size_t)(NT / 2)] += Fp(1);
  }

  // ---- 4. authentication ---------------------------------------------------------------------------
  // Owner-known values of party o: client: [U rows T*d | Uh rows T*d | Rw d | Ry T]; server: [Rout d].
  // Shared values (everything else) are authenticated by all parties. One product per ordered pair.
  const int64_t own_c = 2 * T * d + d + T;
  auto own_n = [&](int o) -> int64_t { return o == 0 ? d : own_c; };
  std::vector<std::vector<Fp>> macOwn((size_t)n);   // MAC shares (at me) of every owner's values
  std::vector<Fp> macS;
  std::vector<Fp> XS;
  if (mal) {
    Scope s(rec, "offline/authenticate");
    XS.reserve((size_t)(T * d + T * B + 2 * T * B + 2 * T * d + 5 * NT + NB));
    for (auto* v : {&vf, &vb, &zf, &zfh, &zb, &zbh, &a, &b, &ah, &c, &ch, &abit}) XS.insert(XS.end(), v->begin(), v->end());
    const int64_t KS = (int64_t)XS.size();
    std::vector<Fp> Xme = XS;
    if (client) {
      for (int64_t t = 0; t < T; ++t)
        for (int64_t j = 0; j < d; ++j) Xme.push_back(u[(size_t)((t * B + i0) * d + j)]);
      for (int64_t t = 0; t < T; ++t)
        for (int64_t j = 0; j < d; ++j) Xme.push_back(uh[(size_t)((t * B + i0) * d + j)]);
      Xme.insert(Xme.end(), rw.begin(), rw.end());
      for (int64_t t = 0; t < T; ++t) Xme.push_back(ry[(size_t)(t * B + i0)]);
    } else {
      Xme.insert(Xme.end(), rout.begin(), rout.end());
    }
    std::vector<std::vector<Fp>> Sx((size_t)n), Rx((size_t)n);
    for_peers([&](int p) {
      PeerLink& L = mesh.peers[(size_t)p];
      for (int dir = 0; dir < 2; ++dir) {
        const int sd = sender_of(p, dir);
        if (me == sd) {
          Sx[(size_t)p].resize(Xme.size());
          product_send(L, 1, (int64_t)Xme.size(), Xme.data(), Sx[(size_t)p].data());
        } else {
          Rx[(size_t)p].resize((size_t)(KS + own_n(sd)));
          product_recv(L, 1, KS + own_n(sd), &M.alpha, Rx[(size_t)p].data());
        }
      }
    });
    macS.resize((size_t)KS);
    for (int64_t k = 0; k < KS; ++k) {
      Fp m = M.alpha * Xme[(size_t)k];
      for (int p = 0; p < n; ++p) if (p != me) m += Sx[(size_t)p][(size_t)k] + Rx[(size_t)p][(size_t)k];
      macS[(size_t)k] = m;
    }
    for (int o = 0; o < n; ++o) {
      macOwn[(size_t)o].resize((size_t)own_n(o));
      for (int64_t k = 0; k < own_n(o); ++k) {
        if (o == me) {
          Fp m = M.alpha * Xme[(size_t)(KS + k)];
          for (int p = 0; p < n; ++p) if (p != me) m += Sx[(size_t)p][(size_t)(KS + k)];
          macOwn[(size_t)o][(size_t)k] = m;
        } else {
          macOwn[(size_t)o][(size_t)k] = Rx[(size_t)o][(size_t)(KS + k)];
        }
      }
    }
  }

  // ---- 5. assemble authenticated shares -------------------------------------------------------------
  std::vector<Sh> Uh, Zfh, Zbh, tah, tch;
  {
    Scope s(rec, "offline/assemble");
    auto mac_of = [&](int o, int64_t k) { return mal ? macOwn[(size_t)o][(size_t)k] : Fp(0); };
    pre.U.resize((size_t)(nrows * d));
    if (mal) Uh.resize((size_t)(nrows * d));
    pre.Ry.resize((size_t)nrows);
    for (int64_t r = 0; r < nrows; ++r) {
      const int o = row_owner(r, (int)B);
      const int64_t t = r / B;
      for (int64_t j = 0; j < d; ++j) {
        pre.U[(size_t)(r * d + j)] = {u[(size_t)(r * d + j)], mac_of(o, t * d + j)};
        if (mal) Uh[(size_t)(r * d + j)] = {uh[(size_t)(r * d + j)], mac_of(o, T * d + t * d + j)};
      }
      pre.Ry[(size_t)r] = {ry[(size_t)r], mac_of(o, 2 * T * d + d + t)};
    }
    pre.Rw.resize((size_t)(B * d));
    for (int o = 1; o <= B; ++o)
      for (int64_t j = 0; j < d; ++j)
        pre.Rw[(size_t)((o - 1) * d + j)] = {o == me ? rw[(size_t)j] : Fp(0), mac_of(o, 2 * T * d + j)};
    pre.Rout.resize((size_t)d);
    for (int64_t j = 0; j < d; ++j) pre.Rout[(size_t)j] = {me == 0 ? rout[(size_t)j] : Fp(0), mac_of(0, j)};
    pre.u_plain = u;
    pre.ry_plain = ry;
    pre.rw_plain = rw.empty() ? std::vector<Fp>((size_t)d, Fp(0)) : rw;
    pre.rout_plain = rout.empty() ? std::vector<Fp>((size_t)d, Fp(0)) : rout;
    size_t off = 0;
    auto shared = [&](const std::vector<Fp>& v, std::vector<Sh>& out) {
      out.resize(v.size());
      for (size_t k = 0; k < v.size(); ++k) out[k] = {v[k], mal ? macS[off + k] : Fp(0)};
      off += v.size();
    };
    shared(vf, pre.Vf); shared(vb, pre.Vb); shared(zf, pre.Zf);
    if (mal) shared(zfh, Zfh);
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
    // round 1: open sigma_k = rho_k a_k - ah_k and S = rho_t U - Uh (matrix)
    std::vector<Sh> to_open;
    to_open.reserve((size_t)(NT + nrows * d));
    for (int64_t k = 0; k < NT; ++k) to_open.push_back(M.sub(M.scale(pre.ta[(size_t)k], rho_s[(size_t)k]), tah[(size_t)k]));
    for (int64_t r = 0; r < nrows; ++r)
      for (int64_t j = 0; j < d; ++j)
        to_open.push_back(M.sub(M.scale(pre.U[(size_t)(r * d + j)], rho_m[(size_t)(r / B)]), Uh[(size_t)(r * d + j)]));
    const std::vector<Fp> opened = M.open(to_open);
    const Fp* sigma = opened.data();
    const Fp* S = opened.data() + NT;
    // round 2: the checks, which must open to zero
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
    par_ranges(mesh.nw(), NB, 512, [&](int64_t lo, int64_t hi) {
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

}  // namespace spz

#endif  // SPZ_PREPROC_H
