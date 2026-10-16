// falcon.h -- the Falcon protocol functions the LR training needs, ported from falcon-public @ e11808c
// (src/Functionalities.cpp = "F.cpp", src/tools.{h,cpp}) and fixed, plus the malicious-security layer.
//
// Upstream function                          -> here                     fixes / additions
//   funcReconstruct (F.cpp:309-384), funcReconstructBit (158-229), funcReconstruct mod p (232-306)
//                                            -> Opener<Ring>             one class for the three rings;
//                                               malicious = value to next + SHA-256 of .second to prev in the
//                                               same step, compared immediately (upstream: 8-bit buffers,
//                                               abort commented out, mod-p variant returns only .first)
//   funcReconstruct3out3 (F.cpp:594-631)     -> inside mat_vec          masked with a PRSS zero-sharing (C4);
//                                               the sum D goes to S0 only (S1 never uses it, SEC-2)
//   funcMatMul (F.cpp:677-735) + matrixMultRSS (tools.cpp:428-518) + dividePlain (tools.cpp:522-543)
//                                            -> mat_vec                 64-bit ring, arithmetic shift instead
//                                               of double division, dealt truncation pair, [M] deferred
//                                               triple check (upstream funcCheckMaliciousMatMul, F.cpp:386-434,
//                                               is a stub)
//   funcDotProduct (F.cpp:739-819), funcDotProduct mod p (823-859), funcMultiplyNeighbours (899-930)
//                                            -> Reshare<Ring>           PRSS zero-sharing added; reshare to prev /
//                                               from next (MultiplyNeighbours sends the wrong way); [M] Beaver-
//                                               triple check (upstream funcCheckMaliciousDotProd is a stub)
//   funcPrivateCompare (F.cpp:1111-1239, parallelFirst/Second 988-1107) + funcCrunchMultiply (933-985)
//                                            -> private_compare         dealer-provided t = 2 eta - 1 and
//                                               s_k = t R_k (C6: no online product, no size/sizeLong bug);
//                                               mask m folded into leaf 0 (R-PC-MASK; upstream opens the raw
//                                               product); [M] tree triples checked BEFORE d is opened
//   funcWrap (F.cpp:1244-1313)               -> relu_prime (body)       PC against X + 1 (C2); delta and beta'
//                                               on component 0 only (C1: upstream also XORs delta into S1's
//                                               .first); eta dealt as an XOR sharing for theta
//   funcRELUPrime (F.cpp:1395-1413)          -> relu_prime              b = MSB(a) (upstream semantics)
//   funcSelectShares (F.cpp:1318-1361)       -> select_shares           a (1 - b); PRSS zero-sharing in the
//                                               dot product; [M] triple check
//   (no upstream counterpart)                -> final_check             [M] checked opening of all deferred
//                                               product checks, all must be 0
//
// Steps (communication rounds) per call: mat_vec 2; relu_prime 1 + PC (6 tree + [M] 1 check + 1 open);
// select_shares 2; final_check 1.
#ifndef FAL_FALCON_H
#define FAL_FALCON_H

#include <emp-tool/runtime/crypto/hash.h>

#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "metrics.h"
#include "net.h"
#include "prep.h"
#include "rss.h"

namespace fal {

// Test-only fault injection: one deviation (+1 on one element) at one server, fired once.
struct Tamper {
  enum Kind { NONE, OPEN_VALUE, OPEN_HASH, REC3, TREE, TREE_NONCANON, MASK, DOT, CHECK, OUTPUT };
  Kind kind = NONE;
  int who = -1;  // the cheating server (0..2)
  bool fired = false;

  static const char* name(Kind k) {
    static const char* n[] = {"none", "open-value", "open-hash", "rec3", "tree", "tree-noncanonical", "mask", "dot", "check", "output"};
    return n[k];
  }
  static Kind parse(const std::string& s) {
    for (int k = 1; k <= OUTPUT; ++k)
      if (s == name((Kind)k)) return (Kind)k;
    throw std::invalid_argument("unknown --tamper kind " + s);
  }
  bool hit(Kind k, int me) {
    if (kind != k || who != me || fired) return false;
    fired = true;
    std::fprintf(stderr, "[S%d] TAMPER fired: %s\n", me, name(k));
    return true;
  }
};

// Per-server protocol context.
struct Ctx {
  Net& net;
  int me;
  bool mal;
  Tamper& tamper;
  Recorder* rec;
  Prss prss;
  uint64_t open_ctr = 0;  // checked openings so far (domain separation of the hashes)
  Rv64 deferred;          // [M] product check values, opened by final_check()
  Ctx(Net& n, bool malicious, Tamper& t, Recorder* r) : net(n), me(n.me()), mal(malicious), tamper(t), rec(r) {}
  int next() const { return next_of(me); }
  int prev() const { return prev_of(me); }
};

inline void open_hash(uint64_t ctr, const void* p, size_t n, u8* out32) {
  emp::Hash h;
  h.put(&ctr, 8);
  if (n) h.put(p, (int64_t)n);
  h.digest(out32);
}

// Opening of an RSS vector (port of funcReconstruct / funcReconstructBit / funcReconstruct mod p,
// F.cpp:158-384). Semi-honest: send .first to next, receive the missing component from prev. Malicious
// ("checked"): in the same step also send SHA-256(op || .second) to prev and compare the hash received from
// next with SHA-256(op || received component). Each server receives its missing component from one neighbour
// and its hash from the other; one of them is honest, so any inconsistency is caught by an honest server.
// Received Z_67 elements / bits must be canonical (SEC-1), in both modes. x must outlive finish().
template <class Rg>
class Opener {
 public:
  using T = typename Rg::T;
  Opener(Ctx& cx, const std::vector<RSS<T>>& x, Tamper::Kind vk = Tamper::NONE, Tamper::Kind hk = Tamper::NONE)
      : cx_(cx), x_(x), ctr_(++cx.open_ctr), first_(x.size()), recv_(x.size()) {
    std::vector<T> second(x.size());
    for (size_t i = 0; i < x.size(); ++i) { first_[i] = x[i].first; second[i] = x[i].second; }
    if (!x.empty() && vk != Tamper::NONE && cx.tamper.hit(vk, cx.me)) first_[0] = Rg::plus_one(first_[0]);
    if (cx.mal) {
      open_hash(ctr_, second.data(), second.size() * sizeof(T), hout_);
      if (hk != Tamper::NONE && cx.tamper.hit(hk, cx.me)) hout_[0] ^= 1;
    }
  }
  void add(Step& s) {
    s.send_vec(cx_.next(), first_);
    if (cx_.mal) s.send(cx_.prev(), hout_, 32);
    s.recv_vec(cx_.prev(), recv_);
    if (cx_.mal) s.recv(cx_.next(), hin_, 32);
  }
  std::vector<T> finish() {
    for (T v : recv_)
      if (!Rg::canonical(v)) throw AbortError("non-canonical field element");
    if (cx_.mal) {
      u8 h[32];
      open_hash(ctr_, recv_.data(), recv_.size() * sizeof(T), h);
      if (std::memcmp(h, hin_, 32) != 0) throw AbortError("checked opening mismatch (op " + std::to_string(ctr_) + ")");
    }
    std::vector<T> out(x_.size());
    for (size_t i = 0; i < x_.size(); ++i) out[i] = Rg::add(Rg::add(x_[i].first, x_[i].second), recv_[i]);
    return out;
  }
  const std::vector<T>& missing() const { return recv_; }  // the component this server did not hold

 private:
  Ctx& cx_;
  const std::vector<RSS<T>>& x_;
  uint64_t ctr_;
  std::vector<T> first_, recv_;
  u8 hout_[32] = {0}, hin_[32] = {0};
};

// Resharing of 3-out-of-3 shares z_i (already masked with a zero-sharing) into RSS: send z_i to prev,
// receive z_{i+1} from next (port of the reshare of funcDotProduct, F.cpp:749-773 / 823-859, with the
// direction of funcMultiplyNeighbours, F.cpp:899-930, fixed).
template <class Rg>
class Reshare {
 public:
  using T = typename Rg::T;
  Reshare(Ctx& cx, std::vector<T> z, Tamper::Kind plus_one = Tamper::NONE, Tamper::Kind plus_p = Tamper::NONE)
      : cx_(cx), z_(std::move(z)), out_(z_), in_(z_.size()) {
    if (!out_.empty() && plus_one != Tamper::NONE && cx.tamper.hit(plus_one, cx.me)) out_[0] = Rg::plus_one(out_[0]);
    if (!out_.empty() && plus_p != Tamper::NONE && cx.tamper.hit(plus_p, cx.me)) out_[0] = (T)(out_[0] + kP);  // non-canonical
  }
  void add(Step& s) {
    s.send_vec(cx_.prev(), out_);
    s.recv_vec(cx_.next(), in_);
  }
  std::vector<RSS<T>> finish() {
    for (T v : in_)
      if (!Rg::canonical(v)) throw AbortError("non-canonical field element");
    std::vector<RSS<T>> r(z_.size());
    for (size_t i = 0; i < z_.size(); ++i) r[i] = {z_[i], in_[i]};
    return r;
  }

 private:
  Ctx& cx_;
  std::vector<T> z_, out_, in_;
};

// [M] what the upstream stubs funcCheckMaliciousDotProd (F.cpp:436-537) intend, done right.
// Beaver-triple check of a product c = p q computed by resharing: with rho = p - x and sigma = q - y
// opened, c - z - rho y - sigma x - rho sigma = 0 for honest shares (rho sigma on component 0).
template <class Rg>
RSS<typename Rg::T> triple_check(int me, const RSS<typename Rg::T>& c, const RSS<typename Rg::T>& x, const RSS<typename Rg::T>& y,
                                 const RSS<typename Rg::T>& z, typename Rg::T rho, typename Rg::T sigma) {
  RSS<typename Rg::T> k = rsub<Rg>(rsub<Rg>(rsub<Rg>(c, z), rscale<Rg>(rho, y)), rscale<Rg>(sigma, x));
  add_const<Rg>(me, k, Rg::sub(0, Rg::mul(rho, sigma)));
  return k;
}

// [M] what mat_vec needs to check its product: the dealt triple (U, v, Z = op(U) v) and E = X_t - U
// (public, sent by the client), stored like the matrix (B x d).
struct MatCheck {
  const std::vector<u64>* E;
  const Rv64* U;
  const Rv64* v;
  const Rv64* Z;
};

// c = (A b) >> shift for an RSS matrix A (rows x cd; with transA the stored matrix is cd x rows and A is its
// transpose) and an RSS vector b. Port of funcMatMul (F.cpp:677-735) with matrixMultRSS (tools.cpp:428-518),
// funcReconstruct3out3 (F.cpp:594-631) and the truncation pair of getDividedShares (Precompute.cpp:11-27);
// fixes: C4 (PRSS zero-sharing on the 3-out-of-3 share), SEC-2 (D to S0 only), 64-bit arithmetic shift
// instead of dividePlain's double division (tools.cpp:522-543); [M] deferred triple check (the upstream
// stub funcCheckMaliciousMatMul, F.cpp:386-434, checks nothing).
//   step 1: T_i = z_i + alpha_i - r'_i; S0, S1 send T to S2 ([M] + checked opening of sigma = b - v)
//   step 2: S2 sends D = sum T = A b - R' to S0
//   c = r + (D >> shift) on component 0
//   [M] chk = (D + [r']) - [Z] - E [v] - [U] sigma - E sigma = (claimed product) - A b, deferred
inline Rv64 mat_vec(Ctx& cx, const Rv64& A, const Rv64& b, size_t rows, size_t cd, bool transA, int shift, const TruncPrep& tp,
                    const MatCheck* mc, const char* phase) {
  Scope sc(cx.rec, phase);
  const int me = cx.me;
  auto at = [&](size_t r, size_t k) { return transA ? k * rows + r : r * cd + k; };
  std::vector<u64> T(rows), T0(rows), T1(rows), D(rows, 0);
  for (size_t r = 0; r < rows; ++r) {
    u64 z = 0;
    for (size_t k = 0; k < cd; ++k) {
      const RSS<u64>& a = A[at(r, k)];
      z += a.first * (b[k].first + b[k].second) + a.second * b[k].first;
    }
    T[r] = z + cx.prss.zero64() - tp.rp[r].first;
  }
  Rv64 sig_sh;
  std::optional<Opener<Ring64>> op;
  if (mc) {
    sig_sh.resize(cd);
    for (size_t k = 0; k < cd; ++k) sig_sh[k] = rsub<Ring64>(b[k], (*mc->v)[k]);
    op.emplace(cx, sig_sh);
  }
  {
    Step s(cx.net);
    if (me != 2) {
      std::vector<u64> Ts = T;
      if (rows && cx.tamper.hit(Tamper::REC3, me)) Ts[0] += u64(1) << shift;
      s.send_vec(S2, Ts);
    } else {
      s.recv_vec(S0, T0);
      s.recv_vec(S1, T1);
    }
    if (op) op->add(s);
    s.run();
  }
  std::vector<u64> sigma;
  if (op) sigma = op->finish();
  {
    Step s(cx.net);
    if (me == 2) {
      for (size_t r = 0; r < rows; ++r) D[r] = T[r] + T0[r] + T1[r];
      std::vector<u64> Ds = D;
      if (rows && cx.tamper.hit(Tamper::REC3, me)) Ds[0] += u64(1) << shift;  // S0 gets a different D
      s.send_vec(S0, Ds);
    } else if (me == 0) {
      s.recv_vec(S2, D);
    }
    s.run();
  }
  Rv64 c(rows);
  for (size_t r = 0; r < rows; ++r) {
    c[r] = tp.r[r];
    add_const<Ring64>(me, c[r], (u64)((int64_t)D[r] >> shift));
  }
  if (mc) {
    const std::vector<u64>& E = *mc->E;
    const Rv64& U = *mc->U;
    const Rv64& v = *mc->v;
    for (size_t r = 0; r < rows; ++r) {
      RSS<u64> k = rsub<Ring64>(tp.rp[r], (*mc->Z)[r]);
      u64 pub = D[r];
      for (size_t q = 0; q < cd; ++q) {
        const size_t i = at(r, q);
        k = rsub<Ring64>(k, rscale<Ring64>(E[i], v[q]));
        k = rsub<Ring64>(k, rscale<Ring64>(sigma[q], U[i]));
        pub -= E[i] * sigma[q];
      }
      add_const<Ring64>(me, k, pub);
      cx.deferred.push_back(k);
    }
  }
  return c;
}

// beta'[e] = [d_e == 0] for public X'[e] and the secret R of the WrapPrep: beta' = [R > X'] if eta = 0 and
// [R < X'] if eta = 1. Port of funcPrivateCompare (F.cpp:1111-1239; parallelFirst/Second 988-1107; paper
// Alg. 1) and funcCrunchMultiply / funcMultiplyNeighbours (F.cpp:899-985); fixes: C6 (dealt t = 2 eta - 1 and
// s_k = t R_k, so u_k = s_k - r_k t is local: removes the online product and its size-vs-sizeLong bug),
// R-PC-MASK (leaf 0 = m c_0 = m s_0 - r_0 m t + m, so d = m prod c_k is 0 or uniform in Z_67^*; upstream
// opens the raw product), MultiplyNeighbours' reshare direction, PRSS zero-sharing, [M] all 63 tree products
// checked with dealt triples before d is opened (d is the only opened value that is not uniformly masked).
//   c_k = s_k - r_k t + 1 + sum_{j<k} (R_j xor r_j)       (k = 0..63, MSB first; leaf 0 times m)
//   6 tree steps (64 -> 1), [M] 1 check step, 1 opening of d
inline std::vector<u8> private_compare(Ctx& cx, const WrapPrep& wp, const std::vector<u64>& Xp) {
  const int me = cx.me;
  const size_t N = Xp.size();
  Rv8 L(64 * N), pc_chk;
  {
    Scope sc(cx.rec, "online/drelu_pc_tree");
    for (size_t e = 0; e < N; ++e) {
      RSS<u8> pre{};  // prefix sum of w_j = R_j xor r_j over the more significant bits
      for (int k = 0; k < 64; ++k) {
        const u8 rk = (u8)((Xp[e] >> (63 - k)) & 1);
        RSS<u8> c;
        if (k == 0) {
          c = radd<Field67>(wp.ms0[e], wp.m[e]);
          if (rk) c = rsub<Field67>(c, wp.mt[e]);
          if (cx.tamper.hit(Tamper::MASK, me)) c.first = add67(c.first, 1);
        } else {
          c = radd<Field67>(wp.s[e * 63 + (size_t)(k - 1)], pre);
          if (rk) c = rsub<Field67>(c, wp.t[e]);
          add_const<Field67>(me, c, 1);
        }
        L[e * 64 + (size_t)k] = c;
        if (k < 63) {  // w_k = R_k xor r_k = r_k ? 1 - R_k : R_k   (XORPublicModPrime, tools.h:206-227)
          RSS<u8> w = wp.bits[e * 63 + (size_t)k];
          if (rk) { w = {neg67(w.first), neg67(w.second)}; add_const<Field67>(me, w, 1); }
          pre = radd<Field67>(pre, w);
        }
      }
    }
    size_t toff = 0;
    for (int level = 0; level < 6; ++level) {
      const size_t n_out = L.size() / 2;
      std::vector<u8> z(n_out);
      for (size_t j = 0; j < n_out; ++j) z[j] = add67(prod3<Field67>(L[2 * j], L[2 * j + 1]), cx.prss.zero67());
      Reshare<Field67> rs(cx, std::move(z), Tamper::TREE, Tamper::TREE_NONCANON);
      Rv8 rhosig;
      std::optional<Opener<Field67>> op;
      if (cx.mal) {
        rhosig.resize(2 * n_out);
        for (size_t j = 0; j < n_out; ++j) {
          rhosig[j] = rsub<Field67>(L[2 * j], wp.tx[toff + j]);
          rhosig[n_out + j] = rsub<Field67>(L[2 * j + 1], wp.ty[toff + j]);
        }
        op.emplace(cx, rhosig);
      }
      Step s(cx.net);
      rs.add(s);
      if (op) op->add(s);
      s.run();
      Rv8 out = rs.finish();
      if (op) {
        const std::vector<u8> v = op->finish();
        for (size_t j = 0; j < n_out; ++j)
          pc_chk.push_back(triple_check<Field67>(me, out[j], wp.tx[toff + j], wp.ty[toff + j], wp.tz[toff + j], v[j], v[n_out + j]));
      }
      toff += n_out;
      L.swap(out);
    }
  }
  if (cx.mal) {  // check before reveal
    Scope sc(cx.rec, "online/drelu_pc_check");
    Opener<Field67> op(cx, pc_chk);
    Step s(cx.net);
    op.add(s);
    s.run();
    for (u8 v : op.finish())
      if (v != 0) throw AbortError("PC check failed");
  }
  Scope sc(cx.rec, "online/drelu_pc_open");
  Opener<Field67> op(cx, L);
  Step s(cx.net);
  op.add(s);
  s.run();
  const std::vector<u8> d = op.finish();
  std::vector<u8> beta(N);
  for (size_t e = 0; e < N; ++e) beta[e] = d[e] == 0 ? 1 : 0;
  return beta;
}

// XOR sharing of MSB(a) for an RSS vector a. Port of funcRELUPrime (F.cpp:1395-1413) around funcWrap
// (F.cpp:1244-1313, paper Alg. 2); fixes: C2 (PC against X + 1, so a = 0 and a = 2^63 are exact; the only
// bad input of the wrap, -1, is odd and the wrap's input 2a is even), C1 (delta xor beta' on component 0
// only), eta dealt as an XOR sharing (upstream XORs its Z_67 shares).
//   x_j = 2 a_j + r_j, beta_j = carry(2 a_j + r_j)        per held component
//   open X (1 step); delta = wrap3(X_0, X_1, X_2)
//   beta' = PC(R, X + 1, eta)
//   theta = beta xor alpha xor eta xor (delta xor beta' on component 0) = wrap3(2a_0, 2a_1, 2a_2)
//   b = theta xor MSB(a_j) per component = MSB(a)
inline Rv8 relu_prime(Ctx& cx, const Rv64& a, const WrapPrep& wp) {
  const int me = cx.me;
  const size_t N = a.size();
  Rv64 x(N);
  Rv8 beta(N);
  for (size_t i = 0; i < N; ++i) {
    const u64 f = a[i].first << 1, g = a[i].second << 1;
    x[i] = {f + wp.r[i].first, g + wp.r[i].second};
    beta[i] = {wrap_around(f, wp.r[i].first), wrap_around(g, wp.r[i].second)};
  }
  std::vector<u64> X, miss;
  {
    Scope sc(cx.rec, "online/drelu_open");
    Opener<Ring64> op(cx, x, Tamper::OPEN_VALUE, Tamper::OPEN_HASH);
    Step s(cx.net);
    op.add(s);
    s.run();
    X = op.finish();
    miss = op.missing();
  }
  std::vector<u64> Xp(N);
  for (size_t i = 0; i < N; ++i) Xp[i] = X[i] + 1;
  const std::vector<u8> bp = private_compare(cx, wp, Xp);
  Rv8 b(N);
  for (size_t i = 0; i < N; ++i) {
    RSS<u8> th = {(u8)(beta[i].first ^ wp.alpha[i].first ^ wp.eta[i].first), (u8)(beta[i].second ^ wp.alpha[i].second ^ wp.eta[i].second)};
    add_const<Bits>(me, th, (u8)(wrap3(x[i].first, x[i].second, miss[i]) ^ bp[i]));
    b[i] = {(u8)(th.first ^ msb(a[i].first)), (u8)(th.second ^ msb(a[i].second))};
  }
  return b;
}

// a (1 - b) for an RSS vector a and an XOR-shared bit vector b (= ReLU(a) when b = MSB(a)). Port of
// funcSelectShares (F.cpp:1318-1361) with funcReconstructBit (F.cpp:158-229) and funcDotProduct
// (F.cpp:739-819, truncation = false); fixes: PRSS zero-sharing in the reshare (C4), [M] Beaver-triple check
// (deferred; upstream funcCheckMaliciousDotProd, F.cpp:436-485, is a stub).
//   step 1: open e = b xor c (c dealt with its ring copy m_c); m = e ? m_c : 1 - m_c  (= 1 - b in Z_2^64)
//   step 2: reshare a m ([M] + checked opening of rho = a - x, sigma = m - y)
inline Rv64 select_shares(Ctx& cx, const Rv64& a, const Rv8& b, const SelPrep& sp) {
  Scope sc(cx.rec, "online/select");
  const int me = cx.me;
  const size_t N = a.size();
  Rv8 bc(N);
  for (size_t i = 0; i < N; ++i) bc[i] = {(u8)(b[i].first ^ sp.c[i].first), (u8)(b[i].second ^ sp.c[i].second)};
  std::vector<u8> e;
  {
    Opener<Bits> op(cx, bc);
    Step s(cx.net);
    op.add(s);
    s.run();
    e = op.finish();
  }
  Rv64 m(N);
  std::vector<u64> z(N);
  for (size_t i = 0; i < N; ++i) {
    m[i] = sp.mc[i];
    if (e[i] == 0) {
      m[i] = {(u64)0 - m[i].first, (u64)0 - m[i].second};
      add_const<Ring64>(me, m[i], 1);
    }
    z[i] = prod3<Ring64>(a[i], m[i]) + cx.prss.zero64();
  }
  Reshare<Ring64> rs(cx, std::move(z), Tamper::DOT);
  Rv64 rhosig;
  std::optional<Opener<Ring64>> op;
  if (cx.mal) {
    rhosig.resize(2 * N);
    for (size_t i = 0; i < N; ++i) {
      rhosig[i] = rsub<Ring64>(a[i], sp.x[i]);
      rhosig[N + i] = rsub<Ring64>(m[i], sp.y[i]);
    }
    op.emplace(cx, rhosig);
  }
  Step s(cx.net);
  rs.add(s);
  if (op) op->add(s);
  s.run();
  Rv64 c = rs.finish();
  if (op) {
    const std::vector<u64> v = op->finish();
    for (size_t i = 0; i < N; ++i) cx.deferred.push_back(triple_check<Ring64>(me, c[i], sp.x[i], sp.y[i], sp.z[i], v[i], v[N + i]));
  }
  return c;
}

// [M] Opens every deferred product check value (checked opening); all must be 0.
inline void final_check(Ctx& cx) {
  Scope sc(cx.rec, "online/final_check");
  Opener<Ring64> op(cx, cx.deferred, Tamper::CHECK);
  Step s(cx.net);
  op.add(s);
  s.run();
  for (u64 v : op.finish())
    if (v != 0) throw AbortError("final check failed");
  cx.deferred.clear();
}

}  // namespace fal

#endif  // FAL_FALCON_H
