// prep.h -- correlated randomness: the client as dealer, and the servers' PRSS zero-sharings.
//
// Dealer (replaces upstream's Precompute.cpp, which returns all-zero "randomness" for every item, so that
// upstream reveals every product, wrap and comparison in the clear):
//   The client draws three 16-byte seeds k_0, k_1, k_2 and sends (k_i, k_{i+1}) to S_i (setup/seeds), so
//   the seed of component j goes to exactly the two holders of component j (S_j and S_{j-1}). Component j
//   of every dealt value comes from the AES-CTR stream of k_j (emp::PRG), consumed in one fixed, public
//   order (deal_iteration below) by everyone who holds j; the client holds all three streams.
//     random item      all three components from the streams; no communication
//     correlated item  (the value v is fixed by a relation with other items) components 1 and 2 from the
//                      streams, component 0 = v - c_1 - c_2 (ring / mod 67 / XOR) is the *correction*,
//                      which the client sends to S0 and S2 (the holders of component 0); stream 0 is not
//                      consumed for it
//   All corrections of the whole run are computed in offline/deal and sent as one message per recipient
//   (the same bytes to S0 and S2); S1 receives nothing but its seeds. Each server learns exactly its two
//   components of every item. The client is honest (it is the only party with inputs and outputs), so the
//   dealt items are correct by construction.
//   Draw rules: ring element = one 64-bit PRG word; Z_67 element = word % 67; bit = word & 1.
//
// PRSS (Falcon's "3-out-of-3 randomness" of Pi_Prep, missing upstream, so that upstream's reshares leak a
// linear relation on the missing components, see C4): S_i draws a key kappa_i and sends it to S_{i+1}
// (setup/keys); S_i holds (kappa_i, kappa_{i-1}) and uses alpha_i = F(kappa_i, ctr) - F(kappa_{i-1}, ctr),
// sum_i alpha_i = 0. One stream per ring, advanced identically at all three servers.
//
// Correction bytes per iteration (B = batch, d = features, N = 2B ReLU inputs; README "dealer bytes"):
//   truncation pairs   8 (B + d)                     (the floor(R'/2^t) sharing of each mask)
//   Wrap/PC per elem   1 (alpha) + 63 (bits of R) + 1 (t) + 63 (s_k) + 3 (m, m s_0, m t) = 131
//                      [M] + 63 (the z of the 63 product-tree triples)                    = 194
//   Select per elem    8 (the ring copy of the selector bit) [M] + 8 (z of the ring triple)
//   [M] matrix triple  8 (B + d)                     (Z_fwd = U v_f, Z_bwd = U^T v_b)
//   semi-honest  286 B + 8 d        malicious  436 B + 16 d       (bytes per iteration, to S0 and to S2)
#ifndef FAL_PREP_H
#define FAL_PREP_H

#include <emp-tool/runtime/crypto/prg.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

#include "net.h"
#include "params.h"
#include "rss.h"

namespace fal {

// One AES-CTR stream (emp::PRG keyed by a 16-byte seed and a stream id), counting the words drawn.
class Stream {
 public:
  Stream(const u8* key16, int id) : prg_(key16, id) {}
  u64 next() { ++used_; return prg_(); }
  u8 z67() { return (u8)(next() % kP); }
  u8 bit() { return (u8)(next() & 1); }
  uint64_t used() const { return used_; }

 private:
  emp::PRG prg_;
  uint64_t used_ = 0;
};

// The three components of a dealt value as seen by one party (components it does not hold are 0).
template <class T>
struct Sh3 {
  T c[3] = {0, 0, 0};
};
inline u64 val(const Sh3<u64>& s) { return s.c[0] + s.c[1] + s.c[2]; }
inline u8 val67(const Sh3<u8>& s) { return add67(add67(s.c[0], s.c[1]), s.c[2]); }
inline u8 valbit(const Sh3<u8>& s) { return s.c[0] ^ s.c[1] ^ s.c[2]; }

class Dealer {
 public:
  int me = CL;
  std::optional<Stream> st[3];
  std::vector<u8> corr;  // client: corrections written; S0 / S2: corrections received
  size_t rpos = 0;

  // client: seeds = k_0 || k_1 || k_2 (48 bytes)
  void init_client(const u8* seeds) {
    me = CL;
    for (int j = 0; j < 3; ++j) st[j].emplace(seeds + 16 * j, 0);
    priv_.emplace();  // the client's private randomness (RDSEED): the masks m
  }
  // server i: two = k_i || k_{i+1} (32 bytes)
  void init_server(int i, const u8* two) {
    me = i;
    st[i].emplace(two, 0);
    st[next_of(i)].emplace(two + 16, 0);
  }
  bool client() const { return me == CL; }
  uint64_t position(int j) const { return st[j] ? st[j]->used() : 0; }
  // the server's RSS view of a dealt value
  template <class T>
  RSS<T> view(const Sh3<T>& s) const { return {s.c[me], s.c[next_of(me)]}; }

  // ---- items. The value arguments of the correlated items are used by the client only. ----
  Sh3<u64> rand64() {
    Sh3<u64> s;
    for (int j = 0; j < 3; ++j) if (st[j]) s.c[j] = st[j]->next();
    return s;
  }
  Sh3<u64> corr64(u64 v) {
    Sh3<u64> s;
    if (st[1]) s.c[1] = st[1]->next();
    if (st[2]) s.c[2] = st[2]->next();
    if (client()) { s.c[0] = v - s.c[1] - s.c[2]; put(&s.c[0], 8); }
    else if (st[0]) get(&s.c[0], 8);
    return s;
  }
  Sh3<u8> rand67() {
    Sh3<u8> s;
    for (int j = 0; j < 3; ++j) if (st[j]) s.c[j] = st[j]->z67();
    return s;
  }
  Sh3<u8> corr67(u8 v) {
    Sh3<u8> s;
    if (st[1]) s.c[1] = st[1]->z67();
    if (st[2]) s.c[2] = st[2]->z67();
    if (client()) { s.c[0] = sub67(sub67(v, s.c[1]), s.c[2]); put(&s.c[0], 1); }
    else if (st[0]) { get(&s.c[0], 1); if (s.c[0] >= kP) throw AbortError("non-canonical field element"); }
    return s;
  }
  Sh3<u8> randbit() {
    Sh3<u8> s;
    for (int j = 0; j < 3; ++j) if (st[j]) s.c[j] = st[j]->bit();
    return s;
  }
  Sh3<u8> corrbit(u8 v) {
    Sh3<u8> s;
    if (st[1]) s.c[1] = st[1]->bit();
    if (st[2]) s.c[2] = st[2]->bit();
    if (client()) { s.c[0] = (u8)(v ^ s.c[1] ^ s.c[2]); put(&s.c[0], 1); }
    else if (st[0]) { get(&s.c[0], 1); if (s.c[0] > 1) throw AbortError("non-canonical field element"); }
    return s;
  }
  // uniform in Z_67^* (client's private randomness; 0 at the servers, which ignore it)
  u8 nonzero67() { return client() ? (u8)(1 + (*priv_)() % (kP - 1)) : 0; }

  // Servers: every received correction byte must have been used.
  void expect_consumed(const char* what) const {
    if (!client() && st[0] && rpos != corr.size())
      throw AbortError(std::string("dealer: ") + what + " corrections have the wrong length");
  }

 private:
  std::optional<emp::PRG> priv_;
  void put(const void* p, size_t n) {
    const u8* b = (const u8*)p;
    corr.insert(corr.end(), b, b + n);
  }
  void get(void* p, size_t n) {
    if (rpos + n > corr.size()) throw AbortError("dealer: corrections too short");
    std::memcpy(p, corr.data() + rpos, n);
    rpos += n;
  }
};

// ---- PRSS zero-sharings among the servers (Pi_Prep "3-out-of-3 randomness") ------------------------------
// replaces upstream's never-used PRSS keys (main.cpp:44-46, AESObject keyAB/keyBC/keyAC, makefile:66-68)
class Prss {
 public:
  // mine = kappa_i (sent to next), from_prev = kappa_{i-1}
  void init(const u8* mine, const u8* from_prev) {
    for (int r = 0; r < 3; ++r) {
      own_[r].emplace(mine, 1 + r);
      prv_[r].emplace(from_prev, 1 + r);
    }
  }
  u64 zero64() { return own_[0]->next() - prv_[0]->next(); }
  u8 zero67() { return sub67(own_[1]->z67(), prv_[1]->z67()); }
  u8 zerobit() { return own_[2]->bit() ^ prv_[2]->bit(); }

 private:
  std::optional<Stream> own_[3], prv_[3];
};

// ---- preprocessing items (server views; the client keeps only what it needs, see ClientView) ------------
// Truncation pair, port of getDividedShares (Precompute.cpp:11-27; upstream returns zeros): rp = sharing of
// a uniform R', r = sharing of floor(R' / 2^t) (arithmetic shift, as the servers' shift of D).
struct TruncPrep {
  Rv64 rp, r;
};
// [M] matrix triple of one iteration (getTriplets, Precompute.cpp:67-82): U (B x d, the mask of the data:
// the client sends E = X_t - U), v_f (d), Z_f = U v_f (B), v_b (B), Z_b = U^T v_b (d).
struct MatPrep {
  Rv64 U, vf, Zf, vb, Zb;
};
// Wrap / PrivateCompare objects for N elements (getShareConvertObjects + getRandomBitShares, Precompute.cpp:
// 29-63, plus the items the port adds).
struct WrapPrep {
  Rv64 r;            // N    mask R = r_0 + r_1 + r_2 (uniform)
  Rv8 alpha;         // N    XOR sharing of wrap3(r_0, r_1, r_2) (on the dealt components)
  Rv8 bits;          // 63N  Z_67 sharings of bit (63-k) of R, k = 0..62 (MSB first; the LSB enters only via s)
  Rv8 eta;           // N    XOR sharing of the PC padding bit eta
  Rv8 t;             // N    Z_67 sharing of 2 eta - 1                         (C6)
  Rv8 s;             // 63N  Z_67 sharing of t * bit_k(R), k = 1..63 (index k-1) (C6)
  Rv8 m, ms0, mt;    // N    m uniform in Z_67^*, m * t * bit_0(R), m * t       (R-PC-MASK)
  Rv8 tx, ty, tz;    // 63N  [M] Z_67 triples z = x y for the product tree
};
// Selector bit (getSelectorBitShares, Precompute.cpp:39-48) and [M] ring triple (Precompute.cpp:85-97).
struct SelPrep {
  Rv8 c;             // N  XOR sharing of a random bit c
  Rv64 mc;           // N  the same bit c shared in Z_2^64
  Rv64 x, y, z;      // N  [M] z = x y
};
struct IterPrep {
  MatPrep mat;
  TruncPrep fwd, bwd;
  WrapPrep wrap;
  SelPrep sel;
};
// What the client keeps: the masks of every truncation (for the bit-exact reference) and U (to form E).
struct ClientView {
  std::vector<u64> rp_fwd, rp_bwd;  // iters*B, iters*d
  std::vector<u64> U;               // iters*B*d [M]
};
// Client-side plaintext of a WrapPrep (unit tests only).
struct WrapFull {
  std::vector<u64> R;
  std::vector<u8> eta;
};

template <class T>
inline void put_view(const Dealer& D, std::vector<RSS<T>>& v, size_t i, const Sh3<T>& s) {
  if (!D.client()) v[i] = D.view(s);
}
template <class T>
inline void size_for(const Dealer& D, std::vector<RSS<T>>& v, size_t n) {
  if (!D.client()) v.assign(n, RSS<T>{});
}

// port of getDividedShares (Precompute.cpp:11-27; zeros upstream); fix: r = (int64)R' >> t (same shift as the servers)
inline void deal_trunc(Dealer& D, size_t n, int shift, TruncPrep& P, std::vector<u64>* rp_full) {
  size_for(D, P.rp, n);
  size_for(D, P.r, n);
  for (size_t i = 0; i < n; ++i) {
    const Sh3<u64> a = D.rand64();
    const u64 R = val(a);
    put_view(D, P.rp, i, a);
    put_view(D, P.r, i, D.corr64((u64)((int64_t)R >> shift)));
    if (D.client() && rp_full) rp_full->push_back(R);
  }
}

// port of getTriplets, matrix form (Precompute.cpp:67-82; zeros upstream); one U per iteration serves both products
inline void deal_mat(Dealer& D, int B, int d, MatPrep& P, std::vector<u64>* U_full) {
  const size_t nb = (size_t)B, nd = (size_t)d;
  std::vector<u64> U(nb * nd), vf(nd), vb(nb);
  size_for(D, P.U, nb * nd);
  size_for(D, P.vf, nd);
  size_for(D, P.Zf, nb);
  size_for(D, P.vb, nb);
  size_for(D, P.Zb, nd);
  for (size_t k = 0; k < nb * nd; ++k) { const auto s = D.rand64(); U[k] = val(s); put_view(D, P.U, k, s); }
  for (size_t j = 0; j < nd; ++j) { const auto s = D.rand64(); vf[j] = val(s); put_view(D, P.vf, j, s); }
  for (size_t i = 0; i < nb; ++i) {
    u64 z = 0;
    if (D.client()) for (size_t j = 0; j < nd; ++j) z += U[i * nd + j] * vf[j];
    put_view(D, P.Zf, i, D.corr64(z));
  }
  for (size_t i = 0; i < nb; ++i) { const auto s = D.rand64(); vb[i] = val(s); put_view(D, P.vb, i, s); }
  for (size_t j = 0; j < nd; ++j) {
    u64 z = 0;
    if (D.client()) for (size_t i = 0; i < nb; ++i) z += U[i * nd + j] * vb[i];
    put_view(D, P.Zb, j, D.corr64(z));
  }
  if (D.client() && U_full) U_full->insert(U_full->end(), U.begin(), U.end());
}

// port of getShareConvertObjects + getRandomBitShares (Precompute.cpp:29-63; zeros upstream) and getTriplets mod p
// (Precompute.cpp:100-112); adds eta as an XOR sharing, t and s_k (C6), m, m s_0, m t (R-PC-MASK)
inline void deal_wrap(Dealer& D, size_t N, bool mal, WrapPrep& P, WrapFull* full = nullptr) {
  size_for(D, P.r, N);
  size_for(D, P.alpha, N);
  size_for(D, P.bits, 63 * N);
  size_for(D, P.eta, N);
  size_for(D, P.t, N);
  size_for(D, P.s, 63 * N);
  size_for(D, P.m, N);
  size_for(D, P.ms0, N);
  size_for(D, P.mt, N);
  if (mal) { size_for(D, P.tx, 63 * N); size_for(D, P.ty, 63 * N); size_for(D, P.tz, 63 * N); }
  for (size_t e = 0; e < N; ++e) {
    const Sh3<u64> r = D.rand64();
    const u64 R = val(r);
    put_view(D, P.r, e, r);
    put_view(D, P.alpha, e, D.corrbit(wrap3(r.c[0], r.c[1], r.c[2])));
    for (int k = 0; k < 63; ++k) put_view(D, P.bits, e * 63 + k, D.corr67((u8)((R >> (63 - k)) & 1)));
    const Sh3<u8> eta = D.randbit();
    put_view(D, P.eta, e, eta);
    const u8 t = valbit(eta) ? 1 : (u8)(kP - 1);
    put_view(D, P.t, e, D.corr67(t));
    for (int k = 1; k < 64; ++k) put_view(D, P.s, e * 63 + (k - 1), D.corr67(((R >> (63 - k)) & 1) ? t : 0));
    const u8 m = D.nonzero67();
    put_view(D, P.m, e, D.corr67(m));
    put_view(D, P.ms0, e, D.corr67((R >> 63) ? mul67(m, t) : 0));
    put_view(D, P.mt, e, D.corr67(mul67(m, t)));
    if (mal)
      for (int k = 0; k < 63; ++k) {
        const Sh3<u8> x = D.rand67(), y = D.rand67();
        put_view(D, P.tx, e * 63 + k, x);
        put_view(D, P.ty, e * 63 + k, y);
        put_view(D, P.tz, e * 63 + k, D.corr67(mul67(val67(x), val67(y))));
      }
    if (D.client() && full) { full->R.push_back(R); full->eta.push_back(valbit(eta)); }
  }
}

// port of getSelectorBitShares (Precompute.cpp:39-48) + [M] getTriplets, vector form (Precompute.cpp:85-97); zeros upstream
inline void deal_select(Dealer& D, size_t N, bool mal, SelPrep& P) {
  size_for(D, P.c, N);
  size_for(D, P.mc, N);
  if (mal) { size_for(D, P.x, N); size_for(D, P.y, N); size_for(D, P.z, N); }
  for (size_t e = 0; e < N; ++e) {
    const Sh3<u8> c = D.randbit();
    put_view(D, P.c, e, c);
    put_view(D, P.mc, e, D.corr64(valbit(c)));
    if (mal) {
      const Sh3<u64> x = D.rand64(), y = D.rand64();
      put_view(D, P.x, e, x);
      put_view(D, P.y, e, y);
      put_view(D, P.z, e, D.corr64(val(x) * val(y)));
    }
  }
}

// The fixed, public dealing order of one iteration.
inline void deal_iteration(Dealer& D, const Params& P, bool mal, IterPrep& I, ClientView* cv) {
  if (mal) deal_mat(D, P.batch, P.d, I.mat, cv ? &cv->U : nullptr);
  deal_trunc(D, (size_t)P.batch, P.shift_fwd(), I.fwd, cv ? &cv->rp_fwd : nullptr);
  deal_wrap(D, 2 * (size_t)P.batch, mal, I.wrap);
  deal_select(D, 2 * (size_t)P.batch, mal, I.sel);
  deal_trunc(D, (size_t)P.d, P.shift_bwd(), I.bwd, cv ? &cv->rp_bwd : nullptr);
}

inline void deal_all(Dealer& D, const Params& P, bool mal, std::vector<IterPrep>& prep, ClientView* cv) {
  prep.assign((size_t)P.iters, IterPrep{});
  for (int t = 0; t < P.iters; ++t) deal_iteration(D, P, mal, prep[(size_t)t], cv);
}

// ---- byte formulas (payload bytes; every message adds an 8-byte frame header) --------------------------
inline uint64_t wrap_corr_bytes_per_elem(bool mal) { return mal ? 194 : 131; }
inline uint64_t sel_corr_bytes_per_elem(bool mal) { return mal ? 16 : 8; }
inline uint64_t deal_bytes_per_iter(const Params& P, bool mal) {
  const uint64_t B = (uint64_t)P.batch, d = (uint64_t)P.d, N = 2 * B;
  uint64_t b = 8 * (B + d) + N * wrap_corr_bytes_per_elem(mal) + N * sel_corr_bytes_per_elem(mal);
  if (mal) b += 8 * (B + d);
  return b;  // semi-honest 286 B + 8 d, malicious 436 B + 16 d
}
inline uint64_t deal_payload_bytes(const Params& P, bool mal) { return (uint64_t)P.iters * deal_bytes_per_iter(P, mal); }
// online/input payload from the client to server i: semi-honest = corrections of X, y, w0 (S0, S2 only);
// malicious = E = X - U to every server, plus the corrections of y and w0 to S0 and S2.
inline uint64_t input_payload_bytes(const Params& P, bool mal, int i) {
  const uint64_t nd = (uint64_t)P.n * (uint64_t)P.d, ny = (uint64_t)P.n + (uint64_t)P.d;
  if (mal) return 8 * nd + (i == 1 ? 0 : 8 * ny);
  return i == 1 ? 0 : 8 * (nd + ny);
}

}  // namespace fal

#endif  // FAL_PREP_H
