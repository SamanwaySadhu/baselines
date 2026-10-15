// mpc.h -- authenticated additive sharing over F_p (SPDZ-style), openings and the batched MAC check.
//
// A shared value x is held as  x = x_0 + x_1  together with a MAC  alpha * x = m_0 + m_1,  where
// alpha = alpha_0 + alpha_1 is a global key that neither party knows. Party i holds Sh{v = x_i, m = m_i}
// and alpha_i. Linear operations (add, subtract, multiply by a public constant, add a public
// constant) are local and keep the MAC relation. Opening a value exchanges the v-shares; the (value,
// MAC share) pair is remembered and verified in a batch by mac_check():
//
//   chi_k  = jointly random coefficients (coin toss: both parties commit to a seed, then open)
//   sigma_i = sum_k chi_k * m_{i,k}  -  alpha_i * sum_k chi_k * x_k
//   each party commits to sigma_i, then both open; accept iff sigma_0 + sigma_1 = 0.
//
// If any opened value was wrong (an additive error e != 0), sigma_0 + sigma_1 = alpha * sum chi_k e_k
// which is non-zero except with probability 2^-127 (alpha is unknown to the cheater and the chi_k are
// chosen after the values were opened). In the semi-honest mode all MAC work is skipped.
#ifndef SML_MPC_H
#define SML_MPC_H

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "fp.h"
#include "link.h"

namespace sml {

struct Sh {
  Fp v, m;
};

struct AbortError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Test-only fault injection: one corrupted value at a chosen stage of one party.
struct Tamper {
  enum Kind { NONE, OPEN_PRE, OPEN_ONLINE, TRIPLE_C, MATRIX_Z, OT_INPUT, MAC_VALUE, MAC_SIGMA } kind = NONE;
  int who = -1;      // party that cheats (0 client, 1 server)
  bool fired = false;
  bool hit(Kind k, int me) {
    if (kind == k && who == me && !fired) { fired = true; return true; }
    return false;
  }
};

class Mpc {
 public:
  Link& lk;
  const int me;
  const bool mal;
  bool online_phase = false;   // set by the online protocol (selects which OPEN tamper fires)
  Fp alpha;
  emp::PRG prg;
  Tamper tamper;

  Mpc(Link& l, bool malicious) : lk(l), me(l.me), mal(malicious) {
    alpha = random_fp(prg);
  }

  // ---------------------------------------------------------------- main channel
  void send_bytes(const void* p, size_t n) { lk.main->send_data(p, (int64_t)n); }
  void recv_bytes(void* p, size_t n) { lk.main->recv_data(p, (int64_t)n); }
  void flush() { lk.main->flush(); }

  // Symmetric exchange of n bytes: afterwards each party has the other's buffer. Small messages are
  // sent by both parties at once; large ones are ordered (party 0 sends first) so that two parties
  // never block in send() at the same time.
  void swap(const void* mine, void* theirs, size_t n) {
    if (n <= (1u << 15)) {
      send_bytes(mine, n); flush(); recv_bytes(theirs, n);
    } else if (me == 0) {
      send_bytes(mine, n); flush(); recv_bytes(theirs, n);
    } else {
      recv_bytes(theirs, n); send_bytes(mine, n); flush();
    }
  }

  // ---------------------------------------------------------------- local operations
  Fp one() const { return me == 0 ? Fp(1) : Fp(0); }  // the public constant 1 as a share
  Sh add(const Sh& a, const Sh& b) const { return {a.v + b.v, a.m + b.m}; }
  Sh sub(const Sh& a, const Sh& b) const { return {a.v - b.v, a.m - b.m}; }
  Sh scale(const Sh& a, Fp c) const { return {a.v * c, a.m * c}; }
  Sh add_const(const Sh& a, Fp c) const { return {me == 0 ? a.v + c : a.v, a.m + alpha * c}; }
  Sh from_const(Fp c) const { return {me == 0 ? c : Fp(0), alpha * c}; }
  Sh zero() const { return {Fp(0), Fp(0)}; }

  // ---------------------------------------------------------------- opening
  // Opens a vector (one exchange). In malicious mode the opened values are queued for mac_check().
  std::vector<Fp> open(const std::vector<Sh>& x) {
    const size_t n = x.size();
    std::vector<Fp> mine(n), theirs(n), out(n);
    for (size_t i = 0; i < n; ++i) mine[i] = x[i].v;
    if (n && tamper.hit(online_phase ? Tamper::OPEN_ONLINE : Tamper::OPEN_PRE, me)) mine[n / 2] += Fp(1);
    swap(mine.data(), theirs.data(), n * sizeof(Fp));
    for (size_t i = 0; i < n; ++i) out[i] = x[i].v + theirs[i];
    if (mal) {
      chk_x.insert(chk_x.end(), out.begin(), out.end());
      for (size_t i = 0; i < n; ++i) chk_m.push_back(x[i].m);
      if (n && tamper.hit(Tamper::MAC_VALUE, me)) chk_m[chk_m.size() - n / 2 - 1] += Fp(1);  // a wrong MAC share
    }
    return out;
  }

  // ---------------------------------------------------------------- MAC check
  size_t pending_checks() const { return chk_x.size(); }

  // Verifies every value opened since the last check. Throws AbortError on failure. 4 exchanges.
  void mac_check() {
    if (!mal) return;
    // 1. coin toss for the coefficients chi (commit, then open)
    emp::block seed_me = random_block(), nonce_me = random_block();
    std::vector<uint8_t> com_me = commit(seed_me, nonce_me, me), com_th(32);
    swap(com_me.data(), com_th.data(), 32);
    emp::block pair_me[2] = {seed_me, nonce_me}, pair_th[2];
    swap(pair_me, pair_th, sizeof pair_me);
    if (commit(pair_th[0], pair_th[1], 1 - me) != com_th) throw AbortError("MAC check: coin-toss commitment mismatch");
    const emp::block seed = seed_me ^ pair_th[0];
    // 2. sigma_i = sum chi_k m_k - alpha_i sum chi_k x_k
    emp::PRG chi_prg(&seed);
    Fp sm(0), sx(0);
    for (size_t k = 0; k < chk_x.size(); ++k) {
      const Fp chi = random_fp(chi_prg);
      sm += chi * chk_m[k];
      sx += chi * chk_x[k];
    }
    Fp sigma = sm - alpha * sx;
    if (tamper.hit(Tamper::MAC_SIGMA, me)) sigma += Fp(1);
    // 3. commit to sigma, then open
    emp::block nonce2 = random_block();
    std::vector<uint8_t> c2_me = commit_fp(sigma, nonce2, me), c2_th(32);
    swap(c2_me.data(), c2_th.data(), 32);
    struct Open { Fp s; emp::block nonce; } o_me{sigma, nonce2}, o_th;
    swap(&o_me, &o_th, sizeof o_me);
    if (commit_fp(o_th.s, o_th.nonce, 1 - me) != c2_th) throw AbortError("MAC check: sigma commitment mismatch");
    if (sigma + o_th.s != Fp(0)) throw AbortError("MAC check failed: an opened value was inconsistent with its MAC");
    chk_x.clear(); chk_m.clear();
  }

  // Joint random 128-bit value (commit-then-open coin toss); used to derive public challenges.
  emp::block coin() {
    emp::block s_me = random_block(), n_me = random_block();
    std::vector<uint8_t> c_me = commit(s_me, n_me, me), c_th(32);
    swap(c_me.data(), c_th.data(), 32);
    emp::block p_me[2] = {s_me, n_me}, p_th[2];
    swap(p_me, p_th, sizeof p_me);
    if (commit(p_th[0], p_th[1], 1 - me) != c_th) throw AbortError("coin toss: commitment mismatch");
    return s_me ^ p_th[0];
  }

  emp::block random_block() {
    emp::block b;
    prg.random_block(&b, 1);
    return b;
  }

 private:
  std::vector<Fp> chk_x, chk_m;

  // The committer's identity is hashed in, so a party cannot reflect the other's commitment.
  static std::vector<uint8_t> commit(const emp::block& a, const emp::block& b, int who) {
    uint8_t buf[33];
    std::memcpy(buf, &a, 16);
    std::memcpy(buf + 16, &b, 16);
    buf[32] = (uint8_t)who;
    std::vector<uint8_t> d(32);
    emp::Hash::hash_once(d.data(), buf, 33);
    return d;
  }
  static std::vector<uint8_t> commit_fp(const Fp& s, const emp::block& nonce, int who) {
    uint8_t buf[33];
    std::memcpy(buf, &s.v, 16);
    std::memcpy(buf + 16, &nonce, 16);
    buf[32] = (uint8_t)who;
    std::vector<uint8_t> d(32);
    emp::Hash::hash_once(d.data(), buf, 33);
    return d;
  }
};

// Applies f(lo, hi) over [0, n) on up to `threads` threads when n is large enough to pay for them.
inline int parts_for(int threads, int64_t n, int64_t grain) {
  return (int)std::max<int64_t>(1, std::min<int64_t>(threads, n / std::max<int64_t>(1, grain)));
}
inline void par_parts(int parts, int64_t n, const std::function<void(int, int64_t, int64_t)>& f) {
  if (parts <= 1) { f(0, 0, n); return; }
  parallel(parts, [&](int p) {
    auto [lo, hi] = split_range(n, parts, p);
    f(p, lo, hi);
  });
}
inline void par_ranges(int threads, int64_t n, int64_t grain, const std::function<void(int64_t, int64_t)>& f) {
  par_parts(parts_for(threads, n, grain), n, [&](int, int64_t lo, int64_t hi) { f(lo, hi); });
}

}  // namespace sml

#endif  // SML_MPC_H
