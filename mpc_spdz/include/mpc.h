// mpc.h -- authenticated additive sharing over F_p among n parties (SPDZ style), openings and the
// batched MAC check.
//
// A shared value x is held as  x = x_0 + ... + x_{n-1}  together with a MAC  alpha * x = m_0 + ... + m_{n-1},
// where alpha = alpha_0 + ... + alpha_{n-1} is a global key that no coalition of fewer than n parties
// knows. Party i holds Sh{v = x_i, m = m_i} and alpha_i. Linear operations (add, subtract, multiplication
// by a public constant, adding a public constant) are local and keep the MAC relation. Opening a value
// is one round in which every party sends its v-share to every other party; the (value, MAC share) pair
// is remembered and verified in a batch by mac_check():
//
//   chi_k   = jointly random coefficients (coin toss: every party commits to a seed, then all open)
//   sigma_i = sum_k chi_k * m_{i,k}  -  alpha_i * sum_k chi_k * x_k          (x_k = the value party i saw)
//   every party commits to sigma_i, then all open; accept iff sigma_0 + ... + sigma_{n-1} = 0.
//
// If any opened value was wrong at any honest party (an additive error e != 0), the sum of the sigmas
// is alpha * sum chi_k e_k plus terms the adversary cannot cancel without knowing the honest parties'
// key shares, which is non-zero except with probability ~2^-127. This also covers a party that sends
// DIFFERENT shares to different parties (equivocation): no broadcast channel is needed for openings.
// In the semi-honest mode all MAC work is skipped.
#ifndef SPZ_MPC_H
#define SPZ_MPC_H

#include <atomic>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "fp.h"
#include "mesh.h"

namespace spz {

struct Sh {
  Fp v, m;
};

struct AbortError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Test-only fault injection: one corrupted value at a chosen stage of one party.
struct Tamper {
  enum Kind {
    NONE, OPEN_PRE, OPEN_ONLINE, OPEN_EQUIV, TRIPLE_C, MATRIX_Z, OT_SEND, OT_CHOOSE, MAC_VALUE, MAC_SIGMA, INPUT_EQUIV
  } kind = NONE;
  int who = -1;  // party that cheats (0 server, 1.. clients)
  std::atomic<bool> fired{false};
  bool hit(Kind k, int me) {
    return kind == k && who == me && !fired.exchange(true);
  }
};

class Mpc {
 public:
  Mesh& mesh;
  const int me, n;
  const bool mal;
  bool online_phase = false;  // set by the online protocol (selects which OPEN tamper fires)
  Fp alpha;
  emp::PRG prg;
  Tamper tamper;

  Mpc(Mesh& m, bool malicious) : mesh(m), me(m.me), n(m.n), mal(malicious) { alpha = random_fp(prg); }

  // ---------------------------------------------------------------- local operations
  // The public constant c is held by party 0 (value share) and by everybody as a MAC share alpha_i * c.
  Sh add(const Sh& a, const Sh& b) const { return {a.v + b.v, a.m + b.m}; }
  Sh sub(const Sh& a, const Sh& b) const { return {a.v - b.v, a.m - b.m}; }
  Sh scale(const Sh& a, Fp c) const { return {a.v * c, a.m * c}; }
  Sh add_const(const Sh& a, Fp c) const { return {me == 0 ? a.v + c : a.v, a.m + alpha * c}; }
  Sh from_const(Fp c) const { return {me == 0 ? c : Fp(0), alpha * c}; }
  Sh zero() const { return {Fp(0), Fp(0)}; }

  // ---------------------------------------------------------------- opening (1 round)
  std::vector<Fp> open(const std::vector<Sh>& x) {
    const size_t k = x.size();
    std::vector<Fp> out(k);
    if (k == 0) return out;
    std::vector<Fp> mine(k), alt;
    for (size_t i = 0; i < k; ++i) mine[i] = x[i].v;
    if (tamper.hit(online_phase ? Tamper::OPEN_ONLINE : Tamper::OPEN_PRE, me)) mine[k / 2] += Fp(1);
    int equiv_peer = -1;
    if (online_phase && tamper.hit(Tamper::OPEN_EQUIV, me)) {  // a different share to one peer only
      alt = mine;
      alt[k / 2] += Fp(1);
      equiv_peer = me == 0 ? 1 : 0;
    }
    std::vector<std::vector<Fp>> theirs((size_t)n);
    std::vector<Mesh::Buf> s((size_t)n);
    std::vector<Mesh::RBuf> r((size_t)n);
    for (int p = 0; p < n; ++p) {
      if (p == me) continue;
      s[(size_t)p] = {p == equiv_peer ? alt.data() : mine.data(), k * sizeof(Fp)};
      theirs[(size_t)p].resize(k);
      r[(size_t)p] = {theirs[(size_t)p].data(), k * sizeof(Fp)};
    }
    mesh.exchange(s, r);
    for (size_t i = 0; i < k; ++i) {
      Fp v = x[i].v;
      for (int p = 0; p < n; ++p) if (p != me) v += canonical(theirs[(size_t)p][i]);
      out[i] = v;
    }
    if (mal) {
      chk_x.insert(chk_x.end(), out.begin(), out.end());
      for (size_t i = 0; i < k; ++i) chk_m.push_back(x[i].m);
      if (tamper.hit(Tamper::MAC_VALUE, me)) chk_m[chk_m.size() - k / 2 - 1] += Fp(1);  // a wrong MAC share
    }
    return out;
  }

  // ---------------------------------------------------------------- MAC check (4 rounds)
  size_t pending_checks() const { return chk_x.size(); }

  // Verifies every value opened since the last check. Throws AbortError on failure.
  void mac_check() {
    if (!mal) return;
    // 1. coin toss for the coefficients chi (2 rounds)
    const emp::block seed = coin();
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
    // 3. commit to sigma, then open (2 rounds); every commitment must verify and the sigmas must cancel
    emp::block nonce = random_block();
    const std::vector<uint8_t> c_me = commit_fp(sigma, nonce, me);
    std::vector<std::vector<uint8_t>> c_th;
    mesh.all_to_all(c_me.data(), 32, c_th);
    struct Open { Fp s; emp::block nonce; } o_me{sigma, nonce};
    std::vector<std::vector<uint8_t>> o_th;
    mesh.all_to_all(&o_me, sizeof o_me, o_th);
    Fp total = sigma;
    for (int p = 0; p < n; ++p) {
      if (p == me) continue;
      Open o;
      std::memcpy(&o, o_th[(size_t)p].data(), sizeof o);
      if (o.s.v >= kP || commit_fp(o.s, o.nonce, p) != c_th[(size_t)p])
        throw AbortError("MAC check: sigma commitment of party " + std::to_string(p) + " does not verify");
      total += o.s;
    }
    if (total != Fp(0)) throw AbortError("MAC check failed: an opened value was inconsistent with its MAC");
    chk_x.clear(); chk_m.clear();
  }

  // Joint random 128-bit value (commit-then-open coin toss, 2 rounds); derives public challenges.
  emp::block coin() {
    const emp::block s_me = random_block(), n_me = random_block();
    const std::vector<uint8_t> c_me = commit(s_me, n_me, me);
    std::vector<std::vector<uint8_t>> c_th;
    mesh.all_to_all(c_me.data(), 32, c_th);
    emp::block p_me[2] = {s_me, n_me};
    std::vector<std::vector<uint8_t>> p_th;
    mesh.all_to_all(p_me, sizeof p_me, p_th);
    emp::block seed = s_me;
    for (int p = 0; p < n; ++p) {
      if (p == me) continue;
      emp::block pr[2];
      std::memcpy(pr, p_th[(size_t)p].data(), sizeof pr);
      if (commit(pr[0], pr[1], p) != c_th[(size_t)p])
        throw AbortError("coin toss: commitment of party " + std::to_string(p) + " does not verify");
      seed = seed ^ pr[0];
    }
    return seed;
  }

  emp::block random_block() {
    emp::block b;
    prg.random_block(&b, 1);
    return b;
  }

 private:
  std::vector<Fp> chk_x, chk_m;

  // Received field elements must be canonical (< p); otherwise two encodings of one value could be
  // used to make honest parties' views differ without being caught by arithmetic alone.
  static Fp canonical(Fp x) {
    if (x.v >= kP) throw AbortError("received a non-canonical field element");
    return x;
  }

  // The committer's identity is hashed in, so a party cannot reflect another party's commitment.
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

}  // namespace spz

#endif  // SPZ_MPC_H
