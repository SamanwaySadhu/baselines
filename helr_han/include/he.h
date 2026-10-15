// he.h -- glue around the HEAAN library: randomness, parallel execution, the set of evaluation keys, and the
// (de)serialization of keys and ciphertexts with validation of everything received from the other party.
#ifndef HAN_HE_H
#define HAN_HE_H

#include <NTL/BasicThreadPool.h>
#include <NTL/ZZ.h>
#include <openssl/rand.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <set>
#include <stdexcept>
#include <vector>

#include "Ciphertext.h"
#include "EvaluatorUtils.h"
#include "Key.h"
#include "Ring.h"
#include "Scheme.h"
#include "SecretKey.h"
#include "metrics.h"
#include "net.h"
#include "params.h"

namespace han {

// A protocol violation by the other party (malformed message, bad parameters, failed check).
struct AbortError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ---- randomness ----------------------------------------------------------------------------------------
// HEAAN samples with NTL's generator, which is per thread; seed it from the OS in every thread that samples.
inline void seed_ntl() {
  unsigned char b[32];
  if (RAND_bytes(b, sizeof b) != 1) throw std::runtime_error("RAND_bytes failed");
  NTL::ZZ s;
  NTL::ZZFromBytes(s, b, sizeof b);
  NTL::SetSeed(s);
}

// ---- parallel execution -------------------------------------------------------------------------------
// NTL's thread pool is both the outer pool (one task per ciphertext) and the inner pool of HEAAN's ring
// operations; a task already running in the pool executes the nested NTL_EXEC_RANGE sequentially.
inline void init_threads(int threads) { NTL::SetNumThreads(threads); }
inline void par_for(long n, const std::function<void(long)>& f) {
  NTL::BasicThreadPool* pool = NTL::GetThreadPool();
  if (!pool || n <= 1) {
    for (long i = 0; i < n; ++i) f(i);
    return;
  }
  // static split of [0, n) over the pool's threads (each thread takes a contiguous range)
  pool->exec_range(n, [&](long first, long last) {
    for (long i = first; i < last; ++i) f(i);
  });
}

// ---- keys ---------------------------------------------------------------------------------------------
inline double prime_bit_size() { return std::log2((double)NTL_SP_BOUND); }
inline long key_np(const Ring& r) { return (long)std::ceil((2.0 * r.logQQ + r.logN + 2) / (prime_bit_size() - 1)); }

// The left-rotation keys of the circuit: powers of two (inner product sums, sample sums, bootstrapping), the
// right rotations of the replication step (N/2 - 2^i), and the extra rotations of the bootstrapping at
// bBits slots (those of Scheme::addBootKey).
inline std::vector<long> required_rot_keys(const Params& P, const Ring& ring) {
  std::set<long> s;
  for (long i = 0; i < ring.logN - 1; ++i) s.insert(1L << i);
  for (long i = 0; i < P.bBits; ++i) s.insert(ring.Nh - (1L << i));
  const long logl = P.bBits, loglh = logl / 2, k = 1L << loglh, m = 1L << (logl - loglh);
  for (long i = 1; i < k; ++i) s.insert(i);
  for (long i = 1; i < m; ++i) s.insert(i * k);
  return std::vector<long>(s.begin(), s.end());
}

inline long boot_logp(const Params& P) { return P.logq + P.logI; }

// Client side: everything the evaluator needs besides the encryption key (Scheme's constructor made the
// encryption and multiplication keys): the bootstrapping context, the conjugation key and the rotation keys.
inline void generate_eval_keys(const Params& P, SecretKey& sk, Scheme& scheme) {
  scheme.ring.addBootContext(P.bBits, boot_logp(P));
  scheme.addConjKey(sk);
  for (long idx : required_rot_keys(P, scheme.ring))
    if (scheme.leftRotKeyMap.find(idx) == scheme.leftRotKeyMap.end()) scheme.addLeftRotKey(sk, idx);
}

enum KeyTag : uint64_t { TAG_MULT = 1, TAG_CONJ = 2, TAG_ROT = 3 };

inline void send_key(Channel& ch, uint64_t tag, uint64_t idx, const Key& k) {
  const uint64_t hdr[4] = {tag, idx, (uint64_t)k.N, (uint64_t)k.np};
  ch.send(hdr, sizeof hdr);
  ch.send(k.rax, (size_t)k.N * k.np * sizeof(uint64_t));
  ch.send(k.rbx, (size_t)k.N * k.np * sizeof(uint64_t));
}

// Receives one key and validates it against the receiver's own ring: the declared dimensions must be the
// expected ones (so the sizes are not attacker-controlled) and every NTT residue must be reduced.
inline Key* recv_key(Channel& ch, const Ring& ring, uint64_t& tag, uint64_t& idx) {
  uint64_t hdr[4];
  ch.recv(hdr, sizeof hdr);
  tag = hdr[0]; idx = hdr[1];
  const long np = key_np(ring);
  if ((long)hdr[2] != ring.N || (long)hdr[3] != np) throw AbortError("evaluation key with unexpected dimensions");
  if (tag < TAG_MULT || tag > TAG_ROT) throw AbortError("evaluation key with an unknown tag");
  const size_t cnt = (size_t)ring.N * np;
  std::unique_ptr<uint64_t[]> ra(new uint64_t[cnt]), rb(new uint64_t[cnt]);
  ch.recv(ra.get(), cnt * sizeof(uint64_t));
  ch.recv(rb.get(), cnt * sizeof(uint64_t));
  std::atomic<bool> bad{false};
  par_for(np, [&](long i) {
    const uint64_t p = ring.multiplier.pVec[i];
    const uint64_t* a = ra.get() + (size_t)i * ring.N;
    const uint64_t* b = rb.get() + (size_t)i * ring.N;
    for (long n = 0; n < ring.N; ++n)
      if (a[n] > p || b[n] > p) { bad = true; break; }
  });
  if (bad) throw AbortError("evaluation key with unreduced residues");
  return new Key(ra.release(), rb.release(), ring.N, np);
}

// ---- ciphertexts ----------------------------------------------------------------------------------------
// Wire format: 4 x int64 header (N, n, logp, logq), then ax and bx, N coefficients each, every coefficient reduced
// mod q = 2^logq and written as ceil(logq / 8) little-endian bytes.
inline size_t ct_coef_bytes(long logq) { return (size_t)((logq + 7) / 8); }
inline size_t ct_body_bytes(long N, long logq) { return 2 * (size_t)N * ct_coef_bytes(logq); }
inline constexpr size_t kCtHeader = 4 * sizeof(int64_t);

inline std::vector<unsigned char> serialize_ct(const Ciphertext& c) {
  const size_t nb = ct_coef_bytes(c.logq);
  std::vector<unsigned char> out(kCtHeader + ct_body_bytes(c.N, c.logq));
  const int64_t hdr[4] = {c.N, c.n, c.logp, c.logq};
  std::memcpy(out.data(), hdr, sizeof hdr);
  NTL::ZZ q = NTL::power2_ZZ(c.logq);
  unsigned char* p = out.data() + kCtHeader;
  for (const NTL::ZZ* poly : {c.ax, c.bx})
    for (long i = 0; i < c.N; ++i) {
      NTL::ZZ v = poly[i] % q;
      NTL::BytesFromZZ(p, v, (long)nb);
      p += nb;
    }
  return out;
}

struct CtShape { long N, n, logp, logq; };

// Checks the header of a received ciphertext against what the protocol expects at this point.
inline void check_ct_header(const unsigned char* hdr, const CtShape& e) {
  int64_t h[4];
  std::memcpy(h, hdr, sizeof h);
  if (h[0] != e.N || h[1] != e.n || h[2] != e.logp || h[3] != e.logq) throw AbortError("ciphertext with an unexpected shape or level");
}

// Parses a ciphertext whose header matched; every coefficient must be < 2^logq (padding bits zero).
inline Ciphertext deserialize_ct(const unsigned char* buf, const CtShape& e) {
  check_ct_header(buf, e);
  const size_t nb = ct_coef_bytes(e.logq);
  const int extra = (int)(nb * 8 - e.logq);  // padding bits of the top byte
  const unsigned char topmask = extra ? (unsigned char)(0xFFu << (8 - extra)) : 0;
  NTL::ZZ* ax = new NTL::ZZ[e.N];
  NTL::ZZ* bx = new NTL::ZZ[e.N];
  const unsigned char* p = buf + kCtHeader;
  for (NTL::ZZ* poly : {ax, bx})
    for (long i = 0; i < e.N; ++i) {
      if (topmask && (p[nb - 1] & topmask)) { delete[] ax; delete[] bx; throw AbortError("ciphertext coefficient out of range"); }
      NTL::ZZFromBytes(poly[i], p, (long)nb);
      p += nb;
    }
  return Ciphertext(ax, bx, e.logp, e.logq, e.N, e.n);
}

// ---- the circuit's level schedule (public, data independent) ------------------------------------------------
inline long iter_cost(const Params& P) { return 3L * P.wBits + 2L * P.pBits; }
inline bool boot_after(const Params& P, int t) { return t % P.iterPerBoot == P.iterPerBoot - 1 && t < P.iters - 1; }
// Modulus of a bootstrapped ciphertext. The bootstrapping consumes logNh bits for the slot-sum division, and the
// authors' logQBoot formula is written for logN = 16, so a smaller test ring ends 16 - logN bits higher.
inline long boot_out_logq(const Params& P) { return P.logQ + (16 - P.logN); }
// logq of the final weight ciphertexts
inline long final_logq(const Params& P) {
  long L = P.logQ;
  for (int t = 0; t < P.iters; ++t) {
    L -= iter_cost(P);
    if (boot_after(P, t)) L = boot_out_logq(P);
  }
  return L;
}

}  // namespace han

#endif  // HAN_HE_H
