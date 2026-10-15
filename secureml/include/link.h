// link.h -- the party's connections and the OT-based product primitive.
//
// A party owns one main channel (the protocol's own messages: openings, MAC checks, inputs) and
// `nw` worker channels. Each worker channel carries two SoftSpoken OT-extension sessions, one per
// role, over the SAME socket: `snd` (this party is the OT sender) and `rcv` (the OT receiver).
// SoftSpoken is run in its maliciously secure mode (CSW base OT + consistency check) unless
// --semi-honest is given. Every worker is used by one thread at a time, so all heavy preprocessing
// is spread over `nw` threads and `nw` TCP connections (default: all hardware threads).
//
// OT product (Gilboa, vectorized). The "sender" holds, for each of J items, a vector A[j][0..L); the
// "receiver" holds one scalar b[j] per item. After the call the sender holds S and the receiver R
// with  S[j][k] + R[j][k] = A[j][k] * b[j]  (mod p).  Per item: 127 random OTs, one per bit of b[j];
// per OT the sender sends the correction vector corr_t[k] = m0_t[k] - m1_t[k] + 2^t A[k]. The OT
// output is expanded to L field elements with AES counter mode, so one OT serves all L messages
// that are multiplied by the same bit (the SecureML matrix vectorization: the same selection bit
// masks a whole column of the matrix).
#ifndef SML_LINK_H
#define SML_LINK_H

#include <emp-ot/emp-ot.h>
#include <emp-tool/emp-tool.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "fp.h"

namespace sml {

using emp::block;
using emp::NetIO;

struct Worker {
  std::unique_ptr<NetIO> io;
  std::unique_ptr<emp::SoftSpoken<4>> snd, rcv;  // this party as OT sender / OT receiver
  uint64_t rot_count = 0;                         // OTs consumed (for the report)
};

struct Link {
  int me = 0;          // 0 = client, 1 = server
  bool mal = true;
  std::unique_ptr<NetIO> main;
  std::vector<Worker> W;
  int nw() const { return (int)W.size(); }

  // Server listens on port, port+1, ...; client connects. Both create channels in the same order.
  Link(int me_, const std::string& host, int port, int nworkers, bool malicious) : me(me_), mal(malicious) {
    auto make = [&](int p) -> std::unique_ptr<NetIO> {
      return me == 1 ? NetIO::listen(p, true) : NetIO::connect(host.c_str(), p, true);
    };
    main = make(port);
    // Both parties use the smaller of the two requested worker counts (the OT connections pair up 1:1).
    {
      int32_t mine = nworkers, theirs = 0;
      if (me == 0) { main->send_data(&mine, 4); main->flush(); main->recv_data(&theirs, 4); }
      else { main->recv_data(&theirs, 4); main->send_data(&mine, 4); main->flush(); }
      nworkers = std::max(1, std::min<int>(mine, theirs));
    }
    W.resize((size_t)nworkers);
    for (int w = 0; w < nworkers; ++w) {
      W[w].io = make(port + 1 + w);
      W[w].snd = std::make_unique<emp::SoftSpoken<4>>(emp::ALICE, W[w].io.get(), malicious);
      W[w].rcv = std::make_unique<emp::SoftSpoken<4>>(emp::BOB, W[w].io.get(), malicious);
    }
  }

  // Byte / flight counters. A flight is a maximal run of same-direction traffic on one channel.
  struct Counters {
    uint64_t main_sent = 0, main_recv = 0, main_flights = 0;
    uint64_t w_sent = 0, w_recv = 0, w_flights_sum = 0, w_flights_max = 0, w_rots = 0;
  };
  Counters counters() const {
    Counters c;
    c.main_sent = main->send_counter; c.main_recv = main->recv_counter; c.main_flights = main->rounds;
    for (const auto& w : W) {
      c.w_sent += w.io->send_counter; c.w_recv += w.io->recv_counter;
      c.w_flights_sum += w.io->rounds;
      c.w_flights_max = std::max<uint64_t>(c.w_flights_max, w.io->rounds);
      c.w_rots += w.rot_count;
    }
    return c;
  }
};

// Runs f(w) for w in [0, n) on n threads (w = 0 on the calling thread); rethrows the first error.
inline void parallel(int n, const std::function<void(int)>& f) {
  if (n <= 1) { f(0); return; }
  std::vector<std::thread> th;
  std::vector<std::exception_ptr> err((size_t)n);
  for (int w = 1; w < n; ++w)
    th.emplace_back([&, w] { try { f(w); } catch (...) { err[w] = std::current_exception(); } });
  try { f(0); } catch (...) { err[0] = std::current_exception(); }
  for (auto& t : th) t.join();
  for (auto& e : err) if (e) std::rethrow_exception(e);
}

// Even split of [0, n) into `parts`: the range of part p.
inline std::pair<int64_t, int64_t> split_range(int64_t n, int parts, int p) {
  const int64_t base = n / parts, extra = n % parts;
  const int64_t lo = p * base + std::min<int64_t>(p, extra);
  return {lo, lo + base + (p < extra ? 1 : 0)};
}

namespace detail {
inline constexpr int64_t kRotItems = 256;      // items per OT batch (x127 OTs)
inline constexpr int64_t kChunkBudget = 8192;  // items*L per correction chunk (~16 MiB of messages)
inline constexpr int64_t kLChunk = 4096;

inline void plan(int64_t L, int64_t& lc, int64_t& ib) {
  lc = std::min<int64_t>(L, kLChunk);
  ib = std::max<int64_t>(1, std::min<int64_t>(kRotItems, kChunkBudget / std::max<int64_t>(1, lc)));
}
}  // namespace detail

// Sender side of an OT product. A, S: J x L, row strides lda / lds. S is overwritten.
inline void prod_send(Worker& w, int64_t J, int64_t L, const Fp* A, int64_t lda, Fp* S, int64_t lds) {
  using namespace detail;
  if (J <= 0 || L <= 0) return;
  int64_t lc, ib;
  plan(L, lc, ib);
  std::vector<block> r0, r1;
  std::vector<Fp> corr;
  std::vector<emp::PRG> g0, g1;
  std::vector<block> t0((size_t)lc), t1((size_t)lc);
  for (int64_t j0 = 0; j0 < J; j0 += ib) {
    const int64_t cnt = std::min(ib, J - j0), n = cnt * kBits;
    r0.resize((size_t)n); r1.resize((size_t)n);
    w.snd->send_rot(r0.data(), r1.data(), n);
    w.rot_count += (uint64_t)n;
    for (int64_t jj = 0; jj < cnt; ++jj)
      for (int64_t k = 0; k < L; ++k) S[(j0 + jj) * lds + k] = Fp();
    if (L > 1) {
      g0.clear(); g1.clear();
      g0.reserve((size_t)n); g1.reserve((size_t)n);
      for (int64_t i = 0; i < n; ++i) { g0.emplace_back(&r0[(size_t)i]); g1.emplace_back(&r1[(size_t)i]); }
    }
    for (int64_t k0 = 0; k0 < L; k0 += lc) {
      const int64_t kl = std::min(lc, L - k0);
      corr.resize((size_t)(n * kl));
      for (int64_t jj = 0; jj < cnt; ++jj) {
        const Fp* a = A + (j0 + jj) * lda + k0;
        Fp* s = S + (j0 + jj) * lds + k0;
        for (int t = 0; t < kBits; ++t) {
          const int64_t i = jj * kBits + t;
          Fp* c = &corr[(size_t)(i * kl)];
          const Fp p2 = pow2(t);
          if (L == 1) {
            const Fp m0 = from_block(r0[(size_t)i]), m1 = from_block(r1[(size_t)i]);
            s[0] -= m0;
            c[0] = m0 - m1 + p2 * a[0];
          } else {
            g0[(size_t)i].random_block(t0.data(), kl);
            g1[(size_t)i].random_block(t1.data(), kl);
            for (int64_t k = 0; k < kl; ++k) {
              const Fp m0 = from_block(t0[(size_t)k]), m1 = from_block(t1[(size_t)k]);
              s[k] -= m0;
              c[k] = m0 - m1 + p2 * a[k];
            }
          }
        }
      }
      w.io->send_data(corr.data(), (int64_t)(n * kl * (int64_t)sizeof(Fp)));
    }
  }
  w.io->flush();
}

// Receiver side. b: J scalars. R: J x L (row stride ldr), overwritten.
inline void prod_recv(Worker& w, int64_t J, int64_t L, const Fp* b, Fp* R, int64_t ldr) {
  using namespace detail;
  if (J <= 0 || L <= 0) return;
  int64_t lc, ib;
  plan(L, lc, ib);
  std::vector<block> rc;
  std::vector<Fp> corr;
  std::vector<emp::PRG> g;
  std::vector<block> tb((size_t)lc);
  std::unique_ptr<bool[]> bits;
  for (int64_t j0 = 0; j0 < J; j0 += ib) {
    const int64_t cnt = std::min(ib, J - j0), n = cnt * kBits;
    rc.resize((size_t)n);
    bits.reset(new bool[(size_t)n]);
    for (int64_t jj = 0; jj < cnt; ++jj)
      for (int t = 0; t < kBits; ++t) bits[jj * kBits + t] = bit_of(b[j0 + jj], t);
    w.rcv->recv_rot(rc.data(), bits.get(), n);
    w.rot_count += (uint64_t)n;
    for (int64_t jj = 0; jj < cnt; ++jj)
      for (int64_t k = 0; k < L; ++k) R[(j0 + jj) * ldr + k] = Fp();
    if (L > 1) {
      g.clear(); g.reserve((size_t)n);
      for (int64_t i = 0; i < n; ++i) g.emplace_back(&rc[(size_t)i]);
    }
    for (int64_t k0 = 0; k0 < L; k0 += lc) {
      const int64_t kl = std::min(lc, L - k0);
      corr.resize((size_t)(n * kl));
      w.io->recv_data(corr.data(), (int64_t)(n * kl * (int64_t)sizeof(Fp)));
      for (int64_t jj = 0; jj < cnt; ++jj) {
        Fp* r = R + (j0 + jj) * ldr + k0;
        for (int t = 0; t < kBits; ++t) {
          const int64_t i = jj * kBits + t;
          const Fp* c = &corr[(size_t)(i * kl)];
          const bool bit = bits[i];
          if (L == 1) {
            Fp m = from_block(rc[(size_t)i]);
            if (bit) m += c[0];
            r[0] += m;
          } else {
            g[(size_t)i].random_block(tb.data(), kl);
            for (int64_t k = 0; k < kl; ++k) {
              Fp m = from_block(tb[(size_t)k]);
              if (bit) m += c[k];
              r[k] += m;
            }
          }
        }
      }
    }
  }
}

// Parallel OT product over the party's workers. Sender: A (J x L) -> S. Receiver: b (J) -> R (J x L).
// Items are split over workers when J >= nw, otherwise the L columns are (each worker repeats the
// 127 OTs of an item, which is cheap when J is small). Both parties derive the same split.
inline void product_send(Link& lk, int64_t J, int64_t L, const Fp* A, Fp* S) {
  const int nw = lk.nw();
  if (J >= nw) {
    parallel(nw, [&](int w) {
      auto [lo, hi] = split_range(J, nw, w);
      prod_send(lk.W[w], hi - lo, L, A + lo * L, L, S + lo * L, L);
    });
  } else {
    parallel(nw, [&](int w) {
      auto [lo, hi] = split_range(L, nw, w);
      prod_send(lk.W[w], J, hi - lo, A + lo, L, S + lo, L);
    });
  }
}
inline void product_recv(Link& lk, int64_t J, int64_t L, const Fp* b, Fp* R) {
  const int nw = lk.nw();
  if (J >= nw) {
    parallel(nw, [&](int w) {
      auto [lo, hi] = split_range(J, nw, w);
      prod_recv(lk.W[w], hi - lo, L, b + lo, R + lo * L, L);
    });
  } else {
    parallel(nw, [&](int w) {
      auto [lo, hi] = split_range(L, nw, w);
      prod_recv(lk.W[w], J, hi - lo, b, R + lo, L);
    });
  }
}

}  // namespace sml

#endif  // SML_LINK_H
