// mesh.h -- the full mesh of n = batch + 1 parties and the OT-based product primitive.
//
// Party 0 is the server, parties 1..batch are the clients. Every pair of parties shares
//   * one MAIN channel (the protocol's own messages: openings, MAC checks, inputs), and
//   * `nw` WORKER channels. Each worker channel carries two SoftSpoken OT-extension sessions, one per
//     role, over the SAME socket: `snd` (this party is the OT sender) and `rcv` (the OT receiver).
// SoftSpoken runs in its maliciously secure mode (CSW base OT + consistency check) unless --semi-honest
// is given. For the pair (i, j), i < j, party i listens and party j connects; pairs are connected in
// lexicographic order, which makes the start-up deadlock free.
//
// OT product (Gilboa, vectorized), exactly as in ../secureml/include/link.h. The "sender" holds, for each
// of J items, a vector A[j][0..L); the "receiver" holds one scalar b[j] per item. Afterwards the sender
// holds S and the receiver R with  S[j][k] + R[j][k] = A[j][k] * b[j]  (mod p).  Per item: 127 random
// OTs, one per bit of b[j]; the OT output is expanded to L field elements with AES counter mode, so one
// OT serves all L messages that are multiplied by the same bit.
#ifndef SPZ_MESH_H
#define SPZ_MESH_H

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

namespace spz {

using emp::block;
using emp::NetIO;

inline constexpr int kMaxParties = 8;
inline constexpr int kMaxWorkers = 16;  // port stride per pair: 1 main + kMaxWorkers workers

struct Worker {
  std::unique_ptr<NetIO> io;
  std::unique_ptr<emp::SoftSpoken<4>> snd, rcv;  // this party as OT sender / OT receiver
  uint64_t rot_count = 0;                         // OTs consumed (for the report)
};

struct PeerLink {
  std::unique_ptr<NetIO> main;
  std::vector<Worker> W;
  int nw() const { return (int)W.size(); }
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

struct Mesh {
  int me = 0, n = 0;
  bool mal = true;
  std::vector<PeerLink> peers;   // indexed by party id; peers[me] is empty
  uint64_t rounds = 0;           // protocol rounds on the main channels (one per all-parties exchange)

  static int port_of(int base, int n, int i, int j, int w) { return base + (i * n + j) * (1 + kMaxWorkers) + w; }
  static int max_port(int base, int n) { return port_of(base, n, n - 1, n - 1, kMaxWorkers); }

  // hosts[i] = address of party i. The pair channels are created in lexicographic pair order.
  Mesh(int me_, const std::vector<std::string>& hosts, int base_port, int nworkers, bool malicious)
      : me(me_), n((int)hosts.size()), mal(malicious) {
    if (n < 2 || n > kMaxParties) throw std::invalid_argument("number of parties out of range");
    if (nworkers < 1 || nworkers > kMaxWorkers) throw std::invalid_argument("--threads must be in [1, 16]");
    peers.resize((size_t)n);
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) {
        if (i != me && j != me) continue;
        const int peer = i == me ? j : i;
        const bool listener = (i == me);  // the lower id listens
        auto make = [&](int w) -> std::unique_ptr<NetIO> {
          const int p = port_of(base_port, n, i, j, w);
          return listener ? NetIO::listen(p, true) : NetIO::connect(hosts[(size_t)i].c_str(), p, true);
        };
        PeerLink& L = peers[(size_t)peer];
        L.main = make(0);
        // both sides use the smaller of the two worker counts (the OT connections pair up 1:1)
        int32_t mine = nworkers, theirs = 0;
        if (listener) { L.main->send_data(&mine, 4); L.main->flush(); L.main->recv_data(&theirs, 4); }
        else { L.main->recv_data(&theirs, 4); L.main->send_data(&mine, 4); L.main->flush(); }
        const int nw = std::max(1, std::min<int>(mine, theirs));
        L.W.resize((size_t)nw);
        for (int w = 0; w < nw; ++w) {
          L.W[(size_t)w].io = make(1 + w);
          L.W[(size_t)w].snd = std::make_unique<emp::SoftSpoken<4>>(emp::ALICE, L.W[(size_t)w].io.get(), malicious);
          L.W[(size_t)w].rcv = std::make_unique<emp::SoftSpoken<4>>(emp::BOB, L.W[(size_t)w].io.get(), malicious);
        }
      }
    // Every pair must use the same number of workers: agree on the minimum over all pairs of this party.
  }

  int nw() const {
    int m = kMaxWorkers;
    for (int p = 0; p < n; ++p) if (p != me) m = std::min(m, peers[(size_t)p].nw());
    return m;
  }

  // ---- counters --------------------------------------------------------------------------------
  struct Snapshot {
    uint64_t sent[kMaxParties] = {0}, recv[kMaxParties] = {0};
    uint64_t rounds = 0, rots = 0;
    std::vector<uint64_t> ot_flights;  // per worker channel
  };
  Snapshot snapshot() const {
    Snapshot s;
    s.rounds = rounds;
    for (int p = 0; p < n; ++p) {
      if (p == me) continue;
      const PeerLink& L = peers[(size_t)p];
      s.sent[p] = L.main->send_counter; s.recv[p] = L.main->recv_counter;
      for (const auto& w : L.W) {
        s.sent[p] += w.io->send_counter; s.recv[p] += w.io->recv_counter;
        s.rots += w.rot_count;
        s.ot_flights.push_back(w.io->rounds);
      }
    }
    return s;
  }

  // ---- the all-parties exchange: one protocol round --------------------------------------------
  // send[p] / recv[p]: the bytes this party sends to / expects from party p (either may be empty).
  // Small exchanges are sent to everybody and then received (the messages fit in the socket buffers);
  // large ones use one thread per peer, with the lower id sending first on each pair, so no two
  // parties are ever blocked in send() on the same channel.
  struct Buf { const void* p = nullptr; size_t len = 0; };
  struct RBuf { void* p = nullptr; size_t len = 0; };
  void exchange(const std::vector<Buf>& send, const std::vector<RBuf>& recv) {
    ++rounds;
    size_t tot = 0;
    for (int p = 0; p < n; ++p) if (p != me) tot += send[(size_t)p].len + recv[(size_t)p].len;
    if (tot <= (1u << 15)) {
      for (int p = 0; p < n; ++p)
        if (p != me && send[(size_t)p].len) { peers[(size_t)p].main->send_data(send[(size_t)p].p, (int64_t)send[(size_t)p].len); peers[(size_t)p].main->flush(); }
      for (int p = 0; p < n; ++p)
        if (p != me && recv[(size_t)p].len) peers[(size_t)p].main->recv_data(recv[(size_t)p].p, (int64_t)recv[(size_t)p].len);
      return;
    }
    parallel(n - 1, [&](int idx) {
      const int p = idx < me ? idx : idx + 1;
      auto& io = *peers[(size_t)p].main;
      const Buf& s = send[(size_t)p];
      const RBuf& r = recv[(size_t)p];
      auto do_send = [&] { if (s.len) { io.send_data(s.p, (int64_t)s.len); io.flush(); } };
      auto do_recv = [&] { if (r.len) io.recv_data(r.p, (int64_t)r.len); };
      if (me < p) { do_send(); do_recv(); } else { do_recv(); do_send(); }
    });
  }
  // The same bytes to every peer; the peers' buffers of equal length are received.
  void all_to_all(const void* mine, size_t len, std::vector<std::vector<uint8_t>>& theirs) {
    theirs.assign((size_t)n, {});
    std::vector<Buf> s((size_t)n);
    std::vector<RBuf> r((size_t)n);
    for (int p = 0; p < n; ++p) {
      if (p == me) continue;
      s[(size_t)p] = {mine, len};
      theirs[(size_t)p].resize(len);
      r[(size_t)p] = {theirs[(size_t)p].data(), len};
    }
    exchange(s, r);
  }
};

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

// OT product over ONE peer link, spread over that link's workers (items when J >= nw, else columns).
// Both ends derive the same split. Sender: A (J x L) -> S. Receiver: b (J) -> R (J x L).
inline void product_send(PeerLink& lk, int64_t J, int64_t L, const Fp* A, Fp* S) {
  const int nw = lk.nw();
  if (J >= nw) {
    parallel(nw, [&](int w) {
      auto [lo, hi] = split_range(J, nw, w);
      prod_send(lk.W[(size_t)w], hi - lo, L, A + lo * L, L, S + lo * L, L);
    });
  } else {
    parallel(nw, [&](int w) {
      auto [lo, hi] = split_range(L, nw, w);
      prod_send(lk.W[(size_t)w], J, hi - lo, A + lo, L, S + lo, L);
    });
  }
}
inline void product_recv(PeerLink& lk, int64_t J, int64_t L, const Fp* b, Fp* R) {
  const int nw = lk.nw();
  if (J >= nw) {
    parallel(nw, [&](int w) {
      auto [lo, hi] = split_range(J, nw, w);
      prod_recv(lk.W[(size_t)w], hi - lo, L, b + lo, R + lo * L, L);
    });
  } else {
    parallel(nw, [&](int w) {
      auto [lo, hi] = split_range(L, nw, w);
      prod_recv(lk.W[(size_t)w], J, hi - lo, b, R + lo, L);
    });
  }
}

}  // namespace spz

#endif  // SPZ_MESH_H
