// lr_fss.cc -- benchmark driver for "Communication-Efficient Secure Logistic Regression" (Agarwal, Peceny, Raykova,
// Schoppmann, Seth; EuroS&P 2024) using the authors' code, google/fss_machine_learning, UNMODIFIED:
//   applications/logistic_regression  (GradientDescentParty{Zero,One}: SGD, matrix products with correlated Beaver triples)
//   applications/secure_sigmoid       (v1: FSS-based MIC gate sigmoid, 4 rounds, trusted-dealer offline phase)
//   applications/secure_sigmoid_new_mic (v2: constant-round comparison instead of MIC, 6 rounds, offline phase that can be
//                                       done by 2PC)
//
// Setting. Party 0 = DATA OWNER: holds the plaintext table (X, y) and wants to hide it. Party 1 = ANALYST: holds nothing and
// ends up with the model. The owner secret-shares its data (the analyst's share R is uniformly random and independent of
// the data), then both parties run the paper's protocol on the shares. The library is only an in-process API, so this
// driver supplies what the library leaves out:
//   * two parties as separate threads that communicate ONLY through a byte channel carrying the library's real protobuf
//     round messages (Channel below); the channel counts bytes and rounds and can emulate a WAN (RTT, bandwidth);
//   * the trusted-dealer offline phase. The dealer is the data owner: the analyst must not be the dealer (it would know the
//     masks A and could unmask X - A), and a dealer that is the owner leaks nothing about X to the analyst;
//   * per-party, per-phase accounting of CPU time (all threads), wall time, wait time, bytes and rounds;
//   * parallelism: the sigmoid batch is cut into chunks that are processed on all cores (every chunk has its own
//     preprocessing; the per-round messages of all chunks travel in ONE frame, so the round count is unchanged), and the
//     dealer's work runs on all cores as well.
//
// Not changed on purpose: the library's own matrix products (single-threaded, O(n d) per iteration).
#include <errno.h>
#include <sys/random.h>
#include <sys/resource.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <thread>
#include <vector>

#include "applications/logistic_regression/gradient_descent.h"
#include "applications/secure_sigmoid/secure_sigmoid.h"
#include "applications/secure_sigmoid_new_mic/secure_sigmoid_new_mic.h"
#include "poisson_regression/beaver_triple_utils.h"
#include "secret_sharing_mpc/gates/correlated_matrix_product.h"

namespace pjc = private_join_and_compute;
namespace lr = private_join_and_compute::logistic_regression;
namespace app = private_join_and_compute::applications;

namespace {

using Bytes = std::vector<uint8_t>;

[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "lr_fss: %s\n", m.c_str());
  std::fflush(stderr);
  std::_Exit(2);
}
template <class T>
T ok(absl::StatusOr<T> s, const char* what) {
  if (!s.ok()) die(std::string(what) + ": " + std::string(s.status().message()));
  return std::move(s).value();
}
void ok_s(const absl::Status& s, const char* what) {
  if (!s.ok()) die(std::string(what) + ": " + std::string(s.message()));
}

double wall_now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
uint64_t thread_cpu_ns() {
  timespec ts;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// ------------------------------------------------------------------------------------------------ configuration
struct Cfg {
  int variant = 1;             // 1: MIC sigmoid (trusted offline), 2: comparison sigmoid (new MIC, 2PC-friendly offline)
  size_t batch = 4;            // rows per iteration (minibatch)
  size_t n = 0;                // rows in the owner's table = iters * batch (set in main)
  size_t d = 5;                // features INCLUDING the bias column
  size_t iters = 20;           // iterations, each on its own minibatch
  double alpha = 1.0, lambda = 0.0001;
  int frac = 20, ring = 63, exp_bound = 13;
  size_t block_length = 16, num_splits = 4;  // v2 only
  int threads = 0;             // per party, online (0 = hardware_concurrency / 2)
  int dealer_threads = 0;      // offline, owner only (0 = hardware_concurrency)
  size_t min_chunk = 64;       // smallest sigmoid chunk
  double rtt_ms = 0, bw_mbps = 0;
  uint64_t seed = 1;
  bool verify = true;
  std::string json;
};

// ------------------------------------------------------------------------------------------------ channel
// Two endpoints connected by a simulated link. send() never blocks; the message becomes visible to the receiver at
//   deliver = max(now, link_free) + bytes/bandwidth + rtt/2,
// and recv() blocks until it is delivered. Hence a symmetric exchange (both send, then both receive) costs rtt/2 of latency
// per round and a one-way transfer costs rtt/2.
struct Net {
  struct Msg { Bytes data; double at; };
  struct Dir {
    std::mutex m;
    std::condition_variable cv;
    std::deque<Msg> q;
    double link_free = 0;
  } dir[2];  // dir[i] = messages travelling TO party i
  double one_way_s = 0, bytes_per_s = 0;
};

struct Endpoint {
  int id;
  Net* net;
  double wait_peer_s = 0;  // blocked waiting for the peer to send (peer compute + its own wait)
  double wait_net_s = 0;   // slept for the modelled latency / transfer time
  uint64_t sent = 0, recv = 0, rounds = 0;

  void send(const Bytes& b) {
    Net::Dir& to = net->dir[1 - id];
    const double now = wall_now();
    std::lock_guard<std::mutex> g(to.m);
    double start = std::max(now, to.link_free);
    if (net->bytes_per_s > 0) start += (double)b.size() / net->bytes_per_s;
    to.link_free = start;
    to.q.push_back({b, start + net->one_way_s});
    sent += b.size();
    to.cv.notify_one();
  }
  Bytes recv_msg() {
    Net::Dir& me = net->dir[id];
    const double t0 = wall_now();
    Net::Msg m;
    {
      std::unique_lock<std::mutex> lk(me.m);
      me.cv.wait(lk, [&] { return !me.q.empty(); });
      m = std::move(me.q.front());
      me.q.pop_front();
    }
    const double t1 = wall_now();
    if (m.at > t1) std::this_thread::sleep_for(std::chrono::duration<double>(m.at - t1));
    const double t2 = wall_now();
    wait_peer_s += t1 - t0;
    wait_net_s += t2 - t1;
    recv += m.data.size();
    return std::move(m.data);
  }
  // Symmetric exchange: 1 round.
  Bytes exchange(const Bytes& b) {
    send(b);
    Bytes r = recv_msg();
    ++rounds;
    return r;
  }
  // One-way transfer: 1 round for both ends.
  void send_oneway(const Bytes& b) { send(b); ++rounds; }
  Bytes recv_oneway() { Bytes r = recv_msg(); ++rounds; return r; }
};

struct Barrier {
  std::mutex m;
  std::condition_variable cv;
  int n, count = 0, gen = 0;
  explicit Barrier(int n_) : n(n_) {}
  void wait() {
    std::unique_lock<std::mutex> lk(m);
    const int g = gen;
    if (++count == n) { count = 0; ++gen; cv.notify_all(); return; }
    cv.wait(lk, [&] { return g != gen; });
  }
};

// ------------------------------------------------------------------------------------------------ accounting
struct Phase {
  double wall = 0, cpu = 0, wait_peer = 0, wait_net = 0;
  uint64_t sent = 0, recv = 0, rounds = 0;
};

struct Party {
  int id;
  Endpoint ep;
  int threads = 1;
  std::atomic<uint64_t> worker_cpu_ns{0};  // CPU of helper threads started by parallel_for
  Phase off, on;
  std::vector<uint64_t> theta_share;
  std::vector<uint64_t> dbg_u, dbg_s;  // iteration-0 sigmoid input/output shares (harness-only check)

  struct Mark { double wall; uint64_t cpu; double wp, wn; uint64_t sent, recv, rounds; };
  Mark mark() {
    return {wall_now(), thread_cpu_ns() + worker_cpu_ns.load(), ep.wait_peer_s, ep.wait_net_s, ep.sent, ep.recv, ep.rounds};
  }
  void end(Phase& p, const Mark& m) {
    const Mark n = mark();
    p.wall += n.wall - m.wall;
    p.cpu += (double)(n.cpu - m.cpu) * 1e-9;
    p.wait_peer += n.wp - m.wp;
    p.wait_net += n.wn - m.wn;
    p.sent += n.sent - m.sent;
    p.recv += n.recv - m.recv;
    p.rounds += n.rounds - m.rounds;
  }
};

// Runs f(0..count-1) on up to `threads` threads (the caller is one of them). Helper-thread CPU is added to the party.
template <class F>
void parallel_for(Party& P, int threads, size_t count, F f) {
  const size_t nt = std::max<size_t>(1, std::min<size_t>((size_t)threads, count));
  std::atomic<size_t> next{0};
  auto loop = [&] {
    for (size_t i; (i = next.fetch_add(1)) < count;) f(i);
  };
  if (nt == 1) { loop(); return; }
  std::vector<std::thread> ts;
  for (size_t t = 1; t < nt; ++t)
    ts.emplace_back([&] {
      const uint64_t c0 = thread_cpu_ns();
      loop();
      P.worker_cpu_ns += thread_cpu_ns() - c0;
    });
  loop();
  for (auto& t : ts) t.join();
}

// ------------------------------------------------------------------------------------------------ framing
template <class M>
Bytes pack(const std::vector<M>& v) {
  Bytes out;
  auto put64 = [&](uint64_t x) { for (int i = 0; i < 8; ++i) out.push_back((uint8_t)(x >> (8 * i))); };
  put64(v.size());
  for (const M& m : v) {
    std::string s;
    if (!m.SerializeToString(&s)) die("serialize");
    put64(s.size());
    out.insert(out.end(), s.begin(), s.end());
  }
  return out;
}
template <class M>
std::vector<M> unpack(const Bytes& b) {
  size_t pos = 0;
  auto get64 = [&]() { if (pos + 8 > b.size()) die("short frame"); uint64_t x = 0; for (int i = 0; i < 8; ++i) x |= (uint64_t)b[pos + i] << (8 * i); pos += 8; return x; };
  std::vector<M> v((size_t)get64());
  for (M& m : v) {
    const size_t len = (size_t)get64();
    if (pos + len > b.size() || !m.ParseFromArray(b.data() + pos, (int)len)) die("parse");
    pos += len;
  }
  return v;
}
// One round: every chunk computes its message with make(c) (in parallel), all go in one frame, the peer's come back.
template <class M, class F>
std::vector<M> round_xchg(Party& P, size_t chunks, F make) {
  std::vector<M> mine(chunks);
  parallel_for(P, P.threads, chunks, [&](size_t c) { mine[c] = make(c); });
  return unpack<M>(P.ep.exchange(pack(mine)));
}
template <class M>
M xchg_one(Party& P, const M& m) {
  std::vector<M> v = unpack<M>(P.ep.exchange(pack(std::vector<M>{m})));
  if (v.size() != 1) die("expected one message");
  return std::move(v[0]);
}

// ------------------------------------------------------------------------------------------------ randomness, data
void fill_random(uint64_t* p, size_t count) {
  uint8_t* b = reinterpret_cast<uint8_t*>(p);
  size_t left = count * 8;
  while (left) {
    ssize_t r = getrandom(b, std::min<size_t>(left, 1 << 20), 0);
    if (r < 0) { if (errno == EINTR) continue; die("getrandom"); }
    b += r;
    left -= (size_t)r;
  }
}

struct SplitMix64 {
  uint64_t s;
  explicit SplitMix64(uint64_t seed) : s(seed) {}
  uint64_t next() {
    uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  double unit() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }  // [0,1)
};

// The owner's table. x is quantised to 2^-frac; the last column is the constant 1 (bias). Labels come from a hidden
// logistic model so that the trained weights are meaningful.
struct Data {
  std::vector<int64_t> x;  // n*d, scale 2^frac
  std::vector<int64_t> y;  // n, 0 or 2^frac
};
Data generate_data(const Cfg& c) {
  Data D;
  D.x.resize(c.n * c.d);
  D.y.resize(c.n);
  SplitMix64 rng(c.seed * 0x2545F4914F6CDD1DULL + 1), rw(c.seed * 0x9E3779B97F4A7C15ULL + 2);
  std::vector<double> wstar(c.d);
  const double sc = 3.0 / std::sqrt((double)c.d);
  for (auto& w : wstar) w = (2 * rw.unit() - 1) * sc * 2;
  const double one = (double)(1LL << c.frac);
  for (size_t r = 0; r < c.n; ++r) {
    double z = 0;
    for (size_t j = 0; j < c.d; ++j) {
      const int64_t q = (j + 1 == c.d) ? (int64_t)one : std::llround((2 * rng.unit() - 1) * one);
      D.x[r * c.d + j] = q;
      z += (double)q / one * wstar[j];
    }
    D.y[r] = (rng.unit() < 1.0 / (1.0 + std::exp(-z))) ? (int64_t)one : 0;
  }
  return D;
}

inline uint64_t to_ring(int64_t v, uint64_t mod) { return (uint64_t)v & (mod - 1); }  // mod = 2^ring
inline double from_ring(uint64_t v, uint64_t mod, int frac) {
  int64_t s = (v >= mod / 2) ? (int64_t)v - (int64_t)mod : (int64_t)v;
  return (double)s / (double)(1LL << frac);
}

// Plaintext twin: double-precision full-batch SGD with the exact sigmoid; same update rule as the library.
std::vector<double> reference_train(const Cfg& c, const Data& D) {
  const double one = (double)(1LL << c.frac);
  const size_t B = c.batch;
  std::vector<double> th(c.d, 0.0), g(c.d);
  for (size_t t = 0; t < c.iters; ++t) {
    std::fill(g.begin(), g.end(), 0.0);
    for (size_t r = t * B; r < (t + 1) * B; ++r) {
      double z = 0;
      for (size_t j = 0; j < c.d; ++j) z += (double)D.x[r * c.d + j] / one * th[j];
      const double e = 1.0 / (1.0 + std::exp(-z)) - (double)D.y[r] / one;
      for (size_t j = 0; j < c.d; ++j) g[j] += (double)D.x[r * c.d + j] / one * e;
    }
    for (size_t j = 0; j < c.d; ++j) th[j] -= c.alpha / (double)B * g[j] + c.alpha * c.lambda / (double)B * th[j];
  }
  return th;
}

// ------------------------------------------------------------------------------------------------ sigmoid variants
const std::vector<double> kLo = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
const std::vector<double> kHi = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
const std::vector<double> kSlope = {0.24979187478940013, 0.24854809833537939, 0.24608519499181072, 0.24245143300792976,
                                    0.23771671089402596, 0.23196975023940808, 0.2253146594237077,  0.2178670895944635,
                                    0.20975021497391394, 0.2010907600500101};
const std::vector<double> kInter = {0.5, 0.5001243776454021, 0.5006169583141158, 0.5017070869092801, 0.5036009757548416,
                                    0.5064744560821506, 0.5104675105715708, 0.5156808094520418, 0.5221743091484814,
                                    0.5299678185799949};
constexpr size_t kLogGroup = 63, kIntervals = 10, kTaylor = 10;
constexpr uint64_t kLargePrime = 9223372036854775783ULL;

app::SecureSplineParameters spline_params(const Cfg& c) {
  return app::SecureSplineParameters{kLogGroup, kIntervals, (uint8_t)c.frac, kLo, kHi, kSlope, kInter};
}
pjc::ExponentiationParams exp_params(const Cfg& c) { return pjc::ExponentiationParams{(uint8_t)c.exp_bound, kLargePrime}; }

struct V1 {
  using Sig = app::SecureSigmoid;
  using Pre = app::SigmoidPrecomputedValue;
  static std::unique_ptr<Sig> create(size_t n, const Cfg& c) {
    app::SecureSigmoidParameters p{kLogGroup, spline_params(c), (uint8_t)c.frac, kTaylor, exp_params(c)};
    return ok(Sig::Create(n, p), "SecureSigmoid::Create");
  }
  static std::pair<Pre, Pre> precompute(Sig& s) { return ok(s.PerformSigmoidPrecomputation(), "sigmoid precompute"); }
  static constexpr int kRounds = 4;

  // 4 exchanged messages, then a local result. in[c] = this party's share of the sigmoid inputs of chunk c.
  static std::vector<std::vector<uint64_t>> run(Party& P, const Cfg&, std::vector<std::unique_ptr<Sig>>& sig,
                                                std::vector<Pre>& pre, std::vector<std::vector<uint64_t>>& in) {
    const size_t C = in.size();
    std::vector<app::RoundOneSigmoidState> s1(C);
    std::vector<app::RoundTwoSigmoidState> s2(C);
    std::vector<app::RoundThreeSigmoidState> s3(C);
    std::vector<app::RoundFourSigmoidState> s4(C);
    auto m1 = round_xchg<app::RoundOneSigmoidMessage>(P, C, [&](size_t c) {
      auto r = ok(sig[c]->GenerateSigmoidRoundOneMessage(P.id, pre[c], in[c]), "sigmoid r1");
      s1[c] = std::move(r.first);
      return std::move(r.second);
    });
    auto m2 = round_xchg<app::RoundTwoSigmoidMessage>(P, C, [&](size_t c) {
      auto r = ok(sig[c]->GenerateSigmoidRoundTwoMessage(P.id, pre[c], s1[c], m1[c]), "sigmoid r2");
      s2[c] = std::move(r.first);
      return std::move(r.second);
    });
    auto m3 = round_xchg<app::RoundThreeSigmoidMessage>(P, C, [&](size_t c) {
      auto r = ok(sig[c]->GenerateSigmoidRoundThreeMessage(P.id, pre[c], s2[c], m2[c]), "sigmoid r3");
      s3[c] = std::move(r.first);
      return std::move(r.second);
    });
    auto m4 = round_xchg<app::RoundFourSigmoidMessage>(P, C, [&](size_t c) {
      auto r = ok(sig[c]->GenerateSigmoidRoundFourMessage(P.id, pre[c], s3[c], m3[c]), "sigmoid r4");
      s4[c] = std::move(r.first);
      return std::move(r.second);
    });
    std::vector<std::vector<uint64_t>> out(C);
    parallel_for(P, P.threads, C, [&](size_t c) {
      out[c] = ok(sig[c]->GenerateSigmoidResult(P.id, pre[c], s4[c], m4[c]), "sigmoid result");
    });
    return out;
  }
};

struct V2 {
  using Sig = app::SecureSigmoidNewMic;
  using Pre = app::SigmoidPrecomputedValueNewMic;
  static std::unique_ptr<Sig> create(size_t n, const Cfg& c) {
    app::SecureSigmoidNewMicParameters p{kLogGroup, spline_params(c), (uint8_t)c.frac, kTaylor, exp_params(c),
                                         c.block_length, c.num_splits};
    return ok(Sig::Create(n, p), "SecureSigmoidNewMic::Create");
  }
  static std::pair<Pre, Pre> precompute(Sig& s) { return ok(s.PerformSigmoidPrecomputation(), "sigmoid precompute"); }
  static constexpr int kRounds = 6;

  // 6 exchanged messages (rounds 1, 2, 3, 3.5, 4, 5), then a local result.
  static std::vector<std::vector<uint64_t>> run(Party& P, const Cfg&, std::vector<std::unique_ptr<Sig>>& sig,
                                                std::vector<Pre>& pre, std::vector<std::vector<uint64_t>>& in) {
    const size_t C = in.size();
    std::vector<app::RoundOneSigmoidNewMicState> s1(C);
    std::vector<app::RoundTwoSigmoidNewMicState> s2(C);
    std::vector<app::RoundThreeSigmoidNewMicState> s3(C);
    std::vector<app::RoundThreePointFiveSigmoidNewMicState> s35(C);
    std::vector<app::RoundFourSigmoidNewMicState> s4(C);
    std::vector<app::RoundFiveSigmoidNewMicState> s5(C);
    auto m1 = round_xchg<app::RoundOneSigmoidNewMicMessage>(P, C, [&](size_t c) {
      auto r = ok(sig[c]->GenerateSigmoidRoundOneMessage(P.id, pre[c], in[c]), "sigmoid r1");
      s1[c] = std::move(r.first);
      return std::move(r.second);
    });
    auto m2 = round_xchg<app::RoundTwoSigmoidNewMicMessage>(P, C, [&](size_t c) {
      auto r = ok(sig[c]->GenerateSigmoidRoundTwoMessage(P.id, pre[c], s1[c], m1[c]), "sigmoid r2");
      s2[c] = std::move(r.first);
      return std::move(r.second);
    });
    auto m3 = round_xchg<app::RoundThreeSigmoidNewMicMessage>(P, C, [&](size_t c) {
      auto r = ok(sig[c]->GenerateSigmoidRoundThreeMessage(P.id, pre[c], s2[c], m2[c]), "sigmoid r3");
      s3[c] = std::move(r.first);
      return std::move(r.second);
    });
    auto m35 = round_xchg<app::RoundThreePointFiveSigmoidNewMicMessage>(P, C, [&](size_t c) {
      auto r = ok(sig[c]->GenerateSigmoidRoundThreePointFiveMessage(P.id, pre[c], s3[c], m3[c]), "sigmoid r3.5");
      s35[c] = std::move(r.first);
      return std::move(r.second);
    });
    auto m4 = round_xchg<app::RoundFourSigmoidNewMicMessage>(P, C, [&](size_t c) {
      auto r = ok(sig[c]->GenerateSigmoidRoundFourMessage(P.id, pre[c], s35[c], m35[c]), "sigmoid r4");
      s4[c] = std::move(r.first);
      return std::move(r.second);
    });
    auto m5 = round_xchg<app::RoundFiveSigmoidNewMicMessage>(P, C, [&](size_t c) {
      auto r = ok(sig[c]->GenerateSigmoidRoundFiveMessage(P.id, pre[c], s4[c], m4[c]), "sigmoid r5");
      s5[c] = std::move(r.first);
      return std::move(r.second);
    });
    std::vector<std::vector<uint64_t>> out(C);
    parallel_for(P, P.threads, C, [&](size_t c) {
      out[c] = ok(sig[c]->GenerateSigmoidResult(P.id, pre[c], s5[c], m5[c]), "sigmoid result");
    });
    return out;
  }
};

// ------------------------------------------------------------------------------------------------ the protocol
using Mat = std::pair<std::vector<uint64_t>, std::vector<uint64_t>>;  // (B, C) of one Beaver matrix triple

struct Chunks {
  size_t base = 0, count = 0;  // chunk c = rows [c*base, min(n, (c+1)*base))
  size_t lo(size_t c) const { return c * base; }
  size_t size(size_t c, size_t n) const { return std::min(n, (c + 1) * base) - c * base; }
};
Chunks plan_chunks(const Cfg& c, int max_threads) {
  Chunks k;
  const size_t want = std::max<size_t>(1, std::min<size_t>((size_t)max_threads, (c.batch + c.min_chunk - 1) / c.min_chunk));
  k.base = (c.batch + want - 1) / want;
  k.count = (c.batch + k.base - 1) / k.base;
  return k;
}

// What the dealer hands to the analyst without serialisation (the library's sigmoid preprocessing is a C++ struct).
template <class V>
struct SigmoidHandoff {
  std::vector<std::vector<typename V::Pre>> pre;  // [iteration][chunk]
};

struct Shared {
  Cfg cfg;
  Chunks chunks;
  int dealer_threads = 1;
  Barrier bar{2};
  void* p1_sigmoid = nullptr;  // SigmoidHandoff<V>*
};

std::vector<uint64_t> to_u64(const Bytes& b) {
  if (b.size() % 8) die("blob size");
  std::vector<uint64_t> v(b.size() / 8);
  std::memcpy(v.data(), b.data(), b.size());
  return v;
}

// Frames that carry several messages of different types in one exchange (one round).
Bytes join(const std::vector<Bytes>& parts) {
  Bytes out;
  for (const Bytes& p : parts) {
    for (int i = 0; i < 8; ++i) out.push_back((uint8_t)((uint64_t)p.size() >> (8 * i)));
    out.insert(out.end(), p.begin(), p.end());
  }
  return out;
}
std::vector<Bytes> split(const Bytes& b, size_t parts) {
  std::vector<Bytes> r;
  size_t pos = 0;
  for (size_t k = 0; k < parts; ++k) {
    if (pos + 8 > b.size()) die("short joined frame");
    uint64_t l = 0;
    for (int i = 0; i < 8; ++i) l |= (uint64_t)b[pos + i] << (8 * i);
    pos += 8;
    if (pos + l > b.size()) die("bad joined frame");
    r.emplace_back(b.begin() + pos, b.begin() + pos + l);
    pos += l;
  }
  return r;
}

// Training: c.iters iterations, each on its OWN minibatch of c.batch rows (rows [t*B, (t+1)*B) of the owner's table).
// Every iteration is one run of the library's full-batch step on B rows, with fresh Beaver masks A for that batch (reusing
// a mask across different batches would reveal the difference of the batches), and the model share carried over.
template <class V, class GD>
void run_party(Party& P, Shared& S, const Data* data) {
  const Cfg& c = S.cfg;
  const size_t B = c.batch, T = c.iters;
  const uint64_t mod = 1ULL << c.ring;
  const lr::GradientDescentParams gdp = {B, c.d, 1, (uint8_t)c.ring, (uint8_t)c.frac, c.alpha, c.lambda, mod};
  const pjc::FixedPointElementFactory::Params fpe = {(uint8_t)c.frac, (uint8_t)c.ring, 1ULL << c.frac,
                                                    1ULL << (c.ring - c.frac), 1ULL << c.ring};
  const size_t bd = B * c.d;
  const size_t C = S.chunks.count;

  S.bar.wait();
  // ================================================================== OFFLINE
  Party::Mark m0 = P.mark();
  std::vector<uint64_t> share_x, share_y;      // all n rows
  std::vector<uint64_t> theta0;                // this party's share of the initial model: a RANDOM sharing of zero (the library's
                                               // local truncation needs random shares; an all-zero share breaks it)
  std::vector<std::vector<uint64_t>> a1(T), a3(T);  // per-iteration masks
  std::vector<Mat> bc1, bc3;                   // per-iteration (B, C)
  SigmoidHandoff<V> mine;                      // this party's sigmoid preprocessing, [iteration][chunk]
  SigmoidHandoff<V>* theirs = nullptr;

  if (P.id == 0) {
    // ---- dealer (data owner). Everything here uses all dealer threads.
    SigmoidHandoff<V>* p1 = new SigmoidHandoff<V>();
    mine.pre.assign(T, std::vector<typename V::Pre>(C));
    p1->pre.assign(T, std::vector<typename V::Pre>(C));
    parallel_for(P, S.dealer_threads, T * C, [&](size_t task) {
      thread_local std::map<size_t, std::unique_ptr<typename V::Sig>> inst;  // one generator per thread and chunk size
      const size_t it = task / C, ch = task % C;
      const size_t sz = S.chunks.size(ch, B);
      auto& g = inst[sz];
      if (!g) g = V::create(sz, c);
      auto pr = V::precompute(*g);
      mine.pre[it][ch] = std::move(pr.first);
      p1->pre[it][ch] = std::move(pr.second);
    });
    // Matrix triples per iteration (the library's own samplers): A for X and X^T, then (B, C) for each product.
    std::vector<std::pair<Mat, Mat>> t1(T), t3(T);
    std::vector<std::vector<uint64_t>> a1p1(T), a3p1(T);
    parallel_for(P, S.dealer_threads, T, [&](size_t it) {
      auto ta1 = ok(pjc::SampleBeaverTripleMatrixA(B, c.d, mod), "triple A (X)");
      auto ta3 = ok(pjc::SampleBeaverTripleMatrixA(c.d, B, mod), "triple A (X^T)");
      t1[it] = ok(pjc::SampleBeaverTripleMatrixBandC(ta1, B, c.d, 1, mod), "triple B,C (round 1)");
      t3[it] = ok(pjc::SampleBeaverTripleMatrixBandC(ta3, c.d, B, 1, mod), "triple B,C (round 3)");
      a1[it] = std::get<0>(ta1); a1p1[it] = std::get<1>(ta1);
      a3[it] = std::get<0>(ta3); a3p1[it] = std::get<1>(ta3);
    });
    for (size_t it = 0; it < T; ++it) { bc1.push_back(t1[it].first); bc3.push_back(t3[it].first); }
    // ---- input sharing: the analyst's share R is uniform and independent of the data; the owner keeps data - R.
    const size_t nd = c.n * c.d;
    std::vector<uint64_t> rx(nd), ry(c.n);
    fill_random(rx.data(), nd);
    fill_random(ry.data(), c.n);
    share_x.resize(nd);
    share_y.resize(c.n);
    for (size_t i = 0; i < nd; ++i) { rx[i] &= mod - 1; share_x[i] = (to_ring(data->x[i], mod) - rx[i]) & (mod - 1); }
    for (size_t i = 0; i < c.n; ++i) { ry[i] &= mod - 1; share_y[i] = (to_ring(data->y[i], mod) - ry[i]) & (mod - 1); }
    // ---- hand the analyst its material: R and its triple shares (serialised: real offline traffic) ...
    std::vector<uint64_t> blob;
    auto app_v = [&](const std::vector<uint64_t>& v) { blob.insert(blob.end(), v.begin(), v.end()); };
    theta0.resize(c.d);
    fill_random(theta0.data(), c.d);
    std::vector<uint64_t> th1(c.d);
    for (size_t j = 0; j < c.d; ++j) { theta0[j] &= mod - 1; th1[j] = (mod - theta0[j]) & (mod - 1); }
    app_v(rx); app_v(ry); app_v(th1);
    for (size_t it = 0; it < T; ++it) {
      app_v(a1p1[it]); app_v(a3p1[it]);
      app_v(t1[it].second.first); app_v(t1[it].second.second);
      app_v(t3[it].second.first); app_v(t3[it].second.second);
    }
    // ... and, in memory, its sigmoid preprocessing (not serialisable without touching the library; see README).
    S.p1_sigmoid = p1;
    Bytes raw(blob.size() * 8);
    std::memcpy(raw.data(), blob.data(), raw.size());
    P.ep.send_oneway(raw);
  } else {
    // ---- analyst: receive and unpack.
    std::vector<uint64_t> blob = to_u64(P.ep.recv_oneway());
    size_t pos = 0;
    auto take = [&](size_t k) {
      if (pos + k > blob.size()) die("blob too short");
      std::vector<uint64_t> v(blob.begin() + pos, blob.begin() + pos + k);
      pos += k;
      return v;
    };
    share_x = take(c.n * c.d);
    share_y = take(c.n);
    theta0 = take(c.d);
    for (size_t it = 0; it < T; ++it) {
      a1[it] = take(bd);
      a3[it] = take(bd);
      Mat m1; m1.first = take(c.d); m1.second = take(B); bc1.push_back(std::move(m1));
      Mat m3; m3.first = take(B); m3.second = take(c.d); bc3.push_back(std::move(m3));
    }
    if (pos != blob.size()) die("blob length mismatch");
    theirs = static_cast<SigmoidHandoff<V>*>(S.p1_sigmoid);
  }
  SigmoidHandoff<V>& pre = (P.id == 0) ? mine : *theirs;

  // Sigmoid engines, one per chunk (this party's own).
  std::vector<std::unique_ptr<typename V::Sig>> sig(C);
  parallel_for(P, P.threads, C, [&](size_t ch) { sig[ch] = V::create(S.chunks.size(ch, B), c); });
  P.end(P.off, m0);

  S.bar.wait();
  // ================================================================== ONLINE
  Party::Mark m1 = P.mark();
  std::vector<uint64_t> theta = theta0;  // this party's share of the model (the initial model is zero)

  for (size_t it = 0; it < T; ++it) {
    // Fresh GD instance on this iteration's rows (local setup: share provider, transpose).
    std::vector<uint64_t> sx(share_x.begin() + it * bd, share_x.begin() + (it + 1) * bd);
    std::vector<uint64_t> sy(share_y.begin() + it * B, share_y.begin() + (it + 1) * B);
    std::vector<Mat> b1{bc1[it]}, b3{bc3[it]};
    auto made = (c.variant == 1)
                    ? lr::LogRegShareProvider::Create(a1[it], b1, std::vector<app::SigmoidPrecomputedValue>(1), a3[it], b3, B,
                                                      c.d, 1)
                    : lr::LogRegShareProvider::CreateNewMic(a1[it], b1, std::vector<app::SigmoidPrecomputedValueNewMic>(1),
                                                            a3[it], b3, B, c.d, 1);
    std::unique_ptr<lr::LogRegShareProvider> provider(new lr::LogRegShareProvider(ok(std::move(made), "share provider")));
    std::unique_ptr<GD> gd(new GD(ok(GD::Init(sx, sy, theta, std::move(provider), fpe, gdp), "GD::Init")));

    // Round 1: [X-A], [X^T-A] (this batch) and [theta-B] travel together.
    lr::StateMaskedX st_x;
    lr::StateMaskedXTranspose st_xt;
    pjc::MaskedXMessage xm_mine;
    pjc::MaskedXTransposeMessage xtm_mine;
    std::tie(st_x, xm_mine) = ok(gd->GenerateCorrelatedProductMessageForX(), "X-A");
    std::tie(st_xt, xtm_mine) = ok(gd->GenerateCorrelatedProductMessageForXTranspose(), "X^T-A");
    lr::StateRound1 st1;
    pjc::RoundOneGradientDescentMessage g1_mine;
    std::tie(st1, g1_mine) = ok(gd->GenerateGradientDescentRoundOneMessage(st_x), "GD round 1");
    std::vector<Bytes> got = split(P.ep.exchange(join({pack(std::vector<pjc::MaskedXMessage>{xm_mine}),
                                                      pack(std::vector<pjc::MaskedXTransposeMessage>{xtm_mine}),
                                                      pack(std::vector<pjc::RoundOneGradientDescentMessage>{g1_mine})})),
                                   3);
    auto va = unpack<pjc::MaskedXMessage>(got[0]);
    auto vb = unpack<pjc::MaskedXTransposeMessage>(got[1]);
    auto vc = unpack<pjc::RoundOneGradientDescentMessage>(got[2]);
    if (va.size() != 1 || vb.size() != 1 || vc.size() != 1) die("round 1 frame content");
    lr::SigmoidInput sin = ok(gd->GenerateSigmoidInput(st1, va[0], vc[0]), "sigmoid input");

    // s = sigmoid(u)  (4 or 6 rounds), chunked over the cores
    std::vector<std::vector<uint64_t>> in(C);
    for (size_t ch = 0; ch < C; ++ch) {
      const size_t lo = S.chunks.lo(ch), sz = S.chunks.size(ch, B);
      in[ch].assign(sin.sigmoid_input.begin() + lo, sin.sigmoid_input.begin() + lo + sz);
    }
    std::vector<std::vector<uint64_t>> out = V::run(P, c, sig, pre.pre[it], in);
    lr::SigmoidOutput sout;
    for (auto& o : out) sout.sigmoid_output.insert(sout.sigmoid_output.end(), o.begin(), o.end());
    if (it == 0) { P.dbg_u = sin.sigmoid_input; P.dbg_s = sout.sigmoid_output; }

    // g = X^T (s - y); theta -= alpha/B g + ...  (1 round)
    lr::StateRound3 st3;
    pjc::RoundThreeGradientDescentMessage g3_mine;
    std::tie(st3, g3_mine) = ok(gd->GenerateGradientDescentRoundThreeMessage(sout, st_xt), "GD round 3");
    pjc::RoundThreeGradientDescentMessage g3_peer = xchg_one(P, g3_mine);
    ok_s(gd->ComputeGradientUpdate(st3, vb[0], g3_peer), "gradient update");
    theta = gd->GetTheta();
  }

  // Output: the owner sends its share of theta to the analyst (1 round), who learns the model. The owner learns nothing new.
  P.theta_share = theta;
  if (P.id == 0) {
    Bytes raw(theta.size() * 8);
    std::memcpy(raw.data(), theta.data(), raw.size());
    P.ep.send_oneway(raw);
  } else {
    std::vector<uint64_t> o = to_u64(P.ep.recv_oneway());
    if (o.size() != theta.size()) die("theta size");
    for (size_t j = 0; j < o.size(); ++j) P.theta_share[j] = (theta[j] + o[j]) & (mod - 1);  // analyst: full theta
  }
  P.end(P.on, m1);
  if (P.id == 0) delete static_cast<SigmoidHandoff<V>*>(S.p1_sigmoid);
}

// ------------------------------------------------------------------------------------------------ driver
size_t hw() { return std::max(1u, std::thread::hardware_concurrency()); }

void usage() {
  std::puts(
      "lr_fss [options]   two-party secure logistic regression (Agarwal et al. 2024), data owner + analyst\n"
      "  --variant 1|2      1: MIC sigmoid, trusted-dealer offline (paper's v1, default)  2: comparison sigmoid (v2)\n"
      "  --d D              features INCLUDING the bias column (default 5)\n"
      "  --batch B          rows per iteration (default 4); the table has iters*B rows\n"
      "  --iters T          iterations, each on its own minibatch (default 20)\n"
      "  --alpha A --lambda L   learning rate (default 1), L2 regularisation (default 1e-4)\n"
      "  --frac F --ring R  fixed-point fraction bits (20) and ring bits (63); the library's tuned values\n"
      "  --threads N        threads per party in the online phase (default: cores/2)\n"
      "  --dealer-threads N threads the owner uses as dealer in the offline phase (default: all cores)\n"
      "  --rtt-ms X --bw-mbps B   emulated link (default 0 / unlimited = LAN-like loopback)\n"
      "  --seed S           data seed;  --no-verify  skip the plaintext comparison;  --json FILE  write all metrics");
}

void json_phase(FILE* f, const char* name, const Phase& p) {
  std::fprintf(f, "    \"%s\": {\"wall_s\": %.6f, \"cpu_s\": %.6f, \"wait_peer_s\": %.6f, \"wait_net_s\": %.6f, \"sent_bytes\": %llu, \"recv_bytes\": %llu, \"rounds\": %llu}",
               name, p.wall, p.cpu, p.wait_peer, p.wait_net, (unsigned long long)p.sent, (unsigned long long)p.recv,
               (unsigned long long)p.rounds);
}
std::string hb(uint64_t b) {
  const char* u[] = {"B", "KiB", "MiB", "GiB"};
  double v = (double)b;
  int i = 0;
  while (v >= 1024 && i < 3) { v /= 1024; ++i; }
  char t[40];
  if (i == 0) std::snprintf(t, sizeof t, "%llu B", (unsigned long long)b);
  else std::snprintf(t, sizeof t, "%.2f %s", v, u[i]);
  return t;
}

template <class V>
int go(Cfg& cfg) {
  Shared S;
  S.cfg = cfg;
  S.dealer_threads = cfg.dealer_threads;
  S.chunks = plan_chunks(cfg, std::max(cfg.threads, cfg.dealer_threads));
  Net net;
  net.one_way_s = cfg.rtt_ms * 1e-3 / 2;
  net.bytes_per_s = cfg.bw_mbps * 1e6 / 8;
  Party owner, analyst;
  owner.id = 0; owner.ep = Endpoint{0, &net}; owner.threads = cfg.threads;
  analyst.id = 1; analyst.ep = Endpoint{1, &net}; analyst.threads = cfg.threads;

  const Data data = generate_data(cfg);  // owner's private table; only the owner thread receives a pointer to it
  const double t0 = wall_now();
  std::thread ta([&] { run_party<V, lr::GradientDescentPartyOne>(analyst, S, nullptr); });
  run_party<V, lr::GradientDescentPartyZero>(owner, S, &data);
  ta.join();
  const double total = wall_now() - t0;

  // ---- result check (test harness only: sees both parties' outputs)
  const uint64_t mod = 1ULL << cfg.ring;
  std::vector<double> th(cfg.d);
  for (size_t j = 0; j < cfg.d; ++j) th[j] = from_ring(analyst.theta_share[j], mod, cfg.frac);
  double maxdiff = -1, acc = -1, acc_ref = -1;
  if (cfg.verify) {
    std::vector<double> ref = reference_train(cfg, data);
    maxdiff = 0;
    for (size_t j = 0; j < cfg.d; ++j) maxdiff = std::max(maxdiff, std::fabs(th[j] - ref[j]));
    size_t ok1 = 0, ok2 = 0;
    const double one = (double)(1LL << cfg.frac);
    for (size_t r = 0; r < cfg.n; ++r) {
      double z1 = 0, z2 = 0;
      for (size_t j = 0; j < cfg.d; ++j) { const double x = (double)data.x[r * cfg.d + j] / one; z1 += x * th[j]; z2 += x * ref[j]; }
      const bool y = data.y[r] != 0;
      ok1 += ((z1 > 0) == y);
      ok2 += ((z2 > 0) == y);
    }
    acc = (double)ok1 / cfg.n;
    acc_ref = (double)ok2 / cfg.n;
  }
  if (std::getenv("LRFSS_DEBUG")) {
    const uint64_t m = 1ULL << cfg.ring;
    for (size_t i = 0; i < owner.dbg_u.size(); ++i)
      std::fprintf(stderr, "dbg row %zu: u=%.6f sigmoid=%.6f\n", i, from_ring((owner.dbg_u[i] + analyst.dbg_u[i]) & (m - 1), m, cfg.frac),
                   from_ring((owner.dbg_s[i] + analyst.dbg_s[i]) & (m - 1), m, cfg.frac));
  }
  rusage ru;
  getrusage(RUSAGE_SELF, &ru);

  const double rtt = cfg.rtt_ms;
  std::printf("\n=== lr_fss: variant v%d (%s), %zu rows d=%zu iters=%zu x batch %zu  link: RTT %.1f ms%s  threads/party=%d dealer=%d chunks=%zu ===\n",
              cfg.variant, cfg.variant == 1 ? "MIC sigmoid, 4 sigmoid rounds" : "comparison sigmoid, 6 sigmoid rounds", cfg.n,
              cfg.d, cfg.iters, cfg.batch, rtt, cfg.bw_mbps > 0 ? "" : ", bandwidth unlimited", cfg.threads, cfg.dealer_threads,
              S.chunks.count);
  auto row = [&](const char* label, const std::string& a, const std::string& b) {
    std::printf("  %-38s %18s %18s\n", label, a.c_str(), b.c_str());
  };
  auto f = [](double v) { char t[32]; std::snprintf(t, sizeof t, "%.3f", v); return std::string(t); };
  auto u = [](uint64_t v) { return std::to_string((unsigned long long)v); };
  for (int ph = 0; ph < 2; ++ph) {
    const Phase& a = ph == 0 ? owner.off : owner.on;
    const Phase& b = ph == 0 ? analyst.off : analyst.on;
    std::printf("\n%s\n", ph == 0 ? "OFFLINE (data independent except input sharing; dealer = data owner)" : "ONLINE");
    row("", "data owner", "analyst");
    row("CPU time, all threads (s)", f(a.cpu), f(b.cpu));
    row("wall time (s)", f(a.wall), f(b.wall));
    row("wait: for the peer to send (s)", f(a.wait_peer), f(b.wait_peer));
    row("wait: network latency/transfer (s)", f(a.wait_net), f(b.wait_net));
    row("compute wall = wall - all waits (s)", f(a.wall - a.wait_peer - a.wait_net), f(b.wall - b.wait_peer - b.wait_net));
    row("bytes sent", hb(a.sent), hb(b.sent));
    row("bytes received", hb(a.recv), hb(b.recv));
    row("rounds of interaction", u(a.rounds), u(b.rounds));
  }
  std::printf("\nonline rounds = %zu x (1 [X-A, X^T-A, theta-B] + %d sigmoid + 1 [gradient]) + 1 (reveal to analyst) = %llu\n", cfg.iters,
              cfg.variant == 1 ? V1::kRounds : V2::kRounds, (unsigned long long)analyst.on.rounds);
  std::printf("online wall (owner/analyst) = %.3f s / %.3f s;  latency alone at %.0f ms RTT = rounds x RTT/2 = %.3f s\n",
              owner.on.wall, analyst.on.wall, rtt, (double)analyst.on.rounds * rtt / 2e3);
  std::printf("total wall (offline+online) %.3f s;  peak RSS %.1f MiB (both parties, one process)\n", total, ru.ru_maxrss / 1024.0);
  const double one_sig = (cfg.variant == 1) ? 2.95 : 205.0;
  std::printf("paper's analytic offline cost per sigmoid (Sec. 7.1, NOT measured here): %s = %.1f KiB  ->  x %zu sigmoids = %s\n",
              cfg.variant == 1 ? "v1 FSS material stored per party" : "v2 offline communication of the 2PC offline phase", one_sig,
              cfg.n, hb((uint64_t)(one_sig * 1024 * cfg.n)).c_str());
  if (cfg.verify)
    std::printf("model check: max |theta_secure - theta_plaintext_double| = %.2e;  train accuracy secure %.4f vs plaintext %.4f\n",
                maxdiff, acc, acc_ref);
  std::printf("analyst's model (first 5 of %zu):", cfg.d);
  for (size_t j = 0; j < std::min<size_t>(5, cfg.d); ++j) std::printf(" %.5f", th[j]);
  std::printf("\n");

  if (!cfg.json.empty()) {
    FILE* jf = std::fopen(cfg.json.c_str(), "w");
    if (!jf) die("cannot write " + cfg.json);
    std::fprintf(jf, "{\n  \"variant\": %d, \"n\": %zu, \"d\": %zu, \"iters\": %zu, \"batch\": %zu, \"rtt_ms\": %g, \"bw_mbps\": %g, \"threads\": %d, \"dealer_threads\": %d, \"chunks\": %zu,\n",
                 cfg.variant, cfg.n, cfg.d, cfg.iters, cfg.batch, cfg.rtt_ms, cfg.bw_mbps, cfg.threads, cfg.dealer_threads, S.chunks.count);
    std::fprintf(jf, "  \"owner\": {\n"); json_phase(jf, "offline", owner.off); std::fprintf(jf, ",\n"); json_phase(jf, "online", owner.on);
    std::fprintf(jf, "\n  },\n  \"analyst\": {\n"); json_phase(jf, "offline", analyst.off); std::fprintf(jf, ",\n"); json_phase(jf, "online", analyst.on);
    std::fprintf(jf, "\n  },\n  \"total_wall_s\": %.6f, \"max_abs_theta_diff\": %g, \"accuracy\": %g, \"accuracy_plaintext\": %g, \"peak_rss_kib\": %ld\n}\n",
                 total, maxdiff, acc, acc_ref, ru.ru_maxrss);
    std::fclose(jf);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Cfg c;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&]() -> const char* { if (i + 1 >= argc) die("missing value for " + a); return argv[++i]; };
    if (a == "--variant") c.variant = std::atoi(val());
    else if (a == "--batch") c.batch = (size_t)std::atoll(val());
    else if (a == "--d") c.d = (size_t)std::atoll(val());
    else if (a == "--iters") c.iters = (size_t)std::atoll(val());
    else if (a == "--alpha") c.alpha = std::atof(val());
    else if (a == "--lambda") c.lambda = std::atof(val());
    else if (a == "--frac") c.frac = std::atoi(val());
    else if (a == "--ring") c.ring = std::atoi(val());
    else if (a == "--threads") c.threads = std::atoi(val());
    else if (a == "--dealer-threads") c.dealer_threads = std::atoi(val());
    else if (a == "--rtt-ms") c.rtt_ms = std::atof(val());
    else if (a == "--bw-mbps") c.bw_mbps = std::atof(val());
    else if (a == "--seed") c.seed = (uint64_t)std::atoll(val());
    else if (a == "--json") c.json = val();
    else if (a == "--no-verify") c.verify = false;
    else if (a == "-h" || a == "--help") { usage(); return 0; }
    else die("unknown option " + a);
  }
  if (c.variant != 1 && c.variant != 2) die("--variant must be 1 or 2");
  if (c.batch == 0 || c.d < 2 || c.iters == 0) die("need batch >= 1, d >= 2, iters >= 1");
  c.n = c.iters * c.batch;
  if (c.ring != 63 || c.frac != 20) std::fprintf(stderr, "warning: the library's sigmoid constants are tuned for ring 63, frac 20\n");
  if (c.threads <= 0) c.threads = (int)std::max<size_t>(1, hw() / 2);
  if (c.dealer_threads <= 0) c.dealer_threads = (int)hw();
  return c.variant == 1 ? go<V1>(c) : go<V2>(c);
}
