// lr_falcon_selftest -- unit and end-to-end tests. The four parties (S0, S1, S2, client) run as threads of
// this process, connected over loopback TCP, through exactly the code of lr_falcon (Net, Dealer, Ctx, the
// protocol functions of falcon.h, server_run / client_run of lr.h). Every case runs under a watchdog: a
// case that hangs fails the whole test (exit 1) instead of blocking.
//
//   ./lr_falcon_selftest            full run: + end-to-end configurations x modes, every tamper kind x cheater
//   ./lr_falcon_selftest --quick    ring identities, dealer consistency, PrivateCompare, RELUPrime,
//                                   SelectShares, MatMul + truncation, one short end-to-end run per mode
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include "falcon.h"
#include "lr.h"
#include "metrics.h"
#include "net.h"
#include "params.h"
#include "prep.h"
#include "rss.h"

using namespace fal;

static int g_fail = 0;
#define CHECK(c, ...)                                               \
  do {                                                              \
    if (!(c)) {                                                     \
      ++g_fail;                                                     \
      std::printf("  FAIL %s:%d: ", __FILE__, __LINE__);            \
      std::printf(__VA_ARGS__);                                     \
      std::printf("\n");                                            \
    }                                                               \
  } while (0)

static std::mt19937_64 g_rng(0x5eed);
static int g_port = 0;
static int next_port() { g_port += 8; return g_port; }

// ---- watchdog: a case that does not finish in time fails the test --------------------------------------
class Watchdog {
 public:
  Watchdog(std::string name, int seconds) : name_(std::move(name)) {
    th_ = std::thread([this, seconds] {
      std::unique_lock<std::mutex> lk(mu_);
      if (!cv_.wait_for(lk, std::chrono::seconds(seconds), [this] { return done_; })) {
        std::printf("  FAIL: case '%s' hung for %d s (watchdog)\nFAILED\n", name_.c_str(), seconds);
        std::fflush(stdout);
        std::_Exit(1);
      }
    });
  }
  ~Watchdog() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      done_ = true;
    }
    cv_.notify_all();
    th_.join();
  }

 private:
  std::string name_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool done_ = false;
  std::thread th_;
};

// ---- the four parties as threads ---------------------------------------------------------------------------
struct Party {
  int code = 0;  // 0 ok, 1 error, 3 AbortError, 4 PeerError (the exit codes of lr_falcon)
  std::string msg;
};
using PartyFn = std::function<void(Net&, Recorder&)>;

static std::array<Party, NP> run_parties(const std::array<PartyFn, NP>& fn, int timeout_s = 20) {
  std::array<Party, NP> R;
  const int port = next_port();
  auto body = [&](int me) {
    NetConfig c;
    c.me = me;
    c.port = port;
    c.run_id = (uint64_t)port;
    c.connect_timeout_s = timeout_s;
    c.timeout_s = timeout_s;
    Net net(c);
    Recorder rec(net);
    auto fail = [&](const std::string& m) {
      if (me == CL) for (int i = 0; i < 3; ++i) net.notify_abort(i, m);
      else net.notify_abort(CL, m);
      net.abort_all();
    };
    try {
      {
        Scope sc(rec, "net/connect");
        net.connect();
      }
      fn[(size_t)me](net, rec);
      net.abort_all();
    } catch (const AbortError& e) {
      R[(size_t)me] = {3, e.what()};
      fail(e.what());
    } catch (const PeerError& e) {
      R[(size_t)me] = {4, e.what()};
      fail(e.what());
    } catch (const std::exception& e) {
      R[(size_t)me] = {1, e.what()};
      fail(e.what());
    }
  };
  std::vector<std::thread> th;
  for (int i = 0; i < 3; ++i) th.emplace_back(body, i);
  body(CL);
  for (auto& t : th) t.join();
  return R;
}
static bool all_ok(const std::array<Party, NP>& R, std::string* why = nullptr) {
  bool ok = true;
  for (int p = 0; p < NP; ++p)
    if (R[(size_t)p].code != 0) {
      ok = false;
      if (why) *why += std::string(party_name(p)) + ": " + R[(size_t)p].msg + "; ";
    }
  return ok;
}

// Unit-test harness: setup (handshake, seeds, PRSS keys) then the given bodies.
using CliFn = std::function<void(Net&, Dealer&)>;
using SrvFn = std::function<void(Net&, Ctx&, Dealer&)>;
static std::array<Party, NP> run_unit(bool mal, const CliFn& cf, const SrvFn& sf) {
  Params P;
  P.finalize();
  std::array<Tamper, 3> tp;
  std::array<PartyFn, NP> fn;
  for (int i = 0; i < 3; ++i)
    fn[(size_t)i] = [&, i](Net& net, Recorder&) {
      Dealer D;
      Ctx cx(net, mal, tp[(size_t)i], nullptr);
      server_setup(net, nullptr, P, mal, D, cx);
      sf(net, cx, D);
    };
  fn[CL] = [&](Net& net, Recorder&) {
    Dealer D;
    client_setup(net, nullptr, P, mal, D);
    cf(net, D);
  };
  return run_parties(fn);
}
// server -> client: (.first..., .second...)
template <class T>
static void send_shares(Net& net, const std::vector<RSS<T>>& v) {
  std::vector<T> b(2 * v.size());
  for (size_t i = 0; i < v.size(); ++i) { b[i] = v[i].first; b[v.size() + i] = v[i].second; }
  Step s(net);
  s.send_vec(CL, b);
  s.run();
}
// client: receives the shares of n values from the three servers, checks the replication, reconstructs.
template <class Rg>
static std::vector<typename Rg::T> collect(Net& net, size_t n, bool* consistent) {
  using T = typename Rg::T;
  std::array<std::vector<T>, 3> b;
  Step s(net);
  for (int i = 0; i < 3; ++i) { b[(size_t)i].resize(2 * n); s.recv_vec(i, b[(size_t)i]); }
  s.run();
  std::vector<T> v(n);
  *consistent = true;
  for (size_t k = 0; k < n; ++k) {
    for (int i = 0; i < 3; ++i)
      if (b[(size_t)i][n + k] != b[(size_t)next_of(i)][k]) *consistent = false;  // S_i.second == S_{i+1}.first
    v[k] = Rg::add(Rg::add(b[0][k], b[1][k]), b[2][k]);
  }
  return v;
}

// ---- 1. local identities -------------------------------------------------------------------------------------
static void test_ring() {
  std::printf("[ring Z_2^64, field Z_67, wrap3 / carry / MSB, truncation pair, 3-out-of-3 products]\n");
  const u64 edge[] = {0, 1, 2, ~0ull, ~0ull - 1, 1ull << 63, (1ull << 63) - 1, (1ull << 63) + 1, 12345};
  int n = 0;
  auto chk_wrap = [&](u64 a, u64 b, u64 c) {
    const unsigned __int128 s = (unsigned __int128)a + b + c;
    const u8 par = (u8)((s >> 64) & 1);
    CHECK(wrap3(a, b, c) == par, "wrap3 parity");
    CHECK(wrap3(a, b, c) == wrap3(c, a, b) && wrap3(a, b, c) == wrap3(b, a, c), "wrap3 symmetric");
    CHECK(wrap_around(a, b) == (u8)((((unsigned __int128)a + b) >> 64) & 1), "carry");
    ++n;
  };
  for (u64 a : edge) for (u64 b : edge) for (u64 c : edge) chk_wrap(a, b, c);
  for (int i = 0; i < 20000; ++i) chk_wrap(g_rng(), g_rng(), g_rng());
  for (int i = 0; i < 2000; ++i) { const u64 a = g_rng(); CHECK(msb(a) == ((int64_t)a < 0), "msb"); }
  for (int a = 0; a < kP; ++a)
    for (int b = 0; b < kP; ++b) {
      CHECK(add67((u8)a, (u8)b) == (a + b) % kP && sub67((u8)a, (u8)b) == (a - b + kP) % kP && mul67((u8)a, (u8)b) == a * b % kP, "Z_67");
    }
  for (int i = 0; i < 20000; ++i) {
    const int t = 8 + (int)(g_rng() % 20);
    const int64_t z = (int64_t)(g_rng() % (1ull << 40)) - (int64_t)(1ull << 39);
    const u64 r = trunc_pair((u64)z, g_rng(), t);
    const int64_t fl = z >> t;
    CHECK((int64_t)r == fl || (int64_t)r == fl - 1, "truncation pair z=%lld t=%d", (long long)z, t);
  }
  for (int i = 0; i < 2000; ++i) {  // sum_i prod3(view_i(x), view_i(y)) = x y
    u64 x[3], y[3];
    u8 p[3], q[3], b[3], c[3];
    for (int j = 0; j < 3; ++j) { x[j] = g_rng(); y[j] = g_rng(); p[j] = (u8)(g_rng() % kP); q[j] = (u8)(g_rng() % kP); b[j] = g_rng() & 1; c[j] = g_rng() & 1; }
    u64 s = 0;
    u8 s67 = 0, sb = 0;
    for (int i2 = 0; i2 < 3; ++i2) {
      const int j = next_of(i2);
      s += prod3<Ring64>({x[i2], x[j]}, {y[i2], y[j]});
      s67 = add67(s67, prod3<Field67>({p[i2], p[j]}, {q[i2], q[j]}));
      sb ^= prod3<Bits>({b[i2], b[j]}, {c[i2], c[j]});
    }
    CHECK(s == (x[0] + x[1] + x[2]) * (y[0] + y[1] + y[2]), "prod3 ring");
    CHECK(s67 == mul67(add67(add67(p[0], p[1]), p[2]), add67(add67(q[0], q[1]), q[2])), "prod3 Z_67");
    CHECK(sb == ((b[0] ^ b[1] ^ b[2]) & (c[0] ^ c[1] ^ c[2])), "prod3 bits");
  }
  // PRSS: the three alpha_i sum to zero
  u8 k[3][16];
  for (auto& kk : k) for (auto& v : kk) v = (u8)g_rng();
  Prss pr[3];
  for (int i = 0; i < 3; ++i) pr[i].init(k[i], k[prev_of(i)]);
  bool zero = true;
  for (int it = 0; it < 1000; ++it) {
    u64 a = 0;
    u8 b = 0, c = 0;
    for (int i = 0; i < 3; ++i) { a += pr[i].zero64(); b = add67(b, pr[i].zero67()); c ^= pr[i].zerobit(); }
    zero = zero && a == 0 && b == 0 && c == 0;
  }
  CHECK(zero, "PRSS zero-sharings must sum to 0");
  std::printf("  %d wrap3 cases, Z_67 tables, truncation pairs, products, PRSS zero-sharings ok\n", n);
}

// ---- 2. dealer consistency (no network: the client's corrections are handed over in memory) ----------------
template <class T, class Add>
static std::vector<T> recon(const std::vector<RSS<T>>* v[3], Add add, bool* ok, const char* what, bool field = false, bool bit = false) {
  const size_t n = v[0]->size();
  std::vector<T> out(n);
  for (int i = 0; i < 3; ++i)
    if (v[i]->size() != n) { *ok = false; std::printf("  size mismatch %s\n", what); return out; }
  for (size_t k = 0; k < n; ++k) {
    for (int i = 0; i < 3; ++i) {
      const RSS<T>& a = (*v[i])[k];
      if ((*v[next_of(i)])[k].first != a.second) { *ok = false; }
      if (field && (a.first >= kP || a.second >= kP)) *ok = false;
      if (bit && (a.first > 1 || a.second > 1)) *ok = false;
    }
    out[k] = add(add((*v[0])[k].first, (*v[1])[k].first), (*v[2])[k].first);
  }
  return out;
}

static void test_dealer() {
  std::printf("[dealer: three server views of every item are consistent and satisfy their relation]\n");
  struct Cfg { int d, iters, batch; double lr; int frac; };
  const Cfg cfgs[] = {{5, 3, 4, 0.5, 13}, {1, 2, 1, 0.25, 13}, {7, 2, 2, 1.0, 10}, {16, 2, 8, 2.0, 16}};
  auto r64 = [](u64 a, u64 b) { return a + b; };
  auto r67 = [](u8 a, u8 b) { return add67(a, b); };
  auto rx = [](u8 a, u8 b) { return (u8)(a ^ b); };
  for (const Cfg& c : cfgs)
    for (bool mal : {true, false}) {
      Params P;
      P.d = c.d; P.iters = c.iters; P.batch = c.batch; P.lr = c.lr; P.frac = c.frac;
      P.finalize();
      u8 seeds[48];
      for (auto& s : seeds) s = (u8)g_rng();
      Dealer C;
      C.init_client(seeds);
      ClientView cv;
      std::vector<IterPrep> pc;
      deal_all(C, P, mal, pc, &cv);
      CHECK(C.corr.size() == deal_payload_bytes(P, mal), "correction bytes %zu != formula %llu", C.corr.size(), (unsigned long long)deal_payload_bytes(P, mal));
      CHECK(deal_bytes_per_iter(P, mal) == (mal ? 436ull * P.batch + 16ull * P.d : 286ull * P.batch + 8ull * P.d), "formula");
      Dealer S[3];
      std::vector<IterPrep> ps[3];
      bool consumed = true;
      for (int i = 0; i < 3; ++i) {
        u8 two[32];
        std::memcpy(two, seeds + 16 * i, 16);
        std::memcpy(two + 16, seeds + 16 * next_of(i), 16);
        S[i].init_server(i, two);
        if (i != S1) S[i].corr = C.corr;  // S1 receives nothing but its seeds
        deal_all(S[i], P, mal, ps[i], nullptr);
        try { S[i].expect_consumed("offline"); } catch (...) { consumed = false; }
      }
      CHECK(consumed && S[1].corr.empty() && S[1].rpos == 0, "corrections: S0/S2 consume all, S1 none");
      for (int j = 0; j < 3; ++j)
        CHECK(C.position(j) == S[j].position(j) && C.position(j) == S[prev_of(j)].position(j) && S[next_of(j)].position(j) == 0,
              "stream %d positions differ: client %llu, holders %llu / %llu", j, (unsigned long long)C.position(j),
              (unsigned long long)S[j].position(j), (unsigned long long)S[prev_of(j)].position(j));
      bool ok = true;
      const size_t B = (size_t)P.batch, d = (size_t)P.d, N = 2 * B;
      for (int t = 0; t < P.iters; ++t) {
        const IterPrep* I[3] = {&ps[0][(size_t)t], &ps[1][(size_t)t], &ps[2][(size_t)t]};
        const Rv64* frp[3] = {&I[0]->fwd.rp, &I[1]->fwd.rp, &I[2]->fwd.rp};
        const Rv64* fr[3] = {&I[0]->fwd.r, &I[1]->fwd.r, &I[2]->fwd.r};
        const Rv64* brp[3] = {&I[0]->bwd.rp, &I[1]->bwd.rp, &I[2]->bwd.rp};
        const Rv64* br[3] = {&I[0]->bwd.r, &I[1]->bwd.r, &I[2]->bwd.r};
        const auto RPf = recon(frp, r64, &ok, "fwd.rp"), Rf = recon(fr, r64, &ok, "fwd.r");
        const auto RPb = recon(brp, r64, &ok, "bwd.rp"), Rb = recon(br, r64, &ok, "bwd.r");
        for (size_t i = 0; i < B; ++i) {
          ok = ok && Rf[i] == (u64)((int64_t)RPf[i] >> P.shift_fwd()) && RPf[i] == cv.rp_fwd[(size_t)t * B + i];
        }
        for (size_t j = 0; j < d; ++j) ok = ok && Rb[j] == (u64)((int64_t)RPb[j] >> P.shift_bwd()) && RPb[j] == cv.rp_bwd[(size_t)t * d + j];
        // wrap / PC
        const Rv64* wr[3] = {&I[0]->wrap.r, &I[1]->wrap.r, &I[2]->wrap.r};
#define P8(name, fld) const Rv8* name[3] = {&I[0]->fld, &I[1]->fld, &I[2]->fld}
        P8(wa, wrap.alpha); P8(wb, wrap.bits); P8(we, wrap.eta); P8(wt, wrap.t); P8(ws, wrap.s); P8(wm, wrap.m); P8(wms, wrap.ms0);
        P8(wmt, wrap.mt); P8(sc, sel.c);
        const auto R = recon(wr, r64, &ok, "wrap.r");
        const auto A = recon(wa, rx, &ok, "alpha", false, true), Bt = recon(wb, r67, &ok, "bits", true);
        const auto Et = recon(we, rx, &ok, "eta", false, true), Tt = recon(wt, r67, &ok, "t", true), St = recon(ws, r67, &ok, "s", true);
        const auto M = recon(wm, r67, &ok, "m", true), MS = recon(wms, r67, &ok, "ms0", true), MT = recon(wmt, r67, &ok, "mt", true);
        for (size_t e = 0; e < N; ++e) {
          const u64 r0 = (*wr[0])[e].first, r1 = (*wr[1])[e].first, r2 = (*wr[2])[e].first;
          ok = ok && A[e] == wrap3(r0, r1, r2);
          for (int k = 0; k < 63; ++k) ok = ok && Bt[e * 63 + (size_t)k] == ((R[e] >> (63 - k)) & 1);
          ok = ok && Tt[e] == (Et[e] ? 1 : kP - 1);
          for (int k = 1; k < 64; ++k) ok = ok && St[e * 63 + (size_t)(k - 1)] == (((R[e] >> (63 - k)) & 1) ? Tt[e] : 0);
          ok = ok && M[e] != 0 && MT[e] == mul67(M[e], Tt[e]) && MS[e] == ((R[e] >> 63) ? mul67(M[e], Tt[e]) : 0);
        }
        if (mal) {
          P8(tx, wrap.tx); P8(ty, wrap.ty); P8(tz, wrap.tz);
          const auto X = recon(tx, r67, &ok, "tx", true), Y = recon(ty, r67, &ok, "ty", true), Z = recon(tz, r67, &ok, "tz", true);
          ok = ok && X.size() == 63 * N;
          for (size_t k = 0; k < X.size(); ++k) ok = ok && Z[k] == mul67(X[k], Y[k]);
        }
        // select
        const Rv64* smc[3] = {&I[0]->sel.mc, &I[1]->sel.mc, &I[2]->sel.mc};
        const auto Cb = recon(sc, rx, &ok, "c", false, true);
        const auto MC = recon(smc, r64, &ok, "mc");
        for (size_t e = 0; e < N; ++e) ok = ok && MC[e] == Cb[e];
        if (mal) {
          const Rv64* sx[3] = {&I[0]->sel.x, &I[1]->sel.x, &I[2]->sel.x};
          const Rv64* sy[3] = {&I[0]->sel.y, &I[1]->sel.y, &I[2]->sel.y};
          const Rv64* sz[3] = {&I[0]->sel.z, &I[1]->sel.z, &I[2]->sel.z};
          const auto X = recon(sx, r64, &ok, "sel.x"), Y = recon(sy, r64, &ok, "sel.y"), Z = recon(sz, r64, &ok, "sel.z");
          for (size_t e = 0; e < N; ++e) ok = ok && Z[e] == X[e] * Y[e];
          // matrix triple
          const Rv64* mu[3] = {&I[0]->mat.U, &I[1]->mat.U, &I[2]->mat.U};
          const Rv64* mvf[3] = {&I[0]->mat.vf, &I[1]->mat.vf, &I[2]->mat.vf};
          const Rv64* mzf[3] = {&I[0]->mat.Zf, &I[1]->mat.Zf, &I[2]->mat.Zf};
          const Rv64* mvb[3] = {&I[0]->mat.vb, &I[1]->mat.vb, &I[2]->mat.vb};
          const Rv64* mzb[3] = {&I[0]->mat.Zb, &I[1]->mat.Zb, &I[2]->mat.Zb};
          const auto U = recon(mu, r64, &ok, "U"), VF = recon(mvf, r64, &ok, "vf"), ZF = recon(mzf, r64, &ok, "Zf");
          const auto VB = recon(mvb, r64, &ok, "vb"), ZB = recon(mzb, r64, &ok, "Zb");
          for (size_t k = 0; k < B * d; ++k) ok = ok && U[k] == cv.U[(size_t)t * B * d + k];
          for (size_t i = 0; i < B; ++i) {
            u64 z = 0;
            for (size_t j = 0; j < d; ++j) z += U[i * d + j] * VF[j];
            ok = ok && ZF[i] == z;
          }
          for (size_t j = 0; j < d; ++j) {
            u64 z = 0;
            for (size_t i = 0; i < B; ++i) z += U[i * d + j] * VB[i];
            ok = ok && ZB[j] == z;
          }
        }
#undef P8
      }
      CHECK(ok, "dealt items inconsistent: d=%d batch=%d mal=%d", P.d, P.batch, (int)mal);
      std::printf("  d=%-2d batch=%d iters=%d %s: %zu correction bytes (= formula), stream positions equal, relations hold %s\n", P.d, P.batch,
                  P.iters, mal ? "malicious  " : "semi-honest", C.corr.size(), ok ? "ok" : "FAIL");
    }
}

// ---- 3. protocol functions over the network -----------------------------------------------------------------
static void test_pc(bool mal) {
  Watchdog wd("PrivateCompare", 60);
  const size_t N = 96;
  WrapFull wf;
  std::vector<u64> Xp(N);
  std::vector<u8> got;
  bool cons = true;
  auto R = run_unit(
      mal,
      [&](Net& net, Dealer& D) {
        WrapPrep wp;
        deal_wrap(D, N, mal, wp, &wf);
        send_corrections(net, D.corr);
        for (size_t e = 0; e < N; ++e) {
          const u64 r = wf.R[e];
          switch (e % 12) {
            case 0: Xp[e] = r; break;
            case 1: Xp[e] = r + 1; break;
            case 2: Xp[e] = r - 1; break;
            case 3: Xp[e] = 0; break;
            case 4: Xp[e] = ~0ull; break;
            case 5: Xp[e] = r ^ (1ull << (g_rng() % 64)); break;
            case 6: Xp[e] = r ^ (1ull << 63); break;
            case 7: Xp[e] = r ^ 1; break;
            case 8: Xp[e] = r + 2; break;
            case 9: Xp[e] = r - 2; break;
            default: Xp[e] = g_rng();
          }
        }
        Step s(net);
        for (int i = 0; i < 3; ++i) s.send_vec(i, Xp);
        s.run();
        std::array<std::vector<u8>, 3> b;
        Step r(net);
        for (int i = 0; i < 3; ++i) { b[(size_t)i].resize(N); r.recv_vec(i, b[(size_t)i]); }
        r.run();
        cons = b[0] == b[1] && b[1] == b[2];
        got = b[0];
      },
      [&](Net& net, Ctx& cx, Dealer& D) {
        recv_corrections(net, D, N * wrap_corr_bytes_per_elem(mal));
        WrapPrep wp;
        deal_wrap(D, N, mal, wp);
        D.expect_consumed("pc");
        std::vector<u64> X(N);
        Step s(net);
        s.recv_vec(CL, X);
        s.run();
        const std::vector<u8> bp = private_compare(cx, wp, X);
        Step o(net);
        o.send_vec(CL, bp);
        o.run();
      });
  std::string why;
  CHECK(all_ok(R, &why), "PC aborted: %s", why.c_str());
  int bad = 0, eq = 0;
  for (size_t e = 0; e < N && got.size() == N; ++e) {
    const u8 want = wf.eta[e] ? (wf.R[e] < Xp[e]) : (wf.R[e] > Xp[e]);  // upstream semantics: beta' = [d == 0]
    if (got[e] != want) ++bad;
    if (wf.R[e] == Xp[e]) ++eq;
  }
  CHECK(cons && got.size() == N && bad == 0, "PC: %d of %zu wrong (servers agree: %d)", bad, N, (int)cons);
  std::printf("  PrivateCompare %s: %zu inputs (R = X', R = X' -+ 1, -+ 2, 0, 2^64-1, bit flips, random; %d equal): %s\n",
              mal ? "malicious  " : "semi-honest", N, eq, bad == 0 && cons ? "ok" : "FAIL");
}

static void test_relu_prime(bool mal) {
  Watchdog wd("RELUPrime", 60);
  std::vector<u64> a;
  const int F = 13;
  const u64 edge[] = {0, 1, ~0ull, 2, ~0ull - 1, 1ull << (F - 1), (u64)0 - (1ull << (F - 1)), 1ull << 63, (1ull << 63) - 1,
                      (1ull << 63) + 1, 1ull << 62, (u64)0 - (1ull << 62), (1ull << 62) - 1, 1ull << F, (u64)0 - (1ull << F)};
  for (int rep = 0; rep < 3; ++rep) for (u64 v : edge) a.push_back(v);  // each edge value with 3 different masks
  for (int i = 0; i < 40; ++i) a.push_back(g_rng());
  for (int i = 0; i < 20; ++i) a.push_back((u64)((int64_t)(g_rng() % 100000) - 50000));
  const size_t N = a.size();
  std::vector<u8> b;
  bool cons = true;
  auto R = run_unit(
      mal,
      [&](Net& net, Dealer& D) {
        WrapPrep wp;
        deal_wrap(D, N, mal, wp);
        for (u64 v : a) D.corr64(v);
        send_corrections(net, D.corr);
        b = collect<Bits>(net, N, &cons);
      },
      [&](Net& net, Ctx& cx, Dealer& D) {
        recv_corrections(net, D, N * wrap_corr_bytes_per_elem(mal) + 8 * N);
        WrapPrep wp;
        deal_wrap(D, N, mal, wp);
        Rv64 x(N);
        for (size_t i = 0; i < N; ++i) x[i] = D.view(D.corr64(0));
        D.expect_consumed("relu");
        send_shares(net, relu_prime(cx, x, wp));
      });
  std::string why;
  CHECK(all_ok(R, &why), "RELUPrime aborted: %s", why.c_str());
  int bad = 0;
  for (size_t i = 0; i < N && b.size() == N; ++i)
    if (b[i] != msb(a[i])) { ++bad; std::printf("    MSB wrong for a = %llu\n", (unsigned long long)a[i]); }
  CHECK(cons && b.size() == N && bad == 0, "RELUPrime: %d of %zu wrong, consistent %d", bad, N, (int)cons);
  std::printf("  RELUPrime %s: %zu inputs (0, +-1, +-2, +-2^(F-1), 2^63, 2^63 -+ 1, +-2^62, random): %s\n", mal ? "malicious  " : "semi-honest", N,
              bad == 0 && cons ? "ok" : "FAIL");
}

static void test_select(bool mal) {
  Watchdog wd("SelectShares", 60);
  const size_t N = 40;
  std::vector<u64> a(N), c;
  std::vector<u8> bit(N);
  for (size_t i = 0; i < N; ++i) { a[i] = i < 4 ? (u64)i * 0x8000000000000000ull : g_rng(); bit[i] = (u8)(g_rng() & 1); }
  bool cons = true;
  auto R = run_unit(
      mal,
      [&](Net& net, Dealer& D) {
        SelPrep sp;
        deal_select(D, N, mal, sp);
        for (size_t i = 0; i < N; ++i) D.corr64(a[i]);
        for (size_t i = 0; i < N; ++i) D.corrbit(bit[i]);
        send_corrections(net, D.corr);
        c = collect<Ring64>(net, N, &cons);
      },
      [&](Net& net, Ctx& cx, Dealer& D) {
        recv_corrections(net, D, N * sel_corr_bytes_per_elem(mal) + 9 * N);
        SelPrep sp;
        deal_select(D, N, mal, sp);
        Rv64 x(N);
        Rv8 b(N);
        for (size_t i = 0; i < N; ++i) x[i] = D.view(D.corr64(0));
        for (size_t i = 0; i < N; ++i) b[i] = D.view(D.corrbit(0));
        D.expect_consumed("select");
        const Rv64 out = select_shares(cx, x, b, sp);
        if (cx.mal) final_check(cx);  // the deferred triple checks must pass
        send_shares(net, out);
      });
  std::string why;
  CHECK(all_ok(R, &why), "SelectShares aborted: %s", why.c_str());
  int bad = 0;
  for (size_t i = 0; i < N && c.size() == N; ++i) bad += c[i] != (bit[i] ? 0 : a[i]);
  CHECK(cons && c.size() == N && bad == 0, "SelectShares: %d wrong", bad);
  std::printf("  SelectShares %s: %zu inputs, a (1 - b) exact%s: %s\n", mal ? "malicious  " : "semi-honest", N, mal ? ", final check passed" : "",
              bad == 0 && cons ? "ok" : "FAIL");
}

static void test_matmul(bool mal) {
  Watchdog wd("MatMul", 60);
  const int B = 4, d = 9, F = 13, Fb = 16;
  const size_t nb = B, nd = d;
  std::vector<u64> X(nb * nd), w(nd), e(nb), rpf, rpb, U, cf, cb;
  auto fp = [&] { return (u64)((int64_t)(g_rng() % (2u << F)) - (int64_t)(1u << F)); };  // in [-1, 1) at scale F
  for (auto& v : X) v = fp();
  for (auto& v : w) v = fp();
  for (auto& v : e) v = fp();
  bool cons1 = true, cons2 = true;
  auto R = run_unit(
      mal,
      [&](Net& net, Dealer& D) {
        MatPrep mp;
        TruncPrep tf, tb;
        if (mal) deal_mat(D, B, d, mp, &U);
        deal_trunc(D, nb, F, tf, &rpf);
        deal_trunc(D, nd, Fb, tb, &rpb);
        send_corrections(net, D.corr);
        D.corr.clear();
        std::vector<u64> E(nb * nd);
        if (mal) for (size_t k = 0; k < nb * nd; ++k) E[k] = X[k] - U[k];
        else for (size_t k = 0; k < nb * nd; ++k) D.corr64(X[k]);
        for (u64 v : w) D.corr64(v);
        for (u64 v : e) D.corr64(v);
        Step s(net);
        for (int i = 0; i < 3; ++i) {
          std::vector<u8> m;
          if (mal) m.insert(m.end(), (const u8*)E.data(), (const u8*)(E.data() + E.size()));
          if (i != S1) m.insert(m.end(), D.corr.begin(), D.corr.end());
          if (!m.empty()) s.send_vec(i, m);
        }
        s.run();
        cf = collect<Ring64>(net, nb, &cons1);
        cb = collect<Ring64>(net, nd, &cons2);
      },
      [&](Net& net, Ctx& cx, Dealer& D) {
        recv_corrections(net, D, 8 * (nb + nd) * (mal ? 2 : 1));
        MatPrep mp;
        TruncPrep tf, tb;
        if (mal) deal_mat(D, B, d, mp, nullptr);
        deal_trunc(D, nb, F, tf, nullptr);
        deal_trunc(D, nd, Fb, tb, nullptr);
        D.expect_consumed("matmul");
        const size_t len = (mal ? 8 * nb * nd : 0) + (cx.me == S1 ? 0 : 8 * (nb + nd + (mal ? 0 : nb * nd)));
        std::vector<u8> buf(len);
        Step s(net);
        if (len) s.recv_vec(CL, buf);
        s.run();
        std::vector<u64> E;
        size_t off = 0;
        if (mal) { E.resize(nb * nd); std::memcpy(E.data(), buf.data(), 8 * nb * nd); off = 8 * nb * nd; }
        D.corr.assign(buf.begin() + (std::ptrdiff_t)off, buf.end());
        D.rpos = 0;
        Rv64 A(nb * nd), wv(nd), ev(nb);
        for (size_t k = 0; k < nb * nd; ++k) {
          if (mal) { A[k] = mp.U[k]; add_const<Ring64>(cx.me, A[k], E[k]); }
          else A[k] = D.view(D.corr64(0));
        }
        for (auto& v : wv) v = D.view(D.corr64(0));
        for (auto& v : ev) v = D.view(D.corr64(0));
        D.expect_consumed("matmul input");
        MatCheck mf{&E, &mp.U, &mp.vf, &mp.Zf}, mb{&E, &mp.U, &mp.vb, &mp.Zb};
        const Rv64 c1 = mat_vec(cx, A, wv, nb, nd, false, F, tf, mal ? &mf : nullptr, "x");
        const Rv64 c2 = mat_vec(cx, A, ev, nd, nb, true, Fb, tb, mal ? &mb : nullptr, "x");
        if (cx.mal) final_check(cx);
        send_shares(net, c1);
        send_shares(net, c2);
      });
  std::string why;
  CHECK(all_ok(R, &why), "MatMul aborted: %s", why.c_str());
  int bad = 0, off_nominal = 0;
  for (size_t i = 0; i < nb && cf.size() == nb; ++i) {
    u64 z = 0;
    for (size_t j = 0; j < nd; ++j) z += X[i * nd + j] * w[j];
    bad += cf[i] != trunc_pair(z, rpf[i], F);
    off_nominal += cf[i] != trunc_exact(z, F);
  }
  for (size_t j = 0; j < nd && cb.size() == nd; ++j) {
    u64 z = 0;
    for (size_t i = 0; i < nb; ++i) z += X[i * nd + j] * e[i];
    bad += cb[j] != trunc_pair(z, rpb[j], Fb);
    off_nominal += cb[j] != trunc_exact(z, Fb);
  }
  CHECK(cons1 && cons2 && cf.size() == nb && cb.size() == nd && bad == 0, "MatMul: %d wrong", bad);
  std::printf("  MatMul+truncation %s: X (4x9) w and X^T e == replayed truncation pairs (%d of 13 are 1 ulp below the exact floor)%s: %s\n",
              mal ? "malicious  " : "semi-honest", off_nominal, mal ? ", final check passed" : "", bad == 0 ? "ok" : "FAIL");
}

// ---- 4. end to end -------------------------------------------------------------------------------------------
struct E2E {
  std::array<Party, NP> p;
  ClientOutcome out;
  std::array<bool, 3> fired{false, false, false};
};
static E2E run_lr(const Params& P, bool mal, Tamper::Kind tk = Tamper::NONE, int cheater = -1) {
  E2E R;
  RunCfg cfg;
  cfg.P = P;
  cfg.mal = mal;
  std::array<Tamper, 3> tp;
  for (auto& t : tp) { t.kind = tk; t.who = cheater; }
  const Dataset data = generate_client_data(P);
  std::array<PartyFn, NP> fn;
  for (int i = 0; i < 3; ++i) fn[(size_t)i] = [&, i](Net& net, Recorder& rec) { server_run(net, rec, cfg, tp[(size_t)i]); };
  fn[CL] = [&](Net& net, Recorder& rec) { R.out = client_run(net, rec, cfg, data); };
  R.p = run_parties(fn);
  for (int i = 0; i < 3; ++i) R.fired[(size_t)i] = tp[(size_t)i].fired;
  return R;
}
static Params make_params(int d, int iters, int batch, double lr, int frac, uint64_t seed, bool bias = true, bool zero = false) {
  Params P;
  P.d = d; P.iters = iters; P.batch = batch; P.lr = lr; P.frac = frac; P.seed = seed; P.bias = bias; P.zero_init = zero;
  P.finalize();
  return P;
}

// The end-to-end data must reach all three pieces of the activation (yhat = 0, u + 1/2, 1).
static void test_branches() {
  std::printf("[activation pieces reached by the end-to-end data (plaintext replay)]\n");
  for (auto [d, seed] : {std::pair<int, uint64_t>{5, 1}, {12, 21}}) {
    const Params P = make_params(d, 20, 4, 0.5, 13, seed);
    const Dataset D = generate_client_data(P);
    const int64_t half = int64_t(1) << (P.frac - 1);
    int lo = 0, mid = 0, hi = 0;
    train_ring(P, D, [&](u64 z, int which, size_t, int t) {
      const u64 u = trunc_exact(z, t);
      if (which == 0) {
        if ((int64_t)u < -half) ++lo;
        else if ((int64_t)u >= half) ++hi;
        else ++mid;
      }
      return u;
    });
    std::printf("  d=%-2d seed=%-2llu: samples below / inside / above the linear piece: %d / %d / %d\n", d, (unsigned long long)seed, lo, mid, hi);
    if (d == 12) CHECK(lo > 0 && mid > 0 && hi > 0, "the d=12 seed=21 data must exercise all three pieces of the activation");
  }
}

static void check_e2e(const Params& P, bool mal) {
  Watchdog wd("end-to-end", 120);
  E2E R = run_lr(P, mal);
  std::string why;
  CHECK(all_ok(R.p, &why), "unexpected abort d=%d mal=%d: %s", P.d, (int)mal, why.c_str());
  if (!all_ok(R.p)) return;
  const Dataset D = generate_client_data(P);
  const std::vector<u64> ref = reference_train(P, D, R.out.cv.rp_fwd, R.out.cv.rp_bwd);
  CHECK(R.out.w == ref, "secure result differs from the reference: d=%d iters=%d batch=%d F=%d mal=%d", P.d, P.iters, P.batch, P.frac, (int)mal);
  const Totals T = analyze(R.out, P, mal);
  CHECK(T.error.empty(), "report invariants: %s", T.error.c_str());
  // dealer bytes: client -> S0 and -> S2 = formula + 8 (frame header), nothing to S1; seeds = 32 + 8 each
  const PhaseRow* deal = find_row(R.out.rows[CL], "offline/deal");
  const PhaseRow* seeds = find_row(R.out.rows[CL], "setup/seeds");
  const uint64_t f = 8 + deal_payload_bytes(P, mal);
  CHECK(deal && deal->sent[S0] == f && deal->sent[S2] == f && deal->sent[S1] == 0, "deal bytes");
  CHECK(seeds && seeds->sent[S0] == 40 && seeds->sent[S1] == 40 && seeds->sent[S2] == 40, "seed bytes");
  const PhaseRow* s1deal = find_row(R.out.rows[S1], "offline/deal");
  CHECK(s1deal && s1deal->recv[CL] == 0, "S1 must receive only seeds from the client");
  const PhaseRow* inp = find_row(R.out.rows[CL], "online/input");
  for (int i = 0; i < 3; ++i) {
    const uint64_t want = input_payload_bytes(P, mal, i) ? 8 + input_payload_bytes(P, mal, i) : 0;
    CHECK(inp && inp->sent[i] == want, "input bytes to S%d", i);
  }
  std::printf("  d=%-3d iters=%-2d batch=%d lr=%-4g F=%-2d%s%s %s: bit-exact, %llu online steps (S0=S1=S2 per phase), deal %llu B to S0/S2, total %s ok\n",
              P.d, P.iters, P.batch, P.lr, P.frac, P.bias ? "" : " no-bias", P.zero_init ? " zero-init" : "", mal ? "malicious  " : "semi-honest",
              (unsigned long long)T.online_steps[0], (unsigned long long)f, human_bytes(T.total).c_str());
}

static void test_end_to_end(bool quick) {
  std::printf("[end-to-end: secure result == plaintext reference with the replayed truncation masks, bit for bit]\n");
  struct Cfg { int d, iters, batch; double lr; int frac; uint64_t seed; bool bias, zero; };
  const Cfg quick_cfgs[] = {{5, 3, 4, 0.5, 13, 1, true, false}};
  const Cfg full_cfgs[] = {
      {5, 20, 4, 0.5, 13, 1, true, false},   // the default benchmark setting
      {1, 3, 1, 0.25, 13, 2, true, false},   // single feature (bias only)
      {7, 6, 2, 1.0, 10, 3, false, true},    // zero init, no bias column, F = 10
      {16, 5, 8, 2.0, 16, 4, true, false},   // batch 8, lr 2.0, F = 16
      {40, 4, 4, 0.5, 13, 5, true, false},   // more features
      {12, 20, 4, 0.5, 13, 21, true, false}, // exercises all three pieces of the activation
  };
  const Cfg* b = quick ? quick_cfgs : full_cfgs;
  const size_t n = quick ? 1 : sizeof full_cfgs / sizeof full_cfgs[0];
  for (size_t k = 0; k < n; ++k)
    for (bool mal : {true, false}) check_e2e(make_params(b[k].d, b[k].iters, b[k].batch, b[k].lr, b[k].frac, b[k].seed, b[k].bias, b[k].zero), mal);
}

// ---- 5. tamper ---------------------------------------------------------------------------------------------
static void test_tamper() {
  std::printf("[malicious security: every deviation x cheater aborts with the right detection; semi-honest does not abort]\n");
  const Params P = make_params(4, 3, 2, 0.5, 13, 8);
  struct K { Tamper::Kind k; const char* frag; bool semi; } kinds[] = {
      {Tamper::OPEN_VALUE, "checked opening mismatch", true},
      {Tamper::OPEN_HASH, "checked opening mismatch", false},
      {Tamper::REC3, "final check failed|checked opening mismatch", true},
      {Tamper::TREE, "PC check failed|checked opening mismatch", true},
      {Tamper::TREE_NONCANON, "non-canonical field element", false},
      {Tamper::MASK, "PC check failed|checked opening mismatch", true},
      {Tamper::DOT, "final check failed|checked opening mismatch", true},
      {Tamper::CHECK, "checked opening mismatch", false},
      {Tamper::OUTPUT, "output components disagree", true},
  };
  for (const K& k : kinds)
    for (int ch = 0; ch < 3; ++ch) {
      Watchdog wd(std::string("tamper ") + Tamper::name(k.k), 120);
      E2E R = run_lr(P, true, k.k, ch);
      const std::regex re(k.frag);
      bool ok = R.out.w.empty() && (R.p[CL].code == 3 || R.p[CL].code == 4) && R.fired[(size_t)ch];
      for (int i = 0; i < 3; ++i) ok = ok && (R.p[(size_t)i].code == 3 || R.p[(size_t)i].code == 4);
      bool detected = false;  // an honest party itself detected it, with the kind's message
      if (k.k == Tamper::OUTPUT) detected = R.p[CL].code == 3 && std::regex_search(R.p[CL].msg, re);
      for (int i = 0; i < 3 && k.k != Tamper::OUTPUT; ++i)
        if (i != ch && R.p[(size_t)i].code == 3 && std::regex_search(R.p[(size_t)i].msg, re)) detected = true;
      CHECK(ok && detected, "tamper %s by S%d: client %d (%s), S0 %d, S1 %d, S2 %d, fired %d, detected %d", Tamper::name(k.k), ch, R.p[CL].code,
            R.p[CL].msg.c_str(), R.p[0].code, R.p[1].code, R.p[2].code, (int)R.fired[(size_t)ch], (int)detected);
      std::string det;
      for (int p = 0; p < NP; ++p)
        if (R.p[(size_t)p].code == 3) { det = std::string(party_name(p)) + ": " + R.p[(size_t)p].msg; break; }
      std::printf("  %-18s by S%d -> abort (exit classes C/S0/S1/S2 = %d/%d/%d/%d; %s)\n", Tamper::name(k.k), ch, R.p[CL].code, R.p[0].code, R.p[1].code,
                  R.p[2].code, det.c_str());
    }
  int semi_runs = 0;
  for (const K& k : kinds) {
    for (int ch = 0; ch < 3; ++ch) {
      if (k.k == Tamper::OPEN_HASH || k.k == Tamper::CHECK) continue;  // exist only in malicious mode
      Watchdog wd(std::string("semi-honest tamper ") + Tamper::name(k.k), 120);
      E2E R = run_lr(P, false, k.k, ch);
      ++semi_runs;
      if (k.semi) {
        std::string why;
        CHECK(all_ok(R.p, &why) && R.out.w.size() == (size_t)P.d, "semi-honest tamper %s by S%d must not abort: %s", Tamper::name(k.k), ch, why.c_str());
      } else {  // a non-canonical field element is a parse error in both modes
        CHECK(R.p[CL].code != 0 && std::regex_search(R.p[(size_t)prev_of(ch)].msg, std::regex("non-canonical")),
              "semi-honest tree-noncanonical by S%d must abort", ch);
      }
    }
  }
  std::printf("  semi-honest: %d runs with the same deviations: no aborts (tree-noncanonical: parse error, aborts in both modes) ok\n", semi_runs);
}

int main(int argc, char** argv) {
  const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
  g_rng.seed((uint64_t)std::chrono::steady_clock::now().time_since_epoch().count());
  // below the Linux ephemeral range (32768-60999), so that no listening port collides with the source port of
  // an outgoing connection; ~70 cases x 8 ports
  g_port = 20000 + (int)(((uint64_t)getpid() * 7919u + (uint64_t)g_rng()) % 1400) * 8;
  test_ring();
  test_dealer();
  std::printf("[protocol functions over loopback]\n");
  for (bool mal : {true, false}) {
    test_pc(mal);
    test_relu_prime(mal);
    test_select(mal);
    test_matmul(mal);
  }
  test_branches();
  test_end_to_end(quick);
  if (!quick) test_tamper();
  std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}
