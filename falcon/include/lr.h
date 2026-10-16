// lr.h -- logistic-regression training with Falcon: the servers' side, the client's side (dealer, input
// owner, output receiver) and the analysis of the four parties' phase tables.
//
// Per iteration t (rows t*B .. t*B+B-1 of the client's data; Falcon FCLayer::forward / updateEquations,
// FCLayer.cpp:60-128, and ReLULayer, with SecureML's piecewise sigmoid written as a difference of ReLUs):
//   u   = trunc_F(X_t w)                          mat_vec                     online/fwd_matmul
//   a   = [u + 1/2 ; u - 1/2]                     public constants on component 0
//   b   = MSB(a)                                  relu_prime (2B elements)    online/drelu_*
//   r   = a (1 - b)                               select_shares (= ReLU)      online/select
//   e   = (r_1 - r_2) - y_t                       yhat = clamp(u + 1/2, 0, 1)
//   g   = trunc_{F+k}(X_t^T e)                    mat_vec (transposed)        online/bwd_matmul
//   w  -= g
// then [M] final_check, and the output: each server sends the client its status and its component(s) of w;
// the client checks ([M]: every component arrives twice and must agree), sums, and sends the verdict.
//
// Input. Semi-honest: X, y and w0 are dealt as RSS (correlated items: corrections to S0 and S2). Malicious:
// X enters as E = X - U, sent to all three servers (U = the uniform mask of the matrix triples), so that
// [X] = E + [U] and E is the public "rho" of both matrix checks of the iteration; y and w0 as RSS.
#ifndef FAL_LR_H
#define FAL_LR_H

#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "falcon.h"
#include "metrics.h"
#include "net.h"
#include "params.h"
#include "prep.h"
#include "rss.h"

namespace fal {

struct RunCfg {
  Params P;
  bool mal = true;
  bool quiet = true;
};

constexpr u8 kStatusOk = 1;
constexpr u8 kAccept = 1;

// Public parameters every party must agree on (the client's --seed / --zero-init are private and not sent).
struct Handshake {
  uint32_t magic, version;
  int32_t d, iters, batch, frac, qbits, bias, mal, reserved;
  double lr;
};

// The phases a server may report (the client rejects anything else).
inline const std::vector<std::string>& server_phase_names() {
  static const std::vector<std::string> n = {
      "net/connect",         "setup/handshake",      "setup/seeds",         "setup/keys",       "offline/deal",
      "online/input",        "online/fwd_matmul",    "online/drelu_open",   "online/drelu_pc_tree",
      "online/drelu_pc_check", "online/drelu_pc_open", "online/select",     "online/bwd_matmul", "online/final_check",
      "online/output"};
  return n;
}
inline const std::vector<std::string>& client_phase_names() {
  static const std::vector<std::string> n = {"net/connect", "setup/handshake", "setup/seeds", "offline/deal",
                                             "online/input", "online/wait", "online/output"};
  return n;
}
// Expected online steps of every server per phase.
inline uint64_t expected_steps(const std::string& phase, const Params& P, bool mal) {
  const uint64_t T = (uint64_t)P.iters;
  if (phase == "online/input") return 1;
  if (phase == "online/fwd_matmul" || phase == "online/bwd_matmul" || phase == "online/select") return 2 * T;
  if (phase == "online/drelu_open" || phase == "online/drelu_pc_open") return T;
  if (phase == "online/drelu_pc_tree") return 6 * T;
  if (phase == "online/drelu_pc_check") return mal ? T : 0;
  if (phase == "online/final_check") return mal ? 1 : 0;
  if (phase == "online/output") return 2;
  return 0;
}
inline uint64_t expected_online_steps(const Params& P, bool mal) {
  return mal ? 15 * (uint64_t)P.iters + 4 : 14 * (uint64_t)P.iters + 3;
}

// ---- setup (shared by both sides and by the unit tests) --------------------------------------------------
inline void handshake(Net& net, Recorder* rec, const Params& P, bool mal) {
  Scope sc(rec, "setup/handshake");
  Handshake h;
  std::memset(&h, 0, sizeof h);  // the padding must be zero for the memcmp below
  h.magic = 0x464C434Eu;
  h.version = 1;
  h.d = P.d; h.iters = P.iters; h.batch = P.batch; h.frac = P.frac; h.qbits = P.qbits;
  h.bias = P.bias ? 1 : 0; h.mal = mal ? 1 : 0; h.lr = P.lr;
  std::array<Handshake, NP> g;
  std::memset(g.data(), 0, sizeof g);
  Step s(net);
  for (int p = 0; p < NP; ++p)
    if (p != net.me()) { s.send(p, &h, sizeof h); s.recv(p, &g[(size_t)p], sizeof h); }
  s.run();
  for (int p = 0; p < NP; ++p)
    if (p != net.me() && std::memcmp(&h, &g[(size_t)p], sizeof h) != 0)
      throw AbortError(std::string(party_name(p)) + " was started with different parameters (d, iters, batch, frac, qbits, lr, bias, security mode)");
}

// client: handshake + seeds (k_0, k_1, k_2; S_i gets k_i || k_{i+1})
inline void client_setup(Net& net, Recorder* rec, const Params& P, bool mal, Dealer& D) {
  handshake(net, rec, P, mal);
  Scope sc(rec, "setup/seeds");
  u8 seeds[48];
  emp::PRG().random_data_unaligned(seeds, 48);
  Step s(net);
  for (int i = 0; i < 3; ++i) {
    u8 two[32];
    std::memcpy(two, seeds + 16 * i, 16);
    std::memcpy(two + 16, seeds + 16 * next_of(i), 16);
    s.send(i, two, 32);
  }
  s.run();
  D.init_client(seeds);
}
// server: handshake + seeds + PRSS keys (S_i sends kappa_i to next, receives kappa_{i-1} from prev)
inline void server_setup(Net& net, Recorder* rec, const Params& P, bool mal, Dealer& D, Ctx& cx) {
  handshake(net, rec, P, mal);
  {
    Scope sc(rec, "setup/seeds");
    u8 two[32];
    Step s(net);
    s.recv(CL, two, 32);
    s.run();
    D.init_server(net.me(), two);
  }
  Scope sc(rec, "setup/keys");
  u8 mine[16], got[16];
  emp::PRG().random_data_unaligned(mine, 16);
  Step s(net);
  s.send(cx.next(), mine, 16);
  s.recv(cx.prev(), got, 16);
  s.run();
  cx.prss.init(mine, got);
}
// client: one step, the correction bytes to S0 and S2 (the holders of component 0)
inline void send_corrections(Net& net, const std::vector<u8>& corr) {
  Step s(net);
  s.send_vec(S0, corr);
  s.send_vec(S2, corr);
  s.run();
}
// server: one step (empty at S1), exactly `len` correction bytes
inline void recv_corrections(Net& net, Dealer& D, size_t len) {
  Step s(net);
  D.corr.assign(net.me() == S1 ? 0 : len, 0);
  D.rpos = 0;
  if (net.me() != S1) s.recv_vec(CL, D.corr);
  s.run();
}

// ---- the servers ---------------------------------------------------------------------------------------
// LR training loop after Falcon's FCLayer::forward / updateEquations (FCLayer.cpp:60-128) and ReLULayer (ReLULayer.cpp:32-50)
inline void server_run(Net& net, Recorder& rec, const RunCfg& cfg, Tamper& tamper) {
  const Params& P = cfg.P;
  const bool mal = cfg.mal;
  const int me = net.me();
  const size_t B = (size_t)P.batch, d = (size_t)P.d, n = (size_t)P.n;
  Dealer D;
  Ctx cx(net, mal, tamper, &rec);
  server_setup(net, &rec, P, mal, D, cx);

  std::vector<IterPrep> prep;
  {
    Scope sc(rec, "offline/deal");
    recv_corrections(net, D, deal_payload_bytes(P, mal));
    deal_all(D, P, mal, prep, nullptr);
    D.expect_consumed("offline");
  }
  if (!cfg.quiet) std::fprintf(stderr, "[S%d] offline phase done\n", me);

  Rv64 X(n * d), Y(n), w(d);
  std::vector<u64> E;
  {
    Scope sc(rec, "online/input");
    const uint64_t len = input_payload_bytes(P, mal, me);
    std::vector<u8> buf(len);
    Step s(net);
    if (len) s.recv_vec(CL, buf);
    s.run();
    size_t off = 0;
    if (mal) {
      E.resize(n * d);
      std::memcpy(E.data(), buf.data(), 8 * n * d);
      off = 8 * n * d;
    }
    D.corr.assign(buf.begin() + (std::ptrdiff_t)off, buf.end());
    D.rpos = 0;
    if (!mal)
      for (size_t i = 0; i < n * d; ++i) X[i] = D.view(D.corr64(0));
    for (size_t i = 0; i < n; ++i) Y[i] = D.view(D.corr64(0));
    for (size_t j = 0; j < d; ++j) w[j] = D.view(D.corr64(0));
    D.expect_consumed("input");
    if (mal)
      for (size_t t = 0; t < (size_t)P.iters; ++t)
        for (size_t k = 0; k < B * d; ++k) {
          RSS<u64> x = prep[t].mat.U[k];
          add_const<Ring64>(me, x, E[t * B * d + k]);
          X[t * B * d + k] = x;
        }
  }

  const u64 half = u64(1) << (P.frac - 1);
  for (int t = 0; t < P.iters; ++t) {
    IterPrep& I = prep[(size_t)t];
    const Rv64 Xt(X.begin() + (std::ptrdiff_t)((size_t)t * B * d), X.begin() + (std::ptrdiff_t)((size_t)(t + 1) * B * d));
    std::vector<u64> Et;
    MatCheck mf{}, mb{};
    if (mal) {
      Et.assign(E.begin() + (std::ptrdiff_t)((size_t)t * B * d), E.begin() + (std::ptrdiff_t)((size_t)(t + 1) * B * d));
      mf = {&Et, &I.mat.U, &I.mat.vf, &I.mat.Zf};
      mb = {&Et, &I.mat.U, &I.mat.vb, &I.mat.Zb};
    }
    const Rv64 u = mat_vec(cx, Xt, w, B, d, false, P.shift_fwd(), I.fwd, mal ? &mf : nullptr, "online/fwd_matmul");
    Rv64 a(2 * B);
    for (size_t i = 0; i < B; ++i) {
      a[i] = u[i];
      add_const<Ring64>(me, a[i], half);
      a[B + i] = u[i];
      add_const<Ring64>(me, a[B + i], (u64)0 - half);
    }
    const Rv8 b = relu_prime(cx, a, I.wrap);
    const Rv64 r = select_shares(cx, a, b, I.sel);
    Rv64 e(B);
    for (size_t i = 0; i < B; ++i) e[i] = rsub<Ring64>(rsub<Ring64>(r[i], r[B + i]), Y[(size_t)t * B + i]);
    const Rv64 g = mat_vec(cx, Xt, e, d, B, true, P.shift_bwd(), I.bwd, mal ? &mb : nullptr, "online/bwd_matmul");
    for (size_t j = 0; j < d; ++j) w[j] = rsub<Ring64>(w[j], g[j]);
    I = IterPrep{};  // free this iteration's randomness
    if (!cfg.quiet && (t == 0 || (t + 1) % 5 == 0 || t + 1 == P.iters))
      std::fprintf(stderr, "[S%d] iteration %d/%d done\n", me, t + 1, P.iters);
  }
  if (mal) final_check(cx);

  {
    Scope sc(rec, "online/output");
    std::vector<u8> out(1 + 8 * d * (mal ? 2 : 1));
    out[0] = kStatusOk;
    for (size_t j = 0; j < d; ++j) {
      std::memcpy(out.data() + 1 + 8 * j, &w[j].first, 8);
      if (mal) std::memcpy(out.data() + 1 + 8 * (d + j), &w[j].second, 8);
    }
    if (tamper.hit(Tamper::OUTPUT, me)) out[1] ^= 1;
    {
      Step s(net);
      s.send_vec(CL, out);
      s.run();
    }
    u8 verdict = 0;
    Step s(net);
    s.recv(CL, &verdict, 1);
    s.run();
    if (verdict != kAccept) throw PeerError("peer client reported abort (verdict)");
  }

  // report (after ACCEPT, outside the measured phases): row count, rows, peak RSS
  const auto& rows = rec.rows();
  std::vector<u8> rep(4 + rows.size() * sizeof(PhaseRow) + 8);
  const uint32_t cnt = (uint32_t)rows.size();
  const int64_t rss = peak_rss_kib();
  std::memcpy(rep.data(), &cnt, 4);
  std::memcpy(rep.data() + 4, rows.data(), rows.size() * sizeof(PhaseRow));
  std::memcpy(rep.data() + 4 + rows.size() * sizeof(PhaseRow), &rss, 8);
  Step s(net);
  s.send_vec(CL, rep);
  s.run();
}

// ---- the client --------------------------------------------------------------------------------------
struct ClientOutcome {
  std::vector<u64> w;                 // trained weights (ring elements, scale F)
  ClientView cv;                      // the dealt truncation masks (for the bit-exact reference)
  std::array<std::vector<PhaseRow>, NP> rows;
  std::array<long, NP> rss_kib{0, 0, 0, 0};
};

inline ClientOutcome client_run(Net& net, Recorder& rec, const RunCfg& cfg, const Dataset& data) {
  const Params& P = cfg.P;
  const bool mal = cfg.mal;
  const size_t d = (size_t)P.d, n = (size_t)P.n;
  ClientOutcome O;
  Dealer D;
  client_setup(net, &rec, P, mal, D);
  {
    Scope sc(rec, "offline/deal");
    std::vector<IterPrep> prep;
    deal_all(D, P, mal, prep, &O.cv);
    if (D.corr.size() != deal_payload_bytes(P, mal)) throw std::logic_error("dealer byte formula mismatch");
    send_corrections(net, D.corr);
  }
  if (!cfg.quiet) std::fprintf(stderr, "[client] offline phase done: %zu correction bytes to S0 and to S2\n", D.corr.size());
  {
    Scope sc(rec, "online/input");
    D.corr.clear();
    std::vector<u64> E;
    if (mal) {
      E.resize(n * d);
      for (size_t i = 0; i < n * d; ++i) E[i] = (u64)data.x[i] - O.cv.U[i];
    } else {
      for (size_t i = 0; i < n * d; ++i) D.corr64((u64)data.x[i]);
    }
    for (size_t i = 0; i < n; ++i) D.corr64((u64)data.y[i] << P.frac);
    for (size_t j = 0; j < d; ++j) D.corr64((u64)data.w0[j]);
    Step s(net);
    for (int i = 0; i < 3; ++i) {
      std::vector<u8> m;
      if (mal) m.insert(m.end(), (const u8*)E.data(), (const u8*)(E.data() + E.size()));
      if (i != S1) m.insert(m.end(), D.corr.begin(), D.corr.end());
      if (m.size() != input_payload_bytes(P, mal, i)) throw std::logic_error("input byte formula mismatch");
      if (!m.empty()) s.send_vec(i, m);
    }
    s.run();
  }
  {
    Scope sc(rec, "online/wait");  // idle while the servers train
    net.wait_any_readable({S0, S1, S2});
  }
  {
    Scope sc(rec, "online/output");
    const size_t per = 1 + 8 * d * (mal ? 2 : 1);
    std::array<std::vector<u8>, 3> got;
    Step s(net);
    for (int i = 0; i < 3; ++i) { got[(size_t)i].resize(per); s.recv_vec(i, got[(size_t)i]); }
    s.run();
    for (int i = 0; i < 3; ++i)
      if (got[(size_t)i][0] != kStatusOk) throw AbortError(std::string("bad output status from ") + party_name(i));
    auto comp = [&](int srv, int which, size_t j) {  // which 0 = .first, 1 = .second
      u64 v;
      std::memcpy(&v, got[(size_t)srv].data() + 1 + 8 * (which * d + j), 8);
      return v;
    };
    O.w.assign(d, 0);
    for (size_t j = 0; j < d; ++j)
      for (int c = 0; c < 3; ++c) {
        const u64 v = comp(c, 0, j);  // component c from S_c (.first)
        if (mal && comp(prev_of(c), 1, j) != v) throw AbortError("output components disagree");  // ... and from S_{c-1} (.second)
        O.w[j] += v;
      }
    Step v(net);
    for (int i = 0; i < 3; ++i) v.send(i, &kAccept, 1);
    v.run();
  }
  // reports of the servers (after ACCEPT, outside the measured phases)
  std::array<std::vector<u8>, 3> rep;
  Step s(net);
  for (int i = 0; i < 3; ++i) s.recv_any(i, &rep[(size_t)i]);
  s.run();
  for (int i = 0; i < 3; ++i) {
    const auto& b = rep[(size_t)i];
    uint32_t cnt = 0;
    if (b.size() < 12) throw std::runtime_error("bad report from " + std::string(party_name(i)));
    std::memcpy(&cnt, b.data(), 4);
    if (cnt > server_phase_names().size() || b.size() != 4 + cnt * sizeof(PhaseRow) + 8)
      throw std::runtime_error("bad report from " + std::string(party_name(i)));
    O.rows[(size_t)i].resize(cnt);
    std::memcpy(O.rows[(size_t)i].data(), b.data() + 4, cnt * sizeof(PhaseRow));
    int64_t rss = 0;
    std::memcpy(&rss, b.data() + 4 + cnt * sizeof(PhaseRow), 8);
    O.rss_kib[(size_t)i] = (long)rss;
    for (auto& r : O.rows[(size_t)i]) {
      r.name[39] = 0;
      bool known = false;
      for (const auto& nm : server_phase_names()) known = known || nm == r.name;
      if (!known) throw std::runtime_error("unknown phase '" + std::string(r.name) + "' in the report of " + party_name(i));
    }
  }
  O.rows[CL] = rec.rows();
  O.rss_kib[CL] = peak_rss_kib();
  return O;
}

// ---- analysis of the four phase tables ------------------------------------------------------------------
struct Totals {
  StageTotals st[NP][3];                 // [party][setup, offline, online]
  StageTotals client_online_compute;     // client online without online/wait
  double client_wait_s = 0;
  uint64_t c2s_off = 0, s2c_off = 0, s2s_off = 0, c2s_on = 0, s2c_on = 0, s2s_on = 0, total = 0;
  uint64_t sum_sent = 0, sum_recv = 0;
  uint64_t online_steps[3] = {0, 0, 0};
  std::string error;                     // empty if every invariant holds
};

inline Totals analyze(const ClientOutcome& O, const Params& P, bool mal) {
  Totals T;
  const char* stages[] = {"setup", "offline", "online"};
  for (int p = 0; p < NP; ++p)
    for (int s = 0; s < 3; ++s) T.st[p][s] = stage_totals(O.rows[(size_t)p], stages[s]);
  T.client_online_compute = stage_totals(O.rows[CL], "online", "online/wait");
  if (const PhaseRow* r = find_row(O.rows[CL], "online/wait")) T.client_wait_s = r->wall_s;
  for (int s = 0; s < 3; ++s) {
    uint64_t& c2s = s < 2 ? T.c2s_off : T.c2s_on;
    uint64_t& s2c = s < 2 ? T.s2c_off : T.s2c_on;
    uint64_t& s2s = s < 2 ? T.s2s_off : T.s2s_on;
    c2s += T.st[CL][s].sent_total();
    for (int i = 0; i < 3; ++i) {
      s2c += T.st[i][s].sent[CL];
      s2s += T.st[i][s].sent[0] + T.st[i][s].sent[1] + T.st[i][s].sent[2];
    }
    for (int p = 0; p < NP; ++p) { T.sum_sent += T.st[p][s].sent_total(); T.sum_recv += T.st[p][s].recv_total(); }
  }
  T.total = T.c2s_off + T.s2c_off + T.s2s_off + T.c2s_on + T.s2c_on + T.s2s_on;
  char buf[256];
  if (T.total != T.sum_sent || T.total != T.sum_recv) {
    std::snprintf(buf, sizeof buf, "communication totals disagree: columns %" PRIu64 ", sent %" PRIu64 ", received %" PRIu64 "; ", T.total, T.sum_sent, T.sum_recv);
    T.error += buf;
  }
  // every link: bytes sent by p to q (setup..online) == bytes received by q from p
  for (int p = 0; p < NP; ++p)
    for (int q = 0; q < NP; ++q) {
      if (p == q) continue;
      uint64_t sp = 0, rq = 0;
      for (int s = 0; s < 3; ++s) { sp += T.st[p][s].sent[q]; rq += T.st[q][s].recv[p]; }
      if (sp != rq) {
        std::snprintf(buf, sizeof buf, "%s->%s: %" PRIu64 " bytes sent, %" PRIu64 " received; ", party_name(p), party_name(q), sp, rq);
        T.error += buf;
      }
    }
  // the three servers run the same step schedule; the online steps are as designed
  for (int i = 0; i < 3; ++i) T.online_steps[i] = T.st[i][2].steps;
  for (const auto& r : O.rows[S0]) {
    const PhaseRow* a = find_row(O.rows[S1], r.name);
    const PhaseRow* b = find_row(O.rows[S2], r.name);
    if (!a || !b || a->steps != r.steps || b->steps != r.steps) {
      std::snprintf(buf, sizeof buf, "steps of %s differ between the servers; ", r.name);
      T.error += buf;
    }
    if (std::strncmp(r.name, "online/", 7) == 0 && r.steps != expected_steps(r.name, P, mal)) {
      std::snprintf(buf, sizeof buf, "%s: %" PRIu64 " steps, expected %" PRIu64 "; ", r.name, r.steps, expected_steps(r.name, P, mal));
      T.error += buf;
    }
  }
  if (O.rows[S0].size() != O.rows[S1].size() || O.rows[S0].size() != O.rows[S2].size()) T.error += "the servers report different phases; ";
  if (T.online_steps[0] != expected_online_steps(P, mal)) {
    std::snprintf(buf, sizeof buf, "online steps %" PRIu64 ", expected %" PRIu64 "; ", T.online_steps[0], expected_online_steps(P, mal));
    T.error += buf;
  }
  return T;
}

}  // namespace fal

#endif  // FAL_LR_H
