// driver.h -- one party of the protocol, start to finish: connect, handshake, OT bootstrap, offline phase,
// online phase, report. Used by the lr_spdz CLI (one process per party) and by the selftest (one forked
// process per party).
//
//   party 0          the server: no input; learns the trained weights (and only it) and prints the report
//   party 1..batch   clients: client k holds datapoint k of every batch and one share of the initial weights
//
// Exit codes: 0 ok, 1 error, 2 verify mismatch, 3 this party detected a deviation (ABORT).
#ifndef SPZ_DRIVER_H
#define SPZ_DRIVER_H

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "metrics.h"
#include "mpc.h"
#include "online.h"
#include "params.h"
#include "preproc.h"

namespace spz {

struct PartyConfig {
  Params P;
  int party = 0;                       // 0 = server, 1..batch = client
  std::vector<std::string> hosts;      // hosts[i] = address of party i
  int port = 12345;
  int threads = 1;                     // OT connections / worker threads per pair of parties
  bool semi_honest = false;
  bool verify = false, quiet = false;
  std::string json_path, csv_path;
  Tamper::Kind tamper = Tamper::NONE;
  int cheater = -1;
  bool print_report = true;            // the server prints the report
};

namespace detail {

struct Handshake {
  int32_t d, iters, batch, frac, qbits, ssp, threads, mal, bias;
  double lr;
};

inline std::string ms(double s) { char b[32]; std::snprintf(b, sizeof b, "%.1f", s * 1e3); return b; }

struct Report {
  int n = 0;
  std::vector<PhaseRow> rows[kMaxParties];  // [party]
  long rss_kib[kMaxParties] = {0};
};

inline const char* party_name(int p, char* buf) {
  if (p == 0) std::snprintf(buf, 16, "server");
  else std::snprintf(buf, 16, "client %d", p);
  return buf;
}

}  // namespace detail

inline int run_party(const PartyConfig& C) {
  using namespace detail;
  const Params& P = C.P;
  const bool mal = !C.semi_honest;
  const int me = C.party, n = P.batch + 1;
  char who[16];
  party_name(me, who);
  if (!C.quiet)
    std::fprintf(stderr, "[%s] %s | %s | threads=%d\n", who, mal ? "MALICIOUS security (MACs + sacrifice)" : "semi-honest", params_string(P).c_str(), C.threads);
  try {
    if ((int)C.hosts.size() != n) throw std::invalid_argument("--hosts needs one address per party (server first)");
    if (Mesh::max_port(C.port, n) > 65535) throw std::invalid_argument("--port out of range");
    Mesh mesh(me, C.hosts, C.port, C.threads, mal);
    Mpc M(mesh, mal);
    if (C.tamper != Tamper::NONE) { M.tamper.kind = C.tamper; M.tamper.who = C.cheater; }
    Recorder rec(mesh);
    auto for_peers = [&](const std::function<void(int)>& f) { parallel(n - 1, [&](int idx) { f(idx < me ? idx : idx + 1); }); };

    // ---- setup: agree on parameters, bring up the OT-extension sessions of every channel -------------
    {
      Scope sc(rec, "setup/handshake");
      Handshake h;
      std::memset(&h, 0, sizeof h);  // the tail padding is otherwise indeterminate and would break the memcmp
      h.d = P.d; h.iters = P.iters; h.batch = P.batch; h.frac = P.frac; h.qbits = P.qbits; h.ssp = P.ssp;
      h.threads = mesh.nw(); h.mal = mal ? 1 : 0; h.bias = P.bias ? 1 : 0; h.lr = P.lr;
      std::vector<std::vector<uint8_t>> th;
      mesh.all_to_all(&h, sizeof h, th);
      for (int p = 0; p < n; ++p)
        if (p != me && std::memcmp(&h, th[(size_t)p].data(), sizeof h) != 0)
          throw AbortError("party " + std::to_string(p) + " was started with different parameters (d, iters, batch, frac, qbits, ssp, lr, bias, threads, mode)");
    }
    {
      Scope sc(rec, "setup/ot_base");  // base OTs + OT-extension bootstrap of every channel, both directions
      for_peers([&](int p) {
        PeerLink& L = mesh.peers[(size_t)p];
        const int nw = L.nw();
        std::vector<Fp> A((size_t)nw * 2, Fp(1)), S((size_t)nw * 2), b((size_t)nw, Fp(1)), R((size_t)nw * 2);
        for (int dir = 0; dir < 2; ++dir) {
          if (me == (dir == 0 ? std::min(me, p) : std::max(me, p))) product_send(L, nw, 2, A.data(), S.data());
          else product_recv(L, nw, 2, b.data(), R.data());
        }
      });
    }

    // ---- offline ---------------------------------------------------------------------------------------
    Pre pre;
    generate_preprocessing(M, P, pre, rec);
    if (!C.quiet) std::fprintf(stderr, "[%s] offline phase done: %" PRId64 " triples, %" PRId64 " random bits\n", who, pre.NT, pre.NB);

    // ---- online ----------------------------------------------------------------------------------------
    Dataset D;
    if (me >= 1 || C.verify) D = generate_client_data(P);
    Online on(M, P, pre, rec);
    on.input(me >= 1 ? &D : nullptr);
    on.train(C.quiet);
    const std::vector<i128> w = on.output();

    // ---- every party sends its table to the server (not part of any phase) ------------------------------
    Report R;
    R.n = n;
    R.rows[me] = rec.rows();
    R.rss_kib[me] = peak_rss_kib();
    if (me != 0) {
      NetIO& io = *mesh.peers[0].main;
      uint64_t cnt = rec.rows().size();
      io.send_data(&cnt, sizeof cnt);
      io.send_data(rec.rows().data(), (int64_t)(cnt * sizeof(PhaseRow)));
      io.send_data(&R.rss_kib[me], sizeof(long));
      io.flush();
    } else {
      for (int p = 1; p < n; ++p) {
        NetIO& io = *mesh.peers[(size_t)p].main;
        uint64_t cnt = 0;
        io.recv_data(&cnt, sizeof cnt);
        if (cnt != rec.rows().size()) throw std::runtime_error("report mismatch");
        R.rows[p].resize(cnt);
        io.recv_data(R.rows[p].data(), (int64_t)(cnt * sizeof(PhaseRow)));
        io.recv_data(&R.rss_kib[p], sizeof(long));
      }
    }
    if (me != 0) return 0;

    // ============================ the server prints the report =============================================
    int rc = 0;
    const char* stages[] = {"setup", "offline", "online"};
    StageTotals T[3][kMaxParties];
    for (int s = 0; s < 3; ++s)
      for (int p = 0; p < n; ++p) T[s][p] = stage_totals(R.rows[p], stages[s]);

    // invariant: what p sent to q equals what q received from p (over all phases)
    bool comm_ok = true;
    uint64_t total_sent = 0, total_recv = 0;
    for (int p = 0; p < n; ++p)
      for (int q = 0; q < n; ++q) {
        if (p == q) continue;
        uint64_t s = 0, r = 0;
        for (int st = 0; st < 3; ++st) { s += T[st][p].sent_to[q]; r += T[st][q].recv_from[p]; }
        if (s != r) comm_ok = false;
        total_sent += s;
        total_recv += r;
      }
    // invariant: every party counts the same number of rounds in every stage
    bool rounds_ok = true;
    for (int s = 0; s < 3; ++s)
      for (int p = 1; p < n; ++p) if (T[s][p].rounds != T[s][0].rounds) rounds_ok = false;

    std::vector<i128> wref;
    bool exact = true, in_range = true;
    std::vector<double> dref;
    double maxd = 0;
    if (C.verify) {
      wref = reference_train(P, D, &in_range);
      for (int j = 0; j < P.d; ++j) if (wref[(size_t)j] != w[(size_t)j]) exact = false;
      dref = reference_train_double(P, D);
      for (int j = 0; j < P.d; ++j) maxd = std::max(maxd, std::fabs(std::ldexp((double)w[(size_t)j], -P.sw) - dref[(size_t)j]));
      if (!exact || !in_range) rc = 2;
    }
    if (!comm_ok && rc == 0) rc = 1;

    auto stage_all = [&](int s, int p) { return T[s][p]; };
    (void)stage_all;
    // computation per party and stage
    auto comp_wall = [&](int s, int p) { return T[s][p].wall_s; };
    auto comp_cpu = [&](int s, int p) { return T[s][p].cpu_s; };
    auto cli_max = [&](auto f) { double m = 0; for (int p = 1; p < n; ++p) m = std::max(m, f(p)); return m; };
    auto cli_mean = [&](auto f) { double m = 0; for (int p = 1; p < n; ++p) m += f(p); return m / (n - 1); };
    auto sent_of = [&](int s, int p) { return T[s][p].sent; };
    auto recv_of = [&](int s, int p) { return T[s][p].recv; };

    if (C.print_report) {
      std::printf("\nSPDZ-style %d-party logistic regression (%d clients + 1 server), %s\n%s\n", n, n - 1,
                  mal ? "MALICIOUS security (dishonest majority, security with abort)" : "semi-honest", params_string(P).c_str());
      std::printf("trained weights (value = int / 2^%d):", P.sw);
      for (int j = 0; j < P.d && j < 8; ++j) std::printf(" %.5f", std::ldexp((double)w[(size_t)j], -P.sw));
      if (P.d > 8) std::printf(" ... (%d weights)", P.d);
      std::printf("\n");
      if (C.verify)
        std::printf("verify: %s the plaintext integer reference%s; max |w - w_double| = %.3g\n",
                    exact ? "EXACT MATCH with" : "MISMATCH against", in_range ? "" : " (gadget range exceeded!)", maxd);

      std::printf("\n%-26s | %9s %9s | %9s %9s | %9s %9s | %7s %7s\n", "phase (times in ms)", "srv wall", "srv cpu", "cli wall", "cli cpu",
                  "srv sent", "cli sent", "rounds", "ot-fl");
      std::printf("%-26s | %19s | %19s | %19s | %15s\n", "", "", "(max of clients)", "(mean of clients)", "");
      for (size_t i = 0; i < R.rows[0].size(); ++i) {
        const PhaseRow& s = R.rows[0][i];
        double cw = 0, cc = 0;
        uint64_t cs = 0;
        for (int p = 1; p < n; ++p) {
          cw = std::max(cw, R.rows[p][i].wall_s); cc = std::max(cc, R.rows[p][i].cpu_s);
          cs += R.rows[p][i].sent_total();
        }
        std::printf("%-26s | %9s %9s | %9s %9s | %9s %9s | %7" PRIu64 " %7" PRIu64 "\n", s.name, ms(s.wall_s).c_str(), ms(s.cpu_s).c_str(),
                    ms(cw).c_str(), ms(cc).c_str(), human_bytes(s.sent_total()).c_str(), human_bytes(cs / (uint64_t)(n - 1)).c_str(),
                    s.rounds, s.ot_flights_max);
      }

      const uint64_t on_rounds = T[2][0].rounds, off_rounds = T[1][0].rounds, setup_rounds = T[0][0].rounds;
      std::printf("\nSUMMARY (%s, d=%d, iters=%d, batch=%d => %d clients + 1 server, %d OT connection(s) per pair)\n",
                  mal ? "malicious" : "semi-honest", P.d, P.iters, P.batch, P.batch, mesh.nw());
      std::printf("\n  COMMUNICATION ROUNDS  (a round = every party sends to every other party at once; all parties count the same)\n");
      std::printf("    online   %" PRIu64 " rounds  (%.1f per iteration; input 1, output opening 1, final MAC check %d)\n", on_rounds,
                  (double)(on_rounds - 2 - (mal ? 4 : 0)) / P.iters, mal ? 4 : 0);
      std::printf("    offline  %" PRIu64 " main-channel rounds, plus %" PRIu64 " OT-channel flights on the busiest channel (OT products run in parallel)\n",
                  off_rounds, T[1][0].ot_flights_max);
      std::printf("    setup    %" PRIu64 " main-channel rounds, plus %" PRIu64 " OT-channel flights (base OTs)\n", setup_rounds, T[0][0].ot_flights_max);
      std::printf("    total    %" PRIu64 " main-channel rounds (setup + offline + online)\n", on_rounds + off_rounds + setup_rounds);

      std::printf("\n  TOTAL COMMUNICATION   %s  (sum of bytes sent over all %d parties; = bytes received: %s)\n", human_bytes(total_sent).c_str(), n,
                  comm_ok ? "verified" : "MISMATCH");
      uint64_t st[3] = {0, 0, 0};
      for (int s = 0; s < 3; ++s) for (int p = 0; p < n; ++p) st[s] += T[s][p].sent;
      std::printf("    setup %s | offline %s | online %s\n", human_bytes(st[0]).c_str(), human_bytes(st[1]).c_str(), human_bytes(st[2]).c_str());

      std::printf("\n  PER-PARTY COMMUNICATION (payload bytes, headers excluded)\n");
      std::printf("    %-9s | %12s %12s | %12s %12s | %12s %12s\n", "party", "sent", "received", "offline sent", "offline recv", "online sent", "online recv");
      for (int p = 0; p < n; ++p) {
        char nm[16];
        std::printf("    %-9s | %12s %12s | %12s %12s | %12s %12s\n", party_name(p, nm),
                    human_bytes(sent_of(0, p) + sent_of(1, p) + sent_of(2, p)).c_str(), human_bytes(recv_of(0, p) + recv_of(1, p) + recv_of(2, p)).c_str(),
                    human_bytes(sent_of(1, p)).c_str(), human_bytes(recv_of(1, p)).c_str(), human_bytes(sent_of(2, p)).c_str(), human_bytes(recv_of(2, p)).c_str());
      }

      std::printf("\n  PER-PARTY COMPUTATION (ms; cpu = whole process, all threads; wall includes waiting for the others)\n");
      std::printf("    %-9s | %10s %10s | %10s %10s | %10s %10s | %9s\n", "party", "setup wall", "setup cpu", "offl. wall", "offl. cpu", "onl. wall", "onl. cpu", "peak RSS");
      for (int p = 0; p < n; ++p) {
        char nm[16];
        std::printf("    %-9s | %10s %10s | %10s %10s | %10s %10s | %6.0f MiB\n", party_name(p, nm), ms(comp_wall(0, p)).c_str(), ms(comp_cpu(0, p)).c_str(),
                    ms(comp_wall(1, p)).c_str(), ms(comp_cpu(1, p)).c_str(), ms(comp_wall(2, p)).c_str(), ms(comp_cpu(2, p)).c_str(), R.rss_kib[p] / 1024.0);
      }
      std::printf("    server cpu offline+online: %s ms; per client (mean): %s ms, (max): %s ms\n",
                  ms(comp_cpu(1, 0) + comp_cpu(2, 0)).c_str(), ms(cli_mean([&](int p) { return comp_cpu(1, p) + comp_cpu(2, p); })).c_str(),
                  ms(cli_max([&](int p) { return comp_cpu(1, p) + comp_cpu(2, p); })).c_str());
      std::printf("    random OTs consumed (summed over all parties as sender+receiver counted once per party): %" PRIu64 "\n",
                  [&] { uint64_t o = 0; for (int s = 0; s < 3; ++s) for (int p = 0; p < n; ++p) o += T[s][p].ots; return o; }());
      if (!rounds_ok) std::printf("  WARNING: the parties counted different numbers of rounds\n");
    }

    // ---- JSON / CSV (server) ---------------------------------------------------------------------------
    if (!C.json_path.empty()) {
      std::ofstream f(C.json_path);
      f << "{\n  \"protocol\": \"mpc_spdz\",\n  \"security\": \"" << (mal ? "malicious" : "semi-honest") << "\",\n  \"d\": " << P.d
        << ", \"iters\": " << P.iters << ", \"batch\": " << P.batch << ", \"parties\": " << n << ", \"frac\": " << P.frac
        << ", \"threads\": " << mesh.nw() << ",\n  \"phases\": [\n";
      bool first = true;
      for (int p = 0; p < n; ++p)
        for (const auto& r : R.rows[p]) {
          if (!first) f << ",\n";
          first = false;
          f << "    {\"party\": " << p << ", \"phase\": \"" << r.name << "\", \"wall_s\": " << r.wall_s << ", \"cpu_s\": " << r.cpu_s
            << ", \"rounds\": " << r.rounds << ", \"ot_flights_max\": " << r.ot_flights_max << ", \"ots\": " << r.ots << ", \"sent_to\": [";
          for (int q = 0; q < n; ++q) f << (q ? ", " : "") << r.sent[q];
          f << "], \"recv_from\": [";
          for (int q = 0; q < n; ++q) f << (q ? ", " : "") << r.recv[q];
          f << "]}";
        }
      f << "\n  ]\n}\n";
    }
    if (!C.csv_path.empty()) {
      const std::string header =
          "protocol,security,d,iters,batch,parties,frac,threads,verified,"
          "rounds_setup,rounds_offline_main,rounds_offline_ot_flights,rounds_online,"
          "comm_total_bytes,comm_setup_bytes,comm_offline_bytes,comm_online_bytes,"
          "server_sent_bytes,server_recv_bytes,client_sent_mean_bytes,client_sent_max_bytes,client_recv_mean_bytes,client_recv_max_bytes,"
          "server_setup_cpu_ms,server_offline_wall_ms,server_offline_cpu_ms,server_online_wall_ms,server_online_cpu_ms,"
          "client_offline_wall_max_ms,client_offline_cpu_mean_ms,client_offline_cpu_max_ms,client_online_wall_max_ms,client_online_cpu_mean_ms,client_online_cpu_max_ms";
      bool exists = false;
      {
        std::ifstream in(C.csv_path);
        std::string first;
        if (in && std::getline(in, first)) {
          exists = true;
          if (first != header) { std::fprintf(stderr, "error: %s has a different header\n", C.csv_path.c_str()); return 1; }
        }
      }
      std::ofstream f(C.csv_path, std::ios::app);
      if (!exists) f << header << "\n";
      auto cs_mean = [&](bool recv) { double m = 0; for (int p = 1; p < n; ++p) m += (double)(recv ? T[0][p].recv + T[1][p].recv + T[2][p].recv : T[0][p].sent + T[1][p].sent + T[2][p].sent); return m / (n - 1); };
      auto cs_max = [&](bool recv) { uint64_t m = 0; for (int p = 1; p < n; ++p) m = std::max<uint64_t>(m, recv ? T[0][p].recv + T[1][p].recv + T[2][p].recv : T[0][p].sent + T[1][p].sent + T[2][p].sent); return m; };
      uint64_t st[3] = {0, 0, 0};
      for (int s = 0; s < 3; ++s) for (int p = 0; p < n; ++p) st[s] += T[s][p].sent;
      char row[2048];
      std::snprintf(row, sizeof row,
                    "mpc_spdz,%s,%d,%d,%d,%d,%d,%d,%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                    ",%" PRIu64 ",%" PRIu64 ",%.0f,%" PRIu64 ",%.0f,%" PRIu64 ",%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
                    mal ? "malicious" : "semi-honest", P.d, P.iters, P.batch, n, P.frac, mesh.nw(), C.verify ? (rc == 0 ? "exact" : "FAIL") : "no",
                    T[0][0].rounds, T[1][0].rounds, T[1][0].ot_flights_max, T[2][0].rounds, total_sent, st[0], st[1], st[2],
                    T[0][0].sent + T[1][0].sent + T[2][0].sent, T[0][0].recv + T[1][0].recv + T[2][0].recv, cs_mean(false), cs_max(false), cs_mean(true), cs_max(true),
                    comp_cpu(0, 0) * 1e3, comp_wall(1, 0) * 1e3, comp_cpu(1, 0) * 1e3, comp_wall(2, 0) * 1e3, comp_cpu(2, 0) * 1e3,
                    cli_max([&](int p) { return comp_wall(1, p); }) * 1e3, cli_mean([&](int p) { return comp_cpu(1, p); }) * 1e3,
                    cli_max([&](int p) { return comp_cpu(1, p); }) * 1e3, cli_max([&](int p) { return comp_wall(2, p); }) * 1e3,
                    cli_mean([&](int p) { return comp_cpu(2, p); }) * 1e3, cli_max([&](int p) { return comp_cpu(2, p); }) * 1e3);
      f << row << "\n";
    }
    return rc;
  } catch (const AbortError& e) {
    std::fprintf(stderr, "[%s] ABORT: %s\n", who, e.what());
    return 3;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[%s] error: %s\n", who, e.what());
    return 1;
  }
}

}  // namespace spz

#endif  // SPZ_DRIVER_H
