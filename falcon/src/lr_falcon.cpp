// lr_falcon -- Falcon (Wagh, Tople, Benhamouda, Kushilevitz, Mittal, Rabin; PETS 2021) logistic-regression
// training on three servers with a client that deals all correlated randomness, with malicious security.
//
//   client (party 3): owns every private input -- the training data (x, y) and the initial weights w0 --,
//                     deals all correlated randomness, and is the only party that learns the trained weights
//   servers S0, S1, S2 (parties 0, 1, 2 = Falcon's PARTY_A/B/C): no input, learn nothing; they run Falcon's
//                     3-party replicated-secret-sharing protocols (honest majority: at most one is corrupt)
//
// The protocol functions are ported from falcon-public (with fixes) in include/falcon.h; the malicious
// layer (checked openings, triple checks, check-before-reveal in PrivateCompare, the final check and the
// checked output) is added on top, and --semi-honest turns it off. See README.md.
//
// usage: lr_falcon client [options]  |  lr_falcon server --id I [options]          (--help)
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "falcon.h"
#include "lr.h"
#include "metrics.h"
#include "net.h"
#include "params.h"

using namespace fal;

namespace {

struct Options {
  int me = CL;
  Params P;
  NetConfig net;
  bool semi_honest = false, verify = false, quiet = false;
  int threads = 1;
  std::string json_path, csv_path;
  Tamper::Kind tamper = Tamper::NONE;
  int cheater = -1;
};

void usage(const char* prog) {
  std::printf(
      "usage: %s client [options]\n"
      "       %s server --id I [options]          (I = 0, 1, 2)\n"
      "  client = owns the data and the initial weights, deals the correlated randomness, learns the weights\n"
      "  servers = run the protocol, have no input and learn nothing\n\n"
      "training (all four parties must pass the same values; --seed / --zero-init only shape the client's\n"
      "private inputs and are not sent to the servers):\n"
      "  --d N            features incl. the bias column (default 5)\n"
      "  --iters N        SGD iterations (default 20)\n"
      "  --batch N        datapoints per iteration (default 4)\n"
      "  --frac N         fractional bits F (default 13 = Falcon's FLOAT_PRECISION)\n"
      "  --lr X           learning rate; batch/lr must be a power of two (default 0.5)\n"
      "  --qbits N        quantization of the client's random raw values (default 6, as in the sibling baselines)\n"
      "  --no-bias        the last feature is a regular client feature, not the bias\n"
      "  --zero-init      client's initial weights are 0 (default: uniform random)\n"
      "  --seed N         seed of the client's random data / initial weights (default 1)\n\n"
      "protocol / network:\n"
      "  --semi-honest    Falcon's semi-honest protocol: no checked openings, no triple checks, no final check\n"
      "  --hosts H0,H1,H2,HC   IPv4 addresses of S0, S1, S2 and the client (default all 127.0.0.1)\n"
      "  --port N         base TCP port; the mesh uses N .. N+5 (default 12345)\n"
      "  --run-id N       nonce that all four parties must share (default 0); stale connections are dropped\n"
      "  --connect-timeout S   seconds to establish the mesh (default 60)\n"
      "  --timeout S      seconds a single receive / send may wait (default 300; the client's idle wait\n"
      "                   while the servers train has no limit, the servers time each other out)\n"
      "  --threads N      accepted and recorded; the port is single-threaded (default 1)\n\n"
      "output (client):\n"
      "  --verify         check the result against the bit-exact plaintext reference (the client owns the data,\n"
      "                   so this needs no test-only leak); exit code 2 on a mismatch\n"
      "  --json FILE      write all metrics of all four parties as JSON\n"
      "  --csv FILE       append one summary row (header written if the file is new)\n"
      "  --quiet          no progress lines\n"
      "  --tamper KIND --cheater I    TEST ONLY: server I deviates once (pass the same to all parties).\n"
      "                   KIND: open-value open-hash rec3 tree tree-noncanonical mask dot check output\n\n"
      "exit codes: 0 ok, 1 error, 2 verify mismatch, 3 this party detected a deviation, 4 a peer aborted / closed\n"
      "            / timed out\n",
      prog, prog);
}

Options parse_options(int argc, char** argv) {
  Options O;
  if (argc < 2) throw std::invalid_argument("missing role");
  const std::string role = argv[1];
  if (role == "--help" || role == "-h") { usage(argv[0]); std::exit(0); }
  if (role == "client") O.me = CL;
  else if (role == "server") O.me = -1;
  else throw std::invalid_argument("unknown role '" + role + "' (use client or server --id I)");
  ArgReader r{argc, argv, 2};
  for (; r.has(); ++r.i) {
    const std::string a = argv[r.i];
    if (parse_param_flag(r, O.P)) continue;
    if (a == "--id") {
      const int id = (int)parse_int(r.next_value(a), a);
      if (role != "server" || id < 0 || id > 2) throw std::invalid_argument("--id must be 0, 1 or 2 (servers only)");
      O.me = id;
    } else if (a == "--hosts") {
      std::stringstream ss(r.next_value(a));
      std::string h;
      int k = 0;
      while (std::getline(ss, h, ',')) {
        if (k >= NP) throw std::invalid_argument("--hosts takes 4 addresses: S0,S1,S2,client");
        O.net.hosts[(size_t)k++] = h;
      }
      if (k != NP) throw std::invalid_argument("--hosts takes 4 addresses: S0,S1,S2,client");
    } else if (a == "--port") O.net.port = (int)parse_int(r.next_value(a), a);
    else if (a == "--run-id") O.net.run_id = (uint64_t)parse_int(r.next_value(a), a);
    else if (a == "--connect-timeout") O.net.connect_timeout_s = (int)parse_int(r.next_value(a), a);
    else if (a == "--timeout") O.net.timeout_s = (int)parse_int(r.next_value(a), a);
    else if (a == "--threads") O.threads = (int)parse_int(r.next_value(a), a);
    else if (a == "--semi-honest") O.semi_honest = true;
    else if (a == "--verify") O.verify = true;
    else if (a == "--quiet") O.quiet = true;
    else if (a == "--json") O.json_path = r.next_value(a);
    else if (a == "--csv") O.csv_path = r.next_value(a);
    else if (a == "--tamper") O.tamper = Tamper::parse(r.next_value(a));
    else if (a == "--cheater") {
      O.cheater = (int)parse_int(r.next_value(a), a);
      if (O.cheater < 0 || O.cheater > 2) throw std::invalid_argument("--cheater must be 0, 1 or 2");
    } else if (a == "--help" || a == "-h") { usage(argv[0]); std::exit(0); }
    else throw std::invalid_argument("unknown option " + a);
  }
  if (O.me < 0) throw std::invalid_argument("server needs --id 0|1|2");
  if (O.net.port < 1024 || O.net.port + 5 > 65535) throw std::invalid_argument("--port must be in [1024, 65530]");
  if (O.net.connect_timeout_s < 1 || O.net.timeout_s < 1) throw std::invalid_argument("timeouts must be >= 1 s");
  if (O.threads < 1) O.threads = 1;
  if ((O.tamper != Tamper::NONE) != (O.cheater >= 0)) throw std::invalid_argument("--tamper and --cheater go together");
  O.net.me = O.me;
  O.P.finalize();
  return O;
}

std::string ms(double s) { char b[32]; std::snprintf(b, sizeof b, "%.1f", s * 1e3); return b; }
double sv(const fal::u64 w, int F) { return std::ldexp((double)(int64_t)w, -F); }

const std::vector<std::string>& all_phases() {
  static const std::vector<std::string> n = {
      "net/connect",         "setup/handshake",     "setup/seeds",          "setup/keys",           "offline/deal",
      "online/input",        "online/wait",         "online/fwd_matmul",    "online/drelu_open",    "online/drelu_pc_tree",
      "online/drelu_pc_check", "online/drelu_pc_open", "online/select",     "online/bwd_matmul",    "online/final_check",
      "online/output"};
  return n;
}

void print_tables(const ClientOutcome& O, const Totals& T) {
  std::printf("\n%-22s | %9s %9s | %9s %9s | %9s %9s | %9s %9s | %11s %11s %11s | %5s\n", "phase (times in ms)", "cli wall", "cli cpu",
              "S0 wall", "S0 cpu", "S1 wall", "S1 cpu", "S2 wall", "S2 cpu", "c->s", "s->c", "s<->s", "steps");
  std::printf("%.*s\n", 160, "----------------------------------------------------------------------------------------------------------------------------------------------------------------");
  for (const auto& nm : all_phases()) {
    const PhaseRow* r[NP];
    bool any = false;
    for (int p = 0; p < NP; ++p) { r[p] = find_row(O.rows[(size_t)p], nm.c_str()); any = any || r[p]; }
    if (!any) continue;
    auto w = [&](int p) { return r[p] ? ms(r[p]->wall_s) : std::string("-"); };
    auto c = [&](int p) { return r[p] ? ms(r[p]->cpu_s) : std::string("-"); };
    uint64_t c2s = 0, s2c = 0, s2s = 0;
    if (r[CL]) c2s = r[CL]->sent[0] + r[CL]->sent[1] + r[CL]->sent[2];
    for (int i = 0; i < 3; ++i)
      if (r[i]) { s2c += r[i]->sent[CL]; s2s += r[i]->sent[0] + r[i]->sent[1] + r[i]->sent[2]; }
    const uint64_t steps = r[S0] ? r[S0]->steps : 0;
    std::printf("%-22s | %9s %9s | %9s %9s | %9s %9s | %9s %9s | %11s %11s %11s | %5" PRIu64 "\n", nm.c_str(), w(CL).c_str(), c(CL).c_str(),
                w(S0).c_str(), c(S0).c_str(), w(S1).c_str(), c(S1).c_str(), w(S2).c_str(), c(S2).c_str(), human_bytes(c2s).c_str(),
                human_bytes(s2c).c_str(), human_bytes(s2s).c_str(), steps);
  }
  std::printf("steps = communication rounds at S0 (every server runs the same schedule); bytes include 8-byte frame headers\n");
  const char* stages[] = {"setup", "offline", "online"};
  std::printf("\n%-8s | %9s %9s | %11s %11s | %11s %11s %11s | %5s\n", "stage", "cli wall", "cli cpu", "srv wall*", "srv cpu*", "c->s", "s->c",
              "s<->s", "steps");
  for (int s = 0; s < 3; ++s) {
    double sw = 0, sc = 0;
    uint64_t s2c = 0, s2s = 0;
    for (int i = 0; i < 3; ++i) {
      sw = std::max(sw, T.st[i][s].wall_s);
      sc = std::max(sc, T.st[i][s].cpu_s);
      s2c += T.st[i][s].sent[CL];
      s2s += T.st[i][s].sent[0] + T.st[i][s].sent[1] + T.st[i][s].sent[2];
    }
    std::printf("%-8s | %9s %9s | %11s %11s | %11s %11s %11s | %5" PRIu64 "\n", stages[s], ms(T.st[CL][s].wall_s).c_str(), ms(T.st[CL][s].cpu_s).c_str(),
                ms(sw).c_str(), ms(sc).c_str(), human_bytes(T.st[CL][s].sent_total()).c_str(), human_bytes(s2c).c_str(), human_bytes(s2s).c_str(),
                T.st[S0][s].steps);
  }
  std::printf("* max over S0, S1, S2\n");
}

}  // namespace

int main(int argc, char** argv) {
  Options O;
  try {
    O = parse_options(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n\n", e.what());
    usage(argv[0]);
    return 1;
  }
  const bool mal = !O.semi_honest;
  const Params& P = O.P;
  const std::string who = O.me == CL ? "client" : "S" + std::to_string(O.me);
  if (!O.quiet)
    std::fprintf(stderr, "[%s] %s | %s\n", who.c_str(), mal ? "MALICIOUS security (checked openings + triple checks)" : "semi-honest",
                 params_string(P).c_str());
  RunCfg cfg;
  cfg.P = P;
  cfg.mal = mal;
  cfg.quiet = O.quiet;
  Tamper tamper;
  tamper.kind = O.tamper;
  tamper.who = O.cheater;

  Dataset data;
  if (O.me == CL) data = generate_client_data(P);
  Net net(O.net);
  Recorder rec(net);
  auto fail = [&](const std::string& msg) {
    if (O.me == CL) for (int i = 0; i < 3; ++i) net.notify_abort(i, msg);
    else net.notify_abort(CL, msg);
    net.abort_all();
  };
  try {
    {
      Scope sc(rec, "net/connect");
      net.connect();
    }
    if (O.me != CL) {
      server_run(net, rec, cfg, tamper);
      net.abort_all();
      if (!O.quiet) std::fprintf(stderr, "[%s] done (client accepted the output)\n", who.c_str());
      return 0;
    }

    // ---- client ----------------------------------------------------------------------------------------
    const ClientOutcome R = client_run(net, rec, cfg, data);
    net.abort_all();
    const Totals T = analyze(R, P, mal);
    int rc = 0;
    std::printf("\nFalcon logistic regression, %s\n%s\n", mal ? "MALICIOUS security" : "semi-honest", params_string(P).c_str());
    std::printf("trained weights (value = int / 2^%d):", P.frac);
    for (int j = 0; j < P.d && j < 8; ++j) std::printf(" %.5f", sv(R.w[(size_t)j], P.frac));
    if (P.d > 8) std::printf(" ... (%d weights)", P.d);
    std::printf("\n");
    if (O.verify) {
      const std::vector<u64> ref = reference_train(P, data, R.cv.rp_fwd, R.cv.rp_bwd);
      const std::vector<u64> nom = reference_train_nominal(P, data);
      const std::vector<double> dbl = reference_train_double(P, data);
      uint64_t ulp = 0;
      double maxd = 0;
      for (int j = 0; j < P.d; ++j) {
        const int64_t df = (int64_t)(R.w[(size_t)j] - nom[(size_t)j]);
        ulp = std::max<uint64_t>(ulp, (uint64_t)(df < 0 ? -df : df));
        maxd = std::max(maxd, std::fabs(sv(R.w[(size_t)j], P.frac) - dbl[(size_t)j]));
      }
      const bool exact = ref == R.w;
      std::printf("verify: %s the plaintext reference (replayed truncation masks); max |w - w_nominal| = %" PRIu64 " ulp; max |w - w_double| = %.3g\n",
                  exact ? "EXACT MATCH with" : "MISMATCH against", ulp, maxd);
      if (!exact) rc = 2;
    }
    if (!T.error.empty()) {
      std::fprintf(stderr, "[client] error: report invariant violated: %s\n", T.error.c_str());
      rc = rc ? rc : 1;
    }
    print_tables(R, T);

    double so_w = 0, so_c = 0, sn_w = 0, sn_c = 0, so_csum = 0, sn_csum = 0;
    long srss = 0;
    for (int i = 0; i < 3; ++i) {
      so_w = std::max(so_w, T.st[i][0].wall_s + T.st[i][1].wall_s);
      so_c = std::max(so_c, T.st[i][0].cpu_s + T.st[i][1].cpu_s);
      sn_w = std::max(sn_w, T.st[i][2].wall_s);
      sn_c = std::max(sn_c, T.st[i][2].cpu_s);
      so_csum += T.st[i][0].cpu_s + T.st[i][1].cpu_s;
      sn_csum += T.st[i][2].cpu_s;
      srss = std::max(srss, R.rss_kib[(size_t)i]);
    }
    const PhaseRow* deal = find_row(R.rows[CL], "offline/deal");
    const PhaseRow* inp = find_row(R.rows[CL], "online/input");
    const uint64_t deal_b = deal ? deal->sent[0] + deal->sent[1] + deal->sent[2] : 0;
    const uint64_t inp_b = inp ? inp->sent[0] + inp->sent[1] + inp->sent[2] : 0;
    const uint64_t per_iter_steps = mal ? 15 : 14;
    std::printf("\nSUMMARY (%s, d=%d, iters=%d, batch=%d)\n", mal ? "malicious" : "semi-honest", P.d, P.iters, P.batch);
    std::printf("  computation  client: offline %s ms wall / %s ms cpu (dealing), online %s ms wall / %s ms cpu (input + output; idle wait %s ms excluded)\n",
                ms(T.st[CL][0].wall_s + T.st[CL][1].wall_s).c_str(), ms(T.st[CL][0].cpu_s + T.st[CL][1].cpu_s).c_str(),
                ms(T.client_online_compute.wall_s).c_str(), ms(T.client_online_compute.cpu_s).c_str(), ms(T.client_wait_s).c_str());
    std::printf("               servers: offline %s ms wall / %s ms cpu, online %s ms wall / %s ms cpu (max over S0..S2)\n", ms(so_w).c_str(),
                ms(so_c).c_str(), ms(sn_w).c_str(), ms(sn_c).c_str());
    std::printf("               per server (online wall / cpu ms): S0 %s / %s, S1 %s / %s, S2 %s / %s; peak RSS client %.0f MiB, servers %.0f MiB\n",
                ms(T.st[0][2].wall_s).c_str(), ms(T.st[0][2].cpu_s).c_str(), ms(T.st[1][2].wall_s).c_str(), ms(T.st[1][2].cpu_s).c_str(),
                ms(T.st[2][2].wall_s).c_str(), ms(T.st[2][2].cpu_s).c_str(), R.rss_kib[CL] / 1024.0, srss / 1024.0);
    std::printf("  communication total %s; client->servers %s (offline %s: seeds + deal %s; online input %s), servers->client %s,\n",
                human_bytes(T.total).c_str(), human_bytes(T.c2s_off + T.c2s_on).c_str(), human_bytes(T.c2s_off).c_str(), human_bytes(deal_b).c_str(),
                human_bytes(inp_b).c_str(), human_bytes(T.s2c_off + T.s2c_on).c_str());
    std::printf("               server<->server %s (setup %s, online %s)\n", human_bytes(T.s2s_off + T.s2s_on).c_str(), human_bytes(T.s2s_off).c_str(),
                human_bytes(T.s2s_on).c_str());
    std::printf("  rounds       online %" PRIu64 " steps at the servers (%" PRIu64 " per iteration + %d); S0/S1/S2 = %" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
                T.online_steps[0], per_iter_steps, mal ? 4 : 3, T.online_steps[0], T.online_steps[1], T.online_steps[2]);
    std::printf("  per iteration (online)  %.2f ms wall at the servers (max); %s server<->server per iteration\n", sn_w * 1e3 / P.iters,
                human_bytes(T.s2s_on / (uint64_t)P.iters).c_str());
    std::printf("  check        comm total = sum of sent bytes = sum of received bytes over all four parties: %s\n", T.error.empty() ? "ok" : "VIOLATED");

    if (!O.json_path.empty()) {
      std::ofstream f(O.json_path);
      f << "{\n  \"protocol\": \"falcon\",\n  \"security\": \"" << (mal ? "malicious" : "semi-honest") << "\",\n  \"d\": " << P.d << ", \"iters\": " << P.iters
        << ", \"batch\": " << P.batch << ", \"frac\": " << P.frac << ", \"threads\": " << O.threads << ",\n  \"comm_total_bytes\": " << T.total
        << ", \"online_steps\": " << T.online_steps[0] << ",\n  \"phases\": [\n";
      const char* pn[] = {"s0", "s1", "s2", "client"};
      bool first = true;
      for (int p = 0; p < NP; ++p)
        for (const auto& r : R.rows[(size_t)p]) {
          if (!first) f << ",\n";
          first = false;
          f << "    {\"party\": \"" << party_name(p) << "\", \"phase\": \"" << r.name << "\", \"wall_s\": " << r.wall_s << ", \"cpu_s\": " << r.cpu_s;
          for (int q = 0; q < NP; ++q) f << ", \"sent_to_" << pn[q] << "\": " << r.sent[q];
          for (int q = 0; q < NP; ++q) f << ", \"recv_from_" << pn[q] << "\": " << r.recv[q];
          f << ", \"steps\": " << r.steps << ", \"calls\": " << r.calls << "}";
        }
      f << "\n  ]\n}\n";
    }
    if (!O.csv_path.empty()) {
      const std::string header =
          "protocol,security,d,iters,batch,frac,threads,verified,client_offline_wall_ms,client_offline_cpu_ms,"
          "client_online_wall_ms,client_online_cpu_ms,client_idle_wait_ms,server_offline_wall_ms,server_offline_cpu_ms,"
          "server_online_wall_ms,server_online_cpu_ms,server_offline_cpu_ms_sum,server_online_cpu_ms_sum,comm_offline_c2s_bytes,"
          "comm_offline_s2c_bytes,comm_offline_s2s_bytes,comm_online_c2s_bytes,comm_online_s2c_bytes,comm_online_s2s_bytes,"
          "comm_total_bytes,online_steps,client_peak_rss_mib,server_peak_rss_mib";
      bool exists = false;
      {
        std::ifstream in(O.csv_path);
        std::string first;
        if (in && std::getline(in, first)) {
          exists = true;
          if (first != header) { std::fprintf(stderr, "error: %s has a different header\n", O.csv_path.c_str()); return 1; }
        }
      }
      std::ofstream f(O.csv_path, std::ios::app);
      if (!exists) f << header << "\n";
      char row[2048];
      std::snprintf(row, sizeof row,
                    "falcon,%s,%d,%d,%d,%d,%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                    ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.1f,%.1f",
                    mal ? "malicious" : "semi-honest", P.d, P.iters, P.batch, P.frac, O.threads, O.verify ? (rc == 0 ? "exact" : "FAIL") : "no",
                    (T.st[CL][0].wall_s + T.st[CL][1].wall_s) * 1e3, (T.st[CL][0].cpu_s + T.st[CL][1].cpu_s) * 1e3, T.client_online_compute.wall_s * 1e3,
                    T.client_online_compute.cpu_s * 1e3, T.client_wait_s * 1e3, so_w * 1e3, so_c * 1e3, sn_w * 1e3, sn_c * 1e3, so_csum * 1e3,
                    sn_csum * 1e3, T.c2s_off, T.s2c_off, T.s2s_off, T.c2s_on, T.s2c_on, T.s2s_on, T.total, T.online_steps[0],
                    R.rss_kib[CL] / 1024.0, srss / 1024.0);
      f << row << "\n";
    }
    return rc;
  } catch (const AbortError& e) {
    std::fprintf(stderr, "[%s] ABORT: %s\n", who.c_str(), e.what());
    fail(e.what());
    return 3;
  } catch (const PeerError& e) {
    std::fprintf(stderr, "[%s] ABORT: %s\n", who.c_str(), e.what());
    fail(e.what());
    return 4;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[%s] error: %s\n", who.c_str(), e.what());
    fail(e.what());
    return 1;
  }
}
