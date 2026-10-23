// lr_secureml -- SecureML (Mohassel & Zhang, S&P 2017) logistic-regression training between a client and
// a server, with malicious security.
//
//   client (party 0): owns every private input -- the training data (x, y) and the initial weights w0 --
//                     and learns nothing
//   server (party 1): has no input; ends with the trained weights (only it learns them)
//
// The two parties act as the two non-colluding SecureML servers: the data is additively shared with
// information-theoretic MACs (SPDZ style), the multiplications use Beaver triples in matrix form, and
// the activation is SecureML's piecewise function. See README.md for the protocol, the malicious-security
// layer (OT-based authenticated preprocessing + sacrifice + MAC checks), the cost report and the
// differences from the paper.
//
// usage: lr_secureml <client|server> [options]          (--help)
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>

#include "metrics.h"
#include "mpc.h"
#include "online.h"
#include "params.h"
#include "preproc.h"

using namespace sml;

namespace {

struct Options {
  int me = 0;
  Params P;
  std::string host = "127.0.0.1";
  int port = 12345;
  int connect_port = 0;  // client: first port to connect to (0 = port); a WAN proxy listens there
  int threads = 0;  // 0 = all hardware threads
  bool semi_honest = false;
  bool verify = false, quiet = false;
  std::string json_path, csv_path;
  Tamper::Kind tamper = Tamper::NONE;
  int cheater = -1;
};

void usage(const char* prog) {
  std::printf(
      "usage: %s <client|server> [options]\n"
      "  client = party 0 (owns the data and the initial weights), server = party 1 (learns the weights)\n\n"
      "training (both parties must pass the same values; --seed / --zero-init only shape the client's\n"
      "private inputs and are used by the server only for --verify):\n"
      "  --d N            features incl. the bias column (default 5)\n"
      "  --iters N        SGD iterations (default 20)\n"
      "  --batch N        datapoints per iteration (default 4)\n"
      "  --frac N         fractional bits F of features / labels / sigmoid output (default 13)\n"
      "  --lr X           learning rate; batch/lr must be a power of two (default 0.5)\n"
      "  --qbits N        quantization of the client's random raw values (default 6, as in the sibling baselines)\n"
      "  --no-bias        the last feature is a regular client feature, not the bias\n"
      "  --zero-init      client's initial weights are 0 (default: uniform random)\n"
      "  --seed N         seed of the client's random data / initial weights (default 1)\n\n"
      "protocol / network:\n"
      "  --host IP        server address the client connects to (default 127.0.0.1)\n"
      "  --port N         first TCP port; the protocol uses port .. port+threads (default 12345)\n"
      "  --connect-port N client only: first port to connect to (default = --port); scripts/run_local.sh points it at wan/wanproxy\n"
      "  --threads N      worker threads AND OT connections per party (default: all hardware threads)\n"
      "  --semi-honest    the paper's original security level: no MACs, no sacrifice, semi-honest OT\n"
      "  --ssp N          statistical security parameter of the masks (default 40)\n\n"
      "output:\n"
      "  --verify         TEST ONLY: the server regenerates the client's data from --seed and checks the\n"
      "                   result against the plaintext integer reference (must be exact)\n"
      "  --json FILE      write all metrics as JSON (server)\n"
      "  --csv FILE       append one summary row (server; header written if the file is new)\n"
      "  --quiet          no progress lines\n"
      "  --tamper KIND --cheater client|server    TEST ONLY: one party deviates once; the run must abort.\n"
      "                   KIND: open-pre open-online triple matrix ot-input mac-value mac-sigma\n",
      prog);
}

Options parse_options(int argc, char** argv) {
  Options O;
  if (argc < 2) throw std::invalid_argument("missing role");
  const std::string role = argv[1];
  if (role == "client" || role == "0") O.me = 0;
  else if (role == "server" || role == "1") O.me = 1;
  else throw std::invalid_argument("unknown role '" + role + "' (use client or server)");
  ArgReader r{argc, argv, 2};
  for (; r.has(); ++r.i) {
    const std::string a = argv[r.i];
    if (parse_param_flag(r, O.P)) continue;
    if (a == "--host") O.host = r.next_value(a);
    else if (a == "--port") O.port = (int)parse_int(r.next_value(a), a);
    else if (a == "--connect-port") O.connect_port = (int)parse_int(r.next_value(a), a);
    else if (a == "--threads") O.threads = (int)parse_int(r.next_value(a), a);
    else if (a == "--semi-honest") O.semi_honest = true;
    else if (a == "--verify") O.verify = true;
    else if (a == "--quiet") O.quiet = true;
    else if (a == "--json") O.json_path = r.next_value(a);
    else if (a == "--csv") O.csv_path = r.next_value(a);
    else if (a == "--tamper") {
      const std::string k = r.next_value(a);
      if (k == "open-pre") O.tamper = Tamper::OPEN_PRE;
      else if (k == "open-online") O.tamper = Tamper::OPEN_ONLINE;
      else if (k == "triple") O.tamper = Tamper::TRIPLE_C;
      else if (k == "matrix") O.tamper = Tamper::MATRIX_Z;
      else if (k == "ot-input") O.tamper = Tamper::OT_INPUT;
      else if (k == "mac-value") O.tamper = Tamper::MAC_VALUE;
      else if (k == "mac-sigma") O.tamper = Tamper::MAC_SIGMA;
      else throw std::invalid_argument("unknown --tamper kind " + k);
    } else if (a == "--cheater") {
      const std::string c = r.next_value(a);
      O.cheater = c == "client" ? 0 : (c == "server" ? 1 : -2);
      if (O.cheater < 0) throw std::invalid_argument("--cheater must be client or server");
    } else if (a == "--help" || a == "-h") { usage(argv[0]); std::exit(0); }
    else throw std::invalid_argument("unknown option " + a);
  }
  if (O.threads <= 0) O.threads = (int)std::max(2u, std::thread::hardware_concurrency());
  if (O.port < 1024 || O.port + O.threads + 1 > 65535) throw std::invalid_argument("--port out of range");
  O.P.finalize();
  return O;
}

// Public parameters both parties must agree on.
struct Handshake {
  int32_t d, iters, batch, frac, qbits, ssp, threads, mal;
  double lr;
  int32_t bias;
};

std::string ms(double s) { char b[32]; std::snprintf(b, sizeof b, "%.1f", s * 1e3); return b; }

struct Report {
  std::vector<PhaseRow> rows[2];  // [0] client, [1] server
  long rss_kib[2] = {0, 0};
};

void print_phase_table(const Report& R) {
  const auto& C = R.rows[0];
  const auto& S = R.rows[1];
  std::printf("\n%-28s | %10s %10s | %10s %10s | %12s %12s | %6s %6s\n", "phase (times in ms)", "client wall", "client cpu",
              "server wall", "server cpu", "c->s", "s->c", "rt", "ot-rt");
  std::printf("%.*s\n", 126, "------------------------------------------------------------------------------------------------------------------------------");
  for (size_t i = 0; i < C.size() && i < S.size(); ++i) {
    const auto& c = C[i];
    const auto& s = S[i];
    const uint64_t c2s = c.main_sent + c.ot_sent, s2c = s.main_sent + s.ot_sent;
    std::printf("%-28s | %10s %10s | %10s %10s | %12s %12s | %6" PRIu64 " %6" PRIu64 "\n", c.name, ms(c.wall_s).c_str(),
                ms(c.cpu_s).c_str(), ms(s.wall_s).c_str(), ms(s.cpu_s).c_str(), human_bytes(c2s).c_str(),
                human_bytes(s2c).c_str(), c.main_flights / 2, c.ot_flights_max / 2);
  }
}

// Network wait: wall time each party's protocol thread spent blocked in socket receives (wan/libwanshim.so).
void print_wait(const Report& R, const StageTotals (&T)[3][2]) {
  if (!wan::shim_loaded()) return;
  const char* stages[] = {"setup", "offline", "online"};
  std::printf("\nNETWORK WAIT (ms): wall = compute + wait; wait = time the protocol thread is blocked in receives\n");
  std::printf("%-8s | %10s %10s %6s | %10s %10s %6s\n", "stage", "cli wall", "cli wait", "wait%", "srv wall", "srv wait", "wait%");
  double tw[2] = {0, 0}, tt[2] = {0, 0};
  for (int s = 0; s < 3; ++s) {
    std::printf("%-8s |", stages[s]);
    for (int p = 0; p < 2; ++p) {
      const double w = T[s][p].wall_s, t = T[s][p].wait_s;
      tw[p] += w; tt[p] += t;
      std::printf(" %10s %10s %5.1f%% %s", ms(w).c_str(), ms(t).c_str(), w > 0 ? 100.0 * t / w : 0.0, p == 0 ? "|" : "\n");
    }
  }
  std::printf("%-8s | %10s %10s %5.1f%% | %10s %10s %5.1f%%\n", "total", ms(tw[0]).c_str(), ms(tt[0]).c_str(), tw[0] > 0 ? 100.0 * tt[0] / tw[0] : 0.0,
              ms(tw[1]).c_str(), ms(tt[1]).c_str(), tw[1] > 0 ? 100.0 * tt[1] / tw[1] : 0.0);
  std::printf("\n%-28s | %10s %10s | %10s %10s\n", "phase (wait in ms)", "cli wall", "cli wait", "srv wall", "srv wait");
  for (size_t i = 0; i < R.rows[0].size() && i < R.rows[1].size(); ++i)
    std::printf("%-28s | %10s %10s | %10s %10s\n", R.rows[0][i].name, ms(R.rows[0][i].wall_s).c_str(), ms(R.rows[0][i].wait_s).c_str(),
                ms(R.rows[1][i].wall_s).c_str(), ms(R.rows[1][i].wait_s).c_str());
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
  const char* who = O.me == 0 ? "client" : "server";
  if (!O.quiet) std::fprintf(stderr, "[%s] %s | %s | threads=%d\n", who, mal ? "MALICIOUS security (MACs + sacrifice)" : "semi-honest", params_string(P).c_str(), O.threads);

  try {
    Link lk(O.me, O.host, O.port, O.threads, mal, O.connect_port);
    Mpc M(lk, mal);
    if (O.tamper != Tamper::NONE) { M.tamper.kind = O.tamper; M.tamper.who = O.cheater; }
    Recorder rec(lk);

    // ---- setup: agree on parameters, bring up the OT-extension sessions of every worker --------------
    {
      Scope sc(rec, "setup/handshake");
      // zero the whole struct first: the tail padding is otherwise indeterminate and would break the memcmp below
      Handshake h, g;
      std::memset(&h, 0, sizeof h);
      std::memset(&g, 0, sizeof g);
      h.d = P.d; h.iters = P.iters; h.batch = P.batch; h.frac = P.frac; h.qbits = P.qbits; h.ssp = P.ssp;
      h.threads = lk.nw(); h.mal = mal ? 1 : 0; h.lr = P.lr; h.bias = P.bias ? 1 : 0;
      M.swap(&h, &g, sizeof h);
      if (std::memcmp(&h, &g, sizeof h) != 0) throw AbortError("the parties were started with different parameters (d, iters, batch, frac, qbits, ssp, lr, bias, threads, mode)");
    }
    {
      Scope sc(rec, "setup/ot_base");  // base OTs + OT-extension bootstrap of every worker, both directions
      std::vector<Fp> A((size_t)lk.nw() * 2, Fp(1)), S((size_t)lk.nw() * 2), b((size_t)lk.nw(), Fp(1)), R((size_t)lk.nw() * 2);
      if (O.me == 0) { product_send(lk, lk.nw(), 2, A.data(), S.data()); product_recv(lk, lk.nw(), 2, b.data(), R.data()); }
      else { product_recv(lk, lk.nw(), 2, b.data(), R.data()); product_send(lk, lk.nw(), 2, A.data(), S.data()); }
    }

    // ---- offline ---------------------------------------------------------------------------------------
    Pre pre;
    generate_preprocessing(M, P, pre, rec);
    if (!O.quiet) std::fprintf(stderr, "[%s] offline phase done: %" PRId64 " triples, %" PRId64 " random bits\n", who, pre.NT, pre.NB);

    // ---- online ----------------------------------------------------------------------------------------
    Dataset D;
    if (O.me == 0) D = generate_client_data(P);
    Online on(M, P, pre, rec);
    on.input(O.me == 0 ? &D : nullptr);
    on.train(O.quiet);
    const std::vector<i128> w = on.output();

    // ---- exchange reports ---------------------------------------------------------------------------------
    Report R;
    R.rows[O.me] = rec.rows();
    R.rss_kib[O.me] = peak_rss_kib();
    {
      uint64_t cnt = rec.rows().size(), cnt_th = 0;
      M.swap(&cnt, &cnt_th, sizeof cnt);
      if (cnt != cnt_th) throw std::runtime_error("report mismatch");
      R.rows[1 - O.me].resize(cnt);
      M.swap(rec.rows().data(), R.rows[1 - O.me].data(), cnt * sizeof(PhaseRow));
      long r_th = 0;
      M.swap(&R.rss_kib[O.me], &r_th, sizeof r_th);
      R.rss_kib[1 - O.me] = r_th;
    }

    // ---- stage totals ---------------------------------------------------------------------------------------
    const char* stages[] = {"setup", "offline", "online"};
    StageTotals T[3][2];
    for (int s = 0; s < 3; ++s)
      for (int p = 0; p < 2; ++p) T[s][p] = stage_totals(R.rows[p], stages[s]);

    // ---- result and report (server) -----------------------------------------------------------------------------
    int rc = 0;
    if (O.me == 1) {
      std::printf("\nSecureML logistic regression, %s\n%s\n", mal ? "MALICIOUS security" : "semi-honest", params_string(P).c_str());
      std::printf("trained weights (value = int / 2^%d):", P.sw);
      for (int j = 0; j < P.d && j < 8; ++j) std::printf(" %.5f", std::ldexp((double)w[(size_t)j], -P.sw));
      if (P.d > 8) std::printf(" ... (%d weights)", P.d);
      std::printf("\n");
      if (O.verify) {
        const Dataset V = generate_client_data(P);
        bool in_range = true;
        const std::vector<i128> ref = reference_train(P, V, &in_range);
        bool exact = true;
        for (int j = 0; j < P.d; ++j) if (ref[(size_t)j] != w[(size_t)j]) exact = false;
        const std::vector<double> dref = reference_train_double(P, V);
        double maxd = 0;
        for (int j = 0; j < P.d; ++j) maxd = std::max(maxd, std::fabs(std::ldexp((double)w[(size_t)j], -P.sw) - dref[(size_t)j]));
        std::printf("verify: %s the plaintext integer reference%s; max |w - w_double| = %.3g\n",
                    exact ? "EXACT MATCH with" : "MISMATCH against", in_range ? "" : " (gadget range exceeded!)", maxd);
        if (!exact || !in_range) rc = 2;
      }
      print_phase_table(R);

      std::printf("\n%-8s | %10s %10s %10s %10s | %14s %14s | %8s %8s | %12s\n", "stage", "cli wall", "cli cpu", "srv wall", "srv cpu",
                  "c->s", "s->c", "flights", "ot-fl", "OTs");
      for (int s = 0; s < 3; ++s)
        std::printf("%-8s | %10s %10s %10s %10s | %14s %14s | %8" PRIu64 " %8" PRIu64 " | %12" PRIu64 "\n", stages[s], ms(T[s][0].wall_s).c_str(),
                    ms(T[s][0].cpu_s).c_str(), ms(T[s][1].wall_s).c_str(), ms(T[s][1].cpu_s).c_str(), human_bytes(T[s][0].sent).c_str(),
                    human_bytes(T[s][1].sent).c_str(), T[s][0].main_flights, T[s][0].ot_flights_max, T[s][0].ots);
      print_wait(R, T);
      std::printf("flights = main-channel direction runs (a symmetric exchange = 2); ot-fl = direction runs on the busiest OT channel\n");

      double wall[2] = {0, 0}, cpu[2] = {0, 0};
      uint64_t sent[2] = {0, 0};
      for (int p = 0; p < 2; ++p) for (int s = 0; s < 3; ++s) { wall[p] += T[s][p].wall_s; cpu[p] += T[s][p].cpu_s; sent[p] += T[s][p].sent; }
      const uint64_t on_fl = T[2][0].main_flights, off_fl = T[1][0].main_flights + T[0][0].main_flights;
      const uint64_t off_ot = T[1][0].ot_flights_max + T[0][0].ot_flights_max;
      std::printf("\nSUMMARY (%s, d=%d, iters=%d, batch=%d, %d threads per party)\n", mal ? "malicious" : "semi-honest", P.d, P.iters, P.batch, lk.nw());
      std::printf("  computation  client: offline %s ms wall / %s ms cpu, online %s ms wall / %s ms cpu\n",
                  ms(T[0][0].wall_s + T[1][0].wall_s).c_str(), ms(T[0][0].cpu_s + T[1][0].cpu_s).c_str(), ms(T[2][0].wall_s).c_str(), ms(T[2][0].cpu_s).c_str());
      std::printf("               server: offline %s ms wall / %s ms cpu, online %s ms wall / %s ms cpu\n",
                  ms(T[0][1].wall_s + T[1][1].wall_s).c_str(), ms(T[0][1].cpu_s + T[1][1].cpu_s).c_str(), ms(T[2][1].wall_s).c_str(), ms(T[2][1].cpu_s).c_str());
      std::printf("               cores kept busy (cpu/wall): client %.1f, server %.1f; peak RSS %.0f / %.0f MiB\n", cpu[0] / std::max(1e-9, wall[0]),
                  cpu[1] / std::max(1e-9, wall[1]), R.rss_kib[0] / 1024.0, R.rss_kib[1] / 1024.0);
      std::printf("  communication offline %s (client->server %s, server->client %s); online %s (client->server %s, server->client %s)\n",
                  human_bytes(T[0][0].sent + T[1][0].sent + T[0][1].sent + T[1][1].sent).c_str(), human_bytes(T[0][0].sent + T[1][0].sent).c_str(),
                  human_bytes(T[0][1].sent + T[1][1].sent).c_str(), human_bytes(T[2][0].sent + T[2][1].sent).c_str(), human_bytes(T[2][0].sent).c_str(),
                  human_bytes(T[2][1].sent).c_str());
      std::printf("  rounds       online %" PRIu64 " flights = %" PRIu64 " round trips (%.1f per iteration); offline %" PRIu64 " main + %" PRIu64 " OT-channel flights\n",
                  on_fl, (on_fl + 1) / 2, (double)((on_fl + 1) / 2) / P.iters, off_fl, off_ot);
      std::printf("  per iteration (online)  %.2f ms wall at the client, %.2f ms at the server; %s sent per iteration (both directions)\n",
                  T[2][0].wall_s * 1e3 / P.iters, T[2][1].wall_s * 1e3 / P.iters, human_bytes((T[2][0].sent + T[2][1].sent) / P.iters).c_str());
      std::printf("  total        %s transferred, %" PRIu64 " random OTs\n",human_bytes(sent[0] + sent[1]).c_str(), T[0][0].ots + T[1][0].ots + T[2][0].ots);
    }

    // ---- JSON / CSV (server) ---------------------------------------------------------------------------------
    if (O.me == 1 && !O.json_path.empty()) {
      std::ofstream f(O.json_path);
      f << "{\n  \"protocol\": \"secureml\",\n  \"security\": \"" << (mal ? "malicious" : "semi-honest") << "\",\n  \"d\": " << P.d
        << ", \"iters\": " << P.iters << ", \"batch\": " << P.batch << ", \"frac\": " << P.frac << ", \"threads\": " << lk.nw()
        << ",\n  \"phases\": [\n";
      bool first = true;
      for (int p = 0; p < 2; ++p)
        for (const auto& r : R.rows[p]) {
          if (!first) f << ",\n";
          first = false;
          f << "    {\"party\": \"" << (p == 0 ? "client" : "server") << "\", \"phase\": \"" << r.name << "\", \"wall_s\": " << r.wall_s << ", \"wait_s\": " << r.wait_s
            << ", \"cpu_s\": " << r.cpu_s << ", \"main_sent\": " << r.main_sent << ", \"main_recv\": " << r.main_recv
            << ", \"ot_sent\": " << r.ot_sent << ", \"ot_recv\": " << r.ot_recv << ", \"main_flights\": " << r.main_flights
            << ", \"ot_flights_max\": " << r.ot_flights_max << ", \"ots\": " << r.ots << "}";
        }
      f << "\n  ]\n}\n";
    }
    if (O.me == 1 && !O.csv_path.empty()) {
      const std::string header =
          "protocol,security,d,iters,batch,frac,threads,verified,"
          "client_offline_wall_ms,client_offline_cpu_ms,client_online_wall_ms,client_online_cpu_ms,"
          "server_offline_wall_ms,server_offline_cpu_ms,server_online_wall_ms,server_online_cpu_ms,"
          "comm_offline_c2s_bytes,comm_offline_s2c_bytes,comm_online_c2s_bytes,comm_online_s2c_bytes,"
          "rounds_online_roundtrips,rounds_offline_main_roundtrips,rounds_offline_ot_roundtrips,ots";
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
      char row[1024];
      std::snprintf(row, sizeof row,
                    "secureml,%s,%d,%d,%d,%d,%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64,
                    mal ? "malicious" : "semi-honest", P.d, P.iters, P.batch, P.frac, lk.nw(), O.verify ? (rc == 0 ? "exact" : "FAIL") : "no",
                    T[1][0].wall_s * 1e3 + T[0][0].wall_s * 1e3, T[1][0].cpu_s * 1e3 + T[0][0].cpu_s * 1e3, T[2][0].wall_s * 1e3, T[2][0].cpu_s * 1e3,
                    T[1][1].wall_s * 1e3 + T[0][1].wall_s * 1e3, T[1][1].cpu_s * 1e3 + T[0][1].cpu_s * 1e3, T[2][1].wall_s * 1e3, T[2][1].cpu_s * 1e3,
                    T[0][0].sent + T[1][0].sent, T[0][1].sent + T[1][1].sent, T[2][0].sent, T[2][1].sent, (T[2][0].main_flights + 1) / 2,
                    (T[1][0].main_flights + T[0][0].main_flights + 1) / 2, T[1][0].ot_flights_max / 2, T[0][0].ots + T[1][0].ots + T[2][0].ots);
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
