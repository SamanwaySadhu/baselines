// lr_gc_agmpc -- logistic regression trained with MULTIPARTY authenticated garbling (WRK17 "Global-Scale Secure
// Multiparty Computation", emp-agmpc, hardened copy in third_party/emp-agmpc): B clients jointly garble, the server
// evaluates and learns only the trained weights. Malicious security against any number of corrupted parties but not
// all (dishonest majority), with abort.
//
//   lr_gc_agmpc server            [options]     emp-agmpc party 1 (evaluator)
//   lr_gc_agmpc client --id K     [options]     emp-agmpc party K+1, K = 1..batch (client K holds datapoint K of
//                                               every batch and one XOR share of the initial weights)
#include "emp-agmpc/cmpc_config.h"
#include "run_cfg.h"

#include <csignal>
#include <cstdio>
#include <sstream>
#include <string>

using namespace lrgc;

#ifndef AGMPC_NP_LIST
#define AGMPC_NP_LIST ""
#endif

static void usage() {
  std::printf(
      "usage: lr_gc_agmpc server [options]\n"
      "       lr_gc_agmpc client --id K [options]      (K = 1..batch)\n"
      "training (every party must pass the same values):\n"
      "  --d N          features including the bias column (default 5)\n"
      "  --iters N      SGD iterations (default 20)\n"
      "  --batch N      datapoints per iteration = number of clients (default 4)\n"
      "  --qbits N  --lr X  --no-bias  --zero-init  --seed N   (as in gc_ag2pc; --seed/--zero-init shape the clients' test data)\n"
      "protocol:\n"
      "  --hosts H1,...,Hn   IPv4 of party 1 (server), 2 (client 1), ... (default all 127.0.0.1)\n"
      "  --port BASE    first TCP port; the run uses about 4 (n^2 + n) ports from BASE (default 12345)\n"
      "  --ssp N        statistical security parameter (default 40)\n"
      "  --seg-ands N   ANDs per emp-agmpc circuit; each iteration is cut into segments (default 280000)\n"
      "  --threads N    thread-pool size per party (default and minimum 2 (n - 1) + 2)\n"
      "  --connect-timeout S  --timeout S   (defaults 60 / 600 s)\n"
      "output (server):  --verify (TEST ONLY: regenerate the clients' data and compare)  --json F  --csv F  --quiet\n"
      "test only:        --tamper KIND --cheater P   (P = 1 server, 2.. clients)\n"
      "  KIND: gt-row xy-share d-bit e-bit abit-equivocate in-mask input-equivocate chain-share label out-label out-mask echo\n"
      "built for batch sizes: %s\n"
      "exit codes: 0 ok, 1 error, 2 verify mismatch, 3 this party detected cheating, 4 a peer aborted/closed/timed out\n",
      AGMPC_NP_LIST);
}

static int dispatch(const RunCfg& C) {
  switch (C.P.batch + 1) {
#ifdef AGMPC_HAVE_NP2
    case 2: return run_party<2>(C);
#endif
#ifdef AGMPC_HAVE_NP3
    case 3: return run_party<3>(C);
#endif
#ifdef AGMPC_HAVE_NP4
    case 4: return run_party<4>(C);
#endif
#ifdef AGMPC_HAVE_NP5
    case 5: return run_party<5>(C);
#endif
#ifdef AGMPC_HAVE_NP6
    case 6: return run_party<6>(C);
#endif
#ifdef AGMPC_HAVE_NP7
    case 7: return run_party<7>(C);
#endif
#ifdef AGMPC_HAVE_NP8
    case 8: return run_party<8>(C);
#endif
#ifdef AGMPC_HAVE_NP9
    case 9: return run_party<9>(C);
#endif
    default:
      throw std::invalid_argument("this build does not support batch " + std::to_string(C.P.batch) +
                                  " (rebuild with -DAGMPC_NP_LIST including " + std::to_string(C.P.batch + 1) + ")");
  }
}

int main(int argc, char** argv) {
  std::signal(SIGPIPE, SIG_IGN);
  RunCfg C;
  std::string hosts, tamper;
  int cheater = 0, id = 0;
  try {
    if (argc < 2) { usage(); return 1; }
    const std::string role = argv[1];
    if (role == "--help" || role == "-h") { usage(); return 0; }
    if (role != "server" && role != "client") throw std::invalid_argument("first argument must be server or client");
    ArgReader r{argc, argv, 2};
    for (; r.has(); ++r.i) {
      const std::string a = argv[r.i];
      if (parse_param_flag(r, C.P)) continue;
      if (a == "--id") id = (int)parse_int(r.next_value(a), a);
      else if (a == "--hosts") hosts = r.next_value(a);
      else if (a == "--port") C.port = (int)parse_int(r.next_value(a), a);
      else if (a == "--ssp") C.ssp = (int)parse_int(r.next_value(a), a);
      else if (a == "--seg-ands") C.seg_ands = (uint64_t)parse_int(r.next_value(a), a);
      else if (a == "--threads") C.threads = (int)parse_int(r.next_value(a), a);
      else if (a == "--connect-timeout") g_agmpc.connect_timeout_s = (int)parse_int(r.next_value(a), a);
      else if (a == "--timeout") g_agmpc.timeout_s = (int)parse_int(r.next_value(a), a);
      else if (a == "--verify") C.verify = true;
      else if (a == "--json") C.json = r.next_value(a);
      else if (a == "--csv") C.csv = r.next_value(a);
      else if (a == "--quiet") C.quiet = true;
      else if (a == "--tamper") tamper = r.next_value(a);
      else if (a == "--cheater") cheater = (int)parse_int(r.next_value(a), a);
      else if (a == "--help" || a == "-h") { usage(); return 0; }
      else throw std::invalid_argument("unknown option " + a);
    }
    C.P.finalize();
    if (role == "server") {
      if (id != 0) throw std::invalid_argument("--id is for clients");
      C.party = 1;
    } else {
      if (id < 1 || id > C.P.batch) throw std::invalid_argument("client --id must be in [1, batch]");
      C.party = id + 1;
      if (C.verify || !C.csv.empty() || !C.json.empty()) throw std::invalid_argument("--verify/--csv/--json are server options");
    }
    if (C.ssp < 1 || C.ssp > 128) throw std::invalid_argument("--ssp must be in [1, 128]");
    if (C.seg_ands < 1) throw std::invalid_argument("--seg-ands must be >= 1");
    if (C.port < 1024 || C.port + 4 * (C.P.batch + 1) * (C.P.batch + 2) + 8 > 65535) throw std::invalid_argument("--port out of range");
    if (g_agmpc.connect_timeout_s < 1 || g_agmpc.timeout_s < 1) throw std::invalid_argument("timeouts must be >= 1 s");
    g_agmpc.hosts.assign((size_t)C.P.batch + 2, "127.0.0.1");
    if (!hosts.empty()) {
      std::stringstream ss(hosts);
      std::string h;
      size_t k = 1;
      while (std::getline(ss, h, ',')) { if (k >= g_agmpc.hosts.size()) throw std::invalid_argument("--hosts has more entries than parties"); g_agmpc.hosts[k++] = h; }
      if (k != g_agmpc.hosts.size()) throw std::invalid_argument("--hosts needs one IPv4 per party (server first)");
    }
    if (!tamper.empty()) { g_tamper.kind = tamper; g_tamper.who = cheater; }
    g_agmpc.party = C.party;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    usage();
    return 1;
  }
  try {
    return dispatch(C);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[%s] error: %s\n", C.party == 1 ? "server" : ("client " + std::to_string(C.party - 1)).c_str(), e.what());
    return 1;
  }
}
