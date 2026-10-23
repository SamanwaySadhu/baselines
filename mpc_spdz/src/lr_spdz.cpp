// lr_spdz -- logistic-regression training with a GENERIC dishonest-majority MPC protocol (SPDZ / MASCOT style:
// additive secret sharing with information-theoretic MACs, Beaver triples from OT) among 4 clients and 1 server.
//
//   lr_spdz server            [options]    party 0: no input; learns only the trained weights
//   lr_spdz client --id K     [options]    party K, K = 1..batch: holds datapoint K of every batch and one
//                                          additive share of the initial weights
//
// Malicious security against any set of corrupted parties but not all of them, with abort. See README.md.
#include <csignal>
#include <cstdio>
#include <sstream>
#include <string>
#include <thread>

#include "driver.h"

using namespace spz;

static void usage(const char* prog) {
  std::printf(
      "usage: %s server [options]\n"
      "       %s client --id K [options]        (K = 1..batch)\n\n"
      "training (every party must pass the same values; --seed / --zero-init shape the clients' test data and\n"
      "are used by the server only for --verify):\n"
      "  --d N            features incl. the bias column: the dimensionality of the data (default 5)\n"
      "  --iters N        SGD iterations (default 20)\n"
      "  --batch N        datapoints per iteration = number of clients (default 4)\n"
      "  --frac N         fractional bits F of features / labels / sigmoid output (default 13)\n"
      "  --lr X           learning rate; batch/lr must be a power of two (default 0.5)\n"
      "  --qbits N        quantization of the clients' random raw values (default 6, as in the sibling baselines)\n"
      "  --no-bias        the last feature is a regular feature, not the bias column\n"
      "  --zero-init      initial weights are 0 (default: uniform random)\n"
      "  --seed N         seed of the clients' random data / initial weights (default 1)\n\n"
      "protocol / network:\n"
      "  --hosts H0,..,Hn address of party 0 (server), 1 (client 1), ... (default all 127.0.0.1)\n"
      "  --port N         first TCP port; the run uses ports in [N, N + 17 * n^2) (default 12345)\n"
      "  --threads N      OT connections and worker threads per pair of parties (default 1; max 16)\n"
      "  --semi-honest    no MACs, no sacrifice, semi-honest OT: the same arithmetic without malicious security\n"
      "  --ssp N          statistical security parameter of the masks (default 40)\n\n"
      "output (server):\n"
      "  --verify         TEST ONLY: regenerate the clients' data from --seed and check the result against the\n"
      "                   plaintext integer reference (must be exact)\n"
      "  --json FILE      write all metrics as JSON\n"
      "  --csv FILE       append one summary row (header written if the file is new)\n"
      "  --quiet          no progress lines\n"
      "test only:  --tamper KIND --cheater P      (P = 0 server, 1.. clients) one party deviates once; the run must abort\n"
      "  KIND: open-pre open-online open-equiv triple matrix ot-send ot-choose mac-value mac-sigma input-equiv\n\n"
      "exit codes: 0 ok, 1 error / a peer aborted (connection closed), 2 verify mismatch, 3 this party detected a deviation\n",
      prog, prog);
}

int main(int argc, char** argv) {
  std::signal(SIGPIPE, SIG_IGN);
  PartyConfig C;
  std::string hosts;
  int id = 0;
  try {
    if (argc < 2) { usage(argv[0]); return 1; }
    const std::string role = argv[1];
    if (role == "--help" || role == "-h") { usage(argv[0]); return 0; }
    if (role != "server" && role != "client") throw std::invalid_argument("first argument must be server or client");
    ArgReader r{argc, argv, 2};
    for (; r.has(); ++r.i) {
      const std::string a = argv[r.i];
      if (parse_param_flag(r, C.P)) continue;
      if (a == "--id") id = (int)parse_int(r.next_value(a), a);
      else if (a == "--hosts") hosts = r.next_value(a);
      else if (a == "--port") C.port = (int)parse_int(r.next_value(a), a);
      else if (a == "--threads") C.threads = (int)parse_int(r.next_value(a), a);
      else if (a == "--semi-honest") C.semi_honest = true;
      else if (a == "--verify") C.verify = true;
      else if (a == "--quiet") C.quiet = true;
      else if (a == "--json") C.json_path = r.next_value(a);
      else if (a == "--csv") C.csv_path = r.next_value(a);
      else if (a == "--tamper") {
        const std::string k = r.next_value(a);
        if (k == "open-pre") C.tamper = Tamper::OPEN_PRE;
        else if (k == "open-online") C.tamper = Tamper::OPEN_ONLINE;
        else if (k == "open-equiv") C.tamper = Tamper::OPEN_EQUIV;
        else if (k == "triple") C.tamper = Tamper::TRIPLE_C;
        else if (k == "matrix") C.tamper = Tamper::MATRIX_Z;
        else if (k == "ot-send") C.tamper = Tamper::OT_SEND;
        else if (k == "ot-choose") C.tamper = Tamper::OT_CHOOSE;
        else if (k == "mac-value") C.tamper = Tamper::MAC_VALUE;
        else if (k == "mac-sigma") C.tamper = Tamper::MAC_SIGMA;
        else if (k == "input-equiv") C.tamper = Tamper::INPUT_EQUIV;
        else throw std::invalid_argument("unknown --tamper kind " + k);
      } else if (a == "--cheater") C.cheater = (int)parse_int(r.next_value(a), a);
      else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
      else throw std::invalid_argument("unknown option " + a);
    }
    C.P.finalize();
    const int n = C.P.batch + 1;
    if (n > kMaxParties) throw std::invalid_argument("--batch must be <= 7");
    if (role == "server") {
      if (id != 0) throw std::invalid_argument("--id is for clients");
      C.party = 0;
    } else {
      if (id < 1 || id > C.P.batch) throw std::invalid_argument("client --id must be in [1, batch]");
      C.party = id;
      if (C.verify || !C.csv_path.empty() || !C.json_path.empty()) throw std::invalid_argument("--verify/--csv/--json are server options");
    }
    if (C.threads < 1 || C.threads > kMaxWorkers) throw std::invalid_argument("--threads must be in [1, 16]");
    if (C.port < 1024 || Mesh::max_port(C.port, n) > 65535) throw std::invalid_argument("--port out of range");
    if (C.tamper != Tamper::NONE && (C.cheater < 0 || C.cheater >= n)) throw std::invalid_argument("--tamper needs --cheater P in [0, batch]");
    C.hosts.assign((size_t)n, "127.0.0.1");
    if (!hosts.empty()) {
      std::stringstream ss(hosts);
      std::string h;
      size_t k = 0;
      while (std::getline(ss, h, ',')) {
        if (k >= C.hosts.size()) throw std::invalid_argument("--hosts has more entries than parties");
        C.hosts[k++] = h;
      }
      if (k != C.hosts.size()) throw std::invalid_argument("--hosts needs one address per party (server first)");
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    usage(argv[0]);
    return 1;
  }
  return run_party(C);
}
