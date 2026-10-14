// lr_mdml -- maliciously secure two-party baseline for logistic-regression training
// with MD-ML (online phase; preprocessing from lr_mdml_dealer).
//
//   party 0 = CLIENT: holds the training data (X, y) and the initial weights w_0
//   party 1 = SERVER: holds nothing; the only party that learns the trained weights
//
// Training: `iters` mini-batch SGD iterations of `batch` datapoints, `d` features
// (incl. bias), fixed point with F fractional bits in Z_{2^64}, piecewise-linear
// sigmoid (see lr_circuit.h). Security: SPDZ2k (k = s = 64) with MD-ML's
// circuit-dependent preprocessing; every opened value is MAC-checked (batched,
// Pi_MACCheck) before the server accepts the output.
//
// Reported per party and per protocol phase: wall/CPU time, bytes sent and received
// (payload bytes, as counted by MD-ML's Party), communication rounds; plus peak RSS.
//
// usage: lr_mdml <client|server> [options]        (see --help)

#include "lr_circuit.h"
#include "lr_params.h"
#include "mdml_ext.h"

#include <sys/resource.h>

#include <algorithm>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace lrmd;

namespace {

struct Options {
  int party = -1;
  Params P;
  std::string host = "127.0.0.1";
  int port = 12345;
  std::string dir = "results/offline", job = "lr";
  bool verify = false, quiet = false;
  Tamper tamper = Tamper::NONE;
  std::string json_path, csv_path;
};

void usage(const char* prog) {
  std::printf(
      "usage: %s <client|server> [options]\n"
      "  client = party 0 (data + initial weights), server = party 1 (learns the trained weights)\n"
      "  Run lr_mdml_dealer with the same training flags first (single-use preprocessing).\n\n"
      "%s\n"
      "protocol / network:\n"
      "  --host IP          the OTHER party's address (default 127.0.0.1); both parties connect\n"
      "  --port N           base TCP port; ports N..N+3 are used (default 12345)\n"
      "  --offline-dir DIR  preprocessing directory (default results/offline)\n"
      "  --job NAME         preprocessing file prefix (default lr)\n\n"
      "output:\n"
      "  --verify           TEST ONLY: the server regenerates the client's data from --seed and\n"
      "                     checks the result bit-exactly (needs the dealer's <job>-verify.txt)\n"
      "  --tamper arith|bin TEST ONLY: the client corrupts one share it sends; the MAC check\n"
      "                     must then abort the protocol\n"
      "  --json FILE        write all metrics as JSON\n"
      "  --csv FILE         append one summary row (header written if the file is new)\n"
      "  --quiet            do not print the trained weights\n",
      prog, kParamHelp);
}

Options parse_options(int argc, char** argv) {
  Options O;
  if (argc < 2) throw std::invalid_argument("missing role");
  const std::string role = argv[1];
  if (role == "client" || role == "0") O.party = 0;
  else if (role == "server" || role == "1") O.party = 1;
  else throw std::invalid_argument("unknown role '" + role + "' (use client or server)");
  ArgReader r{argc, argv, 2};
  for (; r.has(); ++r.i) {
    const std::string a = argv[r.i];
    if (parse_param_flag(r, O.P)) continue;
    if (a == "--host") O.host = r.next_value(a);
    else if (a == "--port") O.port = (int)parse_int(r.next_value(a), a);
    else if (a == "--offline-dir") O.dir = r.next_value(a);
    else if (a == "--job") O.job = r.next_value(a);
    else if (a == "--verify") O.verify = true;
    else if (a == "--quiet") O.quiet = true;
    else if (a == "--json") O.json_path = r.next_value(a);
    else if (a == "--csv") O.csv_path = r.next_value(a);
    else if (a == "--tamper") {
      const std::string v = r.next_value(a);
      if (v == "arith") O.tamper = Tamper::ARITH;
      else if (v == "bin") O.tamper = Tamper::BIN;
      else throw std::invalid_argument("--tamper takes arith or bin");
    } else throw std::invalid_argument("unknown option " + a);
  }
  if (O.port < 1 || O.port > 65532) throw std::invalid_argument("--port out of range");
  O.P.finalize();
  return O;
}

// What each party reports (and sends to the other at the end).
struct PartyReport {
  PhaseStat cat[NCAT];
  PhaseStat online;    // inputs -> MAC check passed
  double connect_s = 0, load_s = 0, peak_rss_mib = 0;
  double and_gates = 0, ltz = 0, opened_arith = 0, opened_bin = 0;

  std::vector<u64> serialize() const {
    std::vector<u64> v;
    auto put = [&](double x) { v.push_back(std::bit_cast<u64>(x)); };
    auto put_stat = [&](const PhaseStat& s) {
      put(s.wall_s); put(s.cpu_s); v.push_back(s.sent); v.push_back(s.rounds);
    };
    for (const auto& s : cat) put_stat(s);
    put_stat(online);
    for (double x : {connect_s, load_s, peak_rss_mib, and_gates, ltz, opened_arith, opened_bin}) put(x);
    return v;
  }
  static size_t size() { return 4 * (NCAT + 1) + 7; }
  static PartyReport deserialize(const std::vector<u64>& v) {
    PartyReport r;
    size_t k = 0;
    auto get = [&] { return std::bit_cast<double>(v[k++]); };
    auto get_stat = [&](PhaseStat& s) {
      s.wall_s = get(); s.cpu_s = get(); s.sent = v[k++]; s.rounds = v[k++];
    };
    for (auto& s : r.cat) get_stat(s);
    get_stat(r.online);
    for (double* x : {&r.connect_s, &r.load_s, &r.peak_rss_mib, &r.and_gates, &r.ltz, &r.opened_arith, &r.opened_bin})
      *x = get();
    return r;
  }
};

double peak_rss_mib() {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_maxrss / 1024.0;
}

std::vector<std::vector<u64>> read_verify_file(const std::string& path, uint64_t fingerprint) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("--verify: cannot open " + path);
  uint64_t fp = 0;
  size_t n = 0;
  f >> fp >> n;
  if (fp != fingerprint) throw std::runtime_error("--verify: " + path + " is for different training flags");
  std::vector<std::vector<u64>> out(n);
  for (auto& v : out) {
    size_t m = 0;
    f >> m;
    v.resize(m);
    for (auto& x : v) f >> x;
  }
  if (!f) throw std::runtime_error("--verify: " + path + " is truncated");
  return out;
}

void json_stat(std::ostringstream& js, const PhaseStat& s) {
  js << "{\"wall_s\":" << s.wall_s << ",\"cpu_s\":" << s.cpu_s << ",\"sent\":" << s.sent << ",\"recv\":" << s.recv
     << ",\"rounds\":" << s.rounds << "}";
}
void json_party(std::ostringstream& js, const PartyReport& r) {
  js << "{\"phases\":{";
  for (int c = 0; c < NCAT; ++c) {
    js << (c ? "," : "") << "\"" << kCatKey[c] << "\":";
    json_stat(js, r.cat[c]);
  }
  js << "},\"online\":";
  json_stat(js, r.online);
  js << ",\"connect_s\":" << r.connect_s << ",\"load_preprocessing_s\":" << r.load_s
     << ",\"peak_rss_mib\":" << r.peak_rss_mib << "}";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    usage(argv[0]);
    return 0;
  }
  Options O;
  try {
    O = parse_options(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s (see --help)\n", e.what());
    return 1;
  }
  const Params& P = O.P;
  const bool is_server = O.party == 1;
  const char* me = is_server ? "server" : "client";

  try {
    namespace fs = std::filesystem;
    const std::string base = (fs::absolute(O.dir) / O.job).string();
    const std::string my_file = base + "-party-" + std::to_string(O.party) + ".txt";
    if (!fs::exists(my_file)) throw std::runtime_error("no preprocessing at " + my_file + " (run lr_mdml_dealer first)");
    ::setenv("MDML_PEER_HOST", O.host.c_str(), 1);

    PartyReport rep;
    double t = now_s();
    MdParty party((size_t)O.party, 2, (size_t)O.port, base);  // connects, opens the file, reads alpha
    rep.connect_s = now_s() - t;

    t = now_s();
    const auto hdr = party.ReadClear(2);  // beta share, circuit fingerprint
    if (!party.input_file() || hdr[1] != P.fingerprint())
      throw std::runtime_error(my_file + " was generated for different training flags");
    OnlineCtx ctx(party, hdr[0]);
    ctx.tamper = O.tamper;
    OnlineBackend be{ctx};
    auto C = build_lr(be, P);
    C.out->readOfflineFromFile();
    if (!party.input_file()) throw std::runtime_error(my_file + " is truncated");
    rep.load_s = now_s() - t;

    if (!is_server) {
      const auto D = generate_client_data(P);
      const auto I = encode_client_inputs(P, D, generate_initial_weights(P));
      std::static_pointer_cast<InputGateU64>(C.w0)->setInput(I.w0);
      for (int i = 0; i < P.iters; ++i) {
        std::static_pointer_cast<InputGateU64>(C.X[i])->setInput(I.X[i]);
        std::static_pointer_cast<InputGateU64>(C.Y[i])->setInput(I.Y[i]);
      }
    }
    std::fprintf(stderr, "[%s] connected (%.2f s), preprocessing loaded (%.2f s); %s\n", me, rep.connect_s,
                 rep.load_s, params_string(P).c_str());

    // ---------------- online phase ----------------
    ctx.exchange(std::vector<uint8_t>{1}, C_LOCAL);  // start together
    ctx.stat[C_LOCAL] = PhaseStat{};
    const uint64_t sent0 = party.bytes_sent();
    const double w0 = now_s(), c0 = cpu_now_s();
    for (auto& g : be.inputs) g->RunOnline();  // the client streams all inputs first
    ctx.stat[C_INPUT].rounds = 1;
    C.out->RunOnline();
    const bool mac_ok = mac_check(ctx);
    rep.online.wall_s = now_s() - w0;
    rep.online.cpu_s = cpu_now_s() - c0;
    rep.online.sent = party.bytes_sent() - sent0;
    for (int c = 0; c < NCAT; ++c) {
      rep.cat[c] = ctx.stat[c];
      rep.online.rounds += ctx.stat[c].rounds;
    }
    rep.peak_rss_mib = peak_rss_mib();
    rep.and_gates = (double)ctx.and_gates;
    rep.ltz = (double)ctx.ltz_count;
    rep.opened_arith = (double)ctx.arith_open.size();
    rep.opened_bin = (double)ctx.bin_open.size();

    // exchange reports (after the measurement)
    auto peer = PartyReport::deserialize(ctx.exchange(rep.serialize(), C_LOCAL));
    for (int c = 0; c < NCAT; ++c) { rep.cat[c].recv = peer.cat[c].sent; peer.cat[c].recv = rep.cat[c].sent; }
    rep.online.recv = peer.online.sent;
    peer.online.recv = rep.online.sent;
    const PartyReport& cl = is_server ? peer : rep;
    const PartyReport& sv = is_server ? rep : peer;

    if (!mac_ok) {
      std::printf("[%s] ABORT: MAC check failed -- a party deviated; the output is rejected\n", me);
      return 3;
    }

    // ---------------- result ----------------
    std::vector<u64> w;
    std::string verify_str = "not run";
    double max_err_double = NAN;
    int rc = 0;
    if (is_server) {
      w = be.output->result();
      if (O.verify) {
        const auto D = generate_client_data(P);
        const auto w0q = generate_initial_weights(P);
        const auto I = encode_client_inputs(P, D, w0q);
        const auto expect = emulate_protocol(P, I, read_verify_file(base + "-verify.txt", P.fingerprint()));
        size_t bad = 0;
        for (int j = 0; j < P.d; ++j) bad += expect[j] != w[j];
        verify_str = bad ? "FAIL" : "PASS";
        if (bad) rc = 4;
        const auto ref = reference_double(P, D, w0q);
        max_err_double = 0;
        for (int j = 0; j < P.d; ++j) max_err_double = std::max(max_err_double, std::fabs(fx_to_double(P, w[j]) - ref[j]));
      }
    }

    // ---------------- report ----------------
    std::printf("\n=== MD-ML logistic regression (2PC, malicious, online phase) -- %s view ===\n", me);
    std::printf("%s\n", params_string(P).c_str());
    std::printf("SPDZ2k k=s=64; binary SPDZ2k k=1,s=64 for LTZ; MAC check: %s\n\n", mac_ok ? "PASS" : "FAIL");
    std::printf("%-36s | %10s %10s | %10s %10s | %10s %10s | %6s\n", "phase (online)", "client ms", "client cpu",
                "server ms", "server cpu", "c->s KiB", "s->c KiB", "rounds");
    std::printf("%s\n", std::string(116, '-').c_str());
    for (int c = 0; c < NCAT; ++c)
      std::printf("%-36s | %10.2f %10.2f | %10.2f %10.2f | %10.2f %10.2f | %6llu\n", kCatLabel[c], cl.cat[c].wall_s * 1e3,
                  cl.cat[c].cpu_s * 1e3, sv.cat[c].wall_s * 1e3, sv.cat[c].cpu_s * 1e3, cl.cat[c].sent / 1024.0,
                  sv.cat[c].sent / 1024.0, (unsigned long long)std::max(cl.cat[c].rounds, sv.cat[c].rounds));
    std::printf("%s\n", std::string(116, '-').c_str());
    std::printf("%-36s | %10.2f %10.2f | %10.2f %10.2f | %10.2f %10.2f | %6llu\n", "TOTAL online (inputs -> MAC check)",
                cl.online.wall_s * 1e3, cl.online.cpu_s * 1e3, sv.online.wall_s * 1e3, sv.online.cpu_s * 1e3,
                cl.online.sent / 1024.0, sv.online.sent / 1024.0,
                (unsigned long long)std::max(cl.online.rounds, sv.online.rounds));
    const uint64_t total = cl.online.sent + sv.online.sent;
    std::printf("\ncommunication: total %.3f MiB (client->server %.3f MiB, server->client %.3f MiB)\n",
                total / 1048576.0, cl.online.sent / 1048576.0, sv.online.sent / 1048576.0);
    std::printf("work: %.0f LTZ comparisons, %.0f binary AND gates; MAC-checked openings: %.0f arithmetic, %.0f binary\n",
                sv.ltz, sv.and_gates, sv.opened_arith, sv.opened_bin);
    std::printf("not in the online total: connect client %.2f s / server %.2f s; loading preprocessing from file "
                "client %.2f s / server %.2f s\n",
                cl.connect_s, sv.connect_s, cl.load_s, sv.load_s);
    std::printf("peak RSS: client %.1f MiB, server %.1f MiB\n", cl.peak_rss_mib, sv.peak_rss_mib);
    if (is_server) {
      if (!O.quiet) {
        std::printf("\ntrained weights (server only):");
        for (int j = 0; j < P.d && j < 16; ++j) std::printf(" %.5f", fx_to_double(P, w[j]));
        if (P.d > 16) std::printf(" ... (%d more)", P.d - 16);
        std::printf("\n");
      }
      if (O.verify)
        std::printf("verify (bit-exact vs. protocol emulation): %s; max |w - w_double| = %.3g\n", verify_str.c_str(),
                    max_err_double);
    } else {
      std::printf("\n(client learns no output)\n");
    }

    if (!O.json_path.empty()) {
      std::ostringstream js;
      js << "{\"view\":\"" << me << "\",\"d\":" << P.d << ",\"iters\":" << P.iters << ",\"batch\":" << P.batch
         << ",\"lr\":" << P.lr << ",\"frac_bits\":" << P.frac << ",\"seed\":" << P.seed
         << ",\"mac_check\":" << (mac_ok ? "true" : "false") << ",\"verify\":\"" << verify_str << "\""
         << ",\"ltz\":" << sv.ltz << ",\"and_gates\":" << sv.and_gates << ",\"comm_total_bytes\":" << total
         << ",\"client\":";
      json_party(js, cl);
      js << ",\"server\":";
      json_party(js, sv);
      if (is_server) {
        js << ",\"weights\":[";
        for (int j = 0; j < P.d; ++j) js << (j ? "," : "") << fx_to_double(P, w[j]);
        js << "]";
      }
      js << "}\n";
      std::ofstream(O.json_path) << js.str();
    }
    if (!O.csv_path.empty()) {
      const bool fresh = !fs::exists(O.csv_path);
      std::ofstream csv(O.csv_path, std::ios::app);
      if (fresh)
        csv << "view,d,iters,batch,lr,frac_bits,client_online_ms,server_online_ms,comm_total_bytes,c2s_bytes,s2c_bytes,"
               "rounds,forward_ms,comparison_ms,sigmoid_mul_ms,gradient_ms,output_ms,mac_check_ms,input_ms,"
               "ltz,and_gates,client_load_ms,server_load_ms,client_rss_mib,server_rss_mib,mac_check,verify\n";
      char row[1024];
      std::snprintf(row, sizeof row,
                    "%s,%d,%d,%d,%g,%d,%.3f,%.3f,%llu,%llu,%llu,%llu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.0f,%.0f,%.3f,"
                    "%.3f,%.1f,%.1f,%s,%s\n",
                    me, P.d, P.iters, P.batch, P.lr, P.frac, cl.online.wall_s * 1e3, sv.online.wall_s * 1e3,
                    (unsigned long long)total, (unsigned long long)cl.online.sent, (unsigned long long)sv.online.sent,
                    (unsigned long long)std::max(cl.online.rounds, sv.online.rounds), sv.cat[C_FWD].wall_s * 1e3,
                    sv.cat[C_CMP].wall_s * 1e3, sv.cat[C_SIGMUL].wall_s * 1e3, sv.cat[C_BWD].wall_s * 1e3,
                    sv.cat[C_OUTPUT].wall_s * 1e3, sv.cat[C_MACCHECK].wall_s * 1e3, sv.cat[C_INPUT].wall_s * 1e3,
                    sv.ltz, sv.and_gates, cl.load_s * 1e3, sv.load_s * 1e3, cl.peak_rss_mib, sv.peak_rss_mib,
                    mac_ok ? "PASS" : "FAIL", verify_str.c_str());
      csv << row;
    }
    return rc;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[%s] error: %s\n", me, e.what());
    return 1;
  }
}
