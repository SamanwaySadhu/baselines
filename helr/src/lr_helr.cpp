// lr_helr -- HELR baseline (Kim, Song, Wang, Xia, Jiang, JMIR Med Inform 2018): logistic
// regression training by homomorphic gradient descent on CKKS/HEAAN ciphertexts, as a
// two-party protocol with malicious-security checks.
//
//   CLIENT = data owner and HE key owner: holds the training data (x, y), the initial
//            weights w0 (random for now) and the HE secret key; learns nothing
//   SERVER = evaluator: runs the whole training on ciphertexts; learns only the trained
//            weights (the client decrypts them for it, with a MAC the server checks)
//
// Phases (helr_protocol.h has the messages):
//   offline  client: coin toss for the public key, keygen, encryption of the data and of w0,
//                    the plaintext run it will check the result against, ONE upload flight
//            server: parameter setup, coin toss, receive + deserialize the upload
//   online   server: the homomorphic training (iters x [inner products, polynomial sigmoid,
//                    per-feature gradient, AllSum, update]) with no interaction, then the
//                    output phase (MAC ciphertexts, commit / seed / open: 2 round trips)
//
// Reported per party: wall and process-CPU time of every phase, payload bytes in each
// direction, flights, peak RSS; the parties exchange their numbers at the end.
//
// usage: lr_helr <client|server> [options]       (see --help)

#include "helr_protocol.h"
#include "lr_params.h"
#include "lr_reference.h"
#include "metrics.h"
#include "net.h"

#include <NTL/version.h>
#include <openssl/opensslv.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using lrhe::Channel;
using lrhe::Metrics;
using lrhe::PhaseStat;

namespace {

// --------------------------------------------------------------- options
struct Options {
  bool is_client = false;
  lrhe::Params P;
  lrhe::HeOptions H;
  std::string host = "127.0.0.1";
  int port = 23456;
  double tol = std::ldexp(1.0, -8);
  bool verify = false;
  bool quiet = false;
  std::string tamper;
  std::string json_path, csv_path;
};

void usage(const char* prog) {
  std::printf(
      "usage: %s <client|server> [options]\n"
      "  client = data owner and HE key owner (holds data, w0 and the secret key)\n"
      "  server = evaluator (runs the training on ciphertexts; learns only the final weights)\n\n"
      "training (both parties must pass the same values; --seed / --zero-init only shape the\n"
      "client's private inputs and are used by the server only for --verify):\n"
      "  --d N            features incl. the bias column (default 5)\n"
      "  --iters N        iterations, each on its own mini-batch (default 20)\n"
      "  --batch N        datapoints per iteration (default 4)\n"
      "  --qbits N        quantization bits of the inputs, 2..12 (default 6: integers in [-32, 31]\n"
      "                   at scale 2^5), as in the other baselines\n"
      "  --lr X           learning rate; batch/lr must be a power of two (default 0.5)\n"
      "  --no-bias        the last feature is a regular client feature, not the bias\n"
      "  --zero-init      client's initial weights are 0 (default: uniform random)\n"
      "  --seed N         seed of the client's random data / initial weights (default 1)\n\n"
      "homomorphic encryption (both parties must pass the same values):\n"
      "  --deg 3|7        degree of the least-squares sigmoid polynomial (default 3)\n"
      "  --logr N         the polynomial is fitted on [-R, R], R = 2^N (default: from d, at least 3,\n"
      "                   i.e. R = 8 as in the paper, so that |z^T w| <= R holds)\n"
      "  --logp N         scaling bits (default 28, as HELR)\n"
      "  --logn N|auto    ring dimension 2^N (default auto: the smallest meeting --lambda by HELR's\n"
      "                   formula, N >= (lambda + 110) / 7.2 * log(qQ); 17 for the default shape)\n"
      "  --lambda N       target security level for --logn auto (default 128)\n"
      "  --mac-bits N     extra output-modulus bits for the decryption MAC (default 48: forging\n"
      "                   probability <= 2^-(N + 2) per weight for the default range)\n"
      "  --insecure       allow a ring dimension below --lambda (tests only)\n\n"
      "protocol / network:\n"
      "  --host IP        server address the client connects to (default 127.0.0.1)\n"
      "  --port N         TCP port; the server listens on it (default 23456)\n"
      "  --threads N      threads per party: key generation, encryption, and the per-feature part\n"
      "                   of every iteration (default 0 = all cores, limited so that ~1 GB per thread\n"
      "                   at N = 2^17 fits in 70%% of available memory).\n"
      "  --tol X          client: abort unless the decrypted weights are within X of its own plaintext\n"
      "                   run of the same training (default 2^-8 = 0.0039)\n\n"
      "output:\n"
      "  --verify         TEST ONLY (server): regenerate the client's data and w0 from --seed and\n"
      "                   compare the trained weights with the plaintext polynomial training, the\n"
      "                   exact-sigmoid training and the fixed-point baselines' training. Outside the\n"
      "                   protocol: a real server never has the client's seed.\n"
      "  --json FILE      write all metrics as JSON\n"
      "  --csv FILE       server: append one summary row (header written if the file is new)\n"
      "  --quiet          no per-iteration progress lines\n"
      "  --tamper WHAT    TEST ONLY: deviate from the protocol, which the other party must detect.\n"
      "                   client: opening (lie about a weight) | commit (open another value);\n"
      "                   server: c1 (return a wrong weight) | c2 (send a c2 not derived from its seed)\n",
      prog);
}

Options parse_options(int argc, char** argv) {
  Options O;
  if (argc < 2) throw std::invalid_argument("missing role");
  const std::string role = argv[1];
  if (role == "client") O.is_client = true;
  else if (role == "server") O.is_client = false;
  else throw std::invalid_argument("unknown role '" + role + "' (use client or server)");
  lrhe::ArgReader r{argc, argv, 2};
  for (; r.has(); ++r.i) {
    const std::string a = argv[r.i];
    if (lrhe::parse_param_flag(r, O.P)) continue;
    if (a == "--host") O.host = r.next_value(a);
    else if (a == "--port") O.port = (int)lrhe::parse_int(r.next_value(a), a);
    else if (a == "--threads") O.H.threads = (int)lrhe::parse_int(r.next_value(a), a);
    else if (a == "--deg") O.H.deg = (int)lrhe::parse_int(r.next_value(a), a);
    else if (a == "--logr") O.H.logR = (int)lrhe::parse_int(r.next_value(a), a);
    else if (a == "--logp") O.H.logp = (int)lrhe::parse_int(r.next_value(a), a);
    else if (a == "--logn") {
      const std::string v = r.next_value(a);
      O.H.logN = v == "auto" ? 0 : (int)lrhe::parse_int(v, a);
    }
    else if (a == "--lambda") O.H.lambda = (int)lrhe::parse_int(r.next_value(a), a);
    else if (a == "--mac-bits") O.H.mac_bits = (int)lrhe::parse_int(r.next_value(a), a);
    else if (a == "--insecure") O.H.insecure_ok = true;
    else if (a == "--tol") O.tol = lrhe::parse_double(r.next_value(a), a);
    else if (a == "--verify") O.verify = true;
    else if (a == "--quiet") O.quiet = true;
    else if (a == "--tamper") O.tamper = r.next_value(a);
    else if (a == "--json") O.json_path = r.next_value(a);
    else if (a == "--csv") O.csv_path = r.next_value(a);
    else throw std::invalid_argument("unknown option " + a);
  }
  if (O.port < 1 || O.port > 65535) throw std::invalid_argument("--port out of range");
  if (O.H.threads < 0 || O.H.threads > 256) throw std::invalid_argument("--threads must be in [0, 256] (0 = automatic)");
  if (!(O.tol > 0)) throw std::invalid_argument("--tol must be > 0");
  O.P.finalize();
  const std::string t = O.tamper;
  if (!t.empty() && !((O.is_client && (t == "opening" || t == "commit")) || (!O.is_client && (t == "c1" || t == "c2"))))
    throw std::invalid_argument("--tamper " + t + " is not valid for the " + (O.is_client ? "client" : "server"));
  return O;
}

// ---------------------------------------------------------------- report
// What each party reports (and sends to the other at the end): its phases and a few
// scalars, as text lines "P name wall cpu sent recv flights_out flights_in calls" and
// "S key value".
struct PartyReport {
  std::map<std::string, PhaseStat> ph;
  std::map<std::string, double> sc;

  PhaseStat get(const std::string& k) const {
    auto it = ph.find(k);
    return it == ph.end() ? PhaseStat{} : it->second;
  }
  double val(const std::string& k, double dflt = 0) const {
    auto it = sc.find(k);
    return it == sc.end() ? dflt : it->second;
  }

  std::vector<uint8_t> serialize() const {
    std::ostringstream os;
    os.precision(17);
    for (const auto& [k, s] : ph)
      os << "P " << k << ' ' << s.wall_s << ' ' << s.cpu_s << ' ' << s.sent << ' ' << s.recv << ' ' << s.flights_out
         << ' ' << s.flights_in << ' ' << s.calls << '\n';
    for (const auto& [k, v] : sc) os << "S " << k << ' ' << v << '\n';
    const std::string t = os.str();
    return std::vector<uint8_t>(t.begin(), t.end());
  }
  static PartyReport deserialize(const std::vector<uint8_t>& b) {
    PartyReport r;
    std::istringstream is(std::string(b.begin(), b.end()));
    std::string tag, k;
    while (is >> tag >> k) {
      if (tag == "P") {
        PhaseStat s;
        is >> s.wall_s >> s.cpu_s >> s.sent >> s.recv >> s.flights_out >> s.flights_in >> s.calls;
        s.flights = s.flights_out + s.flights_in;
        r.ph[k] = s;
      } else if (tag == "S") {
        double v;
        is >> v;
        r.sc[k] = v;
      } else {
        throw std::runtime_error("malformed peer report");
      }
    }
    return r;
  }
};

struct Row {
  const char* key;
  const char* label;
};
const Row kClientOffline[] = {{"coin_toss", "coin toss (public-key element a)"},
                              {"setup", "HE context (roots of unity)"},
                              {"keygen", "keygen (pk, mult key, rot keys)"},
                              {"encrypt", "encryption (data columns + w0)"},
                              {"expected_ref", "plaintext run (for the final check)"},
                              {"upload_send", "send upload (1 flight)"}};
const Row kServerOffline[] = {{"setup", "HE context (roots of unity)"},
                              {"coin_toss", "coin toss (public-key element a)"},
                              {"upload_recv", "receive + deserialize upload"}};
const Row kServerOnline[] = {{"eval", "HE training (all iterations)"},
                             {"mac_derive", "MAC ciphertexts c2 (d)"},
                             {"out_send", "send c1, c2"},
                             {"out_wait", "wait: client decrypts + checks"},
                             {"out_send_seeds", "send seeds"},
                             {"out_recv_open", "receive opening"},
                             {"mac_verify", "verify commitment + MAC + range"}};
const Row kClientOnline[] = {{"out_recv", "wait for server evaluation + recv c1, c2"},
                             {"out_decrypt", "decrypt c1, c2 (2d)"},
                             {"out_check", "check against own plaintext run"},
                             {"out_commit", "commit"},
                             {"out_recv_seeds", "recv seeds"},
                             {"out_verify_c2", "re-derive + compare c2 (d)"},
                             {"out_open", "send opening"}};

std::string secs(double s) {
  char b[32];
  std::snprintf(b, sizeof b, "%.2f s", s);
  return b;
}


void print_report(const Options& O, const lrhe::HeConfig& C, const PartyReport& cl, const PartyReport& sv) {
  const lrhe::Params& P = O.P;
  const lrhe::CommLayout A = lrhe::comm_layout(C);
  const PhaseStat c_off = cl.get("offline"), c_on = cl.get("online"), s_off = sv.get("offline"), s_on = sv.get("online");
  std::printf("\n============ HELR (HEAAN / CKKS homomorphic gradient descent): logistic-regression training ============\n");
  std::printf("config   : %s\n", lrhe::params_string(P).c_str());
  std::printf("HE       : N = 2^%d, log q = %d (%d iterations x %d bits + %d output bits), p = 2^%d, degree-%d sigmoid polynomial on "
              "[-%g, %g], %d slots (%d batches x %d samples, padded)\n",
              C.logN, C.logq, C.T, C.lev_iter, C.out_bits, C.logp, C.deg, C.R(), C.R(), C.nslots, C.T, C.B);
  std::printf("security : HELR's estimate (N >= (lambda+110)/7.2 log(qQ), Q = q) gives lambda ~ %d bits; sparse ternary secret (h = %d) as in\n"
              "           HEAAN; sigma = 3.2. Use the lattice estimator for a current figure. %s, NTL %s\n",
              C.lambda_est(), (int)sv.val("hamming_weight"), OPENSSL_VERSION_TEXT, NTL_VERSION);
  std::printf("protocol : client = data owner + key owner, server = evaluator; malicious security: decryption MAC (2^-%d forgery),\n"
              "           verified decryption, coin-tossed public key (see README); %d thread(s)/party (client %d, server %d)\n",
              (int)lrhe::mac_forgery_bits(C.out_bits, C.logp, O.P), O.H.threads, (int)cl.val("threads"), (int)sv.val("threads"));

  // ---- per-party table ----
  std::printf("\nper-party cost (wall = elapsed at that party; cpu = that process's CPU time, all threads; sent = payload bytes it sent)\n");
  std::printf("  %-44s | %12s %12s %11s | %12s %12s %11s\n", "", "client", "", "", "server", "", "");
  std::printf("  %-44s | %12s %12s %11s | %12s %12s %11s\n", "stage", "wall", "cpu", "sent", "wall", "cpu", "sent");
  const std::string rule(2 + 44 + 3 + 38 + 3 + 38, '-');
  std::printf("%s\n", rule.c_str());
  auto cell = [](const PhaseStat* s) {
    char b[96];
    if (!s) std::snprintf(b, sizeof b, "%12s %12s %11s", "-", "-", "-");
    else std::snprintf(b, sizeof b, "%12s %12s %11s", secs(s->wall_s).c_str(), secs(s->cpu_s).c_str(),
                       lrhe::human_bytes(s->sent).c_str());
    return std::string(b);
  };
  auto row = [&](const std::string& name, const PhaseStat* a, const PhaseStat* b) {
    std::printf("  %-44s | %s | %s\n", name.c_str(), cell(a).c_str(), cell(b).c_str());
  };
  for (const Row& r : kClientOffline) { PhaseStat s = cl.get(r.key); row(std::string("  ") + r.label, &s, nullptr); }
  for (const Row& r : kServerOffline) { PhaseStat s = sv.get(r.key); row(std::string("  ") + r.label, nullptr, &s); }
  row("OFFLINE total", &c_off, &s_off);
  for (const Row& r : kServerOnline) { PhaseStat s = sv.get(r.key); row(std::string("  ") + r.label, nullptr, &s); }
  for (const Row& r : kClientOnline) { PhaseStat s = cl.get(r.key); row(std::string("  ") + r.label, &s, nullptr); }
  row("ONLINE total", &c_on, &s_on);
  PhaseStat c_tot = c_off, s_tot = s_off;
  c_tot.add(c_on);
  s_tot.add(s_on);
  row("TOTAL (offline + online)", &c_tot, &s_tot);
  std::printf("  (client online includes the wait for the server's evaluation; the client's own online compute is\n"
              "   decrypt + check + re-derive: %s)\n",
              secs(c_on.wall_s - cl.get("out_recv").wall_s).c_str());

  // ---- summary: the three numbers a comparison table needs ----
  {
    const double c_on_compute = c_on.wall_s - cl.get("out_recv").wall_s;  // without waiting for the server
    const double s_on_compute = s_on.wall_s - sv.get("out_wait").wall_s;  // without waiting for the client
    std::printf("\n  SUMMARY\n");
    std::printf("    preprocessing (offline) : client %s wall / %s cpu | server %s wall / %s cpu\n", secs(c_off.wall_s).c_str(),
                secs(c_off.cpu_s).c_str(), secs(s_off.wall_s).c_str(), secs(s_off.cpu_s).c_str());
    std::printf("    online computation      : client %s (own compute; %s incl. waiting for the server) | server %s wall / %s cpu "
                "(%s excluding the wait for the client)\n",
                secs(c_on_compute).c_str(), secs(c_on.wall_s).c_str(), secs(s_on.wall_s).c_str(), secs(s_on.cpu_s).c_str(),
                secs(s_on_compute).c_str());
    std::printf("    total communication     : %s = offline %s + online %s  (client->server %s, server->client %s)\n",
                lrhe::human_bytes(c_off.sent + s_off.sent + c_on.sent + s_on.sent).c_str(),
                lrhe::human_bytes(c_off.sent + s_off.sent).c_str(), lrhe::human_bytes(c_on.sent + s_on.sent).c_str(),
                lrhe::human_bytes(c_off.sent + c_on.sent).c_str(), lrhe::human_bytes(s_off.sent + s_on.sent).c_str());
  }

  // ---- where the evaluation time goes ----
  std::printf("\n  HE EVALUATION (server): thread-seconds per operation class, summed over threads (they exceed the wall time with --threads > 1)\n");
  const char* ops[] = {"mult", "square", "rotate", "mult_by_plaintext", "mult_by_const", "rescale", "add", "embed/copy"};
  for (const char* k : ops) {
    const double t = sv.val(std::string("evalop_s_") + k), n = sv.val(std::string("evalop_n_") + k);
    if (n > 0) std::printf("      %-18s %10.2f s  %8.0f ops  (%.3f s each)\n", k, t, n, t / n);
  }
  if (sv.val("iter_mean_s") > 0)
    std::printf("      iterations: mean %.2f s, first %.2f s, last %.2f s (the modulus shrinks by %d bits per iteration)\n",
                sv.val("iter_mean_s"), sv.val("iter_first_s"), sv.val("iter_last_s"), C.lev_iter);

  // ---- communication ----
  const uint64_t off_c2s = c_off.sent, off_s2c = s_off.sent, on_c2s = c_on.sent, on_s2c = s_on.sent;
  const uint64_t c2s = off_c2s + on_c2s, s2c = off_s2c + on_s2c;
  std::printf("\n  COMMUNICATION: total %s (%" PRIu64 " bytes); client->server %s, server->client %s\n",
              lrhe::human_bytes(c2s + s2c).c_str(), c2s + s2c, lrhe::human_bytes(c2s).c_str(), lrhe::human_bytes(s2c).c_str());
  std::printf("      offline: client->server %s (%" PRIu64 " B), server->client %s (%" PRIu64 " B)\n",
              lrhe::human_bytes(off_c2s).c_str(), off_c2s, lrhe::human_bytes(off_s2c).c_str(), off_s2c);
  std::printf("               upload = header+seed %" PRIu64 " B + keys %s (pk.b + mult key + %d rotation keys) + data %s (%d x %s)\n"
              "                        + w0 %s (%d x %s)\n",
              A.coin_c2s, lrhe::human_bytes(A.keys).c_str(), C.nrot, lrhe::human_bytes(A.data).c_str(), C.d,
              lrhe::human_bytes(lrhe::cipher_bytes(C.N(), C.logq)).c_str(), lrhe::human_bytes(A.weights).c_str(), C.d,
              lrhe::human_bytes(lrhe::cipher_bytes(C.N(), C.logq)).c_str());
  std::printf("      online : client->server %s (%" PRIu64 " B), server->client %s (%" PRIu64 " B)\n",
              lrhe::human_bytes(on_c2s).c_str(), on_c2s, lrhe::human_bytes(on_s2c).c_str(), on_s2c);
  std::printf("               output = c1 + c2 for %d weights (%d x %s) + %" PRIu64 " B seeds; client->server: commitment + opening\n",
              C.d, 2 * C.d, lrhe::human_bytes(lrhe::cipher_bytes(C.N(), C.out_bits)).c_str(), A.seeds);
  const bool match = off_c2s == A.offline_c2s() && off_s2c == A.offline_s2c() && on_c2s == A.online_c2s() && on_s2c == A.online_s2c();
  std::printf("      analytic: offline c->s = 96 + keys + 2d ct(q); online s->c = 2d ct(2^%d) + 32d, c->s = 64 + 2d*%zu [%s]\n",
              C.out_bits, lrhe::coeff_bytes(C.out_bits), match ? "measured == analytic" : "MISMATCH");
  const uint64_t off_fl = c_off.flights_out + s_off.flights_out, on_fl = c_on.flights_out + s_on.flights_out;
  std::printf("  ROUNDS: offline %" PRIu64 " flights (coin toss 3 + upload 1); online %" PRIu64
              " flights (output, commit, seeds, opening = 2 round trips); the HE evaluation itself is non-interactive\n",
              off_fl, on_fl);
  std::printf("  peak RSS: client %.1f MiB, server %.1f MiB\n", cl.val("peak_rss_mib"), sv.val("peak_rss_mib"));
  if (cl.get("offline").sent + cl.get("online").sent != sv.get("offline").recv + sv.get("online").recv ||
      sv.get("offline").sent + sv.get("online").sent != cl.get("offline").recv + cl.get("online").recv)
    std::printf("  WARNING: byte counters of the two parties disagree\n");
  std::printf("  client check: max |decrypted - own plaintext run| = %.3g (tolerance %.3g); the polynomial stayed within its fit interval: %s "
              "(max |z^T w| / R = %.3f)\n",
              cl.val("worst_deviation"), cl.val("tol"), cl.val("max_w_ratio") <= 1.0 ? "yes" : "NO", cl.val("max_w_ratio"));

  // ---- leakage ----
  std::printf("\n  LEAKAGE. The server sees ciphertexts under the client's key and, at the end, the decryptions of the final weights\n"
              "  (c1: u1_j = round(w_j 2^%d) + HE noise; c2: uniform) -- nothing else. The client learns nothing about the server's\n"
              "  inputs (it has none). Malicious-security scope and the residual leakage (one abort bit, tolerance window, no ZK proofs\n"
              "  of well-formed client ciphertexts / keys): README \"Security model\".\n", C.logp);
}

void json_phase(std::ostringstream& js, const PhaseStat& s) {
  js << "{\"wall_ms\":" << s.wall_s * 1e3 << ",\"cpu_ms\":" << s.cpu_s * 1e3 << ",\"sent_bytes\":" << s.sent
     << ",\"recv_bytes\":" << s.recv << ",\"flights_out\":" << s.flights_out << ",\"flights_in\":" << s.flights_in << "}";
}
void json_party(std::ostringstream& js, const PartyReport& r) {
  js << "{\"phases\":{";
  bool first = true;
  for (const auto& [k, s] : r.ph) {
    js << (first ? "" : ",") << "\"" << k << "\":";
    json_phase(js, s);
    first = false;
  }
  js << "},\"values\":{";
  first = true;
  for (const auto& [k, v] : r.sc) {
    js << (first ? "" : ",") << "\"" << k << "\":" << v;
    first = false;
  }
  js << "}}";
}

}  // namespace

int main(int argc, char** argv) {
  Options O;
  try {
    for (int i = 1; i < argc; ++i)
      if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
    O = parse_options(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n\n", e.what());
    usage(argv[0]);
    return 2;
  }
  const lrhe::Params& P = O.P;
  const char* me = O.is_client ? "client" : "server";
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  lrhe::seed_ntl_from_os();

  lrhe::HeConfig C;
  try {
    C = lrhe::make_he_config(P, O.H);
    if (O.H.threads == 0) O.H.threads = lrhe::auto_threads(C.N());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }

  Metrics M;
  PartyReport rep;
  std::unique_ptr<Channel> ch;
  std::vector<double> weights;
  std::vector<double> iter_wall_s;
  std::map<std::string, double> op_s, op_n;
  double worst_dev = 0, max_w_ratio = 0;
  int rc = 0;
  try {
    std::printf("[%s] %s\n", me, lrhe::params_string(P).c_str());
    std::printf("[%s] HE: N=2^%d logq=%d logp=%d deg=%d R=%g slots=%d, lambda~%d by HELR's formula\n", me, C.logN, C.logq,
                C.logp, C.deg, C.R(), C.nslots, C.lambda_est());
    const lrhe::Tamper tamper{O.tamper};

    if (O.is_client) {
      // ================================================================ CLIENT
      lrhe::HeClient Cl(P, O.H, lrhe::generate_client_data(P), lrhe::generate_initial_weights(P), tamper);
      Cl.tol = O.tol;
      const lrhe::Mark m0 = M.now();
      ch = Channel::connect(O.host, O.port, 1800.0);
      // offline phases are timed individually below; connecting (waiting for the server's own
      // setup) belongs to neither party's cost
      const lrhe::Mark mo_start = M.now(ch.get());
      Cl.coin_toss(*ch, &M);
      Cl.setup(&M);
      Cl.upload(*ch, &M);
      PhaseStat off;
      for (const Row& r : kClientOffline) off.add(M.get(r.key));
      off.calls = 1;
      M.add("offline", off);
      if (!O.quiet)
        std::printf("[client] offline done in %.2f s (keygen %.2f s, encrypt %.2f s); uploaded %s; waiting for the server\n",
                    off.wall_s, M.get("keygen").wall_s, M.get("encrypt").wall_s,
                    lrhe::human_bytes(M.get("upload_send").sent + M.get("coin_toss").sent).c_str());
      (void)m0;
      (void)mo_start;
      // ---- online ----
      const lrhe::Mark mn = M.now(ch.get());
      Cl.output(*ch, &M);
      PhaseStat on = lrhe::phase_diff(mn, M.now(ch.get()));
      // bytes / flights of the online phase are counted from the end of the upload
      M.add("online", on);
      worst_dev = Cl.worst_deviation();
      max_w_ratio = Cl.max_w_ratio();
      weights = Cl.decrypted();
    } else {
      // ================================================================ SERVER
      lrhe::HeServer S(P, O.H, tamper);
      S.setup(&M);
      std::printf("[server] listening on port %d\n", O.port);
      ch = Channel::listen(O.port);
      S.coin_toss(*ch, &M);
      S.receive(*ch, &M);
      PhaseStat off;
      for (const Row& r : kServerOffline) off.add(M.get(r.key));
      off.calls = 1;
      M.add("offline", off);
      if (!O.quiet)
        std::printf("[server] received the upload (%s) in %.2f s; evaluating %d iterations on ciphertexts\n",
                    lrhe::human_bytes(M.get("upload_recv").recv).c_str(), M.get("upload_recv").wall_s, P.iters);
      S.progress = [&](int t, double s) {
        if (!O.quiet) std::printf("[server] iteration %d/%d done (%.1f s)\n", t + 1, P.iters, s);
      };
      const lrhe::Mark mn = M.now(ch.get());
      S.evaluate(&M);
      S.output(*ch, &M);
      PhaseStat on = lrhe::phase_diff(mn, M.now(ch.get()));
      M.add("online", on);
      weights = S.weights();
      iter_wall_s = S.stats().iter_wall_s;
      for (const auto& [k, v] : S.stats().thread_s) op_s[k] = v;
      for (const auto& [k, v] : S.stats().ops) op_n[k] = (double)v;
    }
  } catch (const lrhe::ProtocolError& e) {
    std::fprintf(stderr, "[%s] ABORT: %s\n", me, e.what());
    return 3;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[%s] ERROR: %s\n", me, e.what());
    return 1;
  }

  // ---- exchange reports over the same socket (after all measurements) ----
  for (const auto& [k, s] : M.phases()) rep.ph[k] = s;
  rep.sc["peak_rss_mib"] = (double)lrhe::peak_rss_kib() / 1024.0;
  rep.sc["threads"] = O.H.threads;
  rep.sc["d"] = P.d; rep.sc["iters"] = P.iters; rep.sc["batch"] = P.batch; rep.sc["qbits"] = P.qbits;
  rep.sc["lr"] = P.lr; rep.sc["bias"] = P.bias ? 1 : 0;
  rep.sc["hamming_weight"] = (double)std::min<long>(64, std::max<long>(2, C.N() / 4));
  if (O.is_client) {
    rep.sc["worst_deviation"] = worst_dev;
    rep.sc["max_w_ratio"] = max_w_ratio;
    rep.sc["tol"] = O.tol;
  } else {
    for (const auto& [k, v] : op_s) rep.sc["evalop_s_" + k] = v;
    for (const auto& [k, v] : op_n) rep.sc["evalop_n_" + k] = v;
    if (!iter_wall_s.empty()) {
      double sum = 0;
      for (double v : iter_wall_s) sum += v;
      rep.sc["iter_mean_s"] = sum / (double)iter_wall_s.size();
      rep.sc["iter_first_s"] = iter_wall_s.front();
      rep.sc["iter_last_s"] = iter_wall_s.back();
    }
  }
  PartyReport peer;
  try {
    if (O.is_client) {
      ch->send_blob(rep.serialize());
      peer = PartyReport::deserialize(ch->recv_blob(1 << 20));
    } else {
      peer = PartyReport::deserialize(ch->recv_blob(1 << 20));
      ch->send_blob(rep.serialize());
      ch->flush();
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[%s] ERROR: report exchange failed: %s\n", me, e.what());
    return 1;
  }
  for (const char* k : {"d", "iters", "batch", "qbits", "lr", "bias"})
    if (peer.val(k, -1) != rep.val(k)) {
      std::fprintf(stderr, "[%s] ERROR: parameter %s differs between the parties (%g here, %g at the peer)\n", me, k,
                   rep.val(k), peer.val(k, -1));
      return 1;
    }
  const PartyReport& cl = O.is_client ? rep : peer;
  const PartyReport& sv = O.is_client ? peer : rep;
  print_report(O, C, cl, sv);

  // ---- trained weights (server) / verification ----
  std::string verified = "not-checked";
  double dev_poly = -1, dev_sigmoid = -1, dev_fixed = -1;
  if (!O.is_client) {
    std::printf("\n[server] trained weights:");
    for (size_t j = 0; j < weights.size() && j < 12; ++j) std::printf(" %.5f", weights[j]);
    std::printf("%s\n", weights.size() > 12 ? " ..." : "");
    if (O.verify) {
      const lrhe::Dataset Dv = lrhe::generate_client_data(P);
      const std::vector<int64_t> w0 = lrhe::generate_initial_weights(P);
      const lrhe::HPoly H = lrhe::fit_sigmoid_poly(C.deg, C.logR);
      const std::vector<double> ref_poly = lrhe::poly_train(P, Dv, w0, H);
      const std::vector<double> ref_sig = lrhe::sigmoid_train(P, Dv, w0);
      const std::vector<int64_t> fx = lrhe::reference_train(P, Dv, w0);
      auto maxdiff = [&](const std::vector<double>& r) {
        double m = 0;
        for (size_t j = 0; j < weights.size(); ++j) m = std::max(m, std::fabs(weights[j] - r[j]));
        return m;
      };
      std::vector<double> fxd(fx.size());
      for (size_t j = 0; j < fx.size(); ++j) fxd[j] = (double)fx[j] / std::ldexp(1.0, P.f);
      dev_poly = maxdiff(ref_poly);
      dev_sigmoid = maxdiff(ref_sig);
      dev_fixed = maxdiff(fxd);
      const bool ok = dev_poly <= O.tol;
      verified = ok ? "pass" : "FAIL";
      std::printf("[server] verify: %s -- max |HE - plaintext polynomial training| = %.3g (tolerance %.3g)\n", ok ? "PASS" : "FAIL",
                  dev_poly, O.tol);
      std::printf("[server]   distance of the HE weights to: exact-sigmoid training %.3g, the other baselines' fixed-point training %.3g\n"
                  "[server]   (the polynomial sigmoid and approximate arithmetic make this a different model from theirs; compare costs, not weights)\n",
                  dev_sigmoid, dev_fixed);
      if (!ok) rc = 1;
    }
  } else {
    std::printf("\n[client] done: the server holds the trained weights; it saw only ciphertexts and the checked decryptions\n");
  }

  const lrhe::CommLayout A = lrhe::comm_layout(C);
  const PhaseStat c_off = cl.get("offline"), c_on = cl.get("online"), s_off = sv.get("offline"), s_on = sv.get("online");
  if (!O.json_path.empty()) {
    std::ostringstream js;
    js.precision(10);
    js << "{\"party\":\"" << me << "\",\"baseline\":\"helr\",\"params\":{\"d\":" << P.d << ",\"iters\":" << P.iters
       << ",\"batch\":" << P.batch << ",\"lr\":" << P.lr << ",\"qbits\":" << P.qbits << ",\"f\":" << P.f
       << ",\"bias\":" << (P.bias ? "true" : "false") << ",\"zero_init\":" << (P.zero_init ? "true" : "false")
       << ",\"seed\":" << P.seed << ",\"threads\":" << O.H.threads << ",\"logN\":" << C.logN << ",\"logq\":" << C.logq
       << ",\"logp\":" << C.logp << ",\"deg\":" << C.deg << ",\"logR\":" << C.logR << ",\"nslots\":" << C.nslots
       << ",\"mac_bits\":" << C.mac_bits << ",\"out_bits\":" << C.out_bits << ",\"lambda_est\":" << C.lambda_est()
       << "},\"verified\":\"" << verified << "\",\"communication\":{\"offline_c2s_bytes\":" << c_off.sent
       << ",\"offline_s2c_bytes\":" << s_off.sent << ",\"online_c2s_bytes\":" << c_on.sent << ",\"online_s2c_bytes\":"
       << s_on.sent << ",\"total_bytes\":" << c_off.sent + s_off.sent + c_on.sent + s_on.sent
       << ",\"upload\":{\"keys\":" << A.keys << ",\"data\":" << A.data << ",\"weights\":" << A.weights
       << "}},\"rounds\":{\"offline_flights\":" << c_off.flights_out + s_off.flights_out << ",\"online_flights\":"
       << c_on.flights_out + s_on.flights_out << "}";
    if (!O.is_client) {
      js << ",\"weights\":[";
      for (size_t j = 0; j < weights.size(); ++j) js << (j ? "," : "") << weights[j];
      js << "],\"server_iteration_wall_s\":[";
      for (size_t i = 0; i < iter_wall_s.size(); ++i) js << (i ? "," : "") << iter_wall_s[i];
      js << "],\"deviation\":{\"poly\":" << dev_poly << ",\"sigmoid\":" << dev_sigmoid << ",\"fixed_point_baselines\":" << dev_fixed << "}";
    }
    js << ",\"client\":";
    json_party(js, cl);
    js << ",\"server\":";
    json_party(js, sv);
    js << "}\n";
    std::ofstream(O.json_path) << js.str();
  }
  if (!O.csv_path.empty() && !O.is_client) {
    const std::string header =
        "party,d,iters,batch,qbits,lr,threads,logN,logq,deg,logR,nslots,"
        "client_keygen_s,client_encrypt_s,client_upload_send_s,client_offline_s,client_offline_cpu_s,"
        "client_wait_s,client_online_s,client_online_cpu_s,"
        "server_setup_s,server_upload_recv_s,server_offline_s,server_offline_cpu_s,"
        "server_eval_s,server_eval_cpu_s,server_online_s,server_online_cpu_s,"
        "comm_offline_c2s_bytes,comm_offline_s2c_bytes,comm_online_c2s_bytes,comm_online_s2c_bytes,comm_total_bytes,"
        "offline_flights,online_flights,client_peak_rss_mib,server_peak_rss_mib,dev_poly,verified";
    std::ifstream probe(O.csv_path);
    std::string first;
    const bool fresh = !probe.good() || !std::getline(probe, first) || first.empty();
    probe.close();
    if (!fresh && first != header) {
      std::fprintf(stderr, "[server] ERROR: %s has a different column layout; row not written. Use a new file.\n", O.csv_path.c_str());
      return rc ? rc : 1;
    }
    std::ofstream csv(O.csv_path, std::ios::app);
    if (fresh) csv << header << "\n";
    auto w = [&](const PartyReport& r, const char* k) { return r.get(k).wall_s; };
    const PhaseStat ev = sv.get("eval");
    char row[4096];
    std::snprintf(row, sizeof row,
                  "server,%d,%d,%d,%d,%g,%d,%d,%d,%d,%d,%d,"
                  "%.3f,%.3f,%.3f,%.3f,%.3f,"
                  "%.3f,%.3f,%.3f,"
                  "%.3f,%.3f,%.3f,%.3f,"
                  "%.3f,%.3f,%.3f,%.3f,"
                  "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
                  "%" PRIu64 ",%" PRIu64 ",%.1f,%.1f,%.3g,%s\n",
                  P.d, P.iters, P.batch, P.qbits, P.lr, O.H.threads, C.logN, C.logq, C.deg, C.logR, C.nslots,
                  w(cl, "keygen"), w(cl, "encrypt"), w(cl, "upload_send"), c_off.wall_s, c_off.cpu_s,
                  w(cl, "out_recv"), c_on.wall_s, c_on.cpu_s,
                  w(sv, "setup"), w(sv, "upload_recv"), s_off.wall_s, s_off.cpu_s,
                  ev.wall_s, ev.cpu_s, s_on.wall_s, s_on.cpu_s,
                  c_off.sent, s_off.sent, c_on.sent, s_on.sent, c_off.sent + s_off.sent + c_on.sent + s_on.sent,
                  c_off.flights_out + s_off.flights_out, c_on.flights_out + s_on.flights_out, cl.val("peak_rss_mib"),
                  sv.val("peak_rss_mib"), dev_poly, verified.c_str());
    csv << row;
  }
  return rc;
}
