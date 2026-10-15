// lr_cryptonn -- CryptoNN baseline (Xu, Joshi, Li, ICDCS 2019) for fixed-point (quantized)
// logistic-regression training: the server holds the model in the clear and learns the
// data-dependent quantities through functional encryption -- FEIP (Abdalla et al., PKC
// 2015) for the inner products and the paper's FEBO for the label subtraction -- over
// NIST P-256 (OpenSSL 3).
//
//   CLIENT = data owner + CryptoNN authority: owns the training data (x, y) and the
//            initial weights w0 (random for now), holds every master secret key and
//            derives the function keys the server asks for
//   SERVER = model holder: receives w0 in the clear, runs the training, ends with the
//            trained weights (it sees every intermediate weight vector)
//
// Phases (see cryptonn_protocol.h for the message formats):
//   offline  client: keygen (3 key sets), encryption of the whole training set, ONE
//                    upload flight (header, w0, all ciphertexts)
//            server: BSGS baby-step table (input independent), receive + validate upload
//   online   3 round trips per iteration: w -> sk_w | yhat -> FEBO keys | e -> sk_e,
//            with B + B + d_data decryptions and bounded discrete logs at the server
//
// Reported per party: wall and process-CPU time of every phase, payload bytes in each
// direction (offline / online, and a breakdown of the upload), flights / round trips,
// peak RSS; both parties exchange their numbers at the end (after all measurements).
// Semi-honest security with a trusted authority (the client); see the leakage summary.
//
// usage: lr_cryptonn <client|server> [options]       (see --help)

#include "cryptonn_protocol.h"
#include "lr_params.h"
#include "lr_reference.h"
#include "metrics.h"
#include "net.h"

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

using lrcn::Channel;
using lrcn::Mark;
using lrcn::Metrics;
using lrcn::Params;
using lrcn::PhaseStat;

namespace {

// --------------------------------------------------------------- options
struct Options {
  bool is_client = false;
  Params P;
  std::string host = "127.0.0.1";
  int port = 23456;
  int threads = 1;
  int bsgs_bits = 0;  // 0 = auto
  bool enc_with_msk = false;
  bool verify = false;
  bool quiet = false;
  long long test_bound_cap = -1;
  std::string json_path, csv_path;
};

void usage(const char* prog) {
  std::printf(
      "usage: %s <client|server> [options]\n"
      "  client = data owner + CryptoNN authority (holds data, w0 and every master secret key)\n"
      "  server = model holder (runs the training on FE ciphertexts, holds w in the clear)\n\n"
      "training (both parties must pass the same values; --seed / --zero-init only shape the\n"
      "client's private inputs and are used by the server only for --verify):\n"
      "  --d N            features incl. the bias column (default 5)\n"
      "  --iters N        SGD iterations (default 20)\n"
      "  --batch N        datapoints per iteration (default 4)\n"
      "  --qbits N        quantization bits of every input, 2..12 (default 6: integers in\n"
      "                   [-32, 31] at scale 2^5); dot products are truncated back to it\n"
      "  --lr X           learning rate; batch/lr must be a power of two (default 0.5)\n"
      "  --no-bias        the last feature is a regular client feature, not the bias\n"
      "  --zero-init      client's initial weights are 0 (default: uniform random)\n"
      "  --seed N         seed of the client's random data / initial weights (default 1)\n\n"
      "protocol / network:\n"
      "  --host IP        server address the client connects to (default 127.0.0.1)\n"
      "  --port N         TCP port; the server listens on it (default 23456)\n"
      "  --threads N      threads for encryption (client) / decryption + dlog (server)\n"
      "                   (default 1; results are identical for any N)\n"
      "  --bsgs-bits N|auto  server: baby-step table of 2^N entries (default auto: covers the\n"
      "                   worst-case forward value of the initial weights in one lookup,\n"
      "                   clamped to 12..20 bits)\n"
      "  --encrypt-with-msk  client: encrypt with the master secret key (one fixed-base\n"
      "                   multiplication per component; same ciphertexts in distribution)\n"
      "                   instead of the paper's public-key encryption\n\n"
      "output:\n"
      "  --verify         TEST ONLY: the server regenerates the client's data and initial\n"
      "                   weights from --seed and checks the trained weights against the\n"
      "                   plaintext fixed-point reference (exact). This is outside the\n"
      "                   protocol: a real server never has the client's seed.\n"
      "  --json FILE      write all metrics as JSON\n"
      "  --csv FILE       server: append one summary row (header written if the file is new)\n"
      "  --quiet          no per-iteration progress lines\n"
      "  --test-bound-cap N  TEST ONLY (server): cap every discrete-log bound at N, to show\n"
      "                   that a too small bound is detected (the run aborts)\n",
      prog);
}

Options parse_options(int argc, char** argv) {
  Options O;
  if (argc < 2) throw std::invalid_argument("missing role");
  const std::string role = argv[1];
  if (role == "client") O.is_client = true;
  else if (role == "server") O.is_client = false;
  else throw std::invalid_argument("unknown role '" + role + "' (use client or server)");
  lrcn::ArgReader r{argc, argv, 2};
  for (; r.has(); ++r.i) {
    const std::string a = argv[r.i];
    if (lrcn::parse_param_flag(r, O.P)) continue;
    if (a == "--host") O.host = r.next_value(a);
    else if (a == "--port") O.port = (int)lrcn::parse_int(r.next_value(a), a);
    else if (a == "--threads") O.threads = (int)lrcn::parse_int(r.next_value(a), a);
    else if (a == "--bsgs-bits") {
      const std::string v = r.next_value(a);
      O.bsgs_bits = v == "auto" ? 0 : (int)lrcn::parse_int(v, a);
      if (v != "auto" && (O.bsgs_bits < 4 || O.bsgs_bits > 26))
        throw std::invalid_argument("--bsgs-bits must be auto or in [4, 26]");
    }
    else if (a == "--encrypt-with-msk") O.enc_with_msk = true;
    else if (a == "--verify") O.verify = true;
    else if (a == "--quiet") O.quiet = true;
    else if (a == "--json") O.json_path = r.next_value(a);
    else if (a == "--csv") O.csv_path = r.next_value(a);
    else if (a == "--test-bound-cap") O.test_bound_cap = lrcn::parse_int(r.next_value(a), a);
    else throw std::invalid_argument("unknown option " + a);
  }
  if (O.port < 1 || O.port > 65535) throw std::invalid_argument("--port out of range");
  if (O.threads < 1 || O.threads > 256) throw std::invalid_argument("--threads must be in [1, 256]");
  O.P.finalize();
  lrcn::check_protocol_params(O.P);
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
  bool has(const std::string& k) const { return ph.count(k) != 0; }

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

// Phase keys and labels. Offline / online totals are measured as their own phases.
struct Row {
  const char* key;
  const char* label;
};
const Row kClientOffline[] = {{"keygen", "keygen (setup of 3 key sets)"},
                              {"encrypt", "encryption (all ciphertexts)"},
                              {"upload_send", "send upload (1 flight)"}};
const Row kServerOffline[] = {{"bsgs_table", "BSGS baby-step table"},
                              {"upload_recv", "receive upload"},
                              {"upload_parse", "deserialize + validate upload"}};
const Row kServerOnline[] = {{"fwd_decrypt", "FEIP fwd decrypt (multi-exp)"},
                             {"fwd_dlog", "dlog fwd (BSGS)"},
                             {"febo_decrypt", "FEBO decrypt"},
                             {"febo_dlog", "dlog FEBO (BSGS)"},
                             {"bwd_decrypt", "FEIP bwd decrypt (multi-exp)"},
                             {"bwd_dlog", "dlog bwd (BSGS)"},
                             {"plain_lr", "plaintext LR (sigmoid, update)"}};

std::string ms(double s) {
  char b[32];
  std::snprintf(b, sizeof b, "%.2f", s * 1e3);
  return b;
}

PhaseStat compute_sum(const PartyReport& r, const Row* rows, size_t n) {
  PhaseStat s;
  for (size_t i = 0; i < n; ++i) s.add(r.get(rows[i].key));
  return s;
}

// The analytic payload sizes (cryptonn_protocol.h).
struct Analytic {
  lrcn::UploadLayout L;
  uint64_t online_s2c = 0, online_c2s = 0;
};
Analytic analytic(const Params& P) {
  Analytic a;
  a.L = lrcn::upload_layout(P);
  a.online_s2c = (uint64_t)P.iters * lrcn::online_s2c_per_iter(P);
  a.online_c2s = (uint64_t)P.iters * lrcn::online_c2s_per_iter(P);
  return a;
}

void print_report(const Options& O, const PartyReport& cl, const PartyReport& sv) {
  const Params& P = O.P;
  const Analytic A = analytic(P);
  const PhaseStat c_off = cl.get("offline"), c_on = cl.get("online"), s_off = sv.get("offline"), s_on = sv.get("online");
  std::printf("\n================= CryptoNN (FEIP + FEBO over NIST P-256): fixed-point LR training (qbits=%d) "
              "=================\n", P.qbits);
  std::printf("config   : %s\n", lrcn::params_string(P).c_str());
  std::printf("protocol : client = data owner + FE authority, server = model holder; semi-honest, trusted authority;\n"
              "           %d thread(s)/party (client %d, server %d); client encryption with the %s; %s\n",
              O.threads, (int)cl.val("threads"), (int)sv.val("threads"),
              cl.val("enc_with_msk") != 0 ? "master secret key (--encrypt-with-msk)" : "public key (as in the paper)",
              OPENSSL_VERSION_TEXT);
  std::printf("BSGS     : 2^%d-entry table (%s), one lookup covers |v| <= %.0f, giant stride %.0f; "
              "worst bounds used: fwd %.0f, FEBO %.0f, bwd %.0f\n",
              (int)sv.val("bsgs_bits"), lrcn::human_bytes((uint64_t)sv.val("bsgs_mem_bytes")).c_str(),
              sv.val("bsgs_half_width"), sv.val("bsgs_stride"), sv.val("dlog_fwd_max_bound"),
              sv.val("dlog_febo_max_bound"), sv.val("dlog_bwd_max_bound"));
  std::printf("dlogs    : fwd %.0f, FEBO %.0f, bwd %.0f solved (largest |v| / bound: %.3f, %.3f, %.3f)\n",
              sv.val("dlog_fwd_calls"), sv.val("dlog_febo_calls"), sv.val("dlog_bwd_calls"),
              sv.val("dlog_fwd_max_ratio"), sv.val("dlog_febo_max_ratio"), sv.val("dlog_bwd_max_ratio"));

  // ---- per-party table ----
  std::printf("\nper-party cost (wall = elapsed at that party; cpu = that process's CPU time, all threads; "
              "sent = payload bytes it sent)\n");
  std::printf("  %-34s | %10s %10s %11s | %10s %10s %11s\n", "", "client", "", "", "server", "", "");
  std::printf("  %-34s | %10s %10s %11s | %10s %10s %11s\n", "stage", "wall ms", "cpu ms", "sent", "wall ms", "cpu ms",
              "sent");
  const std::string rule(2 + 34 + 3 + 33 + 3 + 33, '-');
  std::printf("%s\n", rule.c_str());
  auto cell = [](const PhaseStat* s) {
    char b[64];
    if (!s) std::snprintf(b, sizeof b, "%10s %10s %11s", "-", "-", "-");
    else std::snprintf(b, sizeof b, "%10s %10s %11s", ms(s->wall_s).c_str(), ms(s->cpu_s).c_str(),
                       lrcn::human_bytes(s->sent).c_str());
    return std::string(b);
  };
  auto row = [&](const char* name, const PhaseStat* a, const PhaseStat* b) {
    std::printf("  %-34s | %s | %s\n", name, cell(a).c_str(), cell(b).c_str());
  };
  for (const Row& r : kClientOffline) { PhaseStat s = cl.get(r.key); row(("  " + std::string(r.label)).c_str(), &s, nullptr); }
  for (const Row& r : kServerOffline) { PhaseStat s = sv.get(r.key); row(("  " + std::string(r.label)).c_str(), nullptr, &s); }
  row("OFFLINE total", &c_off, &s_off);
  {
    PhaseStat kd = cl.get("keyderive");
    row("  key derivation (compute only)", &kd, nullptr);
  }
  for (const Row& r : kServerOnline) { PhaseStat s = sv.get(r.key); row(("  " + std::string(r.label)).c_str(), nullptr, &s); }
  {
    PhaseStat cw = c_on, sw = s_on;
    const PhaseStat kd = cl.get("keyderive"), sc = compute_sum(sv, kServerOnline, std::size(kServerOnline));
    cw.wall_s = std::max(0.0, c_on.wall_s - kd.wall_s);
    cw.cpu_s = std::max(0.0, c_on.cpu_s - kd.cpu_s);
    sw.wall_s = std::max(0.0, s_on.wall_s - sc.wall_s);
    sw.cpu_s = std::max(0.0, s_on.cpu_s - sc.cpu_s);
    row("  messages + waiting for the peer", &cw, &sw);
  }
  const std::string online_label = "ONLINE total (" + std::to_string(P.iters) + " iterations)";
  row(online_label.c_str(), &c_on, &s_on);
  PhaseStat c_tot = c_off, s_tot = s_off;
  c_tot.add(c_on);
  s_tot.add(s_on);
  row("TOTAL (offline + online)", &c_tot, &s_tot);
  std::printf("  (client idle between its upload and the first request, i.e. while the server finishes its offline\n"
              "   phase: %s ms; counted in neither phase)\n", ms(cl.get("idle_wait").wall_s).c_str());

  // ---- communication ----
  const uint64_t off_c2s = c_off.sent, off_s2c = s_off.sent, on_c2s = c_on.sent, on_s2c = s_on.sent;
  const uint64_t c2s = off_c2s + on_c2s, s2c = off_s2c + on_s2c;
  std::printf("\n  COMMUNICATION: total %s (%" PRIu64 " bytes); client->server %s, server->client %s\n",
              lrcn::human_bytes(c2s + s2c).c_str(), c2s + s2c, lrcn::human_bytes(c2s).c_str(),
              lrcn::human_bytes(s2c).c_str());
  std::printf("      offline: client->server %s (%" PRIu64 " B), server->client %s\n",
              lrcn::human_bytes(off_c2s).c_str(), off_c2s, lrcn::human_bytes(off_s2c).c_str());
  std::printf("               upload = header %zu B + w0 %zu B + FEIP fwd %s (%d x %zu B) + FEIP bwd %s (%d x %zu B)"
              " + FEBO %s (%d x %zu B)\n",
              A.L.header, A.L.w0, lrcn::human_bytes(A.L.fwd).c_str(), P.iters * P.batch, A.L.fwd_ct,
              lrcn::human_bytes(A.L.bwd).c_str(), P.iters * P.d_data, A.L.bwd_ct, lrcn::human_bytes(A.L.febo).c_str(),
              P.iters * P.batch, A.L.febo_ct);
  std::printf("      online : client->server %s (%" PRIu64 " B), server->client %s (%" PRIu64 " B); per iteration "
              "c->s %zu B, s->c %zu B\n",
              lrcn::human_bytes(on_c2s).c_str(), on_c2s, lrcn::human_bytes(on_s2c).c_str(), on_s2c,
              lrcn::online_c2s_per_iter(P), lrcn::online_s2c_per_iter(P));
  const bool match = off_c2s == A.L.total && off_s2c == 0 && on_c2s == A.online_c2s && on_s2c == A.online_s2c;
  std::printf("      analytic: offline c->s = 32 + 4d + iters*B*(d_data+1)*33 + iters*d_data*(B+1)*33 + iters*B*66 = %zu;"
              " online c->s = iters*(64+33B) = %" PRIu64 ", s->c = iters*4*(d_data+2B) = %" PRIu64 " [%s]\n",
              A.L.total, A.online_c2s, A.online_s2c, match ? "measured == analytic" : "MISMATCH");
  const uint64_t off_fl = c_off.flights_out + s_off.flights_out, on_fl = c_on.flights_out + s_on.flights_out;
  std::printf("  ROUNDS: offline %" PRIu64 " flight (client->server); online %" PRIu64 " flights = %" PRIu64
              " round trips (3 per iteration x %d); total %" PRIu64 " flights\n",
              off_fl, on_fl, s_on.flights_out, P.iters, off_fl + on_fl);
  std::printf("  peak RSS: client %.1f MiB, server %.1f MiB\n", cl.val("peak_rss_mib"), sv.val("peak_rss_mib"));
  if (cl.get("offline").sent + cl.get("online").sent != sv.get("offline").recv + sv.get("online").recv ||
      sv.get("offline").sent + sv.get("online").sent != cl.get("offline").recv + cl.get("online").recv)
    std::printf("  WARNING: byte counters of the two parties disagree\n");

  // ---- leakage ----
  std::printf("\n  LEAKAGE (CryptoNN is semi-honest with a trusted authority; no malicious security, unlike the GC and\n"
              "  MD-ML baselines). The server learns in the clear: the initial weights w0 (sent at setup) and every\n"
              "  w_t; per sample z_i (= <w, x_i> before truncation), yhat_i, the label y_i (FEBO gives label - yhat_i\n"
              "  and yhat_i is the server's own), e_i; per iteration the gradient inner products gfull_j = <e, x_.j>.\n"
              "  The client (authority) learns every w_t, yhat_i and e_i through the key requests. FEIP keys reveal\n"
              "  the function vector (w or e) to the decryptor; the user's FE protocol uses function-hiding IPE instead.\n");
}

void json_phase(std::ostringstream& js, const PhaseStat& s) {
  js << "{\"wall_ms\":" << s.wall_s * 1e3 << ",\"cpu_ms\":" << s.cpu_s * 1e3 << ",\"sent_bytes\":" << s.sent
     << ",\"recv_bytes\":" << s.recv << ",\"flights_out\":" << s.flights_out << ",\"flights_in\":" << s.flights_in
     << "}";
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
  const Params& P = O.P;
  const char* me = O.is_client ? "client" : "server";
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  Metrics M;
  PartyReport rep;
  std::unique_ptr<Channel> ch;
  std::vector<int64_t> trained;
  std::vector<double> iter_wall_s;  // server: wall time of each online iteration
  int rc = 0;
  try {
    std::printf("[%s] %s\n", me, lrcn::params_string(P).c_str());
    (void)lrcn::EcGroup::get();  // group setup outside the timed phases

    if (O.is_client) {
      // ================================================================ CLIENT
      // The client's private inputs (the server never generates them; only --verify,
      // test-only, regenerates them from the seed after the protocol ends).
      lrcn::CnClient C(P, lrcn::generate_client_data(P), lrcn::generate_initial_weights(P), O.threads,
                       O.enc_with_msk);
      // ---- offline: keygen + encryption (before connecting), then one upload flight ----
      Mark m0 = M.now();
      Mark m = m0;
      C.keygen();
      M.record("keygen", m);
      C.encrypt();
      M.record("encrypt", m);
      std::printf("[client] encrypted %s in %.2f ms; connecting to %s:%d\n",
                  lrcn::human_bytes(C.upload().size()).c_str(), M.get("encrypt").wall_s * 1e3, O.host.c_str(), O.port);
      ch = Channel::connect(O.host, O.port, 120.0);
      Mark mc = M.now(ch.get());
      ch->send(C.upload().data(), C.upload().size());
      ch->flush();
      PhaseStat up = M.record("upload_send", mc, ch.get());
      PhaseStat off = lrcn::phase_diff(m0, mc);  // keygen + encrypt + connect + send ...
      off.wall_s = M.get("keygen").wall_s + M.get("encrypt").wall_s + up.wall_s;  // ... minus the connect wait
      off.cpu_s = M.get("keygen").cpu_s + M.get("encrypt").cpu_s + up.cpu_s;
      off.sent = up.sent; off.recv = up.recv; off.flights_out = up.flights_out; off.flights_in = up.flights_in;
      off.flights = up.flights;
      M.add("offline", off);

      // ---- online: answer the server's key requests ----
      // The online clock starts when the first request arrives: until then the client
      // only waits for the server to finish ITS offline phase (receive + validate the
      // upload), which is reported separately ("idle_wait"). Bytes and flights of the
      // online phase are counted from the end of the upload, so they include R1 of t=0.
      const Mark mo_net = M.now(ch.get());
      Mark mo = mo_net;
      uint8_t key[32];
      std::vector<uint8_t> keys((size_t)P.batch * lrcn::febo::kKeyBytes);
      for (int t = 0; t < P.iters; ++t) {
        const std::vector<int64_t> w = ch->recv_i32s((size_t)P.d_data);
        if (t == 0) {
          mo = M.now(ch.get());
          PhaseStat idle = lrcn::phase_diff(mo_net, mo);
          idle.sent = idle.recv = idle.flights = idle.flights_out = idle.flights_in = 0;
          M.add("idle_wait", idle);
        }
        { lrcn::ScopedPhase k(M, "keyderive"); C.fwd_key(w, key); }
        ch->send(key, 32);
        const std::vector<int64_t> yhat = ch->recv_i32s((size_t)P.batch, 0, P.maxq);
        { lrcn::ScopedPhase k(M, "keyderive"); C.eval_keys(yhat, keys.data()); }
        ch->send(keys.data(), keys.size());
        const std::vector<int64_t> e = ch->recv_i32s((size_t)P.batch, -P.maxq, P.maxq);
        { lrcn::ScopedPhase k(M, "keyderive"); C.bwd_key(e, key); }
        ch->send(key, 32);
        if (!O.quiet && (t + 1 == P.iters || (t + 1) % std::max(1, P.iters / 5) == 0))
          std::printf("[client] iteration %d/%d: keys sent\n", t + 1, P.iters);
      }
      ch->flush();
      const Mark mend = M.now(ch.get());
      PhaseStat on = lrcn::phase_diff(mo, mend);
      const PhaseStat on_net = lrcn::phase_diff(mo_net, mend);
      on.sent = on_net.sent; on.recv = on_net.recv; on.flights = on_net.flights;
      on.flights_out = on_net.flights_out; on.flights_in = on_net.flights_in;
      M.add("online", on);
      rep.sc["enc_with_msk"] = O.enc_with_msk ? 1 : 0;
    } else {
      // ================================================================ SERVER
      // ---- offline: input-independent BSGS table, then the upload ----
      const int bits = O.bsgs_bits > 0 ? O.bsgs_bits : lrcn::auto_bsgs_bits(P);
      Mark m = M.now();
      lrcn::Dlog dl(bits, O.threads);
      M.record("bsgs_table", m);
      std::printf("[server] BSGS table 2^%d entries (%s) in %.2f ms; listening on port %d\n", bits,
                  lrcn::human_bytes(dl.memory_bytes()).c_str(), M.get("bsgs_table").wall_s * 1e3, O.port);
      ch = Channel::listen(O.port);
      lrcn::CnServer S(P, dl, O.threads);
      if (O.test_bound_cap >= 0) {
        S.set_test_bound_cap(O.test_bound_cap);
        std::printf("[server] TEST ONLY: every dlog bound is capped at %lld\n", O.test_bound_cap);
      }
      const lrcn::UploadLayout L = lrcn::upload_layout(P);
      Mark mr = M.now(ch.get());
      std::vector<uint8_t> up(L.total);
      ch->recv(up.data(), lrcn::kHeaderBytes);
      lrcn::check_header(up.data(), P);  // the client's public parameters must equal ours
      ch->recv(up.data() + lrcn::kHeaderBytes, L.total - lrcn::kHeaderBytes);
      M.record("upload_recv", mr, ch.get());
      S.parse_upload(up.data(), up.size());
      M.record("upload_parse", mr, ch.get());
      std::vector<uint8_t>().swap(up);
      PhaseStat off;
      for (const Row& r : kServerOffline) off.add(M.get(r.key));
      off.calls = 1;
      M.add("offline", off);

      // ---- online: 3 round trips per iteration ----
      Mark mo = M.now(ch.get());
      uint8_t key[32];
      std::vector<uint8_t> keys((size_t)P.batch * lrcn::febo::kKeyBytes);
      for (int t = 0; t < P.iters; ++t) {
        const double it0 = lrcn::wall_seconds();
        ch->send_i32s(S.fwd_request());
        ch->recv(key, 32);
        S.fwd_finish(key, &M);
        ch->send_i32s(S.yhat());
        ch->recv(keys.data(), keys.size());
        S.eval_finish(keys.data(), &M);
        ch->send_i32s(S.err());
        ch->recv(key, 32);
        S.bwd_finish(key, &M);
        iter_wall_s.push_back(lrcn::wall_seconds() - it0);
        if (!O.quiet && (t + 1 == P.iters || (t + 1) % std::max(1, P.iters / 5) == 0)) {
          std::printf("[server] iteration %d/%d done (%.2f ms); w:", t + 1, P.iters, iter_wall_s.back() * 1e3);
          for (int j = 0; j < P.d && j < 8; ++j) std::printf(" %" PRId64, S.weights()[(size_t)j]);
          std::printf("%s\n", P.d > 8 ? " ..." : "");
        }
      }
      M.record("online", mo, ch.get());
      trained = S.weights();
      rep.sc["bsgs_bits"] = bits;
      rep.sc["bsgs_mem_bytes"] = (double)dl.memory_bytes();
      rep.sc["bsgs_entries"] = (double)dl.table_entries();
      rep.sc["bsgs_half_width"] = (double)dl.half_width();
      rep.sc["bsgs_stride"] = (double)dl.giant_stride();
      const char* kn[3] = {"fwd", "febo", "bwd"};
      for (int k = 0; k < 3; ++k) {
        const lrcn::DlogStats& st = S.stats((lrcn::CnServer::Kind)k);
        rep.sc[std::string("dlog_") + kn[k] + "_calls"] = (double)st.calls;
        rep.sc[std::string("dlog_") + kn[k] + "_max_ratio"] = st.max_ratio;
        rep.sc[std::string("dlog_") + kn[k] + "_max_bound"] = (double)st.max_bound;
        rep.sc[std::string("dlog_") + kn[k] + "_max_abs"] = (double)st.max_abs;
      }
    }
  } catch (const lrcn::DlogError& e) {
    std::fprintf(stderr, "[%s] ERROR: %s\n", me, e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[%s] ERROR: %s\n", me, e.what());
    return 1;
  }

  // ---- exchange reports over the same socket (after all measurements) ----
  for (const auto& [k, s] : M.phases()) rep.ph[k] = s;
  rep.sc["peak_rss_mib"] = (double)lrcn::peak_rss_kib() / 1024.0;
  rep.sc["threads"] = O.threads;
  rep.sc["d"] = P.d; rep.sc["iters"] = P.iters; rep.sc["batch"] = P.batch; rep.sc["qbits"] = P.qbits;
  rep.sc["lr"] = P.lr; rep.sc["bias"] = P.bias ? 1 : 0;
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
  // Cross-check of the public parameters (the server already checked the upload header).
  for (const char* k : {"d", "iters", "batch", "qbits", "lr", "bias"})
    if (peer.val(k, -1) != rep.val(k)) {
      std::fprintf(stderr, "[%s] ERROR: parameter %s differs between the parties (%g here, %g at the peer)\n", me, k,
                   rep.val(k), peer.val(k, -1));
      return 1;
    }
  const PartyReport& cl = O.is_client ? rep : peer;
  const PartyReport& sv = O.is_client ? peer : rep;
  print_report(O, cl, sv);

  // ---- trained weights (server) / verification ----
  std::string verified = "not-checked";
  if (!O.is_client) {
    std::printf("\n[server] trained weights (integers, scale 2^%d):", P.f);
    for (size_t j = 0; j < trained.size() && j < 16; ++j) std::printf(" %" PRId64, trained[j]);
    std::printf("%s\n", trained.size() > 16 ? " ..." : "");
    if (O.verify) {
      const lrcn::Dataset Dv = lrcn::generate_client_data(P);
      const std::vector<int64_t> ref = lrcn::reference_train(P, Dv, lrcn::generate_initial_weights(P));
      const bool ok = ref == trained;
      verified = ok ? "pass" : "FAIL";
      std::printf("[server] verify: %s (CryptoNN result %s the plaintext fixed-point reference)\n",
                  ok ? "PASS" : "FAIL", ok ? "equals" : "DIFFERS FROM");
      if (!ok) {
        rc = 1;
        std::printf("[server] reference weights:");
        for (size_t j = 0; j < ref.size() && j < 16; ++j) std::printf(" %" PRId64, ref[j]);
        std::printf("%s\n", ref.size() > 16 ? " ..." : "");
      }
    }
  } else {
    std::printf("\n[client] done: the server holds the trained weights (it saw every w_t in the clear)\n");
  }

  const Analytic A = analytic(P);
  const PhaseStat c_off = cl.get("offline"), c_on = cl.get("online"), s_off = sv.get("offline"), s_on = sv.get("online");
  if (!O.json_path.empty()) {
    std::ostringstream js;
    js.precision(10);
    js << "{\"party\":\"" << me << "\",\"baseline\":\"cryptonn\",\"params\":{\"d\":" << P.d << ",\"d_data\":"
       << P.d_data << ",\"iters\":" << P.iters << ",\"batch\":" << P.batch << ",\"lr\":" << P.lr
       << ",\"qbits\":" << P.qbits << ",\"f\":" << P.f << ",\"bias\":" << (P.bias ? "true" : "false")
       << ",\"zero_init\":" << (P.zero_init ? "true" : "false") << ",\"seed\":" << P.seed
       << ",\"threads\":" << O.threads << ",\"encrypt_with_msk\":" << (cl.val("enc_with_msk") != 0 ? "true" : "false")
       << ",\"bsgs_bits\":" << sv.val("bsgs_bits") << ",\"group\":\"P-256\"},\"verified\":\"" << verified
       << "\",\"communication\":{\"offline_c2s_bytes\":" << c_off.sent << ",\"offline_s2c_bytes\":" << s_off.sent
       << ",\"online_c2s_bytes\":" << c_on.sent << ",\"online_s2c_bytes\":" << s_on.sent
       << ",\"total_bytes\":" << c_off.sent + s_off.sent + c_on.sent + s_on.sent
       << ",\"upload\":{\"header\":" << A.L.header << ",\"w0\":" << A.L.w0 << ",\"feip_fwd\":" << A.L.fwd
       << ",\"feip_bwd\":" << A.L.bwd << ",\"febo\":" << A.L.febo << "}},\"rounds\":{\"offline_flights\":"
       << c_off.flights_out + s_off.flights_out << ",\"online_flights\":" << c_on.flights_out + s_on.flights_out
       << ",\"online_round_trips\":" << s_on.flights_out << "}";
    if (!O.is_client) {
      js << ",\"server_iteration_wall_ms\":[";
      for (size_t i = 0; i < iter_wall_s.size(); ++i) js << (i ? "," : "") << iter_wall_s[i] * 1e3;
      js << "]";
    }
    js << ",\"client\":";
    json_party(js, cl);
    js << ",\"server\":";
    json_party(js, sv);
    js << "}\n";
    std::ofstream(O.json_path) << js.str();
  }
  if (!O.csv_path.empty() && !O.is_client) {  // both parties hold the same report; the server writes the row
    const std::string header =
        "party,d,iters,batch,qbits,lr,threads,bsgs_bits,encrypt_with_msk,"
        "client_keygen_ms,client_encrypt_ms,client_upload_send_ms,client_offline_ms,client_offline_cpu_ms,"
        "client_keyderive_ms,client_online_ms,client_online_cpu_ms,client_idle_wait_ms,"
        "server_bsgs_table_ms,server_upload_recv_ms,server_upload_parse_ms,server_offline_ms,server_offline_cpu_ms,"
        "server_fwd_decrypt_ms,server_fwd_dlog_ms,server_febo_decrypt_ms,server_febo_dlog_ms,"
        "server_bwd_decrypt_ms,server_bwd_dlog_ms,server_plain_lr_ms,server_online_ms,server_online_cpu_ms,"
        "comm_offline_c2s_bytes,comm_offline_s2c_bytes,comm_online_c2s_bytes,comm_online_s2c_bytes,comm_total_bytes,"
        "offline_flights,online_flights,online_round_trips,client_peak_rss_mib,server_peak_rss_mib,verified";
    std::ifstream probe(O.csv_path);
    std::string first;
    const bool fresh = !probe.good() || !std::getline(probe, first) || first.empty();
    probe.close();
    if (!fresh && first != header) {
      std::fprintf(stderr, "[server] ERROR: %s has a different column layout; row not written. Use a new file.\n",
                   O.csv_path.c_str());
      return rc ? rc : 1;
    }
    std::ofstream csv(O.csv_path, std::ios::app);
    if (fresh) csv << header << "\n";
    auto w = [&](const PartyReport& r, const char* k) { return r.get(k).wall_s * 1e3; };
    char row[4096];
    std::snprintf(row, sizeof row,
                  "server,%d,%d,%d,%d,%g,%d,%d,%d,"
                  "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
                  "%.3f,%.3f,%.3f,%.3f,%.3f,"
                  "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
                  "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
                  "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.1f,%.1f,%s\n",
                  P.d, P.iters, P.batch, P.qbits, P.lr, O.threads, (int)sv.val("bsgs_bits"),
                  (int)cl.val("enc_with_msk"),
                  w(cl, "keygen"), w(cl, "encrypt"), w(cl, "upload_send"), c_off.wall_s * 1e3, c_off.cpu_s * 1e3,
                  w(cl, "keyderive"), c_on.wall_s * 1e3, c_on.cpu_s * 1e3, w(cl, "idle_wait"),
                  w(sv, "bsgs_table"), w(sv, "upload_recv"), w(sv, "upload_parse"), s_off.wall_s * 1e3,
                  s_off.cpu_s * 1e3,
                  w(sv, "fwd_decrypt"), w(sv, "fwd_dlog"), w(sv, "febo_decrypt"), w(sv, "febo_dlog"),
                  w(sv, "bwd_decrypt"), w(sv, "bwd_dlog"), w(sv, "plain_lr"), s_on.wall_s * 1e3, s_on.cpu_s * 1e3,
                  c_off.sent, s_off.sent, c_on.sent, s_on.sent, c_off.sent + s_off.sent + c_on.sent + s_on.sent,
                  c_off.flights_out + s_off.flights_out, c_on.flights_out + s_on.flights_out, s_on.flights_out,
                  cl.val("peak_rss_mib"), sv.val("peak_rss_mib"), verified.c_str());
    csv << row;
  }
  return rc;
}
