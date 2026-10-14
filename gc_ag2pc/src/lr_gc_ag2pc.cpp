// lr_gc_ag2pc -- maliciously secure two-party baseline for float32 logistic-regression
// training using authenticated garbling (emp-ag2pc: WRK17 + KRRW18).
//
//   party 1 = CLIENT = garbler   (emp ALICE): owns every private input -- the training
//                                             data (x, y) and the initial weights --
//                                             and learns nothing
//   party 2 = SERVER = evaluator (emp BOB):   has no private input; evaluates and is the
//                                             only party that learns the trained weights
//
// `iters` mini-batch SGD iterations (batch 4, lr 0.01, binary32, exact sigmoid; see
// lr_reference.h), one protocol chunk per iteration by default. The client's inputs
// enter through the WRK17/KRRW input phase: each input wire's mask lambda is an
// authenticated bit from the OT-extension (COT) session, the server opens its mask
// share to the client only, and the client sends x xor lambda. This needs the patched
// emp-ag2pc in third_party/emp-ag2pc (upstream sends inputs in the clear). The
// weights are opened to the server only.
//
// Reported per party: wall/CPU time and bytes sent/received of every protocol phase
// -- setup, function-dependent preprocessing (authenticated AND triples), input
// processing, garbling (client) / evaluation (server), consistency checks, output
// decoding -- plus peak RSS. Offline = setup + triples (input-independent);
// online = input processing + garbling/evaluation + checks + output.
//
// usage: lr_gc_ag2pc <client|server> [options]       (see --help)

#include "ag2pc_metrics.h"  // must precede every emp-ag2pc header (phase hooks)

#include "emp-ag2pc/emp-ag2pc.h"
#ifndef GCAG2PC_INPUT_PRIVACY_FIX
#error "built against an unpatched emp-ag2pc: its input phase sends private inputs in the clear; use third_party/emp-ag2pc"
#endif

#include "lr_circuit.h"
#include "lr_params.h"
#include "lr_reference.h"

#include "emp-tool/ir/context/count.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using lrgc::Metrics;
using lrgc::Params;
using lrgc::PhaseStat;

using Ctx = emp::AG2PCSession::ctx_t;
using Wire = Ctx::Wire;
using Fl = emp::Float_T<Ctx, 32>;

namespace {

// --------------------------------------------------------------- options
struct Options {
  int party = 0;  // emp::ALICE (client) or emp::BOB (server)
  Params P;
  std::string host = "127.0.0.1";
  int port = 12345;
  int threads = 4;
  int ssp = 40;
  int chunk_iters = 1;
  bool verify = false;
  bool quiet = false;
  std::string json_path, csv_path;
};

void usage(const char* prog) {
  std::printf(
      "usage: %s <client|server> [options]\n"
      "  client = party 1 = garbler (emp ALICE), server = party 2 = evaluator (emp BOB)\n"
      "  (1 / 2, garbler / evaluator and alice / bob are accepted too)\n\n"
      "training (batch = 4, lr = 0.01, float32 are fixed; both parties must pass the same --d / --iters):\n"
      "  --d N            features incl. the bias column, >= 2 (default 5)\n"
      "  --iters N        SGD iterations (default 20)\n"
      "  --seed N         client: seed of its random data / initial weights (default 1);\n"
      "                   server: only used by --verify\n\n"
      "protocol / network:\n"
      "  --host IP        server address the client connects to (default 127.0.0.1)\n"
      "  --port N         TCP port; the server listens on it (default 12345)\n"
      "  --threads N      worker threads per party, >= 2 (default 4)\n"
      "  --ssp N          statistical security parameter (default 40)\n"
      "  --chunk-iters N  iterations per protocol chunk / checkpoint (default 1)\n\n"
      "output:\n"
      "  --verify         TEST ONLY: the server regenerates the client's data and initial\n"
      "                   weights from --seed and checks the trained weights against the\n"
      "                   plaintext float reference (bit-exact). This is outside the\n"
      "                   protocol: a real server never has the client's seed.\n"
      "  --json FILE      write all metrics as JSON\n"
      "  --csv FILE       append one summary row (header written if the file is new)\n"
      "  --quiet          no per-iteration progress lines\n",
      prog);
}

Options parse_options(int argc, char** argv) {
  Options O;
  if (argc < 2) throw std::invalid_argument("missing role");
  const std::string role = argv[1];
  if (role == "client" || role == "1" || role == "garbler" || role == "alice") O.party = emp::ALICE;
  else if (role == "server" || role == "2" || role == "evaluator" || role == "bob") O.party = emp::BOB;
  else throw std::invalid_argument("unknown role '" + role + "' (use client or server)");
  lrgc::ArgReader r{argc, argv, 2};
  for (; r.has(); ++r.i) {
    const std::string a = argv[r.i];
    if (lrgc::parse_param_flag(r, O.P)) continue;
    if (a == "--host") O.host = r.next_value(a);
    else if (a == "--port") O.port = (int)lrgc::parse_int(r.next_value(a), a);
    else if (a == "--threads") O.threads = (int)lrgc::parse_int(r.next_value(a), a);
    else if (a == "--ssp") O.ssp = (int)lrgc::parse_int(r.next_value(a), a);
    else if (a == "--chunk-iters") O.chunk_iters = (int)lrgc::parse_int(r.next_value(a), a);
    else if (a == "--verify") O.verify = true;
    else if (a == "--quiet") O.quiet = true;
    else if (a == "--json") O.json_path = r.next_value(a);
    else if (a == "--csv") O.csv_path = r.next_value(a);
    else throw std::invalid_argument("unknown option " + a);
  }
  if (O.port < 1 || O.port > 65535) throw std::invalid_argument("--port out of range");
  // emp-ag2pc runs a send task and a receive task concurrently on the pool; with a
  // single thread both parties block in send (deadlock), so require >= 2.
  if (O.threads < 2) throw std::invalid_argument("--threads must be >= 2 (emp-ag2pc deadlocks with 1)");
  if (O.ssp < 40) throw std::invalid_argument("--ssp must be >= 40 (smaller values weaken the bucketing bound below 2^-40)");
  if (O.chunk_iters < 1) throw std::invalid_argument("--chunk-iters must be >= 1");
  O.P.finalize();
  return O;
}

// ----------------------------------------------------- phase categories
// emp-ag2pc phase names (engine.h / protocol.h at the pinned commit) -> report rows.
enum Cat { SETUP, PREPROC, INPUTS, GC, CHECKS, OUTPUT, OTHER, LOCAL, NCAT };
const char* const kCatLabel[NCAT] = {
    "one-time setup (base OT + COT session)",
    "fn-dependent preprocessing (auth. ANDs)",
    "input processing (client inputs, KRRW Fig.3)",
    "garbling (client) / evaluation (server)",
    "consistency check (COT check)",
    "output decoding (reveal to server)",
    "other emp-ag2pc phases",
    "local: circuit construction + planning",
};
const char* const kCatKey[NCAT] = {"setup", "preprocessing", "inputs", "garble_eval",
                                   "checks", "output", "other", "local"};

// liveness+load_inputs and gather_outputs are local bookkeeping (no crypto, no
// communication), so they go to LOCAL, which is computed as the untracked remainder.
Cat category_of(const std::string& phase) {
  if (phase == "process_inputs") return INPUTS;
  if (phase == "fused[size+collect_masks]" || phase == "inplace_triples[step5b]") return PREPROC;
  if (phase == "garble_evaluate+c_gamma[step6-10]") return GC;
  if (phase == "cot_check") return CHECKS;
  if (phase == "decode[step14]") return OUTPUT;
  if (phase == "liveness+load_inputs" || phase == "gather_outputs") return LOCAL;
  return OTHER;
}

// What each party reports (and sends to the other at the end).
struct PartyReport {
  PhaseStat cat[NCAT];
  PhaseStat total;
  double peak_rss_mib = 0;
  double num_and = 0;
  std::vector<double> iter_wall_s;

  std::vector<double> serialize() const {
    std::vector<double> v;
    for (int c = 0; c < NCAT; ++c) {
      v.push_back(cat[c].wall_s); v.push_back(cat[c].cpu_s);
      v.push_back((double)cat[c].sent); v.push_back((double)cat[c].recv);
    }
    v.push_back(total.wall_s); v.push_back(total.cpu_s);
    v.push_back((double)total.sent); v.push_back((double)total.recv);
    v.push_back(peak_rss_mib);
    v.push_back(num_and);
    v.insert(v.end(), iter_wall_s.begin(), iter_wall_s.end());
    return v;
  }
  static PartyReport deserialize(const std::vector<double>& v, int iters) {
    PartyReport r;
    size_t k = 0;
    for (int c = 0; c < NCAT; ++c) {
      r.cat[c].wall_s = v[k++]; r.cat[c].cpu_s = v[k++];
      r.cat[c].sent = (uint64_t)v[k++]; r.cat[c].recv = (uint64_t)v[k++];
    }
    r.total.wall_s = v[k++]; r.total.cpu_s = v[k++];
    r.total.sent = (uint64_t)v[k++]; r.total.recv = (uint64_t)v[k++];
    r.peak_rss_mib = v[k++];
    r.num_and = v[k++];
    r.iter_wall_s.assign(v.begin() + (long)k, v.begin() + (long)k + iters);
    return r;
  }
  static size_t serialized_size(int iters) { return (size_t)NCAT * 4 + 4 + 2 + (size_t)iters; }
};

// --------------------------------------- weights <-> fixed-width emp values
// checkpoint(keep...) and reveal(v, ...) take values of compile-time width, while the
// number of weights is a run-time choice. The weights' wires are therefore packed
// into one BitVec_T<ctx_t, N> (N = smallest capacity that fits, padded with a
// public-zero wire). Wire ids are stable across a checkpoint.
template <class F>
void with_capacity(size_t n, F&& f) {
  if (n <= 64) f(std::integral_constant<int, 64>{});
  else if (n <= 256) f(std::integral_constant<int, 256>{});
  else if (n <= 1024) f(std::integral_constant<int, 1024>{});
  else if (n <= 4096) f(std::integral_constant<int, 4096>{});
  else if (n <= 16384) f(std::integral_constant<int, 16384>{});
  else if (n <= 65536) f(std::integral_constant<int, 65536>{});
  else throw std::runtime_error("too many weight bits to keep (> 65536); reduce --d");
}

struct WeightPack {
  std::vector<Wire> secret;  // unique wires of the weights, in pack order
  std::vector<Wire> ids;     // the packed value's wires (secret + padding), size = N
};

WeightPack checkpoint_weights(emp::AG2PCSession& sess, const std::vector<Fl>& w) {
  WeightPack pk;
  std::unordered_set<Wire> seen;
  for (const auto& v : w)
    for (const Wire& b : v.w)
      if (seen.insert(b).second) pk.secret.push_back(b);
  with_capacity(pk.secret.size(), [&](auto cap) {
    constexpr int N = decltype(cap)::value;
    pk.ids = pk.secret;
    if ((int)pk.ids.size() < N) pk.ids.resize(N, sess.ctx().public_bit(false));
    auto bv = std::make_unique<emp::BitVec_T<Ctx, N>>(emp::BitVec_T<Ctx, N>::from_wires(sess.ctx(), pk.ids.data()));
    sess.checkpoint(*bv);
  });
  return pk;
}

// Opens the (checkpointed, hence materialized) weights to `recipient`.
std::optional<std::vector<float>> reveal_weights(emp::AG2PCSession& sess, const WeightPack& pk,
                                                 const std::vector<Fl>& w, int recipient) {
  std::optional<std::vector<uint8_t>> bits;
  with_capacity(pk.ids.size(), [&](auto cap) {
    constexpr int N = decltype(cap)::value;
    auto bv = std::make_unique<emp::BitVec_T<Ctx, N>>(emp::BitVec_T<Ctx, N>::from_wires(sess.ctx(), pk.ids.data()));
    auto out = std::make_unique<std::optional<std::array<bool, N>>>(sess.reveal(*bv, recipient));
    if (out->has_value()) bits.emplace((*out)->begin(), (*out)->end());
  });
  if (!bits) return std::nullopt;
  std::unordered_map<Wire, size_t> pos;
  for (size_t k = 0; k < pk.secret.size(); ++k) pos[pk.secret[k]] = k;
  std::vector<float> vals;
  for (const auto& v : w) {
    uint32_t u = 0;
    for (int i = 0; i < 32; ++i)
      if ((*bits)[pos.at(v.w[(size_t)i])] != 0) u |= uint32_t(1) << i;
    vals.push_back(lrgc::u2f(u));
  }
  return vals;
}

// -------------------------------------------------------------- printing
std::string fmt_ms(double s) {
  char b[32];
  std::snprintf(b, sizeof b, "%.2f", s * 1e3);
  return b;
}
std::string fmt_mib(uint64_t bytes) {
  char b[32];
  std::snprintf(b, sizeof b, "%.3f", (double)bytes / (1024.0 * 1024.0));
  return b;
}
std::string human_bytes(uint64_t bytes) {
  const char* u[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double v = (double)bytes;
  int i = 0;
  while (v >= 1024.0 && i < 4) { v /= 1024.0; ++i; }
  char b[48];
  std::snprintf(b, sizeof b, "%.2f %s", v, u[i]);
  return b;
}

// One party's cost split by protocol stage:
//   offline = setup + function-dependent preprocessing (auth. ANDs): input-independent
//   online  = input processing + garbling (client) / evaluation (server) + consistency
//             check + output decoding
//   local   = gate recording, flush planning, liveness (no crypto, no communication)
// emp-ag2pc runs each chunk's preprocessing right after that chunk's inputs, so
// "offline" means input-independent work, not work done before the inputs exist.
// offline + online + local = total.
struct Stages {
  PhaseStat offline, online, local;
};
Stages stages(const PartyReport& r) {
  Stages g;
  for (int c : {SETUP, PREPROC}) g.offline.add(r.cat[c]);
  for (int c : {INPUTS, GC, CHECKS, OUTPUT, OTHER}) g.online.add(r.cat[c]);
  g.local = r.cat[LOCAL];
  return g;
}

void print_report(const Options& O, const PartyReport& cl, const PartyReport& sv, uint64_t predicted_and,
                  uint64_t in_client_bits, uint64_t in_server_bits, uint64_t out_bits) {
  const Params& P = O.P;
  std::printf("\n================= emp-ag2pc authenticated garbling (WRK17 + KRRW18): float32 LR training =================\n");
  std::printf("config   : %s\n", lrgc::params_string(P).c_str());
  std::printf("protocol : client = party 1 = garbler, server = party 2 = evaluator; ssp=%d; %d thread(s)/party; "
              "%d iteration(s)/chunk\n", O.ssp, O.threads, O.chunk_iters);
  std::printf("circuit  : %.0f AND gates garbled (predicted %" PRIu64 "); private inputs: client %" PRIu64
              " bits, server %" PRIu64 " bits; output: %" PRIu64 " bits to the server only\n",
              cl.num_and, predicted_and, in_client_bits, in_server_bits, out_bits);

  // ---- per-phase table ----
  std::printf("\n%-46s | %11s %11s | %11s %11s | %12s %12s\n", "phase (summed over all chunks)", "client ms",
              "client cpu", "server ms", "server cpu", "c->s MiB", "s->c MiB");
  const std::string rule(46 + 3 + 23 + 3 + 23 + 3 + 25, '-');
  std::printf("%s\n", rule.c_str());
  for (int c = 0; c < NCAT; ++c) {
    if (c == OTHER && cl.cat[c].calls + sv.cat[c].calls == 0 && cl.cat[c].wall_s == 0 && sv.cat[c].wall_s == 0)
      continue;
    std::printf("%-46s | %11s %11s | %11s %11s | %12s %12s\n", kCatLabel[c], fmt_ms(cl.cat[c].wall_s).c_str(),
                fmt_ms(cl.cat[c].cpu_s).c_str(), fmt_ms(sv.cat[c].wall_s).c_str(), fmt_ms(sv.cat[c].cpu_s).c_str(),
                fmt_mib(cl.cat[c].sent).c_str(), fmt_mib(sv.cat[c].sent).c_str());
  }
  std::printf("%s\n", rule.c_str());
  std::printf("%-46s | %11s %11s | %11s %11s | %12s %12s\n", "TOTAL (connected -> output opened)",
              fmt_ms(cl.total.wall_s).c_str(), fmt_ms(cl.total.cpu_s).c_str(), fmt_ms(sv.total.wall_s).c_str(),
              fmt_ms(sv.total.cpu_s).c_str(), fmt_mib(cl.total.sent).c_str(), fmt_mib(sv.total.sent).c_str());

  // ---- per-party cost by stage ----
  const Stages gc = stages(cl), gs = stages(sv);
  std::printf("\nper-party cost (wall = elapsed at that party; cpu = that process's own computation; sent = bytes it sent)\n");
  std::printf("  %-34s | %10s %10s %11s | %10s %10s %11s\n", "", "client", "(garbler)", "", "server", "(evaluator)", "");
  std::printf("  %-34s | %10s %10s %11s | %10s %10s %11s\n", "stage", "wall ms", "cpu ms", "sent", "wall ms", "cpu ms", "sent");
  auto row = [&](const char* name, const PhaseStat& a, const PhaseStat& b) {
    std::printf("  %-34s | %10s %10s %11s | %10s %10s %11s\n", name, fmt_ms(a.wall_s).c_str(), fmt_ms(a.cpu_s).c_str(),
                human_bytes(a.sent).c_str(), fmt_ms(b.wall_s).c_str(), fmt_ms(b.cpu_s).c_str(), human_bytes(b.sent).c_str());
  };
  row("  setup (base OT + COT session)", cl.cat[SETUP], sv.cat[SETUP]);
  row("  authenticated AND triples", cl.cat[PREPROC], sv.cat[PREPROC]);
  row("OFFLINE total", gc.offline, gs.offline);
  row("  input processing", cl.cat[INPUTS], sv.cat[INPUTS]);
  row("  garbling / evaluation", cl.cat[GC], sv.cat[GC]);
  PhaseStat cfin = cl.cat[CHECKS], sfin = sv.cat[CHECKS];
  cfin.add(cl.cat[OUTPUT]); sfin.add(sv.cat[OUTPUT]);
  cfin.add(cl.cat[OTHER]); sfin.add(sv.cat[OTHER]);
  row("  COT check + output decoding", cfin, sfin);
  row("ONLINE total", gc.online, gs.online);
  row("LOCAL (gate recording, planning)", gc.local, gs.local);
  row("TOTAL", cl.total, sv.total);

  const uint64_t c2s = cl.total.sent, s2c = sv.total.sent;
  std::printf("\n  COMMUNICATION: total %s (%" PRIu64 " bytes); client->server %s, server->client %s\n",
              human_bytes(c2s + s2c).c_str(), c2s + s2c, human_bytes(c2s).c_str(), human_bytes(s2c).c_str());
  std::printf("      offline %s, online %s (of which garbled tables client->server %s)\n",
              human_bytes(gc.offline.sent + gs.offline.sent).c_str(), human_bytes(gc.online.sent + gs.online.sent).c_str(),
              human_bytes(cl.cat[GC].sent).c_str());
  std::printf("  Note: in authenticated garbling the authenticated AND triples are generated jointly by both parties\n"
              "  with OT extension, so the offline cost is (by design) about the same at both parties. The client's\n"
              "  extra work is the garbling itself; garbling and evaluation are pipelined over the socket, so their\n"
              "  wall times overlap and the cpu columns show each party's own computation.\n");
  std::printf("  CPU cores kept busy on average (cpu / wall): client %.2f, server %.2f (%d --threads each)\n",
              cl.total.cpu_s / cl.total.wall_s, sv.total.cpu_s / sv.total.wall_s, O.threads);
  if (cl.num_and > 0)
    std::printf("  %.1f bytes per AND gate (all phases); peak RSS: client %.1f MiB, server %.1f MiB\n",
                (double)(c2s + s2c) / cl.num_and, cl.peak_rss_mib, sv.peak_rss_mib);
  if (cl.total.sent != sv.total.recv || sv.total.sent != cl.total.recv)
    std::printf("  WARNING: byte counters disagree (client sent %" PRIu64 " / server recv %" PRIu64 ")\n",
                cl.total.sent, sv.total.recv);
}

void json_phase(std::ostringstream& js, const PhaseStat& s) {
  js << "{\"wall_ms\":" << s.wall_s * 1e3 << ",\"cpu_ms\":" << s.cpu_s * 1e3 << ",\"sent_bytes\":" << s.sent
     << ",\"recv_bytes\":" << s.recv << "}";
}

void json_party(std::ostringstream& js, const PartyReport& r) {
  js << "{\"categories\":{";
  for (int c = 0; c < NCAT; ++c) {
    js << (c ? "," : "") << "\"" << kCatKey[c] << "\":";
    json_phase(js, r.cat[c]);
  }
  js << "},\"total\":";
  json_phase(js, r.total);
  js << ",\"peak_rss_mib\":" << r.peak_rss_mib << ",\"iteration_wall_ms\":[";
  for (size_t i = 0; i < r.iter_wall_s.size(); ++i) js << (i ? "," : "") << r.iter_wall_s[i] * 1e3;
  js << "]}";
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
  // emp's test mode makes every default-seeded PRG deterministic (Delta, labels, OT
  // choices): a benchmark run in it is not secure. Refuse rather than measure it.
  if (emp::is_test_mode()) {
    std::fprintf(stderr, "error: EMP_TEST_MODE is set; it makes all randomness deterministic. Unset it.\n");
    return 2;
  }
  const Params& P = O.P;
  const bool is_client = O.party == emp::ALICE;
  const char* me = is_client ? "client" : "server";
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  // The client's private inputs (the server has none). The server never generates them;
  // only --verify (test-only) regenerates them from the seed after the protocol ends.
  lrgc::Dataset D;
  std::vector<float> w0;
  if (is_client) {
    D = lrgc::generate_client_data(P);
    w0 = lrgc::generate_initial_weights(P);
  }

  // Circuit size, from a dry run over emp::CountCtx on all-zero data of the right
  // shape (the count is data-independent; the server never sees the client's data).
  uint64_t predicted_and = 0;
  {
    lrgc::Dataset Z;
    Z.iters = P.iters;
    Z.d = P.d;
    Z.x.assign((size_t)P.iters * lrgc::kBatch * P.d, 0.0f);
    Z.y.assign((size_t)P.iters * lrgc::kBatch, 0);
    emp::CountCtx cc;
    (void)lrgc::train_offline(cc, P, Z, std::vector<float>((size_t)P.d, 0.0f));
    predicted_and = cc.ands;
  }
  const uint64_t in_client_bits = (uint64_t)P.d * 32 + (uint64_t)P.iters * lrgc::kBatch * ((uint64_t)P.d_data() * 32 + 1);
  const uint64_t in_server_bits = 0;
  std::printf("[%s] %s\n[%s] circuit: %" PRIu64 " AND gates; %s\n", me, lrgc::params_string(P).c_str(), me,
              predicted_and,
              is_client ? ("connecting to " + O.host + ":" + std::to_string(O.port)).c_str()
                        : ("listening on port " + std::to_string(O.port)).c_str());

  // Network: the server listens, the client connects (TCP role is independent of
  // the protocol role; the session opens a second socket on the same port).
  std::unique_ptr<emp::NetIO> io = is_client ? emp::NetIO::connect(O.host.c_str(), O.port, /*quiet=*/true)
                                             : emp::NetIO::listen(O.port, /*quiet=*/true);
  ::ThreadPool pool(O.threads);  // emp-tool/third_party/ThreadPool.h (global namespace)
  Metrics& M = Metrics::get();
  PartyReport rep;

  // ---- one-time setup: sibling socket + base OTs + COT session (TriplePool) ----
  const Metrics::Mark t_start = M.now();
  const uint64_t s0 = io->send_counter, r0 = io->recv_counter;
  auto sess = std::make_unique<emp::AG2PCSession>(io.get(), &pool, O.party, O.ssp);
  ::AG2PCProtocol& proto = sess->protocol();  // declared outside namespace emp at the pinned commit
  M.attach({&proto.io->send_counter, &proto.sib->send_counter}, {&proto.io->recv_counter, &proto.sib->recv_counter});
  M.reset();  // the setup row below covers the whole constructor
  const Metrics::Mark t_setup = M.now();
  rep.cat[SETUP] = Metrics::diff(t_start, t_setup);
  rep.cat[SETUP].sent = t_setup.sent - s0;  // the sibling socket started at 0
  rep.cat[SETUP].recv = t_setup.recv - r0;

  // ---- training ----
  // Iteration t's chunk: the client inputs batch t (and, at t = 0, the initial weights),
  // the circuit for one SGD step is recorded, and the weights are checkpointed.
  std::vector<Fl> w;
  WeightPack pack;
  for (int t = 0; t < P.iters; ++t) {
    const Metrics::Mark it0 = M.now();
    auto batch = sess->input_batch();
    if (t == 0)
      for (int j = 0; j < P.d; ++j) w.push_back(batch.add<Fl>(emp::ALICE, is_client ? w0[(size_t)j] : 0.0f));
    std::vector<std::vector<Fl>> x((size_t)lrgc::kBatch);
    std::vector<emp::Bit_T<Ctx>> yb;
    for (int i = 0; i < lrgc::kBatch; ++i) {
      for (int j = 0; j < P.d_data(); ++j) x[(size_t)i].push_back(batch.add<Fl>(emp::ALICE, is_client ? D.X(t, i, j) : 0.0f));
      yb.push_back(batch.add<emp::Bit_T<Ctx>>(emp::ALICE, is_client && D.Y(t, i) != 0));
    }
    batch.finish();

    lrgc::FloatLR<Ctx> lr(sess->ctx(), P.d);
    std::vector<Fl> y;
    for (const auto& b : yb) y.push_back(lr.label(b.w));
    w = lr.step(w, x, y);
    if ((t + 1) % O.chunk_iters == 0 || t + 1 == P.iters) pack = checkpoint_weights(*sess, w);

    const double it_s = Metrics::diff(it0, M.now()).wall_s;
    rep.iter_wall_s.push_back(it_s);
    if (!O.quiet) std::printf("[%s] iteration %2d/%d done (%.2f ms)\n", me, t + 1, P.iters, it_s * 1e3);
  }

  // ---- output: the trained weights are opened to the server only ----
  const std::optional<std::vector<float>> trained = reveal_weights(*sess, pack, w, emp::BOB);
  const Metrics::Mark t_end = M.now();
  rep.total = Metrics::diff(t_start, t_end);
  rep.total.sent = t_end.sent - s0;
  rep.total.recv = t_end.recv - r0;
  rep.peak_rss_mib = (double)Metrics::peak_rss_kib() / 1024.0;
  rep.num_and = (double)sess->num_and();

  // Sort emp-ag2pc's phases into report categories; the remainder of the total is
  // local work (gate recording, flush planning, carried-state bookkeeping).
  const auto phases = M.phases();
  if (phases.empty())
    std::fprintf(stderr, "[%s] WARNING: emp-ag2pc phase hooks did not fire -- was ag2pc_metrics.h included first?\n", me);
  for (const auto& [name, st] : phases) {
    const Cat c = category_of(name);
    if (c == OTHER) std::fprintf(stderr, "[%s] note: unrecognized emp-ag2pc phase '%s'\n", me, name.c_str());
    rep.cat[c].add(st);
  }
  PhaseStat tracked;  // LOCAL = everything not in a crypto phase (incl. the local phases above)
  for (int c = 0; c < LOCAL; ++c) tracked.add(rep.cat[c]);
  rep.cat[LOCAL].wall_s = std::max(0.0, rep.total.wall_s - tracked.wall_s);
  rep.cat[LOCAL].cpu_s = std::max(0.0, rep.total.cpu_s - tracked.cpu_s);
  rep.cat[LOCAL].sent = rep.total.sent - std::min(rep.total.sent, tracked.sent);
  rep.cat[LOCAL].recv = rep.total.recv - std::min(rep.total.recv, tracked.recv);

  // ---- exchange reports over the primary socket (after all measurements) ----
  const size_t n = PartyReport::serialized_size(P.iters);
  std::vector<double> mine = rep.serialize(), theirs(n);
  if (mine.size() != n) { std::fprintf(stderr, "internal error: report size\n"); return 1; }
  if (is_client) {
    io->send_data(mine.data(), (int64_t)(n * sizeof(double)));
    io->flush();
    io->recv_data(theirs.data(), (int64_t)(n * sizeof(double)));
  } else {
    io->recv_data(theirs.data(), (int64_t)(n * sizeof(double)));
    io->send_data(mine.data(), (int64_t)(n * sizeof(double)));
    io->flush();
  }
  const PartyReport peer = PartyReport::deserialize(theirs, P.iters);
  const PartyReport& cl = is_client ? rep : peer;
  const PartyReport& sv = is_client ? peer : rep;
  const uint64_t out_bits = (uint64_t)P.d * 32;

  print_report(O, cl, sv, predicted_and, in_client_bits, in_server_bits, out_bits);

  // ---- trained weights (server) / verification ----
  int rc = 0;
  std::string verified = "not-checked";
  if (is_client) {
    std::printf("\n[client] output hidden from the garbler: %s\n", trained ? "NO (unexpected!)" : "yes");
    if (trained) rc = 1;
  } else {
    if (!trained) { std::fprintf(stderr, "[server] ERROR: no output at the server\n"); return 1; }
    std::printf("\n[server] trained weights (float32):");
    for (size_t j = 0; j < trained->size() && j < 16; ++j) std::printf(" %.9g", (double)(*trained)[j]);
    std::printf("%s\n", trained->size() > 16 ? " ..." : "");
    if (O.verify) {
      const std::vector<float> w0_ref = lrgc::generate_initial_weights(P);
      const std::vector<float> ref = lrgc::reference_train(P, lrgc::generate_client_data(P), w0_ref);
      bool ok = ref.size() == trained->size();
      for (size_t j = 0; ok && j < ref.size(); ++j) ok = lrgc::f2u(ref[j]) == lrgc::f2u((*trained)[j]);
      verified = ok ? "pass" : "FAIL";
      std::printf("[server] --verify: %s (secure result %s the plaintext float reference, bit for bit)\n",
                  ok ? "PASS" : "FAIL", ok ? "equals" : "DIFFERS FROM");
      if (!ok) rc = 1;
      const std::vector<double> wd = lrgc::reference_train_double(P, lrgc::generate_client_data(P),
                                                                 std::vector<double>(w0_ref.begin(), w0_ref.end()));
      double worst = 0;
      for (size_t j = 0; j < wd.size(); ++j) worst = std::max(worst, std::fabs((double)(*trained)[j] - wd[j]));
      std::printf("[server] max |w_float32 - w_float64 (true-exp) reference| = %.3e\n", worst);
    }
  }

  // Per-phase detail of this party (emp-ag2pc phase names, incl. triple sub-steps).
  if (!O.quiet) {
    std::printf("\n[%s] emp-ag2pc phases at this party (wall ms / cpu ms / sent / recv / calls):\n", me);
    for (const auto& [name, st] : phases)
      std::printf("    %-36s %10s %10s %12s %12s %6" PRIu64 "\n", name.c_str(), fmt_ms(st.wall_s).c_str(),
                  fmt_ms(st.cpu_s).c_str(), human_bytes(st.sent).c_str(), human_bytes(st.recv).c_str(), st.calls);
    for (const auto& [name, st] : M.subphases())
      std::printf("      triples: %-27s %10s %10s %12s %12s %6" PRIu64 "\n", name.c_str(), fmt_ms(st.wall_s).c_str(),
                  fmt_ms(st.cpu_s).c_str(), human_bytes(st.sent).c_str(), human_bytes(st.recv).c_str(), st.calls);
  }

  if (!O.json_path.empty()) {
    std::ostringstream js;
    js.precision(10);
    js << "{\"party\":\"" << me << "\",\"params\":{\"d\":" << P.d << ",\"iters\":" << P.iters << ",\"batch\":"
       << lrgc::kBatch << ",\"lr\":" << lrgc::kLr << ",\"number_format\":\"float32\""
       << ",\"seed\":" << P.seed << ",\"threads\":" << O.threads << ",\"ssp\":" << O.ssp
       << ",\"chunk_iters\":" << O.chunk_iters << "},\"and_gates\":" << cl.num_and
       << ",\"predicted_and_gates\":" << predicted_and << ",\"input_bits\":{\"client\":" << in_client_bits
       << ",\"server\":" << in_server_bits << "},\"output_bits\":" << out_bits << ",\"verified\":\"" << verified
       << "\",\"client\":";
    json_party(js, cl);
    js << ",\"server\":";
    json_party(js, sv);
    js << ",\"this_party_phases\":{";
    bool first = true;
    for (const auto& [name, st] : phases) {
      js << (first ? "" : ",") << "\"" << name << "\":";
      json_phase(js, st);
      first = false;
    }
    for (const auto& [name, st] : M.subphases()) {
      js << (first ? "" : ",") << "\"triples/" << name << "\":";
      json_phase(js, st);
      first = false;
    }
    js << "}}\n";
    std::ofstream(O.json_path) << js.str();
  }
  if (!O.csv_path.empty()) {
    const std::string header =
        "party,d,iters,threads,ssp,chunk_iters,and_gates,"
        "client_offline_ms,client_online_ms,client_local_ms,client_total_ms,"
        "client_offline_cpu_ms,client_online_cpu_ms,client_local_cpu_ms,client_total_cpu_ms,"
        "server_offline_ms,server_online_ms,server_local_ms,server_total_ms,"
        "server_offline_cpu_ms,server_online_cpu_ms,server_local_cpu_ms,server_total_cpu_ms,"
        "client_garble_ms,client_garble_cpu_ms,server_eval_ms,server_eval_cpu_ms,"
        "comm_client_to_server_bytes,comm_server_to_client_bytes,comm_total_bytes,"
        "comm_offline_bytes,comm_online_bytes,client_peak_rss_mib,server_peak_rss_mib,verified";
    std::ifstream probe(O.csv_path);
    std::string first;
    const bool fresh = !probe.good() || !std::getline(probe, first) || first.empty();
    probe.close();
    if (!fresh && first != header) {
      std::fprintf(stderr, "[%s] ERROR: %s has a different column layout (written by an older version); "
                           "row not written. Use a new file.\n", me, O.csv_path.c_str());
      return rc ? rc : 1;
    }
    if (!is_client) {  // both parties hold the same combined report; only the server writes the CSV
    std::ofstream csv(O.csv_path, std::ios::app);
    if (fresh) csv << header << "\n";
    const Stages gc = stages(cl), gs = stages(sv);
    char row[2048];
    std::snprintf(row, sizeof row,
                  "%s,%d,%d,%d,%d,%d,%.0f,"
                  "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
                  "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
                  "%.3f,%.3f,%.3f,%.3f,"
                  "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.1f,%.1f,%s\n",
                  me, P.d, P.iters, O.threads, O.ssp, O.chunk_iters, cl.num_and,
                  gc.offline.wall_s * 1e3, gc.online.wall_s * 1e3, gc.local.wall_s * 1e3, cl.total.wall_s * 1e3,
                  gc.offline.cpu_s * 1e3, gc.online.cpu_s * 1e3, gc.local.cpu_s * 1e3, cl.total.cpu_s * 1e3,
                  gs.offline.wall_s * 1e3, gs.online.wall_s * 1e3, gs.local.wall_s * 1e3, sv.total.wall_s * 1e3,
                  gs.offline.cpu_s * 1e3, gs.online.cpu_s * 1e3, gs.local.cpu_s * 1e3, sv.total.cpu_s * 1e3,
                  cl.cat[GC].wall_s * 1e3, cl.cat[GC].cpu_s * 1e3, sv.cat[GC].wall_s * 1e3, sv.cat[GC].cpu_s * 1e3,
                  cl.total.sent, sv.total.sent, cl.total.sent + sv.total.sent, gc.offline.sent + gs.offline.sent,
                  gc.online.sent + gs.online.sent, cl.peak_rss_mib, sv.peak_rss_mib, verified.c_str());
    csv << row;
    }
  }

  sess.reset();
  return rc;
}
