// lr_gc_ag2pc -- maliciously secure two-party baseline for quantized logistic-
// regression training using authenticated garbling (emp-ag2pc: WRK17 + KRRW18).
//
//   party 1 = CLIENT = garbler   (emp ALICE): holds the training data (x, y)
//   party 2 = SERVER = evaluator (emp BOB):   holds the initial weights and is the
//                                             only party that learns the trained weights
//
// Training: `iters` mini-batch SGD iterations of `batch` datapoints each, `d`
// features (incl. the bias column), fixed-point semantics of lr_reference.h.
// Each iteration authenticates its inputs, records its gates, and checkpoints the
// weights (one protocol chunk per iteration by default), which bounds memory.
//
// Reported per party: wall/CPU time and bytes sent/received of every protocol
// phase -- setup, input authentication, function-dependent preprocessing
// (authenticated AND triples), garbling (client) / evaluation (server),
// consistency checks, output decoding -- plus peak RSS.
//
// usage: lr_gc_ag2pc <client|server> [options]       (see --help)

#include "ag2pc_metrics.h"  // must precede every emp-ag2pc header (phase hooks)

#include "emp-ag2pc/emp-ag2pc.h"

#include "lr_circuit.h"
#include "lr_params.h"
#include "lr_reference.h"
#include "sbit.h"

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

using lrgc::Builder;
using lrgc::Metrics;
using lrgc::Params;
using lrgc::PhaseStat;
using lrgc::SBit;
using lrgc::SInt;

using Ctx = emp::AG2PCSession::ctx_t;
using Wire = Ctx::Wire;

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
      "training (both parties must pass the same values):\n"
      "  --d N            features incl. the bias column (default 5)\n"
      "  --iters N        SGD iterations (default 20)\n"
      "  --batch N        datapoints per iteration (default 4)\n"
      "  --qbits N        quantization bits, 2..12 (default 6)\n"
      "  --lr X           learning rate; batch/lr must be a power of two (default 0.5)\n"
      "  --no-bias        the last feature is a regular client feature, not the bias\n"
      "  --zero-init      server's initial weights are 0 (default: uniform random)\n"
      "  --seed N         data seed (default 1)\n\n"
      "protocol / network:\n"
      "  --host IP        server address the client connects to (default 127.0.0.1)\n"
      "  --port N         TCP port; the server listens on it (default 12345)\n"
      "  --threads N      worker threads per party, >= 2 (default 4)\n"
      "  --ssp N          statistical security parameter (default 40)\n"
      "  --chunk-iters N  iterations per protocol chunk / checkpoint (default 1)\n\n"
      "output:\n"
      "  --verify         TEST ONLY: the server regenerates the client's data from --seed\n"
      "                   and checks the trained weights against the plaintext reference\n"
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
  if (O.ssp < 1) throw std::invalid_argument("--ssp must be >= 1");
  if (O.chunk_iters < 1) throw std::invalid_argument("--chunk-iters must be >= 1");
  O.P.finalize();
  return O;
}

// ----------------------------------------------------- phase categories
// emp-ag2pc phase names (engine.h / protocol.h at the pinned commit) -> report rows.
enum Cat { SETUP, INPUTS, PREPROC, GC, CHECKS, OUTPUT, OTHER, LOCAL, NCAT };
const char* const kCatLabel[NCAT] = {
    "one-time setup (base OT + COT session)",
    "input authentication (KRRW Fig.3)",
    "fn-dependent preprocessing (auth. ANDs)",
    "garbling (client) / evaluation (server)",
    "consistency checks (COT check, gather)",
    "output decoding (reveal to server)",
    "other emp-ag2pc phases",
    "local: circuit construction + planning",
};
const char* const kCatKey[NCAT] = {"setup", "inputs", "preprocessing", "garble_eval",
                                   "checks", "output", "other", "local"};

Cat category_of(const std::string& phase) {
  if (phase == "process_inputs") return INPUTS;
  if (phase == "liveness+load_inputs" || phase == "fused[size+collect_masks]" ||
      phase == "inplace_triples[step5b]")
    return PREPROC;
  if (phase == "garble_evaluate+c_gamma[step6-10]") return GC;
  if (phase == "cot_check" || phase == "gather_outputs") return CHECKS;
  if (phase == "decode[step14]") return OUTPUT;
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
// checkpoint(keep...) and reveal(v, ...) take values of compile-time width, while
// the weights' widths are known only at run time. The weights' secret wires are
// therefore packed into one BitVec_T<ctx_t, N> (N = smallest capacity that fits,
// padded with a public-zero wire). Wire ids are stable across a checkpoint.
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
  std::vector<Wire> secret;  // unique secret wires of the weights, in pack order
  std::vector<Wire> ids;     // the packed value's wires (secret + padding), size = N
};

WeightPack checkpoint_weights(emp::AG2PCSession& sess, Builder<Ctx>& B, const std::vector<SInt<Ctx>>& w) {
  WeightPack pk;
  std::unordered_set<Wire> seen;
  for (const auto& v : w)
    for (const auto& b : v.b)
      if (!b.is_const() && seen.insert(b.w).second) pk.secret.push_back(b.w);
  if (pk.secret.empty()) throw std::runtime_error("weights are public constants (nothing to keep)");
  with_capacity(pk.secret.size(), [&](auto cap) {
    constexpr int N = decltype(cap)::value;
    pk.ids = pk.secret;
    if ((int)pk.ids.size() < N) pk.ids.resize(N, B.materialize(Builder<Ctx>::cst(false)));
    auto bv = std::make_unique<emp::BitVec_T<Ctx, N>>(emp::BitVec_T<Ctx, N>::from_wires(sess.ctx(), pk.ids.data()));
    sess.checkpoint(*bv);
  });
  return pk;
}

// Opens the (checkpointed, hence materialized) weights to `recipient`.
std::optional<std::vector<int64_t>> reveal_weights(emp::AG2PCSession& sess, const WeightPack& pk,
                                                   const std::vector<SInt<Ctx>>& w, int recipient) {
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
  std::vector<int64_t> vals;
  for (const auto& v : w) {
    uint64_t u = 0;
    const int n = v.width();
    for (int i = 0; i < n; ++i) {
      const auto& b = v.b[(size_t)i];
      const bool bit = b.is_const() ? b.k == 1 : (*bits)[pos.at(b.w)] != 0;
      if (bit) u |= uint64_t(1) << i;
    }
    if (n < 64 && ((u >> (n - 1)) & 1)) u |= ~((uint64_t(1) << n) - 1);
    vals.push_back((int64_t)u);
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

void print_report(const Options& O, const PartyReport& cl, const PartyReport& sv, uint64_t predicted_and,
                  uint64_t in_client_bits, uint64_t in_server_bits, uint64_t out_bits) {
  const Params& P = O.P;
  std::printf("\n================= emp-ag2pc authenticated garbling (WRK17 + KRRW18): LR training =================\n");
  std::printf("config   : %s\n", lrgc::params_string(P).c_str());
  std::printf("protocol : client = party 1 = garbler, server = party 2 = evaluator; ssp=%d; %d thread(s)/party; "
              "%d iteration(s)/chunk\n", O.ssp, O.threads, O.chunk_iters);
  std::printf("circuit  : %.0f AND gates garbled (predicted %" PRIu64 "); inputs: client %" PRIu64
              " bits, server %" PRIu64 " bits; output: %" PRIu64 " bits to the server only\n",
              cl.num_and, predicted_and, in_client_bits, in_server_bits, out_bits);
  std::printf("\n%-41s | %11s %11s | %11s %11s | %12s %12s\n", "phase (summed over all chunks)", "client ms",
              "client cpu", "server ms", "server cpu", "c->s MiB", "s->c MiB");
  std::printf("%s\n", std::string(41 + 3 + 23 + 3 + 23 + 3 + 25, '-').c_str());
  for (int c = 0; c < NCAT; ++c) {
    if (c == OTHER && cl.cat[c].calls + sv.cat[c].calls == 0 && cl.cat[c].wall_s == 0 && sv.cat[c].wall_s == 0)
      continue;
    std::printf("%-41s | %11s %11s | %11s %11s | %12s %12s\n", kCatLabel[c], fmt_ms(cl.cat[c].wall_s).c_str(),
                fmt_ms(cl.cat[c].cpu_s).c_str(), fmt_ms(sv.cat[c].wall_s).c_str(), fmt_ms(sv.cat[c].cpu_s).c_str(),
                fmt_mib(cl.cat[c].sent).c_str(), fmt_mib(sv.cat[c].sent).c_str());
  }
  std::printf("%s\n", std::string(41 + 3 + 23 + 3 + 23 + 3 + 25, '-').c_str());
  std::printf("%-41s | %11s %11s | %11s %11s | %12s %12s\n", "TOTAL (connected -> output opened)",
              fmt_ms(cl.total.wall_s).c_str(), fmt_ms(cl.total.cpu_s).c_str(), fmt_ms(sv.total.wall_s).c_str(),
              fmt_ms(sv.total.cpu_s).c_str(), fmt_mib(cl.total.sent).c_str(), fmt_mib(sv.total.sent).c_str());

  const double cl_pre = cl.cat[SETUP].wall_s + cl.cat[PREPROC].wall_s + cl.cat[GC].wall_s;
  const double cl_pre_cpu = cl.cat[SETUP].cpu_s + cl.cat[PREPROC].cpu_s + cl.cat[GC].cpu_s;
  const double sv_online = sv.cat[INPUTS].wall_s + sv.cat[GC].wall_s + sv.cat[CHECKS].wall_s + sv.cat[OUTPUT].wall_s;
  const uint64_t c2s = cl.total.sent, s2c = sv.total.sent;
  std::printf("\nheadline metrics\n");
  std::printf("  client garbling time (garble pass + c_gamma) .......... %10s ms   (CPU %s ms)\n",
              fmt_ms(cl.cat[GC].wall_s).c_str(), fmt_ms(cl.cat[GC].cpu_s).c_str());
  std::printf("  client preprocessing total (setup + auth. ANDs + garbling) %7s ms   (CPU %s ms)\n",
              fmt_ms(cl_pre).c_str(), fmt_ms(cl_pre_cpu).c_str());
  std::printf("  server evaluation time (evaluate pass + c_gamma) ...... %10s ms   (CPU %s ms)\n",
              fmt_ms(sv.cat[GC].wall_s).c_str(), fmt_ms(sv.cat[GC].cpu_s).c_str());
  std::printf("  server online total (inputs + evaluation + checks + output) %6s ms\n", fmt_ms(sv_online).c_str());
  std::printf("  server preprocessing (setup + auth. ANDs; interactive) .. %8s ms\n",
              fmt_ms(sv.cat[SETUP].wall_s + sv.cat[PREPROC].wall_s).c_str());
  std::printf("  communication: client->server %s, server->client %s, total %s (%" PRIu64 " bytes)\n",
              human_bytes(c2s).c_str(), human_bytes(s2c).c_str(), human_bytes(c2s + s2c).c_str(), c2s + s2c);
  if (cl.num_and > 0)
    std::printf("                 %.1f bytes per AND gate (all phases)\n", (double)(c2s + s2c) / cl.num_and);
  std::printf("  peak RSS: client %.1f MiB, server %.1f MiB\n", cl.peak_rss_mib, sv.peak_rss_mib);
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
  const Params& P = O.P;
  const bool is_client = O.party == emp::ALICE;
  const char* me = is_client ? "client" : "server";
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  const lrgc::SigmoidTable S = lrgc::make_sigmoid_table(P.f);

  // Each party's private input. (The generator is shared only so that --verify can
  // recompute the expected result; in a real deployment the server never has it.)
  lrgc::Dataset D;
  std::vector<int64_t> w0;
  if (is_client) D = lrgc::generate_client_data(P);
  else w0 = lrgc::generate_server_weights(P);

  // Circuit size, from a dry run over emp::CountCtx (data-independent).
  uint64_t predicted_and = 0;
  {
    lrgc::Params Pz = P;
    emp::CountCtx cc;
    Builder<emp::CountCtx> Bc(cc);
    const lrgc::Dataset Dz = lrgc::generate_client_data(Pz);
    (void)lrgc::train_offline(Bc, Pz, S, Dz, std::vector<int64_t>((size_t)P.d, 0));
    predicted_and = cc.ands;
  }
  const uint64_t in_client_bits = (uint64_t)P.iters * P.batch * ((uint64_t)P.d_data * P.qbits + 1);
  const uint64_t in_server_bits = (uint64_t)P.d * P.qbits;
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
  Builder<Ctx> B(sess->ctx());
  lrgc::LRCircuit<Ctx> lr(B, P, S);
  using BitV = emp::Bit_T<Ctx>;
  std::vector<SInt<Ctx>> w;
  WeightPack pack;
  for (int t = 0; t < P.iters; ++t) {
    const Metrics::Mark it0 = M.now();
    auto batch = sess->input_batch();
    std::vector<std::vector<Wire>> w0w;
    if (t == 0) {
      w0w.resize((size_t)P.d);
      for (int j = 0; j < P.d; ++j) {
        const std::vector<bool> bits = lrgc::to_bits(is_client ? 0 : w0[(size_t)j], P.qbits);
        for (int k = 0; k < P.qbits; ++k) w0w[(size_t)j].push_back(batch.add<BitV>(emp::BOB, (bool)bits[(size_t)k]).w);
      }
    }
    std::vector<std::vector<std::vector<Wire>>> xw((size_t)P.batch, std::vector<std::vector<Wire>>((size_t)P.d_data));
    std::vector<Wire> yw((size_t)P.batch);
    for (int i = 0; i < P.batch; ++i) {
      for (int j = 0; j < P.d_data; ++j) {
        const std::vector<bool> bits = lrgc::to_bits(is_client ? D.X(t, i, j) : 0, P.qbits);
        for (int k = 0; k < P.qbits; ++k)
          xw[(size_t)i][(size_t)j].push_back(batch.add<BitV>(emp::ALICE, (bool)bits[(size_t)k]).w);
      }
      yw[(size_t)i] = batch.add<BitV>(emp::ALICE, is_client && D.Y(t, i) != 0).w;
    }
    batch.finish();

    if (t == 0)
      for (int j = 0; j < P.d; ++j) w.push_back(lr.quantized(w0w[(size_t)j]));
    std::vector<std::vector<SInt<Ctx>>> x;
    std::vector<SBit<Ctx>> y;
    for (int i = 0; i < P.batch; ++i) {
      x.push_back(lr.feature_row(xw[(size_t)i]));
      y.push_back(Builder<Ctx>::wire(yw[(size_t)i]));
    }
    w = lr.step(w, x, y);
    if ((t + 1) % O.chunk_iters == 0 || t + 1 == P.iters) pack = checkpoint_weights(*sess, B, w);

    const double it_s = Metrics::diff(it0, M.now()).wall_s;
    rep.iter_wall_s.push_back(it_s);
    if (!O.quiet)
      std::printf("[%s] iteration %2d/%d done (%.2f ms; weights now %d..%d bits)\n", me, t + 1, P.iters, it_s * 1e3,
                  std::min_element(w.begin(), w.end(), [](auto& a, auto& b) { return a.width() < b.width(); })->width(),
                  std::max_element(w.begin(), w.end(), [](auto& a, auto& b) { return a.width() < b.width(); })->width());
  }

  // ---- output: the trained weights are opened to the server only ----
  const std::optional<std::vector<int64_t>> trained = reveal_weights(*sess, pack, w, emp::BOB);
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
  PhaseStat tracked;
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
  const uint64_t out_bits = [&] { uint64_t s = 0; for (const auto& v : w) s += (uint64_t)v.width(); return s; }();

  print_report(O, cl, sv, predicted_and, in_client_bits, in_server_bits, out_bits);

  // ---- trained weights (server) / verification ----
  int rc = 0;
  std::string verified = "not-checked";
  if (is_client) {
    std::printf("\n[client] output hidden from the garbler: %s\n", trained ? "NO (unexpected!)" : "yes");
    if (trained) rc = 1;
  } else {
    if (!trained) { std::fprintf(stderr, "[server] ERROR: no output at the server\n"); return 1; }
    std::printf("\n[server] trained weights (integers, scale 2^%d):", P.f);
    for (size_t j = 0; j < trained->size() && j < 16; ++j) std::printf(" %" PRId64, (*trained)[j]);
    std::printf("%s\n", trained->size() > 16 ? " ..." : "");
    if (O.verify) {
      const std::vector<int64_t> ref =
          lrgc::reference_train(P, lrgc::generate_client_data(P), lrgc::generate_server_weights(P));
      const bool ok = ref == *trained;
      verified = ok ? "pass" : "FAIL";
      std::printf("[server] --verify: %s (secure result %s the plaintext reference)\n", ok ? "PASS" : "FAIL",
                  ok ? "equals" : "DIFFERS FROM");
      if (!ok) rc = 1;
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
       << P.batch << ",\"qbits\":" << P.qbits << ",\"lr\":" << P.lr << ",\"bias\":" << (P.bias ? "true" : "false")
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
    std::ifstream probe(O.csv_path);
    const bool fresh = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
    probe.close();
    std::ofstream csv(O.csv_path, std::ios::app);
    if (fresh)
      csv << "party,d,iters,batch,qbits,lr,threads,ssp,chunk_iters,and_gates,client_garble_ms,client_garble_cpu_ms,"
             "client_preprocessing_total_ms,server_eval_ms,server_eval_cpu_ms,server_online_total_ms,"
             "client_total_ms,server_total_ms,comm_client_to_server_bytes,comm_server_to_client_bytes,"
             "comm_total_bytes,client_peak_rss_mib,server_peak_rss_mib,verified\n";
    const double cl_pre = cl.cat[SETUP].wall_s + cl.cat[PREPROC].wall_s + cl.cat[GC].wall_s;
    const double sv_online = sv.cat[INPUTS].wall_s + sv.cat[GC].wall_s + sv.cat[CHECKS].wall_s + sv.cat[OUTPUT].wall_s;
    char row[1024];
    std::snprintf(row, sizeof row,
                  "%s,%d,%d,%d,%d,%g,%d,%d,%d,%.0f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%" PRIu64 ",%" PRIu64
                  ",%" PRIu64 ",%.1f,%.1f,%s\n",
                  me, P.d, P.iters, P.batch, P.qbits, P.lr, O.threads, O.ssp, O.chunk_iters, cl.num_and,
                  cl.cat[GC].wall_s * 1e3, cl.cat[GC].cpu_s * 1e3, cl_pre * 1e3, sv.cat[GC].wall_s * 1e3,
                  sv.cat[GC].cpu_s * 1e3, sv_online * 1e3, cl.total.wall_s * 1e3, sv.total.wall_s * 1e3,
                  cl.total.sent, sv.total.sent, cl.total.sent + sv.total.sent, cl.peak_rss_mib, sv.peak_rss_mib,
                  verified.c_str());
    csv << row;
  }

  sess.reset();
  return rc;
}
