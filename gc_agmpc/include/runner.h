// runner.h -- one party (server = emp-agmpc party 1, client k = party k+1) of the multiparty authenticated-garbling LR
// training (gc_agmpc baseline). Templated on nP = clients + 1; instantiated per nP in src/runner_np*.cpp.
//
// Per iteration t: build iteration t's circuit (from the public layout of the weights leaving t-1), cut it into
// segments of <= --seg-ands ANDs, and run each segment as one emp-agmpc CMPC instance:
//   function_independent (authenticated triples + aBits)  -> offline/func_indep
//   function_dependent   (mask assignment, x/y openings, garbled tables to the server) -> offline/func_dep
//   online(FlexIn, FlexOut): private inputs / carried authenticated shares in, evaluation at the server,
//                            authenticated shares out (or, for the last segment of the last iteration, the weights
//                            revealed to the server only)                                 -> online/{input,evaluate,output}
// One FpreMP (base OTs, COT extenders, each party's Delta) is shared by every instance (PATCHES.md P9), so carried
// authenticated shares remain valid from one instance to the next.
#ifndef LR_AGMPC_RUNNER_H
#define LR_AGMPC_RUNNER_H

#include <emp-tool/emp-tool.h>
#include "emp-agmpc/emp-agmpc.h"

#include "lr_agmpc_circuit.h"
#include "metrics.h"
#include "run_cfg.h"

#include <array>

#include <cinttypes>
#include <cmath>
#include <fstream>
#include <memory>
#include <unordered_map>

namespace lrgc {

// Parameters every party must agree on (the handshake compares them byte for byte).
struct Handshake {
  int32_t magic, version, nP, d, iters, batch, qbits, bias, ssp;
  int64_t lr_micro, seg_ands;
};

inline std::string party_label(int p) { return p == 1 ? "server" : "client " + std::to_string(p - 1); }

inline const char* kPhaseOrder[] = {"net/connect",        "local/circuit",    "setup/handshake", "setup/base_ot",
                                    "offline/func_indep", "offline/func_dep", "online/input",    "online/evaluate",
                                    "online/output"};
constexpr int kNumPhases = 9;

struct PartyReport {
  std::vector<PhaseRow> rows;
  long rss_kib = 0;
};

struct RunStats {
  uint64_t ands = 0, segments = 0, carried_in = 0, max_seg_ands = 0, echo_rounds = 0;
};

template <int nP>
int run_party(const RunCfg& C) {
  static_assert(nP >= 2 && nP + 1 <= kMaxParties, "unsupported number of parties");
  const Params& P = C.P;
  const int party = C.party;
  g_agmpc.party = party;
  if (P.batch + 1 != nP) throw std::logic_error("batch must equal the number of clients");

  NetIOMP<nP>* io = nullptr;
  NetIOMP<nP>* io2 = nullptr;
  auto snap = [&](uint64_t* s, uint64_t* r) {
    for (int j = 1; j <= nP; ++j)
      if (j != party) {
        s[j] = (io ? io->sent_to(j) : 0) + (io2 ? io2->sent_to(j) : 0);
        r[j] = (io ? io->recv_from(j) : 0) + (io2 ? io2->recv_from(j) : 0);
      }
  };
  Recorder rec(snap);
  static Recorder* g_rec = nullptr;
  g_rec = &rec;
  g_phase_hook = [](const char* n) { g_rec->enter(n); };
  auto say = [&](const std::string& m) { if (!C.quiet) std::fprintf(stderr, "[%s] %s\n", party_label(party).c_str(), m.c_str()); };

  // ---- the client's private inputs (random test data from --seed; client k uses only datapoint k of each batch
  //      and its own share of w0)
  const SigmoidTable S = make_sigmoid_table(P.f);
  Dataset D;
  std::vector<std::vector<int64_t>> sh;
  if (party >= 2) {
    rec.enter("local/circuit");
    D = generate_client_data(P);
    sh = w0_shares(P, generate_initial_weights(P));
  }

  // ---- network
  rec.enter("net/connect");
  io = new NetIOMP<nP>(party, C.port);
  io2 = new NetIOMP<nP>(party, C.port + NetIOMP<nP>::port_span());
  rec.set_snap(snap);
  NetIOMP<nP>* ios[2] = {io, io2};
  ThreadPool pool((size_t)(C.threads > 0 ? std::max(C.threads, 2 * (nP - 1) + 2) : 2 * (nP - 1) + 2));

  rec.enter("setup/handshake");
  {
    Handshake mine;
    std::memset(&mine, 0, sizeof mine);  // padding bytes are compared too
    mine.magic = 0x41474d43; mine.version = 1; mine.nP = nP; mine.d = P.d; mine.iters = P.iters; mine.batch = P.batch;
    mine.qbits = P.qbits; mine.bias = P.bias ? 1 : 0; mine.ssp = C.ssp; mine.lr_micro = std::llround(P.lr * 1e6);
    mine.seg_ands = (int64_t)C.seg_ands;
    for (int j = 1; j <= nP; ++j) if (j != party) { io->send_data(j, &mine, sizeof mine); io->flush(j); }
    for (int j = 1; j <= nP; ++j) if (j != party) {
      Handshake th;
      io->recv_data(j, &th, sizeof th);
      if (std::memcmp(&th, &mine, sizeof mine) != 0)
        agmpc_abort(1, party_label(j) + " was started with different parameters (d, iters, batch, qbits, lr, bias, ssp, seg-ands)");
    }
  }
  rec.enter("setup/base_ot");
  NetIOMP<nP>::reset_transcripts();
  bool delta[128];
  { PRG prg; prg.random_bool(delta, 128); }
  std::unique_ptr<FpreMP<nP>> fpre(new FpreMP<nP>(ios, &pool, party, delta, C.ssp));
  say("setup done (" + std::to_string(nP - 1) + " clients, ssp " + std::to_string(C.ssp) + ", segments of <= " +
      std::to_string(C.seg_ands) + " ANDs)");

  RunStats st;
  Layout prev;
  std::vector<AuthBitShare<nP>> carried;  // authenticated shares of the weights entering iteration t
  std::vector<uint8_t> final_bits;
  Layout final_layout;
  for (int t = 0; t < P.iters; ++t) {
    rec.enter("local/circuit");
    IterCircuit IC = build_iteration(P, S, t, prev);
    std::vector<Seg> segs = segment(IC.C, C.seg_ands);
    std::vector<uint8_t> mine;
    if (party >= 2) client_bits(mine, P, D, sh, party - 2, t);
    st.ands += IC.C.ands;
    std::unordered_map<uint32_t, AuthBitShare<nP>> shares;  // carried shares produced inside this iteration
    for (size_t si = 0; si < segs.size(); ++si) {
      const Seg& sg = segs[si];
      const bool final_seg = (t == P.iters - 1) && (si + 1 == segs.size());
      rec.enter("offline/func_indep");
      BristolFormat cf((int)(sg.gates.size() / 4), sg.num_wire, sg.n_in, 0, sg.n_out, const_cast<int*>(sg.gates.data()));
      std::unique_ptr<CMPC<nP>> mpc(new CMPC<nP>(ios, &pool, party, &cf, fpre.get(), C.ssp));
      mpc->function_independent();
      rec.enter("offline/func_dep");
      mpc->function_dependent();
      rec.enter("online/input");
      FlexIn<nP> in(sg.n_in, party);
      FlexOut<nP> out(sg.n_out, party);
      for (int i = 0; i < sg.n_in; ++i) {
        const uint32_t w = sg.in_g[(size_t)i];
        auto it = shares.find(w);
        if (it != shares.end()) { in.assign_party(i, -1); in.assign_authenticated_bitshare(i, &it->second); ++st.carried_in; }
        else if (w < IC.carried) { in.assign_party(i, -1); in.assign_authenticated_bitshare(i, &carried[w]); ++st.carried_in; }
        else {
          const int o = IC.owner(w);
          in.assign_party(i, o);
          if (o == party) in.assign_plaintext_bit(i, mine[w - IC.start[(size_t)party]] != 0);
        }
      }
      for (int i = 0; i < sg.n_out; ++i) out.assign_party(i, final_seg ? 1 : -1);
      mpc->online(&in, &out);
      for (int i = 0; i < sg.n_out; ++i) {
        if (final_seg) { if (party == 1) final_bits.push_back(out.get_plaintext_bit(i) ? 1 : 0); }
        else shares[sg.out_g[(size_t)i]] = out.get_authenticated_bitshare(i);
      }
      st.segments++;
      st.max_seg_ands = std::max<uint64_t>(st.max_seg_ands, (uint64_t)sg.ands);
    }
    if (t + 1 < P.iters) {
      carried.assign((size_t)IC.C.n_out, AuthBitShare<nP>{});
      for (uint32_t s = 0; s < IC.C.n_out; ++s) carried[s] = shares.at(IC.C.num_wire - IC.C.n_out + s);
    } else {
      final_layout = IC.out_w;
    }
    prev = IC.out_w;
    if (t == 0 || (t + 1) % 5 == 0 || t + 1 == P.iters)
      say("iteration " + std::to_string(t + 1) + "/" + std::to_string(P.iters) + " done (" + std::to_string(segs.size()) + " segment" +
          (segs.size() == 1 ? "" : "s") + ")");
  }
  rec.close();
  st.echo_rounds = g_echo_rounds;
  g_phase_hook = nullptr;

  // ---- report: the clients send their tables to the server (outside the measured phases)
  std::vector<PartyReport> R((size_t)nP + 1);
  R[(size_t)party].rows = rec.rows();
  R[(size_t)party].rss_kib = peak_rss_kib();
  if (party != 1) {
    const uint32_t n = (uint32_t)rec.rows().size();
    io->send_data(1, &n, 4);
    io->send_data(1, rec.rows().data(), n * sizeof(PhaseRow));
    const int64_t rss = R[(size_t)party].rss_kib;
    io->send_data(1, &rss, 8);
    io->flush(1);
    char ack = 0;
    io->recv_data(1, &ack, 1);
    delete io2;
    delete io;
    say("done (the server learned the trained weights)");
    return 0;
  }
  for (int j = 2; j <= nP; ++j) {
    uint32_t n = 0;
    io->recv_data(j, &n, 4);
    if (n > 64) agmpc_abort(1, "bad report from " + party_label(j));
    R[(size_t)j].rows.resize(n);
    io->recv_data(j, R[(size_t)j].rows.data(), n * sizeof(PhaseRow));
    for (auto& r : R[(size_t)j].rows) r.name[sizeof r.name - 1] = 0;
    int64_t rss = 0;
    io->recv_data(j, &rss, 8);
    R[(size_t)j].rss_kib = rss;
  }
  for (int j = 2; j <= nP; ++j) { const char ack = 1; io->send_data(j, &ack, 1); io->flush(j); }
  delete io2;
  delete io;

  // ---- the trained weights (the server's only output)
  const std::vector<int64_t> w = decode(final_layout, final_bits);
  std::printf("Multiparty authenticated garbling (emp-agmpc, WRK17) logistic regression, MALICIOUS security, %d clients + server\n", nP - 1);
  std::printf("%s\n", params_string(P).c_str());
  std::printf("trained weights (value = int / 2^%d):", P.f);
  for (size_t j = 0; j < w.size() && j < 8; ++j) std::printf(" %" PRId64, w[j]);
  if (w.size() > 8) std::printf(" ... (%zu weights)", w.size());
  std::printf("\n");

  int rc = 0;
  std::string verified = "no";
  if (C.verify) {  // TEST ONLY: the server regenerates the clients' data from --seed
    const Dataset D2 = generate_client_data(P);
    const std::vector<int64_t> ref = reference_train(P, D2, generate_initial_weights(P));
    const bool ok = ref == w;
    verified = ok ? "exact" : "FAIL";
    std::printf("verify: %s the plaintext reference reference_train (test only: the server regenerated the clients' data)\n",
                ok ? "EXACT MATCH with" : "MISMATCH against");
    if (!ok) rc = 2;
  }

  // ---- tables
  auto find = [](const std::vector<PhaseRow>& rows, const char* n) -> const PhaseRow* {
    for (const auto& r : rows) if (std::strcmp(r.name, n) == 0) return &r;
    return nullptr;
  };
  std::printf("\nphase (ms; bytes = sent by that party) ");
  for (int p = 1; p <= nP; ++p) std::printf("| %-9s wall      cpu      sent ", party == p ? "server" : ("client" + std::to_string(p - 1)).c_str());
  std::printf("\n");
  for (int k = 0; k < kNumPhases; ++k) {
    std::printf("%-38s", kPhaseOrder[k]);
    for (int p = 1; p <= nP; ++p) {
      const PhaseRow* r = find(R[(size_t)p].rows, kPhaseOrder[k]);
      if (r) std::printf("| %9.1f %9.1f %10s ", r->wall_s * 1e3, r->cpu_s * 1e3, human_bytes((double)r->sent_total()).c_str());
      else std::printf("| %9s %9s %10s ", "-", "-", "-");
    }
    std::printf("\n");
  }
  const char* stages[] = {"local", "setup", "offline", "online"};
  std::vector<std::array<StageTotals, 4>> T((size_t)nP + 1);
  for (int p = 1; p <= nP; ++p)
    for (int s = 0; s < 4; ++s) T[(size_t)p][(size_t)s] = stage_totals(R[(size_t)p].rows, stages[s]);
  auto tot = [&](int p, bool cpu) { double v = 0; for (int s = 0; s < 4; ++s) v += cpu ? T[(size_t)p][(size_t)s].cpu_s : T[(size_t)p][(size_t)s].wall_s; return v; };
  auto sent = [&](int p) { uint64_t v = 0; for (int s = 0; s < 4; ++s) v += T[(size_t)p][(size_t)s].sent; return v; };
  auto recv = [&](int p) { uint64_t v = 0; for (int s = 0; s < 4; ++s) v += T[(size_t)p][(size_t)s].recv; return v; };
  uint64_t total = 0, total_recv = 0;
  for (int p = 1; p <= nP; ++p) { total += sent(p); total_recv += recv(p); }
  if (total != total_recv) { std::fprintf(stderr, "[server] error: report invariant violated: sum sent %" PRIu64 " != sum received %" PRIu64 "\n", total, total_recv); rc = rc ? rc : 1; }
  uint64_t stage_comm[4] = {0, 0, 0, 0};
  for (int p = 1; p <= nP; ++p) for (int s = 0; s < 4; ++s) stage_comm[s] += T[(size_t)p][(size_t)s].sent;

  std::printf("\nSUMMARY (malicious, d=%d, iters=%d, batch=%d = %d clients, %" PRIu64 " ANDs in %" PRIu64 " segments of <= %" PRIu64 " ANDs)\n",
              P.d, P.iters, P.batch, nP - 1, st.ands, st.segments, C.seg_ands);
  std::printf("  computation (wall / cpu, ms)   local          setup          offline              online               total\n");
  for (int p = 1; p <= nP; ++p)
    std::printf("    %-9s %13.1f/%-8.1f %8.1f/%-8.1f %10.1f/%-10.1f %9.1f/%-9.1f %10.1f/%-10.1f\n", party_label(p).c_str(),
                T[(size_t)p][0].wall_s * 1e3, T[(size_t)p][0].cpu_s * 1e3, T[(size_t)p][1].wall_s * 1e3, T[(size_t)p][1].cpu_s * 1e3,
                T[(size_t)p][2].wall_s * 1e3, T[(size_t)p][2].cpu_s * 1e3, T[(size_t)p][3].wall_s * 1e3, T[(size_t)p][3].cpu_s * 1e3,
                tot(p, false) * 1e3, tot(p, true) * 1e3);
  std::printf("  communication (sent / received; setup | offline | online)\n");
  for (int p = 1; p <= nP; ++p)
    std::printf("    %-9s %11s / %-11s  (sent: %s | %s | %s)\n", party_label(p).c_str(), human_bytes((double)sent(p)).c_str(),
                human_bytes((double)recv(p)).c_str(), human_bytes((double)T[(size_t)p][1].sent).c_str(), human_bytes((double)T[(size_t)p][2].sent).c_str(),
                human_bytes((double)T[(size_t)p][3].sent).c_str());
  std::printf("  total communication %s (setup %s, offline %s, online %s); check: sum sent = sum received %s\n", human_bytes((double)total).c_str(),
              human_bytes((double)stage_comm[1]).c_str(), human_bytes((double)stage_comm[2]).c_str(), human_bytes((double)stage_comm[3]).c_str(),
              total == total_recv ? "ok" : "VIOLATED");
  long max_client_rss = 0;
  for (int p = 2; p <= nP; ++p) max_client_rss = std::max(max_client_rss, R[(size_t)p].rss_kib);
  std::printf("  peak RSS server %.0f MiB, clients max %.0f MiB; %" PRIu64 " carried authenticated input bits; %" PRIu64 " echo-broadcast checks at the server\n",
              R[1].rss_kib / 1024.0, max_client_rss / 1024.0, st.carried_in, st.echo_rounds);
  std::printf("  (connect time is not counted; offline = input-independent work, interleaved per segment and summed)\n");

  if (!C.json.empty()) {
    std::ofstream f(C.json);
    f << "{\"protocol\": \"gc_agmpc\", \"security\": \"malicious\", \"d\": " << P.d << ", \"iters\": " << P.iters << ", \"batch\": " << P.batch
      << ", \"clients\": " << nP - 1 << ", \"ssp\": " << C.ssp << ", \"seg_ands\": " << C.seg_ands << ", \"ands_total\": " << st.ands
      << ", \"segments\": " << st.segments << ", \"verified\": \"" << verified << "\", \"comm_total_bytes\": " << total << ", \"phases\": [";
    bool first = true;
    for (int p = 1; p <= nP; ++p)
      for (const auto& r : R[(size_t)p].rows) {
        f << (first ? "" : ",") << "\n  {\"party\": \"" << party_label(p) << "\", \"phase\": \"" << r.name << "\", \"wall_s\": " << r.wall_s
          << ", \"cpu_s\": " << r.cpu_s;
        for (int q = 1; q <= nP; ++q) if (q != p) f << ", \"sent_to_" << (q == 1 ? std::string("server") : "client" + std::to_string(q - 1)) << "\": " << r.sent[q];
        for (int q = 1; q <= nP; ++q) if (q != p) f << ", \"recv_from_" << (q == 1 ? std::string("server") : "client" + std::to_string(q - 1)) << "\": " << r.recv[q];
        f << "}";
        first = false;
      }
    f << "\n]}\n";
  }
  if (!C.csv.empty()) {
    std::string hdr = "protocol,security,d,iters,batch,qbits,clients,ssp,seg_ands,ands_total,segments,verified";
    auto party_cols = [&](const std::string& pre) {
      for (const char* s : {"local", "setup", "offline", "online", "total"}) hdr += "," + pre + "_" + s + "_wall_ms," + pre + "_" + s + "_cpu_ms";
      hdr += "," + pre + "_sent_bytes," + pre + "_recv_bytes," + pre + "_peak_rss_mib";
    };
    party_cols("server");
    for (int c = 1; c <= kMaxParties - 2; ++c) party_cols("client" + std::to_string(c));
    hdr += ",comm_setup_bytes,comm_offline_bytes,comm_online_bytes,comm_total_bytes";
    std::string row;
    char b[64];
    auto add = [&](double v, const char* fmt) { std::snprintf(b, sizeof b, fmt, v); row += ","; row += b; };
    row = "gc_agmpc,malicious," + std::to_string(P.d) + "," + std::to_string(P.iters) + "," + std::to_string(P.batch) + "," + std::to_string(P.qbits) + "," +
          std::to_string(nP - 1) + "," + std::to_string(C.ssp) + "," + std::to_string(C.seg_ands) + "," + std::to_string(st.ands) + "," +
          std::to_string(st.segments) + "," + verified;
    for (int p = 1; p <= kMaxParties - 1; ++p) {
      if (p > nP) { for (int k = 0; k < 13; ++k) row += ","; continue; }
      for (int s = 0; s < 4; ++s) { add(T[(size_t)p][(size_t)s].wall_s * 1e3, "%.3f"); add(T[(size_t)p][(size_t)s].cpu_s * 1e3, "%.3f"); }
      add(tot(p, false) * 1e3, "%.3f");
      add(tot(p, true) * 1e3, "%.3f");
      add((double)sent(p), "%.0f");
      add((double)recv(p), "%.0f");
      add(R[(size_t)p].rss_kib / 1024.0, "%.1f");
    }
    for (int s = 1; s < 4; ++s) add((double)stage_comm[s], "%.0f");
    add((double)total, "%.0f");
    std::string first_line;
    { std::ifstream in(C.csv); std::getline(in, first_line); }
    if (!first_line.empty() && first_line != hdr) { std::fprintf(stderr, "[server] error: %s has a different header\n", C.csv.c_str()); return 1; }
    std::ofstream f(C.csv, std::ios::app);
    if (first_line.empty()) f << hdr << "\n";
    f << row << "\n";
  }
  return rc;
}

}  // namespace lrgc

#endif  // LR_AGMPC_RUNNER_H
