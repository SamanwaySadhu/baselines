// lr_agmpc_circuit.h -- the float32 logistic-regression training circuit for emp-agmpc.
//
// The circuit of ONE SGD iteration (gen/gen_circuit.cpp, emp-tool 1.0, gc_ag2pc's float32 LR) is generated once per
// dimension d into two Bristol-style files and loaded here: "first" (iteration 0: the clients input XOR shares of the
// initial weights) and "next" (iterations 1..: the weights come in as carried authenticated shares). All iterations
// after the first are the same circuit. Each is cut into segments of at most N AND gates, run as one emp-agmpc
// circuit each, chained through authenticated shares.
//
// Parties (emp-agmpc numbering): party 1 = SERVER (evaluator, no input); party k+1 = CLIENT k (k = 0..3 here is
// datapoint k of every batch) holding one XOR share of the initial weights.
//
// Input wire layout of an iteration: [carried weights: 32 d bits (iterations >= 1 only)]
//   then per client: [share of w0: 32 d bits (iteration 0 only)] [features x_j, j < d-1: 32 (d-1) bits] [label: 1 bit].
// Output: the 32 d bits of the new weights (IEEE binary32, LSB first, weight 0 first).
#ifndef LR_AGMPC_CIRCUIT_H
#define LR_AGMPC_CIRCUIT_H

#include "bristol.h"
#include "lr_params.h"
#include "lr_reference.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace lrgc {

inline std::string circuit_path(const std::string& dir, int d, bool first) {
  return dir + "/lr_f32_d" + std::to_string(d) + (first ? "_first.bc" : "_next.bc");
}

struct IterCircuit {
  Circuit C;
  uint32_t carried = 0;               // input wires [0, carried) = the previous iteration's output bits
  std::vector<uint32_t> start, end;   // per emp-agmpc party 1..kBatch+1 (index 0, 1 unused): its private input wires
  int owner(uint32_t w) const {
    for (size_t p = 2; p < start.size(); ++p) if (w >= start[p] && w < end[p]) return (int)p;
    return -1;
  }
};

inline IterCircuit load_iteration(const std::string& dir, const Params& P, bool first) {
  IterCircuit IC;
  IC.C = load_circuit(circuit_path(dir, P.d, first), (uint32_t)P.d, first ? 1u : 0u);
  const uint32_t share = first ? 32u * (uint32_t)P.d : 0u, feat = 32u * (uint32_t)P.d_data(), blk = share + feat + 1;
  IC.carried = first ? 0 : 32u * (uint32_t)P.d;
  IC.start.assign((size_t)kBatch + 2, 0);
  IC.end.assign((size_t)kBatch + 2, 0);
  for (int k = 0; k < kBatch; ++k) {
    IC.start[(size_t)k + 2] = IC.carried + (uint32_t)k * blk;
    IC.end[(size_t)k + 2] = IC.start[(size_t)k + 2] + blk;
  }
  if (IC.C.n_in != IC.carried + (uint32_t)kBatch * blk || IC.C.n_out != 32u * (uint32_t)P.d)
    throw std::runtime_error("circuit file does not match the input layout for d=" + std::to_string(P.d));
  return IC;
}

// ---------------------------------------------------------------- segmentation
// One segment = one emp-agmpc circuit (CMPC instance). Its gate array uses emp-agmpc's layout {a, b, out, op}; its
// inputs are the global wires it reads (in_g: carried authenticated shares or private inputs entered here, in the
// first segment that reads them) and its outputs the wires it produces that later segments (or the iteration's
// outputs) read (out_g). "Direct" mode: a carried share waits outside any circuit until a segment reads it.
struct Seg {
  std::vector<int> gates;
  int n_in = 0, num_wire = 0, n_out = 0, ands = 0;
  std::vector<uint32_t> in_g, out_g;
};

inline std::vector<Seg> segment(const Circuit& C, uint64_t N) {
  const uint32_t W = C.num_wire, G = (uint32_t)C.gates.size();
  std::vector<int> gseg(G);
  int s = 0;
  uint64_t a = 0;
  for (uint32_t i = 0; i < G; ++i) {
    if (C.gates[i].op == 0) { if (a == N) { ++s; a = 0; } ++a; }
    gseg[i] = s;
  }
  const int S = s + 1;
  std::vector<int> home(W, -1), last(W, -1);
  for (uint32_t i = 0; i < G; ++i) {
    const auto& g = C.gates[i];
    const int sg = gseg[i];
    home[g.o] = sg;
    auto use = [&](uint32_t w) {
      if (w < C.n_in && home[w] < 0) home[w] = sg;
      last[w] = std::max(last[w], sg);
    };
    use(g.a);
    if (g.op != 2) use(g.b);
  }
  for (uint32_t w = W - C.n_out; w < W; ++w) last[w] = S;  // the iteration's outputs leave the last segment
  std::vector<std::vector<uint32_t>> reads((size_t)S), outs((size_t)S);
  std::vector<int> seen(W, -1);
  for (uint32_t i = 0; i < G; ++i) {
    const auto& g = C.gates[i];
    const int sg = gseg[i];
    for (int k = 0; k < (g.op == 2 ? 1 : 2); ++k) {
      const uint32_t w = k ? g.b : g.a;
      if (home[w] == sg && w >= C.n_in) continue;  // produced in this segment
      if (seen[w] != sg) { seen[w] = sg; reads[(size_t)sg].push_back(w); }
    }
  }
  for (uint32_t w = 0; w < W; ++w)
    if (home[w] >= 0 && last[w] > home[w]) outs[(size_t)home[w]].push_back(w);
  std::vector<Seg> segs((size_t)S);
  for (int sg = 0; sg < S; ++sg) {
    Seg& Z = segs[(size_t)sg];
    Z.in_g = reads[(size_t)sg];
    std::sort(Z.in_g.begin(), Z.in_g.end());
    Z.out_g = outs[(size_t)sg];
    std::sort(Z.out_g.begin(), Z.out_g.end());
    std::unordered_map<uint32_t, int> loc;
    int nx = 0;
    for (uint32_t w : Z.in_g) loc[w] = nx++;
    Z.n_in = nx;
    const int z = nx++;
    Z.gates.insert(Z.gates.end(), {0, 0, z, 1});
    for (uint32_t i = 0; i < G; ++i)
      if (gseg[i] == sg) {
        const auto& g = C.gates[i];
        const int la = loc.at(g.a), lb = g.op == 2 ? 0 : loc.at(g.b);
        loc[g.o] = nx;
        Z.gates.insert(Z.gates.end(), {la, lb, nx++, (int)g.op});
        Z.ands += g.op == 0;
      }
    for (uint32_t w : Z.out_g) Z.gates.insert(Z.gates.end(), {loc.at(w), z, nx++, 1});
    Z.num_wire = nx;
    Z.n_out = (int)Z.out_g.size();
  }
  return segs;
}

// Plaintext evaluation of one segment (selftest and segmentation check).
inline std::vector<uint8_t> eval_segment(const Seg& sg, const std::vector<uint8_t>& in) {
  std::vector<uint8_t> w((size_t)sg.num_wire, 2);
  std::copy(in.begin(), in.end(), w.begin());
  for (size_t i = 0; i < sg.gates.size() / 4; ++i) {
    const int* g = &sg.gates[4 * i];
    if (g[3] == 2) w[(size_t)g[2]] = w[(size_t)g[0]] ^ 1;
    else if (g[3] == 1) w[(size_t)g[2]] = w[(size_t)g[0]] ^ w[(size_t)g[1]];
    else w[(size_t)g[2]] = w[(size_t)g[0]] & w[(size_t)g[1]];
  }
  return std::vector<uint8_t>(w.end() - sg.n_out, w.end());
}

// ---------------------------------------------------------------- the clients' inputs (random test data)
// Client k's XOR share of w0 (the 32-bit pattern of each initial weight). Test-data convention: clients 0..2 draw
// their share from the seed, the last client's share completes w0, so the same --seed gives the siblings' w0. In a
// deployment each client would bring its own share.
inline std::vector<std::vector<uint32_t>> w0_shares(const Params& P, const std::vector<float>& w0) {
  SplitMix64 rng(P.seed * 0xD1B54A32D192ED03ULL + 7);
  std::vector<std::vector<uint32_t>> sh((size_t)kBatch, std::vector<uint32_t>((size_t)P.d));
  for (int j = 0; j < P.d; ++j) {
    uint32_t acc = f2u(w0[(size_t)j]);
    for (int k = 0; k + 1 < kBatch; ++k) {
      const uint32_t r = (uint32_t)rng.next();
      sh[(size_t)k][(size_t)j] = r;
      acc ^= r;
    }
    sh[(size_t)kBatch - 1][(size_t)j] = acc;
  }
  return sh;
}

inline void push_u32(std::vector<uint8_t>& v, uint32_t x) { for (int i = 0; i < 32; ++i) v.push_back((x >> i) & 1); }

// Client k's (0-based) input bits for iteration t, in the order of its input block.
inline void client_bits(std::vector<uint8_t>& v, const Params& P, const Dataset& D, const std::vector<std::vector<uint32_t>>& sh, int k, int t) {
  if (t == 0)
    for (int j = 0; j < P.d; ++j) push_u32(v, sh[(size_t)k][(size_t)j]);
  for (int j = 0; j < P.d_data(); ++j) push_u32(v, f2u(D.X(t, k, j)));
  v.push_back(D.Y(t, k) ? 1 : 0);
}

}  // namespace lrgc

#endif  // LR_AGMPC_CIRCUIT_H
