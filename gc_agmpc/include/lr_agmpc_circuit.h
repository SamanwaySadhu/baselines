// lr_agmpc_circuit.h -- gc_ag2pc's exact fixed-point LR training circuit (sbit.h + lr_circuit.h, unchanged), recorded
// as Bristol-style gate lists for emp-agmpc, one circuit per SGD iteration, and cut into segments of at most N AND
// gates. Standard library only (no emp): the recorder plays the role of an emp-tool context for sbit.h.
//
// Parties (emp-agmpc numbering): party 1 = SERVER (evaluator, no input); party k+1 = CLIENT k (k = 1..batch). Client
// k holds datapoint k-1 of every batch and one XOR share of the initial weights.
#ifndef LR_AGMPC_CIRCUIT_H
#define LR_AGMPC_CIRCUIT_H

#include "lr_circuit.h"
#include "lr_params.h"
#include "lr_reference.h"
#include "sbit.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace lrgc {

// ---------------------------------------------------------------- recording context
// Wire ids are allocated sequentially: inputs first ([0, n_in)), then one id per gate. Ops use emp-agmpc's codes:
// AND 0, XOR 1, NOT 2. Bristol has no constants: 0 = XOR(w0, w0) and 1 = NOT(0), both free in authenticated garbling.
struct BristolRec {
  using Wire = uint32_t;
  struct G { uint32_t a, b, o; uint8_t op; };
  std::vector<G> g;
  uint32_t next = 0, n_in = 0;
  int64_t zero = -1, one = -1;
  bool closed = false;
  Wire input(uint32_t n) {
    if (closed) throw std::logic_error("inputs must be allocated before any gate");
    const Wire b = next;
    next += n;
    n_in += n;
    return b;
  }
  Wire emit(uint32_t a, uint32_t b, uint8_t op) { closed = true; g.push_back({a, b, next, op}); return next++; }
  Wire public_bit(bool v) {
    if (zero < 0) zero = emit(0, 0, 1);
    if (!v) return (Wire)zero;
    if (one < 0) one = emit((uint32_t)zero, 0, 2);
    return (Wire)one;
  }
  Wire and_gate(Wire a, Wire b) { return emit(a, b, 0); }
  Wire xor_gate(Wire a, Wire b) { return emit(a, b, 1); }
  Wire not_gate(Wire a) { return emit(a, 0, 2); }
};
using RecInt = SInt<BristolRec>;

struct Circuit {  // a finalized circuit: inputs are wires [0, n_in), outputs the last n_out wires
  uint32_t num_wire = 0, n_in = 0, n_out = 0;
  std::vector<BristolRec::G> gates;
  uint64_t ands = 0;
};

// Dead-gate elimination and renumbering; each output becomes a fresh XOR(src, zero) copy at the end.
inline Circuit finalize(const BristolRec& R, const std::vector<uint32_t>& outs) {
  const uint32_t N = R.next;
  std::vector<char> live(N, 0);
  for (uint32_t o : outs) live[o] = 1;
  for (size_t k = R.g.size(); k-- > 0;) {
    const auto& gt = R.g[k];
    if (!live[gt.o]) continue;
    live[gt.a] = 1;
    if (gt.op != 2) live[gt.b] = 1;
  }
  Circuit C;
  C.n_in = R.n_in;
  std::vector<uint32_t> id(N, UINT32_MAX);
  for (uint32_t i = 0; i < R.n_in; ++i) id[i] = i;
  uint32_t nx = R.n_in;
  const uint32_t z = nx++;
  C.gates.push_back({0, 0, z, 1});
  for (const auto& gt : R.g) {
    if (!live[gt.o]) continue;
    id[gt.o] = nx++;
    C.gates.push_back({id[gt.a], gt.op == 2 ? 0u : id[gt.b], id[gt.o], gt.op});
  }
  for (uint32_t o : outs) C.gates.push_back({id[o], z, nx++, 1});
  C.num_wire = nx;
  C.n_out = (uint32_t)outs.size();
  for (const auto& gt : C.gates) C.ands += gt.op == 0;
  return C;
}

// ---------------------------------------------------------------- weight layout across iterations
// Public, data-independent description of the weight vector crossing an iteration boundary: per weight its range
// and, per bit, a constant or a slot in the vector of unique secret bits (gc_ag2pc's checkpoint order).
struct Layout {
  struct W { int64_t lo, hi; std::vector<int> bit; };  // bit: -1 const 0, -2 const 1, >= 0 slot
  std::vector<W> w;
  int slots = 0;
};

inline Layout pack(const std::vector<RecInt>& w, std::vector<uint32_t>& wires) {
  Layout L;
  std::unordered_map<uint32_t, int> slot;
  wires.clear();
  for (const auto& v : w) {
    Layout::W lw{v.lo, v.hi, {}};
    for (const auto& b : v.b) {
      if (b.is_const()) { lw.bit.push_back(b.k ? -2 : -1); continue; }
      auto [it, fresh] = slot.emplace(b.w, (int)wires.size());
      if (fresh) wires.push_back(b.w);
      lw.bit.push_back(it->second);
    }
    L.w.push_back(lw);
  }
  L.slots = (int)wires.size();
  return L;
}

inline std::vector<RecInt> unpack(const Layout& L, uint32_t base) {
  std::vector<RecInt> w;
  for (const auto& lw : L.w) {
    RecInt v;
    v.lo = lw.lo;
    v.hi = lw.hi;
    for (int s : lw.bit) v.b.push_back(s >= 0 ? Builder<BristolRec>::wire(base + (uint32_t)s) : Builder<BristolRec>::cst(s == -2));
    w.push_back(v);
  }
  return w;
}

inline std::vector<int64_t> decode(const Layout& L, const std::vector<uint8_t>& bits) {
  std::vector<int64_t> r;
  for (const auto& lw : L.w) {
    uint64_t u = 0;
    const int n = (int)lw.bit.size();
    for (int i = 0; i < n; ++i) {
      const int s = lw.bit[(size_t)i];
      if (s == -2 || (s >= 0 && bits[(size_t)s])) u |= uint64_t(1) << i;
    }
    if (n < 64 && ((u >> (n - 1)) & 1)) u |= ~((uint64_t(1) << n) - 1);
    r.push_back((int64_t)u);
  }
  return r;
}

// ---------------------------------------------------------------- one iteration's circuit
struct ClientBlock { uint32_t share = 0, feat = 0, label = 0; };

inline std::vector<uint32_t> range_wires(uint32_t base, int n) {
  std::vector<uint32_t> r;
  for (int i = 0; i < n; ++i) r.push_back(base + (uint32_t)i);
  return r;
}

// w0 = XOR of the clients' shares (free XORs), read as a qbits-bit quantized value.
inline std::vector<RecInt> initial_weights(Builder<BristolRec>& B, LRCircuit<BristolRec>& lr, const Params& P, const std::vector<ClientBlock>& cb) {
  std::vector<RecInt> w;
  for (int j = 0; j < P.d; ++j) {
    std::vector<uint32_t> bits;
    for (int q = 0; q < P.qbits; ++q) {
      Builder<BristolRec>::Bit acc = Builder<BristolRec>::wire(cb[0].share + (uint32_t)(j * P.qbits + q));
      for (size_t k = 1; k < cb.size(); ++k) acc = B.XOR(acc, Builder<BristolRec>::wire(cb[k].share + (uint32_t)(j * P.qbits + q)));
      bits.push_back(acc.w);
    }
    w.push_back(lr.quantized(bits));
  }
  return w;
}

struct IterCircuit {
  Circuit C;
  Layout out_w;                       // layout of the weights leaving this iteration
  uint32_t carried = 0;               // input wires [0, carried) = previous iteration's output slots
  std::vector<uint32_t> start, end;   // per emp-agmpc party 1..batch+1 (index 0 unused): its private input wires
  int owner(uint32_t w) const {
    for (size_t p = 2; p < start.size(); ++p) if (w >= start[p] && w < end[p]) return (int)p;
    return -1;
  }
};

// Iteration t. `prev` is the layout of the incoming weights (ignored at t = 0, where the clients input XOR shares).
inline IterCircuit build_iteration(const Params& P, const SigmoidTable& S, int t, const Layout& prev) {
  IterCircuit IC;
  BristolRec R;
  IC.carried = t == 0 ? 0 : (uint32_t)prev.slots;
  const uint32_t wbase = IC.carried ? R.input(IC.carried) : 0;
  IC.start.assign((size_t)P.batch + 2, 0);
  IC.end.assign((size_t)P.batch + 2, 0);
  std::vector<ClientBlock> cb((size_t)P.batch);
  for (int k = 0; k < P.batch; ++k) {
    const uint32_t s = R.next;
    if (t == 0) cb[(size_t)k].share = R.input((uint32_t)(P.d * P.qbits));
    cb[(size_t)k].feat = R.input((uint32_t)(P.d_data * P.qbits));
    cb[(size_t)k].label = R.input(1);
    IC.start[(size_t)k + 2] = s;
    IC.end[(size_t)k + 2] = R.next;
  }
  Builder<BristolRec> B(R);
  LRCircuit<BristolRec> lr(B, P, S);
  std::vector<RecInt> w = t == 0 ? initial_weights(B, lr, P, cb) : unpack(prev, wbase);
  std::vector<std::vector<RecInt>> x;
  std::vector<SBit<BristolRec>> y;
  for (int i = 0; i < P.batch; ++i) {
    std::vector<std::vector<uint32_t>> cols;
    for (int j = 0; j < P.d_data; ++j) cols.push_back(range_wires(cb[(size_t)i].feat + (uint32_t)(j * P.qbits), P.qbits));
    x.push_back(lr.feature_row(cols));
    y.push_back(Builder<BristolRec>::wire(cb[(size_t)i].label));
  }
  w = lr.step(w, x, y);
  std::vector<uint32_t> outs;
  IC.out_w = pack(w, outs);
  IC.C = finalize(R, outs);
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
// Client k's XOR share of w0 (qbits bits per weight). Test-data convention: clients 1..batch-1 draw their share from
// the seed, the last client's share completes w0, so the same --seed gives the siblings' w0. In a deployment each
// client would bring its own share.
inline std::vector<std::vector<int64_t>> w0_shares(const Params& P, const std::vector<int64_t>& w0) {
  SplitMix64 rng(P.seed * 0xD1B54A32D192ED03ULL + 7);
  const uint64_t m = (uint64_t(1) << P.qbits) - 1;
  std::vector<std::vector<int64_t>> sh((size_t)P.batch, std::vector<int64_t>((size_t)P.d));
  for (int j = 0; j < P.d; ++j) {
    uint64_t acc = (uint64_t)w0[(size_t)j] & m;
    for (int k = 0; k + 1 < P.batch; ++k) {
      const uint64_t r = (uint64_t)rng.uniform(0, (int64_t)m);
      sh[(size_t)k][(size_t)j] = (int64_t)r;
      acc ^= r;
    }
    sh[(size_t)P.batch - 1][(size_t)j] = (int64_t)acc;
  }
  return sh;
}

inline void push_q(std::vector<uint8_t>& v, int64_t x, int q) { for (bool b : to_bits(x, q)) v.push_back(b); }

// Client k's (0-based) input bits for iteration t, in the order of its input block.
inline void client_bits(std::vector<uint8_t>& v, const Params& P, const Dataset& D, const std::vector<std::vector<int64_t>>& sh, int k, int t) {
  if (t == 0)
    for (int j = 0; j < P.d; ++j) push_q(v, sh[(size_t)k][(size_t)j], P.qbits);
  for (int j = 0; j < P.d_data; ++j) push_q(v, D.X(t, k, j), P.qbits);
  v.push_back(D.Y(t, k) ? 1 : 0);
}

}  // namespace lrgc

#endif  // LR_AGMPC_CIRCUIT_H
