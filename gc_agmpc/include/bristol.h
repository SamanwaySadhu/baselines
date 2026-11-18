// bristol.h -- gate-list circuits for emp-agmpc. Standard library only, so that both the generator (built against
// emp-tool 1.0, gen/gen_circuit.cpp) and the runner (built against the legacy emp-tool that emp-agmpc needs) include it.
//
// BristolRec is a BooleanContext (emp-tool 1.0 concept: Wire, public_bit, and_gate, xor_gate, not_gate) that records
// gates. It folds constants and trivial identities (AND with 0/1, XOR with 0/1, x&x, x^x), so multiplications by
// public constants cost what they should. Ops use emp-agmpc's codes: AND 0, XOR 1, NOT 2. Wire ids: inputs first
// [0, n_in), then one id per gate.
#ifndef LR_AGMPC_BRISTOL_H
#define LR_AGMPC_BRISTOL_H

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace lrgc {

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
  bool is0(Wire a) const { return zero >= 0 && a == (Wire)zero; }
  bool is1(Wire a) const { return one >= 0 && a == (Wire)one; }
  Wire not_gate(Wire a) {
    if (is0(a)) return public_bit(true);
    if (is1(a)) return public_bit(false);
    return emit(a, 0, 2);
  }
  Wire and_gate(Wire a, Wire b) {
    if (is0(a) || is0(b)) return public_bit(false);
    if (is1(a)) return b;
    if (is1(b)) return a;
    if (a == b) return a;
    return emit(a, b, 0);
  }
  Wire xor_gate(Wire a, Wire b) {
    if (is0(a)) return b;
    if (is0(b)) return a;
    if (is1(a)) return not_gate(b);
    if (is1(b)) return not_gate(a);
    if (a == b) return public_bit(false);
    return emit(a, b, 1);
  }
};

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

// Plaintext evaluation of a whole circuit.
inline std::vector<uint8_t> eval_circuit(const Circuit& C, const std::vector<uint8_t>& in) {
  std::vector<uint8_t> w((size_t)C.num_wire, 2);
  std::copy(in.begin(), in.end(), w.begin());
  for (const auto& g : C.gates) {
    if (g.op == 2) w[g.o] = w[g.a] ^ 1;
    else if (g.op == 1) w[g.o] = w[g.a] ^ w[g.b];
    else w[g.o] = w[g.a] & w[g.b];
  }
  return std::vector<uint8_t>(w.end() - C.n_out, w.end());
}

// ---- file format: 8 x u32 header {magic, d, variant, n_in, n_out, num_wire, ngates, 0} then ngates x {a, b, o, op} (u32)
constexpr uint32_t kCircuitMagic = 0x4c524743;  // "LRGC"

inline void save_circuit(const std::string& path, const Circuit& C, uint32_t d, uint32_t variant) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + path);
  const uint32_t h[8] = {kCircuitMagic, d, variant, C.n_in, C.n_out, C.num_wire, (uint32_t)C.gates.size(), 0};
  std::fwrite(h, 4, 8, f);
  std::vector<uint32_t> buf;
  buf.reserve(C.gates.size() * 4);
  for (const auto& g : C.gates) { buf.push_back(g.a); buf.push_back(g.b); buf.push_back(g.o); buf.push_back(g.op); }
  std::fwrite(buf.data(), 4, buf.size(), f);
  std::fclose(f);
}

inline Circuit load_circuit(const std::string& path, uint32_t d, uint32_t variant) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open circuit file " + path + " (run scripts/gen_circuits.sh --d " + std::to_string(d) + ")");
  uint32_t h[8];
  if (std::fread(h, 4, 8, f) != 8 || h[0] != kCircuitMagic || h[1] != d || h[2] != variant) {
    std::fclose(f);
    throw std::runtime_error("bad circuit file " + path);
  }
  Circuit C;
  C.n_in = h[3]; C.n_out = h[4]; C.num_wire = h[5];
  std::vector<uint32_t> buf((size_t)h[6] * 4);
  if (std::fread(buf.data(), 4, buf.size(), f) != buf.size()) { std::fclose(f); throw std::runtime_error("truncated " + path); }
  std::fclose(f);
  C.gates.resize(h[6]);
  for (size_t i = 0; i < C.gates.size(); ++i) {
    C.gates[i] = {buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], (uint8_t)buf[4 * i + 3]};
    C.ands += C.gates[i].op == 0;
  }
  return C;
}

// FNV-1a over the gate list: parties compare it in the handshake so they all run the same circuit.
inline uint64_t circuit_hash(const Circuit& C) {
  uint64_t h = 1469598103934665603ULL;
  auto mix = [&](uint32_t v) { h = (h ^ v) * 1099511628211ULL; };
  mix(C.n_in); mix(C.n_out); mix(C.num_wire);
  for (const auto& g : C.gates) { mix(g.a); mix(g.b); mix(g.o); mix(g.op); }
  return h;
}

}  // namespace lrgc

#endif  // LR_AGMPC_BRISTOL_H
