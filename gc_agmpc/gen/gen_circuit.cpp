// gen_circuit -- writes the float32 LR iteration circuits (first / next) for one dimension d, and checks them.
// Built against emp-tool 1.0 (like gc_ag2pc); the runner itself uses the legacy emp-tool that emp-agmpc needs.
//   gen_circuit --d 5 --out circuits [--check]
// --check evaluates the circuits in the clear for a few iterations and compares with reference_train() bit for bit.
#include "lr_circuit_f32.h"
#include "bristol.h"
#include "lr_agmpc_circuit.h"

#include <chrono>
#include <cstdio>
#include <cstring>

using namespace lrgc;

static Circuit build(const Params& P, bool first) {
  BristolRec R;
  const uint32_t d = (uint32_t)P.d, dd = (uint32_t)P.d_data();
  const uint32_t carried = first ? 0 : 32 * d;
  const uint32_t wbase = first ? 0 : R.input(carried);
  std::vector<uint32_t> share(kBatch), feat(kBatch), label(kBatch);
  for (int k = 0; k < kBatch; ++k) {
    if (first) share[(size_t)k] = R.input(32 * d);
    feat[(size_t)k] = R.input(32 * dd);
    label[(size_t)k] = R.input(1);
  }
  using Fl = emp::Float_T<BristolRec, 32>;
  FloatLR<BristolRec> lr(R, P.d);
  std::vector<Fl> w;
  for (uint32_t j = 0; j < d; ++j) {
    Fl f(R);
    for (uint32_t b = 0; b < 32; ++b) {
      if (first) {
        uint32_t acc = share[0] + 32 * j + b;
        for (int k = 1; k < kBatch; ++k) acc = R.xor_gate(acc, share[(size_t)k] + 32 * j + b);
        f.w[b] = acc;
      } else {
        f.w[b] = wbase + 32 * j + b;
      }
    }
    w.push_back(f);
  }
  std::vector<std::vector<Fl>> x;
  std::vector<Fl> y;
  for (int k = 0; k < kBatch; ++k) {
    std::vector<Fl> row;
    for (uint32_t j = 0; j < dd; ++j) {
      Fl f(R);
      for (uint32_t b = 0; b < 32; ++b) f.w[b] = feat[(size_t)k] + 32 * j + b;
      row.push_back(f);
    }
    x.push_back(row);
    y.push_back(lr.label(label[(size_t)k]));
  }
  std::vector<Fl> nw = lr.step(w, x, y);
  std::vector<uint32_t> outs;
  for (const auto& f : nw) for (int b = 0; b < 32; ++b) outs.push_back(f.w[(size_t)b]);
  return finalize(R, outs);
}

int main(int argc, char** argv) {
  Params P;
  std::string out = "circuits";
  bool check = false;
  try {
    ArgReader r{argc, argv, 1};
    for (; r.has(); ++r.i) {
      const std::string a = argv[r.i];
      if (parse_param_flag(r, P)) continue;
      if (a == "--out") out = r.next_value(a);
      else if (a == "--check") check = true;
      else throw std::invalid_argument("unknown option " + a);
    }
    P.finalize();
  } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
  const auto t0 = std::chrono::steady_clock::now();
  const Circuit first = build(P, true), next = build(P, false);
  std::string cmd = "mkdir -p " + out;
  if (std::system(cmd.c_str()) != 0) return 1;
  save_circuit(circuit_path(out, P.d, true), first, (uint32_t)P.d, 1);
  save_circuit(circuit_path(out, P.d, false), next, (uint32_t)P.d, 0);
  std::printf("d=%d: first iteration %llu ANDs (%zu gates), later iterations %llu ANDs (%zu gates), %.1f s\n", P.d,
              (unsigned long long)first.ands, first.gates.size(), (unsigned long long)next.ands, next.gates.size(),
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  if (!check) return 0;

  // ---- plaintext check against the reference, through the same input layout the runner uses
  const int iters = P.iters;
  const Dataset D = generate_client_data(P);
  const std::vector<float> w0 = generate_initial_weights(P);
  const auto sh = w0_shares(P, w0);
  const std::vector<float> ref = reference_train(P, D, w0);
  std::vector<uint8_t> carried;
  for (int t = 0; t < iters; ++t) {
    const IterCircuit IC = load_iteration(out, P, t == 0);
    std::vector<uint8_t> in = carried;
    for (int k = 0; k < kBatch; ++k) client_bits(in, P, D, sh, k, t);
    carried = eval_circuit(IC.C, in);
  }
  int bad = 0;
  for (int j = 0; j < P.d; ++j) {
    uint32_t u = 0;
    for (int b = 0; b < 32; ++b) u |= (uint32_t)carried[(size_t)(32 * j + b)] << b;
    if (u != f2u(ref[(size_t)j])) ++bad;
  }
  std::printf("check: circuit vs reference_train after %d iterations: %s (%d/%d weights differ)\n", iters, bad ? "MISMATCH" : "bit-exact", bad, P.d);
  return bad ? 2 : 0;
}
