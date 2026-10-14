// lr_mdml_selftest -- in-process tests: dealer + both parties (two threads, real
// sockets on localhost).
//   1. LTZ gate on edge values (0, +-1, +-2^62, INT64_MIN/MAX, random)
//   2. element-wise multiply, linear maps, matmul-then-trunc against exact values
//   3. full LR training, bit-exact against the protocol emulation, for several d
//   4. malicious client (corrupted arithmetic / binary share): the MAC check must fail
//
// usage: lr_mdml_selftest [--quick] [--port N]

#include "lr_circuit.h"
#include "lr_params.h"
#include "mdml_dealer.h"
#include "mdml_ext.h"

#include <cstdio>
#include <exception>
#include <filesystem>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace lrmd;

namespace {

int g_port = 41000;
int g_fail = 0;
std::string g_dir;

struct RunResult {
  bool mac_ok[2] = {false, false};
  std::vector<u64> out;                          // server's output
  std::vector<std::vector<u64>> trunc_lambdas;   // dealer's
  std::string error;
};

// build(be) -> output wire (via be.output_to); feed(inputs) sets the client's inputs.
template <class Build>
RunResult run_two_party(const std::string& job, Build build,
                        const std::function<void(std::vector<std::shared_ptr<InputGateU64>>&)>& feed,
                        Tamper tamper = Tamper::NONE) {
  RunResult R;
  const std::string base = (std::filesystem::path(g_dir) / job).string();
  constexpr uint64_t kFp = 42;
  {
    FakePartyT fp(base);
    DealerCtx dc(fp, kFp);
    DealerBackend be{dc};
    build(be)->runOffline();
    R.trunc_lambdas = dc.trunc_lambdas;
  }
  const int port = g_port;
  g_port += 4;
  std::string err[2];
  auto party_main = [&](int id) {
    try {
      MdParty party((size_t)id, 2, (size_t)port, base);
      const auto hdr = party.ReadClear(2);
      if (hdr[1] != kFp) throw std::runtime_error("bad header");
      OnlineCtx ctx(party, hdr[0]);
      ctx.tamper = tamper;
      OnlineBackend be{ctx};
      auto out = build(be);
      out->readOfflineFromFile();
      if (!party.input_file()) throw std::runtime_error("preprocessing file truncated");
      if (id == 0) feed(be.inputs);
      for (auto& g : be.inputs) g->RunOnline();
      out->RunOnline();
      R.mac_ok[id] = mac_check(ctx);
      if (id == 1) R.out = be.output->result();
    } catch (const std::exception& e) {
      err[id] = e.what();
    }
  };
  std::thread t0(party_main, 0), t1(party_main, 1);
  t0.join();
  t1.join();
  R.error = err[0].empty() ? err[1] : err[0];
  return R;
}

void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what.c_str());
  if (!ok) ++g_fail;
}

void test_ltz() {
  std::vector<u64> xs = {0, 1, (u64)-1, 2, (u64)-2, u64(1) << 62, (u64)-(int64_t(1) << 62), (u64)INT64_MAX,
                         (u64)INT64_MIN, u64(1) << 15, (u64)-(int64_t(1) << 15), 0x0123456789abcdefULL};
  std::mt19937_64 rng(7);
  for (int i = 0; i < 52; ++i) xs.push_back(rng() >> (rng() % 64));
  for (int i = 0; i < 16; ++i) xs.push_back((u64) - (int64_t)(rng() >> (1 + rng() % 63)));
  const size_t n = xs.size();
  auto build = [n](auto& be) {
    auto x = be.input(n, 1, kClient);
    return be.output_to(be.geqz(x), kServer);
  };
  auto R = run_two_party("t_ltz", build, [&](auto& in) { in[0]->setInput(xs); });
  size_t bad = 0;
  for (size_t i = 0; i < n && R.error.empty(); ++i) bad += R.out[i] != (u64)((int64_t)xs[i] >= 0);
  check(R.error.empty() && R.mac_ok[0] && R.mac_ok[1] && bad == 0,
        "LTZ: " + std::to_string(n) + " values (edge cases + random), result == (x >= 0)" + (R.error.empty() ? "" : " -- " + R.error));
}

void test_arith() {
  // out = [ (A*B) >> 8 ; a .* b + 3 a^T-ish linear ]: matmul-trunc, elementwise, linear with constant
  const size_t r = 3, m = 5, c = 2;
  std::mt19937_64 rng(11);
  auto small = [&] { return (u64)((int64_t)(rng() % 2001) - 1000); };
  std::vector<u64> A(r * m), Bm(m * c), a(r * c), b(r * c);
  for (auto* v : {&A, &Bm, &a, &b}) for (auto& x : *v) x = small();
  const int shift = 8;
  auto build = [=](auto& be) {
    auto xA = be.input(r, m, kClient), xB = be.input(m, c, kClient);
    auto xa = be.input(r, c, kClient), xb = be.input(r, c, kClient);
    auto p = be.matmul(xA, xB, shift, C_FWD);
    auto q = be.elem_mul(xa, xb);
    LinFn comb = [](const std::vector<u128>& x, const std::vector<u128>& y) {
      std::vector<u128> o(2 * x.size());
      for (size_t i = 0; i < x.size(); ++i) { o[i] = x[i]; o[x.size() + i] = 3 * y[i] - x[i]; }
      return o;
    };
    std::vector<u64> k(2 * r * c, 0);
    for (size_t i = r * c; i < 2 * r * c; ++i) k[i] = 7;
    return be.output_to(be.linear(p, q, 2 * r * c, 1, comb, k), kServer);
  };
  auto R = run_two_party("t_arith", build, [&](auto& in) {
    in[0]->setInput(A); in[1]->setInput(Bm); in[2]->setInput(a); in[3]->setInput(b);
  });
  bool ok = R.error.empty() && R.mac_ok[0] && R.mac_ok[1];
  if (ok) {
    const auto& lam = R.trunc_lambdas.at(0);
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j) {
        u64 s = 0;
        for (size_t k = 0; k < m; ++k) s += A[i * m + k] * Bm[k * c + j];
        const u64 l = lam[i * c + j];
        const u64 p = ((s + l) >> shift) - (l >> shift);       // probabilistic truncation
        const int64_t exact = (int64_t)s >> shift;            // |error| <= 1 vs. floor
        ok &= R.out[i * c + j] == p && ((int64_t)p - exact == 0 || (int64_t)p - exact == 1);
        const u64 q = a[i * c + j] * b[i * c + j];
        ok &= R.out[r * c + i * c + j] == 3 * q - p + 7;
      }
  }
  check(ok, "matmul-then-trunc, element-wise mult, linear map + constant" + (R.error.empty() ? "" : " -- " + R.error));
}

void test_lr(int d, int iters, Tamper tamper = Tamper::NONE) {
  Params P;
  P.d = d;
  P.iters = iters;
  P.seed = 1000 + d;
  P.finalize();
  const auto D = generate_client_data(P);
  const auto w0q = generate_initial_weights(P);
  const auto I = encode_client_inputs(P, D, w0q);
  auto build = [&P](auto& be) { return build_lr(be, P).out; };
  auto R = run_two_party("t_lr" + std::to_string(d), build,
                         [&](auto& in) {
                           // creation order: w0, then (X_t, Y_t) per iteration
                           in[0]->setInput(I.w0);
                           for (int t = 0; t < P.iters; ++t) {
                             in[1 + 2 * t]->setInput(I.X[t]);
                             in[2 + 2 * t]->setInput(I.Y[t]);
                           }
                         },
                         tamper);
  const std::string tag = "LR d=" + std::to_string(d) + " iters=" + std::to_string(iters);
  if (tamper != Tamper::NONE) {
    check(R.error.empty() && !R.mac_ok[0] && !R.mac_ok[1],
          tag + ": client corrupts one " + (tamper == Tamper::ARITH ? "arithmetic" : "binary") +
              " share -> both parties' MAC check fails" + (R.error.empty() ? "" : " -- " + R.error));
    return;
  }
  bool ok = R.error.empty() && R.mac_ok[0] && R.mac_ok[1];
  double err = 0;
  if (ok) {
    const auto expect = emulate_protocol(P, I, R.trunc_lambdas);
    const auto ref = reference_double(P, D, w0q);
    for (int j = 0; j < d; ++j) {
      ok &= R.out[j] == expect[j];
      err = std::max(err, std::fabs(fx_to_double(P, R.out[j]) - ref[j]));
    }
  }
  char buf[160];
  std::snprintf(buf, sizeof buf, "%s: bit-exact vs. emulation, MAC check passes (max |w - w_double| = %.2g)",
                tag.c_str(), err);
  check(ok, buf + (R.error.empty() ? std::string() : " -- " + R.error));
}

}  // namespace

int main(int argc, char** argv) {
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--quick") quick = true;
    else if (a == "--port" && i + 1 < argc) g_port = std::stoi(argv[++i]);
    else { std::fprintf(stderr, "usage: %s [--quick] [--port N]\n", argv[0]); return 1; }
  }
  g_dir = (std::filesystem::temp_directory_path() / ("lr_mdml_selftest_" + std::to_string(::getpid()))).string();
  std::filesystem::create_directories(g_dir);
  std::printf("lr_mdml_selftest (files in %s)\n", g_dir.c_str());
  std::printf("LTZ circuit: %d AND gates per comparison (%d-bit carry)\n", ltz_ands(), kRingBits - 1);

  test_ltz();
  test_arith();
  for (int d : quick ? std::vector<int>{2, 5} : std::vector<int>{1, 2, 5, 17, 64}) test_lr(d, quick ? 5 : 20);
  test_lr(5, 3, Tamper::ARITH);
  test_lr(5, 3, Tamper::BIN);

  std::filesystem::remove_all(g_dir);
  std::printf(g_fail ? "%d test(s) FAILED\n" : "all tests passed\n", g_fail);
  return g_fail ? 1 : 0;
}
