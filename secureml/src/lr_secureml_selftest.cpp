// lr_secureml_selftest -- unit and end-to-end tests. Both parties run as two threads of this process,
// connected over loopback TCP, through exactly the code of lr_secureml (Link, Mpc, preprocessing, Online).
//
//   ./lr_secureml_selftest            full run (~1 min)
//   ./lr_secureml_selftest --quick    field, OT products, MAC check, activation branches only
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

#include "fp.h"
#include "link.h"
#include "metrics.h"
#include "mpc.h"
#include "online.h"
#include "params.h"
#include "preproc.h"

using namespace sml;

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static Fp slow_mul(Fp a, Fp b) {  // double-and-add reference
  Fp r(0), x = a;
  for (int i = 0; i < 127; ++i) { if (bit_of(b, i)) r = r + x; x = x + x; }
  return r;
}

static void test_field() {
  std::printf("[field F_p, p = 2^127 - 1]\n");
  emp::PRG prg;
  for (int it = 0; it < 2000; ++it) {
    Fp a = random_fp(prg), b = random_fp(prg), c = random_fp(prg);
    CHECK(a * b == slow_mul(a, b), "mul");
    CHECK((a + b) * c == a * c + b * c, "distrib");
    CHECK(a - a == Fp(0) && (a + (-a)).is_zero(), "neg");
  }
  Fp edge[] = {Fp(0), Fp(1), Fp(kP - 1), Fp(kP >> 1), Fp((kP >> 1) + 1)};
  for (Fp a : edge) for (Fp b : edge) CHECK(a * b == slow_mul(a, b), "edge mul");
  for (int it = 0; it < 50; ++it) {
    Fp a = random_fp(prg);
    if (a.is_zero()) continue;
    CHECK(a * inv(a) == Fp(1), "inv");
    Fp r = sqrt_qr(a * a);
    CHECK(r == a || r == -a, "sqrt");
  }
  CHECK(Fp::from_int(-5) + Fp(5) == Fp(0), "from_int");
  CHECK(Fp::from_int(-5).to_signed() == -5, "to_signed");
}

static void run_pair(int nw, bool mal, int port, const std::function<void(Link&)>& f0, const std::function<void(Link&)>& f1) {
  std::thread t([&] { Link lk(1, "127.0.0.1", port, nw, mal); f1(lk); });
  Link lk(0, "127.0.0.1", port, nw, mal);
  f0(lk);
  t.join();
}

static void test_product(int nw, bool mal, int port, int64_t J, int64_t L, bool client_sends) {
  emp::PRG prg;
  std::vector<Fp> A((size_t)(J * L)), b((size_t)J), S((size_t)(J * L)), R((size_t)(J * L));
  for (auto& x : A) x = random_fp(prg);
  for (auto& x : b) x = random_fp(prg);
  run_pair(nw, mal, port,
           [&](Link& lk) { if (client_sends) product_send(lk, J, L, A.data(), S.data()); else product_recv(lk, J, L, b.data(), R.data()); },
           [&](Link& lk) { if (client_sends) product_recv(lk, J, L, b.data(), R.data()); else product_send(lk, J, L, A.data(), S.data()); });
  bool ok = true;
  for (int64_t j = 0; j < J && ok; ++j)
    for (int64_t k = 0; k < L; ++k)
      if (S[(size_t)(j * L + k)] + R[(size_t)(j * L + k)] != A[(size_t)(j * L + k)] * b[(size_t)j]) { ok = false; break; }
  CHECK(ok, "product nw=%d mal=%d J=%lld L=%lld", nw, (int)mal, (long long)J, (long long)L);
  std::printf("  nw=%d %s J=%-4lld L=%-5lld %s\n", nw, mal ? "malicious  " : "semi-honest", (long long)J, (long long)L, ok ? "ok" : "FAIL");
}

// ------------------------------------------------------------------------------------------------
// Both parties of the real protocol, as two threads over loopback TCP.
struct PairResult {
  bool aborted[2] = {false, false};
  std::string msg[2];
  std::vector<i128> w;   // server's output
  size_t tri_left = 0;
};

static PairResult run_protocol(const Params& P, bool mal, int nw, int port, Tamper::Kind tk = Tamper::NONE, int cheater = -1) {
  PairResult R;
  auto party = [&](int me) {
    try {
      Link lk(me, "127.0.0.1", port, nw, mal);
      Mpc M(lk, mal);
      if (tk != Tamper::NONE) { M.tamper.kind = tk; M.tamper.who = cheater; }
      Recorder rec(lk);
      Pre pre;
      generate_preprocessing(M, P, pre, rec);
      Dataset D;
      if (me == 0) D = generate_client_data(P);
      Online on(M, P, pre, rec);
      on.input(me == 0 ? &D : nullptr);
      on.train(true);
      std::vector<i128> w = on.output();
      if (me == 1) { R.w = w; R.tri_left = on.triples_left(); }
    } catch (const AbortError& e) {
      R.aborted[me] = true;
      R.msg[me] = e.what();
    }
  };
  std::thread t([&] { party(1); });
  party(0);
  t.join();
  return R;
}

static Params make_params(int d, int iters, int batch, double lr, int frac, uint64_t seed, bool bias = true, bool zero = false) {
  Params P;
  P.d = d; P.iters = iters; P.batch = batch; P.lr = lr; P.frac = frac; P.seed = seed; P.bias = bias; P.zero_init = zero;
  P.finalize();
  return P;
}

static void test_end_to_end(int& port) {
  std::printf("[end-to-end: secure result == plaintext integer reference, bit for bit]\n");
  struct Cfg { int d, iters, batch; double lr; int frac; uint64_t seed; bool bias, zero; };
  const Cfg cfgs[] = {
      {5, 20, 4, 0.5, 13, 1, true, false},   // the default benchmark setting
      {1, 3, 1, 0.25, 13, 2, true, false},   // single feature (bias only)
      {7, 6, 2, 1.0, 10, 3, false, true},    // zero init, no bias column
      {16, 5, 8, 2.0, 16, 4, true, false},   // larger batch, finer fixed point
      {40, 4, 4, 0.5, 8, 5, true, false},    // more features than any gadget width
  };
  for (const Cfg& c : cfgs)
    for (bool mal : {true, false}) {
      const Params P = make_params(c.d, c.iters, c.batch, c.lr, c.frac, c.seed, c.bias, c.zero);
      PairResult R = run_protocol(P, mal, 3, port);
      port += 8;
      const Dataset D = generate_client_data(P);
      bool in_range = true;
      const std::vector<i128> ref = reference_train(P, D, &in_range);
      CHECK(!R.aborted[0] && !R.aborted[1], "unexpected abort d=%d mal=%d: %s %s", c.d, (int)mal, R.msg[0].c_str(), R.msg[1].c_str());
      CHECK(in_range, "reference left the gadget range d=%d", c.d);
      CHECK(R.w == ref, "secure result differs from the reference: d=%d iters=%d batch=%d mal=%d", c.d, c.iters, c.batch, (int)mal);
      // every triple of the pool is consumed exactly (the schedule is data independent)
      CHECK(R.tri_left == 0, "triple pool not fully consumed (%zu left)", R.tri_left);
      std::printf("  d=%-3d iters=%-2d batch=%d F=%-2d %s ok\n", c.d, c.iters, c.batch, c.frac, mal ? "malicious  " : "semi-honest");
    }
}

static void test_branches() {
  std::printf("[activation branches in the reference]\n");
  const Params P = make_params(12, 20, 4, 0.5, 13, 21);
  const Dataset D = generate_client_data(P);
  std::vector<i128> w(D.w0.begin(), D.w0.end());
  int lo = 0, mid = 0, hi = 0;
  const i128 half = (i128)1 << (P.zs - 1), one = (i128)1 << P.frac;
  for (int t = 0; t < P.iters; ++t) {
    std::vector<i128> e((size_t)P.batch);
    for (int i = 0; i < P.batch; ++i) {
      const int64_t r = (int64_t)t * P.batch + i;
      i128 z = 0;
      for (int j = 0; j < P.d; ++j) z += (i128)D.x[(size_t)(r * P.d + j)] * w[(size_t)j];
      i128 q = (z + half) >> P.tr;
      if (q < 0) { ++lo; q = 0; } else if (q > one) { ++hi; q = one; } else ++mid;
      e[(size_t)i] = q - one * D.y[(size_t)r];
    }
    for (int j = 0; j < P.d; ++j) {
      i128 g = 0;
      for (int i = 0; i < P.batch; ++i) g += e[(size_t)i] * D.x[(size_t)(((int64_t)t * P.batch + i) * P.d + j)];
      w[(size_t)j] -= g;
    }
  }
  std::printf("  samples below / inside / above the linear piece: %d / %d / %d\n", lo, mid, hi);
  CHECK(lo > 0 && mid > 0 && hi > 0, "the test data must exercise all three branches of the activation");
}

static void test_mac_check(int& port) {
  std::printf("[MAC check]\n");
  for (int tamper = 0; tamper < 2; ++tamper) {
    bool aborted[2] = {false, false};
    auto party = [&](int me) {
      try {
        Link lk(me, "127.0.0.1", port, 2, true);
        Mpc M(lk, true);
        // a shared value x = 7 with a valid MAC; both parties derive the same random sharing from a common seed
        const Fp x(7);
        emp::block seed = emp::makeBlock(123, 456);
        emp::PRG c(&seed);
        const Fp a0 = random_fp(c), al0 = random_fp(c), al1 = random_fp(c), m0 = random_fp(c);
        const Fp a1 = x - a0, m1 = (al0 + al1) * x - m0;
        M.alpha = me == 0 ? al0 : al1;
        Sh s{me == 0 ? a0 : a1, me == 0 ? m0 : m1};
        if (tamper && me == 1) s.m += Fp(1);  // a MAC share that no longer matches
        const std::vector<Fp> o = M.open({s});
        CHECK(o[0] == x, "opening");
        M.mac_check();
      } catch (const AbortError&) { aborted[me] = true; }
    };
    std::thread t([&] { party(1); });
    party(0);
    t.join();
    port += 8;
    CHECK(aborted[0] == (tamper != 0) && aborted[1] == (tamper != 0), "MAC check tamper=%d: aborted = %d/%d", tamper, (int)aborted[0], (int)aborted[1]);
    std::printf("  %s: %s\n", tamper ? "wrong MAC share" : "valid MAC      ", aborted[0] ? "rejected by both parties" : "accepted");
  }
}

static void test_tamper(int& port) {
  std::printf("[malicious security: every deviation aborts, with no output]\n");
  const Params P = make_params(4, 3, 2, 0.5, 13, 8);
  struct T { Tamper::Kind k; const char* name; } kinds[] = {
      {Tamper::OPEN_PRE, "wrong opening (offline)"}, {Tamper::OPEN_ONLINE, "wrong opening (online)"},
      {Tamper::TRIPLE_C, "bad scalar triple"},       {Tamper::MATRIX_Z, "bad matrix triple"},
      {Tamper::OT_INPUT, "inconsistent OT input"},   {Tamper::MAC_VALUE, "wrong MAC share"},
      {Tamper::MAC_SIGMA, "wrong MAC-check value"}};
  for (const T& t : kinds)
    for (int cheater = 0; cheater < 2; ++cheater) {
      PairResult R = run_protocol(P, true, 2, port, t.k, cheater);
      port += 8;
      CHECK(R.aborted[0] && R.aborted[1] && R.w.empty(), "%s by party %d was not detected / produced output", t.name, cheater);
      std::printf("  %-26s by %-6s -> abort\n", t.name, cheater == 0 ? "client" : "server");
    }
  // the same deviation in semi-honest mode is (by design) not detected: that mode has no checks
  PairResult R = run_protocol(P, false, 2, port, Tamper::OPEN_ONLINE, 1);
  port += 8;
  CHECK(!R.aborted[0] && !R.aborted[1], "semi-honest mode must not abort");
}

int main(int argc, char** argv) {
  const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
  int port = 24000 + (std::rand() % 2000) * 8;
  test_field();
  std::printf("[OT product]\n");
  test_product(1, true, port, 1, 1, true); port += 4;
  test_product(3, true, port, 5, 7, false); port += 8;
  test_product(3, true, port, 300, 1, true); port += 8;
  test_product(3, true, port, 2, 5000, true); port += 8;
  test_product(2, false, port, 40, 9, false); port += 8;
  test_product(4, true, port, 20, 8, true); port += 8;
  test_mac_check(port);
  test_branches();
  if (!quick) {
    test_end_to_end(port);
    test_tamper(port);
  }
  std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}
