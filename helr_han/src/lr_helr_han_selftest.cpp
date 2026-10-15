// lr_helr_han_selftest -- tests that need no network: the least-squares sigmoid, the wire format and its
// validation, and the homomorphic circuit against the plaintext reference (client and evaluator in one process,
// on a tiny INSECURE ring so that it runs in about a minute; the circuit, the bootstrapping schedule and the
// serialization are the real code).
//
//   ./lr_helr_han_selftest            all tests
//   ./lr_helr_han_selftest --quick    polynomial + serialization only
#include <cstdio>
#include <cstring>
#include <string>

#include "circuit.h"
#include "he.h"
#include "params.h"

using namespace han;

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static void test_poly() {
  std::printf("[least-squares sigmoid]\n");
  const Poly3 p = fit_sigmoid(8.0);
  std::printf("  R=8: a0=%.5f a1=%.5f a3=%.6f   (paper / authors' code: 0.5, -0.15012, 0.001593)\n", p.a0, p.a1, p.a3);
  CHECK(std::fabs(p.a0 - 0.5) < 1e-12 && std::fabs(p.a1 + 0.15012) < 2e-4 && std::fabs(p.a3 - 0.001593) < 3e-6, "coefficients differ from the paper's");
  // it is the least-squares optimum: perturbing a coefficient must increase the squared error
  auto err = [&](double a1, double a3) {
    double e = 0;
    for (double u = -8; u <= 8; u += 0.001) { const double d = 0.5 + a1 * u + a3 * u * u * u - 1 / (1 + std::exp(u)); e += d * d; }
    return e;
  };
  CHECK(err(p.a1, p.a3) < err(p.a1 * 1.01, p.a3) && err(p.a1, p.a3) < err(p.a1, p.a3 * 1.01) && err(p.a1, p.a3) < err(p.a1 * 0.99, p.a3 * 0.99), "not a minimum");
}

static void test_params() {
  std::printf("[parameters]\n");
  Params P;
  P.d = 5; P.threads = 12;
  P.finalize();
  std::printf("  %s\n", params_string(P).c_str());
  CHECK(P.logQ == 495 && P.logQBoot == 1155 && P.logN == 16, "paper parameters: logQ 495, logQBoot 1155, logN 16 (got %d %d %d)", P.logQ, P.logQBoot, P.logN);
  CHECK(P.cnum * P.bsz == P.factorNum && P.slots == P.batch * P.bsz, "packing");
  Params Q;
  Q.d = 1000; Q.threads = 12;
  Q.finalize();
  CHECK(Q.cnum == 8 && Q.bsz == 128 && Q.slots == 512, "d=1000 packing: %d x %d, %d slots", Q.cnum, Q.bsz, Q.slots);
  bool threw = false;
  try { Params B; B.batch = 3; B.finalize(); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw, "batch must be a power of two");
  Params S;
  S.d = 5; S.iters = 20; S.finalize();
  CHECK(final_logq(S) == S.logQ - 2 * iter_cost(S), "final level for 20 iterations (last bootstrap after iteration 17)");
}

static void test_serialization() {
  std::printf("[wire format and validation]\n");
  Ring ring(8, 80);
  SecretKey sk(ring);
  Scheme scheme(sk, ring);
  seed_ntl();
  std::vector<double> v(16);
  for (size_t i = 0; i < v.size(); ++i) v[i] = 0.05 * (double)i - 0.3;
  Ciphertext c = scheme.encrypt(v.data(), 16, 30, 60);
  std::vector<unsigned char> buf = serialize_ct(c);
  const CtShape shape{ring.N, 16, 30, 60};
  Ciphertext r = deserialize_ct(buf.data(), shape);
  std::complex<double>* d = scheme.decrypt(sk, r);
  double dev = 0;
  for (size_t i = 0; i < v.size(); ++i) dev = std::max(dev, std::fabs(d[i].real() - v[i]));
  delete[] d;
  CHECK(dev < 1e-6, "round trip deviation %g", dev);
  auto rejects = [&](const std::vector<unsigned char>& b, const CtShape& s) {
    try { Ciphertext x = deserialize_ct(b.data(), s); (void)x; } catch (const AbortError&) { return true; }
    return false;
  };
  CHECK(rejects(buf, CtShape{ring.N, 8, 30, 60}), "wrong slot count accepted");
  CHECK(rejects(buf, CtShape{ring.N, 16, 30, 59}), "wrong level accepted");
  CHECK(rejects(buf, CtShape{ring.N * 2, 16, 30, 60}), "wrong ring degree accepted");
  std::vector<unsigned char> bad = buf;
  bad[kCtHeader + ct_coef_bytes(60) - 1] |= 0xF0;  // a coefficient >= 2^60
  CHECK(rejects(bad, shape), "out-of-range coefficient accepted");
  std::printf("  round trip ok, shape / level / range checks reject\n");
}

struct Case { int d, iters, batch, cnum; double lr; uint64_t seed; bool zero; };

static void test_circuit(const Case& c, int logn) {
  Params P;
  P.d = c.d; P.iters = c.iters; P.batch = c.batch; P.lr = c.lr; P.seed = c.seed; P.zero_init = c.zero;
  P.cnum_override = c.cnum; P.logN_override = logn; P.threads = 4;
  P.finalize();
  const Poly3 H = fit_sigmoid(P.range);
  const Dataset D = generate_client_data(P);
  const RefResult ref = reference_train(P, D, H);

  Ring ring(P.logN, P.logQBoot);
  SecretKey sk(ring);
  seed_ntl();
  Scheme scheme(sk, ring);
  generate_eval_keys(P, sk, scheme);

  Evaluator ev(scheme, P, H);
  std::vector<std::vector<Ciphertext>> Z((size_t)P.iters);
  for (int t = 0; t < P.iters; ++t)
    for (int j = 0; j < P.cnum; ++j) {
      std::vector<double> v = pack_z(P, D, t, j);
      Z[(size_t)t].push_back(scheme.encrypt(v.data(), P.slots, P.wBits, P.logQ + P.wBits));
    }
  for (int j = 0; j < P.cnum; ++j) {
    std::vector<double> v = pack_w(P, D.w0, j);
    ev.W.push_back(scheme.encrypt(v.data(), P.slots, P.wBits, P.logQ));
    ev.V.push_back(scheme.encrypt(v.data(), P.slots, P.wBits, P.logQ));
  }
  const std::vector<double> eta = nag_alphas_eta(P.iters);
  int boots = 0;
  for (int t = 0; t < P.iters; ++t) {
    ev.iteration(Z[(size_t)t], eta[(size_t)t], nullptr);
    if (boot_after(P, t)) { ev.bootstrap_all(nullptr); ++boots; }
  }
  double rep = 0;
  const std::vector<double> w = decrypt_weights(scheme, sk, ev.W, P, &rep);
  double dev = 0;
  for (int j = 0; j < P.d; ++j) dev = std::max(dev, std::fabs(w[(size_t)j] - ref.w[(size_t)j]));
  const bool level_ok = ev.W[0].logq == final_logq(P);
  std::printf("  d=%-3d iters=%-2d batch=%d cnum=%d (bsz %d, %d slots) logN=%d: %d bootstraps, max |HE - plaintext| = %.2e, replica spread %.1e, final level %ld (expected %ld)\n",
              c.d, c.iters, c.batch, P.cnum, P.bsz, P.slots, P.logN, boots, dev, rep, (long)ev.W[0].logq, final_logq(P));
  CHECK(dev < 5e-3, "HE result deviates from the plaintext reference by %g", dev);
  CHECK(level_ok, "final modulus level differs from the schedule");
  CHECK(ref.max_abs_u < P.range, "test data left the polynomial's range (|u| = %g)", ref.max_abs_u);
}

int main(int argc, char** argv) {
  const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
  init_threads(4);
  test_poly();
  test_params();
  test_serialization();
  if (!quick) {
    std::printf("[homomorphic circuit vs plaintext reference (insecure ring N = 2^12, tests the code, not the security)]\n");
    const Case cases[] = {
        {5, 4, 4, 2, 0.5, 1, false},   // one bootstrapping round, two ciphertexts per vector
        {1, 4, 4, 1, 0.5, 2, false},   // one feature (bsz = 1, one-slot bootstrapping)
        {9, 5, 2, 2, 0.5, 3, true},    // batch 2, zero init, padded to 16 features
        {12, 7, 8, 4, 1.0, 4, false},  // batch 8, two bootstrapping rounds
    };
    for (const Case& c : cases) test_circuit(c, 12);
  }
  std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}
