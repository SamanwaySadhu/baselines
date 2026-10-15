// lr_helr_selftest -- offline tests of the HELR baseline (no TCP; the two parties run on two
// threads of this process over a socketpair). The HE parameters are tiny and INSECURE
// (logN = 10): the circuit, the serialization and the protocol are the production code, only
// the ring is smaller, so a run takes seconds instead of hours.
//
//   [1] least-squares sigmoid polynomial against the constants of the paper / HELR
//   [2] serialization of ciphertexts and keys; malformed encodings are rejected
//   [3] randomness: independent NTL streams per thread; coin-tossed public-key element
//   [4] the HE training circuit against the double-precision reference, degree 3 and 7,
//       several shapes (d, iterations, learning rate, R)
//   [5] the decryption MAC (derive_c2) and its failure cases
//   [6] the whole protocol in-process: honest run, and every deviation it must catch
//
// usage: lr_helr_selftest [--quick]
#include "helr_protocol.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <set>
#include <sstream>
#include <thread>

namespace lrhe {

namespace {

int g_checks = 0, g_failed = 0;
#define CHECK(cond, ...)                                                      \
  do {                                                                        \
    ++g_checks;                                                               \
    if (!(cond)) {                                                            \
      ++g_failed;                                                             \
      std::printf("  FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond);         \
      std::printf(__VA_ARGS__);                                               \
      std::printf("\n");                                                      \
    }                                                                         \
  } while (0)

Params make_params(int d, int iters, int batch = 4, double lr = 0.5, uint64_t seed = 1) {
  Params P;
  P.d = d;
  P.iters = iters;
  P.batch = batch;
  P.lr = lr;
  P.seed = seed;
  P.finalize();
  return P;
}

HeOptions tiny_opts(int logN = 10, int deg = 3, int logR = -1) {
  HeOptions O;
  O.logN = logN;
  O.deg = deg;
  O.logR = logR;
  O.insecure_ok = true;
  return O;
}

// A context with keys for a configuration (no coin toss: pk.ax uniform).
struct Rig {
  HeContext ctx;
  SecKey sk;
  explicit Rig(const HeConfig& c) : ctx(c), sk(ctx.params) {
    NumUtils::sampleUniform2(ctx.pk.ax, ctx.params.N, ctx.params.logq);
    keygen(ctx, sk, 1);
  }
};

// ---------------------------------------------------------------------------------- [1]
void test_poly_fit() {
  std::printf("[1] least-squares sigmoid polynomial\n");
  const HPoly p3 = fit_sigmoid_poly(3, 3);
  // paper / HELR: sigmoid(x) ~ 1/2 + a1 (x/8) + a3 (x/8)^3, a1 = 1.20096, a3 = -0.81562; H = 2h
  CHECK(std::fabs(p3.c[1] / 2 - 1.20096) < 2e-5, "a1 = %.6f", p3.c[1] / 2);
  CHECK(std::fabs(p3.c[3] / 2 + 0.81562) < 2e-5, "a3 = %.6f", p3.c[3] / 2);
  CHECK(std::fabs(p3.c[1] - 2.401926612) < 1e-4 && std::fabs(p3.c[3] + 1.631249824) < 1e-4,
        "HELR constants c1 = %.9f c3 = %.9f", p3.c[1], p3.c[3]);
  const HPoly p7 = fit_sigmoid_poly(7, 3);
  const double b[4] = {1.73496, -4.19407, 5.43402, -2.50739};  // paper, Sec. 3.1 (b7 is -2.5 in the HELR code)
  for (int k = 0; k < 4; ++k)
    CHECK(std::fabs(p7.c[2 * k + 1] / 2 - b[k]) < 5e-3, "b%d = %.6f vs %.5f", 2 * k + 1, p7.c[2 * k + 1] / 2, b[k]);  // the paper's values (printed to 5 digits) differ by <= 2e-3: it fits on a discrete grid
  {  // my coefficients are the least-squares optimum: no worse than the paper's printed ones
    auto mse = [&](const double* c) {
      double s = 0;
      const int n = 20000;
      for (int i = -n; i <= n; ++i) {
        const double w = (double)i / n, w2 = w * w;
        const double H = c[1] * w + c[3] * w * w2 + c[5] * w * w2 * w2 + c[7] * w * w2 * w2 * w2;
        const double f = std::tanh(4.0 * w);
        s += (H - f) * (H - f);
      }
      return s / (2 * n + 1);
    };
    const double paper[8] = {0, 2 * b[0], 0, 2 * b[1], 0, 2 * b[2], 0, 2 * b[3]};
    CHECK(mse(p7.c) <= mse(paper) * (1 + 1e-9), "LS fit (%.3e) is worse than the paper's coefficients (%.3e)", mse(p7.c), mse(paper));
    std::printf("      degree 7, R = 8: MSE of H vs tanh(4w): ours %.3e, paper's printed coefficients %.3e\n", mse(p7.c), mse(paper));
  }
  // quality: max error of 1/2 (1 - H(w)) against sigmoid(-R w) on [-1, 1]
  for (int logR : {3, 4, 6}) {
    const HPoly p = fit_sigmoid_poly(3, logR);
    double mx = 0;
    for (int i = -100; i <= 100; ++i) {
      const double w = i / 100.0;
      mx = std::max(mx, std::fabs(0.5 * (1 - p.eval(w)) - 1.0 / (1.0 + std::exp(p.R * w))));
    }
    std::printf("      degree 3, R = %2.0f: max |g - sigmoid| on [-R, R] = %.4f\n", p.R, mx);
    CHECK(mx < 0.45, "poor fit at R = %g: %.3f", p.R, mx);
  }
  // the coefficients both parties compute independently must agree exactly
  const HPoly q3 = fit_sigmoid_poly(3, 3);
  CHECK(std::memcmp(p3.c, q3.c, sizeof p3.c) == 0, "non-deterministic fit");
}

// ---------------------------------------------------------------------------------- [2]
void test_serialization() {
  std::printf("[2] serialization\n");
  const Params P = make_params(2, 2);
  Rig r(make_he_config(P, tiny_opts()));
  const HeConfig& C = r.ctx.cfg;
  std::vector<NTL::ZZ> v((size_t)C.nslots);
  for (size_t k = 0; k < v.size(); ++k) v[k] = scale_up(0.25 * (double)k - 1.0, C.logp);
  Cipher c = encrypt_scaled(r.ctx, v);
  const auto bytes = cipher_to_bytes(c, C.N());
  CHECK(bytes.size() == cipher_bytes(C.N(), C.logq), "size %zu", bytes.size());
  Cipher c2 = cipher_from_bytes(bytes.data(), C.N(), C.logq, C.nslots);
  CHECK(cipher_to_bytes(c2, C.N()) == bytes, "round trip");
  CHECK(decrypt_coeff(r.ctx, r.sk, c, 0) == decrypt_coeff(r.ctx, r.sk, c2, 0), "decryption differs after the round trip");
  // padding bits: logq % 8 != 0 here, so the last byte of every coefficient has spare bits
  CHECK(C.logq % 8 != 0, "test needs logq %% 8 != 0 (logq = %d)", C.logq);
  auto bad = bytes;
  bad[coeff_bytes(C.logq) - 1] |= 0x80;
  bool threw = false;
  try { cipher_from_bytes(bad.data(), C.N(), C.logq, C.nslots); } catch (const std::runtime_error&) { threw = true; }
  CHECK(threw, "padding bits accepted");
  // a ciphertext with negative / unreduced coefficients serializes into [0, 2^cbits)
  Cipher neg = c;
  neg.ax.rep[3] = NTL::ZZ(-5);
  neg.bx.rep[4] = neg.mod + NTL::ZZ(7);
  Cipher back = cipher_from_bytes(cipher_to_bytes(neg, C.N()).data(), C.N(), C.logq, C.nslots);
  CHECK(back.ax.rep[3] == neg.mod - NTL::ZZ(5) && back.bx.rep[4] == NTL::ZZ(7), "reduction of unreduced coefficients");
}

// ---------------------------------------------------------------------------------- [3]
void test_randomness() {
  std::printf("[3] randomness\n");
  std::vector<std::string> got(8);
  parallel_for(8, 4, [&](int i) {
    std::ostringstream os;
    os << NTL::RandomBits_ZZ(128);
    got[(size_t)i] = os.str();
  });
  std::set<std::string> uniq(got.begin(), got.end());
  CHECK(uniq.size() == 8, "tasks share a random stream (%zu distinct of 8)", uniq.size());
  const std::vector<uint8_t> a = os_random_bytes(32), b = os_random_bytes(32);
  CHECK(a != b, "os_random repeats");
  // coin-tossed a: deterministic, depends on both seeds
  const NTL::ZZX x = derive_pk_a(a, b, 64, 100), y = derive_pk_a(a, b, 64, 100), z = derive_pk_a(b, a, 64, 100);
  CHECK(x == y, "derive_pk_a not deterministic");
  CHECK(!(x == z), "derive_pk_a ignores the order of the seeds");
  bool in_range = true;
  for (long i = 0; i < 64; ++i) in_range &= x.rep[i] >= 0 && NTL::NumBits(x.rep[i]) <= 100;
  CHECK(in_range, "derive_pk_a out of range");
}

// ---------------------------------------------------------------------------------- [4]
double circuit_error(const Params& P, const HeOptions& O, bool verbose, double* max_w = nullptr) {
  const HeConfig C = make_he_config(P, O);
  Rig r(C);
  const Dataset D = generate_client_data(P);
  const std::vector<int64_t> w0 = generate_initial_weights(P);
  const HPoly H = fit_sigmoid_poly(C.deg, C.logR);
  EncInputs in;
  encrypt_inputs(r.ctx, P, D, w0, 1, in);
  EvalStats st;
  Evaluator ev(r.ctx, H, &st);
  std::vector<Cipher> out = ev.run(in.z, in.beta, 1);
  const std::vector<double> ref = poly_train(P, D, w0, H, max_w);
  double worst = 0;
  for (int j = 0; j < C.d; ++j) {
    CHECK(out[(size_t)j].cbits == C.out_bits, "output level %ld != %d", out[(size_t)j].cbits, C.out_bits);
    const double he = zz_to_double(decrypt_coeff(r.ctx, r.sk, out[(size_t)j])) / std::ldexp(1.0, C.logp);
    worst = std::max(worst, std::fabs(he - ref[(size_t)j]));
    // every slot holds the same weight: slot 0 via the full decoder
    if (j == 0) {
      CZZ* s = r.ctx.scheme.decrypt(r.sk, out[0]);
      const double s0 = zz_to_double(s[0].r) / std::ldexp(1.0, C.logp);
      const double sl = zz_to_double(s[C.nslots - 1].r) / std::ldexp(1.0, C.logp);
      CHECK(std::fabs(s0 - he) < 1e-3 && std::fabs(sl - he) < 1e-3, "slots disagree: %.6f %.6f %.6f", s0, sl, he);
      delete[] s;
    }
  }
  if (verbose)
    std::printf("      deg %d d=%d T=%d B=%d K=%d logR=%d logq=%d: max |HE - reference| = %.2e (ops: %" PRIu64 " mult, %" PRIu64
                " rot)\n",
                C.deg, C.d, C.T, C.B, 1 << C.k_shift, C.logR, C.logq, worst, st.ops["mult"] + st.ops["square"], st.ops["rotate"]);
  return worst;
}

void test_circuit(bool quick) {
  std::printf("[4] HE circuit vs double-precision reference (logN = 10, insecure)\n");
  struct Case { int d, T, batch; double lr; int deg, logR; };
  std::vector<Case> cases = {{2, 1, 4, 0.5, 3, 3},  {3, 3, 4, 0.5, 3, 3},  {5, 3, 4, 0.5, 3, -1},
                             {3, 2, 4, 0.5, 7, 3},  {4, 3, 4, 0.5, 7, 4},  {3, 3, 4, 1.0, 3, 3},
                             {3, 3, 4, 0.25, 3, 3}, {3, 3, 2, 0.5, 3, 3},  {3, 5, 4, 0.5, 3, 3}};
  if (quick) cases.resize(5);
  for (const Case& c : cases) {
    const Params P = make_params(c.d, c.T, c.batch, c.lr);
    double mw = 0;
    const double err = circuit_error(P, tiny_opts(10, c.deg, c.logR), true, &mw);
    CHECK(err < 2e-3, "HE circuit error %.3e (deg %d d=%d T=%d)", err, c.deg, c.d, c.T);
    CHECK(mw <= 1.0, "the test data leaves the fit interval (max |w| = %.2f)", mw);
  }
}

// ---------------------------------------------------------------------------------- [5]
void test_mac() {
  std::printf("[5] decryption MAC\n");
  const Params P = make_params(2, 2);
  Rig r(make_he_config(P, tiny_opts()));
  const HeConfig& C = r.ctx.cfg;
  const long L = C.out_bits;
  const NTL::ZZ mod = NTL::power2_ZZ(L);
  // a ciphertext at the output level holding the weight 0.75 in every slot
  Cipher c1 = encrypt_scaled(r.ctx, std::vector<NTL::ZZ>((size_t)C.nslots, scale_up(0.75, C.logp)));
  r.ctx.scheme.modEmbedAndEqual(c1, C.logq - L);
  const PkAtLevel pk = pk_at_level(r.ctx, L);
  const NTL::ZZ E = enc_noise_bound(r.ctx);
  int accepted = 0, rejected_shift = 0, rejected_alpha = 0, same = 0;
  const int trials = 40;
  for (int t = 0; t < trials; ++t) {
    const auto seed = os_random_bytes(32);
    MacSecret s;
    const Cipher c2 = derive_c2(r.ctx, pk, c1, seed, &s);
    same += cipher_to_bytes(derive_c2(r.ctx, pk, c1, seed, nullptr), C.N()) == cipher_to_bytes(c2, C.N());
    const NTL::ZZ u1 = decrypt_coeff(r.ctx, r.sk, c1), u2 = decrypt_coeff(r.ctx, r.sk, c2);
    CHECK(NTL::abs(u1 - scale_up(0.75, C.logp)) < 4096, "decryption of c1 off by %g", zz_to_double(u1 - scale_up(0.75, C.logp)));
    // honest
    NTL::ZZ D = center(u2 - s.alpha * u1 - s.r0, mod);
    accepted += NTL::abs(D) <= E;
    // a client that changes the weight by delta but cannot adjust u2
    for (long dlt : {1L, 7L, 1L << 12, 1L << 20}) {
      D = center(u2 - s.alpha * (u1 + dlt) - s.r0, mod);
      rejected_shift += NTL::abs(D) > E;
    }
    // a client that guesses alpha wrongly: u2' = alpha' (u1 + delta) + (u2 - alpha u1)
    const NTL::ZZ alpha_guess = (s.alpha + NTL::RandomBits_ZZ(L)) | NTL::ZZ(1);
    const NTL::ZZ delta(1000);
    const NTL::ZZ u2f = u2 + alpha_guess * delta;  // what the client would send
    D = center(u2f - s.alpha * (u1 + delta) - s.r0, mod);
    rejected_alpha += NTL::abs(D) > E;
    // u2 is uniform: it does not reveal alpha (it is not, e.g., small)
    CHECK(NTL::abs(center(u2, mod)) > NTL::power2_ZZ(L - 12) || t > 4, "u2 suspiciously small");
  }
  CHECK(accepted == trials, "honest MAC rejected (%d / %d)", accepted, trials);
  CHECK(rejected_shift == 4 * trials, "forged shifts accepted (%d / %d rejected)", rejected_shift, 4 * trials);
  CHECK(rejected_alpha == trials, "wrong-alpha forgery accepted (%d / %d rejected)", rejected_alpha, trials);
  CHECK(same == trials, "derive_c2 is not deterministic in the seed");
  // the noise bound is not vacuous: the honest |D| is far below E and E << 2^L
  CHECK(E < NTL::power2_ZZ(30), "noise bound too large");
  // different seeds give different (alpha, r)
  MacSecret a, b;
  derive_c2(r.ctx, pk, c1, os_random_bytes(32), &a);
  derive_c2(r.ctx, pk, c1, os_random_bytes(32), &b);
  CHECK(a.alpha != b.alpha && a.r0 != b.r0 && NTL::bit(a.alpha, 0) == 1 && NTL::bit(b.alpha, 0) == 1, "alpha/r repeat or alpha even");
  std::printf("      %d honest checks accepted, %d forged shifts and %d wrong-alpha forgeries rejected; E = 2^%.1f, L = %ld\n",
              accepted, rejected_shift, rejected_alpha, std::log2(zz_to_double(E)), L);
}

// ---------------------------------------------------------------------------------- [6]
struct RunResult {
  bool client_ok = false, server_ok = false;
  std::string client_err, server_err;
  std::vector<double> weights, expected;
  uint64_t c2s = 0, s2c = 0;
};

RunResult run_protocol(const Params& P, const HeOptions& O, Tamper tc = {}, Tamper ts = {}, const Params* server_P = nullptr,
                       double tol = -1) {
  RunResult R;
  auto [cch, sch] = Channel::pair();
  const HeOptions Os = O;
  std::thread server([&] {
    try {
      seed_ntl_from_os();
      HeServer S(server_P ? *server_P : P, Os, ts);
      S.setup();
      S.coin_toss(*sch);
      S.receive(*sch);
      S.evaluate();
      S.output(*sch);
      R.weights = S.weights();
      R.server_ok = true;
      R.s2c = sch->bytes_sent();
    } catch (const std::exception& e) {
      R.server_err = e.what();
    }
    sch.reset();
  });
  std::thread client([&] {
    try {
      seed_ntl_from_os();
      HeClient C(P, O, generate_client_data(P), generate_initial_weights(P), tc);
      if (tol > 0) C.tol = tol;
      C.coin_toss(*cch);
      C.setup();
      C.upload(*cch);
      C.output(*cch);
      R.expected = C.expected();
      R.client_ok = true;
      R.c2s = cch->bytes_sent();
    } catch (const std::exception& e) {
      R.client_err = e.what();
    }
    cch.reset();
  });
  client.join();
  server.join();
  return R;
}

bool contains(const std::string& s, const char* sub) { return s.find(sub) != std::string::npos; }

void test_protocol(bool quick) {
  std::printf("[6] full protocol in-process (logN = 10, insecure)\n");
  for (int deg : {3, 7}) {
    if (quick && deg == 7) break;
    const Params P = make_params(3, 3);
    const RunResult R = run_protocol(P, tiny_opts(10, deg));
    CHECK(R.client_ok && R.server_ok, "honest run failed: client [%s] server [%s]", R.client_err.c_str(), R.server_err.c_str());
    if (R.client_ok && R.server_ok) {
      double worst = 0;
      for (size_t j = 0; j < R.weights.size(); ++j) worst = std::max(worst, std::fabs(R.weights[j] - R.expected[j]));
      CHECK(worst < 2e-3, "server weights differ from the plaintext run by %.3e", worst);
      std::printf("      deg %d honest run: server weights within %.2e of the client's plaintext run; c->s %.1f KiB, s->c %.1f KiB\n",
                  deg, worst, R.c2s / 1024.0, R.s2c / 1024.0);
      const HeConfig C = make_he_config(P, tiny_opts(10, deg));
      const CommLayout L = comm_layout(C);
      CHECK(R.c2s == L.offline_c2s() + L.online_c2s(), "client bytes %" PRIu64 " != analytic %" PRIu64, R.c2s,
            L.offline_c2s() + L.online_c2s());
      CHECK(R.s2c == L.offline_s2c() + L.online_s2c(), "server bytes %" PRIu64 " != analytic %" PRIu64, R.s2c,
            L.offline_s2c() + L.online_s2c());
    }
  }
  const Params P = make_params(2, 2);
  {  // the client lies about a weight (and commits to the lie): the MAC catches it
    const RunResult R = run_protocol(P, tiny_opts(), Tamper{"opening"});
    CHECK(!R.server_ok && contains(R.server_err, "MAC check failed"), "client lie not caught: [%s]", R.server_err.c_str());
  }
  {  // the client opens a value other than the one it committed to
    const RunResult R = run_protocol(P, tiny_opts(), Tamper{"commit"});
    CHECK(!R.server_ok && contains(R.server_err, "commitment"), "commitment break not caught: [%s]", R.server_err.c_str());
  }
  {  // the server returns a wrong weight: the client refuses to decrypt / release
    const RunResult R = run_protocol(P, tiny_opts(), {}, Tamper{"c1"});
    CHECK(!R.client_ok && contains(R.client_err, "differ from the client's own computation"), "server c1 not caught: [%s]",
          R.client_err.c_str());
    CHECK(!R.server_ok && R.weights.empty(), "server got an output although the client aborted");
  }
  {  // the server sends a c2 that is not the honest MAC ciphertext (a decryption oracle probe)
    const RunResult R = run_protocol(P, tiny_opts(), {}, Tamper{"c2"});
    CHECK(!R.client_ok && contains(R.client_err, "not the honest MAC ciphertext"), "server c2 not caught: [%s]", R.client_err.c_str());
    CHECK(!R.server_ok && R.weights.empty(), "server got an output although the client aborted");
  }
  {  // a client tolerance that is too tight aborts an honest run (and releases nothing)
    const RunResult R = run_protocol(P, tiny_opts(), {}, {}, nullptr, 1e-12);
    CHECK(!R.client_ok && !R.server_ok && contains(R.client_err, "verification failed"), "tight tolerance: [%s]",
          R.client_err.c_str());
  }
  {  // disagreeing public parameters are caught at the first message
    const Params Q = make_params(2, 3);
    const RunResult R = run_protocol(P, tiny_opts(), {}, {}, &Q);
    CHECK(!R.server_ok && contains(R.server_err, "disagree on iters"), "parameter mismatch: [%s]", R.server_err.c_str());
  }
}

}  // namespace
}  // namespace lrhe

int main(int argc, char** argv) {
  using namespace lrhe;
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--quick")) quick = true;
    else { std::printf("usage: %s [--quick]\n", argv[0]); return 2; }
  }
  seed_ntl_from_os();
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  try {
    test_poly_fit();
    test_serialization();
    test_randomness();
    test_circuit(quick);
    test_mac();
    test_protocol(quick);
  } catch (const std::exception& e) {
    std::printf("EXCEPTION: %s\n", e.what());
    return 1;
  }
  std::printf("\n%d checks, %d failed\n", g_checks, g_failed);
  return g_failed ? 1 : 0;
}
