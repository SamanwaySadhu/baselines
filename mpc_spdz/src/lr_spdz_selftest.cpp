// lr_spdz_selftest -- unit and end-to-end tests.
//
//   * field, OT products, the n-party MAC check and the offline phase run as threads of this process
//     (one Mesh per party, loopback TCP), through exactly the code of lr_spdz;
//   * end-to-end runs and every deviation run as one FORKED PROCESS per party (the real driver), because a
//     party that aborts closes its sockets and a peer blocked on them terminates its whole process.
//
//   ./lr_spdz_selftest --quick    field, OT products, MAC check, offline-phase consistency (a few seconds)
//   ./lr_spdz_selftest            + end-to-end configurations in both modes and every deviation (~2 min)
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "driver.h"
#include "fp.h"
#include "mesh.h"
#include "metrics.h"
#include "mpc.h"
#include "online.h"
#include "params.h"
#include "preproc.h"

using namespace spz;

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// Every run gets its own block of ports. They wrap around inside [20000, 31800): above that, Linux hands out
// ephemeral source ports (32768+) that could collide with a listener.
static int next_port_block(int& port) {
  const int p = port;
  port += 600;
  if (port > 31200) port = 20000 + (port % 600);
  return p;
}

static std::vector<std::string> localhost(int n) { return std::vector<std::string>((size_t)n, "127.0.0.1"); }

// ---------------------------------------------------------------------------------------------------
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

// n parties as threads, each with its own Mesh.
static void run_parties(int n, int nw, bool mal, int port, const std::function<void(int, Mesh&)>& f) {
  std::vector<std::thread> th;
  std::vector<std::exception_ptr> err((size_t)n);
  auto party = [&](int me) {
    try {
      Mesh mesh(me, localhost(n), port, nw, mal);
      f(me, mesh);
    } catch (...) { err[(size_t)me] = std::current_exception(); }
  };
  for (int p = 1; p < n; ++p) th.emplace_back(party, p);
  party(0);
  for (auto& t : th) t.join();
  for (auto& e : err) if (e) std::rethrow_exception(e);
}

static void test_product(int nw, bool mal, int port, int64_t J, int64_t L) {
  emp::PRG prg;
  std::vector<Fp> A((size_t)(J * L)), b((size_t)J), S((size_t)(J * L)), R((size_t)(J * L));
  for (auto& x : A) x = random_fp(prg);
  for (auto& x : b) x = random_fp(prg);
  run_parties(2, nw, mal, port, [&](int me, Mesh& m) {
    if (me == 0) product_send(m.peers[1], J, L, A.data(), S.data());
    else product_recv(m.peers[0], J, L, b.data(), R.data());
  });
  bool ok = true;
  for (int64_t j = 0; j < J && ok; ++j)
    for (int64_t k = 0; k < L; ++k)
      if (S[(size_t)(j * L + k)] + R[(size_t)(j * L + k)] != A[(size_t)(j * L + k)] * b[(size_t)j]) { ok = false; break; }
  CHECK(ok, "product nw=%d mal=%d J=%lld L=%lld", nw, (int)mal, (long long)J, (long long)L);
  std::printf("  nw=%d %s J=%-4lld L=%-5lld %s\n", nw, mal ? "malicious  " : "semi-honest", (long long)J, (long long)L, ok ? "ok" : "FAIL");
}

// ---------------------------------------------------------------------------------------------------
// The n-party MAC check: a valid sharing is accepted; a wrong MAC share anywhere, a wrong opened value
// at ONE party, and a wrong sigma are rejected, at every party.
static void test_mac_check(int n, int& port) {
  std::printf("[MAC check, %d parties]\n", n);
  struct Case { const char* name; int kind; };  // 0 valid, 1 wrong MAC share, 2 equivocated opening, 3 wrong sigma
  for (Case c : {Case{"valid", 0}, Case{"wrong MAC share of one party", 1}, Case{"one party sends a different share to one peer", 2}, Case{"wrong sigma", 3}}) {
    std::vector<int> aborted((size_t)n, 0);
    run_parties(n, 1, true, port, [&](int me, Mesh& mesh) {
      Mpc M(mesh, true);
      // a shared x = 7 with a valid MAC under key alpha = sum alpha_i; every party derives the same sharing from a common seed
      const Fp x(7);
      emp::block seed = emp::makeBlock(123, 456);
      emp::PRG cp(&seed);
      std::vector<Fp> al((size_t)n), xs((size_t)n), ms((size_t)n);
      Fp asum(0), xsum(0), msum(0);
      for (int i = 0; i < n; ++i) al[(size_t)i] = random_fp(cp), asum += al[(size_t)i];
      for (int i = 0; i + 1 < n; ++i) { xs[(size_t)i] = random_fp(cp); ms[(size_t)i] = random_fp(cp); xsum += xs[(size_t)i]; msum += ms[(size_t)i]; }
      xs[(size_t)n - 1] = x - xsum;
      ms[(size_t)n - 1] = asum * x - msum;
      M.alpha = al[(size_t)me];
      Sh s{xs[(size_t)me], ms[(size_t)me]};
      if (c.kind == 1 && me == n - 1) s.m += Fp(1);
      if (c.kind == 2) { M.tamper.kind = Tamper::OPEN_EQUIV; M.tamper.who = 1; M.online_phase = true; }
      if (c.kind == 3) { M.tamper.kind = Tamper::MAC_SIGMA; M.tamper.who = 1; }
      try {
        const std::vector<Fp> o = M.open({s});
        if (c.kind != 2) CHECK(o[0] == x, "opening");
        M.mac_check();
      } catch (const AbortError&) { aborted[(size_t)me] = 1; }
    });
    (void)next_port_block(port);
    bool all = true, none = true;
    for (int a : aborted) { all = all && a; none = none && !a; }
    CHECK(c.kind == 0 ? none : all, "MAC check '%s': aborted = %d of %d parties", c.name, (int)std::count(aborted.begin(), aborted.end(), 1), n);
    std::printf("  %-46s %s\n", c.name, all ? "rejected by every party" : (none ? "accepted" : "MIXED"));
  }
}

// ---------------------------------------------------------------------------------------------------
// The offline phase itself: every triple, matrix triple and bit must satisfy its relation, and the MACs
// must verify, whatever the number of parties.
static Params make_params(int d, int iters, int batch, double lr, int frac, uint64_t seed, bool bias = true, bool zero = false) {
  Params P;
  P.d = d; P.iters = iters; P.batch = batch; P.lr = lr; P.frac = frac; P.seed = seed; P.bias = bias; P.zero_init = zero;
  P.finalize();
  return P;
}

static void test_offline(int& port, int batch, bool mal) {
  const Params P = make_params(3, 2, batch, batch / 8.0, 13, 5);
  const int n = batch + 1;
  bool ok_triples = true, ok_bits = true, ok_mat = true, ok_masks = true;
  run_parties(n, 1, mal, port, [&](int me, Mesh& mesh) {
    Mpc M(mesh, mal);
    Recorder rec(mesh);
    Pre pre;
    generate_preprocessing(M, P, pre, rec);
    // open everything in one go (this also queues the MAC checks)
    const size_t NT = (size_t)pre.NT, NB = (size_t)pre.NB;
    std::vector<Sh> all;
    for (size_t k = 0; k < NT; ++k) { all.push_back(pre.ta[k]); all.push_back(pre.tb[k]); all.push_back(pre.tc[k]); }
    for (size_t k = 0; k < NB; ++k) all.push_back(pre.bits[k]);
    const size_t o_u = all.size();
    all.insert(all.end(), pre.U.begin(), pre.U.end());
    const size_t o_vf = all.size();
    all.insert(all.end(), pre.Vf.begin(), pre.Vf.end());
    all.insert(all.end(), pre.Vb.begin(), pre.Vb.end());
    const size_t o_zf = all.size();
    all.insert(all.end(), pre.Zf.begin(), pre.Zf.end());
    all.insert(all.end(), pre.Zb.begin(), pre.Zb.end());
    const size_t o_m = all.size();
    all.insert(all.end(), pre.Rw.begin(), pre.Rw.end());
    all.insert(all.end(), pre.Ry.begin(), pre.Ry.end());
    all.insert(all.end(), pre.Rout.begin(), pre.Rout.end());
    const std::vector<Fp> v = M.open(all);
    M.mac_check();
    if (me != 0) return;
    for (size_t k = 0; k < NT; ++k) if (v[3 * k] * v[3 * k + 1] != v[3 * k + 2]) ok_triples = false;
    for (size_t k = 0; k < NB; ++k) if (v[3 * NT + k].v > 1) ok_bits = false;
    const int64_t d = P.d, B = P.batch, T = P.iters;
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t i = 0; i < B; ++i) {  // Zf_t[i] = sum_j U[tB+i][j] Vf_t[j]
        Fp acc(0);
        for (int64_t j = 0; j < d; ++j) acc += v[o_u + (size_t)((t * B + i) * d + j)] * v[o_vf + (size_t)(t * d + j)];
        if (acc != v[o_zf + (size_t)(t * B + i)]) ok_mat = false;
      }
      for (int64_t j = 0; j < d; ++j) {  // Zb_t[j] = sum_i U[tB+i][j] Vb_t[i]
        Fp acc(0);
        for (int64_t i = 0; i < B; ++i) acc += v[o_u + (size_t)((t * B + i) * d + j)] * v[o_vf + (size_t)(T * d + t * B + i)];
        if (acc != v[o_zf + (size_t)(T * B + t * d + j)]) ok_mat = false;
      }
    }
    // the masks: each is known to its owner alone, so its opened value must equal the owner's plain value ... checked
    // indirectly by the end-to-end tests; here only that they opened and verified.
    (void)o_m;
    (void)ok_masks;
  });
  (void)next_port_block(port);
  CHECK(ok_triples, "scalar triples c = a*b (batch=%d mal=%d)", batch, (int)mal);
  CHECK(ok_bits, "random bits are bits (batch=%d mal=%d)", batch, (int)mal);
  CHECK(ok_mat, "matrix triples (batch=%d mal=%d)", batch, (int)mal);
  std::printf("  %d parties %s: triples %s, bits %s, matrix triples %s, MACs verified\n", n, mal ? "malicious  " : "semi-honest",
              ok_triples ? "ok" : "FAIL", ok_bits ? "ok" : "FAIL", ok_mat ? "ok" : "FAIL");
}

// ---------------------------------------------------------------------------------------------------
// End to end and deviations: one forked process per party, running the real driver.
struct ForkResult {
  std::vector<int> rc;
  std::vector<std::string> out, err;
};

static std::string slurp(const std::string& path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static ForkResult run_forked(const Params& P, bool mal, int port, Tamper::Kind tk = Tamper::NONE, int cheater = -1, bool verify = true) {
  const int n = P.batch + 1;
  const auto dir = std::filesystem::temp_directory_path();
  std::vector<pid_t> pids;
  std::fflush(stdout);
  std::fflush(stderr);
  for (int p = 0; p < n; ++p) {
    const pid_t pid = fork();
    if (pid == 0) {
      const std::string o = (dir / ("spz_st_" + std::to_string(getppid()) + "_" + std::to_string(p) + ".out")).string();
      const std::string e = (dir / ("spz_st_" + std::to_string(getppid()) + "_" + std::to_string(p) + ".err")).string();
      if (!std::freopen(o.c_str(), "w", stdout) || !std::freopen(e.c_str(), "w", stderr)) _exit(99);
      std::signal(SIGPIPE, SIG_IGN);
      PartyConfig C;
      C.P = P;
      C.party = p;
      C.hosts = localhost(n);
      C.port = port;
      C.threads = 1;
      C.semi_honest = !mal;
      C.verify = verify && p == 0;
      C.quiet = true;
      C.tamper = tk;
      C.cheater = cheater;
      const int rc = run_party(C);
      std::fflush(stdout);
      std::fflush(stderr);
      _exit(rc);
    }
    pids.push_back(pid);
  }
  ForkResult R;
  for (int p = 0; p < n; ++p) {
    int st = 0;
    waitpid(pids[(size_t)p], &st, 0);
    R.rc.push_back(WIFEXITED(st) ? WEXITSTATUS(st) : 100 + WTERMSIG(st));
    const std::string o = (dir / ("spz_st_" + std::to_string(getpid()) + "_" + std::to_string(p) + ".out")).string();
    const std::string e = (dir / ("spz_st_" + std::to_string(getpid()) + "_" + std::to_string(p) + ".err")).string();
    R.out.push_back(slurp(o));
    R.err.push_back(slurp(e));
    std::filesystem::remove(o);
    std::filesystem::remove(e);
  }
  return R;
}

static void test_end_to_end(int& port) {
  std::printf("[end-to-end: secure result == plaintext integer reference, bit for bit]\n");
  struct Cfg { int d, iters, batch; double lr; int frac; uint64_t seed; bool bias, zero; };
  const Cfg cfgs[] = {
      {5, 20, 4, 0.5, 13, 1, true, false},    // the default benchmark setting: 4 clients, 20 iterations, batch 4
      {1, 3, 1, 0.25, 13, 2, true, false},    // one client, bias only
      {7, 6, 2, 1.0, 10, 3, false, true},     // two clients, zero init, no bias column
      {9, 5, 3, 0.375, 12, 6, true, false},   // three clients
      {40, 4, 4, 0.5, 8, 5, true, false},     // more features than any gadget width
  };
  for (const Cfg& c : cfgs)
    for (bool mal : {true, false}) {
      const Params P = make_params(c.d, c.iters, c.batch, c.lr, c.frac, c.seed, c.bias, c.zero);
      const ForkResult R = run_forked(P, mal, port);
      (void)next_port_block(port);
      bool ok = true;
      for (int rc : R.rc) ok = ok && rc == 0;
      CHECK(ok, "end-to-end d=%d iters=%d batch=%d mal=%d: exit codes %d %d %d %d %d (server out: %s)", c.d, c.iters, c.batch, (int)mal,
            R.rc[0], R.rc.size() > 1 ? R.rc[1] : -1, R.rc.size() > 2 ? R.rc[2] : -1, R.rc.size() > 3 ? R.rc[3] : -1, R.rc.size() > 4 ? R.rc[4] : -1,
            (R.err[0] + R.out[0]).substr(0, 300).c_str());
      CHECK(R.out[0].find("EXACT MATCH") != std::string::npos, "server did not report an exact match");
      std::printf("  d=%-3d iters=%-2d clients=%d F=%-2d %s %s\n", c.d, c.iters, c.batch, c.frac, mal ? "malicious  " : "semi-honest", ok ? "exact" : "FAIL");
    }
}

static void test_tamper(int& port) {
  std::printf("[malicious security: every deviation aborts, with no output]\n");
  const Params P = make_params(3, 2, 4, 0.5, 13, 8);
  struct K { Tamper::Kind k; const char* name; bool clients_only; } kinds[] = {
      {Tamper::OPEN_PRE, "wrong opening (offline)", false}, {Tamper::OPEN_ONLINE, "wrong opening (online)", false},
      {Tamper::OPEN_EQUIV, "equivocated opening", false},   {Tamper::TRIPLE_C, "bad scalar triple", false},
      {Tamper::MATRIX_Z, "bad matrix triple", false},       {Tamper::OT_SEND, "inconsistent OT message", true},
      {Tamper::OT_CHOOSE, "inconsistent OT choice", false}, {Tamper::MAC_VALUE, "wrong MAC share", false},
      {Tamper::MAC_SIGMA, "wrong MAC-check value", false},  {Tamper::INPUT_EQUIV, "equivocated client input", true}};
  for (const K& t : kinds)
    for (int cheater : {0, 1, 4}) {
      if (t.clients_only && cheater == 0) continue;
      const ForkResult R = run_forked(P, true, port, t.k, cheater, false);
      (void)next_port_block(port);
      bool all_nonzero = true, any_abort = false;
      for (size_t p = 0; p < R.rc.size(); ++p) {
        all_nonzero = all_nonzero && R.rc[p] != 0;
        if (R.rc[p] == 3 && R.err[p].find("ABORT") != std::string::npos) any_abort = true;
      }
      const bool no_output = R.out[0].find("trained weights") == std::string::npos;
      CHECK(all_nonzero && any_abort && no_output, "%s by party %d was not detected (exit codes %d %d %d %d %d, output=%d)", t.name, cheater, R.rc[0], R.rc[1], R.rc[2],
            R.rc[3], R.rc[4], (int)!no_output);
      std::string msg;
      for (const auto& e : R.err) { const auto k = e.find("ABORT:"); if (k != std::string::npos) { msg = e.substr(k, 90); break; } }
      std::printf("  %-26s by party %d -> abort (%s)\n", t.name, cheater, msg.substr(0, msg.find('\n')).c_str());
    }
  // the same deviation in semi-honest mode is (by design) not detected: that mode has no checks
  const ForkResult R = run_forked(P, false, port, Tamper::OPEN_ONLINE, 1, false);
  (void)next_port_block(port);
  bool none = true;
  for (int rc : R.rc) none = none && rc == 0;
  CHECK(none, "semi-honest mode must not abort");
}

int main(int argc, char** argv) {
  std::signal(SIGPIPE, SIG_IGN);
  const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
  int port = 20000 + (std::rand() % 18) * 600;
  // forked tests first: the process must be single-threaded when it forks
  if (!quick) {
    test_end_to_end(port);
    test_tamper(port);
  }
  test_field();
  std::printf("[OT product]\n");
  test_product(1, true, port, 1, 1); port += 40;
  test_product(3, true, port, 5, 7); port += 80;
  test_product(3, true, port, 300, 1); port += 80;
  test_product(3, true, port, 2, 5000); port += 80;
  test_product(2, false, port, 40, 9); port += 80;
  test_mac_check(2, port);
  test_mac_check(5, port);
  std::printf("[offline phase: triples, bits and matrix triples are consistent, MACs verify]\n");
  for (int batch : {1, 2, 4})
    for (bool mal : {true, false}) test_offline(port, batch, mal);
  std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}
