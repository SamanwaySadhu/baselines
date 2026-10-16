// lr_helr_han -- logistic regression on homomorphically encrypted data (Han, Hong, Cheon, Park, IAAI-19) as a
// two-party protocol, with malicious-security checks.
//
//   client: owns the training data (x, y), the initial weights and the HEAAN secret key; encrypts everything
//           and uploads it; at the end decrypts the trained weights
//   server: owns nothing; evaluates the paper's Nesterov-accelerated mini-batch gradient descent on the
//           ciphertexts, including the approximate bootstrapping, and returns the encrypted weights
//
// Protocol (one TCP connection):
//   C -> S  HELLO (parameters)              S -> C  HELLO              (agree on every public parameter)
//   C -> S  UPLOAD: evaluation keys (multiplication, conjugation, rotations), `iters x cnum` encrypted data
//           ciphertexts, `cnum` encrypted w0 and `cnum` encrypted v0                          [offline]
//   S       evaluates: for each iteration  inner product, sigmoid gradient, NAG update; every `iterPerBoot`
//           iterations bootstraps w and v (no interaction)                                  [online]
//   S -> C  RESULT: the `cnum` encrypted final weights
//   C       decrypts, and checks the weights against the plaintext training it runs itself (it owns the inputs).
//
// The server learns nothing but the sizes. The client learns the trained model, as the decryptor does in the paper.
// See README.md for the security model.
//
// usage: lr_helr_han <client|server> [options]       (--help)
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

#include "circuit.h"
#include "he.h"
#include "metrics.h"
#include "net.h"
#include "params.h"

using namespace han;

namespace {

constexpr uint64_t kMagicHello = 0x48414e4845303031ULL;   // "HANHE001"
constexpr uint64_t kMagicUpload = 0x48414e5550303031ULL;  // "HANUP001"
constexpr uint64_t kMagicResult = 0x48414e5253303031ULL;  // "HANRS001"

struct Tamper {
  enum Kind { NONE, SERVER_RESULT, SERVER_ORACLE, SERVER_LEVEL, CLIENT_RANGE, CLIENT_SHAPE, CLIENT_KEY, CLIENT_MISSING_KEY } kind = NONE;
};

struct Options {
  int role = 0;  // 0 client, 1 server
  Params P;
  std::string host = "127.0.0.1";
  int port = 12345;
  double tol = 1e-2;
  bool quiet = false;
  std::string json_path, csv_path;
  Tamper tamper;
};

void usage(const char* prog) {
  std::printf(
      "usage: %s <client|server> [options]\n"
      "  client = owns the data and the secret key and receives the trained weights; server = evaluates on ciphertexts\n\n"
      "training (both parties must pass the same values; --seed / --zero-init / --qbits only shape the client's\n"
      "private inputs):\n"
      "  --d N            features incl. the bias column (default 5)\n"
      "  --iters N        NAG iterations (default 20)\n"
      "  --batch N        samples per iteration = mini-batch block size, a power of two (default 4)\n"
      "  --lr X           learning rate alpha; gamma = alpha / batch (default 0.5)\n"
      "  --no-bias        the last feature is a regular client feature, not the bias\n"
      "  --zero-init      client's initial weights are 0 (default: uniform random)\n"
      "  --qbits N        quantization of the client's raw random values (default 6, as in the sibling baselines)\n"
      "  --seed N         seed of the client's random data / initial weights (default 1)\n"
      "  --range X        the sigmoid polynomial is fitted on [-X, X] (default 8, the paper's)\n\n"
      "homomorphic encryption (the paper's defaults; both parties must agree):\n"
      "  --wbits N --pbits N          scaling bits of the data / of constants (default 40 / 15)\n"
      "  --iter-per-boot N            iterations between bootstrappings (default 3)\n"
      "  --lambda N                   security level used to derive the ring degree (default 80, the paper's;\n"
      "                               128 needs N = 2^17 and about 4 times the memory)\n"
      "  --logn N                     override the ring degree 2^N. Below the bound of --lambda this is INSECURE\n"
      "                               and only for tests\n"
      "  --cnum N                     ciphertexts per weight vector (power of two). Default: enough that the 2*cnum\n"
      "                               bootstrapped ciphertexts keep all threads busy. The client's value is adopted\n"
      "                               by the server\n\n"
      "protocol / network:\n"
      "  --host IP --port N           (default 127.0.0.1:12345)\n"
      "  --threads N                  worker threads (default: all hardware threads)\n"
      "  --tol X                      client: abort if a decrypted weight differs from the plaintext\n"
      "                               computation by more than X (default 1e-2)\n\n"
      "output:\n"
      "  --json FILE  --csv FILE      (client) metrics as JSON / one summary row\n"
      "  --quiet                      no progress lines\n"
      "  --tamper KIND                TEST ONLY: this party deviates once; the run must abort.\n"
      "      server: result oracle level      client: range shape key missing-key\n",
      prog);
}

Options parse_options(int argc, char** argv) {
  Options O;
  if (argc < 2) throw std::invalid_argument("missing role");
  const std::string role = argv[1];
  if (role == "client") O.role = 0;
  else if (role == "server") O.role = 1;
  else throw std::invalid_argument("unknown role '" + role + "' (use client or server)");
  ArgReader r{argc, argv, 2};
  for (; r.has(); ++r.i) {
    const std::string a = argv[r.i];
    if (parse_param_flag(r, O.P)) continue;
    if (a == "--host") O.host = r.next_value(a);
    else if (a == "--port") O.port = (int)parse_int(r.next_value(a), a);
    else if (a == "--tol") O.tol = parse_double(r.next_value(a), a);
    else if (a == "--quiet") O.quiet = true;
    else if (a == "--json") O.json_path = r.next_value(a);
    else if (a == "--csv") O.csv_path = r.next_value(a);
    else if (a == "--tamper") {
      const std::string k = r.next_value(a);
      if (k == "result") O.tamper.kind = Tamper::SERVER_RESULT;
      else if (k == "oracle") O.tamper.kind = Tamper::SERVER_ORACLE;
      else if (k == "level") O.tamper.kind = Tamper::SERVER_LEVEL;
      else if (k == "range") O.tamper.kind = Tamper::CLIENT_RANGE;
      else if (k == "shape") O.tamper.kind = Tamper::CLIENT_SHAPE;
      else if (k == "key") O.tamper.kind = Tamper::CLIENT_KEY;
      else if (k == "missing-key") O.tamper.kind = Tamper::CLIENT_MISSING_KEY;
      else throw std::invalid_argument("unknown --tamper kind " + k);
    } else if (a == "--help" || a == "-h") { usage(argv[0]); std::exit(0); }
    else throw std::invalid_argument("unknown option " + a);
  }
  if (O.port < 1024 || O.port > 65535) throw std::invalid_argument("--port out of range");
  const bool is_server_kind = O.tamper.kind == Tamper::SERVER_RESULT || O.tamper.kind == Tamper::SERVER_ORACLE || O.tamper.kind == Tamper::SERVER_LEVEL;
  if (O.tamper.kind != Tamper::NONE && is_server_kind != (O.role == 1)) throw std::invalid_argument("this --tamper kind belongs to the other role");
  O.P.finalize();
  return O;
}

struct Hello {
  uint64_t magic;
  int32_t d, iters, batch, bias, qbits, wBits, pBits, lBits, iterPerBoot, lambda, logN, cnum, bsz, slots;
  double lr, range;
};
Hello make_hello(const Params& P) {
  Hello h{};
  h.magic = kMagicHello;
  h.d = P.d; h.iters = P.iters; h.batch = P.batch; h.bias = P.bias; h.qbits = P.qbits; h.wBits = P.wBits; h.pBits = P.pBits;
  h.lBits = P.lBits; h.iterPerBoot = P.iterPerBoot; h.lambda = P.lambda; h.logN = P.logN; h.cnum = P.cnum; h.bsz = P.bsz; h.slots = P.slots;
  h.lr = P.lr; h.range = P.range;
  return h;
}

std::vector<unsigned char> recv_ct_bytes(Channel& ch, const CtShape& e) {
  std::vector<unsigned char> buf(kCtHeader + ct_body_bytes(e.N, e.logq));
  ch.recv(buf.data(), kCtHeader);
  check_ct_header(buf.data(), e);  // before the body is read: sizes never come from the peer
  ch.recv(buf.data() + kCtHeader, buf.size() - kCtHeader);
  return buf;
}

// Padding-bit scan only (the full parse happens when the ciphertext is used).
void check_ct_padding(const std::vector<unsigned char>& buf, const CtShape& e) {
  const size_t nb = ct_coef_bytes(e.logq);
  const int extra = (int)(nb * 8 - e.logq);
  if (!extra) return;
  const unsigned char topmask = (unsigned char)(0xFFu << (8 - extra));
  const unsigned char* p = buf.data() + kCtHeader + nb - 1;
  for (long i = 0; i < 2 * e.N; ++i, p += nb)
    if (*p & topmask) throw AbortError("ciphertext coefficient out of range");
}

struct Report {
  std::vector<PhaseRow> rows[2];  // [0] client, [1] server
  long rss_kib[2] = {0, 0};
};

void exchange_report(Channel& ch, int role, const std::vector<PhaseRow>& mine, Report& R) {
  R.rows[role] = mine;
  R.rss_kib[role] = peak_rss_kib();
  auto send_mine = [&] {
    ch.send_u64(mine.size());
    ch.send(mine.data(), mine.size() * sizeof(PhaseRow));
    ch.send_pod(R.rss_kib[role]);
    ch.flush();
  };
  auto recv_theirs = [&] {
    const uint64_t n = ch.recv_u64();
    if (n > 256) throw std::runtime_error("report: implausible row count");
    R.rows[1 - role].resize(n);
    ch.recv(R.rows[1 - role].data(), n * sizeof(PhaseRow));
    R.rss_kib[1 - role] = ch.recv_pod<long>();
  };
  if (role == 0) { send_mine(); recv_theirs(); } else { recv_theirs(); send_mine(); }
}

std::string ms(double s) { return fmt_s(s); }

void print_phase_table(const Report& R, int party) {
  std::printf("\n%s phases:\n%-26s %12s %12s %12s %12s %8s\n", party == 0 ? "client" : "server", "phase", "wall", "cpu", "sent", "received", "flights");
  for (const auto& r : R.rows[party])
    std::printf("%-26s %12s %12s %12s %12s %8" PRIu64 "\n", r.name, ms(r.wall_s).c_str(), ms(r.cpu_s).c_str(), human_bytes(r.sent).c_str(),
                human_bytes(r.recv).c_str(), r.flights);
}

}  // namespace

int main(int argc, char** argv) {
  Options O;
  try {
    O = parse_options(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n\n", e.what());
    usage(argv[0]);
    return 1;
  }
  const char* who = O.role == 0 ? "client" : "server";
  Params& P = O.P;
  if (!O.quiet) std::fprintf(stderr, "[%s] %s | threads=%d\n", who, params_string(P).c_str(), P.threads);
  if (P.insecure && !O.quiet) std::fprintf(stderr, "[%s] WARNING: ring degree below the security bound of lambda=%d: test only\n", who, P.lambda);

  Recorder rec;
  int rc = 0;
  try {
    init_threads(P.threads);
    seed_ntl();
    std::unique_ptr<Channel> chp = O.role == 1 ? Channel::listen(O.port) : Channel::connect(O.host, O.port);
    Channel& ch = *chp;
    rec.attach(&ch);
    const Poly3 H = fit_sigmoid(P.range);

    // ---------------------------------------------------------------- handshake
    {
      Scope sc(rec, "other/handshake");
      if (O.role == 0) {
        Hello mine = make_hello(P);
        ch.send_pod(mine);
        ch.flush();
        const Hello theirs = ch.recv_pod<Hello>();
        if (std::memcmp(&mine, &theirs, sizeof mine) != 0) throw AbortError("the server runs with different parameters");
      } else {
        const Hello theirs = ch.recv_pod<Hello>();
        if (theirs.magic != kMagicHello) throw AbortError("bad hello");
        // the packing (cnum) is the client's choice; everything else must match what this server was started with
        if (theirs.cnum < 1 || theirs.cnum > 1024) throw AbortError("implausible cnum");
        P.cnum_override = theirs.cnum;
        P.finalize();
        const Hello mine = make_hello(P);
        if (std::memcmp(&mine, &theirs, sizeof mine) != 0) throw AbortError("the client runs with different parameters");
        ch.send_pod(mine);
        ch.flush();
        if (!O.quiet) std::fprintf(stderr, "[server] packing adopted from the client: %d ciphertexts x %d features, %d slots\n", P.cnum, P.bsz, P.slots);
      }
    }

    const std::vector<double> eta = nag_alphas_eta(P.iters);
    const CtShape z_shape{1L << P.logN, P.slots, P.wBits, (long)P.logQ + P.wBits};
    const CtShape w_shape{1L << P.logN, P.slots, P.wBits, (long)P.logQ};
    const CtShape out_shape{1L << P.logN, P.slots, P.wBits, final_logq(P)};

    if (O.role == 0) {
      // ================================================================ CLIENT
      Dataset D = generate_client_data(P);
      RefResult ref, ref_exact;
      {
        Scope sc(rec, "other/plaintext_reference");  // the client's own plaintext training: used to check the server
        ref = reference_train(P, D, H);
        ref_exact = reference_train(P, D, H, true);
      }
      if (ref.max_abs_u > P.range && !O.quiet) std::fprintf(stderr, "[client] note: |z.v| reached %.2f > %.1f, outside the polynomial's fitted range\n", ref.max_abs_u, P.range);

      std::unique_ptr<Ring> ring;
      std::unique_ptr<SecretKey> sk;
      std::unique_ptr<Scheme> scheme;
      {
        Scope sc(rec, "offline/context");
        ring.reset(new Ring(P.logN, P.logQBoot));
        ring->addBootContext(P.bBits, boot_logp(P));
        sk.reset(new SecretKey(*ring));
      }
      {
        Scope sc(rec, "offline/keygen");
        scheme.reset(new Scheme(*sk, *ring));  // encryption (public) key and multiplication key
        generate_eval_keys(P, *sk, *scheme);
      }
      std::vector<std::vector<unsigned char>> zbuf((size_t)P.iters * P.cnum), wbuf((size_t)P.cnum), vbuf((size_t)P.cnum);
      {
        Scope sc(rec, "offline/encrypt");
        const long total = (long)zbuf.size() + 2L * P.cnum;
        par_for(total, [&](long idx) {
          seed_ntl();
          std::vector<double> v;
          long logq;
          std::vector<unsigned char>* dst;
          if (idx < (long)zbuf.size()) {
            const int t = (int)(idx / P.cnum), j = (int)(idx % P.cnum);
            v = pack_z(P, D, t, j);
            logq = P.logQ + P.wBits;
            dst = &zbuf[(size_t)idx];
          } else {
            const long k = idx - (long)zbuf.size();
            const int j = (int)(k % P.cnum);
            v = pack_w(P, D.w0, j);
            logq = P.logQ;
            dst = k < P.cnum ? &wbuf[(size_t)j] : &vbuf[(size_t)j];
          }
          Ciphertext c = scheme->encrypt(v.data(), P.slots, P.wBits, logq);
          *dst = serialize_ct(c);
        });
      }
      // test hooks: a malicious client
      if (O.tamper.kind == Tamper::CLIENT_RANGE) zbuf[0][kCtHeader + ct_coef_bytes(z_shape.logq) - 1] |= 0x80;
      if (O.tamper.kind == Tamper::CLIENT_SHAPE) { int64_t n = P.slots * 2; std::memcpy(zbuf[1].data() + 8, &n, 8); }
      {
        Scope sc(rec, "offline/upload");
        ch.send_u64(kMagicUpload);
        // keys: multiplication, conjugation, rotations
        std::vector<long> rots = required_rot_keys(P, *ring);
        if (O.tamper.kind == Tamper::CLIENT_MISSING_KEY) rots.pop_back();
        ch.send_u64(2 + rots.size());
        send_key(ch, TAG_MULT, 0, *scheme->keyMap.at(MULTIPLICATION));
        if (O.tamper.kind == Tamper::CLIENT_KEY) {
          Key* k = scheme->keyMap.at(CONJUGATION);
          k->rax[5] = ~0ULL;  // an unreduced residue
        }
        send_key(ch, TAG_CONJ, 0, *scheme->keyMap.at(CONJUGATION));
        for (long idx : rots) send_key(ch, TAG_ROT, (uint64_t)idx, *scheme->leftRotKeyMap.at(idx));
        for (auto& b : zbuf) ch.send(b.data(), b.size());
        for (auto& b : wbuf) ch.send(b.data(), b.size());
        for (auto& b : vbuf) ch.send(b.data(), b.size());
        ch.flush();
        // the evaluator keys are not needed any more
        for (auto& kv : scheme->leftRotKeyMap) delete kv.second;
        scheme->leftRotKeyMap.clear();
      }
      if (!O.quiet) std::fprintf(stderr, "[client] upload done, waiting for the server's evaluation\n");

      // ---------------------------------------------------------------- online: wait, receive, decrypt, verify
      {
        Scope sc(rec, "online/wait_server");
        if (ch.recv_u64() != kMagicResult) throw AbortError("bad result marker");
      }
      std::vector<Ciphertext> Wf((size_t)P.cnum);
      {
        Scope sc(rec, "online/receive");
        std::vector<std::vector<unsigned char>> rb((size_t)P.cnum);
        for (int j = 0; j < P.cnum; ++j) rb[(size_t)j] = recv_ct_bytes(ch, out_shape);
        for (int j = 0; j < P.cnum; ++j) Wf[(size_t)j] = deserialize_ct(rb[(size_t)j].data(), out_shape);
      }
      std::vector<double> w;
      double dev = 0, dev_exact = 0, rep = 0;
      {
        Scope sc(rec, "online/decrypt_verify");
        w = decrypt_weights(*scheme, *sk, Wf, P, &rep);
        for (int j = 0; j < P.d; ++j) {
          dev = std::max(dev, std::fabs(w[(size_t)j] - ref.w[(size_t)j]));
          dev_exact = std::max(dev_exact, std::fabs(w[(size_t)j] - ref_exact.w[(size_t)j]));
        }
        for (int j = P.d; j < P.factorNum; ++j) dev = std::max(dev, std::fabs(w[(size_t)j]));  // padding weights stay 0
        if (!(dev <= O.tol) || !(rep <= O.tol)) {
          char b[256];
          std::snprintf(b, sizeof b, "the decrypted weights differ from the client's own plaintext training by %.3g (replicas %.3g) > tolerance %.3g: the result is rejected and nothing is released", dev, rep, O.tol);
          throw AbortError(b);
        }
      }

      Report R;
      exchange_report(ch, 0, rec.rows(), R);
      std::printf("\nHELR (Han et al.) logistic regression on encrypted data: %s\n", params_string(P).c_str());
      std::printf("trained weights (client, decrypted):");
      for (int j = 0; j < P.d && j < 8; ++j) std::printf(" %.5f", w[(size_t)j]);
      if (P.d > 8) std::printf(" ... (%d weights)", P.d);
      std::printf("\nverify: max |HE - plaintext polynomial training| = %.3g (tolerance %.3g, replica spread %.2g); distance to exact-sigmoid training %.3g\n", dev, O.tol, rep, dev_exact);
      print_phase_table(R, 0);
      print_phase_table(R, 1);

      // ---- stage totals and summary
      const char* stages[] = {"offline", "online", "other"};
      StageTotals T[3][2];
      for (int s = 0; s < 3; ++s)
        for (int p = 0; p < 2; ++p) T[s][p] = stage_totals(R.rows[p], stages[s]);
      std::printf("\n%-8s | %-34s | %-34s | %-26s\n", "stage", "client: wall (compute) / cpu", "server: wall (compute) / cpu", "combined compute wall / cpu");
      double cw = 0, cc = 0, cx = 0;
      StageTotals tot[2];
      for (int s = 0; s < 3; ++s) {
        for (int p = 0; p < 2; ++p) {
          tot[p].wall_s += T[s][p].wall_s; tot[p].wait_s += T[s][p].wait_s; tot[p].cpu_s += T[s][p].cpu_s;
          tot[p].sent += T[s][p].sent;
        }
        std::printf("%-8s | %10s (%9s) / %9s | %10s (%9s) / %9s | %10s / %9s\n", stages[s], ms(T[s][0].wall_s).c_str(), ms(T[s][0].compute_s()).c_str(), ms(T[s][0].cpu_s).c_str(),
                    ms(T[s][1].wall_s).c_str(), ms(T[s][1].compute_s()).c_str(), ms(T[s][1].cpu_s).c_str(),
                    ms(T[s][0].compute_s() + T[s][1].compute_s()).c_str(), ms(T[s][0].cpu_s + T[s][1].cpu_s).c_str());
        cw += T[s][0].compute_s() + T[s][1].compute_s();
        cc += T[s][0].cpu_s + T[s][1].cpu_s;
        cx += 0;
      }
      std::printf("%-8s | %10s (%9s) / %9s | %10s (%9s) / %9s | %10s / %9s\n", "total", ms(tot[0].wall_s).c_str(), ms(tot[0].wall_s - tot[0].wait_s).c_str(), ms(tot[0].cpu_s).c_str(),
                  ms(tot[1].wall_s).c_str(), ms(tot[1].wall_s - tot[1].wait_s).c_str(), ms(tot[1].cpu_s).c_str(), ms(cw).c_str(), ms(cc).c_str());
      (void)cx;
      std::printf("(compute = wall minus the time spent blocked on the other party; cpu = process CPU time over all threads)\n");

      const double wall_e2e = tot[0].wall_s;
      std::printf("\nSUMMARY (d=%d, iters=%d, batch=%d, %d threads per party, N=2^%d, %d+%d ciphertexts per weight vector / data block)\n", P.d, P.iters, P.batch, P.threads, P.logN, P.cnum, P.cnum);
      for (int s = 0; s < 3; ++s)
        std::printf("  %-8s client %9s compute / %9s cpu   server %9s compute / %9s cpu   combined %9s compute / %9s cpu\n", stages[s], ms(T[s][0].compute_s()).c_str(),
                    ms(T[s][0].cpu_s).c_str(), ms(T[s][1].compute_s()).c_str(), ms(T[s][1].cpu_s).c_str(), ms(T[s][0].compute_s() + T[s][1].compute_s()).c_str(),
                    ms(T[s][0].cpu_s + T[s][1].cpu_s).c_str());
      std::printf("  end-to-end elapsed %s (client's wall clock); peak RSS client %.0f MiB / server %.0f MiB\n", ms(wall_e2e).c_str(), R.rss_kib[0] / 1024.0, R.rss_kib[1] / 1024.0);
      const uint64_t c2s = tot[0].sent, s2c = tot[1].sent;
      std::printf("  communication offline: client->server %s, server->client %s | online: client->server %s, server->client %s | other: %s / %s | total %s\n",
                  human_bytes(T[0][0].sent).c_str(), human_bytes(T[0][1].sent).c_str(), human_bytes(T[1][0].sent).c_str(), human_bytes(T[1][1].sent).c_str(),
                  human_bytes(T[2][0].sent).c_str(), human_bytes(T[2][1].sent).c_str(), human_bytes(c2s + s2c).c_str());
      uint64_t fl[3];
      for (int s = 0; s < 3; ++s) fl[s] = T[s][0].flights;
      std::printf("  rounds: offline %" PRIu64 " flights, online %" PRIu64 " flights, other %" PRIu64 " flights; total %" PRIu64 " flights = %" PRIu64 " round trips\n", fl[0], fl[1], fl[2],
                  fl[0] + fl[1] + fl[2], (fl[0] + fl[1] + fl[2] + 1) / 2);

      if (!O.json_path.empty()) {
        std::ofstream f(O.json_path);
        f << "{\n  \"protocol\": \"helr_han\",\n  \"d\": " << P.d << ", \"iters\": " << P.iters << ", \"batch\": " << P.batch << ", \"logN\": " << P.logN
          << ", \"cnum\": " << P.cnum << ", \"threads\": " << P.threads << ", \"max_dev\": " << dev << ",\n  \"phases\": [\n";
        bool first = true;
        for (int p = 0; p < 2; ++p)
          for (const auto& r : R.rows[p]) {
            if (!first) f << ",\n";
            first = false;
            f << "    {\"party\": \"" << (p == 0 ? "client" : "server") << "\", \"phase\": \"" << r.name << "\", \"wall_s\": " << r.wall_s << ", \"cpu_s\": " << r.cpu_s
              << ", \"sent\": " << r.sent << ", \"recv\": " << r.recv << ", \"flights\": " << r.flights << "}";
          }
        f << "\n  ]\n}\n";
      }
      if (!O.csv_path.empty()) {
        const std::string header =
            "protocol,d,iters,batch,logN,cnum,threads,max_dev,"
            "client_offline_compute_s,client_offline_cpu_s,client_online_compute_s,client_online_cpu_s,client_other_compute_s,client_other_cpu_s,"
            "server_offline_compute_s,server_offline_cpu_s,server_online_compute_s,server_online_cpu_s,server_other_compute_s,server_other_cpu_s,"
            "comm_offline_c2s_bytes,comm_offline_s2c_bytes,comm_online_c2s_bytes,comm_online_s2c_bytes,comm_other_c2s_bytes,comm_other_s2c_bytes,"
            "flights_offline,flights_online,flights_other,elapsed_s";
        bool exists = false;
        {
          std::ifstream in(O.csv_path);
          std::string first;
          if (in && std::getline(in, first)) {
            exists = true;
            if (first != header) { std::fprintf(stderr, "error: %s has a different header\n", O.csv_path.c_str()); return 1; }
          }
        }
        std::ofstream f(O.csv_path, std::ios::app);
        if (!exists) f << header << "\n";
        char row[1536];
        std::snprintf(row, sizeof row,
                      "helr_han,%d,%d,%d,%d,%d,%d,%.3g,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.3f",
                      P.d, P.iters, P.batch, P.logN, P.cnum, P.threads, dev, T[0][0].compute_s(), T[0][0].cpu_s, T[1][0].compute_s(), T[1][0].cpu_s, T[2][0].compute_s(), T[2][0].cpu_s,
                      T[0][1].compute_s(), T[0][1].cpu_s, T[1][1].compute_s(), T[1][1].cpu_s, T[2][1].compute_s(), T[2][1].cpu_s, T[0][0].sent, T[0][1].sent, T[1][0].sent,
                      T[1][1].sent, T[2][0].sent, T[2][1].sent, fl[0], fl[1], fl[2], wall_e2e);
        f << row << "\n";
      }
    } else {
      // ================================================================ SERVER
      std::unique_ptr<Ring> ring;
      std::unique_ptr<Scheme> scheme;
      {
        Scope sc(rec, "offline/context");
        ring.reset(new Ring(P.logN, P.logQBoot));
        ring->addBootContext(P.bBits, boot_logp(P));
        scheme.reset(new Scheme(*ring));  // no keys of its own: the client's evaluation keys are installed below
      }
      {
        Scope sc(rec, "offline/wait_client");  // the client generates keys and encrypts
        if (ch.recv_u64() != kMagicUpload) throw AbortError("bad upload marker");
      }
      {
        Scope sc(rec, "offline/receive_keys");
        const std::vector<long> need = required_rot_keys(P, *ring);
        const uint64_t nkeys = ch.recv_u64();
        if (nkeys != 2 + need.size()) throw AbortError("the client sent the wrong number of evaluation keys");
        std::set<long> have;
        bool have_mult = false, have_conj = false;
        for (uint64_t i = 0; i < nkeys; ++i) {
          uint64_t tag, idx;
          std::unique_ptr<Key> k(recv_key(ch, *ring, tag, idx));
          if (tag == TAG_MULT && !have_mult) { have_mult = true; scheme->keyMap[MULTIPLICATION] = k.release(); }
          else if (tag == TAG_CONJ && !have_conj) { have_conj = true; scheme->keyMap[CONJUGATION] = k.release(); }
          else if (tag == TAG_ROT && std::binary_search(need.begin(), need.end(), (long)idx) && !have.count((long)idx)) {
            have.insert((long)idx);
            scheme->leftRotKeyMap[(long)idx] = k.release();
          } else throw AbortError("unexpected or duplicate evaluation key");
        }
        if (!have_mult || !have_conj || have.size() != need.size()) throw AbortError("missing evaluation key");
      }
      Evaluator ev(*scheme, P, H);
      std::vector<std::vector<unsigned char>> zbuf((size_t)P.iters * P.cnum);
      {
        Scope sc(rec, "offline/receive_data");
        for (auto& b : zbuf) { b = recv_ct_bytes(ch, z_shape); check_ct_padding(b, z_shape); }
        std::vector<std::vector<unsigned char>> wb((size_t)P.cnum), vb((size_t)P.cnum);
        for (auto& b : wb) { b = recv_ct_bytes(ch, w_shape); check_ct_padding(b, w_shape); }
        for (auto& b : vb) { b = recv_ct_bytes(ch, w_shape); check_ct_padding(b, w_shape); }
        ev.W.resize((size_t)P.cnum);
        ev.V.resize((size_t)P.cnum);
        for (int j = 0; j < P.cnum; ++j) {
          ev.W[(size_t)j] = deserialize_ct(wb[(size_t)j].data(), w_shape);
          ev.V[(size_t)j] = deserialize_ct(vb[(size_t)j].data(), w_shape);
        }
      }
      if (!O.quiet) std::fprintf(stderr, "[server] upload received (%s), evaluating\n", human_bytes(ch.counters().recv).c_str());

      std::vector<unsigned char> oracle_copy;
      if (O.tamper.kind == Tamper::SERVER_ORACLE) oracle_copy = zbuf[0];
      for (int t = 0; t < P.iters; ++t) {
        std::vector<Ciphertext> Z((size_t)P.cnum);
        {
          Scope sc(rec, "online/load_data");
          par_for(P.cnum, [&](long j) { Z[(size_t)j] = deserialize_ct(zbuf[(size_t)t * P.cnum + (size_t)j].data(), z_shape); });
        }
        ev.iteration(Z, eta[(size_t)t], &rec);
        Z.clear();
        for (int j = 0; j < P.cnum; ++j) { zbuf[(size_t)t * P.cnum + j].clear(); zbuf[(size_t)t * P.cnum + j].shrink_to_fit(); }
        if (boot_after(P, t)) ev.bootstrap_all(&rec);
        if (!O.quiet) std::fprintf(stderr, "[server] iteration %d/%d done%s\n", t + 1, P.iters, boot_after(P, t) ? " (bootstrapped)" : "");
      }
      {
        Scope sc(rec, "online/send_result");
        if (O.tamper.kind == Tamper::SERVER_RESULT) scheme->addConstAndEqual(ev.W[0], 0.5, P.wBits);  // a wrong result
        if (O.tamper.kind == Tamper::SERVER_ORACLE) {  // tries to get the client to decrypt one of its own data ciphertexts
          Ciphertext c = deserialize_ct(oracle_copy.data(), z_shape);
          scheme->modDownToAndEqual(c, out_shape.logq);
          ev.W[0] = c;
        }
        if (O.tamper.kind == Tamper::SERVER_LEVEL) scheme->modDownByAndEqual(ev.W[0], 5);
        ch.send_u64(kMagicResult);
        for (int j = 0; j < P.cnum; ++j) {
          std::vector<unsigned char> b = serialize_ct(ev.W[(size_t)j]);
          ch.send(b.data(), b.size());
        }
        ch.flush();
      }
      Report R;
      exchange_report(ch, 1, rec.rows(), R);
      std::printf("\nHELR (Han et al.) server: evaluated %d iterations on encrypted data (%d bootstrapping rounds); learned nothing but the sizes\n", P.iters,
                  [&] { int b = 0; for (int t = 0; t < P.iters; ++t) b += boot_after(P, t); return b; }());
      print_phase_table(R, 1);
    }
  } catch (const AbortError& e) {
    std::fprintf(stderr, "[%s] ABORT: %s\n", who, e.what());
    rc = 3;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[%s] error: %s\n", who, e.what());
    rc = 1;
  }
  return rc;
}
