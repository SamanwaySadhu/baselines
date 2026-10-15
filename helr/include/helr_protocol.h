// helr_protocol.h -- the two-party protocol around HELR's homomorphic training, as
// message-level state machines shared by the program and the selftest.
//
//   CLIENT  data owner and key owner: holds the training data, the initial weights and the
//           HE secret key
//   SERVER  evaluator: no input; ends with the trained weights and nothing else
//
//   S -> C  com_S = H(sigma_S)                      coin toss for the public-key element a
//   C -> S  header, sigma_C                         (a = XOF(sigma_S || sigma_C), see below)
//   S -> C  sigma_S                                 (client checks the commitment)
//   C -> S  UPLOAD: pk.bx, multiplication key, rotation keys, the d encrypted data columns,
//           the d encrypted initial weights                                      [offline]
//           ... the server evaluates HELR's gradient descent on ciphertexts ...  [online]
//   S -> C  c1_j (final weight j), c2_j = alpha_j * c1_j + Enc_pk(r_j)     for every j < d
//   C -> S  commitment to the decryptions (u1_j, u2_j) of c1_j and c2_j
//   S -> C  seed_j: everything the server drew for c2_j (alpha_j, r_j, encryption randomness)
//   C -> S  opening: the nonce and every (u1_j, u2_j)
//   server: checks the commitment, the MAC  u2_j = alpha_j u1_j + r_j (+ small noise)  and the
//           weight range, and outputs w_j = u1_j / 2^logp.
//
// Malicious security: what each message is for (README: "Security model" has the full
// argument and what is NOT covered).
//  * Decryption MAC (server's integrity against a lying client). The client must report
//    u1_j = Dec(c1_j)[0], the final weight. c2_j is a fresh encryption of the one-time-MAC
//    of c1_j under the server's secret alpha_j, masked by a uniform r_j; the client commits to
//    (u1_j, u2_j) BEFORE it learns alpha_j, r_j, so a wrong u1_j needs a matching u2_j =
//    alpha_j (u1_j + delta) + r_j, i.e. alpha_j, which the client's view does not depend on
//    (u2_j is uniform because of r_j; c2_j hides alpha_j under RLWE given a uniform `a`).
//    Forging probability <= 2^-(mac_bits + 2) per weight (mac_forgery_bits in helr_reference.h).
//  * Coin-tossed `a` (so that the MAC mask is sound against a client-chosen public key).
//  * Verified decryption (client's privacy against a lying server). The client decrypts only
//    after (i) re-deriving c2_j from seed_j, i.e. checking that c2_j is exactly the honest MAC
//    ciphertext, and (ii) checking that Dec(c1_j) is within `tol` of the weight the client
//    computes itself by running the same training in the clear (it owns the inputs). Nothing is
//    revealed before both pass; the opening is sent only afterwards.
//  * Server checks: weight range, every ring element well formed, parameters equal.
#ifndef LRHE_HELR_PROTOCOL_H
#define LRHE_HELR_PROTOCOL_H

#include "helr_circuit.h"
#include "lr_reference.h"

#include <array>
#include <memory>
#include <optional>

namespace lrhe {

struct ProtocolError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ----------------------------------------------------------------------- header
constexpr size_t kHeaderBytes = 64;
constexpr uint32_t kMagic = 0x52454C48;  // "HELR"
constexpr uint32_t kVersion = 1;

inline std::vector<uint8_t> make_header(const HeConfig& c) {
  const uint32_t f[16] = {kMagic, kVersion, (uint32_t)c.logN, (uint32_t)c.logp, (uint32_t)c.deg, (uint32_t)c.mac_bits,
                          (uint32_t)c.logR, (uint32_t)c.T, (uint32_t)c.B, (uint32_t)c.d, (uint32_t)c.k_shift,
                          (uint32_t)c.nslots, (uint32_t)c.logq, (uint32_t)c.nrot, 0, 0};
  std::vector<uint8_t> h(kHeaderBytes);
  for (int i = 0; i < 16; ++i)
    for (int b = 0; b < 4; ++b) h[(size_t)i * 4 + b] = (uint8_t)(f[i] >> (8 * b));
  return h;
}
inline void check_header(const std::vector<uint8_t>& h, const HeConfig& c) {
  if (h.size() != kHeaderBytes) throw ProtocolError("bad header length");
  uint32_t f[16];
  for (int i = 0; i < 16; ++i) {
    f[i] = 0;
    for (int b = 0; b < 4; ++b) f[i] |= (uint32_t)h[(size_t)i * 4 + b] << (8 * b);
  }
  if (f[0] != kMagic || f[1] != kVersion) throw ProtocolError("not a HELR header (magic / version)");
  const auto want = make_header(c);
  if (std::memcmp(want.data(), h.data(), kHeaderBytes) != 0) {
    const char* nm[16] = {"magic", "version", "logN", "logp", "deg", "mac_bits", "logR", "iters", "batch", "d",
                          "k_shift", "nslots", "logq", "nrot", "-", "-"};
    for (int i = 2; i < 14; ++i) {
      uint32_t w = 0;
      for (int b = 0; b < 4; ++b) w |= (uint32_t)want[(size_t)i * 4 + b] << (8 * b);
      if (w != f[i])
        throw ProtocolError(std::string("client and server disagree on ") + nm[i] + " (client " + std::to_string(f[i]) +
                            ", server " + std::to_string(w) + ")");
    }
    throw ProtocolError("header mismatch");
  }
}

// ------------------------------------------------------------- analytic sizes
struct CommLayout {
  uint64_t coin_s2c = 64, coin_c2s = 0;  // com_S + sigma_S ; header + sigma_C
  uint64_t keys = 0, data = 0, weights = 0;
  uint64_t out_s2c = 0, seeds = 0, commit = 32, opening = 0;
  uint64_t offline_c2s() const { return coin_c2s + keys + data + weights; }
  uint64_t offline_s2c() const { return coin_s2c; }
  uint64_t online_s2c() const { return out_s2c + seeds; }
  uint64_t online_c2s() const { return commit + opening; }
};
inline CommLayout comm_layout(const HeConfig& c) {
  CommLayout L;
  L.coin_c2s = kHeaderBytes + 32;
  L.keys = keys_bytes(c);
  L.data = (uint64_t)c.d * cipher_bytes(c.N(), c.logq);
  L.weights = L.data;
  L.out_s2c = 2ull * c.d * cipher_bytes(c.N(), c.out_bits);
  L.seeds = 32ull * c.d;
  L.opening = 32 + 2ull * c.d * coeff_bytes(c.out_bits);
  return L;
}

// ------------------------------------------------------------------- the MAC
// Fresh-encryption noise at coefficient 0 of Enc_pk(r): (v e_pk + e0 + e1 s)[0]. The client's
// public key error has sigma 3.2, v is ternary (E v^2 = 1/2), s has Hamming weight h and e0,
// e1 are centered binomial (variance 10): a 12-sigma bound.
inline NTL::ZZ enc_noise_bound(const HeContext& c) {
  const double N = (double)c.params.N, h = (double)c.params.h;
  const double var = 3.2 * 3.2 * N / 2.0 + 10.0 * (h + 1.0);
  return NTL::to_ZZ((long)std::ceil(12.0 * std::sqrt(var)) + 2);
}

// The server's secrets for one output.
struct MacSecret {
  NTL::ZZ alpha;  // odd, uniform in [1, 2^L)
  NTL::ZZ r0;     // coefficient 0 of the uniform mask r (in [0, 2^L))
};

// pk mod 2^L (valid: q is a multiple of 2^L), computed once per output modulus.
struct PkAtLevel {
  NTL::ZZX ax, bx;
};
inline PkAtLevel pk_at_level(const HeContext& c, long L) {
  PkAtLevel p;
  NTL::ZZ mod = NTL::power2_ZZ(L);
  Ring2Utils::mod(p.ax, const_cast<NTL::ZZX&>(c.pk.ax), mod, c.params.N);
  Ring2Utils::mod(p.bx, const_cast<NTL::ZZX&>(c.pk.bx), mod, c.params.N);
  return p;
}

namespace detail {
inline void fill_bytes(std::vector<uint8_t>& buf, size_t n) {
  // n bytes from NTL's current stream
  NTL::ZZ big = NTL::RandomBits_ZZ((long)(8 * n));
  buf.assign(n, 0);
  NTL::BytesFromZZ(buf.data(), big, (long)n);
}
inline void sample_ternary(NTL::ZZX& v, long N) {
  std::vector<uint8_t> b;
  fill_bytes(b, (size_t)(2 * N + 7) / 8);
  v.SetLength(N);
  for (long i = 0; i < N; ++i) {
    const int b0 = (b[(size_t)(2 * i) >> 3] >> ((2 * i) & 7)) & 1;
    const int b1 = (b[(size_t)(2 * i + 1) >> 3] >> ((2 * i + 1) & 7)) & 1;
    v.rep[i] = NTL::ZZ(b0 == 0 ? 0 : (b1 == 0 ? 1 : -1));
  }
}
// centered binomial, eta = 20 (|e| <= 20, variance 10): integer-only, platform independent
inline void sample_cbd(NTL::ZZX& e, long N) {
  std::vector<uint8_t> b;
  fill_bytes(b, (size_t)(5 * N));
  e.SetLength(N);
  for (long i = 0; i < N; ++i) {
    uint64_t x = 0;
    for (int k = 0; k < 5; ++k) x |= (uint64_t)b[(size_t)(5 * i + k)] << (8 * k);
    e.rep[i] = NTL::ZZ((long)__builtin_popcountll(x & 0xFFFFF) - (long)__builtin_popcountll((x >> 20) & 0xFFFFF));
  }
}
}  // namespace detail

// c2 = alpha * c1 + Enc_pk(r) with alpha, r and the encryption randomness drawn from
// `seed`. Deterministic given (pk, c1, seed): the server builds c2 with it, the client re-derives
// and compares.
inline Cipher derive_c2(const HeContext& ctx, const PkAtLevel& pk, const Cipher& c1, const std::vector<uint8_t>& seed,
                        MacSecret* secret) {
  const long N = ctx.params.N, L = c1.cbits;
  NTL::ZZ mod = c1.mod;
  NTL::RandomStreamPush push;
  NTL::SetSeed(seed.data(), (long)seed.size());
  MacSecret s;
  s.alpha = NTL::RandomBits_ZZ(L);
  NTL::SetBit(s.alpha, 0);
  NTL::ZZX r;
  r.SetLength(N);
  for (long i = 0; i < N; ++i) r.rep[i] = NTL::RandomBits_ZZ(L);
  NTL::ZZX v, e0, e1;
  detail::sample_ternary(v, N);
  detail::sample_cbd(e0, N);
  detail::sample_cbd(e1, N);
  Ring2Utils::modAndEqual(e0, mod, N);  // into [0, 2^L): HEAAN's AddMod expects reduced inputs
  Ring2Utils::modAndEqual(e1, mod, N);
  // Enc_pk(r) = (ax, bx) = (v a + e1, v b + e0 + r)
  NTL::ZZX pa = pk.ax, pb = pk.bx, ax, bx;
  Ring2Utils::mult(ax, v, pa, mod, N);
  Ring2Utils::addAndEqual(ax, e1, mod, N);
  Ring2Utils::mult(bx, v, pb, mod, N);
  Ring2Utils::addAndEqual(bx, e0, mod, N);
  Ring2Utils::addAndEqual(bx, r, mod, N);
  // alpha * c1 (coefficient-wise)
  NTL::ZZX c1a = c1.ax, c1b = c1.bx, ax2, bx2;
  Ring2Utils::modAndEqual(c1a, mod, N);
  Ring2Utils::modAndEqual(c1b, mod, N);
  Ring2Utils::multByConst(ax2, c1a, s.alpha, mod, N);
  Ring2Utils::multByConst(bx2, c1b, s.alpha, mod, N);
  Ring2Utils::addAndEqual(ax2, ax, mod, N);
  Ring2Utils::addAndEqual(bx2, bx, mod, N);
  s.r0 = r.rep[0] % mod;
  if (secret) *secret = s;
  return Cipher(ax2, bx2, mod, L, c1.slots);
}

inline NTL::ZZ center(const NTL::ZZ& x, const NTL::ZZ& mod) {
  NTL::ZZ v = x % mod;
  if (v >= (mod >> 1)) v -= mod;
  return v;
}

inline void put_residue(std::vector<uint8_t>& out, const NTL::ZZ& x, const NTL::ZZ& mod, long bits) {
  const size_t nb = coeff_bytes(bits);
  const size_t at = out.size();
  out.resize(at + nb);
  NTL::BytesFromZZ(out.data() + at, x % mod, (long)nb);
}
inline NTL::ZZ get_residue(const uint8_t* p, long bits) {
  const size_t nb = coeff_bytes(bits);
  const int pad = (int)(nb * 8 - (size_t)bits);
  if (pad && (p[nb - 1] >> (8 - pad)) != 0) throw ProtocolError("malformed residue (padding bits set)");
  NTL::ZZ x;
  NTL::ZZFromBytes(x, p, (long)nb);
  return x;
}

inline std::vector<uint8_t> commitment(const std::vector<uint8_t>& nonce, const std::vector<uint8_t>& opened_values,
                                       int d, int L) {
  Sha256 h;
  h.update("HELR-open");
  const uint32_t meta[2] = {(uint32_t)d, (uint32_t)L};
  h.update(meta, sizeof meta).update(nonce).update(opened_values);
  return h.final();
}

// ---------------------------------------------------------------- test hooks
// TEST ONLY: deviations that the protocol must catch (see the selftest and `--tamper`).
struct Tamper {
  std::string what;  // client: "opening" | "commit";  server: "c1" | "c2"
  bool is(const char* w) const { return what == w; }
};

// ------------------------------------------------------------------------ client
class HeClient {
 public:
  HeClient(const Params& P, const HeOptions& O, Dataset D, std::vector<int64_t> w0q, Tamper tamper = {})
      : P_(P), O_(O), cfg_(make_he_config(P, O)), D_(std::move(D)), w0q_(std::move(w0q)), tamper_(std::move(tamper)),
        H_(fit_sigmoid_poly(cfg_.deg, cfg_.logR)) {}

  const HeConfig& cfg() const { return cfg_; }
  double tol = std::ldexp(1.0, -8);  // largest accepted |Dec(c1_j) - expected weight_j|
  // Largest |z^T beta| / R of the client's own plaintext run (> 1: the polynomial left its fit interval).
  double max_w_ratio() const { return max_w_; }
  const std::vector<double>& expected() const { return expected_; }
  const std::vector<double>& decrypted() const { return decrypted_; }  // |Dec(c1_j)| / 2^logp, as checked

  // The coin toss. The server's first message arrives first, so the client connects before it
  // generates anything.
  void coin_toss(Channel& ch, Metrics* M = nullptr) {
    ScopedOpt p(M, "coin_toss", &ch);
    const std::vector<uint8_t> com = ch.recv_bytes(32);
    sigma_c_ = os_random_bytes(32);
    ch.send_bytes(make_header(cfg_));
    ch.send_bytes(sigma_c_);
    const std::vector<uint8_t> sigma_s = ch.recv_bytes(32);
    if (Sha256().update("HELR-coin").update(sigma_s).final() != com)
      throw ProtocolError("server's coin-toss opening does not match its commitment");
    sigma_s_ = sigma_s;
  }

  // Keys, encryption of the inputs, and the plaintext run that the final check compares with.
  void setup(Metrics* M = nullptr) {
    {
      ScopedOpt p(M, "setup", nullptr);  // SchemeAux (roots of unity), parameters
      ctx_ = std::make_unique<HeContext>(cfg_);
    }
    {
      ScopedOpt p(M, "keygen", nullptr);
      ctx_->pk.ax = derive_pk_a(sigma_s_, sigma_c_, cfg_.N(), cfg_.logq);
      sk_ = std::make_unique<SecKey>(ctx_->params);
      keygen(*ctx_, *sk_, O_.threads);
    }
    {
      ScopedOpt p(M, "encrypt", nullptr);
      encrypt_inputs(*ctx_, P_, D_, w0q_, O_.threads, in_);
    }
    {
      ScopedOpt p(M, "expected_ref", nullptr);
      expected_ = poly_train(P_, D_, w0q_, H_, &max_w_);
    }
  }

  void upload(Channel& ch, Metrics* M = nullptr) {
    ScopedOpt p(M, "upload_send", &ch);
    send_keys(ch, *ctx_);
    for (const Cipher& c : in_.z) send_cipher(ch, c, cfg_.N());
    for (const Cipher& c : in_.beta) send_cipher(ch, c, cfg_.N());
    ch.flush();
    in_ = EncInputs();
    free_eval_keys(*ctx_);
  }

  // The output phase. Throws ProtocolError (and sends nothing further) if the server deviates.
  void output(Channel& ch, Metrics* M = nullptr) {
    const long L = cfg_.out_bits, N = cfg_.N();
    const int d = cfg_.d;
    const NTL::ZZ mod = NTL::power2_ZZ(L);
    std::vector<Cipher> c1((size_t)d), c2((size_t)d);
    {
      ScopedOpt p(M, "out_recv", &ch);
      for (int j = 0; j < d; ++j) c1[(size_t)j] = recv_cipher(ch, N, L, cfg_.nslots);
      for (int j = 0; j < d; ++j) c2[(size_t)j] = recv_cipher(ch, N, L, cfg_.nslots);
    }
    // 1. decrypt, and check the weights against the client's own plaintext run
    std::vector<NTL::ZZ> u1((size_t)d), u2((size_t)d);
    {
      ScopedOpt p(M, "out_decrypt", nullptr);
      parallel_for(d, O_.threads, [&](int j) {
        u1[(size_t)j] = decrypt_coeff(*ctx_, *sk_, c1[(size_t)j]);
        u2[(size_t)j] = decrypt_coeff(*ctx_, *sk_, c2[(size_t)j]);
      });
    }
    {
      ScopedOpt p(M, "out_check", nullptr);
      decrypted_.assign((size_t)d, 0.0);
      double worst = 0;
      for (int j = 0; j < d; ++j) {
        const double w = zz_to_double(u1[(size_t)j]) / std::ldexp(1.0, cfg_.logp);
        decrypted_[(size_t)j] = w;
        worst = std::max(worst, std::fabs(w - expected_[(size_t)j]));
      }
      worst_dev_ = worst;
      if (!(worst <= tol))
        throw ProtocolError("verification failed: the decrypted weights differ from the client's own computation by " +
                            std::to_string(worst) + " (tolerance " + std::to_string(tol) + "); aborting, nothing released");
    }
    // 2. c2_j must be exactly the honest MAC ciphertext -- but its seed is not known yet, so
    //    commit to the decryptions first and check after the server reveals the seeds.
    std::vector<uint8_t> opened;
    for (int j = 0; j < d; ++j) {
      put_residue(opened, u1[(size_t)j], mod, L);
      put_residue(opened, u2[(size_t)j], mod, L);
    }
    if (tamper_.is("opening")) opened[0] ^= 1;  // lie about weight 0 (u1_0), consistently with the commitment
    const std::vector<uint8_t> nonce = os_random_bytes(32);
    {
      ScopedOpt p(M, "out_commit", &ch);
      ch.send_bytes(commitment(nonce, opened, d, (int)L));
      ch.flush();
    }
    std::vector<std::vector<uint8_t>> seeds((size_t)d);
    {
      ScopedOpt p(M, "out_recv_seeds", &ch);
      for (int j = 0; j < d; ++j) seeds[(size_t)j] = ch.recv_bytes(32);
    }
    {
      ScopedOpt p(M, "out_verify_c2", nullptr);
      const PkAtLevel pk = pk_at_level(*ctx_, L);
      std::atomic<int> bad{0};
      parallel_for(d, O_.threads, [&](int j) {
        const Cipher want = derive_c2(*ctx_, pk, c1[(size_t)j], seeds[(size_t)j], nullptr);
        if (cipher_to_bytes(want, N) != cipher_to_bytes(c2[(size_t)j], N)) bad.fetch_add(1);
      });
      if (bad.load())
        throw ProtocolError("verification failed: the server's c2 is not the honest MAC ciphertext of c1 for " +
                            std::to_string(bad.load()) + " weight(s); aborting, nothing released");
    }
    // 3. open
    {
      ScopedOpt p(M, "out_open", &ch);
      if (tamper_.is("commit")) opened[0] ^= 1;  // open a value other than the committed one
      ch.send_bytes(nonce);
      ch.send_bytes(opened);
      ch.flush();
    }
  }

  double worst_deviation() const { return worst_dev_; }

 private:
  struct ScopedOpt {
    std::optional<ScopedPhase> p;
    ScopedOpt(Metrics* M, const char* name, const Channel* ch) {
      if (M) p.emplace(*M, name, ch);
    }
  };
  Params P_;
  HeOptions O_;
  HeConfig cfg_;
  Dataset D_;
  std::vector<int64_t> w0q_;
  Tamper tamper_;
  HPoly H_;
  std::vector<uint8_t> sigma_c_, sigma_s_;
  std::unique_ptr<HeContext> ctx_;
  std::unique_ptr<SecKey> sk_;
  EncInputs in_;
  std::vector<double> expected_, decrypted_;
  double max_w_ = 0, worst_dev_ = 0;
};

// ------------------------------------------------------------------------ server
class HeServer {
 public:
  HeServer(const Params& P, const HeOptions& O, Tamper tamper = {})
      : P_(P), O_(O), cfg_(make_he_config(P, O)), tamper_(std::move(tamper)), H_(fit_sigmoid_poly(cfg_.deg, cfg_.logR)) {}

  const HeConfig& cfg() const { return cfg_; }
  const std::vector<double>& weights() const { return w_; }
  const std::vector<NTL::ZZ>& raw_weights() const { return u1_; }
  EvalStats& stats() { return stats_; }
  std::function<void(int, double)> progress;  // per-iteration callback

  // Input-independent setup (parameters, roots of unity), before the client connects.
  void setup(Metrics* M = nullptr) {
    ScopedOpt p(M, "setup", nullptr);
    ctx_ = std::make_unique<HeContext>(cfg_);
  }

  void coin_toss(Channel& ch, Metrics* M = nullptr) {
    ScopedOpt p(M, "coin_toss", &ch);
    const std::vector<uint8_t> sigma_s = os_random_bytes(32);
    ch.send_bytes(Sha256().update("HELR-coin").update(sigma_s).final());
    ch.flush();
    const std::vector<uint8_t> hdr = ch.recv_bytes(kHeaderBytes);
    check_header(hdr, cfg_);
    const std::vector<uint8_t> sigma_c = ch.recv_bytes(32);
    ch.send_bytes(sigma_s);
    ch.flush();
    ctx_->pk.ax = derive_pk_a(sigma_s, sigma_c, cfg_.N(), cfg_.logq);
  }

  void receive(Channel& ch, Metrics* M = nullptr) {
    const long N = cfg_.N();
    {
      ScopedOpt p(M, "upload_recv", &ch);
      recv_keys(ch, *ctx_);
      z_.resize((size_t)cfg_.d);
      beta_.resize((size_t)cfg_.d);
      for (auto& c : z_) c = recv_cipher(ch, N, cfg_.logq, cfg_.nslots);
      for (auto& c : beta_) c = recv_cipher(ch, N, cfg_.logq, cfg_.nslots);
    }
  }

  void evaluate(Metrics* M = nullptr) {
    ScopedOpt p(M, "eval", nullptr);
    Evaluator ev(*ctx_, H_, &stats_);
    ev.progress = progress;
    final_ = ev.run(z_, std::move(beta_), O_.threads);
    z_.clear();
    z_.shrink_to_fit();
    beta_.clear();
  }

  // Output phase; throws ProtocolError (abort) on a failed check.
  void output(Channel& ch, Metrics* M = nullptr) {
    const long L = cfg_.out_bits, N = cfg_.N();
    const int d = cfg_.d;
    const NTL::ZZ mod = NTL::power2_ZZ(L);
    std::vector<std::vector<uint8_t>> seeds((size_t)d);
    std::vector<MacSecret> sec((size_t)d);
    std::vector<Cipher> c2((size_t)d);
    {
      ScopedOpt p(M, "mac_derive", nullptr);
      if (tamper_.is("c1")) {  // shift the weight 0 by 2^(logp-3), far beyond the client's tolerance
        NTL::ZZ delta = NTL::power2_ZZ(cfg_.logp - 3);
        NTL::AddMod(final_[0].bx.rep[0], final_[0].bx.rep[0], delta % final_[0].mod, final_[0].mod);
      }
      const PkAtLevel pk = pk_at_level(*ctx_, L);
      for (int j = 0; j < d; ++j) seeds[(size_t)j] = os_random_bytes(32);
      parallel_for(d, O_.threads, [&](int j) {
        c2[(size_t)j] = derive_c2(*ctx_, pk, final_[(size_t)j], seeds[(size_t)j], &sec[(size_t)j]);
      });
      if (tamper_.is("c2")) {  // a c2 that is not derived from the seed
        c2[0].bx.rep[1] += 1;
      }
    }
    {
      ScopedOpt p(M, "out_send", &ch);
      for (int j = 0; j < d; ++j) send_cipher(ch, final_[(size_t)j], N);
      for (int j = 0; j < d; ++j) send_cipher(ch, c2[(size_t)j], N);
      ch.flush();
    }
    std::vector<uint8_t> com;
    {
      ScopedOpt p(M, "out_wait", &ch);  // the client decrypts and checks
      com = ch.recv_bytes(32);
    }
    {
      ScopedOpt p(M, "out_send_seeds", &ch);
      for (int j = 0; j < d; ++j) ch.send_bytes(seeds[(size_t)j]);
      ch.flush();
    }
    std::vector<uint8_t> nonce, opened;
    {
      ScopedOpt p(M, "out_recv_open", &ch);
      nonce = ch.recv_bytes(32);
      opened = ch.recv_bytes((size_t)2 * d * coeff_bytes(L));
    }
    {
      ScopedOpt p(M, "mac_verify", nullptr);
      if (commitment(nonce, opened, d, (int)L) != com)
        throw ProtocolError("client's opening does not match its commitment; aborting");
      const NTL::ZZ E = enc_noise_bound(*ctx_);
      const NTL::ZZ wmax = scale_up(weight_bound(P_), cfg_.logp);
      const size_t nb = coeff_bytes(L);
      u1_.assign((size_t)d, NTL::ZZ(0));
      w_.assign((size_t)d, 0.0);
      for (int j = 0; j < d; ++j) {
        const NTL::ZZ U1 = get_residue(opened.data() + (size_t)(2 * j) * nb, (int)L);
        const NTL::ZZ U2 = get_residue(opened.data() + (size_t)(2 * j + 1) * nb, (int)L);
        const NTL::ZZ D = center(U2 - sec[(size_t)j].alpha * U1 - sec[(size_t)j].r0, mod);
        if (NTL::abs(D) > E)
          throw ProtocolError("MAC check failed for weight " + std::to_string(j) + ": the client's decryption is inconsistent");
        const NTL::ZZ u = center(U1, mod);
        if (NTL::abs(u) > wmax)
          throw ProtocolError("weight " + std::to_string(j) + " is outside the range an honest run can produce");
        u1_[(size_t)j] = u;
        w_[(size_t)j] = zz_to_double(u) / std::ldexp(1.0, cfg_.logp);
      }
    }
  }

 private:
  struct ScopedOpt {
    std::optional<ScopedPhase> p;
    ScopedOpt(Metrics* M, const char* name, const Channel* ch) {
      if (M) p.emplace(*M, name, ch);
    }
  };
  Params P_;
  HeOptions O_;
  HeConfig cfg_;
  Tamper tamper_;
  HPoly H_;
  std::unique_ptr<HeContext> ctx_;
  std::vector<Cipher> z_, beta_, final_;
  std::vector<NTL::ZZ> u1_;
  std::vector<double> w_;
  EvalStats stats_;
};

}  // namespace lrhe

#endif  // LRHE_HELR_PROTOCOL_H
