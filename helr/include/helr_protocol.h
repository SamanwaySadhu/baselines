// helr_protocol.h -- the two-party protocol around HELR's homomorphic training, as
// message-level state machines shared by the program and the selftest. This is the
// semi-honest outsourcing protocol of the paper (Kim et al. 2018, Fig. 1(a)):
//
//   CLIENT  data owner and key owner: holds the training data, the initial weights and the
//           HE secret key; ends with the trained weights (it is the decryptor, as in the paper)
//   SERVER  evaluator: no input; sees only ciphertexts and evaluation keys, and learns nothing
//
//   C -> S  UPLOAD: header (parameters), multiplication key, rotation keys 2^0..2^(logn-1),
//           the d encrypted data columns, the d encrypted initial weights          [offline]
//           ... the server evaluates HELR's gradient descent on ciphertexts ...    [online]
//   S -> C  the d encrypted final weights
//   client: decrypts
//
// Security: semi-honest (honest-but-curious) parties, as in the paper: privacy follows from the
// semantic security of the HE scheme (RLWE); nothing in the protocol checks that the other
// party followed it. There are no MACs, commitments, proofs or consistency checks.
#ifndef LRHE_HELR_PROTOCOL_H
#define LRHE_HELR_PROTOCOL_H

#include "helr_circuit.h"
#include "lr_reference.h"

#include <memory>
#include <optional>

namespace lrhe {

struct ProtocolError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ----------------------------------------------------------------------- header
// The client's parameters, sent first so that a configuration mismatch fails loudly instead of
// producing garbage (a convenience, not a security check).
constexpr size_t kHeaderBytes = 64;
constexpr uint32_t kMagic = 0x52454C48;  // "HELR"
constexpr uint32_t kVersion = 2;

inline std::vector<uint8_t> make_header(const HeConfig& c) {
  const uint32_t f[16] = {kMagic, kVersion, (uint32_t)c.logN, (uint32_t)c.logp, (uint32_t)c.deg, 0,
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
    const char* nm[16] = {"magic", "version", "logN", "logp", "deg", "-", "logR", "iters", "batch", "d",
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
  uint64_t header = kHeaderBytes;
  uint64_t keys = 0, data = 0, weights = 0;
  uint64_t out_s2c = 0;
  uint64_t offline_c2s() const { return header + keys + data + weights; }
  uint64_t offline_s2c() const { return 0; }
  uint64_t online_s2c() const { return out_s2c; }
  uint64_t online_c2s() const { return 0; }
};
inline CommLayout comm_layout(const HeConfig& c) {
  CommLayout L;
  L.keys = keys_bytes(c);
  L.data = (uint64_t)c.d * cipher_bytes(c.N(), c.logq);
  L.weights = L.data;
  L.out_s2c = (uint64_t)c.d * cipher_bytes(c.N(), c.out_bits);
  return L;
}

namespace detail {
struct ScopedOpt {
  std::optional<ScopedPhase> p;
  ScopedOpt(Metrics* M, const char* name, const Channel* ch) {
    if (M) p.emplace(*M, name, ch);
  }
};
}  // namespace detail

// ------------------------------------------------------------------------ client
class HeClient {
 public:
  HeClient(const Params& P, const HeOptions& O, Dataset D, std::vector<int64_t> w0q)
      : P_(P), O_(O), cfg_(make_he_config(P, O)), D_(std::move(D)), w0q_(std::move(w0q)) {}

  const HeConfig& cfg() const { return cfg_; }
  const std::vector<double>& weights() const { return weights_; }  // the decrypted trained weights

  // Keys and encryption of the inputs.
  void setup(Metrics* M = nullptr) {
    {
      detail::ScopedOpt p(M, "setup", nullptr);  // SchemeAux (roots of unity), parameters
      ctx_ = std::make_unique<HeContext>(cfg_);
    }
    {
      detail::ScopedOpt p(M, "keygen", nullptr);
      sk_ = std::make_unique<SecKey>(ctx_->params);
      keygen(*ctx_, *sk_, O_.threads);
    }
    {
      detail::ScopedOpt p(M, "encrypt", nullptr);
      encrypt_inputs(*ctx_, P_, D_, w0q_, O_.threads, in_);
    }
  }

  void upload(Channel& ch, Metrics* M = nullptr) {
    detail::ScopedOpt p(M, "upload_send", &ch);
    ch.send_bytes(make_header(cfg_));
    send_keys(ch, *ctx_);
    for (const Cipher& c : in_.z) send_cipher(ch, c, cfg_.N());
    for (const Cipher& c : in_.beta) send_cipher(ch, c, cfg_.N());
    ch.flush();
    in_ = EncInputs();
    free_eval_keys(*ctx_);
  }

  // Receive the encrypted weights and decrypt them.
  void output(Channel& ch, Metrics* M = nullptr) {
    const long L = cfg_.out_bits, N = cfg_.N();
    const int d = cfg_.d;
    std::vector<Cipher> c((size_t)d);
    {
      detail::ScopedOpt p(M, "out_recv", &ch);
      for (int j = 0; j < d; ++j) c[(size_t)j] = recv_cipher(ch, N, L, cfg_.nslots);
    }
    {
      detail::ScopedOpt p(M, "out_decrypt", nullptr);
      weights_.assign((size_t)d, 0.0);
      parallel_for(d, O_.threads, [&](int j) { weights_[(size_t)j] = decrypt_weight(*ctx_, *sk_, c[(size_t)j]); });
    }
  }

 private:
  Params P_;
  HeOptions O_;
  HeConfig cfg_;
  Dataset D_;
  std::vector<int64_t> w0q_;
  std::unique_ptr<HeContext> ctx_;
  std::unique_ptr<SecKey> sk_;
  EncInputs in_;
  std::vector<double> weights_;
};

// ------------------------------------------------------------------------ server
class HeServer {
 public:
  HeServer(const Params& P, const HeOptions& O)
      : P_(P), O_(O), cfg_(make_he_config(P, O)), H_(fit_sigmoid_poly(cfg_.deg, cfg_.logR)) {}

  const HeConfig& cfg() const { return cfg_; }
  EvalStats& stats() { return stats_; }
  std::function<void(int, double)> progress;  // per-iteration callback

  // Input-independent setup (parameters, roots of unity), before the client connects.
  void setup(Metrics* M = nullptr) {
    detail::ScopedOpt p(M, "setup", nullptr);
    ctx_ = std::make_unique<HeContext>(cfg_);
  }

  void receive(Channel& ch, Metrics* M = nullptr) {
    const long N = cfg_.N();
    detail::ScopedOpt p(M, "upload_recv", &ch);
    check_header(ch.recv_bytes(kHeaderBytes), cfg_);
    recv_keys(ch, *ctx_);
    z_.resize((size_t)cfg_.d);
    beta_.resize((size_t)cfg_.d);
    for (auto& c : z_) c = recv_cipher(ch, N, cfg_.logq, cfg_.nslots);
    for (auto& c : beta_) c = recv_cipher(ch, N, cfg_.logq, cfg_.nslots);
  }

  void evaluate(Metrics* M = nullptr) {
    detail::ScopedOpt p(M, "eval", nullptr);
    Evaluator ev(*ctx_, H_, &stats_);
    ev.progress = progress;
    final_ = ev.run(z_, std::move(beta_), O_.threads);
    z_.clear();
    z_.shrink_to_fit();
    beta_.clear();
  }

  // Send the encrypted weights to the client.
  void output(Channel& ch, Metrics* M = nullptr) {
    detail::ScopedOpt p(M, "out_send", &ch);
    for (const Cipher& c : final_) send_cipher(ch, c, cfg_.N());
    ch.flush();
  }

 private:
  Params P_;
  HeOptions O_;
  HeConfig cfg_;
  HPoly H_;
  std::unique_ptr<HeContext> ctx_;
  std::vector<Cipher> z_, beta_, final_;
  EvalStats stats_;
};

}  // namespace lrhe

#endif  // LRHE_HELR_PROTOCOL_H
