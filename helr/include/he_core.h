// he_core.h -- the HEAAN layer of the HELR baseline: parameter selection, OS-seeded
// randomness, key generation (only the keys the circuit uses, in parallel), strict
// (de)serialization of ciphertexts and keys, and a small thread helper.
//
// The scheme is HELR's snapshot of HEAAN (Cheon-Kim-Kim-Song, ASIACRYPT 2017): a
// ciphertext is (ax, bx) over Z_{2^cbits}[X]/(X^N+1) with bx + ax*sx = m + e (mod 2^cbits);
// multiplication / rotation use key-switching keys modulo 2^(2 logq); modSwitch divides
// by a power of two. HELR's headers do `using namespace std/NTL`; everything here is
// qualified and lives in namespace lrhe.
#ifndef LRHE_HE_CORE_H
#define LRHE_HE_CORE_H

#include "helr_reference.h"
#include "lr_params.h"
#include "net.h"

#include <openssl/evp.h>
#include <openssl/rand.h>

#include "Cipher.h"
#include "Message.h"
#include "NumUtils.h"
#include "Params.h"
#include "PubKey.h"
#include "Ring2Utils.h"
#include "Scheme.h"
#include "SchemeAux.h"
#include "SecKey.h"

#include <NTL/ZZ.h>
#include <NTL/ZZX.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace lrhe {

// ------------------------------------------------------------------ randomness
inline void os_random(void* p, size_t n) {
  if (RAND_bytes(static_cast<unsigned char*>(p), (int)n) != 1) throw std::runtime_error("RAND_bytes failed");
}
inline std::vector<uint8_t> os_random_bytes(size_t n) {
  std::vector<uint8_t> b(n);
  os_random(b.data(), n);
  return b;
}
// NTL's generator (used by every HEAAN sampler) is thread-local; seed the calling
// thread's from the OS. Done for the main thread at startup and by parallel_for for every
// worker, so no two threads (or runs) ever share a stream.
inline void seed_ntl_from_os() {
  unsigned char b[32];
  os_random(b, sizeof b);
  NTL::SetSeed(b, (long)sizeof b);
}

// f(i) for i in [0, n) on up to `threads` threads (the caller's thread included).
// Exceptions are propagated (the first one wins).
template <class F>
void parallel_for(int n, int threads, F&& f) {
  if (n <= 0) return;
  threads = std::max(1, std::min(threads, n));
  if (threads == 1) {
    for (int i = 0; i < n; ++i) f(i);
    return;
  }
  std::atomic<int> next{0};
  std::exception_ptr err;
  std::mutex mu;
  auto worker = [&](bool reseed) {
    try {
      if (reseed) seed_ntl_from_os();
      for (;;) {
        const int i = next.fetch_add(1);
        if (i >= n) break;
        f(i);
      }
    } catch (...) {
      std::lock_guard<std::mutex> g(mu);
      if (!err) err = std::current_exception();
      next.store(n);
    }
  };
  std::vector<std::thread> ts;
  for (int t = 1; t < threads; ++t) ts.emplace_back(worker, true);
  worker(false);
  for (auto& t : ts) t.join();
  if (err) std::rethrow_exception(err);
}

// --------------------------------------------------------------------- config
// Everything both parties must agree on (the client sends it in the upload header, the
// server recomputes it from its own flags and compares).
struct HeConfig {
  // training shape
  int T = 0, B = 0, d = 0;       // iterations, batch, features (all encrypted, bias included)
  int k_shift = 0;               // K = B / lr = 2^k_shift
  int nslots = 0, logn = 0;      // slots = pow2 >= T*B ; logn = log2(nslots)
  // HE
  int logN = 0;                  // ring dimension N = 2^logN
  int logp = 28;                 // scaling bits (HELR: 28)
  int deg = 3;                   // degree of the least-squares sigmoid polynomial: 3 or 7
  int mac_bits = 48;             // extra modulus bits kept at the output for the decryption MAC
  int lambda = 128;              // target security level (HELR's formula, see lambda_est)
  int logR = 3;                  // the polynomial is fitted on [-R, R], R = 2^logR (HELR: 8)
  // derived
  int lev_iter = 0;              // modulus bits one iteration consumes: (deg == 3 ? 3 : 4) * logp
  int out_bits = 0;              // modulus of the output ciphertexts: logp + 10 + mac_bits
  int logq = 0;                  // fresh-ciphertext modulus: T * lev_iter + out_bits
  int nrot = 0;                  // rotation keys 2^0 .. 2^(logn-1)

  long N() const { return 1L << logN; }
  double R() const { return std::ldexp(1.0, logR); }
  // HELR's estimate (its Sec. 4.1, from Gentry-Halevi-Smart): N >= (lambda + 110) / 7.2 * log(qQ),
  // here with the key-switching modulus Q = q, i.e. log(qQ) = 2 logq.
  static int lambda_est(int logN, int logq) {
    return (int)std::floor(7.2 * std::ldexp(1.0, logN) / (2.0 * logq)) - 110;
  }
  int lambda_est() const { return lambda_est(logN, logq); }

  bool operator==(const HeConfig& o) const {
    return T == o.T && B == o.B && d == o.d && k_shift == o.k_shift && nslots == o.nslots && logN == o.logN &&
           logp == o.logp && deg == o.deg && mac_bits == o.mac_bits && logR == o.logR && logq == o.logq;
  }
};

struct HeOptions {
  int logN = 0;        // 0 = smallest meeting `lambda`
  int logp = 28;
  int deg = 3;
  int mac_bits = 48;
  int lambda = 128;
  int logR = -1;       // -1 = automatic from d (see auto_logR)
  int threads = 1;
  bool insecure_ok = false;  // allow a logN below the target (tests only)
};

// R = 2^logR so that |z^T w| <= R holds with high probability for the generated data
// (features and weights uniform in [-1, 1): z^T w has std sqrt(d)/3): R >= 2 sqrt(d), at least 8.
inline int auto_logR(int d) {
  int r = 3;
  while (std::ldexp(1.0, r) < 2.0 * std::sqrt((double)d)) ++r;
  return r;
}

inline HeConfig make_he_config(const Params& P, const HeOptions& O) {
  if (O.deg != 3 && O.deg != 7) throw std::invalid_argument("--deg must be 3 or 7");
  if (O.logp < 16 || O.logp > 40) throw std::invalid_argument("--logp must be in [16, 40]");
  if (O.mac_bits < 24 || O.mac_bits > 200) throw std::invalid_argument("--mac-bits must be in [24, 200]");
  HeConfig C;
  C.T = P.iters;
  C.B = P.batch;
  C.d = P.d;
  C.k_shift = P.k_shift;
  C.logp = O.logp;
  C.deg = O.deg;
  C.mac_bits = O.mac_bits;
  C.lambda = O.lambda;
  C.logR = O.logR >= 0 ? O.logR : auto_logR(P.d);
  if (C.logR > 20) throw std::invalid_argument("--logr too large");
  const long total = (long)C.T * C.B;
  C.nslots = 1;
  while (C.nslots < total) C.nslots <<= 1;
  C.logn = 0;
  while ((1 << C.logn) < C.nslots) ++C.logn;
  C.nrot = C.logn;
  C.lev_iter = (C.deg == 3 ? 3 : 4) * C.logp;
  C.out_bits = C.logp + 10 + C.mac_bits;
  C.logq = C.T * C.lev_iter + C.out_bits;
  if (O.logN > 0) {
    C.logN = O.logN;
  } else {
    C.logN = 10;
    while (C.lambda_est() < C.lambda || (1L << C.logN) < 2L * C.nslots) ++C.logN;
  }
  if (mac_forgery_bits(C.out_bits, C.logp, P) < 40)
    throw std::invalid_argument("--mac-bits " + std::to_string(C.mac_bits) + " gives a decryption-MAC forgery probability of only 2^-" +
                                std::to_string((int)mac_forgery_bits(C.out_bits, C.logp, P)) + " (need 2^-40); raise --mac-bits");
  if (C.logN < 6 || C.logN > 20) throw std::invalid_argument("--logn must be in [6, 20]");
  if (2L * C.nslots > C.N()) throw std::invalid_argument("too many slots for this ring dimension");
  if (!O.insecure_ok && C.lambda_est() < C.lambda)
    throw std::invalid_argument("logN = " + std::to_string(C.logN) + " gives only ~" + std::to_string(C.lambda_est()) +
                                " bits by HELR's estimate (< --lambda " + std::to_string(C.lambda) +
                                "); raise --logn or pass --insecure for tests");
  return C;
}

// ------------------------------------------------------------- HEAAN context
// Params + SchemeAux + PubKey + Scheme with stable addresses (Scheme keeps references).
struct HeContext {
  HeConfig cfg;
  long logN_;
  ::Params params;
  SchemeAux aux;
  PubKey pk;
  Scheme scheme;

  explicit HeContext(const HeConfig& c)
      : cfg(c), logN_(c.logN), params(c.logN, c.logq, 3.2, std::min<long>(64, std::max<long>(2, c.N() / 4))),
        aux(logN_), pk(), scheme(params, pk, aux) {
    pk.axLeftRot = new ZZX[params.logN - 1];
    pk.bxLeftRot = new ZZX[params.logN - 1];
    pk.axRightRot = new ZZX[params.logN - 1];
    pk.bxRightRot = new ZZX[params.logN - 1];
  }
  ~HeContext() {
    delete[] pk.axLeftRot;
    delete[] pk.bxLeftRot;
    delete[] pk.axRightRot;
    delete[] pk.bxRightRot;
  }
  HeContext(const HeContext&) = delete;
  HeContext& operator=(const HeContext&) = delete;
  long N() const { return params.N; }
};

// Drop the evaluation keys (the client does not need them after the upload).
inline void free_eval_keys(HeContext& c) {
  c.pk.axStar.kill();
  c.pk.bxStar.kill();
  for (int i = 0; i < c.cfg.nrot; ++i) {
    c.pk.axLeftRot[i].kill();
    c.pk.bxLeftRot[i].kill();
  }
}

// Key generation as in HELR's PubKey constructor, restricted to the keys the circuit needs:
// the public key, the multiplication key and the left-rotation keys 2^0 .. 2^(nrot-1).
// (No conjugation and no right-rotation keys; HELR's constructor makes ~5 GB of keys.)
// The keys are independent, so they are generated in parallel.
inline void keygen(HeContext& c, const SecKey& sk, int threads) {
  ::Params& params = c.params;
  PubKey& pk = c.pk;
  const int n_tasks = 2 + c.cfg.nrot;  // pk, mult, rot_0 .. rot_{nrot-1}
  parallel_for(n_tasks, threads, [&](int task) {
    ZZX ex;
    ZZX sx = sk.sx;
    if (task == 0) {
      // pk.ax is NOT sampled here: the caller sets it to the jointly random element of the
      // coin toss (derive_pk_a), so that the server's decryption MAC (helr_protocol.h) is
      // sound even against a client that tries to choose a structured public key.
      if (pk.ax.rep.length() != params.N) throw std::logic_error("keygen: pk.ax must be set first");
      NumUtils::sampleGauss(ex, params.N, params.sigma);
      Ring2Utils::mult(pk.bx, sx, pk.ax, params.q, params.N);
      Ring2Utils::sub(pk.bx, ex, pk.bx, params.q, params.N);
    } else if (task == 1) {
      ZZX sxsx;
      Ring2Utils::mult(sxsx, sx, sx, params.q, params.N);
      Ring2Utils::leftShiftAndEqual(sxsx, params.logq, params.qq, params.N);
      NumUtils::sampleUniform2(pk.axStar, params.N, params.logqq);
      NumUtils::sampleGauss(ex, params.N, params.sigma);
      Ring2Utils::addAndEqual(ex, sxsx, params.qq, params.N);
      Ring2Utils::mult(pk.bxStar, sx, pk.axStar, params.qq, params.N);
      Ring2Utils::sub(pk.bxStar, ex, pk.bxStar, params.qq, params.N);
    } else {
      const int i = task - 2;
      ZZX spow;
      const long ipow = 1L << i;
      Ring2Utils::inpower(spow, sx, params.rotGroup[ipow], params.q, params.N);
      Ring2Utils::leftShiftAndEqual(spow, params.logq, params.qq, params.N);
      NumUtils::sampleUniform2(pk.axLeftRot[i], params.N, params.logqq);
      NumUtils::sampleGauss(ex, params.N, params.sigma);
      Ring2Utils::addAndEqual(ex, spow, params.qq, params.N);
      Ring2Utils::mult(pk.bxLeftRot[i], sx, pk.axLeftRot[i], params.qq, params.N);
      Ring2Utils::sub(pk.bxLeftRot[i], ex, pk.bxLeftRot[i], params.qq, params.N);
    }
  });
}

// ---------------------------------------------------------------- serialization
// A ring element over Z_{2^bits}: N coefficients, each ceil(bits/8) bytes little-endian,
// reduced into [0, 2^bits). The receiver rejects any element with set padding bits.
inline size_t coeff_bytes(long bits) { return (size_t)((bits + 7) / 8); }
inline size_t elem_bytes(long N, long bits) { return (size_t)N * coeff_bytes(bits); }

inline void put_elem(uint8_t* out, const NTL::ZZX& x, long N, const NTL::ZZ& mod, long bits) {
  const size_t nb = coeff_bytes(bits);
  const long len = x.rep.length();
  for (long i = 0; i < N; ++i) {
    NTL::ZZ c = i < len ? x.rep[i] : NTL::ZZ(0);
    c %= mod;  // NTL: result in [0, mod)
    NTL::BytesFromZZ(out + (size_t)i * nb, c, (long)nb);
  }
}
inline void get_elem(const uint8_t* in, NTL::ZZX& x, long N, long bits) {
  const size_t nb = coeff_bytes(bits);
  const int pad = (int)(nb * 8 - (size_t)bits);
  x.SetLength(N);
  for (long i = 0; i < N; ++i) {
    const uint8_t* p = in + (size_t)i * nb;
    if (pad && (p[nb - 1] >> (8 - pad)) != 0) throw std::runtime_error("malformed ring element (padding bits set)");
    NTL::ZZFromBytes(x.rep[i], p, (long)nb);
  }
}

inline size_t cipher_bytes(long N, long bits) { return 2 * elem_bytes(N, bits); }

inline std::vector<uint8_t> cipher_to_bytes(const Cipher& c, long N) {
  std::vector<uint8_t> b(cipher_bytes(N, c.cbits));
  const size_t e = elem_bytes(N, c.cbits);
  put_elem(b.data(), c.ax, N, c.mod, c.cbits);
  put_elem(b.data() + e, c.bx, N, c.mod, c.cbits);
  return b;
}
inline Cipher cipher_from_bytes(const uint8_t* b, long N, long bits, long slots) {
  Cipher c;
  const size_t e = elem_bytes(N, bits);
  get_elem(b, c.ax, N, bits);
  get_elem(b + e, c.bx, N, bits);
  c.mod = NTL::power2_ZZ(bits);
  c.cbits = bits;
  c.slots = slots;
  return c;
}

inline void send_cipher(Channel& ch, const Cipher& c, long N) { ch.send_bytes(cipher_to_bytes(c, N)); }
inline Cipher recv_cipher(Channel& ch, long N, long bits, long slots) {
  const auto b = ch.recv_bytes(cipher_bytes(N, bits));
  return cipher_from_bytes(b.data(), N, bits, slots);
}

// Key material, in this order: the public key's bx (one element mod q; its ax is the
// coin-tossed element both parties derive), the multiplication key and the nrot
// left-rotation keys (2 elements each, mod q^2).
inline uint64_t keys_bytes(const HeConfig& c) {
  return (uint64_t)elem_bytes(c.N(), c.logq) + (uint64_t)(1 + c.nrot) * cipher_bytes(c.N(), 2L * c.logq);
}
inline void send_keys(Channel& ch, HeContext& c) {
  const ::Params& p = c.params;
  const long N = p.N;
  auto put1 = [&](const NTL::ZZX& a, const NTL::ZZ& mod, long bits) {
    std::vector<uint8_t> buf(elem_bytes(N, bits));
    put_elem(buf.data(), a, N, mod, bits);
    ch.send_bytes(buf);
  };
  put1(c.pk.bx, p.q, p.logq);
  put1(c.pk.axStar, p.qq, p.logqq);
  put1(c.pk.bxStar, p.qq, p.logqq);
  for (int i = 0; i < c.cfg.nrot; ++i) {
    put1(c.pk.axLeftRot[i], p.qq, p.logqq);
    put1(c.pk.bxLeftRot[i], p.qq, p.logqq);
  }
}
// pk.ax must already be set (derive_pk_a).
inline void recv_keys(Channel& ch, HeContext& c) {
  const long N = c.params.N;
  auto get1 = [&](NTL::ZZX& a, long bits) {
    const auto buf = ch.recv_bytes(elem_bytes(N, bits));
    get_elem(buf.data(), a, N, bits);
  };
  get1(c.pk.bx, c.params.logq);
  get1(c.pk.axStar, c.params.logqq);
  get1(c.pk.bxStar, c.params.logqq);
  for (int i = 0; i < c.cfg.nrot; ++i) {
    get1(c.pk.axLeftRot[i], c.params.logqq);
    get1(c.pk.bxLeftRot[i], c.params.logqq);
  }
}

// ------------------------------------------------------------------- encoding
// round(v * 2^bits) as an exact integer (|v| < 2^(62-bits)).
inline NTL::ZZ scale_up(double v, int bits) {
  const long double s = std::ldexp((long double)v, bits);
  if (!(std::fabs((double)s) < 4.0e18)) throw std::runtime_error("scale_up: value out of range");
  return NTL::to_ZZ((long)std::llround(s));
}

// Encrypt a vector of nslots integers (already scaled) at the top modulus.
inline Cipher encrypt_scaled(HeContext& c, const std::vector<NTL::ZZ>& v) {
  const long slots = c.cfg.nslots;
  CZZ* vals = new CZZ[slots];
  for (long k = 0; k < slots; ++k) vals[k].r = k < (long)v.size() ? v[k] : NTL::ZZ(0);
  Cipher ct = c.scheme.encrypt(vals, slots);
  delete[] vals;
  return ct;
}

// Plaintext polynomial with the given (scaled) slot values, for multByPoly.
inline Message encode_scaled(HeContext& c, const std::vector<NTL::ZZ>& v) {
  const long slots = c.cfg.nslots;
  CZZ* vals = new CZZ[slots];
  for (long k = 0; k < slots; ++k) vals[k].r = k < (long)v.size() ? v[k] : NTL::ZZ(0);
  CZZ* g = c.scheme.groupidx(vals, slots);
  Message m = c.scheme.encode(g, slots);
  delete[] g;
  delete[] vals;
  return m;
}

// Centered coefficient i of the decryption polynomial of `ct`: bx + ax*sx mod 2^cbits, in
// (-2^(cbits-1), 2^(cbits-1)]. For a ciphertext whose slots all hold the same real v this
// is round(v * 2^logp) + noise at i = 0 (the other coefficients on the slot grid are pure noise).
inline NTL::ZZ decrypt_coeff(HeContext& c, const SecKey& sk, const Cipher& ct, long i = 0) {
  Message m = c.scheme.decryptMsg(const_cast<SecKey&>(sk), const_cast<Cipher&>(ct));
  NTL::ZZ v = m.mx.rep[i] % ct.mod;
  if (v >= (ct.mod >> 1)) v -= ct.mod;
  return v;
}

inline double zz_to_double(const NTL::ZZ& z) {
  double d;
  NTL::conv(d, z);
  return d;
}

// ------------------------------------------------------------------ SHA-256
class Sha256 {
 public:
  Sha256() : c_(EVP_MD_CTX_new()) {
    if (!c_ || EVP_DigestInit_ex(c_, EVP_sha256(), nullptr) != 1) throw std::runtime_error("SHA-256 init failed");
  }
  ~Sha256() { EVP_MD_CTX_free(c_); }
  Sha256(const Sha256&) = delete;
  Sha256& operator=(const Sha256&) = delete;
  Sha256& update(const void* p, size_t n) {
    if (EVP_DigestUpdate(c_, p, n) != 1) throw std::runtime_error("SHA-256 update failed");
    return *this;
  }
  Sha256& update(const std::vector<uint8_t>& v) { return update(v.data(), v.size()); }
  Sha256& update(const char* s) { return update(s, std::strlen(s)); }
  std::vector<uint8_t> final() {
    std::vector<uint8_t> out(32);
    unsigned len = 0;
    if (EVP_DigestFinal_ex(c_, out.data(), &len) != 1 || len != 32) throw std::runtime_error("SHA-256 final failed");
    return out;
  }

 private:
  EVP_MD_CTX* c_;
};

// The jointly random public-key element a of the coin toss: both parties derive it from
// the server's committed seed and the client's seed. NTL's generator (ChaCha20 keyed by
// the SHA-256 of the two seeds) expands it; the sampling is integer-only, so it is the
// same on every platform.
inline NTL::ZZX derive_pk_a(const std::vector<uint8_t>& seed_server, const std::vector<uint8_t>& seed_client, long N,
                            long logq) {
  Sha256 h;
  h.update("HELR-pk-a").update(seed_server).update(seed_client);
  const std::vector<uint8_t> key = h.final();
  NTL::RandomStreamPush push;
  NTL::SetSeed(key.data(), (long)key.size());
  NTL::ZZX a;
  a.SetLength(N);
  for (long i = 0; i < N; ++i) a.rep[i] = NTL::RandomBits_ZZ(logq);
  return a;
}

}  // namespace lrhe

#endif  // LRHE_HE_CORE_H
