// cryptonn_protocol.h -- the CryptoNN logistic-regression protocol (Xu, Joshi, Li,
// "CryptoNN: Training Neural Networks over Encrypted Data", ICDCS 2019), as message-level
// state machines for the two parties. lr_cryptonn.cpp drives them over TCP and
// lr_cryptonn_selftest.cpp in-process, so both run exactly this code.
//
//   CLIENT (CnClient) = data owner AND the paper's trusted authority: holds the data
//                       (x, y), the initial weights w0 and every master secret key, and
//                       derives function keys when the server asks for them.
//   SERVER (CnServer) = model holder: keeps w in the clear and runs the paper's
//                       Algorithm 2, using FE only for the steps that touch the data.
//
// Fixed-point training of lr_reference.h, per iteration t (batch i < B, f = qbits-1,
// MAXQ = 2^f - 1, K = batch/lr = 2^k):
//   zfull_i = <w_data, x_i>  [FEIP fwd]  + w_bias*MAXQ  [public bias column, plaintext]
//   z_i = zfull_i >> f;  yhat_i = qsigmoid_ref(z_i, f)   [plaintext at the server]
//   e_i = yhat_i - MAXQ*y_i  [FEBO '-': decrypts to MAXQ*y_i - yhat_i, the server negates]
//   gfull_j = <e, (x_0j..x_{B-1}j)>  [FEIP bwd], j < d_data;  gfull_bias = MAXQ*sum_i e_i
//   w_j <- (K*w_j - (gfull_j >> f)) / K   (C++ '/': toward zero)
// so the result equals reference_train bit for bit. With --no-bias every one of the d
// columns is client data (d_data = d) and there are no plaintext bias terms.
//
// Offline upload (ONE client->server flight), layout from upload_layout():
//   header (32 B: magic "CNN1", d, iters, batch, qbits, flags, lr as IEEE bits)
//   w0       d x int32 LE                        (CryptoNN's server needs w in the clear)
//   FEIP fwd iters*B ciphertexts of x_i (eta = d_data), (d_data+1)*33 B each
//   FEIP bwd iters*d_data ciphertexts of column j of batch t (eta = B), (B+1)*33 B each
//   FEBO     iters*B ciphertexts (cmt, ct) of MAXQ*y_i, 66 B each
// The three key sets (FEIP fwd, FEIP bwd, FEBO) are independent. mpk is not sent:
// decryption does not use it, and the client both encrypts and derives keys.
//
// Online, per iteration (3 round trips):
//   R1 S->C w_0..w_{d_data-1} (int32)  | C->S sk_w = <w, s_fwd> mod q        (32 B)
//   R2 S->C yhat_0..yhat_{B-1} (int32) | C->S B FEBO '-' keys                (33 B each)
//   R3 S->C e_0..e_{B-1} (int32)       | C->S sk_e = <e, s_bwd> mod q        (32 B)
//
// Every decryption ends in a bounded discrete log with an EXACT bound (|x| <= 2^f):
//   forward  2^f * sum_j |w_j|,   FEBO  2*MAXQ,   backward  2^f * sum_i |e_i|
// so a solve can only fail if the protocol is broken; a failure throws DlogError (it is
// never silently replaced by a wrong value). set_test_bound_cap() shrinks the bounds to
// demonstrate this (tests only).
//
// Threads: encryption (client) and decryption + dlog (server) are split over --threads
// std::threads by index; every result is independent of the thread count.
#ifndef LRCN_CRYPTONN_PROTOCOL_H
#define LRCN_CRYPTONN_PROTOCOL_H

#include "bsgs.h"
#include "ec_group.h"
#include "fe_bo.h"
#include "fe_ip.h"
#include "lr_params.h"
#include "lr_reference.h"
#include "metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace lrcn {

// ------------------------------------------------------------- wire layout
constexpr uint32_t kUploadMagic = 0x314E4E43u;  // "CNN1" little-endian
constexpr size_t kHeaderBytes = 32;

struct UploadLayout {
  size_t header = 0, w0 = 0, fwd = 0, bwd = 0, febo = 0, total = 0;  // section sizes
  size_t off_w0 = 0, off_fwd = 0, off_bwd = 0, off_febo = 0;         // section offsets
  size_t fwd_ct = 0, bwd_ct = 0, febo_ct = 0;                        // one ciphertext
};

inline UploadLayout upload_layout(const Params& P) {
  UploadLayout L;
  L.fwd_ct = feip::ct_bytes(P.d_data);
  L.bwd_ct = feip::ct_bytes(P.batch);
  L.febo_ct = febo::kCtBytes;
  L.header = kHeaderBytes;
  L.w0 = 4 * (size_t)P.d;
  L.fwd = (size_t)P.iters * P.batch * L.fwd_ct;
  L.bwd = (size_t)P.iters * P.d_data * L.bwd_ct;
  L.febo = (size_t)P.iters * P.batch * L.febo_ct;
  L.off_w0 = L.header;
  L.off_fwd = L.off_w0 + L.w0;
  L.off_bwd = L.off_fwd + L.fwd;
  L.off_febo = L.off_bwd + L.bwd;
  L.total = L.off_febo + L.febo;
  return L;
}
// Online payload per iteration (both directions).
inline size_t online_s2c_per_iter(const Params& P) { return 4 * ((size_t)P.d_data + 2 * (size_t)P.batch); }
inline size_t online_c2s_per_iter(const Params& P) {
  return kScalarBytes + febo::kKeyBytes * (size_t)P.batch + kScalarBytes;
}

inline void put_u32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i)); }
inline uint32_t get_u32(const uint8_t* p) {
  uint32_t v = 0;
  for (int i = 0; i < 4; ++i) v |= (uint32_t)p[i] << (8 * i);
  return v;
}
inline void put_u64(uint8_t* p, uint64_t v) { for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i)); }
inline uint64_t get_u64(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= (uint64_t)p[i] << (8 * i);
  return v;
}
inline void put_i32(uint8_t* p, int64_t v) {
  if (v < std::numeric_limits<int32_t>::min() || v > std::numeric_limits<int32_t>::max())
    throw std::runtime_error("value " + std::to_string(v) + " does not fit in int32");
  put_u32(p, (uint32_t)(int32_t)v);
}
inline int64_t get_i32(const uint8_t* p) { return (int32_t)get_u32(p); }

// The public training parameters, so the two parties can check that they agree.
inline void encode_header(const Params& P, uint8_t* h) {
  uint64_t lr_bits;
  std::memcpy(&lr_bits, &P.lr, 8);
  put_u32(h + 0, kUploadMagic);
  put_u32(h + 4, (uint32_t)P.d);
  put_u32(h + 8, (uint32_t)P.iters);
  put_u32(h + 12, (uint32_t)P.batch);
  put_u32(h + 16, (uint32_t)P.qbits);
  put_u32(h + 20, P.bias ? 1u : 0u);
  put_u64(h + 24, lr_bits);
}
// Throws std::runtime_error naming every field that differs from P.
inline void check_header(const uint8_t* h, const Params& P) {
  if (get_u32(h) != kUploadMagic) throw std::runtime_error("upload header: bad magic (not a CryptoNN client?)");
  double lr;
  const uint64_t lr_bits = get_u64(h + 24);
  std::memcpy(&lr, &lr_bits, 8);
  std::string diff;
  auto cmp = [&](const char* name, long long theirs, long long mine) {
    if (theirs != mine) diff += std::string(" ") + name + "=" + std::to_string(theirs) + " (here " + std::to_string(mine) + ")";
  };
  cmp("--d", get_u32(h + 4), P.d);
  cmp("--iters", get_u32(h + 8), P.iters);
  cmp("--batch", get_u32(h + 12), P.batch);
  cmp("--qbits", get_u32(h + 16), P.qbits);
  cmp("bias", get_u32(h + 20), P.bias ? 1 : 0);
  if (lr != P.lr) diff += " --lr=" + std::to_string(lr) + " (here " + std::to_string(P.lr) + ")";
  if (!diff.empty()) throw std::runtime_error("the client's training parameters differ:" + diff);
}

// ------------------------------------------------------------- threading
// f(i) for i in [0, n), split into contiguous blocks over `threads` std::threads; the
// first exception (by block) is rethrown after all threads have joined.
template <class F>
void parallel_for(int threads, size_t n, F&& f) {
  if (threads <= 1 || n <= 1) {
    for (size_t i = 0; i < n; ++i) f(i);
    return;
  }
  const size_t T = std::min<size_t>((size_t)threads, n);
  std::vector<std::thread> th;
  std::vector<std::exception_ptr> err(T);
  th.reserve(T);
  for (size_t k = 0; k < T; ++k)
    th.emplace_back([&, k] {
      try {
        for (size_t i = n * k / T; i < n * (k + 1) / T; ++i) f(i);
      } catch (...) {
        err[k] = std::current_exception();
      }
    });
  for (auto& t : th) t.join();
  for (auto& e : err)
    if (e) std::rethrow_exception(e);
}

// A bounded discrete log found no value: the protocol is broken (or a test shrank the
// bound). Never ignored.
struct DlogError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Rejects parameter combinations this baseline does not support.
inline void check_protocol_params(const Params& P) {
  if (P.d_data < 1)
    throw std::invalid_argument("CryptoNN needs at least one client feature column (--d >= 2 with the bias column)");
}

// ============================================================= the client
class CnClient {
 public:
  // enc_with_msk: encrypt with the master secret key (ct_i = (r*s_i + x_i)*g, one
  // fixed-base multiplication per component). Same ciphertext distribution as the
  // paper's public-key encryption (ct_i = x_i*g + r*h_i); available because the client
  // is also the authority. Default: public key, as in the paper.
  CnClient(const Params& P, Dataset D, std::vector<int64_t> w0, int threads, bool enc_with_msk = false)
      : P_(P), D_(std::move(D)), w0_(std::move(w0)), threads_(std::max(1, threads)), msk_enc_(enc_with_msk) {
    check_protocol_params(P_);
    if ((int)w0_.size() != P_.d) throw std::invalid_argument("CnClient: |w0| != d");
  }

  // Setup of the three independent key sets (FEIP fwd: eta = d_data; FEIP bwd: eta = B; FEBO).
  void keygen() {
    std::tie(fwd_mpk_, fwd_msk_) = feip::setup(P_.d_data);
    std::tie(bwd_mpk_, bwd_msk_) = feip::setup(P_.batch);
    std::tie(bo_mpk_, bo_msk_) = febo::setup();
    have_keys_ = true;
  }

  // Encrypts the whole training set into the upload buffer (see the file comment).
  void encrypt() {
    if (!have_keys_) throw std::logic_error("CnClient::encrypt before keygen");
    const UploadLayout L = upload_layout(P_);
    upload_.assign(L.total, 0);
    encode_header(P_, upload_.data());
    for (int j = 0; j < P_.d; ++j) put_i32(upload_.data() + L.off_w0 + 4 * (size_t)j, w0_[(size_t)j]);
    const size_t n_fwd = (size_t)P_.iters * P_.batch, n_bwd = (size_t)P_.iters * P_.d_data, n_bo = n_fwd;
    cmt_.assign(n_bo, Point());
    // One flat index space so that the threads share the work evenly.
    parallel_for(threads_, n_fwd + n_bwd + n_bo, [&](size_t k) {
      if (k < n_fwd) {  // x_i of batch t (data columns only)
        const int t = (int)(k / P_.batch), i = (int)(k % P_.batch);
        std::vector<int64_t> x((size_t)P_.d_data);
        for (int j = 0; j < P_.d_data; ++j) x[(size_t)j] = D_.X(t, i, j);
        const feip::Ct c = msk_enc_ ? feip::encrypt_with_msk(fwd_msk_, x) : feip::encrypt(fwd_mpk_, x);
        feip::serialize(c, upload_.data() + L.off_fwd + k * L.fwd_ct);
      } else if (k < n_fwd + n_bwd) {  // column j of batch t
        const size_t kk = k - n_fwd;
        const int t = (int)(kk / P_.d_data), j = (int)(kk % P_.d_data);
        std::vector<int64_t> col((size_t)P_.batch);
        for (int i = 0; i < P_.batch; ++i) col[(size_t)i] = D_.X(t, i, j);
        const feip::Ct c = msk_enc_ ? feip::encrypt_with_msk(bwd_msk_, col) : feip::encrypt(bwd_mpk_, col);
        feip::serialize(c, upload_.data() + L.off_bwd + kk * L.bwd_ct);
      } else {  // label value MAXQ*y of sample (t, i)
        const size_t kk = k - n_fwd - n_bwd;
        const int t = (int)(kk / P_.batch), i = (int)(kk % P_.batch);
        const int64_t lab = P_.maxq * D_.Y(t, i);
        febo::Ct c;
        if (msk_enc_) {
          const Scalar r = Scalar::random();
          c.cmt = Point::gen_mul(r);
          c.ct = Point::gen_mul(r.mul(bo_msk_.s).add(Scalar::from_i64(lab)));  // r*h + lab*g
        } else {
          c = febo::encrypt(bo_mpk_, lab);
        }
        febo::serialize(c, upload_.data() + L.off_febo + kk * L.febo_ct);
        cmt_[kk] = c.cmt;  // kept to answer the FEBO key request of R2
      }
    });
  }
  const std::vector<uint8_t>& upload() const { return upload_; }

  // R1: sk_w = <w_data, s_fwd> mod q (32 bytes into out).
  void fwd_key(const std::vector<int64_t>& w_data, uint8_t* out) {
    expect(0, "forward key");
    if ((int)w_data.size() != P_.d_data) throw std::runtime_error("forward key request: wrong length");
    feip::serialize_key(feip::key_derive(fwd_msk_, w_data), out);
    stage_ = 1;
  }
  // R2: one FEBO '-' key per sample of batch t for y = yhat_i (B * 33 bytes into out).
  void eval_keys(const std::vector<int64_t>& yhat, uint8_t* out) {
    expect(1, "FEBO keys");
    if ((int)yhat.size() != P_.batch) throw std::runtime_error("FEBO key request: wrong length");
    for (int i = 0; i < P_.batch; ++i) {
      if (yhat[(size_t)i] < 0 || yhat[(size_t)i] > P_.maxq)
        throw std::runtime_error("FEBO key request: yhat outside [0, MAXQ]");
      const Point& cmt = cmt_[(size_t)t_ * P_.batch + i];
      febo::serialize_key(febo::key_derive(bo_msk_, cmt, febo::Op::Sub, yhat[(size_t)i]),
                          out + (size_t)i * febo::kKeyBytes);
    }
    yhat_ = yhat;
    stage_ = 2;
  }
  // R3: sk_e = <e, s_bwd> mod q (32 bytes into out). The authority knows y and the yhat
  // it was asked about, so it also checks e = yhat - MAXQ*y (a cheap sanity check; the
  // model is semi-honest).
  void bwd_key(const std::vector<int64_t>& e, uint8_t* out) {
    expect(2, "backward key");
    if ((int)e.size() != P_.batch) throw std::runtime_error("backward key request: wrong length");
    for (int i = 0; i < P_.batch; ++i)
      if (e[(size_t)i] != yhat_[(size_t)i] - P_.maxq * D_.Y(t_, i))
        throw std::runtime_error("backward key request: e is not yhat - label");
    feip::serialize_key(feip::key_derive(bwd_msk_, e), out);
    stage_ = 0;
    ++t_;
  }
  int iteration() const { return t_; }

 private:
  void expect(int stage, const char* what) const {
    if (t_ >= P_.iters) throw std::runtime_error(std::string(what) + " requested after the last iteration");
    if (stage_ != stage) throw std::runtime_error(std::string(what) + " requested out of order");
  }

  Params P_;
  Dataset D_;
  std::vector<int64_t> w0_;
  int threads_;
  bool msk_enc_;
  bool have_keys_ = false;
  feip::Mpk fwd_mpk_, bwd_mpk_;
  feip::Msk fwd_msk_, bwd_msk_;
  febo::Mpk bo_mpk_;
  febo::Msk bo_msk_;
  std::vector<Point> cmt_;  // FEBO commitments g^r, [t][i]
  std::vector<uint8_t> upload_;
  std::vector<int64_t> yhat_;
  int t_ = 0, stage_ = 0;
};

// ============================================================= the server
// Per-kind statistics of the bounded discrete logs (for the bound-safety tests/report).
struct DlogStats {
  uint64_t calls = 0;
  int64_t max_abs = 0;     // largest |value| recovered
  int64_t max_bound = 0;   // largest bound used
  double max_ratio = 0;    // max |value| / bound (<= 1)
  void add(int64_t v, int64_t bound) {
    ++calls;
    const int64_t a = v < 0 ? -v : v;
    max_abs = std::max(max_abs, a);
    max_bound = std::max(max_bound, bound);
    if (bound > 0) max_ratio = std::max(max_ratio, (double)a / (double)bound);
  }
};

class CnServer {
 public:
  enum Kind { FWD = 0, FEBO = 1, BWD = 2 };

  CnServer(const Params& P, const Dlog& dl, int threads) : P_(P), dl_(dl), threads_(std::max(1, threads)) {
    check_protocol_params(P_);
    K_ = int64_t(1) << P_.k_shift;
  }

  // Header check, w0, and every ciphertext deserialized (each point validated on P-256).
  void parse_upload(const uint8_t* data, size_t n) {
    const UploadLayout L = upload_layout(P_);
    if (n != L.total)
      throw std::runtime_error("upload has " + std::to_string(n) + " bytes, expected " + std::to_string(L.total));
    check_header(data, P_);
    w_.assign((size_t)P_.d, 0);
    for (int j = 0; j < P_.d; ++j) w_[(size_t)j] = get_i32(data + L.off_w0 + 4 * (size_t)j);
    const size_t n_fwd = (size_t)P_.iters * P_.batch, n_bwd = (size_t)P_.iters * P_.d_data, n_bo = n_fwd;
    fwd_.assign(n_fwd, feip::Ct{});
    bwd_.assign(n_bwd, feip::Ct{});
    bo_.assign(n_bo, febo::Ct{});
    parallel_for(threads_, n_fwd + n_bwd + n_bo, [&](size_t k) {
      if (k < n_fwd)
        fwd_[k] = feip::deserialize(data + L.off_fwd + k * L.fwd_ct, P_.d_data);
      else if (k < n_fwd + n_bwd)
        bwd_[k - n_fwd] = feip::deserialize(data + L.off_bwd + (k - n_fwd) * L.bwd_ct, P_.batch);
      else
        bo_[k - n_fwd - n_bwd] = febo::deserialize(data + L.off_febo + (k - n_fwd - n_bwd) * L.febo_ct);
    });
    w0_ = w_;
    parsed_ = true;
  }

  const std::vector<int64_t>& weights() const { return w_; }
  const std::vector<int64_t>& initial_weights() const { return w0_; }
  int iteration() const { return t_; }

  // R1 request: the data-column weights w_0..w_{d_data-1}.
  std::vector<int64_t> fwd_request() const {
    expect(0, "forward request");
    return std::vector<int64_t>(w_.begin(), w_.begin() + P_.d_data);
  }
  // R1 response: B FEIP decryptions + dlogs, then z and yhat in the clear.
  void fwd_finish(const uint8_t* key, Metrics* M) {
    expect(0, "forward decryption");
    const Scalar sk = feip::deserialize_key(key);
    const std::vector<int64_t> wd = fwd_request();
    int64_t bound = 0;
    for (int64_t v : wd) bound += v < 0 ? -v : v;
    bound <<= P_.f;  // |x| <= 2^f
    Mark m = mark_now();
    std::vector<Point> g((size_t)P_.batch);
    parallel_for(threads_, (size_t)P_.batch, [&](size_t i) {
      g[i] = feip::decrypt_to_group(fwd_[(size_t)t_ * P_.batch + i], sk, wd);
    });
    if (M) M->record("fwd_decrypt", m);
    std::vector<int64_t> dot((size_t)P_.batch);
    solve_all(g, bound, FWD, dot, "forward <w, x_i>");
    if (M) M->record("fwd_dlog", m);
    const int64_t wb = P_.bias ? w_[(size_t)P_.d - 1] * P_.maxq : 0;  // bias column x = MAXQ
    yhat_.assign((size_t)P_.batch, 0);
    z_.assign((size_t)P_.batch, 0);
    for (int i = 0; i < P_.batch; ++i) {
      z_[(size_t)i] = trunc_f(dot[(size_t)i] + wb, P_.f);
      yhat_[(size_t)i] = qsigmoid_ref(z_[(size_t)i], P_.f);
    }
    if (M) M->record("plain_lr", m);
    stage_ = 1;
  }
  const std::vector<int64_t>& yhat() const { return yhat_; }
  const std::vector<int64_t>& z() const { return z_; }

  // R2 response: B FEBO '-' decryptions (g^{MAXQ*y_i - yhat_i}) + dlogs; e = -(that).
  void eval_finish(const uint8_t* keys, Metrics* M) {
    expect(1, "FEBO decryption");
    Mark m = mark_now();
    std::vector<Point> g((size_t)P_.batch);
    parallel_for(threads_, (size_t)P_.batch, [&](size_t i) {
      const Point sk = febo::deserialize_key(keys + i * febo::kKeyBytes);
      g[i] = febo::decrypt_to_group(bo_[(size_t)t_ * P_.batch + i], sk, febo::Op::Sub, yhat_[i]);
    });
    if (M) M->record("febo_decrypt", m);
    std::vector<int64_t> diff((size_t)P_.batch);
    solve_all(g, P_.maxq + P_.maxq, FEBO, diff, "FEBO label - yhat");
    if (M) M->record("febo_dlog", m);
    e_.assign((size_t)P_.batch, 0);
    for (int i = 0; i < P_.batch; ++i) e_[(size_t)i] = -diff[(size_t)i];
    if (M) M->record("plain_lr", m);
    stage_ = 2;
  }
  const std::vector<int64_t>& err() const { return e_; }

  // R3 response: d_data FEIP decryptions (<e, column j>) + dlogs, then the update.
  void bwd_finish(const uint8_t* key, Metrics* M) {
    expect(2, "backward decryption");
    const Scalar sk = feip::deserialize_key(key);
    int64_t bound = 0, esum = 0;
    for (int64_t v : e_) bound += v < 0 ? -v : v, esum += v;
    bound <<= P_.f;
    Mark m = mark_now();
    std::vector<Point> g((size_t)P_.d_data);
    parallel_for(threads_, (size_t)P_.d_data, [&](size_t j) {
      g[j] = feip::decrypt_to_group(bwd_[(size_t)t_ * P_.d_data + j], sk, e_);
    });
    if (M) M->record("bwd_decrypt", m);
    std::vector<int64_t> gfull((size_t)P_.d_data);
    solve_all(g, bound, BWD, gfull, "backward <e, x_.j>");
    if (M) M->record("bwd_dlog", m);
    if (P_.bias) gfull.push_back(P_.maxq * esum);  // bias column x = MAXQ
    for (int j = 0; j < P_.d; ++j)
      w_[(size_t)j] = (K_ * w_[(size_t)j] - trunc_f(gfull[(size_t)j], P_.f)) / K_;
    if (M) M->record("plain_lr", m);
    stage_ = 0;
    ++t_;
  }

  // TEST ONLY: every dlog bound becomes min(bound, cap), cap >= 0 (to show that a too
  // small bound is detected). A negative cap disables it.
  void set_test_bound_cap(int64_t cap) { cap_ = cap; }
  const DlogStats& stats(Kind k) const { return stats_[k]; }

 private:
  void expect(int stage, const char* what) const {
    if (!parsed_) throw std::logic_error(std::string(what) + " before the upload was parsed");
    if (t_ >= P_.iters) throw std::logic_error(std::string(what) + " after the last iteration");
    if (stage_ != stage) throw std::logic_error(std::string(what) + " out of order");
  }

  void solve_all(const std::vector<Point>& g, int64_t bound, Kind kind, std::vector<int64_t>& out,
                 const char* what) {
    const int64_t b = cap_ >= 0 ? std::min(bound, cap_) : bound;
    std::vector<char> ok(g.size(), 0);
    parallel_for(threads_, g.size(), [&](size_t k) {
      const auto v = dl_.solve(g[k], b);
      if (v) out[k] = *v, ok[k] = 1;
    });
    for (size_t k = 0; k < g.size(); ++k) {
      if (!ok[k])
        throw DlogError("bounded discrete log FAILED for " + std::string(what) + " (iteration " +
                        std::to_string(t_) + ", index " + std::to_string(k) + "): no value in [-" +
                        std::to_string(b) + ", " + std::to_string(b) + "]" +
                        (b < bound ? " -- the bound was capped by --test-bound-cap (test only)" : ""));
      stats_[kind].add(out[k], b);
    }
  }

  Params P_;
  const Dlog& dl_;
  int threads_;
  int64_t K_ = 1;
  int64_t cap_ = -1;
  bool parsed_ = false;
  std::vector<int64_t> w_, w0_, yhat_, z_, e_;
  std::vector<feip::Ct> fwd_, bwd_;
  std::vector<febo::Ct> bo_;
  int t_ = 0, stage_ = 0;
  DlogStats stats_[3];
};

// BSGS table size for --bsgs-bits auto: one lookup covers the worst-case forward value
// of the INITIAL weights (|w_j| <= 2^f, |x| <= 2^f: 2^(2f) * d_data), clamped to
// [12, 20] bits. Larger values (weights grow during training, or qbits > 6 with large d)
// cost ~1 us per extra giant step instead of a bigger table (20 bits: ~1 s, 24 MiB).
inline int auto_bsgs_bits(const Params& P) {
  const int64_t bound = (int64_t(1) << (2 * P.f)) * (int64_t)P.d_data;
  return Dlog::bits_for_bound(bound, 12, 20);
}

}  // namespace lrcn

#endif  // LRCN_CRYPTONN_PROTOCOL_H
