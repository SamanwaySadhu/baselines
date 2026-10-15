// cryptonn_protocol.h -- the two-party CryptoNN logistic-regression protocol (client and
// server code), shared by lr_cryptonn (the program) and lr_cryptonn_selftest (which runs
// both parties in two threads over loopback TCP).
//
// CryptoNN (Xu, Joshi, Li, ICDCS 2019) trains a model held IN THE CLEAR by a server on
// data that the data owners only ever send functionally encrypted. A trusted authority
// holds the master secret keys and hands the server function keys for the vectors the
// training needs. Here the authority is merged into the client, the data owner:
//
//   CLIENT = data owner + FE authority: owns the training data (x, y), the initial
//            weights w0 and every master secret key (msk_ff, msk_bp, msk_bo).
//   SERVER = model trainer: receives ciphertexts only, holds the model w in the clear,
//            requests function keys and decrypts the inner products / differences.
//
// Notation: f = qbits - 1, MAXQ = 2^f - 1, K = batch / lr = 2^k_shift; batch t holds the
// rows i < batch of the Dataset (lr_params.h); the label value is MAXQ * y.
//
// PREPROCESSING (paper Algorithm 1, pre-process-encryption)
//   0. handshake: client -> server the public parameters (kHeaderBytes, NOT the seed);
//      server -> client one status byte (0 = the server's parameters are identical).
//   1. client: FEIP.Setup(eta = d) (msk_ff, for sample rows), FEIP.Setup(eta = batch)
//      (msk_bp, for the per-batch feature columns), FEBO.Setup (msk_bo, for labels), and
//      encryption with the PUBLIC keys, as a data owner would (--enc-with-msk: the
//      authority's equivalent shortcut feip::encrypt_with_msk):
//        ct_ff[t][i] = FEIP_ff.Enc(x_{t,i,:})     d components
//        ct_bp[t][j] = FEIP_bp.Enc(x_{t,:,j})     batch components
//        ct_bo[t][i] = FEBO.Enc(MAXQ * y_{t,i})
//      All ciphertexts and w0 (d int32) go to the server in ONE flight. Revealing w0 is
//      inherent to CryptoNN: the server holds the model in plaintext, because it asks for
//      function keys for w and FEIP decryption takes y = w as input. mpk is not sent (the
//      server never encrypts and decryption does not use it).
//   2. server: BSGS table (input independent, overlaps the client's encryption), receive
//      and deserialize (validates every point) the ciphertexts.
//
// ONLINE, iteration t (paper Algorithm 2, LR case): 3 round trips
//   a. secure feed-forward:  s -> c  w (d int32);   c -> s  sk_w = FEIP.KeyDerive(msk_ff, w)
//      server: zfull_i = dlog(FEIP.Dec(ct_ff[t][i], sk_w, w)), |zfull_i| <= sum_j |w_j| 2^f;
//              z_i = trunc_f(zfull_i), yhat_i = qsigmoid_ref(z_i)
//   b. secure evaluation:    s -> c  yhat (batch int32);
//                            c -> s  per i FEBO.KeyDerive(msk_bo, cmt_{t,i}, '-', yhat_i)
//      server: d_i = dlog(FEBO.Dec) = MAXQ y_i - yhat_i (|d_i| <= MAXQ); e_i = -d_i
//   c. secure back-propagation: s -> c  e (batch int32); c -> s  sk_e = FEIP.KeyDerive(msk_bp, e)
//      server: gfull_j = dlog(FEIP.Dec(ct_bp[t][j], sk_e, e)), |gfull_j| <= sum_i |e_i| 2^f;
//              g_j = trunc_f(gfull_j);  w_j = (K w_j - g_j) / K (C++ '/')
//   This is exactly reference_train (lr_reference.h), bit for bit. The client keeps the
//   FEBO commitments cmt_{t,i} it generated, so a key request carries only yhat (in the
//   paper the server sends cmt along with the request).
// OUTPUT: the server ends holding the trained model. With --send-model it also sends w to
//   the client (s -> c, 4d bytes), counted in a separate 'output' stage.
//
// What each party learns (semi-honest): the server learns w0 and every w, z, yhat, e, g,
// and hence every label (y = (yhat - e) / MAXQ) -- inherent to CryptoNN's LR (the paper's
// random label mapping is for one-hot multi-class labels and is not applied to binary
// labels). The client, being the authority, learns every w, yhat and e it is asked to
// key. A malicious server could request keys for unit vectors and read features
// directly; the authority's range checks below do not prevent that (as in the paper).
//
// Cost accounting (metrics.h): each party records contiguous phases named
// "<stage>/<sub-phase>" with stage in {pre, online, output}; sub-phases whose name starts
// with "wait" are blocking receives (waiting for the peer), the rest is local compute
// plus sends. Every flight is flushed explicitly where it ends so that its send cost is
// attributed to the phase that produced it.
#ifndef LRCN_CRYPTONN_PROTOCOL_H
#define LRCN_CRYPTONN_PROTOCOL_H

#include "bsgs.h"
#include "ec_group.h"
#include "fe_bo.h"
#include "fe_ip.h"
#include "lr_params.h"
#include "lr_reference.h"
#include "metrics.h"
#include "net.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace lrcn {

// ---------------------------------------------------------------- worker pool
// Persistent helper threads for parallel_for (the caller participates; dynamic
// scheduling by an atomic index). Keeps thread creation out of the timed phases and
// gives each thread a long-lived BN_CTX. threads = 1: everything runs inline.
class WorkerPool {
 public:
  explicit WorkerPool(int threads) {
    const int helpers = std::max(1, threads) - 1;
    for (int k = 0; k < helpers; ++k) th_.emplace_back([this] { loop(); });
  }
  ~WorkerPool() {
    {
      std::lock_guard<std::mutex> g(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : th_) t.join();
  }
  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;

  int threads() const { return int(th_.size()) + 1; }

  // fn(i) for every i in [0, n); rethrows the first exception (remaining indices skipped).
  template <class F>
  void parallel_for(size_t n, F&& fn) {
    if (n == 0) return;
    if (th_.empty() || n == 1) {
      for (size_t i = 0; i < n; ++i) fn(i);
      return;
    }
    const std::function<void(size_t)> job = [&fn](size_t i) { fn(i); };
    {
      std::lock_guard<std::mutex> g(mu_);
      job_ = &job;
      n_ = n;
      next_.store(0);
      err_ = nullptr;
      pending_ = th_.size();
      ++gen_;
    }
    cv_.notify_all();
    run_job();
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [this] { return pending_ == 0; });
    job_ = nullptr;
    if (err_) {
      std::exception_ptr e = err_;
      err_ = nullptr;
      std::rethrow_exception(e);
    }
  }

 private:
  void run_job() {
    for (;;) {
      const size_t i = next_.fetch_add(1);
      if (i >= n_) return;
      try {
        (*job_)(i);
      } catch (...) {
        std::lock_guard<std::mutex> g(mu_);
        if (!err_) err_ = std::current_exception();
        next_.store(n_);
      }
    }
  }
  void loop() {
    uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
        if (stop_) return;
        seen = gen_;
      }
      run_job();
      std::lock_guard<std::mutex> g(mu_);
      if (--pending_ == 0) done_cv_.notify_one();
    }
  }

  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(size_t)>* job_ = nullptr;
  size_t n_ = 0;
  std::atomic<size_t> next_{0};
  size_t pending_ = 0;
  uint64_t gen_ = 0;
  bool stop_ = false;
  std::exception_ptr err_;
  std::vector<std::thread> th_;
};

// ------------------------------------------------------------------ options
struct ProtoOptions {
  Params P;                   // finalized; both parties must agree (checked by the handshake)
  int threads = 1;            // per party: encryption / (de)serialization / decryption + dlog
  bool enc_with_msk = false;  // client: authority shortcut feip::encrypt_with_msk (same distribution)
  int dlog_bits = 0;          // server: BSGS table bits; 0 = auto_dlog_bits(P)
  bool send_model = false;    // both: output stage, the server sends the trained w to the client
  bool quiet = false;         // no per-iteration progress lines
};

// ------------------------------------------------------- public-params header
// magic "CRNN" | version u32 | d | iters | batch | qbits | k_shift (u32 each) |
// lr (IEEE double bits, u64) | flags u32 (bit 0: bias column, bit 1: send_model).
// The seed and --zero-init are the client's private choices and are not sent.
constexpr size_t kHeaderBytes = 40;
constexpr uint32_t kProtoVersion = 1;

struct Header {
  bool magic_ok = false;
  uint32_t version = 0, d = 0, iters = 0, batch = 0, qbits = 0, k_shift = 0, flags = 0;
  double lr = 0;
  std::string str() const {
    char b[256];
    std::snprintf(b, sizeof b, "d=%u iters=%u batch=%u qbits=%u lr=%g (k_shift=%u) bias=%s send_model=%s%s",
                  d, iters, batch, qbits, lr, k_shift, (flags & 1) ? "yes" : "no", (flags & 2) ? "yes" : "no",
                  magic_ok && version == kProtoVersion ? "" : " [bad magic/version]");
    return b;
  }
};

namespace detail {
inline void put_u32(uint8_t* p, uint32_t v) {
  for (int i = 0; i < 4; ++i) p[i] = uint8_t(v >> (8 * i));
}
inline uint32_t get_u32(const uint8_t* p) {
  uint32_t v = 0;
  for (int i = 0; i < 4; ++i) v |= uint32_t(p[i]) << (8 * i);
  return v;
}
inline void put_i32(uint8_t* p, int64_t v, const char* what) {
  if (v < INT32_MIN || v > INT32_MAX)
    throw std::runtime_error(std::string(what) + ": value " + std::to_string(v) + " does not fit in int32");
  put_u32(p, uint32_t(int32_t(v)));
}
inline int64_t get_i32(const uint8_t* p) { return int32_t(get_u32(p)); }
}  // namespace detail

inline std::array<uint8_t, kHeaderBytes> encode_header(const Params& P, bool send_model) {
  std::array<uint8_t, kHeaderBytes> h{};
  std::memcpy(h.data(), "CRNN", 4);
  detail::put_u32(&h[4], kProtoVersion);
  detail::put_u32(&h[8], uint32_t(P.d));
  detail::put_u32(&h[12], uint32_t(P.iters));
  detail::put_u32(&h[16], uint32_t(P.batch));
  detail::put_u32(&h[20], uint32_t(P.qbits));
  detail::put_u32(&h[24], uint32_t(P.k_shift));
  uint64_t lr_bits;
  std::memcpy(&lr_bits, &P.lr, 8);
  detail::put_u32(&h[28], uint32_t(lr_bits));
  detail::put_u32(&h[32], uint32_t(lr_bits >> 32));
  detail::put_u32(&h[36], (P.bias ? 1u : 0u) | (send_model ? 2u : 0u));
  return h;
}
inline Header decode_header(const uint8_t* h) {
  Header r;
  r.magic_ok = std::memcmp(h, "CRNN", 4) == 0;
  r.version = detail::get_u32(h + 4);
  r.d = detail::get_u32(h + 8);
  r.iters = detail::get_u32(h + 12);
  r.batch = detail::get_u32(h + 16);
  r.qbits = detail::get_u32(h + 20);
  r.k_shift = detail::get_u32(h + 24);
  const uint64_t lr_bits = uint64_t(detail::get_u32(h + 28)) | (uint64_t(detail::get_u32(h + 32)) << 32);
  std::memcpy(&r.lr, &lr_bits, 8);
  r.flags = detail::get_u32(h + 36);
  return r;
}

// ------------------------------------------------------ closed-form traffic
// Exact payload bytes / flights of a run, from the message formats (the selftest
// asserts measured == expected; the program warns on a mismatch).
struct CommFormula {
  uint64_t header = kHeaderBytes, ack = 1;
  uint64_t upload_ff = 0, upload_bp = 0, upload_bo = 0, upload_w0 = 0, upload = 0;
  uint64_t iter_c2s = 0, iter_s2c = 0;  // per online iteration
  uint64_t pre_c2s = 0, pre_s2c = 0, online_c2s = 0, online_s2c = 0, output_s2c = 0;
  uint64_t pre_flights = 3, online_flights = 0, output_flights = 0;
  uint64_t c2s() const { return pre_c2s + online_c2s; }
  uint64_t s2c() const { return pre_s2c + online_s2c + output_s2c; }
  uint64_t total() const { return c2s() + s2c(); }
  uint64_t flights() const { return pre_flights + online_flights + output_flights; }
};

inline CommFormula comm_formula(const Params& P, bool send_model) {
  CommFormula c;
  const uint64_t T = uint64_t(P.iters), B = uint64_t(P.batch), d = uint64_t(P.d);
  c.upload_ff = T * B * feip::ct_bytes(P.d);       // iters*batch*(d+1)*33
  c.upload_bp = T * d * feip::ct_bytes(P.batch);   // iters*d*(batch+1)*33
  c.upload_bo = T * B * febo::kCtBytes;            // iters*batch*66
  c.upload_w0 = 4 * d;
  c.upload = c.upload_ff + c.upload_bp + c.upload_bo + c.upload_w0;
  c.pre_c2s = c.header + c.upload;
  c.pre_s2c = c.ack;
  c.iter_s2c = 4 * d + 4 * B + 4 * B;                                   // w, yhat, e
  c.iter_c2s = kScalarBytes + B * febo::kKeyBytes + kScalarBytes;       // sk_w, sk_bo[i], sk_e
  c.online_c2s = T * c.iter_c2s;
  c.online_s2c = T * c.iter_s2c;
  c.online_flights = 6 * T;
  c.output_s2c = send_model ? 4 * d : 0;
  c.output_flights = send_model ? 1 : 0;
  return c;
}

// ------------------------------------------------------------ public bounds
// |w_j| before iteration t (0-based): |w0_j| <= 2^f and |g_j| <= batch*MAXQ, so one
// update moves a weight by at most floor(batch*MAXQ / K). Both parties check the
// weights against it (the client as authority before it derives a key for w).
inline int64_t weight_bound(const Params& P, int t) {
  const int64_t K = int64_t(1) << P.k_shift;
  return (int64_t(1) << P.f) + int64_t(t) * ((int64_t(P.batch) * P.maxq) / K);
}
// BSGS table bits covering max(d, batch) * 2^(2f) in one lookup, capped at 22 bits
// (4M entries, ~96 MiB); larger values are found by giant steps.
inline int auto_dlog_bits(const Params& P) {
  const int64_t bound = int64_t(std::max(P.d, P.batch)) << (2 * P.f);
  return Dlog::bits_for_bound(bound, 8, 22);
}

// ------------------------------------------------------------ party report
// One party's measurements, exchanged at the end (the report exchange itself is not
// counted anywhere).
struct PartyReport {
  std::vector<std::pair<std::string, PhaseStat>> phases;  // "<stage>/<sub-phase>", first-seen order
  std::vector<double> iter_wall_s;                        // per online iteration
  double peak_rss_mib = 0;
  NetCounters net;  // channel counters at the end of the protocol
  uint64_t threads = 0;       // --threads of that party
  uint64_t enc_with_msk = 0;  // client: 1 if --enc-with-msk

  static bool is_wait(const std::string& name) {
    const size_t s = name.find('/');
    return name.compare(s == std::string::npos ? 0 : s + 1, 4, "wait") == 0;
  }
  static bool in_stage(const std::string& name, const std::string& stage) {
    return stage.empty() || name.compare(0, stage.size() + 1, stage + "/") == 0;
  }
  // Sum of the stage's phases ("" = all); compute_only drops the wait phases.
  PhaseStat stage(const std::string& st, bool compute_only = false) const {
    PhaseStat s;
    for (const auto& [n, p] : phases)
      if (in_stage(n, st) && !(compute_only && is_wait(n))) s.add(p);
    return s;
  }

  std::vector<uint8_t> serialize() const {
    std::vector<uint8_t> b;
    auto u64 = [&](uint64_t v) {
      for (int i = 0; i < 8; ++i) b.push_back(uint8_t(v >> (8 * i)));
    };
    auto dbl = [&](double v) {
      uint64_t u;
      std::memcpy(&u, &v, 8);
      u64(u);
    };
    u64(phases.size());
    for (const auto& [n, p] : phases) {
      u64(n.size());
      b.insert(b.end(), n.begin(), n.end());
      dbl(p.wall_s); dbl(p.cpu_s);
      u64(p.sent); u64(p.recv); u64(p.flights); u64(p.flights_out); u64(p.flights_in); u64(p.calls);
    }
    u64(iter_wall_s.size());
    for (double v : iter_wall_s) dbl(v);
    dbl(peak_rss_mib);
    u64(net.sent); u64(net.recv); u64(net.flights_out); u64(net.flights_in);
    u64(threads); u64(enc_with_msk);
    return b;
  }
  static PartyReport deserialize(const std::vector<uint8_t>& b) {
    size_t k = 0;
    auto u64 = [&]() {
      if (k + 8 > b.size()) throw std::runtime_error("report: truncated");
      uint64_t v = 0;
      for (int i = 0; i < 8; ++i) v |= uint64_t(b[k + size_t(i)]) << (8 * i);
      k += 8;
      return v;
    };
    auto dbl = [&]() {
      const uint64_t u = u64();
      double v;
      std::memcpy(&v, &u, 8);
      return v;
    };
    PartyReport r;
    const uint64_t np = u64();
    if (np > 4096) throw std::runtime_error("report: too many phases");
    for (uint64_t q = 0; q < np; ++q) {
      const uint64_t len = u64();
      if (len > 256 || k + len > b.size()) throw std::runtime_error("report: bad phase name");
      std::string n(b.begin() + long(k), b.begin() + long(k + len));
      k += len;
      PhaseStat p;
      p.wall_s = dbl(); p.cpu_s = dbl();
      p.sent = u64(); p.recv = u64(); p.flights = u64(); p.flights_out = u64(); p.flights_in = u64(); p.calls = u64();
      r.phases.emplace_back(std::move(n), p);
    }
    const uint64_t ni = u64();
    if (ni > (uint64_t(1) << 24)) throw std::runtime_error("report: too many iterations");
    for (uint64_t q = 0; q < ni; ++q) r.iter_wall_s.push_back(dbl());
    r.peak_rss_mib = dbl();
    r.net.sent = u64(); r.net.recv = u64(); r.net.flights_out = u64(); r.net.flights_in = u64();
    r.threads = u64(); r.enc_with_msk = u64();
    if (k != b.size()) throw std::runtime_error("report: trailing bytes");
    return r;
  }
};

inline void send_report(Channel& ch, const PartyReport& r) {
  ch.send_blob(r.serialize());
  ch.flush();
}
inline PartyReport recv_report(Channel& ch) { return PartyReport::deserialize(ch.recv_blob(uint64_t(1) << 24)); }

// ------------------------------------------------------------- dlog stats
struct DlogStats {
  int table_bits = 0;
  size_t entries = 0, memory_bytes = 0;
  int64_t half_width = 0;
  int64_t max_bound_ff = 0, max_bound_eval = 0, max_bound_bp = 0;  // largest bound passed to solve()
  int64_t max_abs_ff = 0, max_abs_eval = 0, max_abs_bp = 0;        // largest |value| recovered
  int64_t max_steps_needed = 0;  // giant-step lookups the hardest solve needed (0 = table hit)
  int64_t worst_case_steps = 0;  // Dlog::max_giant_steps(largest bound)
  uint64_t solves = 0;
};

// Giant-step lookups solve() performs before finding v (k = 0 first, then +1, -1, +2, ...).
inline int64_t giant_steps_needed(const Dlog& L, int64_t v) {
  const int64_t m = L.half_width(), M = L.giant_stride();
  const int64_t k = v >= 0 ? (v + m) / M : -((-v + m) / M);
  return k == 0 ? 0 : (k > 0 ? 2 * k - 1 : -2 * k);
}

// --------------------------------------------------------------- results
struct ClientResult {
  PartyReport rep;
  std::vector<int64_t> final_w;  // only with send_model
};
struct ServerResult {
  PartyReport rep;
  std::vector<int64_t> w0, w;                  // received initial weights, trained weights
  std::vector<std::vector<int64_t>> w_after;   // weights after each iteration
  DlogStats dlog;
};

// ================================================================== CLIENT
inline ClientResult run_client(Channel& ch, const ProtoOptions& O, const Dataset& D,
                               const std::vector<int64_t>& w0) {
  const Params& P = O.P;
  if (D.iters != P.iters || D.batch != P.batch || D.d != P.d)
    throw std::invalid_argument("run_client: dataset shape does not match the parameters");
  if (w0.size() != size_t(P.d)) throw std::invalid_argument("run_client: |w0| != d");
  for (int64_t v : w0)
    if (v < P.minq || v > P.maxq) throw std::invalid_argument("run_client: w0 is not a qbits-bit quantized vector");
  for (int64_t v : D.x)
    if (v < P.minq || v > P.maxq) throw std::invalid_argument("run_client: feature outside [MINQ, MAXQ]");

  WorkerPool pool(O.threads);
  const size_t T = size_t(P.iters), B = size_t(P.batch), d = size_t(P.d);
  const CommFormula F = comm_formula(P, O.send_model);
  Metrics M;
  ClientResult R;
  Mark m = M.now(&ch);

  // ---- 0. handshake ----
  const auto hdr = encode_header(P, O.send_model);
  ch.send(hdr.data(), hdr.size());
  ch.flush();
  M.record("pre/handshake", m, &ch);
  uint8_t status = 0xff;
  ch.recv(&status, 1);
  M.record("pre/wait_ack", m, &ch);
  if (status != 0)
    throw std::runtime_error("the server rejected the public parameters (they differ from its own; see the "
                             "server log). client: " + decode_header(hdr.data()).str());

  // ---- 1. FE setup (the client is the authority) ----
  const auto fe_ff = feip::setup(P.d), fe_bp = feip::setup(P.batch);
  const auto fe_bo = febo::setup();
  const feip::Mpk& mpk_ff = fe_ff.first;
  const feip::Msk& msk_ff = fe_ff.second;
  const feip::Mpk& mpk_bp = fe_bp.first;
  const feip::Msk& msk_bp = fe_bp.second;
  const febo::Mpk& mpk_bo = fe_bo.first;
  const febo::Msk& msk_bo = fe_bo.second;
  M.record("pre/fe_setup", m, &ch);

  // ---- encryption (Algorithm 1), parallel over ciphertexts ----
  const size_t n_ff = T * B, n_bp = T * d, n_bo = T * B;
  std::vector<feip::Ct> ct_ff(n_ff), ct_bp(n_bp);
  std::vector<febo::Ct> ct_bo(n_bo);
  pool.parallel_for(n_ff + n_bp + n_bo, [&](size_t k) {
    if (k < n_ff) {  // row (t, i): x_{t,i,:}
      const int t = int(k / B), i = int(k % B);
      std::vector<int64_t> x(d);
      for (size_t j = 0; j < d; ++j) x[j] = D.X(t, i, int(j));
      ct_ff[k] = O.enc_with_msk ? feip::encrypt_with_msk(msk_ff, x) : feip::encrypt(mpk_ff, x);
    } else if (k < n_ff + n_bp) {  // column (t, j): x_{t,:,j}
      const size_t q = k - n_ff;
      const int t = int(q / d), j = int(q % d);
      std::vector<int64_t> x(B);
      for (size_t i = 0; i < B; ++i) x[i] = D.X(t, int(i), j);
      ct_bp[q] = O.enc_with_msk ? feip::encrypt_with_msk(msk_bp, x) : feip::encrypt(mpk_bp, x);
    } else {  // label (t, i)
      const size_t q = k - n_ff - n_bp;
      ct_bo[q] = febo::encrypt(mpk_bo, P.maxq * D.Y(int(q / B), int(q % B)));
    }
  });
  M.record("pre/encrypt", m, &ch);

  // ---- serialization: ct_ff[t][i] | ct_bp[t][j] | ct_bo[t][i] | w0 (int32 LE) ----
  std::vector<uint8_t> up(F.upload);
  const size_t sz_ff = feip::ct_bytes(P.d), sz_bp = feip::ct_bytes(P.batch);
  const size_t off_bp = n_ff * sz_ff, off_bo = off_bp + n_bp * sz_bp, off_w = off_bo + n_bo * febo::kCtBytes;
  pool.parallel_for(n_ff + n_bp + n_bo, [&](size_t k) {
    if (k < n_ff) feip::serialize(ct_ff[k], up.data() + k * sz_ff);
    else if (k < n_ff + n_bp) feip::serialize(ct_bp[k - n_ff], up.data() + off_bp + (k - n_ff) * sz_bp);
    else febo::serialize(ct_bo[k - n_ff - n_bp], up.data() + off_bo + (k - n_ff - n_bp) * febo::kCtBytes);
  });
  for (size_t j = 0; j < d; ++j) detail::put_i32(up.data() + off_w + 4 * j, w0[j], "w0");
  if (off_w + 4 * d != up.size()) throw std::logic_error("run_client: upload layout");
  // Only the FEBO commitments are needed later (key derivation for '-').
  std::vector<Point> cmt(n_bo);
  for (size_t k = 0; k < n_bo; ++k) cmt[k] = std::move(ct_bo[k].cmt);
  std::vector<feip::Ct>().swap(ct_ff);
  std::vector<feip::Ct>().swap(ct_bp);
  std::vector<febo::Ct>().swap(ct_bo);
  M.record("pre/serialize", m, &ch);

  ch.send(up.data(), up.size());  // the whole upload is one flight
  ch.flush();
  std::vector<uint8_t>().swap(up);
  M.record("pre/upload", m, &ch);

  // ---- ONLINE: answer the server's key requests (the client is the authority) ----
  std::vector<uint8_t> keys(B * febo::kKeyBytes);
  uint8_t sk[kScalarBytes];
  for (int t = 0; t < P.iters; ++t) {
    const Mark it0 = m;
    // a. secure feed-forward: sk_w for the current model
    const int64_t wb = weight_bound(P, t);
    const std::vector<int64_t> w = ch.recv_i32s(d, -wb, wb);
    M.record("online/wait", m, &ch);
    feip::serialize_key(feip::key_derive(msk_ff, w), sk);
    M.record("online/keyderive_ff", m, &ch);
    ch.send(sk, kScalarBytes);
    ch.flush();
    M.record("online/send", m, &ch);
    // b. secure evaluation: FEBO '-' keys for yhat_i
    const std::vector<int64_t> yhat = ch.recv_i32s(B, 0, P.maxq);
    M.record("online/wait", m, &ch);
    pool.parallel_for(B, [&](size_t i) {
      febo::serialize_key(febo::key_derive(msk_bo, cmt[size_t(t) * B + i], febo::Op::Sub, yhat[i]),
                          keys.data() + i * febo::kKeyBytes);
    });
    M.record("online/keyderive_eval", m, &ch);
    ch.send(keys.data(), keys.size());
    ch.flush();
    M.record("online/send", m, &ch);
    // c. secure back-propagation: sk_e for the batch errors
    const std::vector<int64_t> e = ch.recv_i32s(B, -P.maxq, P.maxq);
    M.record("online/wait", m, &ch);
    feip::serialize_key(feip::key_derive(msk_bp, e), sk);
    M.record("online/keyderive_bp", m, &ch);
    ch.send(sk, kScalarBytes);
    ch.flush();
    M.record("online/send", m, &ch);
    R.rep.iter_wall_s.push_back(phase_diff(it0, m).wall_s);
    if (!O.quiet) std::printf("[client] iteration %2d/%d keyed (%.2f ms)\n", t + 1, P.iters, R.rep.iter_wall_s.back() * 1e3);
  }

  if (O.send_model) {
    const int64_t wb = weight_bound(P, P.iters);
    R.final_w = ch.recv_i32s(d, -wb, wb);
    M.record("output/wait_model", m, &ch);
  }

  R.rep.phases = M.phases();
  R.rep.net = ch.counters();
  R.rep.peak_rss_mib = double(peak_rss_kib()) / 1024.0;
  R.rep.threads = uint64_t(pool.threads());
  R.rep.enc_with_msk = O.enc_with_msk ? 1 : 0;
  return R;
}

// ================================================================== SERVER
inline ServerResult run_server(Channel& ch, const ProtoOptions& O) {
  const Params& P = O.P;
  WorkerPool pool(O.threads);
  const size_t T = size_t(P.iters), B = size_t(P.batch), d = size_t(P.d);
  const CommFormula F = comm_formula(P, O.send_model);
  Metrics M;
  ServerResult R;
  Mark m = M.now(&ch);

  // ---- 0. handshake ----
  uint8_t hb[kHeaderBytes];
  ch.recv(hb, kHeaderBytes);
  M.record("pre/wait_header", m, &ch);
  const auto mine = encode_header(P, O.send_model);
  const bool same = std::memcmp(hb, mine.data(), kHeaderBytes) == 0;
  const uint8_t status = same ? 0 : 1;
  ch.send(&status, 1);
  ch.flush();
  M.record("pre/handshake", m, &ch);
  if (!same)
    throw std::runtime_error("public parameter mismatch: client sent [" + decode_header(hb).str() +
                             "], server has [" + decode_header(mine.data()).str() +
                             "]; both parties must pass the same --d --iters --batch --qbits --lr "
                             "--no-bias --send-model");

  // ---- 2. BSGS table (input independent; overlaps the client's encryption) ----
  const Dlog L(O.dlog_bits > 0 ? O.dlog_bits : auto_dlog_bits(P), O.threads);
  M.record("pre/dlog_table", m, &ch);

  // ---- receive + deserialize (validates every point) the upload ----
  const std::vector<uint8_t> up = ch.recv_bytes(F.upload);
  M.record("pre/wait_upload", m, &ch);
  const size_t n_ff = T * B, n_bp = T * d, n_bo = T * B;
  const size_t sz_ff = feip::ct_bytes(P.d), sz_bp = feip::ct_bytes(P.batch);
  const size_t off_bp = n_ff * sz_ff, off_bo = off_bp + n_bp * sz_bp, off_w = off_bo + n_bo * febo::kCtBytes;
  std::vector<feip::Ct> ct_ff(n_ff), ct_bp(n_bp);
  std::vector<febo::Ct> ct_bo(n_bo);
  pool.parallel_for(n_ff + n_bp + n_bo, [&](size_t k) {
    if (k < n_ff) ct_ff[k] = feip::deserialize(up.data() + k * sz_ff, P.d);
    else if (k < n_ff + n_bp) ct_bp[k - n_ff] = feip::deserialize(up.data() + off_bp + (k - n_ff) * sz_bp, P.batch);
    else ct_bo[k - n_ff - n_bp] = febo::deserialize(up.data() + off_bo + (k - n_ff - n_bp) * febo::kCtBytes);
  });
  R.w0.resize(d);
  for (size_t j = 0; j < d; ++j) {
    R.w0[j] = detail::get_i32(up.data() + off_w + 4 * j);
    if (R.w0[j] < P.minq || R.w0[j] > P.maxq)
      throw std::runtime_error("received w0[" + std::to_string(j) + "] = " + std::to_string(R.w0[j]) +
                               " is not a qbits-bit quantized value");
  }
  M.record("pre/deserialize", m, &ch);

  // ---- ONLINE ----
  DlogStats& S = R.dlog;
  S.table_bits = L.table_bits();
  S.entries = L.table_entries();
  S.memory_bytes = L.memory_bytes();
  S.half_width = L.half_width();
  auto solve = [&L](const Point& Pt, int64_t bound, const char* what, int t, size_t idx) {
    const std::optional<int64_t> v = L.solve(Pt, bound);
    if (!v)
      throw std::runtime_error(std::string("dlog failed (") + what + ", iteration " + std::to_string(t + 1) +
                               ", index " + std::to_string(idx) + ", bound " + std::to_string(bound) + ")");
    return *v;
  };
  auto note = [&](const std::vector<int64_t>& vals, int64_t bound, int64_t& max_bound, int64_t& max_abs) {
    max_bound = std::max(max_bound, bound);
    for (int64_t v : vals) {
      max_abs = std::max(max_abs, v < 0 ? -v : v);
      S.max_steps_needed = std::max(S.max_steps_needed, giant_steps_needed(L, v));
      ++S.solves;
    }
  };

  const int64_t K = int64_t(1) << P.k_shift;
  std::vector<int64_t> w = R.w0, zfull(B), yhat(B), e(B), gfull(d);
  std::vector<uint8_t> keys(B * febo::kKeyBytes);
  uint8_t sk[kScalarBytes];
  for (int t = 0; t < P.iters; ++t) {
    const Mark it0 = m;
    // a. secure feed-forward
    ch.send_i32s(w);
    ch.flush();
    M.record("online/send", m, &ch);
    ch.recv(sk, kScalarBytes);
    M.record("online/wait", m, &ch);
    {
      const Scalar sk_w = feip::deserialize_key(sk);
      int64_t bound = 0;
      for (int64_t v : w) bound += (v < 0 ? -v : v) << P.f;  // |x| <= 2^f
      pool.parallel_for(B, [&](size_t i) {
        zfull[i] = solve(feip::decrypt_to_group(ct_ff[size_t(t) * B + i], sk_w, w), bound, "feed-forward", t, i);
      });
      note(zfull, bound, S.max_bound_ff, S.max_abs_ff);
    }
    M.record("online/decrypt_ff", m, &ch);
    for (size_t i = 0; i < B; ++i) yhat[i] = qsigmoid_ref(trunc_f(zfull[i], P.f), P.f);
    M.record("online/sigmoid", m, &ch);
    // b. secure evaluation (FEBO '-'): label_i - yhat_i
    ch.send_i32s(yhat);
    ch.flush();
    M.record("online/send", m, &ch);
    ch.recv(keys.data(), keys.size());
    M.record("online/wait", m, &ch);
    {
      std::vector<int64_t> diff(B);
      pool.parallel_for(B, [&](size_t i) {
        const Point key = febo::deserialize_key(keys.data() + i * febo::kKeyBytes);
        diff[i] = solve(febo::decrypt_to_group(ct_bo[size_t(t) * B + i], key, febo::Op::Sub, yhat[i]), P.maxq,
                        "evaluation", t, i);
      });
      note(diff, P.maxq, S.max_bound_eval, S.max_abs_eval);
      for (size_t i = 0; i < B; ++i) {
        const int64_t label = diff[i] + yhat[i];
        if (label != 0 && label != P.maxq)
          throw std::runtime_error("evaluation: decrypted label " + std::to_string(label) + " is not 0 or MAXQ");
        e[i] = -diff[i];  // yhat_i - label_i
      }
    }
    M.record("online/decrypt_eval", m, &ch);
    // c. secure back-propagation
    ch.send_i32s(e);
    ch.flush();
    M.record("online/send", m, &ch);
    ch.recv(sk, kScalarBytes);
    M.record("online/wait", m, &ch);
    {
      const Scalar sk_e = feip::deserialize_key(sk);
      int64_t bound = 0;
      for (int64_t v : e) bound += (v < 0 ? -v : v) << P.f;
      pool.parallel_for(d, [&](size_t j) {
        gfull[j] = solve(feip::decrypt_to_group(ct_bp[size_t(t) * d + j], sk_e, e), bound, "back-propagation", t, j);
      });
      note(gfull, bound, S.max_bound_bp, S.max_abs_bp);
    }
    M.record("online/decrypt_bp", m, &ch);
    const int64_t wb = weight_bound(P, t + 1);
    for (size_t j = 0; j < d; ++j) {
      w[j] = (K * w[j] - trunc_f(gfull[j], P.f)) / K;  // C++ '/' truncates toward zero
      if (w[j] < -wb || w[j] > wb) throw std::logic_error("updated weight outside its public bound");
    }
    R.w_after.push_back(w);
    M.record("online/update", m, &ch);
    R.rep.iter_wall_s.push_back(phase_diff(it0, m).wall_s);
    if (!O.quiet) {
      std::printf("[server] iteration %2d/%d done (%.2f ms) w =", t + 1, P.iters, R.rep.iter_wall_s.back() * 1e3);
      for (size_t j = 0; j < d && j < 8; ++j) std::printf(" %lld", (long long)w[j]);
      std::printf("%s\n", d > 8 ? " ..." : "");
    }
  }
  S.worst_case_steps = L.max_giant_steps(std::max({S.max_bound_ff, S.max_bound_eval, S.max_bound_bp}));

  if (O.send_model) {
    ch.send_i32s(w);
    ch.flush();
    M.record("output/send_model", m, &ch);
  }
  R.w = w;
  R.rep.phases = M.phases();
  R.rep.net = ch.counters();
  R.rep.peak_rss_mib = double(peak_rss_kib()) / 1024.0;
  R.rep.threads = uint64_t(pool.threads());
  return R;
}

}  // namespace lrcn

#endif  // LRCN_CRYPTONN_PROTOCOL_H
