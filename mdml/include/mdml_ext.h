// mdml_ext.h -- online-phase extensions of MD-ML (NemoYuan2008/MD-ML @ 65ad2d2) needed
// for maliciously secure logistic-regression training between two parties.
//
// Everything here follows the MD-ML paper (Yuan et al., USENIX Security 2024) and
// builds on MD-ML's own gate framework (md_ml::Gate, PartyWithFakeOffline, the
// fake-offline file format, Spdz2kShare64 = SPDZ2k with k = s = 64):
//
//   wire x:  public Delta_x = x + lambda_x (mod 2^k), SPDZ2k-shared [lambda_x] with MAC
//
//   InputGateU64    Delta_x = x + lambda_x sent by the owner (k = 64 bits per element;
//                   MD-ML's InputGate sends the 128-bit container)
//   MulGate         Pi_OnlineArith multiplication / Pi_MultTrunc (paper Sec. 4.1):
//                   one opening of the (k+s)-bit [Delta_z'], then local truncation by
//                   `shift`. Matrix (Eigen) or element-wise. Same algorithm as MD-ML's
//                   MultiplyGate / ElemMultiplyGate, plus a run-time shift and MAC logging.
//   LinearGate      any public linear map (+ public constant): local, no communication
//   GeqZeroGate     z = (x >= 0) = 1 - LTZ(x), paper Pi_LTZ with the "LTZ right after
//                   MultTrunc" optimization: BitLT(Delta_x, bits of lambda_x) on
//                   SPDZ2k *binary* shares ([.]_2: k = 1, s = 64, i.e. Z_{2^65}) with
//                   authenticated AND triples; carry circuit of MD-ML's GtzGate
//                   (CarryOutCin, log-depth); then B2A with a daBit and the opening of
//                   Delta_z. MD-ML's GtzGate fakes the binary triples as (0,0,0), sends
//                   all-zero MACs and opens the comparison bit in the clear; this gate
//                   uses real authenticated triples and MACs at the same message sizes.
//   OutputToGate    output to one party only: open [lambda_x] - [r] where r is known
//                   to the receiver (MD-ML's OutputGate opens to both parties)
//   mac_check()     Pi_MACCheck (paper App. B.1, from SPDZ2k) over every opened value:
//                   coin tossing by commit-and-open (SHA-256), random r_i in Z_{2^s},
//                   commitments to z_j = m_j - alpha_j * y. Run before the output is
//                   accepted. MD-ML's code computes the MAC shares but never checks them.
//
// All preprocessing comes from a trusted dealer (lr_mdml_dealer, MD-ML's "fake
// offline" approach); the dealer side of every gate is in mdml_dealer.h.
#ifndef LRMDML_MDML_EXT_H
#define LRMDML_MDML_EXT_H

#include "wan_wait.h"
#include "protocols/Gate.h"
#include "protocols/PartyWithFakeOffline.h"
#include "share/Spdz2kShare.h"
#include "utils/linear_algebra.h"

#include <omp.h>

#include <openssl/evp.h>
#include <openssl/rand.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace lrmd {

using u64 = uint64_t;
using u128 = __uint128_t;
using Shr = md_ml::Spdz2kShare64;
using MdParty = md_ml::PartyWithFakeOffline<Shr>;
using GateT = md_ml::Gate<Shr>;
using GatePtr = std::shared_ptr<GateT>;

constexpr int kRingBits = 64;                       // k
constexpr int kBinBits = 65;                        // 1 + s for the binary domain
constexpr u128 kMask64 = ((u128)1 << 64) - 1;
constexpr u128 kMask65 = ((u128)1 << 65) - 1;
inline u128 m65(u128 x) { return x & kMask65; }

// LinearGate map: out = f(x, y) (y empty for one-input gates). f must be linear over
// the integers (it is applied to Delta, to lambda shares and to MAC shares alike).
using LinFn = std::function<std::vector<u128>(const std::vector<u128>& x, const std::vector<u128>& y)>;

// ------------------------------------------------------------- metrics
enum Cat { C_INPUT, C_FWD, C_CMP, C_SIGMUL, C_BWD, C_LOCAL, C_OUTPUT, C_MACCHECK, NCAT };
inline const char* const kCatLabel[NCAT] = {
    "input sharing (client -> server)",
    "forward X*w (mult-then-trunc)",
    "comparison LTZ (BitLT + B2A)",
    "sigmoid (element-wise mult)",
    "gradient X^T*e (mult-then-trunc)",
    "local linear ops (no comm.)",
    "output to server (open lam-r)",
    "batched MAC check",
};
inline const char* const kCatKey[NCAT] = {"input", "forward", "comparison", "sigmoid_mul",
                                          "gradient", "local", "output", "mac_check"};

struct PhaseStat {
  double wall_s = 0, cpu_s = 0;
  double wait_s = 0;            // wall time the protocol thread is blocked in socket receives (wan/libwanshim.so; else 0)
  uint64_t sent = 0, recv = 0;  // recv is filled from the peer's counters at the end
  uint64_t rounds = 0;          // synchronous exchanges (both parties send, then receive)
  PhaseStat& operator+=(const PhaseStat& o) {
    wall_s += o.wall_s; cpu_s += o.cpu_s; wait_s += o.wait_s; sent += o.sent; recv += o.recv; rounds += o.rounds;
    return *this;
  }
};

inline double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline double cpu_now_s() {
  timespec ts{};
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

// Multi-core: loops over at least kParMin independent elements run on all OpenMP threads
// (set by --threads; default = every core). Smaller loops stay serial, where fork/join
// would cost more than the work.
constexpr size_t kParMin = 256;
// n <= 0 picks min(cores, kDefaultThreads): measured scaling stops at ~4 threads per party
// (memory/network bound), and idle OpenMP threads only burn CPU beyond that.
constexpr int kDefaultThreads = 4;
inline int set_threads(int n) {
  ::setenv("OMP_WAIT_POLICY", "passive", 0);  // do not spin between the many short regions
  const int hw = std::max(1, (int)std::thread::hardware_concurrency());
  if (n <= 0) n = std::min(hw, kDefaultThreads);
  omp_set_num_threads(n);
  return n;
}

// Allocator whose resize() leaves new elements uninitialised (the MAC logs are written
// in parallel right after growing; zero-filling them serially would dominate).
template <class T>
struct NoInitAlloc : std::allocator<T> {
  template <class U> struct rebind { using other = NoInitAlloc<U>; };
  using std::allocator<T>::allocator;
  template <class U> void construct(U* p) noexcept { ::new ((void*)p) U; }
  template <class U, class... A> void construct(U* p, A&&... a) { ::new ((void*)p) U(std::forward<A>(a)...); }
};
using OpenLog = std::vector<u128, NoInitAlloc<u128>>;

enum class Tamper { NONE, ARITH, BIN };  // TEST ONLY: corrupt one share sent by the client

// ------------------------------------------------------------- online context
// Per-party protocol state shared by all gates: MAC keys, the log of opened values
// (for the batched MAC check) and the metrics.
struct OnlineCtx {
  MdParty& party;
  int id;            // 0 = client, 1 = server
  u128 alpha;        // this party's share of the arithmetic MAC key (Z_{2^128} MACs)
  u128 beta;         // this party's share of the binary MAC key (Z_{2^65} MACs)
  Tamper tamper = Tamper::NONE;

  // opened values (full ring element) and this party's MAC share on them
  std::vector<u128> arith_open, arith_mac;  // mod 2^(k+s) = 2^128
  OpenLog bin_open, bin_mac;                // mod 2^(1+s) = 2^65

  PhaseStat stat[NCAT];
  uint64_t and_gates = 0, ltz_count = 0;

  // Pre-size the logs so the online phase never reallocates them (outside the timed phase).
  void reserve_openings(size_t arith, size_t bin) {
    arith_open.reserve(arith); arith_mac.reserve(arith);
    bin_open.reserve(bin); bin_mac.reserve(bin);
  }

  OnlineCtx(MdParty& p, u128 beta_shr) : party(p), id((int)p.my_id()), alpha(p.global_key_shr()), beta(beta_shr) {}

  // --- networking (both parties call these in the same order) ---
  template <typename T>
  std::vector<T> exchange(const std::vector<T>& mine, Cat cat) {
    std::vector<T> theirs;
    std::thread sender([&] { party.SendVecToOther(mine); });
    theirs = party.template ReceiveVecFromOther<T>(mine.size());
    sender.join();
    ++stat[cat].rounds;
    return theirs;
  }
  // Exchange (1+s) = 65-bit ring elements as one message: n low 64-bit words, then
  // the n top bits packed into bytes.
  std::vector<u128> exchange65(const std::vector<u128>& mine, Cat cat) {
    const size_t n = mine.size(), lo_bytes = 8 * n, hi_bytes = (n + 7) / 8;
    std::vector<uint8_t> msg(lo_bytes + hi_bytes);
#pragma omp parallel for schedule(static) if (n >= 16 * kParMin)
    for (size_t i = 0; i < n; ++i) {
      const u64 lo = (u64)mine[i];
      std::memcpy(&msg[8 * i], &lo, 8);
    }
#pragma omp parallel for schedule(static) if (n >= 16 * kParMin)
    for (size_t j = 0; j < hi_bytes; ++j) {
      uint8_t v = 0;
      for (size_t k = 0; k < 8 && 8 * j + k < n; ++k) v |= (uint8_t)(((mine[8 * j + k] >> 64) & 1) << k);
      msg[lo_bytes + j] = v;
    }
    const auto th = exchange(msg, cat);
    std::vector<u128> out(n);
#pragma omp parallel for schedule(static) if (n >= 16 * kParMin)
    for (size_t i = 0; i < n; ++i) {
      u64 lo;
      std::memcpy(&lo, &th[8 * i], 8);
      out[i] = (u128)lo | ((u128)((th[lo_bytes + (i >> 3)] >> (i & 7)) & 1) << 64);
    }
    return out;
  }
  bool take_tamper(Tamper kind) {
    if (id != 0 || tamper != kind) return false;
    tamper = Tamper::NONE;  // one corrupted element is enough
    return true;
  }
};

// Accumulates wall/CPU time and bytes sent into one category.
class Scope {
 public:
  Scope(OnlineCtx& c, Cat k) : c_(c), k_(k), t0_(now_s()), cpu0_(cpu_now_s()), sent0_(c.party.bytes_sent()), wait0_(wan::wait_s()) {}
  ~Scope() {
    c_.stat[k_].wall_s += now_s() - t0_;
    c_.stat[k_].cpu_s += cpu_now_s() - cpu0_;
    c_.stat[k_].wait_s += wan::wait_s() - wait0_;
    c_.stat[k_].sent += c_.party.bytes_sent() - sent0_;
  }
 private:
  OnlineCtx& c_;
  Cat k_;
  double t0_, cpu0_;
  uint64_t sent0_;
  double wait0_;
};

inline void reduce64(std::vector<u128>& v) {
  for (auto& x : v) x &= kMask64;
}

// ------------------------------------------------------------- input
class InputGateU64 : public GateT {
 public:
  InputGateU64(OnlineCtx& c, size_t rows, size_t cols, size_t owner)
      : GateT(c.party, rows, cols), ctx_(c), owner_(owner) {}

  void setInput(const std::vector<u64>& v) {
    if (ctx_.id != (int)owner_) throw std::logic_error("only the owner sets an input");
    if (v.size() != dim_row() * dim_col()) throw std::invalid_argument("input size mismatch");
    value_ = v;
  }

 private:
  void doReadOfflineFromFile() override {
    const size_t n = dim_row() * dim_col();
    if (ctx_.id == (int)owner_) lambda_clear_ = party().ReadShares(n);
    lambda_shr() = party().ReadShares(n);
    lambda_shr_mac() = party().ReadShares(n);
  }
  void doRunOnline() override {
    Scope sc(ctx_, C_INPUT);
    const size_t n = dim_row() * dim_col();
    std::vector<u64> delta(n);
    if (ctx_.id == (int)owner_) {
      if (value_.size() != n) throw std::logic_error("input not set");
      for (size_t i = 0; i < n; ++i) delta[i] = value_[i] + (u64)lambda_clear_[i];
      party().SendVecToOther(delta);
    } else {
      delta = party().template ReceiveVecFromOther<u64>(n);
    }
    Delta_clear().assign(delta.begin(), delta.end());
    lambda_clear_.clear();
    lambda_clear_.shrink_to_fit();
  }

  OnlineCtx& ctx_;
  size_t owner_;
  std::vector<u64> value_;
  std::vector<u128> lambda_clear_;
};

// ------------------------------------------------------------- multiplication
// Delta_z' = (Dx+dx)(Dy+dy) - (Dy+dy)[a] - (Dx+dx)[b] + [c] + [lambda_z'], opened;
// then Delta_z = Delta_z' >> shift and [lambda_z] = [lambda_z' >> shift] (dealer).
class MulGate : public GateT {
 public:
  enum Mode { MATMUL, ELEMENTWISE };

  MulGate(OnlineCtx& c, const GatePtr& x, const GatePtr& y, Mode mode, int shift, Cat cat)
      : GateT(x, y), ctx_(c), mode_(mode), shift_(shift), cat_(cat) {
    if (mode == MATMUL) {
      if (x->dim_col() != y->dim_row()) throw std::invalid_argument("MulGate: incompatible matrix dimensions");
      mid_ = x->dim_col();
      set_dim_row(x->dim_row());
      set_dim_col(y->dim_col());
    } else {
      if (x->dim_row() != y->dim_row() || x->dim_col() != y->dim_col())
        throw std::invalid_argument("MulGate: element-wise inputs must have equal shapes");
      if (shift != 0) throw std::invalid_argument("MulGate: element-wise truncation not supported");
      set_dim_row(x->dim_row());
      set_dim_col(x->dim_col());
    }
  }

 private:
  size_t lhs_size() const { return mode_ == MATMUL ? dim_row() * mid_ : dim_row() * dim_col(); }
  size_t rhs_size() const { return mode_ == MATMUL ? mid_ * dim_col() : dim_row() * dim_col(); }

  // Same order as md_ml::FakeMultiplyGate / FakeElemMultiplyGate (+ the truncation pair).
  void doReadOfflineFromFile() override {
    const size_t nl = lhs_size(), nr = rhs_size(), no = dim_row() * dim_col();
    auto& P = party();
    a_ = P.ReadShares(nl); a_mac_ = P.ReadShares(nl);
    b_ = P.ReadShares(nr); b_mac_ = P.ReadShares(nr);
    c_ = P.ReadShares(no); c_mac_ = P.ReadShares(no);
    lam_pre_ = P.ReadShares(no); lam_pre_mac_ = P.ReadShares(no);
    dx_ = P.ReadShares(nl); dy_ = P.ReadShares(nr);
    if (shift_ > 0) {
      lambda_shr() = P.ReadShares(no);
      lambda_shr_mac() = P.ReadShares(no);
    } else {
      lambda_shr() = lam_pre_;
      lambda_shr_mac() = lam_pre_mac_;
    }
  }

  std::vector<u128> mul(const std::vector<u128>& l, const std::vector<u128>& r) const {
    if (mode_ == MATMUL) {  // row-parallel i-k-j product mod 2^128
      const size_t R = dim_row(), M = mid_, C = dim_col();
      std::vector<u128> o(R * C, 0);
#pragma omp parallel for schedule(static) if (R * M * C >= (size_t)1 << 20 && R > 1)
      for (size_t i = 0; i < R; ++i)
        for (size_t k = 0; k < M; ++k) {
          const u128 a = l[i * M + k];
          for (size_t j = 0; j < C; ++j) o[i * C + j] += a * r[k * C + j];
        }
      return o;
    }
    std::vector<u128> o(l.size());
#pragma omp parallel for schedule(static) if (l.size() >= kParMin * 16)
    for (size_t i = 0; i < l.size(); ++i) o[i] = l[i] * r[i];
    return o;
  }

  void doRunOnline() override {
    Scope sc(ctx_, cat_);
    using namespace md_ml;
    auto tx = matrixAdd(input_x()->Delta_clear(), dx_);  // Delta_x + delta_x
    auto ty = matrixAdd(input_y()->Delta_clear(), dy_);  // Delta_y + delta_y
    auto txy = mul(tx, ty);

    auto D = matrixAdd(c_, lam_pre_);
    matrixSubtractAssign(D, mul(a_, ty));
    matrixSubtractAssign(D, mul(tx, b_));
    if (ctx_.id == 0) matrixAddAssign(D, txy);

    auto M = std::move(txy);
    matrixScalarAssign(M, ctx_.alpha);
    matrixAddAssign(M, c_mac_);
    matrixAddAssign(M, lam_pre_mac_);
    matrixSubtractAssign(M, mul(a_mac_, ty));
    matrixSubtractAssign(M, mul(tx, b_mac_));

    std::vector<u128> sendv = D;
    if (ctx_.take_tamper(Tamper::ARITH)) sendv[0] += 1;
    auto theirs = ctx_.exchange(sendv, cat_);
    matrixAddAssign(D, theirs);  // full (k+s)-bit Delta_z'
    ctx_.arith_open.insert(ctx_.arith_open.end(), D.begin(), D.end());
    ctx_.arith_mac.insert(ctx_.arith_mac.end(), M.begin(), M.end());

    reduce64(D);
    if (shift_ > 0)
      for (auto& v : D) v >>= shift_;  // Delta_z = Delta_z' / 2^shift (lambda_z = lambda_z' / 2^shift)
    Delta_clear() = std::move(D);

    for (auto* v : {&a_, &a_mac_, &b_, &b_mac_, &c_, &c_mac_, &lam_pre_, &lam_pre_mac_, &dx_, &dy_}) {
      v->clear();
      v->shrink_to_fit();
    }
  }

  OnlineCtx& ctx_;
  Mode mode_;
  int shift_;
  Cat cat_;
  size_t mid_ = 0;
  std::vector<u128> a_, a_mac_, b_, b_mac_, c_, c_mac_, lam_pre_, lam_pre_mac_, dx_, dy_;
};

// ------------------------------------------------------------- local linear map
class LinearGate : public GateT {
 public:
  LinearGate(OnlineCtx& c, const GatePtr& x, const GatePtr& y, size_t rows, size_t cols, LinFn f,
             std::vector<u64> konst)
      : GateT(x, y), ctx_(c), f_(std::move(f)), konst_(std::move(konst)) {
    set_dim_row(rows);
    set_dim_col(cols);
    if (!konst_.empty() && konst_.size() != rows * cols) throw std::invalid_argument("LinearGate: constant size");
  }

 private:
  static const std::vector<u128>& none() { static const std::vector<u128> e; return e; }
  void doReadOfflineFromFile() override {  // [lambda_z] = f([lambda_x], [lambda_y]), no preprocessing data
    lambda_shr() = f_(input_x()->lambda_shr(), input_y() ? input_y()->lambda_shr() : none());
    lambda_shr_mac() = f_(input_x()->lambda_shr_mac(), input_y() ? input_y()->lambda_shr_mac() : none());
  }
  void doRunOnline() override {
    Scope sc(ctx_, C_LOCAL);
    auto D = f_(input_x()->Delta_clear(), input_y() ? input_y()->Delta_clear() : none());
    for (size_t i = 0; i < konst_.size(); ++i) D[i] += konst_[i];
    reduce64(D);
    Delta_clear() = std::move(D);
  }

  OnlineCtx& ctx_;
  LinFn f_;
  std::vector<u64> konst_;
};

// ------------------------------------------------------------- comparison
// Number of AND gates of the carry circuit over `bits` bits: (p,g) pairs combined
// level by level; the lowest node of a level never needs its p, so its pair costs
// one AND instead of two (63 bits: 118 ANDs in 6 rounds; the paper counts 2k-2).
inline int carry_and_count(int bits) {
  int n = bits, ands = 0;
  while (n > 1) {
    const int pairs = n / 2;
    ands += 2 * pairs - 1;
    n = pairs + (n % 2);
  }
  return ands;
}
constexpr int kLtzBits = kRingBits;               // bit decomposition of lambda_x
inline int ltz_ands() { return carry_and_count(kRingBits - 1); }

struct BShare {  // SPDZ2k binary share, k = 1, s = 64: value and MAC in Z_{2^65}
  u128 v = 0, m = 0;
};

class GeqZeroGate : public GateT {
 public:
  GeqZeroGate(OnlineCtx& c, const GatePtr& x) : GateT(x, nullptr), ctx_(c) {
    set_dim_row(x->dim_row());
    set_dim_col(x->dim_col());
  }

 private:
  // Order written by FakeGeqZeroGate.
  void doReadOfflineFromFile() override {
    const size_t n = dim_row() * dim_col(), na = (size_t)ltz_ands() * n;
    auto& P = party();
    lambda_shr() = P.ReadShares(n);
    lambda_shr_mac() = P.ReadShares(n);
    bits_ = read_bin(n * kLtzBits);
    ta_ = read_bin(na); tb_ = read_bin(na); tc_ = read_bin(na);
    rb_ = read_bin(n);
    ra_v_ = P.ReadShares(n);
    ra_m_ = P.ReadShares(n);
  }
  std::vector<BShare> read_bin(size_t n) {
    auto v = party().ReadShares(n), m = party().ReadShares(n);
    std::vector<BShare> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = {m65(v[i]), m65(m[i])};
    return out;
  }

  // --- local operations on binary shares (low bit = the boolean value) ---
  BShare add(const BShare& x, const BShare& y) const { return {m65(x.v + y.v), m65(x.m + y.m)}; }  // XOR
  BShare add_pub(const BShare& x, unsigned c) const {                                            // XOR public bit
    return {m65(x.v + (ctx_.id == 0 ? (u128)c : 0)), m65(x.m + ctx_.beta * c)};
  }
  BShare bnot(const BShare& x) const { return add_pub(x, 1); }
  static BShare and_pub(const BShare& x, unsigned c) { return c ? x : BShare{}; }

  // One round of Beaver ANDs: z_i = x_i * y_i with triples taken in order.
  std::vector<BShare> and_batch(const std::vector<BShare>& X, const std::vector<BShare>& Y) {
    const size_t n = X.size();
    std::vector<u128> de(2 * n);
#pragma omp parallel for schedule(static) if (n >= kParMin)
    for (size_t i = 0; i < n; ++i) {
      de[2 * i] = m65(X[i].v - ta_[next_ + i].v);
      de[2 * i + 1] = m65(Y[i].v - tb_[next_ + i].v);
    }
    std::vector<u128> sendv = de;
    if (!sendv.empty() && ctx_.take_tamper(Tamper::BIN)) sendv[0] = m65(sendv[0] + 1);
    auto theirs = ctx_.exchange65(sendv, C_CMP);
    std::vector<BShare> Z(n);
    const size_t base = ctx_.bin_open.size();  // openings are logged at fixed offsets: 2 per AND
    ctx_.bin_open.resize(base + 2 * n);
    ctx_.bin_mac.resize(base + 2 * n);
#pragma omp parallel for schedule(static) if (n >= kParMin)
    for (size_t i = 0; i < n; ++i) {
      const BShare &a = ta_[next_ + i], &b = tb_[next_ + i], &c = tc_[next_ + i];
      const u128 d = m65(de[2 * i] + theirs[2 * i]), e = m65(de[2 * i + 1] + theirs[2 * i + 1]);
      ctx_.bin_open[base + 2 * i] = d;      ctx_.bin_mac[base + 2 * i] = m65(X[i].m - a.m);
      ctx_.bin_open[base + 2 * i + 1] = e;  ctx_.bin_mac[base + 2 * i + 1] = m65(Y[i].m - b.m);
      Z[i].v = m65(c.v + d * b.v + e * a.v + (ctx_.id == 0 ? d * e : 0));
      Z[i].m = m65(c.m + d * b.m + e * a.m + ctx_.beta * d * e);
    }
    next_ += n;
    ctx_.and_gates += n;
    return Z;
  }

  void doRunOnline() override {
    Scope sc(ctx_, C_CMP);
    const size_t n = dim_row() * dim_col();
    const auto& Dx = input_x()->Delta_clear();
    constexpr int L = kRingBits - 1;  // carry out of the low 63 bits
    // x = Delta - lambda = Delta + ~lambda + 1 (mod 2^64): generate/propagate per bit,
    // with A = Delta (public) and B = ~lambda (shared): p = A^B, g = A&B, carry-in 1.
    std::vector<std::vector<BShare>> P(n), G(n);
    std::vector<BShare> Bmsb(n);
    std::vector<unsigned> Amsb(n);
#pragma omp parallel for schedule(static) if (n >= 8)
    for (size_t e = 0; e < n; ++e) {
      const u64 A = (u64)Dx[e];
      P[e].resize(L); G[e].resize(L);
      for (int i = 0; i < L; ++i) {
        const unsigned ai = (A >> i) & 1;
        const BShare Bi = bnot(bits_[e * kLtzBits + i]);
        P[e][i] = add_pub(Bi, ai);
        G[e][i] = and_pub(Bi, ai);
      }
      G[e][0] = add(G[e][0], P[e][0]);  // g0 += p0 * cin (g0, p0 exclusive)
      Bmsb[e] = bnot(bits_[e * kLtzBits + L]);
      Amsb[e] = (A >> L) & 1;
    }
    // log-depth carry tree: (p_hi, g_hi) o (p_lo, g_lo) = (p_hi p_lo, g_hi ^ p_hi g_lo)
    int len = L;
    while (len > 1) {
      const int pairs = len / 2;
      const size_t per = (size_t)(2 * pairs - 1);  // ANDs per element at this level
      std::vector<BShare> X(n * per), Y(n * per);
#pragma omp parallel for schedule(static) if (n >= 8)
      for (size_t e = 0; e < n; ++e) {
        size_t k = e * per;
        for (int j = 0; j < pairs; ++j) {
          X[k] = P[e][2 * j + 1]; Y[k++] = G[e][2 * j];      // p_hi * g_lo
          if (j > 0) { X[k] = P[e][2 * j + 1]; Y[k++] = P[e][2 * j]; }  // p_hi * p_lo
        }
      }
      auto Z = and_batch(X, Y);
#pragma omp parallel for schedule(static) if (n >= 8)
      for (size_t e = 0; e < n; ++e) {
        size_t k = e * per;
        std::vector<BShare> P2(pairs + len % 2), G2(pairs + len % 2);
        for (int j = 0; j < pairs; ++j) {
          G2[j] = add(G[e][2 * j + 1], Z[k++]);
          if (j > 0) P2[j] = Z[k++];
        }
        if (len % 2) { P2[pairs] = P[e][len - 1]; G2[pairs] = G[e][len - 1]; }
        P[e] = std::move(P2); G[e] = std::move(G2);
      }
      len = pairs + len % 2;
    }
    // msb(x) = A63 ^ B63 ^ carry63;  z = (x >= 0) = 1 ^ msb(x)
    std::vector<BShare> z(n);
    for (size_t e = 0; e < n; ++e) z[e] = bnot(add_pub(add(G[e][0], Bmsb[e]), Amsb[e]));

    // B2A with a daBit (r in both domains): open v = z ^ r (a (1+s)-bit element),
    // [z] = v + (1 - 2v) [r]; then open [Delta_z] = [z] + [lambda_z].
    std::vector<u128> vsh(n), vmac(n);
    for (size_t e = 0; e < n; ++e) {
      const BShare t = add(z[e], rb_[e]);
      vsh[e] = t.v; vmac[e] = t.m;
    }
    auto vth = ctx_.exchange65(vsh, C_CMP);
    std::vector<u128> D(n), M(n);
    const size_t bbase = ctx_.bin_open.size();
    ctx_.bin_open.resize(bbase + n);
    ctx_.bin_mac.resize(bbase + n);
#pragma omp parallel for schedule(static) if (n >= kParMin)
    for (size_t e = 0; e < n; ++e) {
      const u128 v = m65(vsh[e] + vth[e]);
      ctx_.bin_open[bbase + e] = v; ctx_.bin_mac[bbase + e] = vmac[e];
      const unsigned vb = (unsigned)(v & 1);
      const u128 zv = (ctx_.id == 0 ? (u128)vb : 0) + (vb ? -ra_v_[e] : ra_v_[e]);
      const u128 zm = ctx_.alpha * vb + (vb ? -ra_m_[e] : ra_m_[e]);
      D[e] = zv + lambda_shr()[e];
      M[e] = zm + lambda_shr_mac()[e];
    }
    auto Dth = ctx_.exchange(D, C_CMP);
    for (size_t e = 0; e < n; ++e) {
      D[e] += Dth[e];
      ctx_.arith_open.push_back(D[e]); ctx_.arith_mac.push_back(M[e]);
    }
    reduce64(D);
    Delta_clear() = std::move(D);
    ctx_.ltz_count += n;

    bits_.clear(); ta_.clear(); tb_.clear(); tc_.clear(); rb_.clear(); ra_v_.clear(); ra_m_.clear();
    bits_.shrink_to_fit(); ta_.shrink_to_fit(); tb_.shrink_to_fit(); tc_.shrink_to_fit();
  }

  OnlineCtx& ctx_;
  std::vector<BShare> bits_, ta_, tb_, tc_, rb_;
  std::vector<u128> ra_v_, ra_m_;
  size_t next_ = 0;
};

// ------------------------------------------------------------- output to one party
class OutputToGate : public GateT {
 public:
  OutputToGate(OnlineCtx& c, const GatePtr& x, size_t owner) : GateT(x, nullptr), ctx_(c), owner_(owner) {
    set_dim_row(x->dim_row());
    set_dim_col(x->dim_col());
  }
  // The receiver's result; meaningful only after mac_check() has passed.
  const std::vector<u64>& result() const { return result_; }

 private:
  void doReadOfflineFromFile() override {
    const size_t n = dim_row() * dim_col();
    if (ctx_.id == (int)owner_) r_clear_ = party().ReadClear(n);
    r_ = party().ReadShares(n);
    r_mac_ = party().ReadShares(n);
  }
  void doRunOnline() override {
    Scope sc(ctx_, C_OUTPUT);
    const size_t n = dim_row() * dim_col();
    const auto& lam = input_x()->lambda_shr();
    const auto& lam_mac = input_x()->lambda_shr_mac();
    std::vector<u128> u(n), um(n);
    for (size_t i = 0; i < n; ++i) { u[i] = lam[i] - r_[i]; um[i] = lam_mac[i] - r_mac_[i]; }
    auto th = ctx_.exchange(u, C_OUTPUT);
    for (size_t i = 0; i < n; ++i) {
      u[i] += th[i];  // lambda_x - r: uniform to everyone but the receiver
      ctx_.arith_open.push_back(u[i]); ctx_.arith_mac.push_back(um[i]);
    }
    if (ctx_.id == (int)owner_) {
      result_.resize(n);
      const auto& D = input_x()->Delta_clear();
      for (size_t i = 0; i < n; ++i) result_[i] = (u64)D[i] - ((u64)u[i] + r_clear_[i]);  // x = Delta - lambda
    }
  }

  OnlineCtx& ctx_;
  size_t owner_;
  std::vector<u64> r_clear_, result_;
  std::vector<u128> r_, r_mac_;
};

// ------------------------------------------------------------- MAC check
inline std::array<uint8_t, 32> sha256(const std::vector<uint8_t>& msg) {
  std::array<uint8_t, 32> h{};
  unsigned len = 0;
  if (EVP_Digest(msg.data(), msg.size(), h.data(), &len, EVP_sha256(), nullptr) != 1 || len != 32)
    throw std::runtime_error("SHA-256 failed");
  return h;
}
inline std::vector<uint8_t> random_bytes(size_t n) {
  std::vector<uint8_t> b(n);
  if (RAND_bytes(b.data(), (int)n) != 1) throw std::runtime_error("RAND_bytes failed");
  return b;
}
// n pseudorandom 64-bit words from AES-128-CTR keyed by `key` (16 bytes).
inline std::vector<u64> prg_words(const std::vector<uint8_t>& key, size_t n) {
  std::vector<u64> out(n, 0);
  if (n == 0) return out;
  EVP_CIPHER_CTX* c = EVP_CIPHER_CTX_new();
  const uint8_t iv[16] = {0};
  int len = 0;
  const bool ok = c && EVP_EncryptInit_ex(c, EVP_aes_128_ctr(), nullptr, key.data(), iv) == 1 &&
                  EVP_EncryptUpdate(c, reinterpret_cast<uint8_t*>(out.data()), &len,
                                    reinterpret_cast<const uint8_t*>(out.data()), (int)(n * 8)) == 1;
  EVP_CIPHER_CTX_free(c);
  if (!ok) throw std::runtime_error("AES-CTR failed");
  return out;
}

// Commit-and-open: send H(msg || nonce), then (msg || nonce); returns the peer's msg
// after checking it against the peer's commitment.
inline std::vector<uint8_t> commit_and_open(OnlineCtx& c, const std::vector<uint8_t>& msg) {
  std::vector<uint8_t> opening = msg;
  auto nonce = random_bytes(16);
  opening.insert(opening.end(), nonce.begin(), nonce.end());
  const auto h = sha256(opening);
  auto their_h = c.exchange(std::vector<uint8_t>(h.begin(), h.end()), C_MACCHECK);
  auto their_open = c.exchange(opening, C_MACCHECK);
  const auto h2 = sha256(their_open);
  if (!std::equal(h2.begin(), h2.end(), their_h.begin())) throw std::runtime_error("peer opened a different commitment");
  their_open.resize(msg.size());
  return their_open;
}

inline void put_u128(std::vector<uint8_t>& b, u128 v) {
  for (int i = 0; i < 16; ++i) b.push_back((uint8_t)(v >> (8 * i)));
}
inline u128 get_u128(const std::vector<uint8_t>& b, size_t off) {
  u128 v = 0;
  for (int i = 15; i >= 0; --i) v = (v << 8) | b[off + i];
  return v;
}

// Pi_MACCheck over all values opened so far, in both domains at once.
// Returns true iff the MACs verify; on false the parties must abort.
inline bool mac_check(OnlineCtx& c) {
  Scope sc(c, C_MACCHECK);
  // 1. coin tossing: joint seed for the public coefficients r_i in Z_{2^s}
  auto my_seed = random_bytes(16);
  auto their_seed = commit_and_open(c, my_seed);
  std::vector<uint8_t> seed(16);
  for (int i = 0; i < 16; ++i) seed[i] = my_seed[i] ^ their_seed[i];
  const size_t na = c.arith_open.size(), nb = c.bin_open.size();
  const auto r = prg_words(seed, na + nb);
  // 2. y = sum r_i x_i;  z_j = sum r_i m_ij - alpha_j y   (mod 2^(k+s), per domain)
  u128 ya = 0, ma = 0, yb = 0, mb = 0;
#pragma omp parallel if (na + nb >= 16 * kParMin)
  {
    u128 la = 0, lma = 0, lb = 0, lmb = 0;  // per-thread partial sums (mod 2^128)
#pragma omp for schedule(static) nowait
    for (size_t i = 0; i < na; ++i) { la += (u128)r[i] * c.arith_open[i]; lma += (u128)r[i] * c.arith_mac[i]; }
#pragma omp for schedule(static) nowait
    for (size_t i = 0; i < nb; ++i) { lb += (u128)r[na + i] * c.bin_open[i]; lmb += (u128)r[na + i] * c.bin_mac[i]; }
#pragma omp critical(lrmd_macsum)
    { ya += la; ma += lma; yb += lb; mb += lmb; }
  }
  const u128 za = ma - c.alpha * ya;
  const u128 zb = m65(mb - c.beta * m65(yb));
  // 3. commit to z_j, open, check sum_j z_j == 0
  std::vector<uint8_t> msg;
  put_u128(msg, za);
  put_u128(msg, zb);
  auto th = commit_and_open(c, msg);
  const bool ok_a = (u128)(za + get_u128(th, 0)) == 0;
  const bool ok_b = m65(zb + get_u128(th, 16)) == 0;
  return ok_a && ok_b;
}

}  // namespace lrmd

#endif  // LRMDML_MDML_EXT_H
