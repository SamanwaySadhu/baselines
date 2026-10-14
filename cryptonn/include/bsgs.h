// bsgs.h -- bounded discrete logarithm in P-256 by baby-step giant-step (BSGS).
//
// FEIP / FEBO decryption ends in g^v for a small signed integer v (an inner product of
// quantized values); recovering v is a bounded discrete log, which CryptoNN (like every
// DDH-based FE scheme) solves by BSGS over a precomputed table.
//
// Table (input independent, built once by the server in its offline phase): the baby
// steps j*g for j = 1 .. m, m = 2^table_bits - 1, keyed by the low 64 bits of the
// affine x-coordinate (= the last 8 bytes of the compressed encoding). Since j*g and
// -j*g share x, one entry serves both signs: the table covers [-m, m] and the giant
// stride is M = 2m + 1. Storage is a flat open-addressing hash (linear probing, load
// <= 1/2): 8-byte keys + 4-byte values per slot.
//
// solve(P, bound): v in [-bound, bound] with v*g == P. Giant steps visit
// Q_k = P - k*M*g in the order k = 0, +1, -1, +2, -2, ... (early exit; values near 0
// are found first) for |k| <= ceil((bound - m) / M); a table hit j gives v = k*M +- j,
// the sign coming from the y-parity. Every hit is verified against j*g (rules out
// 64-bit key collisions) and v is checked against the bound; since the discrete log is
// unique mod q, a verified v outside the bound means there is no solution (nullopt).
// Worst-case giant steps per solve: 2 * ceil(max(0, bound - m) / M).
//
// Speed: x-coordinates require affine points. OpenSSL's nistz256 implementation
// always inverts when asked for affine coordinates, so the table build and the giant
// steps run in a second EC_GROUP for the same curve built from its explicit parameters
// (generic GFp method), where EC_POINTs_make_affine (one shared inversion per batch)
// followed by EC_POINT_get_affine_coordinates (free once Z = 1) is cheap. Points cross
// between the two groups via affine coordinates.
//
// Thread safety: after construction the object is immutable; concurrent solve() calls
// are safe (each uses its thread's BN_CTX and local points). The constructor can
// spread the table build over `threads` std::threads (identical table for any count).
#ifndef LRCN_BSGS_H
#define LRCN_BSGS_H

#include "ec_group.h"

#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace lrcn {

class Dlog {
 public:
  explicit Dlog(int table_bits, int threads = 1) : bits_(table_bits) {
    if (table_bits < 1 || table_bits > 30)
      throw std::invalid_argument("Dlog: table_bits must be in [1, 30]");
    m_ = (int64_t(1) << table_bits) - 1;
    stride_ = 2 * m_ + 1;
    init_generic_group();
    build(threads < 1 ? 1 : threads);
  }
  ~Dlog() {
    EC_POINT_free(g2_step_neg_);
    EC_POINT_free(g2_step_pos_);
    EC_GROUP_free(g2_);
  }
  Dlog(const Dlog&) = delete;
  Dlog& operator=(const Dlog&) = delete;

  // v in [-bound, bound] with v*g == P, or nullopt if there is none.
  std::optional<int64_t> solve(const Point& P, int64_t bound) const;

  size_t table_entries() const { return size_t(m_) + 1; }  // j = 0 .. m (0 = identity)
  size_t memory_bytes() const {
    return keys_.capacity() * sizeof(uint64_t) + vals_.capacity() * sizeof(uint32_t);
  }
  int table_bits() const { return bits_; }
  int64_t half_width() const { return m_; }   // a single lookup covers [-m, m]
  int64_t giant_stride() const { return stride_; }
  // Worst-case number of giant steps (excluding the k = 0 lookup) for a bound.
  int64_t max_giant_steps(int64_t bound) const {
    return bound > m_ ? 2 * ((bound - m_ + stride_ - 1) / stride_) : 0;
  }
  // Smallest table_bits in [min_bits, max_bits] whose single lookup covers `bound`
  // (max_bits when none does).
  static int bits_for_bound(int64_t bound, int min_bits = 8, int max_bits = 22) {
    int b = min_bits;
    while (b < max_bits && (int64_t(1) << b) - 1 < bound) ++b;
    return b;
  }

 private:
  using EcPointPtr = std::unique_ptr<EC_POINT, decltype(&EC_POINT_free)>;

  static uint64_t key_of_x(const BIGNUM* x) {
    uint8_t le[32];
    if (BN_bn2lebinpad(x, le, 32) != 32) ossl_fail("BN_bn2lebinpad");
    uint64_t k = 0;
    for (int i = 7; i >= 0; --i) k = (k << 8) | le[i];
    return k ? k : 1;  // 0 marks an empty slot
  }

  void init_generic_group() {
    const auto& G = EcGroup::get();
    BN_CTX* ctx = bn_ctx();
    BN_CTX_start(ctx);
    BIGNUM* p = BN_CTX_get(ctx);
    BIGNUM* a = BN_CTX_get(ctx);
    BIGNUM* b = BN_CTX_get(ctx);
    BIGNUM* x = BN_CTX_get(ctx);
    BIGNUM* y = BN_CTX_get(ctx);
    if (!y) ossl_fail("BN_CTX_get");
    ossl_check(EC_GROUP_get_curve(G.group(), p, a, b, ctx), "EC_GROUP_get_curve");
    g2_ = EC_GROUP_new_curve_GFp(p, a, b, ctx);
    if (!g2_) ossl_fail("EC_GROUP_new_curve_GFp");
    ossl_check(EC_POINT_get_affine_coordinates(G.group(), G.gen(), x, y, ctx), "get_affine(g)");
    EcPointPtr gen2(EC_POINT_new(g2_), EC_POINT_free);
    if (!gen2) ossl_fail("EC_POINT_new");
    ossl_check(EC_POINT_set_affine_coordinates(g2_, gen2.get(), x, y, ctx), "set_affine(g)");
    ossl_check(EC_GROUP_set_generator(g2_, gen2.get(), G.order(), BN_value_one()),
               "EC_GROUP_set_generator");
    // Giant steps +-M*g in the generic group, affine (Z = 1: cheaper mixed additions).
    Point s = Point::gen_mul_i64(stride_);
    ossl_check(EC_POINT_get_affine_coordinates(G.group(), s.p, x, y, ctx), "get_affine(M*g)");
    g2_step_pos_ = EC_POINT_new(g2_);
    g2_step_neg_ = EC_POINT_new(g2_);
    if (!g2_step_pos_ || !g2_step_neg_) ossl_fail("EC_POINT_new");
    ossl_check(EC_POINT_set_affine_coordinates(g2_, g2_step_pos_, x, y, ctx), "set_affine(M*g)");
    ossl_check(EC_POINT_copy(g2_step_neg_, g2_step_pos_), "EC_POINT_copy");
    ossl_check(EC_POINT_invert(g2_, g2_step_neg_, ctx), "EC_POINT_invert");
    BN_CTX_end(ctx);
  }

  // Keys of j*g for j in [lo, hi) (lo >= 1) into keys[j].
  void compute_keys(int64_t lo, int64_t hi, std::vector<uint64_t>& keys) const {
    if (lo >= hi) return;
    BN_CTX* ctx = bn_ctx();
    const EC_POINT* g = EC_GROUP_get0_generator(g2_);
    constexpr int kBatch = 512;
    std::vector<EC_POINT*> batch(kBatch);
    for (auto& b : batch)
      if (!(b = EC_POINT_new(g2_))) ossl_fail("EC_POINT_new");
    EcPointPtr cur(EC_POINT_new(g2_), EC_POINT_free);
    if (!cur) ossl_fail("EC_POINT_new");
    BN_CTX_start(ctx);
    BIGNUM* x = BN_CTX_get(ctx);
    BIGNUM* s = BN_CTX_get(ctx);
    if (!x) ossl_fail("BN_CTX_get");
    ossl_check(BN_set_word(s, static_cast<BN_ULONG>(lo)), "BN_set_word");
    ossl_check(EC_POINT_mul(g2_, cur.get(), s, nullptr, nullptr, ctx), "EC_POINT_mul");
    for (int64_t j = lo; j < hi; j += kBatch) {
      const int n = int(std::min<int64_t>(kBatch, hi - j));
      for (int i = 0; i < n; ++i) {
        ossl_check(EC_POINT_copy(batch[i], cur.get()), "EC_POINT_copy");
        ossl_check(EC_POINT_add(g2_, cur.get(), cur.get(), g, ctx), "EC_POINT_add");
      }
      LRCN_DEPRECATED_BEGIN
      ossl_check(EC_POINTs_make_affine(g2_, size_t(n), batch.data(), ctx), "EC_POINTs_make_affine");
      LRCN_DEPRECATED_END
      for (int i = 0; i < n; ++i) {
        ossl_check(EC_POINT_get_affine_coordinates(g2_, batch[i], x, nullptr, ctx), "get_affine");
        keys[size_t(j + i)] = key_of_x(x);
      }
    }
    BN_CTX_end(ctx);
    for (auto* b : batch) EC_POINT_free(b);
  }

  void build(int threads) {
    std::vector<uint64_t> keys(size_t(m_) + 1, 0);
    const int64_t n = m_;  // j = 1 .. m
    threads = int(std::min<int64_t>(threads, std::max<int64_t>(1, n / 4096)));
    if (threads <= 1) {
      compute_keys(1, m_ + 1, keys);
    } else {
      std::vector<std::thread> th;
      std::vector<std::exception_ptr> err(static_cast<size_t>(threads));
      for (int t = 0; t < threads; ++t) {
        const int64_t lo = 1 + n * t / threads, hi = 1 + n * (t + 1) / threads;
        th.emplace_back([this, lo, hi, &keys, &err, t] {
          try {
            compute_keys(lo, hi, keys);
          } catch (...) {
            err[size_t(t)] = std::current_exception();
          }
        });
      }
      for (auto& x : th) x.join();
      for (auto& e : err)
        if (e) std::rethrow_exception(e);
    }
    size_t cap = 1;
    while (cap < 2 * size_t(m_)) cap <<= 1;
    keys_.assign(cap, 0);
    vals_.assign(cap, 0);
    mask_ = cap - 1;
    for (int64_t j = 1; j <= m_; ++j) {
      size_t s = size_t(keys[size_t(j)]) & mask_;
      while (keys_[s]) s = (s + 1) & mask_;
      keys_[s] = keys[size_t(j)];
      vals_[s] = uint32_t(j);
    }
  }

  // Table lookup of an affine point (x, y) != identity: the signed j in [-m, m] with
  // (x, y) == j*g, verified against j*g.
  std::optional<int64_t> lookup(const BIGNUM* x, const BIGNUM* y, BN_CTX* ctx) const {
    const uint64_t key = key_of_x(x);
    for (size_t s = size_t(key) & mask_; keys_[s]; s = (s + 1) & mask_) {
      if (keys_[s] != key) continue;
      const int64_t j = vals_[s];
      const Point C = Point::gen_mul_i64(j);
      BN_CTX_start(ctx);
      BIGNUM* xc = BN_CTX_get(ctx);
      BIGNUM* yc = BN_CTX_get(ctx);
      if (!yc) ossl_fail("BN_CTX_get");
      ossl_check(EC_POINT_get_affine_coordinates(EcGroup::get().group(), C.p, xc, yc, ctx),
                 "get_affine");
      const bool hit = BN_cmp(xc, x) == 0;
      const bool same_sign = BN_is_odd(yc) == BN_is_odd(y);
      BN_CTX_end(ctx);
      if (hit) return same_sign ? j : -j;
    }
    return std::nullopt;
  }

  int bits_ = 0;
  int64_t m_ = 0, stride_ = 0;
  size_t mask_ = 0;
  std::vector<uint64_t> keys_;
  std::vector<uint32_t> vals_;
  EC_GROUP* g2_ = nullptr;  // same curve, generic GFp method
  EC_POINT* g2_step_pos_ = nullptr;  // +M*g in g2_
  EC_POINT* g2_step_neg_ = nullptr;  // -M*g in g2_
};

inline std::optional<int64_t> Dlog::solve(const Point& P, int64_t bound) const {
  if (bound < 0) return std::nullopt;
  const auto& G = EcGroup::get();
  auto within = [bound](int64_t v) -> std::optional<int64_t> {
    if (v < -bound || v > bound) return std::nullopt;
    return v;
  };
  if (P.is_identity()) return 0;
  BN_CTX* ctx = bn_ctx();
  BN_CTX_start(ctx);
  struct CtxEnd {
    BN_CTX* c;
    ~CtxEnd() { BN_CTX_end(c); }
  } ctx_end{ctx};
  BIGNUM* x = BN_CTX_get(ctx);
  BIGNUM* y = BN_CTX_get(ctx);
  if (!y) ossl_fail("BN_CTX_get");

  // k = 0: one affine conversion (nistz256 inversion) and a lookup.
  ossl_check(EC_POINT_get_affine_coordinates(G.group(), P.p, x, y, ctx), "get_affine(P)");
  if (auto j = lookup(x, y, ctx)) return within(*j);
  const int64_t kmax = bound > m_ ? (bound - m_ + stride_ - 1) / stride_ : 0;
  if (kmax == 0) return std::nullopt;

  // Giant steps in the generic group: Qp = P - k*M*g (v = k*M + j) and
  // Qn = P + k*M*g (v = -k*M + j), batched with growing batch sizes.
  EcPointPtr qp(EC_POINT_new(g2_), EC_POINT_free), qn(EC_POINT_new(g2_), EC_POINT_free);
  if (!qp || !qn) ossl_fail("EC_POINT_new");
  ossl_check(EC_POINT_set_affine_coordinates(g2_, qp.get(), x, y, ctx), "set_affine(P)");
  ossl_check(EC_POINT_copy(qn.get(), qp.get()), "EC_POINT_copy");
  constexpr int kMaxBatch = 64;  // k values per batch (2 points each)
  std::vector<EC_POINT*> pts;
  pts.reserve(2 * kMaxBatch);
  struct Free {
    std::vector<EC_POINT*>& v;
    ~Free() {
      for (auto* p : v) EC_POINT_free(p);
    }
  } free_pts{pts};
  int batch = 2;
  for (int64_t k0 = 1; k0 <= kmax; k0 += batch, batch = std::min(2 * batch, kMaxBatch)) {
    const int n = int(std::min<int64_t>(batch, kmax - k0 + 1));
    while (pts.size() < size_t(2 * n)) {
      EC_POINT* e = EC_POINT_new(g2_);
      if (!e) ossl_fail("EC_POINT_new");
      pts.push_back(e);
    }
    for (int i = 0; i < n; ++i) {
      ossl_check(EC_POINT_add(g2_, qp.get(), qp.get(), g2_step_neg_, ctx), "EC_POINT_add");
      ossl_check(EC_POINT_add(g2_, qn.get(), qn.get(), g2_step_pos_, ctx), "EC_POINT_add");
      ossl_check(EC_POINT_copy(pts[size_t(2 * i)], qp.get()), "EC_POINT_copy");
      ossl_check(EC_POINT_copy(pts[size_t(2 * i + 1)], qn.get()), "EC_POINT_copy");
    }
    LRCN_DEPRECATED_BEGIN
    ossl_check(EC_POINTs_make_affine(g2_, size_t(2 * n), pts.data(), ctx), "EC_POINTs_make_affine");
    LRCN_DEPRECATED_END
    for (int i = 0; i < 2 * n; ++i) {
      const int64_t k = k0 + i / 2;
      const int64_t base = (i % 2 == 0) ? k * stride_ : -k * stride_;
      if (EC_POINT_is_at_infinity(g2_, pts[size_t(i)])) return within(base);
      ossl_check(EC_POINT_get_affine_coordinates(g2_, pts[size_t(i)], x, y, ctx), "get_affine");
      if (auto j = lookup(x, y, ctx)) return within(base + *j);
    }
  }
  return std::nullopt;
}

}  // namespace lrcn

#endif  // LRCN_BSGS_H
