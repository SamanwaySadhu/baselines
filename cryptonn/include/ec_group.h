// ec_group.h -- NIST P-256 group context and RAII scalar / point wrappers (OpenSSL 3).
//
// Every CryptoNN primitive in this baseline (FEIP, FEBO, bounded discrete log) works in
// the prime-order group of NIST P-256 (128-bit security), written additively here:
// the paper's g^x is Point::gen_mul(x), h^r * g^x is r*h + x*g, and a / b is a - b.
//
//   * EcGroup  -- singleton owning the EC_GROUP (OpenSSL picks the constant-time
//                 nistz256 x86-64 implementation for P-256, which has a built-in
//                 fixed-base table for g; EC_GROUP_precompute_mult is called anyway).
//   * Scalar   -- RAII BIGNUM reduced mod q (the group order). Signed integers map to
//                 scalars mod q (from_i64(-5) = q - 5).
//   * Point    -- RAII, copyable EC_POINT. Serialized compressed (kPointBytes = 33);
//                 the identity is encoded as 33 zero bytes and is rejected by
//                 from_bytes() unless explicitly allowed. P-256 has cofactor 1, so an
//                 on-curve check is a complete subgroup check.
//
// Thread safety: the EC_GROUP is immutable after construction and may be shared by any
// number of threads. Each thread uses its own BN_CTX (bn_ctx(), thread_local). Scalar
// and Point objects are not synchronized: share them read-only or not at all.
//
// Multi-scalar multiplication (Point::multi_mul) splits its terms: terms whose scalar
// is a small signed integer (|k| < 2^40, e.g. the weights / errors used as FEIP
// function vectors) go through a bucket (Pippenger) method built from plain point
// additions, the rest through OpenSSL's interleaved EC_POINTs_mul; the cheaper route
// is chosen from an operation-count estimate. The result is the same either way.
//
// Several of the batch APIs used here (EC_POINTs_mul, EC_POINTs_make_affine,
// EC_GROUP_precompute_mult) are marked deprecated in OpenSSL 3.0 but are still
// provided; their deprecation warnings are silenced locally.
#ifndef LRCN_EC_GROUP_H
#define LRCN_EC_GROUP_H

#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/err.h>
#include <openssl/obj_mac.h>
#include <openssl/rand.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define LRCN_DEPRECATED_BEGIN \
  _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")
#define LRCN_DEPRECATED_END _Pragma("GCC diagnostic pop")

namespace lrcn {

static_assert(sizeof(BN_ULONG) >= 8, "64-bit BN_ULONG expected (BN_get_word of 40-bit values)");

constexpr size_t kPointBytes = 33;   // compressed SEC1 encoding
constexpr size_t kScalarBytes = 32;  // big-endian, < q

// Throws std::runtime_error carrying the oldest queued OpenSSL error.
[[noreturn]] inline void ossl_fail(const char* what) {
  const unsigned long e = ERR_get_error();
  char buf[256] = {0};
  if (e) ERR_error_string_n(e, buf, sizeof buf);
  ERR_clear_error();
  throw std::runtime_error(std::string("OpenSSL: ") + what + (e ? std::string(": ") + buf : ""));
}
inline void ossl_check(int ok, const char* what) {
  if (ok != 1) ossl_fail(what);
}

// Per-thread BN_CTX (scratch BIGNUMs for every OpenSSL arithmetic call).
inline BN_CTX* bn_ctx() {
  struct Holder {
    BN_CTX* c;
    Holder() : c(BN_CTX_new()) {
      if (!c) ossl_fail("BN_CTX_new");
    }
    ~Holder() { BN_CTX_free(c); }
    Holder(const Holder&) = delete;
    Holder& operator=(const Holder&) = delete;
  };
  thread_local Holder h;
  return h.c;
}

// RAII BN_CTX_start / BN_CTX_end frame: the frame is closed even when an OpenSSL
// failure throws between the two (keeps the thread_local BN_CTX balanced).
struct BnCtxFrame {
  BN_CTX* c;
  explicit BnCtxFrame(BN_CTX* ctx) : c(ctx) { BN_CTX_start(c); }
  ~BnCtxFrame() { BN_CTX_end(c); }
  BnCtxFrame(const BnCtxFrame&) = delete;
  BnCtxFrame& operator=(const BnCtxFrame&) = delete;
};

// ------------------------------------------------------------------ group
class EcGroup {
 public:
  static const EcGroup& get() {
    static const EcGroup g;  // thread-safe one-time initialization
    return g;
  }
  const EC_GROUP* group() const { return group_; }
  const BIGNUM* order() const { return order_; }
  const EC_POINT* gen() const { return EC_GROUP_get0_generator(group_); }

  ~EcGroup() { EC_GROUP_free(group_); }
  EcGroup(const EcGroup&) = delete;
  EcGroup& operator=(const EcGroup&) = delete;

 private:
  EcGroup() {
    group_ = EC_GROUP_new_by_curve_name(NID_X9_62_prime256v1);
    if (!group_) ossl_fail("EC_GROUP_new_by_curve_name(P-256)");
    LRCN_DEPRECATED_BEGIN
    ossl_check(EC_GROUP_precompute_mult(group_, bn_ctx()), "EC_GROUP_precompute_mult");
    LRCN_DEPRECATED_END
    order_ = EC_GROUP_get0_order(group_);  // owned by group_
  }
  EC_GROUP* group_ = nullptr;
  const BIGNUM* order_ = nullptr;
};

// ----------------------------------------------------------------- scalar
struct Scalar {
  BIGNUM* bn = nullptr;

  Scalar() : bn(BN_new()) {
    if (!bn) ossl_fail("BN_new");
    BN_zero(bn);
  }
  Scalar(const Scalar& o) : bn(BN_dup(o.bn)) {
    if (!bn) ossl_fail("BN_dup");
  }
  Scalar(Scalar&& o) noexcept : bn(o.bn) { o.bn = nullptr; }
  Scalar& operator=(const Scalar& o) {  // also revives a moved-from object
    if (this == &o) return *this;
    if (!bn) {
      if (!(bn = BN_dup(o.bn))) ossl_fail("BN_dup");
    } else if (!BN_copy(bn, o.bn)) {
      ossl_fail("BN_copy");
    }
    return *this;
  }
  Scalar& operator=(Scalar&& o) noexcept {
    std::swap(bn, o.bn);
    return *this;
  }
  ~Scalar() { BN_clear_free(bn); }

  const BIGNUM* get() const { return bn; }

  // Uniform in [1, q-1].
  static Scalar random() {
    Scalar s;
    do {
      ossl_check(BN_rand_range(s.bn, EcGroup::get().order()), "BN_rand_range");
    } while (BN_is_zero(s.bn));
    return s;
  }
  // v mod q (negative v -> q - |v|).
  static Scalar from_i64(int64_t v) {
    Scalar s;
    const uint64_t mag = v < 0 ? uint64_t(0) - uint64_t(v) : uint64_t(v);
    uint8_t be[8];
    for (int i = 0; i < 8; ++i) be[i] = uint8_t(mag >> (56 - 8 * i));
    if (!BN_bin2bn(be, 8, s.bn)) ossl_fail("BN_bin2bn");
    if (v < 0 && !BN_is_zero(s.bn))
      ossl_check(BN_sub(s.bn, EcGroup::get().order(), s.bn), "BN_sub");
    return s;
  }

  Scalar add(const Scalar& o) const {
    Scalar r;
    ossl_check(BN_mod_add(r.bn, bn, o.bn, EcGroup::get().order(), bn_ctx()), "BN_mod_add");
    return r;
  }
  Scalar sub(const Scalar& o) const {
    Scalar r;
    ossl_check(BN_mod_sub(r.bn, bn, o.bn, EcGroup::get().order(), bn_ctx()), "BN_mod_sub");
    return r;
  }
  Scalar mul(const Scalar& o) const {
    Scalar r;
    ossl_check(BN_mod_mul(r.bn, bn, o.bn, EcGroup::get().order(), bn_ctx()), "BN_mod_mul");
    return r;
  }
  Scalar neg() const {
    Scalar r;
    if (!BN_is_zero(bn)) ossl_check(BN_sub(r.bn, EcGroup::get().order(), bn), "BN_sub");
    return r;
  }
  // Multiplicative inverse mod q; throws on zero.
  Scalar inv() const {
    if (BN_is_zero(bn)) throw std::domain_error("Scalar::inv of zero");
    Scalar r;
    if (!BN_mod_inverse(r.bn, bn, EcGroup::get().order(), bn_ctx())) ossl_fail("BN_mod_inverse");
    return r;
  }
  bool is_zero() const { return BN_is_zero(bn); }
  bool operator==(const Scalar& o) const { return BN_cmp(bn, o.bn) == 0; }

  void to_bytes32(uint8_t* out) const {
    if (BN_bn2binpad(bn, out, int(kScalarBytes)) != int(kScalarBytes)) ossl_fail("BN_bn2binpad");
  }
  std::array<uint8_t, kScalarBytes> to_bytes32() const {
    std::array<uint8_t, kScalarBytes> a;
    to_bytes32(a.data());
    return a;
  }
  // Rejects encodings >= q (std::invalid_argument).
  static Scalar from_bytes32(const uint8_t* in) {
    Scalar s;
    if (!BN_bin2bn(in, int(kScalarBytes), s.bn)) ossl_fail("BN_bin2bn");
    if (BN_cmp(s.bn, EcGroup::get().order()) >= 0)
      throw std::invalid_argument("Scalar::from_bytes32: value >= group order");
    return s;
  }
};

// ------------------------------------------------------------------ point
struct Point {
  EC_POINT* p = nullptr;

  Point() : p(EC_POINT_new(EcGroup::get().group())) {
    if (!p) ossl_fail("EC_POINT_new");
    ossl_check(EC_POINT_set_to_infinity(EcGroup::get().group(), p), "EC_POINT_set_to_infinity");
  }
  Point(const Point& o) : p(EC_POINT_dup(o.p, EcGroup::get().group())) {
    if (!p) ossl_fail("EC_POINT_dup");
  }
  Point(Point&& o) noexcept : p(o.p) { o.p = nullptr; }
  Point& operator=(const Point& o) {  // also revives a moved-from object
    if (this == &o) return *this;
    if (!p) {
      if (!(p = EC_POINT_dup(o.p, EcGroup::get().group()))) ossl_fail("EC_POINT_dup");
    } else {
      ossl_check(EC_POINT_copy(p, o.p), "EC_POINT_copy");
    }
    return *this;
  }
  Point& operator=(Point&& o) noexcept {
    std::swap(p, o.p);
    return *this;
  }
  ~Point() { EC_POINT_free(p); }

  static Point identity() { return Point(); }

  // k*g (fixed-base: nistz256 built-in / precomputed generator table).
  static Point gen_mul(const Scalar& k) {
    const auto& G = EcGroup::get();
    Point r;
    ossl_check(EC_POINT_mul(G.group(), r.p, k.bn, nullptr, nullptr, bn_ctx()), "EC_POINT_mul(g)");
    return r;
  }
  static Point gen_mul_i64(int64_t v) { return gen_mul(Scalar::from_i64(v)); }
  // a*g + b*P in one double-scalar EC_POINT_mul.
  static Point gen_mul_add(const Scalar& a, const Point& P, const Scalar& b) {
    const auto& G = EcGroup::get();
    Point r;
    ossl_check(EC_POINT_mul(G.group(), r.p, a.bn, P.p, b.bn, bn_ctx()), "EC_POINT_mul(g,P)");
    return r;
  }
  // k*P (variable base).
  Point mul(const Scalar& k) const {
    const auto& G = EcGroup::get();
    Point r;
    ossl_check(EC_POINT_mul(G.group(), r.p, nullptr, p, k.bn, bn_ctx()), "EC_POINT_mul(P)");
    return r;
  }
  Point add(const Point& o) const {
    Point r;
    ossl_check(EC_POINT_add(EcGroup::get().group(), r.p, p, o.p, bn_ctx()), "EC_POINT_add");
    return r;
  }
  Point neg() const {
    Point r(*this);
    ossl_check(EC_POINT_invert(EcGroup::get().group(), r.p, bn_ctx()), "EC_POINT_invert");
    return r;
  }
  Point sub(const Point& o) const { return add(o.neg()); }
  Point dbl() const {
    Point r;
    ossl_check(EC_POINT_dbl(EcGroup::get().group(), r.p, p, bn_ctx()), "EC_POINT_dbl");
    return r;
  }
  bool is_identity() const { return EC_POINT_is_at_infinity(EcGroup::get().group(), p) == 1; }
  bool operator==(const Point& o) const {
    const int c = EC_POINT_cmp(EcGroup::get().group(), p, o.p, bn_ctx());
    if (c < 0) ossl_fail("EC_POINT_cmp");
    return c == 0;
  }
  bool operator!=(const Point& o) const { return !(*this == o); }

  // sum_i k_i * P_i (identity for no terms). See the file comment for the strategy.
  static Point multi_mul(const std::vector<const Point*>& pts, const std::vector<const Scalar*>& ks);
  // sum_i v_i * P_i for signed 64-bit coefficients (same algorithm, no BIGNUMs needed).
  static Point multi_mul_i64(const std::vector<const Point*>& pts, const std::vector<int64_t>& vs);

  // Compressed SEC1 (0x02/0x03 || x); the identity encodes as 33 zero bytes.
  void to_bytes(uint8_t* out) const {
    const auto& G = EcGroup::get();
    if (is_identity()) {
      std::memset(out, 0, kPointBytes);
      return;
    }
    if (EC_POINT_point2oct(G.group(), p, POINT_CONVERSION_COMPRESSED, out, kPointBytes, bn_ctx()) !=
        kPointBytes)
      ossl_fail("EC_POINT_point2oct");
  }
  std::array<uint8_t, kPointBytes> to_bytes() const {
    std::array<uint8_t, kPointBytes> a;
    to_bytes(a.data());
    return a;
  }
  // Decodes and validates (well-formed, on the curve; not the identity unless allowed).
  static std::optional<Point> try_from_bytes(const uint8_t* in, bool allow_identity = false) {
    const auto& G = EcGroup::get();
    bool all_zero = true;
    for (size_t i = 0; i < kPointBytes; ++i) all_zero &= (in[i] == 0);
    if (all_zero) {
      if (allow_identity) return Point();
      return std::nullopt;
    }
    if (in[0] != 0x02 && in[0] != 0x03) return std::nullopt;
    Point r;
    if (EC_POINT_oct2point(G.group(), r.p, in, kPointBytes, bn_ctx()) != 1) {
      ERR_clear_error();
      return std::nullopt;
    }
    if (EC_POINT_is_on_curve(G.group(), r.p, bn_ctx()) != 1 || r.is_identity()) {
      ERR_clear_error();
      return std::nullopt;
    }
    return r;
  }
  // As try_from_bytes, but throws std::invalid_argument on an invalid encoding.
  static Point from_bytes(const uint8_t* in, bool allow_identity = false) {
    auto r = try_from_bytes(in, allow_identity);
    if (!r) throw std::invalid_argument("Point::from_bytes: invalid P-256 point encoding");
    return std::move(*r);
  }
};

namespace detail {

// Bucket (Pippenger) multi-scalar multiplication for small signed coefficients, using
// only point additions and doublings: sum_i sign_i * mag_i * P_i with mag_i < 2^bits.
// Window c digits; per window: one addition per nonzero digit into its bucket, then
// the running-sum trick (2 * 2^c additions), then c doublings of the accumulator.
inline int pippenger_window(size_t n, int bits) {
  int best_c = 1;
  double best = 1e300;
  for (int c = 1; c <= std::min(bits, 16); ++c) {
    const int W = (bits + c - 1) / c;
    const double cost = double(W) * (double(n) + double(2u << c)) + bits;
    if (cost < best) best = cost, best_c = c;
  }
  return best_c;
}
// Estimated costs in units of one generic point addition (EC_POINT_add, ~0.55 us),
// calibrated on x86-64 / OpenSSL 3.0.13: the bucket method runs at ~0.6x its operation
// count (bucket fills are often copies, doublings are cheaper than additions); the
// interleaved nistz256 multi-scalar mult costs ~31 us + ~10 us per full-width term.
inline double pippenger_cost(size_t n, int bits) {
  if (n == 0 || bits == 0) return 0;
  const int c = pippenger_window(n, bits);
  const int W = (bits + c - 1) / c;
  return 0.6 * (double(W) * (double(n) + double(2u << c)) + bits);
}
inline double interleaved_cost(size_t n) { return n == 0 ? 0 : 38.0 + 18.5 * double(n); }

inline void pippenger_small(EC_POINT* out, const std::vector<const EC_POINT*>& pts,
                            const std::vector<uint64_t>& mags, int bits) {
  const auto& G = EcGroup::get();
  const EC_GROUP* grp = G.group();
  BN_CTX* ctx = bn_ctx();
  ossl_check(EC_POINT_set_to_infinity(grp, out), "EC_POINT_set_to_infinity");
  if (pts.empty() || bits == 0) return;
  const int c = pippenger_window(pts.size(), bits);
  const int W = (bits + c - 1) / c;
  const uint64_t mask = (uint64_t(1) << c) - 1;
  std::vector<Point> buckets(size_t(mask) + 1);
  std::vector<char> used(size_t(mask) + 1);
  Point running, sum;
  for (int w = W - 1; w >= 0; --w) {
    for (int i = 0; i < c && !EC_POINT_is_at_infinity(grp, out); ++i)
      ossl_check(EC_POINT_dbl(grp, out, out, ctx), "EC_POINT_dbl");
    std::fill(used.begin(), used.end(), 0);
    for (size_t i = 0; i < pts.size(); ++i) {
      const uint64_t dgt = (mags[i] >> (w * c)) & mask;
      if (!dgt) continue;
      if (used[dgt]) {
        ossl_check(EC_POINT_add(grp, buckets[dgt].p, buckets[dgt].p, pts[i], ctx), "EC_POINT_add");
      } else {
        ossl_check(EC_POINT_copy(buckets[dgt].p, pts[i]), "EC_POINT_copy");
        used[dgt] = 1;
      }
    }
    // sum_d d * B_d = sum_{d} (B_top + ... + B_d), accumulated from the top digit down.
    bool have_run = false, have_sum = false;
    for (uint64_t dgt = mask; dgt >= 1; --dgt) {
      if (used[dgt]) {
        if (have_run) {
          ossl_check(EC_POINT_add(grp, running.p, running.p, buckets[dgt].p, ctx), "EC_POINT_add");
        } else {
          ossl_check(EC_POINT_copy(running.p, buckets[dgt].p), "EC_POINT_copy");
          have_run = true;
        }
      }
      if (have_run) {
        if (have_sum) {
          ossl_check(EC_POINT_add(grp, sum.p, sum.p, running.p, ctx), "EC_POINT_add");
        } else {
          ossl_check(EC_POINT_copy(sum.p, running.p), "EC_POINT_copy");
          have_sum = true;
        }
      }
    }
    if (have_sum) ossl_check(EC_POINT_add(grp, out, out, sum.p, ctx), "EC_POINT_add");
  }
}

inline int bit_length(uint64_t v) { return v ? 64 - __builtin_clzll(v) : 0; }

// Shared driver: small terms (pts_s, signed mags) and big terms (pts_b, ks_b).
inline Point multi_mul_split(const std::vector<const Point*>& pts_s, const std::vector<int64_t>& vs,
                             std::vector<const EC_POINT*>& pts_b, std::vector<const BIGNUM*>& ks_b) {
  const auto& G = EcGroup::get();
  // Small terms: negate points with negative coefficients (copies).
  std::vector<Point> negs;
  negs.reserve(pts_s.size());
  std::vector<const EC_POINT*> sp;
  std::vector<uint64_t> mags;
  sp.reserve(pts_s.size());
  mags.reserve(pts_s.size());
  int bits = 0;
  for (size_t i = 0; i < pts_s.size(); ++i) {
    const int64_t v = vs[i];
    if (v == 0) continue;
    const uint64_t mag = v < 0 ? uint64_t(0) - uint64_t(v) : uint64_t(v);
    if (v < 0) {
      negs.push_back(pts_s[i]->neg());
      sp.push_back(negs.back().p);
    } else {
      sp.push_back(pts_s[i]->p);
    }
    mags.push_back(mag);
    bits = std::max(bits, bit_length(mag));
  }
  // Route the small terms through the interleaved path too when that is cheaper.
  std::vector<Scalar> moved;
  if (!sp.empty() && interleaved_cost(pts_b.size() + sp.size()) <
                         interleaved_cost(pts_b.size()) + pippenger_cost(sp.size(), bits)) {
    moved.reserve(sp.size());
    for (size_t i = 0; i < sp.size(); ++i) {
      moved.push_back(Scalar::from_i64(int64_t(mags[i])));  // point already negated
      pts_b.push_back(sp[i]);
      ks_b.push_back(moved.back().bn);
    }
    sp.clear();
    mags.clear();
  }
  Point r;
  pippenger_small(r.p, sp, mags, bits);
  if (!pts_b.empty()) {
    Point big;
    LRCN_DEPRECATED_BEGIN
    ossl_check(EC_POINTs_mul(G.group(), big.p, nullptr, pts_b.size(), pts_b.data(), ks_b.data(),
                             bn_ctx()),
               "EC_POINTs_mul");
    LRCN_DEPRECATED_END
    if (r.is_identity())
      r = std::move(big);
    else
      ossl_check(EC_POINT_add(G.group(), r.p, r.p, big.p, bn_ctx()), "EC_POINT_add");
  }
  return r;
}

constexpr int kSmallBits = 40;  // |k| < 2^40 counts as a small coefficient

}  // namespace detail

inline Point Point::multi_mul(const std::vector<const Point*>& pts,
                              const std::vector<const Scalar*>& ks) {
  if (pts.size() != ks.size()) throw std::invalid_argument("Point::multi_mul: size mismatch");
  const BIGNUM* q = EcGroup::get().order();
  std::vector<const Point*> pts_s;
  std::vector<int64_t> vs;
  std::vector<const EC_POINT*> pts_b;
  std::vector<const BIGNUM*> ks_b;
  BN_CTX* ctx = bn_ctx();
  std::optional<BnCtxFrame> frame(std::in_place, ctx);
  BIGNUM* t = BN_CTX_get(ctx);
  if (!t) ossl_fail("BN_CTX_get");
  auto small_value = [&](const BIGNUM* b, int64_t& out) -> bool {
    if (BN_num_bits(b) <= detail::kSmallBits) {
      out = int64_t(BN_get_word(b));
      return true;
    }
    if (!BN_sub(t, q, b)) ossl_fail("BN_sub");
    if (BN_num_bits(t) <= detail::kSmallBits) {
      out = -int64_t(BN_get_word(t));
      return true;
    }
    return false;
  };
  for (size_t i = 0; i < pts.size(); ++i) {
    int64_t v;
    if (small_value(ks[i]->bn, v)) {
      pts_s.push_back(pts[i]);
      vs.push_back(v);
    } else {
      pts_b.push_back(pts[i]->p);
      ks_b.push_back(ks[i]->bn);
    }
  }
  frame.reset();  // closes the frame before multi_mul_split uses the same BN_CTX
  return detail::multi_mul_split(pts_s, vs, pts_b, ks_b);
}

inline Point Point::multi_mul_i64(const std::vector<const Point*>& pts,
                                  const std::vector<int64_t>& vs) {
  if (pts.size() != vs.size()) throw std::invalid_argument("Point::multi_mul_i64: size mismatch");
  std::vector<const Point*> pts_s;
  std::vector<int64_t> vs_s;
  std::vector<const EC_POINT*> pts_b;
  std::vector<const BIGNUM*> ks_b;
  std::vector<Scalar> big;
  big.reserve(vs.size());
  for (size_t i = 0; i < pts.size(); ++i) {
    const uint64_t mag = vs[i] < 0 ? uint64_t(0) - uint64_t(vs[i]) : uint64_t(vs[i]);
    if (detail::bit_length(mag) <= detail::kSmallBits) {
      pts_s.push_back(pts[i]);
      vs_s.push_back(vs[i]);
    } else {
      big.push_back(Scalar::from_i64(vs[i]));
      pts_b.push_back(pts[i]->p);
      ks_b.push_back(big.back().bn);
    }
  }
  return detail::multi_mul_split(pts_s, vs_s, pts_b, ks_b);
}

}  // namespace lrcn

#endif  // LRCN_EC_GROUP_H
