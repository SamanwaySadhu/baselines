// fe_ip.h -- FEIP: DDH-based functional encryption for inner products
// (Abdalla, Bourse, De Caro, Pointcheval, PKC 2015), as used by CryptoNN (Sec. II-B).
//
//   Setup(eta):          s <- Z_q^eta;  mpk = (g, h_i = g^{s_i});  msk = s
//   KeyDerive(msk, y):   sk_y = <y, s> mod q
//   Encrypt(mpk, x):     r <- Z_q;  ct0 = g^r,  ct_i = h_i^r * g^{x_i}
//   Decrypt(ct, sk, y):  g^{<x,y>} = prod_i ct_i^{y_i} / ct0^{sk_y}   (then bounded dlog)
//
// In additive notation: ct_i = r*h_i + x_i*g, and decryption is the multi-scalar sum
// sum_i y_i*ct_i - sk_y*ct0. x and y are signed integers mapped to Z_q. The discrete log
// is NOT taken here (decrypt_to_group returns g^{<x,y>}); see bsgs.h.
//
// Encryption: ct0 by the fixed-base g table, each ct_i by one double-scalar
// EC_POINT_mul (x_i*g + r*h_i). encrypt_with_msk() is an equivalent shortcut for a
// party that also holds msk (in this two-party baseline the client is both data owner
// and authority): ct_i = (r*s_i + x_i)*g, one fixed-base multiplication per component,
// same ciphertext distribution.
//
// Decryption: Point::multi_mul -- the y_i are small integers (weights / errors), so
// the eta terms go through the bucket method and the single full-width term -sk*ct0
// through OpenSSL's multiplication (see ec_group.h).
//
// Wire format: ct0 || ct_0 .. ct_{eta-1}, each kPointBytes (compressed);
// ct_bytes(eta) = (eta + 1) * 33. Function keys: kScalarBytes (32).
#ifndef LRCN_FE_IP_H
#define LRCN_FE_IP_H

#include "ec_group.h"

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lrcn::feip {

struct Msk {
  std::vector<Scalar> s;
};
struct Mpk {
  std::vector<Point> h;
};
struct Ct {
  Point ct0;
  std::vector<Point> ct;
};

inline std::pair<Mpk, Msk> setup(int eta) {
  if (eta < 1) throw std::invalid_argument("feip::setup: eta must be >= 1");
  Mpk mpk;
  Msk msk;
  mpk.h.reserve(size_t(eta));
  msk.s.reserve(size_t(eta));
  for (int i = 0; i < eta; ++i) {
    msk.s.push_back(Scalar::random());
    mpk.h.push_back(Point::gen_mul(msk.s.back()));
  }
  return {std::move(mpk), std::move(msk)};
}

inline Ct encrypt(const Mpk& mpk, const std::vector<int64_t>& x) {
  if (x.size() != mpk.h.size()) throw std::invalid_argument("feip::encrypt: |x| != eta");
  const Scalar r = Scalar::random();
  Ct c;
  c.ct0 = Point::gen_mul(r);
  c.ct.reserve(x.size());
  for (size_t i = 0; i < x.size(); ++i)
    c.ct.push_back(Point::gen_mul_add(Scalar::from_i64(x[i]), mpk.h[i], r));
  return c;
}

// Same distribution as encrypt(mpk, x) with mpk = g^msk; needs msk.
inline Ct encrypt_with_msk(const Msk& msk, const std::vector<int64_t>& x) {
  if (x.size() != msk.s.size()) throw std::invalid_argument("feip::encrypt_with_msk: |x| != eta");
  const Scalar r = Scalar::random();
  Ct c;
  c.ct0 = Point::gen_mul(r);
  c.ct.reserve(x.size());
  for (size_t i = 0; i < x.size(); ++i)
    c.ct.push_back(Point::gen_mul(r.mul(msk.s[i]).add(Scalar::from_i64(x[i]))));
  return c;
}

inline Scalar key_derive(const Msk& msk, const std::vector<int64_t>& y) {
  if (y.size() != msk.s.size()) throw std::invalid_argument("feip::key_derive: |y| != eta");
  Scalar acc;
  for (size_t i = 0; i < y.size(); ++i)
    if (y[i] != 0) acc = acc.add(msk.s[i].mul(Scalar::from_i64(y[i])));
  return acc;
}

// g^{<x,y>} = sum_i y_i*ct_i - sk*ct0.
inline Point decrypt_to_group(const Ct& c, const Scalar& sk, const std::vector<int64_t>& y) {
  if (y.size() != c.ct.size()) throw std::invalid_argument("feip::decrypt: |y| != eta");
  std::vector<Scalar> ys;
  ys.reserve(y.size() + 1);
  std::vector<const Point*> pts;
  std::vector<const Scalar*> ks;
  pts.reserve(y.size() + 1);
  ks.reserve(y.size() + 1);
  for (size_t i = 0; i < y.size(); ++i) {
    if (y[i] == 0) continue;
    ys.push_back(Scalar::from_i64(y[i]));
    pts.push_back(&c.ct[i]);
    ks.push_back(&ys.back());
  }
  ys.push_back(sk.neg());
  pts.push_back(&c.ct0);
  ks.push_back(&ys.back());
  return Point::multi_mul(pts, ks);
}

// ------------------------------------------------------------ serialization
constexpr size_t ct_bytes(int eta) { return size_t(eta + 1) * kPointBytes; }

inline void serialize(const Ct& c, uint8_t* out) {
  c.ct0.to_bytes(out);
  for (size_t i = 0; i < c.ct.size(); ++i) c.ct[i].to_bytes(out + (i + 1) * kPointBytes);
}
inline std::vector<uint8_t> serialize(const Ct& c) {
  std::vector<uint8_t> v(ct_bytes(int(c.ct.size())));
  serialize(c, v.data());
  return v;
}
// Validates every point (std::invalid_argument on a malformed / off-curve / identity point).
inline Ct deserialize(const uint8_t* in, int eta) {
  if (eta < 1) throw std::invalid_argument("feip::deserialize: eta must be >= 1");
  Ct c;
  c.ct0 = Point::from_bytes(in);
  c.ct.reserve(size_t(eta));
  for (int i = 0; i < eta; ++i) c.ct.push_back(Point::from_bytes(in + size_t(i + 1) * kPointBytes));
  return c;
}

inline void serialize_key(const Scalar& sk, uint8_t* out) { sk.to_bytes32(out); }
inline Scalar deserialize_key(const uint8_t* in) { return Scalar::from_bytes32(in); }

}  // namespace lrcn::feip

#endif  // LRCN_FE_IP_H
