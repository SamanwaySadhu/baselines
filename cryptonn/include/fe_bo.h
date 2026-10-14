// fe_bo.h -- FEBO: CryptoNN's functional encryption for basic operations (Sec. III-B),
// an ElGamal-like scheme whose function key for (op, y) lets the holder learn x op y.
//
//   Setup:                          s <- Z_q;  mpk = (g, h = g^s);  msk = s
//   Encrypt(mpk, x):                r <- Z_q;  cmt = g^r,  ct = h^r * g^x
//   KeyDerive(msk, cmt, op, y):     '+': cmt^s * g^{-y}     '-': cmt^s * g^{y}
//                                   '*': (cmt^s)^y          '/': (cmt^s)^{y^{-1}}
//   Decrypt(ct, sk, op, y):         '+','-': ct / sk          = g^{x + y}, g^{x - y}
//                                   '*':     ct^y / sk        = g^{x * y}
//                                   '/':     ct^{y^{-1}} / sk = g^{x * y^{-1} mod q}
//
// Additive notation: cmt^s = s*cmt (= r*h), '/' is a point subtraction. '/' is
// meaningful only when y divides x (then g^{x/y} has a small dlog); y = 0 is rejected
// for '/'. The discrete log is NOT taken here (decrypt_to_group returns g^{x op y}).
// The CryptoNN protocol in this baseline uses '-' (secure evaluation of e = yhat - y).
//
// Wire format: ciphertext cmt || ct (2 * kPointBytes = 66 bytes); function key one
// point (kPointBytes = 33 bytes).
#ifndef LRCN_FE_BO_H
#define LRCN_FE_BO_H

#include "ec_group.h"

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace lrcn::febo {

enum class Op { Add, Sub, Mul, Div };

struct Msk {
  Scalar s;
};
struct Mpk {
  Point h;
};
struct Ct {
  Point cmt, ct;
};

inline std::pair<Mpk, Msk> setup() {
  Msk msk{Scalar::random()};
  Mpk mpk{Point::gen_mul(msk.s)};
  return {std::move(mpk), std::move(msk)};
}

inline Ct encrypt(const Mpk& mpk, int64_t x) {
  const Scalar r = Scalar::random();
  Ct c;
  c.cmt = Point::gen_mul(r);
  c.ct = Point::gen_mul_add(Scalar::from_i64(x), mpk.h, r);  // x*g + r*h
  return c;
}

inline Point key_derive(const Msk& msk, const Point& cmt, Op op, int64_t y) {
  switch (op) {
    case Op::Add:  // s*cmt - y*g
      return Point::gen_mul_add(Scalar::from_i64(y).neg(), cmt, msk.s);
    case Op::Sub:  // s*cmt + y*g
      return Point::gen_mul_add(Scalar::from_i64(y), cmt, msk.s);
    case Op::Mul:  // (s*y)*cmt
      return cmt.mul(msk.s.mul(Scalar::from_i64(y)));
    case Op::Div:  // (s*y^{-1})*cmt
      if (y == 0) throw std::domain_error("febo::key_derive: division by zero");
      return cmt.mul(msk.s.mul(Scalar::from_i64(y).inv()));
  }
  throw std::invalid_argument("febo::key_derive: bad op");
}

inline Point decrypt_to_group(const Ct& c, const Point& sk, Op op, int64_t y) {
  switch (op) {
    case Op::Add:
    case Op::Sub:
      return c.ct.sub(sk);
    case Op::Mul:
      return c.ct.mul(Scalar::from_i64(y)).sub(sk);
    case Op::Div:
      if (y == 0) throw std::domain_error("febo::decrypt: division by zero");
      return c.ct.mul(Scalar::from_i64(y).inv()).sub(sk);
  }
  throw std::invalid_argument("febo::decrypt: bad op");
}

// ------------------------------------------------------------ serialization
constexpr size_t kCtBytes = 2 * kPointBytes;
constexpr size_t kKeyBytes = kPointBytes;

inline void serialize(const Ct& c, uint8_t* out) {
  c.cmt.to_bytes(out);
  c.ct.to_bytes(out + kPointBytes);
}
// Validates both points (std::invalid_argument).
inline Ct deserialize(const uint8_t* in) {
  Ct c;
  c.cmt = Point::from_bytes(in);
  c.ct = Point::from_bytes(in + kPointBytes);
  return c;
}
inline void serialize_key(const Point& sk, uint8_t* out) { sk.to_bytes(out); }
// A key may legitimately be the identity (e.g. '*' with y = 0), so it is allowed here.
inline Point deserialize_key(const uint8_t* in) { return Point::from_bytes(in, true); }

}  // namespace lrcn::febo

#endif  // LRCN_FE_BO_H
