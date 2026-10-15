// fp.h -- the prime field F_p, p = 2^127 - 1 (a Mersenne prime, p = 3 mod 4), on unsigned __int128.
//
// Every secret in the protocol (data, weights, masks, MACs) is an element of F_p. The field is
// large enough for two things at once: (i) the MAC check has soundness error ~2^-127, and
// (ii) a fixed-point value of up to ~2^60 can be hidden by a mask of 40 more random bits without
// the sum wrapping around p (the comparison gadget in online.h relies on this).
#ifndef SML_FP_H
#define SML_FP_H

#include <emp-tool/emp-tool.h>

#include <cstdint>
#include <cstring>

namespace sml {

using u128 = unsigned __int128;
inline constexpr u128 kP = (((u128)1) << 127) - 1;
inline constexpr int kBits = 127;  // bits of an element = OTs per product

struct Fp {
  u128 v = 0;
  constexpr Fp() = default;
  constexpr explicit Fp(u128 x) : v(x) {}
  static Fp from_int(int64_t x) { return x >= 0 ? Fp((u128)(uint64_t)x) : Fp(kP - (u128)(uint64_t)(-(x + 1)) - 1); }
  static Fp from_u128_reduce(u128 x) {  // any 128-bit value -> canonical
    x = (x & kP) + (x >> 127);
    if (x >= kP) x -= kP;
    return Fp(x);
  }
  bool operator==(const Fp& o) const { return v == o.v; }
  bool operator!=(const Fp& o) const { return v != o.v; }
  bool is_zero() const { return v == 0; }
  // Signed value assuming |x| < p/2.
  __int128 to_signed() const { return v > (kP >> 1) ? -(__int128)(kP - v) : (__int128)v; }
};

inline Fp operator+(Fp a, Fp b) {
  u128 s = a.v + b.v;
  if (s >= kP) s -= kP;
  return Fp(s);
}
inline Fp operator-(Fp a, Fp b) { return Fp(a.v >= b.v ? a.v - b.v : a.v + kP - b.v); }
inline Fp operator-(Fp a) { return Fp(a.v == 0 ? 0 : kP - a.v); }
inline Fp& operator+=(Fp& a, Fp b) { return a = a + b; }
inline Fp& operator-=(Fp& a, Fp b) { return a = a - b; }

inline Fp operator*(Fp a, Fp b) {
  const uint64_t a0 = (uint64_t)a.v, a1 = (uint64_t)(a.v >> 64);
  const uint64_t b0 = (uint64_t)b.v, b1 = (uint64_t)(b.v >> 64);
  const u128 p00 = (u128)a0 * b0, p01 = (u128)a0 * b1, p10 = (u128)a1 * b0, p11 = (u128)a1 * b1;
  const u128 mid = p01 + p10;  // < 2^128 because a1, b1 < 2^63
  const u128 lo = p00 + (mid << 64);
  const u128 carry = lo < p00 ? 1 : 0;
  const u128 hi = p11 + (mid >> 64) + carry;  // < 2^126
  // a*b = hi * 2^128 + lo, and 2^128 = 2 (mod p)
  u128 r = (lo & kP) + (lo >> 127) + (hi << 1);
  r = (r & kP) + (r >> 127);
  if (r >= kP) r -= kP;
  return Fp(r);
}
inline Fp& operator*=(Fp& a, Fp b) { return a = a * b; }

inline Fp pow(Fp b, u128 e) {
  Fp r(1);
  while (e) {
    if (e & 1) r = r * b;
    b = b * b;
    e >>= 1;
  }
  return r;
}
inline Fp inv(Fp a) { return pow(a, kP - 2); }
inline Fp sqrt_qr(Fp a) { return pow(a, (kP + 1) / 4); }  // a square: a^((p+1)/4) squared = a
inline Fp pow2(int t) { return Fp(((u128)1) << t); }

inline Fp from_block(const emp::block& b) {
  u128 x;
  std::memcpy(&x, &b, 16);
  return Fp::from_u128_reduce(x);
}
inline bool bit_of(Fp a, int t) { return (a.v >> t) & 1; }

// Uniform field element from a PRG.
inline Fp random_fp(emp::PRG& prg) {
  for (;;) {
    u128 x;
    prg.random_data(&x, 16);
    x &= kP;
    if (x != kP) return Fp(x);  // rejection: uniform on [0, p)
  }
}

}  // namespace sml

#endif  // SML_FP_H
