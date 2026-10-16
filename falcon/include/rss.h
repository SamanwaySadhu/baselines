// rss.h -- replicated secret sharing (Falcon's convention) over the three rings of the protocol, and the
// local operations on shares.
//
// A value x = x_0 + x_1 + x_2 (in Z_2^64, in Z_67, or XOR for bits) is shared among the servers S0, S1, S2
// so that S_i holds (.first, .second) = (x_i, x_{i+1}), indices mod 3. So S0 = (x_0, x_1), S1 = (x_1, x_2),
// S2 = (x_2, x_0): every component is held by two servers, S_i is missing x_{i-1}, and component 0 is held
// by S0 (.first) and S2 (.second). A public constant is added to component 0 only (S0 .first, S2 .second),
// the only party-specific pattern of upstream (tools.h:193-227, F.cpp:704-734 ...).
//
// The 3-out-of-3 product share z_i = x_i y_i + x_{i+1} y_i + x_i y_{i+1} (upstream operator*, tools.h:
// 276-279): the nine terms of z_0 + z_1 + z_2 cover every x_j y_k once, so the z_i sum to x y. Resharing
// z_i into RSS: send z_i (plus a zero-sharing alpha_i) to prev, receive z_{i+1} from next.
//
// Z_67 arithmetic is done in int with % 67 (upstream's addition/multiplicationModPrime lookup tables,
// secondary.cpp:53-59, are not ported: they read out of bounds on non-canonical inputs).
#ifndef FAL_RSS_H
#define FAL_RSS_H

#include <cstdint>
#include <vector>

namespace fal {

using u64 = uint64_t;
using u8 = uint8_t;

template <class T>
struct RSS {
  T first = 0, second = 0;
};
using Rv64 = std::vector<RSS<u64>>;
using Rv8 = std::vector<RSS<u8>>;

inline int next_of(int i) { return (i + 1) % 3; }  // port of nextParty, tools.cpp:545-559
inline int prev_of(int i) { return (i + 2) % 3; }  // port of prevParty, tools.cpp:561-575

// ---- Z_67 (PRIME_NUMBER, globals.h:37); elements are bytes in [0, 66] -----------------------------
constexpr int kP = 67;
inline u8 add67(u8 a, u8 b) { return (u8)(((int)a + (int)b) % kP); }
inline u8 sub67(u8 a, u8 b) { return (u8)(((int)a + kP - (int)b) % kP); }
inline u8 mul67(u8 a, u8 b) { return (u8)(((int)a * (int)b) % kP); }
inline u8 neg67(u8 a) { return (u8)((kP - (int)a) % kP); }

// ---- ring traits: T, +, -, *, canonical-form test, "add 1" (tamper) ------------------------------
struct Ring64 {
  using T = u64;
  static T add(T a, T b) { return a + b; }
  static T sub(T a, T b) { return a - b; }
  static T mul(T a, T b) { return a * b; }
  static bool canonical(T) { return true; }
  static T plus_one(T a) { return a + 1; }
};
struct Field67 {
  using T = u8;
  static T add(T a, T b) { return add67(a, b); }
  static T sub(T a, T b) { return sub67(a, b); }
  static T mul(T a, T b) { return mul67(a, b); }
  static bool canonical(T a) { return a < kP; }
  static T plus_one(T a) { return add67(a, 1); }
};
struct Bits {
  using T = u8;
  static T add(T a, T b) { return a ^ b; }
  static T sub(T a, T b) { return a ^ b; }
  static T mul(T a, T b) { return a & b; }
  static bool canonical(T a) { return a <= 1; }
  static T plus_one(T a) { return a ^ 1; }
};

// ---- local operations ------------------------------------------------------------------------------
template <class Rg>
RSS<typename Rg::T> radd(const RSS<typename Rg::T>& a, const RSS<typename Rg::T>& b) {
  return {Rg::add(a.first, b.first), Rg::add(a.second, b.second)};
}
template <class Rg>
RSS<typename Rg::T> rsub(const RSS<typename Rg::T>& a, const RSS<typename Rg::T>& b) {
  return {Rg::sub(a.first, b.first), Rg::sub(a.second, b.second)};
}
template <class Rg>
RSS<typename Rg::T> rscale(typename Rg::T k, const RSS<typename Rg::T>& a) {  // public * shared (tools.h:282)
  return {Rg::mul(k, a.first), Rg::mul(k, a.second)};
}
// x + k for a public k: component 0 only (S0 .first, S2 .second)
template <class Rg>
void add_const(int me, RSS<typename Rg::T>& x, typename Rg::T k) {
  if (me == 0) x.first = Rg::add(x.first, k);
  else if (me == 2) x.second = Rg::add(x.second, k);
}
// 3-out-of-3 product share (port of operator*, tools.h:276-279)
template <class Rg>
typename Rg::T prod3(const RSS<typename Rg::T>& x, const RSS<typename Rg::T>& y) {
  return Rg::add(Rg::add(Rg::mul(x.first, y.first), Rg::mul(x.second, y.first)), Rg::mul(x.first, y.second));
}

// ---- wrap helpers (port of tools.h:179-240) ---------------------------------------------------------
inline u8 msb(u64 a) { return (u8)(a >> 63); }                  // getMSB, tools.h:179-182
inline u8 wrap_around(u64 a, u64 b) { return (u8)(a > ~b); }    // wrapAround, tools.h:230-231: carry of a+b
// wrap3, tools.h:233-240: parity of floor((a+b+c) / 2^64) (symmetric in its arguments)
inline u8 wrap3(u64 a, u64 b, u64 c) { return wrap_around(a, b) ^ wrap_around(a + b, c); }

}  // namespace fal

#endif  // FAL_RSS_H
