// lr_cryptonn_selftest -- offline tests of the CryptoNN baseline (no network).
//
//   [1] P-256 wrappers: scalar arithmetic mod q, point laws, multi-scalar multiplication
//       (bucket and interleaved routes) against a naive sum
//   [2] serialization: points (compressed, identity, invalid encodings), FEIP / FEBO
//       ciphertexts and keys, the upload header (parameter mismatch detection)
//   [3] FEIP (Abdalla et al.): decryption = g^<x,y> for eta 1..64, signed x / y, public-
//       key and master-key encryption, recovered through BSGS; wrong keys do not decrypt
//   [4] FEBO (CryptoNN Sec. III-B): '+', '-', '*', '/' with signed operands, and the
//       protocol's case (label in {0, MAXQ}, yhat in [0, MAXQ], '-') exhaustively
//   [5] BSGS: table sizes 1..16 bits, bounds around the table / stride edges, values just
//       outside the bound rejected, thread-count independence, concurrent solves
//   [6] the whole CryptoNN training in-process (CnClient / CnServer from
//       cryptonn_protocol.h, the code lr_cryptonn runs over TCP; the plaintext messages
//       go through the same int32 encoding) == reference_train bit for bit, after every
//       iteration: d in {2,5,8,17}, iters {3,20}, seeds 1..3, and zero-init, --no-bias,
//       qbits 4 / 8, batch 1 / 8, lr 0.25 / 1, master-key encryption, threads 1 vs 4
//   [7] dlog bound safety: the exact bounds never fail and are attained (worst-case data
//       hits |v| = bound); a capped bound and a wrong function key both raise DlogError
//       (never a silently wrong value)
//
// usage: lr_cryptonn_selftest [--quick]          (exit 0 iff every check passes)
#include "bsgs.h"
#include "cryptonn_protocol.h"
#include "ec_group.h"
#include "fe_bo.h"
#include "fe_ip.h"
#include "lr_params.h"
#include "lr_reference.h"
#include "metrics.h"

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace lrcn;

namespace {

int g_fail = 0, g_pass = 0;
#define CHECK(c)                                                                      \
  do {                                                                                \
    if (c) ++g_pass;                                                                  \
    else { ++g_fail; std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); }     \
  } while (0)

std::mt19937_64 rng(12345);
int64_t rnd(int64_t lo, int64_t hi) { return std::uniform_int_distribution<int64_t>(lo, hi)(rng); }

Point naive_multi(const std::vector<Point>& P, const std::vector<Scalar>& k) {
  Point r;
  for (size_t i = 0; i < P.size(); ++i) r = r.add(P[i].mul(k[i]));
  return r;
}

template <class E, class F>
bool throws(F&& f) {
  try { f(); } catch (const E&) { return true; } catch (...) { return false; }
  return false;
}

// One BSGS table per size, shared by every test (built with 4 threads).
const Dlog& table(int bits) {
  static std::map<int, std::unique_ptr<Dlog>> cache;
  auto& p = cache[bits];
  if (!p) p = std::make_unique<Dlog>(bits, 4);
  return *p;
}

// ------------------------------------------------------------- [1] group
void test_group(bool quick) {
  Scalar a = Scalar::from_i64(-5), b = Scalar::from_i64(5);
  CHECK(a.add(b).is_zero());
  CHECK(b.neg() == a);
  CHECK(Scalar::from_i64(-7).mul(Scalar::from_i64(3)) == Scalar::from_i64(-21));
  CHECK(Scalar::from_i64(3).sub(Scalar::from_i64(10)) == Scalar::from_i64(-7));
  CHECK(Scalar::from_i64(INT64_MIN).add(Scalar::from_i64(INT64_MAX)) == Scalar::from_i64(-1));
  for (int i = 0; i < 20; ++i) {
    Scalar r = Scalar::random();
    CHECK(!r.is_zero());
    CHECK(r.mul(r.inv()) == Scalar::from_i64(1));
    CHECK(Scalar::from_bytes32(r.to_bytes32().data()) == r);
  }
  uint8_t ff[32];
  std::memset(ff, 0xff, 32);
  CHECK(throws<std::invalid_argument>([&] { (void)Scalar::from_bytes32(ff); }));
  CHECK(throws<std::domain_error>([&] { (void)Scalar::from_i64(0).inv(); }));

  for (int i = 0; i < 10; ++i) {
    Scalar x = Scalar::random(), y = Scalar::random();
    CHECK(Point::gen_mul(x.add(y)) == Point::gen_mul(x).add(Point::gen_mul(y)));
    CHECK(Point::gen_mul(x.sub(y)) == Point::gen_mul(x).sub(Point::gen_mul(y)));
    CHECK(Point::gen_mul(x).mul(y) == Point::gen_mul(x.mul(y)));
    Point P = Point::gen_mul(Scalar::random());
    CHECK(Point::gen_mul_add(x, P, y) == Point::gen_mul(x).add(P.mul(y)));
    CHECK(P.add(P.neg()).is_identity());
  }
  CHECK(Point::gen_mul_i64(0).is_identity());
  CHECK(Point::gen_mul_i64(-3) == Point::gen_mul_i64(3).neg());

  CHECK(Point::multi_mul({}, {}).is_identity());
  std::vector<int> ns = quick ? std::vector<int>{1, 5, 33} : std::vector<int>{1, 2, 3, 5, 8, 17, 32, 65, 100};
  for (int n : ns)
    for (int kind = 0; kind < 4; ++kind) {
      std::vector<Point> pts;
      std::vector<Scalar> ks;
      std::vector<int64_t> vi;
      for (int i = 0; i < n; ++i) {
        pts.push_back(Point::gen_mul(Scalar::random()));
        int64_t v = 0;
        if (kind == 0) ks.push_back(Scalar::random());
        else if (kind == 1) { v = rnd(-40, 40); ks.push_back(Scalar::from_i64(v)); }
        else if (kind == 2) { v = rnd(-(int64_t(1) << 39), int64_t(1) << 39); ks.push_back(Scalar::from_i64(v)); }
        else if (i % 3 == 0) ks.push_back(Scalar::random());
        else { v = rnd(-3000, 3000); ks.push_back(Scalar::from_i64(v)); }
        vi.push_back(v);
      }
      std::vector<const Point*> pp;
      std::vector<const Scalar*> kp;
      for (int i = 0; i < n; ++i) pp.push_back(&pts[(size_t)i]), kp.push_back(&ks[(size_t)i]);
      const Point expect = naive_multi(pts, ks);
      CHECK(Point::multi_mul(pp, kp) == expect);
      if (kind == 1 || kind == 2) CHECK(Point::multi_mul_i64(pp, vi) == expect);
    }
  // repeated point, cancelling coefficients
  Point A = Point::gen_mul_i64(123456789);
  std::vector<Point> rep(6, A);
  std::vector<const Point*> rp;
  for (auto& p : rep) rp.push_back(&p);
  CHECK(Point::multi_mul_i64(rp, {5, 5, -5, 7, 1, -13}).is_identity());
  CHECK(Point::multi_mul_i64(rp, {1, 1, 1, 1, 1, 1}) == A.mul(Scalar::from_i64(6)));
}

// ----------------------------------------------------- [2] serialization
void test_serialization() {
  for (int i = 0; i < 20; ++i) {
    Point P = Point::gen_mul(Scalar::random());
    auto b = P.to_bytes();
    CHECK(b[0] == 2 || b[0] == 3);
    CHECK(Point::from_bytes(b.data()) == P);
  }
  auto z = Point::identity().to_bytes();
  CHECK(!Point::try_from_bytes(z.data()).has_value());
  CHECK(Point::try_from_bytes(z.data(), true).has_value());
  auto b = Point::gen_mul_i64(5).to_bytes();
  for (uint8_t pre : {0x00, 0x01, 0x04, 0x05, 0xff}) {
    auto c = b;
    c[0] = pre;
    CHECK(!Point::try_from_bytes(c.data()).has_value());
  }
  std::array<uint8_t, 33> big;
  big[0] = 2;
  std::memset(big.data() + 1, 0xff, 32);
  CHECK(!Point::try_from_bytes(big.data()).has_value());
  int rej = 0, acc = 0;  // random x: about half are not on the curve
  for (int i = 0; i < 100; ++i) {
    std::array<uint8_t, 33> c;
    c[0] = uint8_t(2 + (i & 1));
    for (int j = 1; j < 33; ++j) c[(size_t)j] = uint8_t(rng());
    c[1] &= 0x7f;
    (Point::try_from_bytes(c.data()) ? acc : rej)++;
  }
  CHECK(rej > 20 && acc > 20);

  auto [mpk, msk] = feip::setup(7);
  const std::vector<int64_t> x = {1, -2, 3, -4, 31, -32, 0};
  const feip::Ct ct = feip::encrypt(mpk, x);
  auto bytes = feip::serialize(ct);
  CHECK(bytes.size() == feip::ct_bytes(7) && feip::ct_bytes(7) == 8 * 33);
  const feip::Ct ct2 = feip::deserialize(bytes.data(), 7);
  CHECK(ct2.ct0 == ct.ct0);
  for (int i = 0; i < 7; ++i) CHECK(ct2.ct[(size_t)i] == ct.ct[(size_t)i]);
  bytes[33 * 3] = 0x07;
  CHECK(throws<std::invalid_argument>([&] { (void)feip::deserialize(bytes.data(), 7); }));
  uint8_t kb[32];
  const Scalar sk = feip::key_derive(msk, x);
  feip::serialize_key(sk, kb);
  CHECK(feip::deserialize_key(kb) == sk);

  auto [bmpk, bmsk] = febo::setup();
  const febo::Ct bc = febo::encrypt(bmpk, -17);
  uint8_t bb[febo::kCtBytes];
  febo::serialize(bc, bb);
  const febo::Ct bc2 = febo::deserialize(bb);
  CHECK(bc2.cmt == bc.cmt && bc2.ct == bc.ct);
  const Point bk = febo::key_derive(bmsk, bc.cmt, febo::Op::Sub, 9);
  uint8_t kk[febo::kKeyBytes];
  febo::serialize_key(bk, kk);
  CHECK(febo::decrypt_to_group(bc2, febo::deserialize_key(kk), febo::Op::Sub, 9) == Point::gen_mul_i64(-26));

  // upload header: equal parameters pass, every differing field is reported
  Params P;
  P.finalize();
  uint8_t h[kHeaderBytes];
  encode_header(P, h);
  CHECK(!throws<std::runtime_error>([&] { check_header(h, P); }));
  for (int k = 0; k < 6; ++k) {
    Params Q = P;
    if (k == 0) Q.d = 6;
    if (k == 1) Q.iters = 21;
    if (k == 2) Q.batch = 8;
    if (k == 3) Q.qbits = 7;
    if (k == 4) Q.lr = 0.25;
    if (k == 5) Q.bias = false;
    Q.finalize();
    CHECK(throws<std::runtime_error>([&] { check_header(h, Q); }));
  }
  h[0] ^= 1;
  CHECK(throws<std::runtime_error>([&] { check_header(h, P); }));
  // int32 wire values
  uint8_t w4[4];
  put_i32(w4, -123456);
  CHECK(get_i32(w4) == -123456);
  CHECK(throws<std::runtime_error>([&] { put_i32(w4, int64_t(1) << 31); }));
}

// --------------------------------------------------------------- [3] FEIP
void test_feip(bool quick) {
  const Dlog& dl = table(16);
  int via_bsgs = 0;
  const int max_eta = quick ? 12 : 64;
  for (int eta = 1; eta <= max_eta; ++eta) {
    auto [mpk, msk] = feip::setup(eta);
    CHECK((int)mpk.h.size() == eta && (int)msk.s.size() == eta);
    for (int r = 0; r < 3; ++r) {
      std::vector<int64_t> x((size_t)eta), y((size_t)eta);
      const int64_t X = r == 0 ? 32 : 1000, Y = r == 2 ? 1 << 20 : 64;
      int64_t ip = 0, bound = 0;
      for (int i = 0; i < eta; ++i) {
        x[(size_t)i] = rnd(-X, X), y[(size_t)i] = rnd(-Y, Y);
        ip += x[(size_t)i] * y[(size_t)i], bound += X * std::abs(y[(size_t)i]);
      }
      const feip::Ct ct = r == 1 ? feip::encrypt_with_msk(msk, x) : feip::encrypt(mpk, x);
      const Point g = feip::decrypt_to_group(ct, feip::key_derive(msk, y), y);
      CHECK(g == Point::gen_mul_i64(ip));
      if (bound < (int64_t(1) << 26)) {
        const auto v = dl.solve(g, bound);
        CHECK(v && *v == ip);
        ++via_bsgs;
      }
      std::vector<int64_t> y2 = y;
      y2[0] += 1;  // key for another vector
      CHECK(feip::decrypt_to_group(ct, feip::key_derive(msk, y2), y) != g || x[0] == 0);
    }
  }
  auto [mpk, msk] = feip::setup(5);
  const std::vector<int64_t> x = {-32, 31, -1, 0, 7}, y0(5, 0), yn = {-1, -2, -3, -4, -5};
  const feip::Ct ct = feip::encrypt(mpk, x);
  CHECK(feip::decrypt_to_group(ct, feip::key_derive(msk, y0), y0).is_identity());
  CHECK(dl.solve(feip::decrypt_to_group(ct, feip::key_derive(msk, yn), yn), 1000) == 32 - 62 + 3 - 35);
  std::printf("  FEIP: eta 1..%d, %d decryptions also recovered through BSGS\n", max_eta, via_bsgs);
}

// --------------------------------------------------------------- [4] FEBO
void test_febo(bool quick) {
  using febo::Op;
  const Dlog& dl = table(16);
  auto [mpk, msk] = febo::setup();
  for (int r = 0; r < (quick ? 15 : 60); ++r) {
    const int64_t x = rnd(-1000, 1000), y = rnd(-1000, 1000);
    const febo::Ct c = febo::encrypt(mpk, x);
    auto dec = [&](Op op, int64_t yy) { return febo::decrypt_to_group(c, febo::key_derive(msk, c.cmt, op, yy), op, yy); };
    CHECK(dl.solve(dec(Op::Add, y), 2000) == x + y);
    CHECK(dl.solve(dec(Op::Sub, y), 2000) == x - y);
    CHECK(dl.solve(dec(Op::Mul, y), 1000000) == x * y);
    const int64_t dv = rnd(1, 40) * (rnd(0, 1) ? 1 : -1), xd = dv * rnd(-50, 50);
    const febo::Ct cd = febo::encrypt(mpk, xd);
    CHECK(dl.solve(febo::decrypt_to_group(cd, febo::key_derive(msk, cd.cmt, Op::Div, dv), Op::Div, dv), 100) == xd / dv);
    CHECK(febo::decrypt_to_group(c, febo::key_derive(msk, c.cmt, Op::Sub, y + 1), Op::Sub, y) !=
          Point::gen_mul_i64(x - y));
  }
  // The protocol's use: label value in {0, MAXQ}, '-' yhat in [0, MAXQ], bound 2*MAXQ.
  for (int64_t maxq : {7, 31, 127})
    for (int64_t lab : {int64_t(0), maxq})
      for (int64_t yh = 0; yh <= maxq; yh += (quick && maxq > 31 ? 9 : 1)) {
        const febo::Ct c = febo::encrypt(mpk, lab);
        const auto v = dl.solve(febo::decrypt_to_group(c, febo::key_derive(msk, c.cmt, Op::Sub, yh), Op::Sub, yh),
                                2 * maxq);
        CHECK(v && *v == lab - yh);
      }
  CHECK(throws<std::domain_error>([&] { (void)febo::key_derive(msk, mpk.h, Op::Div, 0); }));
  const febo::Ct c = febo::encrypt(mpk, 5);
  CHECK(febo::key_derive(msk, c.cmt, Op::Mul, 0).is_identity());
}

// --------------------------------------------------------------- [5] BSGS
void test_bsgs(bool quick) {
  for (int bits : {1, 2, 4, 8, 12, 16}) {
    if (quick && (bits == 2 || bits == 16)) continue;
    const Dlog& dl = table(bits);
    const int64_t m = dl.half_width();
    CHECK(m == (int64_t(1) << bits) - 1);
    for (int64_t bound : {int64_t(0), int64_t(1), int64_t(5), m / 2, m, m + 1, 2 * m + 1, 3 * m, 10 * m + 7, 100 * m + 3}) {
      if (bound > 2000000) continue;
      std::vector<int64_t> vs = {0, bound, -bound, bound / 2, -bound / 3};
      for (int i = 0; i < (quick ? 5 : 20); ++i) vs.push_back(rnd(-bound, bound));
      for (int64_t v : vs) {
        const auto r = dl.solve(Point::gen_mul_i64(v), bound);
        CHECK(r && *r == v);
      }
      for (int64_t v : {bound + 1, -bound - 1, bound + 1 + rnd(0, 3 * m + 5)})
        CHECK(!dl.solve(Point::gen_mul_i64(v), bound).has_value());
    }
    CHECK(!dl.solve(Point::gen_mul(Scalar::random()), 1000).has_value());
  }
  {  // single- and multi-threaded table builds agree
    Dlog a(14, 1), b(14, 6);
    for (int i = 0; i < (quick ? 50 : 300); ++i) {
      const int64_t v = rnd(-200000, 200000);
      const Point P = Point::gen_mul_i64(v);
      CHECK(a.solve(P, 200000) == v && b.solve(P, 200000) == v);
    }
  }
  {  // concurrent solves and FEIP calls on shared objects
    const Dlog& dl = table(12);
    auto [mpk, msk] = feip::setup(8);
    std::atomic<int> bad{0};
    std::vector<std::thread> th;
    for (int t = 0; t < 8; ++t)
      th.emplace_back([&, t] {
        std::mt19937_64 r(1000 + t);
        for (int i = 0; i < (quick ? 40 : 200); ++i) {
          const int64_t v = std::uniform_int_distribution<int64_t>(-300000, 300000)(r);
          if (dl.solve(Point::gen_mul_i64(v), 300000) != v) ++bad;
          std::vector<int64_t> x(8), y(8);
          int64_t ip = 0;
          for (int j = 0; j < 8; ++j) x[j] = int64_t(r() % 64) - 32, y[j] = int64_t(r() % 64) - 32, ip += x[j] * y[j];
          if (dl.solve(feip::decrypt_to_group(feip::encrypt(mpk, x), feip::key_derive(msk, y), y), 8 * 32 * 32) != ip)
            ++bad;
        }
      });
    for (auto& x : th) x.join();
    CHECK(bad == 0);
  }
}

// ---------------------------------------------------- [6] protocol runs
// int32 wire encoding of a plaintext message, as Channel::send_i32s / recv_i32s do.
std::vector<int64_t> wire(const std::vector<int64_t>& v) {
  std::vector<uint8_t> b(4 * v.size());
  for (size_t k = 0; k < v.size(); ++k) put_i32(b.data() + 4 * k, v[k]);
  std::vector<int64_t> out(v.size());
  for (size_t k = 0; k < v.size(); ++k) out[k] = get_i32(b.data() + 4 * k);
  return out;
}

struct RunOut {
  std::vector<std::vector<int64_t>> w_after;  // server weights after each iteration
  size_t upload_bytes = 0;
  DlogStats st[3];
};

// The protocol of lr_cryptonn without the socket: the same CnClient / CnServer calls in
// the same order. corrupt_key_at >= 0 flips one bit of the forward key of that iteration.
RunOut run_inprocess(const Params& P, const Dataset& D, const std::vector<int64_t>& w0, int threads, bool msk_enc,
                     int64_t cap = -1, int corrupt_key_at = -1) {
  CnClient C(P, D, w0, threads, msk_enc);
  C.keygen();
  C.encrypt();
  const Dlog& dl = table(auto_bsgs_bits(P));
  CnServer S(P, dl, threads);
  S.set_test_bound_cap(cap);
  S.parse_upload(C.upload().data(), C.upload().size());
  RunOut out;
  out.upload_bytes = C.upload().size();
  if (S.initial_weights() != w0) throw std::runtime_error("w0 not transferred intact");
  uint8_t key[32];
  std::vector<uint8_t> keys((size_t)P.batch * febo::kKeyBytes);
  for (int t = 0; t < P.iters; ++t) {
    C.fwd_key(wire(S.fwd_request()), key);
    if (t == corrupt_key_at) key[31] ^= 1;
    S.fwd_finish(key, nullptr);
    C.eval_keys(wire(S.yhat()), keys.data());
    S.eval_finish(keys.data(), nullptr);
    C.bwd_key(wire(S.err()), key);
    S.bwd_finish(key, nullptr);
    out.w_after.push_back(S.weights());
  }
  for (int k = 0; k < 3; ++k) out.st[k] = S.stats((CnServer::Kind)k);
  return out;
}

struct Cfg {
  int d = 5, iters = 20, batch = 4, qbits = 6;
  double lr = 0.5;
  bool bias = true, zero_init = false, msk = false;
  uint64_t seed = 1;
  int threads = 1;
  std::string name() const {
    char b[200];
    std::snprintf(b, sizeof b, "d=%d iters=%d batch=%d qbits=%d lr=%g%s%s seed=%llu threads=%d%s", d, iters, batch,
                  qbits, lr, bias ? "" : " no-bias", zero_init ? " zero-init" : "", (unsigned long long)seed, threads,
                  msk ? " enc-msk" : "");
    return b;
  }
  Params params() const {
    Params P;
    P.d = d; P.iters = iters; P.batch = batch; P.qbits = qbits; P.lr = lr; P.bias = bias;
    P.zero_init = zero_init; P.seed = seed;
    P.finalize();
    return P;
  }
};

DlogStats g_stats[3];  // over every protocol run (for [7])

std::vector<int64_t> check_run(const Cfg& c) {
  const Params P = c.params();
  const Dataset D = generate_client_data(P);
  const std::vector<int64_t> w0 = generate_initial_weights(P);
  RefTrace tr;
  const std::vector<int64_t> ref = reference_train(P, D, w0, &tr);
  const double t0 = wall_seconds();
  RunOut r;
  bool ok = true;
  try {
    r = run_inprocess(P, D, w0, c.threads, c.msk);
  } catch (const std::exception& e) {
    std::printf("  FAIL %s: exception %s\n", c.name().c_str(), e.what());
    ++g_fail;
    return {};
  }
  ok = r.w_after == tr.w_after;
  CHECK(r.w_after == tr.w_after);
  CHECK(r.upload_bytes == upload_layout(P).total);
  for (int k = 0; k < 3; ++k) {
    CHECK(r.st[k].max_ratio <= 1.0);
    g_stats[k].calls += r.st[k].calls;
    g_stats[k].max_ratio = std::max(g_stats[k].max_ratio, r.st[k].max_ratio);
    g_stats[k].max_abs = std::max(g_stats[k].max_abs, r.st[k].max_abs);
    g_stats[k].max_bound = std::max(g_stats[k].max_bound, r.st[k].max_bound);
  }
  CHECK(r.st[0].calls == (uint64_t)P.iters * P.batch && r.st[1].calls == (uint64_t)P.iters * P.batch &&
        r.st[2].calls == (uint64_t)P.iters * P.d_data);
  std::printf("  %-66s %s (%.0f ms, upload %zu B, w:", c.name().c_str(), ok ? "== reference" : "DIFFERS",
              1e3 * (wall_seconds() - t0), r.upload_bytes);
  for (size_t j = 0; j < ref.size() && j < 5; ++j) std::printf(" %" PRId64, ref[j]);
  std::printf("%s)\n", ref.size() > 5 ? " ..." : "");
  return r.w_after.empty() ? std::vector<int64_t>{} : r.w_after.back();
}

void test_protocol(bool quick) {
  std::vector<Cfg> cfgs;
  const std::vector<int> ds = quick ? std::vector<int>{2, 5} : std::vector<int>{2, 5, 8, 17};
  const std::vector<int> its = quick ? std::vector<int>{3} : std::vector<int>{3, 20};
  const int nseeds = quick ? 1 : 3;
  for (int d : ds)
    for (int it : its)
      for (int s = 1; s <= nseeds; ++s) {
        Cfg c;
        c.d = d, c.iters = it, c.seed = (uint64_t)s;
        cfgs.push_back(c);
      }
  const std::vector<int> vds = quick ? std::vector<int>{5} : ds;
  const int vit = quick ? 3 : 20;
  for (int d : vds)
    for (int v = 0; v < 9; ++v) {
      Cfg c;
      c.d = d, c.iters = vit;
      switch (v) {
        case 0: c.zero_init = true; break;
        case 1: c.bias = false; break;
        case 2: c.qbits = 4; break;
        case 3: c.qbits = 8; break;
        case 4: c.batch = 1; break;
        case 5: c.batch = 8; break;
        case 6: c.lr = 0.25; break;
        case 7: c.lr = 1; break;
        case 8: c.msk = true; break;
      }
      cfgs.push_back(c);
    }
  for (const Cfg& c : cfgs) check_run(c);
  // threads 1 vs 4: identical trained weights (and both == reference, checked inside)
  for (int d : vds) {
    Cfg c;
    c.d = d, c.iters = vit, c.seed = 7;
    const auto w1 = check_run(c);
    c.threads = 4;
    const auto w4 = check_run(c);
    CHECK(!w1.empty() && w1 == w4);
  }
}

// ------------------------------------------------------ [7] bound safety
void test_bounds(bool quick) {
  const char* kn[3] = {"forward", "FEBO", "backward"};
  for (int k = 0; k < 3; ++k)
    std::printf("  over all runs above: %-8s %6" PRIu64 " dlogs, no failure, largest |v| = %" PRId64
                ", largest bound = %" PRId64 ", max |v|/bound = %.3f\n",
                kn[k], g_stats[k].calls, g_stats[k].max_abs, g_stats[k].max_bound, g_stats[k].max_ratio);
  CHECK(g_stats[0].calls > 0 && g_stats[0].max_ratio <= 1.0 && g_stats[1].max_ratio <= 1.0 &&
        g_stats[2].max_ratio <= 1.0);

  // Worst-case data: every data feature MINQ, every weight MAXQ -> the first forward
  // value is exactly -bound (the bound is attained, not exceeded).
  for (int d : quick ? std::vector<int>{5} : std::vector<int>{2, 5, 17}) {
    Cfg c;
    c.d = d, c.iters = 2;
    const Params P = c.params();
    Dataset D = generate_client_data(P);
    for (int t = 0; t < P.iters; ++t)
      for (int i = 0; i < P.batch; ++i)
        for (int j = 0; j < P.d_data; ++j) D.X(t, i, j) = P.minq;
    const std::vector<int64_t> w0((size_t)P.d, P.maxq);
    RefTrace tr;
    (void)reference_train(P, D, w0, &tr);
    const RunOut r = run_inprocess(P, D, w0, 1, false);
    CHECK(r.w_after == tr.w_after);
    CHECK(r.st[CnServer::FWD].max_ratio == 1.0);
    std::printf("  worst-case data d=%d: forward |v| / bound = %.3f (bound attained), result == reference\n", d,
                r.st[CnServer::FWD].max_ratio);
  }

  // A too small bound must be an error, never a wrong value.
  {
    Cfg c;
    c.d = 5, c.iters = 3;
    const Params P = c.params();
    const Dataset D = generate_client_data(P);
    const std::vector<int64_t> w0 = generate_initial_weights(P);
    std::string msg;
    try {
      (void)run_inprocess(P, D, w0, 1, false, /*cap=*/10);
    } catch (const DlogError& e) {
      msg = e.what();
    }
    CHECK(!msg.empty());
    CHECK(msg.find("bounded discrete log FAILED") != std::string::npos && msg.find("capped") != std::string::npos);
    std::printf("  capped bound (10): DlogError \"%s\"\n", msg.c_str());
    msg.clear();
    try {
      (void)run_inprocess(P, D, w0, 4, false, -1, /*corrupt_key_at=*/1);
    } catch (const DlogError& e) {
      msg = e.what();
    }
    CHECK(!msg.empty() && msg.find("forward") != std::string::npos);
    std::printf("  wrong forward key at iteration 1: DlogError \"%s\"\n", msg.c_str());
  }
  // Protocol misuse is rejected: requests out of order / beyond the last iteration.
  {
    Cfg c;
    c.d = 3, c.iters = 1;
    const Params P = c.params();
    CnClient C(P, generate_client_data(P), generate_initial_weights(P), 1);
    C.keygen();
    C.encrypt();
    uint8_t key[32], keys[4 * 33];
    CHECK(throws<std::runtime_error>([&] { C.eval_keys({0, 0, 0, 0}, keys); }));
    C.fwd_key({1, 2}, key);
    CHECK(throws<std::runtime_error>([&] { C.eval_keys({0, 0, 99, 0}, keys); }));  // yhat > MAXQ
    std::vector<int64_t> yh = {1, 2, 3, 4};
    C.eval_keys(yh, keys);
    CHECK(throws<std::runtime_error>([&] { C.bwd_key({0, 0, 0, 0}, key); }));  // e != yhat - label (labels are random)
    const Dataset D = generate_client_data(P);
    std::vector<int64_t> e(4);
    for (int i = 0; i < 4; ++i) e[(size_t)i] = yh[(size_t)i] - P.maxq * D.Y(0, i);
    C.bwd_key(e, key);
    CHECK(throws<std::runtime_error>([&] { C.fwd_key({1, 2}, key); }));  // after the last iteration
    Params Q = P;
    Q.d = 4;
    Q.finalize();
    CnServer S(Q, table(12), 1);
    CHECK(throws<std::runtime_error>([&] { S.parse_upload(C.upload().data(), C.upload().size()); }));
    Params B = P;
    B.d = 1;
    B.finalize();
    CHECK(throws<std::invalid_argument>([&] { CnServer S2(B, table(12), 1); }));
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--quick")) quick = true;
    else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
      std::printf("usage: %s [--quick]\n", argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "unknown option %s\n", argv[i]);
      return 2;
    }
  }
  const double t0 = wall_seconds();
  try {
    std::printf("[1] P-256 scalars, points, multi-scalar multiplication\n");
    test_group(quick);
    std::printf("[2] serialization (points, FEIP / FEBO ciphertexts and keys, upload header)\n");
    test_serialization();
    std::printf("[3] FEIP\n");
    test_feip(quick);
    std::printf("[4] FEBO ('+', '-', '*', '/'; protocol case exhaustive)\n");
    test_febo(quick);
    std::printf("[5] BSGS bounded discrete log\n");
    test_bsgs(quick);
    std::printf("[6] CryptoNN training in-process == reference_train (bit-exact, every iteration)\n");
    test_protocol(quick);
    std::printf("[7] dlog bound safety and misuse detection\n");
    test_bounds(quick);
  } catch (const std::exception& e) {
    std::printf("FAIL: unexpected exception: %s\n", e.what());
    ++g_fail;
  }
  std::printf("\n%s: %d checks passed, %d failed (%.1f s%s)\n", g_fail ? "SELFTEST FAILED" : "SELFTEST PASSED", g_pass,
              g_fail, wall_seconds() - t0, quick ? ", --quick" : "");
  return g_fail ? 1 : 0;
}
