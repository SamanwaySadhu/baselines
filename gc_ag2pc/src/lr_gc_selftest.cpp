// lr_gc_selftest -- offline correctness tests and gate counts (no networking).
//
//   1. emp-tool's binary32 add / mul / div are bit-exact against the host's floats.
//   2. The circuit's sigmoid equals sigmoid_f32() bit for bit, and sigmoid_f32() is
//      within a few ulp of the true sigmoid (long double) over the whole range.
//   3. Whole training run (all iterations) over emp::ClearCtx equals the plaintext
//      reference bit for bit, for several (d, iters, seed); distance to double-precision
//      logistic regression is reported.
//   4. AND-gate counts over emp::CountCtx.
//
// usage: lr_gc_selftest [--quick]
#include "lr_circuit.h"
#include "lr_params.h"
#include "lr_reference.h"

#include "emp-tool/emp-tool.h"
#include "emp-tool/ir/context/clear.h"
#include "emp-tool/ir/context/count.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

using namespace lrgc;
using CX = emp::ClearCtx;
using FC = emp::Float_T<CX, 32>;

namespace {
int failures = 0;
void check(bool ok, const char* what) {
  std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what);
  if (!ok) ++failures;
}

float ulp_of(float v) { return std::nextafter(std::fabs(v), INFINITY) - std::fabs(v); }

void test_fp_ops(int N) {
  std::printf("1. binary32 add / sub / mul / div vs host floats (%d random operand pairs each)\n", N);
  CX cx;
  std::mt19937 g(7);
  std::uniform_real_distribution<float> U(-4, 4);
  long bad[4] = {0, 0, 0, 0};
  for (int i = 0; i < N; ++i) {
    const float x = U(g) * (i % 5 == 0 ? 1e-4f : 1.0f), y = U(g);
    const FC X = FC::constant(cx, x), Y = FC::constant(cx, y);
    volatile float a = x + y, s = x - y, m = x * y, d = x / y;
    bad[0] += f2u(clear_float(X + Y)) != f2u(a);
    bad[1] += f2u(clear_float(X - Y)) != f2u(s);
    bad[2] += f2u(clear_float(X * Y)) != f2u(m);
    bad[3] += f2u(clear_float(X / Y)) != f2u(d);
  }
  check(bad[0] + bad[1] + bad[2] + bad[3] == 0, "add, sub, mul, div: 0 mismatches");
}

void test_sigmoid(int N) {
  std::printf("2. sigmoid (%d inputs)\n", N);
  CX cx;
  FloatLR<CX> lr(cx, 2);
  std::mt19937 g(11);
  std::uniform_real_distribution<float> U(-1, 1);
  long mism = 0;
  double worst_rel = 0;  // relative to the true sigmoid, in float32 ulps
  float worst_z = 0;
  for (int i = 0; i < N; ++i) {
    float z;
    switch (i % 4) {
      case 0: z = U(g) * 4; break;
      case 1: z = U(g) * 20; break;
      case 2: z = U(g) * 90; break;
      default: z = U(g) * 1e-3f;
    }
    const float ref = sigmoid_f32(z);
    const float got = clear_float(lr.sigmoid(FC::constant(cx, z)));
    mism += f2u(ref) != f2u(got);
    const long double exact = sigmoid_ld((long double)z);
    if (std::fabs(z) <= 80 && exact > 1e-30L) {  // beyond that the saturation documented in lr_reference.h applies
      const double ulps = (double)(std::fabs((long double)ref - exact) / ulp_of(ref));
      if (ulps > worst_rel) { worst_rel = ulps; worst_z = z; }
    }
  }
  check(mism == 0, "circuit sigmoid == sigmoid_f32() bit for bit");
  std::printf("      worst error vs the true sigmoid: %.2f ulp (at z = %g)\n", worst_rel, (double)worst_z);
  check(worst_rel < 4.0, "sigmoid_f32() within 4 ulp of the true sigmoid");
  check(sigmoid_f32(1e30f) == 1.0f && sigmoid_f32(-1e30f) > 0 && sigmoid_f32(-1e30f) < 2e-38f &&
            sigmoid_f32(0.0f) == 0.5f,
        "saturation: sigmoid(+huge) = 1, sigmoid(-huge) ~ 1.6e-38, sigmoid(0) = 0.5");
}

void test_training(bool quick) {
  std::printf("3. whole training run over ClearCtx vs the plaintext float reference\n");
  struct Cfg { int d, iters; uint64_t seed; };
  std::vector<Cfg> cfgs = {{2, 3, 1}, {5, 20, 1}, {5, 20, 2}, {8, 6, 3}};
  if (!quick) cfgs.push_back({17, 20, 4});
  for (const Cfg& c : cfgs) {
    Params P;
    P.d = c.d; P.iters = c.iters; P.seed = c.seed;
    P.finalize();
    const Dataset D = generate_client_data(P);
    const std::vector<float> w0 = generate_initial_weights(P);
    CX cx;
    const auto w = train_offline(cx, P, D, w0);
    const std::vector<float> ref = reference_train(P, D, w0);
    bool same = true;
    for (int j = 0; j < P.d; ++j) same = same && f2u(clear_float(w[j])) == f2u(ref[j]);
    const std::vector<double> wd = reference_train_double(P, D, std::vector<double>(w0.begin(), w0.end()));
    double worst = 0, scale = 0;
    for (int j = 0; j < P.d; ++j) { worst = std::max(worst, std::fabs((double)ref[j] - wd[j])); scale = std::max(scale, std::fabs(wd[j])); }
    char msg[160];
    std::snprintf(msg, sizeof msg, "d=%-2d iters=%-2d seed=%llu: circuit == float reference (bit-exact); |w_f32 - w_f64| <= %.2e",
                  c.d, c.iters, (unsigned long long)c.seed, worst);
    check(same && worst < 1e-5 * std::max(1.0, scale), msg);
  }
}

void test_counts(bool quick) {
  std::printf("4. AND-gate counts (batch = 4)\n");
  std::printf("      %5s | %14s | %16s\n", "d", "per iteration", "total, 20 iters");
  std::vector<int> ds = {2, 5, 10, 20, 50};
  if (!quick) ds.push_back(100);
  for (int d : ds) {
    Params P;
    P.d = d;
    P.iters = 1;
    const std::vector<float> w0((size_t)d, 0.0f);
    emp::CountCtx c1;
    (void)train_offline(c1, P, generate_client_data(P), w0);
    P.iters = 20;
    emp::CountCtx cn;
    (void)train_offline(cn, P, generate_client_data(P), w0);
    std::printf("      %5d | %14llu | %16llu\n", d, (unsigned long long)c1.ands, (unsigned long long)cn.ands);
  }
}
}  // namespace

int main(int argc, char** argv) {
  const bool quick = argc > 1 && !std::strcmp(argv[1], "--quick");
  test_fp_ops(quick ? 3000 : 20000);
  test_sigmoid(quick ? 20000 : 200000);
  test_training(quick);
  test_counts(quick);
  std::printf("\n%s\n", failures ? "SELFTEST FAILED" : "all selftests passed");
  return failures ? 1 : 0;
}
