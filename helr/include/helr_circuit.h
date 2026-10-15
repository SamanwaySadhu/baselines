// helr_circuit.h -- HELR's homomorphic gradient descent (Kim et al. 2018, Alg. 2 and the
// optimizations of Sec. 3.2), adapted to this baseline's setting:
//   * the client's initial weights are ciphertexts too (HELR starts from beta = 0), so every
//     iteration is the general one;
//   * every iteration trains on its own mini-batch of B samples (the paper uses the whole
//     training set each time). All T*B samples are packed once, sample (t, i) in slot t*B+i
//     of one ciphertext per feature; iteration t multiplies its terms by a public 0/constant
//     mask, which folds into the constant HELR multiplies by anyway, so it costs no depth.
//
// Data layout (HELR's "feature-wise packing"): ciphertext z_j holds p * z_ij / R for every
// sample in its slots (z_i = (2 y_i - 1) x_i, label sign folded in); ciphertext beta_j holds
// p * beta_j replicated in every slot. p = 2^logp.
//
// Per iteration t (degree 3; each line's modulus drop in bits):
//   w      = sum_j z_j * beta_j                                   [z^T beta / R, slotwise]   p
//   w2c    = w^2 + c1/c3                                                                      2p
//   A_j    = ((gamma c3 mask_t) * z_j) * w * w2c  = gamma mask_t (z_j/R) H(w)                  3p
//   B_j    = (gamma mask_t) * z_j                = gamma mask_t (z_j/R)
//   beta_j += AllSum(B_j - A_j)          (AllSum: log2(slots) rotations)
// where H(w) = c1 w + c3 w^3 is the least-squares polynomial (helr_reference.h) and
// gamma = R / (2K): the update beta += (1/K) sum_i (1 - H(w_i))/2 * z_i. Degree 7 uses
// HELR's evaluation order (w^2, w^4, then w * w^2 * (w^4 + c5/c7 w^2 + c3/c7)), 4p per iteration.
// (HELR's published degree-7 code drops the c7 multiplications and adds ciphertexts of
// different moduli; this is the corrected evaluation.)
//
// The server runs this on ciphertexts only; it needs the multiplication key and the
// rotation keys 2^0 .. 2^(logn-1) of the client's key set.
#ifndef LRHE_HELR_CIRCUIT_H
#define LRHE_HELR_CIRCUIT_H

#include "he_core.h"
#include "helr_reference.h"
#include "metrics.h"

#include <chrono>
#include <functional>
#include <map>

namespace lrhe {

// ------------------------------------------------------------ client: encryption
struct EncInputs {
  std::vector<Cipher> z;     // d columns, fresh (modulus 2^logq)
  std::vector<Cipher> beta;  // d initial weights
};

inline std::vector<NTL::ZZ> pack_z_column(const HeConfig& C, const Params& P, const Dataset& D, int j) {
  std::vector<NTL::ZZ> v((size_t)C.nslots, NTL::ZZ(0));
  for (int t = 0; t < P.iters; ++t)
    for (int i = 0; i < P.batch; ++i)
      v[(size_t)t * P.batch + i] = scale_up(z_value(P, D, t, i, j) / C.R(), C.logp);
  return v;
}

inline void encrypt_inputs(HeContext& ctx, const Params& P, const Dataset& D, const std::vector<int64_t>& w0q,
                           int threads, EncInputs& out) {
  const HeConfig& C = ctx.cfg;
  out.z.assign((size_t)C.d, Cipher());
  out.beta.assign((size_t)C.d, Cipher());
  parallel_for(2 * C.d, threads, [&](int k) {
    if (k < C.d) {
      out.z[(size_t)k] = encrypt_scaled(ctx, pack_z_column(C, P, D, k));
    } else {
      const int j = k - C.d;
      const NTL::ZZ w = scale_up((double)w0q[(size_t)j] / std::ldexp(1.0, P.f), C.logp);
      out.beta[(size_t)j] = encrypt_scaled(ctx, std::vector<NTL::ZZ>((size_t)C.nslots, w));
    }
  });
}

// ------------------------------------------------------------- server: evaluation
// Where the evaluation time goes: thread-seconds per operation class (summed over the
// threads, so they exceed the wall time when threads > 1) and operation counts.
struct EvalStats {
  std::map<std::string, double> thread_s;
  std::map<std::string, uint64_t> ops;
  std::vector<double> iter_wall_s;
  std::mutex mu;
  void add(const std::string& k, double s, uint64_t n = 1) {
    std::lock_guard<std::mutex> g(mu);
    thread_s[k] += s;
    ops[k] += n;
  }
};

namespace detail {
struct OpTimer {
  EvalStats* st;
  const char* k;
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  OpTimer(EvalStats* s, const char* key) : st(s), k(key) {}
  ~OpTimer() {
    if (st) st->add(k, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
};
}  // namespace detail

class Evaluator {
 public:
  Evaluator(HeContext& ctx, const HPoly& H, EvalStats* stats = nullptr) : ctx_(ctx), sc_(ctx.scheme), cfg_(ctx.cfg), H_(H), st_(stats) {
    if (H.deg != cfg_.deg) throw std::invalid_argument("polynomial degree does not match the configuration");
  }

  // beta (d ciphertexts at the top modulus) -> final weights (d ciphertexts at out_bits).
  std::vector<Cipher> run(std::vector<Cipher>& z, std::vector<Cipher> beta, int threads) {
    for (int t = 0; t < cfg_.T; ++t) {
      const auto t0 = std::chrono::steady_clock::now();
      iteration(t, z, beta, threads);
      if (st_) {
        std::lock_guard<std::mutex> g(st_->mu);
        st_->iter_wall_s.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      }
      if (progress) progress(t, st_ ? st_->iter_wall_s.back() : 0.0);
    }
    return beta;
  }

  std::function<void(int, double)> progress;  // called after every iteration

 private:
  HeContext& ctx_;
  Scheme& sc_;
  const HeConfig& cfg_;
  const HPoly& H_;
  EvalStats* st_;

  // ---- thin wrappers: operation timing, level changes on copies
  Cipher embed(const Cipher& c, long bits) {
    detail::OpTimer ot(st_, "embed/copy");
    Cipher r = c;
    if (bits > 0) sc_.modEmbedAndEqual(r, bits);
    return r;
  }
  void mult(Cipher& a, Cipher& b) { detail::OpTimer ot(st_, "mult"); sc_.multAndEqual(a, b); }
  Cipher square(Cipher& a) { detail::OpTimer ot(st_, "square"); return sc_.square(a); }
  Cipher mult_poly(Cipher& a, Message& m) { detail::OpTimer ot(st_, "mult_by_plaintext"); return sc_.multByPoly(a, m); }
  Cipher mult_const(Cipher& a, NTL::ZZ& c) { detail::OpTimer ot(st_, "mult_by_const"); return sc_.multByConst(a, c); }
  void rescale(Cipher& a) { detail::OpTimer ot(st_, "rescale"); sc_.modSwitchAndEqual(a, cfg_.logp); }
  void add_const(Cipher& a, double v) {
    detail::OpTimer ot(st_, "add");
    NTL::ZZ c = scale_up(v, cfg_.logp);
    c %= a.mod;
    sc_.addConstAndEqual(a, c);
  }
  void add(Cipher& a, Cipher& b) { detail::OpTimer ot(st_, "add"); sc_.addAndEqual(a, b); }
  void sub(Cipher& a, Cipher& b) { detail::OpTimer ot(st_, "add"); sc_.subAndEqual(a, b); }

  // sum over all slots, in every slot (HELR's AllSum, Alg. 1)
  void all_sum(Cipher& c) {
    for (int r = 0; r < cfg_.logn; ++r) {
      Cipher rot;
      {
        detail::OpTimer ot(st_, "rotate");
        rot = sc_.leftRotateByPo2(c, r);
      }
      add(c, rot);
    }
  }

  Message mask_poly(int t, double value) {
    std::vector<NTL::ZZ> v((size_t)cfg_.nslots, NTL::ZZ(0));
    const NTL::ZZ s = scale_up(value, cfg_.logp);
    for (int i = 0; i < cfg_.B; ++i) v[(size_t)t * cfg_.B + i] = s;
    return encode_scaled(ctx_, v);
  }

  void iteration(int t, std::vector<Cipher>& z, std::vector<Cipher>& beta, int threads) {
    const int d = cfg_.d, p = cfg_.logp;
    const long tl = beta[0].cbits;
    const double gamma = H_.R / (2.0 * std::ldexp(1.0, cfg_.k_shift));  // R / (2K)
    const double *c = H_.c;

    // 1. w = sum_j z_j * beta_j   (level tl - p); accumulated as the products finish, so only
    //    O(threads) products are alive at once
    Cipher w;
    bool have_w = false;
    std::mutex wmu;
    parallel_for(d, threads, [&](int j) {
      Cipher a = embed(z[(size_t)j], z[(size_t)j].cbits - tl);
      mult(a, beta[(size_t)j]);
      std::lock_guard<std::mutex> g(wmu);
      if (!have_w) { w = std::move(a); have_w = true; }
      else add(w, a);
    });
    rescale(w);

    if (cfg_.deg == 3) {
      // 2. w2c = w^2 + c1/c3   (level tl - 2p)
      Cipher w2c = square(w);
      rescale(w2c);
      add_const(w2c, c[1] / c[3]);
      Message mC = mask_poly(t, gamma * c[3]);
      Message mG = mask_poly(t, gamma);
      // 3. per feature
      parallel_for(d, threads, [&](int j) {
        Cipher zl = embed(z[(size_t)j], z[(size_t)j].cbits - tl);
        Cipher A = mult_poly(zl, mC);
        rescale(A);        // tl - p
        mult(A, w);
        rescale(A);        // tl - 2p
        mult(A, w2c);
        rescale(A);        // tl - 3p
        Cipher Bc = mult_poly(zl, mG);
        shrink(z[(size_t)j], zl, t);
        rescale(Bc);
        sc_.modEmbedAndEqual(Bc, 2 * p);
        sub(Bc, A);        // gamma mask (z/R) (1 - H(w))
        update(beta[(size_t)j], Bc, 3 * p);
      });
    } else {
      // 2. w2 = w^2 (tl - 2p), w4 = w2^2 (tl - 3p), u = w4 + (c5/c7) w2 + c3/c7 (tl - 3p)
      Cipher w2 = square(w);
      rescale(w2);
      Cipher u = square(w2);
      rescale(u);
      {
        NTL::ZZ k5 = scale_up(c[5] / c[7], p);
        k5 %= w2.mod;
        Cipher t5 = mult_const(w2, k5);
        rescale(t5);
        add(u, t5);
        add_const(u, c[3] / c[7]);
      }
      Message m7 = mask_poly(t, gamma * c[7]);
      Message mG = mask_poly(t, gamma);
      parallel_for(d, threads, [&](int j) {
        Cipher zl = embed(z[(size_t)j], z[(size_t)j].cbits - tl);
        Cipher t1 = mult_poly(zl, m7);
        rescale(t1);
        mult(t1, w);
        rescale(t1);       // tl - 2p : gamma c7 mask (z/R) w
        Cipher t4;
        {
          NTL::ZZ k1 = scale_up(c[1] / c[7], p);
          k1 %= t1.mod;
          t4 = mult_const(t1, k1);
          rescale(t4);     // tl - 3p : gamma c1 mask (z/R) w
          sc_.modEmbedAndEqual(t4, p);
        }
        mult(t1, w2);
        rescale(t1);       // tl - 3p : gamma c7 mask (z/R) w^3
        mult(t1, u);
        rescale(t1);       // tl - 4p
        add(t1, t4);       // gamma mask (z/R) H(w)
        Cipher Bc = mult_poly(zl, mG);
        shrink(z[(size_t)j], zl, t);
        rescale(Bc);
        sc_.modEmbedAndEqual(Bc, 3 * p);
        sub(Bc, t1);
        update(beta[(size_t)j], Bc, 4 * p);
      });
    }
  }

  // The data column is only ever used at the level of the current weights, which drops by
  // lev_iter bits per iteration: keep a compact copy at the next iteration's level.
  void shrink(Cipher& dst, Cipher& zl, int t) {
    if (t + 1 < cfg_.T) {
      sc_.modEmbedAndEqual(zl, cfg_.lev_iter);
      Cipher compact(zl);
      dst = std::move(compact);
    }
    zl = Cipher();
  }

  // beta_j += AllSum(res); beta drops `drop` bits of modulus to match res.
  void update(Cipher& beta, Cipher& res, long drop) {
    all_sum(res);
    {
      detail::OpTimer ot(st_, "embed/copy");
      sc_.modEmbedAndEqual(beta, drop);
    }
    add(beta, res);
    Cipher compact(beta);  // NTL keeps the capacity of the larger modulus otherwise
    beta = std::move(compact);
  }
};

}  // namespace lrhe

#endif  // LRHE_HELR_CIRCUIT_H
