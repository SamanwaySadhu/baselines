// online.h -- the online (data-dependent) phase among the server (party 0) and the clients (parties 1..batch).
//
// Input. Client k owns datapoint k of every batch (rows r = t*batch + k - 1) and one additive share
// w0_k of the initial weights (w0 = sum_k w0_k). In ONE round it sends every other party, the server
// included,  E = X - U  for its rows (U = its secret mask from preprocessing), w0_k - Rw_k and the
// masked labels y - Ry. All of this is uniformly random to anybody who does not know the masks. The
// shares of w and y are then  w = sum_k ([Rw_k] + (w0_k - Rw_k))  and  [Ry] + (y - Ry); X itself is never
// shared, only E (public) and [U] are used (SecureML's trick: one mask U per data entry, for the whole
// training). A client that sends different values to different parties is caught by the MAC check:
// the public constant would enter the MAC relation differently at different parties.
//
// Iteration t (batch rows r = tB .. tB+B-1), all on authenticated shares:
//   1. F = open(w - Vf_t)                                  1 exchange
//   2. z_i = Zf_i + sum_j F_j [U_ij] + sum_j E_ij [Vf_j] + sum_j E_ij F_j       local (Beaver, x_ij w_j = (U+E)(V+F))
//   3. yhat_i = activation(z_i)                             1 + ceil(log2 N) + 2 exchanges (see below)
//   4. F' = open(e - Vb_t),  e_i = yhat_i - y_i             1 exchange
//   5. g_j = Zb_j + sum_i F'_i [U_ij] + sum_i E_ij [Vb_i] + sum_i E_ij F'_i      local
//   6. w_j -= g_j                                            local (lr/batch = 2^-k is folded into the scale of w)
//
// Activation (SecureML's piecewise function, computed exactly). With v = z + 1/2 (scale zs = 3F + k)
// and T = 2F + k, the result is yhat = clamp(floor(v / 2^T), 0, 2^F). Using ONE masked opening
//     c = v + 2^(N-1) + r,   r = sum_{i < N + ssp} 2^i r_i  with random shared bits r_i,
// (|v| < 2^(N-2); the top ssp bits statistically hide v), everything follows from c and the bits r_i:
//   q  = floor(v / 2^T) = floor(c / 2^T) - 2^(N-1-T) - sum_{i>=T} 2^(i-T) r_i - [c mod 2^T < r mod 2^T]
//   g1 = [v >= 0]  = bit N-1 of (c - r) mod 2^N     = c_{N-1} xor r_{N-1} xor [c mod 2^(N-1) < r mod 2^(N-1)]
//   g2 = [q >= 2^F] = [v >= 2^zs]                   = same with c replaced by c - 2^zs
//   yhat = g2 * 2^F + (g1 - g2) * q
// The three "[public < secret bits]" comparisons are borrow chains evaluated as log-depth trees of
// Beaver multiplications. q is exact, so the secure result equals the plaintext integer reference
// bit for bit.
#ifndef SPZ_ONLINE_H
#define SPZ_ONLINE_H

#include <algorithm>
#include <vector>

#include "metrics.h"
#include "mpc.h"
#include "params.h"
#include "preproc.h"

namespace spz {

// Client k's additive share of the initial weights (w0 = sum of the clients' shares, in F_p). Test convention,
// as in the sibling baselines: shares 1..batch-1 are pseudo-random field elements derived from --seed, and the
// last client's share is what is left, so every client can derive its own share without talking to the others.
inline std::vector<Fp> client_w0_share(const Params& P, const Dataset& D, int k) {
  const int B = P.batch;
  auto rand_share = [&](int kk, int64_t j) {
    SplitMix64 g(P.seed * 0xD1B54A32D192ED03ULL + (uint64_t)kk * 0x9E3779B97F4A7C15ULL + (uint64_t)j * 0xBF58476D1CE4E5B9ULL + 7);
    const u128 hi = g.next();
    return Fp::from_u128_reduce((hi << 64) | g.next());
  };
  std::vector<Fp> out((size_t)P.d);
  for (int64_t j = 0; j < P.d; ++j) {
    if (k < B) out[(size_t)j] = rand_share(k, j);
    else {
      Fp rest = Fp::from_int(D.w0[(size_t)j]);
      for (int kk = 1; kk < B; ++kk) rest -= rand_share(kk, j);
      out[(size_t)j] = rest;
    }
  }
  return out;
}

class Online {
 public:
  Online(Mpc& m, const Params& p, Pre& pre, Recorder& rec) : M(m), P(p), pre_(pre), rec_(rec) {
    M.online_phase = true;
  }

  // Client: pass the data (all rows; it uses its own). Server: pass nullptr (it only receives).
  void input(const Dataset* D) {
    Scope sc(rec_, "online/input");
    const int64_t d = P.d, n = P.n, T = P.iters, B = P.batch;
    const int me = M.me, np = M.n;
    const size_t per = (size_t)(T * d + d + T);  // one client's message: [E rows | w0_k - Rw_k | y - Ry]
    std::vector<std::vector<Fp>> msg((size_t)np);
    for (int o = 1; o < np; ++o) msg[(size_t)o].assign(per, Fp(0));
    if (me >= 1) {
      const int64_t i0 = me - 1;
      std::vector<Fp>& m = msg[(size_t)me];
      const std::vector<Fp> w0s = client_w0_share(P, *D, me);
      for (int64_t t = 0; t < T; ++t) {
        const int64_t r = t * B + i0;
        for (int64_t j = 0; j < d; ++j)
          m[(size_t)(t * d + j)] = Fp::from_int(D->x[(size_t)(r * d + j)]) - pre_.u_plain[(size_t)(r * d + j)];
        m[(size_t)(T * d + d + t)] = Fp::from_int((int64_t)D->y[(size_t)r] << P.frac) - pre_.ry_plain[(size_t)r];
      }
      for (int64_t j = 0; j < d; ++j) m[(size_t)(T * d + j)] = w0s[(size_t)j] - pre_.rw_plain[(size_t)j];
    }
    {
      std::vector<Mesh::Buf> s((size_t)np);
      std::vector<Mesh::RBuf> r((size_t)np);
      std::vector<Fp> alt;
      int equiv_peer = -1;
      if (me >= 1 && M.tamper.hit(Tamper::INPUT_EQUIV, me)) {  // a different input to one peer only
        alt = msg[(size_t)me];
        alt[0] += Fp(1);
        equiv_peer = me == 1 ? 0 : 1;
      }
      for (int p = 0; p < np; ++p) {
        if (p == me) continue;
        if (me >= 1) s[(size_t)p] = {p == equiv_peer ? alt.data() : msg[(size_t)me].data(), per * sizeof(Fp)};
        if (p >= 1) r[(size_t)p] = {msg[(size_t)p].data(), per * sizeof(Fp)};
      }
      M.mesh.exchange(s, r);
    }
    E_.assign((size_t)(n * d), Fp(0));
    w_.resize((size_t)d);
    y_.resize((size_t)n);
    std::vector<Fp> wc((size_t)d, Fp(0));
    for (int o = 1; o < np; ++o) {
      for (const Fp& x : msg[(size_t)o]) if (x.v >= kP) throw AbortError("input: non-canonical field element from party " + std::to_string(o));
      const std::vector<Fp>& m = msg[(size_t)o];
      for (int64_t t = 0; t < T; ++t) {
        const int64_t r = t * B + (o - 1);
        for (int64_t j = 0; j < d; ++j) E_[(size_t)(r * d + j)] = m[(size_t)(t * d + j)];
        y_[(size_t)r] = M.add_const(pre_.Ry[(size_t)r], m[(size_t)(T * d + d + t)]);
      }
      for (int64_t j = 0; j < d; ++j) wc[(size_t)j] += m[(size_t)(T * d + j)];
    }
    for (int64_t j = 0; j < d; ++j) {
      Sh acc = M.zero();
      for (int o = 1; o < np; ++o) acc = M.add(acc, pre_.Rw[(size_t)((o - 1) * d + j)]);
      w_[(size_t)j] = M.add_const(acc, wc[(size_t)j]);
    }
  }

  void train(bool quiet) {
    for (int t = 0; t < P.iters; ++t) {
      iteration(t);
      if (!quiet && (t == 0 || t + 1 == P.iters || (t + 1) % 5 == 0))
        std::fprintf(stderr, "  [%s] iteration %d/%d done\n", M.me == 0 ? "server" : "client", t + 1, P.iters);
    }
  }

  // Opens w + Rout (public), then verifies every opening of the run. Returns the server's weights
  // (scale S_w); the client gets an empty vector (it only sees w + Rout, which is uniform).
  std::vector<i128> output() {
    std::vector<Sh> masked((size_t)P.d);
    {
      Scope sc(rec_, "online/output");
      for (int j = 0; j < P.d; ++j) masked[(size_t)j] = M.add(w_[(size_t)j], pre_.Rout[(size_t)j]);
    }
    std::vector<Fp> opened;
    {
      Scope sc(rec_, "online/output");
      opened = M.open(masked);
    }
    {
      Scope sc(rec_, "online/mac_check");
      M.mac_check();  // nothing is accepted before every opened value has been verified
    }
    std::vector<i128> w;
    if (M.me == 0)
      for (int j = 0; j < P.d; ++j) w.push_back((opened[(size_t)j] - pre_.rout_plain[(size_t)j]).to_signed());
    return w;
  }

  size_t triples_left() const { return pre_.tc.size() - pre_.tri_next; }

 private:
  Mpc& M;
  const Params& P;
  Pre& pre_;
  Recorder& rec_;
  std::vector<Fp> E_;
  std::vector<Sh> w_, y_;

  // ---- Beaver multiplication of a batch (one exchange) ----------------------------------------------
  std::vector<Sh> mul_batch(const std::vector<Sh>& xs, const std::vector<Sh>& ys) {
    const size_t K = xs.size();
    if (K == 0) return {};
    if (pre_.tri_next + K > pre_.tc.size()) throw std::runtime_error("out of multiplication triples");
    const size_t i0 = pre_.tri_next;
    pre_.tri_next += K;
    std::vector<Sh> de(2 * K);
    for (size_t k = 0; k < K; ++k) {
      de[k] = M.sub(xs[k], pre_.ta[i0 + k]);
      de[K + k] = M.sub(ys[k], pre_.tb[i0 + k]);
    }
    const std::vector<Fp> o = M.open(de);
    std::vector<Sh> z(K);
    for (size_t k = 0; k < K; ++k) {
      const Fp dd = o[k], ee = o[K + k];
      Sh s = M.add(pre_.tc[i0 + k], M.add(M.scale(pre_.tb[i0 + k], dd), M.scale(pre_.ta[i0 + k], ee)));
      z[k] = M.add_const(s, dd * ee);
    }
    return z;
  }

  struct Seg { Sh G, P; };

  // ---- the activation gadget on a batch of B pre-activations -----------------------------------------
  std::vector<Sh> activation(const std::vector<Sh>& z) {
    const int B = (int)z.size();
    const int N = P.nbits, mb = P.mask_bits(), T = P.tr, F = P.frac;
    const Fp half = pow2(P.zs - 1);
    const u128 maskN = (((u128)1) << N) - 1;
    const u128 theta = ((u128)1) << P.zs;
    const Sh* bits = &pre_.bits[pre_.bit_next];
    if (pre_.bit_next + (size_t)B * mb > pre_.bits.size()) throw std::runtime_error("out of random bits");
    pre_.bit_next += (size_t)B * mb;
    auto R = [&](int s, int i) -> const Sh& { return bits[(size_t)s * mb + i]; };

    // 1. masked opening c = v + 2^(N-1) + r
    std::vector<Sh> csh((size_t)B);
    {
      Scope sc(rec_, "online/activation_local");
      const Fp k0 = half + pow2(N - 1);
      for (int s = 0; s < B; ++s) {
        Sh acc = M.add_const(z[(size_t)s], k0);
        for (int i = 0; i < mb; ++i) acc = M.add(acc, M.scale(R(s, i), pow2(i)));
        csh[(size_t)s] = acc;
      }
    }
    std::vector<Fp> cv;
    {
      Scope sc(rec_, "online/activation_open");
      cv = M.open(csh);
    }

    // 2. borrow chains. chain 0: low T bits of c; chain 1: low N-1 bits of c; chain 2: low N-1 bits of c - 2^zs.
    Scope sc2(rec_, "online/activation_chains");
    std::vector<u128> cu((size_t)B), cth((size_t)B);
    for (int s = 0; s < B; ++s) {
      cu[(size_t)s] = cv[(size_t)s].v;
      cth[(size_t)s] = ((cu[(size_t)s] & maskN) + (((u128)1) << N) - (theta & maskN)) & maskN;
    }
    const int nbit[3] = {T, N - 1, N - 1};
    std::vector<std::vector<Seg>> seg((size_t)B * 3);
    for (int s = 0; s < B; ++s)
      for (int c = 0; c < 3; ++c) {
        auto& v = seg[(size_t)s * 3 + c];
        v.resize((size_t)nbit[c]);
        for (int i = 0; i < nbit[c]; ++i) {
          const bool Cb = ((c == 2 ? cth[(size_t)s] : cu[(size_t)s]) >> i) & 1;
          const Sh& r = R(s, i);
          if (!Cb) v[(size_t)i] = {r, M.sub(M.from_const(Fp(1)), r)};
          else v[(size_t)i] = {M.zero(), r};
        }
      }
    for (;;) {
      std::vector<Sh> xs, ys;
      bool any = false;
      for (auto& v : seg) {
        const size_t m = v.size();
        if (m <= 1) continue;
        any = true;
        for (size_t p = 0; p < m / 2; ++p) {
          xs.push_back(v[2 * p + 1].P); ys.push_back(v[2 * p].G);
          if (p > 0) { xs.push_back(v[2 * p + 1].P); ys.push_back(v[2 * p].P); }
        }
      }
      if (!any) break;
      const std::vector<Sh> prod = mul_batch(xs, ys);
      size_t q = 0;
      for (auto& v : seg) {
        const size_t m = v.size();
        if (m <= 1) continue;
        std::vector<Seg> nv;
        for (size_t p = 0; p < m / 2; ++p) {
          Seg s;
          s.G = M.add(v[2 * p + 1].G, prod[q++]);
          if (p > 0) s.P = prod[q++]; else s.P = M.zero();
          nv.push_back(s);
        }
        if (m & 1) nv.push_back(v[m - 1]);
        v = std::move(nv);
      }
    }

    // 3. xor with the top bits: g = c_top xor r_top xor borrow  (one product each)
    std::vector<Sh> xs, ys;
    for (int s = 0; s < B; ++s)
      for (int c = 1; c <= 2; ++c) { xs.push_back(R(s, N - 1)); ys.push_back(seg[(size_t)s * 3 + c][0].G); }
    const std::vector<Sh> rb = mul_batch(xs, ys);
    std::vector<Sh> g1((size_t)B), g2((size_t)B), diff((size_t)B), q((size_t)B);
    for (int s = 0; s < B; ++s) {
      for (int c = 1; c <= 2; ++c) {
        const Sh& beta = seg[(size_t)s * 3 + c][0].G;
        const Sh& prod = rb[(size_t)s * 2 + (c - 1)];
        Sh t = M.sub(M.add(R(s, N - 1), beta), M.scale(prod, Fp(2)));  // r xor beta
        const bool top = ((c == 2 ? cth[(size_t)s] : cu[(size_t)s]) >> (N - 1)) & 1;
        if (top) t = M.sub(M.from_const(Fp(1)), t);
        (c == 1 ? g1 : g2)[(size_t)s] = t;
      }
      diff[(size_t)s] = M.sub(g1[(size_t)s], g2[(size_t)s]);
      // q = floor(c / 2^T) - 2^(N-1-T) - sum_{i >= T} 2^(i-T) r_i - borrow_T
      Sh qq = M.from_const(Fp::from_u128_reduce(cu[(size_t)s] >> T) - pow2(N - 1 - T));
      for (int i = T; i < mb; ++i) qq = M.sub(qq, M.scale(R(s, i), pow2(i - T)));
      q[(size_t)s] = M.sub(qq, seg[(size_t)s * 3 + 0][0].G);
    }
    // 4. yhat = g2 * 2^F + (g1 - g2) * q
    const std::vector<Sh> mid = mul_batch(diff, q);
    std::vector<Sh> yhat((size_t)B);
    for (int s = 0; s < B; ++s) yhat[(size_t)s] = M.add(M.scale(g2[(size_t)s], pow2(F)), mid[(size_t)s]);
    return yhat;
  }

  // ---- one SGD iteration ----------------------------------------------------------------------------
  void iteration(int t) {
    const int64_t d = P.d, B = P.batch;
    const int threads = M.mesh.nw();
    const int64_t grain = 4096;

    // 1. F = open(w - Vf_t)
    std::vector<Fp> Fv;
    {
      Scope sc(rec_, "online/open_w");
      std::vector<Sh> tmp((size_t)d);
      for (int64_t j = 0; j < d; ++j) tmp[(size_t)j] = M.sub(w_[(size_t)j], pre_.Vf[(size_t)(t * d + j)]);
      Fv = M.open(tmp);
    }
    // 2. z_i
    std::vector<Sh> z((size_t)B);
    {
      Scope sc(rec_, "online/forward_local");
      const int parts = parts_for(threads, d, grain);
      for (int64_t i = 0; i < B; ++i) {
        const int64_t r = t * B + i;
        std::vector<Fp> pv((size_t)parts), pm((size_t)parts), pc((size_t)parts);
        par_parts(parts, d, [&](int p, int64_t lo, int64_t hi) {
          Fp sv(0), sm(0), sc2(0);
          for (int64_t j = lo; j < hi; ++j) {
            const Sh& u = pre_.U[(size_t)(r * d + j)];
            const Sh& v = pre_.Vf[(size_t)(t * d + j)];
            const Fp e = E_[(size_t)(r * d + j)], f = Fv[(size_t)j];
            sv += f * u.v + e * v.v;
            sm += f * u.m + e * v.m;
            sc2 += e * f;
          }
          pv[(size_t)p] = sv; pm[(size_t)p] = sm; pc[(size_t)p] = sc2;
        });
        Sh acc = pre_.Zf[(size_t)r];
        Fp cst(0);
        for (int p = 0; p < parts; ++p) { acc.v += pv[(size_t)p]; acc.m += pm[(size_t)p]; cst += pc[(size_t)p]; }
        z[(size_t)i] = M.add_const(acc, cst);
      }
    }
    // 3. activation
    std::vector<Sh> yhat = activation(z);
    // 4. F' = open(e - Vb_t)
    std::vector<Fp> Fb;
    {
      Scope sc(rec_, "online/open_e");
      std::vector<Sh> tmp((size_t)B);
      for (int64_t i = 0; i < B; ++i)
        tmp[(size_t)i] = M.sub(M.sub(yhat[(size_t)i], y_[(size_t)(t * B + i)]), pre_.Vb[(size_t)(t * B + i)]);
      Fb = M.open(tmp);
    }
    // 5-6. g_j and the update (parallel over j)
    {
      Scope sc(rec_, "online/backward_local");
      par_ranges(threads, d, grain / 4, [&](int64_t lo, int64_t hi) {
        for (int64_t j = lo; j < hi; ++j) {
          Sh acc = pre_.Zb[(size_t)(t * d + j)];
          Fp cst(0);
          for (int64_t i = 0; i < B; ++i) {
            const int64_t r = t * B + i;
            const Sh& u = pre_.U[(size_t)(r * d + j)];
            const Sh& v = pre_.Vb[(size_t)(t * B + i)];
            const Fp e = E_[(size_t)(r * d + j)], f = Fb[(size_t)i];
            acc.v += f * u.v + e * v.v;
            acc.m += f * u.m + e * v.m;
            cst += e * f;
          }
          acc = M.add_const(acc, cst);
          w_[(size_t)j] = M.sub(w_[(size_t)j], acc);
        }
      });
    }
  }
};

}  // namespace spz

#endif  // SPZ_ONLINE_H
