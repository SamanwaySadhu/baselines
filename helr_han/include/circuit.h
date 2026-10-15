// circuit.h -- the homomorphic NAG logistic regression of the paper, on HEAAN ciphertexts.
//
// State: the weight vectors w and v are `cnum` ciphertexts each; ciphertext j holds the weights of feature group j,
// replicated for every sample slot group: slot (bsz*k + l) = w[j*bsz + l]. The data of iteration t is `cnum`
// ciphertexts Z_{t,j} with slot (bsz*k + l) = z[t*B + k][j*bsz + l].
//
// One iteration (levels counted as consumed bits of the ciphertext modulus; p = pBits, w = wBits):
//   InnerProduct   u_k = sum over j,l of Z.v : slot-wise product, rotate-and-add inside each group of bsz slots, add the
//                  cnum ciphertexts, mask the group heads with a 0/1 plaintext, rotate-and-add to replicate.     p + w
//   Sigmoid        g = gamma z (a0 + a1 u + a3 u^3) = z * [ (gamma a3) u (u^2 + a1/a3) + gamma a0 ], then sum the
//                  B sample groups with rotate-and-add (the mini-batch sum).                                    w + w + w
//   Update         w' = v + g;  v' = (1 - eta) w' + eta w   (Nesterov)                                          p
// i.e. 3w + 2p per iteration. After `iterPerBoot` iterations the weights are at logq = w + lBits and the 2 cnum
// ciphertexts are bootstrapped in parallel (approximate bootstrapping; precision is checked by the client at the end).
#ifndef HAN_CIRCUIT_H
#define HAN_CIRCUIT_H

#include <complex>
#include <vector>

#include "he.h"
#include "metrics.h"
#include "params.h"

namespace han {

// ---- packing (client) ------------------------------------------------------------------------------------
inline std::vector<double> pack_z(const Params& P, const Dataset& D, int t, int j) {
  std::vector<double> v((size_t)P.slots, 0.0);
  for (int k = 0; k < P.batch; ++k)
    for (int l = 0; l < P.bsz; ++l) v[(size_t)(P.bsz * k + l)] = D.z[(size_t)(((int64_t)t * P.batch + k) * P.factorNum + j * P.bsz + l)];
  return v;
}
inline std::vector<double> pack_w(const Params& P, const std::vector<double>& w, int j) {
  std::vector<double> v((size_t)P.slots, 0.0);
  for (int k = 0; k < P.batch; ++k)
    for (int l = 0; l < P.bsz; ++l) v[(size_t)(P.bsz * k + l)] = w[(size_t)(j * P.bsz + l)];
  return v;
}

// Decrypts the weight ciphertexts. Also reports the largest disagreement between the replicas of a weight.
inline std::vector<double> decrypt_weights(Scheme& scheme, SecretKey& sk, std::vector<Ciphertext>& W, const Params& P, double* replica_dev = nullptr) {
  std::vector<double> w((size_t)P.factorNum, 0.0);
  double dev = 0;
  for (int j = 0; j < P.cnum; ++j) {
    std::complex<double>* dec = scheme.decrypt(sk, W[(size_t)j]);
    for (int l = 0; l < P.bsz; ++l) {
      w[(size_t)(j * P.bsz + l)] = dec[l].real();
      for (int k = 1; k < P.batch; ++k) dev = std::max(dev, std::fabs(dec[P.bsz * k + l].real() - dec[l].real()));
    }
    delete[] dec;
  }
  if (replica_dev) *replica_dev = dev;
  return w;
}

class Evaluator {
 public:
  Scheme& scheme;
  const Params& P;
  Poly3 H;
  std::vector<Ciphertext> W, V;

  Evaluator(Scheme& s, const Params& p, Poly3 h) : scheme(s), P(p), H(h) {
    // the 0/1 plaintext that keeps the first slot of every group of bsz slots
    std::complex<double>* pv = new std::complex<double>[P.slots]();
    for (long i = 0; i < P.slots; i += P.bsz) pv[i].real(1.0);
    dummy_ = new NTL::ZZ[scheme.ring.N];
    scheme.ring.encode(dummy_, pv, P.slots, P.pBits);
    delete[] pv;
  }
  ~Evaluator() { delete[] dummy_; }
  Evaluator(const Evaluator&) = delete;
  Evaluator& operator=(const Evaluator&) = delete;

  // One NAG iteration on the data ciphertexts Z of this block. rec may be null.
  void iteration(std::vector<Ciphertext>& Z, double eta, Recorder* rec) {
    Ciphertext ip;
    std::vector<Ciphertext> grad((size_t)P.cnum);
    {
      auto sc = rec ? std::make_unique<Scope>(*rec, "online/inner_product") : nullptr;
      ip = inner_product(Z);
    }
    {
      auto sc = rec ? std::make_unique<Scope>(*rec, "online/sigmoid_gradient") : nullptr;
      sigmoid_gradient(grad, Z, ip);
    }
    {
      auto sc = rec ? std::make_unique<Scope>(*rec, "online/nag_update") : nullptr;
      update(grad, eta);
    }
  }

  // Bootstraps w and v (2 cnum independent ciphertexts, in parallel). The ciphertexts hold bsz distinct values
  // replicated batch times, so they are bootstrapped as bsz-slot ciphertexts and re-interpreted afterwards.
  void bootstrap_all(Recorder* rec) {
    auto sc = rec ? std::make_unique<Scope>(*rec, "online/bootstrap") : nullptr;
    par_for(2L * P.cnum, [&](long idx) {
      Ciphertext& c = idx < P.cnum ? W[(size_t)idx] : V[(size_t)(idx - P.cnum)];
      c.n = P.bsz;
      scheme.bootstrapAndEqual(c, P.logq, P.logQBoot, P.logT, P.logI);
      c.n = P.slots;
    });
  }

 private:
  NTL::ZZ* dummy_ = nullptr;

  Ciphertext inner_product(std::vector<Ciphertext>& Z) {
    std::vector<Ciphertext> ipv((size_t)P.cnum);
    par_for(P.cnum, [&](long i) {
      Ciphertext tmp = scheme.modDownTo(Z[(size_t)i], V[(size_t)i].logq);
      ipv[(size_t)i] = scheme.mult(tmp, V[(size_t)i]);  // logp 2w
      for (long j = 0; j < P.bBits; ++j) {
        Ciphertext rot = scheme.leftRotateFast(ipv[(size_t)i], 1L << j);
        scheme.addAndEqual(ipv[(size_t)i], rot);
      }
    });
    Ciphertext ip = ipv[0];
    for (int i = 1; i < P.cnum; ++i) scheme.addAndEqual(ip, ipv[(size_t)i]);
    scheme.multByPolyAndEqual(ip, dummy_, P.pBits);  // logp 2w + p
    for (long i = 0; i < P.bBits; ++i) {
      Ciphertext tmp = scheme.rightRotateFast(ip, 1L << i);
      scheme.addAndEqual(ip, tmp);
    }
    scheme.reScaleByAndEqual(ip, P.pBits + P.wBits);  // logp w, modulus -(p + w)
    return ip;
  }

  void sigmoid_gradient(std::vector<Ciphertext>& grad, std::vector<Ciphertext>& Z, Ciphertext& ip) {
    const double gamma = P.lr / P.batch;
    Ciphertext sq = scheme.square(ip);
    scheme.addConstAndEqual(sq, H.a1 / H.a3, 2 * P.wBits);
    scheme.reScaleByAndEqual(sq, P.wBits);  // u^2 + a1/a3
    par_for(P.cnum, [&](long i) {
      Ciphertext& g = grad[(size_t)i];
      g = scheme.multByConst(Z[(size_t)i], gamma * H.a3, 2 * P.wBits);
      scheme.reScaleByAndEqual(g, 2 * P.wBits);
      scheme.modDownToAndEqual(g, ip.logq);
      scheme.multAndEqual(g, ip);
      scheme.reScaleByAndEqual(g, P.wBits);
      scheme.multAndEqual(g, sq);
      scheme.reScaleByAndEqual(g, P.wBits);
      Ciphertext tmp = scheme.multByConst(Z[(size_t)i], gamma * H.a0, 2 * P.wBits);
      scheme.reScaleByAndEqual(tmp, 2 * P.wBits);
      scheme.modDownToAndEqual(tmp, g.logq);
      scheme.addAndEqual(g, tmp);
      for (long l = P.bBits; l < P.sBits; ++l) {  // the mini-batch sum
        Ciphertext rot = scheme.leftRotateFast(g, 1L << l);
        scheme.addAndEqual(g, rot);
      }
    });
  }

  void update(std::vector<Ciphertext>& grad, double eta) {
    par_for(P.cnum, [&](long i) {
      Ciphertext w1 = scheme.modDownTo(V[(size_t)i], grad[(size_t)i].logq);
      scheme.addAndEqual(w1, grad[(size_t)i]);                                  // w' = v + g
      Ciphertext ew = scheme.multByConst(W[(size_t)i], eta, P.wBits + P.pBits);
      scheme.reScaleByAndEqual(ew, P.wBits + P.pBits);                          // eta * w
      V[(size_t)i] = scheme.multByConst(w1, 1.0 - eta, P.pBits);
      scheme.reScaleByAndEqual(V[(size_t)i], P.pBits);                          // (1 - eta) w'
      scheme.modDownToAndEqual(ew, V[(size_t)i].logq);
      scheme.addAndEqual(V[(size_t)i], ew);                                     // v' = (1-eta) w' + eta w
      W[(size_t)i] = w1;
      scheme.modDownToAndEqual(W[(size_t)i], V[(size_t)i].logq);
    });
  }
};

}  // namespace han

#endif  // HAN_CIRCUIT_H
