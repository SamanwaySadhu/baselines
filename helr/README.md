# Baseline 4: HELR (homomorphic-encryption logistic regression), with malicious-security checks

This baseline trains logistic regression the way **HELR** does it (Kim, Song, Wang, Xia, Jiang, *Secure Logistic
Regression Based on Homomorphic Encryption: Design and Evaluation*, JMIR Med Inform 2018, ePrint 2018/074; code
[K-miran/HELR](https://github.com/K-miran/HELR)): the data is encrypted under CKKS/HEAAN, and the server runs
gradient descent **entirely on ciphertexts**, with the sigmoid replaced by a least-squares polynomial.

| role | holds | learns |
|---|---|---|
| **client** | training data (x, y), initial weights w0 (random for now), the HE secret key | nothing |
| **server** | nothing | only the trained weights |

The training parameters are those of the other baselines: **20 iterations, batch 4, learning rate 0.5**, 6-bit quantized
inputs, the bias as the last of the `--d` features. The data generator is baseline 1's: the same `--seed` gives the same
data and initial weights in all four baselines.

## What is computed

HELR's formulation: with `z_i = (2 y_i - 1) x_i` (label sign folded into the features) one iteration is

```
w_i  = z_i . beta / R                    R = 2^logR (HELR: 8); the data is encrypted as z / R
g_i  = (1 - H(w_i)) / 2                  ~ sigmoid(-z_i . beta);  H = least-squares odd polynomial of degree 3 (or 7)
beta = beta + (1/K) sum_{i in batch} g_i z_i,        K = batch / lr = 8
```

`H` is the least-squares fit of `tanh(R w / 2)` on `[-1, 1]` (HELR: "least squares approximation of the sigmoid over
[-8, 8]"; for R = 8 the degree-3 polynomial is the paper's `1/2 + 1.20096 (x/8) - 0.81562 (x/8)^3`).

**Differences from the paper's setting, all forced by this baseline's task:**
- **Initial weights are ciphertexts.** HELR starts from beta = 0; here w0 is the client's private input, so every iteration is the general one.
- **Mini-batches.** HELR uses the whole training set in every iteration. Here iteration t uses its own batch of 4 samples:
  all `iters x batch` samples are packed once (sample (t, i) in slot `t*batch + i` of one ciphertext per feature) and
  iteration t multiplies by a public 0/constant mask. The mask folds into the constant HELR multiplies by anyway, so it
  costs no multiplicative depth.
- **R depends on d.** The polynomial is only accurate for `|z . beta| <= R`. Random features and weights give
  `|z . beta| ~ sqrt(d)/3`, so R = 2^logR is the smallest power of two `>= 2 sqrt(d)` (at least 8, the paper's value; 8 for d <= 16,
  64 for d = 1000). `--logr` overrides it. A larger R means a coarser cubic fit (the paper's own caveat, Sec. 5.2); the run
  reports whether `|z . beta| / R` stayed within 1.
- **Degree 7 is corrected.** HELR's published degree-7 code discards the `c7` multiplications (it calls `multByConst`, not
  `multByConstAndEqual`) and adds ciphertexts of different moduli. `--deg 7` here is a clean implementation of the
  paper's evaluation order (4 levels per iteration); its coefficients are recomputed (the paper's printed ones differ by up to 2e-3; ours have
  the lower mean-square error, checked in the selftest).
- **Approximate arithmetic, polynomial sigmoid.** The result is close to, not equal to, the fixed-point baselines' weights
  (those use the exact sigmoid and truncations). `--verify` reports the distance to all of them. Compare costs, not weights.

### Parameters

HELR's: logp = 28 scaling bits, a fresh ciphertext modulus of `T * (3 logp) + (logp + 10 + mac_bits)` bits (3 levels of p per
iteration at degree 3; `mac_bits` = 48 extra output bits for the decryption MAC), sparse ternary secret (h = 64), sigma = 3.2.
The ring dimension is the smallest meeting `--lambda 128` by HELR's own formula `N >= (lambda + 110)/7.2 * log(qQ)` (with Q = q), which gives
N = 2^17 for the default shape (log q = 1766, lambda ~ 157 by that formula). The formula is from 2012 and the sparse secret is
weaker than a dense one; use the lattice estimator if you need a current figure. Ciphertexts are 2 x N x log q bits (58 MiB fresh).

## Install on WSL (Ubuntu 22.04/24.04)

```bash
sudo apt install -y build-essential cmake git libntl-dev libgmp-dev libssl-dev
cd baselines/helr
bash scripts/fetch_deps.sh      # HELR at a pinned commit + patches/heaan.patch -> third_party/HELR
bash scripts/build.sh
./build/lr_helr_selftest        # ~15 s: tiny INSECURE ring (N = 2^10); the circuit, MAC and protocol are the real code
bash scripts/run_local.sh --d 3 --iters 3 --logn 11 --insecure     # the whole flow in seconds
bash scripts/run_local.sh --d 2                                     # real parameters (N = 2^17): tens of minutes, ~4 GB per party
```

By hand: `./build/lr_helr server --port 23456 --d 5` and `./build/lr_helr client --host <server-ip> --port 23456 --d 5`
(same flags at both parties). Options: training `--d --iters --batch --qbits --lr --seed --no-bias --zero-init`; HE
`--deg --logr --logp --logn --lambda --mac-bits --insecure`; protocol `--host --port --threads --tol`; output `--verify --json --csv --quiet`;
test-only `--tamper`. `--help` has the details. The patch to HELR: an empty `PubKey` constructor (HELR's makes all 32
rotation keys, ~5 GB) and move constructors for `Cipher` / `Message`. Everything else in `third_party/HELR/src` is
unmodified; the prebuilt `.o`/`.a` files in the upstream repository are not used.

## Protocol

```
S -> C  com_S = H(sigma_S)                    coin toss for the public-key element a
C -> S  header, sigma_C                       a = XOF(sigma_S || sigma_C)
S -> C  sigma_S
C -> S  UPLOAD  pk.b, mult key, rotation keys 2^0..2^(logn-1), d encrypted data columns, d encrypted w0     [offline]
        server: HELR's gradient descent on ciphertexts, no interaction                                      [online]
S -> C  c1_j (final weight j), c2_j = alpha_j c1_j + Enc_pk(r_j)          for every j < d
C -> S  commitment to the decryptions (u1_j, u2_j) of c1_j and c2_j
S -> C  seed_j (everything the server drew for c2_j)
C -> S  opening: nonce and every (u1_j, u2_j)
server: checks commitment, MAC and weight range; outputs w_j = u1_j / 2^logp
```

## Security model

**Read this before quoting the baseline as "malicious secure".** HE has no cheap analogue of the MAC checks of baselines 1
and 2. Fully simulatable malicious security needs zero-knowledge proofs that the client's ciphertexts and evaluation keys are
well formed (lattice proofs of plaintext knowledge and key validity) and verifiable evaluation. None of that is implemented here.
What the baseline adds on top of HELR, and what it guarantees:

**Against a malicious server (privacy of the client's data).**
- The server sees only RLWE ciphertexts under the client's key, `pk.a` is jointly random (commit-and-reveal coin toss), and
  nothing else of the client until the output phase.
- The client never decrypts anything the server could use as a decryption oracle. It decrypts c1_j and c2_j internally, then
  (i) checks that the decrypted weights are within `--tol` (default 2^-8) of the weights **it computes itself by running the same
  training in the clear** (it owns the inputs; this costs microseconds against hours of HE), and (ii) commits to the decryptions.
  Only after the server has revealed `seed_j` does it re-derive c2_j exactly (alpha_j, r_j, the encryption randomness) and compare; a c2 that is not
  the honest MAC ciphertext of c1_j aborts the run with nothing released. Only then does it open.
- Residual leakage: **one abort bit per run** (the server can make the client abort on a predicate of its choosing), and the
  released u1_j may differ from the honest decryption by any amount below `--tol` (a server-chosen, data-dependent perturbation
  inside the tolerance window: at most log2(tol * 2^logp) = 20 bits per weight beyond the output). Approximate-arithmetic HE cannot avoid
  a window like this without proofs of correct evaluation.
- The opening reveals one exact linear equation on the secret key per weight (coefficient 0 of the decryption of c1_j and of c2_j). The key is
  fresh per run and sparse (h = 64) so d equations are far from enough to recover it for the supported d, but the equations are exact.
  Do not extend the opening to more coefficients.

**Against a malicious client (integrity of the server's output).**
- *Decryption MAC.* The client must report u1_j = Dec(c1_j)[0], the weight. The server sends c2_j = alpha_j * c1_j + Enc_pk(r_j) with a
  fresh odd alpha_j and uniform mask r_j (all derived from seed_j), the client **commits to (u1_j, u2_j) before it learns alpha_j and r_j**, and the
  server accepts only if `u2_j = alpha_j u1_j + r_j` up to the fresh-encryption noise bound. u2_j is uniform because of r_j, and c2_j hides alpha_j
  under RLWE given a uniform `a` (hence the coin toss: a client-chosen `a` would let it read alpha_j off c2_j). A wrong u1_j needs
  alpha_j * delta. Forging probability per weight is `<= 2^-(mac_bits+2)` = 2^-50 by default (`mac_forgery_bits`; the run refuses
  parameters below 2^-40). The checks are exact: a forged shift of 1 is rejected (selftest [5]).
- The server rejects any weight outside the range an honest run can produce, and any malformed ring element (padding bits).
- **Not covered:** a malicious client can still send ciphertexts or evaluation keys that are not well formed, so the server's output is
  guaranteed to be the decryption of *the server's own homomorphic computation on what the client sent*, not that it equals
  the training on some in-range dataset. (Out-of-range inputs, or a malformed multiplication key, can steer the result.) Closing this needs the
  zero-knowledge proofs above; their cost (several extra ciphertext-sized messages per batch) would be small next to the
  evaluation, but the **security** claim of this baseline stops here. Equivalently: what is enforced is correctness of decryption and
  privacy against a malicious server; input well-formedness is trusted.

`lr_helr_selftest` runs every deviation it can: a client that lies about a weight (MAC), opens something else than it committed
(commitment), a server that returns a wrong weight or a c2 not derived from its seed (client aborts, server gets nothing), a
too-tight tolerance, mismatching parameters. `--tamper` does the same over TCP (exit code 3 = protocol abort).

Other hardening of HELR's code: NTL's generator is seeded from the OS in every thread (HELR relies on NTL's default seeding),
only the keys the circuit needs are generated, and every received ring element is length- and padding-checked.

## Metrics reported

The report has the same shape as baseline 3's: per party wall and process-CPU time of every phase, payload bytes in each
direction, flights, peak RSS, plus the server's HE operation counts and thread-seconds per operation class (mult, rotate, ...). The parties
exchange their numbers at the end.

- **Offline** = before the evaluation starts: coin toss, keygen, encryption, the client's plaintext run, one upload flight (keys, data, w0); the server's
  setup and receive. **Online** = the HE evaluation and the output phase (MAC ciphertexts, 2 round trips). The HE evaluation itself is non-interactive.
- The client's online time is dominated by waiting for the server's evaluation; its own online compute is decrypt + check + re-derive (reported separately).
- Bytes are payload bytes (no TCP/IP headers) and equal the analytic formulas (`comm_layout`); the run prints `[measured == analytic]`.

RESULTS_PLACEHOLDER

## Files

| file | contents |
|---|---|
| `src/lr_helr.cpp` | the two-party program: coin toss, offline upload, evaluation, output phase, report exchange, report, JSON/CSV, `--verify` |
| `src/lr_helr_selftest.cpp` | polynomial fit, serialization, HE circuit vs the double-precision reference (deg 3 and 7, several shapes), MAC, full in-process protocol and every deviation |
| `include/helr_protocol.h` | the protocol as state machines (`HeClient`, `HeServer`), the decryption MAC (`derive_c2`), commitments, analytic sizes |
| `include/helr_circuit.h` | HELR's homomorphic gradient descent (`Evaluator`), input encryption |
| `include/helr_reference.h` | the least-squares polynomial, the circuit's function in double precision, the exact-sigmoid training |
| `include/he_core.h` | parameter choice, OS-seeded randomness, key generation, ciphertext/key (de)serialization, coin-tossed `a` |
| `include/net.h`, `include/metrics.h`, `include/lr_params.h`, `include/lr_reference.h` | copies of baseline 3's channel / metrics / data generator, and baseline 1's fixed-point reference |
| `patches/heaan.patch`, `scripts/` | patch to HELR's sources; `fetch_deps.sh`, `build.sh`, `run_local.sh`, `bench_sweep.sh` |
