# Baseline 4: HELR (homomorphic-encryption logistic regression), semi-honest

This baseline trains logistic regression the way **HELR** does it (Kim, Song, Wang, Xia, Jiang, *Secure Logistic
Regression Based on Homomorphic Encryption: Design and Evaluation*, JMIR Med Inform 2018, ePrint 2018/074; code
[K-miran/HELR](https://github.com/K-miran/HELR)): the data is encrypted under CKKS/HEAAN, and the server runs
gradient descent **entirely on ciphertexts**, with the sigmoid replaced by a least-squares polynomial. The protocol is the
paper's: **semi-honest** secure outsourcing (its Fig. 1(a)); there is no malicious-security machinery (MACs, commitments,
coin tossing, consistency checks, proofs). An earlier version of this baseline had that machinery; it was removed to match the paper.

| role | holds | learns |
|---|---|---|
| **client** | training data (x, y), initial weights w0 (random for now), the HE secret key | the trained weights (it is the decryptor, as in the paper) |
| **server** | nothing | nothing but the sizes: it sees only ciphertexts and evaluation keys |

As in the paper (and in baseline 6, `helr_han`), the data owner is the only party that can decrypt, so **the client, not the
server, ends up with the model**. This differs from the secret-sharing / garbled-circuit baselines, where the server receives it.

The training parameters are those of the other baselines: **20 iterations, batch 4, learning rate 0.5**, 6-bit quantized
inputs, the bias as the last of the `--d` features. The data generator is baseline 1's: the same `--seed` gives the same
data and initial weights in all baselines.

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

HELR's (`Test_HELR.cpp`, `LRtest.cpp`): logN = 17, logp = 28 scaling bits, output modulus `logp + logl` with logl = 10, a fresh
ciphertext modulus of `T * (3 logp) + logp + 10` bits (3 levels of p per iteration at degree 3, 4 at degree 7), sparse ternary
secret (h = 64), sigma = 3.2, `Params(logN, logq)` and HEAAN's key generation (uniform public-key element `a`). The ring
dimension is the smallest meeting `--lambda 128` by HELR's own formula `N >= (lambda + 110)/7.2 * log(qQ)` (with Q = q), which gives
N = 2^17 for the default shape (log q = 1718, lambda ~ 164 by that formula). The formula is from 2012 and the sparse secret is
weaker than a dense one; use the lattice estimator if you need a current figure. Ciphertexts are 2 x N x log q bits (~56 MiB fresh).

## Install on WSL (Ubuntu 22.04/24.04)

```bash
sudo apt install -y build-essential cmake git libntl-dev libgmp-dev libssl-dev
cd baselines/helr
bash scripts/fetch_deps.sh      # HELR at a pinned commit + patches/heaan.patch -> third_party/HELR
bash scripts/build.sh
./build/lr_helr_selftest        # ~6 s: tiny INSECURE ring (N = 2^10); the circuit and protocol are the real code
bash scripts/run_local.sh --d 3 --iters 3 --logn 11 --insecure     # the whole flow in seconds
bash scripts/run_local.sh --d 2                                     # real parameters (N = 2^17): tens of minutes, ~4 GB per party
```

By hand: `./build/lr_helr server --port 23456 --d 5` and `./build/lr_helr client --host <server-ip> --port 23456 --d 5`
(same flags at both parties). Options: training `--d --iters --batch --qbits --lr --seed --no-bias --zero-init`; HE
`--deg --logr --logp --logn --lambda --insecure`; protocol `--host --port --threads`; output `--verify --tol --json --csv --quiet`.
`--help` has the details. The patch to HELR: an empty `PubKey` constructor (HELR's makes all 32
rotation keys, ~5 GB) and move constructors for `Cipher` / `Message`. Everything else in `third_party/HELR/src` is
unmodified; the prebuilt `.o`/`.a` files in the upstream repository are not used.

## Protocol

```
C -> S  UPLOAD  header, mult key, rotation keys 2^0..2^(logn-1), d encrypted data columns, d encrypted w0     [offline]
        server: HELR's gradient descent on ciphertexts, no interaction                                      [online]
S -> C  the d encrypted final weights
client: decrypts (HELR's Scheme::decrypt, slot 0, divided by 2^logp)
```

One flight in each direction. The header carries the public parameters so that a configuration mismatch fails loudly; it is not
a security check. The server never receives the public key (evaluation does not use it).

## Security model

**Semi-honest, exactly as in the paper.** Both parties follow the protocol and try to learn from what they see. The server sees
only RLWE ciphertexts and evaluation keys under the client's key; privacy of the client's data and weights is the semantic
security of the HE scheme (RLWE, sparse ternary secret, sigma = 3.2). The server has no input, so there is nothing for the client
to learn about it. Nothing checks that either party followed the protocol: a malicious server could return wrong ciphertexts, and a
malicious client could send ill-formed ciphertexts or keys. **Do not quote this baseline as malicious secure**; compare it with the
semi-honest modes of the other baselines (e.g. `secureml --semi-honest`).

What this baseline does *not* change from HELR: HELR's security estimate is the 2012 formula quoted above; its key generation uses
NTL's generator (here seeded from the OS in every thread, since NTL's default seeding is deterministic); only the keys the circuit
needs are generated (HELR's constructor makes ~5 GB of keys, see the patch), which changes key-generation time and upload size
relative to the paper's numbers but not the evaluation.

`lr_helr_selftest` checks: the polynomial fit against the paper's / HELR's constants, serialization, the HE circuit against the
double-precision reference for degree 3 and 7 over several shapes, and the whole protocol in-process (decrypted weights, measured
bytes against the analytic formulas, parameter mismatch).

## Metrics reported

The report has the same shape as baseline 3's: per party wall and process-CPU time of every phase, payload bytes in each
direction, flights, peak RSS, plus the server's HE operation counts and thread-seconds per operation class (mult, rotate, ...). The parties
exchange their numbers at the end.

- **Offline** = before the evaluation starts: keygen, encryption, one upload flight (keys, data, w0); the server's
  setup and receive. **Online** = the HE evaluation and returning the encrypted weights (one flight). The HE evaluation itself is non-interactive.
- The client's online time is dominated by waiting for the server's evaluation; its own online compute is the decryption (reported separately).
- Bytes are payload bytes (no TCP/IP headers) and equal the analytic formulas (`comm_layout`); the run prints `[measured == analytic]`.
- `--verify` (client, test only) compares the decrypted weights with the plaintext polynomial training, the exact-sigmoid training
  and the fixed-point baselines' training; it runs outside the measured phases.

## Results

@@RESULTS@@

## Files

| file | contents |
|---|---|
| `src/lr_helr.cpp` | the two-party program: offline upload, evaluation, output, report exchange, report, JSON/CSV, `--verify` |
| `src/lr_helr_selftest.cpp` | polynomial fit, serialization, HE circuit vs the double-precision reference (deg 3 and 7, several shapes), full in-process protocol |
| `include/helr_protocol.h` | the protocol as state machines (`HeClient`, `HeServer`), header, analytic sizes |
| `include/helr_circuit.h` | HELR's homomorphic gradient descent (`Evaluator`), input encryption |
| `include/helr_reference.h` | the least-squares polynomial, the circuit's function in double precision, the exact-sigmoid training |
| `include/he_core.h` | parameter choice, OS-seeded randomness, key generation, ciphertext/key (de)serialization, decryption |
| `include/net.h`, `include/metrics.h`, `include/lr_params.h`, `include/lr_reference.h` | copies of baseline 3's channel / metrics / data generator, and baseline 1's fixed-point reference |
| `patches/heaan.patch`, `scripts/` | patch to HELR's sources; `fetch_deps.sh`, `build.sh`, `run_local.sh`, `bench_sweep.sh` |
