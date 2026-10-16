# Baseline 6: Logistic regression on homomorphic encrypted data (Han, Hong, Cheon, Park, IAAI-19)

This baseline implements **"Logistic Regression on Homomorphic Encrypted Data at Scale"** (Han, Hong, Cheon, Park, AAAI
IAAI-19) from the paper, on the HEAAN library with approximate bootstrapping, as a client/server protocol with
malicious-security checks. The client's data and weights are encrypted; the server trains on ciphertexts.

| role | holds | learns |
|---|---|---|
| **client** | the training data (x, y), the initial weights w0 (random for now), the HEAAN secret key | the trained weights (it is the decryptor, as in the paper) |
| **server** | nothing | nothing but the sizes: it never sees a plaintext, a key, or the weights |

Unlike the secret-sharing and garbled-circuit baselines, in an HE protocol the data owner is the only party that can
decrypt, so **the client, not the server, ends up with the model** (that is the paper's setting: "this model can be
decrypted by a decryptor"). The server learns *less* than in the other baselines. The client could of course also compute
the model locally; that is what makes the malicious-security check below possible.

Training defaults match the sibling baselines: **20 iterations, mini-batch 4, learning rate 0.5**, `--d` features including
the bias column, and the same random data for the same `--seed` (quantized values in [-1, 31/32]).

## The algorithm (paper, Sec. 3)

With `z_i = (2 y_i - 1) x_i` the gradient is `sum_i sigma(-z_i . w) z_i`. One iteration on a mini-batch of `B` samples:

```
u_i = z_i . v                              v = Nesterov look-ahead weights
g   = (lr / B) * sum_i z_i * H(u_i)        H(u) = a0 + a1 u + a3 u^3, least-squares fit of sigma(-u) on [-8, 8]
w'  = v + g
v'  = (1 - eta) w' + eta w ;  w = w'       Nesterov accelerated gradient: eta_t = (1 - alpha_t) / alpha_{t+1}
```

`H` is fitted by `fit_sigmoid` (a0 = 0.5, a1 = -0.15012, a3 = 0.001593 for [-8, 8], the paper's polynomial; checked in
the selftest). The weights are bootstrapped every 3 iterations, since an iteration consumes `3 wBits + 2 pBits` bits of
modulus.

**HE-specific design of the paper, all implemented:**
- *Approximate HE (HEAAN)*: real numbers with `wBits = 40` scaling bits, constants at `pBits = 15` / `2 wBits`, rescaling after every
  multiplication, approximate bootstrapping (`logq = wBits + lBits = 45`, `logI = 4`, `logT = 3`).
- *Vectorization*: a mini-batch of `B` samples times a group of `bsz` features fills one ciphertext (slot `bsz·k + l` = feature
  `j·bsz + l` of sample `k`). The inner product is a slot-wise multiplication, `log2 bsz` rotate-and-adds inside each group,
  a 0/1 plaintext mask on the group heads and `log2 bsz` right rotations to replicate; the mini-batch sum is `log2 B`
  rotate-and-adds. The weights are replicated for all `B` sample groups.
- *Parallelized bootstrapping*: the `d` features (padded to a power of two) are split into `cnum` ciphertexts per weight vector,
  so each bootstrapping round has `2·cnum` independent ciphertexts (w and v) that run in parallel, and every ciphertext is
  bootstrapped as a `bsz`-slot ciphertext (the values are replicated, so the sparse-slot bootstrapping is enough and the
  re-interpretation as `B·bsz` slots is free). The data blocks keep the same split, so no ciphertext is ever re-assembled.
- *Mini-batch data blocks*: iteration `t` uses its own block of `B` fresh samples, encrypted in advance (`iters × cnum` data
  ciphertexts), so the whole data set, not only a batch, is uploaded once.

Deviations: (i) the initial weights are the client's private input, so `w` and `v` start as encryptions of `w0` (the paper
starts from 0); (ii) the paper picks a random block per iteration, here iteration `t` uses block `t`; (iii) the packing is
`B = 4` samples per ciphertext (the task's batch size) instead of `N/2 / bsz`, so a ciphertext has few used slots; the cost
per iteration therefore does not depend on how many slots are used, only on `cnum` (see Results).

## Protocol

```
C -> S  HELLO (every public parameter; the packing `cnum` is the client's choice)      S -> C  HELLO
C       Ring + bootstrapping context, secret key, evaluation keys, encryption of the data blocks, w0, v0     [offline]
C -> S  UPLOAD: multiplication key, conjugation key, rotation keys, iters*cnum data ciphertexts, cnum w0, cnum v0
S       for each iteration: inner product, sigmoid gradient, NAG update; every 3 iterations bootstrap w and v    [online]
S -> C  RESULT: the cnum encrypted final weights
C       decrypt, check against its own plaintext training, accept or ABORT
```

The server never needs the public encryption key (it never encrypts), so the public key stays at the client. The only
interaction is the handshake, the upload and the result: **2 round trips**; the evaluation itself is non-interactive.

## Malicious security

HE gives no cheap analogue of MACs, and a fully simulatable malicious-secure *computation* would need zero-knowledge proofs
of well-formed ciphertexts and keys, which are not implemented. What this baseline enforces, and what it does not:

**Against a malicious server (privacy and correctness for the client).**
- The server sees only RLWE ciphertexts and evaluation keys under the client's key (semantic security, and the usual
  circular-security assumption for the evaluation keys); it receives **no output at all**.
- The client **never releases a decryption**. It decrypts the result locally and accepts it only if every weight is within
  `--tol` (default 1e-2) of the weights it computes itself by running the same training in the clear (it owns every input; this costs
  microseconds against the HE evaluation), and the replicas of each weight agree. Otherwise it aborts and prints nothing. So a
  server that returns a wrong result, or that tries to turn the client into a decryption oracle (for instance by sending one of the
  client's own data ciphertexts, modulus-switched to the expected level), is detected, and nothing the client decrypts is ever sent
  back. The only thing a malicious server can learn is the one bit "did the client abort".
- The result ciphertexts must have exactly the expected ring degree, slot count, scale and **modulus level** (the schedule is
  public and deterministic), and every coefficient must be reduced.

**Against a malicious client (the server must not misbehave or leak).**
- The server has no secret and returns nothing that depends on a secret, so the client can learn nothing about the server.
- Everything received is validated before use: all message sizes are derived from the server's own parameters (never from the
  peer), ciphertext headers (ring degree, slots, scale, level) are compared with the schedule before the body is read, every
  coefficient and every NTT residue of every key must be reduced, the number, indices and tags of the evaluation keys must be exactly
  the required set with no duplicates, and the parameters in the handshake must equal the server's.
- **Not covered**: a malicious client can still send ciphertexts or keys that are well-formed in format but not honestly generated;
  the server then returns the result of its (correct) computation on whatever it received, only to that client.

`scripts/test_tamper.sh` runs seven deviations with the real binaries on a tiny insecure ring: server returns a wrong result
(`result`), returns a client data ciphertext at the expected level (`oracle`), returns a ciphertext at a wrong level
(`level`); client sends an out-of-range coefficient (`range`), a wrong-shape ciphertext (`shape`), an unreduced key
(`key`), a missing key (`missing-key`). Every one aborts at the detecting party (exit code 3) and the client prints no weights.

**Security parameters.** The paper's: `lambda = 80` with HEAAN's estimate `N >= logQBoot (lambda + 110) / 3.6`, which gives
`N = 2^16` (logQBoot = 1155), with the sparse ternary secret (hamming weight 64) of HEAAN. That is the paper's level, **not**
128 bits; the estimate is from 2012 and the sparse secret is weaker than a dense one. `--lambda 128` gives `N = 2^17`
(about 4 times the memory and time) and `--logn` overrides the degree (below the bound it is flagged INSECURE and used by the
selftest and the tamper test). The RNG of NTL is seeded from the OS in every thread that samples.

## Install and run (WSL / Linux)

```bash
sudo apt install -y build-essential cmake libntl-dev libgmp-dev libssl-dev
cd baselines/helr_han
bash scripts/build.sh
./build/lr_helr_han_selftest --quick      # polynomial, parameters, wire format (seconds)
./build/lr_helr_han_selftest              # + the homomorphic circuit vs the plaintext reference (insecure ring, ~40 s)
bash scripts/test_tamper.sh               # seven deviations, real binaries, tiny insecure ring
bash scripts/run_local.sh --d 3 --iters 4 --logn 12      # the whole flow in seconds (INSECURE ring, test only)
bash scripts/run_local.sh --d 5                           # real parameters (N = 2^16): read the memory note below
```

By hand: `./build/lr_helr_han server --port 12345 --d 5` and `./build/lr_helr_han client --host <server-ip> --port 12345 --d 5`.
Options (`--help`): `--d --iters --batch --lr --seed --no-bias --zero-init --qbits --range`; HE `--wbits --pbits --iter-per-boot --lambda --logn
--cnum`; `--threads`, `--tol`; `--json --csv --quiet`; test-only `--tamper`.

**HEAAN.** `third_party/HEAAN` is the HEAAN snapshot with bootstrapping that the authors' code uses (kimandrik/HEAAN v1.1, with
the authors' `PRIME_BIT_SIZE` change), CryptoLab's non-commercial licence (`third_party/HEAAN/LICENSE`). One patch: a
`Scheme(Ring&)` constructor that generates no keys, so that the server can hold the client's evaluation keys; the other files
are unmodified apart from deleting the demo `HEAAN.cpp` / `TestScheme.*`.

**All CPU cores.** `--threads` defaults to all hardware threads and sets NTL's thread pool, which is used at two levels: one
task per ciphertext (the `cnum` feature groups of each step, the `2 cnum` bootstrappings, the encryption of the data blocks, the
validation of the keys) and, inside each ring operation, the NTT over the RNS primes. `cnum` defaults to the smallest power of two
with `2 cnum >= threads` (at least 2 features per ciphertext), so the bootstrapping round keeps all threads busy; the server adopts
the client's `cnum`. HEAAN's inner parallelism alone is weak (one bootstrapping: 34 s on 1 thread, 14 s on 12), which is why the
paper splits the features across ciphertexts.

**Memory.** The evaluation keys alone are ~1.4 GB per party (17 keys of ~84 MB at N = 2^16); every parallel bootstrapping needs
several hundred MB more, and the encrypted data is `iters × cnum × 9 MB`. With both parties on one machine give each party about
half of it: on a 7 GB machine `--threads 2` to `3` fits, higher thread counts start to swap.

## Metrics

Per party, per phase: wall time, process CPU time (all threads), payload bytes sent / received, flights. Phases are
`offline/*` (everything before the evaluation: HE context and bootstrapping tables, key generation, encryption, upload on the
client; context, key and data reception on the server), `online/*` (the evaluation on the server, the wait and the decryption
on the client) and `other/*` (handshake, the client's own plaintext training used for the check). Time spent blocked on the other
party (`offline/wait_client`, `online/wait_server`) is **not computation**: the tables show it separately and the
stage totals report **wall (compute)** = wall minus waiting. The SUMMARY gives, for client and server separately and
**combined** (sum of the two parties' computation), the offline / online / other computation and cpu time, the communication in
each direction per stage, the flights and the end-to-end elapsed time. A flight is a maximal run of messages in one direction; a
round trip is 2.

## Results

One run, this machine: WSL2 with 12 hardware threads and 7 GB RAM (the real-parameter run swaps), localhost, both parties on
the machine with 6 threads each. `d = 5` (padded to 8, `cnum = 4` ciphertexts of 2 features), iters = 20, batch = 4, lr = 0.5,
**N = 2^16, logQ = 495, logQBoot = 1155** (the paper's parameters), 6 bootstrapping rounds of 8 ciphertexts.

| | client | server | combined |
|---|---|---|---|
| offline: compute wall / cpu | 35.9 s / 163.7 s (HE context 0.2 s, key generation 10.6 s, encryption 19.1 s, upload 6.0 s) | 6.5 s / 6.9 s (context 0.2 s, key reception 2.1 s, data reception 4.2 s) | 42.4 s / 170.6 s |
| online: compute wall / cpu | 0.7 s / 0.8 s (receive, decrypt, verify) | 15.0 min / 74.3 min (inner products 59 s, sigmoid gradients 101 s, NAG updates 2.7 s, **bootstrapping 12.2 min**) | 15.0 min / 74.3 min |
| other: compute | ~0 | ~0 | ~0 |
| waiting for the other party | 15.0 min online | 29.7 s offline | |
| total compute wall / cpu | 36.6 s / 164.5 s | 15.1 min / 74.4 min | 15.7 min / 77.2 min |
| peak RSS | 2.9 GiB | 5.7 GiB | |

- **Communication**: offline client->server **2.10 GiB** (keys 1.39 GiB, data 732 MiB, w0 and v0), online server->client
  **12.5 MiB** (the 4 result ciphertexts), handshake 80 B each way; total 2.12 GiB.
- **Rounds**: 4 flights = **2 round trips** (handshake, upload + result); the evaluation is non-interactive.
- **Accuracy**: max |HE weights - plaintext polynomial training| = 3.2e-3 (client tolerance 1e-2), replicas agree to 5e-9;
  0.18 away from training with the exact sigmoid (the cost of the cubic fit with random data and weights).
- The bootstrapping (12.2 of the 15.0 minutes, in 6 rounds) dominates, as in the paper. The per-iteration work is small by comparison.
  With 6 threads and 8 ciphertexts per round, memory pressure (swapping) made the run slower than the CPU time alone would
  suggest; a machine with more RAM would run it faster. HEAAN v1.1 and the sparse-slot bootstrapping are used as the authors do.
- Cost scales with `cnum`, not with the number of used slots: every ciphertext costs a full `N = 2^16` operation however few
  slots it uses. `d = 5` is the worst case for the packing (4 ciphertexts for 5 features).

## Files

| file | contents |
|---|---|
| `src/lr_helr_han.cpp` | the two-party program: handshake, upload, evaluation, result, verification, report, JSON / CSV, `--tamper` |
| `src/lr_helr_han_selftest.cpp` | polynomial fit, parameters, wire format and validation, the HE circuit against the plaintext reference |
| `include/params.h` | parameters (the paper's), packing, the client's data, the least-squares sigmoid, the plaintext reference, CLI |
| `include/circuit.h` | the homomorphic NAG iteration and bootstrapping (`Evaluator`), packing, decryption |
| `include/he.h` | randomness, thread pool, evaluation-key set, (de)serialization and validation, the level schedule |
| `include/net.h`, `include/metrics.h` | the TCP channel with exact counters and flights; per-phase metrics |
| `third_party/HEAAN/` | the HEAAN library (one patch, see above) |
