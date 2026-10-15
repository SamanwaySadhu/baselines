# Baseline 3: CryptoNN (functional encryption, semi-honest with a trusted authority)

This baseline trains logistic regression the way **CryptoNN** does it (Xu, Joshi, Li, *CryptoNN: Training Neural
Networks over Encrypted Data*, ICDCS 2019, arXiv:1904.07303):
- the server holds the model **in the clear**;
- the steps that touch the training data use **functional encryption**:
  - **FEIP**, the DDH inner-product scheme of Abdalla, Bourse, De Caro and Pointcheval (PKC 2015), paper Sec. II-B;
  - **FEBO**, the paper's FE for basic operations (Sec. III-B).

The group is NIST P-256 (128-bit security), through the OpenSSL 3 libcrypto EC API.

| role | holds | learns |
|---|---|---|
| **client** = data owner **and** CryptoNN authority | the training data (x, y), the initial weights w0 (random for now), every master secret key | every w_t, yhat_i and e_i (from the key requests) |
| **server** = model holder | the model, in the clear (it receives w0 at setup) | the trained weights and every intermediate value listed under [Leakage](#security-model-and-leakage) |

CryptoNN has three roles: the authority, the clients and the server. The two-party setting merges the authority into
the client. The client generates every key and answers the server's function-key requests. Nobody else holds a master
secret key.

The training parameters are those of the FE code and of the other two baselines:
- **6 quantization bits** (`QUANTIZATION_BITS=6`), **batch = 4**, **learning rate = 0.5**;
- **20 iterations**, with the bias as the last feature (`x = MAXQ`);
- `--d` is the number of features, **including** the bias column.

`--iters`, `--qbits`, `--batch`, `--lr`, `--no-bias`, `--zero-init` and `--seed` change the rest. The data generator is
baseline 1's, so the same `--seed` gives the same data and initial weights in all three baselines.

## What is computed

Exactly the fixed-point training of baseline 1 (`include/lr_reference.h`, a copy of `gc_ag2pc/include/lr_reference.h`).
Values are qbits-bit integers in `[MINQ, MAXQ] = [-32, 31]` at scale `2^f`, `f = 5`. Labels are `y in {0, 1}`, with the
label value `MAXQ * y`. `K = batch / lr = 8`. Per iteration t, for the B = 4 samples i of the batch:

```
zfull_i = sum_{j<d_data} w_j x_ij              FEIP fwd (server decrypts with sk_w)
          + w_bias * MAXQ                      bias column is public: plaintext
z_i     = zfull_i >> f                         truncation (floor), plaintext
yhat_i  = min(floor(2^f sigmoid(z_i/2^f)), MAXQ)   exact quantized sigmoid, plaintext at the server
e_i     = yhat_i - MAXQ y_i                    FEBO '-': decrypts to MAXQ y_i - yhat_i, the server negates it
gfull_j = sum_i e_i x_ij,  j < d_data          FEIP bwd over the encrypted feature column (server decrypts with sk_e)
gfull_bias = MAXQ * sum_i e_i                  plaintext
w_j     = (K w_j - (gfull_j >> f)) / K         C++ '/', i.e. toward zero; plaintext
```

Everything is exact integer arithmetic. The decryptions recover the exact inner products, so the trained weights equal
`reference_train` **bit for bit**. `--verify` checks this on every real run, and `lr_cryptonn_selftest` checks it after
every iteration for 68 configurations, plus worst-case data. With `--no-bias`, all d columns are client data
(`d_data = d`) and there are no plaintext bias terms.

This is the same function as baseline 1 (the exact sigmoid, the FE code's update). Baseline 2 (MD-ML) uses a
piecewise-linear sigmoid with probabilistic truncation.

## The CryptoNN mapping

The paper's Algorithm 2 runs the training at the server. It uses the secure matrix computation of Algorithm 1 only
where the data appears. For logistic regression, a one-layer network, those places are:

| step | paper | here |
|---|---|---|
| secure feed-forward `w . x_i` | FEIP: the data owner encrypts x_i, the server asks the authority for `sk_w` and decrypts `g^<w,x_i>` | FEIP fwd, eta = d_data, one ciphertext per sample |
| secure evaluation `yhat - y` | FEBO: the label is encrypted, the server asks for the '-' key for its own `yhat_i` | FEBO, one ciphertext per sample, '-' key per sample |
| first-layer weight gradient `X^T e` | FEIP over the feature columns, function vector e | FEIP bwd, eta = B, one ciphertext per (t, column j) |
| sigmoid, truncations, update | plaintext at the server | plaintext at the server (`qsigmoid_ref`, `>> f`, `/ K`) |

FEBO implements all four operations ('+', '-', '*', '/'), and `[4]` of the selftest tests them; the protocol needs only '-'.

**The schemes**, written multiplicatively as in the paper; the code is additive, see `include/ec_group.h`:
- **FEIP**:
  - Setup: `s <- Z_q^eta`, `mpk = (g, h_i = g^{s_i})`.
  - KeyDerive: `sk_y = <y, s> mod q`.
  - Encrypt: `ct0 = g^r`, `ct_i = h_i^r g^{x_i}`.
  - Decrypt: `g^{<x,y>} = prod ct_i^{y_i} / ct0^{sk_y}`, then a bounded discrete log.
- **FEBO**:
  - Setup: `h = g^s`.
  - Encrypt: `cmt = g^r`, `ct = h^r g^x`.
  - '-' key: `cmt^s g^{y}`; decrypt: `ct / sk = g^{x - y}`.
  - The other operations are in `include/fe_bo.h`.

**Decisions in this two-party version:**
- **The client is the authority.** It holds the three master secret keys and derives every function key on request.
- **w0 goes to the server in the clear** at setup. CryptoNN's server must know w to decrypt `<w, x_i>`. This is
  inherent to the scheme, and it is reported as leakage.
- **Three independent key sets:** FEIP fwd (eta = d_data), FEIP bwd (eta = B) and FEBO.
- **mpk is not sent.** Decryption does not use it, and the client both encrypts and derives keys.
- **Encryption uses the public key, as in the paper**: one double-scalar multiplication per component.
  `--encrypt-with-msk` encrypts with the master secret key instead: `ct_i = (r s_i + x_i) g`, one fixed-base
  multiplication per component. The ciphertext distribution is the same, and it is about 3.8 times faster at
  d = 1000. It is possible only because the data owner is also the authority.
- **Encodings:** points are compressed (33 bytes) and scalars are 32 bytes. Every received point is validated on the
  curve; P-256 has cofactor 1, so this is a complete subgroup check.

### Encryption layout (the client's offline upload, one flight)

| section | count | size each | d = 5 (d_data = 4) | d = 100 | d = 1000 |
|---|---|---|---|---|---|
| header (magic, d, iters, batch, qbits, bias, lr) | 1 | 32 B | 32 B | 32 B | 32 B |
| w0 (int32) | d | 4 B | 20 B | 400 B | 4000 B |
| FEIP fwd: `x_i` of every sample | iters x B | (d_data+1) x 33 B | 80 x 165 B | 80 x 3300 B | 80 x 33000 B |
| FEIP bwd: column `(x_0j..x_{B-1}j)` of every batch | iters x d_data | (B+1) x 33 = 165 B | 80 x 165 B | 1980 x 165 B | 19980 x 165 B |
| FEBO: label value `MAXQ y_i` of every sample | iters x B | 2 x 33 = 66 B | 80 x 66 B | 80 x 66 B | 80 x 66 B |
| **total** | | | **31732 B (30.99 KiB)** | **596412 B (582.43 KiB)** | **5946012 B (5.67 MiB)** |

```
offline client->server = 32 + 4d + iters*B*(d_data+1)*33 + iters*d_data*(B+1)*33 + iters*B*66     (server->client: 0)
                       ~ 33 * iters * (2B+1) * d  for large d   (both FEIP encodings of the data: rows and columns)
```

### Message flow per iteration (online)

| round trip | server -> client | client -> server | server work |
|---|---|---|---|
| R1 | `w_0..w_{d_data-1}`: 4·d_data B | `sk_w = <w, s_fwd>`: 32 B | B FEIP decryptions (multi-exp of d_data+1 points) + B dlogs, bound `2^f sum_j abs(w_j)`; z, yhat in the clear |
| R2 | `yhat_0..yhat_{B-1}`: 4B B | B FEBO '-' keys: 33B B | B FEBO decryptions (one point subtraction each) + B dlogs, bound `2 MAXQ` |
| R3 | `e_0..e_{B-1}`: 4B B | `sk_e = <e, s_bwd>`: 32 B | d_data FEIP decryptions (multi-exp of B+1 points) + d_data dlogs, bound `2^f sum_i abs(e_i)`; update |

```
online per iteration:  server->client 4*(d_data + 2B) B,   client->server 32 + 33B + 32 = 196 B (B = 4)
rounds:                offline 1 flight; online 3 round trips per iteration = 60 round trips (120 flights) for 20 iterations
```

The client answers every request without waiting for anything else. The server holds the trained weights at the end,
and nothing more is sent. The client checks each request: lengths, `yhat` in `[0, MAXQ]`, and `e = yhat - MAXQ y`,
which it can compute itself. Both parties check that `d / iters / batch / qbits / lr / bias` agree: the server checks
the upload header, and both sides check again when the reports are exchanged. A mismatch aborts with a message.

**Discrete logs.** Every decryption ends in `g^v` with a small v. The server solves it by baby-step giant-step (`include/bsgs.h`).
- The table holds j·g for `j < 2^b`. Since j·g and -j·g share the x-coordinate, one lookup covers `[-(2^b - 1), 2^b - 1]`.
  Giant steps go outward from 0.
- **The bounds are exact**, because `abs(x) <= 2^f`. A solve can fail only if the protocol is broken. A failure aborts the run
  (`DlogError`), it is never replaced by a wrong value:
  - the selftest shows that a capped bound and a corrupted key are both detected;
  - `--test-bound-cap 100` shows it on a real run: `[server] ERROR: bounded discrete log FAILED for forward <w, x_i> ...`,
    and both parties exit 1.
  - On worst-case data (all features MINQ, all weights MAXQ), the forward value reaches the bound exactly.
- **Table size** (`--bsgs-bits auto`): the smallest b in [12, 20] whose single lookup covers the worst-case forward value of
  the *initial* weights, `2^(2f) d_data`. This gives 13, 14, 16, 17, 19 and 20 bits for d = 5, 10, 50, 100, 500 and 1000.
  The weights grow during training. A larger value costs about 1 µs per extra giant step instead of a bigger table:
  the 20-bit table takes about 1 s to build and 24 MiB.

**Threads** (`--threads N`, default 1). The client's encryption and the server's decryptions, discrete logs, upload
validation and table build are split by index over N std::threads, as the paper parallelizes decryption. The results
are identical for any N; this is checked by the selftest and by the d = 10 `--threads 4` run.

## Security model and leakage

- **Semi-honest, with a trusted authority.** The authority, here the client, is trusted to derive only the keys the protocol needs.
  The server is trusted to follow the protocol. **There is no malicious security**, unlike the other two baselines:
  - baseline 1 is authenticated garbling;
  - baseline 2 is SPDZ2k with MAC checks.

  A malicious server could simply ask for keys of other function vectors.
- **What the server learns in the clear:**
  - w0 and every w_t;
  - per sample: `z_i` (and the inner product `<w, x_i>`), `yhat_i`, the label `y_i` (FEBO gives `MAXQ y_i - yhat_i`,
    and `yhat_i` is its own), and `e_i`;
  - per iteration: the gradient inner products `gfull_j = <e, x_.j>` for every data column;
  - the trained weights.
  Over many iterations these equations can reveal much of the data. CryptoNN accepts this by design: the model is the server's.
- **What the client (authority) learns:** every w_t (R1), every `yhat_i` (R2) and every `e_i` (R3), from the key requests.
- **Comparison with the FE protocol under evaluation** (`Functional_Encryption_for_Logistic_Regression`): that protocol
  uses **function-hiding** inner-product FE (Kim et al.), so a function key does not reveal its vector, and the weights
  stay hidden from the decryptor. CryptoNN's FEIP is not function-hiding. The decryptor must know y (the weights, the
  errors) to decrypt, so CryptoNN cannot hide the model. This is the main qualitative difference to keep in mind
  when comparing the numbers.

## Install on WSL (Ubuntu 22.04/24.04)

```bash
# 1. toolchain: C++20 compiler, CMake >= 3.16, OpenSSL 3 headers (nothing else; no third-party sources)
sudo apt update
sudo apt install -y build-essential cmake libssl-dev

# 2. build (lr_cryptonn, lr_cryptonn_selftest into ./build; BUILD_DIR=... overrides)
cd baselines/cryptonn
bash scripts/build.sh

# 3. offline tests (--quick: ~0.3 s, 1179 checks; full: ~6 s, 3857 checks)
./build/lr_cryptonn_selftest --quick

# 4. both parties locally, with the server checking the result (test only)
bash scripts/run_local.sh --d 5
```

By hand, in two terminals or on two machines:

```bash
./build/lr_cryptonn server --port 23456 --d 5                     # builds the BSGS table, then listens
./build/lr_cryptonn client --host <server-ip> --port 23456 --d 5   # encrypts, then connects (retries for 120 s)
```

Both parties must pass the same training flags. The options are:
- training: `--d --iters --batch --qbits --lr --seed --no-bias --zero-init`
- protocol: `--host --port --threads N --bsgs-bits N|auto` (server), `--encrypt-with-msk` (client)
- output: `--json FILE`, `--csv FILE` (server), `--quiet`
- test only: `--verify` (server: regenerates the client's data from `--seed`), and `--test-bound-cap N` (server)

Run `--help` for details. The protocol uses one TCP port.

The scripts:
- `run_local.sh`: `PORT=... THREADS=... VERIFY=0|1`. It writes `results/{server,client}.log`, `results/last_run.json`
  and appends a row to `results/results.csv`.
- `bench_sweep.sh`: `D_LIST="5 10 50 100" REPS=3 [BASE_PORT=...] bash scripts/bench_sweep.sh` writes `results/sweep.csv`.
  Each run is checked with `--verify`.

## Metrics reported

Both parties time every phase: wall time, and process CPU time summed over all threads. Payload bytes and flights come
from the channel's counters (`include/net.h`, without TCP/IP headers). After all measurements, the parties exchange their
numbers over the same socket, and each prints the combined report.

| party | stage | rows |
|---|---|---|
| client | **offline** | keygen (setup of the 3 key sets), encryption of all ciphertexts, send of the upload |
| | **online** | key derivation (compute only); the rest of the online wall is messages and waiting for the server |
| server | **offline** | BSGS baby-step table (input independent), receive of the upload, deserialization + point validation |
| | **online** | FEIP fwd decryption (multi-exp), dlog fwd, FEBO decryption, dlog FEBO, FEIP bwd decryption, dlog bwd, plaintext LR (sigmoid, update), messages + waiting |

The client encrypts **before connecting**, and the server builds its table before accepting. So neither offline phase
includes waiting for the other party. The client's online clock starts when the first request arrives. The time
between its upload and that request, while the server validates the upload, is printed separately ("idle") and counted
in neither phase.

The report also gives:
- **communication**: each direction, offline and online, the upload broken down by section, and a check
  against the analytic formulas above (`[measured == analytic]`);
- **rounds**: offline flights, online flights and round trips;
- peak RSS per party; the BSGS table; the number of discrete logs of each kind and the largest |v| / bound;
- the leakage summary.

`--json` has every phase of both parties and the server's per-iteration wall times. The `--csv` row has
`d, iters, batch, qbits, lr, threads, bsgs_bits`, every timing, the bytes in each direction offline and online, flights,
round trips, peak RSS and `verified`.

"Offline" here means *before the interactive training*. The client's encryption depends on its data but not on the
model. The server's table depends on nothing.

## Results

This laptop: Intel Core 7 150U (2 performance + 8 efficiency cores, 12 threads), WSL2 Ubuntu 24.04, OpenSSL 3.0.13,
localhost. Defaults: 20 iterations, batch 4, qbits 6, lr 0.5, 1 thread per party, public-key encryption. Each cell is
the median of 3 runs from `bench_sweep.sh`. Every run passed `--verify`, and every byte count equalled the analytic
formula.

| d | 5 | 10 | 50 | 100 | 500 | 1000 |
|---|---|---|---|---|---|---|
| client offline (ms): keygen + encrypt + send | 32.8 | 69.6 | 353 | 704 | 3419 | 8286 |
| client online (ms, 20 iterations) | 15.9 | 21.6 | 71.0 | 116 | 570 | 1205 |
| server offline (ms): BSGS table + recv + validate | 16.5 | 33.4 | 149 | 276 | 1309 | 3086 |
| server online (ms, 20 iterations) | 16.4 | 22.2 | 73.7 | 121 | 598 | 1266 |
| offline client->server | 30.99 KiB | 60.01 KiB | 292.20 KiB | 582.43 KiB | 2.84 MiB | 5.67 MiB |
| offline server->client | 0 | 0 | 0 | 0 | 0 | 0 |
| online client->server | 3.83 KiB | 3.83 KiB | 3.83 KiB | 3.83 KiB | 3.83 KiB | 3.83 KiB |
| online server->client | 960 B | 1.33 KiB | 4.45 KiB | 8.36 KiB | 39.61 KiB | 78.67 KiB |
| rounds (offline flights + online round trips) | 1 + 60 | 1 + 60 | 1 + 60 | 1 + 60 | 1 + 60 | 1 + 60 |
| BSGS table bits (auto) | 13 | 14 | 16 | 17 | 19 | 20 |
| peak RSS client / server (MiB) | 8.0 / 8.4 | 8.1 / 8.9 | 8.2 / 12.4 | 8.6 / 16.9 | 11.4 / 49.9 | 14.9 / 91.9 |

Where the time goes (d = 1000, median run):
- **client offline**: public-key encryption, 8.3 s. That is about `iters * 2B * d_data` = 160K double-scalar multiplications
  (~50 µs each, in an `EC_POINT_mul`) plus 20K fixed-base ones. Keygen takes 8 ms.
- **server offline**: BSGS table 1.96 s and upload validation 1.12 s, which decompresses and checks 180K points.
  The table took ~1 s on an idle machine; here it is built while the client encrypts on the same laptop.
- **server online**, 1.27 s:
  - FEIP bwd decryption 920 ms (20 x 999 multi-exps of 5 points) and dlog bwd 257 ms;
  - FEIP fwd decryption 61 ms (20 x 4 multi-exps of 1000 points with 6- to 10-bit coefficients, by a bucket method);
  - FEBO under 3 ms.
- **client online**: key derivation 13 ms in total; the rest is waiting for the server.
- **The backward pass dominates online**: it costs one decryption per feature column per iteration. The forward pass
  costs one per sample.

Variants, d = 1000, one run each:

| configuration | client offline | client online | server offline | server online |
|---|---|---|---|---|
| defaults (1 thread, public-key encryption) | 8286 ms | 1205 ms | 3086 ms | 1266 ms |
| `--encrypt-with-msk` | 2191 ms | 1316 ms | 3285 ms | 1384 ms |
| `--threads 12` | 5214 ms | 453 ms | 1947 ms | 478 ms |
| `--threads 12 --encrypt-with-msk` | 1488 ms | 411 ms | 1799 ms | 432 ms |

- **Thread scaling is limited on this laptop.** It has 2 performance and 8 efficiency cores. In a local run, both
  parties share them, and the server's table build overlaps the client's encryption. Expect better scaling on a
  server-class machine.
- **Communication** does not depend on the thread count or the encryption mode.
- **Online time on localhost** is mostly server computation plus 60 round trips. On a WAN, add about 60 × RTT.
  Raw localhost RTT is about 25 µs here.

Compared with the other baselines, CryptoNN is **much cheaper**, because it gives much **less security**: the server sees
the model and every intermediate value, and there is no malicious security. Its cost is dominated by the one-time
encryption of the data, about 33·iters·(2B+1)·d bytes. The online phase is small: 196 B per iteration from the client
and `4(d_data + 2B)` B from the server, in 3 round trips per iteration.

## Files

| file | contents |
|---|---|
| `src/lr_cryptonn.cpp` | the two-party program: offline upload, 3 round trips per iteration, report exchange, report, JSON/CSV, `--verify` |
| `src/lr_cryptonn_selftest.cpp` | group / FEIP / FEBO / BSGS tests, in-process bit-exact training runs (68 configurations + worst-case data), dlog-bound safety |
| `include/cryptonn_protocol.h` | the protocol as message-level state machines (`CnClient`, `CnServer`), shared by the program and the selftest; upload layout, header check |
| `include/fe_ip.h` | FEIP (Abdalla et al. PKC 2015): setup, encrypt (mpk or msk), key derivation, decryption to `g^<x,y>`, serialization |
| `include/fe_bo.h` | FEBO (CryptoNN Sec. III-B): '+', '-', '*', '/', serialization |
| `include/bsgs.h` | bounded discrete log by baby-step giant-step (thread-safe, signed values, exact bounds) |
| `include/ec_group.h` | P-256 context, RAII `Scalar` / `Point`, compressed encoding with validation, multi-scalar multiplication |
| `include/net.h` | TCP channel with exact payload byte and flight counters |
| `include/metrics.h` | per-phase wall / CPU / traffic accounting, peak RSS, byte formatting |
| `include/lr_reference.h`, `include/lr_params.h` | copies of baseline 1's fixed-point specification and data generator (namespace `lrcn`) |
| `scripts/` | `build.sh`, `run_local.sh`, `bench_sweep.sh` |
