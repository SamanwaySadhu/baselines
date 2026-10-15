# Baseline 3: CryptoNN (functional encryption)

This baseline trains logistic regression the way **CryptoNN** does (Xu, Joshi, Li, *CryptoNN: Training Neural Networks
over Encrypted Data*, ICDCS 2019). The server trains a model that it holds **in the clear**. The data reaches it only
**functionally encrypted**, and a trusted authority gives it function keys for exactly the inner products and
differences that training needs. The schemes are FEIP, the DDH-based inner-product FE of Abdalla et al. (PKC 2015), and
the paper's FEBO for basic operations. Both run over NIST P-256 (OpenSSL 3 libcrypto). There are no third-party
sources.

In this two-party setting, the FE authority is merged into the client:

| role | holds | learns |
|---|---|---|
| **client** = data owner + FE authority | the training data (x, y), the initial weights w0 (random for now), every master secret key (`msk_ff`, `msk_bp`, `msk_bo`) | every function input it is asked to key: w, ŷ, e |
| **server** = model trainer | the ciphertexts, the model w in the clear | w0, every intermediate w, z, ŷ, e, g, and therefore the labels; ends with the trained model |

The training is the same quantized fixed-point LR as the sibling baselines (`include/lr_params.h` and
`include/lr_reference.h` are copies of the GC baseline's). The same `--seed` gives the same data and initial weights in
all three baselines. The defaults are qbits = 6 (values in [-32, 31]/32), batch = 4, lr = 0.5 (K = batch/lr = 8),
20 iterations, and d features including the bias column. The protocol's result equals `reference_train` **bit for
bit**. The selftest checks this, and so does `--verify` on a real run, after every iteration.

## Protocol

Notation: f = qbits − 1, MAXQ = 2^f − 1, K = 2^k_shift = batch/lr. Batch t holds rows i < batch, and the label value is
MAXQ·y.

**Preprocessing** (paper Algorithm 1, *pre-process-encryption*):

1. **Handshake.** The client sends a 40-byte public-parameter header (d, iters, batch, qbits, lr/k_shift, bias and
   send-model flags; *not* the seed). The server replies with one status byte. If the parameters differ, both parties
   stop with a clear error and a non-zero exit.
2. **Client setup and encryption.** The client runs three setups: `FEIP.Setup(η = d)` → `msk_ff` (sample rows),
   `FEIP.Setup(η = batch)` → `msk_bp` (the per-batch feature columns), and `FEBO.Setup` → `msk_bo` (labels). It then
   encrypts with the **public** keys, as a data owner would:
   - `ct_ff[t][i] = FEIP_ff.Enc(x_{t,i,:})`: d components
   - `ct_bp[t][j] = FEIP_bp.Enc(x_{t,:,j})`: batch components
   - `ct_bo[t][i] = FEBO.Enc(MAXQ·y_{t,i})`

   `--enc-with-msk` switches to the authority's equivalent shortcut `encrypt_with_msk`, which gives the same ciphertext
   distribution and is about 5× faster (d = 50, 1 thread: 76 ms instead of 424 ms). The client parallelizes encryption over `--threads`. All ciphertexts and w0
   (d int32) go to the server in **one flight**.

   **w0 must be revealed.** In CryptoNN the server holds the model in plaintext: it requests function keys for w, and
   FEIP decryption takes y = w as an input. mpk is not sent, because the server never encrypts and decryption does not
   use it.
3. **Server preprocessing.** The server builds the BSGS discrete-log table. This is input-independent and overlaps the
   client's encryption. It then receives and deserializes the ciphertexts, which validates every point.

**Online, iteration t** (paper Algorithm 2, the LR case): 3 round trips.

| step | server → client | client → server | server computes |
|---|---|---|---|
| a. secure feed-forward | w (4d B) | `sk_w = FEIP.KeyDerive(msk_ff, w)` (32 B) | `zfull_i = dlog(FEIP.Dec(ct_ff[t][i], sk_w, w))`, bound Σ\|w_j\|·2^f; `z_i = zfull_i >> f`; `ŷ_i = qsigmoid(z_i)` |
| b. secure evaluation (FEBO '−') | ŷ (4·batch B) | per i `FEBO.KeyDerive(msk_bo, cmt_{t,i}, −, ŷ_i)` (33 B each) | `MAXQ·y_i − ŷ_i = dlog(FEBO.Dec)`, bound MAXQ; `e_i = ŷ_i − MAXQ·y_i` |
| c. secure back-propagation | e (4·batch B) | `sk_e = FEIP.KeyDerive(msk_bp, e)` (32 B) | `gfull_j = dlog(FEIP.Dec(ct_bp[t][j], sk_e, e))`, bound Σ\|e_i\|·2^f; `g_j = gfull_j >> f`; `w_j = (K·w_j − g_j)/K` (C++ `/`) |

- The server parallelizes decryption and discrete logs over `--threads`, as the paper does: over the batch rows in (a)
  and (b), and over the d columns in (c).
- A failed discrete log is a hard error.
- **Range checks.** The client is the authority, so it checks every function input before it derives a key:
  |w_j| ≤ 2^f + t·⌊batch·MAXQ/K⌋, ŷ ∈ [0, MAXQ], |e| ≤ MAXQ. The server checks that the decrypted label is 0 or MAXQ,
  and that the updated weights stay inside the same public bound.
- **FEBO commitments.** The client keeps the commitments `cmt_{t,i}` it generated, so a FEBO key request carries only ŷ.
  In the paper, the server sends `cmt` along with the request.

**Output.** The server ends holding the trained model, which is CryptoNN's semantics, and prints it. With `--send-model`,
it also sends w to the client: 4d bytes, counted in a separate **output** stage.

**Report exchange** (not counted anywhere). The client sends its measurements to the server, and the server prints the
combined report.

## What each party learns (semi-honest)

- **The server** learns w0 and every intermediate w, z, ŷ, e and g. Since e = ŷ − MAXQ·y and it knows ŷ, it also learns
  **every label**. This is inherent to CryptoNN's LR: the server holds the model in the clear and decrypts these values
  by design.
  - The paper hides labels with a random label mapping, but that mapping is for one-hot multi-class labels. It is
    **not applied here** for binary labels, because a permutation of {0, 1} would not hide anything once e is
    decrypted.
  - The server never receives x or y in the clear. It sees x only through the inner products ⟨w, x_i⟩ and ⟨e, x_{:,j}⟩
    it is given keys for.
- **The client (authority)** learns every w, ŷ and e it is asked to key, because the authority is merged into the
  client.
- **Not covered:** a malicious server. It could request keys for unit vectors and read features directly. The
  authority's range checks do not prevent that, which is also the case in the paper. The baseline is semi-honest, like
  the paper.

## Building and running

```bash
bash scripts/build.sh                  # needs a C++20 compiler, CMake >= 3.16, OpenSSL 3 headers (libssl-dev)
./build/lr_cryptonn_selftest           # ~1 s; must print "selftest: ... 0 failure(s): PASS"
scripts/run_local.sh --d 5             # both parties on this machine; the server checks the result (--verify)
THREADS=4 scripts/run_local.sh --d 100 --iters 20
```

By hand, in two terminals or on two machines:

```bash
./build/lr_cryptonn server --port 12345 --d 5            # listens
./build/lr_cryptonn client --host <server-ip> --port 12345 --d 5
```

Both parties must pass the same training flags; the handshake checks them. The options are:

- **Training:** `--d` (features incl. the bias column), `--iters` (default 20), `--batch` (default 4), `--qbits`
  (default 6), `--lr` (default 0.5), `--no-bias`, `--send-model`, `--seed`, `--zero-init`.
- **Protocol:**
  - `--threads N` per party (default 1)
  - `--enc-with-msk` (client)
  - `--dlog-bits N` (server). The default covers max(d, batch)·2^(2f) in one lookup, capped at 22 bits (4M entries,
    96 MiB). Larger values are found by giant steps.
- **Output:**
  - `--verify` (server, test only): it regenerates the client's data and w0 from `--seed`, checks the received w0 and
    the weights after every iteration bit for bit, prints `verify: PASS|FAIL`, and exits non-zero on FAIL.
  - `--json FILE`
  - `--csv FILE` (server). It appends a row and writes the header for a new file. It refuses a file whose header differs.
  - `--quiet`

To sweep dimensions: `D_LIST="5 11 13 31" REPS=2 OUT=results/sweep.csv bash scripts/bench_sweep.sh`. Use `THREADS=4` for 4
threads per party, and `VERIFY=0` to skip the check.

## Cost reporting

Each party measures itself with `include/metrics.h`, in contiguous phases named `stage/sub-phase`. The server prints both
parties' tables.

| party | stage | sub-phases |
|---|---|---|
| client | **preprocessing** | `handshake`, `wait_ack`, `fe_setup` (3 setups), `encrypt`, `serialize`, `upload` (sending the one upload flight; it can block while the server builds its table) |
| client | **online** | `keyderive_ff`, `keyderive_eval`, `keyderive_bp`, `send`, `wait` (blocked on the server) |
| server | **preprocessing** | `wait_header`, `handshake`, `dlog_table`, `wait_upload` (mostly the client's encryption time), `deserialize` (point decompression and validation) |
| server | **online** | `decrypt_ff` / `decrypt_eval` / `decrypt_bp` (decryption + discrete logs), `sigmoid`, `update`, `send`, `wait` (blocked on the client's keys) |

- **Columns.** For each stage the report gives three numbers: **wall** (elapsed at that party), **compute** (wall minus
  the `wait*` sub-phases) and **cpu** (that process's CPU time summed over all its threads, so it is about N × wall with
  N busy threads). It also gives the **amortized online cost per iteration** (online / iters).
- **Communication.** Payload bytes are counted exactly by `net.h`, per direction and per stage. They are printed in B,
  KB/MB (decimal) and KiB/MiB (binary), next to the bytes **expected from closed-form formulas**. The program warns on a
  mismatch, and the selftest asserts equality.
  - preprocessing c→s = 40 + iters·batch·(d+1)·33 + iters·d·(batch+1)·33 + iters·batch·66 + 4d; s→c = 1
  - online, per iteration: s→c = 4d + 8·batch; c→s = 32 + 33·batch + 32
  - output (only with `--send-model`): s→c = 4d
- **Rounds.** A *flight* is a maximal run of messages in one direction, and a round trip is 2 flights. Online is exactly
  **3·iters round trips** (6·iters flights). Preprocessing is 3 flights (handshake round trip + upload), and output is
  1 flight.
- **Discrete logs.** The report shows the BSGS table bits and memory, the largest bound and |value| per step, and the
  giant-step lookups needed by the hardest solve against the worst case for the largest bound.

Example from `results/sweep_impl.csv` (this machine, iters = 20, batch = 4, qbits = 6, public-key encryption, 1 thread
per party, localhost; times in ms):

| d | client pre wall | client online wall / cpu | server pre compute | server online wall / cpu | c→s pre | c→s online | s→c online | total |
|---|---|---|---|---|---|---|---|---|
| 5 | 53.3 | 37.6 / 5.4 | 22.6 | 24.3 / 15.5 | 37.68 KB | 3.92 KB | 1.04 KB | 42.64 KB |
| 11 | 107.5 | 66.4 / 5.7 | 43.5 | 41.0 / 26.3 | 73.34 KB | 3.92 KB | 1.52 KB | 78.79 KB |
| 31 | 304.0 | 127.5 / 5.6 | 131.5 | 61.2 / 55.2 | 192.22 KB | 3.92 KB | 3.12 KB | 199.27 KB |

The client's work is almost entirely preprocessing: encryption grows with iters·batch·d. Its online work is 2 + batch
key derivations per iteration. The client's online *wall* time is mostly waiting for the server's decryptions and
discrete logs. Online traffic does not depend on the data and grows only with d (through w).

## Files

| file | contents |
|---|---|
| `src/lr_cryptonn.cpp` | the program: CLI, reports, JSON / CSV, `--verify` |
| `include/cryptonn_protocol.h` | the protocol (`run_client`, `run_server`), header, traffic formulas, report blob, worker pool |
| `src/lr_cryptonn_selftest.cpp` | group / BSGS / FEIP / FEBO tests, and end-to-end runs of the real protocol over loopback TCP |
| `include/ec_group.h` | P-256 scalars and points (OpenSSL), multi-scalar multiplication |
| `include/bsgs.h` | bounded discrete log (baby-step giant-step) |
| `include/fe_ip.h`, `include/fe_bo.h` | FEIP (Abdalla et al.) and FEBO (CryptoNN Sec. III-B) |
| `include/lr_params.h`, `include/lr_reference.h` | parameters and data generation; the plaintext fixed-point reference (copies of the GC baseline's) |
| `include/metrics.h`, `include/net.h` | per-phase wall / CPU / bytes / flights; the TCP channel with exact counters |

## Limitations

- Semi-honest only (see above). There is no protection against a server that requests keys for chosen vectors.
- The client generates the master keys itself, so there is no separate authority process. Merging the roles removes the
  server ↔ authority hop that CryptoNN has. In the paper's three-party deployment, each key request would cost an
  extra round trip to the authority.
- The discrete-log bound is Σ|w_j|·2^f. It is public but loose; actual values are much smaller, and the solve stops at
  the first hit. With a large `--qbits` and a small `--dlog-bits`, giant steps dominate the server's online time.
- Values on the wire are int32. The program refuses parameters whose public weight bound would not fit.
