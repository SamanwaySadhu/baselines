# Baseline 9: generic dishonest-majority MPC (SPDZ / MASCOT style), 4 clients + 1 server

This baseline trains logistic regression with a **generic arithmetic MPC protocol** among **5 parties**: 4 clients and
1 server. Values are additively secret-shared over a prime field with information-theoretic MACs (SPDZ). The
multiplication triples come from OT-based, MASCOT-style preprocessing. It is the n-party generalisation of
`../secureml` (same field, same exact activation, same OT-product code), and it is the arithmetic-sharing counterpart
of `../gc_agmpc`, which solves the same 4-clients-plus-server problem with garbled circuits.

| role | party | holds | learns |
|---|---|---|---|
| **server** | 0 | no input | **only the trained weights** |
| **client k** (k = 1..4) | k | datapoint k of every batch (20 rows (x, y) of dimension `--d`) and one additive share of the initial weights | nothing |

Training defaults match the sibling baselines: **20 iterations, batch 4, learning rate 0.5**, so each batch has one
datapoint from each of the 4 clients. **The dimensionality is an input: `--d N`** (features including the bias column,
default 5; every party must pass the same value, and the handshake checks it). Data and initial weights are random for
now: the clients regenerate their rows from the siblings' generator, so the same `--seed` gives the same quantised
data as in every other baseline. Client k's share of w0 is derived from `--seed` as well (test convention; a real client
brings its own).

## What is computed

Exactly `../secureml`'s fixed-point logistic regression, in the prime field F_p, p = 2^127 − 1, with F = 13 fractional
bits and scales chosen so that nothing is truncated except one **exact** floor inside the activation:

```
z_i  = sum_j x_ij w_j                              Beaver, matrix form (exact)
ŷ_i  = clamp(floor((z_i + 1/2) / 2^(2F+k)), 0, 2^F)   SecureML's piecewise sigmoid, by bit decomposition
e_i  = ŷ_i − y_i
g_j  = sum_i e_i x_ij                              Beaver, matrix form
w_j -= g_j                                         lr/batch = 2^-k is folded into the scale of w
```

The trained weights equal the plaintext integer reference (`reference_train`) **bit for bit**. `--verify` (server, test
only) and the selftest check this; `--verify` also prints the distance to double-precision SGD (about 6e-5 at d = 5).

## Protocol

Notation: `[x]` is an authenticated share: party i holds x_i and m_i with Σx_i = x and Σm_i = α·x, where
α = Σα_i is a global MAC key. B, d, T are batch (= number of clients), features and iterations; n = B + 1 parties.

### Offline phase (data independent)

1. **Masks.** Client k draws U (one mask per entry of its rows), Rw_k (masks its share of w0) and Ry (masks its
   labels). The server draws Rout (masks the output). Every party draws its own share of Vf_t (masks w) and Vb_t
   (masks e) for each iteration.
2. **Matrix triples by OT.** Per iteration, Zf_t = U_t·Vf_t and Zb_t = Uᵀ_t·Vb_t. Row i of U_t is known to its owner
   (client i+1) alone, so the owner is the OT **sender** to every other party, whose share of V gives the choice bits.
   This is one OT product per (owner, other party), vectorised over the row (SecureML Sec. 4.2).
3. **Scalar triples and bits for the activation.** c = (Σa_i)(Σb_j) takes one OT product per **ordered** pair of
   parties (20 for 5 parties). Random bits come from `a`, by opening a² (one Beaver product), as in `../secureml`.
4. **Authentication.** A shared value costs one OT product per ordered pair with the receiver's key α_j as the choice.
   An owner-known value costs one product from the owner to each other party. Everything an ordered pair has to
   authenticate goes into **one** product call.
5. **Sacrifice** (malicious only). Every triple has a partner (Uh, Zh = Uh·V). After a coin toss for ρ,
   ρU − Uh is opened and ρZ − Zh − (ρU − Uh)V must open to 0, for the matrix triples and for every scalar triple.
6. **MAC check** of everything opened so far.

### Online phase

* **Input** (1 round). Each client sends every other party `E = X − U` for its rows, `w0_k − Rw_k` and `y − Ry`,
  all uniformly random to anybody who does not know its masks. Then w = Σ_k([Rw_k] + (w0_k − Rw_k)). X itself is never
  shared, only E (public) and [U] are used (SecureML's trick: one mask per data entry for the whole training).
* **Per iteration, 11 rounds**, because every opening is one round in which every party sends its share to every other
  party:

| step | rounds |
|---|---|
| open F = w − Vf_t | 1 |
| z_i = Zf_i + Σ_j F_j [U_ij] + Σ_j E_ij [Vf_j] + Σ_j E_ij F_j (local) | 0 |
| activation: masked opening c = z + 1/2 + 2^(N−1) + r with N + ssp random shared bits r | 1 |
| three borrow chains as log-depth trees of Beaver products (⌈log2(N−1)⌉ levels) | 6 |
| the two xor products, then ŷ = g2·2^F + (g1 − g2)·q | 2 |
| open F' = e − Vb_t | 1 |
| g_j = Zb_j + Σ_i F'_i [U_ij] + Σ_i E_ij [Vb_i] + Σ_i E_ij F'_i, then w_j −= g_j (local) | 0 |

* **Output.** Open w + Rout (so only the server, who knows Rout, learns w), then the final MAC check (4 rounds). The
  server prints the weights only after the check has passed.

Online rounds in total: **11·T + 2** semi-honest and **11·T + 6** malicious (222 / 226 at T = 20). They do not depend
on d.

## Malicious security

**Model.** A static, active adversary corrupts **any set of the 5 parties except all of them** (dishonest majority,
as in WRK17 / `../gc_agmpc`). For example, the server and 3 clients can collude. The protocol is secure with abort: no
fairness and no guaranteed output. The server learns only the weights; a coalition learns nothing about an honest
client's rows beyond what the weights reveal. Pairwise channels are assumed private and authenticated, as in the
siblings; plain TCP stands in for them in benchmarks.

| deviation (by any party) | caught by |
|---|---|
| wrong value in any opening (Beaver masks, F, F', c, w + Rout, ...) | MAC check: all openings are verified in a batch with jointly random coefficients; error about 2^-127 |
| sending **different** shares to different parties (equivocation); sending different inputs E to different parties | the same MAC check. The sigmas of the honest parties are computed from what each of them saw, so they no longer cancel. No broadcast channel is needed |
| wrong MAC share, or a chooser using bits other than its key α | MAC check |
| inconsistent inputs in an OT product (sender messages or chooser bits), wrong cross terms | sacrifice check (matrix and scalar triples) |
| inconsistent OT-extension setup or messages | SoftSpoken's maliciously secure mode (CSW base OT + consistency check) |
| cheating in the coin tosses and in the check itself | commit-then-open, with the committer's identity hashed in; non-canonical field elements are rejected on receipt |
| a non-bit "random bit" | a² is opened and MAC-checked |

`lr_spdz_selftest` and `scripts/test_tamper.sh` inject 10 kinds of deviations (wrong opening offline and online,
equivocated opening, bad scalar and matrix triple, inconsistent OT message and choice, wrong MAC share, wrong MAC-check
value, equivocated client input), with the server, client 1 and client 4 as the cheater. Every run aborted: the
server printed no weights, every party exited non-zero, and at least one exited 3 with an `ABORT:` line. The same
deviation in `--semi-honest` mode is not detected, as designed.

**What this does and does not claim.**
- This is the standard SPDZ/MASCOT construction, applied to SecureML's computation. There is **no formal proof for this
  code and no third-party audit**. The OT extension is emp-ot's SoftSpoken in its malicious mode, and the checks above
  were validated only by the tests listed.
- As in every dishonest-majority SPDZ-style protocol, a corrupted OT sender can make the MAC check pass or fail
  depending on a bit of an honest party's key. A failed check ends the run, so this leaks at most one bit of α per
  execution, which only lowers the 127-bit soundness by one bit.
- There is no fairness: a malicious party can abort at any point. The server uses the masked output only if the final
  check passes.
- **Input ranges are not enforced.** The activation gadget hides z + 1/2 statistically only if |z| stays below its
  design bound, which holds for inputs in [−1, 1] and weights of bounded size. In the 2-party SecureML baseline the
  only data owner can merely corrupt its own result. With several clients, a malicious client that sends an
  out-of-range x (or an out-of-range share of w0) can break that bound and learn something about intermediate weights,
  which depend on the other clients' data. A fix is a range proof on every input, by authenticated bit
  decomposition. It would add about (F+1)·d triples per datapoint to the offline phase, and it is **not implemented**.
  `../gc_agmpc` does not have this issue, because its circuit only accepts 6-bit inputs.

## Install and run (WSL / Linux)

The baseline needs emp-tool and emp-ot (the same pinned versions as `../gc_ag2pc`; install with
`../gc_ag2pc/scripts/install_emp.sh`, default prefix `$HOME/emp`), CMake >= 3.21, a C++20 compiler and OpenSSL 3.

```bash
cd baselines/mpc_spdz
EMP_PREFIX=$HOME/emp bash scripts/build.sh

./build/lr_spdz_selftest --quick     # field, OT products, MAC check, offline consistency (a few seconds)
./build/lr_spdz_selftest             # + end-to-end configs in both modes and every deviation (about 1 minute)

bash scripts/run_local.sh --d 5      # server + 4 clients on this machine; the server prints the report
bash scripts/run_local.sh --d 100 --semi-honest
bash scripts/test_tamper.sh --d 3 --iters 2   # real binaries: every deviation must abort
D_LIST="5 50 500" REPS=2 bash scripts/bench_sweep.sh   # results/sweep.csv
```

By hand, in five terminals or on five machines (hosts: server first, then clients 1..4):

```bash
H=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4,10.0.0.5
./build/lr_spdz server        --hosts $H --port 23456 --d 100
./build/lr_spdz client --id 1 --hosts $H --port 23456 --d 100     # ... --id 2, 3, 4
```

Options (`--help`):
- training: `--d --iters --batch --frac --lr --qbits --seed --no-bias --zero-init`. All parties must pass the same
  values; `--batch` is the number of clients (1..7);
- protocol: `--hosts --port --threads --semi-honest --ssp`. `--threads` is the number of OT connections per pair of
  parties (default 1, max 16);
- server output: `--verify` (test only), `--json FILE`, `--csv FILE`, `--quiet`;
- test only: `--tamper KIND --cheater P`.

Exit codes: 0 ok, 1 error or a peer closed the connection, 2 verify mismatch, 3 this party detected a deviation.

## Cost report

Each party measures itself in contiguous phases named `stage/sub-phase` (`setup`: `handshake`, `ot_base`; `offline`:
`randomness`, `matrix_products`, `triple_products`, `authenticate`, `assemble`, `sacrifice`, `random_bits`,
`mac_check`; `online`: `input`, `open_w`, `forward_local`, `activation_*`, `open_e`, `backward_local`, `output`,
`mac_check`). Connection set-up is not counted. At the end every client sends its table to the server, which prints:

- **Communication rounds.** A round is one step in which every party sends to every other party at once. All five
  parties count the same number, and the run asserts it. Reported separately for online, offline and setup. The OT
  products run in parallel with these; for them the report gives the growth of the busiest channel's **flight**
  counter (a maximal run of same-direction traffic). That number is inflated by the chunking of large OT batches and
  is not a latency bound.
- **Total communication.** The sum of bytes sent over all 5 parties, checked equal to the sum received pair by pair,
  split by setup / offline / online. Payload bytes only.
- **Per-party communication.** Sent and received, offline and online, for the server and each client.
- **Per-party computation.** Wall and cpu time per stage, for the server and each client, and peak RSS. cpu is the whole
  process, all threads. Wall includes waiting for the others.

`--csv` (one row per run) has `rounds_{setup,offline_main,offline_ot_flights,online}`, `comm_{total,setup,offline,online}_bytes`,
`server_{sent,recv}_bytes`, `client_{sent,recv}_{mean,max}_bytes`, the server's and the clients' offline/online wall/cpu,
and so on. `--json` has every phase of every party with bytes sent to and received from each peer.

## Results

WSL2 on one machine (12 hardware threads, all 5 processes on loopback, 1 OT connection per pair), T = 20, batch = 4
(4 clients), F = 13, one run each (`scripts/bench_sweep.sh`). Every run was checked with `--verify` (bit-exact).
Client columns are the mean over the 4 clients; the clients are symmetric, within about 3%.

| d | mode | online rounds | offline rounds (main / OT flights) | total comm. (MiB) | of which online (MiB) | per client sent / received (MiB) | server sent (MiB) | server cpu offline / online (ms) | per client cpu offline / online (ms) | offline wall (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | malicious | 226 | 14 / 412 | 7639 | 11.5 | 1529 / 1528 | 1524 | 23729 / 48 | 24137 / 49 | 10452 |
| 10 | malicious | 226 | 14 / 416 | 7787 | 11.8 | 1559 / 1557 | 1550 | 26459 / 41 | 26179 / 41 | 11781 |
| 50 | malicious | 226 | 14 / 442 | 8386 | 12.8 | 1687 / 1675 | 1640 | 26650 / 52 | 27246 / 52 | 11406 |
| 100 | malicious | 226 | 14 / 458 | 8817 | 13.6 | 1782 / 1759 | 1689 | 27527 / 61 | 28196 / 52 | 12464 |
| 500 | malicious | 226 | 14 / 538 | 11562 | 18.6 | 2405 / 2293 | 1943 | 33440 / 74 | 37148 / 68 | 16131 |
| 5 | semi-honest | 222 | 2 / 410 | 1282 | 11.5 | 257 / 256 | 255 | 3119 / 49 | 3126 / 49 | 1507 |
| 10 | semi-honest | 222 | 2 / 414 | 1308 | 11.8 | 262 / 262 | 259 | 2984 / 49 | 3064 / 46 | 1475 |
| 50 | semi-honest | 222 | 2 / 440 | 1424 | 12.8 | 288 / 284 | 273 | 3380 / 68 | 3437 / 61 | 1606 |
| 100 | semi-honest | 222 | 2 / 456 | 1514 | 13.6 | 309 / 302 | 279 | 3469 / 68 | 3606 / 67 | 1668 |
| 500 | semi-honest | 222 | 2 / 534 | 2116 | 18.6 | 453 / 419 | 302 | 4586 / 80 | 5307 / 80 | 2661 |

(the online wall time is 65-130 ms in every row, about 0.35 ms per round on loopback.)

- **The offline phase is over 99% of the traffic and of the CPU time.** It is dominated by MAC authentication
  (127 field elements per authenticated value, per ordered pair) and by the scalar triples of the activation gadget.
  Both depend on the number of data points, not on d, so the cost is nearly flat in d: about 7.5 GiB at d = 5 and
  11.3 GiB at d = 500.
- **Malicious security costs about 6× the traffic and about 8× the CPU** of the semi-honest mode, as in `../secureml`:
  every value gets a MAC and every triple a partner.
- **Everyone does about the same work.** All 5 parties take part in the triple generation for every ordered pair, so
  the server and the clients have almost the same communication and CPU. The only asymmetry is that the clients also
  send their inputs and the server also reads the output; both are negligible (the online phase is about 12 MiB in
  total).
- **Compared with 2 parties**, 5 parties have 20 ordered pairs instead of 2, so the total traffic is about 10× that
  of `../secureml` (about 770 MiB), and each party sends about 4× as much as either SecureML party.
- **Latency.** The online phase takes 226 rounds regardless of d. Over a real network, rounds × latency dominates the
  online time.

## Files

| file | contents |
|---|---|
| `src/lr_spdz.cpp` | CLI and roles |
| `include/driver.h` | one party start to finish: handshake, OT bootstrap, offline, online, the report (rounds, total / per-party communication, per-party computation), JSON / CSV |
| `include/mesh.h` | the full mesh of parties: main + OT channels per pair, the all-parties exchange (one round), the OT-product primitive |
| `include/mpc.h` | authenticated shares, n-party opening, coin toss, the batched MAC check, fault injection |
| `include/preproc.h` | the offline phase (matrix triples, scalar triples, authentication, sacrifice, random bits) |
| `include/online.h` | the online phase (client inputs, Beaver matrix products, activation gadget, update, output) |
| `include/params.h`, `fp.h` | parameters, fixed-point formats, data generator, plaintext reference; the field F_{2^127−1} (as in `../secureml`) |
| `include/metrics.h` | per-phase wall / cpu / bytes per peer / rounds |
| `src/lr_spdz_selftest.cpp` | field, OT products, n-party MAC check, offline consistency, forked end-to-end runs and every deviation |
| `scripts/` | `build.sh`, `run_local.sh`, `bench_sweep.sh`, `test_tamper.sh` |
