# Baseline 7: Falcon (3-server honest-majority MPC), with a client dealer and malicious-security checks

This baseline trains logistic regression with the protocols of **Falcon** (Wagh, Tople, Benhamouda, Kushilevitz, Mittal,
Rabin, *Falcon: Honest-Majority Maliciously Secure Framework for Private Deep Learning*, PETS 2021). Falcon's code is
[snwagh/falcon-public](https://github.com/snwagh/falcon-public), and this baseline is ported from it, pinned at commit
`e11808c`. Three servers hold every value in 2-out-of-3 replicated secret sharing (RSS) over Z_2^64. They compute
products with Falcon's resharing multiplication and comparisons with Falcon's wrap3 / PrivateCompare / ReLU protocols.
The client owns the data and also deals all correlated randomness. Upstream's malicious mode only adds traffic and never
checks anything; here **real malicious-security checks** are added on top. `--semi-honest` switches them off, so the cost
of malicious security can be measured on the same code.

| role | holds | learns |
|---|---|---|
| **client** C | the training data (x, y), the initial weights w0 (random for now), and the dealer randomness | the trained weights (only the client) |
| **servers** S0, S1, S2 (Falcon's parties A, B, C) | no input | nothing: every value they receive is uniformly masked, and the weights are output to the client |

Training defaults match the sibling baselines: **20 iterations, batch 4, learning rate 0.5**. `--d` sets the number of
features, including the bias column. The client's random data and initial weights come from the siblings' generator, so
the same `--seed` gives the same quantized values. They are rescaled to Falcon's fixed point with F = 13 fractional bits
(`FLOAT_PRECISION`).

> **Upstream is not private as shipped.** At `e11808c`, every function in `Precompute.cpp` returns all-zero shares.
> `funcMatMul` / `funcTruncate` therefore reveal every product in the clear, and `funcWrap` reveals the compared value.
> The malicious checks (`funcCheckMalicious*`) are stubs: an empty "Do the abort thing" branch, and the reconstruct
> aborts are commented out. The upstream README itself warns of "known bugs (especially in the malicious
> implementation)". Upstream also never trains end to end. This baseline keeps Falcon's protocol structure and fixes
> every function it uses. Section [Port of falcon-public](#port-of-falcon-public) lists the changes, each with the
> upstream file:line it fixes.

## What is computed

Values live in the ring Z_2^64 in two's complement. Features, labels, weights and ŷ are at scale 2^F. Iteration t uses
rows t·B .. t·B+B−1:

```
u   = trunc_F( X_t · w )                           funcMatMul (fused truncation)
a   = [u + 1/2 ; u − 1/2]                           public constants on component 0
b   = MSB(a)                                        funcRELUPrime = funcWrap(2a) ⊕ MSB of the components (2B elements, one call)
r   = a · (1 − b)                                   funcSelectShares, i.e. ReLU(a)
ŷ   = r_1 − r_2 = clamp(u + 1/2, 0, 1)              SecureML's piecewise sigmoid (the mdml baseline uses the same one)
e   = ŷ − y_t
g   = trunc_{F+k}( X_tᵀ · e )                       funcMatMul with transposed A; lr/B = 2^-k folded into the shift, as in FCLayer
w  -= g
```

* **Falcon has no sigmoid.** LR needs one, so the sigmoid is built as ReLU(u + ½) − ReLU(u − ½), which equals the
  SecureML/ABY3 piecewise function exactly. Both ReLUs run as one 2B-element call.
* **Truncation** uses Falcon's truncation pair (getDividedShares, ABY3 Π_trunc2):
  - the dealer deals r' uniform and r = ⌊r'/2^t⌋;
  - the servers open D = z − r' and set z/2^t := r + ⌊D/2^t⌋;
  - the result is ⌊z/2^t⌋ or one less, and is wrong only with probability ≈ |z|/2^64.
* **Why Z_2^64 rather than Falcon's Z_2^32.** In Z_2^32 with F = 13, the truncation pair fails for a few percent of
  elements, which wrecks training. In Z_2^64 the failure probability is about 2^-30 or smaller.
* **Exactness.** The client dealt every r', so it can replay the training exactly. `--verify` checks the trained weights
  against `reference_train` (plaintext, same masks) **bit for bit**. It also prints the distance to exact-floor training
  (`reference_train_nominal`: 4 ulp at d = 5, 82 ulp at d = 1000) and to double-precision SGD with the same sigmoid
  (9e-4 at d = 5).
* **Backward pass.** As in the sibling baselines, it uses the LR gradient (ŷ − y)·x, not the derivative of the piecewise
  sigmoid.

## Protocol

Notation: `[x]` is an RSS of x. Server S_i holds components (x_i, x_{i+1}), indices mod 3, and public constants go on
component 0. B, d and T are batch, features and iterations; N = 2B is the number of comparisons per iteration.

### Setup and offline phase

1. **Handshake** (all six links). Each party compares the full parameter set, including the security mode, with every
   peer's and aborts on a mismatch.
2. **Dealer seeds.** The client draws seeds k_0, k_1, k_2 and sends (k_i, k_{i+1}) to S_i. Each component's seed goes to
   exactly the two servers that hold that component.
3. **Server zero-sharings** (Falcon Π_Prep "3-out-of-3 randomness", unused upstream). S_i sends a key κ_i to S_{i+1} and
   sets α_i = F(κ_i) − F(κ_{i−1}), so the three α_i sum to 0. One stream each for Z_2^64, Z_67 and bits. Every 3-out-of-3
   share that leaves a server is masked with a fresh α.
4. **Dealing** (`offline/deal`, one message from C to S0 and one to S2).
   - A random item (r', the wrap masks R, triple masks, η, c, …) is expanded from the seeds and costs no traffic.
   - A correlated item gets two components from the seeds. The client sends the third (component 0) to its two holders,
     S0 and S2.

   Per iteration the client deals:
   - truncation pairs: B for the forward pass, d for the backward pass;
   - per comparison:
     * R;
     * 63 bits of R in Z_67;
     * α = wrap3(R_0, R_1, R_2);
     * η;
     * t = 2η − 1 and s_k = t·R_k, which make PrivateCompare's first product local;
     * the mask m ∈ Z_67^* folded into the first leaf (m·s_0, m·t, m);
   - per ReLU: the selector bit c (as an XOR sharing and in Z_2^64);
   - **[malicious]** in addition:
     * a matrix triple (U, v_f, U·v_f) / (Uᵀ, v_b, Uᵀ·v_b) per iteration;
     * 63 Z_67 triples per comparison (one per tree product);
     * a ring triple per ReLU product.

### Online phase

* **Input** (1 step).
  - Semi-honest: the client deals X, y and w0 as RSS.
  - **[malicious]** X instead enters as **E = X − U**, sent to all three servers, with [X] = E + [U]. E is also the public
    "ρ" of both MatMul checks of the iteration (SecureML's trick).
* **Per iteration: 14 steps semi-honest, 15 malicious.**

| step | what | steps |
|---|---|---|
| forward MatMul | T_i = (3-out-of-3 share of X_t·w) + α_i − r'_i. S0 and S1 send T to S2 ([M] in the same step: checked opening of σ = w − v_f). S2 sends D = ΣT to S0. u = r + ⌊D/2^F⌋ | 2 |
| DReLU: open | X = 2a + R (checked in [M]); δ = wrap3(X_0, X_1, X_2) | 1 |
| DReLU: PrivateCompare tree | leaves c_k = s_k − r_k·t + 1 + Σ_{j<k}(R_j ⊕ r_j) against the public bits r of X + 1 (leaf 0 times m); 64 → 32 → … → 1 by resharing products ([M] with the ρ, σ of each product's triple opened in the same step) | 6 |
| DReLU: PC check **[M]** | checked opening of all 63 tree-product check values; all must be 0. This happens **before** d is opened | 1 |
| DReLU: open d | β' = [d = 0]; θ = β ⊕ α ⊕ η ⊕ (δ ⊕ β' on component 0); b = θ ⊕ MSB(a_j) | 1 |
| SelectShares | open e = b ⊕ c; m = 1 − b; reshare a·m ([M] with the triple's ρ, σ opened in the same step) | 2 |
| backward MatMul | the same as forward, with Xᵀ and e, shift F + k | 2 |

* **Final check [M]** (1 step). One checked opening of every deferred product check: both MatMuls and every Select
  product of every iteration. All values must be 0.
* **Output** (2 steps).
  - Semi-honest: S_i sends component i to the client.
  - **[malicious]** S_i sends both of its components, so the client receives every component twice and compares.
  - The client then sends ACCEPT or ABORT to all servers. Servers stop only after this verdict, so every honest party
    ends in the same state.

Online steps in total: **14·T + 3** semi-honest and **15·T + 4** malicious (283 / 304 at T = 20). Neither depends on d.

## Malicious security

**Model.** A static, active adversary corrupts **one of the three servers** (honest majority, as in Falcon). The
protocol is secure with abort: no fairness and no guaranteed output.

The client is honest. It is the only party with inputs or outputs, so a malicious client can only hurt itself. As the
data owner it is also trusted to deal correlated randomness. This is the "client-aided" model, and it replaces Falcon's
server-side Π_Prep, which upstream does not implement. Pairwise channels are assumed to be private and authenticated, as
in Falcon. The plain TCP used here stands in for that in benchmarks.

| deviation by a server | caught by |
|---|---|
| wrong value in any opening (X, d, e, σ, ρ, check values) | **checked opening**: the value goes to next, and SHA-256(op ‖ second component) goes to prev in the same step. Every server receives its missing component from one neighbour and its hash from the other, so an honest server sees any inconsistency immediately |
| error injected into a reshared product (PrivateCompare tree, Select) or into a MatMul 3-out-of-3 contribution | **Beaver-triple check** against a triple dealt by the client: c − z − ρy − σx − ρσ must open to 0. This is what upstream's `funcCheckMalicious*` stubs were meant to do. It is exact: there are no random linear combinations, so it is sound over Z_2^64 and Z_67 |
| S2 equivocating on D in MatMul | the comp-0 copies at S0 and S2 disagree at the next checked opening that uses them |
| non-canonical Z_67 element or bit (≥ 67, > 1) | rejected on receipt (`non-canonical field element`). Upstream indexes 67×67 lookup tables with received bytes, which would make the abort depend on an honest party's share |
| wrong output component | the client gets every component from two servers |
| a server that stops responding or closes | `--timeout` at its peers; then ABORT is relayed to everyone |

**Selective failure.** Every check recomputes the claimed value exactly against honestly dealt triples, and every
received element is canonical. Every value opened before a check is uniformly masked by fresh dealt randomness (D by r',
X by R, ρ/σ by triple masks, e by c) or padded (β' by η). The one opened value that is not uniform is PrivateCompare's
d. It is opened only after the check of the tree that produced it. Each abort is therefore a deterministic function of
errors the adversary chose, so an abort leaks nothing.

`scripts/test_tamper.sh` runs every deviation kind with each server as the cheater:
- wrong opening value;
- wrong hash;
- wrong MatMul contribution;
- wrong D;
- tree reshare error;
- non-canonical element;
- mask leaf;
- Select reshare error;
- wrong check value;
- wrong output share.

A malicious run passes only if all of the following hold:
- no party prints `trained weights`;
- the client and the honest servers exit with 3 (detected) or 4 (a peer aborted);
- the cheater logged that its tamper fired;
- an honest party logged the kind-specific detection message.

The same deviations in `--semi-honest` mode must not abort. `lr_falcon_selftest` does the same in-process. It also
checks dealer consistency (it reconstructs every dealt item and checks its relation), PrivateCompare and DReLU at edge
inputs (0, ±1, ±2^(F−1), 2^63, 2^63−1, random) and the exact byte formulas.

**What this does and does not claim.**
- This is Falcon's protocol, with the checks Falcon's code skeleton intended, implemented in the standard
  honest-majority way. There is **no formal proof for this code and no third-party audit**. The checks were validated
  only by the tests above.
- The client is trusted as dealer. A deployment without that trust would need Falcon's server-side preprocessing
  (ABY3-style bit injection and triple generation with cut-and-choose), which would add a lot of offline cost.
- The protocol does not check that the client's inputs are in range. It does not need to: the client can only corrupt
  its own result.

## Port of falcon-public

Every ported function in `include/falcon.h`, `rss.h` and `prep.h` cites its upstream origin in a comment. The
correctness fixes below were found by reading upstream and confirmed with 64-bit simulations.

| upstream (`Functionalities.cpp` unless noted) | here | changes |
|---|---|---|
| funcReconstruct / ReconstructBit / mod-p (158-384) | `Opener<Ring64/Field67/Bits>` | one class for all three rings; checked opening in [M]. Upstream's malicious path uses 8-bit buffers for 32-bit values, has the abort commented out, and returns only `.first` for mod p |
| funcMatMul (677-735), matrixMultRSS, dividePlain (tools.cpp), getDividedShares (Precompute.cpp) | `mat_vec` | real truncation pair; arithmetic shift instead of division through a `double` (lossy at 64 bit); a zero-sharing on the 3-out-of-3 share sent to party C (otherwise C learns a cross term); D is sent to S0 only (upstream also sends it to B, which never uses it); [M] deferred triple check |
| funcDotProduct (739-859), funcMultiplyNeighbours (899-930) | `Reshare<…>` | zero-sharing (upstream has none, so each reshare leaks); MultiplyNeighbours resharing in the correct direction (upstream builds a mirrored sharing that reconstructs wrongly) |
| funcPrivateCompare (988-1239), funcCrunchMultiply (933-985) | `private_compare` | the first product made local via dealt t, s_k (this also removes upstream's bug of resharing `size` instead of `size·ℓ` elements); a nonzero mask m folded into leaf 0 (upstream opens the raw product d, which leaks); [M] tree checked before d is opened |
| funcWrap (1244-1313), funcRELUPrime (1395-1413) | `relu_prime` | compare against X + 1 as in the paper (upstream compares against X, which makes MSB(0) wrong half the time); δ only on component 0 (upstream also XORs δ into party B's share, which makes the result inconsistent once shares are nonzero); η dealt as an XOR sharing (upstream XORs Z_67 shares) |
| funcSelectShares (1318-1361) | `select_shares` | zero-sharing; [M] checked e and triple check. Upstream's 2-step structure is kept (a 1-step variant exists but is not used) |
| Precompute.cpp (all zeros) | `Dealer`, `deal_*` (prep.h) | the client deals everything, using seed expansion plus one correction component per correlated item |
| AESObject keys (never used) | `Prss` (prep.h) | PRSS zero-sharings |
| FCLayer forward / updateEquations | `server_run` (lr.h) | shifts F and F + log2(B/lr), as upstream |

`scripts/fetch_upstream.sh` clones the pinned upstream into `third_party/` (gitignored) for side-by-side reading. It is
not compiled.

## Install and run (WSL / Linux)

The baseline needs emp-tool only, for AES-CTR (PRG) and SHA-256. Use the version pinned in `../gc_ag2pc`: install it with
`../gc_ag2pc/scripts/install_emp.sh` (default prefix `$HOME/emp`). You also need CMake >= 3.21, a C++20 compiler and
OpenSSL 3. Networking uses plain POSIX sockets.

```bash
cd baselines/falcon
EMP_PREFIX=$HOME/emp bash scripts/build.sh

./build/lr_falcon_selftest --quick     # identities, dealer consistency, PC/DReLU edge cases, one end-to-end run (~0.1 s)
./build/lr_falcon_selftest             # + end-to-end configs in both modes, every tamper kind x cheater (~1 s)

bash scripts/run_local.sh --d 5        # 3 servers + client on this machine; the client prints the report
bash scripts/run_local.sh --d 1000 --semi-honest
bash scripts/test_tamper.sh --d 5      # real binaries: every deviation must abort (malicious), none (semi-honest)
D_LIST="5 50 500" REPS=2 bash scripts/bench_sweep.sh   # results/sweep.csv
```

By hand, in four terminals or on four machines (the hosts are listed in the order S0, S1, S2, client; the parties may
start in any order within `--connect-timeout`):

```bash
H=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4
./build/lr_falcon server --id 0 --hosts $H --port 23456 --d 100
./build/lr_falcon server --id 1 --hosts $H --port 23456 --d 100
./build/lr_falcon server --id 2 --hosts $H --port 23456 --d 100
./build/lr_falcon client       --hosts $H --port 23456 --d 100 --verify
```

Options (`--help`):
- training: `--d --iters --batch --frac --lr --qbits --seed --no-bias --zero-init` (all four parties must pass the same
  training flags; `--seed` and `--zero-init` only matter at the client);
- protocol: `--semi-honest --hosts --port` (uses ports BASE .. BASE+5, one per pair), `--run-id` (rejects stray
  connections from another run), `--connect-timeout S` (60), `--timeout S` (300, per receive);
- client output: `--verify --json FILE --csv FILE --quiet`;
- test only: `--tamper KIND --cheater I`.

Exit codes: 0 ok, 1 error, 2 verify mismatch, **3** this party detected a deviation, **4** a peer aborted, closed or
timed out.

The protocol code is single-threaded. `--threads` is accepted and recorded but has no effect. The online phase is
latency-bound (about 15 small messages per iteration), and the local work is tiny.

## Cost report

Each party measures itself in contiguous phases named `stage/sub-phase`. After the client accepts, the servers send it
their tables, and the client prints everything.

| stage | client phases | server phases |
|---|---|---|
| (not counted) | `net/connect` | `net/connect` |
| **setup** | `handshake`, `seeds` | `handshake`, `seeds`, `keys` |
| **offline** | `deal` (computes and sends the corrections) | `deal` (receives and expands) |
| **online** | `input`, `wait` (idle while the servers train; excluded from the client's computation), `output` | `input`, `fwd_matmul`, `drelu_open`, `drelu_pc_tree`, `drelu_pc_check` [M], `drelu_pc_open`, `select`, `bwd_matmul`, `final_check` [M], `output` |

Each phase records:
- **wall** time;
- **cpu** time (the whole process, including the sender threads);
- bytes sent and received **per peer**, including the 8-byte frame header of every message;
- **steps**.

A *step* is one communication round on the servers' critical path, that is, one one-way message latency. All three
servers run the same schedule, and every run asserts that S0, S1 and S2 count the same steps in every phase. One step is
about the same as one sibling "round trip" made of simultaneous swaps.

The **SUMMARY** gives:
- **client computation**: dealing (offline), and input plus output (online);
- **server computation**: max over the servers, plus each server's own figure;
- **total communication**, split into client → servers (seeds, deal, input), servers → client (output) and server ↔ server;
- the online steps.

Every run also asserts that the total equals the sum of bytes sent and the sum of bytes received over all four parties.
CSV (`--csv`, one row per run): `client_*` and `server_*` wall/cpu (server = max over S0..S2, plus `_sum` columns),
`client_idle_wait_ms`, `comm_{offline,online}_{c2s,s2c,s2s}_bytes`, `comm_total_bytes` (the sum of those six),
`online_steps`, and peak RSS.

**Exact byte counts** (headers included; B = batch, d = features, T = iterations):
- client → S0 and → S2, `offline/deal`:
  - `8 + T·(286·B + 8·d)` semi-honest;
  - `8 + T·(436·B + 16·d)` malicious;
  - S1 receives only its two 16-byte seeds;
- client input:
  - semi-honest: `8 + 8·(n·d + n + d)` to S0 and S2;
  - malicious: E (`8·n·d`) to all three, plus the y/w0 corrections to S0 and S2;
- server ↔ server per iteration:
  - `288 + 510·B + 24·d` semi-honest;
  - `1888 + 1764·B + 48·d` malicious, plus a final check of `144 + 24·T·(3B + d)`.

## Results

Measured on WSL2 on one machine: 12 hardware threads, all four processes on loopback. T = 20, batch = 4, F = 13, mean of
2 runs (`scripts/bench_sweep.sh`). Every run was checked with `--verify` (bit-exact). Times are in ms. Server figures are
the max over S0..S2.

| d | mode | client offline wall / cpu (dealing) | client online cpu (input + output) | server offline cpu | server online wall / cpu | client→servers (KiB) | servers→client (KiB) | server↔server (KiB) | total (KiB) | online steps |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | malicious | 2.10 / 2.59 | 0.90 | 2.56 | 87.9 / 75.1 | 82.3 | 0.42 | 187.9 | 270.6 | 304 |
| 10 | malicious | 2.57 / 2.86 | 0.83 | 2.14 | 79.8 / 63.0 | 94.9 | 0.66 | 194.9 | 290.5 | 304 |
| 50 | malicious | 2.71 / 3.07 | 1.02 | 2.05 | 74.4 / 58.8 | 195.5 | 2.53 | 251.2 | 449.2 | 304 |
| 100 | malicious | 2.38 / 2.71 | 1.12 | 1.71 | 69.4 / 53.3 | 321.3 | 4.88 | 321.5 | 647.6 | 304 |
| 500 | malicious | 4.25 / 4.43 | 3.01 | 4.97 | 81.5 / 72.9 | 1327.5 | 23.63 | 884.0 | 2235.1 | 304 |
| 1000 | malicious | 5.89 / 6.85 | 7.12 | 6.71 | 94.3 / 82.5 | 2585.3 | 47.07 | 1587.1 | 4219.5 | 304 |
| 5 | semi-honest | 1.42 / 1.75 | 0.85 | 1.61 | 54.4 / 36.7 | 54.2 | 0.31 | 48.2 | 102.7 | 283 |
| 10 | semi-honest | 1.66 / 1.76 | 0.59 | 1.97 | 42.5 / 27.8 | 62.1 | 0.42 | 50.6 | 113.0 | 283 |
| 50 | semi-honest | 1.32 / 1.73 | 0.76 | 1.43 | 40.4 / 26.9 | 125.2 | 1.36 | 69.3 | 195.8 | 283 |
| 100 | semi-honest | 1.47 / 1.74 | 0.87 | 1.15 | 40.7 / 28.2 | 204.1 | 2.53 | 92.7 | 299.4 | 283 |
| 500 | semi-honest | 3.42 / 3.94 | 5.38 | 2.68 | 74.1 / 54.3 | 835.3 | 11.91 | 280.2 | 1127.5 | 283 |
| 1000 | semi-honest | 3.86 / 4.43 | 6.78 | 3.68 | 73.8 / 49.0 | 1624.4 | 23.63 | 514.6 | 2162.6 | 283 |

- **The whole protocol is cheap compared with the OT- and HE-based siblings.** There is no public-key or OT work; the
  honest-majority setting and the dealer remove it. The run is latency-bound: 283 or 304 steps take about 40-90 ms on
  loopback (about 0.2-0.3 ms per step). Over a real network, the step count times the latency dominates.
- **Client cost** is the dealing (a few ms) plus the input. For each comparison it sends 131 bytes (semi-honest) or
  194 bytes (malicious) of wrap/PrivateCompare corrections to each of S0 and S2, with 2B = 8 comparisons per
  iteration. On top of that comes the data: 16 bytes per value semi-honest (one correction to each of S0 and S2) and
  24 malicious (E goes to all three servers).
- **Malicious security roughly doubles the total traffic** at large d and costs about 2.6× at small d. The extra traffic
  is the triple material, the checked-opening hashes (32 B per opening), the ρ/σ openings and the final check. It adds
  **1 step per iteration plus 1**, and about 1.5-2× the server CPU.
- **d mainly affects the data-sized terms** (input, truncation pairs, MatMul triples and openings). The comparison part
  does not depend on d.

## Files

| file | contents |
|---|---|
| `src/lr_falcon.cpp` | CLI, roles, report (phase table, SUMMARY), JSON / CSV, `--verify`, exit codes |
| `src/lr_falcon_selftest.cpp` | 4 parties as threads over loopback: identities, dealer consistency, edge cases, end-to-end, every tamper kind |
| `include/falcon.h` | the ported Falcon protocols (openings, resharing, MatMul, PrivateCompare, wrap/RELUPrime, SelectShares) and the malicious checks |
| `include/prep.h` | the client dealer (seed streams, corrections), PRSS zero-sharings, preprocessing structures, byte formulas |
| `include/lr.h` | the training driver: setup, input, iterations, final check, output and verdict, report invariants |
| `include/rss.h` | RSS over Z_2^64, Z_67 and bits; wrap/MSB helpers |
| `include/net.h` | 4-party TCP mesh: a sender thread per peer, framed messages, abort propagation, timeouts |
| `include/metrics.h` | per-phase wall / cpu / bytes per peer / steps |
| `include/params.h` | parameters, the client's data (the siblings' generator), the plaintext references, CLI helpers |
| `scripts/` | `build.sh`, `run_local.sh`, `bench_sweep.sh`, `test_tamper.sh`, `fetch_upstream.sh` |
