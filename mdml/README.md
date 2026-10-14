# Baseline 2: maliciously secure 2PC with MD-ML (SPDZ2k, dishonest majority)

This baseline trains logistic regression with [MD-ML](https://github.com/NemoYuan2008/MD-ML)
(Yuan et al., *MD-ML: Super Fast Privacy-Preserving Machine Learning for Malicious Security with a Dishonest
Majority*, USENIX Security 2024). It uses SPDZ2k secret sharing over Z_{2^64} (k = s = 64) with MD-ML's
circuit-dependent preprocessing.

| role | MD-ML party | holds | learns |
|---|---|---|---|
| **client** | party 0 | training data (x, y) **and** the initial weights, uniformly random for now | nothing |
| **server** | party 1 | nothing | only the trained weights |

The training defaults are 20 SGD iterations with a batch of 4 datapoints each. The number of features is set
with `--d`, which counts the bias column.

## What is computed

Values are fixed point in Z_{2^64} with F fractional bits (`--frac-bits`, default 16: MD-ML's code default; the
paper uses 20). Each iteration t does:

```
z  = trunc_F(X_t w)                      multiply-then-truncate (paper Sec. 4.1)       1 round
b1 = (z + 1/2 >= 0),  b2 = (z - 1/2 >= 0)   one LTZ gate on 2B values (paper Sec. 4.3)  8 rounds
s  = (b1 - b2)(z + 1/2) + b2             = clamp(z + 1/2, 0, 1)                        1 round
e  = s - y_t                             local
w  = w - trunc_{F+k}(X_t^T e)            lr/B = 2^-k folded into the truncation        1 round
```

- **Sigmoid.** s is the piecewise-linear sigmoid of SecureML/ABY3. MD-ML has no sigmoid, and the paper
  evaluates SVM and neural networks, not logistic regression.
- **Truncation is probabilistic** (MD-ML's): each truncation may be off by 1 in the last bit. The trained
  weights match a double-precision run of the same algorithm to about 1e-5.
- **Same data as baseline 1.** The data generator is baseline 1's: same `--seed`, same quantized
  features and initial weights, scaled exactly to F bits; labels are y in {0, 1}.
- **Same constraint:** `batch / lr` must be a power of two.

**This is not the same function as baseline 1**, which computes the FE code's exact quantized semantics with an
exact sigmoid. The difference is the usual one between MPC baselines (fixed point with a piecewise sigmoid), so
compare costs, not model bits.

## What comes from MD-ML, and what was added

**Used unchanged from MD-ML** (pinned commit `65ad2d2`, `scripts/fetch_deps.sh`):
- networking (`Party`)
- SPDZ2k share types and the dealer's share and MAC generation (`FakeParty`)
- the gate framework (`Gate` / `FakeGate`, with Delta / lambda wires and the fake-offline file pipeline)
- `FakeInputGate`, `FakeMultiplyGate` and `FakeElemMultiplyGate`
- Eigen-based matrix multiplication

**Patched** (`patches/mdml-network.patch`, two small changes to `Party.cpp`):
- the peer address comes from `MDML_PEER_HOST` (it was hard-coded to 127.0.0.1)
- `TCP_NODELAY` is set. Without it, Nagle's algorithm plus delayed ACKs add ~40 ms stalls to the small
  messages of the MAC check (emp-tool, used by baseline 1, also disables Nagle).

**Added** in `include/mdml_ext.h`, with the dealer side in `include/mdml_dealer.h`. The MD-ML repository is a
research prototype: it has no logistic regression, and several protocol pieces are placeholders.

- **MAC check.** The repo computes MAC shares but never checks them. This baseline adds the paper's
  `Pi_MACCheck` (App. B.1, from SPDZ2k), batched over every opened value in both domains:
  - coin tossing by commit-and-open (SHA-256)
  - random coefficients in Z_{2^s}
  - commitments to `z_j = m_j - alpha_j * y`
  - the server accepts the output only after the check passes
- **Comparison (LTZ).** The repo's `GtzGate` fakes the binary triples as (0,0,0), which opens the input
  bits, sends all-zero "MACs", and opens the comparison bit in the clear. This baseline:
  - runs MD-ML's carry circuit (`CarryOutCin`) on real SPDZ2k binary shares (k = 1, s = 64)
  - uses authenticated AND triples and the dealer's binary decomposition of lambda_x (the paper's
    "LTZ right after MultTrunc" optimization)
  - does B2A with a daBit, then opens Delta_z
  - costs 118 ANDs in 6 rounds per comparison (the paper counts 2k-2 = 126). Message sizes match the
    paper's formula.
- **Output to the server only.** The repo's `OutputGate` opens the result to both parties. Here the
  parties open `[lambda_w] - [r]`, where only the server knows r.
- **Multiply-then-truncate with any shift.** The repo fixes the shift at 16 bits; the update needs F + k.
- **Local linear gates** for transpose, stacking, slicing and constants. The repo's `SubtractGate` desyncs
  the preprocessing files.
- **Inputs send k = 64 bits per element.** The repo's `InputGate` sends the 128-bit container.

## Security model, and what the numbers cover

- **Malicious security.** The two parties are malicious, with a dishonest majority: either party may
  deviate. Every opened value is MAC-checked before the output is released (`--tamper` demonstrates an
  abort).
- **Preprocessing comes from a trusted dealer** (`lr_mdml_dealer`). This is MD-ML's own "fake offline"
  method; the paper's online numbers are measured the same way.
  - The dealer writes single-use text files for each party.
  - The reported times and bytes are the **online phase only**: from input sharing to a passing MAC check.
  - Loading the preprocessing files is reported separately and is not part of the online total.
- **Baseline 1 includes its interactive preprocessing**, so the totals of the two baselines are not like
  for like. The dealer reports how much material a real offline phase would have to produce (for d=5:
  1400 triple elements, 180 truncation pairs, 18880 binary AND triples, 10240 lambda bits, 160 daBits).
  Use these counts with the paper's preprocessing costs (Table 7) if you need an offline estimate.

## Install on WSL (Ubuntu 22.04/24.04)

```bash
# 1. toolchain (CMake >= 3.16, GCC >= 10, header-only Boost >= 1.70, OpenSSL)
sudo apt install -y build-essential cmake git libboost-dev libssl-dev

# 2. MD-ML (pinned + patched) and Eigen 3.4.0 into ./third_party (no sudo)
cd baselines/mdml
bash scripts/fetch_deps.sh

# 3. build
bash scripts/build.sh

# 4. in-process tests: LTZ edge cases, arithmetic gates, bit-exact LR for several d, tamper detection
./build/lr_mdml_selftest --quick        # ~6 s (full run: ~10 s)

# 5. dealer + both parties locally, with the server checking the result (test-only)
bash scripts/run_local.sh --d 5
```

By hand: the dealer runs first, then the two parties, each in its own terminal. All three must pass the same
training flags.

```bash
./build/lr_mdml_dealer --d 5 --offline-dir results/offline --job lr      # writes lr-party-{0,1}.txt
./build/lr_mdml server --d 5 --port 12345 --job lr --host <client-ip>
./build/lr_mdml client --d 5 --port 12345 --job lr --host <server-ip>
```

The options are:
- training: `--iters --batch --lr --frac-bits --qbits --seed --no-bias --zero-init`
- output: `--verify` (server, test only: bit-exact check), `--json FILE`, `--csv FILE`, `--quiet`
- testing: `--tamper arith|bin` (client, test only: corrupts one share; both parties must abort with exit 3)

Run `--help` for details.

**Networking.** MD-ML connects both ways, so each party's `--host` is the *other* party's address, and the
ports `N..N+3` must be open. For two machines, copy `<job>-party-0.txt` to the client and `<job>-party-1.txt`
to the server. Preprocessing is single-use, so run the dealer again before every run (`run_local.sh` does this).

To sweep dimensions: `D_LIST="5 10 50 100" REPS=3 bash scripts/bench_sweep.sh` writes `results/sweep.csv`.

## Metrics reported

Every gate is timed at both parties, recording wall time, process CPU time, bytes sent, and communication
rounds. The two parties exchange their numbers at the end, and each prints the combined table. Bytes are payload
bytes as counted by MD-ML's `Party`, without TCP/IP headers (the paper counts headers).

| row | contents |
|---|---|
| input sharing | client sends Delta = x + lambda for w_0 and every (X_t, y_t), streamed in one flight |
| forward X*w | `Pi_MultTrunc`: one (k+s)-bit opening per output element |
| comparison LTZ | BitLT on binary shares (6 rounds of ANDs), B2A opening, Delta_z opening |
| sigmoid | element-wise multiplication |
| gradient X^T*e | `Pi_MultTrunc` with shift F + k |
| local linear ops | transpose, stack, constants (no communication) |
| output to server | opening of lambda_w - r |
| batched MAC check | coin toss + commitment to z_j (2 commit-and-open exchanges) |

### Example (this laptop: Intel Core 7 150U, WSL2, localhost, d=5)

```
phase (online)                       |  client ms client cpu |  server ms server cpu |   c->s KiB   s->c KiB | rounds
input sharing (client -> server)     |       0.10       0.10 |       0.04       0.04 |       3.79       0.00 |      1
forward X*w (mult-then-trunc)        |       2.55       0.84 |       1.61       0.71 |       1.25       1.25 |     20
comparison LTZ (BitLT + B2A)         |      17.55       9.90 |      17.59       9.10 |     303.38     303.38 |    160
sigmoid (element-wise mult)          |       1.61       0.97 |       1.61       0.76 |       1.25       1.25 |     20
gradient X^T*e (mult-then-trunc)     |       1.63       0.77 |       2.55       0.69 |       1.56       1.56 |     20
local linear ops (no comm.)          |       0.06       0.06 |       0.07       0.08 |       0.00       0.00 |      0
output to server (open lam-r)        |       0.07       0.02 |       0.06       0.03 |       0.08       0.08 |      1
batched MAC check                    |       1.33       1.20 |       1.34       1.17 |       0.14       0.14 |      4
TOTAL online (inputs -> MAC check)   |      25.05      13.97 |      25.01      12.68 |     311.45     307.66 |    226
communication: total 0.605 MiB (client->server 0.304 MiB, server->client 0.300 MiB)
```

Scaling on the same machine (median of 3 runs; every run passed the MAC check and the bit-exact `--verify`):

| d | 2 | 5 | 10 | 20 | 50 | 100 | 200 | 500 | 1000 |
|---|---|---|---|---|---|---|---|---|---|
| online (ms) | 29 | 31 | 26 | 26 | 35 | 25 | 26 | 30 | 29 |
| comm. total (MiB) | 0.60 | 0.61 | 0.61 | 0.62 | 0.66 | 0.72 | 0.85 | 1.23 | 1.86 |
| peak RSS/party (MiB) | 11 | 11 | 11 | 11 | 12 | 13 | 14 | 20 | 28 |
| preprocessing load (ms) | 159 | 117 | 136 | 162 | 194 | 247 | 349 | 673 | 1253 |

- **Online time on localhost is dominated by the 226 rounds**, of which 160 are the 20 LTZ gates (8 each).
  On a WAN, expect roughly 226 × RTT.
- **The comparisons dominate communication** at small d: about 1942 bytes per comparison per direction,
  160 comparisons in total.
- **Communication grows by 16·d bytes per iteration per direction** for the gradient, plus 8·4·d bytes from the
  client for the inputs.

## Files

| file | contents |
|---|---|
| `src/lr_mdml.cpp` | the two-party program: inputs, online phase, MAC check, output to the server, report |
| `src/lr_mdml_dealer.cpp` | trusted dealer (fake offline): writes the preprocessing; material counts |
| `src/lr_mdml_selftest.cpp` | in-process tests (dealer + two parties over localhost) |
| `include/lr_circuit.h` | the LR training circuit (shared by dealer and parties), bit-exact emulation, double reference |
| `include/mdml_ext.h` | online gates added to MD-ML: input, mult(-trunc), linear, LTZ, output-to-one, MAC check |
| `include/mdml_dealer.h` | the dealer side of those gates |
| `include/lr_params.h` | parameters, data generation (identical to baseline 1), CLI helpers |
| `patches/mdml-network.patch` | peer host + TCP_NODELAY for MD-ML's `Party` |
| `scripts/` | `fetch_deps.sh`, `build.sh`, `run_local.sh`, `bench_sweep.sh` |
