# Baseline 1: two-party authenticated garbled circuits (emp-ag2pc)

This baseline trains logistic regression with **maliciously secure 2PC via authenticated garbling**. It uses
[emp-ag2pc](https://github.com/emp-toolkit/emp-ag2pc), which implements WRK17 with the KRRW18
optimizations, as the comparison point for the FE protocol.

| role | emp party | holds | learns |
|---|---|---|---|
| **client** = garbler | party 1 (ALICE) | training data (x, y), uniformly random for now | nothing |
| **server** = evaluator | party 2 (BOB) | initial weights | the trained weights |

The training defaults are 20 SGD iterations with a batch of 4 datapoints each. The number of features is set
with `--d`, which counts the bias column.

## What is computed

The circuit uses the fixed-point semantics of the FE code (`QUANTIZATION_BITS=6`, `LR=0.5`, `BATCH_SIZE=4`,
bias as the last feature with `x = MAX_X`). With `f = qbits-1` and `K = batch/lr`, each iteration does:

```
z_i    = sum_j w_j x_ij
yhat_i = min(floor(2^f * sigmoid(z_i / 2^(2f))), 2^f - 1)
e_i    = yhat_i - MAXQ*y_i
g_j    = sum_i e_i x_ij
w_j    = trunc((K*w_j - floor(g_j / 2^f)) / K)
```

This is the update in `backward_prop_kim.cpp`. The sigmoid applies the factor `2^f` outside
the sigmoid (see the note at the end). The arithmetic is exact: the weights grow from 6 to about 10 bits over 20
iterations, and there is no wrap-around.

The output is bit-exact with the plaintext reference in `include/lr_reference.h`. This is checked by
`lr_gc_selftest`, and by `--verify` on a real run.

The circuit is sized for a fair baseline:
- Every integer carries a value range, and each operation uses the minimal bit width, so no AND gates are wasted.
- Constants are folded: the bias column and the shifts by `K` cost nothing.
- Products use schoolbook multiplication; constants use CSD (canonical signed digit) multiplication.
- The sigmoid is exact. It is a binary search over the 31 sigmoid thresholds, costing about 5 comparisons per evaluation.

AND gates (iters=20, batch=4, qbits=6), from `lr_gc_selftest`:

| d | 2 | 5 | 10 | 20 | 50 | 100 | 200 | 500 | 1000 |
|---|---|---|---|---|---|---|---|---|---|
| ANDs | 24.7K | 73.1K | 153K | 314K | 795K | 1.60M | 3.20M | 8.01M | 16.0M |

## Why emp-ag2pc

- **emp-ag2pc** is the standard C++ implementation of authenticated garbling (WRK17). Its 2026 `main` branch adds the KRRW18
  optimizations (function-dependent half-gate leaky-AND, batched F_eq, bucketing) and SoftSpoken OT, which makes it
  the strongest baseline.
  - It supports checkpoint chunking: here there is one protocol chunk per iteration, so memory stays flat as `d` grows.
  - Its README warns that the 2026 rewrite is "AI-assisted, not yet audited". That is acceptable for benchmarking, and the
    outputs are verified against the plaintext reference.
- **emp-ag2pc branch `v0.3.x`** is the classic WRK17 `C2PC` over Bristol circuits, the version used in many papers.
  - It materializes the whole circuit in memory, so it does not scale to large `d`.
  - It lacks the KRRW18 optimizations.
- **MP-SPDZ** offers malicious binary protocols (Tiny/Tinier, BMR), but its Yao implementation is semi-honest only.
- **emp-agmpc** is the multi-party variant.
- **Semi-honest GC** (emp-sh2pc, ABY, MOTION, TinyGarble2) is not authenticated.

## Install on WSL (Ubuntu 22.04/24.04)

```bash
# 1. toolchain (CMake >= 3.21, C++20 compiler, OpenSSL >= 3)
sudo apt update
sudo apt install -y build-essential cmake git libssl-dev pkg-config

# 2. EMP, pinned to a mutually compatible snapshot (about 5 min).
#    Default prefix is $HOME/emp (no sudo); PREFIX=/usr/local for system-wide.
cd baselines/gc_ag2pc
bash scripts/install_emp.sh

# 3. build the baseline
EMP_PREFIX=$HOME/emp bash scripts/build.sh

# 4. offline correctness tests + gate counts (--quick: ~10 s; full: ~2.5 min, 11M checks)
./build/lr_gc_selftest --quick

# 5. run both parties locally, with the server checking the result (test-only)
bash scripts/run_local.sh --d 5
```

**Why the EMP versions are pinned** (`install_emp.sh` handles this):
- Pinned commits: emp-tool `2ea73a3`, emp-ot `842da83`, emp-ag2pc `a245ca0`, all from 2026-06-22.
- emp-ag2pc `main` does not compile against newer emp-tool (`joinNclean` / `bit0_mask` become ambiguous).
- emp-ag2pc has no install rules, so the script copies its headers into `$PREFIX/include`.
- To use a git checkout instead, pass `-DEMP_AG2PC_DIR=/path/to/emp-ag2pc` to CMake.

The parties can also be run by hand, in two terminals or on two machines:

```bash
./build/lr_gc_ag2pc server --port 12345 --d 5                     # listens
./build/lr_gc_ag2pc client --host <server-ip> --port 12345 --d 5   # connects (retries for 60 s)
```

Both parties must pass the same training flags. The options are:
- training: `--iters --batch --qbits --lr --seed --no-bias --zero-init`
- protocol: `--threads N` (per party, default 4), `--ssp 40`, `--chunk-iters N`
- output: `--json FILE`, `--csv FILE`, `--quiet`

Run `--help` for details. The protocol uses one TCP port; the session opens its second socket on the same port.

To sweep dimensions: `D_LIST="5 10 50 100" REPS=3 bash scripts/bench_sweep.sh` writes `results/sweep.csv`.

### WSL notes

- **Memory.** WSL gets half of the host RAM by default; this machine's WSL has 7 GB. A local run hosts both parties on one
  machine. One chunk per iteration keeps peak RSS small. To raise the limit, add `[wsl2]` / `memory=12GB` to
  `%UserProfile%\.wslconfig`, then run `wsl --shutdown`.
- **Two machines.** Use `networkingMode=mirrored` in `.wslconfig`, or forward the port, and allow the port in the
  Windows firewall.
- **LAN/WAN emulation** (optional; needs the `sch_netem` module in the WSL kernel):
  `sudo tc qdisc add dev lo root netem delay 10ms rate 1gbit`. Remove it with `sudo tc qdisc del dev lo root`.
  With a delay on `lo`, the round-trip time is twice the delay.

## Metrics reported

The program hooks emp-ag2pc's phase markers (`include/ag2pc_metrics.h`), so every phase is timed **at both parties**. Each
phase reports wall time, process CPU time, and bytes sent and received on both sockets. The two parties exchange
their numbers at the end, and each prints the combined table.

| row | emp-ag2pc phases |
|---|---|
| one-time setup | session construction: second socket, base OT, SoftSpoken COT |
| input authentication | `process_inputs` |
| fn-dependent preprocessing | liveness, mask collection, `inplace_triples` (authenticated AND triples) |
| **garbling (client) / evaluation (server)** | `garble_evaluate+c_gamma` |
| consistency checks | `cot_check`, `gather_outputs` |
| output decoding | `decode` (to the server only) |
| local | gate recording + flush planning (no crypto) |

The headline metrics map to these rows as follows:
- **Client garbling time** is the garble pass.
- **Client preprocessing total** is setup + authenticated ANDs + garbling.
- **Server evaluation time** is the evaluate pass.
- **Communication** is reported per direction and in total. Both parties' counters are cross-checked.

In authenticated garbling the preprocessing is interactive, so the server also spends time there; that time is shown too.
Garbling and evaluation are pipelined over the network, so the CPU time column shows pure compute.

### Example (this laptop: Intel Core 7 150U, WSL2, localhost, 4 threads/party, d=5)

```
phase (summed over all chunks)            |   client ms  client cpu |   server ms  server cpu |     c->s MiB     s->c MiB
one-time setup (base OT + COT session)    |      121.76       15.44 |      122.22       16.32 |        0.013        0.013
input authentication (KRRW Fig.3)         |        9.02        9.23 |        9.49        7.99 |        0.641        0.608
fn-dependent preprocessing (auth. ANDs)   |      101.62      100.30 |      101.75       99.41 |       10.062       10.061
garbling (client) / evaluation (server)   |       11.57        6.97 |       11.72        6.29 |        2.300        0.009
...
TOTAL (connected -> output opened)        |      309.64      179.95 |      309.55      177.28 |       13.029       10.703
communication: total 23.73 MiB (340.6 bytes per AND gate)
```

Scaling on the same machine (default chunking, all runs `--verify` PASS):

| d | ANDs | client garbling | server evaluation | total wall | comm (total) | peak RSS/party |
|---|---|---|---|---|---|---|
| 5 | 73K | 12 ms | 12 ms | 0.31 s | 23.7 MiB | 12 MiB |
| 17 | 266K | 46 ms | 47 ms | 0.64 s | 64.4 MiB | 21 MiB |
| 50 | 795K | 119 ms | 119 ms | 1.35 s | 191 MiB | 42 MiB |
| 100 | 1.60M | 193 ms | 194 ms | 2.55 s | 382 MiB | 75 MiB |

`--chunk-iters 20` (a single chunk) uses less communication (d=100: 295 MiB) but far more memory (d=100: 1.17 GiB per party).
`--threads` must be at least 2: emp-ag2pc deadlocks with a single pool thread.

## Files

| file | contents |
|---|---|
| `src/lr_gc_ag2pc.cpp` | the two-party program: inputs, per-iteration checkpoints, reveal to the server, report |
| `src/lr_gc_selftest.cpp` | exhaustive op tests, sigmoid tests, bit-exact training checks, gate counts |
| `include/lr_circuit.h` | the LR training circuit, generic over any emp-tool context |
| `include/sbit.h` | constant-folding bits + range-tracked signed integers (add/sub/mul/compare/shift) |
| `include/lr_reference.h` | plaintext specification + sigmoid threshold table |
| `include/lr_params.h` | parameters, deterministic data generation (splitmix64) |
| `include/ag2pc_metrics.h` | per-phase metrics hooks (must be included before emp-ag2pc) |

## Note on the FE code

In `encrypted_forward_prop_kim.cpp` and `encrypted_finetune_forward_kim.cpp`, `non_linear_transform` computes
`static_cast<int>(sigmoid(x / (1 << 10) * (1 << 5)))`: the factor `2^5` lands **inside** the sigmoid, so the
result is always 0 (or 1). The intended expression is presumably `static_cast<int>(sigmoid(x / (1 << 10)) * (1 << 5))`, and that
intended version is what this baseline computes.
