# Baseline 1: two-party authenticated garbled circuits (emp-ag2pc)

This baseline trains logistic regression with **maliciously secure 2PC via authenticated garbling**. It uses
[emp-ag2pc](https://github.com/emp-toolkit/emp-ag2pc), which implements WRK17 with the KRRW18
optimizations, as the comparison point for the FE protocol.

| role | emp party | holds | learns |
|---|---|---|---|
| **client** = garbler | party 1 (ALICE) | every private input: the training data (x, y) and the initial weights (random for now) | nothing |
| **server** = evaluator | party 2 (BOB) | no private input | the trained weights |

**How the client's inputs enter.** This is the WRK17/KRRW input phase. Every input wire has a random mask λ_w, which is
an IT-MAC-authenticated bit produced by the OT-extension (COT) session. The server opens its share of λ_w **to the client
only**, and the client sends Γ_w = x_w ⊕ λ_w. The garbler then sends the label for Γ_w.

Yao-style protocols need an OT per input bit only for the *evaluator's* inputs. The server has none here, so the client's
inputs travel masked, and OT is used to generate the masks and every authenticated triple.

> **Security fixes.** The pinned upstream emp-ag2pc (`a245ca0`) had two problems. It sent every private input to the other
> party **in the clear**. Its triple generation also let a malicious party learn wire-mask shares through selective aborts.
> This baseline builds against a patched copy in `third_party/emp-ag2pc`, whose README has the evidence and the patches.
> `scripts/check_input_privacy.sh` checks that the client's inputs never appear in its traffic. Results produced before the
> fix (for example an old `results/sweep.csv`) came from the leaky library. Their timings are about the same, but they were not private.

The circuit uses the fixed-point semantics of the FE code: **6 quantization bits** (`QUANTIZATION_BITS=6`),
**batch = 4**, **learning rate = 0.5**, and the bias as the last feature with `x = MAXQ`. `--d` sets the number of features
(including the bias column) and `--iters` the SGD iterations (default 20). `--qbits`, `--batch`, `--lr`, `--no-bias`,
`--zero-init` and `--seed` change the rest.

## What is computed

Every input (feature, initial weight) is a **qbits-bit signed integer** in `[MINQ, MAXQ] = [-32, 31]` that stands for
`v / 2^f`, `f = qbits - 1 = 5`, so the values are in `[-1, 31/32]`. Labels are `y in {0, 1}` with the label value
`MAXQ * y`. With `K = batch / lr = 8`, each iteration on its batch of 4 datapoints does:

```
z_i    = trunc_f( sum_j w_j x_ij )                     dot product (scale 2^10), truncated to scale 2^5
yhat_i = min( floor(2^f * sigmoid(z_i / 2^f)), MAXQ )  exact sigmoid, 6-bit output
e_i    = yhat_i - MAXQ * y_i
g_j    = trunc_f( sum_i e_i x_ij )                     dot product (scale 2^10), truncated to scale 2^5
w_j    = trunc0( (K * w_j - g_j) / K )                 the update of backward_prop_kim.cpp
```

- **Truncation after every dot product.** A product of two scale-2^5 values has scale 2^10. Each dot product is computed
  exactly and then truncated back to scale 2^5 by an arithmetic shift right by `f = 5`: `trunc_f(v) = v >> 5 = floor(v / 32)`
  (floor, also for negative values). The shift drops the low 5 bits and costs no gates. `lr_gc_dot_sigmoid` shows this for
  one dot product, product by product.
- **"Truncate to 6 bits" means the scale, not the width.** The shift keeps the integer bits: there is no clamping to
  `[-32, 31]` and nothing wraps around. `z` can exceed the 6-bit range (for example `z = 240` for d = 8 with all inputs 31),
  and the weights grow over training (about 7 bits after one iteration and 10 bits after 20 at d = 5). Every value carries
  its exact range, and each operation uses the minimal width that holds it.
- **The update has one more rounding.** `(K * w_j - g_j) / K` rounds **toward zero** (C's `/`), as the FE code does, so
  `lr / batch = 1/8` is applied exactly up to that one rounding. `K` must be a power of two.
- **The sigmoid is exact** (no approximation): `yhat` is the true sigmoid of `z / 2^5`, floored to 6 bits, and clamped to
  `MAXQ` only where the sigmoid rounds to 1. The circuit finds it by binary search over the 31 public thresholds
  `t[v] = ceil(32 * ln(v / (32 - v)))` (`yhat >= v` iff `z >= t[v]`): 5 comparisons plus a few ANDs for the one-hot
  prefix lookup, 47 to 58 ANDs per sigmoid at d = 5.
- The arithmetic is exact, so the secure result equals the plaintext reference `reference_train` in `include/lr_reference.h`
  **bit for bit**. This is checked by `lr_gc_selftest` and by `--verify` on a real run, which also prints the distance to
  unquantized double-precision SGD on the same inputs.

The circuit is sized for a fair baseline: constants are folded (the bias column and the shifts by `K` cost nothing),
products use schoolbook multiplication over the operand with fewer secret bits, and constant multiples use CSD recoding.

AND gates (iters = 20, batch = 4, qbits = 6), from `lr_gc_selftest`. A run needs about `iters x 800 x d` ANDs for large d:

| d | 2 | 5 | 10 | 20 | 50 | 100 | 200 | 500 | 1000 |
|---|---|---|---|---|---|---|---|---|---|
| ANDs | 23.1K | 71.5K | 152K | 312K | 794K | 1.60M | 3.20M | 8.01M | 16.0M |

The float32 version of this baseline (IEEE binary32 with a polynomial sigmoid, 40 to 90 times as many ANDs) is in the git
history.

## Why emp-ag2pc

- **emp-ag2pc** is the standard C++ implementation of authenticated garbling (WRK17). Its 2026 `main` branch adds the KRRW18
  optimizations (function-dependent half-gate leaky-AND, batched F_eq, bucketing) and SoftSpoken OT, which makes it
  the strongest baseline.
  - It supports checkpoint chunking: here there is one protocol chunk per iteration, so memory stays flat as `d` grows.
  - Its README warns that the 2026 rewrite is "AI-assisted, not yet audited". That warning was justified: the input
    phase leaked every private input (fixed here, see `third_party/emp-ag2pc`). The outputs are verified against the
    plaintext reference.
- **emp-ag2pc branch `v0.3.x`** is the classic WRK17 `C2PC` over Bristol circuits, the version used in many papers.
  - It materializes the whole circuit in memory, so it does not scale to large `d`.
  - It lacks the KRRW18 optimizations.
- **MP-SPDZ** offers malicious binary protocols (Tiny/Tinier, BMR), but its Yao implementation is semi-honest only.
- **emp-agmpc** is the multi-party variant.
- **Semi-honest GC** (emp-sh2pc, ABY, MOTION, TinyGarble2) is not authenticated.

## Install on WSL (Ubuntu 22.04/24.04)

```bash
# 1. toolchain (CMake >= 3.25 for the pinned EMP libraries, C++20 compiler, OpenSSL >= 3)
sudo apt update
sudo apt install -y build-essential cmake git libssl-dev pkg-config

# 2. EMP, pinned to a mutually compatible snapshot (about 5 min).
#    Default prefix is $HOME/emp (no sudo); PREFIX=/usr/local for system-wide.
cd baselines/gc_ag2pc
bash scripts/install_emp.sh

# 3. build the baseline
EMP_PREFIX=$HOME/emp bash scripts/build.sh

# 4. offline correctness tests + gate counts (--quick: ~1 s; full: ~15 s, 8.3M checks)
./build/lr_gc_selftest --quick

# 5. run both parties locally, with the server checking the result (test-only)
bash scripts/run_local.sh --d 5

# 6. small circuit: one dot product, its truncation and one sigmoid, checked bit for bit
./build/lr_gc_dot_sigmoid local --d 8                      # emp::ClearCtx + 2000 random vectors
./build/lr_gc_dot_sigmoid server --d 8 & ./build/lr_gc_dot_sigmoid client --d 8   # over AG2PC
#    explicit inputs are integers in [-32, 31] (value * 32): --w -32,17,5 --x 31,-9,30
#    (pass the same flags to both parties)
```

**Why the EMP versions are pinned** (`install_emp.sh` handles this):
- Pinned commits: emp-tool `2ea73a3`, emp-ot `842da83`, emp-ag2pc `a245ca0`, all from 2026-06-22.
- emp-ag2pc `main` does not compile against newer emp-tool (`joinNclean` / `bit0_mask` become ambiguous).
- emp-ag2pc has no install rules, so the script copies its headers into `$PREFIX/include` and runs its tests. The
  baseline does **not** use those headers: it builds against the patched copy in `third_party/emp-ag2pc`, and refuses to compile
  against an unpatched one.

The parties can also be run by hand, in two terminals or on two machines. Locally, `run_local.sh` starts the server and then
the client as two processes on this machine, talking over localhost:

```bash
./build/lr_gc_ag2pc server --port 12345 --d 5                     # listens
./build/lr_gc_ag2pc client --host <server-ip> --port 12345 --d 5   # connects (retries for 60 s)
```

Both parties must pass the same training flags. The options are:
- training: `--d --iters --seed`, and `--qbits` (6), `--batch` (4), `--lr` (0.5; `batch / lr` must be a power of two),
  `--no-bias`, `--zero-init`
- protocol: `--threads N` (per party, default 4), `--ssp 40`, `--chunk-iters N`
- output: `--json FILE`, `--csv FILE`, `--quiet`, and `--verify` (test only)

Run `--help` for details. The protocol uses one TCP port; the session opens its second socket on the same port.

To sweep dimensions: `D_LIST="5 10 50" REPS=3 bash scripts/bench_sweep.sh` writes `results/sweep_fx.csv` (the CSV has `batch`, `qbits` and `lr` columns). Each run is checked with `--verify`.

To check input privacy: `bash scripts/check_input_privacy.sh` must print `input privacy: PASS`.

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
their numbers at the end, and each prints the combined tables.

| stage | rows | contents |
|---|---|---|
| **offline** (input-independent) | setup | session construction: second socket, base OT, SoftSpoken COT |
| | authenticated AND triples | mask collection and `inplace_triples` (leaky-AND, bucketing, F_eq) |
| **online** | input processing | `process_inputs`: the server opens its mask shares to the client, the client sends Γ = x ⊕ λ, input labels |
| | garbling (client) / evaluation (server) | `garble_evaluate+c_gamma` |
| | COT check + output decoding | `cot_check`, `decode` (to the server only) |
| **local** | gate recording + planning | liveness, `gather_outputs`, recording, flush planning: no crypto and no communication |

emp-ag2pc runs each chunk's triple generation right after that chunk's inputs. So "offline" here means *input-independent*
work measured inside the same run, not a separate phase completed before the data exists.

The per-party table gives, for the client (garbler) and the server (evaluator) separately, the wall time, CPU time and bytes
sent of each stage. It also gives the totals, with offline + online + local = total. The CSV has the same split as
`client_{offline,online,local,total}_ms`, the matching `_cpu_ms` columns, the same for `server_`, and `comm_{offline,online,total}_bytes`.

**Why the client and the server cost about the same.** In authenticated garbling the garbled circuit is not produced by
the garbler alone. Both parties hold shares of every wire mask, and they jointly generate an authenticated AND triple per
gate with OT extension. Together with the one-time setup, that symmetric step is most of the time and communication. The garbler-only work is computing the
garbled tables (about 33 B per AND, client to server), and the evaluator-only work is evaluating them. The asymmetry shows in
the online rows (the client sends the garbled tables, the server sends almost nothing online), not in the
totals. A protocol where the garbler bears most of the cost would be semi-honest Yao (not malicious) or
cut-and-choose (a different protocol family).

Authenticated garbling is interactive, so both parties are online throughout. Garbling and evaluation are pipelined over
the socket, so their wall times overlap; the CPU columns show each party's own computation.

`--threads` must be at least 2: emp-ag2pc deadlocks with a single pool thread. More than about 4 does not help: emp-ag2pc keeps only a
few tasks in flight per party (the two OT directions, the garble/evaluate pass and its network task), so each party uses about 1.2 cores
(measured with 4 to 224 threads, the run got slightly slower). The report prints the average cores kept busy.

## Files

| file | contents |
|---|---|
| `src/lr_gc_ag2pc.cpp` | the two-party program: client inputs, per-iteration checkpoints, reveal to the server, report |
| `src/lr_gc_dot_sigmoid.cpp` | test circuit: one dot product + truncation + one sigmoid (ClearCtx or AG2PC), checked against the spec |
| `src/lr_gc_selftest.cpp` | primitive exactness, truncation, sigmoid thresholds, bit-exact training checks, gate counts |
| `include/sbit.h` | constant-folding bits and range-tracked two's-complement integers over any emp-tool context |
| `include/lr_circuit.h` | the fixed-point LR training circuit (dot, truncate, qsigmoid, step), generic over any emp-tool context |
| `include/lr_reference.h` | plaintext fixed-point specification (bit-exact twin of the circuit) and the sigmoid thresholds |
| `include/lr_params.h` | parameters, deterministic generation of the client's data and initial weights (splitmix64), CLI helpers |
| `include/ag2pc_metrics.h` | per-phase metrics hooks (must be included before emp-ag2pc) |
| `src/lr_gc_client_inputs.cpp`, `scripts/check_input_privacy.sh` | input-privacy regression test (recording proxy) |
| `third_party/emp-ag2pc/` | emp-ag2pc `a245ca0` headers with the input-privacy fix (`input-privacy.patch`, `README.md`) |
