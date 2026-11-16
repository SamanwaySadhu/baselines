# Baseline 10: Agarwal et al., "Communication-Efficient Secure Logistic Regression" (EuroS&P 2024)

Two parties, **data owner** and **analyst**, train logistic regression on the owner's table using the authors' own code
(`google/fss_machine_learning`, commit `4e369f0`, used unmodified). Setting: the owner holds `(X, y)` in the clear and wants to
hide it; the analyst holds nothing and ends with the model. Semi-honest, like the paper.

Defaults match the sibling baselines: **20 iterations, batch of 4 rows per iteration** (table = 80 rows), `--d` features
including the bias column (default 5), learning rate 1, L2 1e-4, 20 fractional bits, 63-bit ring. Dimensions are inputs:
`--d`, `--batch`, `--iters`.

## What the driver (`src/lr_fss.cc`) adds around the library
The library is an in-process API (its gRPC client even says "purposely incorrect" about preprocessing). The driver supplies:

* **Two parties as threads sharing nothing but a byte channel** that carries the library's real protobuf round messages.
  The channel counts bytes and rounds and emulates a link: one-way delay RTT/2 per flight, optional bandwidth cap.
  A symmetric exchange is 1 round and costs RTT/2 of latency, like `../wan`.
* **Input sharing:** analyst's share `R` is uniform and independent of the data (`getrandom`), owner keeps `data - R`.
  Initial model = a random sharing of zero (an all-zero share breaks the library's local truncation; found by testing).
* **Trusted-dealer offline phase, dealer = data owner** (v1 of the paper). The analyst must not be the dealer: it would
  know the masks `A` and could unmask `X - A`. The dealer's work runs on all cores.
* **Per iteration a fresh GD instance on that iteration's 4 rows with fresh Beaver masks** (reusing one mask across
  different batches would reveal their difference), model share carried over.
* **Accounting per party and phase:** CPU time of all threads, wall, wait (for the peer / modelled network), bytes, rounds.
* **Parallelism:** the sigmoid batch is split into chunks on all cores (messages of all chunks share one frame, so the round
  count is unchanged); dealer work is parallel over iterations and chunks. With a batch of 4 there is one chunk, so the
  online phase is effectively single-threaded; larger `--batch` uses the cores. The library's matrix products are single-threaded.

## Variants
`--variant 1` (default): MIC-gate sigmoid, 4 sigmoid rounds. `--variant 2`: comparison-based sigmoid, 6 sigmoid rounds
(the paper's variant whose offline phase can be done by 2PC; here its offline phase is still produced by the dealer, as in the
authors' code).

Online rounds = `iters x (1 + sigmoid rounds + 1) + 1` (round 1 carries `X-A`, `X^T-A`, `theta-B`; last round = gradient;
final round = owner reveals its model share to the analyst): **121** for v1, **161** for v2 at 20 iterations.

## Build and run (Linux; tested on Ubuntu 24.04 / WSL2)
```
scripts/build.sh                       # needs bazelisk as `bazel`, git, gcc; uses Bazel 5.4.1 (the repo's WORKSPACE is 2022-era)
scripts/run_local.sh                   # LAN run (RTT 0) then WAN run (RTT 50 ms), v1;  VARIANT=2 for v2
./lr_fss --d 20 --rtt-ms 50 --json out.json     # any dimensions; ./lr_fss --help
WAN_RTT_MS=80 WAN_BW_MBPS=100 scripts/run_local.sh --d 20
```
Results of the default run are in `results/` (`run_default.txt`, JSON per run).

## Results, defaults (80 rows, d = 5, 20 x 4), 12-core machine
| | v1 LAN | v1 WAN 50 ms | v2 LAN | v2 WAN 50 ms |
|---|---|---|---|---|
| online rounds | 121 | 121 | 161 | 161 |
| online wall | 0.043 s | 3.105 s | 0.057 s | 4.166 s |
| online CPU, owner / analyst | 0.040 / 0.040 s | 0.108 / 0.111 s | 0.052 / 0.053 s | 0.122 / 0.105 s |
| online network wait (modelled latency) | 0 | 2.95 s | 0 | 3.15 s |
| online traffic per party (each way) | 32.5 KiB | 32.5 KiB | 61.5 KiB | 61.6 KiB |
| offline CPU owner (dealer) | 0.12 s | 0.13 s | 0.59 s | 0.74 s |
| offline traffic (owner -> analyst, measured part) | 12.85 KiB | | 12.85 KiB | |

Secure model vs double-precision plaintext SGD: max |difference| about 7e-5, identical accuracy (0.7375 on this data).

## Caveats (read before citing numbers)
* **Offline traffic and analyst offline CPU are partial.** Input shares and matrix triples travel over the channel
  (measured). The library's sigmoid preprocessing (FSS keys etc.) is a C++ struct with no serialiser; it is handed to the
  analyst in memory, so its transfer and the analyst's deserialisation are not measured. The report prints the paper's
  analytic figures (Sec. 7.1: 2.95 KiB stored per party per sigmoid for v1; 205 KiB offline communication per sigmoid for v2)
  and labels them as not measured.
* Both parties share one machine and one process (threads); per-thread CPU is measured separately. For online threads the
  default is cores/2 per party.
* Data is synthetic (hidden logistic model, features in [-1, 1]) from `--seed`; the analyst thread never sees it. The
  plaintext check runs in the harness after the protocol.
* Input masks use `getrandom`; the library's own samplers produce the triples. No audit, no malicious security.
