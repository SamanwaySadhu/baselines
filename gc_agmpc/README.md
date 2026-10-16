# Baseline 8: multiparty authenticated garbling (WRK17, emp-agmpc), 4 clients + 1 server

This baseline trains logistic regression with **maliciously secure multiparty garbled circuits**: the authenticated
garbling of Wang, Ranellucci and Katz, *Global-Scale Secure Multiparty Computation* (CCS 2017), as implemented in
[emp-agmpc](https://github.com/emp-toolkit/emp-agmpc). It is the multiparty counterpart of `../gc_ag2pc`, which uses
the 2-party version, emp-ag2pc. It uses gc_ag2pc's exact fixed-point LR, so the two produce the same weights bit for
bit.

| role | emp-agmpc party | holds | learns |
|---|---|---|---|
| **client k** (k = 1..4) | party k+1, garbler | datapoint k of every batch: 20 rows (x, y) of dimension `--d`, plus one XOR share of the initial weights | nothing |
| **server** | party 1, evaluator | no input | **only the trained weights** |

The four clients and the server **jointly** prepare the authenticated garbled circuit. The clients hold the garbling
keys, and the server evaluates. The data and the initial weights are random for now: client k uses rows t·4 + (k−1)
of the siblings' generator, so the same `--seed` gives the same data as in the other baselines. Training is **20
iterations with a batch of 4**, one datapoint per client in every batch, lr 0.5. `--d` is the dimension, including the
bias column. `--batch B` runs B clients, for B = 1, 2 or 4 in the default build.

> **emp-agmpc is not secure as shipped, so this baseline uses a hardened copy** (`third_party/emp-agmpc`; patch list in
> [`PATCHES.md`](third_party/emp-agmpc/PATCHES.md)).
> - **Passive leaks.** An honest-but-curious server recovers every client's global MAC key Δ, and with it every input,
>   by inverting the garbled-row hash, which is fixed-key AES without feed-forward. A second leak gives it 40 parities
>   of real wire values per circuit, through a skipped tail in a MAC check.
> - **Active gaps.** Several openings are not authenticated, a garbled-row mismatch prints a warning instead of
>   aborting, nothing checks broadcast consistency, and a missing flush deadlocks multi-circuit runs.
>
> All of these were found by a code audit and confirmed with probes against upstream. Each fix is covered by
> `lr_gc_agmpc_selftest`, `scripts/test_e2e.sh` or `scripts/test_tamper.sh`.

## What is computed

The function is exactly gc_ag2pc's, with the same `include/lr_params.h`, `lr_reference.h`, `lr_circuit.h` and `sbit.h`:
- 6-bit quantized values in [−1, 31/32] (`--qbits 6`);
- an exact truncation after every dot product;
- the **exact sigmoid**, found by binary search over the 31 thresholds;
- the update `w = trunc((K·w − g)/K)` with K = batch/lr = 8.

The initial weights are `w0 = share_1 ⊕ … ⊕ share_4`, computed inside the circuit with free XORs. The server's output
equals `reference_train` **bit for bit**. `--verify`, a test-only option where the server regenerates the clients'
data from `--seed`, checks this, as do the selftest and `scripts/test_e2e.sh`.

**Circuit.** The training circuit is generated in-process from gc_ag2pc's `lr_circuit.h` by a recording context
(`include/lr_agmpc_circuit.h`). It is built one circuit per iteration. The weights leaving iteration t are carried
into iteration t+1 as **authenticated shares**: emp-agmpc's FlexOut −1 output feeds the next circuit's FlexIn −1
input, which is valid because every party keeps the same Δ. Each iteration's circuit is further cut into segments of at
most `--seg-ands` ANDs (default 280,000), chained the same way, so memory stays bounded as d grows.

| d | 5 | 50 | 100 | 1000 |
|---|---|---|---|---|
| ANDs (20 iterations) | 71.5K | 794K | 1.60M | 16.0M |
| emp-agmpc circuits | 20 | 20 | 20 | 65 |

## Protocol (WRK17, per circuit)

| phase | what | who sends what |
|---|---|---|
| **setup** (once) | base OTs and IKNP COT extension between every pair of the 5 parties; each party's global key Δ | all pairs |
| **offline / func_indep** | authenticated AND triples (leaky ANDs, random-linear-combination check, bucketing with MAC-checked `d`) and authenticated random wire masks (aBits, with check1/check2) | all pairs; about 7 KB per AND in total at 5 parties (1.4 KiB per party) |
| **offline / func_dep** | masks assigned to wires; the AND-gate openings x, y are MAC-checked and echoed; every client sends its share of each garbled table to the server (4 rows × 5 blocks per AND) | clients → server |
| **online / input** | owners' inputs: input-mask shares are opened to the owner (MAC-checked); the owner sends x ⊕ λ to everyone (echoed); carried shares are opened (MAC-checked); the clients send the input labels to the server | all pairs, clients → server |
| **online / evaluate** | the server evaluates the multiparty garbled circuit; any row matching neither key aborts | – |
| **online / output** | the server sends the output labels to the clients, who decode x ⊕ λ, so carried shares stay authenticated; for the final weights, every client opens its λ shares to the server only, MAC-checked | server → clients, clients → server |

All "offline" work is input-independent. emp-agmpc runs a circuit's offline phases just before its online phase, so
offline and online alternate per circuit, and the report sums each phase.

## Malicious security

**Model** (WRK17):
- a static, active adversary that corrupts **any subset of the 5 parties except all of them**;
- a dishonest majority: for example 4 clients colluding with each other, or the server colluding with up to 3 clients;
- security with abort, with no fairness or guaranteed output.

The server learns only the trained weights. A coalition learns nothing about an honest client's rows beyond what the
weights reveal. Pairwise channels are assumed to be private and authenticated, as in emp-agmpc; plain TCP stands in for
them in benchmarks.

| deviation (any party) | caught by | `test_tamper.sh` kind |
|---|---|---|
| wrong garbled row or wrong input label (client) | server: row matches neither key → abort | `gt-row`, `label` |
| wrong input-mask share opened to an input owner | owner: MAC digest | `in-mask` |
| owner sends different masked inputs to different parties | echo broadcast | `input-equivocate` |
| wrong opening of a carried authenticated share | MAC digest | `chain-share` |
| wrong AND-gate opening x, y | MAC digest | `xy-share` |
| wrong bucketing bit `d` / triple-generation bit `e` | MAC digest / echo | `d-bit`, `e-bit` |
| inconsistent aBit check values | check1 MAC check + echo | `abit-equivocate` |
| server sends a wrong output label | client: label matches neither label | `out-label` |
| wrong output-mask share to the server | server: MAC digest | `out-mask` |
| any inconsistent "broadcast" | echo check (all pairs compare a digest of every sender's transcript) | `echo` |
| a party that stops or disconnects | `--timeout`, then a cascade of exit 4 | – |

Over 52 runs (every kind, with the server and each client as cheater), every honest party exits 3 or 4, at least one
honest party exits 3 with an `ABORT:` line naming the failed check, and the server never prints weights.

**What this does and does not claim.**
- This is WRK17's protocol in emp-agmpc's implementation, with the gaps above fixed. There is **no formal proof for
  this code and no third-party audit**; the patches were validated by the tests listed and by reviewing them against
  the paper.
- The pinned emp-ot's KOS consistency check is itself unsound: the receiver chooses its own challenge. So the OT
  extension runs in emp-agmpc's default semi-honest mode, and aBit soundness rests on emp-agmpc's own `check1`/`check2`
  (with the tail fix and the echo). An OT-level malicious check would add defense in depth.
- Statistical security is ssp = 40 per circuit (bucket size 3–5 by circuit size). With S chained circuits the bound
  is about S·2^-40.
- The test-data convention derives every client's rows and share from `--seed`; a real client brings its own.

## Install and run (WSL / Linux)

emp-agmpc needs the **old** emp-tool / emp-ot API. They are installed into a separate prefix, `$HOME/emp-legacy`, so
the emp-tool 1.0 used by gc_ag2pc, secureml and falcon (`$HOME/emp`) is not touched.

```bash
cd baselines/gc_agmpc
bash scripts/install_emp_legacy.sh          # emp-tool 80cb3b4 + emp-ot ff28eb5 -> ~/emp-legacy (a few minutes)
bash scripts/build.sh                       # party counts 2;3;5 (batch 1, 2, 4); NP_LIST="5" builds only batch 4

./build/lr_gc_agmpc_selftest --quick        # hash maps, circuits + segmentation vs reference_train (< 1 s)
bash scripts/run_local.sh --d 5             # server + 4 clients on this machine; the server prints the report
bash scripts/test_e2e.sh                    # 8 configurations, bit-exact (about a minute)
bash scripts/test_tamper.sh --d 3 --iters 2 # every deviation x every cheater must abort (about 3 minutes)
D_LIST="5 10 50 100" bash scripts/bench_sweep.sh   # results/sweep.csv
```

By hand, in five terminals or on five machines. Hosts are listed server first, then clients 1..4; ports run from BASE
to about BASE+130.

```bash
H=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4,10.0.0.5
./build/lr_gc_agmpc server       --hosts $H --port 23456 --d 100
./build/lr_gc_agmpc client --id 1 --hosts $H --port 23456 --d 100     # ... --id 2, 3, 4
```

Options (`--help`):
- training: `--d --iters --batch --qbits --lr --no-bias --zero-init --seed`; all parties must agree, and the handshake
  checks it;
- protocol: `--hosts --port --ssp --seg-ands --threads --connect-timeout --timeout`;
- server output: `--verify` (test only), `--json`, `--csv`, `--quiet`;
- test only: `--tamper KIND --cheater P`.

Exit codes: 0 ok, 1 error, 2 verify mismatch, 3 this party detected cheating, 4 a peer aborted, closed or timed out.

## Cost report

Every party measures itself per phase and sends its table to the server at the end; the server prints the report.

| stage | phases | |
|---|---|---|
| net | `connect` | **not counted** in any total (process start skew) |
| **local** | `circuit` | generating and segmenting the circuits; the clients also prepare their inputs |
| **setup** | `handshake`, `base_ot` | one time |
| **offline** | `func_indep`, `func_dep` | input-independent, summed over all circuits |
| **online** | `input`, `evaluate`, `output` | |

Each phase records:
- **wall** time;
- **cpu** time (the whole process, including the thread pool);
- bytes **sent to** and **received from** each peer (payload).

The SUMMARY gives, **per client and for the server**:
- local / setup / offline / online / total computation (wall and cpu);
- bytes sent and received;
- the **total communication** (Σ bytes sent over the 5 parties, checked equal to Σ received), split by stage.

The CSV has, for the server and each client, `{local,setup,offline,online,total}_{wall,cpu}_ms`, `sent_bytes`,
`recv_bytes` and `peak_rss_mib`, plus `comm_{setup,offline,online,total}_bytes`.

## Results

Measured on WSL2 on one machine (12 hardware threads, 7.8 GB RAM), with all 5 processes on loopback. Settings: 20
iterations, batch 4 (4 clients), qbits 6, ssp 40, `--seg-ands 280000`. The figures are the mean of 2 runs
(`scripts/bench_sweep.sh`); d = 1000 is a single run (`run_local.sh`). Every run was checked with `--verify`, bit-exact.
Client columns give the largest of the 4 clients.

| d | ANDs | circuits | server cpu offline / online (s) | client cpu offline / online (s) | wall per party (s) | per client sent / received | server sent / received | total communication | of which online |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 71,472 | 20 | 3.6 / 0.23 | 3.0 / 0.10 | 2.8 | 101.8 MiB / 80.3 MiB | 80.3 MiB / 166.6 MiB | 486.1 MiB | 0.3 MiB |
| 10 | 151,844 | 20 | 5.9 / 0.30 | 4.9 / 0.12 | 4.1 | 212.9 MiB / 167.1 MiB | 167.2 MiB / 350.5 MiB | 1015.4 MiB | 0.6 MiB |
| 50 | 793,692 | 20 | 33.5 / 0.94 | 25.4 / 0.09 | 18.9 | 1.08 GiB / 866.7 MiB | 867.1 MiB / 1.78 GiB | 5.15 GiB | 3.0 MiB |
| 100 | 1,595,524 | 20 | 68.6 / 1.94 | 54.3 / 0.52 | 38.8 | 2.17 GiB / 1.70 GiB | 1.70 GiB / 3.58 GiB | 10.35 GiB | 5.9 MiB |
| 1000 | 16,025,052 | 65 | 758.4 / 27.9 | 601.9 / 8.4 | 439.9 | 18.73 GiB / 13.99 GiB | 14.03 GiB / 32.98 GiB | 88.65 GiB | 184.5 MiB |

Local (circuit generation) and setup (one-time base OTs, about 0.1 s and 51 KiB per party) are in `results/*.csv`;
local is about 0.06 s at d = 5 and 16 s at d = 1000. Peak RSS was 74 MiB (server) and 58 MiB (client) at d = 5, and
1.7 GiB and 1.1 GiB at d = 1000. The whole d = 1000 run peaked at 5.7 GB of system memory.

- **The cost is the offline phase**, which is input-independent. It is over 99.8% of the traffic and 88–95% of the CPU
  time (more at larger d). Triple generation runs between all 5 pairs, so it costs about 7 KB per AND in total and grows with the number
  of parties. The online phase is cheap: input, evaluation and output use 0.3 MiB at d = 5 and 185 MiB at d = 1000.
- **Clients and server cost about the same**, as in gc_ag2pc's 2-party authenticated garbling. Everyone takes part in
  the triple generation. The asymmetry is in who sends what: each client sends its share of every garbled table to the
  server (4 rows × 5 blocks per AND), so the server receives about twice what it sends. The server also does the
  evaluation, which is its online CPU.
- **Per AND, d = 1000 is cheaper than d = 100.** Its circuits reach 280,000 ANDs, so emp-agmpc uses bucket size 3
  instead of 4 in the triple generation. That is why `--seg-ands` defaults to 280,000; smaller segments save memory but
  cost about 30% more offline.
- **Comparison with gc_ag2pc.** Both run the same circuit, so the costs compare directly (run
  `../gc_ag2pc/scripts/bench_sweep.sh` with the same `D_LIST`). Here every pair of the 5 parties runs its own OT
  extension, and every wire is authenticated toward every other party.

## Files

| file | contents |
|---|---|
| `src/lr_gc_agmpc.cpp` | CLI, roles, dispatch on the number of parties |
| `include/runner.h` | one party: circuits per iteration, segments chained through authenticated shares, phases, report, CSV/JSON |
| `src/runner_np.cpp` | one explicit instantiation per party count (compiled once per entry of `AGMPC_NP_LIST`) |
| `include/lr_agmpc_circuit.h` | recording context, per-iteration circuits, weight layout, segmentation, the clients' inputs |
| `include/lr_params.h`, `lr_reference.h`, `lr_circuit.h`, `sbit.h` | copied unchanged from `../gc_ag2pc` |
| `include/metrics.h`, `run_cfg.h` | per-phase wall/cpu/bytes per peer; configuration |
| `src/lr_gc_agmpc_selftest.cpp` | GF(2^128) hash maps, circuits and segmentation vs the reference, AND counts vs gc_ag2pc |
| `third_party/emp-agmpc/` | hardened emp-agmpc headers, `PATCHES.md`, `hardening.patch`, upstream `LICENSE` |
| `scripts/` | `install_emp_legacy.sh`, `build.sh`, `run_local.sh`, `test_e2e.sh`, `test_tamper.sh`, `bench_sweep.sh` |
