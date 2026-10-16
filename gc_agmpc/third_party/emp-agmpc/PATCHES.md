# Hardened emp-agmpc

This directory contains a patched copy of the headers of [emp-agmpc](https://github.com/emp-toolkit/emp-agmpc), pinned at
master `39fd6e2`. Its headers are identical to `43e4953` (2021-07-20), the last code change. emp-agmpc implements
Wang, Ranellucci and Katz, *Global-Scale Secure Multiparty Computation*, CCS 2017 (WRK17).

- `hardening.patch` is the complete diff against upstream.
- `LICENSE` is upstream's.
- The copy builds against emp-tool `80cb3b4` and emp-ot `ff28eb5`, the pinned legacy API (see `../../scripts/install_emp_legacy.sh`).

As shipped, upstream is **not even passively secure** in our setting. An honest-but-curious evaluator (the server)
recovers every garbler's global key Δ, and with it every client's input (P1 below). A second passive leak hands the
server 40 parities of real wire values per circuit (P3). The other gaps let a malicious party change the result
undetected or desynchronise the parties.

The gaps were found by a code audit and checked with probes run against the upstream code. Every patch except the
pure-robustness and API changes (P9, P10, P11, P13, P14) has a deviation in `scripts/test_tamper.sh` that it
must catch.

| id | upstream (file:line @ 39fd6e2) | problem | patch | test |
|---|---|---|---|---|
| **P1** | `mpc.h:441-466` `CMPC::Hash`, with `PRP()` = AES under the all-zero key | The garbled-row hash is fixed-key AES with **no feed-forward**: `H = π(σ(a) ⊕ σ²(b) ⊕ tweak)`. The evaluator inverts it and recovers each garbler's Δ, then every mask and input. A probe recovered 4/4 Δs, 6800/6800 gate masks and 256/256 input bits on AES. Adding feed-forward to the same input map is **still broken**: σ ⊕ σ² is the identity, so for the row opposite the evaluated one, the Δ inside the hash cancels against the z·Δ in the payload. A reviewer recovered Δ on about 50% of gates that way. | `K = 2·a ⊕ 4·b ⊕ T` in GF(2^128), `H = π(K) ⊕ K` (MMO). The tweak `T` is unique per (instance, gate, row, slot). Row differences are c·Δ with c ∈ {2, 4, 6}, and with the payload term they become (c ⊕ z)·Δ, which is never 0. This is BHKR13's A4-style dual-key cipher in the random-permutation model. | selftest: ranks of 2, 4, 6, 3, 5, 7 are all 128, and σ ⊕ σ² ⊕ I has rank 0 |
| **P2** | `mpc.h:404, 577, 661` | A garbled row matching neither key only prints `no match GT!`, then evaluation continues with a wrong value | abort (exit 3) | `gt-row`, `label` |
| **P3** | `abitmp.h:150` `check1` | The loop runs only to `length/2048`, so the last `length mod 2048` rows are skipped, including the 3·ssp masking rows. The opened combinations then give the server 40 parities of real wire values per circuit (probe: 40/40). A one-past-the-end read also sits at `:144`. | loop to `ceil(length/2048)`; treat the row after the last as 0 | (passive leak; covered by the e2e runs and the `abit-equivocate` run) |
| **P4** | `mpc.h:182-205` | The AND-gate openings x, y carry no MAC. A party can flip a gate's output undetected. | Each sender sends its bits plus one SHA-256 digest of its MACs under the receiver's key; the receiver recomputes the digest from its keys. Then echo (P12). | `xy-share` |
| **P5** | `fpremp.h:252-286` | The bucketing bits d carry no MAC (WRK17 Fig. 18 step 3a: "reveal d with its MACs checked") | authenticated opening as in P4, plus echo | `d-bit` |
| **P6** | `abitmp.h` (`IKNP` constructed semi-honest) | The pinned emp-ot's `malicious=true` KOS check is unsound: the receiver picks the challenge seed itself, and the coefficients repeat every 2048 rows. | Left semi-honest, **on purpose**. aBit soundness rests on `check1` with full row coverage (P3), joint coins from `sampleRandom`, and the echo (P12). | (`abit-equivocate`) |
| **P7** | `flexible_input_output.h:120-421` FlexIn | The masked private inputs are sent point-to-point with no consistency check. The `-2` mode is unauthenticated. | Only private (>0) and authenticated-share (−1) inputs are allowed. All openings are MAC-checked, the masked inputs are echoed, and `echo_check` runs before any label is sent. | `in-mask`, `input-equivocate`, `chain-share` |
| **P8** | `flexible_input_output.h` FlexIn/FlexOut | Every bit was sent as a 32-byte struct with 15 uninitialized padding bytes, to every peer, in every mode (about 98 bytes per bit per pair) | 1 byte per bit, only to the parties that need it, plus one 32-byte digest. Outputs go to one party (the server), MAC-checked. | `out-mask`, `out-label` |
| **P9** | `mpc.h:58` | Every CMPC instance builds a new `FpreMP` (base OTs, COT extenders, Δ) | One `FpreMP` is shared by every instance of the run, as WRK17's one-time setup; Δ must stay the same for carried shares anyway. Tweaks are domain-separated per instance (P1, P15). | e2e runs with up to 21 chained circuits per iteration |
| **P10** | `emp::error()` = `exit(1)`; `helper.h:149` `sampleRandom` `exit(0)` on cheating; legacy `NetIO::recv` busy-loops forever on EOF | Aborts were not coordinated or distinguishable, and peers hung | `agmpc_cheat` prints `ABORT: <reason>` and exits **3**. A peer that closes or times out gives exit **4** (`SafeNetIO` replaces `NetIO`: EOF, error and `SO_RCVTIMEO` handling, connect deadlines, receive counters). | every tamper run ends with 3/4 at every honest party |
| **P11** | `cmpc_config.h:12-15` `IP[]` (3 entries) | hard-coded loopback hosts | hosts, ports and timeouts come from the command line | – |
| **P12** | `abitmp.h:104-199` `check1` and the other broadcasts | WRK17 assumes a broadcast channel (Π_naBit, Fig. 14 broadcasts the combined bits). Here every value is sent point-to-point, so a party can open different bits to different peers, and per-receiver MAC checks do not prevent that. | **Echo broadcast**: every broadcast value (the check1 combinations, check2 commitments, `sampleRandom` commitments and openings, Fpre `e`, the LaAND commitments, `d`, x/y, the masked inputs, the opened shares) goes into a running per-sender transcript. All pairs compare `H(T_1 ‖ … ‖ T_n)`. | `echo`, `e-bit`, `input-equivocate` |
| **P13** | `abitmp.h:136-138` | `check1` leaks about 0.5 MB+ per call (two calls per circuit) | freed | – |
| **P14** | `abitmp.h:255-260` `check2` | A send task returns without flushing, which deadlocked 7 of 10 multi-circuit runs | flush | e2e runs with many segments |
| **P15** | `fpremp.h` `garble` / `evaluate` / `HnID` | The HaAND and Φ hashes queried the same fixed-key CRH with colliding tweaks (`4k` vs `2k`), and every call restarted the tweaks at 0 under the same Δ | tweak high word = (domain, call counter) | – |
| – | `strncmp` on digests (`mpc.h:503`, `fpremp.h:221`, `helper.h:143`, `abitmp.h:270, 302, 304`) | stops at the first NUL byte (about 1/256 per byte) | constant-time `digest_eq` | – |

Not changed (out of scope, documented in the README):
- WRK17's statistical security is per circuit. With S chained circuits the failure bound grows to about S·2^-ssp.
- The bucket size depends only on the circuit size (`fpremp.h:40-48`), not on `ssp`.
