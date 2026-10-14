# emp-ag2pc (patched copy)

This directory contains the headers of [emp-ag2pc](https://github.com/emp-toolkit/emp-ag2pc) at commit
`a245ca0` (2026-06-22), licensed under Apache-2.0 (`LICENSE`), with **three modifications** to two files. The baseline builds against this copy.
The headers that `scripts/install_emp.sh` installs are not used.

## Modification 1: input privacy in `AG2PCProtocol::process_inputs`

File: `emp-ag2pc/backend/protocol.h`. The diff against upstream is in `input-privacy.patch`.

**Upstream behaviour (a245ca0).** For every input wire, both parties sent their share of the wire mask λ_w to each
other, so both learned λ_w. The wire's owner then sent its **raw input bit x_w**. The other party therefore received
every private input in the clear. With the client as garbler, the server received all of the client's training data and
weights. We confirmed this by recording what the server receives in a real run: it recovered all 15 private values
exactly (d=3, one iteration).

**Patched behaviour.** This follows the input phase of WRK17 (Wang, Ranellucci, Katz, CCS 2017) and KRRW18:
1. For each wire, the non-owner opens its λ share **to the owner only**. The open is authenticated: a hash of the
   matching IT-MACs is checked against the owner's keys, and the protocol aborts on a mismatch.
2. The owner sends Γ_w = x_w ⊕ λ_w, never x_w. Γ_w is uniform to the other party because it never learns λ_w.
3. The garbler sends the label m_{w,Γ_w}. This step is unchanged.

The outputs (`Lambda`, `label0` / `eval_label`, `wire_bundle`) have the same meaning as before. Everything downstream,
including the c_γ check that binds Γ_w, is unchanged. The cost is one extra one-way message per input batch.

**Checks:**
- emp-ag2pc's own test suite (7 tests) passes with the patched headers.
- The training run matches the plaintext reference bit for bit (`--verify`).
- Recording the server's received bytes again shows about 50% agreement with the client's input bits, which is chance level.

`src/lr_gc_ag2pc.cpp` refuses to compile unless `GCAG2PC_INPUT_PRIVACY_FIX` is defined. Only this copy defines it.

**Regression test.** `scripts/check_input_privacy.sh` puts a recording TCP proxy between the parties and searches the client's
traffic for its input bits:
- With this copy it passes.
- Built against upstream (a negative control), it finds all 5 input records in plaintext and fails. `--verify` still passes in that build.

## Modification 2: the leaky-AND's leaky input in `TriplePool::compute_inplace`

File: `emp-ag2pc/backend/triple_pool.h`. The diff, together with modification 3, is in `leaky-and-mask.patch`.

**Upstream behaviour.** The function-dependent triple generation used the AND gate's real input masks as the
representative leaky-AND's inputs: region a = λ_α and region b = λ_β. In `leaky_and_halfgate` the evaluator half selects
on its region-a share bit. A malicious peer that adds an error to its half-gate ciphertext therefore makes the batched
F_eq check abort exactly when the honest party's a-share is 1. With a = λ_α, that selective failure tells the adversary the
honest party's share of a real wire mask, and with the public masked value Λ, the wire's actual value. It is caught only
half the time. WRK17 bucketing protects only a leaky input that is fresh randomness combined across the bucket.

**Patched behaviour.** Region a of layer 0 is fresh COT randomness. Region b stays λ_β, because b is the kept, non-leaky
input. The existing Beaver reconcile then opens xb = λ_α ⊕ X, which is uniform because X is the bucketed fresh
randomness. That generic path was already in the code, and yb stays 0. Cost: one extra COT per AND gate (3B-1 instead of
3B-2). That is about 3.6% more triple traffic at d=5 with 3 iterations: 101.7 instead of 98.1 MiB each way.

## Modification 3: leaky-AND hash seed per call

The leaky-AND's MITCCRH seed was a constant, and `setS` restarts the gate counter, so hash tweaks repeated across protocol
chunks (one chunk per SGD iteration here). The seed now includes a per-call counter, which both parties advance identically.

## Known issues not fixed (not on this baseline's path, or low severity)

These come from the audit:
- **Garbler's Λ in AND-free chunks.** In a chunk with zero AND gates, the garbler's Λ on XOR/NOT outputs is wrong. This only matters for `reveal(v, ALICE)` or `reveal(v, PUBLIC)` of such values. This baseline reveals only to BOB, and every chunk has AND gates.
- **Unaudited bucketing variant.** The bucketing uses a per-layer random cyclic shift, not a random permutation. Its security argument is stated in comments but has not been independently verified.
- **Base-OT session ids.** The CSW base-OT instances use the default session id. (low)
- **Invariant checks compiled out.** Some invariants are plain `assert`s, which a Release build removes. (low)

