#!/usr/bin/env bash
# Privacy regression test: runs both parties through a recording TCP proxy and checks
# that the client's private inputs never appear in what the client sends.
#
# The upstream emp-ag2pc input phase shipped every input bit as one 0/1 byte. This test
# looks for the client's first-chunk inputs (the initial weights, and each datapoint's
# qbits-bit feature columns + label bit, as printed by lr_gc_client_inputs) in that
# encoding and in a few other plain encodings: the bits packed 8 per byte, and each
# value as a little-endian int8 / int32 / int64. With the patched library the client
# sends only x xor lambda, so nothing is found. --verify must still pass, and the
# client must exit 0 with the trained weights hidden from it.
# Patterns carrying fewer than 40 input bits are skipped: the traffic legitimately holds
# runs of thousands of 0/1 bytes (masked bits), which a random 13-bit record matches
# about one time in three and an 18-bit one about one time in a hundred. d therefore
# defaults to 8 (a datapoint = 7 six-bit columns + the label = 43 bits); at d=3 only the
# all-inputs record is long enough.
# Limits: this finds PLAINTEXT inputs only. It cannot see a leak in masked form (for
# example opening the owner's lambda share to the non-owner), and it covers one chunk.
# It is a regression test for the specific bug fixed in third_party/emp-ag2pc, not a
# proof of security.
#   scripts/check_input_privacy.sh                       # d=8, batch 4, qbits 6, 1 iteration
#   BUILD_DIR=$HOME/gc_build scripts/check_input_privacy.sh
#   BIN=/path/to/lr_gc_ag2pc HELPER=/path/to/lr_gc_client_inputs scripts/check_input_privacy.sh
#   SPORT=23480 PPORT=23481 scripts/check_input_privacy.sh   # fixed ports (default: free ones)
set -uo pipefail
cd "$(dirname "$0")/.."
BUILD_DIR="${BUILD_DIR:-build}"
BIN="${BIN:-$BUILD_DIR/lr_gc_ag2pc}"
HELPER="${HELPER:-$BUILD_DIR/lr_gc_client_inputs}"
D="${D:-8}"; BATCH="${BATCH:-4}"; QBITS="${QBITS:-6}"; SEED="${SEED:-7}"
FLAGS=(--d "$D" --batch "$BATCH" --qbits "$QBITS" --iters 1 --seed "$SEED")
OUT="$(mktemp -d "${TMPDIR:-/tmp}/privcheck.XXXXXX")"
free_port() { python3 -I -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])'; }
SPORT="${SPORT:-$(free_port)}"; PPORT="${PPORT:-$(free_port)}"

"$HELPER" "${FLAGS[@]}" > "$OUT/inputs.txt" || exit 2

python3 -I - "$PPORT" "$SPORT" "$OUT" <<'PY' &
import socket, sys, threading, time
pport, sport, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
ls = socket.socket(); ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
ls.bind(("127.0.0.1", pport)); ls.listen(8); ls.settimeout(120)
def pump(src, dst, f):
    try:
        while True:
            b = src.recv(1 << 16)
            if not b: break
            if f: f.write(b)
            dst.sendall(b)
    except OSError: pass
    finally:
        if f: f.close()
        try: dst.shutdown(socket.SHUT_WR)
        except OSError: pass
n = 0
try:
    while True:
        c, _ = ls.accept()
        for _ in range(600):  # the server may not be listening yet (or is re-listening for the sibling)
            try: s = socket.create_connection(("127.0.0.1", sport)); break
            except OSError: time.sleep(0.1)
        else: raise SystemExit("proxy: server never accepted")
        for x in (c, s): x.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        f = open(f"{out}/c2s_{n}.bin", "wb"); n += 1
        threading.Thread(target=pump, args=(c, s, f), daemon=True).start()
        threading.Thread(target=pump, args=(s, c, None), daemon=False).start()
except socket.timeout: pass
PY
PROXY=$!
"$BIN" server --port "$SPORT" "${FLAGS[@]}" --verify --quiet > "$OUT/server.log" 2>&1 &
SPID=$!
"$BIN" client --port "$PPORT" "${FLAGS[@]}" --quiet --json "$OUT/client.json" > "$OUT/client.log" 2>&1
CRC=$?
wait "$SPID"; SRC=$?
kill "$PROXY" 2>/dev/null; wait "$PROXY" 2>/dev/null

python3 -I - "$OUT" "$D" "$BATCH" "$QBITS" <<'PY'
import glob, json, sys
out, d, batch, q = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
d_data = d - 1  # the bias column is a public constant, not a client input
vals = [l.strip() for l in open(f"{out}/inputs.txt") if l.strip()]
if len(vals) != d + batch * (d_data + 1) or any(len(v) != q for i, v in enumerate(vals)
                                              if i < d or (i - d) % (d_data + 1) != d_data):
    print(f"input privacy: ERROR (unexpected helper output: {len(vals)} values)"); sys.exit(2)
stream = b"".join(open(p, "rb").read() for p in sorted(glob.glob(f"{out}/c2s_*.bin")))

def value(s):  # LSB-first bit string -> integer (two's complement; a label bit stays 0/1)
    u = int(s[::-1], 2)
    return u - (1 << len(s)) if len(s) > 1 and s[-1] == "1" else u
def packed(bits):  # bits packed LSB first, 8 per byte (whole bytes only)
    return bytes(int(bits[k:k + 8][::-1], 2) for k in range(0, len(bits) - 7, 8))
encodings = (
    ("one byte per bit", lambda rec: bytes(int(c) for c in "".join(rec))),
    ("packed bits", lambda rec: packed("".join(rec))),
    ("int8 per value", lambda rec: b"".join(value(v).to_bytes(1, "little", signed=True) for v in rec)),
    ("int32 LE per value", lambda rec: b"".join(value(v).to_bytes(4, "little", signed=True) for v in rec)),
    ("int64 LE per value", lambda rec: b"".join(value(v).to_bytes(8, "little", signed=True) for v in rec)),
)
records = [("initial weights", vals[:d])]
rest = vals[d:]
for i in range(batch):
    records.append((f"datapoint {i} (features + label)", rest[i * (d_data + 1):(i + 1) * (d_data + 1)]))
records.append(("all first-chunk inputs", vals))
found = checked = skipped = 0
for name, rec in records:
    for enc, f in encodings:
        pat = f(rec)
        info = 8 * len(pat) if enc == "packed bits" else sum(map(len, rec))
        if info < 40: skipped += 1; continue  # too few bits to tell a leak from chance
        checked += 1
        if stream.find(pat) >= 0:
            print(f"LEAK: {name} found in client->server traffic ({enc})"); found += 1
# The proxy must have seen everything the client reports sending (it also sees a few
# bytes sent before emp's counters start).
try: sent = json.load(open(f"{out}/client.json"))["client"]["total"]["sent_bytes"]
except (OSError, ValueError, KeyError): sent = None
if sent is None or len(stream) < sent:
    print(f"input privacy: ERROR (captured {len(stream)} bytes of client->server traffic, client reports "
          f"{sent} sent; the client failed or the proxy missed the run)")
    sys.exit(2)
print(f"checked {len(records)} input records ({checked} patterns, {skipped} skipped as too short; {len(vals)} "
      f"input values, {sum(map(len, vals))} bits) against {len(stream)} bytes of client->server traffic "
      f"(client reports {sent} sent)")
print("input privacy: FAIL" if found else "input privacy: PASS (no client input found in the traffic)")
sys.exit(1 if found else 0)
PY
PRC=$?
grep -m1 -E "verify: (PASS|FAIL)" "$OUT/server.log" || { echo "server log:"; tail -5 "$OUT/server.log"; }
# The client exits non-zero if the trained weights reached it (output privacy) or the
# run failed; either fails this test too.
if [ "$CRC" -ne 0 ] || ! grep -q "output hidden from the garbler: yes" "$OUT/client.log"; then
  echo "client: FAIL (exit $CRC)"; tail -5 "$OUT/client.log"; CRC=1
else
  echo "client: output hidden from the garbler: yes"
fi
rm -rf "$OUT"
exit $(( PRC | SRC | CRC ))
