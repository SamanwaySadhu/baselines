#!/usr/bin/env bash
# Privacy regression test: runs both parties through a recording TCP proxy and checks
# that the client's private inputs never appear in what the client sends.
#
# The upstream emp-ag2pc input phase shipped every input bit as one 0/1 byte; this
# test looks for exactly that encoding (and for the packed little-endian float32
# bytes) of the client's first-chunk inputs: the initial weights, and each
# datapoint's features + label. With the patched library the client sends only
# x xor lambda, so nothing is found. --verify must still pass.
# Limits: this finds PLAINTEXT inputs only. It cannot see a leak in masked form (for
# example opening the owner's lambda share to the non-owner), and it covers d=3 with
# one chunk. It is a regression test for the specific bug fixed in third_party/emp-ag2pc,
# not a proof of security.
#   scripts/check_input_privacy.sh                 # d=3, 1 iteration
#   BIN=/path/to/lr_gc_ag2pc scripts/check_input_privacy.sh
set -uo pipefail
cd "$(dirname "$0")/.."
BIN="${BIN:-build/lr_gc_ag2pc}"
HELPER="${HELPER:-build/lr_gc_client_inputs}"
D="${D:-3}"; SEED="${SEED:-7}"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/privcheck.XXXXXX")"
free_port() { python3 -I -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])'; }
SPORT=$(free_port); PPORT=$(free_port)

"$HELPER" --d "$D" --iters 1 --seed "$SEED" > "$OUT/inputs.txt" || exit 2

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
"$BIN" server --port "$SPORT" --d "$D" --iters 1 --seed "$SEED" --verify --quiet > "$OUT/server.log" 2>&1 &
SPID=$!
"$BIN" client --port "$PPORT" --d "$D" --iters 1 --seed "$SEED" --quiet > "$OUT/client.log" 2>&1
wait "$SPID"; SRC=$?
kill "$PROXY" 2>/dev/null; wait "$PROXY" 2>/dev/null

python3 -I - "$OUT" "$D" <<'PY'
import glob, sys
out, d = sys.argv[1], int(sys.argv[2])
vals = [l.strip() for l in open(f"{out}/inputs.txt") if l.strip()]
stream = b"".join(open(p, "rb").read() for p in sorted(glob.glob(f"{out}/c2s_*.bin")))
b01 = lambda s: bytes(int(c) for c in s)
packed = lambda s: int(s[::-1], 2).to_bytes(4, "little")
records = [("initial weights", vals[:d])]
rest = vals[d:]
for i in range(4):
    records.append((f"datapoint {i} (features + label)", rest[i * d:(i + 1) * d]))
found = 0
for name, rec in records:
    for enc, pat in (("one byte per bit", b01("".join(rec))),
                     ("packed float32", b"".join(packed(v) for v in rec if len(v) == 32))):
        if len(pat) >= 8 and stream.find(pat) >= 0:
            print(f"LEAK: {name} found in client->server traffic ({enc})"); found += 1
if len(stream) < 1_000_000:
    print(f"input privacy: ERROR (captured only {len(stream)} bytes of client->server traffic; the proxy missed the run)")
    sys.exit(2)
print(f"checked {len(records)} input records against {len(stream)} bytes of client->server traffic")
print("input privacy: FAIL" if found else "input privacy: PASS (no client input found in the traffic)")
sys.exit(1 if found else 0)
PY
PRC=$?
grep -m1 -E "verify: (PASS|FAIL)" "$OUT/server.log" || { echo "server log:"; tail -5 "$OUT/server.log"; }
rm -rf "$OUT"
exit $(( PRC | SRC ))
