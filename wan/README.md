# WAN simulation for the falcon, secureml and mdml baselines

No root, `tc` or `netem` is needed. Two small pieces, shared by all three baselines:

| file | what it does |
|---|---|
| `wanproxy.cpp` | TCP forwarder between the parties. Every byte is delayed by the one-way latency (RTT/2) and passes through a bandwidth-limited pipe (one pipe per link and direction). FIFO order and pipelining are kept. No TCP window or loss model; the proxy buffers without bound. |
| `wanshim.c` → `libwanshim.so` | `LD_PRELOAD` shim. It times every blocking socket receive (`recv`, `recvmsg`, `read`, `readv`, `fread` on a socket) of the protocol thread (tid == pid; worker 0 in SecureML's parallel phases). |
| `wan_wait.h` | the baselines' `Recorder`s read that counter at phase boundaries, so each phase reports `wall`, `wait` and `compute = wall - wait`. 0 without the shim. |
| `wan.sh` | sourced by each baseline's `scripts/run_local.sh`; starts/stops the proxy and arms the shim. |
| `run_all.sh` | runs the three baselines and prints their NETWORK WAIT reports. |

## Use

```
WAN_RTT_MS=40 WAN_BW_MBPS=100 falcon/scripts/run_local.sh --d 5     # same for secureml/ and mdml/
WAN_RTT_MS=0  falcon/scripts/run_local.sh --d 5                      # LAN reference, wait accounting only
wan/run_all.sh --rtt-ms 80 --bw-mbps 50 -- --d 100
```

`WAN_RTT_MS` unset leaves a run exactly as before. `WAN_BW_MBPS` unset = unlimited bandwidth. `WAN_STATS=file`
writes the proxy's byte totals.

## What "wait" means

Wait = wall time the party's protocol thread is blocked in a receive. It contains the network latency, the transfer
time at the emulated bandwidth, and time spent waiting for a slower peer (this last part is also present on the LAN;
compare with a `WAN_RTT_MS=0` run to isolate the WAN part). Compute = wall - wait. In Falcon's client column the
online wait includes the idle time while the servers train.

## Per-baseline wiring

- **falcon**: new `--connect-port` (connectors dial the proxy, listeners stay on `--port`). Six links, one pipe each.
- **secureml**: new `--connect-port` (client only). All worker sockets share one pipe per direction.
- **mdml**: `MDML_PEER_PORT` env var in the patched MD-ML `Party.cpp` (`patches/mdml-network.patch`; re-run
  `scripts/fetch_deps.sh` on a fresh checkout). MD-ML opens one socket per direction, each its own link.
