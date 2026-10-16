#ifndef __CMPC_CONFIG
#define __CMPC_CONFIG
// HARDENED COPY (gc_agmpc baseline). Upstream emp-agmpc 39fd6e2 hard-codes 3 loopback IPs here and aborts
// with emp::error() = exit(1). This version holds the runtime configuration (hosts, ports, timeouts), the
// abort helpers (exit 3 = this party detected cheating, exit 4 = a peer closed / timed out), the test-only
// fault-injection hook, and the GF(2^128) doubling used by the garbling hash (PATCHES.md P1).
#include <emp-tool/emp-tool.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#define EMP_AGMPC_HARDENED 1

const static int abit_block_size = 1024;
const static int fpre_threads = 1;
const static bool lan_network = false;

struct AgmpcConfig {
	std::vector<std::string> hosts;  // hosts[1..nP] (index 0 unused); default 127.0.0.1
	int connect_timeout_s = 60;
	int timeout_s = 600;             // per receive / send
	int party = 0;
};
inline AgmpcConfig g_agmpc;

inline std::string agmpc_party_name(int p) {
	return p == 1 ? std::string("server") : "client " + std::to_string(p - 1);
}

// Exit codes shared with the program: 3 = this party detected a deviation, 4 = a peer aborted / closed / timed out.
[[noreturn]] inline void agmpc_abort(int code, const std::string& msg) {
	std::fprintf(stderr, "[%s] ABORT: %s\n", agmpc_party_name(g_agmpc.party).c_str(), msg.c_str());
	std::fflush(stderr);
	std::fflush(stdout);
	_exit(code);
}
[[noreturn]] inline void agmpc_cheat(const std::string& msg) { agmpc_abort(3, msg); }

// Test-only fault injection (--tamper KIND --cheater P): fires once, at the cheating party only.
struct AgmpcTamper {
	std::string kind;
	int who = 0;
	std::atomic<bool> fired{false};
	bool hit(const char* k) {
		if (kind != k || who != g_agmpc.party) return false;
		bool exp = false;
		if (!fired.compare_exchange_strong(exp, true)) return false;
		std::fprintf(stderr, "[%s] TAMPER fired: %s\n", agmpc_party_name(g_agmpc.party).c_str(), k);
		std::fflush(stderr);
		return true;
	}
};
inline AgmpcTamper g_tamper;

// Phase marker for the cost report (set by the program; called inside CMPC::online between input / evaluation / output).
inline void (*g_phase_hook)(const char*) = nullptr;
inline void agmpc_phase(const char* name) { if (g_phase_hook) g_phase_hook(name); }

// Constant-time equality of two digests (upstream used strncmp, which stops at the first NUL byte).
inline bool digest_eq(const void* a, const void* b, size_t n = emp::Hash::DIGEST_SIZE) {
	const unsigned char *x = (const unsigned char*)a, *y = (const unsigned char*)b;
	unsigned char d = 0;
	for (size_t i = 0; i < n; ++i) d |= x[i] ^ y[i];
	return d == 0;
}

// Multiplication by x in GF(2^128) mod x^128 + x^7 + x^2 + x + 1 (bit 0 of the low word is the constant term).
inline emp::block gf128_double(const emp::block& a) {
	uint64_t lo = (uint64_t)_mm_extract_epi64(a, 0), hi = (uint64_t)_mm_extract_epi64(a, 1);
	const uint64_t carry = hi >> 63;
	hi = (hi << 1) | (lo >> 63);
	lo = (lo << 1) ^ (carry ? 0x87ULL : 0ULL);
	return emp::makeBlock(hi, lo);
}
#endif  // __CMPC_CONFIG
