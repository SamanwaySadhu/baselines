#ifndef NETIOMP_H__
#define NETIOMP_H__
// HARDENED COPY (gc_agmpc baseline), rewritten from upstream netmp.h (39fd6e2). Changes (PATCHES.md P10, P11, P12):
//  * SafeNetIO replaces emp::NetIO: on EOF the legacy NetIO busy-loops forever in fread; here EOF, a socket
//    error or a --timeout (SO_RCVTIMEO / SO_SNDTIMEO) ends the process with "ABORT: peer ... closed / timed
//    out" (exit 4). Connects and accepts have a --connect-timeout deadline. Received bytes are counted.
//  * Hosts and ports come from g_agmpc (CLI) instead of the hard-coded IP[] table; the LOCALHOST port scheme
//    of upstream (one port per ordered pair) is used for every host, so parties can run on one or many machines.
//  * Echo broadcast (GL05-style, WRK17 Fig. 14/15 assume a broadcast channel): every value the protocol
//    broadcasts is appended to a running per-sender transcript T_s; echo_check() exchanges a digest of
//    (T_1..T_nP) between all pairs and aborts on any mismatch, so a party cannot send different "broadcast"
//    values to different peers.
#include <emp-tool/emp-tool.h>
#include "cmpc_config.h"
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <mutex>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
using namespace emp;

class SafeNetIO : public IOChannel<SafeNetIO> { public:
	int consocket = -1;
	FILE* stream = nullptr;
	char* buffer = nullptr;
	bool has_sent = false;
	int peer = 0;
	uint64_t recv_counter = 0;

	SafeNetIO(const char* address, int port, int peer_) : peer(peer_) {
		using clk = std::chrono::steady_clock;
		const auto deadline = clk::now() + std::chrono::seconds(g_agmpc.connect_timeout_s);
		auto left_ms = [&]() { return (long)std::chrono::duration_cast<std::chrono::milliseconds>(deadline - clk::now()).count(); };
		const std::string who = agmpc_party_name(peer) + " (port " + std::to_string(port) + ")";
		if (address == nullptr) {
			int ls = socket(AF_INET, SOCK_STREAM, 0);
			int reuse = 1;
			setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));
			sockaddr_in serv{};
			serv.sin_family = AF_INET;
			serv.sin_addr.s_addr = htonl(INADDR_ANY);
			serv.sin_port = htons((uint16_t)port);
			while (bind(ls, (sockaddr*)&serv, sizeof(serv)) < 0) {
				if (errno != EADDRINUSE || left_ms() <= 0) agmpc_abort(1, "cannot bind port for " + who + ": " + std::strerror(errno));
				usleep(20000);
			}
			if (listen(ls, 1) < 0) agmpc_abort(1, "listen failed for " + who);
			for (;;) {
				pollfd p{ls, POLLIN, 0};
				const long l = left_ms();
				if (l <= 0) agmpc_abort(1, "connect timeout waiting for " + who);
				const int r = poll(&p, 1, (int)std::min<long>(l, 1000));
				if (r < 0 && errno != EINTR) agmpc_abort(1, "poll failed for " + who);
				if (r > 0) { consocket = accept(ls, nullptr, nullptr); if (consocket >= 0) break; }
			}
			close(ls);
		} else {
			sockaddr_in dest{};
			dest.sin_family = AF_INET;
			dest.sin_addr.s_addr = inet_addr(address);
			dest.sin_port = htons((uint16_t)port);
			for (;;) {
				consocket = socket(AF_INET, SOCK_STREAM, 0);
				if (connect(consocket, (sockaddr*)&dest, sizeof(dest)) == 0) break;
				close(consocket);
				if (left_ms() <= 0) agmpc_abort(1, "connect timeout connecting to " + who + " at " + address);
				usleep(2000);
			}
		}
		const int one = 1;
		setsockopt(consocket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
		timeval tv{};
		tv.tv_sec = g_agmpc.timeout_s;
		setsockopt(consocket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
		setsockopt(consocket, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
		stream = fdopen(consocket, "wb+");
		buffer = new char[NETWORK_BUFFER_SIZE];
		memset(buffer, 0, NETWORK_BUFFER_SIZE);
		setvbuf(stream, buffer, _IOFBF, NETWORK_BUFFER_SIZE);
	}
	~SafeNetIO() {
		if (stream) { fflush(stream); fclose(stream); }
		delete[] buffer;
	}
	[[noreturn]] void peer_failed(const char* what) {
		const int e = errno;
		const bool timeout = (e == EAGAIN || e == EWOULDBLOCK);
		agmpc_abort(4, "peer " + agmpc_party_name(peer) + (timeout ? " timed out" : std::string(" closed the connection")) +
		                   " (" + what + (timeout ? "" : std::string(", ") + std::strerror(e)) + ")");
	}
	void flush() {
		if (fflush(stream) != 0) peer_failed("send");
	}
	void send_data_internal(const void* data, int len) {
		int sent = 0;
		while (sent < len) {
			const size_t res = fwrite(sent + (const char*)data, 1, (size_t)(len - sent), stream);
			if (res == 0) peer_failed("send");
			sent += (int)res;
		}
		has_sent = true;
	}
	void recv_data_internal(void* data, int len) {
		if (has_sent) flush();
		has_sent = false;
		int got = 0;
		while (got < len) {
			errno = 0;
			const size_t res = fread(got + (char*)data, 1, (size_t)(len - got), stream);
			if (res == 0) {
				if (errno == EINTR) { clearerr(stream); continue; }
				peer_failed("receive");
			}
			got += (int)res;
		}
		recv_counter += (uint64_t)len;
	}
	void sync() {}
};

// Running per-sender transcripts of broadcast values (process-wide: the protocol uses two NetIOMP objects).
inline unsigned char g_bcast_T[16][32];
inline std::mutex g_bcast_mu[16];
inline uint64_t g_echo_rounds = 0;

template<int nP>
class NetIOMP { public:
	SafeNetIO* ios[nP+1];
	SafeNetIO* ios2[nP+1];
	int party;
	bool sent[nP+1];
	NetIOMP(int party, int port) {
		static_assert(nP + 1 <= 16, "at most 15 parties");
		this->party = party;
		memset(sent, false, nP+1);
		for (int i = 0; i <= nP; ++i) { ios[i] = nullptr; ios2[i] = nullptr; }
		auto host = [&](int j) { return (j < (int)g_agmpc.hosts.size() && !g_agmpc.hosts[j].empty()) ? g_agmpc.hosts[j].c_str() : "127.0.0.1"; };
		// Same pair order and port scheme as upstream's LOCALHOST build (deadlock-free: every party walks the pairs
		// in the same order; the lower id connects first on the even port, the higher id first listens on it).
		for (int i = 1; i <= nP; ++i) for (int j = 1; j <= nP; ++j) if (i < j) {
			if (i == party) {
				ios[j] = new SafeNetIO(host(j), port + 2*(i*nP+j), j);
				ios2[j] = new SafeNetIO(nullptr, port + 2*(i*nP+j) + 1, j);
			} else if (j == party) {
				ios[i] = new SafeNetIO(nullptr, port + 2*(i*nP+j), i);
				ios2[i] = new SafeNetIO(host(i), port + 2*(i*nP+j) + 1, i);
			}
		}
	}
	static int port_span() { return 2*(nP*nP + nP) + 2; }
	int64_t count() {
		int64_t res = 0;
		for (int i = 1; i <= nP; ++i) if (i != party) res += ios[i]->counter + ios2[i]->counter;
		return res;
	}
	uint64_t sent_to(int j) const { return ios[j]->counter + ios2[j]->counter; }
	uint64_t recv_from(int j) const { return ios[j]->recv_counter + ios2[j]->recv_counter; }

	~NetIOMP() {
		for (int i = 1; i <= nP; ++i)
			if (i != party) {
				delete ios[i];
				delete ios2[i];
			}
	}
	void send_data(int dst, const void * data, size_t len) {
		if (dst != 0 and dst != party) {
			if (party < dst)
				ios[dst]->send_data(data, (int)len);
			else
				ios2[dst]->send_data(data, (int)len);
			sent[dst] = true;
		}
#ifdef __MORE_FLUSH
		flush(dst);
#endif
	}
	void recv_data(int src, void * data, size_t len) {
		if (src != 0 and src != party) {
			if (sent[src]) flush(src);
			if (src < party)
				ios[src]->recv_data(data, (int)len);
			else
				ios2[src]->recv_data(data, (int)len);
		}
	}
	SafeNetIO*& get(size_t idx, bool b = false) {
		if (b)
			return ios[idx];
		else return ios2[idx];
	}
	void flush(int idx = 0) {
		if (idx == 0) {
			for (int i = 1; i <= nP; ++i)
				if (i != party) {
					ios[i]->flush();
					ios2[i]->flush();
				}
		} else {
			if (party < idx)
				ios[idx]->flush();
			else
				ios2[idx]->flush();
		}
	}
	void sync() {}

	// Append `len` bytes broadcast by `sender` to its transcript (called by the sender for its own value, and by
	// every receiver for the value it received).
	static void note(int sender, const void* data, size_t len) {
		std::lock_guard<std::mutex> lk(g_bcast_mu[sender]);
		emp::Hash h;
		h.put(g_bcast_T[sender], 32);
		uint64_t l = len;
		h.put(&l, 8);
		if (len) h.put(data, (int)len);
		h.digest((char*)g_bcast_T[sender]);
	}
	static void reset_transcripts() { memset(g_bcast_T, 0, sizeof(g_bcast_T)); }
	// All pairs exchange H(T_1 || ... || T_nP) and abort (exit 3) on any mismatch.
	void echo_check(const char* where) {
		unsigned char D[32], R[32];
		{
			emp::Hash h;
			for (int s = 1; s <= nP; ++s) { std::lock_guard<std::mutex> lk(g_bcast_mu[s]); h.put(g_bcast_T[s], 32); }
			h.digest((char*)D);
		}
		if (g_tamper.hit("echo")) D[0] ^= 1;
		for (int j = 1; j <= nP; ++j) if (j != party) { send_data(j, D, 32); flush(j); }
		for (int j = 1; j <= nP; ++j) if (j != party) {
			recv_data(j, R, 32);
			if (!digest_eq(D, R, 32)) agmpc_cheat(std::string("broadcast echo mismatch with ") + agmpc_party_name(j) + " (" + where + ")");
		}
		++g_echo_rounds;
	}
};
#endif //NETIOMP_H__
