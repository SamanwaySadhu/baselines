// net.h -- the 4-party mesh (servers S0, S1, S2 and the client) on raw POSIX TCP sockets.
//
// Why not emp's NetIO: NetIO _Exit(1)s when a peer closes the connection, so an abort could not be
// propagated as an exception with a meaningful exit code (3 = I detected a deviation, 4 = a peer
// aborted). emp-tool is used only for AES (PRG) and SHA-256 (Hash).
//
// Topology: one TCP connection per unordered pair of parties, port = base + pair index (6 ports):
//   (S0,S1)=0 (S0,S2)=1 (S0,C)=2 (S1,S2)=3 (S1,C)=4 (S2,C)=5.
// The lower id listens, the higher id connects. Every party first binds all of its listening ports, then
// walks the pairs in lexicographic order (accept with poll / connect with a 5 ms retry), all under one
// --connect-timeout deadline. A hello {magic, id, run id} is exchanged both ways; a connection with the
// wrong hello is dropped and the party keeps accepting / reconnecting until the deadline. Deadlock-free:
// the smallest pair that is not yet connected has both ends waiting for it.
//
// Traffic: messages are framed {u32 op counter, u32 length, payload}. Each direction of each link has its
// own op counter; the receiver checks counter and length (a mismatch is a protocol deviation: AbortError).
// Every link has a dedicated sender thread (FIFO queue of owned buffers); receives run on the calling
// thread. A protocol *step* (class Step) = everything this party sends in one communication round (one
// frame per peer, all pieces concatenated in a canonical order) followed by everything it receives in
// that round; the sends are joined before the step returns. Because all sends are asynchronous, a step
// cannot deadlock for any message size.
//
// Errors: recv() == 0, ECONNRESET, EPIPE, ENOTCONN -> PeerError "closed the connection"; a receive or send
// that exceeds --timeout (SO_RCVTIMEO / SO_SNDTIMEO) -> PeerError "timed out"; a special ABORT frame
// (op = 0xFFFFFFFF, payload = reason) -> PeerError "reported abort". abort_all() is noexcept and
// idempotent: it stops the senders, shuts the sockets down (FIN after queued data), drains the receive
// buffers (so that close() does not reset the connection) and closes everything.
#ifndef FAL_NET_H
#define FAL_NET_H

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fal {

enum : int { S0 = 0, S1 = 1, S2 = 2, CL = 3, NP = 4 };
inline const char* party_name(int p) {
  static const char* n[] = {"S0", "S1", "S2", "client"};
  return (p >= 0 && p < NP) ? n[p] : "?";
}

// This party itself detected a deviation from the protocol (exit code 3).
struct AbortError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
// A peer closed the connection, timed out or reported an abort (exit code 4).
struct PeerError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline int pair_index(int a, int b) {
  static const int idx[NP][NP] = {{-1, 0, 1, 2}, {0, -1, 3, 4}, {1, 3, -1, 5}, {2, 4, 5, -1}};
  return idx[a][b];
}

struct NetConfig {
  int me = 0;
  std::array<std::string, NP> hosts{"127.0.0.1", "127.0.0.1", "127.0.0.1", "127.0.0.1"};
  int port = 12345;              // base port; the mesh uses port .. port+5
  int connect_port = 0;          // base port this party *connects* to (0 = port); lets a WAN proxy sit in front of the listeners
  uint64_t run_id = 0;           // must match at all four parties
  int connect_timeout_s = 60;
  int timeout_s = 300;           // per receive / send
};

struct Counters {
  uint64_t sent[NP] = {0, 0, 0, 0}, recv[NP] = {0, 0, 0, 0};  // bytes incl. the 8-byte frame headers
  uint64_t steps = 0;                                          // communication steps (Step::run calls)
};

class Net {
 public:
  static constexpr uint32_t kAbortOp = 0xFFFFFFFFu;
  static constexpr uint32_t kMagic = 0x464C434Eu;  // "FLCN"
  static constexpr uint32_t kMaxFrame = 1u << 30;

  explicit Net(const NetConfig& c) : cfg_(c) {
    if (c.me < 0 || c.me >= NP) throw std::invalid_argument("bad party id");
    if (c.port < 1024 || c.port + 5 > 65535) throw std::invalid_argument("--port must be in [1024, 65530]");
    ::signal(SIGPIPE, SIG_IGN);
  }
  ~Net() { abort_all(); }
  Net(const Net&) = delete;
  Net& operator=(const Net&) = delete;

  int me() const { return cfg_.me; }
  const NetConfig& config() const { return cfg_; }
  const Counters& counters() const { return ctr_; }
  void count_step() { ++ctr_.steps; }

  // ---------------------------------------------------------------- setup
  void connect() {
    using clk = std::chrono::steady_clock;
    const auto deadline = clk::now() + std::chrono::seconds(cfg_.connect_timeout_s);
    auto left_ms = [&]() -> int {
      const auto l = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - clk::now()).count();
      return l <= 0 ? 0 : (int)std::min<long long>(l, 1 << 30);
    };
    const int me = cfg_.me;
    try {
      // 1. bind + listen on every port where this party is the lower id
      for (int hi = me + 1; hi < NP; ++hi) {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) throw std::runtime_error(std::string("socket: ") + std::strerror(errno));
        lfd_[hi] = fd;
        int one = 1;
        ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_ANY);
        a.sin_port = htons((uint16_t)(cfg_.port + pair_index(me, hi)));
        while (::bind(fd, (sockaddr*)&a, sizeof a) != 0) {  // EADDRINUSE may be transient: retry until the deadline
          if (errno != EADDRINUSE || left_ms() == 0)
            throw std::runtime_error("bind port " + std::to_string(cfg_.port + pair_index(me, hi)) + ": " + std::strerror(errno));
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (::listen(fd, 8) != 0) throw std::runtime_error(std::string("listen: ") + std::strerror(errno));
      }
      // 2. walk the pairs in lexicographic order
      for (int lo = 0; lo < NP; ++lo)
        for (int hi = lo + 1; hi < NP; ++hi) {
          if (lo != me && hi != me) continue;
          const std::string pair = std::string(party_name(lo)) + "-" + party_name(hi);
          if (lo == me) {
            while (L_[hi].fd < 0) {
              if (left_ms() == 0) throw std::runtime_error("connect timeout on pair " + pair);
              pollfd p{lfd_[hi], POLLIN, 0};
              const int r = ::poll(&p, 1, std::min(left_ms(), 200));
              if (r < 0 && errno != EINTR) throw std::runtime_error(std::string("poll: ") + std::strerror(errno));
              if (r <= 0) continue;
              const int fd = ::accept(lfd_[hi], nullptr, nullptr);
              if (fd < 0) continue;
              if (hello(fd, hi, left_ms())) L_[hi].fd = fd;
              else ::close(fd);
            }
            ::close(lfd_[hi]);
            lfd_[hi] = -1;
          } else {
            sockaddr_in a{};
            a.sin_family = AF_INET;
            a.sin_port = htons((uint16_t)((cfg_.connect_port ? cfg_.connect_port : cfg_.port) + pair_index(lo, hi)));
            if (::inet_pton(AF_INET, cfg_.hosts[lo].c_str(), &a.sin_addr) != 1)
              throw std::runtime_error("bad IPv4 address for " + std::string(party_name(lo)) + ": " + cfg_.hosts[lo]);
            while (L_[lo].fd < 0) {
              if (left_ms() == 0) throw std::runtime_error("connect timeout on pair " + pair);
              const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
              if (fd < 0) throw std::runtime_error(std::string("socket: ") + std::strerror(errno));
              set_timeouts(fd, std::max(1, std::min(left_ms(), 1000)));
              if (::connect(fd, (sockaddr*)&a, sizeof a) == 0 && hello(fd, lo, left_ms())) {
                L_[lo].fd = fd;
                break;
              }
              const int e = errno;
              ::close(fd);
              if (e != 0 && e != ECONNREFUSED && e != ECONNRESET && e != ETIMEDOUT && e != EAGAIN && e != EINTR &&
                  e != ENETUNREACH && e != EHOSTUNREACH && e != EINPROGRESS)
                throw std::runtime_error("connect to " + std::string(party_name(lo)) + ": " + std::strerror(e));
              std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
          }
        }
    } catch (...) {
      abort_all();
      throw;
    }
    // 3. protocol settings + sender threads
    for (int p = 0; p < NP; ++p) {
      if (p == me) continue;
      int one = 1;
      ::setsockopt(L_[p].fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
      set_timeouts(L_[p].fd, cfg_.timeout_s * 1000);
      L_[p].th = std::thread([this, p] { sender(p); });
    }
  }

  // ---------------------------------------------------------------- frames
  // Queues one frame for `to` (the sender thread writes it). Counted as sent.
  void post(int to, std::vector<uint8_t>&& payload) {
    Link& l = L_[to];
    if (l.fd < 0) throw PeerError("no connection to " + std::string(party_name(to)));
    std::vector<uint8_t> f(8 + payload.size());
    const uint32_t h[2] = {l.tx_op++, (uint32_t)payload.size()};
    std::memcpy(f.data(), h, 8);
    if (!payload.empty()) std::memcpy(f.data() + 8, payload.data(), payload.size());
    ctr_.sent[to] += f.size();
    std::lock_guard<std::mutex> lk(l.mu);
    if (l.err) std::rethrow_exception(l.err);
    l.q.push_back(std::move(f));
    ++l.pending;
    l.cv.notify_one();
  }

  // Reads the next frame from `from`. expect >= 0: the payload must have exactly that length.
  void read_frame(int from, std::vector<uint8_t>& out, int64_t expect) {
    Link& l = L_[from];
    if (l.fd < 0) throw PeerError("no connection to " + std::string(party_name(from)));
    uint32_t h[2];
    recv_all(from, h, 8);
    if (h[0] == kAbortOp) {
      std::string why(std::min<uint32_t>(h[1], 4096), '\0');
      try { recv_all(from, why.data(), why.size()); } catch (...) {}
      throw PeerError("peer " + std::string(party_name(from)) + " reported abort (" + why + ")");
    }
    if (h[0] != l.rx_op)
      throw AbortError("protocol error: frame " + std::to_string(h[0]) + " from " + party_name(from) + ", expected frame " + std::to_string(l.rx_op));
    if (h[1] > kMaxFrame || (expect >= 0 && (int64_t)h[1] != expect))
      throw AbortError("protocol error: frame of " + std::to_string(h[1]) + " bytes from " + party_name(from) +
                       (expect >= 0 ? ", expected " + std::to_string(expect) : std::string(", too large")));
    out.resize(h[1]);
    if (h[1]) recv_all(from, out.data(), h[1]);
    ++l.rx_op;
    ctr_.recv[from] += 8 + (uint64_t)h[1];
  }

  // Waits until every queued frame has been written; rethrows a sender error.
  void drain_sends() {
    for (int p = 0; p < NP; ++p) {
      Link& l = L_[p];
      if (p == cfg_.me || l.fd < 0) continue;
      std::unique_lock<std::mutex> lk(l.mu);
      l.done.wait(lk, [&] { return l.pending == 0 || l.err; });
      if (l.err) std::rethrow_exception(l.err);
    }
  }

  // Blocks, with no deadline, until one of `peers` has data (or EOF) to read. Used by the client while the
  // servers train: training may take arbitrarily long, and liveness comes from the servers themselves (a
  // hung server is timed out by its two peers, which then send the client an ABORT frame or close). The
  // receives that follow use the per-receive --timeout again.
  void wait_any_readable(const std::vector<int>& peers) {
    std::vector<pollfd> p;
    for (int q : peers) p.push_back(pollfd{L_[q].fd, POLLIN, 0});
    for (;;) {
      const int r = ::poll(p.data(), p.size(), -1);
      if (r > 0) return;
      if (r < 0 && errno != EINTR) throw PeerError(std::string("poll: ") + std::strerror(errno));
    }
  }

  // Best effort: tells `to` that this party aborts (and why); waits up to 1 s for the frame to leave.
  void notify_abort(int to, const std::string& why) noexcept {
    try {
      Link& l = L_[to];
      if (to == cfg_.me || l.fd < 0 || closed_) return;
      const std::string w = why.substr(0, 4096);
      std::vector<uint8_t> f(8 + w.size());
      const uint32_t h[2] = {kAbortOp, (uint32_t)w.size()};
      std::memcpy(f.data(), h, 8);
      std::memcpy(f.data() + 8, w.data(), w.size());
      std::unique_lock<std::mutex> lk(l.mu);
      if (l.err || !l.th.joinable()) return;
      l.q.push_back(std::move(f));
      ++l.pending;
      l.cv.notify_one();
      l.done.wait_for(lk, std::chrono::seconds(1), [&] { return l.pending == 0 || l.err; });
    } catch (...) {
    }
  }

  // Stops everything (also the normal end of a run). noexcept, idempotent.
  void abort_all() noexcept {
    if (closed_) return;
    closed_ = true;
    aborting_ = true;
    for (int p = 0; p < NP; ++p) {
      std::lock_guard<std::mutex> lk(L_[p].mu);
      L_[p].cv.notify_all();
    }
    for (int p = 0; p < NP; ++p)
      if (L_[p].fd >= 0) ::shutdown(L_[p].fd, SHUT_WR);  // FIN after the queued data; wakes blocked senders
    for (int p = 0; p < NP; ++p)
      if (L_[p].th.joinable()) L_[p].th.join();
    for (int p = 0; p < NP; ++p) {
      if (L_[p].fd < 0) continue;
      // drain what the peer already sent, so that close() sends FIN rather than RST (an RST could destroy
      // data, e.g. an ABORT frame, that the peer has not read yet)
      char buf[4096];
      for (int k = 0; k < 4096; ++k) {
        const ssize_t r = ::recv(L_[p].fd, buf, sizeof buf, MSG_DONTWAIT);
        if (r <= 0) break;
      }
      ::close(L_[p].fd);
      L_[p].fd = -1;
    }
    for (int p = 0; p < NP; ++p)
      if (lfd_[p] >= 0) { ::close(lfd_[p]); lfd_[p] = -1; }
  }

 private:
  struct Link {
    int fd = -1;
    std::thread th;
    std::mutex mu;
    std::condition_variable cv, done;
    std::deque<std::vector<uint8_t>> q;
    size_t pending = 0;
    std::exception_ptr err;
    uint32_t tx_op = 0, rx_op = 0;
  };
  struct Hello {
    uint32_t magic, id;
    uint64_t run_id;
  };

  NetConfig cfg_;
  std::array<Link, NP> L_;
  std::array<int, NP> lfd_{-1, -1, -1, -1};
  Counters ctr_;
  std::atomic<bool> aborting_{false};
  bool closed_ = false;

  static void set_timeouts(int fd, int ms) {
    timeval tv{ms / 1000, (ms % 1000) * 1000};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
  }

  // Exchanges and checks the hello on a fresh connection to `peer`. false = drop this connection.
  bool hello(int fd, int peer, int ms) {
    set_timeouts(fd, std::max(1, ms));
    const Hello mine{kMagic, (uint32_t)cfg_.me, cfg_.run_id};
    if (::send(fd, &mine, sizeof mine, MSG_NOSIGNAL) != (ssize_t)sizeof mine) return false;
    Hello h{};
    size_t got = 0;
    while (got < sizeof h) {
      const ssize_t r = ::recv(fd, (char*)&h + got, sizeof h - got, 0);
      if (r < 0 && errno == EINTR) continue;
      if (r <= 0) { errno = ECONNRESET; return false; }
      got += (size_t)r;
    }
    if (h.magic != kMagic || h.id != (uint32_t)peer || h.run_id != cfg_.run_id) { errno = ECONNRESET; return false; }
    return true;
  }

  void sender(int p) {
    Link& l = L_[p];
    for (;;) {
      std::vector<uint8_t> job;
      {
        std::unique_lock<std::mutex> lk(l.mu);
        l.cv.wait(lk, [&] { return aborting_ || !l.q.empty(); });
        if (l.q.empty()) return;
        job = std::move(l.q.front());
        l.q.pop_front();
      }
      try {
        send_all(p, job.data(), job.size());
      } catch (...) {
        std::lock_guard<std::mutex> lk(l.mu);
        l.err = std::current_exception();
        l.q.clear();
        l.pending = 0;
        l.done.notify_all();
        return;
      }
      std::lock_guard<std::mutex> lk(l.mu);
      --l.pending;
      l.done.notify_all();
    }
  }

  void send_all(int p, const uint8_t* b, size_t n) {
    while (n > 0) {
      const ssize_t r = ::send(L_[p].fd, b, n, MSG_NOSIGNAL);
      if (r < 0) {
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) throw PeerError("peer " + std::string(party_name(p)) + " timed out");
        throw PeerError("peer " + std::string(party_name(p)) + " closed the connection");
      }
      b += r;
      n -= (size_t)r;
    }
  }

  void recv_all(int p, void* dst, size_t n) {
    char* b = (char*)dst;
    while (n > 0) {
      const ssize_t r = ::recv(L_[p].fd, b, n, 0);
      if (r == 0) throw PeerError("peer " + std::string(party_name(p)) + " closed the connection");
      if (r < 0) {
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) throw PeerError("peer " + std::string(party_name(p)) + " timed out");
        throw PeerError("peer " + std::string(party_name(p)) + " closed the connection");
      }
      b += r;
      n -= (size_t)r;
    }
  }
};

// One communication step: pieces append their outgoing data per peer and register their incoming slots
// per peer, in the same canonical order at every party; run() sends one frame per peer, receives one
// frame per peer and scatters it into the slots, joins the sends and counts one step.
class Step {
 public:
  explicit Step(Net& n) : n_(n) {}
  void send(int to, const void* p, size_t len) {
    has_out_[to] = true;
    const uint8_t* b = (const uint8_t*)p;
    out_[to].insert(out_[to].end(), b, b + len);
  }
  template <class T>
  void send_vec(int to, const std::vector<T>& v) { send(to, v.data(), v.size() * sizeof(T)); }
  void recv(int from, void* p, size_t len) {
    has_in_[from] = true;
    in_[from].push_back({(uint8_t*)p, len});
  }
  template <class T>
  void recv_vec(int from, std::vector<T>& v) { recv(from, v.data(), v.size() * sizeof(T)); }  // v pre-sized
  void recv_any(int from, std::vector<uint8_t>* out) {  // a frame of any length (whole frame)
    has_in_[from] = true;
    any_[from] = out;
  }
  void run() {
    for (int p = 0; p < NP; ++p)
      if (has_out_[p]) n_.post(p, std::move(out_[p]));
    std::vector<uint8_t> buf;
    for (int p = 0; p < NP; ++p) {
      if (!has_in_[p]) continue;
      if (any_[p]) { n_.read_frame(p, *any_[p], -1); continue; }
      size_t total = 0;
      for (const auto& s : in_[p]) total += s.len;
      n_.read_frame(p, buf, (int64_t)total);
      size_t off = 0;
      for (const auto& s : in_[p]) {
        if (s.len) std::memcpy(s.dst, buf.data() + off, s.len);
        off += s.len;
      }
    }
    n_.drain_sends();
    n_.count_step();
  }

 private:
  struct Slot {
    uint8_t* dst;
    size_t len;
  };
  Net& n_;
  std::array<std::vector<uint8_t>, NP> out_;
  std::array<std::vector<Slot>, NP> in_;
  std::array<std::vector<uint8_t>*, NP> any_{nullptr, nullptr, nullptr, nullptr};
  std::array<bool, NP> has_out_{false, false, false, false}, has_in_{false, false, false, false};
};

}  // namespace fal

#endif  // FAL_NET_H
