// net.h -- the TCP channel between the two CryptoNN parties (client and server),
// with exact payload byte counters and flight counting.
//
//   auto ch = Channel::listen(port);              // server: accept ONE connection
//   Channel::listen(0, "127.0.0.1", [](int p) {...}) // ephemeral port; the callback
//                                                  // gets the bound port before accept
//   auto ch = Channel::connect("127.0.0.1", port); // client: retries for ~10 s, so it
//                                                  // may start right after the server
//
// Counters (payload only; no TCP/IP headers):
//   * bytes_sent / bytes_recv: every byte handed to send() / returned by recv();
//   * flights: a FLIGHT is a maximal run of messages in one direction. A send() at the
//     start or after a recv() opens a new outgoing flight (flights_out); a recv() at
//     the start or after a send() opens a new incoming flight (flights_in). So k
//     request/response round trips give flights_out = flights_in = k at both parties,
//     and 2k flights in total on the wire.
//
// Writes are buffered (kBufBytes) and flushed when the channel switches from sending
// to receiving, when the buffer fills up, on flush() and on destruction; TCP_NODELAY is
// on, so a flushed flight leaves at once. All I/O loops until the full length is
// transferred; a closed peer or socket error throws std::runtime_error.
//
// Integers on the wire are little-endian.
#ifndef LRCN_NET_H
#define LRCN_NET_H

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace lrcn {

// Snapshot of a channel's counters; differences give per-phase traffic.
struct NetCounters {
  uint64_t sent = 0, recv = 0;            // payload bytes
  uint64_t flights_out = 0, flights_in = 0;
  uint64_t flights() const { return flights_out + flights_in; }
  NetCounters operator-(const NetCounters& o) const {
    return {sent - o.sent, recv - o.recv, flights_out - o.flights_out, flights_in - o.flights_in};
  }
};

class Channel {
 public:
  static constexpr size_t kBufBytes = 1 << 16;

  // Server side: bind (SO_REUSEADDR) to bind_host:port, accept one connection, close
  // the listening socket. port = 0 binds an ephemeral port; on_listening (if set) is
  // called with the bound port once the socket listens, before accept() blocks.
  static std::unique_ptr<Channel> listen(int port, const std::string& bind_host = "0.0.0.0",
                                         const std::function<void(int)>& on_listening = {}) {
    if (port < 0 || port > 65535) throw std::invalid_argument("net: bad port " + std::to_string(port));
    const int ls = ::socket(AF_INET, SOCK_STREAM, 0);
    if (ls < 0) sys_fail("socket");
    const int one = 1;
    ::setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    if (::inet_pton(AF_INET, bind_host.c_str(), &a.sin_addr) != 1) {
      ::close(ls);
      throw std::runtime_error("net: bad bind address " + bind_host);
    }
    if (::bind(ls, (sockaddr*)&a, sizeof a) < 0) {
      const int e = errno;
      ::close(ls);
      sys_fail("bind to port " + std::to_string(port), e);
    }
    if (::listen(ls, 1) < 0) {
      const int e = errno;
      ::close(ls);
      sys_fail("listen", e);
    }
    if (on_listening) {
      sockaddr_in b{};
      socklen_t blen = sizeof b;
      if (::getsockname(ls, (sockaddr*)&b, &blen) < 0) {
        const int e = errno;
        ::close(ls);
        sys_fail("getsockname", e);
      }
      try {
        on_listening((int)ntohs(b.sin_port));
      } catch (...) {
        ::close(ls);
        throw;
      }
    }
    int fd;
    do { fd = ::accept(ls, nullptr, nullptr); } while (fd < 0 && errno == EINTR);
    const int e = errno;
    ::close(ls);
    if (fd < 0) sys_fail("accept", e);
    return std::unique_ptr<Channel>(new Channel(fd));
  }

  // Client side: connect to host:port, retrying (connection refused / unreachable)
  // until timeout_s has passed.
  static std::unique_ptr<Channel> connect(const std::string& host, int port, double timeout_s = 10.0) {
    check_port(port);
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    const int rc = ::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res);
    if (rc != 0 || !res) throw std::runtime_error("net: cannot resolve " + host + ": " + ::gai_strerror(rc));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
    int last_err = 0;
    for (;;) {
      const int fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
      if (fd < 0) {
        const int e = errno;
        ::freeaddrinfo(res);
        sys_fail("socket", e);
      }
      int r;
      do { r = ::connect(fd, res->ai_addr, res->ai_addrlen); } while (r < 0 && errno == EINTR);
      if (r == 0) {
        ::freeaddrinfo(res);
        return std::unique_ptr<Channel>(new Channel(fd));
      }
      last_err = errno;
      ::close(fd);
      if (std::chrono::steady_clock::now() >= deadline) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ::freeaddrinfo(res);
    sys_fail("connect to " + host + ":" + std::to_string(port) + " (gave up after " +
                 std::to_string(timeout_s) + " s)", last_err);
  }

  ~Channel() {
    try { flush(); } catch (...) {}  // best effort; the peer may already be gone
    ::close(fd_);
  }
  Channel(const Channel&) = delete;
  Channel& operator=(const Channel&) = delete;

  // ---------------------------------------------------------------- raw I/O
  void send(const void* data, size_t n) {
    if (dir_ != Dir::Send) { dir_ = Dir::Send; ++c_.flights_out; }
    c_.sent += n;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    if (wbuf_.size() + n <= kBufBytes) {
      wbuf_.insert(wbuf_.end(), p, p + n);
      return;
    }
    flush();
    if (n >= kBufBytes) write_all(p, n);
    else wbuf_.insert(wbuf_.end(), p, p + n);
  }

  void recv(void* data, size_t n) {
    if (dir_ != Dir::Recv) {
      flush();  // the peer cannot answer before it has our whole flight
      dir_ = Dir::Recv;
      ++c_.flights_in;
    }
    uint8_t* p = static_cast<uint8_t*>(data);
    size_t got = 0;
    while (got < n) {
      const ssize_t r = ::recv(fd_, p + got, n - got, 0);
      if (r > 0) { got += (size_t)r; continue; }
      if (r == 0)
        throw std::runtime_error("net: peer closed the connection (expected " + std::to_string(n - got) +
                                 " more of " + std::to_string(n) + " bytes)");
      if (errno == EINTR) continue;
      sys_fail("recv");
    }
    c_.recv += n;
  }

  void flush() {
    if (wbuf_.empty()) return;
    write_all(wbuf_.data(), wbuf_.size());
    wbuf_.clear();
  }

  // ---------------------------------------------------------------- helpers
  void send_u32(uint32_t v) {
    uint8_t b[4];
    for (int i = 0; i < 4; ++i) b[i] = (uint8_t)(v >> (8 * i));
    send(b, 4);
  }
  uint32_t recv_u32() {
    uint8_t b[4];
    recv(b, 4);
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= (uint32_t)b[i] << (8 * i);
    return v;
  }
  void send_u64(uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) b[i] = (uint8_t)(v >> (8 * i));
    send(b, 8);
  }
  uint64_t recv_u64() {
    uint8_t b[8];
    recv(b, 8);
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= (uint64_t)b[i] << (8 * i);
    return v;
  }

  // n values as int32 LE (4 bytes each). Throws if a value does not fit in int32.
  void send_i32s(const int64_t* v, size_t n) {
    std::vector<uint8_t> b(4 * n);
    for (size_t k = 0; k < n; ++k) {
      if (v[k] < std::numeric_limits<int32_t>::min() || v[k] > std::numeric_limits<int32_t>::max())
        throw std::runtime_error("net: value " + std::to_string(v[k]) + " does not fit in int32");
      const uint32_t u = (uint32_t)(int32_t)v[k];
      for (int i = 0; i < 4; ++i) b[4 * k + i] = (uint8_t)(u >> (8 * i));
    }
    send(b.data(), b.size());
  }
  void send_i32s(const std::vector<int64_t>& v) { send_i32s(v.data(), v.size()); }
  // n int32 LE values, sign-extended; with lo <= hi, throws unless every value is in [lo, hi].
  std::vector<int64_t> recv_i32s(size_t n, int64_t lo = 1, int64_t hi = 0) {
    std::vector<uint8_t> b(4 * n);
    recv(b.data(), b.size());
    std::vector<int64_t> v(n);
    for (size_t k = 0; k < n; ++k) {
      uint32_t u = 0;
      for (int i = 0; i < 4; ++i) u |= (uint32_t)b[4 * k + i] << (8 * i);
      v[k] = (int32_t)u;
      if (lo <= hi && (v[k] < lo || v[k] > hi))
        throw std::runtime_error("net: received value " + std::to_string(v[k]) + " outside [" +
                                 std::to_string(lo) + ", " + std::to_string(hi) + "]");
    }
    return v;
  }

  // Raw byte strings (no length prefix: both sides know the length).
  void send_bytes(const std::vector<uint8_t>& b) { send(b.data(), b.size()); }
  std::vector<uint8_t> recv_bytes(size_t n) {
    std::vector<uint8_t> b(n);
    recv(b.data(), n);
    return b;
  }
  // Length-prefixed (u64) byte strings, for variable-size messages (e.g. reports).
  void send_blob(const std::vector<uint8_t>& b) { send_u64(b.size()); send_bytes(b); }
  std::vector<uint8_t> recv_blob(uint64_t max_len = uint64_t(1) << 32) {
    const uint64_t n = recv_u64();
    if (n > max_len) throw std::runtime_error("net: blob of " + std::to_string(n) + " bytes exceeds the limit");
    return recv_bytes((size_t)n);
  }

  // ---------------------------------------------------------------- counters
  uint64_t bytes_sent() const { return c_.sent; }
  uint64_t bytes_recv() const { return c_.recv; }
  uint64_t flights_out() const { return c_.flights_out; }
  uint64_t flights_in() const { return c_.flights_in; }
  uint64_t flights() const { return c_.flights(); }
  NetCounters counters() const { return c_; }

 private:
  enum class Dir { None, Send, Recv };
  int fd_;
  Dir dir_ = Dir::None;
  NetCounters c_;
  std::vector<uint8_t> wbuf_;

  explicit Channel(int fd) : fd_(fd) {
    const int one = 1;
    ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    wbuf_.reserve(kBufBytes);
  }

  void write_all(const uint8_t* p, size_t n) {
    size_t off = 0;
    while (off < n) {
      const ssize_t r = ::send(fd_, p + off, n - off, MSG_NOSIGNAL);  // no SIGPIPE
      if (r > 0) { off += (size_t)r; continue; }
      if (r < 0 && errno == EINTR) continue;
      if (r < 0 && (errno == EPIPE || errno == ECONNRESET))
        throw std::runtime_error("net: peer closed the connection while sending (" +
                                 std::to_string(n - off) + " bytes unsent)");
      sys_fail("send");
    }
  }

  static void check_port(int port) {
    if (port <= 0 || port > 65535) throw std::invalid_argument("net: bad port " + std::to_string(port));
  }
  [[noreturn]] static void sys_fail(const std::string& what, int e = errno) {
    throw std::runtime_error("net: " + what + " failed: " + std::strerror(e));
  }
};

}  // namespace lrcn

#endif  // LRCN_NET_H
