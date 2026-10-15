// net.h -- the TCP channel between client and server with exact payload counters and flight counting.
//
// A FLIGHT is a maximal run of messages in one direction (a send after a recv, or a recv after a send, opens a
// new one); k request/response round trips give 2k flights. Counters are payload only (no TCP/IP headers).
// Integers on the wire are little-endian. A closed peer or socket error throws std::runtime_error.
#ifndef HAN_NET_H
#define HAN_NET_H

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
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace han {

struct NetCounters {
  uint64_t sent = 0, recv = 0, flights_out = 0, flights_in = 0;
  uint64_t flights() const { return flights_out + flights_in; }
  NetCounters operator-(const NetCounters& o) const {
    return {sent - o.sent, recv - o.recv, flights_out - o.flights_out, flights_in - o.flights_in};
  }
};

class Channel {
 public:
  static constexpr size_t kBuf = 1 << 20;

  static std::unique_ptr<Channel> listen(int port) {
    const int ls = ::socket(AF_INET, SOCK_STREAM, 0);
    if (ls < 0) fail("socket");
    const int one = 1;
    ::setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    if (::bind(ls, (sockaddr*)&a, sizeof a) < 0) { ::close(ls); fail("bind"); }
    if (::listen(ls, 1) < 0) { ::close(ls); fail("listen"); }
    const int fd = ::accept(ls, nullptr, nullptr);
    ::close(ls);
    if (fd < 0) fail("accept");
    return std::unique_ptr<Channel>(new Channel(fd));
  }

  // The client retries for ~60 s, so it may start before the server.
  static std::unique_ptr<Channel> connect(const std::string& host, int port) {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res)
      throw std::runtime_error("net: cannot resolve " + host);
    for (int attempt = 0; attempt < 600; ++attempt) {
      const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
      if (fd < 0) { ::freeaddrinfo(res); fail("socket"); }
      if (::connect(fd, res->ai_addr, res->ai_addrlen) == 0) {
        ::freeaddrinfo(res);
        return std::unique_ptr<Channel>(new Channel(fd));
      }
      ::close(fd);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ::freeaddrinfo(res);
    throw std::runtime_error("net: could not connect to " + host + ":" + std::to_string(port));
  }

  ~Channel() {
    try { flush(); } catch (...) {}
    if (fd_ >= 0) ::close(fd_);
  }

  void send(const void* p, size_t n) {
    if (last_ != Dir::Out) { ++c_.flights_out; last_ = Dir::Out; }
    c_.sent += n;
    const char* s = (const char*)p;
    while (n) {
      const size_t room = kBuf - out_.size();
      const size_t take = n < room ? n : room;
      if (take == kBuf && out_.empty()) {  // big message, no copy
        raw_send(s, kBuf);
      } else {
        out_.insert(out_.end(), s, s + take);
        if (out_.size() == kBuf) flush_buf();
      }
      s += take;
      n -= take;
    }
  }
  void recv(void* p, size_t n) {
    flush();
    if (last_ != Dir::In) { ++c_.flights_in; last_ = Dir::In; }
    c_.recv += n;
    char* d = (char*)p;
    size_t got = 0;
    while (got < n) {
      const ssize_t r = ::recv(fd_, d + got, n - got, 0);
      if (r > 0) got += (size_t)r;
      else if (r < 0 && errno == EINTR) continue;
      else throw std::runtime_error("net: connection closed by the peer");
    }
  }
  void flush() { flush_buf(); }

  template <typename T> void send_pod(const T& v) { send(&v, sizeof v); }
  template <typename T> T recv_pod() { T v; recv(&v, sizeof v); return v; }
  void send_u64(uint64_t v) { send_pod(v); }
  uint64_t recv_u64() { return recv_pod<uint64_t>(); }

  NetCounters counters() const { return c_; }

 private:
  enum class Dir { None, Out, In };
  int fd_;
  NetCounters c_;
  Dir last_ = Dir::None;
  std::vector<char> out_;

  explicit Channel(int fd) : fd_(fd) {
    const int one = 1;
    ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    out_.reserve(kBuf);
  }
  [[noreturn]] static void fail(const char* what) { throw std::runtime_error(std::string("net: ") + what + ": " + std::strerror(errno)); }
  void raw_send(const char* s, size_t n) {
    while (n) {
      const ssize_t w = ::send(fd_, s, n, MSG_NOSIGNAL);
      if (w > 0) { s += w; n -= (size_t)w; }
      else if (w < 0 && errno == EINTR) continue;
      else throw std::runtime_error("net: connection closed by the peer");
    }
  }
  void flush_buf() {
    if (!out_.empty()) { raw_send(out_.data(), out_.size()); out_.clear(); }
  }
};

}  // namespace han

#endif  // HAN_NET_H
