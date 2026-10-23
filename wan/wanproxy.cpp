// wanproxy -- user-space WAN emulator: a TCP forwarder that delays every byte by a fixed one-way latency and
// serialises it through a bandwidth-limited pipe. No root / tc / netem needed.
//
//   wanproxy --latency-ms L --bw-mbps B [--stats FILE] --map FRONT:BACK[:COUNT] [--map ...]
//
// For each port FRONT..FRONT+COUNT-1 the proxy listens on 0.0.0.0; an accepted connection is forwarded to
// 127.0.0.1:(BACK + port - FRONT). One --map entry is one *link*: the bandwidth of each direction (initiator ->
// target, target -> initiator) is a single pipe shared by all connections of that entry. Use one entry for a
// set of sockets that share a physical link (SecureML's worker sockets) and separate entries for links that are
// physically distinct (Falcon's six pairs).
//
// Model, per direction: a chunk read at time t occupies the pipe from max(t, pipe_free) for len/bw seconds and
// is written to the peer L seconds after it leaves the pipe (L = one-way latency, so RTT = 2L). FIFO order and
// pipelining are preserved. The proxy buffers without bound, so there is no TCP window / loss model: it is a
// fixed-delay, fixed-rate link. SIGTERM / SIGINT prints (and optionally writes to --stats) the traffic seen.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;
using tp = clk::time_point;

static double g_lat_s = 0.0, g_bw_Bps = 0.0;  // bw 0 = unlimited

struct Pipe {  // one direction of one link
  std::mutex mu;
  tp free_at = clk::now();
  std::atomic<uint64_t> bytes{0}, chunks{0};
};
struct Link {
  Pipe dir[2];  // 0 = initiator -> target, 1 = target -> initiator
};

struct Item {
  tp at;
  std::vector<uint8_t> data;  // empty = FIN
};
struct Chan {  // reader -> writer queue
  std::mutex mu;
  std::condition_variable cv;
  std::deque<Item> q;
};

static void set_nodelay(int fd) {
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
}

static void pump(int from, int to, Pipe* pipe) {
  Chan ch;
  std::thread writer([&] {
    for (;;) {
      Item it;
      {
        std::unique_lock<std::mutex> lk(ch.mu);
        ch.cv.wait(lk, [&] { return !ch.q.empty(); });
        it = std::move(ch.q.front());
        ch.q.pop_front();
      }
      std::this_thread::sleep_until(it.at);
      if (it.data.empty()) { ::shutdown(to, SHUT_WR); return; }
      size_t off = 0;
      while (off < it.data.size()) {
        const ssize_t r = ::send(to, it.data.data() + off, it.data.size() - off, MSG_NOSIGNAL);
        if (r <= 0) { if (r < 0 && errno == EINTR) continue; ::shutdown(from, SHUT_RD); return; }
        off += (size_t)r;
      }
    }
  });
  std::vector<uint8_t> buf(1 << 18);
  for (;;) {
    const ssize_t r = ::recv(from, buf.data(), buf.size(), 0);
    if (r < 0 && errno == EINTR) continue;
    Item it;
    const tp now = clk::now();
    if (r <= 0) {
      it.at = now + std::chrono::duration_cast<clk::duration>(std::chrono::duration<double>(g_lat_s));
    } else {
      tp done;
      {
        std::lock_guard<std::mutex> lk(pipe->mu);
        const tp start = std::max(now, pipe->free_at);
        done = start + std::chrono::duration_cast<clk::duration>(std::chrono::duration<double>(g_bw_Bps > 0 ? (double)r / g_bw_Bps : 0.0));
        pipe->free_at = done;
      }
      pipe->bytes += (uint64_t)r;
      pipe->chunks += 1;
      it.at = done + std::chrono::duration_cast<clk::duration>(std::chrono::duration<double>(g_lat_s));
      it.data.assign(buf.begin(), buf.begin() + r);
    }
    {
      std::lock_guard<std::mutex> lk(ch.mu);
      ch.q.push_back(std::move(it));
      ch.cv.notify_one();
    }
    if (r <= 0) break;
  }
  writer.join();
}

static void serve(int cfd, int back_port, Link* link) {
  int bfd = -1;
  for (int i = 0; i < 6000 && bfd < 0; ++i) {  // the target may not be listening yet: retry up to 60 s
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)back_port);
    inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    if (::connect(fd, (sockaddr*)&a, sizeof a) == 0) bfd = fd;
    else { ::close(fd); std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
  }
  if (bfd < 0) { ::close(cfd); return; }
  set_nodelay(cfd);
  set_nodelay(bfd);
  std::thread t([&] { pump(bfd, cfd, &link->dir[1]); });
  pump(cfd, bfd, &link->dir[0]);
  t.join();
  ::close(cfd);
  ::close(bfd);
}

struct Map { int front, back, count; std::unique_ptr<Link> link{new Link}; };
static std::vector<std::unique_ptr<Map>> g_maps;
static std::string g_stats;
static volatile sig_atomic_t g_stop = 0;

static void report() {
  uint64_t b0 = 0, b1 = 0, c0 = 0, c1 = 0;
  for (auto& m : g_maps) {
    b0 += m->link->dir[0].bytes; b1 += m->link->dir[1].bytes;
    c0 += m->link->dir[0].chunks; c1 += m->link->dir[1].chunks;
  }
  std::fprintf(stderr, "wanproxy: latency %.2f ms one-way, bw %.1f Mbit/s; forwarded %llu B (initiator->target) + %llu B (target->initiator)\n",
               g_lat_s * 1e3, g_bw_Bps * 8e-6, (unsigned long long)b0, (unsigned long long)b1);
  if (!g_stats.empty()) {
    if (FILE* f = std::fopen(g_stats.c_str(), "w")) {
      std::fprintf(f, "{\"latency_ms\": %.3f, \"bw_mbps\": %.3f, \"bytes_fwd\": %llu, \"bytes_rev\": %llu, \"chunks_fwd\": %llu, \"chunks_rev\": %llu}\n",
                   g_lat_s * 1e3, g_bw_Bps * 8e-6, (unsigned long long)b0, (unsigned long long)b1, (unsigned long long)c0, (unsigned long long)c1);
      std::fclose(f);
    }
  }
}

int main(int argc, char** argv) {
  signal(SIGPIPE, SIG_IGN);
  struct sigaction sa{};
  sa.sa_handler = [](int) { g_stop = 1; };
  sigaction(SIGTERM, &sa, nullptr);
  sigaction(SIGINT, &sa, nullptr);
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&]() -> const char* { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return argv[++i]; };
    if (a == "--latency-ms") g_lat_s = std::atof(val()) * 1e-3;
    else if (a == "--bw-mbps") g_bw_Bps = std::atof(val()) * 1e6 / 8.0;
    else if (a == "--stats") g_stats = val();
    else if (a == "--map") {
      int f = 0, b = 0, c = 1;
      if (std::sscanf(val(), "%d:%d:%d", &f, &b, &c) < 2) { std::fprintf(stderr, "bad --map\n"); return 2; }
      auto m = std::make_unique<Map>();
      m->front = f; m->back = b; m->count = c;
      g_maps.push_back(std::move(m));
    } else { std::fprintf(stderr, "unknown flag %s\n", a.c_str()); return 2; }
  }
  if (g_maps.empty()) { std::fprintf(stderr, "usage: wanproxy --latency-ms L --bw-mbps B --map FRONT:BACK[:COUNT]...\n"); return 2; }
  std::vector<int> lfds;
  std::vector<std::pair<Map*, int>> lmap;
  for (auto& m : g_maps)
    for (int k = 0; k < m->count; ++k) {
      const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
      int one = 1;
      setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
      sockaddr_in a{};
      a.sin_family = AF_INET;
      a.sin_addr.s_addr = htonl(INADDR_ANY);
      a.sin_port = htons((uint16_t)(m->front + k));
      if (::bind(fd, (sockaddr*)&a, sizeof a) != 0 || ::listen(fd, 64) != 0) {
        std::fprintf(stderr, "wanproxy: cannot listen on %d: %s\n", m->front + k, std::strerror(errno));
        return 1;
      }
      lfds.push_back(fd);
      lmap.push_back({m.get(), k});
    }
  std::printf("ready\n");
  std::fflush(stdout);
  std::vector<std::thread> acc;
  for (size_t i = 0; i < lfds.size(); ++i)
    acc.emplace_back([&, i] {
      for (;;) {
        const int c = ::accept(lfds[i], nullptr, nullptr);
        if (c < 0) { if (errno == EINTR) continue; return; }
        Map* m = lmap[i].first;
        std::thread(serve, c, m->back + lmap[i].second, m->link.get()).detach();
      }
    });
  while (!g_stop) ::pause();
  report();
  std::_Exit(0);
}
