// metrics.h -- per-phase wall time, process CPU time, traffic and flights of one party.
//
// Phases are named "stage/sub-phase", stage = offline | online | other. Phases whose sub-phase starts with "wait"
// are time spent blocked on the other party (not computation). Per phase:
//   wall     elapsed time at this party
//   cpu      CLOCK_PROCESS_CPUTIME_ID, summed over all threads of the process (cpu ~ cores x wall when parallel)
//   sent / recv   payload bytes on the channel
//   flights  maximal runs of same-direction traffic (a round trip is 2 flights)
// Repeated records of one name accumulate. The table is plain-old-data so the parties can exchange it.
#ifndef HAN_METRICS_H
#define HAN_METRICS_H

#include <sys/resource.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "net.h"

namespace han {

inline double cpu_seconds() {
  timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
inline double wall_seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline long peak_rss_kib() {
  rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_maxrss;
}
inline std::string human_bytes(uint64_t bytes) {
  const char* u[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double v = (double)bytes;
  int i = 0;
  while (v >= 1024.0 && i < 4) { v /= 1024.0; ++i; }
  char b[48];
  if (i == 0) std::snprintf(b, sizeof b, "%llu B", (unsigned long long)bytes);
  else std::snprintf(b, sizeof b, "%.2f %s", v, u[i]);
  return b;
}
inline std::string fmt_s(double s) {
  char b[32];
  if (s < 1.0) std::snprintf(b, sizeof b, "%.0f ms", s * 1e3);
  else if (s < 600) std::snprintf(b, sizeof b, "%.2f s", s);
  else std::snprintf(b, sizeof b, "%.1f min", s / 60);
  return b;
}

struct PhaseRow {
  char name[40];
  double wall_s, cpu_s;
  uint64_t sent, recv, flights, calls;
};

class Recorder {
 public:
  explicit Recorder(Channel* ch = nullptr) : ch_(ch) {}
  void attach(Channel* ch) { ch_ = ch; }

  struct Mark {
    double wall = 0, cpu = 0;
    NetCounters c;
  };
  Mark now() const { return {wall_seconds(), cpu_seconds(), ch_ ? ch_->counters() : NetCounters{}}; }

  void record(const char* name, Mark& m) {
    const Mark n = now();
    PhaseRow* r = nullptr;
    for (auto& x : rows_) if (std::strcmp(x.name, name) == 0) r = &x;
    if (!r) {
      rows_.emplace_back();
      r = &rows_.back();
      std::memset(r, 0, sizeof *r);
      std::snprintf(r->name, sizeof r->name, "%s", name);
    }
    r->wall_s += n.wall - m.wall;
    r->cpu_s += n.cpu - m.cpu;
    r->sent += n.c.sent - m.c.sent;
    r->recv += n.c.recv - m.c.recv;
    r->flights += n.c.flights() - m.c.flights();
    r->calls += 1;
    m = n;
  }
  const std::vector<PhaseRow>& rows() const { return rows_; }

 private:
  Channel* ch_;
  std::vector<PhaseRow> rows_;
};

class Scope {
 public:
  Scope(Recorder& r, const char* name) : r_(r), name_(name), m_(r.now()) {}
  ~Scope() { r_.record(name_, m_); }
 private:
  Recorder& r_;
  const char* name_;
  Recorder::Mark m_;
};

struct StageTotals {
  double wall_s = 0, cpu_s = 0, wait_s = 0;
  uint64_t sent = 0, recv = 0, flights = 0;
  double compute_s() const { return wall_s - wait_s; }
};
inline StageTotals stage_totals(const std::vector<PhaseRow>& rows, const char* stage) {
  StageTotals t;
  const size_t n = std::strlen(stage);
  for (const auto& r : rows) {
    if (std::strncmp(r.name, stage, n) != 0 || r.name[n] != '/') continue;
    t.wall_s += r.wall_s; t.cpu_s += r.cpu_s;
    if (std::strncmp(r.name + n + 1, "wait", 4) == 0) t.wait_s += r.wall_s;
    t.sent += r.sent; t.recv += r.recv; t.flights += r.flights;
  }
  return t;
}

}  // namespace han

#endif  // HAN_METRICS_H
