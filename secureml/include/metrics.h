// metrics.h -- per-phase wall time, process CPU time, traffic and flights of one party.
//
// Phases are named "stage/sub-phase" (stage = setup | offline | online). Each phase records
//   wall   elapsed time at this party
//   cpu    CLOCK_PROCESS_CPUTIME_ID, summed over all of this party's threads (so a phase spread over
//          N busy threads shows cpu ~ N * wall)
//   sent / recv    payload bytes on the main channel and on all worker (OT) channels
//   flights        maximal runs of same-direction traffic. The main channel carries the protocol's
//                  openings; a symmetric exchange is 2 flights = 1 round trip. For the worker
//                  channels, which run in parallel, the sum over channels and the maximum over
//                  channels (the critical path) are kept.
// Repeated records of one name accumulate.
#ifndef SML_METRICS_H
#define SML_METRICS_H

#include <sys/resource.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "link.h"

namespace sml {

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

// Plain-old-data so both parties can exchange their tables as raw bytes.
struct PhaseRow {
  char name[40];
  double wall_s, cpu_s;
  uint64_t main_sent, main_recv, ot_sent, ot_recv;
  uint64_t main_flights, ot_flights_sum, ot_flights_max;
  uint64_t ots;     // random OTs consumed
  uint64_t calls;
};

class Recorder {
 public:
  explicit Recorder(Link& l) : lk_(l) {}

  struct Mark {
    double wall = 0, cpu = 0;
    Link::Counters c;
  };
  Mark now() const { return {wall_seconds(), cpu_seconds(), lk_.counters()}; }

  // Accumulate [m, now) into phase `name`, then advance m.
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
    r->main_sent += n.c.main_sent - m.c.main_sent;
    r->main_recv += n.c.main_recv - m.c.main_recv;
    r->ot_sent += n.c.w_sent - m.c.w_sent;
    r->ot_recv += n.c.w_recv - m.c.w_recv;
    r->main_flights += n.c.main_flights - m.c.main_flights;
    r->ot_flights_sum += n.c.w_flights_sum - m.c.w_flights_sum;
    // The critical path of parallel channels: the largest per-channel flight count in this phase.
    // Approximated by the growth of the busiest channel (all channels run the same schedule).
    r->ot_flights_max += n.c.w_flights_max - m.c.w_flights_max;
    r->ots += n.c.w_rots - m.c.w_rots;
    r->calls += 1;
    m = n;
  }
  const std::vector<PhaseRow>& rows() const { return rows_; }

 private:
  Link& lk_;
  std::vector<PhaseRow> rows_;
};

// RAII helper: Scope s(rec, "offline/auth");
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
  double wall_s = 0, cpu_s = 0;
  uint64_t sent = 0, recv = 0;                 // main + OT
  uint64_t main_flights = 0, ot_flights_max = 0, ots = 0;
};
inline StageTotals stage_totals(const std::vector<PhaseRow>& rows, const char* stage) {
  StageTotals t;
  const size_t n = std::strlen(stage);
  for (const auto& r : rows) {
    if (std::strncmp(r.name, stage, n) != 0 || r.name[n] != '/') continue;
    t.wall_s += r.wall_s; t.cpu_s += r.cpu_s;
    t.sent += r.main_sent + r.ot_sent;
    t.recv += r.main_recv + r.ot_recv;
    t.main_flights += r.main_flights;
    t.ot_flights_max += r.ot_flights_max;
    t.ots += r.ots;
  }
  return t;
}

}  // namespace sml

#endif  // SML_METRICS_H
