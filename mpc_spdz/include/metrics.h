// metrics.h -- per-phase wall time, process CPU time, traffic per peer and rounds of one party.
//
// Phases are named "stage/sub-phase" (stage = setup | offline | online). Each phase records
//   wall      elapsed time at this party
//   cpu       CLOCK_PROCESS_CPUTIME_ID, summed over all of this party's threads (so a phase spread over
//             N busy threads shows cpu ~ N * wall)
//   sent/recv payload bytes to / from every other party (main channel + all of that pair's OT channels)
//   rounds    protocol rounds on the main channels: one per "everybody sends to everybody" exchange.
//             Every party counts the same number (the schedule is data independent).
//   ot_fl     for the OT channels, which run in parallel: the growth of the busiest channel's flight
//             counter (maximal runs of same-direction traffic) -- the critical path of the OT work
// Repeated records of one name accumulate.
#ifndef SPZ_METRICS_H
#define SPZ_METRICS_H

#include <sys/resource.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "mesh.h"

namespace spz {

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

// Plain-old-data so the parties can send their tables to the server as raw bytes.
struct PhaseRow {
  char name[40];
  double wall_s, cpu_s;
  uint64_t sent[kMaxParties], recv[kMaxParties];
  uint64_t rounds, ot_flights_max, ots, calls;
  uint64_t sent_total() const { uint64_t s = 0; for (int i = 0; i < kMaxParties; ++i) s += sent[i]; return s; }
  uint64_t recv_total() const { uint64_t s = 0; for (int i = 0; i < kMaxParties; ++i) s += recv[i]; return s; }
};

class Recorder {
 public:
  explicit Recorder(Mesh& m) : mesh_(m) {}

  struct Mark {
    double wall = 0, cpu = 0;
    Mesh::Snapshot s;
  };
  Mark now() const { return {wall_seconds(), cpu_seconds(), mesh_.snapshot()}; }

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
    for (int p = 0; p < kMaxParties; ++p) {
      r->sent[p] += n.s.sent[p] - m.s.sent[p];
      r->recv[p] += n.s.recv[p] - m.s.recv[p];
    }
    r->rounds += n.s.rounds - m.s.rounds;
    r->ots += n.s.rots - m.s.rots;
    uint64_t mx = 0;
    for (size_t c = 0; c < n.s.ot_flights.size() && c < m.s.ot_flights.size(); ++c)
      mx = std::max<uint64_t>(mx, n.s.ot_flights[c] - m.s.ot_flights[c]);
    r->ot_flights_max += mx;
    r->calls += 1;
    m = n;
  }
  const std::vector<PhaseRow>& rows() const { return rows_; }

 private:
  Mesh& mesh_;
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
  uint64_t sent = 0, recv = 0;
  uint64_t sent_to[kMaxParties] = {0}, recv_from[kMaxParties] = {0};
  uint64_t rounds = 0, ot_flights_max = 0, ots = 0;
};
inline StageTotals stage_totals(const std::vector<PhaseRow>& rows, const char* stage) {
  StageTotals t;
  const size_t n = std::strlen(stage);
  for (const auto& r : rows) {
    if (std::strncmp(r.name, stage, n) != 0 || r.name[n] != '/') continue;
    t.wall_s += r.wall_s; t.cpu_s += r.cpu_s;
    for (int p = 0; p < kMaxParties; ++p) { t.sent_to[p] += r.sent[p]; t.recv_from[p] += r.recv[p]; }
    t.sent += r.sent_total();
    t.recv += r.recv_total();
    t.rounds += r.rounds;
    t.ot_flights_max += r.ot_flights_max;
    t.ots += r.ots;
  }
  return t;
}

}  // namespace spz

#endif  // SPZ_METRICS_H
