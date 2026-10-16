// metrics.h -- per-phase wall time, process CPU time, traffic per peer and communication steps of one party.
//
// Phases are named "stage/sub-phase" (stage = net | setup | offline | online; "net" = connection setup,
// excluded from every total, like the siblings' Link construction). Each phase records
//   wall    elapsed time at this party (includes waiting for peers)
//   cpu     CLOCK_PROCESS_CPUTIME_ID, summed over all of this party's threads (incl. the sender threads)
//   sent[p] / recv[p]   bytes to / from party p (S0, S1, S2, client), frame headers included
//   steps   communication steps: one Step::run() = one round in which the party sends its messages and
//           receives the messages of that round (a send-only or receive-only round is also one step).
//           All three servers execute the same step schedule, so the online steps of a server are the
//           protocol's round complexity.
// Repeated records of one name accumulate. PhaseRow is plain-old-data so that the servers can send their
// tables to the client as raw bytes.
#ifndef FAL_METRICS_H
#define FAL_METRICS_H

#include <sys/resource.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "net.h"

namespace fal {

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

struct PhaseRow {
  char name[40];
  double wall_s, cpu_s;
  uint64_t sent[NP], recv[NP];
  uint64_t steps;
  uint64_t calls;
};

class Recorder {
 public:
  explicit Recorder(const Net& n) : net_(n) {}

  struct Mark {
    double wall = 0, cpu = 0;
    Counters c;
  };
  Mark now() const { return {wall_seconds(), cpu_seconds(), net_.counters()}; }

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
    for (int p = 0; p < NP; ++p) {
      r->sent[p] += n.c.sent[p] - m.c.sent[p];
      r->recv[p] += n.c.recv[p] - m.c.recv[p];
    }
    r->steps += n.c.steps - m.c.steps;
    r->calls += 1;
    m = n;
  }
  const std::vector<PhaseRow>& rows() const { return rows_; }

 private:
  const Net& net_;
  std::vector<PhaseRow> rows_;
};

// RAII helper: Scope s(rec, "offline/deal"); a null recorder makes it a no-op (unit tests).
class Scope {
 public:
  Scope(Recorder* r, const char* name) : r_(r), name_(name) {
    if (r_) m_ = r_->now();
  }
  Scope(Recorder& r, const char* name) : Scope(&r, name) {}
  ~Scope() {
    if (r_) r_->record(name_, m_);
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

 private:
  Recorder* r_;
  const char* name_;
  Recorder::Mark m_;
};

struct StageTotals {
  double wall_s = 0, cpu_s = 0;
  uint64_t sent[NP] = {0, 0, 0, 0}, recv[NP] = {0, 0, 0, 0};
  uint64_t steps = 0;
  uint64_t sent_total() const { return sent[0] + sent[1] + sent[2] + sent[3]; }
  uint64_t recv_total() const { return recv[0] + recv[1] + recv[2] + recv[3]; }
};
// Sums every row whose name starts with "<stage>/", except the phase `skip` (if given).
inline StageTotals stage_totals(const std::vector<PhaseRow>& rows, const char* stage, const char* skip = nullptr) {
  StageTotals t;
  const size_t n = std::strlen(stage);
  for (const auto& r : rows) {
    if (std::strncmp(r.name, stage, n) != 0 || r.name[n] != '/') continue;
    if (skip && std::strcmp(r.name, skip) == 0) continue;
    t.wall_s += r.wall_s;
    t.cpu_s += r.cpu_s;
    for (int p = 0; p < NP; ++p) { t.sent[p] += r.sent[p]; t.recv[p] += r.recv[p]; }
    t.steps += r.steps;
  }
  return t;
}
inline const PhaseRow* find_row(const std::vector<PhaseRow>& rows, const char* name) {
  for (const auto& r : rows) if (std::strcmp(r.name, name) == 0) return &r;
  return nullptr;
}

}  // namespace fal

#endif  // FAL_METRICS_H
