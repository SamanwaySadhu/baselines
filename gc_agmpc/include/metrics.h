// metrics.h -- per-phase wall / cpu time and bytes per peer, recorded at every party (gc_agmpc baseline).
// Phases are contiguous and named "stage/sub" (stage = net, local, setup, offline, online); repeated names accumulate
// (the protocol interleaves the phases of many segments). Rows are plain-old-data so that the clients can send their
// tables to the server, which prints the report.
#ifndef LR_AGMPC_METRICS_H
#define LR_AGMPC_METRICS_H

#include <sys/resource.h>
#include <time.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace lrgc {

constexpr int kMaxParties = 10;  // emp-agmpc parties 1..9 (index 0 unused)

inline double cpu_seconds() {  // the whole process, all threads
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
inline std::string human_bytes(double b) {
  char s[32];
  if (b < 1024) std::snprintf(s, sizeof s, "%.0f B", b);
  else if (b < 1024.0 * 1024) std::snprintf(s, sizeof s, "%.2f KiB", b / 1024);
  else if (b < 1024.0 * 1024 * 1024) std::snprintf(s, sizeof s, "%.2f MiB", b / 1048576);
  else std::snprintf(s, sizeof s, "%.2f GiB", b / 1073741824);
  return s;
}

struct PhaseRow {
  char name[32];
  double wall_s, cpu_s;
  uint64_t sent[kMaxParties], recv[kMaxParties];  // bytes sent to / received from each party
  uint64_t calls;
  uint64_t sent_total() const { uint64_t s = 0; for (uint64_t v : sent) s += v; return s; }
  uint64_t recv_total() const { uint64_t s = 0; for (uint64_t v : recv) s += v; return s; }
};

class Recorder {
 public:
  // snap(sent, recv): cumulative bytes per peer right now
  explicit Recorder(std::function<void(uint64_t*, uint64_t*)> snap) : snap_(std::move(snap)) {}
  // Closes the current phase (if any) and starts `name`.
  void enter(const char* name) {
    close();
    cur_ = name;
    mark();
  }
  void close() {
    if (cur_.empty()) return;
    uint64_t s[kMaxParties] = {0}, r[kMaxParties] = {0};
    if (snap_) snap_(s, r);
    PhaseRow& R = row(cur_.c_str());
    R.wall_s += wall_seconds() - w0_;
    R.cpu_s += cpu_seconds() - c0_;
    for (int p = 0; p < kMaxParties; ++p) { R.sent[p] += s[p] - s0_[p]; R.recv[p] += r[p] - r0_[p]; }
    R.calls++;
    cur_.clear();
  }
  // Snapshot source changes (e.g. the network objects are created): restart the mark of the current phase.
  void set_snap(std::function<void(uint64_t*, uint64_t*)> snap) { close_keep_name(); snap_ = std::move(snap); mark(); }
  const std::vector<PhaseRow>& rows() const { return rows_; }

 private:
  void close_keep_name() { const std::string n = cur_; close(); cur_ = n; }
  void mark() {
    w0_ = wall_seconds();
    c0_ = cpu_seconds();
    std::memset(s0_, 0, sizeof s0_);
    std::memset(r0_, 0, sizeof r0_);
    if (snap_) snap_(s0_, r0_);
  }
  PhaseRow& row(const char* name) {
    for (auto& r : rows_)
      if (std::strcmp(r.name, name) == 0) return r;
    PhaseRow r;
    std::memset(&r, 0, sizeof r);
    std::snprintf(r.name, sizeof r.name, "%s", name);
    rows_.push_back(r);
    return rows_.back();
  }
  std::function<void(uint64_t*, uint64_t*)> snap_;
  std::vector<PhaseRow> rows_;
  std::string cur_;
  double w0_ = 0, c0_ = 0;
  uint64_t s0_[kMaxParties], r0_[kMaxParties];
};

struct StageTotals {
  double wall_s = 0, cpu_s = 0;
  uint64_t sent = 0, recv = 0;
};
inline StageTotals stage_totals(const std::vector<PhaseRow>& rows, const char* stage) {
  StageTotals t;
  const size_t n = std::strlen(stage);
  for (const auto& r : rows)
    if (std::strncmp(r.name, stage, n) == 0 && r.name[n] == '/') {
      t.wall_s += r.wall_s;
      t.cpu_s += r.cpu_s;
      t.sent += r.sent_total();
      t.recv += r.recv_total();
    }
  return t;
}

}  // namespace lrgc

#endif  // LR_AGMPC_METRICS_H
