// metrics.h -- per-phase wall time, process CPU time and communication of one party,
// plus peak RSS and byte formatting.
//
//   Metrics M;
//   auto m = M.now(ch);              // ch: the party's Channel (or nullptr: no traffic)
//   ... work ...
//   M.record("encrypt", m, ch);      // accumulate [m, now) into "encrypt", advance m
//   { ScopedPhase p(M, "keyderive"); ... }   // the same as RAII (compute-only here)
//
// Phases accumulate over repeated records (calls counts them) and keep first-seen
// order. CPU time is CLOCK_PROCESS_CPUTIME_ID, i.e. summed over all threads of the
// process (so a phase parallelized over N threads can show cpu_s ~ N * wall_s). Traffic
// is the difference of the Channel's payload counters (see net.h for flights).
#ifndef LRHE_METRICS_H
#define LRHE_METRICS_H

#include "net.h"

#include <sys/resource.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace lrhe {

// Process CPU time (all threads), seconds.
inline double cpu_seconds() {
  timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
// Monotonic wall clock, seconds (arbitrary origin).
inline double wall_seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// Peak resident set size of this process (Linux ru_maxrss: KiB).
inline long peak_rss_kib() {
  rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_maxrss;
}

// 1536 -> "1.50 KiB" (binary units).
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
inline std::string fmt_ms(double s) {
  char b[32];
  std::snprintf(b, sizeof b, "%.3f", 1e3 * s);
  return b;
}

struct PhaseStat {
  double wall_s = 0, cpu_s = 0;
  uint64_t sent = 0, recv = 0;              // payload bytes
  uint64_t flights = 0;                     // flights_out + flights_in opened in the phase
  uint64_t flights_out = 0, flights_in = 0;
  uint64_t calls = 0;                       // records accumulated
  void add(const PhaseStat& o) {
    wall_s += o.wall_s; cpu_s += o.cpu_s; sent += o.sent; recv += o.recv;
    flights += o.flights; flights_out += o.flights_out; flights_in += o.flights_in; calls += o.calls;
  }
};

struct Mark {
  std::chrono::steady_clock::time_point t;
  double cpu = 0;
  NetCounters net;
};

inline Mark mark_now(const Channel* ch = nullptr) {
  Mark m;
  m.t = std::chrono::steady_clock::now();
  m.cpu = cpu_seconds();
  if (ch) m.net = ch->counters();
  return m;
}

inline PhaseStat phase_diff(const Mark& a, const Mark& b) {
  PhaseStat s;
  s.wall_s = std::chrono::duration<double>(b.t - a.t).count();
  s.cpu_s = b.cpu - a.cpu;
  const NetCounters d = b.net - a.net;
  s.sent = d.sent;
  s.recv = d.recv;
  s.flights_out = d.flights_out;
  s.flights_in = d.flights_in;
  s.flights = d.flights();
  s.calls = 1;
  return s;
}

class Metrics {
 public:
  Mark now(const Channel* ch = nullptr) const { return mark_now(ch); }

  // Accumulate [m, now) into phase `name`, then advance m to now. Pass the same
  // channel (or nullptr) that m was taken with, or the traffic numbers are garbage.
  PhaseStat record(const std::string& name, Mark& m, const Channel* ch = nullptr) {
    const Mark n = mark_now(ch);
    const PhaseStat s = phase_diff(m, n);
    add(name, s);
    m = n;
    return s;
  }

  // Accumulate an externally measured stat into phase `name` (thread-safe).
  void add(const std::string& name, const PhaseStat& s) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = tbl_.find(name);
    if (it == tbl_.end()) {
      it = tbl_.emplace(name, PhaseStat{}).first;
      order_.push_back(name);
    }
    it->second.add(s);
  }

  bool has(const std::string& name) const {
    std::lock_guard<std::mutex> g(mu_);
    return tbl_.count(name) != 0;
  }
  // The phase's totals (all zero if never recorded).
  PhaseStat get(const std::string& name) const {
    std::lock_guard<std::mutex> g(mu_);
    auto it = tbl_.find(name);
    return it == tbl_.end() ? PhaseStat{} : it->second;
  }
  // Phases in first-seen order.
  std::vector<std::pair<std::string, PhaseStat>> phases() const {
    std::lock_guard<std::mutex> g(mu_);
    std::vector<std::pair<std::string, PhaseStat>> v;
    for (const auto& k : order_) v.emplace_back(k, tbl_.at(k));
    return v;
  }
  void clear() {
    std::lock_guard<std::mutex> g(mu_);
    tbl_.clear();
    order_.clear();
  }

 private:
  mutable std::mutex mu_;
  std::map<std::string, PhaseStat> tbl_;
  std::vector<std::string> order_;
};

// Records [construction, destruction) into M's phase `name`.
class ScopedPhase {
 public:
  ScopedPhase(Metrics& M, std::string name, const Channel* ch = nullptr)
      : M_(M), name_(std::move(name)), ch_(ch), m_(mark_now(ch)) {}
  ~ScopedPhase() { M_.record(name_, m_, ch_); }
  ScopedPhase(const ScopedPhase&) = delete;
  ScopedPhase& operator=(const ScopedPhase&) = delete;

 private:
  Metrics& M_;
  std::string name_;
  const Channel* ch_;
  Mark m_;
};

}  // namespace lrhe

#endif  // LRHE_METRICS_H
