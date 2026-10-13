// ag2pc_metrics.h -- per-phase wall time, CPU time and communication of
// emp-ag2pc's protocol steps, recorded at BOTH parties.
//
// emp-ag2pc marks its protocol steps with AG2PC_PHASE_BEGIN() / AG2PC_PHASE(name)
// (and AG2PC_TP_BEGIN() / AG2PC_TP(name) inside triple generation), defined in
// emp-ag2pc/backend/profiling.h: no-ops in a normal build, printed at party 1 only
// under -DAG2PC_PROFILE. This header pre-empts that file (it defines its include
// guard) and defines the four macros to record into lrgc::Metrics instead, so it
// MUST be included before any emp-ag2pc header, with AG2PC_PROFILE undefined.
// lr_gc_ag2pc checks at run time that the hooks fired.
#ifndef LRGC_AG2PC_METRICS_H
#define LRGC_AG2PC_METRICS_H

#ifdef AG2PC_PROFILING_H__
#error "include ag2pc_metrics.h before any emp-ag2pc header"
#endif
#ifdef AG2PC_PROFILE
#error "ag2pc_metrics.h replaces AG2PC_PROFILE; do not define both"
#endif
#define AG2PC_PROFILING_H__  // emp-ag2pc/backend/profiling.h is now a no-op

#include <sys/resource.h>

#include <chrono>
#include <cstdint>
#include <ctime>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace lrgc {

struct PhaseStat {
  double wall_s = 0, cpu_s = 0;
  uint64_t sent = 0, recv = 0, calls = 0;
  void add(const PhaseStat& o) {
    wall_s += o.wall_s; cpu_s += o.cpu_s; sent += o.sent; recv += o.recv; calls += o.calls;
  }
};

class Metrics {
 public:
  struct Mark {
    std::chrono::steady_clock::time_point t;
    double cpu = 0;
    uint64_t sent = 0, recv = 0;
  };

  static Metrics& get() {
    static Metrics m;
    return m;
  }

  // Process CPU time (all threads) and peak resident set size (Linux: KiB).
  static double cpu_seconds() {
    timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
  }
  static long peak_rss_kib() {
    rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;
  }

  // Byte counters (emp::IOChannel::send_counter / recv_counter) of the session's
  // channels. Before attach(), marks carry zero bytes.
  void attach(std::vector<const uint64_t*> sent, std::vector<const uint64_t*> recv) {
    std::lock_guard<std::mutex> g(mu_);
    sent_ = std::move(sent);
    recv_ = std::move(recv);
  }

  Mark now() const {
    Mark m;
    m.t = std::chrono::steady_clock::now();
    m.cpu = cpu_seconds();
    for (const uint64_t* p : sent_) m.sent += *p;
    for (const uint64_t* p : recv_) m.recv += *p;
    return m;
  }

  static PhaseStat diff(const Mark& a, const Mark& b) {
    PhaseStat s;
    s.wall_s = std::chrono::duration<double>(b.t - a.t).count();
    s.cpu_s = b.cpu - a.cpu;
    s.sent = b.sent - a.sent;
    s.recv = b.recv - a.recv;
    s.calls = 1;
    return s;
  }

  // Accumulate [m, now) into phase `name`, then advance m to now (AG2PC_PHASE).
  void record(const char* name, Mark& m) { record_into(phases_, order_, name, m); }
  // Same for the sub-steps of authenticated-triple generation (AG2PC_TP).
  void record_sub(const char* name, Mark& m) { record_into(sub_, sub_order_, name, m); }

  // Forget all recorded phases (keeps the attached counters).
  void reset() {
    std::lock_guard<std::mutex> g(mu_);
    phases_.clear(); sub_.clear(); order_.clear(); sub_order_.clear();
  }

  // Phases in first-seen order.
  std::vector<std::pair<std::string, PhaseStat>> phases() const { return ordered(phases_, order_); }
  std::vector<std::pair<std::string, PhaseStat>> subphases() const { return ordered(sub_, sub_order_); }

 private:
  Metrics() = default;
  mutable std::mutex mu_;
  std::vector<const uint64_t*> sent_, recv_;
  std::map<std::string, PhaseStat> phases_, sub_;
  std::vector<std::string> order_, sub_order_;

  void record_into(std::map<std::string, PhaseStat>& tbl, std::vector<std::string>& order,
                   const char* name, Mark& m) {
    const Mark n = now();
    std::lock_guard<std::mutex> g(mu_);
    auto it = tbl.find(name);
    if (it == tbl.end()) {
      it = tbl.emplace(name, PhaseStat{}).first;
      order.push_back(name);
    }
    it->second.add(diff(m, n));
    m = n;
  }
  std::vector<std::pair<std::string, PhaseStat>> ordered(const std::map<std::string, PhaseStat>& tbl,
                                                         const std::vector<std::string>& order) const {
    std::lock_guard<std::mutex> g(mu_);
    std::vector<std::pair<std::string, PhaseStat>> v;
    for (const auto& k : order) v.emplace_back(k, tbl.at(k));
    return v;
  }
};

}  // namespace lrgc

#define AG2PC_PHASE_BEGIN() ::lrgc::Metrics::Mark _lrgc_phase_mark = ::lrgc::Metrics::get().now()
#define AG2PC_PHASE(name) ::lrgc::Metrics::get().record((name), _lrgc_phase_mark)
#define AG2PC_TP_BEGIN() ::lrgc::Metrics::Mark _lrgc_tp_mark = ::lrgc::Metrics::get().now()
#define AG2PC_TP(name) ::lrgc::Metrics::get().record_sub((name), _lrgc_tp_mark)

#endif  // LRGC_AG2PC_METRICS_H
