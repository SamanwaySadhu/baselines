// run_cfg.h -- the configuration of one party of the gc_agmpc baseline (shared by main and the per-nP runners).
#ifndef LR_AGMPC_RUN_CFG_H
#define LR_AGMPC_RUN_CFG_H

#include "lr_params.h"

#include <cstdint>
#include <string>

namespace lrgc {

struct RunCfg {
  Params P;
  int party = 0;          // emp-agmpc party: 1 = server, k + 1 = client k
  int port = 12345;
  int ssp = 40;
  uint64_t seg_ands = 280000;
  int threads = 0;        // thread-pool size (0 = 2 (nP - 1) + 2)
  bool verify = false, quiet = false;
  std::string json, csv;
};

template <int nP>
int run_party(const RunCfg& C);  // runner.h, explicitly instantiated per nP in src/runner_np.cpp

}  // namespace lrgc

#endif  // LR_AGMPC_RUN_CFG_H
