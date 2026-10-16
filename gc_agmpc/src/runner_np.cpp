// One explicit instantiation of run_party<NP> per translation unit (CMake compiles this file once per NP), so that
// the heavy emp-agmpc templates of several party counts are never compiled together.
#include "runner.h"

namespace lrgc {
template int run_party<NP>(const RunCfg& C);
}
