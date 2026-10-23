// wan_wait.h -- reads the network wait time that wan/libwanshim.so (LD_PRELOAD) accumulates.
// Both functions return 0 when the shim is not preloaded, so a normal run is unaffected.
#ifndef WAN_WAIT_H
#define WAN_WAIT_H

#include <dlfcn.h>

namespace wan {

// Wall time the protocol thread (tid == pid) has spent blocked in socket receives so far, in seconds.
inline double wait_s() {
  static auto f = (double (*)())dlsym(RTLD_DEFAULT, "wanshim_main_wait_s");
  return f ? f() : 0.0;
}
// The same summed over all threads (can exceed wall time when several threads wait at once).
inline double wait_all_s() {
  static auto f = (double (*)())dlsym(RTLD_DEFAULT, "wanshim_all_wait_s");
  return f ? f() : 0.0;
}
inline bool shim_loaded() { return dlsym(RTLD_DEFAULT, "wanshim_main_wait_s") != nullptr; }

}  // namespace wan

#endif  // WAN_WAIT_H
