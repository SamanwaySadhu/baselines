// libwanshim.so -- LD_PRELOAD timer for network wait time.
//
// Interposes the blocking receive calls (recv, recvfrom, recvmsg, read, readv, fread) and, when the file
// descriptor is a socket, adds the wall time spent inside the call to a counter. Two counters:
//   main  : the thread with tid == pid (the protocol's own thread; for multi-threaded phases this is worker 0,
//           which runs the same schedule as the other workers) -- this is the "wait time" a party reports;
//   all   : the sum over all threads.
// A baseline reads them through wan/wan_wait.h (dlsym); without the preload they read as 0.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

static _Atomic uint64_t g_main_ns, g_all_ns;
static __thread int t_is_main = -1;

static inline uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
static inline int is_sock(int fd) {
  struct stat st;
  return fd >= 0 && fstat(fd, &st) == 0 && S_ISSOCK(st.st_mode);
}
static inline void add(uint64_t t0) {
  const uint64_t d = now_ns() - t0;
  if (t_is_main < 0) t_is_main = ((pid_t)syscall(SYS_gettid) == getpid());
  if (t_is_main) atomic_fetch_add(&g_main_ns, d);
  atomic_fetch_add(&g_all_ns, d);
}

double wanshim_main_wait_s(void) { return (double)atomic_load(&g_main_ns) * 1e-9; }
double wanshim_all_wait_s(void) { return (double)atomic_load(&g_all_ns) * 1e-9; }

#define REAL(name) static __typeof__(&name) real = 0; if (!real) real = (__typeof__(&name))dlsym(RTLD_NEXT, #name)

ssize_t recv(int fd, void* b, size_t n, int fl) {
  REAL(recv);
  const uint64_t t0 = now_ns();
  const ssize_t r = real(fd, b, n, fl);
  if (!(fl & MSG_DONTWAIT)) add(t0);
  return r;
}
ssize_t recvfrom(int fd, void* b, size_t n, int fl, struct sockaddr* a, socklen_t* l) {
  REAL(recvfrom);
  const uint64_t t0 = now_ns();
  const ssize_t r = real(fd, b, n, fl, a, l);
  if (!(fl & MSG_DONTWAIT)) add(t0);
  return r;
}
ssize_t recvmsg(int fd, struct msghdr* m, int fl) {
  REAL(recvmsg);
  const uint64_t t0 = now_ns();
  const ssize_t r = real(fd, m, fl);
  if (!(fl & MSG_DONTWAIT)) add(t0);
  return r;
}
ssize_t read(int fd, void* b, size_t n) {
  REAL(read);
  if (!is_sock(fd)) return real(fd, b, n);
  const uint64_t t0 = now_ns();
  const ssize_t r = real(fd, b, n);
  add(t0);
  return r;
}
ssize_t readv(int fd, const struct iovec* v, int c) {
  REAL(readv);
  if (!is_sock(fd)) return real(fd, v, c);
  const uint64_t t0 = now_ns();
  const ssize_t r = real(fd, v, c);
  add(t0);
  return r;
}
size_t fread(void* b, size_t s, size_t n, FILE* f) {
  REAL(fread);
  if (!is_sock(fileno(f))) return real(b, s, n, f);
  const uint64_t t0 = now_ns();
  const size_t r = real(b, s, n, f);
  add(t0);
  return r;
}
