// lr_gc_client_inputs -- test helper for scripts/check_input_privacy.sh: prints the
// client's private input bits of the first protocol chunk (initial weights, then
// batch 0: per datapoint the d-1 features and the label), as one '0'/'1' line per
// input value, in the order lr_gc_ag2pc feeds them to the protocol.
//
// usage: lr_gc_client_inputs --d N --iters N --seed N
#include "lr_params.h"

#include <cstdio>
#include <exception>

using namespace lrgc;

int main(int argc, char** argv) {
  Params P;
  try {
    ArgReader r{argc, argv, 1};
    for (; r.has(); ++r.i)
      if (!parse_param_flag(r, P)) throw std::invalid_argument(std::string("unknown option ") + argv[r.i]);
    P.finalize();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
  auto bits = [](uint32_t u) { for (int i = 0; i < 32; ++i) std::putchar('0' + ((u >> i) & 1)); std::putchar('\n'); };
  for (float w : generate_initial_weights(P)) bits(f2u(w));
  const Dataset D = generate_client_data(P);
  for (int i = 0; i < kBatch; ++i) {
    for (int j = 0; j < P.d_data(); ++j) bits(f2u(D.X(0, i, j)));
    std::printf("%d\n", D.Y(0, i));
  }
  return 0;
}
