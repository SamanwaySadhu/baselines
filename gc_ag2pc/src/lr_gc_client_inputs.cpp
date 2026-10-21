// lr_gc_client_inputs -- test helper for scripts/check_input_privacy.sh: prints the
// client's private input bits of the first protocol chunk, one '0'/'1' line per input
// value, in exactly the order lr_gc_ag2pc feeds them to the protocol (iteration 0's
// input batch):
//   the d initial weights, each qbits bits (two's complement, LSB first); then for
//   each datapoint i < batch of iteration 0: its d_data feature columns (qbits bits
//   each, LSB first) and its label bit.
// The bias column is the public constant MAXQ, not an input, so it is not printed.
//
// usage: lr_gc_client_inputs [training flags of lr_gc_ag2pc: --d N --batch N --qbits N
//                             --seed N --zero-init --no-bias ...]
#include "lr_params.h"

#include <cstdio>
#include <exception>
#include <string>

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
  auto quantized = [&](int64_t v) {
    for (bool b : to_bits(v, P.qbits)) std::putchar(b ? '1' : '0');
    std::putchar('\n');
  };
  for (int64_t w : generate_initial_weights(P)) quantized(w);
  const Dataset D = generate_client_data(P);
  for (int i = 0; i < P.batch; ++i) {
    for (int j = 0; j < P.d_data; ++j) quantized(D.X(0, i, j));
    std::printf("%d\n", D.Y(0, i) != 0 ? 1 : 0);
  }
  return 0;
}
