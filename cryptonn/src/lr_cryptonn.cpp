// PLACEHOLDER -- lr_cryptonn.cpp: the two-party CryptoNN logistic-regression program
// (client = data owner + FE authority, server = model holder). This stub only exists so
// that the CMake skeleton builds; the protocol implementation replaces this file.
#include "lr_params.h"
#include "lr_reference.h"
#include "metrics.h"
#include "net.h"

#include <openssl/opensslv.h>

#include <cstdio>

int main() {
  std::fprintf(stderr, "lr_cryptonn: PLACEHOLDER build (%s); the protocol is not implemented yet\n",
               OPENSSL_VERSION_TEXT);
  return 2;
}
