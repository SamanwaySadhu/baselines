// PLACEHOLDER -- lr_cryptonn_selftest.cpp: offline tests of the CryptoNN baseline.
// This stub only exists so that the CMake skeleton builds; the real tests replace it.
#include "lr_params.h"
#include "lr_reference.h"
#include "metrics.h"
#include "net.h"

#include <openssl/crypto.h>

#include <cstdio>

int main() {
  std::printf("lr_cryptonn_selftest: PLACEHOLDER (linked against %s); no tests yet\n",
              OpenSSL_version(OPENSSL_VERSION));
  return 0;
}
