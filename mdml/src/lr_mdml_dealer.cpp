// lr_mdml_dealer -- trusted dealer for the MD-ML logistic-regression baseline
// (MD-ML's "fake offline" phase). Generates the circuit-dependent preprocessing for
// one training run and writes it to
//   <dir>/<job>-party-0.txt   (client)      <dir>/<job>-party-1.txt   (server)
//   <dir>/<job>-verify.txt    TEST ONLY: the pre-truncation masks, read by `server --verify`
// The material is single-use: run the dealer again before every training run.
//
// usage: lr_mdml_dealer [training flags] [--offline-dir DIR] [--job NAME] [--json FILE]

#include "lr_circuit.h"
#include "lr_params.h"
#include "mdml_dealer.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace lrmd;

int main(int argc, char** argv) {
  Params P;
  std::string dir = "results/offline", job = "lr", json_path;
  try {
    ArgReader r{argc, argv, 1};
    for (; r.has(); ++r.i) {
      const std::string a = argv[r.i];
      if (parse_param_flag(r, P)) continue;
      if (a == "--offline-dir") dir = r.next_value(a);
      else if (a == "--job") job = r.next_value(a);
      else if (a == "--json") json_path = r.next_value(a);
      else if (a == "--quiet") continue;  // accepted so scripts can pass the same flags to all programs
      else if (a == "--help" || a == "-h") {
        std::printf("usage: %s [options]\n%s\n  --offline-dir DIR  output directory (default results/offline)\n"
                    "  --job NAME         file name prefix (default lr)\n  --json FILE        write the counts as JSON\n",
                    argv[0], kParamHelp);
        return 0;
      } else throw std::invalid_argument("unknown option " + a);
    }
    P.finalize();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s (see --help)\n", e.what());
    return 1;
  }

  namespace fs = std::filesystem;
  fs::create_directories(dir);
  const std::string base = (fs::absolute(dir) / job).string();  // absolute: overrides FAKE_OFFLINE_DIR
  const double t0 = now_s();
  DealerStats st;
  {
    FakePartyT fp(base);
    for (int i = 0; i < 2; ++i)
      if (!fp.ithPartyFile(i)) { std::fprintf(stderr, "error: cannot write %s-party-%d.txt\n", base.c_str(), i); return 1; }
    DealerCtx dc(fp, P.fingerprint());
    DealerBackend be{dc};
    auto C = build_lr(be, P);
    C.out->runOffline();
    st = dc.st;

    std::ofstream vf(base + "-verify.txt");
    vf << P.fingerprint() << '\n' << dc.trunc_lambdas.size() << '\n';
    for (const auto& v : dc.trunc_lambdas) {
      vf << v.size();
      for (u64 x : v) vf << ' ' << x;
      vf << '\n';
    }
  }  // files are flushed and closed here
  const double secs = now_s() - t0;
  const auto sz0 = fs::file_size(base + "-party-0.txt"), sz1 = fs::file_size(base + "-party-1.txt");

  std::printf("MD-ML LR dealer (fake offline): %s\n", params_string(P).c_str());
  std::printf("  input masks %llu | triple elements %llu | truncation pairs %llu | LTZ: %llu lambda bits, "
              "%llu binary AND triples, %llu daBits | output masks %llu\n",
              (unsigned long long)st.input_masks, (unsigned long long)st.triple_elems,
              (unsigned long long)st.trunc_pairs, (unsigned long long)st.lambda_bits,
              (unsigned long long)st.bin_triples, (unsigned long long)st.dabits, (unsigned long long)st.output_masks);
  std::printf("  wrote %s-party-{0,1}.txt (%.2f / %.2f MiB) in %.2f s\n", base.c_str(), sz0 / 1048576.0,
              sz1 / 1048576.0, secs);

  if (!json_path.empty()) {
    std::ostringstream js;
    js << "{\"d\":" << P.d << ",\"iters\":" << P.iters << ",\"batch\":" << P.batch << ",\"frac_bits\":" << P.frac
       << ",\"input_masks\":" << st.input_masks << ",\"triple_elems\":" << st.triple_elems
       << ",\"trunc_pairs\":" << st.trunc_pairs << ",\"lambda_bits\":" << st.lambda_bits
       << ",\"bin_triples\":" << st.bin_triples << ",\"dabits\":" << st.dabits
       << ",\"output_masks\":" << st.output_masks << ",\"dealer_s\":" << secs << ",\"file_bytes\":[" << sz0
       << "," << sz1 << "]}\n";
    std::ofstream(json_path) << js.str();
  }
  return 0;
}
