#include <filesystem>
#include <iostream>

#include "cxxopts.hpp"
#include "minrt/minrt.hpp"

using namespace minrt;

int main(int argc, char* argv[]) {
  cxxopts::Options options("MinRT", "Minimal TensorRT Engine");
  auto options_adder = options.add_options();
  options_adder("help", "Print help");
  options_adder("e,engine", "Engine path", cxxopts::value<fs::path>());
  auto args = options.parse(argc, argv);

  if (args.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  auto engine = Engine::engine_from_path(args["engine"].as<fs::path>());
  engine.create_device_buffer();

  return 0;
}
