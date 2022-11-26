#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "minrt/minrt.hpp"
#include "minrt/utils.hpp"
#include "mnist/mnist_reader_less.hpp"
#include "mnist/mnist_utils.hpp"
#include "spdlog/fmt/bundled/ranges.h"
#include "spdlog/spdlog.h"

using namespace minrt;

static constexpr int num_classes = 10;

int main(int argc, char* argv[]) {
  cxxopts::Options options("mnisrt", "MNIST TensorRT Engine");
  auto options_adder = options.add_options();
  options_adder("help", "Print help");
  options_adder("e,engine", "Engine path", cxxopts::value<fs::path>());
  options_adder("d,data_dir", "Data directory path",
                cxxopts::value<fs::path>()->default_value(MNIST_DATA_LOCATION));
  auto args = options.parse(argc, argv);

  if (args.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  spdlog::info("MNIST data path: {}", args["data_dir"].as<fs::path>().string());

  auto engine = Engine::engine_from_path(args["engine"].as<fs::path>());
  engine.create_device_buffer();

  auto images = ([&]() {
    auto uimages = mnist::read_mnist_image_file(
        args["data_dir"].as<fs::path>() / "train-images-idx3-ubyte");
    std::vector<std::vector<float>> images(
        uimages.size(), std::vector<float>(uimages[0].size()));
    for (std::size_t i = 0; i < uimages.size(); ++i) {
      for (std::size_t j = 0; j < uimages[i].size(); ++j) {
        images[i][j] = uimages[i][j] / 255.;
      }
    }
    return images;
  })();
  const auto labels = mnist::read_mnist_label_file(
      args["data_dir"].as<fs::path>() / "train-labels-idx1-ubyte");

  std::vector<float> result(num_classes);
  std::size_t correct_count = 0;

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();

  for (std::size_t i = 0; i < images.size(); ++i) {
    engine.upload(images[i], 0);
    engine.forward();
    engine.download(result, 0);

    auto digit = std::distance(result.begin(),
                               std::max_element(result.begin(), result.end()));
    if (digit == labels[i]) ++correct_count;
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  spdlog::info("MNIST accuracy: {}", correct_count * 1.f / images.size());
  spdlog::info(
      "MNIST running time: {}s",
      (std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
           .count()) /
          1000000.0);

  return 0;
}
