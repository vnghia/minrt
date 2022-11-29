#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <tuple>
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
  engine.create_device_buffer(
      {{engine.get_input_name(0), Engine::malloc_mode::managed},
       {engine.get_output_name(0), Engine::malloc_mode::managed}});

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

  auto image = engine.get_input_host_ptr(0);
  auto result = static_cast<float*>(engine.get_output_host_ptr(0));

  CudaStream copy_image_stream;
  CudaStream forward_stream;
  CudaEvent input_consumed_event;
  CudaEvent finish_forward_event;
  engine.set_input_consumed_event(input_consumed_event);
  auto input_size = engine.get_input_byte_size(0);

  using forward_stream_data_t =
      std::tuple<float*, const std::vector<unsigned char>*, unsigned long,
                 unsigned long*>;

  auto forward_stream_fn = [](void* raw_data) {
    auto* pdata = static_cast<forward_stream_data_t*>(raw_data);
    auto& data = *pdata;
    auto result = std::get<0>(data);
    auto digit =
        std::distance(result, std::max_element(result, result + num_classes));
    if (digit == (*std::get<1>(data))[std::get<2>(data)])
      ++(*std::get<3>(data));
  };

  std::size_t correct_count = 0;

  std::vector<forward_stream_data_t> forward_stream_data(images.size());
  for (std::size_t i = 0; i < images.size(); ++i) {
    std::get<0>(forward_stream_data[i]) = result;
    std::get<1>(forward_stream_data[i]) = &labels;
    std::get<2>(forward_stream_data[i]) = i;
    std::get<3>(forward_stream_data[i]) = &correct_count;
  }

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();

  for (std::size_t i = 0; i < images.size(); ++i) {
    cudaMemcpyAsync(image, images[i].data(), input_size,
                    cudaMemcpyKind::cudaMemcpyHostToHost, copy_image_stream);
    engine.forward(forward_stream);
    CUDA_EXIT_IF_ERROR(cudaLaunchHostFunc(forward_stream, forward_stream_fn,
                                          &forward_stream_data[i]));
    CUDA_EXIT_IF_ERROR(
        cudaStreamWaitEvent(copy_image_stream, input_consumed_event));
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
