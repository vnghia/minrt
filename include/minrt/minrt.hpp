#pragma once

#include <NvInferRuntime.h>

#include <cstddef>
#include <filesystem>
#include <istream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "NvInfer.h"
#include "minrt/utils.hpp"

namespace minrt {

namespace fs = std::filesystem;

class Logger : public nvinfer1::ILogger {
 public:
  void set_severity(Severity severity);

 private:
  void log(Severity severity, const char* msg) noexcept override;

  Severity severity_ = Severity::kINFO;
};

class Engine {
 public:
  static Engine engine_from_path(const fs::path& path, int32_t profile = 0,
                                 int dla_core = -1);
  static Engine engine_from_stream(std::istream& stream, int32_t profile = 0,
                                   int dla_core = -1);

  void create_input_device_buffer(
      bool use_managed = true,
      const std::unordered_map<std::string, std::shared_ptr<void>>&
          preallocated_buffers = {});
  void create_output_device_buffer(
      bool use_managed = true,
      const std::unordered_map<std::string, std::shared_ptr<void>>&
          preallocated_buffers = {});
  void create_device_buffer(
      bool use_managed = true,
      const std::unordered_map<std::string, std::shared_ptr<void>>&
          preallocated_buffers = {});

  template <typename T>
  void upload(const T& container, std::size_t input_index,
              cudaStream_t stream = 0) {
    upload(container.data(), input_index, stream);
  }

  template <typename T>
  void upload(const T* data, std::size_t input_index, cudaStream_t stream = 0) {
    cuda_upload(input_bindings_[input_index].get(), data,
                input_sizes_[input_index], stream);
  }

  template <typename T>
  void download(T& container, std::size_t output_index,
                cudaStream_t stream = 0) {
    download(container.data(), output_index, stream);
  }

  template <typename T>
  void download(T* data, std::size_t output_index, cudaStream_t stream = 0) {
    cuda_download(data, output_bindings_[output_index].get(),
                  output_sizes_[output_index], stream);
  }

  bool forward(cudaStream_t stream = 0) { return context_->enqueueV3(stream); }

  auto get_input_size(std::size_t input_index) {
    return input_sizes_[input_index];
  }

  auto get_input_binding(std::size_t input_index) {
    return input_bindings_[input_index];
  }

  auto get_output_size(std::size_t output_index) {
    return output_sizes_[output_index];
  }

  auto get_output_binding(std::size_t output_index) {
    return output_bindings_[output_index];
  }

  void set_input_consumed_event(cudaEvent_t input_consumed_event);

 private:
  Engine(Logger& logger, std::unique_ptr<nvinfer1::ICudaEngine>& engine,
         std::unique_ptr<nvinfer1::IExecutionContext>& context,
         int32_t profile);

  void create_device_buffer(
      bool use_managed, const std::vector<std::string>& names,
      std::vector<nvinfer1::Dims>& dims,
      std::vector<nvinfer1::DataType>& dtypes, std::vector<std::size_t>& sizes,
      std::vector<std::shared_ptr<void>>& bindings,
      const std::unordered_map<std::string, std::shared_ptr<void>>&
          preallocated_buffers,
      bool is_input);

  Logger logger_;

  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  std::vector<std::string> input_names_;
  std::vector<nvinfer1::Dims> input_dims_;
  std::vector<nvinfer1::DataType> input_dtypes_;
  std::vector<std::size_t> input_sizes_;
  std::vector<std::shared_ptr<void>> input_bindings_;

  std::vector<std::string> output_names_;
  std::vector<nvinfer1::Dims> outputs_dims_;
  std::vector<nvinfer1::DataType> outputs_dtypes_;
  std::vector<std::size_t> output_sizes_;
  std::vector<std::shared_ptr<void>> output_bindings_;

  int32_t profile_;
};

}  // namespace minrt
