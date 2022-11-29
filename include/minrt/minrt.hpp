#pragma once

#include <NvInferRuntime.h>

#include <cstddef>
#include <filesystem>
#include <istream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
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
  enum class malloc_mode {
    // using cudaMalloc - has only device_ptr
    device,
    // using cudaMallocManaged - has both device_ptr and host_ptr
    managed,
    // using cudaHostAlloc with cudaHostAllocMapped | cudaHostAllocPortable -
    // has both device_ptr and host_ptr
    mapped
  };

  using shared_ptr_or_raw_t = std::variant<std::shared_ptr<void>, void*>;

  using malloc_mode_or_ptr_t =
      std::variant<malloc_mode, std::pair<shared_ptr_or_raw_t,
                                          std::optional<shared_ptr_or_raw_t>>>;

  static Engine engine_from_path(const fs::path& path, int32_t profile = 0,
                                 int dla_core = -1);
  static Engine engine_from_stream(std::istream& stream, int32_t profile = 0,
                                   int dla_core = -1);

  void create_input_device_buffer(
      const std::unordered_map<std::string, malloc_mode_or_ptr_t>&
          buffer_modes = {});
  void create_output_device_buffer(
      const std::unordered_map<std::string, malloc_mode_or_ptr_t>&
          buffer_modes = {});
  void create_device_buffer(
      const std::unordered_map<std::string, malloc_mode_or_ptr_t>&
          buffer_modes = {});

  template <typename T>
  void upload(const T& container, std::size_t input_index,
              cudaStream_t stream = 0) {
    upload(container.data(), input_index, stream);
  }

  template <typename T>
  void upload(const T* data, std::size_t input_index, cudaStream_t stream = 0) {
    cuda_upload(input_device_ptrs_[input_index], data,
                input_byte_sizes_[input_index], stream);
  }

  template <typename T>
  void download(T& container, std::size_t output_index,
                cudaStream_t stream = 0) {
    download(container.data(), output_index, stream);
  }

  template <typename T>
  void download(T* data, std::size_t output_index, cudaStream_t stream = 0) {
    cuda_download(data, output_device_ptrs_[output_index],
                  output_byte_sizes_[output_index], stream);
  }

  bool forward(cudaStream_t stream = 0) { return context_->enqueueV3(stream); }

  const auto& get_input_name(std::size_t input_index) {
    return input_names_[input_index];
  }

  auto& get_input_dim(std::size_t input_index) {
    return input_dims_[input_index];
  }

  auto get_input_size(std::size_t input_index) {
    return input_sizes_[input_index];
  }

  auto get_input_byte_size(std::size_t input_index) {
    return input_byte_sizes_[input_index];
  }

  auto get_input_device_ptr(std::size_t input_index) {
    return input_device_ptrs_[input_index];
  }

  auto get_input_host_ptr(std::size_t input_index) {
    return input_host_ptrs_[input_index];
  }

  auto get_input_device_owned_ptr(std::size_t input_index) {
    return input_device_owned_ptrs_[input_index];
  }

  auto get_input_host_owned_ptr(std::size_t input_index) {
    return input_host_owned_ptrs_[input_index];
  }

  const auto& get_output_name(std::size_t output_index) {
    return output_names_[output_index];
  }

  auto& get_output_dim(std::size_t output_index) {
    return outputs_dims_[output_index];
  }

  auto get_output_size(std::size_t output_index) {
    return output_sizes_[output_index];
  }

  auto get_output_byte_size(std::size_t output_index) {
    return output_byte_sizes_[output_index];
  }

  auto get_output_device_ptr(std::size_t output_index) {
    return output_device_ptrs_[output_index];
  }

  auto get_output_host_ptr(std::size_t output_index) {
    return output_host_ptrs_[output_index];
  }

  auto get_output_device_owned_ptr(std::size_t output_index) {
    return output_device_owned_ptrs_[output_index];
  }

  auto get_output_host_owned_ptr(std::size_t output_index) {
    return output_host_owned_ptrs_[output_index];
  }

  void set_input_consumed_event(cudaEvent_t input_consumed_event);

 private:
  Engine(Logger& logger, std::unique_ptr<nvinfer1::ICudaEngine>& engine,
         std::unique_ptr<nvinfer1::IExecutionContext>& context,
         int32_t profile);

  void load_io_tensor_info(const std::vector<std::string>& names,
                           std::vector<nvinfer1::Dims>& dims,
                           std::vector<nvinfer1::DataType>& dtypes,
                           std::vector<std::size_t>& sizes,
                           std::vector<std::size_t>& byte_sizes, bool is_input);

  void create_device_buffer(
      const std::vector<std::string>& names,
      std::vector<std::size_t>& byte_sizes, std::vector<void*>& device_ptrs,
      std::vector<void*>& host_ptrs,
      std::vector<std::shared_ptr<void>>& device_owned_ptrs,
      std::vector<std::shared_ptr<void>>& host_owned_ptrs,
      const std::unordered_map<std::string, malloc_mode_or_ptr_t>&
          buffer_modes);

  Logger logger_;

  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  std::vector<std::string> input_names_;
  std::vector<nvinfer1::Dims> input_dims_;
  std::vector<nvinfer1::DataType> input_dtypes_;
  std::vector<std::size_t> input_sizes_;
  std::vector<std::size_t> input_byte_sizes_;
  std::vector<void*> input_device_ptrs_;
  std::vector<void*> input_host_ptrs_;
  std::vector<std::shared_ptr<void>> input_device_owned_ptrs_;
  std::vector<std::shared_ptr<void>> input_host_owned_ptrs_;

  std::vector<std::string> output_names_;
  std::vector<nvinfer1::Dims> outputs_dims_;
  std::vector<nvinfer1::DataType> outputs_dtypes_;
  std::vector<std::size_t> output_sizes_;
  std::vector<std::size_t> output_byte_sizes_;
  std::vector<void*> output_device_ptrs_;
  std::vector<void*> output_host_ptrs_;
  std::vector<std::shared_ptr<void>> output_device_owned_ptrs_;
  std::vector<std::shared_ptr<void>> output_host_owned_ptrs_;

  int32_t profile_;
};

}  // namespace minrt
