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
      const std::unordered_map<std::string, std::shared_ptr<void>>&
          preallocated_buffers = {});
  void create_output_device_buffer(
      const std::unordered_map<std::string, std::shared_ptr<void>>&
          preallocated_buffers = {});
  void create_device_buffer(
      const std::unordered_map<std::string, std::shared_ptr<void>>&
          preallocated_buffers = {});

 private:
  Engine(Logger& logger, std::unique_ptr<nvinfer1::ICudaEngine>& engine,
         std::unique_ptr<nvinfer1::IExecutionContext>& context,
         int32_t profile);

  void create_device_buffer(
      const std::vector<std::string>& names, std::vector<nvinfer1::Dims>& dims,
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
