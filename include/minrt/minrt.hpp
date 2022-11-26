#pragma once

#include <NvInferRuntime.h>

#include <filesystem>
#include <istream>
#include <memory>

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

 private:
  Engine(Logger& logger, std::unique_ptr<nvinfer1::ICudaEngine>& engine,
         std::unique_ptr<nvinfer1::IExecutionContext>& context,
         int32_t profile);

  Logger logger_;

  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  int32_t profile_;
};

}  // namespace minrt
