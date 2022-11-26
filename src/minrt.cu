#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "minrt/minrt.hpp"
#include "minrt/utils.hpp"
#include "spdlog/fmt/bundled/format.h"
#include "spdlog/fmt/bundled/ranges.h"
#include "spdlog/spdlog.h"

template <>
struct fmt::formatter<minrt::fs::path> {
  char presentation = 's';

  constexpr auto parse(format_parse_context &ctx) {
    auto it = ctx.begin(), end = ctx.end();
    if (it != end &&
        (*it == '/' || *it == '.' || *it == 'c' || *it == 'f' || *it == 'n'))
      presentation = *it++;

    if (it != end && *it != '}') throw format_error("invalid format");

    return it;
  }

  template <typename FormatContext>
  auto format(const minrt::fs::path &p, FormatContext &ctx) {
    switch (presentation) {
      case '/':
        return format_to(
            ctx.out(), "{}",
            p.has_root_path() ? p.root_path().string() : p.string());
      case '.':
        return format_to(
            ctx.out(), "{}",
            p.has_relative_path() ? p.relative_path().string() : p.string());
      case 'n':
        return format_to(ctx.out(), "{}", p.stem().string());
      case 'f':
        return format_to(ctx.out(), "{}", p.filename().string());
      case 'c':
        return format_to(ctx.out(), "{}",
                         minrt::fs::weakly_canonical(p).string());
      default:
        return format_to(ctx.out(), "{}", p.string());
    }
  }
};

namespace minrt {

void Logger::set_severity(Severity severity) { severity_ = severity; }

void Logger::log(Severity severity, const char *msg) noexcept {
  if (severity <= severity_) {
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        spdlog::critical("[MINRT] {}", msg);
        break;
      case Severity::kERROR:
        spdlog::error("[MINRT] {}", msg);
        break;
      case Severity::kWARNING:
        spdlog::warn("[MINRT] {}", msg);
        break;
      case Severity::kINFO:
        spdlog::info("[MINRT] {}", msg);
        break;
      case Severity::kVERBOSE:
        spdlog::info("[MINRT] {}", msg);
        break;
    }
  }
}

Engine::Engine(Logger &logger, std::unique_ptr<nvinfer1::ICudaEngine> &engine,
               std::unique_ptr<nvinfer1::IExecutionContext> &context,
               int32_t profile)
    : logger_(std::move(logger)),
      engine_(std::move(engine)),
      context_(std::move(context)),
      profile_(profile) {
  if (!context_->setOptimizationProfileAsync(profile_, 0))
    throw std::invalid_argument("set profile index failed");

  for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
    std::string name = engine_->getIOTensorName(i);
    auto io_mode = engine_->getTensorIOMode(name.c_str());
    if (io_mode == nvinfer1::TensorIOMode::kINPUT) {
      input_names_.push_back(name);
    } else if (io_mode == nvinfer1::TensorIOMode::kOUTPUT) {
      output_names_.push_back(name);
    }
  }
  spdlog::info("tensor input name {}", input_names_);
  spdlog::info("tensor output name {}", output_names_);
}

Engine Engine::engine_from_path(const fs::path &path, int32_t profile,
                                int dla_core) {
  spdlog::info("deserialize engine from {:c}", path);
  std::ifstream file(path, std::ifstream::binary);
  file.exceptions(std::ifstream::badbit);
  return engine_from_stream(file, profile, dla_core);
}

Engine Engine::engine_from_stream(std::istream &stream, int32_t profile,
                                  int dla_core) {
  auto const start_pos = stream.tellg();
  stream.ignore(std::numeric_limits<std::streamsize>::max());

  size_t buf_size = stream.gcount();
  stream.seekg(start_pos);
  std::unique_ptr<char[]> engine_buf(new char[buf_size]);
  stream.read(engine_buf.get(), buf_size);

  Logger logger;

  std::unique_ptr<nvinfer1::IRuntime> runtime{
      nvinfer1::createInferRuntime(logger)};
  if (dla_core >= 0) {
    runtime->setDLACore(dla_core);
  }

  std::unique_ptr<nvinfer1::ICudaEngine> engine(
      runtime->deserializeCudaEngine((void *)engine_buf.get(), buf_size));
  if (!engine) throw std::invalid_argument("engine deserialization failed");
  std::unique_ptr<nvinfer1::IExecutionContext> context(
      engine->createExecutionContext());
  if (!context)
    throw std::invalid_argument("execution context creation failed");

  return Engine(logger, engine, context, profile);
}

void Engine::create_device_buffer(
    const std::unordered_map<std::string, std::shared_ptr<void>>
        &preallocated_buffers) {
  create_input_device_buffer(preallocated_buffers);
  create_output_device_buffer(preallocated_buffers);
}

void Engine::create_input_device_buffer(
    const std::unordered_map<std::string, std::shared_ptr<void>>
        &preallocated_buffers) {
  spdlog::info("create input device buffer");
  create_device_buffer(input_names_, input_dims_, input_dtypes_, input_sizes_,
                       input_bindings_, preallocated_buffers, true);
}

void Engine::create_output_device_buffer(
    const std::unordered_map<std::string, std::shared_ptr<void>>
        &preallocated_buffers) {
  spdlog::info("create output device buffer");
  create_device_buffer(output_names_, outputs_dims_, outputs_dtypes_,
                       output_sizes_, output_bindings_, preallocated_buffers,
                       false);
}

void Engine::create_device_buffer(
    const std::vector<std::string> &names, std::vector<nvinfer1::Dims> &dims,
    std::vector<nvinfer1::DataType> &dtypes, std::vector<std::size_t> &sizes,
    std::vector<std::shared_ptr<void>> &bindings,
    const std::unordered_map<std::string, std::shared_ptr<void>>
        &preallocated_buffers,
    bool is_input) {
  dims.resize(names.size());
  dtypes.resize(names.size());
  sizes.resize(names.size());
  bindings.resize(names.size());

  for (std::size_t i = 0; i < names.size(); ++i) {
    const auto &name = names[i];

    auto dim =
        is_input ? engine_->getProfileShape(name.c_str(), profile_,
                                            nvinfer1::OptProfileSelector::kMAX)
                 : engine_->getTensorShape(name.c_str());
    auto dtype = engine_->getTensorDataType(name.c_str());
    auto size = get_total_size(dim) * get_dtype_size(dtype);

    spdlog::info("tensor name=\"{}\" dims=[{}] dtype={}", name,
                 fmt::join(dim.d, dim.d + dim.nbDims, ", "),
                 dtype_to_string(dtype));

    auto preallocated_buffer = preallocated_buffers.find(name);
    if (preallocated_buffer == preallocated_buffers.end()) {
      bindings[i] = cuda_malloc(size);
      spdlog::info("tensor name=\"{}\" allocated {} byte", name, size);
    } else {
      bindings[i] = preallocated_buffer->second;
      spdlog::info("tensor name=\"{}\" use allocated buffer at {}", name,
                   preallocated_buffer->second.get());
    }
    context_->setTensorAddress(name.c_str(), bindings[i].get());
    spdlog::info("tensor name=\"{}\" set address to {}", name,
                 context_->getTensorAddress(name.c_str()));

    dims[i] = dim;
    dtypes[i] = dtype;
    sizes[i] = size;
  }
}

}  // namespace minrt
