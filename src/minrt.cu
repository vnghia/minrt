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
  load_io_tensor_info(input_names_, input_dims_, input_dtypes_, input_sizes_,
                      input_byte_sizes_, true);
  spdlog::info("tensor output name {}", output_names_);
  load_io_tensor_info(output_names_, outputs_dims_, outputs_dtypes_,
                      output_sizes_, output_byte_sizes_, false);
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

void Engine::load_io_tensor_info(const std::vector<std::string> &names,
                                 std::vector<nvinfer1::Dims> &dims,
                                 std::vector<nvinfer1::DataType> &dtypes,
                                 std::vector<std::size_t> &sizes,
                                 std::vector<std::size_t> &byte_sizes,
                                 bool is_input) {
  dims.resize(names.size());
  dtypes.resize(names.size());
  sizes.resize(names.size());
  byte_sizes.resize(names.size());

  for (std::size_t i = 0; i < names.size(); ++i) {
    const auto &name = names[i];

    auto dim =
        is_input ? engine_->getProfileShape(name.c_str(), profile_,
                                            nvinfer1::OptProfileSelector::kMAX)
                 : engine_->getTensorShape(name.c_str());
    auto dtype = engine_->getTensorDataType(name.c_str());
    auto size = get_total_size(dim);
    auto byte_size = size * get_dtype_size(dtype);

    spdlog::info("tensor name=\"{}\" dims=[{}] dtype={}", name,
                 fmt::join(dim.d, dim.d + dim.nbDims, ", "),
                 dtype_to_string(dtype));

    dims[i] = dim;
    dtypes[i] = dtype;
    sizes[i] = size;
    byte_sizes[i] = byte_size;
  }
}

void Engine::create_device_buffer(
    const std::unordered_map<std::string, malloc_mode_or_ptr_t> &buffer_modes) {
  create_input_device_buffer(buffer_modes);
  create_output_device_buffer(buffer_modes);
}

void Engine::create_input_device_buffer(
    const std::unordered_map<std::string, malloc_mode_or_ptr_t> &buffer_modes) {
  spdlog::info("create input device buffer");
  create_device_buffer(input_names_, input_byte_sizes_, input_device_ptrs_,
                       input_host_ptrs_, input_device_owned_ptrs_,
                       input_host_owned_ptrs_, buffer_modes);
}

void Engine::create_output_device_buffer(
    const std::unordered_map<std::string, malloc_mode_or_ptr_t> &buffer_modes) {
  spdlog::info("create output device buffer");
  create_device_buffer(output_names_, output_byte_sizes_, output_device_ptrs_,
                       output_host_ptrs_, output_device_owned_ptrs_,
                       output_host_owned_ptrs_, buffer_modes);
}

void Engine::create_device_buffer(
    const std::vector<std::string> &names, std::vector<std::size_t> &byte_sizes,
    std::vector<void *> &device_ptrs, std::vector<void *> &host_ptrs,
    std::vector<std::shared_ptr<void>> &device_owned_ptrs,
    std::vector<std::shared_ptr<void>> &host_owned_ptrs,
    const std::unordered_map<std::string, malloc_mode_or_ptr_t> &buffer_modes) {
  device_ptrs.resize(names.size());
  host_ptrs.resize(names.size());
  device_owned_ptrs.resize(names.size());
  host_owned_ptrs.resize(names.size());

  const auto assign_ptr = [](std::size_t i, std::vector<void *> &ptrs,
                             std::vector<std::shared_ptr<void>> &owned_ptrs,
                             const shared_ptr_or_raw_t ptr) {
    if (std::holds_alternative<std::shared_ptr<void>>(ptr)) {
      owned_ptrs[i] = std::get<0>(ptr);
      ptrs[i] = owned_ptrs[i].get();
    } else {
      ptrs[i] = std::get<1>(ptr);
    }
  };

  for (std::size_t i = 0; i < names.size(); ++i) {
    const auto &name = names[i];
    auto byte_size = byte_sizes[i];

    auto buffer_mode_or_ptr_it = buffer_modes.find(name);
    auto buffer_mode_or_ptr = (buffer_mode_or_ptr_it == buffer_modes.end())
                                  ? malloc_mode::device
                                  : buffer_mode_or_ptr_it->second;
    if (std::holds_alternative<malloc_mode>(buffer_mode_or_ptr)) {
      auto buffer_mode = std::get<0>(buffer_mode_or_ptr);
      switch (buffer_mode) {
        case malloc_mode::device:
          assign_ptr(i, device_ptrs, device_owned_ptrs, cuda_malloc(byte_size));
          spdlog::info(
              "tensor name=\"{}\" allocated {} byte for device pointer", name,
              byte_size);
          break;
        case malloc_mode::managed:
          assign_ptr(i, device_ptrs, device_owned_ptrs,
                     cuda_malloc_managed(byte_size));
          assign_ptr(i, host_ptrs, host_owned_ptrs, device_ptrs[i]);
          spdlog::info(
              "tensor name=\"{}\" allocated {} byte for device and host "
              "pointer",
              name, byte_size);
          break;
        case malloc_mode::mapped: {
          const auto [host_ptr, device_ptr] = cuda_malloc_mapped(byte_size);
          assign_ptr(i, device_ptrs, device_owned_ptrs, device_ptr);
          assign_ptr(i, host_ptrs, host_owned_ptrs, host_ptr);
        } break;
      }
    } else {
      const auto [device_ptr, host_ptr] = std::get<1>(buffer_mode_or_ptr);
      assign_ptr(i, device_ptrs, device_owned_ptrs, device_ptr);
      spdlog::info(
          "tensor name=\"{}\" use allocated buffer for device pointer at {}",
          name, device_ptrs[i]);
      if (host_ptr) {
        assign_ptr(i, host_ptrs, host_owned_ptrs, host_ptr.value());
        spdlog::info(
            "tensor name=\"{}\" use allocated buffer for host pointer at {}",
            name, host_ptrs[i]);
      }
    }

    context_->setTensorAddress(name.c_str(), device_ptrs[i]);
    spdlog::info("tensor name=\"{}\" set address to {}", name,
                 context_->getTensorAddress(name.c_str()));
  }
}

void Engine::set_input_consumed_event(cudaEvent_t input_consumed_event) {
  context_->setInputConsumedEvent(input_consumed_event);
}

}  // namespace minrt
