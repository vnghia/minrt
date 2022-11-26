#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>

#include "minrt/minrt.hpp"
#include "spdlog/fmt/bundled/format.h"
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

}  // namespace minrt
