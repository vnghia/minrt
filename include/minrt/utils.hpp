#pragma once

#include <cstddef>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

#include "NvInfer.h"
#include "cuda_runtime.h"
#include "spdlog/spdlog.h"

namespace minrt {

#ifndef CUDA_EXIT_IF_ERROR
#define CUDA_EXIT_IF_ERROR(call_str)                                      \
  {                                                                       \
    cudaError_t error_code = call_str;                                    \
    if (error_code != cudaSuccess) {                                      \
      spdlog::critical("[MINRT] error {} at {}:{}", error_code, __FILE__, \
                       __LINE__);                                         \
      exit(error_code);                                                   \
    }                                                                     \
  }
#endif

inline std::size_t get_dtype_size(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT8:
      return 1;
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kBOOL:
      return 1;
    case nvinfer1::DataType::kUINT8:
      return 1;
    default:
      throw std::runtime_error("invalid dtype");
  }
}

inline std::string dtype_to_string(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT:
      return "float32";
    case nvinfer1::DataType::kHALF:
      return "float16";
    case nvinfer1::DataType::kINT8:
      return "int8";
    case nvinfer1::DataType::kINT32:
      return "int32";
    case nvinfer1::DataType::kBOOL:
      return "bool";
    case nvinfer1::DataType::kUINT8:
      return "uint8";
    default:
      throw std::runtime_error("invalid dtype");
  }
}

inline std::size_t get_total_size(const nvinfer1::Dims& d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1,
                         std::multiplies<std::size_t>());
}

inline std::shared_ptr<void> cuda_malloc(size_t size) {
  void* device_mem;
  CUDA_EXIT_IF_ERROR(cudaMalloc(&device_mem, size));
  return std::shared_ptr<void>(
      device_mem, [](void* p) { CUDA_EXIT_IF_ERROR(cudaFree(p)); });
}

class CudaStream {
 public:
  CudaStream(bool non_blocking = true) {
    if (non_blocking) {
      CUDA_EXIT_IF_ERROR(
          cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    } else {
      CUDA_EXIT_IF_ERROR(cudaStreamCreate(&stream_));
    }
  }
  ~CudaStream() { CUDA_EXIT_IF_ERROR(cudaStreamDestroy(stream_)); }

  operator cudaStream_t() { return stream_; }

 private:
  cudaStream_t stream_;
};

}  // namespace minrt
