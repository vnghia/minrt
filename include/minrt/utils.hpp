#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

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

#ifndef MINRT_EXECUTION_TIMER
#define MINRT_EXECUTION_TIMER(tag, call_str)                                \
  {                                                                         \
    std::chrono::steady_clock::time_point begin =                           \
        std::chrono::steady_clock::now();                                   \
    call_str;                                                               \
    std::chrono::steady_clock::time_point end =                             \
        std::chrono::steady_clock::now();                                   \
    spdlog::info(                                                           \
        "{} took {}ms", tag,                                                \
        (std::chrono::duration_cast<std::chrono::milliseconds>(end - begin) \
             .count()));                                                    \
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

template <typename T = void>
inline std::shared_ptr<T> cuda_malloc(size_t size) {
  void* device_mem;
  CUDA_EXIT_IF_ERROR(cudaMalloc(&device_mem, size));
  return std::shared_ptr<T>(static_cast<T*>(device_mem),
                            [](void* p) { CUDA_EXIT_IF_ERROR(cudaFree(p)); });
}

template <typename T = void>
inline std::shared_ptr<T> cuda_malloc_managed(size_t size) {
  void* device_mem;
  CUDA_EXIT_IF_ERROR(cudaMallocManaged(&device_mem, size));
  return std::shared_ptr<T>(static_cast<T*>(device_mem),
                            [](void* p) { CUDA_EXIT_IF_ERROR(cudaFree(p)); });
}

template <typename T = void>
inline std::pair<std::shared_ptr<T>, T*> cuda_malloc_mapped(size_t size) {
  void* host_mem;
  CUDA_EXIT_IF_ERROR(cudaHostAlloc(
      &host_mem, size, cudaHostAllocMapped | cudaHostAllocPortable));
  void* device_mem;
  CUDA_EXIT_IF_ERROR(cudaHostGetDevicePointer(&device_mem, host_mem, 0));
  return std::make_pair(
      std::shared_ptr<void>(
          static_cast<T*>(host_mem),
          [](void* p) { CUDA_EXIT_IF_ERROR(cudaFreeHost(p)); }),
      static_cast<T*>(device_mem));
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

inline void cuda_upload(void* device, const void* host, std::size_t size,
                        cudaStream_t stream = 0) {
  CUDA_EXIT_IF_ERROR(cudaMemcpyAsync(
      device, host, size, cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
}

inline void cuda_download(void* host, const void* device, std::size_t size,
                          cudaStream_t stream = 0) {
  CUDA_EXIT_IF_ERROR(cudaMemcpyAsync(
      host, device, size, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));
}

class CudaEvent {
 public:
  CudaEvent() {
    CUDA_EXIT_IF_ERROR(
        cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }
  ~CudaEvent() { CUDA_EXIT_IF_ERROR(cudaEventDestroy(event_)); }

  operator cudaEvent_t() { return event_; }

 private:
  cudaEvent_t event_;
};

}  // namespace minrt
