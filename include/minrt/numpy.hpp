#pragma once

#include <filesystem>
#include <fstream>
#include <istream>
#include <ostream>
#include <typeindex>
#include <vector>

#include "npy.hpp"

namespace minrt {

template <typename T>
void save_npy(std::ostream &stream, unsigned int n_dims,
              const unsigned long shape[], const T *data,
              bool fortran_order = false) {
  const auto dtype = npy::dtype_map.at(std::type_index(typeid(T)));

  std::vector<npy::ndarray_len_t> shape_v(shape, shape + n_dims);
  npy::header_t header{dtype, fortran_order, shape_v};
  npy::write_header(stream, header);

  auto size = static_cast<size_t>(npy::comp_size(shape_v));

  stream.write(reinterpret_cast<const char *>(data), sizeof(T) * size);
}

template <typename T>
void save_npy(fs::path &path, unsigned int n_dims, const unsigned long shape[],
              const T *data, bool fortran_order = false) {
  std::ofstream stream(path, std::ofstream::binary);
  stream.exceptions(std::ofstream::badbit);
  save_npy(stream, n_dims, shape, data, fortran_order);
}

template <typename T>
void load_npy(std::istream &stream, std::vector<unsigned long> &shape,
              std::vector<T> &data, bool fortran_order = false) {
  auto header_s = npy::read_header(stream);
  auto header = npy::parse_header(header_s);
  auto dtype = npy::dtype_map.at(std::type_index(typeid(T)));

  if (header.dtype.tie() != dtype.tie()) {
    throw std::runtime_error("formatting error: typestrings not matching");
  }

  shape = header.shape;
  fortran_order = header.fortran_order;

  auto size = static_cast<size_t>(npy::comp_size(shape));
  data.resize(size);

  stream.read(reinterpret_cast<char *>(data.data()), sizeof(T) * size);
}

template <typename T>
void load_npy(fs::path &path, std::vector<unsigned long> &shape,
              std::vector<T> &data, bool fortran_order = false) {
  std::ifstream stream(path, std::ifstream::binary);
  stream.exceptions(std::ofstream::badbit);
  load_npy(stream, shape, data, fortran_order);
}

}  // namespace minrt
