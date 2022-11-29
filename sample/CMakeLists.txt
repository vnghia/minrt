cmake_minimum_required(VERSION 3.22)

# mnist
include(minrt/external/mnist)
find_package(MNIST REQUIRED)

add_executable(mnist
  mnist.cpp
)

target_include_directories(mnist PUBLIC
  ${MNIST_INCLUDE_DIR}
)

target_link_libraries(mnist PUBLIC
  ${PROJECT_NAME}
  cxxopts::cxxopts
  spdlog::spdlog
)

target_compile_definitions(mnist PRIVATE
  MNIST_DATA_LOCATION="${MNIST_DATA_DIR}"
)

# mnist_managed
add_executable(mnist-managed
  mnist_managed.cpp
)

target_include_directories(mnist-managed PUBLIC
  ${MNIST_INCLUDE_DIR}
)

target_link_libraries(mnist-managed PUBLIC
  ${PROJECT_NAME}
  cxxopts::cxxopts
  spdlog::spdlog
)

target_compile_definitions(mnist-managed PRIVATE
  MNIST_DATA_LOCATION="${MNIST_DATA_DIR}"
)
