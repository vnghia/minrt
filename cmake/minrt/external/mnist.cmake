include(FetchContent)
FetchContent_Declare(
  mnist
  GIT_REPOSITORY https://github.com/wichtounet/mnist.git
  GIT_TAG 3b65c35ede53b687376c4302eeb44fdf76e0129b
)
FetchContent_MakeAvailable(mnist)
set(MNIST_DIR ${mnist_SOURCE_DIR})
