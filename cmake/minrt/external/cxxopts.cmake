if(TARGET cxxopts::cxxopts)
  return()
endif()

include(FetchContent)
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG v3.0.0
)
FetchContent_MakeAvailable(cxxopts)
