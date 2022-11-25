if(TARGET spdlog::spdlog)
  return()
endif()

include(FetchContent)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG cabbe65be461b95576ca66cdbcb23de1e6de9944
)
FetchContent_MakeAvailable(spdlog)
