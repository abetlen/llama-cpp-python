cmake_minimum_required(VERSION 3.21)

project(llama_cpp)

option(LLAMA_BUILD "Build llama.cpp shared library and install alongside python package" ON)

if (LLAMA_BUILD)
    include(ExternalProject)
    if(NOT DEFINED SKBUILD_PLATLIB_DIR)
        set(SKBUILD_PLATLIB_DIR ${CMAKE_SOURCE_DIR})
    endif()
    ExternalProject_Add(
      llama-cpp
      GIT_REPOSITORY "https://github.com/ggerganov/llama.cpp"
      GIT_TAG f5ef5cf
      CMAKE_ARGS -DBUILD_SHARED_LIBS=ON "-DCMAKE_INSTALL_PREFIX=${SKBUILD_PLATLIB_DIR}/llama_cpp/data"
      USES_TERMINAL_DOWNLOAD 1
      USES_TERMINAL_CONFIGURE 1
      USES_TERMINAL_BUILD 1)
endif()
