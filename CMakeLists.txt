cmake_minimum_required(VERSION 3.4...3.22)

project(llama_cpp)

option(BUILD_LLAMA_CPP "Build llama.cpp shared library and install alongside python package" ON)

if (APPLE)
    set(LLAMA_AVX OFF)
    set(LLAMA_AVX2 OFF)
    set(LLAMA_AVX512 OFF)
    set(LLAMA_AVX512_VBMI OFF)
    set(LLAMA_AVX512_VNNI OFF)
    set(LLAMA_FMA OFF)
    set(LLAMA_F16C OFF)
    set(LLAMA_ACCELERATE OFF)
    set(LLAMA_METAL OFF)
endif()

if (BUILD_LLAMA_CPP)
    set(BUILD_SHARED_LIBS "On")
    add_subdirectory(vendor/llama.cpp)
    install(
        TARGETS llama 
        LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp
        RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp
        ARCHIVE DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp
        FRAMEWORK DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp
        RESOURCE DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp
    )
    # Temporary fix for https://github.com/scikit-build/scikit-build-core/issues/374
    install(
        TARGETS llama 
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp
        RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp
        ARCHIVE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp
        FRAMEWORK DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp
        RESOURCE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp
    )
endif()
