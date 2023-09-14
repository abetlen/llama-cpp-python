cmake_minimum_required(VERSION 3.4...3.22)

project(llama_cpp)

option(LLAMA_BUILD "Build llama.cpp shared library and install alongside python package" ON)


if (LLAMA_BUILD)
    set(BUILD_SHARED_LIBS "On")
    if (APPLE)
        # Need to disable these llama.cpp flags on Apple
        # otherwise users may encounter invalid instruction errors
        set(LLAMA_AVX "Off" CACHE BOOL "llama: enable AVX" FORCE)
        set(LLAMA_AVX2 "Off" CACHE BOOL "llama: enable AVX2" FORCE)
        set(LLAMA_FMA "Off" CACHE BOOL "llama: enable FMA" FORCE)
        set(LLAMA_F16C "Off" CACHE BOOL "llama: enable F16C" FORCE)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mtune=native")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
    endif()
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
