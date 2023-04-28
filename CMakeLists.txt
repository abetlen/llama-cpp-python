set(CMAKE_SYSTEM_PROCESSOR "arm64")

cmake_minimum_required(VERSION 3.4...3.22)

project(llama_cpp)

set(BUILD_SHARED_LIBS "On")
add_subdirectory(vendor/llama.cpp)
install(
    TARGETS llama 
    LIBRARY DESTINATION llama_cpp
    RUNTIME DESTINATION llama_cpp
)
