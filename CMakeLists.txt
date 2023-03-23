cmake_minimum_required(VERSION 3.4...3.22)

project(llama_cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(LLAMA_STATIC "Off")
set(BUILD_SHARED_LIBS "On")

add_subdirectory(vendor/llama.cpp)

install(TARGETS llama LIBRARY DESTINATION llama_cpp)