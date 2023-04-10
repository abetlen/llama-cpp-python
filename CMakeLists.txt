cmake_minimum_required(VERSION 3.4...3.22)

project(llama_cpp)

if (UNIX)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/libllama.so
        COMMAND make libllama.so
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp
    )
    add_custom_target(
        run ALL
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/libllama.so
    )
    install(
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/libllama.so
        DESTINATION llama_cpp
    )
else()
    set(BUILD_SHARED_LIBS "On")
    add_subdirectory(vendor/llama.cpp)
    install(
        TARGETS llama 
        LIBRARY DESTINATION llama_cpp
        RUNTIME DESTINATION llama_cpp
    )
endif(UNIX)
