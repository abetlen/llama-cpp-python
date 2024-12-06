cmake_minimum_required(VERSION 3.21)

project(llama_cpp)

option(LLAMA_BUILD "Build llama.cpp shared library and install alongside python package" ON)
option(LLAVA_BUILD "Build llava shared library and install alongside python package" ON)

function(llama_cpp_python_install_target target)
    if(NOT TARGET ${target})
        return()
    endif()

    install(
        TARGETS ${target}
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
        RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
        ARCHIVE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
        FRAMEWORK DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
        RESOURCE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
    )
    install(
        TARGETS ${target}
        LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
        RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
        ARCHIVE DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
        FRAMEWORK DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
        RESOURCE DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
    )
    set_target_properties(${target} PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    if(UNIX)
        if(APPLE)
            set_target_properties(${target} PROPERTIES
                INSTALL_RPATH "@loader_path"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        else()
            set_target_properties(${target} PROPERTIES
                INSTALL_RPATH "$ORIGIN"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        endif()
    endif()
endfunction()

if (LLAMA_BUILD)
    set(BUILD_SHARED_LIBS "On")

    set(CMAKE_SKIP_BUILD_RPATH FALSE)

    # When building, don't use the install RPATH already
    # (but later on when installing)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
 
    # Add the automatically determined parts of the RPATH
    # which point to directories outside the build tree to the install RPATH
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_SKIP_RPATH FALSE)

    # Enable building of the common library
    set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build llama.cpp common library" FORCE)

    # Architecture detection and settings for Apple platforms
    if (APPLE)
        # Get the target architecture
        execute_process(
            COMMAND uname -m
            OUTPUT_VARIABLE HOST_ARCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        # If CMAKE_OSX_ARCHITECTURES is not set, use the host architecture
        if(NOT CMAKE_OSX_ARCHITECTURES)
            set(CMAKE_OSX_ARCHITECTURES ${HOST_ARCH} CACHE STRING "Build architecture for macOS" FORCE)
        endif()

        message(STATUS "Host architecture: ${HOST_ARCH}")
        message(STATUS "Target architecture: ${CMAKE_OSX_ARCHITECTURES}")

        # Configure based on target architecture
        if(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
            # Intel Mac settings
            set(GGML_AVX "OFF" CACHE BOOL "ggml: enable AVX" FORCE)
            set(GGML_AVX2 "OFF" CACHE BOOL "ggml: enable AVX2" FORCE)
            set(GGML_FMA "OFF" CACHE BOOL "ggml: enable FMA" FORCE)
            set(GGML_F16C "OFF" CACHE BOOL "ggml: enable F16C" FORCE)
        endif()

        # Metal settings (enable for both architectures)
        set(GGML_METAL "ON" CACHE BOOL "ggml: enable Metal" FORCE)
        set(GGML_METAL_EMBED_LIBRARY "ON" CACHE BOOL "ggml: embed metal library" FORCE)
    endif()

    add_subdirectory(vendor/llama.cpp)
    llama_cpp_python_install_target(llama)
    llama_cpp_python_install_target(ggml)

    llama_cpp_python_install_target(ggml-base)

    llama_cpp_python_install_target(ggml-amx)
    llama_cpp_python_install_target(ggml-blas)
    llama_cpp_python_install_target(ggml-can)
    llama_cpp_python_install_target(ggml-cpu)
    llama_cpp_python_install_target(ggml-cuda)
    llama_cpp_python_install_target(ggml-hip)
    llama_cpp_python_install_target(ggml-kompute)
    llama_cpp_python_install_target(ggml-metal)
    llama_cpp_python_install_target(ggml-musa)
    llama_cpp_python_install_target(ggml-rpc)
    llama_cpp_python_install_target(ggml-sycl)
    llama_cpp_python_install_target(ggml-vulkan)

    # Workaround for Windows + CUDA https://github.com/abetlen/llama-cpp-python/issues/563
    if (WIN32)
        install(
            FILES $<TARGET_RUNTIME_DLLS:llama>
            DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
        )
        install(
            FILES $<TARGET_RUNTIME_DLLS:llama>
            DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
        )
        install(
            FILES $<TARGET_RUNTIME_DLLS:ggml>
            DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
        )
        install(
            FILES $<TARGET_RUNTIME_DLLS:ggml>
            DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
        )
    endif()

    if (LLAVA_BUILD)
        if (LLAMA_CUBLAS OR LLAMA_CUDA)
            add_compile_definitions(GGML_USE_CUBLAS)
            add_compile_definitions(GGML_USE_CUDA)
        endif()

        if (LLAMA_METAL)
            add_compile_definitions(GGML_USE_METAL)
        endif()

        # Building llava
        add_subdirectory(vendor/llama.cpp/examples/llava)
        set_target_properties(llava_shared PROPERTIES OUTPUT_NAME "llava")

        if (WIN32)
            set_target_properties(llava_shared PROPERTIES CUDA_ARCHITECTURES OFF)
        endif()
        llama_cpp_python_install_target(llava_shared)
        if (WIN32)
            install(
                FILES $<TARGET_RUNTIME_DLLS:llava_shared>
                DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
            )
            install(
                FILES $<TARGET_RUNTIME_DLLS:llava_shared>
                DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
            )
        endif()

        # Fix for llava build: Add include directory for llama.h
        # Move these commands after the add_subdirectory call
        target_include_directories(llava PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/include)
        target_include_directories(llava PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/ggml/include)

        if (BUILD_SHARED_LIBS)
            target_include_directories(llava_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/include)
            target_include_directories(llava_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/ggml/include)
        endif()

        target_include_directories(llama-llava-cli PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/include)
        target_include_directories(llama-minicpmv-cli PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/include)
    endif()
endif()
