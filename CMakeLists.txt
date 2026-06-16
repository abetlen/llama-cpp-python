cmake_minimum_required(VERSION 3.21)

project(llama_cpp)

option(LLAMA_BUILD "Build llama.cpp shared library and install alongside python package" ON)
option(LLAVA_BUILD "Build llava shared library and install alongside python package" ON)

function(llama_cpp_python_install_target target)
    if(NOT TARGET ${target})
        return()
    endif()

    if(EMSCRIPTEN)
        set_target_properties(${target} PROPERTIES
            OUTPUT_NAME "${target}.cpython-00-wasm32-emscripten"
        )
    endif()

    if(NOT EMSCRIPTEN)
        install(
            TARGETS ${target}
            LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
            RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
            ARCHIVE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
            FRAMEWORK DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
            RESOURCE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
        )
    endif()
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

    # Disable building curl support
    set(LLAMA_CURL OFF CACHE BOOL "llama.cpp: enable curl" FORCE)

    if (EMSCRIPTEN)
        if (DEFINED EMSCRIPTEN_SYSTEM_PROCESSOR)
            set(CMAKE_SYSTEM_PROCESSOR ${EMSCRIPTEN_SYSTEM_PROCESSOR} CACHE STRING "Target processor" FORCE)
        else()
            set(CMAKE_SYSTEM_PROCESSOR wasm32 CACHE STRING "Target processor" FORCE)
        endif()

        set(LLAMA_WASM_MEM64 OFF CACHE BOOL "llama.cpp: enable wasm64 memory" FORCE)
        set(GGML_NATIVE OFF CACHE BOOL "ggml: enable -march=native" FORCE)
        set(GGML_OPENMP OFF CACHE BOOL "ggml: use OpenMP" FORCE)
        set(GGML_METAL OFF CACHE BOOL "ggml: use Metal" FORCE)
        set(GGML_BLAS OFF CACHE BOOL "ggml: use BLAS" FORCE)
        set(GGML_CUDA OFF CACHE BOOL "ggml: use CUDA" FORCE)
        set(GGML_HIP OFF CACHE BOOL "ggml: use HIP" FORCE)
        set(GGML_VULKAN OFF CACHE BOOL "ggml: use Vulkan" FORCE)
        set(GGML_OPENCL OFF CACHE BOOL "ggml: use OpenCL" FORCE)
        set(GGML_RPC OFF CACHE BOOL "ggml: use RPC" FORCE)

        # Pyodide auto-loads side modules from top-level site-packages/lib
        # before Python imports run, so keep upstream installs package-local.
        set(CMAKE_INSTALL_BINDIR llama_cpp/lib CACHE PATH "Install binaries" FORCE)
        set(CMAKE_INSTALL_INCLUDEDIR llama_cpp/include CACHE PATH "Install headers" FORCE)
        set(CMAKE_INSTALL_LIBDIR llama_cpp/lib CACHE PATH "Install libraries" FORCE)
        set(LLAMA_BUILD_COMMON OFF CACHE BOOL "Build llama.cpp common library" FORCE)
    endif()

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

    if (WIN32)
        if (TARGET llama)
            set_target_properties(llama PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
        endif()
    endif()

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

        # Upstream mtmd expects LLAMA_INSTALL_VERSION to be set by llama.cpp's
        # top-level CMakeLists.txt. When we include tools/mtmd directly from the
        # Python package build, that directory scope is skipped.
        if (NOT DEFINED LLAMA_INSTALL_VERSION OR "${LLAMA_INSTALL_VERSION}" STREQUAL "")
            set(LLAMA_INSTALL_VERSION 0.0.0)
            find_package(Git QUIET)
            if (Git_FOUND)
                execute_process(
                    COMMAND ${GIT_EXECUTABLE} rev-list --count HEAD
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp
                    OUTPUT_VARIABLE LLAMA_MTMD_BUILD_NUMBER
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE LLAMA_MTMD_BUILD_NUMBER_RESULT
                )
                if (LLAMA_MTMD_BUILD_NUMBER_RESULT EQUAL 0)
                    set(LLAMA_INSTALL_VERSION 0.0.${LLAMA_MTMD_BUILD_NUMBER})
                endif()
            endif()
        endif()

        # Building llava
        add_subdirectory(vendor/llama.cpp/tools/mtmd)

        # The Python package only ships mtmd as a shared library.
        # Upstream mtmd also defines CLI compatibility wrappers, but those are
        # not installed here and can fail to link in minimal Docker toolchains.
        foreach(target llama-llava-cli llama-gemma3-cli llama-minicpmv-cli llama-qwen2vl-cli llama-mtmd-cli llama-mtmd-debug)
            if (TARGET ${target})
                set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL TRUE)
            endif()
        endforeach()

        if (WIN32)
            set_target_properties(mtmd PROPERTIES CUDA_ARCHITECTURES OFF)
        endif()
        llama_cpp_python_install_target(mtmd)
        if (WIN32)
            install(
                FILES $<TARGET_RUNTIME_DLLS:mtmd>
                DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
            )
            install(
                FILES $<TARGET_RUNTIME_DLLS:mtmd>
                DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
            )
        endif()

        # Fix for mtmd build: Add include directory for llama.h
        # Move these commands after the add_subdirectory call
        target_include_directories(mtmd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/include)
        target_include_directories(mtmd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/ggml/include)

        if (BUILD_SHARED_LIBS)
            target_include_directories(mtmd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/include)
            target_include_directories(mtmd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/ggml/include)
        endif()

        # target_include_directories(llama-llava-cli PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/include)
        # target_include_directories(llama-minicpmv-cli PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/include)
    endif()
endif()
