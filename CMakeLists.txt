cmake_minimum_required(VERSION 3.21)

project(llama_cpp)

option(LLAMA_BUILD "Build llama.cpp shared library and install alongside python package" ON)
option(LLAVA_BUILD "Build llava shared library and install alongside python package" ON)

if(SKBUILD_STATE STREQUAL "editable")
    # Install into the source directory
    # Temporary fix for https://github.com/scikit-build/scikit-build-core/issues/374
    set(LLAMA_CPP_PYTHON_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib)
else()
    set(LLAMA_CPP_PYTHON_INSTALL_DIR ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib)
endif()

message(STATUS "LLAMA_CPP_PYTHON_INSTALL_DIR: ${LLAMA_CPP_PYTHON_INSTALL_DIR}")

if (LLAMA_BUILD)
    set(BUILD_SHARED_LIBS "On")

    set(CMAKE_SKIP_BUILD_RPATH FALSE)

    # When building, don't use the install RPATH already
    # (but later on when installing)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
 
    # Add the automatically determined parts of the RPATH
    # which point to directories outside the build tree to the install RPATH
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")

    # Building llama
    if (APPLE AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        # Need to disable these llama.cpp flags on Apple x86_64,
        # otherwise users may encounter invalid instruction errors
        set(LLAMA_AVX "Off" CACHE BOOL "llama: enable AVX" FORCE)
        set(LLAMA_AVX2 "Off" CACHE BOOL "llama: enable AVX2" FORCE)
        set(LLAMA_FMA "Off" CACHE BOOL "llama: enable FMA" FORCE)
        set(LLAMA_F16C "Off" CACHE BOOL "llama: enable F16C" FORCE)
    endif()

    if (APPLE)
        set(LLAMA_METAL_EMBED_LIBRARY "On" CACHE BOOL "llama: embed metal library" FORCE)
    endif()

    add_subdirectory(vendor/llama.cpp)
    install(
        TARGETS llama
        LIBRARY DESTINATION ${LLAMA_CPP_PYTHON_INSTALL_DIR}
        RUNTIME DESTINATION ${LLAMA_CPP_PYTHON_INSTALL_DIR}
        ARCHIVE DESTINATION ${LLAMA_CPP_PYTHON_INSTALL_DIR}
        FRAMEWORK DESTINATION ${LLAMA_CPP_PYTHON_INSTALL_DIR}
        RESOURCE DESTINATION ${LLAMA_CPP_PYTHON_INSTALL_DIR}
    )
    install(
        TARGETS ggml
        LIBRARY DESTINATION ${LLAMA_CPP_PYTHON_INSTALL_DIR}
        RUNTIME DESTINATION ${LLAMA_CPP_PYTHON_INSTALL_DIR}
        ARCHIVE DESTINATION ${LLAMA_CPP_PYTHON_INSTALL_DIR}
        FRAMEWORK DESTINATION ${LLAMA_CPP_PYTHON_INSTALL_DIR}
        RESOURCE DESTINATION ${LLAMA_CPP_PYTHON_INSTALL_DIR}
    )
    # Workaround for Windows + CUDA https://github.com/abetlen/llama-cpp-python/issues/563
    if (WIN32 AND (LLAMA_CUDA OR LLAMA_CUBLAS))
        install(
            FILES $<TARGET_RUNTIME_DLLS:llama>
            DESTINATION ${LLAMA_CPP_PYTHON_INSTALL_DIR}
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
        # Set CUDA_ARCHITECTURES to OFF on windows
        if (WIN32)
            set_target_properties(llava_shared PROPERTIES CUDA_ARCHITECTURES OFF)
        endif()
        install(
            TARGETS llava_shared
            LIBRARY DESTINATION ${LLAMA_CPP_PYTHON_INSTALL_DIR}
            RUNTIME DESTINATION ${LLAMA_CPP_PYTHON_INSTALL_DIR}
            ARCHIVE DESTINATION ${LLAMA_CPP_PYTHON_INSTALL_DIR}
            FRAMEWORK DESTINATION ${LLAMA_CPP_PYTHON_INSTALL_DIR}
            RESOURCE DESTINATION ${LLAMA_CPP_PYTHON_INSTALL_DIR}
        )
    endif()
endif()
