cmake_minimum_required(VERSION 3.21)

project(llama_cpp)

option(LLAMA_BUILD "Build llama.cpp shared library and install alongside python package" ON)
option(LLAVA_BUILD "Build llava shared library and install alongside python package" ON)

function(llama_cpp_python_install_target target)
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

    # Building llama
    if (APPLE AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        # Need to disable these llama.cpp flags on Apple x86_64,
        # otherwise users may encounter invalid instruction errors
        set(GGML_AVX "Off" CACHE BOOL "ggml: enable AVX" FORCE)
        set(GGML_AVX2 "Off" CACHE BOOL "ggml: enable AVX2" FORCE)
        set(GGML_FMA "Off" CACHE BOOL "gml: enable FMA" FORCE)
        set(GGML_F16C "Off" CACHE BOOL "gml: enable F16C" FORCE)
    endif()

    if (APPLE)
        set(GGML_METAL_EMBED_LIBRARY "On" CACHE BOOL "llama: embed metal library" FORCE)
    endif()

    add_subdirectory(vendor/llama.cpp)
    llama_cpp_python_install_target(llama)
    llama_cpp_python_install_target(ggml)
    
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
        # Set CUDA_ARCHITECTURES to OFF on windows
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
    endif()
endif()
