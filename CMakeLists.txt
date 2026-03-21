cmake_minimum_required(VERSION 3.18)
project(cfireants C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(CFIREANTS_CUDA "Build with CUDA backend" ON)
option(CFIREANTS_WEBGPU "Build with WebGPU backend (future)" OFF)

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_C_FLAGS_RELEASE "-O3 -ffast-math -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find zlib (required for NIfTI gzip support)
find_package(ZLIB REQUIRED)

# Core C library (always built)
add_library(cfireants STATIC
    src/tensor.c
    src/nifti_io.c
    src/image.c
    src/interpolator.c
    src/losses.c
    src/utils.c
    backend/cpu/backend_cpu.c
)

target_include_directories(cfireants PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(cfireants PUBLIC ZLIB::ZLIB m)

# Check for zstd (optional)
find_library(ZSTD_LIB zstd)
if(ZSTD_LIB)
    target_compile_definitions(cfireants PRIVATE HAVE_ZSTD)
    target_link_libraries(cfireants PUBLIC ${ZSTD_LIB})
    message(STATUS "Found zstd: ${ZSTD_LIB}")
endif()

# Always define HAVE_ZLIB since we require it
target_compile_definitions(cfireants PRIVATE HAVE_ZLIB)

# CUDA backend
if(CFIREANTS_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)

    add_library(cfireants_cuda STATIC
        backend/cuda/backend_cuda.cu
        backend/cuda/backend_cuda_ops.cu
        backend/cuda/grid_sample.cu
        backend/cuda/cc_loss.cu
        backend/cuda/fused_cc.cu
        backend/cuda/fused_compose.cu
        backend/cuda/fused_blur.cu
        backend/cuda/warp_inverse.cu
        backend/cuda/mi_loss.cu
        backend/cuda/downsample_fft.cu
        backend/cuda/moments_gpu.cu
        backend/cuda/greedy_gpu.cu
        backend/cuda/syn_gpu.cu
        backend/cuda/linear_gpu.cu
    )

    target_include_directories(cfireants_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/cuda
    )

    target_include_directories(cfireants_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_definitions(cfireants PUBLIC CFIREANTS_HAS_CUDA)
    target_link_libraries(cfireants_cuda PUBLIC cfireants cufft)

    # Set CUDA architecture (auto-detect or default)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()

    message(STATUS "CUDA backend enabled (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
endif()

# Test programs
add_executable(test_nifti_load tests/test_nifti_load.c)
target_link_libraries(test_nifti_load cfireants)
if(CFIREANTS_CUDA)
    target_link_libraries(test_nifti_load cfireants_cuda)
endif()

add_executable(test_phase6 tests/test_phase6.c
    src/registration/moments.c src/registration/rigid.c
    src/registration/affine.c src/registration/greedy.c src/registration/syn.c)
target_link_libraries(test_phase6 cfireants)
if(CFIREANTS_CUDA)
    target_link_libraries(test_phase6 cfireants_cuda)
endif()

add_executable(test_phase5 tests/test_phase5.c src/registration/moments.c src/registration/rigid.c src/registration/affine.c)
target_link_libraries(test_phase5 cfireants)
if(CFIREANTS_CUDA)
    target_link_libraries(test_phase5 cfireants_cuda)
endif()

add_executable(test_phase4 tests/test_phase4.c src/registration/moments.c src/registration/rigid.c)
target_link_libraries(test_phase4 cfireants)
if(CFIREANTS_CUDA)
    target_link_libraries(test_phase4 cfireants_cuda)
endif()

add_executable(test_phase3 tests/test_phase3.c src/registration/moments.c)
target_link_libraries(test_phase3 cfireants)
if(CFIREANTS_CUDA)
    target_link_libraries(test_phase3 cfireants_cuda)
endif()

add_executable(test_phase2 tests/test_phase2.c)
target_link_libraries(test_phase2 cfireants)
if(CFIREANTS_CUDA)
    target_link_libraries(test_phase2 cfireants_cuda)
endif()

add_executable(test_validate_all tests/test_validate_all.c
    src/registration/moments.c src/registration/rigid.c src/registration/affine.c)
target_link_libraries(test_validate_all cfireants)
if(CFIREANTS_CUDA)
    target_link_libraries(test_validate_all cfireants_cuda)
endif()

add_executable(test_gpu_pipeline tests/test_gpu_pipeline.c
    src/registration/moments.c src/registration/rigid.c src/registration/affine.c)
target_link_libraries(test_gpu_pipeline cfireants)
if(CFIREANTS_CUDA)
    target_link_libraries(test_gpu_pipeline cfireants_cuda)
endif()

add_executable(test_cuda_backend tests/test_cuda_backend.c)
target_link_libraries(test_cuda_backend cfireants)
if(CFIREANTS_CUDA)
    target_link_libraries(test_cuda_backend cfireants_cuda)
endif()

# WebGPU backend
if(CFIREANTS_WEBGPU)
    # wgpu-native: expect pre-downloaded in third_party/wgpu/
    set(WGPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/wgpu")
    if(NOT EXISTS "${WGPU_DIR}/lib/libwgpu_native.a")
        message(FATAL_ERROR "wgpu-native not found at ${WGPU_DIR}/lib/libwgpu_native.a\n"
                "Download from https://github.com/gfx-rs/wgpu-native/releases and extract to third_party/wgpu/")
    endif()

    add_library(wgpu_native STATIC IMPORTED)
    set_target_properties(wgpu_native PROPERTIES
        IMPORTED_LOCATION "${WGPU_DIR}/lib/libwgpu_native.a"
        INTERFACE_INCLUDE_DIRECTORIES "${WGPU_DIR}/include"
    )

    add_library(cfireants_webgpu STATIC
        backend/webgpu/webgpu_context.c
        backend/webgpu/backend_webgpu.c
        backend/webgpu/backend_webgpu_ops.c
        backend/webgpu/webgpu_kernels.c
        backend/webgpu/webgpu_kernels_gpu.c
        backend/webgpu/linear_webgpu.c
        backend/webgpu/greedy_webgpu.c
        backend/webgpu/syn_webgpu.c
        backend/webgpu/fft_cpu_fallback.c
        third_party/kissfft/kiss_fft.c
        third_party/kissfft/kiss_fftnd.c
    )

    target_include_directories(cfireants_webgpu PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/webgpu
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/kissfft
        ${WGPU_DIR}/include
    )
    target_include_directories(cfireants_webgpu PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_definitions(cfireants PUBLIC CFIREANTS_HAS_WEBGPU)
    target_link_libraries(cfireants_webgpu PUBLIC cfireants wgpu_native dl pthread m)

    message(STATUS "WebGPU backend enabled (wgpu-native)")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA backend: ${CFIREANTS_CUDA}")
message(STATUS "WebGPU backend: ${CFIREANTS_WEBGPU}")

add_executable(test_validate_greedy tests/test_validate_all.c
    src/registration/moments.c src/registration/rigid.c src/registration/affine.c)
target_compile_definitions(test_validate_greedy PRIVATE USE_GREEDY)
target_link_libraries(test_validate_greedy cfireants)
if(CFIREANTS_CUDA)
    target_link_libraries(test_validate_greedy cfireants_cuda)
endif()

# WebGPU tests
if(CFIREANTS_WEBGPU)
    add_executable(test_webgpu_basic tests/test_webgpu_basic.c)
    target_link_libraries(test_webgpu_basic cfireants cfireants_webgpu)
    if(CFIREANTS_CUDA)
        target_link_libraries(test_webgpu_basic cfireants_cuda)
    endif()

    add_executable(test_fft_fallback tests/test_fft_fallback.c)
    target_link_libraries(test_fft_fallback cfireants cfireants_webgpu)
    if(CFIREANTS_CUDA)
        target_link_libraries(test_fft_fallback cfireants_cuda)
    endif()

    add_executable(test_webgpu_linear tests/test_webgpu_linear.c
        src/registration/moments.c)
    target_link_libraries(test_webgpu_linear cfireants cfireants_webgpu)
    if(CFIREANTS_CUDA)
        target_link_libraries(test_webgpu_linear cfireants_cuda)
    endif()

    add_executable(test_validate_webgpu tests/test_validate_webgpu.c
        src/registration/moments.c)
    target_link_libraries(test_validate_webgpu cfireants cfireants_webgpu)
    if(CFIREANTS_CUDA)
        target_link_libraries(test_validate_webgpu cfireants_cuda)
    endif()
endif()
