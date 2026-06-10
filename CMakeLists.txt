cmake_minimum_required(VERSION 3.18)
project(gpu_png_encoder LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GT 710 = Kepler GK208, Compute Capability 3.5
# CUDA 11.8 still supports SM 3.5 (removed in CUDA 12.0)
set(CMAKE_CUDA_ARCHITECTURES 35)

find_package(PNG    REQUIRED)
find_package(TIFF   REQUIRED)
find_package(ZLIB   REQUIRED)
find_package(libraw CONFIG REQUIRED)
# LibRaw: vcpkg exposes it as an unofficial find module



set(SOURCES
    src/main.cpp
    src/gpu_filter.cu
    src/parallel_deflate.cpp
    src/png_writer.cpp
    src/image_loader.cpp
    src/pipeline.cpp
)

add_executable(gpu_png_encoder ${SOURCES})

target_include_directories(gpu_png_encoder PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)



target_link_libraries(gpu_png_encoder PRIVATE
    PNG::PNG
    TIFF::TIFF
    ZLIB::ZLIB
    libraw::raw_r
)

# NVCC flags – CMAKE_CUDA_ARCHITECTURES already emits the correct -gencode;
# do NOT repeat it here or nvcc will warn about duplicate arch specs.
# -Wno-deprecated-gpu-targets silences the SM 3.5 deprecation warning in
# CUDA 11.x (SM 3.x is fully removed only in CUDA 12.0).
target_compile_options(gpu_png_encoder PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -Wno-deprecated-gpu-targets
        -Xcompiler /W3
    >
    $<$<COMPILE_LANGUAGE:CXX>:/W3 /O2>
)

set_property(TARGET gpu_png_encoder PROPERTY
    CUDA_SEPARABLE_COMPILATION OFF)

# Suppress SM 3.5 deprecation noise from CUDA 11.x
target_compile_definitions(gpu_png_encoder PRIVATE
    _SILENCE_NONFLOAT_CXX_DEPRECATED_WARNINGS)
