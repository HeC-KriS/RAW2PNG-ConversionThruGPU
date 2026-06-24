cmake_minimum_required(VERSION 3.18)
project(gpu_png_encoder LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Language standards
# C++17 is required by DCMTK. CUDA 11.8 supports C++17 for SM 3.5.
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Dual GPU target selection
#
#   LEGACY_GPU=ON  (GT 710, Kepler GK208, SM 3.5, CUDA 11.8 max):
#       CMAKE_CUDA_ARCHITECTURES = 35 (explicit; SM 3.5 removed in CUDA 12.0)
#       GPU deflate / GPU CRC32 / GPU PNG assembly are NOT built -- the
#       legacy pipeline (GPU filter -> CPU deflate -> CPU PNG writer) runs
#       unconditionally regardless of what detect_gpu_capability() reports.
#
#   LEGACY_GPU=OFF (default; RTX 5050 and other SM >= 7.0 parts, CUDA 13.1+):
#       CMAKE_CUDA_ARCHITECTURES = native (requires CMake >= 3.24) -- builds
#       only for the GPU(s) actually present on the build machine.
#       The modern pipeline (custom fixed-Huffman GPU deflate, GPU CRC32,
#       GPU PNG assembly) is built; detect_gpu_capability() still gates
#       which path runs at runtime so the binary degrades gracefully on
#       older GPUs.
#
# Configure with:  cmake -B build -DLEGACY_GPU=ON   (GT 710 box)
#                  cmake -B build                   (RTX 5050 box, default)
# ---------------------------------------------------------------------------
option(LEGACY_GPU "Build for GT710 / SM 3.5 compatibility (CUDA 11.8, no GPU deflate)" OFF)

if(LEGACY_GPU)
    set(CMAKE_CUDA_ARCHITECTURES 35)
    message(STATUS "LEGACY_GPU=ON  -> CMAKE_CUDA_ARCHITECTURES=35, GPU deflate/CRC32/assembly disabled")
else()
    if(CMAKE_VERSION VERSION_LESS 3.24)
        message(FATAL_ERROR
            "CMAKE_CUDA_ARCHITECTURES=native requires CMake >= 3.24 (found ${CMAKE_VERSION}). "
            "Either upgrade CMake, or configure with -DLEGACY_GPU=ON for GT710/SM35 builds.")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES native)
    message(STATUS "LEGACY_GPU=OFF -> CMAKE_CUDA_ARCHITECTURES=native, modern GPU pipeline enabled")
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
find_package(PNG     REQUIRED)
find_package(TIFF    REQUIRED)

# ---------------------------------------------------------------------------
# DEFLATE backend
#
# The deflate stage is CPU-bound on zlib's level-1 speed (~40 MB/s/core); the
# 6 worker threads are already saturated, so the only way to cut deflate time
# materially is a faster zlib-API-compatible codec.
#
# zlib-ng (in zlib-compat mode) is a drop-in replacement: it exports the same
# ZLIB::ZLIB target and the same deflate()/deflateInit2()/adler32() ABI, so NO
# source changes are needed and the Z_FULL_FLUSH / BFINAL=1 correctness
# invariant in parallel_deflate.cpp is preserved. Typical speedup: 1.8-2.5x.
#
# To switch backends, install the compat build and reconfigure:
#     vcpkg install zlib-ng[core,zlib-compat]
# find_package(ZLIB) below then resolves to zlib-ng automatically.
#
# libdeflate is intentionally NOT used: it always emits BFINAL=1 and has no
# Z_FULL_FLUSH equivalent, which breaks the concatenated-stream design.
# ---------------------------------------------------------------------------
find_package(ZLIB    REQUIRED)
find_package(libraw  CONFIG REQUIRED)
find_package(DCMTK   CONFIG REQUIRED)

# OpenJPEG is required for JPEG2000 DICOM decode.
# vcpkg installs it as target 'openjp2' via OpenJPEGConfig.cmake.
find_package(OpenJPEG CONFIG REQUIRED)

# Resolve the OpenJPEG link target (cmake name differs across OPJ versions).
# OpenJPEG 2.x exports 'openjp2'; some patched builds use 'OpenJPEG::openjp2'.
if(TARGET openjp2)
    set(_OPENJP2_TARGET openjp2)
elseif(TARGET OpenJPEG::openjp2)
    set(_OPENJP2_TARGET OpenJPEG::openjp2)
else()
    message(FATAL_ERROR
        "OpenJPEG found but no known cmake target (openjp2 / OpenJPEG::openjp2). "
        "Check your vcpkg OpenJPEG installation.")
endif()
message(STATUS "OpenJPEG target: ${_OPENJP2_TARGET}  version ${OpenJPEG_VERSION}")

# ---------------------------------------------------------------------------
# GPU DEFLATE backend: nvCOMP was evaluated and rejected for this role.
#
# nvCOMP's deflate is a one-shot codec: each call produces a complete,
# independently BFINAL=1-terminated stream with no exposed bit-precise end
# position, which makes splicing several such streams into one continuous
# PNG IDAT bitstream unverifiable without either an undocumented API
# guarantee or a hand-written GPU inflate decode-trace -- neither could be
# checked without nvCOMP's actual headers and real hardware in the authoring
# session. src/gpu_deflate_backend.cu instead implements a small custom
# fixed-Huffman encoder that tracks its own output bit position as it writes,
# so this problem cannot occur by construction. See that file's header
# comment for the full rationale. No nvCOMP dependency is needed as a result.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# dicom_lib — pure C++ static library that owns all DCMTK/OpenJPEG interaction.
#
# Linking DCMTK and OpenJPEG as PRIVATE isolates their INTERFACE_COMPILE_OPTIONS
# (notably DCMTK's /Zc:__cplusplus) from gpu_png_encoder's CUDA compilation.
# ---------------------------------------------------------------------------
add_library(dicom_lib STATIC src/dicom_loader.cpp)

target_include_directories(dicom_lib PRIVATE
    ${CMAKE_SOURCE_DIR}/include)

target_compile_features(dicom_lib PRIVATE cxx_std_17)

target_link_libraries(dicom_lib PRIVATE
    DCMTK::dcmdata
    DCMTK::ofstd
    DCMTK::oflog
    DCMTK::dcmjpeg    # JPEG Baseline / Extended / Lossless
    DCMTK::dcmjpls    # JPEG-LS
    ${_OPENJP2_TARGET} # JPEG 2000 (decoded via OpenJPEG C API, not dcmjp2k)
)

# ---------------------------------------------------------------------------
# Main executable — CUDA + C++ sources, no direct DCMTK or OpenJPEG dependency
# ---------------------------------------------------------------------------
set(SOURCES
    src/main.cpp
    src/gpu_filter.cu
    src/parallel_deflate.cpp
    src/png_writer.cpp
    src/bmp_writer.cpp
    src/image_loader.cpp
    src/pipeline.cpp
    src/gpu_capability.cpp
    src/batch_processor.cpp
    src/file_type.cpp
    src/jls_encoder.cpp
    src/gpu_jls_backend.cu
    src/gpu_bmp_backend.cu
)

# GPU CRC32, GPU Adler32, the custom GPU DEFLATE encoder, and GPU PNG chunk
# assembly only need a real GPU to test (no external library dependency), so
# all four are gated purely on the modern build (NOT LEGACY_GPU). Together
# they form the modern single-image pipeline wired into run_pipeline() behind
# --gpu-deflate (opt-in; off by default -- see PipelineConfig::use_gpu_deflate).
if(NOT LEGACY_GPU)
    list(APPEND SOURCES src/gpu_crc32.cu src/gpu_deflate_backend.cu
                        src/gpu_png_assemble.cu src/gpu_adler32.cu)
endif()

# Gates pipeline.cpp's #include of the four headers above and the modern
# single-image GPU-deflate pipeline they enable -- must match the SOURCES
# condition above exactly, or LEGACY_GPU builds would try to link symbols
# that were never compiled in.
if(NOT LEGACY_GPU)
    list(APPEND GPU_PNG_COMPILE_DEFS GPU_PNG_MODERN_DEFLATE=1)
endif()

add_executable(gpu_png_encoder ${SOURCES})

target_include_directories(gpu_png_encoder PRIVATE
    ${CMAKE_SOURCE_DIR}/include)

target_compile_features(gpu_png_encoder PRIVATE cxx_std_17)

find_library(DCMTK_CHARLS_LIB NAMES dcmtkcharls
    PATHS "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/lib"
    NO_DEFAULT_PATH)
if(NOT DCMTK_CHARLS_LIB)
    message(FATAL_ERROR "dcmtkcharls.lib not found in vcpkg_installed/x64-windows/lib")
endif()

target_link_libraries(gpu_png_encoder PRIVATE
    PNG::PNG
    TIFF::TIFF
    ZLIB::ZLIB
    libraw::raw_r
    dicom_lib       # includes DCMTK + OpenJPEG symbols; no INTERFACE flag leakage
    CUDA::cudart    # exposes cuda_runtime.h to plain .cpp TUs (gpu_capability.cpp)
    ${DCMTK_CHARLS_LIB}
)

target_link_options(gpu_png_encoder PRIVATE /LTCG)
target_link_options(dicom_lib PRIVATE /LTCG)

# ---------------------------------------------------------------------------
# NVCC flags
# ---------------------------------------------------------------------------
target_compile_options(gpu_png_encoder PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -Wno-deprecated-gpu-targets
        -Xcompiler=/W3
    >
    $<$<COMPILE_LANGUAGE:CXX>:/O2 /GL>
)

target_compile_options(dicom_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/O2 /GL>
)

set_property(TARGET gpu_png_encoder PROPERTY
    CUDA_SEPARABLE_COMPILATION OFF)

target_compile_definitions(gpu_png_encoder PRIVATE
    _SILENCE_NONFLOAT_CXX_DEPRECATED_WARNINGS
    ${GPU_PNG_COMPILE_DEFS})
