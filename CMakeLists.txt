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

# GT 710 = Kepler GK208, Compute Capability 3.5
# CUDA 11.8 still supports SM 3.5 (removed in CUDA 12.0)
set(CMAKE_CUDA_ARCHITECTURES 35)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
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
    src/image_loader.cpp
    src/pipeline.cpp
)

add_executable(gpu_png_encoder ${SOURCES})

target_include_directories(gpu_png_encoder PRIVATE
    ${CMAKE_SOURCE_DIR}/include)

target_compile_features(gpu_png_encoder PRIVATE cxx_std_17)

target_link_libraries(gpu_png_encoder PRIVATE
    PNG::PNG
    TIFF::TIFF
    ZLIB::ZLIB
    libraw::raw_r
    dicom_lib       # includes DCMTK + OpenJPEG symbols; no INTERFACE flag leakage
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
    _SILENCE_NONFLOAT_CXX_DEPRECATED_WARNINGS)
