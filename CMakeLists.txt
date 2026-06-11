cmake_minimum_required(VERSION 3.18)
project(gpu_png_encoder LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Language standards
# C++17 is required by DCMTK. CUDA 11.8 supports C++17 for both host and
# device code on SM 3.5. Setting both here ensures consistency.
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
find_package(PNG    REQUIRED)
find_package(TIFF   REQUIRED)
find_package(ZLIB   REQUIRED)
find_package(libraw CONFIG REQUIRED)
find_package(DCMTK  CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# dicom_lib — pure C++ static library that owns all DCMTK interaction.
#
# Linking DCMTK as PRIVATE here is the key architectural decision:
#   - DCMTK's INTERFACE_COMPILE_OPTIONS (including /Zc:__cplusplus) are NOT
#     propagated to any consuming target.
#   - This prevents MSVC-only flags from leaking into NVCC when the main
#     executable (which has .cu sources) links against dicom_lib.
#
# This is the permanent fix for the nvcc fatal:
#   "A single input file is required for a non-link phase when an output
#    file is specified"
# caused by /Zc:__cplusplus appearing in CUDA AdditionalOptions.
# ---------------------------------------------------------------------------
add_library(dicom_lib STATIC src/dicom_loader.cpp)

target_include_directories(dicom_lib PRIVATE
    ${CMAKE_SOURCE_DIR}/include)

target_compile_features(dicom_lib PRIVATE cxx_std_17)

# PRIVATE: DCMTK's interface options stay inside dicom_lib, never reaching
# gpu_png_encoder's CUDA compilation units.
target_link_libraries(dicom_lib PRIVATE
    DCMTK::dcmdata
    DCMTK::ofstd
    DCMTK::oflog)

# ---------------------------------------------------------------------------
# Main executable — CUDA + C++ sources, NO direct DCMTK dependency
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
    dicom_lib           # includes DCMTK symbols; no INTERFACE leakage
)

# ---------------------------------------------------------------------------
# NVCC flags
# CMAKE_CUDA_ARCHITECTURES already emits -gencode arch=compute_35,code=sm_35.
# Do NOT add -gencode manually or nvcc will warn about duplicates.
# -Wno-deprecated-gpu-targets: suppress SM 3.5 deprecation noise (CUDA 11.x).
# ---------------------------------------------------------------------------
target_compile_options(gpu_png_encoder PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -Wno-deprecated-gpu-targets
        -Xcompiler=/W3
    >
    $<$<COMPILE_LANGUAGE:CXX>:/W3 /O2>
)

target_compile_options(dicom_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W3 /O2>
)

set_property(TARGET gpu_png_encoder PROPERTY
    CUDA_SEPARABLE_COMPILATION OFF)

target_compile_definitions(gpu_png_encoder PRIVATE
    _SILENCE_NONFLOAT_CXX_DEPRECATED_WARNINGS)
