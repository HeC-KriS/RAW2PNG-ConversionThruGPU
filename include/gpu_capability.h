#pragma once
// gpu_capability.h
// Runtime CUDA device detection used to select between the legacy pipeline
// (GPU filter -> CPU deflate -> CPU PNG writer; targets GT 710 / SM 3.5) and
// the modern pipeline (GPU filter -> GPU deflate -> GPU CRC32 -> GPU PNG
// assembly; targets RTX 5050 / SM >= 7.0).
//
// Plain C++ header (no CUDA headers) so any .cpp translation unit can include
// it. The .cpp implementation calls the CUDA *runtime* API (cudaGetDeviceProperties
// etc.), which does not require nvcc -- only linking against cudart.

#include <cstddef>
#include <string>

enum class GpuPipelineMode {
    Legacy,   // SM < 7.0: GPU filter -> CPU deflate -> CPU PNG writer
    Modern,   // SM >= 7.0: GPU filter -> GPU deflate -> GPU CRC32 -> GPU PNG assembly
};

struct GpuCapability {
    bool            available        = false;  // false if no CUDA device found
    int             device_id        = -1;
    std::string     name;
    int             compute_major    = 0;
    int             compute_minor    = 0;
    size_t          vram_bytes       = 0;
    int             cuda_runtime_ver = 0;       // e.g. 11080, 13010 (CUDART_VERSION-style)
    GpuPipelineMode mode             = GpuPipelineMode::Legacy;
};

// Query device 0 (the active CUDA device) and decide which pipeline mode to
// run. Selection rule: compute capability >= 7.0 -> Modern, else Legacy.
// If no CUDA device is available, returns available=false, mode=Legacy
// (callers must still handle "no GPU at all" as a hard error elsewhere --
// this struct only decides which GPU pipeline variant to use).
GpuCapability detect_gpu_capability();

// One-line human-readable summary for --verbose / startup logging.
std::string describe_gpu_capability(const GpuCapability& cap);
