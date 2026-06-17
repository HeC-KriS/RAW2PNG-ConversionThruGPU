// gpu_capability.cpp
// Plain C++ translation unit (NOT compiled by nvcc) that calls the CUDA
// *runtime* API to detect the active device. This is safe because functions
// like cudaGetDeviceProperties are ordinary exported functions in cudart;
// only kernel launches (<<<>>>) and __global__/__device__ code require nvcc.

#include "gpu_capability.h"
#include <cuda_runtime.h>
#include <cstdio>

GpuCapability detect_gpu_capability()
{
    GpuCapability cap;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        cap.available = false;
        cap.mode      = GpuPipelineMode::Legacy;
        return cap;
    }

    int dev = 0;
    cudaGetDevice(&dev);  // current device (0 unless the caller changed it)

    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, dev) != cudaSuccess) {
        cap.available = false;
        cap.mode      = GpuPipelineMode::Legacy;
        return cap;
    }

    int runtime_ver = 0;
    cudaRuntimeGetVersion(&runtime_ver);

    cap.available        = true;
    cap.device_id        = dev;
    cap.name              = props.name;
    cap.compute_major    = props.major;
    cap.compute_minor    = props.minor;
    cap.vram_bytes       = props.totalGlobalMem;
    cap.cuda_runtime_ver = runtime_ver;

    // Selection rule: compute capability >= 7.0 (Volta+) gets the modern
    // GPU-compression pipeline. GT 710 (SM 3.5) and similar Kepler/Maxwell
    // parts fall back to the legacy CPU-deflate pipeline.
    const float cc = (float)props.major + (float)props.minor / 10.0f;
    cap.mode = (cc >= 7.0f) ? GpuPipelineMode::Modern : GpuPipelineMode::Legacy;

    return cap;
}

std::string describe_gpu_capability(const GpuCapability& cap)
{
    if (!cap.available) {
        return "GPU: none detected (CUDA unavailable) -- legacy CPU paths only";
    }
    char buf[256];
    snprintf(buf, sizeof(buf),
        "GPU: %s  SM %d.%d  CUDA runtime %d.%d  VRAM %.1f GB  mode=%s",
        cap.name.c_str(), cap.compute_major, cap.compute_minor,
        cap.cuda_runtime_ver / 1000, (cap.cuda_runtime_ver % 1000) / 10,
        (double)cap.vram_bytes / (1024.0 * 1024.0 * 1024.0),
        cap.mode == GpuPipelineMode::Modern ? "modern (GPU deflate)" : "legacy (CPU deflate)");
    return buf;
}
