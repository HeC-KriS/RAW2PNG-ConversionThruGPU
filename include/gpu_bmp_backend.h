#pragma once
#include <cstdint>
#include "dicom_params.h"

// Opaque GPU BMP processing context.
// CUDA types are kept out of this header so plain .cpp translation units
// can include it without the CUDA toolkit headers.
struct GpuBmpContext;

// Per-call GPU timing breakdown (microseconds, from CUDA Events).
struct GpuBmpTimings {
    float h2d_us    = 0.f;  // host-to-device transfer
    float kernel_us = 0.f;  // window/level + 16→8 conversion kernel
    float d2h_us    = 0.f;  // device-to-host transfer
};

// Create a GPU BMP context.
// initial_pixels: expected pixel count (width × height) for pre-allocation.
// Pass 0 to defer allocation until the first gpu_bmp_process call.
GpuBmpContext* gpu_bmp_create(int initial_pixels = 0);
void           gpu_bmp_destroy(GpuBmpContext* ctx);

// Process 16-bit little-endian DICOM pixels → 8-bit grayscale on GPU.
//   h_in16  : caller's width×height uint16_t LE pixel buffer (any host memory)
//   width   : image width in pixels
//   height  : image height in pixels
//   params  : DICOM windowing parameters (bits_stored, pixel_rep, rescale, W/L)
// Returns a pointer to an internal pinned host buffer containing the 8-bit output
// (valid until the next call on this context), or nullptr on failure.
// The caller must NOT free the returned pointer.
const uint8_t* gpu_bmp_process(GpuBmpContext*          ctx,
                               const uint16_t*          h_in16,
                               int                      width,
                               int                      height,
                               const DicomPixelParams&  params);

GpuBmpTimings gpu_bmp_get_timings(const GpuBmpContext* ctx);

// Total VRAM allocated by this context (bytes).
size_t gpu_bmp_vram_bytes(const GpuBmpContext* ctx);

// Force CUDA context creation on the calling thread.
// Call from each encode worker thread before the batch starts to eliminate
// the ~50-200 ms one-time CUDA init overhead from the first processed file.
void gpu_bmp_warmup_thread();
