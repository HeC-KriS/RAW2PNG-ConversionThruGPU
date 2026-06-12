#pragma once
#include <cstddef>
#include <cstdint>
#include "dicom_params.h"

// Opaque handle – CUDA types are kept out of this header so plain .cpp
// translation units can include it without needing the CUDA toolkit headers.
struct GpuFilterContext;

// Per-call GPU timing breakdown (milliseconds, from CUDA Events).
// Populated only when the timings pointer is non-null.
struct GpuTimings {
    float h2d_ms    = 0.f;  // host-to-device transfer
    float kernel_ms = 0.f;  // all GPU kernels (DICOM preprocess + PNG filters)
    float d2h_ms    = 0.f;  // device-to-host transfer
};

GpuFilterContext* gpu_filter_create(int width, int bpp, int strip_height);
void              gpu_filter_destroy(GpuFilterContext* ctx);

// Reset per-image filter state (zeros the prior-row buffer). Call this before
// processing the first strip of a new image/frame when reusing a context, so
// the first row's Up/Paeth predictors see zeros as the PNG spec requires.
void              gpu_filter_reset(GpuFilterContext* ctx);

// Process a strip already in device memory.
// Pass dicom != nullptr to run the DICOM pixel preprocessing kernel before PNG filtering.
// The kernel transforms raw little-endian DICOM pixel values (bit-depth, sign,
// rescale slope/intercept, window/level) to big-endian PNG-ready samples, in-place.
const uint8_t* gpu_filter_process_from_device(
    GpuFilterContext*       ctx,
    const uint8_t*          d_input,
    const uint8_t*          d_prior_row,
    int                     actual_rows,
    GpuTimings*             timings = nullptr,
    const DicomPixelParams* dicom   = nullptr);

// Same as above but uploads h_input from host (pageable) memory first.
const uint8_t* gpu_filter_process_from_host(
    GpuFilterContext*       ctx,
    const uint8_t*          h_input,
    const uint8_t*          h_prior_row,
    int                     actual_rows,
    GpuTimings*             timings = nullptr,
    const DicomPixelParams* dicom   = nullptr);

// Bytes in the output for a strip of actual_rows rows:
//   actual_rows * (width * bpp + 1)   (+1 for the per-row PNG filter byte)
size_t gpu_filter_output_size(const GpuFilterContext* ctx, int actual_rows);
