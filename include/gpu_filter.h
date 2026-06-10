#pragma once
#include <cstddef>
#include <cstdint>

// Opaque handle – CUDA types are kept out of this header so plain .cpp
// translation units can include it without needing the CUDA toolkit headers.
struct GpuFilterContext;

// Per-call GPU timing breakdown (milliseconds, from CUDA Events).
// Populated only when the timings pointer passed to a process function is non-null.
struct GpuTimings {
    float h2d_ms    = 0.f;  // host-to-device transfer
    float kernel_ms = 0.f;  // all three filter kernels
    float d2h_ms    = 0.f;  // device-to-host transfer
};

// Allocate device buffers sized for the given image geometry.
// bpp = bytes per pixel (e.g. 6 for RGB-16).
// strip_height is the maximum number of rows per strip.
GpuFilterContext* gpu_filter_create(int width, int bpp, int strip_height);
void              gpu_filter_destroy(GpuFilterContext* ctx);

// Process a strip whose raw data already lives on the device.
// d_prior_row may be nullptr (context tracks it internally).
const uint8_t* gpu_filter_process_from_device(
    GpuFilterContext*  ctx,
    const uint8_t*     d_input,
    const uint8_t*     d_prior_row,
    int                actual_rows,
    GpuTimings*        timings = nullptr);

// Convenience wrapper: uploads h_input from host first, then runs kernels.
// h_prior_row may be nullptr (context tracks it internally).
// If timings is non-null it is filled with H2D / kernel / D2H milliseconds.
const uint8_t* gpu_filter_process_from_host(
    GpuFilterContext*  ctx,
    const uint8_t*     h_input,
    const uint8_t*     h_prior_row,
    int                actual_rows,
    GpuTimings*        timings = nullptr);

// Number of bytes in the output buffer for a strip of actual_rows rows.
// = actual_rows * (width * bpp + 1)   (+1 for the per-row filter byte)
size_t gpu_filter_output_size(const GpuFilterContext* ctx, int actual_rows);
