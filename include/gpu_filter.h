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

// device_output_only: when true (modern GPU-deflate path) skips allocating the
// pinned h_output buffer entirely. The modern path reads d_selected directly
// via gpu_filter_device_output() and never touches h_output. Pass false (default)
// for the legacy path that returns h_output from gpu_filter_process_from_host().
GpuFilterContext* gpu_filter_create(int width, int bpp, int strip_height,
                                    bool device_output_only = false);
void              gpu_filter_destroy(GpuFilterContext* ctx);

// Allocate/free CUDA pinned (page-locked) host memory. Wraps cudaMallocHost /
// cudaFreeHost so callers in plain .cpp translation units (e.g. pipeline.cpp)
// can use pinned strip buffers without including cuda_runtime.h themselves.
uint8_t* gpu_filter_alloc_pinned(size_t bytes);
void     gpu_filter_free_pinned(uint8_t* ptr);

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
// Set device_output_only=true for the modern GPU-deflate path: skips the
// D2H copy of d_selected → h_output and its associated stream sync, since
// the modern path reads d_selected directly via gpu_filter_device_output()
// and never touches h_output.  gpu_filter_copy_prior_row_to_host() (called
// immediately after) provides the synchronization barrier that the modern
// path needs before using d_selected.
const uint8_t* gpu_filter_process_from_host(
    GpuFilterContext*       ctx,
    const uint8_t*          h_input,
    const uint8_t*          h_prior_row,
    int                     actual_rows,
    GpuTimings*             timings         = nullptr,
    const DicomPixelParams* dicom           = nullptr,
    bool                    device_output_only = false);

// Bytes in the output for a strip of actual_rows rows:
//   actual_rows * (width * bpp + 1)   (+1 for the per-row PNG filter byte)
size_t gpu_filter_output_size(const GpuFilterContext* ctx, int actual_rows);

// Device pointer to the same filtered strip data that gpu_filter_process_*
// already copied to ctx->h_output -- lets the modern (GPU-resident) pipeline
// feed the GPU deflate encoder directly without an extra H2D round trip.
// Valid only for the strip most recently processed by this context; call
// immediately after gpu_filter_process_from_host/_device, before the next call.
const uint8_t* gpu_filter_device_output(const GpuFilterContext* ctx);

// Copy this context's current prior-row buffer (the last preprocessed row,
// width_bytes long -- i.e. the same data PNG Up/Paeth filtering would use as
// "the row above" for the next strip) to host memory. h_dst must have room
// for width_bytes.
//
// Needed when round-robining strips across a POOL of independent
// GpuFilterContext instances (one stream pool slot per concurrently
// in-flight strip): each context's own internal prior-row carry only
// reflects strips THAT SAME context previously handled, which is wrong once
// strips are distributed round-robin across N contexts. The caller must
// explicitly thread the true previous-strip's last row through as the
// h_prior_row argument to the NEXT gpu_filter_process_from_host/_device call
// (on whichever pool context handles it), overriding that context's own
// (possibly stale, from a different strip several pool-slots back) carry.
void gpu_filter_copy_prior_row_to_host(const GpuFilterContext* ctx, uint8_t* h_dst);
