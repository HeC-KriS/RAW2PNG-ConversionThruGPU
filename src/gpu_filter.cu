// gpu_filter.cu
// PNG filter application + adaptive row-filter selection on SM 3.5 (GT 710).
//
// Two kernels:
//   1. dicom_preprocess_kernel  – DICOM pixel transforms (optional, in-place)
//   2. filter_select_kernel     – all 5 PNG filters scored in registers and
//                                 the winner written to d_selected in one pass.
//
// The three-kernel design (multi_filter / row_score / select_assemble) has been
// collapsed into filter_select_kernel.  This eliminates a 5× d_filtered global
// write and read, cutting DDR3 memory traffic by ~3× on the GT 710.  Each block
// handles one complete row; shared memory holds 5×256 uint32 score accumulators
// (5 KB per block, well within the 48 KB SM 3.5 limit).

#include "gpu_filter.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Context definition (opaque to callers)
// ---------------------------------------------------------------------------
struct GpuFilterContext {
    uint8_t*  d_input;      // [strip_height × width_bytes]
    uint8_t*  d_prior;      // [width_bytes]  – last row of previous strip
    uint8_t*  d_selected;   // [strip_height × (width_bytes + 1)]
    uint8_t*  h_output;     // pinned host mirror of d_selected

    int width;
    int bpp;
    int strip_height;
    int width_bytes;        // width * bpp

    cudaStream_t stream;

    // CUDA Events for per-call timing breakdown
    cudaEvent_t ev_start;
    cudaEvent_t ev_h2d_done;
    cudaEvent_t ev_kernels_done;
    cudaEvent_t ev_d2h_done;
};

// ---------------------------------------------------------------------------
// DICOM pixel preprocessing kernel (in-place on d_input)
//
// Transforms raw little-endian DICOM samples to big-endian PNG-ready samples:
//   1. bit-depth right-alignment (shift = HighBit − BitsStored + 1)
//   2. sign extension for pixel_rep=1
//   3. rescale slope / intercept (DICOM: applies to raw signed integer)
//   4. window / level (DICOM PS 3.3 C.7.6.3.1.5 exact formula)
//   5. byte-swap to big-endian (16-bit only)
//
// One thread per sample.  NOT __restrict__ on d_inout: single pointer,
// reads happen before writes for each thread (no intra-kernel aliasing).
// ---------------------------------------------------------------------------
__global__ void dicom_preprocess_kernel(
    uint8_t*     d_inout,
    int          total_samples,
    DicomPixelParams p)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_samples) return;

    const uint8_t* d_in  = d_inout;
    uint8_t*       d_out = d_inout;

    const int out_maxval = (p.bits_allocated == 16) ? 65535 : 255;
    float val;

    if (p.bits_allocated == 16) {
        // Read DICOM native (little-endian) 16-bit sample
        uint16_t raw = (uint16_t)d_in[idx * 2]
                     | ((uint16_t)d_in[idx * 2 + 1] << 8);

        // Right-align: shift = HighBit − BitsStored + 1
        //   Right-aligned (most scanners): HighBit=BitsStored-1 → shift=0 (no-op)
        //   Left-aligned  (rare):          HighBit=BitsAllocated-1 → shift=BitsAllocated-BitsStored
        int bit_shift = p.high_bit - p.bits_stored + 1;
        if (bit_shift > 0) raw = (uint16_t)(raw >> bit_shift);

        // Mask to exactly bits_stored bits — garbage in padding bits is real-world
        if (p.bits_stored < 16) {
            uint32_t mask = (1u << p.bits_stored) - 1u;
            raw = (uint16_t)(raw & (uint16_t)mask);
        }

        if (p.pixel_rep == 1) {
            // Two's complement sign extension; rescale applies to the signed value
            int32_t sv;
            if (p.bits_stored == 16) {
                sv = (int32_t)(int16_t)raw;
            } else {
                int bits = p.bits_stored;
                int half = 1 << (bits - 1);
                sv = (int32_t)raw;
                if (sv >= half) sv -= (1 << bits);
            }
            if (p.apply_rescale)
                val = (float)sv * p.rescale_slope + p.rescale_intercept;
            else
                val = (float)sv;
        } else {
            val = (float)raw;
            if (p.apply_rescale)
                val = val * p.rescale_slope + p.rescale_intercept;
        }
    } else {
        // 8-bit — respect pixel_rep for sign extension
        if (p.pixel_rep == 1)
            val = (float)(int8_t)d_in[idx];
        else
            val = (float)d_in[idx];
        if (p.apply_rescale)
            val = val * p.rescale_slope + p.rescale_intercept;
    }

    // DICOM PS 3.3 C.7.6.3.1.5 VOI window/level:
    //   if (x - c + 0.5) <= -(w-1)/2  →  y = 0
    //   if (x - c + 0.5) >   (w-1)/2  →  y = maxval
    //   else y = ((x - c + 0.5) / (w-1) + 0.5) * maxval
    if (p.apply_window) {
        float wm1 = p.window_width - 1.0f;
        float cen  = p.window_center;
        if (wm1 <= 0.0f) {
            val = (val >= cen) ? (float)out_maxval : 0.0f;
        } else {
            float lo = cen - wm1 * 0.5f - 0.5f;
            float hi = cen + wm1 * 0.5f - 0.5f;
            if (val <= lo)
                val = 0.0f;
            else if (val > hi)
                val = (float)out_maxval;
            else
                val = ((val - cen + 0.5f) / wm1 + 0.5f) * (float)out_maxval;
        }
    }

    val = fmaxf(0.f, fminf((float)out_maxval, val));
    uint32_t ov = (uint32_t)val;

    if (p.bits_allocated == 16) {
        d_out[idx * 2]     = (uint8_t)(ov >> 8);    // big-endian MSB
        d_out[idx * 2 + 1] = (uint8_t)(ov & 0xFF);  // big-endian LSB
    } else {
        d_out[idx] = (uint8_t)ov;
    }
}

// ---------------------------------------------------------------------------
// filter_select_kernel — fused multi-filter + score + select
//
// Grid : (actual_rows, 1)   Block: (256, 1)
//
// One block handles one complete row.  All 5 PNG filter values are computed
// in registers per byte, accumulated into per-thread score accumulators, then
// reduced across the block in shared memory.  The winning filter is then
// re-computed and written directly to d_selected — no d_filtered global buffer.
//
// Shared memory: 5 × 256 × 4 = 5120 bytes + 4 bytes = 5124 bytes per block.
// SM 3.5: 48 KB → 9 blocks max from shared; 2048 threads → 8 blocks max →
// effective occupancy: 8 blocks (100% thread occupancy, 50% block occupancy).
//
// Memory traffic vs. the old 3-kernel design:
//   Old: multi_filter writes 5× strip, row_score reads 5× strip,
//        select_assemble reads 1× strip and writes 1× strip  ≈ 12× strip_bytes
//   New: 2 passes over d_input (score + write), 1 write to d_selected ≈ 3× strip_bytes
// ---------------------------------------------------------------------------
__global__ void filter_select_kernel(
    const uint8_t* __restrict__ d_input,
    const uint8_t* __restrict__ d_prior,
    uint8_t*                    d_selected,
    int                         width_bytes,
    int                         bpp,
    int                         actual_rows)
{
    const int row = blockIdx.x;
    if (row >= actual_rows) return;

    // Pointer to the previous row (either d_prior for row 0 or d_input row-1).
    // row > 0 is constant across all threads in a block (no warp divergence);
    // the compiler hoists this pointer assignment out of both inner loops.
    const uint8_t* prev_in = (row > 0)
                           ? (d_input + (size_t)(row - 1) * width_bytes)
                           : d_prior;

    // -------------------------------------------------------------------------
    // Pass 1: compute per-thread partial score sums for all 5 PNG filters
    // -------------------------------------------------------------------------
    uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;

    for (int i = threadIdx.x; i < width_bytes; i += blockDim.x) {
        const uint8_t x = d_input[(size_t)row * width_bytes + i];
        const uint8_t a = (i >= bpp) ? d_input[(size_t)row * width_bytes + i - bpp] : 0;
        const uint8_t b = prev_in[i];
        const uint8_t c = (i >= bpp) ? prev_in[i - bpp] : 0;

        const int pv  = (int)a + (int)b - (int)c;
        const int pa  = abs(pv - (int)a);
        const int pb  = abs(pv - (int)b);
        const int pc  = abs(pv - (int)c);
        const uint8_t paeth = (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);

        // Compute all 5 candidates (uint8 arithmetic wraps mod 256 — PNG spec)
        const uint8_t f0 = x;
        const uint8_t f1 = x - a;
        const uint8_t f2 = x - b;
        const uint8_t f3 = x - (uint8_t)(((uint16_t)a + (uint16_t)b) >> 1);
        const uint8_t f4 = x - paeth;

        // Byte cost = |signed byte| = min(v, 256-v)
#define COST(v) ((v) <= 128u ? (uint32_t)(v) : (uint32_t)(256u - (v)))
        s0 += COST(f0);
        s1 += COST(f1);
        s2 += COST(f2);
        s3 += COST(f3);
        s4 += COST(f4);
#undef COST
    }

    // -------------------------------------------------------------------------
    // Block-wide reduction of the 5 per-thread partial scores
    // -------------------------------------------------------------------------
    __shared__ uint32_t sdata[5][256];
    sdata[0][threadIdx.x] = s0;
    sdata[1][threadIdx.x] = s1;
    sdata[2][threadIdx.x] = s2;
    sdata[3][threadIdx.x] = s3;
    sdata[4][threadIdx.x] = s4;
    __syncthreads();

#define REDUCE_STEP(half)                                               \
    if (threadIdx.x < (half)) {                                         \
        sdata[0][threadIdx.x] += sdata[0][threadIdx.x + (half)];       \
        sdata[1][threadIdx.x] += sdata[1][threadIdx.x + (half)];       \
        sdata[2][threadIdx.x] += sdata[2][threadIdx.x + (half)];       \
        sdata[3][threadIdx.x] += sdata[3][threadIdx.x + (half)];       \
        sdata[4][threadIdx.x] += sdata[4][threadIdx.x + (half)];       \
    } __syncthreads();

    REDUCE_STEP(128)
    REDUCE_STEP( 64)
    REDUCE_STEP( 32)
    REDUCE_STEP( 16)
    REDUCE_STEP(  8)
    REDUCE_STEP(  4)
    REDUCE_STEP(  2)
    REDUCE_STEP(  1)
#undef REDUCE_STEP

    // Thread 0 picks the minimum-score filter and writes the filter-type byte
    __shared__ int s_best;
    if (threadIdx.x == 0) {
        int      best = 0;
        uint32_t bsc  = sdata[0][0];
        if (sdata[1][0] < bsc) { bsc = sdata[1][0]; best = 1; }
        if (sdata[2][0] < bsc) { bsc = sdata[2][0]; best = 2; }
        if (sdata[3][0] < bsc) { bsc = sdata[3][0]; best = 3; }
        if (sdata[4][0] < bsc) {                    best = 4; }
        s_best = best;
        d_selected[row * (width_bytes + 1)] = (uint8_t)best;
    }
    __syncthreads();

    const int best_f = s_best;
    uint8_t* dst = d_selected + row * (width_bytes + 1) + 1;

    // -------------------------------------------------------------------------
    // Pass 2: recompute and write the winning filtered row
    // -------------------------------------------------------------------------
    for (int i = threadIdx.x; i < width_bytes; i += blockDim.x) {
        const uint8_t x = d_input[(size_t)row * width_bytes + i];
        const uint8_t a = (i >= bpp) ? d_input[(size_t)row * width_bytes + i - bpp] : 0;
        const uint8_t b = prev_in[i];

        uint8_t out_val;
        if (best_f == 0) {
            out_val = x;
        } else if (best_f == 1) {
            out_val = x - a;
        } else if (best_f == 2) {
            out_val = x - b;
        } else if (best_f == 3) {
            out_val = x - (uint8_t)(((uint16_t)a + (uint16_t)b) >> 1);
        } else {
            const uint8_t c = (i >= bpp) ? prev_in[i - bpp] : 0;
            const int pv = (int)a + (int)b - (int)c;
            const int pa = abs(pv - (int)a);
            const int pb = abs(pv - (int)b);
            const int pc = abs(pv - (int)c);
            out_val = x - ((pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c));
        }
        dst[i] = out_val;
    }
}

// ---------------------------------------------------------------------------
// Host-side API
// ---------------------------------------------------------------------------
GpuFilterContext* gpu_filter_create(int width, int bpp, int strip_height,
                                    bool device_output_only)
{
    GpuFilterContext* ctx = new GpuFilterContext();
    ctx->width        = width;
    ctx->bpp          = bpp;
    ctx->strip_height = strip_height;
    ctx->width_bytes  = width * bpp;

    const size_t strip_bytes = (size_t)strip_height * ctx->width_bytes;
    const size_t out_bytes   = (size_t)strip_height * (ctx->width_bytes + 1);

    CHECK_CUDA(cudaMalloc(&ctx->d_input,    strip_bytes));
    CHECK_CUDA(cudaMalloc(&ctx->d_prior,    ctx->width_bytes));
    CHECK_CUDA(cudaMalloc(&ctx->d_selected, out_bytes));
    // h_output is only needed by the legacy path that copies d_selected to host.
    // In the modern GPU-deflate path the compressed reads d_selected directly on
    // the device, so we skip the allocation to save pinned memory.
    if (!device_output_only)
        CHECK_CUDA(cudaMallocHost(&ctx->h_output, out_bytes));

    // Zero the prior-row buffer (PNG spec: first strip's prior row = zeros)
    CHECK_CUDA(cudaMemset(ctx->d_prior, 0, ctx->width_bytes));

    CHECK_CUDA(cudaStreamCreate(&ctx->stream));
    CHECK_CUDA(cudaEventCreate(&ctx->ev_start));
    CHECK_CUDA(cudaEventCreate(&ctx->ev_h2d_done));
    CHECK_CUDA(cudaEventCreate(&ctx->ev_kernels_done));
    CHECK_CUDA(cudaEventCreate(&ctx->ev_d2h_done));

    return ctx;
}

uint8_t* gpu_filter_alloc_pinned(size_t bytes)
{
    uint8_t* p = nullptr;
    CHECK_CUDA(cudaMallocHost(&p, bytes));
    return p;
}

void gpu_filter_free_pinned(uint8_t* ptr)
{
    if (ptr) cudaFreeHost(ptr);
}

void gpu_filter_destroy(GpuFilterContext* ctx)
{
    if (!ctx) return;
    cudaFree(ctx->d_input);
    cudaFree(ctx->d_prior);
    cudaFree(ctx->d_selected);
    if (ctx->h_output) cudaFreeHost(ctx->h_output);
    cudaEventDestroy(ctx->ev_start);
    cudaEventDestroy(ctx->ev_h2d_done);
    cudaEventDestroy(ctx->ev_kernels_done);
    cudaEventDestroy(ctx->ev_d2h_done);
    cudaStreamDestroy(ctx->stream);
    delete ctx;
}

size_t gpu_filter_output_size(const GpuFilterContext* ctx, int actual_rows)
{
    return (size_t)actual_rows * (ctx->width_bytes + 1);
}

const uint8_t* gpu_filter_device_output(const GpuFilterContext* ctx)
{
    return ctx->d_selected;
}

void gpu_filter_copy_prior_row_to_host(const GpuFilterContext* ctx, uint8_t* h_dst)
{
    CHECK_CUDA(cudaMemcpyAsync(h_dst, ctx->d_prior, ctx->width_bytes,
                               cudaMemcpyDeviceToHost, ctx->stream));
    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));
}

void gpu_filter_reset(GpuFilterContext* ctx)
{
    if (!ctx) return;
    // Zero the prior-row buffer so the first strip starts with zeros for its
    // Up/Paeth predictors (PNG spec). The subsequent H2D + kernel launch for
    // the first strip run in the same stream and will see the memset result
    // without a host sync.
    CHECK_CUDA(cudaMemsetAsync(ctx->d_prior, 0, ctx->width_bytes, ctx->stream));
}

// ---------------------------------------------------------------------------
// Internal: launch kernels, DMA result to pinned host, fill timings
// ---------------------------------------------------------------------------
static const uint8_t* run_kernels(GpuFilterContext* ctx, int actual_rows,
                                   GpuTimings* timings, const DicomPixelParams* dicom,
                                   bool skip_d2h = false)
{
    const int W = ctx->width_bytes;

    // Optional DICOM preprocessing (in-place on d_input, must precede filter kernel)
    if (dicom) {
        int total_samples = actual_rows * W / (dicom->bits_allocated / 8);
        dim3 dblk(256);
        dim3 dgrd((total_samples + 255) / 256);
        dicom_preprocess_kernel<<<dgrd, dblk, 0, ctx->stream>>>(
            ctx->d_input, total_samples, *dicom);
    }

    // Fused filter + select: one block per row
    {
        dim3 block(256, 1, 1);
        dim3 grid(actual_rows, 1, 1);
        filter_select_kernel<<<grid, block, 0, ctx->stream>>>(
            ctx->d_input, ctx->d_prior, ctx->d_selected,
            W, ctx->bpp, actual_rows);
    }

    // Save last row of d_input (preprocessed, BE) as prior row for the next strip
    CHECK_CUDA(cudaMemcpyAsync(
        ctx->d_prior,
        ctx->d_input + (size_t)(actual_rows - 1) * W,
        W,
        cudaMemcpyDeviceToDevice,
        ctx->stream));

    CHECK_CUDA(cudaEventRecord(ctx->ev_kernels_done, ctx->stream));

    if (!skip_d2h) {
        const size_t out_bytes = gpu_filter_output_size(ctx, actual_rows);
        CHECK_CUDA(cudaMemcpyAsync(
            ctx->h_output, ctx->d_selected,
            out_bytes,
            cudaMemcpyDeviceToHost,
            ctx->stream));
        CHECK_CUDA(cudaEventRecord(ctx->ev_d2h_done, ctx->stream));
        CHECK_CUDA(cudaStreamSynchronize(ctx->stream));
    } else {
        // Modern GPU-deflate path: d_selected is consumed directly on the
        // device.  Record ev_d2h_done immediately so elapsed-time queries
        // report 0 for d2h (accurate -- no transfer occurred).
        // Synchronization is provided by the gpu_filter_copy_prior_row_to_host()
        // call that follows in modern_filter_stage.
        CHECK_CUDA(cudaEventRecord(ctx->ev_d2h_done, ctx->stream));
    }

    if (timings) {
        cudaEventElapsedTime(&timings->h2d_ms,    ctx->ev_start,        ctx->ev_h2d_done);
        cudaEventElapsedTime(&timings->kernel_ms, ctx->ev_h2d_done,     ctx->ev_kernels_done);
        cudaEventElapsedTime(&timings->d2h_ms,    ctx->ev_kernels_done, ctx->ev_d2h_done);
    }

    return ctx->h_output;
}

const uint8_t* gpu_filter_process_from_device(
    GpuFilterContext*       ctx,
    const uint8_t*          d_input,
    const uint8_t*          d_prior_row,
    int                     actual_rows,
    GpuTimings*             timings,
    const DicomPixelParams* dicom)
{
    CHECK_CUDA(cudaEventRecord(ctx->ev_start, ctx->stream));
    const size_t strip_bytes = (size_t)actual_rows * ctx->width_bytes;
    CHECK_CUDA(cudaMemcpyAsync(ctx->d_input, d_input,
                               strip_bytes, cudaMemcpyDeviceToDevice, ctx->stream));
    if (d_prior_row)
        CHECK_CUDA(cudaMemcpyAsync(ctx->d_prior, d_prior_row,
                                   ctx->width_bytes, cudaMemcpyDeviceToDevice, ctx->stream));
    CHECK_CUDA(cudaEventRecord(ctx->ev_h2d_done, ctx->stream));
    return run_kernels(ctx, actual_rows, timings, dicom);
}

const uint8_t* gpu_filter_process_from_host(
    GpuFilterContext*       ctx,
    const uint8_t*          h_input,
    const uint8_t*          h_prior_row,
    int                     actual_rows,
    GpuTimings*             timings,
    const DicomPixelParams* dicom,
    bool                    device_output_only)
{
    CHECK_CUDA(cudaEventRecord(ctx->ev_start, ctx->stream));
    const size_t strip_bytes = (size_t)actual_rows * ctx->width_bytes;
    CHECK_CUDA(cudaMemcpyAsync(ctx->d_input, h_input,
                               strip_bytes, cudaMemcpyHostToDevice, ctx->stream));
    if (h_prior_row)
        CHECK_CUDA(cudaMemcpyAsync(ctx->d_prior, h_prior_row,
                                   ctx->width_bytes, cudaMemcpyHostToDevice, ctx->stream));
    CHECK_CUDA(cudaEventRecord(ctx->ev_h2d_done, ctx->stream));
    return run_kernels(ctx, actual_rows, timings, dicom, device_output_only);
}
