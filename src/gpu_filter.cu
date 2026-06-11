// gpu_filter.cu
// PNG filter application + adaptive row-filter selection on SM 3.5 (GT 710).
//
// Three kernels:
//   1. multi_filter_kernel  – computes all 5 PNG filters for every byte of
//                             every row in the strip, in one pass.
//   2. row_score_kernel     – for each (filter, row) pair sums |signed(byte)|
//                             using a shared-memory tree reduction.
//   3. select_assemble_kernel – picks the minimum-score filter per row and
//                             writes [filter_byte | filtered_row] to output.
//
// All kernels require only global memory, shared memory, and __ldg().
// No cooperative groups, no tensor cores, no warp-level primitives beyond
// what SM 3.0+ guarantees.  Safe on SM 3.5 with CUDA 11.8.

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
    uint8_t*  d_filtered;   // [5 × strip_height × width_bytes]
    uint32_t* d_scores;     // [5 × strip_height]
    uint8_t*  d_selected;   // [strip_height × (width_bytes + 1)]
    uint8_t*  h_output;     // pinned host mirror of d_selected

    int width;
    int bpp;
    int strip_height;
    int width_bytes;        // width * bpp

    cudaStream_t stream;

    // CUDA Events for per-call timing breakdown
    cudaEvent_t ev_start;        // before H2D upload
    cudaEvent_t ev_h2d_done;     // after H2D, before kernels
    cudaEvent_t ev_kernels_done; // after all kernels + prior-row save
    cudaEvent_t ev_d2h_done;     // after D2H download (= stream sync point)
};

// ---------------------------------------------------------------------------
// Kernel 1 – multi_filter_kernel
//
// Grid : (ceil(width_bytes/256), actual_rows, 1)
// Block: (256, 1, 1)
//
// Indexing inside d_filtered: filter_id * strip_height * width_bytes + row *
// width_bytes + byte_idx.  The strip_height dimension is used for indexing
// even when actual_rows < strip_height; unused rows are never written.
// ---------------------------------------------------------------------------
__global__ void multi_filter_kernel(
    const uint8_t* __restrict__ d_input,
    const uint8_t* __restrict__ d_prior,
    uint8_t*                    d_filtered,
    int                         width_bytes,
    int                         bpp,
    int                         strip_height,  // allocated height (for strides)
    int                         actual_rows)
{
    const int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_idx  = blockIdx.y;
    if (byte_idx >= width_bytes || row_idx >= actual_rows) return;

    const int offset = row_idx * width_bytes + byte_idx;

    const uint8_t x = d_input[offset];

    // a = Sub predecessor  (left neighbour, same row)
    const uint8_t a = (byte_idx >= bpp) ? d_input[offset - bpp] : 0;

    // b = Up predecessor   (same column, previous row)
    // Use __ldg for d_prior (read-only, SM 3.5 supports the L1 read-only path)
    const uint8_t b = (row_idx == 0)
        ? __ldg(d_prior + byte_idx)
        : d_input[(row_idx - 1) * width_bytes + byte_idx];

    // c = Average/Paeth diagonal (previous row, left neighbour)
    const uint8_t c = (row_idx == 0)
        ? ((byte_idx >= bpp) ? __ldg(d_prior + byte_idx - bpp) : 0)
        : ((byte_idx >= bpp) ? d_input[(row_idx - 1) * width_bytes + byte_idx - bpp] : 0);

    // Paeth predictor (PNG spec section 9.4)
    const int p  = (int)a + (int)b - (int)c;
    const int pa = abs(p - (int)a);
    const int pb = abs(p - (int)b);
    const int pc = abs(p - (int)c);
    const uint8_t paeth = (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);

    // Write all 5 candidates.  uint8_t arithmetic wraps mod 256 – correct per PNG spec.
    const int base = strip_height * width_bytes;
    d_filtered[0 * base + offset] = x;          // None
    d_filtered[1 * base + offset] = x - a;      // Sub
    d_filtered[2 * base + offset] = x - b;      // Up
    d_filtered[3 * base + offset] = x - (uint8_t)(((uint16_t)a + (uint16_t)b) >> 1); // Average
    d_filtered[4 * base + offset] = x - paeth;  // Paeth
}

// ---------------------------------------------------------------------------
// Kernel 2 – row_score_kernel
//
// Grid : (actual_rows, 5, 1)
// Block: (256, 1, 1)
//
// Computes sum_of_abs_signed for one (row, filter) pair using a standard
// shared-memory tree reduction.  No warp primitives – works on any SM >= 2.0.
// ---------------------------------------------------------------------------
__global__ void row_score_kernel(
    const uint8_t* __restrict__ d_filtered,
    uint32_t*                   d_scores,
    int                         width_bytes,
    int                         strip_height,
    int                         actual_rows)
{
    const int row_idx    = blockIdx.x;
    const int filter_id  = blockIdx.y;
    if (row_idx >= actual_rows) return;

    const int base   = filter_id * strip_height * width_bytes;
    const uint8_t* row = d_filtered + base + row_idx * width_bytes;

    __shared__ uint32_t sdata[256];

    uint32_t local_sum = 0;
    for (int i = threadIdx.x; i < width_bytes; i += 256) {
        // |signed(byte)| = min(byte, 256 - byte)
        uint8_t b = row[i];
        local_sum += (b <= 128u) ? (uint32_t)b : (uint32_t)(256u - b);
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Tree reduction – explicit to avoid any __syncthreads inside unrolled warp
    if (threadIdx.x < 128) sdata[threadIdx.x] += sdata[threadIdx.x + 128]; __syncthreads();
    if (threadIdx.x <  64) sdata[threadIdx.x] += sdata[threadIdx.x +  64]; __syncthreads();
    if (threadIdx.x <  32) sdata[threadIdx.x] += sdata[threadIdx.x +  32]; __syncthreads();
    if (threadIdx.x <  16) sdata[threadIdx.x] += sdata[threadIdx.x +  16]; __syncthreads();
    if (threadIdx.x <   8) sdata[threadIdx.x] += sdata[threadIdx.x +   8]; __syncthreads();
    if (threadIdx.x <   4) sdata[threadIdx.x] += sdata[threadIdx.x +   4]; __syncthreads();
    if (threadIdx.x <   2) sdata[threadIdx.x] += sdata[threadIdx.x +   2]; __syncthreads();
    if (threadIdx.x <   1) sdata[threadIdx.x] += sdata[threadIdx.x +   1]; __syncthreads();

    if (threadIdx.x == 0)
        d_scores[filter_id * strip_height + row_idx] = sdata[0];
}

// ---------------------------------------------------------------------------
// Kernel 3 – select_assemble_kernel
//
// Grid : (actual_rows, 1, 1)
// Block: (256, 1, 1)
//
// Picks the minimum-score filter per row then parallel-copies the selected
// filtered row + filter byte into d_selected.
// ---------------------------------------------------------------------------
__global__ void select_assemble_kernel(
    const uint8_t*  __restrict__ d_filtered,
    const uint32_t* __restrict__ d_scores,
    uint8_t*                     d_selected,
    int                          width_bytes,
    int                          strip_height,
    int                          actual_rows)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= actual_rows) return;

    __shared__ int s_best;

    // Thread 0 performs the 5-way minimum
    if (threadIdx.x == 0) {
        int    best  = 0;
        uint32_t bsc = d_scores[row_idx];  // filter 0
        for (int f = 1; f < 5; f++) {
            uint32_t s = d_scores[f * strip_height + row_idx];
            if (s < bsc) { bsc = s; best = f; }
        }
        s_best = best;
        // Write filter-type byte at head of output row
        d_selected[row_idx * (width_bytes + 1)] = (uint8_t)best;
    }
    __syncthreads();

    const int best_f = s_best;
    const uint8_t* src = d_filtered + best_f * strip_height * width_bytes
                                    + row_idx * width_bytes;
    uint8_t*       dst = d_selected + row_idx * (width_bytes + 1) + 1;

    for (int i = threadIdx.x; i < width_bytes; i += blockDim.x)
        dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// Host-side API
// ---------------------------------------------------------------------------
GpuFilterContext* gpu_filter_create(int width, int bpp, int strip_height)
{
    GpuFilterContext* ctx = new GpuFilterContext();
    ctx->width        = width;
    ctx->bpp          = bpp;
    ctx->strip_height = strip_height;
    ctx->width_bytes  = width * bpp;

    const size_t strip_bytes = (size_t)strip_height * ctx->width_bytes;

    CHECK_CUDA(cudaMalloc(&ctx->d_input,    strip_bytes));
    CHECK_CUDA(cudaMalloc(&ctx->d_prior,    ctx->width_bytes));
    CHECK_CUDA(cudaMalloc(&ctx->d_filtered, 5 * strip_bytes));
    CHECK_CUDA(cudaMalloc(&ctx->d_scores,   5 * strip_height * sizeof(uint32_t)));

    const size_t out_bytes = (size_t)strip_height * (ctx->width_bytes + 1);
    CHECK_CUDA(cudaMalloc(&ctx->d_selected, out_bytes));
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

void gpu_filter_destroy(GpuFilterContext* ctx)
{
    if (!ctx) return;
    cudaFree(ctx->d_input);
    cudaFree(ctx->d_prior);
    cudaFree(ctx->d_filtered);
    cudaFree(ctx->d_scores);
    cudaFree(ctx->d_selected);
    cudaFreeHost(ctx->h_output);
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

// ---------------------------------------------------------------------------
// DICOM pixel preprocessing kernel
//
// Runs in-place on d_input before the PNG filter kernels.
// Transforms raw little-endian DICOM samples to big-endian PNG-ready samples:
//   1. bit-depth normalisation  (shift away unused high bits)
//   2. signed → unsigned offset binary (if pixel_rep == 1)
//   3. rescale slope / intercept
//   4. window / level clamp
//   5. byte-swap to big-endian (16-bit only)
//
// One thread handles one sample (one colour channel of one pixel).
// Grid: ceil(total_samples / 256) × 1   Block: 256 × 1
// ---------------------------------------------------------------------------
// In-place kernel: reads and writes the same buffer (d_inout).
// NOT __restrict__ on d_inout because d_in and d_out are the same pointer;
// using __restrict__ on an aliased pointer is undefined behaviour.
__global__ void dicom_preprocess_kernel(
    uint8_t*     d_inout,
    int          total_samples,
    DicomPixelParams p)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_samples) return;

    // Local aliases for clarity; all reads must precede writes for this thread.
    const uint8_t* d_in  = d_inout;
    uint8_t*       d_out = d_inout;

    const int out_maxval = (p.bits_allocated == 16) ? 65535 : 255;
    float val;

    if (p.bits_allocated == 16) {
        // Read DICOM native (little-endian) 16-bit sample
        uint16_t raw = (uint16_t)d_in[idx * 2]
                     | ((uint16_t)d_in[idx * 2 + 1] << 8);

        // Right-align: shift = HighBit − BitsStored + 1
        //   Right-aligned (common): HighBit = BitsStored-1 → shift = 0
        //   Left-aligned  (rare):   HighBit = BitsAllocated-1 → shift = BitsAllocated-BitsStored
        int bit_shift = p.high_bit - p.bits_stored + 1;
        if (bit_shift > 0) raw = (uint16_t)(raw >> bit_shift);

        // Mask to exactly bits_stored bits (defensive: DICOM spec requires unused bits = 0
        // but real-world files occasionally have garbage in the padding bits)
        if (p.bits_stored < 16) {
            uint32_t mask = (1u << p.bits_stored) - 1u;
            raw = (uint16_t)(raw & (uint16_t)mask);
        }

        if (p.pixel_rep == 1) {
            // Two's complement sign extension, then apply rescale to the signed value.
            // Rescale slope/intercept in DICOM always applies to the raw signed integer,
            // NOT to an unsigned-shifted representation.
            int32_t sv;
            if (p.bits_stored == 16) {
                sv = (int32_t)(int16_t)raw;
            } else {
                int bits = p.bits_stored;
                int half = 1 << (bits - 1);
                sv = (int32_t)(raw);
                if (sv >= half) sv -= (1 << bits);  // two's complement → signed int
            }
            // Apply rescale to signed value, then shift to unsigned for window/output
            if (p.apply_rescale)
                val = (float)sv * p.rescale_slope + p.rescale_intercept;
            else
                val = (float)sv;
        } else {
            // Unsigned: rescale applies directly to the unsigned integer
            val = (float)raw;
            if (p.apply_rescale)
                val = val * p.rescale_slope + p.rescale_intercept;
        }
    } else {
        // 8-bit: single unsigned byte per sample
        val = (float)d_in[idx];
        if (p.apply_rescale)
            val = val * p.rescale_slope + p.rescale_intercept;
    }

    if (p.apply_window) {
        float lo = p.window_center - p.window_width * 0.5f;
        val = (val - lo) / p.window_width * (float)out_maxval;
    }

    val = fmaxf(0.f, fminf((float)out_maxval, val));
    uint32_t ov = (uint32_t)val;

    if (p.bits_allocated == 16) {
        // Write big-endian (PNG / network byte order)
        d_out[idx * 2]     = (uint8_t)(ov >> 8);
        d_out[idx * 2 + 1] = (uint8_t)(ov & 0xFF);
    } else {
        d_out[idx] = (uint8_t)ov;
    }
}

// ---------------------------------------------------------------------------
// Internal: run kernels, DMA result to pinned host memory, fill timings.
// ev_h2d_done must already be recorded by the caller.
// ---------------------------------------------------------------------------
static const uint8_t* run_kernels(GpuFilterContext* ctx, int actual_rows,
                                   GpuTimings* timings, const DicomPixelParams* dicom)
{
    const int W  = ctx->width_bytes;
    const int SH = ctx->strip_height;

    // Optional DICOM pixel preprocessing (in-place on d_input)
    // Must run before multi_filter_kernel; stream ordering guarantees sequencing.
    if (dicom) {
        // total_samples = number of individual colour-channel values in the strip
        int total_samples = actual_rows * W / (dicom->bits_allocated / 8);
        dim3 dblk(256);
        dim3 dgrd((total_samples + 255) / 256);
        dicom_preprocess_kernel<<<dgrd, dblk, 0, ctx->stream>>>(
            ctx->d_input, total_samples, *dicom);
    }

    // Kernel 1 – multi_filter
    {
        dim3 block(256, 1, 1);
        dim3 grid((W + 255) / 256, actual_rows, 1);
        multi_filter_kernel<<<grid, block, 0, ctx->stream>>>(
            ctx->d_input, ctx->d_prior, ctx->d_filtered,
            W, ctx->bpp, SH, actual_rows);
    }

    // Kernel 2 – row_score  (one block per (row, filter) pair)
    {
        dim3 block(256, 1, 1);
        dim3 grid(actual_rows, 5, 1);
        row_score_kernel<<<grid, block, 0, ctx->stream>>>(
            ctx->d_filtered, ctx->d_scores, W, SH, actual_rows);
    }

    // Kernel 3 – select + assemble
    {
        dim3 block(256, 1, 1);
        dim3 grid(actual_rows, 1, 1);
        select_assemble_kernel<<<grid, block, 0, ctx->stream>>>(
            ctx->d_filtered, ctx->d_scores, ctx->d_selected,
            W, SH, actual_rows);
    }

    // Save the last row of this strip as the prior row for the next strip
    CHECK_CUDA(cudaMemcpyAsync(
        ctx->d_prior,
        ctx->d_input + (size_t)(actual_rows - 1) * W,
        W,
        cudaMemcpyDeviceToDevice,
        ctx->stream));

    // Mark end of kernel phase
    CHECK_CUDA(cudaEventRecord(ctx->ev_kernels_done, ctx->stream));

    // DMA result to pinned host buffer
    const size_t out_bytes = gpu_filter_output_size(ctx, actual_rows);
    CHECK_CUDA(cudaMemcpyAsync(
        ctx->h_output, ctx->d_selected,
        out_bytes,
        cudaMemcpyDeviceToHost,
        ctx->stream));

    // Mark end of D2H phase
    CHECK_CUDA(cudaEventRecord(ctx->ev_d2h_done, ctx->stream));

    // Synchronise – caller receives valid data immediately on return
    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));

    // Fill timing breakdown if requested
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
    const DicomPixelParams* dicom)
{
    CHECK_CUDA(cudaEventRecord(ctx->ev_start, ctx->stream));
    const size_t strip_bytes = (size_t)actual_rows * ctx->width_bytes;
    CHECK_CUDA(cudaMemcpyAsync(ctx->d_input, h_input,
                               strip_bytes, cudaMemcpyHostToDevice, ctx->stream));
    if (h_prior_row)
        CHECK_CUDA(cudaMemcpyAsync(ctx->d_prior, h_prior_row,
                                   ctx->width_bytes, cudaMemcpyHostToDevice, ctx->stream));
    CHECK_CUDA(cudaEventRecord(ctx->ev_h2d_done, ctx->stream));
    return run_kernels(ctx, actual_rows, timings, dicom);
}
