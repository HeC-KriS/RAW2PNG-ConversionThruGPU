// gpu_bmp_backend.cu
// GPU-resident DICOM 16-bit → 8-bit grayscale conversion for BMP export.
//
// Pipeline per file:
//   CPU decode (DCMTK) → H2D → bmp_dicom_to_gray8_kernel → D2H → CPU BMP write
//
// The kernel replicates the DICOM PS 3.3 C.7.6.3.1.5 VOI LUT formula already
// used by dicom_preprocess_kernel in gpu_filter.cu, but outputs 8-bit grayscale
// (for BMP) instead of 16-bit big-endian (for PNG).  Separate file keeps BMP
// and PNG GPU paths independent so each can be tuned without risk to the other.

#include "gpu_bmp_backend.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t _e = (call);                                                  \
        if (_e != cudaSuccess) {                                                  \
            fprintf(stderr, "[gpu_bmp] CUDA error %s:%d  %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                   \
            return false;                                                         \
        }                                                                         \
    } while (0)

#define CHECK_CUDA_CTX(call)                                                      \
    do {                                                                          \
        cudaError_t _e = (call);                                                  \
        if (_e != cudaSuccess) {                                                  \
            fprintf(stderr, "[gpu_bmp] CUDA error %s:%d  %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                   \
        }                                                                         \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel: DICOM 16-bit LE → 8-bit grayscale
//
// Steps (identical to dicom_preprocess_kernel in gpu_filter.cu):
//   1. Right-align: shift raw by (HighBit − BitsStored + 1)
//   2. Mask to bits_stored bits
//   3. Sign-extend if pixel_rep == 1
//   4. Apply rescale slope/intercept if apply_rescale
//   5. Apply VOI window/level if apply_window, else full-range normalize
//   6. Clamp to [0, 255] and write uint8_t
//
// One thread per pixel. Grid: ceil(n_pixels/256) × 1.
// ---------------------------------------------------------------------------
__global__ void bmp_dicom_to_gray8_kernel(
    const uint16_t* __restrict__ d_in,
    uint8_t*        __restrict__ d_out,
    int                          n_pixels,
    DicomPixelParams             p)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pixels) return;

    uint16_t raw = d_in[idx];

    // Step 1: right-align (most CT scanners: high_bit = bits_stored-1 → shift=0)
    const int bit_shift = p.high_bit - p.bits_stored + 1;
    if (bit_shift > 0) raw = (uint16_t)(raw >> bit_shift);

    // Step 2: mask to bits_stored (removes garbage in unused high bits)
    if (p.bits_stored < 16) {
        const uint32_t mask = (1u << p.bits_stored) - 1u;
        raw = (uint16_t)(raw & (uint16_t)mask);
    }

    // Step 3: convert to float with optional sign extension
    float val;
    if (p.pixel_rep == 1) {
        int32_t sv;
        if (p.bits_stored == 16) {
            sv = (int32_t)(int16_t)raw;
        } else {
            const int half = 1 << (p.bits_stored - 1);
            sv = (int32_t)(uint32_t)raw;
            if (sv >= half) sv -= (1 << p.bits_stored);
        }
        val = (float)sv;
    } else {
        val = (float)(uint32_t)raw;
    }

    // Step 4: rescale slope/intercept
    if (p.apply_rescale)
        val = val * p.rescale_slope + p.rescale_intercept;

    // Step 5: VOI window/level (DICOM PS 3.3 C.7.6.3.1.5) or full-range norm
    if (p.apply_window) {
        const float wm1 = p.window_width - 1.0f;
        const float cen = p.window_center;
        if (wm1 <= 0.0f) {
            val = (val >= cen) ? 255.0f : 0.0f;
        } else {
            const float lo = cen - wm1 * 0.5f - 0.5f;
            const float hi = cen + wm1 * 0.5f - 0.5f;
            if (val <= lo)
                val = 0.0f;
            else if (val > hi)
                val = 255.0f;
            else
                val = ((val - cen + 0.5f) / wm1 + 0.5f) * 255.0f;
        }
    } else {
        // No window tag: map full stored-value range to [0, 255]
        const float range = (float)(1u << p.bits_stored);
        if (p.pixel_rep == 1) {
            const float half = range * 0.5f;
            val = (val + half) / range * 255.0f;
        } else {
            val = val / (range - 1.0f) * 255.0f;
        }
    }

    // Step 6: clamp and write
    d_out[idx] = (uint8_t)__float2int_rn(fminf(255.0f, fmaxf(0.0f, val)));
}

// ---------------------------------------------------------------------------
// Context implementation
// ---------------------------------------------------------------------------
struct GpuBmpContext {
    uint16_t*    d_in16   = nullptr;  // device input:  16-bit LE pixels
    uint8_t*     d_out8   = nullptr;  // device output: 8-bit grayscale
    uint8_t*     h_out8   = nullptr;  // pinned host output (returned to caller)
    cudaStream_t stream   = nullptr;
    cudaEvent_t  ev_start = nullptr;
    cudaEvent_t  ev_h2d   = nullptr;
    cudaEvent_t  ev_kern  = nullptr;
    cudaEvent_t  ev_d2h   = nullptr;
    int          cap_px   = 0;       // allocated capacity in pixels

    float last_h2d_us    = 0.f;
    float last_kernel_us = 0.f;
    float last_d2h_us    = 0.f;
};

static void bmp_free_buffers(GpuBmpContext* ctx)
{
    if (ctx->d_in16) { cudaFree(ctx->d_in16);       ctx->d_in16 = nullptr; }
    if (ctx->d_out8) { cudaFree(ctx->d_out8);       ctx->d_out8 = nullptr; }
    if (ctx->h_out8) { cudaFreeHost(ctx->h_out8);   ctx->h_out8 = nullptr; }
    ctx->cap_px = 0;
}

static bool bmp_alloc_buffers(GpuBmpContext* ctx, int n_pixels)
{
    bmp_free_buffers(ctx);
    cudaError_t e;
    e = cudaMalloc(&ctx->d_in16, (size_t)n_pixels * sizeof(uint16_t));
    if (e != cudaSuccess) {
        fprintf(stderr, "[gpu_bmp] cudaMalloc d_in16 failed: %s\n", cudaGetErrorString(e));
        return false;
    }
    e = cudaMalloc(&ctx->d_out8, (size_t)n_pixels * sizeof(uint8_t));
    if (e != cudaSuccess) {
        fprintf(stderr, "[gpu_bmp] cudaMalloc d_out8 failed: %s\n", cudaGetErrorString(e));
        cudaFree(ctx->d_in16); ctx->d_in16 = nullptr;
        return false;
    }
    e = cudaMallocHost(&ctx->h_out8, (size_t)n_pixels * sizeof(uint8_t));
    if (e != cudaSuccess) {
        fprintf(stderr, "[gpu_bmp] cudaMallocHost h_out8 failed: %s\n", cudaGetErrorString(e));
        cudaFree(ctx->d_in16); ctx->d_in16 = nullptr;
        cudaFree(ctx->d_out8); ctx->d_out8 = nullptr;
        return false;
    }
    ctx->cap_px = n_pixels;
    return true;
}

GpuBmpContext* gpu_bmp_create(int initial_pixels)
{
    GpuBmpContext* ctx = new GpuBmpContext();

    cudaError_t e;
    e = cudaStreamCreate(&ctx->stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "[gpu_bmp] cudaStreamCreate failed: %s\n", cudaGetErrorString(e));
        delete ctx; return nullptr;
    }
    e = cudaEventCreate(&ctx->ev_start);
    if (e != cudaSuccess) { cudaStreamDestroy(ctx->stream); delete ctx; return nullptr; }
    e = cudaEventCreate(&ctx->ev_h2d);
    if (e != cudaSuccess) { cudaEventDestroy(ctx->ev_start); cudaStreamDestroy(ctx->stream); delete ctx; return nullptr; }
    e = cudaEventCreate(&ctx->ev_kern);
    if (e != cudaSuccess) { cudaEventDestroy(ctx->ev_start); cudaEventDestroy(ctx->ev_h2d); cudaStreamDestroy(ctx->stream); delete ctx; return nullptr; }
    e = cudaEventCreate(&ctx->ev_d2h);
    if (e != cudaSuccess) { cudaEventDestroy(ctx->ev_start); cudaEventDestroy(ctx->ev_h2d); cudaEventDestroy(ctx->ev_kern); cudaStreamDestroy(ctx->stream); delete ctx; return nullptr; }

    if (initial_pixels > 0 && !bmp_alloc_buffers(ctx, initial_pixels)) {
        // Non-fatal: will allocate on first use
    }
    return ctx;
}

void gpu_bmp_destroy(GpuBmpContext* ctx)
{
    if (!ctx) return;
    if (ctx->stream) cudaStreamSynchronize(ctx->stream);
    bmp_free_buffers(ctx);
    if (ctx->ev_d2h)   cudaEventDestroy(ctx->ev_d2h);
    if (ctx->ev_kern)  cudaEventDestroy(ctx->ev_kern);
    if (ctx->ev_h2d)   cudaEventDestroy(ctx->ev_h2d);
    if (ctx->ev_start) cudaEventDestroy(ctx->ev_start);
    if (ctx->stream)   cudaStreamDestroy(ctx->stream);
    delete ctx;
}

const uint8_t* gpu_bmp_process(GpuBmpContext*         ctx,
                               const uint16_t*         h_in16,
                               int                     width,
                               int                     height,
                               const DicomPixelParams& params)
{
    if (!ctx || !h_in16 || width <= 0 || height <= 0) return nullptr;

    const int n = width * height;

    // (Re-)allocate device/host buffers if image is larger than current capacity
    if (n > ctx->cap_px) {
        if (!bmp_alloc_buffers(ctx, n)) return nullptr;
    }

    const size_t in_bytes  = (size_t)n * sizeof(uint16_t);
    const size_t out_bytes = (size_t)n * sizeof(uint8_t);

    // ── Timing start ─────────────────────────────────────────────────────────
    cudaEventRecord(ctx->ev_start, ctx->stream);

    // ── H2D (synchronous from non-pinned host memory) ────────────────────────
    // cudaMemcpy from pageable host → device is synchronous by definition;
    // using the stream variant here so the subsequent kernel launch on the
    // same stream is ordered correctly.
    cudaError_t e = cudaMemcpyAsync(ctx->d_in16, h_in16, in_bytes,
                                    cudaMemcpyHostToDevice, ctx->stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "[gpu_bmp] H2D failed: %s\n", cudaGetErrorString(e));
        return nullptr;
    }
    cudaEventRecord(ctx->ev_h2d, ctx->stream);

    // ── Kernel ────────────────────────────────────────────────────────────────
    const int block = 256;
    const int grid  = (n + block - 1) / block;
    bmp_dicom_to_gray8_kernel<<<grid, block, 0, ctx->stream>>>(
        ctx->d_in16, ctx->d_out8, n, params);
    cudaEventRecord(ctx->ev_kern, ctx->stream);

    // ── D2H → pinned host output buffer (async) ───────────────────────────
    e = cudaMemcpyAsync(ctx->h_out8, ctx->d_out8, out_bytes,
                        cudaMemcpyDeviceToHost, ctx->stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "[gpu_bmp] D2H failed: %s\n", cudaGetErrorString(e));
        return nullptr;
    }
    cudaEventRecord(ctx->ev_d2h, ctx->stream);

    // ── Synchronize ──────────────────────────────────────────────────────────
    e = cudaStreamSynchronize(ctx->stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "[gpu_bmp] stream sync failed: %s\n", cudaGetErrorString(e));
        return nullptr;
    }

    // ── Collect timings ───────────────────────────────────────────────────────
    float h2d_ms = 0.f, kern_ms = 0.f, d2h_ms = 0.f;
    cudaEventElapsedTime(&h2d_ms,  ctx->ev_start, ctx->ev_h2d);
    cudaEventElapsedTime(&kern_ms, ctx->ev_h2d,   ctx->ev_kern);
    cudaEventElapsedTime(&d2h_ms,  ctx->ev_kern,  ctx->ev_d2h);
    ctx->last_h2d_us    = h2d_ms  * 1000.f;
    ctx->last_kernel_us = kern_ms * 1000.f;
    ctx->last_d2h_us    = d2h_ms  * 1000.f;

    return ctx->h_out8;  // valid until next gpu_bmp_process call on this context
}

GpuBmpTimings gpu_bmp_get_timings(const GpuBmpContext* ctx)
{
    GpuBmpTimings t;
    if (ctx) {
        t.h2d_us    = ctx->last_h2d_us;
        t.kernel_us = ctx->last_kernel_us;
        t.d2h_us    = ctx->last_d2h_us;
    }
    return t;
}

size_t gpu_bmp_vram_bytes(const GpuBmpContext* ctx)
{
    if (!ctx || ctx->cap_px == 0) return 0;
    return (size_t)ctx->cap_px * (sizeof(uint16_t) + sizeof(uint8_t));
}

void gpu_bmp_warmup_thread()
{
    // Force CUDA context creation on the calling thread by performing a
    // trivial allocation + free. This pays the one-time ~50-200 ms CUDA
    // driver init cost up front rather than on the first processed file.
    void* d = nullptr;
    cudaMalloc(&d, 256);
    if (d) cudaFree(d);
}
