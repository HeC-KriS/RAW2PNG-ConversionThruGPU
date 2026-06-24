// gpu_adler32.cu
// See gpu_adler32.h for the design rationale.

#include "gpu_adler32.h"

#include <cuda_runtime.h>
#include <zlib.h>   // adler32_combine() -- reuse zlib's combine algebra verbatim
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(1);                                                         \
        }                                                                     \
    } while (0)

static const uint32_t ADLER_BASE = 65521u;  // largest prime < 2^16, per RFC 1950

struct GpuAdler32Context {
    size_t       max_input_bytes;
    size_t       chunk_bytes;
    int          max_blocks;
    uint32_t*    d_partial;       // [max_blocks] packed (B<<16)|A per sub-chunk
    uint32_t*    h_partial;       // pinned mirror
    cudaStream_t stream;
    // State saved by gpu_adler32_launch() for use by gpu_adler32_collect()
    size_t       pending_len        = 0;
    int          pending_num_blocks = 0;
};

// One thread per block computes its sub-chunk's Adler-32 serially (A/B are
// inherently sequential within a chunk); independent sub-chunks across
// blocks execute in parallel across the GPU's SMs.
__global__ void adler32_partial_kernel(const uint8_t* __restrict__ d_data,
                                        size_t len, size_t chunk_bytes,
                                        uint32_t* __restrict__ d_partial)
{
    const size_t blk   = blockIdx.x;
    const size_t start = blk * chunk_bytes;
    if (start >= len) { if (threadIdx.x == 0) d_partial[blk] = 1u; return; }  // adler32 of empty input = 1
    const size_t end = (start + chunk_bytes < len) ? (start + chunk_bytes) : len;

    if (threadIdx.x == 0) {
        uint32_t a = 1, b = 0;
        for (size_t i = start; i < end; i++) {
            a = (a + d_data[i]) % ADLER_BASE;
            b = (b + a)          % ADLER_BASE;
        }
        d_partial[blk] = (b << 16) | a;
    }
}

GpuAdler32Context* gpu_adler32_create(size_t max_input_bytes, size_t chunk_bytes)
{
    GpuAdler32Context* ctx = new GpuAdler32Context();
    ctx->max_input_bytes = max_input_bytes;
    ctx->chunk_bytes      = chunk_bytes;
    ctx->max_blocks       = (int)((max_input_bytes + chunk_bytes - 1) / chunk_bytes);
    ctx->max_blocks        = std::max(ctx->max_blocks, 1);

    CHECK_CUDA(cudaMalloc(&ctx->d_partial, sizeof(uint32_t) * ctx->max_blocks));
    CHECK_CUDA(cudaMallocHost(&ctx->h_partial, sizeof(uint32_t) * ctx->max_blocks));
    CHECK_CUDA(cudaStreamCreate(&ctx->stream));

    return ctx;
}

void gpu_adler32_destroy(GpuAdler32Context* ctx)
{
    if (!ctx) return;
    cudaFree(ctx->d_partial);
    cudaFreeHost(ctx->h_partial);
    cudaStreamDestroy(ctx->stream);
    delete ctx;
}

void gpu_adler32_launch(GpuAdler32Context* ctx, const uint8_t* d_data, size_t len)
{
    if (len == 0) {
        ctx->pending_len        = 0;
        ctx->pending_num_blocks = 0;
        return;
    }
    if (len > ctx->max_input_bytes) {
        fprintf(stderr,
            "gpu_adler32_launch: len %zu exceeds max_input_bytes %zu from gpu_adler32_create()\n",
            len, ctx->max_input_bytes);
        exit(1);
    }
    const int num_blocks = (int)((len + ctx->chunk_bytes - 1) / ctx->chunk_bytes);
    ctx->pending_len        = len;
    ctx->pending_num_blocks = num_blocks;

    adler32_partial_kernel<<<num_blocks, 1, 0, ctx->stream>>>(
        d_data, len, ctx->chunk_bytes, ctx->d_partial);
    CHECK_CUDA(cudaMemcpyAsync(ctx->h_partial, ctx->d_partial,
                               sizeof(uint32_t) * num_blocks,
                               cudaMemcpyDeviceToHost, ctx->stream));
}

uint32_t gpu_adler32_collect(GpuAdler32Context* ctx)
{
    if (ctx->pending_len == 0) return 1u;
    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));

    const size_t len        = ctx->pending_len;
    const int    num_blocks = ctx->pending_num_blocks;
    ctx->pending_len        = 0;
    ctx->pending_num_blocks = 0;

    uint32_t running = 0;
    for (int b = 0; b < num_blocks; b++) {
        const size_t start = (size_t)b * ctx->chunk_bytes;
        const size_t end   = std::min(start + ctx->chunk_bytes, len);
        const size_t blen  = end - start;
        if (b == 0) running = ctx->h_partial[0];
        else        running = (uint32_t)adler32_combine(running, ctx->h_partial[b], (z_off_t)blen);
    }
    return running;
}

uint32_t gpu_adler32_compute(GpuAdler32Context* ctx, const uint8_t* d_data, size_t len)
{
    gpu_adler32_launch(ctx, d_data, len);
    return gpu_adler32_collect(ctx);
}
