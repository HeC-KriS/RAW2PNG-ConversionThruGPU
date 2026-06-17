// gpu_crc32.cu
// See gpu_crc32.h for the design rationale (block-parallel partial CRCs,
// combined via zlib's crc32_combine() for bit-identical results to the
// existing CPU path).
//
// NOTE ON VERIFICATION: this was written without access to SM >= 7.0
// hardware in the authoring environment. The CRC-32 table is generated at
// runtime from the standard reflected-polynomial algorithm (PNG spec Annex
// D / zlib's crc32.c) rather than hand-transcribed, specifically to avoid an
// unverifiable transcription error silently corrupting every PNG checksum.
// Before relying on this in production, verify gpu_crc32_compute() against
// zlib's crc32() on a known buffer on the actual RTX 5050 target.

#include "gpu_crc32.h"

#include <cuda_runtime.h>
#include <zlib.h>   // crc32_combine() -- reuse zlib's GF(2) combine algebra verbatim
#include <cstdio>
#include <cstdlib>
#include <vector>
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

__device__ __constant__ uint32_t kCrc32Table[256];

// Standard reflected CRC-32 table generator (polynomial 0xEDB88320), per the
// PNG specification Annex D. Generated on the host once, then uploaded to
// the device __constant__ array -- avoids hand-transcribing 256 magic
// numbers, which would be unverifiable without hardware to test against.
static void build_crc32_table_host(uint32_t table[256])
{
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++)
            c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        table[n] = c;
    }
}

struct GpuCrc32Context {
    size_t        max_input_bytes;
    size_t        chunk_bytes;
    int           max_blocks;
    uint32_t*     d_partial;   // [max_blocks] device scratch
    uint32_t*     h_partial;   // [max_blocks] pinned host mirror
    cudaStream_t  stream;
};

// One thread per block computes its sub-chunk's CRC-32 serially via table
// lookup (CRC is inherently sequential within a sub-chunk); independent
// sub-chunks across blocks execute in parallel across the GPU's SMs.
__global__ void crc32_partial_kernel(const uint8_t* __restrict__ d_data,
                                      size_t len, size_t chunk_bytes,
                                      uint32_t* __restrict__ d_partial)
{
    const size_t blk   = blockIdx.x;
    const size_t start = blk * chunk_bytes;
    if (start >= len) { if (threadIdx.x == 0) d_partial[blk] = 0; return; }
    const size_t end = (start + chunk_bytes < len) ? (start + chunk_bytes) : len;

    if (threadIdx.x == 0) {
        uint32_t crc = 0xFFFFFFFFu;
        for (size_t i = start; i < end; i++)
            crc = kCrc32Table[(crc ^ d_data[i]) & 0xFFu] ^ (crc >> 8);
        d_partial[blk] = crc ^ 0xFFFFFFFFu;
    }
}

GpuCrc32Context* gpu_crc32_create(size_t max_input_bytes, size_t chunk_bytes)
{
    static bool table_uploaded = false;
    if (!table_uploaded) {
        uint32_t h_table[256];
        build_crc32_table_host(h_table);
        CHECK_CUDA(cudaMemcpyToSymbol(kCrc32Table, h_table, sizeof(h_table)));
        table_uploaded = true;
    }

    GpuCrc32Context* ctx = new GpuCrc32Context();
    ctx->max_input_bytes = max_input_bytes;
    ctx->chunk_bytes      = chunk_bytes;
    ctx->max_blocks       = (int)((max_input_bytes + chunk_bytes - 1) / chunk_bytes);
    ctx->max_blocks        = std::max(ctx->max_blocks, 1);

    CHECK_CUDA(cudaMalloc(&ctx->d_partial, sizeof(uint32_t) * ctx->max_blocks));
    CHECK_CUDA(cudaMallocHost(&ctx->h_partial, sizeof(uint32_t) * ctx->max_blocks));
    CHECK_CUDA(cudaStreamCreate(&ctx->stream));

    return ctx;
}

void gpu_crc32_destroy(GpuCrc32Context* ctx)
{
    if (!ctx) return;
    cudaFree(ctx->d_partial);
    cudaFreeHost(ctx->h_partial);
    cudaStreamDestroy(ctx->stream);
    delete ctx;
}

uint32_t gpu_crc32_compute(GpuCrc32Context* ctx, const uint8_t* d_data, size_t len)
{
    if (len == 0) return 0;
    if (len > ctx->max_input_bytes) {
        fprintf(stderr,
            "gpu_crc32_compute: len %zu exceeds max_input_bytes %zu from gpu_crc32_create()\n",
            len, ctx->max_input_bytes);
        exit(1);
    }

    const int num_blocks = (int)((len + ctx->chunk_bytes - 1) / ctx->chunk_bytes);

    crc32_partial_kernel<<<num_blocks, 1, 0, ctx->stream>>>(
        d_data, len, ctx->chunk_bytes, ctx->d_partial);

    CHECK_CUDA(cudaMemcpyAsync(ctx->h_partial, ctx->d_partial,
                               sizeof(uint32_t) * num_blocks,
                               cudaMemcpyDeviceToHost, ctx->stream));
    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));

    // Combine partial CRCs in order using zlib's GF(2) combine algebra --
    // O(num_blocks) host-side calls, each cheap; the expensive part (the
    // actual byte-by-byte CRC work) already happened in parallel on the GPU.
    uint32_t running = 0;
    for (int b = 0; b < num_blocks; b++) {
        const size_t start = (size_t)b * ctx->chunk_bytes;
        const size_t end   = std::min(start + ctx->chunk_bytes, len);
        const size_t blen  = end - start;
        if (b == 0) running = ctx->h_partial[0];
        else        running = (uint32_t)crc32_combine(running, ctx->h_partial[b], (z_off_t)blen);
    }
    return running;
}

uint32_t gpu_crc32_combine(uint32_t crc1, uint32_t crc2, size_t len2)
{
    return (uint32_t)crc32_combine(crc1, crc2, (z_off_t)len2);
}
