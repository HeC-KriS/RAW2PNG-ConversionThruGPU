// gpu_png_assemble.cu
// See gpu_png_assemble.h for the design rationale.

#include "gpu_png_assemble.h"
#include "gpu_crc32.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(1);                                                         \
        }                                                                     \
    } while (0)

static void put_u32_be(uint8_t* p, uint32_t v)
{
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)v;
}

struct GpuPngAssembleContext {
    size_t           max_total_bytes;  // 8 (chunk len+type) + max_data + 4 (crc)
    uint8_t*         d_buf;
    GpuCrc32Context*  crc_ctx;
    cudaStream_t      stream;
};

GpuPngAssembleContext* gpu_png_assemble_create(size_t max_chunk_data_bytes)
{
    GpuPngAssembleContext* ctx = new GpuPngAssembleContext();
    ctx->max_total_bytes = 8 + max_chunk_data_bytes + 4;

    CHECK_CUDA(cudaMalloc(&ctx->d_buf, ctx->max_total_bytes));
    // CRC is computed over [type(4) + data] in one call.
    ctx->crc_ctx = gpu_crc32_create(max_chunk_data_bytes + 4);
    CHECK_CUDA(cudaStreamCreate(&ctx->stream));
    return ctx;
}

void gpu_png_assemble_destroy(GpuPngAssembleContext* ctx)
{
    if (!ctx) return;
    cudaFree(ctx->d_buf);
    gpu_crc32_destroy(ctx->crc_ctx);
    cudaStreamDestroy(ctx->stream);
    delete ctx;
}

size_t gpu_png_assemble_idat_chunk(GpuPngAssembleContext* ctx,
                                   const uint8_t* d_compressed_slice, size_t slice_bytes,
                                   bool is_first, bool is_last,
                                   uint32_t running_adler, int zlib_level)
{
    const size_t data_len = (is_first ? 2u : 0u) + slice_bytes + (is_last ? 4u : 0u);

    size_t off = 0;

    uint8_t len_type[8];
    put_u32_be(len_type, (uint32_t)data_len);
    memcpy(len_type + 4, "IDAT", 4);
    CHECK_CUDA(cudaMemcpyAsync(ctx->d_buf + off, len_type, 8, cudaMemcpyHostToDevice, ctx->stream));
    off += 8;
    const size_t data_off = off;  // "IDAT" type field sits 4 bytes before this

    if (is_first) {
        uint8_t zhdr[2];
        const uint8_t CMF = 0x78;
        uint8_t FLG;
        if      (zlib_level == 1)                  FLG = 0x01;
        else if (zlib_level >= 2 && zlib_level <= 5) FLG = 0x5E;
        else if (zlib_level >= 6 && zlib_level <= 7) FLG = 0x9C;
        else                                         FLG = 0xDA;
        zhdr[0] = CMF;
        zhdr[1] = FLG;
        CHECK_CUDA(cudaMemcpyAsync(ctx->d_buf + off, zhdr, 2, cudaMemcpyHostToDevice, ctx->stream));
        off += 2;
    }

    if (slice_bytes > 0) {
        CHECK_CUDA(cudaMemcpyAsync(ctx->d_buf + off, d_compressed_slice, slice_bytes,
                                   cudaMemcpyDeviceToDevice, ctx->stream));
        off += slice_bytes;
    }

    if (is_last) {
        uint8_t atrl[4];
        put_u32_be(atrl, running_adler);
        CHECK_CUDA(cudaMemcpyAsync(ctx->d_buf + off, atrl, 4, cudaMemcpyHostToDevice, ctx->stream));
        off += 4;
    }

    // Everything above must have landed before the CRC kernel reads it.
    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));

    // CRC over [type "IDAT" (4 bytes) + data (data_len bytes)] in one call --
    // the type field sits exactly 4 bytes before data_off in this layout.
    const uint32_t crc_val =
        gpu_crc32_compute(ctx->crc_ctx, ctx->d_buf + data_off - 4, data_len + 4);
    uint8_t crc_bytes[4];
    put_u32_be(crc_bytes, crc_val);
    CHECK_CUDA(cudaMemcpyAsync(ctx->d_buf + off, crc_bytes, 4, cudaMemcpyHostToDevice, ctx->stream));
    off += 4;

    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));
    return off;
}

void gpu_png_assemble_copy_to_host(const GpuPngAssembleContext* ctx,
                                   uint8_t* h_dst, size_t num_bytes)
{
    CHECK_CUDA(cudaMemcpyAsync(h_dst, ctx->d_buf, num_bytes,
                               cudaMemcpyDeviceToHost, ctx->stream));
    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));
}
