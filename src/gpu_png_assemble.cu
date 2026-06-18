// gpu_png_assemble.cu
// See gpu_png_assemble.h for the design rationale.

#include "gpu_png_assemble.h"

#include <cuda_runtime.h>
#include <zlib.h>       // for CPU crc32()
#include <chrono>
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
    cudaStream_t     stream;

    // Sub-phase timing accumulators (microseconds, per image)
    long long acc_chunks        = 0;
    long long acc_header_h2d_us = 0;
    long long acc_data_d2d_us   = 0;
    long long acc_presync_us    = 0;  // kept for API compat; always 0 now (no pre-CRC GPU sync)
    long long acc_crc_us        = 0;  // CPU crc32() time (microseconds)
    long long acc_crc_h2d_us    = 0;  // kept for API compat; always 0 now (CRC computed on host)
    long long acc_copy_d2h_us   = 0;
};

GpuPngAssembleContext* gpu_png_assemble_create(size_t max_chunk_data_bytes)
{
    GpuPngAssembleContext* ctx = new GpuPngAssembleContext();
    ctx->max_total_bytes = 8 + max_chunk_data_bytes + 4;

    CHECK_CUDA(cudaMalloc(&ctx->d_buf, ctx->max_total_bytes));
    CHECK_CUDA(cudaStreamCreate(&ctx->stream));
    return ctx;
}

void gpu_png_assemble_destroy(GpuPngAssembleContext* ctx)
{
    if (!ctx) return;
    cudaFree(ctx->d_buf);
    cudaStreamDestroy(ctx->stream);
    delete ctx;
}

size_t gpu_png_assemble_idat_chunk(GpuPngAssembleContext* ctx,
                                   const uint8_t* d_compressed_slice, size_t slice_bytes,
                                   bool is_first, bool is_last,
                                   uint32_t running_adler, int zlib_level)
{
    using Clock = std::chrono::high_resolution_clock;
    const size_t data_len = (is_first ? 2u : 0u) + slice_bytes + (is_last ? 4u : 0u);

    size_t off = 0;

    // --- Phase 1: IDAT Header H2D (len+type + optional zlib CMF/FLG) ---
    auto t0 = Clock::now();

    uint8_t len_type[8];
    put_u32_be(len_type, (uint32_t)data_len);
    memcpy(len_type + 4, "IDAT", 4);
    CHECK_CUDA(cudaMemcpyAsync(ctx->d_buf + off, len_type, 8, cudaMemcpyHostToDevice, ctx->stream));
    off += 8;

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

    auto t1 = Clock::now();
    ctx->acc_header_h2d_us +=
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // --- Phase 2: IDAT Data D2D (compressed slice) + optional Adler32 H2D ---
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

    auto t2 = Clock::now();
    ctx->acc_data_d2d_us +=
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    // CRC-32 is computed on the host in gpu_png_assemble_copy_to_host() after
    // the D2H transfer completes.  The +4 here reserves space in the caller's
    // host buffer for those CRC bytes (written there by copy_to_host).
    ctx->acc_chunks++;
    return off + 4;
}

void gpu_png_assemble_copy_to_host(const GpuPngAssembleContext* ctx,
                                   uint8_t* h_dst, size_t num_bytes)
{
    using Clock = std::chrono::high_resolution_clock;

    // D2H all bytes except the 4-byte CRC slot (which is not in d_buf;
    // it will be filled here after the transfer).
    auto t0 = Clock::now();
    const size_t device_bytes = num_bytes - 4;  // 8 (hdr) + data_len
    CHECK_CUDA(cudaMemcpyAsync(h_dst, ctx->d_buf, device_bytes,
                               cudaMemcpyDeviceToHost, ctx->stream));
    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));
    auto t1 = Clock::now();
    const_cast<GpuPngAssembleContext*>(ctx)->acc_copy_d2h_us +=
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // CPU CRC-32 over [type(4 bytes) + data(data_len bytes)] = h_dst[4..num_bytes-5].
    // Faster than the 1-thread-per-block GPU CRC kernel by ~10-15× on SIMD CPUs.
    auto t2 = Clock::now();
    uLong crc = crc32(0L, nullptr, 0);
    crc = crc32(crc, h_dst + 4, (uInt)(num_bytes - 8));
    h_dst[num_bytes - 4] = (uint8_t)(crc >> 24);
    h_dst[num_bytes - 3] = (uint8_t)(crc >> 16);
    h_dst[num_bytes - 2] = (uint8_t)(crc >> 8);
    h_dst[num_bytes - 1] = (uint8_t)crc;
    auto t3 = Clock::now();
    const_cast<GpuPngAssembleContext*>(ctx)->acc_crc_us +=
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
}

GpuPngAssembleStats gpu_png_assemble_get_stats(const GpuPngAssembleContext* ctx)
{
    GpuPngAssembleStats s{};
    if (!ctx) return s;
    s.chunks        = ctx->acc_chunks;
    s.header_h2d_us = ctx->acc_header_h2d_us;
    s.data_d2d_us   = ctx->acc_data_d2d_us;
    s.presync_us    = ctx->acc_presync_us;
    s.crc_us        = ctx->acc_crc_us;
    s.crc_h2d_us    = ctx->acc_crc_h2d_us;
    s.copy_d2h_us   = ctx->acc_copy_d2h_us;
    return s;
}

void gpu_png_assemble_reset_stats(GpuPngAssembleContext* ctx)
{
    if (!ctx) return;
    ctx->acc_chunks        = 0;
    ctx->acc_header_h2d_us = 0;
    ctx->acc_data_d2d_us   = 0;
    ctx->acc_presync_us    = 0;
    ctx->acc_crc_us        = 0;
    ctx->acc_crc_h2d_us    = 0;
    ctx->acc_copy_d2h_us   = 0;
}
