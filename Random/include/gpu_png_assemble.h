#pragma once
// gpu_png_assemble.h
// GPU-resident PNG IDAT chunk assembly (Requirement 9), INCREMENTAL form:
// produces one IDAT chunk at a time from a slice of the GPU deflate output,
// so the modern pipeline's writer stage can flush completed chunks to disk
// while LATER strips are still being compressed (see pipeline.cpp's
// run_one_modern() for the producer/consumer design this enables -- this
// was the single biggest lever for closing the gap to the RTX 5050
// optimization PRD's wall-time target, since disk I/O dominated the
// previous strictly-sequential design's wall time).
//
// PNG does not require IDAT chunk boundaries to align with DEFLATE block
// boundaries -- decoders concatenate ALL IDAT chunks' data into one
// continuous buffer before inflating it, so chunking is purely a transport
// convenience. This lets the caller flush whatever new COMPLETE bytes have
// accumulated after each strip (or batch of strips) as its own chunk,
// independent of where deflate block boundaries fall.
//
// CORRECTNESS-CRITICAL: only flush bytes gpu_deflate_flushable_byte_length()
// reports as complete (see that function's doc comment) -- never flush a
// byte that a later gpu_deflate_compress_strip() call might still OR-write
// into, or the copy already on disk will be stale.
//
// The PNG signature, IHDR chunk, and IEND chunk are NOT produced by this
// module: at a few dozen constant-ish bytes each, they don't need to stay
// GPU-resident, so pipeline.cpp constructs them directly on the host (see
// parallel_deflate.h's crc32_of()).
//
// IDAT's CRC-32 is computed with gpu_crc32_compute() over the chunk's type
// field + full data payload in one call (laid out contiguously in the
// assembled buffer).
//
// Opaque handle: no CUDA types in this header, so plain .cpp can include it.

#include <cstddef>
#include <cstdint>

struct GpuPngAssembleContext;

// max_chunk_data_bytes: upper bound on a single IDAT chunk's DATA payload --
// i.e. the largest compressed-byte slice you will ever pass in one call,
// plus 6 bytes of slack for the zlib header (first chunk) / Adler-32
// trailer (last chunk).
GpuPngAssembleContext* gpu_png_assemble_create(size_t max_chunk_data_bytes);
void                   gpu_png_assemble_destroy(GpuPngAssembleContext* ctx);

// Produce ONE complete IDAT chunk (length + type + data + crc) from a slice
// of the GPU deflate output buffer (gpu_deflate_output() + an offset/length
// you track between flushes -- see gpu_deflate_flushable_byte_length()).
//
// is_first: prepend the 2-byte zlib stream header to this chunk's data
//   (true only for the very first chunk flushed for this image).
// is_last: append the 4-byte big-endian Adler-32 trailer after the data
//   (true only for the final chunk; running_adler must be the final,
//   fully-accumulated checksum at that point -- ignored otherwise).
// zlib_level: only affects the cosmetic FLEVEL bits in the zlib header
//   (this encoder has no real compression "levels"); ignored unless is_first.
//
// Returns the chunk's total byte length. Call gpu_png_assemble_copy_to_host()
// with that length immediately after to retrieve it.
size_t gpu_png_assemble_idat_chunk(GpuPngAssembleContext* ctx,
                                   const uint8_t* d_compressed_slice, size_t slice_bytes,
                                   bool is_first, bool is_last,
                                   uint32_t running_adler, int zlib_level);

// Single D2H copy of the most recently assembled chunk (num_bytes =
// gpu_png_assemble_idat_chunk()'s return value), ready for one fwrite().
void gpu_png_assemble_copy_to_host(const GpuPngAssembleContext* ctx,
                                   uint8_t* h_dst, size_t num_bytes);

// Per-image PNG assembly timing breakdown (microseconds, cumulative across all
// chunks produced for one image since the last gpu_png_assemble_reset_stats).
struct GpuPngAssembleStats {
    long long chunks        = 0;  // number of IDAT chunks produced
    long long header_h2d_us = 0;  // H2D: chunk len+type (8B) + optional zlib header (2B)
    long long data_d2d_us   = 0;  // D2D: compressed slice → GPU chunk buffer + adler trailer
    long long presync_us    = 0;  // cudaStreamSync stall before CRC kernel can start
    long long crc_us        = 0;  // CRC32 kernel + partial D2H readback + host combine
    long long crc_h2d_us    = 0;  // H2D: 4-byte CRC → device buffer + final stream sync
    long long copy_d2h_us   = 0;  // D2H: assembled chunk → pinned host buffer for fwrite
};

GpuPngAssembleStats gpu_png_assemble_get_stats(const GpuPngAssembleContext* ctx);
void                gpu_png_assemble_reset_stats(GpuPngAssembleContext* ctx);
