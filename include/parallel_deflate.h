#pragma once
// parallel_deflate.h
// Multi-threaded CPU DEFLATE using a persistent ThreadPool.
//
// Each strip is divided into N chunks.  Chunks are compressed independently
// in parallel (raw DEFLATE, windowBits = -15).  The results are byte-aligned
// at every boundary (Z_FULL_FLUSH for non-terminal chunks, Z_FINISH for the
// absolute last chunk).  Concatenation produces a valid DEFLATE bitstream.
//
// Adler-32 is computed per chunk and combined with adler32_combine() so that
// the caller can append the correct zlib trailer after all strips.

#include "thread_pool.h"
#include <cstddef>
#include <cstdint>
#include <vector>

// Compression settings
struct ParallelDeflateConfig {
    int num_threads = 6;   // chunks per strip (matches i5-9400F physical cores)
    int zlib_level  = 3;   // 1=fastest, 9=best ratio; 3 is recommended default
};

// Per-strip result returned by deflate_strip().
// Carries its own adler32 contribution so the caller can accumulate in order.
struct DeflateResult {
    std::vector<uint8_t> data;         // raw DEFLATE bytes (no zlib header/trailer)
    unsigned long        strip_adler;  // adler32 of uncompressed input bytes
    size_t               input_size;   // byte count of uncompressed input
    // Diagnostic timing (sum across all parallel chunks in this strip, in μs).
    // Wall-time cost = each field / num_threads  (chunks run in parallel).
    long long            init_us     = 0;  // time in z_stream acquire/reset
    long long            compress_us = 0;  // time inside deflate() call
};

// Running checksum accumulated across all strips in sequence.
struct ParallelDeflateState {
    unsigned long running_adler = 1;   // adler32 neutral initial value (RFC 1950)
    bool          started       = false;
};

// Compress one strip's filtered row data using the persistent thread pool.
// is_last must be true for the final strip of the image so the DEFLATE
// bitstream is properly terminated (BFINAL=1).
DeflateResult deflate_strip(
    const uint8_t*               input,
    size_t                       input_size,
    const ParallelDeflateConfig& cfg,
    ThreadPool&                  pool,
    bool                         is_last);

// Accumulate one strip's adler32 contribution into state.
// Must be called in strip order (0, 1, 2, …).
void accum_adler(ParallelDeflateState& state, const DeflateResult& dr);

// Build the 2-byte zlib stream header for the given compression level.
// Write this before the first DEFLATE byte in the IDAT stream.
void zlib_header(int level, uint8_t out[2]);

// Build the 4-byte big-endian Adler-32 trailer.
// Write this after the last DEFLATE byte in the IDAT stream.
void zlib_trailer(const ParallelDeflateState& state, uint8_t out[4]);

// Thin wrapper over zlib's adler32(), so callers outside parallel_deflate.cpp
// (e.g. the modern GPU pipeline in pipeline.cpp) don't need their own
// #include <zlib.h> just to checksum a buffer for accum_adler().
unsigned long adler32_of(const uint8_t* data, size_t len);

// Thin wrapper over zlib's crc32(), for callers constructing tiny fixed PNG
// chunks (signature/IHDR/IEND) directly on the host without their own
// #include <zlib.h> -- e.g. the modern GPU pipeline's run_one_modern(),
// which only needs gpu_png_assemble.h's GPU-resident path for the bulk IDAT
// data, not these few dozen constant-ish bytes.
unsigned long crc32_of(const uint8_t* data, size_t len);

// Standalone DEFLATE throughput benchmark.
// Compresses strip_bytes of synthetic data at levels 0, 1, and 3 using
// num_threads parallel chunks, prints a throughput table to stdout.
// Use --bench-deflate to invoke from the CLI.
void bench_deflate(size_t strip_bytes, int num_threads);
