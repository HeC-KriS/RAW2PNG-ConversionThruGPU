#pragma once
// gpu_crc32.h
// GPU-parallel CRC-32 (zlib/PNG polynomial 0xEDB88320), for computing PNG
// chunk checksums (IDAT, etc.) without a CPU round-trip once chunk data is
// GPU-resident (Requirement 8 / 9 of the modern pipeline).
//
// CRC-32 is serial byte-to-byte, so parallelism comes from splitting the
// input into independent sub-chunks (one GPU block per sub-chunk, computed
// as if each sub-chunk were its own standalone buffer), then algebraically
// combining the partial results -- the same idea as zlib's crc32_combine(),
// which this implementation calls directly to guarantee bit-identical output
// to the existing CPU crc32() path used by png_writer.cpp today.
//
// Opaque handle: no CUDA types in this header, so plain .cpp can include it.

#include <cstddef>
#include <cstdint>

struct GpuCrc32Context;

// max_input_bytes: largest single buffer ever passed to gpu_crc32_compute().
// Sized once at creation time so no cudaMalloc happens in the hot per-call
// path. chunk_bytes: sub-chunk size each GPU block computes independently
// (default 64 KB -- large enough to amortize per-block launch overhead,
// small enough for many blocks to run in parallel on a full strip).
GpuCrc32Context* gpu_crc32_create(size_t max_input_bytes, size_t chunk_bytes = 65536);
void             gpu_crc32_destroy(GpuCrc32Context* ctx);

// Compute the CRC-32 of d_data[0..len), which must already be resident in
// device memory. len must be <= max_input_bytes passed to gpu_crc32_create().
// Blocking call (synchronizes internally before returning).
uint32_t gpu_crc32_compute(GpuCrc32Context* ctx, const uint8_t* d_data, size_t len);

// CRC of (A concatenated with B) given crc(A), crc(B), and len(B) -- exposed
// so GPU PNG assembly (Phase 5) can fold a chunk's type-field CRC together
// with its data CRC, or combine across strips, without re-touching already
// -computed device data. Thin wrapper over zlib's crc32_combine().
uint32_t gpu_crc32_combine(uint32_t crc1, uint32_t crc2, size_t len2);
