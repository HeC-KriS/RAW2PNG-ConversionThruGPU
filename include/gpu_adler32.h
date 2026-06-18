#pragma once
// gpu_adler32.h
// GPU-parallel Adler-32 (RFC 1950 checksum used by the zlib stream trailer),
// computed directly on GPU-resident filtered strip data so the modern
// pipeline never needs a host copy of that data just to checksum it.
//
// Same block-parallel-partial + host-combine design as gpu_crc32.h: split
// the input into independent sub-chunks, compute each as if it were its own
// standalone buffer (A=1, B=0), then combine via zlib's adler32_combine() --
// guarantees bit-identical results to the existing CPU adler32() path.
//
// Deliberately uses a per-byte modulo inside each sub-chunk rather than
// zlib's NMAX-deferred-reduction optimization: simpler to verify correct by
// hand (no need to re-derive zlib's overflow-safety bound), and the
// parallelism across sub-chunks already makes this far faster than the
// single-threaded CPU path it replaces. Revisiting with deferred reduction
// is a valid future optimization once this is verified on hardware.
//
// Opaque handle: no CUDA types in this header, so plain .cpp can include it.

#include <cstddef>
#include <cstdint>

struct GpuAdler32Context;

// max_input_bytes: largest single buffer ever passed to gpu_adler32_compute().
// chunk_bytes: sub-chunk size each GPU block computes independently.
GpuAdler32Context* gpu_adler32_create(size_t max_input_bytes, size_t chunk_bytes = 65536);
void                gpu_adler32_destroy(GpuAdler32Context* ctx);

// Blocking convenience: launch + collect in one call.
uint32_t gpu_adler32_compute(GpuAdler32Context* ctx, const uint8_t* d_data, size_t len);

// Split async form — allows overlap with other GPU work (e.g., deflate kernels
// running in a different stream that also reads d_data).
//
// gpu_adler32_launch: queues the kernel and D2H into ctx->stream. Returns
//   immediately without blocking the host. len is saved in the context.
//
// gpu_adler32_collect: blocks the host until ctx->stream is idle, then
//   combines the partial results and returns the Adler-32.  Must be called
//   exactly once after each gpu_adler32_launch call and before the next launch.
void     gpu_adler32_launch (GpuAdler32Context* ctx, const uint8_t* d_data, size_t len);
uint32_t gpu_adler32_collect(GpuAdler32Context* ctx);
