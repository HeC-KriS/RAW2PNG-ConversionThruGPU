#pragma once
// compression_backend.h
// Abstraction over "compress one filtered PNG strip into a DEFLATE-framed
// byte blob, plus an adler32 contribution for the running zlib trailer."
//
// CPUZlibBackend wraps the existing CPU path (parallel_deflate.cpp) unchanged
// -- same correctness invariant (only the last chunk of the last strip may
// set BFINAL=1; every other chunk ends with Z_FULL_FLUSH).
//
// GPUDeflateBackend (Phase 4) will implement this via nvCOMP. nvCOMP has no
// Z_FULL_FLUSH equivalent -- each call produces a complete, independently
// terminated (BFINAL=1) deflate block. To honor this same interface contract
// without corrupting the single-bitstream PNG IDAT model, GPUDeflateBackend
// must run a bit-level stitch pass after nvCOMP returns: flip each non-final
// strip's BFINAL bit from 1 to 0 and bit-shift-splice it against the next
// strip's bits (deflate blocks are not byte-aligned after a one-shot encode
// the way Z_FULL_FLUSH guarantees). That stitching is what makes
// GPUDeflateBackend's output indistinguishable from CPUZlibBackend's once
// concatenated -- it is NOT optional polish, it is required for correctness.
//
// pipeline.cpp does not dispatch through this interface yet: with only one
// real implementation (CPU) in the tree, switching deflate_stage() to call
// through a base-class pointer would be churn with no behavioral payoff.
// The polymorphic wiring happens when GPUDeflateBackend lands and there is
// an actual choice to make at runtime (selected via detect_gpu_capability()).

#include "parallel_deflate.h"
#include "thread_pool.h"
#include <cstddef>
#include <cstdint>
#include <memory>

class CompressionBackend {
public:
    virtual ~CompressionBackend() = default;

    // Compress one strip of filtered scanline bytes (PNG filter byte + row
    // data, as produced by gpu_filter_*). is_last must be true only for the
    // final strip of the final frame so the output stream's BFINAL bit ends
    // up set correctly once all strips are concatenated.
    virtual DeflateResult compress_strip(const uint8_t* input,
                                         size_t          input_size,
                                         bool            is_last) = 0;

    // Human-readable backend name for logging ("zlib (CPU)", "nvCOMP (GPU)").
    virtual const char* name() const = 0;
};

// Wraps the existing, tested CPU parallel_deflate.cpp path verbatim. Owns the
// ThreadPool it was constructed with.
class CPUZlibBackend : public CompressionBackend {
public:
    CPUZlibBackend(const ParallelDeflateConfig& cfg, int num_threads)
        : cfg_(cfg), pool_(num_threads) {}

    DeflateResult compress_strip(const uint8_t* input, size_t input_size,
                                 bool is_last) override
    {
        return deflate_strip(input, input_size, cfg_, pool_, is_last);
    }

    const char* name() const override { return "zlib (CPU)"; }

private:
    ParallelDeflateConfig cfg_;
    ThreadPool            pool_;
};

// Factory: builds the right backend for the active GpuPipelineMode. Returns
// CPUZlibBackend on both Legacy and Modern until GPUDeflateBackend (Phase 4)
// is implemented -- callers should hold the result as
// std::unique_ptr<CompressionBackend> so swapping in GPUDeflateBackend later
// requires no caller-side changes.
inline std::unique_ptr<CompressionBackend> make_compression_backend(
    const ParallelDeflateConfig& cfg, int num_threads)
{
    return std::make_unique<CPUZlibBackend>(cfg, num_threads);
}
