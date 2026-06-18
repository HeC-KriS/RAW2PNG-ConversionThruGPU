#pragma once
// gpu_deflate_backend.h
// Custom GPU DEFLATE encoder using Fixed Huffman codes (RFC 1951 §3.2.6).
//
// Two encoding paths are available, selected at context-creation time:
//
//   use_lz77=false (default): literal-only. Every filtered byte becomes one
//     Huffman-coded literal. Simple, fast, output is ~57 MB for a
//     17043×11710 RGB-16 image. Correct by construction.
//
//   use_lz77=true: GPU LZ77 match finder + fixed Huffman. A sequential
//     per-row scan (one CUDA block per row, shared-memory hash table,
//     16 KB per block) finds back-references left-to-right within each
//     row, replacing repeated byte sequences with length/distance pairs.
//     Extra VRAM cost: ~200 MB at strip_height=1024 for the 17043px image
//     (d_lz_len + d_lz_dist arrays at 2 bytes × rows × row_bytes each).
//     Controlled by --gpu-lz77 / --no-gpu-lz77.
//
//   Hash table (LZ77 path):
//     12-bit (4096 bucket) shared-memory table, LZ4-style (one slot per
//     bucket, most-recent position wins). Sequential scan guarantees all
//     candidates are back-references. No global hash table allocation.
//
// Strip-height limit: previously 1024 (single-block prefix scan). Now
// unlimited up to 1,048,576 rows — the prefix scan uses a 3-pass
// multi-block approach (partial scan → block totals scan → offset add).
// The only remaining limit is ceil(strip_height/1024) ≤ 1024 which
// corresponds to a theoretical maximum of 1,048,576 rows per strip; no
// practical workload comes close.
//
// OUTPUT FRAMING: raw deflate bytes only. The 2-byte zlib header and
// 4-byte Adler-32 trailer are applied by the caller via zlib_header() /
// zlib_trailer() from parallel_deflate.h, unchanged.
//
// Opaque handle: no CUDA types in this header, so plain .cpp can include it.

#include <cstddef>
#include <cstdint>

struct GpuDeflateContext;

// Accumulated LZ77 match statistics for an entire image (all strips combined).
// Returned by gpu_deflate_lz_stats(); zeros if use_lz77=false.
struct LzStats {
    uint64_t matches_found;    // number of accepted back-references
    uint64_t matched_bytes;    // total bytes covered by matches
    uint32_t max_match_len;    // longest single match (0 if none)
    double   avg_match_len;    // matched_bytes / matches_found  (0.0 if none)
    double   avg_match_dist;   // average back-reference distance (0.0 if none)
    uint64_t total_positions;  // total byte positions processed (sum of strip sizes)
    double   coverage;         // matched_bytes / total_positions  (0.0–1.0)
};

// Create a context.
//   max_rows       : largest strip_height that will ever be passed.
//   max_row_bytes  : width*bpp + 1 (filter byte + filtered pixel bytes).
//   max_total_bits : provable upper bound on total output bits for the whole
//                    image (use total_filtered_bytes*9 + num_strips*10).
//   use_lz77       : if true, allocate LZ77 token buffers and enable the
//                    sequential per-row match finder. Adds ~200 MB VRAM for
//                    the 17043px image at strip_height=1024 (token arrays
//                    only; the hash table lives in shared memory, not VRAM).
GpuDeflateContext* gpu_deflate_create(int max_rows, int max_row_bytes,
                                      size_t max_total_bits,
                                      bool use_lz77 = false);
void               gpu_deflate_destroy(GpuDeflateContext* ctx);

// Reset running bit position, re-zero the output buffer, and clear the
// LZ77 match statistics accumulators. Required before reusing a context
// for a new image.
void gpu_deflate_reset(GpuDeflateContext* ctx);

// Compress one strip. d_filtered must be GPU-resident, laid out as
// actual_rows × row_bytes (filter byte + filtered pixels per row).
// is_last must be true only for the last strip of the last frame.
// This call is FULLY ASYNC — kernels are queued into ctx->stream but the
// host is NOT blocked.  Call gpu_deflate_stream_sync() once the caller has
// launched all concurrent work for this strip (e.g., the adler32 kernel in
// its own stream), then sync before reading d_running_total or d_output.
void gpu_deflate_compress_strip(GpuDeflateContext* ctx,
                                const uint8_t* d_filtered, int actual_rows,
                                int row_bytes, bool is_last);

// Block the host until all work enqueued by the most recent
// gpu_deflate_compress_strip() completes, then collect per-phase kernel
// timings into ctx->acc_* accumulators.  Must be called before:
//   • reading d_output or d_running_total on the host, and
//   • starting a new strip if the next strip reads from d_output.
void gpu_deflate_stream_sync(GpuDeflateContext* ctx);

// Device pointer to the output bitstream.
const uint8_t* gpu_deflate_output(const GpuDeflateContext* ctx);

// Valid byte length so far, rounded up to the next whole byte.
// Call once, after the last gpu_deflate_compress_strip().
size_t gpu_deflate_output_byte_length(const GpuDeflateContext* ctx);

// Guaranteed-complete bytes (floor(total_bits/8)). Use for incremental
// flushing mid-image; never includes a byte that a later strip might
// still OR-write into.
size_t gpu_deflate_flushable_byte_length(const GpuDeflateContext* ctx);

// Copy the first num_bytes of the output bitstream to host memory. Blocking.
void gpu_deflate_copy_to_host(const GpuDeflateContext* ctx, uint8_t* h_dst,
                              size_t num_bytes);

// Return LZ77 match statistics accumulated across all strips processed
// since the last gpu_deflate_reset() call. All fields are zero if
// use_lz77=false or if no strips have been processed yet.
// Expected to be called after the last gpu_deflate_compress_strip().
LzStats gpu_deflate_lz_stats(const GpuDeflateContext* ctx);

// Per-image GPU deflate kernel phase timings (cumulative across all strips
// since the last gpu_deflate_reset / gpu_deflate_reset_phase_stats call).
// Measured via CUDA Events (GPU wall-clock) for kernel phases and host-side
// chrono for sync and stats-D2H overhead.
struct GpuDeflatePhaseStats {
    long long lz77_us      = 0;  // LZ77 match-finder kernel time (0 = literal path)
    long long bitlen_us    = 0;  // kernel A/A': per-row Huffman bit lengths
    long long scan_us      = 0;  // kernel B: 3-pass prefix scan
    long long encode_us    = 0;  // kernels C+D+E: block header + encode + EOB
    long long sync_us      = 0;  // host wait: cudaStreamSync stall per strip
    long long stats_d2h_us = 0;  // LZ77 stats D2H readback (28 B/strip, blocking)
};

GpuDeflatePhaseStats gpu_deflate_get_phase_stats(const GpuDeflateContext* ctx);
void                 gpu_deflate_reset_phase_stats(GpuDeflateContext* ctx);
