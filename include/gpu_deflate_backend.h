#pragma once
// gpu_deflate_backend.h
// Custom GPU DEFLATE encoder using FIXED Huffman literal codes (RFC 1951
// section 3.2.6) -- no LZ77 back-references, no dynamic Huffman tables.
//
// WHY NOT nvCOMP: nvCOMP's deflate is a one-shot codec -- each call produces
// a complete, independently BFINAL=1-terminated bitstream with no exposed
// bit-precise end position. Splicing several such opaque streams into one
// continuous PNG IDAT bitstream requires knowing the *exact bit* where each
// stream's real data ends (trailing bits in the last byte are unspecified
// padding, not zero by guarantee), which would require either an undocumented
// API guarantee or a hand-written GPU inflate decode-trace to discover -- both
// unverifiable without nvCOMP's actual headers and real hardware in this
// session. This encoder sidesteps the problem entirely: it tracks its own
// output bit position as it writes (exactly like zlib's internal bi_buf/
// bi_valid bit writer), so the end-of-data bit position is always known by
// construction, never discovered after the fact.
//
// TRADEOFF: literal-only fixed Huffman gets a meaningfully worse compression
// ratio than zlib's LZ77 + dynamic Huffman (no run-length exploitation of
// repeated byte sequences). This was an explicit, accepted tradeoff in favor
// of correctness that's provable without test hardware. Revisiting with a
// GPU LZ77 matcher is a valid future improvement once this path is verified.
//
// OUTPUT FRAMING: deliberately byte-compatible with the existing CPU path
// (parallel_deflate.h's zlib_header()/zlib_trailer()) -- this encoder only
// produces the *raw deflate bytes*; the existing 2-byte zlib header and
// 4-byte Adler-32 trailer functions are reused unchanged by the modern
// pipeline. gpu_deflate_output_byte_length() rounds up to the next whole
// byte, exactly like zlib's total_out after Z_FINISH -- any unused high bits
// in that last byte are zero (never read by a conformant decoder, which
// stops at the BFINAL+end-of-block code).
//
// VERIFICATION RECIPE (do this on real RTX 5050 hardware before trusting
// this in production): encode a known image through both the CPU path and
// this GPU path, decompress both resulting PNGs with libpng (or any
// standard viewer/tool) and diff the decoded pixels -- they must match
// exactly. Independently, RFC 1951's own worked example states literal 0's
// fixed Huffman code is "00110000"; the table-build algorithm here was
// checked by hand to reproduce that exact value (see code comment in the
// .cu file) as a sanity check before being trusted further.
//
// Opaque handle: no CUDA types in this header, so plain .cpp can include it.

#include <cstddef>
#include <cstdint>

struct GpuDeflateContext;

// max_rows: largest strip_height ever passed. MUST be <= 1024 -- the per-row
// bit-length prefix sum runs as a single-block scan (one thread per row), so
// 1024 (CUDA's per-block thread limit on all architectures this targets) is
// a hard ceiling, not a tuning knob. gpu_deflate_compress_strip() rejects
// larger actual_rows at runtime with a clear error rather than silently
// producing wrong output.
// max_row_bytes: width*bpp + 1 (filter byte + filtered pixel bytes per row).
// max_total_bits: provable upper bound on total output bits across the
// WHOLE image this context will encode. Fixed-Huffman literal codes are at
// most 9 bits, so total_filtered_bytes*9 + num_strips*10 (3-bit header +
// 7-bit EOB per strip) is a safe bound -- never just "probably enough".
GpuDeflateContext* gpu_deflate_create(int max_rows, int max_row_bytes,
                                      size_t max_total_bits);
void               gpu_deflate_destroy(GpuDeflateContext* ctx);

// Reset the running bit position and re-zero the output buffer. Required
// before reusing a context for a new image/frame: this encoder only ever
// OR-writes into the output buffer (see .cu rationale), so leftover '1' bits
// from a previous image would otherwise corrupt the new one.
void gpu_deflate_reset(GpuDeflateContext* ctx);

// Append one strip's filtered data (already GPU-resident: [actual_rows x
// row_bytes], one PNG filter byte + filtered pixel bytes per row) as one
// fixed-Huffman DEFLATE block into the context's running device-resident
// output bitstream. is_last must be true only for the final strip of the
// final frame (sets BFINAL=1 on that block).
//
// No flush/byte-alignment is needed between strips -- unlike the CPU path's
// Z_FULL_FLUSH (needed there because independent zlib streams per chunk lose
// track of bit position across deflateEnd/deflateInit2 boundaries), this
// encoder's bit position is tracked continuously across all calls on one
// context, so each new strip's block starts exactly where the previous one's
// EOB code ended, bit-precise, with no padding required.
//
// PERFORMANCE: this function is pure kernel-launch sequencing -- the running
// bit position and all per-strip bookkeeping live in device memory and are
// read/written by tiny single-thread kernels, never copied to the host
// mid-strip (an earlier version of this file did a D2H/host-loop/H2D round
// trip here; that is gone). The one cudaStreamSynchronize left in this
// function exists only for a cross-stream buffer-reuse hazard with the
// caller's GpuFilterContext (see the .cu file's comment at that call site),
// not for this encoder's own state.
void gpu_deflate_compress_strip(GpuDeflateContext* ctx,
                                const uint8_t* d_filtered, int actual_rows,
                                int row_bytes, bool is_last);

// Device pointer to the output bitstream (byte 0 = bit 0 of the stream).
const uint8_t* gpu_deflate_output(const GpuDeflateContext* ctx);

// Valid byte length so far, rounded up to the next whole byte (matches
// zlib's total_out convention -- see "OUTPUT FRAMING" above). This is the
// ONE point per image where the running bit position is read back to the
// host (a single size_t) -- call it once, after the last gpu_deflate_compress_strip().
size_t gpu_deflate_output_byte_length(const GpuDeflateContext* ctx);

// Bytes that are GUARANTEED COMPLETE -- i.e. floor(total_bits/8) -- meaning
// no future gpu_deflate_compress_strip() call will ever write to them again
// (every write only ever advances forward from the current bit position).
// Use this, NOT gpu_deflate_output_byte_length(), to decide how many bytes
// are safe to flush incrementally to an IDAT chunk mid-image: the latter
// rounds UP and so may include a still-partially-written trailing byte that
// a later strip's encoding will continue to OR-write into.
size_t gpu_deflate_flushable_byte_length(const GpuDeflateContext* ctx);

// Copy the first num_bytes of the output bitstream to host memory (h_dst
// must have room for num_bytes). Blocking call. Keeps all direct CUDA
// runtime calls inside this .cu file, consistent with gpu_filter.h/.cu --
// callers like pipeline.cpp never need their own <cuda_runtime.h> include.
void gpu_deflate_copy_to_host(const GpuDeflateContext* ctx, uint8_t* h_dst,
                              size_t num_bytes);
