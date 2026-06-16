// gpu_deflate_backend.cu
// See gpu_deflate_backend.h for the full design rationale. Summary of the
// per-strip kernel sequence, ALL of it device-resident (no host round trip
// except one, at the very end of the whole image -- see rationale below):
//
//   Kernel A (row_bitlen_kernel)            : one thread per row, sums
//                                              fixed-Huffman code lengths for
//                                              that row's bytes -> d_row_bits[].
//   Kernel B (row_offset_scan_kernel)        : single block, Hillis-Steele
//                                              scan over d_row_bits[] ->
//                                              d_row_offsets[] (exclusive,
//                                              relative to strip start) and
//                                              d_literal_bits[0] (this
//                                              strip's total bit length).
//                                              Requires actual_rows <= 1024
//                                              (one block's thread limit) --
//                                              enforced at the top of
//                                              gpu_deflate_compress_strip().
//   Kernel C (deflate_header_and_bookkeeping_kernel): single thread. Reads
//                                              the running bit position from
//                                              device memory (d_running_total),
//                                              writes the 3-bit block header
//                                              there, then computes and
//                                              stores (still in device
//                                              memory) where this strip's row
//                                              data starts, where its EOB
//                                              code goes, and the new running
//                                              total for the NEXT strip. None
//                                              of this ever touches the host.
//   Kernel D (row_encode_splice_kernel)      : one thread per row, reads its
//                                              starting bit position from
//                                              device memory (no host-passed
//                                              offset needed), re-walks its
//                                              bytes, bit-packs them with a
//                                              standard LSB-first accumulator
//                                              (same structure as zlib's
//                                              internal bi_buf/bi_valid bit
//                                              writer), OR-splices completed
//                                              bytes into the output buffer
//                                              via a sub-word atomicOr.
//   Kernel E (write_small_bits_from_ptr_kernel): single thread, writes the
//                                              7-bit EOB code at the bit
//                                              position Kernel C computed.
//
// EARLIER DESIGN NOTE (superseded): an earlier version of this file did the
// prefix sum on the HOST -- D2H the per-row lengths, a serial host loop, H2D
// the offsets back -- once per strip. That round trip's cost scaled with
// strip_height and forced a host sync mid-strip. Moving the scan and all
// bit-position bookkeeping onto the device (this version) removes that
// host stall entirely; the only remaining sync per strip is the one at the
// end of gpu_deflate_compress_strip(), which is NOT for this encoder's own
// bookkeeping but for a cross-stream hazard (see comment at the call site).
//
// CORRECTNESS-CRITICAL: the output buffer must be zero before any writes,
// because every write here is an OR, never an overwrite (rows splice into
// shared boundary bytes from both sides). gpu_deflate_create() zeroes it
// once; gpu_deflate_reset() re-zeroes it for reuse across images/frames.

#include "gpu_deflate_backend.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(1);                                                         \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Fixed Huffman literal/EOB table (symbols 0..256 only -- no length/distance
// codes since this encoder never emits LZ77 back-references).
//
// Built on the HOST using the canonical Huffman code assignment algorithm
// from RFC 1951 section 3.2.2, from the FOUR length-range constants given
// directly in RFC 1951 section 3.2.6 (these four boundaries are extremely
// well known and independently checkable against any DEFLATE reference).
// Hand-transcribing the resulting ~257 bit patterns would be error-prone and
// unverifiable here; deriving them from the simple length constants instead
// means the only "magic numbers" below are the four range boundaries.
//
// Sanity check performed by hand while writing this: the algorithm assigns
// literal value 0 the code 0b00110000 (8 bits, MSB-first numeric value 48)
// -- this exactly matches RFC 1951's own published worked example for the
// fixed Huffman table, which is the strongest available correctness signal
// without a real decoder to test against in this session.
//
// DEFLATE packs Huffman codes MSB-first into the bitstream while every other
// field (block header bits, etc.) is packed LSB-first; codes are bit-reversed
// here so they can be appended through the same simple LSB-first bit writer
// used for everything else in this file.
// ---------------------------------------------------------------------------
struct HuffEntry { uint32_t code; uint8_t len; };

__device__ __constant__ HuffEntry kFixedHuff[257];

static void build_fixed_huffman_table_host(HuffEntry table[257])
{
    uint8_t lengths[288];
    for (int i = 0;   i <= 143; i++) lengths[i] = 8;
    for (int i = 144; i <= 255; i++) lengths[i] = 9;
    for (int i = 256; i <= 279; i++) lengths[i] = 7;
    for (int i = 280; i <= 287; i++) lengths[i] = 8;

    int length_count[16] = {0};
    for (int i = 0; i < 288; i++) length_count[lengths[i]]++;

    int next_code[16] = {0};
    {
        int code = 0;
        length_count[0] = 0;
        for (int bits = 1; bits <= 15; bits++) {
            code = (code + length_count[bits - 1]) << 1;
            next_code[bits] = code;
        }
    }

    uint32_t raw_code[288];
    for (int i = 0; i < 288; i++) {
        int len = lengths[i];
        raw_code[i] = (uint32_t)next_code[len]++;
    }

    for (int i = 0; i <= 256; i++) {
        const int len = lengths[i];
        uint32_t v = raw_code[i];
        uint32_t r = 0;
        for (int b = 0; b < len; b++) {
            r = (r << 1) | (v & 1u);
            v >>= 1;
        }
        table[i].code = r;
        table[i].len  = (uint8_t)len;
    }
}

// ---------------------------------------------------------------------------
// Sub-byte atomic OR (CUDA has no native 8-bit atomicOr; mask onto the
// containing aligned 32-bit word instead -- standard, well-established trick).
// ---------------------------------------------------------------------------
__device__ inline void atomic_or_byte(uint8_t* addr, uint8_t value)
{
    if (value == 0) return;
    uintptr_t addr_val = (uintptr_t)addr;
    uintptr_t aligned   = addr_val & ~((uintptr_t)3);
    unsigned int byte_offset = (unsigned int)(addr_val - aligned);
    unsigned int shift       = byte_offset * 8;
    unsigned int mask        = ((unsigned int)value) << shift;
    atomicOr((unsigned int*)aligned, mask);
}

__device__ inline void splice_field(uint8_t* g_out, size_t bit_offset,
                                    uint32_t value, int nbits)
{
    const size_t   byte_pos = bit_offset / 8;
    const int      shift     = (int)(bit_offset % 8);
    const uint32_t v = value & ((nbits >= 32) ? 0xFFFFFFFFu : ((1u << nbits) - 1u));
    if (shift == 0) {
        atomic_or_byte(&g_out[byte_pos], (uint8_t)v);
    } else {
        atomic_or_byte(&g_out[byte_pos],     (uint8_t)((v << shift) & 0xFF));
        atomic_or_byte(&g_out[byte_pos + 1], (uint8_t)(v >> (8 - shift)));
    }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------
struct GpuDeflateContext {
    int     max_rows;
    int     max_row_bytes;
    size_t  max_total_bits;
    size_t  max_total_bytes;

    size_t* d_row_bits;       // [max_rows] per-row bit length (Kernel A output)
    size_t* d_row_offsets;    // [max_rows] exclusive prefix sum within the strip
    size_t* d_literal_bits;   // [1] this strip's total bit length (Kernel B output)
    size_t* d_running_total;  // [1] persistent bit position across strips/calls
    size_t* d_strip_base_bit; // [1] where this strip's row data starts
    size_t* d_eob_bit_pos;    // [1] where this strip's EOB code goes

    uint8_t* d_output;        // [max_total_bytes] running output bitstream

    cudaStream_t stream;
};

GpuDeflateContext* gpu_deflate_create(int max_rows, int max_row_bytes,
                                      size_t max_total_bits)
{
    static bool table_uploaded = false;
    if (!table_uploaded) {
        HuffEntry h_table[257];
        build_fixed_huffman_table_host(h_table);
        CHECK_CUDA(cudaMemcpyToSymbol(kFixedHuff, h_table, sizeof(h_table)));
        table_uploaded = true;
    }

    GpuDeflateContext* ctx = new GpuDeflateContext();
    ctx->max_rows        = max_rows;
    ctx->max_row_bytes    = max_row_bytes;
    ctx->max_total_bits  = max_total_bits;
    ctx->max_total_bytes = (max_total_bits + 7) / 8 + 8;  // +8 bytes of slack for boundary writes

    CHECK_CUDA(cudaMalloc(&ctx->d_row_bits,       sizeof(size_t) * max_rows));
    CHECK_CUDA(cudaMalloc(&ctx->d_row_offsets,    sizeof(size_t) * max_rows));
    CHECK_CUDA(cudaMalloc(&ctx->d_literal_bits,   sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&ctx->d_running_total,  sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&ctx->d_strip_base_bit, sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&ctx->d_eob_bit_pos,    sizeof(size_t)));

    CHECK_CUDA(cudaMalloc(&ctx->d_output, ctx->max_total_bytes));
    CHECK_CUDA(cudaMemset(ctx->d_output, 0, ctx->max_total_bytes));
    CHECK_CUDA(cudaMemset(ctx->d_running_total, 0, sizeof(size_t)));

    CHECK_CUDA(cudaStreamCreate(&ctx->stream));

    return ctx;
}

void gpu_deflate_destroy(GpuDeflateContext* ctx)
{
    if (!ctx) return;
    cudaFree(ctx->d_row_bits);
    cudaFree(ctx->d_row_offsets);
    cudaFree(ctx->d_literal_bits);
    cudaFree(ctx->d_running_total);
    cudaFree(ctx->d_strip_base_bit);
    cudaFree(ctx->d_eob_bit_pos);
    cudaFree(ctx->d_output);
    cudaStreamDestroy(ctx->stream);
    delete ctx;
}

void gpu_deflate_reset(GpuDeflateContext* ctx)
{
    if (!ctx) return;
    // Every write in this encoder is an OR, never an overwrite -- the buffer
    // MUST be zero before reuse or stale '1' bits from a previous image
    // would corrupt the new bitstream (OR can only set bits, never clear).
    CHECK_CUDA(cudaMemsetAsync(ctx->d_output, 0, ctx->max_total_bytes, ctx->stream));
    CHECK_CUDA(cudaMemsetAsync(ctx->d_running_total, 0, sizeof(size_t), ctx->stream));
    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));
}

// ---------------------------------------------------------------------------
// Kernel A: per-row fixed-Huffman bit length (no packing yet)
// ---------------------------------------------------------------------------
__global__ void row_bitlen_kernel(const uint8_t* __restrict__ d_filtered,
                                   int row_bytes, int actual_rows,
                                   size_t* __restrict__ d_row_bits)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= actual_rows) return;

    const uint8_t* src = d_filtered + (size_t)row * row_bytes;
    size_t total = 0;
    for (int i = 0; i < row_bytes; i++)
        total += kFixedHuff[src[i]].len;
    d_row_bits[row] = total;
}

// ---------------------------------------------------------------------------
// Kernel B: single-block Hillis-Steele inclusive scan, converted to an
// exclusive prefix sum per row, plus the strip's total bit length. Requires
// blockDim.x >= actual_rows (enforced by the launch in
// gpu_deflate_compress_strip, which also rejects actual_rows > 1024).
// ---------------------------------------------------------------------------
__global__ void row_offset_scan_kernel(const size_t* __restrict__ d_row_bits,
                                        size_t* __restrict__ d_row_offsets,
                                        size_t* __restrict__ d_literal_bits_out,
                                        int actual_rows)
{
    extern __shared__ size_t sdata[];
    const int tid = threadIdx.x;
    const size_t val = (tid < actual_rows) ? d_row_bits[tid] : 0;
    sdata[tid] = val;
    __syncthreads();

    for (int offset = 1; offset < (int)blockDim.x; offset <<= 1) {
        const size_t add = (tid >= offset) ? sdata[tid - offset] : 0;
        __syncthreads();
        sdata[tid] += add;
        __syncthreads();
    }

    if (tid < actual_rows)
        d_row_offsets[tid] = sdata[tid] - val;  // exclusive = inclusive - own value
    if (tid == blockDim.x - 1)
        *d_literal_bits_out = sdata[tid];       // inclusive sum over all real rows
                                                  // (padding rows contributed 0)
}

// ---------------------------------------------------------------------------
// Kernel C: single thread. All bit-position bookkeeping for this strip,
// entirely in device memory -- the host never sees any of these values
// during the per-strip loop (only gpu_deflate_output_byte_length() reads
// d_running_total back, once, after the whole image is done).
// ---------------------------------------------------------------------------
__global__ void deflate_header_and_bookkeeping_kernel(
    uint8_t* __restrict__ g_out,
    const size_t* __restrict__ d_literal_bits,
    size_t* __restrict__ d_running_total,
    size_t* __restrict__ d_strip_base_bit,
    size_t* __restrict__ d_eob_bit_pos,
    int is_last)
{
    const size_t base = *d_running_total;

    // 3-bit block header: bit0=BFINAL, bits1-2=BTYPE (fixed Huffman = value
    // 1, packed LSB-first like any other non-Huffman-code field --
    // transmission order is [BFINAL][1][0], i.e. header value = BFINAL|0b010).
    const uint32_t header_value = is_last ? 0x3u : 0x2u;
    splice_field(g_out, base, header_value, 3);

    const size_t strip_base = base + 3;
    const size_t lit         = *d_literal_bits;
    const size_t eob_pos     = strip_base + lit;

    *d_strip_base_bit = strip_base;
    *d_eob_bit_pos     = eob_pos;
    *d_running_total   = eob_pos + 7;  // ready for the next strip's Kernel C
}

// ---------------------------------------------------------------------------
// Kernel D: re-walk each row's bytes, bit-pack with a standard LSB-first
// accumulator, OR-splice completed bytes directly into the global output.
// One thread per row; rows run fully in parallel across the GPU. Reads its
// starting bit position from device memory (d_strip_base_bit) rather than a
// host-passed argument, so it never needs Kernel C's result on the host.
// ---------------------------------------------------------------------------
__global__ void row_encode_splice_kernel(
    const uint8_t* __restrict__ d_filtered, int row_bytes, int actual_rows,
    const size_t* __restrict__ d_row_offsets,
    const size_t* __restrict__ d_strip_base_bit,
    uint8_t* __restrict__ g_out)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= actual_rows) return;

    const uint8_t* src = d_filtered + (size_t)row * row_bytes;
    size_t global_bit = *d_strip_base_bit + d_row_offsets[row];

    uint32_t acc      = 0;   // LSB-first bit accumulator, like zlib's bi_buf
    int      acc_bits = 0;   // valid bits currently in acc (always < 8 before
                              // the next code is added, since we flush eagerly)

    for (int i = 0; i < row_bytes; i++) {
        const HuffEntry e = kFixedHuff[src[i]];
        acc |= (e.code << acc_bits);
        acc_bits += e.len;

        while (acc_bits >= 8) {
            splice_field(g_out, global_bit, acc & 0xFFu, 8);
            global_bit += 8;
            acc       >>= 8;
            acc_bits   -= 8;
        }
    }

    if (acc_bits > 0)
        splice_field(g_out, global_bit, acc & 0xFFu, acc_bits);
}

// ---------------------------------------------------------------------------
// Kernel E: write a small (<=7-bit) field at a bit offset read from device
// memory. Used for the EOB code, whose position (d_eob_bit_pos) is only
// known after Kernel B/C run -- never touches the host either.
// ---------------------------------------------------------------------------
__global__ void write_small_bits_from_ptr_kernel(uint8_t* g_out,
                                                  const size_t* d_bit_offset,
                                                  uint32_t value, int nbits)
{
    splice_field(g_out, *d_bit_offset, value, nbits);
}

// ---------------------------------------------------------------------------
// Host-side orchestration -- pure kernel-launch sequencing, no host
// computation or data transfer until the single trailing sync (see comment
// at that call site for why it is still needed).
// ---------------------------------------------------------------------------
void gpu_deflate_compress_strip(GpuDeflateContext* ctx,
                                const uint8_t* d_filtered, int actual_rows,
                                int row_bytes, bool is_last)
{
    if (actual_rows > 1024) {
        fprintf(stderr,
            "gpu_deflate_compress_strip: actual_rows %d exceeds the 1024-row "
            "device-scan limit -- reduce --strip-height when using --gpu-deflate.\n",
            actual_rows);
        exit(1);
    }

    const int threads = 256;
    const int blocks   = (actual_rows + threads - 1) / threads;

    int scan_threads = 1;
    while (scan_threads < actual_rows) scan_threads <<= 1;

    row_bitlen_kernel<<<blocks, threads, 0, ctx->stream>>>(
        d_filtered, row_bytes, actual_rows, ctx->d_row_bits);

    row_offset_scan_kernel<<<1, scan_threads, (size_t)scan_threads * sizeof(size_t), ctx->stream>>>(
        ctx->d_row_bits, ctx->d_row_offsets, ctx->d_literal_bits, actual_rows);

    deflate_header_and_bookkeeping_kernel<<<1, 1, 0, ctx->stream>>>(
        ctx->d_output, ctx->d_literal_bits, ctx->d_running_total,
        ctx->d_strip_base_bit, ctx->d_eob_bit_pos, is_last ? 1 : 0);

    row_encode_splice_kernel<<<blocks, threads, 0, ctx->stream>>>(
        d_filtered, row_bytes, actual_rows, ctx->d_row_offsets,
        ctx->d_strip_base_bit, ctx->d_output);

    write_small_bits_from_ptr_kernel<<<1, 1, 0, ctx->stream>>>(
        ctx->d_output, ctx->d_eob_bit_pos, 0u, 7);

    // The only sync in this function. NOT needed for this encoder's own
    // bookkeeping (everything above is device-resident and correctly
    // ordered by single-stream execution) -- needed because gpu_filter's
    // NEXT call will overwrite d_filtered on a DIFFERENT stream, and
    // row_encode_splice_kernel above (reading d_filtered on THIS stream)
    // must be guaranteed complete before that happens.
    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));
}

const uint8_t* gpu_deflate_output(const GpuDeflateContext* ctx)
{
    return ctx->d_output;
}

static size_t read_total_bits(const GpuDeflateContext* ctx)
{
    size_t total_bits = 0;
    CHECK_CUDA(cudaMemcpyAsync(&total_bits, ctx->d_running_total, sizeof(size_t),
                               cudaMemcpyDeviceToHost, ctx->stream));
    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));
    return total_bits;
}

size_t gpu_deflate_output_byte_length(const GpuDeflateContext* ctx)
{
    return (read_total_bits(ctx) + 7) / 8;
}

size_t gpu_deflate_flushable_byte_length(const GpuDeflateContext* ctx)
{
    return read_total_bits(ctx) / 8;
}

void gpu_deflate_copy_to_host(const GpuDeflateContext* ctx, uint8_t* h_dst,
                              size_t num_bytes)
{
    CHECK_CUDA(cudaMemcpyAsync(h_dst, ctx->d_output, num_bytes,
                               cudaMemcpyDeviceToHost, ctx->stream));
    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));
}
