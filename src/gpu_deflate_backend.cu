// gpu_deflate_backend.cu
// GPU DEFLATE encoder using Fixed Huffman codes (RFC 1951 §3.2.6).
//
// Two encoding paths share the same Kernels B/C/E and differ only in how
// tokens are produced (Kernel A/D vs A'/D'):
//
//   Literal-only path (use_lz77=false):
//     A  row_bitlen_kernel        – sums kFixedHuff[byte].len per row
//     B  (3-pass multi-block scan) – exclusive prefix sum + strip total
//     C  deflate_header_and_bookkeeping_kernel – block header, positions
//     D  row_encode_splice_kernel – emits literal Huffman codes
//     E  write_small_bits_from_ptr_kernel – EOB code
//
//   LZ77 path (use_lz77=true):
//     LZ  lz77_row_kernel         – sequential per-row match finder + greedy
//                                   parse + inline stats (one block per row,
//                                   shared-memory hash table, thread 0 scans)
//     A'  row_bitlen_kernel_lz    – sums bits for literals + length/dist codes
//     B   (same 3-pass scan, unchanged)
//     C   (unchanged)
//     D'  row_encode_splice_kernel_lz – emits literal + length/dist codes
//     E   (unchanged)
//
// Strip height is no longer limited to 1024. Kernel B uses a 3-pass
// multi-block scan (partial_scan → block_sum_scan → offset_add) that
// supports any number of rows up to max_rows, provided
// ceil(max_rows/1024) ≤ 1024 (i.e., max_rows ≤ 1,048,576).
//
// The OR-only output invariant and all other correctness properties
// (BFINAL, Z_FULL_FLUSH semantics, zlib framing) are unchanged.

#include "gpu_deflate_backend.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Fixed Huffman literal/length table (symbols 0..285)
//   0-255  : literal bytes
//   256    : EOB
//   257-285: length codes (only used by LZ77 path)
// Built once on host from the four RFC 1951 §3.2.6 length constants and
// copied to device constant memory.
// ---------------------------------------------------------------------------
struct HuffEntry { uint32_t code; uint8_t len; };
__device__ __constant__ HuffEntry kFixedHuff[286];

// Distance code tables (RFC 1951 §3.2.5)
__device__ __constant__ uint8_t  kDistCode[30];   // 5-bit codes, bit-reversed for LSB-first writer
__device__ __constant__ uint8_t  kDistExtra[30];  // extra bits per distance code
__device__ __constant__ uint32_t kDistBase[30];   // base distance per code

// Length code tables (RFC 1951 §3.2.5)
__device__ __constant__ uint8_t  kLenExtra[29];   // extra bits per length code (codes 257-285)
__device__ __constant__ uint32_t kLenBase[29];    // base length per length code
__device__ __constant__ uint8_t  kLenCodeOf[256]; // (length-3) → (code-257), index 0-255

// ---------------------------------------------------------------------------
// Host-side table builders
// ---------------------------------------------------------------------------
static void build_fixed_huffman_table_host(HuffEntry table[286])
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

    for (int i = 0; i <= 285; i++) {
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

static void build_lz77_tables_host(
    uint8_t  dist_code[30], uint8_t  dist_extra[30], uint32_t dist_base[30],
    uint8_t  len_extra[29], uint32_t len_base[29],   uint8_t  len_code_of[256])
{
    // Distance tables (RFC 1951 §3.2.5)
    static const uint32_t DIST_BASE[30] = {
           1,   2,   3,   4,   5,   7,   9,  13,  17,  25,
          33,  49,  65,  97, 129, 193, 257, 385, 513, 769,
        1025,1537,2049,3073,4097,6145,8193,12289,16385,24577
    };
    static const uint8_t DIST_EXTRA[30] = {
        0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13
    };
    for (int d = 0; d < 30; d++) {
        dist_base[d]  = DIST_BASE[d];
        dist_extra[d] = DIST_EXTRA[d];
        // 5-bit code bit-reversed for LSB-first bit writer
        uint8_t v = (uint8_t)d, r = 0;
        for (int b = 0; b < 5; b++) { r = (uint8_t)((r << 1) | (v & 1)); v >>= 1; }
        dist_code[d] = r;
    }

    // Length tables (RFC 1951 §3.2.5)
    static const uint32_t LEN_BASE[29] = {
          3,  4,  5,  6,  7,  8,  9, 10,  // codes 257-264, 0 extra bits
         11, 13, 15, 17,                    // codes 265-268, 1 extra bit
         19, 23, 27, 31,                    // codes 269-272, 2 extra bits
         35, 43, 51, 59,                    // codes 273-276, 3 extra bits
         67, 83, 99,115,                    // codes 277-280, 4 extra bits
        131,163,195,227,                    // codes 281-284, 5 extra bits
        258                                 // code  285,     0 extra bits
    };
    static const uint8_t LEN_EXTRA[29] = {
        0,0,0,0,0,0,0,0,
        1,1,1,1,
        2,2,2,2,
        3,3,3,3,
        4,4,4,4,
        5,5,5,5,
        0
    };
    for (int i = 0; i < 29; i++) {
        len_base[i]  = LEN_BASE[i];
        len_extra[i] = LEN_EXTRA[i];
    }

    // kLenCodeOf: (length-3) → (length_code - 257)
    // Fill by iterating each code's range
    for (int c = 0; c < 29; c++) {
        uint32_t base = LEN_BASE[c];
        int      extra = LEN_EXTRA[c];
        uint32_t end  = (c == 28) ? 258 : LEN_BASE[c + 1] - 1;
        // Extra bits means every value from base to base+(2^extra)-1 maps to this code
        for (uint32_t len = base; len <= end; len++) {
            if (len >= 3 && len <= 258)
                len_code_of[len - 3] = (uint8_t)c;
        }
        (void)extra;
    }
}

// ---------------------------------------------------------------------------
// Sub-byte atomic OR (unchanged from original)
// ---------------------------------------------------------------------------
__device__ inline void atomic_or_byte(uint8_t* addr, uint8_t value)
{
    if (value == 0) return;
    uintptr_t addr_val    = (uintptr_t)addr;
    uintptr_t aligned     = addr_val & ~((uintptr_t)3);
    unsigned int byte_off = (unsigned int)(addr_val - aligned);
    unsigned int shift    = byte_off * 8;
    unsigned int mask     = ((unsigned int)value) << shift;
    atomicOr((unsigned int*)aligned, mask);
}

__device__ inline void splice_field(uint8_t* g_out, size_t bit_offset,
                                    uint32_t value, int nbits)
{
    const size_t   byte_pos = bit_offset / 8;
    const int      shift    = (int)(bit_offset % 8);
    const uint32_t v = value & ((nbits >= 32) ? 0xFFFFFFFFu : ((1u << nbits) - 1u));
    if (shift == 0) {
        atomic_or_byte(&g_out[byte_pos], (uint8_t)v);
    } else {
        atomic_or_byte(&g_out[byte_pos],     (uint8_t)((v << shift) & 0xFF));
        atomic_or_byte(&g_out[byte_pos + 1], (uint8_t)(v >> (8 - shift)));
    }
}

// ---------------------------------------------------------------------------
// LZ77 constants
// ---------------------------------------------------------------------------
static constexpr int LZ_MIN_MATCH  = 3;
static constexpr int LZ_MAX_MATCH  = 258;
static constexpr int LZ_MAX_DIST   = 32768;

// Per-row shared-memory hash table (sequential kernel): 4096 buckets × 4 bytes = 16 KB.
// Each bucket holds the most-recent row-relative position (uint32_t) that hashed there.
// uint32_t handles any row_bytes value without truncation (no uint16_t wrapping risk).
// SM 3.5 (GT 710): 48 KB shared → 3 blocks per SM.
// SM 8.9 (RTX 5050): up to 100 KB shared (configurable) → ≥6 blocks per SM.
//
// 10-bit hash (1024 buckets, 4 KB shared mem) vs. original 12-bit (4096, 16 KB).
// Smaller table allows ~16+ blocks/SM on Blackwell (was 6), ~2.7× occupancy gain.
// Compression ratio impact is negligible (<1%) on photographic PNG data.
static constexpr int SEQ_HASH_BITS = 10;
static constexpr int SEQ_HASH_SIZE = 1 << SEQ_HASH_BITS;   // 1024 buckets

// Inline distance → code (0-29). Uses __clz available on SM 3.5+.
__device__ inline int dist_to_code(int dist)
{
    if (dist <= 4) return dist - 1;
    int d0    = dist - 1;
    int log2d = 31 - __clz((unsigned int)d0);
    int extra = log2d - 1;
    return 2 * extra + 2 + ((d0 >> (log2d - 1)) & 1);
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------
struct GpuDeflateContext {
    int    max_rows;
    int    max_row_bytes;
    size_t max_total_bits;
    size_t max_total_bytes;

    size_t* d_row_bits;       // [max_rows]
    size_t* d_row_offsets;    // [max_rows]
    size_t* d_literal_bits;   // [1]
    size_t* d_running_total;  // [1] -- GPU-resident; shadowed on host by h_running_total
    size_t* h_running_total;  // [1] pinned mirror of d_running_total; valid after stream sync
    size_t* d_strip_base_bit; // [1]
    size_t* d_eob_bit_pos;    // [1]
    size_t* d_block_sums;     // [ceil(max_rows/1024)+1] for multi-block scan

    uint8_t* d_output;

    // LZ77 token arrays (null when use_lz77=false)
    bool      use_lz77;
    uint16_t* d_lz_len;      // [max_rows * max_row_bytes]: 0=skip,1=lit,3-258=match
    uint16_t* d_lz_dist;     // [max_rows * max_row_bytes]: distance for matches

    // LZ77 per-strip stats accumulation (device-side, zeroed before each strip)
    uint32_t* d_stats32;     // [3]: matches_found, matched_bytes, max_match_len
    uint64_t* d_sums64;      // [2]: total_len_sum, total_dist_sum

    // Host-side running totals accumulated across all strips (reset in gpu_deflate_reset)
    uint64_t h_matches;
    uint64_t h_mbytes;
    uint32_t h_max_len;
    uint64_t h_len_sum;
    uint64_t h_dist_sum;
    uint64_t h_total_positions;

    cudaStream_t stream;

    // CUDA events for per-strip phase timing (created once, reused every strip)
    cudaEvent_t ev0;  // before any kernel (start of compress_strip)
    cudaEvent_t ev1;  // after LZ77 kernel (or == ev0 for literal path)
    cudaEvent_t ev2;  // after bitlen kernel A/A'
    cudaEvent_t ev3;  // after prefix scan B
    cudaEvent_t ev4;  // after encode+EOB kernels C+D+E

    // Accumulated phase times in microseconds (reset by gpu_deflate_reset)
    long long acc_lz77_us;
    long long acc_bitlen_us;
    long long acc_scan_us;
    long long acc_encode_us;
    long long acc_sync_us;      // host-side wait time per strip
    long long acc_stats_d2h_us; // LZ77 stats readback per strip
};

// ---------------------------------------------------------------------------
// Create / Destroy / Reset
// ---------------------------------------------------------------------------
GpuDeflateContext* gpu_deflate_create(int max_rows, int max_row_bytes,
                                      size_t max_total_bits, bool use_lz77)
{
    // Upload tables once per process. Static mutex guards against concurrent
    // callers (e.g., multiple batch workers hitting gpu_deflate_create()
    // simultaneously for their first file).
    {
        static std::mutex s_upload_mtx;
        static bool       s_uploaded = false;
        std::lock_guard<std::mutex> lk(s_upload_mtx);
        if (!s_uploaded) {
            HuffEntry  h_huff[286];
            uint8_t    h_dist_code[30];
            uint8_t    h_dist_extra[30];
            uint32_t   h_dist_base[30];
            uint8_t    h_len_extra[29];
            uint32_t   h_len_base[29];
            uint8_t    h_len_code_of[256];

            build_fixed_huffman_table_host(h_huff);
            build_lz77_tables_host(h_dist_code, h_dist_extra, h_dist_base,
                                   h_len_extra, h_len_base,  h_len_code_of);

            CHECK_CUDA(cudaMemcpyToSymbol(kFixedHuff,  h_huff,        sizeof(h_huff)));
            CHECK_CUDA(cudaMemcpyToSymbol(kDistCode,   h_dist_code,   sizeof(h_dist_code)));
            CHECK_CUDA(cudaMemcpyToSymbol(kDistExtra,  h_dist_extra,  sizeof(h_dist_extra)));
            CHECK_CUDA(cudaMemcpyToSymbol(kDistBase,   h_dist_base,   sizeof(h_dist_base)));
            CHECK_CUDA(cudaMemcpyToSymbol(kLenExtra,   h_len_extra,   sizeof(h_len_extra)));
            CHECK_CUDA(cudaMemcpyToSymbol(kLenBase,    h_len_base,    sizeof(h_len_base)));
            CHECK_CUDA(cudaMemcpyToSymbol(kLenCodeOf,  h_len_code_of, sizeof(h_len_code_of)));
            s_uploaded = true;
        }
    }

    GpuDeflateContext* ctx = new GpuDeflateContext();
    ctx->max_rows        = max_rows;
    ctx->max_row_bytes   = max_row_bytes;
    ctx->max_total_bits  = max_total_bits;
    ctx->max_total_bytes = (max_total_bits + 7) / 8 + 8;
    ctx->use_lz77        = use_lz77;

    const int max_blocks = (max_rows + 1023) / 1024 + 1;

    CHECK_CUDA(cudaMalloc(&ctx->d_row_bits,       sizeof(size_t) * max_rows));
    CHECK_CUDA(cudaMalloc(&ctx->d_row_offsets,    sizeof(size_t) * max_rows));
    CHECK_CUDA(cudaMalloc(&ctx->d_literal_bits,   sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&ctx->d_running_total,  sizeof(size_t)));
    CHECK_CUDA(cudaMallocHost(&ctx->h_running_total, sizeof(size_t)));
    *ctx->h_running_total = 0;
    CHECK_CUDA(cudaMalloc(&ctx->d_strip_base_bit, sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&ctx->d_eob_bit_pos,    sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&ctx->d_block_sums,     sizeof(size_t) * max_blocks));

    CHECK_CUDA(cudaMalloc(&ctx->d_output, ctx->max_total_bytes));
    CHECK_CUDA(cudaMemset(ctx->d_output, 0, ctx->max_total_bytes));
    CHECK_CUDA(cudaMemset(ctx->d_running_total, 0, sizeof(size_t)));

    ctx->d_lz_len  = nullptr;
    ctx->d_lz_dist = nullptr;
    ctx->d_stats32 = nullptr;
    ctx->d_sums64  = nullptr;
    ctx->h_matches = ctx->h_mbytes = 0;
    ctx->h_max_len = 0;
    ctx->h_len_sum = ctx->h_dist_sum = ctx->h_total_positions = 0;

    if (use_lz77) {
        const size_t lz_count = (size_t)max_rows * max_row_bytes;
        CHECK_CUDA(cudaMalloc(&ctx->d_lz_len,   sizeof(uint16_t) * lz_count));
        CHECK_CUDA(cudaMalloc(&ctx->d_lz_dist,  sizeof(uint16_t) * lz_count));
        CHECK_CUDA(cudaMalloc(&ctx->d_stats32,   sizeof(uint32_t) * 3));
        CHECK_CUDA(cudaMalloc(&ctx->d_sums64,    sizeof(uint64_t) * 2));
        CHECK_CUDA(cudaMemset(ctx->d_stats32, 0, sizeof(uint32_t) * 3));
        CHECK_CUDA(cudaMemset(ctx->d_sums64,  0, sizeof(uint64_t) * 2));
    }

    CHECK_CUDA(cudaStreamCreate(&ctx->stream));

    CHECK_CUDA(cudaEventCreate(&ctx->ev0));
    CHECK_CUDA(cudaEventCreate(&ctx->ev1));
    CHECK_CUDA(cudaEventCreate(&ctx->ev2));
    CHECK_CUDA(cudaEventCreate(&ctx->ev3));
    CHECK_CUDA(cudaEventCreate(&ctx->ev4));
    ctx->acc_lz77_us = ctx->acc_bitlen_us = ctx->acc_scan_us = ctx->acc_encode_us = 0;
    ctx->acc_sync_us = ctx->acc_stats_d2h_us = 0;

    return ctx;
}

void gpu_deflate_destroy(GpuDeflateContext* ctx)
{
    if (!ctx) return;
    cudaFree(ctx->d_row_bits);
    cudaFree(ctx->d_row_offsets);
    cudaFree(ctx->d_literal_bits);
    cudaFree(ctx->d_running_total);
    cudaFreeHost(ctx->h_running_total);
    cudaFree(ctx->d_strip_base_bit);
    cudaFree(ctx->d_eob_bit_pos);
    cudaFree(ctx->d_block_sums);
    cudaFree(ctx->d_output);
    if (ctx->d_lz_len)  cudaFree(ctx->d_lz_len);
    if (ctx->d_lz_dist) cudaFree(ctx->d_lz_dist);
    if (ctx->d_stats32) cudaFree(ctx->d_stats32);
    if (ctx->d_sums64)  cudaFree(ctx->d_sums64);
    cudaEventDestroy(ctx->ev0);
    cudaEventDestroy(ctx->ev1);
    cudaEventDestroy(ctx->ev2);
    cudaEventDestroy(ctx->ev3);
    cudaEventDestroy(ctx->ev4);
    cudaStreamDestroy(ctx->stream);
    delete ctx;
}

void gpu_deflate_reset(GpuDeflateContext* ctx)
{
    if (!ctx) return;
    // All memsets go into ctx->stream; the next compress_strip call also runs
    // in ctx->stream, so they are ordered by the stream without a host sync.
    *ctx->h_running_total = 0;  // clear host mirror immediately (no data from prev image)
    CHECK_CUDA(cudaMemsetAsync(ctx->d_output, 0, ctx->max_total_bytes, ctx->stream));
    CHECK_CUDA(cudaMemsetAsync(ctx->d_running_total, 0, sizeof(size_t), ctx->stream));
    if (ctx->use_lz77) {
        CHECK_CUDA(cudaMemsetAsync(ctx->d_stats32, 0, sizeof(uint32_t) * 3, ctx->stream));
        CHECK_CUDA(cudaMemsetAsync(ctx->d_sums64,  0, sizeof(uint64_t) * 2, ctx->stream));
    }
    // Reset host-side LZ77 accumulators
    ctx->h_matches = ctx->h_mbytes = 0;
    ctx->h_max_len = 0;
    ctx->h_len_sum = ctx->h_dist_sum = ctx->h_total_positions = 0;
    // Reset phase timing accumulators
    ctx->acc_lz77_us = ctx->acc_bitlen_us = ctx->acc_scan_us = ctx->acc_encode_us = 0;
    ctx->acc_sync_us = ctx->acc_stats_d2h_us = 0;
}

// ---------------------------------------------------------------------------
// Kernel A: per-row bit lengths, literal-only
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
// Kernel A': per-row bit lengths, LZ77 tokens
// ---------------------------------------------------------------------------
__global__ void row_bitlen_kernel_lz(
    const uint8_t*  __restrict__ d_filtered,
    const uint16_t* __restrict__ d_lz_len,
    const uint16_t* __restrict__ d_lz_dist,
    int row_bytes, int actual_rows,
    size_t* __restrict__ d_row_bits)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= actual_rows) return;

    const uint8_t*  src = d_filtered + (size_t)row * row_bytes;
    const uint16_t* llen = d_lz_len  + (size_t)row * row_bytes;
    const uint16_t* ldist = d_lz_dist + (size_t)row * row_bytes;
    size_t total = 0;

    for (int i = 0; i < row_bytes; i++) {
        const uint16_t tlen = llen[i];
        if (tlen == 0) continue;            // inside a match — skip
        if (tlen == 1) {                    // literal
            total += kFixedHuff[src[i]].len;
        } else {                            // match start (length 3-258)
            const int lc = kLenCodeOf[tlen - 3];     // 0-28 (offset from 257)
            total += kFixedHuff[257 + lc].len;        // length code Huffman bits
            total += kLenExtra[lc];                   // extra bits for length
            const int dc = dist_to_code((int)ldist[i]);
            total += 5;                               // fixed distance always 5 bits
            total += kDistExtra[dc];                  // extra bits for distance
        }
    }
    d_row_bits[row] = total;
}

// ---------------------------------------------------------------------------
// Kernel B – 3-pass multi-block exclusive prefix scan
//
// Pass B1: each block of 1024 threads does a local Hillis-Steele inclusive
//          scan, writes local exclusive offsets and the block total.
// Pass B2: single block scans the per-block totals → block offsets + grand total.
// Pass B3: each element adds its block's offset.
//
// Supports any actual_rows up to max_rows, with no hard ceiling as long as
// ceil(actual_rows/1024) <= 1024 (i.e., actual_rows <= 1,048,576).
// ---------------------------------------------------------------------------
__global__ void row_partial_scan_kernel(
    const size_t* __restrict__ d_row_bits,
    size_t* __restrict__       d_row_offsets,
    size_t* __restrict__       d_block_sums,
    int actual_rows)
{
    extern __shared__ size_t sdata[];
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    const size_t val = (gid < actual_rows) ? d_row_bits[gid] : 0;
    sdata[tid] = val;
    __syncthreads();

    // Hillis-Steele inclusive scan (blockDim.x must be a power of 2)
    for (int offset = 1; offset < (int)blockDim.x; offset <<= 1) {
        const size_t add = (tid >= offset) ? sdata[tid - offset] : 0;
        __syncthreads();
        sdata[tid] += add;
        __syncthreads();
    }

    if (tid == blockDim.x - 1)
        d_block_sums[blockIdx.x] = sdata[tid];

    if (gid < actual_rows)
        d_row_offsets[gid] = sdata[tid] - val;  // exclusive = inclusive - own
}

__global__ void block_sum_scan_kernel(
    size_t* __restrict__ d_block_sums,
    size_t* __restrict__ d_literal_bits_out,
    int num_blocks)
{
    extern __shared__ size_t sdata[];
    const int tid = threadIdx.x;

    const size_t val = (tid < num_blocks) ? d_block_sums[tid] : 0;
    sdata[tid] = val;
    __syncthreads();

    for (int offset = 1; offset < (int)blockDim.x; offset <<= 1) {
        const size_t add = (tid >= offset) ? sdata[tid - offset] : 0;
        __syncthreads();
        sdata[tid] += add;
        __syncthreads();
    }

    if (tid == blockDim.x - 1)
        *d_literal_bits_out = sdata[tid];  // grand total

    // Convert each block's inclusive sum to exclusive (= offset for that block)
    if (tid < num_blocks)
        d_block_sums[tid] = sdata[tid] - val;
}

__global__ void row_offset_add_kernel(
    size_t* __restrict__       d_row_offsets,
    const size_t* __restrict__ d_block_sums,
    int actual_rows)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= actual_rows) return;
    d_row_offsets[gid] += d_block_sums[blockIdx.x];
}

// ---------------------------------------------------------------------------
// Kernel C: single-thread block header + bookkeeping (unchanged)
// ---------------------------------------------------------------------------
__global__ void deflate_header_and_bookkeeping_kernel(
    uint8_t* __restrict__        g_out,
    const size_t* __restrict__   d_literal_bits,
    size_t* __restrict__         d_running_total,
    size_t* __restrict__         d_strip_base_bit,
    size_t* __restrict__         d_eob_bit_pos,
    int is_last)
{
    const size_t   base  = *d_running_total;
    const uint32_t hdr   = is_last ? 0x3u : 0x2u;
    splice_field(g_out, base, hdr, 3);

    const size_t strip_base = base + 3;
    const size_t lit        = *d_literal_bits;
    const size_t eob_pos    = strip_base + lit;

    *d_strip_base_bit = strip_base;
    *d_eob_bit_pos    = eob_pos;
    *d_running_total  = eob_pos + 7;
}

// ---------------------------------------------------------------------------
// Kernel D: per-row encode, literal-only (unchanged)
// ---------------------------------------------------------------------------
__global__ void row_encode_splice_kernel(
    const uint8_t* __restrict__ d_filtered,
    int row_bytes, int actual_rows,
    const size_t* __restrict__ d_row_offsets,
    const size_t* __restrict__ d_strip_base_bit,
    uint8_t* __restrict__ g_out)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= actual_rows) return;

    const uint8_t* src = d_filtered + (size_t)row * row_bytes;
    size_t global_bit  = *d_strip_base_bit + d_row_offsets[row];

    uint32_t acc      = 0;
    int      acc_bits = 0;

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
// Kernel D': per-row encode, LZ77 tokens
// ---------------------------------------------------------------------------
__global__ void row_encode_splice_kernel_lz(
    const uint8_t*  __restrict__ d_filtered,
    const uint16_t* __restrict__ d_lz_len,
    const uint16_t* __restrict__ d_lz_dist,
    int row_bytes, int actual_rows,
    const size_t* __restrict__ d_row_offsets,
    const size_t* __restrict__ d_strip_base_bit,
    uint8_t* __restrict__ g_out)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= actual_rows) return;

    const uint8_t*  src   = d_filtered + (size_t)row * row_bytes;
    const uint16_t* llen  = d_lz_len   + (size_t)row * row_bytes;
    const uint16_t* ldist = d_lz_dist  + (size_t)row * row_bytes;
    size_t global_bit = *d_strip_base_bit + d_row_offsets[row];

    uint32_t acc      = 0;
    int      acc_bits = 0;

// Emit `nbits` bits of `value` (LSB-first, standard accumulator pattern)
#define EMIT(value, nbits)                                              \
    do {                                                                \
        uint32_t _v = (value) & ((nbits) >= 32 ? 0xFFFFFFFFu           \
                                               : ((1u << (nbits))-1u));\
        acc |= (_v << acc_bits);                                        \
        acc_bits += (nbits);                                            \
        while (acc_bits >= 8) {                                         \
            splice_field(g_out, global_bit, acc & 0xFFu, 8);           \
            global_bit += 8;                                            \
            acc       >>= 8;                                            \
            acc_bits   -= 8;                                            \
        }                                                               \
    } while (0)

    for (int i = 0; i < row_bytes; i++) {
        const uint16_t tlen = llen[i];
        if (tlen == 0) continue;    // inside match — no bits

        if (tlen == 1) {
            // Literal
            const HuffEntry e = kFixedHuff[src[i]];
            EMIT(e.code, (int)e.len);
        } else {
            // Match: emit length code + extra + distance code + extra
            const int len  = (int)tlen;
            const int dist = (int)ldist[i];

            // Length code
            const int lc = kLenCodeOf[len - 3];   // offset from 257
            const HuffEntry le = kFixedHuff[257 + lc];
            EMIT(le.code, (int)le.len);
            // Length extra bits
            const int l_extra_bits = kLenExtra[lc];
            if (l_extra_bits > 0) {
                const int l_extra_val = len - (int)kLenBase[lc];
                EMIT((uint32_t)l_extra_val, l_extra_bits);
            }
            // Distance code (always 5 bits in fixed Huffman)
            const int dc = dist_to_code(dist);
            EMIT((uint32_t)kDistCode[dc], 5);
            // Distance extra bits
            const int d_extra_bits = kDistExtra[dc];
            if (d_extra_bits > 0) {
                const int d_extra_val = dist - (int)kDistBase[dc];
                EMIT((uint32_t)d_extra_val, d_extra_bits);
            }
        }
    }
#undef EMIT

    if (acc_bits > 0)
        splice_field(g_out, global_bit, acc & 0xFFu, acc_bits);
}

// ---------------------------------------------------------------------------
// Kernel E: write EOB code (unchanged)
// ---------------------------------------------------------------------------
__global__ void write_small_bits_from_ptr_kernel(
    uint8_t* g_out, const size_t* d_bit_offset, uint32_t value, int nbits)
{
    splice_field(g_out, *d_bit_offset, value, nbits);
}

// ---------------------------------------------------------------------------
// LZ77 sequential row kernel
//
// Grid:  (actual_rows, 1) — one block per row
// Block: (32, 1) — one warp; all 32 threads initialize the shared hash table,
//        then only thread 0 performs the sequential scan.
// Shared memory: SEQ_HASH_SIZE × sizeof(uint32_t) = 4 KB per block (1024 buckets).
//
// This is the CORRECT approach for GPU LZ77: by processing each row sequentially
// from left to right, we guarantee:
//   1. Every hash table lookup returns a position strictly less than the current
//      position (no forward references to filter out).
//   2. The greedy parse is done inline — no separate pass needed.
//   3. Rows are independent and run in parallel across the block grid.
//
// Hash strategy (LZ4-style, single-entry per bucket):
//   s_hash[h] = most recent row-relative position with hash h.
//   Sentinel 0xFFFFFFFF = empty bucket.
//   uint32_t handles any row_bytes up to 4 GB.
//
// After this kernel d_lz_len/d_lz_dist are ready for Kernels A' and D'.
//   d_lz_len[pos] == 0  → inside a match (skip)
//   d_lz_len[pos] == 1  → literal
//   d_lz_len[pos] >= 3  → match start of that length
//
// Stats are accumulated into d_stats32 and d_sums64 using one set of atomics
// per row (not per match), keeping contention low.
// ---------------------------------------------------------------------------
__global__ void lz77_row_kernel(
    const uint8_t*  __restrict__ d_strip,
    uint16_t* __restrict__       d_lz_len,
    uint16_t* __restrict__       d_lz_dist,
    int  row_bytes,
    int  actual_rows,
    uint32_t* __restrict__ d_stats32,   // [3]: matches_found, matched_bytes, max_len
    uint64_t* __restrict__ d_sums64     // [2]: total_len_sum, total_dist_sum
)
{
    const int row = blockIdx.x;
    if (row >= actual_rows) return;

    // Shared hash table: 1024 × uint32_t = 4 KB per block
    __shared__ uint32_t s_hash[SEQ_HASH_SIZE];

    // Parallel init: each of the 32 threads clears 32 entries
    for (int i = threadIdx.x; i < SEQ_HASH_SIZE; i += blockDim.x)
        s_hash[i] = 0xFFFFFFFFu;
    __syncthreads();

    if (threadIdx.x != 0) return;   // only thread 0 scans

    const uint8_t* src    = d_strip   + (size_t)row * row_bytes;
    uint16_t*      o_len  = d_lz_len  + (size_t)row * row_bytes;
    uint16_t*      o_dist = d_lz_dist + (size_t)row * row_bytes;

    uint32_t row_matches  = 0;
    uint32_t row_mbytes   = 0;
    uint32_t row_max_len  = 0;
    uint64_t row_len_sum  = 0;
    uint64_t row_dist_sum = 0;

    int pos = 0;
    while (pos < row_bytes) {
        // Need at least LZ_MIN_MATCH bytes remaining to form a 3-byte hash
        if (pos + LZ_MIN_MATCH > row_bytes) {
            o_len[pos] = 1;
            pos++;
            continue;
        }

        // Knuth multiplicative hash over 4 bytes (3-byte fallback near row end)
        uint32_t v;
        if (pos + 4 <= row_bytes) {
            memcpy(&v, src + pos, 4);
        } else {
            // Only 3 bytes available (pos+2 is guaranteed valid above)
            v = (uint32_t)src[pos]
              | ((uint32_t)src[pos + 1] << 8)
              | ((uint32_t)src[pos + 2] << 16);
        }
        const uint32_t h = (v * 2654435761u) >> (32 - SEQ_HASH_BITS);

        const uint32_t cand = s_hash[h];   // most recent occupant of this bucket
        s_hash[h] = (uint32_t)pos;          // update: this position takes the slot

        if (cand == 0xFFFFFFFFu) {
            o_len[pos] = 1;
            pos++;
            continue;
        }

        // Sequential scan guarantees cand < pos (no forward-reference check needed)
        const int dist = (int)pos - (int)cand;
        if (dist > LZ_MAX_DIST) {
            o_len[pos] = 1;
            pos++;
            continue;
        }

        // Quick 3-byte content check (all three positions are guaranteed in-bounds
        // since pos + LZ_MIN_MATCH <= row_bytes, and cand < pos < row_bytes)
        if (src[cand]     != src[pos]     ||
            src[cand + 1] != src[pos + 1] ||
            src[cand + 2] != src[pos + 2]) {
            // Hash collision: bucket holds a different 3-byte sequence
            o_len[pos] = 1;
            pos++;
            continue;
        }

        // Extend match as far as possible up to LZ_MAX_MATCH or row boundary.
        // Bounds proof: pos+len < row_bytes (len < max_len_here = row_bytes-pos),
        //               cand+len < pos+len < row_bytes (because cand < pos).
        const int max_len_here = (row_bytes - pos < LZ_MAX_MATCH)
                                 ? (row_bytes - pos) : LZ_MAX_MATCH;
        int len = LZ_MIN_MATCH;
        while (len < max_len_here && src[cand + len] == src[pos + len])
            len++;

        // Write match token and mark continuation bytes as 0 (skip)
        o_len[pos]  = (uint16_t)len;
        o_dist[pos] = (uint16_t)dist;
        for (int k = 1; k < len; k++) o_len[pos + k] = 0;

        // Accumulate per-row stats in registers (avoids one global atomic per match)
        row_matches++;
        row_mbytes   += (uint32_t)len;
        if ((uint32_t)len > row_max_len) row_max_len = (uint32_t)len;
        row_len_sum  += (uint64_t)len;
        row_dist_sum += (uint64_t)dist;

        pos += len;
    }

    // One global atomic per counter per row (not per match) — low contention
    if (row_matches > 0) {
        atomicAdd(&d_stats32[0], row_matches);
        atomicAdd(&d_stats32[1], row_mbytes);
        atomicMax(&d_stats32[2], row_max_len);
        atomicAdd((unsigned long long*)&d_sums64[0], (unsigned long long)row_len_sum);
        atomicAdd((unsigned long long*)&d_sums64[1], (unsigned long long)row_dist_sum);
    }
}

// ---------------------------------------------------------------------------
// Internal: launch the 3-pass prefix scan (replaces single-block B kernel)
// ---------------------------------------------------------------------------
static void run_prefix_scan(GpuDeflateContext* ctx, int actual_rows)
{
    constexpr int SCAN_BLOCK = 1024;
    const int num_scan_blocks = (actual_rows + SCAN_BLOCK - 1) / SCAN_BLOCK;

    if (num_scan_blocks > 1024) {
        fprintf(stderr,
            "gpu_deflate: actual_rows=%d requires %d scan blocks which exceeds "
            "the 1024-block limit for the second-level scan. "
            "Reduce --strip-height to at most 1,048,576 rows.\n",
            actual_rows, num_scan_blocks);
        exit(1);
    }

    // B1: local scans, one block of 1024 threads each
    row_partial_scan_kernel<<<num_scan_blocks, SCAN_BLOCK,
                              SCAN_BLOCK * sizeof(size_t), ctx->stream>>>(
        ctx->d_row_bits, ctx->d_row_offsets, ctx->d_block_sums, actual_rows);

    // B2: scan the block totals (single block; round up to power of 2)
    int b2_threads = 1;
    while (b2_threads < num_scan_blocks) b2_threads <<= 1;
    block_sum_scan_kernel<<<1, b2_threads,
                            b2_threads * sizeof(size_t), ctx->stream>>>(
        ctx->d_block_sums, ctx->d_literal_bits, num_scan_blocks);

    // B3: add block offsets to per-row offsets
    row_offset_add_kernel<<<num_scan_blocks, SCAN_BLOCK, 0, ctx->stream>>>(
        ctx->d_row_offsets, ctx->d_block_sums, actual_rows);
}

// ---------------------------------------------------------------------------
// Public: compress one strip
// ---------------------------------------------------------------------------
void gpu_deflate_compress_strip(GpuDeflateContext* ctx,
                                const uint8_t* d_filtered, int actual_rows,
                                int row_bytes, bool is_last)
{
    using Clock = std::chrono::high_resolution_clock;
    const int threads = 256;
    const int blocks  = (actual_rows + threads - 1) / threads;

    // --- ev0: start of all kernel work ---
    CHECK_CUDA(cudaEventRecord(ctx->ev0, ctx->stream));

    if (ctx->use_lz77) {
        // d_stats32 and d_sums64 accumulate across all strips of the image;
        // they are zeroed once in gpu_deflate_reset() (not per-strip), so
        // the atomics in the kernel naturally sum up over the full image.
        // A single D2H in gpu_deflate_lz_stats() reads the final totals.

        // LZ: one block per row, thread 0 does the sequential scan.
        // Grid = (actual_rows, 1), Block = (32, 1), shared=4 KB (1024 buckets).
        // NOTE: only 1/32 threads per block does actual work (thread utilization 3.1%).
        lz77_row_kernel<<<actual_rows, 32, 0, ctx->stream>>>(
            d_filtered, ctx->d_lz_len, ctx->d_lz_dist,
            row_bytes, actual_rows, ctx->d_stats32, ctx->d_sums64);
    }

    // --- ev1: after LZ77 (or same time as ev0 for literal path) ---
    CHECK_CUDA(cudaEventRecord(ctx->ev1, ctx->stream));

    if (ctx->use_lz77) {
        // A': compute per-row bit lengths using LZ77 tokens
        row_bitlen_kernel_lz<<<blocks, threads, 0, ctx->stream>>>(
            d_filtered, ctx->d_lz_len, ctx->d_lz_dist,
            row_bytes, actual_rows, ctx->d_row_bits);
    } else {
        // A: literal-only bit lengths
        row_bitlen_kernel<<<blocks, threads, 0, ctx->stream>>>(
            d_filtered, row_bytes, actual_rows, ctx->d_row_bits);
    }

    // --- ev2: after bitlen kernel A/A' ---
    CHECK_CUDA(cudaEventRecord(ctx->ev2, ctx->stream));

    // B: 3-pass prefix scan (supports any actual_rows)
    run_prefix_scan(ctx, actual_rows);

    // --- ev3: after prefix scan B ---
    CHECK_CUDA(cudaEventRecord(ctx->ev3, ctx->stream));

    // C: block header + bit-position bookkeeping
    deflate_header_and_bookkeeping_kernel<<<1, 1, 0, ctx->stream>>>(
        ctx->d_output, ctx->d_literal_bits, ctx->d_running_total,
        ctx->d_strip_base_bit, ctx->d_eob_bit_pos, is_last ? 1 : 0);

    if (ctx->use_lz77) {
        // D': encode using LZ77 tokens
        row_encode_splice_kernel_lz<<<blocks, threads, 0, ctx->stream>>>(
            d_filtered, ctx->d_lz_len, ctx->d_lz_dist,
            row_bytes, actual_rows,
            ctx->d_row_offsets, ctx->d_strip_base_bit, ctx->d_output);
    } else {
        // D: literal-only encode
        row_encode_splice_kernel<<<blocks, threads, 0, ctx->stream>>>(
            d_filtered, row_bytes, actual_rows,
            ctx->d_row_offsets, ctx->d_strip_base_bit, ctx->d_output);
    }

    // E: EOB code (7 bits, value 0)
    write_small_bits_from_ptr_kernel<<<1, 1, 0, ctx->stream>>>(
        ctx->d_output, ctx->d_eob_bit_pos, 0u, 7);

    // --- ev4: after all encode+EOB kernels C+D+E ---
    CHECK_CUDA(cudaEventRecord(ctx->ev4, ctx->stream));

    // Shadow d_running_total to the pinned host mirror so that after
    // gpu_deflate_stream_sync() returns the caller can read h_running_total
    // without an extra D2H round-trip (Phase 3 optimization: eliminates S4 sync).
    CHECK_CUDA(cudaMemcpyAsync(ctx->h_running_total, ctx->d_running_total,
                               sizeof(size_t), cudaMemcpyDeviceToHost, ctx->stream));

    // No host sync here. Call gpu_deflate_stream_sync() after launching any
    // concurrent work (e.g., adler32) in another stream, then sync there.
    // Buffer-reuse safety (filter not overwriting d_filtered): guaranteed by the
    // BoundedQueue capacity (pool_n-2) — the filter cannot reuse this strip's
    // context until compress has popped at least pool_n-2 more strips, which only
    // happens after gpu_deflate_stream_sync() has returned and all reads finished.

    // Accumulate host-side position count (arithmetic only, no D2H).
    if (ctx->use_lz77) {
        ctx->h_total_positions += (uint64_t)actual_rows * (uint64_t)row_bytes;
    }
}

// ---------------------------------------------------------------------------
// Sync the deflate stream and collect per-phase kernel timings.
// Must be called exactly once per gpu_deflate_compress_strip() call, before:
//   - reading d_output / d_running_total on the host, and
//   - launching the next strip's work in any stream that reads d_output.
// ---------------------------------------------------------------------------
void gpu_deflate_stream_sync(GpuDeflateContext* ctx)
{
    using Clock = std::chrono::high_resolution_clock;
    auto h_sync_t0 = Clock::now();
    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));
    auto h_sync_t1 = Clock::now();

    // Query per-phase GPU elapsed times (requires all events to be recorded,
    // which the stream sync above guarantees).
    float ms01 = 0.f, ms12 = 0.f, ms23 = 0.f, ms34 = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms01, ctx->ev0, ctx->ev1));  // LZ77
    CHECK_CUDA(cudaEventElapsedTime(&ms12, ctx->ev1, ctx->ev2));  // bitlen
    CHECK_CUDA(cudaEventElapsedTime(&ms23, ctx->ev2, ctx->ev3));  // scan
    CHECK_CUDA(cudaEventElapsedTime(&ms34, ctx->ev3, ctx->ev4));  // encode+EOB
    ctx->acc_lz77_us   += (long long)(ms01 * 1000.f);
    ctx->acc_bitlen_us += (long long)(ms12 * 1000.f);
    ctx->acc_scan_us   += (long long)(ms23 * 1000.f);
    ctx->acc_encode_us += (long long)(ms34 * 1000.f);
    ctx->acc_sync_us   += std::chrono::duration_cast<std::chrono::microseconds>(
                              h_sync_t1 - h_sync_t0).count();
}

// ---------------------------------------------------------------------------
// Output accessors (unchanged)
// ---------------------------------------------------------------------------
const uint8_t* gpu_deflate_output(const GpuDeflateContext* ctx)
{
    return ctx->d_output;
}

// h_running_total is updated by the last cudaMemcpyAsync queued in
// gpu_deflate_compress_strip (after ev4). After gpu_deflate_stream_sync()
// the value is host-visible; no further D2H or sync is required.
size_t gpu_deflate_output_byte_length(const GpuDeflateContext* ctx)
{
    return (*ctx->h_running_total + 7) / 8;
}

size_t gpu_deflate_flushable_byte_length(const GpuDeflateContext* ctx)
{
    return *ctx->h_running_total / 8;
}

void gpu_deflate_copy_to_host(const GpuDeflateContext* ctx, uint8_t* h_dst,
                              size_t num_bytes)
{
    CHECK_CUDA(cudaMemcpyAsync(h_dst, ctx->d_output, num_bytes,
                               cudaMemcpyDeviceToHost, ctx->stream));
    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));
}

// ---------------------------------------------------------------------------
// Phase statistics accessors
// ---------------------------------------------------------------------------
GpuDeflatePhaseStats gpu_deflate_get_phase_stats(const GpuDeflateContext* ctx)
{
    GpuDeflatePhaseStats s{};
    if (!ctx) return s;
    s.lz77_us      = ctx->acc_lz77_us;
    s.bitlen_us    = ctx->acc_bitlen_us;
    s.scan_us      = ctx->acc_scan_us;
    s.encode_us    = ctx->acc_encode_us;
    s.sync_us      = ctx->acc_sync_us;
    s.stats_d2h_us = ctx->acc_stats_d2h_us;
    return s;
}

void gpu_deflate_reset_phase_stats(GpuDeflateContext* ctx)
{
    if (!ctx) return;
    ctx->acc_lz77_us = ctx->acc_bitlen_us = ctx->acc_scan_us = ctx->acc_encode_us = 0;
    ctx->acc_sync_us = ctx->acc_stats_d2h_us = 0;
}

// ---------------------------------------------------------------------------
// LZ77 statistics accessor
// ---------------------------------------------------------------------------
LzStats gpu_deflate_lz_stats(const GpuDeflateContext* ctx)
{
    LzStats s{};
    if (!ctx || !ctx->use_lz77) return s;

    // Single D2H at end of image (28 bytes). Stream is already synced by the
    // gpu_deflate_copy_to_host() call that precedes this in the pipeline.
    uint32_t h_s32[3] = {0, 0, 0};
    uint64_t h_s64[2] = {0, 0};
    cudaMemcpy(h_s32, ctx->d_stats32, sizeof(h_s32), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_s64, ctx->d_sums64,  sizeof(h_s64), cudaMemcpyDeviceToHost);

    s.matches_found    = (uint64_t)h_s32[0];
    s.matched_bytes    = (uint64_t)h_s32[1];
    s.max_match_len    = h_s32[2];
    s.total_positions  = ctx->h_total_positions;
    s.coverage         = (s.total_positions > 0)
                         ? (double)s.matched_bytes / (double)s.total_positions : 0.0;
    s.avg_match_len    = (s.matches_found > 0)
                         ? (double)h_s64[0] / (double)s.matches_found : 0.0;
    s.avg_match_dist   = (s.matches_found > 0)
                         ? (double)h_s64[1] / (double)s.matches_found : 0.0;
    return s;
}
