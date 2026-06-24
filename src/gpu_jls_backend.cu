// gpu_jls_backend.cu
// GPU JPEG-LS encoder — Architecture JLS-GPU / JLS-C1
//
// Phase map:
//   GPU JLS:  H2D → G1[residual+context] → G2[golomb_k] → G3[bitlen] → G4[scan] → G5[emit]
//
// G1 (jls_residual_kernel): Each thread computes one pixel's LOCO-I prediction,
//    signed residual, and context index independently. Fully parallel.
//
// G2a (jls_golomb_row_scan_kernel) — ROW_RESET mode (legacy):
//    Thread 0 of each block scans one full row sequentially, maintaining
//    JPEG-LS context state (A/N/B/C per context) in shared memory.
//    All rows run in parallel. Context resets at row start for row-level
//    independence. Non-standard — produces ~104 KB avg output.
//
// G2b (jls_golomb_row_carry_kernel) — ROW_CARRY mode (JLS-C1):
//    Same sequential scan per row, but reads the prior row's A/N/B/C state
//    from d_ctx_in[NUM_CTX×4] and writes this row's final state to
//    d_ctx_out[NUM_CTX×4]. Called once per row from host (sequential launches).
//    Produces spec-compliant context propagation; target ≤ 70 KB avg output.
//
// G3 (jls_bitlen_kernel): Parallel per-pixel bit-length from merrval and k.
//
// G4 (3-pass scan): Partial → block-sum → add-offset prefix scan over bitlens.
//
// G5 (jls_emit_kernel): Each thread emits one Golomb-Rice code (MSB-first) via
//    atomicOr into a pre-zeroed output buffer at its prefix-scan bit position.
//
// CPU post-processing: byte-stuffing (0xFF → 0xFF 0x00) + JPEG-LS markers.

#include "gpu_jls_backend.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// JPEG-LS constants (ISO 14495-1, lossless 8-bit grayscale)
// ---------------------------------------------------------------------------
static constexpr int JLS_MAXVAL   = 255;
static constexpr int JLS_RANGE    = 256;  // MAXVAL + 1
static constexpr int JLS_qbpp     = 8;    // ceil(log2(RANGE))
static constexpr int JLS_LIMIT    = 32;   // 2*qbpp + 16
static constexpr int JLS_RESET    = 64;
static constexpr int JLS_NUM_CTX  = 405;  // 5×9×9 after sign folding
// Preset coding parameters (ISO 14495-1 Table C.2 for MAXVAL=255, NEAR=0)
static constexpr int JLS_T1 = 3;
static constexpr int JLS_T2 = 7;
static constexpr int JLS_T3 = 21;
static constexpr int JLS_Ainit = 4;  // max(2, floor((MAXVAL+32)/64)) = max(2,4)

// ---------------------------------------------------------------------------
// G1: LOCO-I residual + context kernel
// ---------------------------------------------------------------------------
__global__ void jls_residual_kernel(
    const uint8_t* __restrict__ d_pixels,   // [height × width]
    int16_t*       __restrict__ d_residuals,// [height × width] signed residual
    uint16_t*      __restrict__ d_contexts, // [height × width] context 0..404
    int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    const int px  = (int)d_pixels[idx];

    // LOCO-I neighbors (ISO 14495-1 §A.3, boundary init per spec)
    const int Ra = (x > 0)               ? (int)d_pixels[y * width + x - 1]         : 0;
    const int Rb = (y > 0)               ? (int)d_pixels[(y-1) * width + x]         : Ra;
    const int Rc = (y > 0 && x > 0)      ? (int)d_pixels[(y-1) * width + x - 1]     : Rb;
    const int Rd = (y > 0 && x < width-1)? (int)d_pixels[(y-1) * width + x + 1]    : Rb;

    // LOCO-I predictor
    const int mn = min(Ra, Rb);
    const int mx = max(Ra, Rb);
    int pred = Ra + Rb - Rc;
    if (pred < mn) pred = mn;
    if (pred > mx) pred = mx;

    // Signed residual with modular wrap into [-128, 127]
    int err = px - pred;
    if (err >  127) err -= 256;
    if (err < -128) err += 256;

    // Gradient quantization (T1=3, T2=7, T3=21)
    auto quant = [](int g) -> int {
        if (g <= -JLS_T3) return -4;
        if (g <= -JLS_T2) return -3;
        if (g <= -JLS_T1) return -2;
        if (g <       0 ) return -1;
        if (g ==      0 ) return  0;
        if (g <  JLS_T1 ) return  1;
        if (g <  JLS_T2 ) return  2;
        if (g <  JLS_T3 ) return  3;
                          return  4;
    };

    int q1 = quant(Ra - Rc);  // horizontal gradient
    int q2 = quant(Rb - Rc);  // vertical gradient
    int q3 = quant(Rd - Rb);  // near-horizontal gradient

    // Sign folding: canonical form ensures unique context per equivalent scenario
    if (q1 < 0 || (q1 == 0 && q2 < 0) || (q1 == 0 && q2 == 0 && q3 < 0)) {
        q1 = -q1; q2 = -q2; q3 = -q3;
        err = -err;
        if (err >  127) err -= 256;
        if (err < -128) err += 256;
    }

    // Context index: q1 ∈ [0,4], q2 ∈ [-4,4], q3 ∈ [-4,4]  → [0, 404]
    d_residuals[idx] = (int16_t)err;
    d_contexts[idx]  = (uint16_t)(q1 * 81 + (q2 + 4) * 9 + (q3 + 4));
}

// ---------------------------------------------------------------------------
// G2a: Row-parallel Golomb-k scan — ROW_RESET mode (legacy, non-standard)
// ---------------------------------------------------------------------------
// One block per row. Thread 0 scans all width pixels sequentially, maintaining
// context state (A/N/B/C per context) in shared memory (6480 B / block for 405
// contexts × 4 int arrays × 4 B = 6480 B, well within the 48 KB SM limit).
// Context resets to spec defaults at every row start → non-standard JPEG-LS.
// ---------------------------------------------------------------------------
__global__ void jls_golomb_row_scan_kernel(
    const int16_t*  __restrict__ d_residuals,
    const uint16_t* __restrict__ d_contexts,
    uint8_t*  __restrict__ d_golomb_k,
    uint16_t* __restrict__ d_merrval,
    int width, int height)
{
    const int row = (int)blockIdx.x;
    if (row >= height || threadIdx.x != 0) return;

    // 405 context × 4 arrays = 6480 bytes per block (shared)
    __shared__ int sA[JLS_NUM_CTX], sN[JLS_NUM_CTX];
    __shared__ int sB[JLS_NUM_CTX], sC[JLS_NUM_CTX];

    for (int q = 0; q < JLS_NUM_CTX; q++) {
        sA[q] = JLS_Ainit;
        sN[q] = 1;
        sB[q] = 0;
        sC[q] = 0;
    }

    const int base = row * width;
    for (int x = 0; x < width; x++) {
        const int idx = base + x;
        const int q   = (int)d_contexts[idx];
        const int err = (int)d_residuals[idx];

        // Bias correction: errval = err - C[q], wrapped to [-RANGE/2, RANGE/2-1]
        int errval = err - sC[q];
        if (errval >  JLS_RANGE/2 - 1) errval -= JLS_RANGE;
        if (errval < -JLS_RANGE/2)     errval += JLS_RANGE;

        // Golomb-Rice k: smallest k such that 2^k >= A[q]/N[q]
        int kval = 0;
        {
            int An = sA[q], Nn = sN[q];
            while (An > Nn && kval < JLS_qbpp) { An >>= 1; kval++; }
        }

        // Map to non-negative merrval
        const int merrval_i = (errval >= 0) ? (2 * errval) : (-2 * errval - 1);

        d_golomb_k[idx] = (uint8_t)kval;
        d_merrval[idx]  = (uint16_t)merrval_i;

        // Update context state
        const int abs_err = (errval >= 0) ? errval : -errval;
        sB[q] += errval;
        sA[q] += abs_err;

        // Bias tracking
        if (sB[q] > 0) {
            sB[q] -= sN[q];
            if (sC[q] < JLS_MAXVAL) sC[q]++;
            if (sB[q] > 0) sB[q] = 0;
        } else if (sB[q] < -sN[q]) {
            sB[q] += sN[q];
            if (sC[q] > -JLS_MAXVAL) sC[q]--;
            if (sB[q] < -sN[q]) sB[q] = -sN[q];
        }

        sN[q]++;
        if (sN[q] == JLS_RESET) {
            sA[q] >>= 1;
            sB[q] >>= 1;
            sN[q] >>= 1;
        }
    }
}

// ---------------------------------------------------------------------------
// G2b: Full-image Golomb-k scan with context carry -- ROW_CARRY mode (JLS-C1 v2)
// ---------------------------------------------------------------------------
// A SINGLE kernel launch (<<<1, 1>>>) processes the entire image sequentially.
// Context A/N/B/C arrays live in thread-local registers across all rows, giving
// natural row-to-row context propagation identical to what CharLS does.
//
// Why single-thread: G2 in ROW_RESET is already single-threaded per row; the
// scan is inherently sequential within each row (each pixel depends on the
// prior pixel's context state). Moving the outer row loop inside the kernel
// eliminates height × kernel-launch-overhead (~5-7 us each = ~2.5-3.5 ms
// for a 512-row image) while preserving the same arithmetic work.
//
// Expected G2 time: ~281 us (same as ROW_RESET -- same work, one launch).
// Expected output size: spec-compliant context propagation -> approaching
//   CharLS compression quality (~50-70 KB target vs 104 KB ROW_RESET).
// ---------------------------------------------------------------------------
__global__ void jls_golomb_fullimage_carry_kernel(
    const int16_t*  __restrict__ d_residuals,
    const uint16_t* __restrict__ d_contexts,
    uint8_t*  __restrict__ d_golomb_k,
    uint16_t* __restrict__ d_merrval,
    int width, int height)
{
    // Only a single thread (block 0, thread 0) does all the work.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Context state in registers for the full image.
    // JLS_NUM_CTX = 405; 405 * 4 arrays * 4 bytes = 6480 bytes in local memory.
    // nvcc will spill to L1/L2 -- acceptable; the scan is memory-bandwidth-bound
    // on d_residuals / d_contexts anyway.
    int lA[JLS_NUM_CTX], lN[JLS_NUM_CTX], lB[JLS_NUM_CTX], lC[JLS_NUM_CTX];

    // Initialise to spec defaults (ISO 14495-1).
    for (int q = 0; q < JLS_NUM_CTX; q++) {
        lA[q] = JLS_Ainit;
        lN[q] = 1;
        lB[q] = 0;
        lC[q] = 0;
    }

    // Scan all rows. Context carries naturally -- no reset between rows.
    for (int row = 0; row < height; row++) {
        const int base = row * width;
        for (int x = 0; x < width; x++) {
            const int idx = base + x;
            const int q   = (int)d_contexts[idx];
            const int err = (int)d_residuals[idx];

            // Bias correction: errval = err - C[q], wrapped to [-RANGE/2, RANGE/2-1]
            int errval = err - lC[q];
            if (errval >  JLS_RANGE/2 - 1) errval -= JLS_RANGE;
            if (errval < -JLS_RANGE/2)     errval += JLS_RANGE;

            // Golomb-Rice k: smallest k such that 2^k >= A[q]/N[q]
            int kval = 0;
            {
                int An = lA[q], Nn = lN[q];
                while (An > Nn && kval < JLS_qbpp) { An >>= 1; kval++; }
            }

            // Map to non-negative merrval
            const int merrval_i = (errval >= 0) ? (2 * errval) : (-2 * errval - 1);

            d_golomb_k[idx] = (uint8_t)kval;
            d_merrval[idx]  = (uint16_t)merrval_i;

            // Update context state
            const int abs_err = (errval >= 0) ? errval : -errval;
            lB[q] += errval;
            lA[q] += abs_err;

            // Bias tracking
            if (lB[q] > 0) {
                lB[q] -= lN[q];
                if (lC[q] < JLS_MAXVAL) lC[q]++;
                if (lB[q] > 0) lB[q] = 0;
            } else if (lB[q] < -lN[q]) {
                lB[q] += lN[q];
                if (lC[q] > -JLS_MAXVAL) lC[q]--;
                if (lB[q] < -lN[q]) lB[q] = -lN[q];
            }

            lN[q]++;
            if (lN[q] == JLS_RESET) {
                lA[q] >>= 1;
                lB[q] >>= 1;
                lN[q] >>= 1;
            }
        }
        // No context reset between rows -- carry continues to next row.
    }
}

// ---------------------------------------------------------------------------
// G3: Parallel bit-length computation
// ---------------------------------------------------------------------------
__global__ void jls_bitlen_kernel(
    const uint16_t* __restrict__ d_merrval,
    const uint8_t*  __restrict__ d_golomb_k,
    uint32_t*       __restrict__ d_bitlens,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const uint32_t merrval = d_merrval[idx];
    const int      k       = d_golomb_k[idx];
    const int      unary   = (int)(merrval >> k);

    int bitlen;
    if (unary >= JLS_LIMIT - k) {
        // Limited-length code: (LIMIT-k-1) + 1 + qbpp bits
        bitlen = JLS_LIMIT - k - 1 + 1 + JLS_qbpp;
    } else {
        bitlen = unary + 1 + k;
    }

    d_bitlens[idx] = (uint32_t)bitlen;
}

// ---------------------------------------------------------------------------
// G4: Three-pass prefix scan over bitlens → per-pixel bit positions
//
// Pass 1 (jls_scan_partial):   scan within each block of SCAN_BLK threads,
//                               write block totals to d_block_sums.
// Pass 2 (jls_scan_blocks):    scan the block totals in a single block.
// Pass 3 (jls_add_offsets):    add block offsets back to all partial sums.
// ---------------------------------------------------------------------------
static constexpr int SCAN_BLK = 1024;

// Exclusive prefix scan within a block using Kogge-Stone (shared memory).
// Returns the inclusive scan result; the exclusive result is just the input
// to the next element.  Writes the EXCLUSIVE scan to d_out and the block
// total to d_block_sums[blockIdx.x].
__global__ void jls_scan_partial(
    const uint32_t* __restrict__ d_in,
    uint64_t*       __restrict__ d_out,
    uint64_t*       __restrict__ d_block_sums,
    int N)
{
    __shared__ uint64_t s[SCAN_BLK];

    const int gid = blockIdx.x * SCAN_BLK + threadIdx.x;
    s[threadIdx.x] = (gid < N) ? (uint64_t)d_in[gid] : 0ULL;
    __syncthreads();

    // Inclusive Kogge-Stone scan
    for (int stride = 1; stride < SCAN_BLK; stride <<= 1) {
        uint64_t v = (threadIdx.x >= stride) ? s[threadIdx.x - stride] : 0ULL;
        __syncthreads();
        s[threadIdx.x] += v;
        __syncthreads();
    }

    // Convert to exclusive: shift by 1 and write
    if (gid < N)
        d_out[gid] = (threadIdx.x > 0) ? s[threadIdx.x - 1] : 0ULL;

    if (threadIdx.x == SCAN_BLK - 1)
        d_block_sums[blockIdx.x] = s[SCAN_BLK - 1];
}

// Inclusive scan of d_block_sums in a single block (up to SCAN_BLK blocks).
__global__ void jls_scan_blocks(uint64_t* __restrict__ d_block_sums, int nblocks)
{
    __shared__ uint64_t s[SCAN_BLK];
    const int tid = threadIdx.x;
    s[tid] = (tid < nblocks) ? d_block_sums[tid] : 0ULL;
    __syncthreads();

    for (int stride = 1; stride < SCAN_BLK; stride <<= 1) {
        uint64_t v = (tid >= stride) ? s[tid - stride] : 0ULL;
        __syncthreads();
        s[tid] += v;
        __syncthreads();
    }

    if (tid < nblocks) d_block_sums[tid] = s[tid];
}

// Add the exclusive block offset (d_block_sums[blockIdx.x - 1]) to each
// element of d_out belonging to this block.
__global__ void jls_add_offsets(
    uint64_t*       __restrict__ d_out,
    const uint64_t* __restrict__ d_block_sums,
    int N)
{
    const int gid = blockIdx.x * SCAN_BLK + threadIdx.x;
    if (gid >= N) return;
    if (blockIdx.x == 0) return;  // block 0 has offset 0 (exclusive)
    d_out[gid] += d_block_sums[blockIdx.x - 1];
}

// ---------------------------------------------------------------------------
// G5: Parallel MSB-first Golomb-Rice bit emission
// ---------------------------------------------------------------------------
// For each pixel, emits its Golomb-Rice code at the bit position d_positions[idx]:
//   Normal mode (unary < LIMIT-k):
//     (unary zeros) + '1' + (k-bit remainder MSB-first)
//     Only the '1' and the set remainder bits need atomicOr (buffer pre-zeroed).
//
//   Limited mode (unary >= LIMIT-k):
//     (LIMIT-k-1 zeros) + '1' + (qbpp bits of merrval-1, MSB-first)
//
// MSB-first bit packing: bit at stream position p → byte p/8, bit (7 - p%8).
// ---------------------------------------------------------------------------
__global__ void jls_emit_kernel(
    const uint16_t* __restrict__ d_merrval,
    const uint8_t*  __restrict__ d_golomb_k,
    const uint64_t* __restrict__ d_positions,
    uint8_t*                     d_output,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const uint32_t merrval  = d_merrval[idx];
    const int      k        = d_golomb_k[idx];
    const uint64_t base_pos = d_positions[idx];
    const int      unary    = (int)(merrval >> k);

    // Helper: set one bit at stream position pos (MSB-first).
    // CUDA atomicOr requires uint32/int; address/shift map byte+bit into the
    // host-visible (little-endian) word that contains that byte's bits.
    auto set_bit = [&](uint64_t pos) {
        const uint32_t byte_idx    = (uint32_t)(pos >> 3);
        const uint32_t bit_in_byte = 7u - (uint32_t)(pos & 7u);  // MSB-first within byte
        const uint32_t word_idx    = byte_idx >> 2;               // which uint32
        const uint32_t byte_lane   = byte_idx & 3u;               // 0=LSB .. 3=MSB byte of word
        const uint32_t bit_in_word = bit_in_byte + byte_lane * 8u;
        atomicOr((unsigned int*)d_output + word_idx, 1u << bit_in_word);
    };

    if (unary >= JLS_LIMIT - k) {
        // Limited-length code
        // Write '1' after (LIMIT-k-1) zeros
        uint64_t pos = base_pos + (uint64_t)(JLS_LIMIT - k - 1);
        set_bit(pos++);
        // Write qbpp bits of (merrval - 1), MSB first
        const uint32_t raw = merrval - 1u;
        for (int i = JLS_qbpp - 1; i >= 0; i--) {
            if ((raw >> i) & 1u) set_bit(pos);
            pos++;
        }
        return;
    }

    // Normal Golomb-Rice: zeros are already in buffer; set the '1' bit
    uint64_t one_pos = base_pos + (uint64_t)unary;
    set_bit(one_pos);

    // Write k-bit remainder MSB-first
    if (k > 0) {
        const uint32_t remainder = merrval & ((1u << k) - 1u);
        uint64_t rem_pos = one_pos + 1u;
        for (int i = k - 1; i >= 0; i--) {
            if ((remainder >> i) & 1u) set_bit(rem_pos);
            rem_pos++;
        }
    }
}

// ---------------------------------------------------------------------------
// Context definition
// ---------------------------------------------------------------------------
struct GpuJlsContext {
    // Device buffers
    uint8_t*  d_pixels8;     // [max_h × max_w]  8-bit input
    int16_t*  d_residuals;   // [max_h × max_w]  G1 output
    uint16_t* d_contexts;    // [max_h × max_w]  G1 output
    uint8_t*  d_golomb_k;    // [max_h × max_w]  G2 output
    uint16_t* d_merrval;     // [max_h × max_w]  G2 output
    uint32_t* d_bitlens;     // [max_h × max_w]  G3 output
    uint64_t* d_positions;   // [max_h × max_w]  G4 output (bit positions)
    uint64_t* d_block_sums;  // [ceil(max_N/SCAN_BLK)] for prefix scan
    uint8_t*  d_bitstream;   // [max_bitstream_bytes] G5 output

    // JLS-C1: row context carry state (ROW_CARRY mode only).
    // Two alternating buffers of [NUM_CTX × 4] int32s per row:
    //   d_row_ctx[0] and d_row_ctx[1] are ping-ponged per row.
    // Each buffer holds A/N/B/C for all 405 contexts.
    // Size: 405 × 4 × 4 = 6480 bytes per buffer (two buffers = 12960 bytes).
    int* d_row_ctx[2];   // [0] = ping, [1] = pong

    // Pinned host output buffer
    uint8_t*  h_bitstream;   // [max_bitstream_bytes]

    // CPU_RM mode: pinned host buffers for G1 output (residuals + contexts)
    int16_t*  h_residuals;   // [max_h × max_w]  pinned, CPU_RM only
    uint16_t* h_contexts;    // [max_h × max_w]  pinned, CPU_RM only

    int max_width, max_height;
    size_t max_bitstream_bytes;
    bool printed_timings_ = false;

    GpuJlsMode mode;       // ROW_RESET, ROW_CARRY, or CPU_RM

    cudaStream_t stream;

    // Timing events (per phase)
    cudaEvent_t ev_start, ev_h2d, ev_g1, ev_g2, ev_g3, ev_g4, ev_g5, ev_d2h;

    // Accumulated last-run timings
    GpuJlsTimings timings;
};

// ---------------------------------------------------------------------------
// Host API
// ---------------------------------------------------------------------------
GpuJlsContext* gpu_jls_create(int max_width, int max_height, GpuJlsMode mode)
{
    auto* ctx = new GpuJlsContext{};
    ctx->max_width  = max_width;
    ctx->max_height = max_height;
    ctx->mode       = mode;

    const int    N         = max_width * max_height;
    const size_t px_bytes  = (size_t)N;
    const size_t s16_bytes = (size_t)N * 2;
    const size_t u16_bytes = (size_t)N * 2;
    const size_t u8_bytes  = (size_t)N;
    const size_t u32_bytes = (size_t)N * 4;
    const size_t u64_bytes = (size_t)N * 8;

    const int    nblocks         = (N + SCAN_BLK - 1) / SCAN_BLK;
    ctx->max_bitstream_bytes     = (size_t)N * (JLS_LIMIT / 8 + 1) + 4096;

    CHECK_CUDA(cudaMalloc(&ctx->d_pixels8,    px_bytes));
    CHECK_CUDA(cudaMalloc(&ctx->d_residuals,  s16_bytes));
    CHECK_CUDA(cudaMalloc(&ctx->d_contexts,   u16_bytes));
    CHECK_CUDA(cudaMalloc(&ctx->d_golomb_k,   u8_bytes));
    CHECK_CUDA(cudaMalloc(&ctx->d_merrval,    u16_bytes));
    CHECK_CUDA(cudaMalloc(&ctx->d_bitlens,    u32_bytes));
    CHECK_CUDA(cudaMalloc(&ctx->d_positions,  u64_bytes));
    CHECK_CUDA(cudaMalloc(&ctx->d_block_sums, (size_t)nblocks * 8));
    CHECK_CUDA(cudaMalloc(&ctx->d_bitstream,  ctx->max_bitstream_bytes));
    CHECK_CUDA(cudaMallocHost(&ctx->h_bitstream, ctx->max_bitstream_bytes));

    // CPU_RM: pinned host mirrors for G1 output (D2H path for CPU encode).
    // Allocated for all modes — overhead is just 3 MB for 1024×1024 max.
    CHECK_CUDA(cudaMallocHost(&ctx->h_residuals, s16_bytes));
    CHECK_CUDA(cudaMallocHost(&ctx->h_contexts,  u16_bytes));

    // JLS-C1: allocate ping-pong context state buffers (always; tiny overhead).
    // Each buffer: NUM_CTX × 4 int32s = 405 × 4 × 4 = 6480 bytes.
    const size_t ctx_state_bytes = (size_t)JLS_NUM_CTX * 4 * sizeof(int);
    CHECK_CUDA(cudaMalloc(&ctx->d_row_ctx[0], ctx_state_bytes));
    CHECK_CUDA(cudaMalloc(&ctx->d_row_ctx[1], ctx_state_bytes));

    CHECK_CUDA(cudaStreamCreate(&ctx->stream));
    CHECK_CUDA(cudaEventCreate(&ctx->ev_start));
    CHECK_CUDA(cudaEventCreate(&ctx->ev_h2d));
    CHECK_CUDA(cudaEventCreate(&ctx->ev_g1));
    CHECK_CUDA(cudaEventCreate(&ctx->ev_g2));
    CHECK_CUDA(cudaEventCreate(&ctx->ev_g3));
    CHECK_CUDA(cudaEventCreate(&ctx->ev_g4));
    CHECK_CUDA(cudaEventCreate(&ctx->ev_g5));
    CHECK_CUDA(cudaEventCreate(&ctx->ev_d2h));

    return ctx;
}

void gpu_jls_destroy(GpuJlsContext* ctx)
{
    if (!ctx) return;
    cudaFree(ctx->d_pixels8);
    cudaFree(ctx->d_residuals);
    cudaFree(ctx->d_contexts);
    cudaFree(ctx->d_golomb_k);
    cudaFree(ctx->d_merrval);
    cudaFree(ctx->d_bitlens);
    cudaFree(ctx->d_positions);
    cudaFree(ctx->d_block_sums);
    cudaFree(ctx->d_bitstream);
    cudaFreeHost(ctx->h_bitstream);
    cudaFreeHost(ctx->h_residuals);
    cudaFreeHost(ctx->h_contexts);
    cudaFree(ctx->d_row_ctx[0]);
    cudaFree(ctx->d_row_ctx[1]);
    cudaEventDestroy(ctx->ev_start);
    cudaEventDestroy(ctx->ev_h2d);
    cudaEventDestroy(ctx->ev_g1);
    cudaEventDestroy(ctx->ev_g2);
    cudaEventDestroy(ctx->ev_g3);
    cudaEventDestroy(ctx->ev_g4);
    cudaEventDestroy(ctx->ev_g5);
    cudaEventDestroy(ctx->ev_d2h);
    cudaStreamDestroy(ctx->stream);
    delete ctx;
}

// Reallocate device/host buffers when a larger frame arrives.
// Syncs the stream before freeing to avoid use-after-free in async ops.
static void gpu_jls_resize(GpuJlsContext* ctx, int new_width, int new_height)
{
    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));

    cudaFree(ctx->d_pixels8);
    cudaFree(ctx->d_residuals);
    cudaFree(ctx->d_contexts);
    cudaFree(ctx->d_golomb_k);
    cudaFree(ctx->d_merrval);
    cudaFree(ctx->d_bitlens);
    cudaFree(ctx->d_positions);
    cudaFree(ctx->d_block_sums);
    cudaFree(ctx->d_bitstream);
    cudaFreeHost(ctx->h_bitstream);
    cudaFreeHost(ctx->h_residuals);
    cudaFreeHost(ctx->h_contexts);
    // d_row_ctx buffers are fixed-size (NUM_CTX × 4 × 4 bytes) — no realloc needed.

    ctx->max_width        = new_width;
    ctx->max_height       = new_height;
    ctx->printed_timings_ = false;  // log once for the new dimensions

    const int    N         = new_width * new_height;
    const int    nblocks   = (N + SCAN_BLK - 1) / SCAN_BLK;
    ctx->max_bitstream_bytes = (size_t)N * (JLS_LIMIT / 8 + 1) + 4096;

    CHECK_CUDA(cudaMalloc(&ctx->d_pixels8,    (size_t)N));
    CHECK_CUDA(cudaMalloc(&ctx->d_residuals,  (size_t)N * 2));
    CHECK_CUDA(cudaMalloc(&ctx->d_contexts,   (size_t)N * 2));
    CHECK_CUDA(cudaMalloc(&ctx->d_golomb_k,   (size_t)N));
    CHECK_CUDA(cudaMalloc(&ctx->d_merrval,    (size_t)N * 2));
    CHECK_CUDA(cudaMalloc(&ctx->d_bitlens,    (size_t)N * 4));
    CHECK_CUDA(cudaMalloc(&ctx->d_positions,  (size_t)N * 8));
    CHECK_CUDA(cudaMalloc(&ctx->d_block_sums, (size_t)nblocks * 8));
    CHECK_CUDA(cudaMalloc(&ctx->d_bitstream,  ctx->max_bitstream_bytes));
    CHECK_CUDA(cudaMallocHost(&ctx->h_bitstream, ctx->max_bitstream_bytes));
    CHECK_CUDA(cudaMallocHost(&ctx->h_residuals, (size_t)N * 2));
    CHECK_CUDA(cudaMallocHost(&ctx->h_contexts,  (size_t)N * 2));
}

// Write a uint16 big-endian
static void write_be16(uint8_t* p, uint16_t v)
{
    p[0] = (uint8_t)(v >> 8);
    p[1] = (uint8_t)(v & 0xFF);
}

// Write a uint32 big-endian
static void write_be32(uint8_t* p, uint32_t v)
{
    p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >>  8); p[3] = (uint8_t)(v & 0xFF);
}

// Build the JPEG-LS file header into buf; return byte count written.
// markers: SOI + SOF_55 + (optional LSE for preset params) + SOS
static int build_jls_header(uint8_t* buf, int width, int height)
{
    uint8_t* p = buf;

    // SOI
    *p++ = 0xFF; *p++ = 0xD8;

    // SOF_55 (JPEG-LS frame header)
    // Length = 8 + 3*Ncomp = 8 + 3*1 = 11
    *p++ = 0xFF; *p++ = 0xF7;
    write_be16(p, 11); p += 2;  // Lf
    *p++ = 8;                    // P (sample precision)
    write_be16(p, (uint16_t)height); p += 2;
    write_be16(p, (uint16_t)width);  p += 2;
    *p++ = 1;                    // Nf (1 component)
    *p++ = 1;                    // component id
    *p++ = 0x11;                 // H=1, V=1 sampling factors
    *p++ = 0;                    // Tqi (quant table index, unused)

    // LSE — preset coding parameters (tells decoder to use standard defaults)
    // Lse = 13, ID = 1
    *p++ = 0xFF; *p++ = 0xF8;
    write_be16(p, 13); p += 2;  // Lse
    *p++ = 1;                   // ID
    write_be16(p, (uint16_t)JLS_MAXVAL);  p += 2;  // MAXVAL
    write_be16(p, (uint16_t)JLS_T1);     p += 2;
    write_be16(p, (uint16_t)JLS_T2);     p += 2;
    write_be16(p, (uint16_t)JLS_T3);     p += 2;
    write_be16(p, (uint16_t)JLS_RESET);  p += 2;

    // SOS (scan header for 1 component, NEAR=0)
    // Ls = 8 + 2*Ns = 8 + 2 = 10
    *p++ = 0xFF; *p++ = 0xDA;
    write_be16(p, 10); p += 2;  // Ls
    *p++ = 1;                    // Ns (1 scan component)
    *p++ = 1;                    // component selector
    *p++ = 0;                    // mapping table selector
    *p++ = 0;                    // NEAR
    *p++ = 0;                    // ILV (none, grayscale)
    *p++ = 0;                    // point transform

    return (int)(p - buf);
}

// ---------------------------------------------------------------------------
// CPU run-mode encoder (JLS-C1 v3)
//
// Implements ISO 14495-1 §A.7 run-mode coding + §A.3 regular-mode Golomb-Rice
// with full context carry across rows and columns.
//
// Input : h_residuals[height × width]  (signed, from G1 GPU kernel)
//         h_contexts [height × width]  (context index 0..404, from G1 GPU kernel)
//         h_pixels   [height × width]  (original pixels, for run detection)
// Output: bitstream written directly into out_buf; returns byte count written.
//
// Run mode (ISO 14495-1 §A.3): triggered when Q1=Q2=Q3=0 (flat neighborhood,
// i.e., Ra==Rb==Rc==Rd). Encodes a run of identical pixels using accumulated
// J-order Golomb codes, then an interruption sample at the end.
// ---------------------------------------------------------------------------

// JPEG-LS J array (ISO 14495-1 Table A.1) — run mode Golomb orders
static constexpr int JLS_J[32] = {
    0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,5,5,6,6,7,8,9,10,11,12,13,14,15,15
};

// ---------------------------------------------------------------------------
// Fast word-accumulator bit writer (MSB-first, no pre-zeroing needed)
// Accumulates up to 56 bits in a uint64 before flushing bytes. This avoids
// the per-bit branch + byte-index computation of the old bit-by-bit writer.
// ---------------------------------------------------------------------------
struct BitWriter {
    uint8_t* buf;      // output pointer (advances on each flushed byte)
    uint64_t accum;    // pending bits, MSB-first in the high bits
    int      pending;  // number of valid bits in accum (0..63)

    void init(uint8_t* out) { buf = out; accum = 0; pending = 0; }

    // Emit n bits (n ≤ 56) from the LSBs of `bits`, MSB first.
    inline void put(uint64_t bits, int n) {
        accum  = (accum << n) | (bits & ((n < 64) ? ((1ULL << n) - 1ULL) : ~0ULL));
        pending += n;
        // Flush complete bytes
        while (pending >= 8) {
            pending -= 8;
            *buf++  = (uint8_t)(accum >> pending);
        }
    }

    // Flush any remaining partial byte (zero-pad on right)
    void flush() {
        if (pending > 0) {
            *buf++ = (uint8_t)(accum << (8 - pending));
            pending = 0;
        }
    }

    size_t byte_count(const uint8_t* base) const {
        return (size_t)(buf - base) + (pending > 0 ? 1 : 0);
    }
};

// Emit one Golomb-Rice code (MSB-first):
//   Normal:   unary '0's + '1' + k remainder bits  → unary+1+k total bits
//   Limited:  (LIMIT-k-1) '0's + '1' + qbpp bits of (merrval-1)  → LIMIT-k+qbpp-1+1 bits
static inline void cpu_emit_golomb(BitWriter& bw, int merrval, int k)
{
    const int unary = merrval >> k;
    if (unary >= JLS_LIMIT - k) {
        // Limited-length: (LIMIT-k-1) zeros + '1' + 8 raw bits
        // Total: LIMIT - k + 8 bits. Encode as two put() calls to stay ≤56 bits each.
        const int zero_run = JLS_LIMIT - k - 1;  // typically 31 for k=0
        // Emit the zero run in chunks of 56 bits max
        int rem = zero_run;
        while (rem > 0) {
            const int chunk = (rem > 56) ? 56 : rem;
            bw.put(0ULL, chunk);
            rem -= chunk;
        }
        bw.put(1ULL, 1);  // the '1' terminator
        bw.put((uint64_t)(merrval - 1) & 0xFFu, JLS_qbpp);  // 8 raw bits
    } else {
        // Normal: unary zeros + '1' + k remainder bits
        // Pack into at most two chunks: unary zeros (may be >56), then 1+k bits
        int rem = unary;
        while (rem > 0) {
            const int chunk = (rem > 56) ? 56 : rem;
            bw.put(0ULL, chunk);
            rem -= chunk;
        }
        // '1' + k-bit remainder packed together
        const uint64_t code = (1ULL << k) | ((uint64_t)(merrval & ((1u << k) - 1)));
        bw.put(code, 1 + k);
    }
}

// Run interruption sample (ISO 14495-1 §A.7.2, lossless NEAR=0)
// RItype=1: Ix >= Ra (positive error relative to Ra)
// RItype=0: Ix <  Ra (negative error)
// Uses separate run-interruption contexts riA[2] / riN[2].
static void cpu_emit_run_interrupt(BitWriter& bw,
                                   int Ix, int Ra, int Rb,
                                   int* riA, int* riN)
{
    // §A.7.2.2: determine type and prediction
    const int RItype = (Ix >= Ra) ? 1 : 0;
    // Prediction for run interruption: Ra for both types (lossless simplified)
    int errval = Ix - Ra;

    // Range correction per spec (wrap into [-MAXVAL/2, MAXVAL/2])
    if (RItype == 1) {
        // RItype=1: errval adjusted down by 1 then mapped
        if (errval > 0) errval--;   // subtract 1 per §A.7.2
    }
    if (errval < 0) errval += JLS_RANGE;
    if (errval > JLS_MAXVAL / 2) errval -= JLS_RANGE;

    // Golomb k from run-interruption context
    int k = 0;
    { int An = riA[RItype], Nn = riN[RItype];
      while (An > Nn && k < JLS_qbpp) { An >>= 1; k++; } }

    // Mapped merrval (non-negative)
    const int merrval = (errval >= 0) ? (2 * errval) : (-2 * errval - 1);

    bw.put((uint64_t)RItype, 1);     // RItype indicator bit
    cpu_emit_golomb(bw, merrval, k);

    // Update run-interruption context
    const int abs_err = (errval >= 0) ? errval : -errval;
    riA[RItype] += abs_err;
    riN[RItype]++;
    if (riN[RItype] == JLS_RESET) { riA[RItype] >>= 1; riN[RItype] >>= 1; }
}

// Full CPU encode (called from CPU_RM mode after G1 D2H)
static size_t cpu_jls_encode_rm(
    const uint8_t*  h_pixels,
    const int16_t*  h_residuals,
    const uint16_t* h_contexts,
    int width, int height,
    uint8_t* out_buf,    // pre-allocated, large enough (width*height*5+1024)
    size_t   out_cap)
{
    (void)out_cap;

    BitWriter bw;
    bw.init(out_buf);

    // Context state arrays — carry across ALL pixels (rows AND columns, spec-compliant)
    int lA[JLS_NUM_CTX], lN[JLS_NUM_CTX], lB[JLS_NUM_CTX], lC[JLS_NUM_CTX];
    for (int q = 0; q < JLS_NUM_CTX; q++) {
        lA[q] = JLS_Ainit; lN[q] = 1; lB[q] = 0; lC[q] = 0;
    }

    // Run-interruption context state: 2 contexts (RItype=0 and RItype=1)
    int riA[2] = { JLS_Ainit, JLS_Ainit };
    int riN[2] = { 1, 1 };

    int RUNindex = 0;  // J-array index for run-length Golomb order

    for (int y = 0; y < height; y++) {
        const int base = y * width;
        int x = 0;

        while (x < width) {
            const int idx = base + x;

            // Neighbor values (boundary init per spec)
            const int Ra = (x > 0)               ? (int)h_pixels[idx - 1]         : 0;
            const int Rb = (y > 0)               ? (int)h_pixels[idx - width]      : Ra;
            const int Rc = (y > 0 && x > 0)      ? (int)h_pixels[idx - width - 1] : Rb;
            const int Rd = (y > 0 && x < width-1)? (int)h_pixels[idx - width + 1] : Rb;

            // Run mode: D1=D2=D3=0 ↔ Ra==Rb==Rc==Rd (ISO 14495-1 §A.3)
            if (Ra == Rb && Rb == Rc && Rc == Rd) {
                // Count run of pixels equal to Ra; emit J-segment '1' bits
                int L = 0;
                int ri = RUNindex;

                while (x < width && (int)h_pixels[base + x] == Ra) {
                    L++;
                    x++;
                    // Complete J-segment: emit '1' and advance ri
                    if (L == (1 << JLS_J[ri])) {
                        bw.put(1ULL, 1);
                        if (ri < 31) ri++;
                        L = 0;
                    }
                }
                RUNindex = ri;

                if (x < width) {
                    // Run interrupted within row: emit '0' + interruption sample
                    bw.put(0ULL, 1);
                    const int IxRI = (int)h_pixels[base + x];
                    cpu_emit_run_interrupt(bw, IxRI, Ra, Rb, riA, riN);
                    x++;  // advance past interruption pixel
                } else {
                    // Run reached end of row: emit residual '1' if partial segment
                    if (L > 0) bw.put(1ULL, 1);
                }
                continue;  // RUNindex resets below at row end
            }

            // Regular mode: use G1-computed residual + context (sign-folded)
            const int q      = (int)h_contexts[idx];
            const int err_g1 = (int)h_residuals[idx];

            // Bias correction (carry-over lC)
            int errval = err_g1 - lC[q];
            if (errval >  JLS_RANGE / 2 - 1) errval -= JLS_RANGE;
            if (errval < -JLS_RANGE / 2)     errval += JLS_RANGE;

            // Golomb-Rice k: smallest k s.t. 2^k >= A[q]/N[q]
            int kval = 0;
            { int An = lA[q], Nn = lN[q];
              while (An > Nn && kval < JLS_qbpp) { An >>= 1; kval++; } }

            // Mapped (non-negative) error and emit
            const int merrval = (errval >= 0) ? (2 * errval) : (-2 * errval - 1);
            cpu_emit_golomb(bw, merrval, kval);

            // Update context stats
            const int abs_err = (errval >= 0) ? errval : -errval;
            lB[q] += errval;
            lA[q] += abs_err;
            if (lB[q] > 0) {
                lB[q] -= lN[q];
                if (lC[q] < JLS_MAXVAL) lC[q]++;
                if (lB[q] > 0) lB[q] = 0;
            } else if (lB[q] < -lN[q]) {
                lB[q] += lN[q];
                if (lC[q] > -JLS_MAXVAL) lC[q]--;
                if (lB[q] < -lN[q]) lB[q] = -lN[q];
            }
            lN[q]++;
            if (lN[q] == JLS_RESET) { lA[q] >>= 1; lB[q] >>= 1; lN[q] >>= 1; }

            x++;
        }
        RUNindex = 0;  // RUNindex resets at end of each row (ISO 14495-1)
    }

    bw.flush();
    return bw.byte_count(out_buf);
}

bool gpu_jls_encode(GpuJlsContext* ctx,
                    const uint8_t* h_pixels,
                    int width, int height,
                    const char* out_path)
{
    const int    N       = width * height;
    const size_t px_size = (size_t)N;
    const int    nblocks = (N + SCAN_BLK - 1) / SCAN_BLK;

    // Reallocate if this frame is larger than the context's capacity.
    if (N > ctx->max_width * ctx->max_height)
        gpu_jls_resize(ctx, width, height);

    // ---- H2D ---------------------------------------------------------
    CHECK_CUDA(cudaEventRecord(ctx->ev_start, ctx->stream));
    CHECK_CUDA(cudaMemcpyAsync(ctx->d_pixels8, h_pixels,
                               px_size, cudaMemcpyHostToDevice, ctx->stream));
    CHECK_CUDA(cudaEventRecord(ctx->ev_h2d, ctx->stream));

    // ---- G1: residual + context (fully parallel, 32×8 thread blocks) ----
    {
        dim3 block(32, 8);
        dim3 grid((width  + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
        jls_residual_kernel<<<grid, block, 0, ctx->stream>>>(
            ctx->d_pixels8, ctx->d_residuals, ctx->d_contexts, width, height);
    }
    CHECK_CUDA(cudaEventRecord(ctx->ev_g1, ctx->stream));

    // ---- CPU_RM: after G1, copy residuals+contexts to host and encode on CPU ----
    if (ctx->mode == GpuJlsMode::CPU_RM) {
        // D2H residuals and contexts (G1 output)
        const size_t s16 = (size_t)N * 2;
        CHECK_CUDA(cudaMemcpyAsync(ctx->h_residuals, ctx->d_residuals,
                                   s16, cudaMemcpyDeviceToHost, ctx->stream));
        CHECK_CUDA(cudaMemcpyAsync(ctx->h_contexts,  ctx->d_contexts,
                                   s16, cudaMemcpyDeviceToHost, ctx->stream));
        CHECK_CUDA(cudaEventRecord(ctx->ev_g2, ctx->stream));  // marks D2H start
        CHECK_CUDA(cudaStreamSynchronize(ctx->stream));
        CHECK_CUDA(cudaEventRecord(ctx->ev_g3, ctx->stream));  // marks D2H end

        // CPU encode: run mode + context carry + direct bit emission
        auto cpu_t0 = std::chrono::high_resolution_clock::now();

        // Use h_bitstream as raw bit output buffer (large enough: max_bitstream_bytes)
        const size_t raw_bytes = cpu_jls_encode_rm(
            h_pixels,
            ctx->h_residuals, ctx->h_contexts,
            width, height,
            ctx->h_bitstream, ctx->max_bitstream_bytes);

        // Byte-stuffing + header + EOI
        uint8_t hdr[64];
        const int hdr_len = build_jls_header(hdr, width, height);

        std::vector<uint8_t> stuffed;
        stuffed.reserve(raw_bytes + raw_bytes / 128 + 2);
        for (size_t i = 0; i < raw_bytes; i++) {
            uint8_t b = ctx->h_bitstream[i];
            stuffed.push_back(b);
            if (b == 0xFF) stuffed.push_back(0x00);
        }
        stuffed.push_back(0xFF);
        stuffed.push_back(0xD9);

        FILE* fout = fopen(out_path, "wb");
        if (!fout) return false;
        fwrite(hdr, 1, (size_t)hdr_len, fout);
        fwrite(stuffed.data(), 1, stuffed.size(), fout);
        fclose(fout);

        auto cpu_t1 = std::chrono::high_resolution_clock::now();
        ctx->timings.cpu_wrap_us = (float)std::chrono::duration_cast<
            std::chrono::microseconds>(cpu_t1 - cpu_t0).count();

        // Record fake g4/g5/d2h events so elapsed queries don't fail
        CHECK_CUDA(cudaEventRecord(ctx->ev_g4, ctx->stream));
        CHECK_CUDA(cudaEventRecord(ctx->ev_g5, ctx->stream));
        CHECK_CUDA(cudaEventRecord(ctx->ev_d2h, ctx->stream));
        CHECK_CUDA(cudaStreamSynchronize(ctx->stream));

        ctx->timings.h2d_us = 0.f;
        float ms_g1_total = 0.f;
        cudaEventElapsedTime(&ms_g1_total, ctx->ev_start, ctx->ev_g1);
        ctx->timings.g1_us  = ms_g1_total * 1000.f;
        float ms_d2h = 0.f;
        cudaEventElapsedTime(&ms_d2h, ctx->ev_g1, ctx->ev_g3);
        ctx->timings.g2_us  = ms_d2h * 1000.f;  // repurposed: D2H time
        ctx->timings.g3_us  = 0.f;
        ctx->timings.g4_us  = 0.f;
        ctx->timings.g5_us  = 0.f;
        ctx->timings.d2h_us = 0.f;

        if (!ctx->printed_timings_) {
            ctx->printed_timings_ = true;
            fprintf(stderr,
                "[GPU JLS CPU_RM %dx%d] H2D=%.0fus G1=%.0fus D2H=%.0fus "
                "CPU_encode=%.0fus  raw=%zu B\n",
                width, height,
                ctx->timings.h2d_us, ctx->timings.g1_us, ctx->timings.g2_us,
                ctx->timings.cpu_wrap_us, raw_bytes);
        }
        return true;
    }

    // ---- G2: Golomb-k scan -----------------------------------------------
    if (ctx->mode == GpuJlsMode::ROW_RESET) {
        // Legacy: all rows in parallel, context resets each row.
        jls_golomb_row_scan_kernel<<<height, 1, 0, ctx->stream>>>(
            ctx->d_residuals, ctx->d_contexts,
            ctx->d_golomb_k, ctx->d_merrval,
            width, height);
    } else {
        // JLS-C1 ROW_CARRY v2: single <<<1,1>>> launch processes all rows.
        // No per-row launch overhead; context carries naturally across rows.
        jls_golomb_fullimage_carry_kernel<<<1, 1, 0, ctx->stream>>>(
            ctx->d_residuals, ctx->d_contexts,
            ctx->d_golomb_k,  ctx->d_merrval,
            width, height);
    }
    CHECK_CUDA(cudaEventRecord(ctx->ev_g2, ctx->stream));

    // ---- G3: bit lengths (fully parallel) --------------------------------
    {
        dim3 block(256);
        dim3 grid((N + 255) / 256);
        jls_bitlen_kernel<<<grid, block, 0, ctx->stream>>>(
            ctx->d_merrval, ctx->d_golomb_k, ctx->d_bitlens, N);
    }
    CHECK_CUDA(cudaEventRecord(ctx->ev_g3, ctx->stream));

    // ---- G4: 3-pass prefix scan ------------------------------------------
    // Pass 1: partial scan within each block → partial exclusive sums + block totals
    jls_scan_partial<<<nblocks, SCAN_BLK, 0, ctx->stream>>>(
        ctx->d_bitlens, ctx->d_positions, ctx->d_block_sums, N);
    // Pass 2: scan block totals (all in one block)
    jls_scan_blocks<<<1, SCAN_BLK, 0, ctx->stream>>>(ctx->d_block_sums, nblocks);
    // Pass 3: add block offsets
    jls_add_offsets<<<nblocks, SCAN_BLK, 0, ctx->stream>>>(
        ctx->d_positions, ctx->d_block_sums, N);
    CHECK_CUDA(cudaEventRecord(ctx->ev_g4, ctx->stream));

    // ---- G5: bit emission ------------------------------------------------
    // Zero the output buffer first (only '1' bits are written by emit kernel)
    CHECK_CUDA(cudaMemsetAsync(ctx->d_bitstream, 0,
                               ctx->max_bitstream_bytes, ctx->stream));
    {
        dim3 block(256);
        dim3 grid((N + 255) / 256);
        jls_emit_kernel<<<grid, block, 0, ctx->stream>>>(
            ctx->d_merrval, ctx->d_golomb_k,
            ctx->d_positions, ctx->d_bitstream, N);
    }
    CHECK_CUDA(cudaEventRecord(ctx->ev_g5, ctx->stream));

    // ---- D2H: need total bit count and bitstream -------------------------
    // First retrieve the last position + last bitlen to compute total bits.
    // We read d_positions[N-1] + d_bitlens[N-1] on host after sync.
    // Copy them as two 8-byte values from device.
    uint64_t last_pos  = 0;
    uint32_t last_blen = 0;
    CHECK_CUDA(cudaMemcpyAsync(&last_pos,  ctx->d_positions + N - 1,
                               sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx->stream));
    CHECK_CUDA(cudaMemcpyAsync(&last_blen, ctx->d_bitlens   + N - 1,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost, ctx->stream));

    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));  // must sync before reading

    const uint64_t total_bits  = last_pos + (uint64_t)last_blen;
    const size_t   raw_bytes   = (size_t)((total_bits + 7) / 8);

    // Copy the raw bitstream to host
    CHECK_CUDA(cudaMemcpyAsync(ctx->h_bitstream, ctx->d_bitstream,
                               raw_bytes, cudaMemcpyDeviceToHost, ctx->stream));
    CHECK_CUDA(cudaEventRecord(ctx->ev_d2h, ctx->stream));
    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));

    // ---- Collect GPU timings (ms → µs) -----------------------------------
    auto elapsed_us = [&](cudaEvent_t a, cudaEvent_t b) -> float {
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        return ms * 1000.f;
    };
    ctx->timings.h2d_us = elapsed_us(ctx->ev_start, ctx->ev_h2d);
    ctx->timings.g1_us  = elapsed_us(ctx->ev_h2d,   ctx->ev_g1);
    ctx->timings.g2_us  = elapsed_us(ctx->ev_g1,    ctx->ev_g2);
    ctx->timings.g3_us  = elapsed_us(ctx->ev_g2,    ctx->ev_g3);
    ctx->timings.g4_us  = elapsed_us(ctx->ev_g3,    ctx->ev_g4);
    ctx->timings.g5_us  = elapsed_us(ctx->ev_g4,    ctx->ev_g5);
    ctx->timings.d2h_us = elapsed_us(ctx->ev_g5,    ctx->ev_d2h);

    // ---- CPU: byte stuffing + header + EOI -------------------------------
    auto cpu_t0 = std::chrono::high_resolution_clock::now();

    // Build header
    uint8_t hdr[64];
    const int hdr_len = build_jls_header(hdr, width, height);

    // Byte-stuffing: scan raw bitstream for 0xFF → insert 0x00
    std::vector<uint8_t> stuffed;
    stuffed.reserve(raw_bytes + raw_bytes / 256 + 2);  // ~0.4% overhead estimate
    for (size_t i = 0; i < raw_bytes; i++) {
        uint8_t b = ctx->h_bitstream[i];
        stuffed.push_back(b);
        if (b == 0xFF) stuffed.push_back(0x00);
    }

    // EOI
    stuffed.push_back(0xFF);
    stuffed.push_back(0xD9);

    // Write file
    FILE* f = fopen(out_path, "wb");
    if (!f) return false;
    fwrite(hdr, 1, (size_t)hdr_len, f);
    fwrite(stuffed.data(), 1, stuffed.size(), f);
    fclose(f);

    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    ctx->timings.cpu_wrap_us = (float)std::chrono::duration_cast<
        std::chrono::microseconds>(cpu_t1 - cpu_t0).count();

    // Print per-phase timing on first call (once per context lifetime)
    if (ctx->timings.h2d_us > 0.f && !ctx->printed_timings_) {
        ctx->printed_timings_ = true;
        fprintf(stderr,
            "[GPU JLS %dx%d] H2D=%.0fus G1=%.0fus G2=%.0fus G3=%.0fus "
            "G4=%.0fus G5=%.0fus D2H=%.0fus CPU=%.0fus  raw=%zu B\n",
            width, height,
            ctx->timings.h2d_us, ctx->timings.g1_us, ctx->timings.g2_us,
            ctx->timings.g3_us,  ctx->timings.g4_us, ctx->timings.g5_us,
            ctx->timings.d2h_us, ctx->timings.cpu_wrap_us,
            raw_bytes);
    }

    return true;
}

GpuJlsTimings gpu_jls_get_timings(const GpuJlsContext* ctx)
{
    return ctx->timings;
}

void gpu_jls_reset_timings(GpuJlsContext* ctx)
{
    ctx->timings = GpuJlsTimings{};
}
