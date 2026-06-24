#pragma once
// gpu_jls_backend.h
// GPU-accelerated JPEG-LS encoder — Architecture JLS-GPU / JLS-C
//
// Five phases (G1–G5):
//
//   G1  jls_residual_kernel       – parallel LOCO-I prediction + residual + context
//                                    (one thread per pixel, fully independent)
//   G2  jls_golomb_row_scan_kernel – Golomb-Rice k scan (one thread per row,
//                                    sequential within row, parallel across rows)
//                                    JLS-C1: ROW_CARRY mode threads row N state
//                                    into row N+1, producing spec-compliant
//                                    context propagation and materially smaller
//                                    output vs the prior ROW_RESET mode.
//   G3  jls_bitlen_kernel          – parallel bit-length computation from k+merrval
//   G4  3-pass prefix scan         – partial → block_sum → add_offset
//   G5  jls_emit_kernel            – parallel MSB-first Golomb-Rice bit emission
//
// CPU wrapper then adds JPEG-LS byte stuffing (0xFF → 0xFF 0x00) and markers.
//
// Phase timings (CUDA Events) are accumulated per-image and queryable via
// gpu_jls_get_timings().
//
// Opaque handle: no CUDA types in this header — plain .cpp can include it.

#include <cstddef>
#include <cstdint>

struct GpuJlsContext;

// Context carry-over mode for G2 (Golomb-k scan).
//
//   ROW_RESET  – Legacy: context A/N/B/C reset to spec defaults at each row start.
//                Allows all rows to run in parallel. Produces ~104 KB avg output.
//                Non-standard JPEG-LS (row-context reset).
//
//   ROW_CARRY  – JLS-C1: context state is carried row-to-row. Thread 0 of row N
//                writes its final A/N/B/C to a device buffer; row N+1 reads it as
//                its initial state. Rows must execute sequentially (one kernel
//                launch per row). Produces spec-compliant context propagation
//                and materially smaller output. Target: ≤ 70 KB avg.
//
enum class GpuJlsMode : int {
    ROW_RESET = 0,  // original behaviour (default for backwards compat)
    ROW_CARRY = 1,  // JLS-C1: row-to-row context carry (collapsed to ~30fps, archived)
    CPU_RM    = 2,  // JLS-C1 v3: GPU for G1 only; CPU does G2+run-mode+emit
                    // Produces ISO 14495-1 run-mode coding → target ~50KB avg output
};

// Per-image phase timings in microseconds (GPU wall-clock via CUDA Events).
struct GpuJlsTimings {
    float h2d_us      = 0.f;  // H2D pixel transfer
    float g1_us       = 0.f;  // G1: residual + context kernel
    float g2_us       = 0.f;  // G2: row Golomb-k scan (all rows combined)
    float g3_us       = 0.f;  // G3: bit-length kernel
    float g4_us       = 0.f;  // G4: prefix scan (3 passes)
    float g5_us       = 0.f;  // G5: bit-emission kernel
    float d2h_us      = 0.f;  // D2H bitstream transfer
    float cpu_wrap_us = 0.f;  // CPU byte-stuffing + header write (host clock)
};

// Allocate a context sized for images up to max_width × max_height pixels.
// mode selects the G2 context carry strategy (default: ROW_RESET for
// backwards compatibility; use ROW_CARRY for JLS-C1 production mode).
GpuJlsContext* gpu_jls_create(int max_width, int max_height,
                               GpuJlsMode mode = GpuJlsMode::ROW_RESET);
void           gpu_jls_destroy(GpuJlsContext* ctx);

// Encode one 8-bit grayscale image to a JPEG-LS-like bitstream and write to
// out_path.  h_pixels is row-major, width × height bytes.
// Returns true on success; ctx accumulates timing stats on each call.
bool gpu_jls_encode(GpuJlsContext* ctx,
                    const uint8_t* h_pixels,
                    int width, int height,
                    const char* out_path);

// Read and reset the accumulated timings from the last gpu_jls_encode call.
GpuJlsTimings gpu_jls_get_timings(const GpuJlsContext* ctx);
void          gpu_jls_reset_timings(GpuJlsContext* ctx);
