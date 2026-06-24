// batch_processor.cpp
// Phase D2: Batch Scheduler (decode-ahead pipeline)
//
// DICOM files use a two-pool architecture:
//   dec_pool (N threads): DicomSource::open + load_frame → PreDecodedFrame queue
//   enc_pool (N threads): pop PreDecodedFrame → encode_source_to_png/bmp
//
// Decode and encode run concurrently. Overlap hides DICOM decode latency
// behind GPU encode time, improving per-file throughput at all worker counts.
//
// TIFF / RAW files continue to use the existing direct encode path (no
// pre-decode step, since those sources don't benefit from it).

#include "batch_processor.h"
#include "bounded_queue.h"
#include "file_type.h"
#include "dicom_loader.h"
#include "image_loader.h"   // tiff_open/tiff_read_strip/tiff_close, raw_open/raw_read_strip/raw_close
#include "image_source.h"
#include "thread_pool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <intrin.h>   // _mm_prefetch for software-prefetch LUT optimization
#include <climits>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::milliseconds;

// ---------------------------------------------------------------------------
// Phase D2: pre-decoded DICOM frame produced by the decode pool.
// ---------------------------------------------------------------------------
struct PreDecodedFrame {
    std::string      input_path;
    std::string      output_path;
    std::string      filename;
    ImageInfo        info;
    DicomPixelParams params;
    std::vector<uint8_t> pixels;
    long long        decode_ms        = 0;
    bool             needs_all_frames = false; // multi-frame PNG → re-open in enc task
    bool             decode_ok        = false;
};

// ---------------------------------------------------------------------------
// In-memory DICOM source: wraps pre-decoded pixel bytes as an ImageSource so
// the existing encode_source_to_png / encode_source_to_bmp can consume them
// without any DCMTK involvement.
// ---------------------------------------------------------------------------
struct MemoryDicomSource : ImageSource {
    ImageInfo        info_;
    DicomPixelParams params_;
    const uint8_t*   pixels_        = nullptr;
    int              next_row_       = 0;
    // When true, dicom_params() returns nullptr so the GPU preprocessing
    // kernel is NOT run. Set this after pre-converting 16-bit pixels to 8-bit
    // on the CPU (the PNG and BMP fast paths).
    bool             suppress_dicom_ = false;

    const ImageInfo&        info()         const override { return info_;    }
    const DicomPixelParams* dicom_params() const override {
        return suppress_dicom_ ? nullptr : &params_;
    }

    int read_strip(uint8_t* out, int max_rows) override {
        const size_t row_bytes = (size_t)info_.width * (size_t)info_.bpp;
        const int remaining    = (int)info_.height - next_row_;
        const int rows         = std::min(max_rows, remaining);
        if (rows <= 0) return 0;
        std::memcpy(out, pixels_ + (size_t)next_row_ * row_bytes,
                    (size_t)rows * row_bytes);
        next_row_ += rows;
        return rows;
    }
};

// ---------------------------------------------------------------------------
// Precompute a uint16→uint8 LUT for DICOM grayscale windowing.
//
// The LUT is indexed by the POST-shift-and-mask value, so only lut_size
// = 2^bits_stored entries are needed (4096 for 12-bit, 65536 for 16-bit).
// For 12-bit CT (very common), 4096 bytes fit in L1 cache — the apply loop
// then has zero L1 misses vs the 64KB-scatter pattern for full 16-bit.
//
// lut[] must have room for lut_size bytes. Always call with lut_size
// = 1 << p.bits_stored.
// ---------------------------------------------------------------------------
static void build_dicom16_lut(uint8_t* lut, const DicomPixelParams& p, int lut_size)
{
    const int mask = lut_size - 1;

    float norm_scale = 1.f, norm_offset = 0.f;
    if (!p.apply_window) {
        if (p.pixel_rep == 1) {
            const float half  = (float)(lut_size >> 1);
            const float range = (float)lut_size;
            norm_scale  = 1.f / range;
            norm_offset = half / range;
        } else {
            norm_scale = 1.f / (float)(lut_size - 1);
        }
    }

    float lo = 0.f, inv_range = 1.f;
    if (p.apply_window) {
        lo             = p.window_center - p.window_width * 0.5f;
        const float hi = p.window_center + p.window_width * 0.5f;
        inv_range      = (hi > lo) ? 1.f / (hi - lo) : 0.f;
    }

    // i is the post-shift-and-mask index (0 … lut_size-1)
    for (int i = 0; i < lut_size; i++) {
        float val;
        if (p.pixel_rep == 1 && p.bits_stored < 16) {
            const int sign_bit = lut_size >> 1;  // 2^(bits_stored-1)
            int32_t s = i;
            if (s & sign_bit) s |= ~mask;        // sign-extend
            val = (float)s;
        } else if (p.pixel_rep == 1) {
            val = (float)(int16_t)(uint16_t)(unsigned)i;
        } else {
            val = (float)(unsigned)i;
        }

        if (p.apply_rescale)
            val = val * p.rescale_slope + p.rescale_intercept;

        const float norm = p.apply_window
            ? (val - lo) * inv_range
            : val * norm_scale + norm_offset;

        const int v = (int)(norm * 255.f + 0.5f);
        lut[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
}

// Apply a precomputed LUT to a little-endian uint16 pixel buffer.
// shift / mask16 compress the raw pixel to a post-mask index matching
// how build_dicom16_lut populated the table.
//
// Software prefetch: issue _mm_prefetch 64 iterations ahead so L3 load
// latency (~15 ns) is hidden behind computation. Without prefetch each
// random LUT access causes an L3 stall; with it the loop runs at ~2 ns/pixel
// (compute-bound) instead of ~15–25 ns/pixel (memory-bound).
static void apply_dicom16_lut(const uint8_t* src_le, uint8_t* dst8,
                               size_t pixel_count, const uint8_t* lut,
                               int shift, uint16_t mask16)
{
    constexpr size_t PF = 64;  // prefetch distance in pixels

    // Warm the prefetch pipeline for the first PF iterations.
    const size_t warm = (pixel_count < PF) ? pixel_count : PF;
    for (size_t i = 0; i < warm; i++) {
        const uint16_t raw = (uint16_t)((unsigned)src_le[i * 2]
                                       | ((unsigned)src_le[i * 2 + 1] << 8));
        _mm_prefetch(reinterpret_cast<const char*>(lut + ((raw >> shift) & mask16)),
                     _MM_HINT_T0);
    }

    for (size_t i = 0; i < pixel_count; i++) {
        // Issue prefetch for pixel i+PF while processing pixel i.
        if (i + PF < pixel_count) {
            const uint16_t pf_raw = (uint16_t)(
                (unsigned)src_le[(i + PF) * 2]
                | ((unsigned)src_le[(i + PF) * 2 + 1] << 8));
            _mm_prefetch(reinterpret_cast<const char*>(
                             lut + ((pf_raw >> shift) & mask16)), _MM_HINT_T0);
        }
        const uint16_t raw = (uint16_t)((unsigned)src_le[i * 2]
                                       | ((unsigned)src_le[i * 2 + 1] << 8));
        dst8[i] = lut[(raw >> shift) & mask16];
    }
}

// ---------------------------------------------------------------------------
// Fast linear windowing: Q15 fixed-point integer arithmetic, no table lookup.
//
// For the common CT case (right-aligned 16-bit pixels, linear window/level):
//   out = clamp(raw × A + B, 0, 255)
// where A = slope×255/window_width (Q15) and B = (intercept-lo)×255/window_width (Q15).
//
// All memory accesses are sequential (hardware-prefetchable). No random
// 64 KB table reads → no L3 misses. MSVC /O2 auto-vectorizes the inner
// loop 8-wide with SSE2: ~0.08 ms for 512×512 vs ~6 ms for the LUT path.
//
// Returns false when params fall outside the covered range (narrow windows,
// non-zero shift, unsigned pixels requiring a different scale, etc.) so the
// caller can fall back to the full LUT.
// ---------------------------------------------------------------------------
static bool try_apply_linear_window(const uint8_t* src_le, uint8_t* dst8,
                                    int pixel_count, const DicomPixelParams& p)
{
    // Only handle right-aligned data (the common case: high_bit = bits_stored-1)
    const int shift = p.high_bit - p.bits_stored + 1;
    if (shift != 0) return false;

    // Compute the affine mapping: out = clamp(raw * A + B, 0, 255)
    // Two sub-cases mirror exactly what dicom16_row_to_gray8 / LUT builder do.
    float A_f, B_f;

    if (p.apply_window && p.window_width > 0.f) {
        // Window/level path (most clinical CT)
        const float lo        = p.window_center - p.window_width * 0.5f;
        const float slope     = p.apply_rescale ? p.rescale_slope     : 1.f;
        const float intercept = p.apply_rescale ? p.rescale_intercept : 0.f;
        A_f = slope * 255.f / p.window_width;
        B_f = (intercept - lo) * 255.f / p.window_width;
    } else if (!p.apply_window) {
        // Full-range normalisation (no WindowCenter/Width tag in the DICOM)
        //   pixel_rep=1: norm = (val + half) / range
        //   pixel_rep=0: norm = val / (range-1)
        // When apply_rescale=1: val = raw * slope + intercept first.
        const float slope     = p.apply_rescale ? p.rescale_slope     : 1.f;
        const float intercept = p.apply_rescale ? p.rescale_intercept : 0.f;
        const float range     = (float)(1u << p.bits_stored);
        if (p.pixel_rep == 1) {
            const float half = range * 0.5f;
            A_f = slope * 255.f / range;
            B_f = (intercept + half) * 255.f / range;
        } else {
            A_f = slope * 255.f / (range - 1.f);
            B_f = intercept * 255.f / (range - 1.f);
        }
    } else {
        return false;
    }

    // Convert A to Q15 — reject if it overflows int16 (very narrow windows)
    const float A_q15 = A_f * 32768.f;
    if (A_q15 < -32767.f || A_q15 > 32767.f) return false;

    const int16_t A_fp = (int16_t)(int)A_q15;
    const int32_t B_fp = (int32_t)(A_f >= 0.f
                                       ? (A_q15 == (int)A_q15 ? B_f * 32768.f
                                                               : B_f * 32768.f + 0.5f)
                                       : B_f * 32768.f - 0.5f);

    if (p.pixel_rep == 1) {
        for (int i = 0; i < pixel_count; i++) {
            const int16_t s = (int16_t)(uint16_t)(
                (unsigned)src_le[i * 2] | ((unsigned)src_le[i * 2 + 1] << 8));
            const int32_t v = ((int32_t)s * A_fp + B_fp) >> 15;
            dst8[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
    } else {
        for (int i = 0; i < pixel_count; i++) {
            const int32_t u = (int32_t)(uint16_t)(
                (unsigned)src_le[i * 2] | ((unsigned)src_le[i * 2 + 1] << 8));
            const int32_t v = (u * A_fp + B_fp) >> 15;
            dst8[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Shared LUT cache for the encode thread pool.
//
// For a homogeneous CT batch (all 739 slices share the same window/level),
// we build the LUT exactly once and share a single read-only copy across all
// 6 encode threads. The 64KB (or 4KB for 12-bit) LUT sits in L2/L3 as
// shared-read cache lines, giving L2 hit rates on all table lookups.
//
// Thread-safety: g_lut.mtx protects valid/params/lut. Callers hold the
// returned shared_ptr; even if params change and lut is replaced, the old
// data stays alive via ref-count until all callers release it.
// ---------------------------------------------------------------------------
static bool params_equal(const DicomPixelParams& a, const DicomPixelParams& b)
{
    return a.bits_stored       == b.bits_stored
        && a.high_bit          == b.high_bit
        && a.pixel_rep         == b.pixel_rep
        && a.apply_rescale     == b.apply_rescale
        && a.rescale_slope     == b.rescale_slope
        && a.rescale_intercept == b.rescale_intercept
        && a.apply_window      == b.apply_window
        && a.window_center     == b.window_center
        && a.window_width      == b.window_width;
}

struct SharedLutState {
    std::mutex              mtx;
    bool                    valid  = false;
    DicomPixelParams        params = {};
    int                     shift  = 0;
    uint16_t                mask16 = 0xFFFFu;
    std::shared_ptr<std::vector<uint8_t>> lut;
};
static SharedLutState g_lut;

// Returns a shared_ptr to the LUT data (ref-count keeps it alive).
// Builds once per unique DicomPixelParams; cache hit acquires and releases
// the mutex in microseconds.
static std::shared_ptr<std::vector<uint8_t>>
get_or_build_shared_lut(const DicomPixelParams& p, int& out_shift, uint16_t& out_mask)
{
    std::lock_guard<std::mutex> lk(g_lut.mtx);
    if (!g_lut.valid || !params_equal(g_lut.params, p)) {
        const int lut_size = 1 << p.bits_stored;
        auto data = std::make_shared<std::vector<uint8_t>>(lut_size);
        build_dicom16_lut(data->data(), p, lut_size);
        g_lut.lut    = std::move(data);
        g_lut.params = p;
        g_lut.shift  = p.high_bit - p.bits_stored + 1;
        g_lut.mask16 = (uint16_t)(lut_size - 1);
        g_lut.valid  = true;
    }
    out_shift = g_lut.shift;
    out_mask  = g_lut.mask16;
    return g_lut.lut;
}

// ---------------------------------------------------------------------------
// Decode one DICOM file (called from dec_pool thread).
// Always produces a frame (even on error) so the paired encode task never
// blocks indefinitely waiting for its pop().
// ---------------------------------------------------------------------------
static PreDecodedFrame decode_dicom_to_frame(const fs::path& file,
                                              const fs::path& outdir,
                                              const PipelineConfig& cfg)
{
    PreDecodedFrame result;
    result.input_path = file.string();
    result.filename   = file.filename().string();

    auto t0 = Clock::now();

    DicomSource src;
    if (!src.open(result.input_path.c_str())) {
        result.decode_ms = std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
        return result;
    }

    // Multi-frame DICOM in PNG mode: route to all_frames path.
    // The encode task re-opens the file via encode_dicom_all_frames_to_png.
    if (src.num_frames() > 1 && cfg.format == ExportFormat::PNG) {
        result.needs_all_frames = true;
        result.decode_ok        = true;
        result.output_path      = (outdir / file.stem()).string();
        result.decode_ms = std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
        return result;
    }

    const int frame_idx = (cfg.frame >= 0) ? cfg.frame : 0;
    if (frame_idx >= src.num_frames() || !src.load_frame(frame_idx)) {
        result.decode_ms = std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
        return result;
    }

    result.info      = src.info_;
    result.params    = src.params_;
    result.pixels    = std::move(src.pixel_data_);  // zero-copy move
    result.decode_ok = true;

    const std::string ext = (cfg.format == ExportFormat::BMP) ? ".bmp" : ".png";
    result.output_path = (outdir / (file.stem().string() + ext)).string();
    result.decode_ms = std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
    return result;
}

// ---------------------------------------------------------------------------
// Encode a pre-decoded DICOM frame (called from enc_pool thread).
// ---------------------------------------------------------------------------
static bool encode_predecoded_dicom(const PreDecodedFrame& frame,
                                    const PipelineConfig& cfg)
{
    if (frame.needs_all_frames) {
        std::error_code ec;
        fs::create_directories(frame.output_path, ec);
        return encode_dicom_all_frames_to_png(frame.input_path.c_str(),
                                              frame.output_path.c_str(), cfg);
    }

    // -----------------------------------------------------------------------
    // 16-bit grayscale DICOM → 8-bit pre-conversion fast path.
    //
    // Both BMP and PNG benefit: do the 16→8 conversion on the CPU here
    // (Q15 fixed-point, ≈0.08 ms for 512×512) so the GPU never runs the
    // dicom_preprocess_kernel inside run_pipeline(). For PNG, this fixes a
    // -32% throughput regression (≈6 ms/file wasted GPU preprocessing).
    //
    // suppress_dicom_ = true tells MemoryDicomSource::dicom_params() to
    // return nullptr, which skips the GPU DICOM kernel unconditionally.
    // -----------------------------------------------------------------------
    if (frame.info.channels == 1 && frame.info.bits_per_sample == 16) {
        const size_t pc = (size_t)frame.info.width * frame.info.height;
        std::vector<uint8_t> px8(pc);

        // Fast path: Q15 fixed-point (SSE2-vectorizable, ~0.08 ms at 512×512)
        if (!try_apply_linear_window(frame.pixels.data(), px8.data(),
                                     (int)pc, frame.params)) {
            // Fallback: shared LUT (built once per unique params, L3-resident)
            int shift; uint16_t mask16;
            auto lut_ref = get_or_build_shared_lut(frame.params, shift, mask16);
            apply_dicom16_lut(frame.pixels.data(), px8.data(), pc,
                              lut_ref->data(), shift, mask16);
        }

        MemoryDicomSource src8;
        src8.info_                 = frame.info;
        src8.info_.bits_per_sample = 8;
        src8.info_.bpp             = 1;
        src8.params_               = frame.params;
        src8.pixels_               = px8.data();
        src8.suppress_dicom_       = true;  // GPU preprocessing kernel must NOT run

        if (cfg.format == ExportFormat::BMP)
            return encode_source_to_bmp(src8, frame.output_path.c_str());
        return encode_source_to_png(src8, frame.output_path.c_str(), cfg);
    }

    // -----------------------------------------------------------------------
    // Non-16-bit or multi-channel: pass raw pixels through as-is.
    // dicom_params() is non-null → GPU DICOM kernel runs inside run_pipeline().
    // This path covers 8-bit DICOM, RGB DICOM, and any future formats.
    // -----------------------------------------------------------------------
    MemoryDicomSource src;
    src.info_   = frame.info;
    src.params_ = frame.params;
    src.pixels_ = frame.pixels.data();

    if (cfg.format == ExportFormat::BMP)
        return encode_source_to_bmp(src, frame.output_path.c_str());
    return encode_source_to_png(src, frame.output_path.c_str(), cfg);
}

// ---------------------------------------------------------------------------
// Direct encode for TIFF / RAW (unchanged from Phase 4).
// ---------------------------------------------------------------------------
static bool process_ndicom_file(const fs::path& file, const fs::path& output_dir,
                                const PipelineConfig& cfg)
{
    const std::string path_str = file.string();
    const char*        path    = path_str.c_str();
    const std::string  stem    = file.stem().string();
    const bool         bmp     = (cfg.format == ExportFormat::BMP);
    const std::string  ext     = bmp ? ".bmp" : ".png";

    if (is_tiff_file(path)) {
        const fs::path outp = output_dir / (stem + ext);
        return bmp ? encode_tiff_to_bmp(path, outp.string().c_str(), cfg)
                   : encode_tiff_to_png(path, outp.string().c_str(), cfg);
    }
    if (is_raw_file(path)) {
        const fs::path outp = output_dir / (stem + ext);
        return bmp ? encode_raw_to_bmp(path, outp.string().c_str(), cfg)
                   : encode_raw_to_png(path, outp.string().c_str(), cfg);
    }
    return false;
}

// ---------------------------------------------------------------------------
// Per-file result collected from encode futures.
// ---------------------------------------------------------------------------
struct FileResult {
    std::string name;
    bool        ok;
    long long   ms;
    int         worker_id;
};

// Thread-local worker index: assigned once per OS thread the first time it
// picks up an encode task. Stable for the pool lifetime.
static std::atomic<int>  s_next_worker_id{0};
static thread_local int  tl_worker_id = -1;

// ---------------------------------------------------------------------------
// encode_folder: main entry point.
// ---------------------------------------------------------------------------
bool encode_folder(const char*           input_dir,
                   const char*           output_dir,
                   const PipelineConfig& cfg,
                   const BatchConfig&    batch_cfg)
{
    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec) {
        fprintf(stderr, "Cannot create output folder '%s': %s\n",
                output_dir, ec.message().c_str());
        return false;
    }

    std::vector<fs::path> files;
    for (auto& entry : fs::directory_iterator(input_dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        const std::string p = entry.path().string();
        if (is_dicom_file(p.c_str()) || is_tiff_file(p.c_str()) || is_raw_file(p.c_str()))
            files.push_back(entry.path());
    }
    if (ec) {
        fprintf(stderr, "Cannot read input folder '%s': %s\n", input_dir, ec.message().c_str());
        return false;
    }
    if (files.empty()) {
        fprintf(stdout, "No supported files found in '%s'\n", input_dir);
        return true;
    }

    std::sort(files.begin(), files.end());

    int workers = batch_cfg.num_workers;
    if (workers <= 0) {
        int hw = (int)std::thread::hardware_concurrency();
        if (hw <= 0) hw = 4;
        workers = std::min(hw, std::min((int)files.size(), 4));
        workers = std::max(workers, 1);
    }

    // Split by type: DICOM goes through the D2 decode-ahead pipeline;
    // TIFF/RAW use the direct encode path.
    std::vector<fs::path> dicom_files, other_files;
    for (auto& f : files) {
        const std::string p = f.string();
        (is_dicom_file(p.c_str()) ? dicom_files : other_files).push_back(f);
    }

    const int   total          = (int)files.size();
    const bool  bmp_mode       = (cfg.format == ExportFormat::BMP);
    const char* fmt_name       = bmp_mode ? "BMP" : "PNG";
    const bool  use_gpu_summary = (cfg.use_gpu_deflate && !bmp_mode);

    fprintf(stdout, "Batch: %d file(s), %d worker(s), format=%s%s\n",
            total, workers, fmt_name,
            batch_cfg.batch_verbose ? "  [--batch-verbose: per-file times shown]" : "");

    if (use_gpu_summary)
        pipeline_reset_gpu_batch_stats();

    s_next_worker_id.store(0);

    const fs::path   outdir(output_dir);
    std::atomic<int> done_count{0};
    std::mutex       print_mtx;

    auto print_progress = [&](int n, const std::string& fname, bool ok,
                              long long ms, int wid) {
        std::lock_guard<std::mutex> lk(print_mtx);
        if (batch_cfg.batch_verbose) {
            fprintf(stdout, "[%d/%d] W%-2d  %-40s  %-6s  (%lld ms)\n",
                    n, total, wid, fname.c_str(), ok ? "OK" : "FAILED", ms);
        } else {
            fprintf(stdout, "[%d/%d] %s  %s\n",
                    n, total, fname.c_str(), ok ? "OK" : "FAILED");
        }
    };

    std::vector<FileResult> results;
    results.reserve(files.size());

    ThreadPool encode_pool(workers);
    auto batch_start = Clock::now();

    // -----------------------------------------------------------------
    // Phase D2: DICOM — decode pool → frame queue → encode pool.
    //
    // Encode futures are submitted FIRST so encode threads are already
    // waiting at frame_q.pop() when decode tasks begin pushing.
    // -----------------------------------------------------------------
    if (!dicom_files.empty()) {
        // Queue capacity: 2× worker count bounds memory use
        // (each decoded CT frame is ~512 KB, so 12 frames × 512 KB = 6 MB at 6W).
        BoundedQueue<PreDecodedFrame> frame_q(std::max(2, workers * 2));

        // Submit encode tasks — each pops exactly one frame then encodes.
        std::vector<std::future<FileResult>> dicom_futs;
        dicom_futs.reserve(dicom_files.size());
        for (size_t i = 0; i < dicom_files.size(); i++) {
            dicom_futs.push_back(encode_pool.submit([&, i]() -> FileResult {
                if (tl_worker_id < 0)
                    tl_worker_id = s_next_worker_id.fetch_add(1);
                const int wid = tl_worker_id;

                PreDecodedFrame frame;
                if (!frame_q.pop(frame)) {
                    // Should not happen — queue is never closed, and we push
                    // exactly dicom_files.size() items (one per task submitted).
                    return FileResult{"??", false, 0, wid};
                }

                auto enc_t0 = Clock::now();
                bool ok = false;
                if (!frame.decode_ok) {
                    fprintf(stderr, "[batch] Decode failed: %s\n",
                            frame.input_path.c_str());
                } else {
                    ok = encode_predecoded_dicom(frame, cfg);
                }
                auto enc_t1 = Clock::now();
                const long long enc_ms =
                    std::chrono::duration_cast<Ms>(enc_t1 - enc_t0).count();

                // Report decode + encode time to GPU batch stats accumulator.
                if (use_gpu_summary)
                    pipeline_record_file_times(frame.decode_ms,
                                               frame.decode_ms + enc_ms);

                const long long total_ms = frame.decode_ms + enc_ms;
                const int n = ++done_count;
                print_progress(n, frame.filename, ok, total_ms, wid);
                return FileResult{frame.filename, ok, total_ms, wid};
            }));
        }

        // Run decode pool concurrently alongside encode pool.
        // Destructor blocks until all decode tasks have pushed their frames.
        {
            ThreadPool dec_pool(std::max(1, workers));
            for (auto& f : dicom_files) {
                dec_pool.submit([&, f]() {
                    try {
                        frame_q.push(decode_dicom_to_frame(f, outdir, cfg));
                    } catch (...) {
                        // Always push a frame to unblock the paired encode task.
                        PreDecodedFrame err;
                        err.input_path = f.string();
                        err.filename   = f.filename().string();
                        frame_q.push(std::move(err));
                    }
                });
            }
        }  // dec_pool join: all decode tasks have pushed; encode tasks still running

        for (auto& fut : dicom_futs)
            results.push_back(fut.get());
    }

    // -----------------------------------------------------------------
    // TIFF / RAW: direct encode (no pre-decode needed).
    // -----------------------------------------------------------------
    if (!other_files.empty()) {
        std::vector<std::future<FileResult>> other_futs;
        other_futs.reserve(other_files.size());
        for (auto& f : other_files) {
            other_futs.push_back(encode_pool.submit([&, f]() -> FileResult {
                if (tl_worker_id < 0)
                    tl_worker_id = s_next_worker_id.fetch_add(1);
                const int wid = tl_worker_id;

                auto t0  = Clock::now();
                bool ok  = process_ndicom_file(f, outdir, cfg);
                auto t1  = Clock::now();
                const long long ms = std::chrono::duration_cast<Ms>(t1 - t0).count();

                const int n = ++done_count;
                print_progress(n, f.filename().string(), ok, ms, wid);
                return FileResult{f.filename().string(), ok, ms, wid};
            }));
        }
        for (auto& fut : other_futs)
            results.push_back(fut.get());
    }

    auto batch_end  = Clock::now();
    const long long batch_ms =
        std::chrono::duration_cast<Ms>(batch_end - batch_start).count();

    long long sum_ms = 0, min_ms = LLONG_MAX, max_ms = 0;
    int succeeded = 0;
    for (auto& r : results) {
        if (r.ok) succeeded++;
        sum_ms += r.ms;
        if (r.ms < min_ms) min_ms = r.ms;
        if (r.ms > max_ms) max_ms = r.ms;
    }
    if (results.empty()) { min_ms = 0; max_ms = 0; }

    const long long avg_ms  = results.empty() ? 0LL
                                               : sum_ms / (long long)results.size();
    const double    fps     = (batch_ms > 0)
                                ? (double)total / ((double)batch_ms / 1000.0)
                                : 0.0;
    const int       failed  = total - succeeded;

    fprintf(stdout, "\n----- Batch Summary ");
    for (int i = 0; i < 41; i++) fputc('-', stdout);
    fputc('\n', stdout);

    fprintf(stdout, "Files       : %d total  |  %d OK  |  %d failed\n",
            total, succeeded, failed);
    fprintf(stdout, "Wall time   : %.1f s\n", (double)batch_ms / 1000.0);
    fprintf(stdout, "Per file    : avg %lld ms  |  min %lld ms  |  max %lld ms\n",
            avg_ms, min_ms, max_ms);
    fprintf(stdout, "Throughput  : %.2f files/s", fps);
    if (workers > 1)
        fprintf(stdout, "  (%d workers)", workers);
    fputc('\n', stdout);

    // Format-specific pipeline annotation
    if (!dicom_files.empty()) {
        const int n_dec = std::max(1, workers);
        if (bmp_mode)
            fprintf(stdout,
                    "Pipeline    : BMP fast path  "
                    "(D2 decode-ahead: %d dec + %d enc threads)\n",
                    n_dec, workers);
        else if (!use_gpu_summary)
            fprintf(stdout,
                    "Pipeline    : PNG CPU-deflate  "
                    "(D2 decode-ahead: %d dec + %d enc threads)\n",
                    n_dec, workers);
    }

    if (failed > 0) {
        fprintf(stdout, "Failed files:\n");
        for (auto& r : results)
            if (!r.ok)
                fprintf(stdout, "  FAILED  %s  (%lld ms)\n",
                        r.name.c_str(), r.ms);
    }

    // GPU deflate breakdown only relevant for PNG + --gpu-deflate.
    // BMP has no GPU deflate; skip to avoid printing all-zero stats.
    if (use_gpu_summary)
        pipeline_print_gpu_batch_summary(total, succeeded,
                                         (double)batch_ms / 1000.0,
                                         sum_ms, workers);

    return (failed == 0);
}

// ===========================================================================
// Decode-only benchmark (--benchmark-decode)
// ===========================================================================

struct DecodeBenchItem {
    std::string filename;
    const char* file_type = "";  // "DICOM", "TIFF", "RAW"
    long long   open_ms   = 0;   // header / metadata parse
    long long   load_ms   = 0;   // pixel data decode
    long long   total_ms  = 0;
    int         width     = 0;
    int         height    = 0;
    int         frames    = 1;
    bool        ok        = false;
};

static DecodeBenchItem bench_decode_dicom(const fs::path& file, const PipelineConfig& cfg)
{
    DecodeBenchItem r;
    r.filename  = file.filename().string();
    r.file_type = "DICOM";

    const auto t0 = Clock::now();
    DicomSource src;
    if (!src.open(file.string().c_str())) {
        r.total_ms = std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
        return r;
    }
    r.open_ms = std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
    r.frames  = src.num_frames();

    const int frame_idx = (cfg.frame >= 0) ? cfg.frame : 0;
    if (frame_idx >= src.num_frames()) {
        r.total_ms = r.open_ms;
        return r;
    }

    const auto tl = Clock::now();
    if (!src.load_frame(frame_idx)) {
        r.total_ms = r.open_ms + std::chrono::duration_cast<Ms>(Clock::now() - tl).count();
        return r;
    }
    r.load_ms  = std::chrono::duration_cast<Ms>(Clock::now() - tl).count();
    r.total_ms = r.open_ms + r.load_ms;
    r.width    = (int)src.info_.width;
    r.height   = (int)src.info_.height;
    r.ok       = true;
    return r;
}

static DecodeBenchItem bench_decode_tiff(const fs::path& file)
{
    DecodeBenchItem r;
    r.filename  = file.filename().string();
    r.file_type = "TIFF";

    const auto t0 = Clock::now();
    ImageInfo    info;
    TiffReader*  tr = tiff_open(file.string().c_str(), info);
    if (!tr) {
        r.total_ms = std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
        return r;
    }
    r.open_ms = std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
    r.width   = (int)info.width;
    r.height  = (int)info.height;

    // Read all strips to benchmark the full libtiff decode path.
    const auto   tl        = Clock::now();
    const size_t row_bytes = (size_t)info.width * (size_t)info.bpp;
    std::vector<uint8_t> strip(64 * row_bytes);
    int got, total_rows = 0;
    while ((got = tiff_read_strip(tr, strip.data(), 64)) > 0)
        total_rows += got;
    tiff_close(tr);

    r.load_ms  = std::chrono::duration_cast<Ms>(Clock::now() - tl).count();
    r.total_ms = r.open_ms + r.load_ms;
    r.ok       = (total_rows > 0);
    return r;
}

static DecodeBenchItem bench_decode_raw(const fs::path& file)
{
    DecodeBenchItem r;
    r.filename  = file.filename().string();
    r.file_type = "RAW";

    // raw_open runs full LibRaw demosaicing; raw_read_strip just copies from buffer.
    const auto t0 = Clock::now();
    ImageInfo   info;
    RawReader*  rr = raw_open(file.string().c_str(), info);
    if (!rr) {
        r.total_ms = std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
        return r;
    }
    r.open_ms  = std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
    r.load_ms  = 0;  // included in open
    r.total_ms = r.open_ms;
    r.width    = (int)info.width;
    r.height   = (int)info.height;

    // Drain one strip to confirm buffer is readable.
    std::vector<uint8_t> row((size_t)info.width * (size_t)info.bpp);
    raw_read_strip(rr, row.data(), 1);
    raw_close(rr);

    r.ok = true;
    return r;
}

// ---------------------------------------------------------------------------
// Print the aggregate decode benchmark report.
// ---------------------------------------------------------------------------
static void print_decode_report(const std::vector<DecodeBenchItem>& results,
                                 long long batch_ms, int workers)
{
    const int total  = (int)results.size();
    int n_ok = 0, n_dicom = 0, n_tiff = 0, n_raw = 0;
    long long sum_open = 0, sum_load = 0, sum_total = 0;
    long long min_total = LLONG_MAX, max_total = 0;

    // Track per-type stats for DICOM (most informative breakdown)
    long long dicom_open_sum = 0, dicom_load_sum = 0;

    for (auto& r : results) {
        if (r.ok) n_ok++;
        if (strcmp(r.file_type, "DICOM") == 0) { n_dicom++; dicom_open_sum += r.open_ms; dicom_load_sum += r.load_ms; }
        else if (strcmp(r.file_type, "TIFF") == 0) n_tiff++;
        else if (strcmp(r.file_type, "RAW")  == 0) n_raw++;
        sum_open  += r.open_ms;
        sum_load  += r.load_ms;
        sum_total += r.total_ms;
        if (r.total_ms < min_total) min_total = r.total_ms;
        if (r.total_ms > max_total) max_total = r.total_ms;
    }
    if (results.empty()) { min_total = 0; max_total = 0; }

    const int       failed   = total - n_ok;
    const long long avg_total = (total > 0) ? sum_total / total : 0;
    const double    fps       = (batch_ms > 0) ? (double)total / ((double)batch_ms / 1000.0) : 0.0;

    // Worker utilization: fraction of wall-time workers spent actively decoding.
    const long long worker_budget_ms = batch_ms * workers;
    const double    utilization =
        (worker_budget_ms > 0) ? (double)sum_total / (double)worker_budget_ms : 0.0;

    // Theoretical decode ceiling
    const double fps_1w = (avg_total > 0) ? 1000.0 / (double)avg_total : 0.0;
    const double fps_nw = fps_1w * workers;

    fprintf(stdout, "\n");
    for (int i = 0; i < 60; i++) fputc('=', stdout);
    fprintf(stdout, "\n  DECODE BENCHMARK REPORT\n");
    for (int i = 0; i < 60; i++) fputc('=', stdout);
    fprintf(stdout, "\n\n");

    fprintf(stdout, "Files         : %d total  |  %d OK  |  %d failed\n",
            total, n_ok, failed);

    if (n_dicom || n_tiff || n_raw) {
        fprintf(stdout, "Types         :");
        if (n_dicom) fprintf(stdout, "  %d DICOM", n_dicom);
        if (n_tiff)  fprintf(stdout, "  %d TIFF",  n_tiff);
        if (n_raw)   fprintf(stdout, "  %d RAW",   n_raw);
        fputc('\n', stdout);
    }

    fprintf(stdout, "Wall time     : %.2f s\n", (double)batch_ms / 1000.0);
    fprintf(stdout, "Throughput    : %.1f files/s  (%d workers)\n", fps, workers);
    fprintf(stdout, "Per file      : avg %lld ms  |  min %lld ms  |  max %lld ms\n",
            avg_total, min_total, max_total);

    // DICOM-specific breakdown (open vs load)
    if (n_dicom > 0) {
        const long long avg_dicom_open = dicom_open_sum / n_dicom;
        const long long avg_dicom_load = dicom_load_sum / n_dicom;
        fprintf(stdout, "\nDICOM timing (%d files):\n", n_dicom);
        fprintf(stdout, "  Avg header parse : %3lld ms  (DicomSource::open)\n",
                avg_dicom_open);
        fprintf(stdout, "  Avg pixel decode : %3lld ms  (DicomSource::load_frame / DCMTK)\n",
                avg_dicom_load);
        fprintf(stdout, "  Avg total        : %3lld ms\n",
                avg_dicom_open + avg_dicom_load);
    }

    // Timing consistency / worker utilization
    fprintf(stdout, "\nWorker utilization:\n");
    fprintf(stdout, "  Sum per-file decode : %lld ms\n", sum_total);
    fprintf(stdout, "  Workers x wall time : %lld ms  (%dW x %.2fs)\n",
            worker_budget_ms, workers, (double)batch_ms / 1000.0);
    fprintf(stdout, "  Utilization         : %.1f%%", utilization * 100.0);
    if (utilization > 0.90)
        fprintf(stdout, "  → DECODE-LIMITED (workers busy >90%% of wall time)\n");
    else if (utilization > 0.70)
        fprintf(stdout, "  → MOSTLY DECODE-LIMITED (~%.0f%% idle)\n",
                (1.0 - utilization) * 100.0);
    else
        fprintf(stdout, "  → I/O-LIMITED (workers idle %.0f%% of wall time; check disk/cache)\n",
                (1.0 - utilization) * 100.0);

    // Theoretical ceiling
    fprintf(stdout, "\nTheoretical decode ceiling:\n");
    fprintf(stdout, "  @ 1 worker  : %5.0f files/s  (= 1000 / %lld ms avg)\n",
            fps_1w, avg_total);
    fprintf(stdout, "  @ %d workers : %5.0f files/s  (= %d x %.0f)  ←  measured %.1f files/s\n",
            workers, fps_nw, workers, fps_1w, fps);

    const double headroom_pct = (fps_nw > 0) ? (fps_nw - fps) / fps_nw * 100.0 : 0.0;
    if (headroom_pct < 5.0)
        fprintf(stdout,
                "  Gap: %.1f%%  → encode overhead is negligible; "
                "DCMTK IS the throughput ceiling.\n", headroom_pct);
    else if (headroom_pct < 25.0)
        fprintf(stdout,
                "  Gap: %.1f%%  → encode adds moderate overhead; "
                "both decode and encode contribute.\n", headroom_pct);
    else
        fprintf(stdout,
                "  Gap: %.1f%%  → encode stage is a significant bottleneck; "
                "optimise encode OR increase decode workers.\n", headroom_pct);

    if (failed > 0) {
        fprintf(stdout, "\nFailed files:\n");
        for (auto& r : results)
            if (!r.ok)
                fprintf(stdout, "  FAILED  %s\n", r.filename.c_str());
    }

    for (int i = 0; i < 60; i++) fputc('=', stdout);
    fputc('\n', stdout);
}

bool benchmark_decode_folder(const char*           input_dir,
                              const PipelineConfig& cfg,
                              const BatchConfig&    batch_cfg)
{
    std::error_code ec;
    std::vector<fs::path> files;
    for (auto& entry : fs::directory_iterator(input_dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        const std::string p = entry.path().string();
        if (is_dicom_file(p.c_str()) || is_tiff_file(p.c_str()) || is_raw_file(p.c_str()))
            files.push_back(entry.path());
    }
    if (ec) {
        fprintf(stderr, "Cannot read folder '%s': %s\n", input_dir, ec.message().c_str());
        return false;
    }
    if (files.empty()) {
        fprintf(stdout, "No supported files found in '%s'\n", input_dir);
        return true;
    }
    std::sort(files.begin(), files.end());

    int workers = batch_cfg.num_workers;
    if (workers <= 0) {
        int hw = (int)std::thread::hardware_concurrency();
        if (hw <= 0) hw = 4;
        workers = std::min(hw, std::min((int)files.size(), 4));
        workers = std::max(workers, 1);
    }

    const int total = (int)files.size();
    fprintf(stdout, "Decode benchmark: %d file(s), %d worker(s)  [no output written]\n",
            total, workers);

    std::atomic<int> done_count{0};
    std::mutex       print_mtx;

    ThreadPool pool(workers);
    std::vector<std::future<DecodeBenchItem>> futs;
    futs.reserve(files.size());

    const auto batch_start = Clock::now();

    for (auto& f : files) {
        futs.push_back(pool.submit([&, f]() -> DecodeBenchItem {
            DecodeBenchItem r;
            const std::string p = f.string();
            if      (is_dicom_file(p.c_str())) r = bench_decode_dicom(f, cfg);
            else if (is_tiff_file(p.c_str()))  r = bench_decode_tiff(f);
            else if (is_raw_file(p.c_str()))   r = bench_decode_raw(f);
            else { r.filename = f.filename().string(); r.file_type = "?"; }

            const int n = ++done_count;
            {
                std::lock_guard<std::mutex> lk(print_mtx);
                fprintf(stdout, "[%d/%d] %s  %s  (%lld ms)\n",
                        n, total, r.filename.c_str(),
                        r.ok ? "OK" : "FAILED",
                        r.total_ms);
            }
            return r;
        }));
    }

    std::vector<DecodeBenchItem> results;
    results.reserve(futs.size());
    for (auto& fut : futs)
        results.push_back(fut.get());

    const auto batch_end = Clock::now();
    const long long batch_ms =
        std::chrono::duration_cast<Ms>(batch_end - batch_start).count();

    print_decode_report(results, batch_ms, workers);
    return true;
}

bool benchmark_decode_file(const char* input_path, const PipelineConfig& cfg)
{
    const fs::path file(input_path);
    DecodeBenchItem r;

    if (is_dicom_file(input_path))     r = bench_decode_dicom(file, cfg);
    else if (is_tiff_file(input_path)) r = bench_decode_tiff(file);
    else if (is_raw_file(input_path))  r = bench_decode_raw(file);
    else {
        fprintf(stderr, "Unsupported file type: %s\n", input_path);
        return false;
    }

    fprintf(stdout, "Decode benchmark: %s\n", input_path);
    fprintf(stdout, "  Type         : %s\n", r.file_type);
    if (r.width > 0)
        fprintf(stdout, "  Dimensions   : %dx%d\n", r.width, r.height);
    if (r.frames > 1)
        fprintf(stdout, "  Frames       : %d\n", r.frames);
    fprintf(stdout, "  Header parse : %lld ms\n", r.open_ms);
    if (r.load_ms > 0)
        fprintf(stdout, "  Pixel decode : %lld ms\n", r.load_ms);
    fprintf(stdout, "  Total        : %lld ms\n", r.total_ms);
    fprintf(stdout, "  Result       : %s\n", r.ok ? "OK" : "FAILED");
    return r.ok;
}
