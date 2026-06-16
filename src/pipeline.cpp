// pipeline.cpp
// 4-stage producer-consumer pipeline:
//
//   [Loader Thread] → Queue A → [GPU Thread] → Queue B
//                             → [Deflate Thread] → Queue C → [Writer Thread]
//
// All four stages run concurrently.  Queue backpressure (cap=2) keeps memory
// bounded regardless of image size.  The DEFLATE thread uses a persistent
// ThreadPool for per-strip chunk parallelism so thread-creation overhead is
// paid only once at startup.
//
// run_pipeline() accepts any ImageSource implementation (TIFF / RAW / DICOM).
// DICOM sources provide a non-null DicomPixelParams pointer that activates
// the GPU preprocessing kernel before PNG filtering.

#include "pipeline.h"
#include "image_source.h"
#include "dicom_loader.h"
#include "gpu_filter.h"
#include "gpu_capability.h"
#include "parallel_deflate.h"
#include "png_writer.h"
#include "image_loader.h"
#include "bounded_queue.h"
#include "strip_job.h"
#include "thread_pool.h"
#if defined(GPU_PNG_MODERN_DEFLATE)
#include "gpu_deflate_backend.h"
#include "gpu_png_assemble.h"
#include "gpu_adler32.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// TiffSource / RawSource – thin ImageSource wrappers (private to this TU)
// ---------------------------------------------------------------------------
struct TiffSource : ImageSource {
    TiffReader* r_ = nullptr;
    ImageInfo   info_;
    bool open(const char* path) {
        r_ = tiff_open(path, info_);
        return r_ != nullptr;
    }
    ~TiffSource() { if (r_) tiff_close(r_); }
    const ImageInfo& info() const override { return info_; }
    int read_strip(uint8_t* out, int rows) override {
        return tiff_read_strip(r_, out, rows);
    }
};

struct RawSource : ImageSource {
    RawReader* r_ = nullptr;
    ImageInfo  info_;
    bool open(const char* path) {
        r_ = raw_open(path, info_);
        return r_ != nullptr;
    }
    ~RawSource() { if (r_) raw_close(r_); }
    const ImageInfo& info() const override { return info_; }
    int read_strip(uint8_t* out, int rows) override {
        return raw_read_strip(r_, out, rows);
    }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static PngColorType channels_to_color_type(int channels)
{
    switch (channels) {
        case 1:  return PngColorType::Gray;
        case 3:  return PngColorType::RGB;
        case 4:  return PngColorType::RGBA;
        default: return PngColorType::RGB;
    }
}

// Format a system_clock time point as "HH:MM:SS.mmm"
static void fmt_timestamp(std::chrono::system_clock::time_point tp,
                           char out[16])
{
    using namespace std::chrono;
    auto ms  = duration_cast<milliseconds>(tp.time_since_epoch()) % 1000;
    std::time_t t = system_clock::to_time_t(tp);
    std::tm tm_buf;
#ifdef _MSC_VER
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    snprintf(out, 16, "%02d:%02d:%02d.%03lld",
             tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
             (long long)ms.count());
}

// ---------------------------------------------------------------------------
// Per-pipeline performance counters
// ---------------------------------------------------------------------------
struct PipelineStats {
    std::atomic<long long> load_ms{0};
    std::atomic<long long> gpu_ms{0};
    std::atomic<long long> deflate_ms{0};
    std::atomic<long long> write_ms{0};

    std::atomic<long long> load_count{0};
    std::atomic<long long> gpu_count{0};
    std::atomic<long long> deflate_count{0};
    std::atomic<long long> write_count{0};

    // GPU sub-phase totals in microseconds (from CUDA Events)
    std::atomic<long long> gpu_h2d_us{0};
    std::atomic<long long> gpu_kernel_us{0};
    std::atomic<long long> gpu_d2h_us{0};

    // Deflate sub-phase totals in microseconds.
    // Summed across all parallel chunks across all strips.
    // Wall cost = sum / num_threads  (chunks run in parallel within each strip).
    std::atomic<long long> deflate_init_us{0};
    std::atomic<long long> deflate_compress_us{0};
};

// ---------------------------------------------------------------------------
// Stage 1 – Loader
// ---------------------------------------------------------------------------
static void loader_stage(
    ImageSource&            src,
    int                     strip_height,
    size_t                  row_bytes,
    uint32_t                image_height,
    BoundedQueue<StripJob>& qa,
    std::atomic<bool>&      error,
    PipelineStats&          stats)
{
    int total_strips = ((int)image_height + strip_height - 1) / strip_height;

    for (int s = 0; s < total_strips && !error.load(); s++) {
        StripJob job;
        job.strip_index = s;
        job.data.resize((size_t)strip_height * row_bytes);

        auto t0 = std::chrono::high_resolution_clock::now();
        job.actual_rows = src.read_strip(job.data.data(), strip_height);
        auto t1 = std::chrono::high_resolution_clock::now();

        stats.load_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        stats.load_count++;

        if (job.actual_rows <= 0) { error.store(true); break; }
        job.is_last = (s == total_strips - 1);
        if (!qa.push(std::move(job))) break;
    }
    qa.close();
}

// ---------------------------------------------------------------------------
// Stage 2 – GPU Filter
// ---------------------------------------------------------------------------
static void gpu_stage(
    GpuFilterContext*          gpu,
    const DicomPixelParams*    dicom,
    BoundedQueue<StripJob>&    qa,
    BoundedQueue<FilteredJob>& qb,
    std::atomic<bool>&         error,
    PipelineStats&             stats)
{
    StripJob job;
    while (!error.load() && qa.pop(job)) {
        auto t0 = std::chrono::high_resolution_clock::now();

        GpuTimings gt;
        const uint8_t* filtered = gpu_filter_process_from_host(
            gpu,
            job.data.data(),
            nullptr,
            job.actual_rows,
            &gt,
            dicom);

        auto t1 = std::chrono::high_resolution_clock::now();

        stats.gpu_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        stats.gpu_h2d_us    += (long long)(gt.h2d_ms    * 1000.f);
        stats.gpu_kernel_us += (long long)(gt.kernel_ms * 1000.f);
        stats.gpu_d2h_us    += (long long)(gt.d2h_ms    * 1000.f);
        stats.gpu_count++;

        const size_t fsize = gpu_filter_output_size(gpu, job.actual_rows);

        FilteredJob fjob;
        fjob.strip_index = job.strip_index;
        fjob.actual_rows = job.actual_rows;
        fjob.is_last     = job.is_last;
        fjob.data.assign(filtered, filtered + fsize);

        if (!qb.push(std::move(fjob))) break;
    }
    qb.close();
}

// ---------------------------------------------------------------------------
// Stage 3 – Deflate
// ---------------------------------------------------------------------------
static void deflate_stage(
    BoundedQueue<FilteredJob>&   qb,
    BoundedQueue<CompressedJob>& qc,
    const ParallelDeflateConfig& dcfg,
    ThreadPool&                  pool,
    std::atomic<bool>&           error,
    PipelineStats&               stats)
{
    FilteredJob fjob;
    while (!error.load() && qb.pop(fjob)) {
        auto t0 = std::chrono::high_resolution_clock::now();

        DeflateResult dr = deflate_strip(
            fjob.data.data(), fjob.data.size(), dcfg, pool, fjob.is_last);

        auto t1 = std::chrono::high_resolution_clock::now();
        stats.deflate_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        stats.deflate_init_us     += dr.init_us;
        stats.deflate_compress_us += dr.compress_us;
        stats.deflate_count++;

        CompressedJob cjob;
        cjob.strip_index = fjob.strip_index;
        cjob.is_last     = fjob.is_last;
        cjob.data        = std::move(dr.data);
        cjob.strip_adler = dr.strip_adler;
        cjob.input_size  = dr.input_size;

        if (!qc.push(std::move(cjob))) break;
    }
    qc.close();
}

// ---------------------------------------------------------------------------
// Stage 4 – Writer
// ---------------------------------------------------------------------------
static void writer_stage(
    BoundedQueue<CompressedJob>& qc,
    PngWriter&                   writer,
    int                          deflate_level,
    std::atomic<bool>&           error,
    PipelineStats&               stats)
{
    uint8_t zhdr[2];
    zlib_header(deflate_level, zhdr);
    writer.write_idat_bytes(zhdr, 2);

    ParallelDeflateState dstate;
    CompressedJob cjob;

    while (!error.load() && qc.pop(cjob)) {
        DeflateResult dr_adler;
        dr_adler.strip_adler = cjob.strip_adler;
        dr_adler.input_size  = cjob.input_size;
        accum_adler(dstate, dr_adler);

        if (!cjob.data.empty()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            writer.write_idat_bytes(cjob.data.data(), cjob.data.size());
            auto t1 = std::chrono::high_resolution_clock::now();
            stats.write_ms +=
                std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            stats.write_count++;
        }
    }
    uint8_t ztrl[4];
    zlib_trailer(dstate, ztrl);
    writer.write_idat_bytes(ztrl, 4);
}

// ---------------------------------------------------------------------------
// Internal: run the 4-stage pipeline for ONE image/frame using a caller-owned
// GPU context and thread pool (both reusable across frames).
//
//   print_header : print the one-line source/DICOM header before running
//   print_report : print the full per-frame timing report after running
//
// The GPU prior-row state is reset here so reused contexts start each frame
// with zeros for the first row's Up/Paeth predictors.
// ---------------------------------------------------------------------------
static bool run_one(
    const PipelineConfig& cfg,
    const char*           output_path,
    const char*           source_label,
    ImageSource&          src,
    GpuFilterContext*     gpu,
    ThreadPool&           pool,
    bool                  print_header,
    bool                  print_report)
{
    const ImageInfo& info   = src.info();
    const int        strip_h = cfg.strip_height;
    const size_t     row_b   = (size_t)info.width * info.bpp;
    const DicomPixelParams* dicom = src.dicom_params();

    // Reset per-image GPU state (prior row) — required when reusing a context
    // across multiple frames.
    gpu_filter_reset(gpu);

    // Wall-clock start timestamp (absolute, for the timing report)
    auto t_abs_start = std::chrono::system_clock::now();

    if (print_header) {
        char ts[16];
        fmt_timestamp(t_abs_start, ts);
        fprintf(stdout, "[%s] %s  %ux%u  ch=%d  bps=%d  strip_h=%d  threads=%d  level=%d\n",
                ts, source_label,
                info.width, info.height, info.channels, info.bits_per_sample,
                cfg.strip_height, cfg.deflate_threads, cfg.deflate_level);
        if (dicom) {
            int shift = dicom->high_bit - dicom->bits_stored + 1;
            fprintf(stdout,
                "         DICOM  frames=%d  bits_alloc=%d  bits_stored=%d  high_bit=%d"
                "  shift=%d  pixel_rep=%d  rescale=%s  window=%s\n",
                src.num_frames(),
                dicom->bits_allocated, dicom->bits_stored, dicom->high_bit,
                shift, dicom->pixel_rep,
                dicom->apply_rescale ? "yes" : "no",
                dicom->apply_window  ? "yes" : "no");
        }
    }

    ParallelDeflateConfig dcfg;
    dcfg.num_threads = cfg.deflate_threads;
    dcfg.zlib_level  = cfg.deflate_level;

    PngWriter writer;
    if (!writer.open(output_path, info.width, info.height,
                     (uint8_t)info.bits_per_sample,
                     channels_to_color_type(info.channels))) {
        fprintf(stderr, "Cannot open output: %s\n", output_path);
        return false;
    }
    writer.begin_idat();

    BoundedQueue<StripJob>      qa(2);
    BoundedQueue<FilteredJob>   qb(2);
    BoundedQueue<CompressedJob> qc(2);
    std::atomic<bool> error(false);
    PipelineStats stats;

    // High-resolution wall-clock timer for throughput calculation
    auto t_wall_start = std::chrono::high_resolution_clock::now();

    std::thread t_load([&]{
        loader_stage(src, strip_h, row_b, info.height, qa, error, stats);
    });
    std::thread t_gpu([&]{
        gpu_stage(gpu, dicom, qa, qb, error, stats);
    });
    std::thread t_def([&]{
        deflate_stage(qb, qc, dcfg, pool, error, stats);
    });
    std::thread t_wri([&]{
        writer_stage(qc, writer, cfg.deflate_level, error, stats);
    });

    t_load.join();
    t_gpu.join();
    t_def.join();
    t_wri.join();

    auto t_wall_end  = std::chrono::high_resolution_clock::now();
    auto t_abs_end   = std::chrono::system_clock::now();

    writer.end_idat();
    writer.close();

    if (print_report) {
        double wall_ms =
            (double)std::chrono::duration_cast<std::chrono::microseconds>(
                t_wall_end - t_wall_start).count() / 1000.0;

        double total_pixels = (double)info.width * (double)info.height;
        double total_mb     = total_pixels * info.bpp / (1024.0 * 1024.0);
        double wall_s       = wall_ms / 1000.0;
        double mpps         = (total_pixels / 1e6) / wall_s;
        double mbs          = total_mb / wall_s;

        char ts_start[16], ts_end[16];
        fmt_timestamp(t_abs_start, ts_start);
        fmt_timestamp(t_abs_end,   ts_end);

        fprintf(stdout, "\n--- Pipeline Timing Report ---\n");
        fprintf(stdout, "Started          : %s\n", ts_start);
        fprintf(stdout, "Finished         : %s\n", ts_end);
        fprintf(stdout, "Wall time        : %.1f ms\n", wall_ms);
        fprintf(stdout, "Throughput       : %.2f MP/s  |  %.1f MB/s (uncompressed)\n",
                mpps, mbs);

        fprintf(stdout, "\nLoader           : %lld ms  (%lld strips)\n",
                stats.load_ms.load(), stats.load_count.load());

        long long gpu_h2d_ms    = stats.gpu_h2d_us.load()    / 1000LL;
        long long gpu_kernel_ms = stats.gpu_kernel_us.load() / 1000LL;
        long long gpu_d2h_ms    = stats.gpu_d2h_us.load()    / 1000LL;
        fprintf(stdout, "GPU (wall)       : %lld ms  (%lld strips)\n",
                stats.gpu_ms.load(), stats.gpu_count.load());
        fprintf(stdout, "  H2D transfer   : %lld ms\n",    gpu_h2d_ms);
        fprintf(stdout, "  Kernels        : %lld ms\n",    gpu_kernel_ms);
        fprintf(stdout, "  D2H transfer   : %lld ms\n",    gpu_d2h_ms);

        fprintf(stdout, "Deflate          : %lld ms  (%lld strips)\n",
                stats.deflate_ms.load(), stats.deflate_count.load());
        {
            long long nt      = (long long)dcfg.num_threads;
            long long init_ms = stats.deflate_init_us.load() / 1000LL;
            long long cmp_ms  = stats.deflate_compress_us.load() / 1000LL;
            // These are sums across all parallel chunks; divide by thread count
            // to get the wall-clock contribution.
            fprintf(stdout,
                "  z_stream init  : %lld ms total (%lld ms wall, %lld chunks)\n",
                init_ms, init_ms / nt,
                stats.deflate_count.load() * nt);
            fprintf(stdout,
                "  deflate()      : %lld ms total (%lld ms wall)\n",
                cmp_ms, cmp_ms / nt);
        }
        fprintf(stdout, "Writer           : %lld ms  (%lld strips)\n",
                stats.write_ms.load(), stats.write_count.load());

        fprintf(stdout, "\n(Stages run concurrently — stage totals exceed wall time)\n");
    }

    return !error.load();
}

#if defined(GPU_PNG_MODERN_DEFLATE)
// ---------------------------------------------------------------------------
// Modern pipeline -- RTX 5050 GPU-deflate, overlapped (PRD "Final Phase").
//
// Three stages on three threads, connected by BoundedQueues, mirroring the
// legacy 4-stage pipeline's spirit but adapted to GPU-resident data:
//
//   [Filter thread] --qa--> [Compress+Adler+Assemble thread] --qb--> [Writer thread]
//
// Filter stage: round-robins strips across a POOL of N independent
// GpuFilterContext instances (N = "CUDA stream pool" size, Requirement 2),
// each with its own stream/buffers. PNG Up/Paeth filtering has an inherent
// strip-to-strip dependency (row R needs row R-1's preprocessed value), so
// the true previous strip's last row is threaded through EXPLICITLY via
// gpu_filter_copy_prior_row_to_host()/h_prior_row, overriding each pool
// context's own (otherwise-wrong-when-round-robined) internal carry. This
// is what makes it safe to distribute strips across independent contexts.
//
// Compress+Adler+Assemble stage: GPU deflate is inherently sequential
// (its running bit position is cumulative across strips -- see
// gpu_deflate_backend.h), so this stage stays single-threaded/single-stream,
// consuming filtered strips in order. After each strip it also runs GPU
// Adler32 (Requirement 3) directly on the GPU-resident filtered data, then
// incrementally flushes whatever compressed bytes are now GUARANTEED
// COMPLETE (gpu_deflate_flushable_byte_length() -- never a byte a later
// strip might still write into) as a standalone IDAT chunk, pushed to the
// writer queue. This is what lets disk I/O for early chunks overlap with
// compression of later strips -- the dominant win, since the benchmark that
// motivated this phase showed "Final Write" alone (~90-100 ms) was over
// half of total wall time in the old strictly-sequential design.
//
// Writer stage: pops finished byte ranges and fwrites them, while the
// compress stage has already moved on to the next strip.
//
// Buffer-reuse safety: the filter thread may run at most (queue_capacity)
// strips ahead of what the compress thread has popped, plus the one strip
// each thread is actively working on right now -- so the pool must satisfy
// N >= queue_capacity + 2. queue_capacity is set to EXACTLY (N - 2) (the
// tightest value that still satisfies this), clamped to >= 1, so N is
// clamped to >= 3 by the caller (run_pipeline()'s dispatch block).
// ---------------------------------------------------------------------------
struct ModernPipelineStats {
    std::atomic<long long> load_ms              {0};
    std::atomic<long long> gpu_ms               {0};
    std::atomic<long long> gpu_h2d_us           {0};
    std::atomic<long long> gpu_kernel_us        {0};
    std::atomic<long long> gpu_d2h_us           {0};
    std::atomic<long long> deflate_ms           {0};
    std::atomic<long long> adler_ms             {0};
    std::atomic<long long> assemble_ms          {0};
    std::atomic<long long> write_ms             {0};
    std::atomic<long long> filter_queue_wait_ms {0};  // time blocked in qa.push()/pop()
    std::atomic<long long> write_queue_wait_ms  {0};  // time blocked in qb.push()/pop()
    std::atomic<long long> filter_queue_depth_sum     {0};
    std::atomic<long long> filter_queue_depth_samples {0};
    std::atomic<long long> write_queue_depth_sum      {0};
    std::atomic<long long> write_queue_depth_samples  {0};
    std::atomic<int>       strip_count          {0};
};

struct GpuFilteredJob {
    int  strip_index = 0;
    int  actual_rows = 0;
    bool is_last      = false;
};

struct ModernWriteJob {
    std::vector<uint8_t> bytes;
};

// ---------------------------------------------------------------------------
// Stage 1: Filter. Reads strips and runs the GPU filter kernel via the
// context pool, threading the true prior-row dependency through explicitly.
// ---------------------------------------------------------------------------
static void modern_filter_stage(
    ImageSource&                     src,
    int                              strip_h,
    size_t                           row_b,
    const DicomPixelParams*          dicom,
    int                              total_strips,
    std::vector<GpuFilterContext*>&  pool,
    BoundedQueue<GpuFilteredJob>&    qa,
    std::atomic<bool>&               error,
    ModernPipelineStats&             stats)
{
    using Clock = std::chrono::high_resolution_clock;
    const int pool_n = (int)pool.size();

    std::vector<uint8_t> strip_buf((size_t)strip_h * row_b);
    std::vector<uint8_t> prev_row(row_b, 0);  // PNG spec: first row's "prior" is zeros

    for (int s = 0; s < total_strips && !error.load(); s++) {
        auto t0 = Clock::now();
        const int actual_rows = src.read_strip(strip_buf.data(), strip_h);
        auto t1 = Clock::now();
        stats.load_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        if (actual_rows <= 0) { error.store(true); break; }

        GpuFilterContext* ctx = pool[s % pool_n];

        GpuTimings gt;
        auto t2 = Clock::now();
        gpu_filter_process_from_host(ctx, strip_buf.data(), prev_row.data(),
                                     actual_rows, &gt, dicom);
        auto t3 = Clock::now();
        stats.gpu_ms        += std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
        stats.gpu_h2d_us    += (long long)(gt.h2d_ms    * 1000.f);
        stats.gpu_kernel_us += (long long)(gt.kernel_ms * 1000.f);
        stats.gpu_d2h_us    += (long long)(gt.d2h_ms    * 1000.f);

        // Thread the true last-row dependency forward explicitly -- see the
        // file-level comment on why this is required when round-robining
        // across an independent context pool.
        gpu_filter_copy_prior_row_to_host(ctx, prev_row.data());

        GpuFilteredJob job;
        job.strip_index = s;
        job.actual_rows = actual_rows;
        job.is_last      = (s == total_strips - 1);

        auto t4 = Clock::now();
        const bool pushed = qa.push(job);
        auto t5 = Clock::now();
        stats.filter_queue_wait_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count();
        if (!pushed) break;

        stats.filter_queue_depth_sum     += (long long)qa.size();
        stats.filter_queue_depth_samples += 1;
    }
    qa.close();
}

// ---------------------------------------------------------------------------
// Stage 2: Compress + Adler32 + incremental PNG IDAT assembly. Inherently
// sequential (GPU deflate's bit position is cumulative across strips), one
// context/stream, consuming strips strictly in order.
// ---------------------------------------------------------------------------
static void modern_compress_stage(
    std::vector<GpuFilterContext*>& pool,
    GpuDeflateContext*               gdef,
    GpuAdler32Context*                gadler,
    GpuPngAssembleContext*            passemble,
    int                               row_bytes_out,
    int                               zlib_level,
    BoundedQueue<GpuFilteredJob>&     qa,
    BoundedQueue<ModernWriteJob>&     qb,
    std::atomic<bool>&                error,
    ModernPipelineStats&              stats,
    uint32_t&                         final_adler_out)
{
    using Clock = std::chrono::high_resolution_clock;
    const int pool_n = (int)pool.size();

    ParallelDeflateState dstate;
    size_t flushed_bytes = 0;

    GpuFilteredJob job;
    for (;;) {
        auto tq0 = Clock::now();
        const bool got = qa.pop(job);
        auto tq1 = Clock::now();
        stats.filter_queue_wait_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(tq1 - tq0).count();
        if (!got || error.load()) break;

        GpuFilterContext* ctx          = pool[job.strip_index % pool_n];
        const uint8_t*    d_filtered    = gpu_filter_device_output(ctx);
        const size_t      filtered_bytes = gpu_filter_output_size(ctx, job.actual_rows);

        auto t0 = Clock::now();
        gpu_deflate_compress_strip(gdef, d_filtered, job.actual_rows, row_bytes_out, job.is_last);
        auto t1 = Clock::now();
        stats.deflate_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        auto t2 = Clock::now();
        const uint32_t strip_adler = gpu_adler32_compute(gadler, d_filtered, filtered_bytes);
        auto t3 = Clock::now();
        stats.adler_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

        DeflateResult dr;
        dr.strip_adler = strip_adler;
        dr.input_size  = filtered_bytes;
        accum_adler(dstate, dr);

        auto t4 = Clock::now();
        const size_t avail = gpu_deflate_flushable_byte_length(gdef);
        if (avail > flushed_bytes) {
            const size_t slice_len = avail - flushed_bytes;
            const bool   is_first  = (flushed_bytes == 0);
            const size_t chunk_len = gpu_png_assemble_idat_chunk(
                passemble, gpu_deflate_output(gdef) + flushed_bytes, slice_len,
                is_first, false, 0, zlib_level);

            ModernWriteJob wj;
            wj.bytes.resize(chunk_len);
            gpu_png_assemble_copy_to_host(passemble, wj.bytes.data(), chunk_len);
            flushed_bytes = avail;

            auto tw0 = Clock::now();
            const bool pushed = qb.push(std::move(wj));
            auto tw1 = Clock::now();
            stats.write_queue_wait_ms +=
                std::chrono::duration_cast<std::chrono::milliseconds>(tw1 - tw0).count();
            if (!pushed) { error.store(true); break; }

            stats.write_queue_depth_sum     += (long long)qb.size();
            stats.write_queue_depth_samples += 1;
        }
        auto t5 = Clock::now();
        stats.assemble_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count();

        stats.strip_count += 1;
    }

    // Final flush: whatever bytes remain (including the byte that was only
    // partially written during the loop, now final since no more strips
    // will ever write to it) + the Adler-32 trailer, as the closing chunk.
    if (!error.load()) {
        const size_t   total_bytes  = gpu_deflate_output_byte_length(gdef);
        const size_t   slice_len    = total_bytes - flushed_bytes;
        const bool     is_first     = (flushed_bytes == 0);
        const uint32_t final_adler  = (uint32_t)dstate.running_adler;

        const size_t chunk_len = gpu_png_assemble_idat_chunk(
            passemble, gpu_deflate_output(gdef) + flushed_bytes, slice_len,
            is_first, true, final_adler, zlib_level);
        ModernWriteJob wj;
        wj.bytes.resize(chunk_len);
        gpu_png_assemble_copy_to_host(passemble, wj.bytes.data(), chunk_len);
        qb.push(std::move(wj));

        final_adler_out = final_adler;
    }

    qb.close();
}

// ---------------------------------------------------------------------------
// Stage 3: Writer. Pops finished byte ranges and fwrites them.
// ---------------------------------------------------------------------------
static void modern_writer_stage(
    BoundedQueue<ModernWriteJob>& qb,
    FILE*                          f,
    std::atomic<bool>&             error,
    ModernPipelineStats&           stats)
{
    using Clock = std::chrono::high_resolution_clock;
    ModernWriteJob job;
    while (qb.pop(job)) {
        if (error.load()) continue;  // drain without writing once a failure is flagged
        auto t0 = Clock::now();
        fwrite(job.bytes.data(), 1, job.bytes.size(), f);
        auto t1 = Clock::now();
        stats.write_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    }
}

static bool run_one_modern(
    const PipelineConfig&            cfg,
    const char*                      output_path,
    const char*                      source_label,
    ImageSource&                     src,
    std::vector<GpuFilterContext*>&  filter_pool,
    GpuDeflateContext*               gdef,
    GpuAdler32Context*                gadler,
    GpuPngAssembleContext*            passemble,
    bool                              print_header,
    bool                              print_report)
{
    using Clock = std::chrono::high_resolution_clock;

    const ImageInfo& info   = src.info();
    const int        strip_h = cfg.strip_height;
    const size_t     row_b   = (size_t)info.width * info.bpp;
    const int        row_bytes_out = (int)(row_b + 1);  // filter byte + row
    const DicomPixelParams* dicom = src.dicom_params();

    for (GpuFilterContext* ctx : filter_pool) gpu_filter_reset(ctx);
    gpu_deflate_reset(gdef);

    auto t_abs_start = std::chrono::system_clock::now();
    if (print_header) {
        char ts[16];
        fmt_timestamp(t_abs_start, ts);
        fprintf(stdout,
            "[%s] %s  %ux%u  ch=%d  bps=%d  strip_h=%d  streams=%d  "
            "[modern: custom GPU deflate, overlapped pipeline, UNVERIFIED on real hardware]\n",
            ts, source_label, info.width, info.height,
            info.channels, info.bits_per_sample, strip_h, (int)filter_pool.size());
    }

    FILE* f = fopen(output_path, "wb");
    if (!f) {
        fprintf(stderr, "Cannot open output: %s\n", output_path);
        return false;
    }

    // PNG signature + IHDR chunk. A few dozen fixed bytes -- constructed
    // directly on the host (see gpu_png_assemble.h for why these stay off
    // the GPU-resident path; only the bulk IDAT data goes through it).
    {
        static const uint8_t PNG_SIG[8] = {137, 80, 78, 71, 13, 10, 26, 10};
        fwrite(PNG_SIG, 1, 8, f);

        uint8_t ihdr[8 + 13 + 4];
        ihdr[0] = 0; ihdr[1] = 0; ihdr[2] = 0; ihdr[3] = 13;
        memcpy(ihdr + 4, "IHDR", 4);
        uint8_t* d = ihdr + 8;
        d[0] = (uint8_t)(info.width  >> 24); d[1] = (uint8_t)(info.width  >> 16);
        d[2] = (uint8_t)(info.width  >> 8);  d[3] = (uint8_t)(info.width);
        d[4] = (uint8_t)(info.height >> 24); d[5] = (uint8_t)(info.height >> 16);
        d[6] = (uint8_t)(info.height >> 8);  d[7] = (uint8_t)(info.height);
        d[8]  = (uint8_t)info.bits_per_sample;
        d[9]  = (uint8_t)channels_to_color_type(info.channels);
        d[10] = 0; d[11] = 0; d[12] = 0;
        const unsigned long ihdr_crc = crc32_of(ihdr + 4, 4 + 13);
        uint8_t* c = ihdr + 8 + 13;
        c[0] = (uint8_t)(ihdr_crc >> 24); c[1] = (uint8_t)(ihdr_crc >> 16);
        c[2] = (uint8_t)(ihdr_crc >> 8);  c[3] = (uint8_t)(ihdr_crc);
        fwrite(ihdr, 1, sizeof(ihdr), f);
    }

    const int total_strips = ((int)info.height + strip_h - 1) / strip_h;

    // Tightest safe queue capacity for the filter->compress queue -- see the
    // file-level comment above on buffer-reuse safety (capacity = N - 2,
    // clamped >= 1; run_pipeline()'s dispatch block clamps N >= 3).
    const size_t qa_capacity = (size_t)std::max(1, (int)filter_pool.size() - 2);
    const size_t qb_capacity = 2;

    BoundedQueue<GpuFilteredJob>  qa(qa_capacity);
    BoundedQueue<ModernWriteJob>  qb(qb_capacity);
    std::atomic<bool> error{false};
    ModernPipelineStats stats;
    uint32_t final_adler = 0;

    auto t_wall_start = Clock::now();

    std::thread t_filter([&] {
        modern_filter_stage(src, strip_h, row_b, dicom, total_strips,
                            filter_pool, qa, error, stats);
    });
    std::thread t_compress([&] {
        modern_compress_stage(filter_pool, gdef, gadler, passemble, row_bytes_out,
                              cfg.deflate_level, qa, qb, error, stats, final_adler);
    });
    std::thread t_writer([&] {
        modern_writer_stage(qb, f, error, stats);
    });

    t_filter.join();
    t_compress.join();
    t_writer.join();

    const bool ok = !error.load();
    if (ok) {
        uint8_t iend[12];
        iend[0] = 0; iend[1] = 0; iend[2] = 0; iend[3] = 0;
        memcpy(iend + 4, "IEND", 4);
        const unsigned long iend_crc = crc32_of(iend + 4, 4);
        iend[8] = (uint8_t)(iend_crc >> 24); iend[9]  = (uint8_t)(iend_crc >> 16);
        iend[10] = (uint8_t)(iend_crc >> 8); iend[11] = (uint8_t)(iend_crc);
        fwrite(iend, 1, 12, f);
    }

    auto t_wall_end = Clock::now();
    fclose(f);
    auto t_abs_end = std::chrono::system_clock::now();

    if (!ok) {
        fprintf(stderr, "Modern pipeline encoding failed for %s\n", output_path);
        return false;
    }

    if (print_report) {
        double wall_ms = (double)std::chrono::duration_cast<std::chrono::microseconds>(
            t_wall_end - t_wall_start).count() / 1000.0;
        double total_pixels = (double)info.width * (double)info.height;
        double total_mb     = total_pixels * info.bpp / (1024.0 * 1024.0);
        double mpps         = (total_pixels / 1e6) / (wall_ms / 1000.0);
        double mbs          = total_mb / (wall_ms / 1000.0);

        char ts_start[16], ts_end[16];
        fmt_timestamp(t_abs_start, ts_start);
        fmt_timestamp(t_abs_end,   ts_end);

        const long long load_ms      = stats.load_ms.load();
        const long long gpu_ms_v     = stats.gpu_ms.load();
        const long long gpu_h2d_ms   = stats.gpu_h2d_us.load()    / 1000LL;
        const long long gpu_kernel_ms = stats.gpu_kernel_us.load() / 1000LL;
        const long long gpu_d2h_ms   = stats.gpu_d2h_us.load()    / 1000LL;
        const long long deflate_ms   = stats.deflate_ms.load();
        const long long adler_ms     = stats.adler_ms.load();
        const long long assemble_ms  = stats.assemble_ms.load();
        const long long write_ms     = stats.write_ms.load();
        const long long fq_wait_ms   = stats.filter_queue_wait_ms.load();
        const long long wq_wait_ms   = stats.write_queue_wait_ms.load();

        const double sum_stage_ms = (double)(load_ms + gpu_ms_v + deflate_ms +
                                             adler_ms + assemble_ms + write_ms);
        const double overlap_efficiency = (sum_stage_ms > 0.0)
            ? std::max(0.0, (sum_stage_ms - wall_ms) / sum_stage_ms) * 100.0
            : 0.0;

        const size_t total_compressed_bytes = gpu_deflate_output_byte_length(gdef);
        const double uncompressed_filtered_bytes = (double)(row_b + 1) * (double)info.height;
        const double compression_ratio = (uncompressed_filtered_bytes > 0.0)
            ? (double)total_compressed_bytes / uncompressed_filtered_bytes : 0.0;
        const double deflate_throughput_gbs = (deflate_ms > 0)
            ? (uncompressed_filtered_bytes / (1024.0*1024.0*1024.0)) / ((double)deflate_ms / 1000.0)
            : 0.0;
        const double h2d_bandwidth_gbs = (gpu_h2d_ms > 0)
            ? (total_mb / 1024.0) / ((double)gpu_h2d_ms / 1000.0) : 0.0;
        const double d2h_bandwidth_gbs = (gpu_d2h_ms > 0)
            ? (total_mb / 1024.0) / ((double)gpu_d2h_ms / 1000.0) : 0.0;

        const long long fq_depth_n = stats.filter_queue_depth_samples.load();
        const long long wq_depth_n = stats.write_queue_depth_samples.load();
        const double avg_fq_depth = fq_depth_n > 0
            ? (double)stats.filter_queue_depth_sum.load() / (double)fq_depth_n : 0.0;
        const double avg_wq_depth = wq_depth_n > 0
            ? (double)stats.write_queue_depth_sum.load() / (double)wq_depth_n : 0.0;

        fprintf(stdout, "\n--- Modern Pipeline Timing Report (overlapped, custom GPU deflate) ---\n");
        fprintf(stdout, "Started             : %s\n", ts_start);
        fprintf(stdout, "Finished            : %s\n", ts_end);
        fprintf(stdout, "Wall time           : %.1f ms\n", wall_ms);
        fprintf(stdout, "Throughput          : %.2f MP/s  |  %.1f MB/s (uncompressed)\n", mpps, mbs);
        // "Stream utilization" here is a coarse proxy (filter stage's own
        // wall-clock share of the whole pipeline's wall time), NOT true SM
        // occupancy -- that needs NVIDIA profiling tools (Nsight Systems/
        // Compute, CUPTI) instrumenting the actual hardware, which this
        // process cannot measure from inside itself. Likewise "concurrent
        // kernel count" is intentionally not reported here for the same
        // reason: fabricating a number without a profiler backing it would
        // be worse than omitting it. Use Nsight Systems against this binary
        // for both if you need hardware-verified figures.
        const double stream_utilization_pct = (wall_ms > 0.0)
            ? std::min(100.0, (double)gpu_ms_v / wall_ms * 100.0) : 0.0;

        fprintf(stdout, "\nGPU Streams         : %d\n", (int)filter_pool.size());
        fprintf(stdout, "Stream Utilization  : %.0f%%  (approx: filter-stage wall time / pipeline "
                        "wall time -- not true SM occupancy; profile with Nsight for that)\n",
                stream_utilization_pct);
        fprintf(stdout, "Overlap Efficiency  : %.0f%%  ((sum of stage times - wall) / sum of stage times)\n",
                overlap_efficiency);
        fprintf(stdout, "\nLoader              : %lld ms  (%d strips)\n", load_ms, stats.strip_count.load());
        fprintf(stdout, "GPU filter (wall)   : %lld ms\n", gpu_ms_v);
        fprintf(stdout, "  H2D transfer      : %lld ms  (%.2f GB/s)\n", gpu_h2d_ms, h2d_bandwidth_gbs);
        fprintf(stdout, "  Kernels           : %lld ms\n", gpu_kernel_ms);
        fprintf(stdout, "  D2H transfer      : %lld ms  (%.2f GB/s -- output unused in this "
                        "path, see gpu_filter.cu; kept because legacy shares this code)\n",
                gpu_d2h_ms, d2h_bandwidth_gbs);
        fprintf(stdout, "GPU deflate         : %lld ms  (%.2f GB/s)\n", deflate_ms, deflate_throughput_gbs);
        fprintf(stdout, "GPU Adler32         : %lld ms\n", adler_ms);
        fprintf(stdout, "GPU PNG assemble    : %lld ms  (%zu compressed bytes)\n",
                assemble_ms, total_compressed_bytes);
        fprintf(stdout, "Writer              : %lld ms\n", write_ms);
        fprintf(stdout, "\nFilter->Compress queue wait : %lld ms  (avg depth %.1f / capacity %zu)\n",
                fq_wait_ms, avg_fq_depth, qa_capacity);
        fprintf(stdout, "Compress->Writer queue wait : %lld ms  (avg depth %.1f / capacity %zu)\n",
                wq_wait_ms, avg_wq_depth, qb_capacity);
        fprintf(stdout, "\nCompression Ratio   : %.2f\n", compression_ratio);
        fprintf(stdout, "Final Adler-32      : 0x%08x  (decode-compare against the CPU path's "
                        "checksum for the same input)\n", final_adler);
        fprintf(stdout, "\n(3 stages overlap on independent threads -- wall time can be, "
                        "and should be, less than the sum of stage times above)\n");
    }

    return true;
}
#endif  // GPU_PNG_MODERN_DEFLATE

// ---------------------------------------------------------------------------
// Internal: single-image convenience wrapper.
// Builds a GPU context (+ deflate backend), runs one image, tears them down.
// Dispatches to the modern GPU-deflate path only when cfg.use_gpu_deflate is
// explicitly set AND a modern GPU is detected; otherwise uses the proven
// CPU-deflate 4-stage pipeline (run_one()) unconditionally. See
// PipelineConfig::use_gpu_deflate for why this defaults to off.
// ---------------------------------------------------------------------------
static bool run_pipeline(
    const PipelineConfig& cfg,
    const char*           output_path,
    const char*           source_label,
    ImageSource&          src)
{
    const ImageInfo& info = src.info();

#if defined(GPU_PNG_MODERN_DEFLATE)
    if (cfg.use_gpu_deflate) {
        GpuCapability gpu_cap = detect_gpu_capability();
        if (gpu_cap.mode == GpuPipelineMode::Modern) {
            // gpu_deflate's per-row bit-length scan is a single CUDA block
            // (one thread per row), so strip_height is hard-capped at 1024.
            // Caught here, before any GPU work starts, instead of deep
            // inside gpu_deflate_compress_strip()'s per-strip loop.
            if (cfg.strip_height > 1024) {
                fprintf(stderr,
                    "--gpu-deflate requires --strip-height <= 1024 (got %d).\n",
                    cfg.strip_height);
                return false;
            }

            // Stream pool size (Requirement 2): explicit --gpu-streams wins;
            // otherwise auto-default 8 for the modern path. Clamped to >= 3
            // so the filter->compress queue capacity (N-2) is always >= 1 --
            // see modern_filter_stage's file-level buffer-reuse-safety comment.
            int pool_size = (cfg.gpu_streams > 0) ? cfg.gpu_streams : 8;
            pool_size = std::max(pool_size, 3);

            std::vector<GpuFilterContext*> filter_pool;
            filter_pool.reserve(pool_size);
            for (int i = 0; i < pool_size; i++) {
                GpuFilterContext* ctx = gpu_filter_create(info.width, info.bpp, cfg.strip_height);
                if (!ctx) {
                    fprintf(stderr, "gpu_filter_create failed (pool slot %d)\n", i);
                    for (GpuFilterContext* c : filter_pool) gpu_filter_destroy(c);
                    return false;
                }
                filter_pool.push_back(ctx);
            }

            const size_t row_bytes      = (size_t)info.width * info.bpp + 1;
            const int    total_strips    = ((int)info.height + cfg.strip_height - 1) / cfg.strip_height;
            const size_t total_filtered = (size_t)info.height * row_bytes;
            // Fixed-Huffman literal codes are at most 9 bits; +10 bits/strip
            // covers the 3-bit header + 7-bit EOB. Provable upper bound, not
            // a heuristic guess -- see gpu_deflate_backend.h.
            const size_t max_total_bits = total_filtered * 9 + (size_t)total_strips * 10;
            const size_t max_compressed_bytes = (max_total_bits + 7) / 8 + 8;
            // Largest single incremental flush is bounded by the whole
            // image's compressed size (the rare single-strip-image case
            // flushes everything in one chunk), plus 6 bytes slack for the
            // zlib header / Adler-32 trailer.
            const size_t max_chunk_data_bytes = max_compressed_bytes + 6;

            GpuDeflateContext* gdef = gpu_deflate_create(
                cfg.strip_height, (int)row_bytes, max_total_bits);
            GpuAdler32Context* gadler = gpu_adler32_create(total_filtered);
            GpuPngAssembleContext* passemble = gpu_png_assemble_create(max_chunk_data_bytes);

            bool ok = run_one_modern(cfg, output_path, source_label, src, filter_pool,
                                     gdef, gadler, passemble, cfg.verbose, cfg.verbose);

            gpu_png_assemble_destroy(passemble);
            gpu_adler32_destroy(gadler);
            gpu_deflate_destroy(gdef);
            for (GpuFilterContext* c : filter_pool) gpu_filter_destroy(c);
            return ok;
        }
        if (cfg.verbose)
            fprintf(stdout, "use_gpu_deflate requested but no modern GPU detected; using CPU deflate.\n");
    }
#endif

    GpuFilterContext* gpu = gpu_filter_create(info.width, info.bpp, cfg.strip_height);
    if (!gpu) {
        fprintf(stderr, "gpu_filter_create failed\n");
        return false;
    }

    ThreadPool pool(cfg.deflate_threads);

    bool ok = run_one(cfg, output_path, source_label, src, gpu, pool,
                      cfg.verbose, cfg.verbose);

    gpu_filter_destroy(gpu);
    return ok;
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------
bool encode_tiff_to_png(const char* input_path,
                        const char* output_path,
                        const PipelineConfig& cfg)
{
    TiffSource src;
    if (!src.open(input_path)) {
        fprintf(stderr, "Cannot open TIFF: %s\n", input_path);
        return false;
    }
    return run_pipeline(cfg, output_path, "TIFF", src);
}

bool encode_raw_to_png(const char* input_path,
                       const char* output_path,
                       const PipelineConfig& cfg)
{
    RawSource src;
    if (!src.open(input_path)) {
        fprintf(stderr, "Cannot open RAW: %s\n", input_path);
        return false;
    }
    return run_pipeline(cfg, output_path, "RAW", src);
}

bool encode_dicom_to_png(const char* input_path,
                         const char* output_path,
                         const PipelineConfig& cfg)
{
    DicomSource src;
    if (!src.open(input_path)) return false;

    const int frame = (cfg.frame >= 0) ? cfg.frame : 0;
    if (frame >= src.num_frames()) {
        fprintf(stderr, "DICOM: requested frame %d but file has %d frame(s)\n",
                frame, src.num_frames());
        return false;
    }
    if (cfg.verbose && src.num_frames() > 1)
        fprintf(stdout, "DICOM: %d frames; exporting frame %d\n",
                src.num_frames(), frame);

    if (!src.load_frame(frame)) return false;
    return run_pipeline(cfg, output_path, "DICOM", src);
}

bool encode_dicom_all_frames_to_png(const char* input_path,
                                    const char* output_dir,
                                    const PipelineConfig& cfg)
{
    DicomSource src;
    if (!src.open(input_path)) return false;

    const int n = src.num_frames();

    // Create the output folder (and any parents).
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    if (ec) {
        fprintf(stderr, "Cannot create output folder '%s': %s\n",
                output_dir, ec.message().c_str());
        return false;
    }

    fprintf(stdout, "Detected %d frame%s\n", n, (n == 1 ? "" : "s"));

    // Decide how many concurrent GPU streams to use. Each worker owns one
    // GpuFilterContext (and therefore one cudaStream_t) plus one deflate
    // ThreadPool for its entire lifetime -- frames it processes share that
    // context/pool exactly the way the original single-worker design did,
    // so this is correct on the legacy pipeline with worker_count=1 (no
    // change in behavior) and adds real concurrency on the modern pipeline.
    int worker_count = cfg.gpu_streams;
    if (worker_count <= 0) {
        GpuCapability gpu_cap = detect_gpu_capability();
        worker_count = (gpu_cap.mode == GpuPipelineMode::Modern) ? 4 : 1;
    }
    worker_count = std::max(1, std::min(worker_count, n));

    // Split deflate threads across workers so total CPU thread count stays
    // close to cfg.deflate_threads regardless of worker_count.
    const int per_worker_deflate_threads =
        std::max(1, cfg.deflate_threads / worker_count);

    fprintf(stdout, "Exporting all frames... (%d concurrent stream%s, %d deflate thread%s/stream)\n",
            worker_count, (worker_count == 1 ? "" : "s"),
            per_worker_deflate_threads, (per_worker_deflate_threads == 1 ? "" : "s"));

    const ImageInfo& info = src.info();
    const std::string output_dir_str(output_dir);

    std::atomic<int>  next_frame{0};
    std::atomic<bool> all_ok{true};
    std::atomic<bool> header_printed{false};
    std::mutex        print_mtx;

    auto t0 = std::chrono::high_resolution_clock::now();

    auto worker_fn = [&]() {
        DicomSource local_src;
        if (!local_src.open(input_path)) { all_ok = false; return; }

        GpuFilterContext* gpu = gpu_filter_create(info.width, info.bpp, cfg.strip_height);
        if (!gpu) {
            fprintf(stderr, "gpu_filter_create failed\n");
            all_ok = false;
            return;
        }
        ThreadPool deflate_pool(per_worker_deflate_threads);

        for (;;) {
            const int i = next_frame.fetch_add(1);
            if (i >= n || !all_ok.load()) break;

            if (!local_src.load_frame(i)) { all_ok = false; break; }

            char fname[32];
            snprintf(fname, sizeof(fname), "slice_%04d.png", i);
            std::filesystem::path outp = std::filesystem::path(output_dir_str) / fname;
            const std::string outs = outp.string();

            // Show the header/report exactly once, for whichever frame happens to
            // reach this point first across workers (nondeterministic with > 1
            // worker; the printed per-frame timing belongs to that frame, not
            // necessarily frame 0 -- fine for a diagnostic sample, not used for
            // correctness).
            bool hdr = false;
            if (cfg.verbose) {
                bool expected = false;
                hdr = header_printed.compare_exchange_strong(expected, true);
            }

            {
                std::lock_guard<std::mutex> lk(print_mtx);
                fprintf(stdout, "[%d/%d] %s\n", i + 1, n, fname);
            }

            if (!run_one(cfg, outs.c_str(), "DICOM", local_src, gpu, deflate_pool, hdr, hdr)) {
                all_ok = false;
                break;
            }
        }

        gpu_filter_destroy(gpu);
    };

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (int w = 0; w < worker_count; w++)
        workers.emplace_back(worker_fn);
    for (auto& t : workers)
        t.join();

    auto t1 = std::chrono::high_resolution_clock::now();

    if (all_ok.load()) {
        double secs =
            (double)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
            / 1000.0;
        fprintf(stdout, "Finished exporting %d PNG%s in %.1f s\n",
                n, (n == 1 ? "" : "s"), secs);
    } else {
        fprintf(stderr, "Export aborted after an error.\n");
    }
    return all_ok.load();
}
