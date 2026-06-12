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
#include "parallel_deflate.h"
#include "png_writer.h"
#include "image_loader.h"
#include "bounded_queue.h"
#include "strip_job.h"
#include "thread_pool.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
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

// ---------------------------------------------------------------------------
// Internal: single-image convenience wrapper.
// Builds a GPU context + thread pool, runs one image, tears them down.
// ---------------------------------------------------------------------------
static bool run_pipeline(
    const PipelineConfig& cfg,
    const char*           output_path,
    const char*           source_label,
    ImageSource&          src)
{
    const ImageInfo& info = src.info();

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
    fprintf(stdout, "Exporting all frames...\n");

    // Build GPU context + thread pool once; reuse across every frame.
    const ImageInfo& info = src.info();
    GpuFilterContext* gpu = gpu_filter_create(info.width, info.bpp, cfg.strip_height);
    if (!gpu) {
        fprintf(stderr, "gpu_filter_create failed\n");
        return false;
    }
    ThreadPool pool(cfg.deflate_threads);

    bool all_ok = true;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i++) {
        if (!src.load_frame(i)) { all_ok = false; break; }

        char fname[32];
        snprintf(fname, sizeof(fname), "slice_%04d.png", i);
        std::filesystem::path outp = std::filesystem::path(output_dir) / fname;
        const std::string outs = outp.string();

        fprintf(stdout, "[%d/%d] %s\n", i + 1, n, fname);

        // Show the header/report only for the first frame in verbose mode.
        const bool hdr = cfg.verbose && (i == 0);
        if (!run_one(cfg, outs.c_str(), "DICOM", src, gpu, pool, hdr, hdr)) {
            all_ok = false;
            break;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    gpu_filter_destroy(gpu);

    if (all_ok) {
        double secs =
            (double)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
            / 1000.0;
        fprintf(stdout, "Finished exporting %d PNG%s in %.1f s\n",
                n, (n == 1 ? "" : "s"), secs);
    } else {
        fprintf(stderr, "Export aborted after an error.\n");
    }
    return all_ok;
}
