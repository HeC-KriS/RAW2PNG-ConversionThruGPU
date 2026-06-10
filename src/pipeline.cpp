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
// Expected steady-state throughput (17043×11710 RGB-16):
//   Loader  : ~16 ms/strip   (bottleneck on spinning HDD)
//   GPU     : ~10 ms/strip
//   Deflate : ~14 ms/strip   (6 threads, zlib level 3)
//   Total   : ~3–4 s  (bottleneck × 183 strips + start/teardown)

#include "pipeline.h"
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
#include <thread>
#include <vector>



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
struct PipelineStats {
    // Wall-clock totals per stage (milliseconds)
    std::atomic<long long> load_ms{0};
    std::atomic<long long> gpu_ms{0};
    std::atomic<long long> deflate_ms{0};
    std::atomic<long long> write_ms{0};

    // Strip counts
    std::atomic<long long> load_count{0};
    std::atomic<long long> gpu_count{0};
    std::atomic<long long> deflate_count{0};
    std::atomic<long long> write_count{0};

    // GPU sub-phase totals in microseconds (from CUDA Events)
    std::atomic<long long> gpu_h2d_us{0};
    std::atomic<long long> gpu_kernel_us{0};
    std::atomic<long long> gpu_d2h_us{0};
};


// ---------------------------------------------------------------------------
// Stage 1 – Loader
// Reads strips from the image source and enqueues them.
// Runs in a dedicated thread so I/O overlaps with GPU and CPU work.
// ---------------------------------------------------------------------------
static void loader_stage(
    int (*read_strip)(void*, uint8_t*, int),
    void* reader,
    int strip_height,
    size_t row_bytes,
    uint32_t image_height,
    BoundedQueue<StripJob>& qa,
    std::atomic<bool>& error,
    PipelineStats& stats)
{
    int total_strips = ((int)image_height + strip_height - 1) / strip_height;

    for (int s = 0; s < total_strips && !error.load(); s++) {
        StripJob job;
        job.strip_index = s;
        job.data.resize((size_t)strip_height * row_bytes);

        auto t0 = std::chrono::high_resolution_clock::now();

        job.actual_rows =
            read_strip(reader, job.data.data(), strip_height);

        auto t1 = std::chrono::high_resolution_clock::now();

        stats.load_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(
                t1 - t0).count();

        stats.load_count++;
        if (job.actual_rows <= 0) {
            error.store(true);
            break;
        }
        job.is_last = (s == total_strips - 1);

        if (!qa.push(std::move(job))) break;  // queue closed
    }
    qa.close();
}

// ---------------------------------------------------------------------------
// Stage 2 – GPU Filter
// Pulls raw strips, applies all 5 PNG filters + adaptive row selection.
// Runs in its own thread to keep the CUDA context isolated.
// ---------------------------------------------------------------------------
static void gpu_stage(
    GpuFilterContext* gpu,
    BoundedQueue<StripJob>& qa,
    BoundedQueue<FilteredJob>& qb,
    std::atomic<bool>& error,
    PipelineStats& stats)
{
    StripJob job;
    while (!error.load() && qa.pop(job)) {
        auto t0 = std::chrono::high_resolution_clock::now();

        GpuTimings gt;
        const uint8_t* filtered = gpu_filter_process_from_host(
            gpu,
            job.data.data(),
            nullptr,          // prior row tracked internally by GpuFilterContext
            job.actual_rows,
            &gt);
        auto t1 = std::chrono::high_resolution_clock::now();

        stats.gpu_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(
                t1 - t0).count();

        // Accumulate CUDA Event sub-phase timings (convert ms → us)
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
// Pulls filtered strips and compresses them using the thread pool.
// Adler-32 is NOT accumulated here; each CompressedJob carries its own
// strip_adler so the writer can combine them in order.
// ---------------------------------------------------------------------------
static void deflate_stage(
    BoundedQueue<FilteredJob>& qb,
    BoundedQueue<CompressedJob>& qc,
    const ParallelDeflateConfig& dcfg,
    ThreadPool& pool,
    std::atomic<bool>& error,
    PipelineStats& stats)
{
    FilteredJob fjob;
    while (!error.load() && qb.pop(fjob)) {
        auto t0 = std::chrono::high_resolution_clock::now();

        DeflateResult dr = deflate_strip(
            fjob.data.data(), fjob.data.size(),
            dcfg, pool, fjob.is_last);

        auto t1 = std::chrono::high_resolution_clock::now();
        stats.deflate_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        stats.deflate_count++;

        CompressedJob cjob;
        cjob.strip_index  = fjob.strip_index;
        cjob.is_last      = fjob.is_last;
        cjob.data         = std::move(dr.data);
        cjob.strip_adler  = dr.strip_adler;
        cjob.input_size   = dr.input_size;

        if (!qc.push(std::move(cjob))) break;
    }
    qc.close();
}

// ---------------------------------------------------------------------------
// Stage 4 – Writer
// Pulls compressed jobs in order and writes PNG IDAT chunks.
// Accumulates the running Adler-32 and writes the zlib trailer at the end.
// ---------------------------------------------------------------------------
static void writer_stage(
    BoundedQueue<CompressedJob>& qc,
    PngWriter& writer,
    int deflate_level,
    std::atomic<bool>& error,
    PipelineStats& stats)
{
    // Prepend zlib stream header before the first DEFLATE byte
    uint8_t zhdr[2];
    zlib_header(deflate_level, zhdr);
    writer.write_idat_bytes(zhdr, 2);

    ParallelDeflateState dstate;
    CompressedJob cjob;

    while (!error.load() && qc.pop(cjob)) {
        // accum_adler only needs strip_adler and input_size; build a
        // lightweight DeflateResult with an empty data field to avoid
        // copying the compressed bytes vector.
        DeflateResult dr_adler;
        dr_adler.strip_adler = cjob.strip_adler;
        dr_adler.input_size  = cjob.input_size;
        accum_adler(dstate, dr_adler);

        if (!cjob.data.empty())
        {
            auto t0 = std::chrono::high_resolution_clock::now();

            writer.write_idat_bytes(
                cjob.data.data(),
                cjob.data.size());

            auto t1 = std::chrono::high_resolution_clock::now();

            stats.write_ms +=
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    t1 - t0).count();

            stats.write_count++;
        }
    }
    // Append zlib Adler-32 trailer
    uint8_t ztrl[4];
    zlib_trailer(dstate, ztrl);
    writer.write_idat_bytes(ztrl, 4);
}

// ---------------------------------------------------------------------------
// Internal: run the 4-stage pipeline for any image source
// ---------------------------------------------------------------------------
static bool run_pipeline(
    const ImageInfo&      info,
    const PipelineConfig& cfg,
    const char*           output_path,
    int  (*read_strip)(void*, uint8_t*, int),
    void*                 reader)
{
    const int    strip_h = cfg.strip_height;
    const size_t row_b   = (size_t)info.width * info.bpp;

    // ---- GPU filter context (prior-row state lives here) ------------------
    GpuFilterContext* gpu = gpu_filter_create(info.width, info.bpp, strip_h);
    if (!gpu) {
        fprintf(stderr, "gpu_filter_create failed\n");
        return false;
    }

    // ---- Persistent thread pool for DEFLATE chunks -----------------------
    ThreadPool pool(cfg.deflate_threads);

    ParallelDeflateConfig dcfg;
    dcfg.num_threads = cfg.deflate_threads;
    dcfg.zlib_level  = cfg.deflate_level;

    // ---- PNG writer -------------------------------------------------------
    PngWriter writer;
    if (!writer.open(output_path, info.width, info.height,
                     (uint8_t)info.bits_per_sample,
                     channels_to_color_type(info.channels))) {
        fprintf(stderr, "Cannot open output: %s\n", output_path);
        gpu_filter_destroy(gpu);
        return false;
    }
    writer.begin_idat();

    // ---- Inter-stage queues (capacity 2 = double-buffer per PRD) ---------
    BoundedQueue<StripJob>      qa(2);
    BoundedQueue<FilteredJob>   qb(2);
    BoundedQueue<CompressedJob> qc(2);

    // ---- Shared error flag ------------------------------------------------
    std::atomic<bool> error(false);
    PipelineStats stats;

    // ---- Timing -----------------------------------------------------------
    clock_t t_start = clock();

    // ---- Launch all four stages concurrently ------------------------------
    std::thread t_load([&]{
        loader_stage(read_strip, reader,
                     strip_h, row_b, info.height,
                     qa, error, stats);
    });

    std::thread t_gpu([&]{
        gpu_stage(gpu, qa, qb, error, stats);
    });

    std::thread t_def([&]{
        deflate_stage(qb, qc, dcfg, pool, error, stats);
    });

    std::thread t_wri([&]{
        writer_stage(qc, writer, cfg.deflate_level, error, stats);
    });

    // ---- Wait for pipeline to drain ---------------------------------------
    t_load.join();
    t_gpu.join();
    t_def.join();
    t_wri.join();

    writer.end_idat();
    writer.close();
    gpu_filter_destroy(gpu);

    if (cfg.verbose) {
        double wall_ms = 1000.0 * (double)(clock() - t_start) / CLOCKS_PER_SEC;

        // Throughput
        double total_pixels = (double)info.width * (double)info.height;
        double total_mb     = total_pixels * info.bpp / (1024.0 * 1024.0);
        double wall_s       = wall_ms / 1000.0;
        double mpps         = (total_pixels / 1e6) / wall_s;
        double mbs          = total_mb / wall_s;

        fprintf(stdout, "\n--- Pipeline Timing Report ---\n");
        fprintf(stdout, "Wall time        : %.1f ms\n", wall_ms);
        fprintf(stdout, "Throughput       : %.2f MP/s  |  %.1f MB/s (uncompressed)\n",
                mpps, mbs);

        fprintf(stdout, "\nLoader           : %lld ms  (%lld strips)\n",
                stats.load_ms.load(), stats.load_count.load());

        // GPU: wall-clock total + CUDA Event breakdown
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
        fprintf(stdout, "Writer           : %lld ms  (%lld strips)\n",
                stats.write_ms.load(), stats.write_count.load());

        fprintf(stdout, "\n(Stages run concurrently — stage totals exceed wall time)\n");
    }

    return !error.load();
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------
static int tiff_strip_cb(void* r, uint8_t* out, int h)
{
    return tiff_read_strip(static_cast<TiffReader*>(r), out, h);
}

bool encode_tiff_to_png(const char* input_path,
                        const char* output_path,
                        const PipelineConfig& cfg)
{
    ImageInfo info;
    TiffReader* r = tiff_open(input_path, info);
    if (!r) { fprintf(stderr, "Cannot open TIFF: %s\n", input_path); return false; }
    if (cfg.verbose)
        fprintf(stdout, "TIFF  %ux%u  ch=%d  bps=%d  strip_h=%d  threads=%d  level=%d\n",
                info.width, info.height, info.channels,
                info.bits_per_sample, cfg.strip_height,
                cfg.deflate_threads, cfg.deflate_level);
    bool ok = run_pipeline(info, cfg, output_path, tiff_strip_cb, r);
    tiff_close(r);
    return ok;
}

static int raw_strip_cb(void* r, uint8_t* out, int h)
{
    return raw_read_strip(static_cast<RawReader*>(r), out, h);
}

bool encode_raw_to_png(const char* input_path,
                       const char* output_path,
                       const PipelineConfig& cfg)
{
    ImageInfo info;
    RawReader* r = raw_open(input_path, info);
    if (!r) { fprintf(stderr, "Cannot open RAW: %s\n", input_path); return false; }
    if (cfg.verbose)
        fprintf(stdout, "RAW  %ux%u  ch=%d  bps=%d\n",
                info.width, info.height, info.channels, info.bits_per_sample);
    bool ok = run_pipeline(info, cfg, output_path, raw_strip_cb, r);
    raw_close(r);
    return ok;
}
