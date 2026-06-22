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
    // Stage wall times (milliseconds)
    std::atomic<long long> load_ms       {0};
    std::atomic<long long> gpu_ms        {0};
    std::atomic<long long> deflate_ms    {0};
    std::atomic<long long> adler_ms      {0};
    std::atomic<long long> assemble_ms   {0};  // gpu_png_assemble_idat_chunk wall time
    std::atomic<long long> write_ms      {0};
    std::atomic<int>       strip_count   {0};

    // GPU filter sub-phases (microseconds, from CUDA Events)
    std::atomic<long long> gpu_h2d_us    {0};  // raw strip H2D
    std::atomic<long long> gpu_kernel_us {0};  // filter kernel
    std::atomic<long long> gpu_d2h_us    {0};  // WASTED: filtered output D2H (unused in modern path)

    // GPU deflate kernel sub-phases (microseconds, from CUDA Events)
    std::atomic<long long> deflate_lz77_us      {0};
    std::atomic<long long> deflate_bitlen_us     {0};
    std::atomic<long long> deflate_scan_us       {0};
    std::atomic<long long> deflate_encode_us     {0};
    std::atomic<long long> deflate_sync_us       {0};  // host wait per strip
    std::atomic<long long> deflate_stats_d2h_us  {0};  // LZ77 stats D2H (28 B/strip)
    std::atomic<long long> flush_query_us        {0};  // flushable_byte_length() D2H cost

    // PNG assembly sub-phases (microseconds)
    std::atomic<long long> assemble_header_us    {0};
    std::atomic<long long> assemble_d2d_us       {0};
    std::atomic<long long> assemble_presync_us   {0};
    std::atomic<long long> assemble_crc_us       {0};
    std::atomic<long long> assemble_crc_h2d_us   {0};
    std::atomic<long long> assemble_d2h_us       {0};  // chunk D2H to host

    // Queue wait times — separate push vs pop (previous code double-counted these)
    std::atomic<long long> filter_push_wait_ms   {0};  // filter blocked on qa.push()
    std::atomic<long long> compress_pop_wait_ms  {0};  // compress blocked on qa.pop()
    std::atomic<long long> write_push_wait_ms    {0};  // compress blocked on qb.push()
    std::atomic<long long> writer_pop_wait_ms    {0};  // writer blocked on qb.pop()

    // Queue depth stats
    std::atomic<long long> filter_queue_depth_sum     {0};
    std::atomic<long long> filter_queue_depth_samples {0};
    std::atomic<long long> filter_queue_max_depth     {0};
    std::atomic<long long> write_queue_depth_sum      {0};
    std::atomic<long long> write_queue_depth_samples  {0};
    std::atomic<long long> write_queue_max_depth      {0};
};

// Global cross-image accumulator for batch GPU summary.
// Zeroed by pipeline_reset_gpu_batch_stats(); read by pipeline_print_gpu_batch_summary().
struct GpuBatchAccum {
    std::atomic<int>       files              {0};
    // pipeline_wall_ms: run_one_modern() only (fopen → fclose).
    // Does NOT include DICOM decode, GPU context alloc, or batch overhead.
    std::atomic<long long> pipeline_wall_ms   {0};
    // file_total_ms: encode_*_to_png() entry → return, includes DICOM decode.
    // Accumulated via pipeline_record_file_times().
    std::atomic<long long> file_total_ms      {0};
    // dicom_decode_ms: DicomSource::open() + load_frame() only.
    std::atomic<long long> dicom_decode_ms    {0};
    std::atomic<long long> load_ms            {0};
    std::atomic<long long> gpu_ms             {0};
    std::atomic<long long> deflate_ms         {0};
    std::atomic<long long> assemble_ms        {0};
    std::atomic<long long> write_ms           {0};
    // queue_wait_ms: sum of all inter-stage queue stalls per file.
    std::atomic<long long> queue_wait_ms      {0};
    std::atomic<long long> gpu_h2d_us         {0};
    std::atomic<long long> gpu_d2h_us         {0};
    std::atomic<long long> deflate_lz77_us    {0};
    std::atomic<long long> deflate_encode_us  {0};
    std::atomic<long long> deflate_sync_us    {0};
    std::atomic<long long> asm_crc_us         {0};
    std::atomic<long long> asm_d2h_us         {0};
    std::atomic<long long> compressed_bytes   {0};
    std::atomic<long long> raw_bytes          {0};
    std::atomic<long long> lz77_matches       {0};
    std::atomic<long long> lz77_matched_bytes {0};
    std::atomic<long long> total_positions    {0};
};
static GpuBatchAccum g_batch;

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
    uint8_t*                         strip_buf,  // caller-owned pinned buffer, strip_h × row_b
    BoundedQueue<GpuFilteredJob>&    qa,
    std::atomic<bool>&               error,
    ModernPipelineStats&             stats)
{
    using Clock = std::chrono::high_resolution_clock;
    const int pool_n = (int)pool.size();

    std::vector<uint8_t> prev_row(row_b, 0);  // PNG spec: first row's "prior" is zeros

    for (int s = 0; s < total_strips && !error.load(); s++) {
        auto t0 = Clock::now();
        const int actual_rows = src.read_strip(strip_buf, strip_h);
        auto t1 = Clock::now();
        stats.load_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        if (actual_rows <= 0) { error.store(true); break; }

        GpuFilterContext* ctx = pool[s % pool_n];

        GpuTimings gt;
        auto t2 = Clock::now();
        gpu_filter_process_from_host(ctx, strip_buf, prev_row.data(),
                                     actual_rows, &gt, dicom,
                                     /*device_output_only=*/true);
        auto t3 = Clock::now();
        stats.gpu_ms        += std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
        stats.gpu_h2d_us    += (long long)(gt.h2d_ms    * 1000.f);
        stats.gpu_kernel_us += (long long)(gt.kernel_ms * 1000.f);
        stats.gpu_d2h_us    += (long long)(gt.d2h_ms    * 1000.f);

        // Thread the true last-row dependency forward explicitly. Skip on the
        // last strip: prev_row won't be read again so the D2H is wasted work.
        // For single-strip images (most CT slices) this eliminates the only
        // prior-row D2H+sync call entirely.
        if (s < total_strips - 1)
            gpu_filter_copy_prior_row_to_host(ctx, prev_row.data());

        GpuFilteredJob job;
        job.strip_index = s;
        job.actual_rows = actual_rows;
        job.is_last      = (s == total_strips - 1);

        auto t4 = Clock::now();
        const bool pushed = qa.push(job);
        auto t5 = Clock::now();
        stats.filter_push_wait_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count();
        if (!pushed) break;

        const long long depth = (long long)qa.size();
        stats.filter_queue_depth_sum     += depth;
        stats.filter_queue_depth_samples += 1;
        // Track maximum depth (compare-and-swap loop for atomics)
        long long cur_max = stats.filter_queue_max_depth.load(std::memory_order_relaxed);
        while (depth > cur_max &&
               !stats.filter_queue_max_depth.compare_exchange_weak(
                   cur_max, depth, std::memory_order_relaxed)) {}
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
        // --- Queue pop wait: compress starved if filter is slow ---
        auto tq0 = Clock::now();
        const bool got = qa.pop(job);
        auto tq1 = Clock::now();
        stats.compress_pop_wait_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(tq1 - tq0).count();
        if (!got || error.load()) break;

        GpuFilterContext* ctx          = pool[job.strip_index % pool_n];
        const uint8_t*    d_filtered    = gpu_filter_device_output(ctx);
        const size_t      filtered_bytes = gpu_filter_output_size(ctx, job.actual_rows);

        // --- Launch GPU Deflate (ASYNC — no host sync inside) ---
        auto t0 = Clock::now();
        gpu_deflate_compress_strip(gdef, d_filtered, job.actual_rows, row_bytes_out, job.is_last);

        // --- Launch GPU Adler32 CONCURRENTLY with deflate ---
        // Both kernels read d_filtered (independent reads from the same device
        // buffer) and write to their own separate device memory. No dependency
        // between them → they can overlap on the GPU.
        // d_filtered is already complete: the filter stage synchronised its
        // stream before pushing the job, and BoundedQueue provides happens-before.
        auto t_adler_start = Clock::now();
        gpu_adler32_launch(gadler, d_filtered, filtered_bytes);

        // --- Sync deflate: wait for GPU kernels + collect timing ---
        gpu_deflate_stream_sync(gdef);
        auto t1 = Clock::now();
        stats.deflate_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        // --- Flush query: D2H of 8-byte running bit count ---
        // (deflate stream is synced; d_running_total is host-readable)
        auto t_fq0 = Clock::now();
        const size_t avail = gpu_deflate_flushable_byte_length(gdef);
        auto t_fq1 = Clock::now();
        stats.flush_query_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(t_fq1 - t_fq0).count();

        if (avail > flushed_bytes) {
            const size_t slice_len = avail - flushed_bytes;
            const bool   is_first  = (flushed_bytes == 0);

            // --- PNG IDAT chunk assembly: queue D2D copy in passemble->stream ---
            // Async: passemble->stream is independent of gadler->stream, so the
            // adler32 kernel may still be running in gadler->stream in parallel.
            auto t_idat0 = Clock::now();
            const size_t chunk_len = gpu_png_assemble_idat_chunk(
                passemble, gpu_deflate_output(gdef) + flushed_bytes, slice_len,
                is_first, false, 0, zlib_level);
            auto t_idat1 = Clock::now();
            stats.assemble_ms +=
                std::chrono::duration_cast<std::chrono::milliseconds>(t_idat1 - t_idat0).count();

            // --- Collect adler32: sync gadler->stream, combine partials ---
            // By the time we get here, deflate took ~30ms and adler ~5ms, so
            // adler is typically already complete — the collect is near-free.
            const uint32_t strip_adler = gpu_adler32_collect(gadler);
            auto t_adler_done = Clock::now();
            stats.adler_ms +=
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    t_adler_done - t_adler_start).count();

            DeflateResult dr;
            dr.strip_adler = strip_adler;
            dr.input_size  = filtered_bytes;
            accum_adler(dstate, dr);

            // --- D2H: assembled chunk → pinned host buffer (syncs passemble->stream) ---
            ModernWriteJob wj;
            wj.bytes.resize(chunk_len);
            gpu_png_assemble_copy_to_host(passemble, wj.bytes.data(), chunk_len);

            flushed_bytes = avail;

            // --- Queue push wait: compress blocked if writer is slow ---
            auto tw0 = Clock::now();
            const bool pushed = qb.push(std::move(wj));
            auto tw1 = Clock::now();
            stats.write_push_wait_ms +=
                std::chrono::duration_cast<std::chrono::milliseconds>(tw1 - tw0).count();
            if (!pushed) { error.store(true); break; }

            const long long wdepth = (long long)qb.size();
            stats.write_queue_depth_sum     += wdepth;
            stats.write_queue_depth_samples += 1;
            long long wmax = stats.write_queue_max_depth.load(std::memory_order_relaxed);
            while (wdepth > wmax &&
                   !stats.write_queue_max_depth.compare_exchange_weak(
                       wmax, wdepth, std::memory_order_relaxed)) {}
        } else {
            // No data flushed this strip — still must collect adler before next launch.
            const uint32_t strip_adler = gpu_adler32_collect(gadler);
            auto t_adler_done = Clock::now();
            stats.adler_ms +=
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    t_adler_done - t_adler_start).count();

            DeflateResult dr;
            dr.strip_adler = strip_adler;
            dr.input_size  = filtered_bytes;
            accum_adler(dstate, dr);
        }

        stats.strip_count += 1;
    }

    // Final flush: remaining bytes (including the last partial byte, now complete)
    // + the Adler-32 trailer.
    if (!error.load()) {
        const size_t   total_bytes  = gpu_deflate_output_byte_length(gdef);
        const size_t   slice_len    = total_bytes - flushed_bytes;
        const bool     is_first     = (flushed_bytes == 0);
        const uint32_t final_adler  = (uint32_t)dstate.running_adler;

        auto t_idat0 = Clock::now();
        const size_t chunk_len = gpu_png_assemble_idat_chunk(
            passemble, gpu_deflate_output(gdef) + flushed_bytes, slice_len,
            is_first, true, final_adler, zlib_level);
        auto t_idat1 = Clock::now();
        stats.assemble_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(t_idat1 - t_idat0).count();

        ModernWriteJob wj;
        wj.bytes.resize(chunk_len);
        gpu_png_assemble_copy_to_host(passemble, wj.bytes.data(), chunk_len);
        qb.push(std::move(wj));

        final_adler_out = final_adler;
    }

    // Collect deflate and assembly sub-phase stats from GPU contexts
    {
        GpuDeflatePhaseStats dps = gpu_deflate_get_phase_stats(gdef);
        stats.deflate_lz77_us     += dps.lz77_us;
        stats.deflate_bitlen_us   += dps.bitlen_us;
        stats.deflate_scan_us     += dps.scan_us;
        stats.deflate_encode_us   += dps.encode_us;
        stats.deflate_sync_us     += dps.sync_us;
        stats.deflate_stats_d2h_us += dps.stats_d2h_us;

        GpuPngAssembleStats aps = gpu_png_assemble_get_stats(passemble);
        stats.assemble_header_us  += aps.header_h2d_us;
        stats.assemble_d2d_us     += aps.data_d2d_us;
        stats.assemble_presync_us += aps.presync_us;
        stats.assemble_crc_us     += aps.crc_us;
        stats.assemble_crc_h2d_us += aps.crc_h2d_us;
        stats.assemble_d2h_us     += aps.copy_d2h_us;
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
    for (;;) {
        // --- Queue pop wait: writer starved if compress is slow ---
        auto tw0 = Clock::now();
        const bool got = qb.pop(job);
        auto tw1 = Clock::now();
        stats.writer_pop_wait_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(tw1 - tw0).count();
        if (!got) break;

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
    uint8_t*                          strip_buf,
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
    gpu_png_assemble_reset_stats(passemble);

    auto t_abs_start = std::chrono::system_clock::now();
    if (print_header) {
        char ts[16];
        fmt_timestamp(t_abs_start, ts);
        fprintf(stdout,
            "[%s] %s  %ux%u  ch=%d  bps=%d  strip_h=%d  streams=%d  "
            "[modern: GPU deflate%s, overlapped pipeline, UNVERIFIED on real hardware]\n",
            ts, source_label, info.width, info.height,
            info.channels, info.bits_per_sample, strip_h, (int)filter_pool.size(),
            cfg.use_gpu_lz77 ? "+LZ77" : " literal-only");
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
                            filter_pool, strip_buf, qa, error, stats);
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

    const double wall_ms_total = (double)std::chrono::duration_cast<std::chrono::microseconds>(
        t_wall_end - t_wall_start).count() / 1000.0;

    // Accumulate into the global batch stats (thread-safe via atomics).
    {
        LzStats lzs = gpu_deflate_lz_stats(gdef);
        g_batch.files             .fetch_add(1);
        g_batch.pipeline_wall_ms  .fetch_add((long long)wall_ms_total);
        g_batch.load_ms           .fetch_add(stats.load_ms.load());
        g_batch.gpu_ms            .fetch_add(stats.gpu_ms.load());
        g_batch.deflate_ms        .fetch_add(stats.deflate_ms.load());
        g_batch.assemble_ms       .fetch_add(stats.assemble_ms.load());
        g_batch.write_ms          .fetch_add(stats.write_ms.load());
        g_batch.queue_wait_ms     .fetch_add(
            stats.filter_push_wait_ms.load() +
            stats.compress_pop_wait_ms.load() +
            stats.write_push_wait_ms.load() +
            stats.writer_pop_wait_ms.load());
        g_batch.gpu_h2d_us        .fetch_add(stats.gpu_h2d_us.load());
        g_batch.gpu_d2h_us        .fetch_add(stats.gpu_d2h_us.load());
        g_batch.deflate_lz77_us   .fetch_add(stats.deflate_lz77_us.load());
        g_batch.deflate_encode_us .fetch_add(stats.deflate_encode_us.load());
        g_batch.deflate_sync_us   .fetch_add(stats.deflate_sync_us.load());
        g_batch.asm_crc_us        .fetch_add(stats.assemble_crc_us.load());
        g_batch.asm_d2h_us        .fetch_add(stats.assemble_d2h_us.load());
        g_batch.compressed_bytes  .fetch_add((long long)gpu_deflate_output_byte_length(gdef));
        const long long filt_b = (long long)((size_t)(info.width * info.bpp + 1) * (size_t)info.height);
        g_batch.raw_bytes         .fetch_add(filt_b);
        g_batch.lz77_matches      .fetch_add((long long)lzs.matches_found);
        g_batch.lz77_matched_bytes.fetch_add((long long)lzs.matched_bytes);
        g_batch.total_positions   .fetch_add((long long)lzs.total_positions);
    }

    if (print_report) {
        double wall_ms = wall_ms_total;
        double total_pixels = (double)info.width * (double)info.height;
        double total_mb     = total_pixels * info.bpp / (1024.0 * 1024.0);
        double mpps         = (total_pixels / 1e6) / (wall_ms / 1000.0);
        double mbs          = total_mb / (wall_ms / 1000.0);

        char ts_start[16], ts_end[16];
        fmt_timestamp(t_abs_start, ts_start);
        fmt_timestamp(t_abs_end,   ts_end);

        // --- Load all raw counters ---
        const long long load_ms_v        = stats.load_ms.load();
        const long long gpu_ms_v         = stats.gpu_ms.load();
        const long long gpu_h2d_us       = stats.gpu_h2d_us.load();
        const long long gpu_kernel_us    = stats.gpu_kernel_us.load();
        const long long gpu_d2h_us       = stats.gpu_d2h_us.load();
        const long long deflate_ms_v     = stats.deflate_ms.load();
        const long long adler_ms_v       = stats.adler_ms.load();
        const long long assemble_ms_v    = stats.assemble_ms.load();
        const long long write_ms_v       = stats.write_ms.load();
        const int       nstrips          = stats.strip_count.load();

        const long long def_lz77_us      = stats.deflate_lz77_us.load();
        const long long def_bitlen_us    = stats.deflate_bitlen_us.load();
        const long long def_scan_us      = stats.deflate_scan_us.load();
        const long long def_encode_us    = stats.deflate_encode_us.load();
        const long long def_sync_us      = stats.deflate_sync_us.load();
        const long long def_statsd2h_us  = stats.deflate_stats_d2h_us.load();
        const long long flush_query_us_v = stats.flush_query_us.load();

        const long long asm_header_us    = stats.assemble_header_us.load();
        const long long asm_d2d_us       = stats.assemble_d2d_us.load();
        const long long asm_presync_us   = stats.assemble_presync_us.load();
        const long long asm_crc_us       = stats.assemble_crc_us.load();
        const long long asm_crch2d_us    = stats.assemble_crc_h2d_us.load();
        const long long asm_d2h_us       = stats.assemble_d2h_us.load();

        const long long fq_push_ms       = stats.filter_push_wait_ms.load();
        const long long cmp_pop_ms       = stats.compress_pop_wait_ms.load();
        const long long wq_push_ms       = stats.write_push_wait_ms.load();
        const long long wrt_pop_ms       = stats.writer_pop_wait_ms.load();

        const long long fq_depth_n       = stats.filter_queue_depth_samples.load();
        const long long wq_depth_n       = stats.write_queue_depth_samples.load();
        const double avg_fq_depth        = fq_depth_n > 0
            ? (double)stats.filter_queue_depth_sum.load() / (double)fq_depth_n : 0.0;
        const double avg_wq_depth        = wq_depth_n > 0
            ? (double)stats.write_queue_depth_sum.load() / (double)wq_depth_n : 0.0;
        const long long max_fq_depth     = stats.filter_queue_max_depth.load();
        const long long max_wq_depth     = stats.write_queue_max_depth.load();

        // Derived metrics
        const long long gpu_h2d_ms       = gpu_h2d_us   / 1000LL;
        const long long gpu_kernel_ms    = gpu_kernel_us / 1000LL;
        const long long gpu_d2h_ms       = gpu_d2h_us   / 1000LL;

        const size_t total_compressed_bytes = gpu_deflate_output_byte_length(gdef);
        const double filtered_bytes = (double)(row_b + 1) * (double)info.height;
        const double compression_ratio = (filtered_bytes > 0.0)
            ? (double)total_compressed_bytes / filtered_bytes : 0.0;

        const double h2d_mb = total_mb;                             // raw image H2D
        const double d2h_wasted_mb = filtered_bytes / (1024.0*1024.0); // filter D2H (unused)
        const double d2h_chunks_mb = (double)total_compressed_bytes / (1024.0*1024.0);
        const double total_h2d_mb  = h2d_mb;
        const double total_d2h_mb  = d2h_wasted_mb + d2h_chunks_mb;

        const double deflate_throughput_gbs = (deflate_ms_v > 0)
            ? (filtered_bytes / (1024.0*1024.0*1024.0)) / ((double)deflate_ms_v / 1000.0) : 0.0;
        const double h2d_bw_gbs = (gpu_h2d_ms > 0)
            ? (total_mb / 1024.0) / ((double)gpu_h2d_ms / 1000.0) : 0.0;
        const double d2h_bw_gbs = (gpu_d2h_ms > 0)
            ? (d2h_wasted_mb / 1024.0) / ((double)gpu_d2h_ms / 1000.0) : 0.0;

        const double sum_stage_ms = (double)(load_ms_v + gpu_ms_v + deflate_ms_v +
                                             adler_ms_v + assemble_ms_v + write_ms_v);
        const double overlap_pct = (sum_stage_ms > 0.0)
            ? std::max(0.0, (sum_stage_ms - wall_ms) / sum_stage_ms) * 100.0 : 0.0;

        // =====================================================================
        fprintf(stdout, "\n--- Modern Pipeline Timing Report (overlapped, custom GPU deflate) ---\n");
        fprintf(stdout, "Started             : %s\n", ts_start);
        fprintf(stdout, "Finished            : %s\n", ts_end);
        fprintf(stdout, "Wall time           : %.1f ms\n", wall_ms);
        fprintf(stdout, "Throughput          : %.2f MP/s  |  %.1f MB/s (uncompressed)\n", mpps, mbs);
        fprintf(stdout, "GPU Streams         : %d  |  Overlap Efficiency: %.0f%%\n",
                (int)filter_pool.size(), overlap_pct);
        fprintf(stdout, "Strips              : %d  (strip_height = %d)\n", nstrips, strip_h);

        // --- Stage Times ---
        fprintf(stdout, "\nStage Times (concurrent -- sum exceeds wall time):\n");
        fprintf(stdout, "  Loader            : %lld ms\n", load_ms_v);
        fprintf(stdout, "  GPU Filter (wall) : %lld ms\n", gpu_ms_v);
        fprintf(stdout, "  GPU Deflate       : %lld ms  (%.2f GB/s throughput)\n",
                deflate_ms_v, deflate_throughput_gbs);
        fprintf(stdout, "  GPU Adler32       : %lld ms\n", adler_ms_v);
        fprintf(stdout, "  PNG Assemble      : %lld ms\n", assemble_ms_v);
        fprintf(stdout, "  Writer            : %lld ms\n", write_ms_v);

        // --- PNG Assembly Breakdown ---
        fprintf(stdout, "\n=== PNG Assembly Breakdown ===\n");
        fprintf(stdout, "  Chunks Produced   : %lld\n",
                (long long)gpu_png_assemble_get_stats(passemble).chunks);
        fprintf(stdout, "  Header H2D        : %lld us  (len+type 8B + optional zlib header 2B)\n",
                asm_header_us);
        fprintf(stdout, "  Data D2D Copy     : %lld us  (compressed slice → GPU chunk buffer)\n",
                asm_d2d_us);
        fprintf(stdout, "  Pre-CRC Sync      : %lld us  (stream sync stall -- actual D2D time)\n",
                asm_presync_us);
        fprintf(stdout, "  CRC Generation    : %lld us  (CPU crc32() on host after D2H -- Phase2 replaced GPU CRC)\n",
                asm_crc_us);
        fprintf(stdout, "  CRC Finalize      : %lld us  (obsolete GPU CRC H2D -- now 0; CPU CRC writes directly to host buf)\n",
                asm_crch2d_us);
        fprintf(stdout, "  Chunk D2H         : %lld us  (%.2f MB assembled chunks → host)\n",
                asm_d2h_us, d2h_chunks_mb);

        // --- Transfer Breakdown ---
        fprintf(stdout, "\n=== Transfer Breakdown ===\n");
        fprintf(stdout, "  Filter H2D        : %lld ms  (%.2f MB raw image → GPU, %.2f GB/s)\n",
                gpu_h2d_ms, h2d_mb, h2d_bw_gbs);
        fprintf(stdout, "  Filter D2H [!]    : %lld ms  (%.2f MB filtered output → host"
                        " -- NOT USED in modern path, WASTED bandwidth, %.2f GB/s)\n",
                gpu_d2h_ms, d2h_wasted_mb, d2h_bw_gbs);
        fprintf(stdout, "  LZ77 Stats D2H    : %lld us  (28 B x %d strips = %d B)\n",
                def_statsd2h_us, nstrips, nstrips * 28);
        fprintf(stdout, "  Flush Query D2H   : %lld us  (8 B x %d strips = %d B -- running bit count)\n",
                flush_query_us_v, nstrips, nstrips * 8);
        fprintf(stdout, "  Assembly D2H      : %lld us  (see Chunk D2H above)\n", asm_d2h_us);
        fprintf(stdout, "  Total H2D         : %.2f MB\n", total_h2d_mb);
        fprintf(stdout, "  Total D2H         : %.2f MB  (%.2f MB wasted filter + %.2f MB chunks)\n",
                total_d2h_mb, d2h_wasted_mb, d2h_chunks_mb);

        // --- Queue Stall Analysis ---
        fprintf(stdout, "\n=== Queue Stall Analysis ===\n");
        fprintf(stdout, "  Filter→Compress queue (capacity %zu):\n", qa_capacity);
        fprintf(stdout, "    Avg depth         : %.1f  |  Max depth: %lld\n",
                avg_fq_depth, max_fq_depth);
        fprintf(stdout, "    Filter push wait  : %lld ms"
                        "  (filter blocked -- compress can't keep up)\n", fq_push_ms);
        fprintf(stdout, "    Compress pop wait : %lld ms"
                        "  (compress starved -- filter too slow)\n", cmp_pop_ms);
        fprintf(stdout, "  Compress→Writer queue (capacity %zu):\n", qb_capacity);
        fprintf(stdout, "    Avg depth         : %.1f  |  Max depth: %lld\n",
                avg_wq_depth, max_wq_depth);
        fprintf(stdout, "    Write push wait   : %lld ms"
                        "  (compress blocked -- writer can't keep up)\n", wq_push_ms);
        fprintf(stdout, "    Writer pop wait   : %lld ms"
                        "  (writer starved -- compress too slow)\n", wrt_pop_ms);

        // --- Deflate Kernel Breakdown ---
        fprintf(stdout, "\n=== GPU Deflate Kernel Breakdown ===\n");
        if (cfg.use_gpu_lz77) {
            fprintf(stdout, "  LZ77 Match Finder : %lld us  (1 active thread/block × %d blocks,"
                            " %d threads launched, util=3.1%%)\n",
                    def_lz77_us, nstrips > 0 ? strip_h : 0, nstrips > 0 ? strip_h * 32 : 0);
        } else {
            fprintf(stdout, "  LZ77 Match Finder : 0 us  (literal-only path)\n");
        }
        fprintf(stdout, "  Bit Length (A/A') : %lld us  (per-row Huffman bit count)\n", def_bitlen_us);
        fprintf(stdout, "  Prefix Scan (B)   : %lld us  (3-pass multi-block scan)\n", def_scan_us);
        fprintf(stdout, "  Encode (C+D+E)    : %lld us  (block header + Huffman emit + EOB)\n", def_encode_us);
        fprintf(stdout, "  Stream Sync       : %lld us  (CPU waiting for GPU per strip)\n", def_sync_us);
        fprintf(stdout, "  LZ77 Stats D2H    : %lld us  (28 B/strip blocking readback)\n", def_statsd2h_us);

        // --- GPU Kernel Utilization (estimated) ---
        fprintf(stdout, "\n=== GPU Kernel Utilization (estimated, not hardware-verified) ===\n");
        if (cfg.use_gpu_lz77) {
            fprintf(stdout, "  LZ77 kernel       : 1/32 threads/block active = 3.1%% thread util\n");
            fprintf(stdout, "                      Shared mem: 4 KB/block (1024×uint32 hash table, Phase2 reduced from 16 KB)\n");
            fprintf(stdout, "                      4 KB/block allows ~16+ blocks/SM on Blackwell (was 6 at 16 KB)\n");
        }
        fprintf(stdout, "  Filter kernel     : 256 threads/block, 5120 B shared per block\n");
        fprintf(stdout, "  Deflate encode    : 256 threads/block, many atomicOr per thread\n");
        fprintf(stdout, "  (Use Nsight Systems/Compute for true SM occupancy and warp efficiency)\n");

        // --- Compression Statistics ---
        LzStats lzs = gpu_deflate_lz_stats(gdef);
        const uint64_t raw_bytes_v   = (uint64_t)(filtered_bytes);
        const uint64_t literal_bytes = (lzs.total_positions > 0)
            ? (lzs.total_positions - lzs.matched_bytes) : raw_bytes_v;

        fprintf(stdout, "\n=== Compression Statistics ===\n");
        fprintf(stdout, "  Raw Bytes         : %.2f MB  (filtered output)\n",
                filtered_bytes / (1024.0*1024.0));
        fprintf(stdout, "  Compressed Bytes  : %.2f MB\n",
                (double)total_compressed_bytes / (1024.0*1024.0));
        fprintf(stdout, "  Compression Ratio : %.3f  (compressed / raw)\n", compression_ratio);
        if (cfg.use_gpu_lz77 && lzs.total_positions > 0) {
            fprintf(stdout, "  Literal Bytes     : %llu  (%.1f%% of input)\n",
                    (unsigned long long)literal_bytes,
                    100.0 * (double)literal_bytes / (double)lzs.total_positions);
            fprintf(stdout, "  Match Count       : %llu\n",
                    (unsigned long long)lzs.matches_found);
            fprintf(stdout, "  Matched Bytes     : %llu  (%.1f%% coverage)\n",
                    (unsigned long long)lzs.matched_bytes, lzs.coverage * 100.0);
            fprintf(stdout, "  Avg Match Length  : %.1f bytes\n", lzs.avg_match_len);
            fprintf(stdout, "  Max Match Length  : %u bytes\n",   lzs.max_match_len);
            fprintf(stdout, "  Avg Match Distance: %.0f bytes\n", lzs.avg_match_dist);
        }
        fprintf(stdout, "  LZ77 Config       : hash=%d buckets, 1 slot/bucket, "
                        "min=%d max=%d bytes\n", 1024, 3, 258);
        fprintf(stdout, "  Final Adler-32    : 0x%08x\n", final_adler);

        // =====================================================================
        // PERFORMANCE SUMMARY with bottleneck identification
        // =====================================================================
        // Identify the dominant bottleneck. We compare stage times that are
        // ON THE CRITICAL PATH of the pipeline (not overlapped work).
        // The filter D2H is a special case: it's wasted but adds to GPU stage time.
        const long long gpu_filter_total_ms = gpu_ms_v;
        const long long gpu_lz77_ms_v       = def_lz77_us / 1000LL;
        const long long gpu_deflate_other_ms = (deflate_ms_v > gpu_lz77_ms_v / 1000LL)
                                               ? deflate_ms_v : deflate_ms_v;
        const long long png_assemble_total_ms = assemble_ms_v
                                               + (long long)(asm_d2h_us / 1000LL);
        const long long writer_total_ms       = write_ms_v;

        struct { const char* name; long long ms; } stages[] = {
            { "GPU Filter",   gpu_filter_total_ms     },
            { "GPU Deflate",  deflate_ms_v            },
            { "GPU LZ77",     gpu_lz77_ms_v           },
            { "PNG Assembly", png_assemble_total_ms   },
            { "Writer",       writer_total_ms         },
            { "Loader",       load_ms_v               },
        };
        int bottleneck_idx = 0;
        for (int i = 1; i < 6; i++)
            if (stages[i].ms > stages[bottleneck_idx].ms) bottleneck_idx = i;

        fprintf(stdout, "\n");
        for (int i = 0; i < 60; i++) fputc('=', stdout); fputc('\n', stdout);
        fprintf(stdout, "PERFORMANCE SUMMARY\n");
        for (int i = 0; i < 60; i++) fputc('=', stdout); fputc('\n', stdout);
        fprintf(stdout, "Files Processed     : 1\n");
        fprintf(stdout, "Total Runtime       : %.3f s  (%.1f ms)\n",
                wall_ms / 1000.0, wall_ms);
        fprintf(stdout, "\n");
        fprintf(stdout, "GPU Filter Time     : %lld ms\n", gpu_filter_total_ms);
        fprintf(stdout, "GPU LZ77 Time       : %lld ms  (GPU event)\n", gpu_lz77_ms_v);
        fprintf(stdout, "GPU Deflate Time    : %lld ms  (wall, includes sync)\n", deflate_ms_v);
        fprintf(stdout, "PNG Assembly Time   : %lld ms  (assemble + D2H)\n", png_assemble_total_ms);
        fprintf(stdout, "Writer Time         : %lld ms\n", writer_total_ms);
        fprintf(stdout, "\n");
        fprintf(stdout, "Total H2D           : %.2f MB\n", total_h2d_mb);
        fprintf(stdout, "Total D2H           : %.2f MB  (%.2f MB wasted filter)\n",
                total_d2h_mb, d2h_wasted_mb);
        fprintf(stdout, "\n");
        fprintf(stdout, "Largest Bottleneck  : %s (%lld ms)\n",
                stages[bottleneck_idx].name, stages[bottleneck_idx].ms);

        // Specific diagnosis
        if (gpu_d2h_ms > (long long)(wall_ms * 0.15)) {
            fprintf(stdout, "  [!] WASTED FILTER D2H: %lld ms (%.0f%% of wall time).\n",
                    gpu_d2h_ms, 100.0 * gpu_d2h_ms / wall_ms);
            fprintf(stdout, "      The modern path never reads h_output from gpu_filter.\n");
            fprintf(stdout, "      Eliminating this transfer will directly cut GPU stage time.\n");
        }
        if (cfg.use_gpu_lz77 && gpu_lz77_ms_v > (long long)(deflate_ms_v * 0.5)) {
            fprintf(stdout, "  [!] LZ77 DOMINATES DEFLATE: %lld ms / %lld ms (%.0f%%).\n",
                    gpu_lz77_ms_v, deflate_ms_v,
                    (deflate_ms_v > 0) ? 100.0 * gpu_lz77_ms_v / deflate_ms_v : 0.0);
            fprintf(stdout, "      LZ77 uses 1 thread per row block (3.1%% utilization).\n");
            fprintf(stdout, "      Consider multi-bucket hash or wider match search.\n");
        }
        if (asm_crc_us > asm_d2d_us && asm_crc_us > asm_presync_us) {
            fprintf(stdout, "  [!] CRC GENERATION DOMINANT IN ASSEMBLY: %lld us.\n", asm_crc_us);
            fprintf(stdout, "      CRC kernel runs 1 thread/block serially over 64KB chunks.\n");
        }
        if (cmp_pop_ms > fq_push_ms && cmp_pop_ms > (long long)(wall_ms * 0.1)) {
            fprintf(stdout, "  [!] COMPRESS STAGE STARVED: %lld ms waiting for filter.\n",
                    cmp_pop_ms);
            fprintf(stdout, "      Filter is the pipeline bottleneck.\n");
        } else if (fq_push_ms > cmp_pop_ms && fq_push_ms > (long long)(wall_ms * 0.1)) {
            fprintf(stdout, "  [!] FILTER STAGE BLOCKED: %lld ms waiting for compress.\n",
                    fq_push_ms);
            fprintf(stdout, "      Compress is the pipeline bottleneck.\n");
        }

        fprintf(stdout, "\n(3 pipeline stages run concurrently -- wall < sum of stage times)\n");
    }

    return true;
}

void pipeline_reset_gpu_batch_stats()
{
    g_batch.files             .store(0);
    g_batch.pipeline_wall_ms  .store(0);
    g_batch.file_total_ms     .store(0);
    g_batch.dicom_decode_ms   .store(0);
    g_batch.load_ms           .store(0);
    g_batch.gpu_ms            .store(0);
    g_batch.deflate_ms        .store(0);
    g_batch.assemble_ms       .store(0);
    g_batch.write_ms          .store(0);
    g_batch.queue_wait_ms     .store(0);
    g_batch.gpu_h2d_us        .store(0);
    g_batch.gpu_d2h_us        .store(0);
    g_batch.deflate_lz77_us   .store(0);
    g_batch.deflate_encode_us .store(0);
    g_batch.deflate_sync_us   .store(0);
    g_batch.asm_crc_us        .store(0);
    g_batch.asm_d2h_us        .store(0);
    g_batch.compressed_bytes  .store(0);
    g_batch.raw_bytes         .store(0);
    g_batch.lz77_matches      .store(0);
    g_batch.lz77_matched_bytes.store(0);
    g_batch.total_positions   .store(0);
}

void pipeline_record_file_times(long long dicom_ms, long long total_ms)
{
    g_batch.dicom_decode_ms.fetch_add(dicom_ms);
    g_batch.file_total_ms  .fetch_add(total_ms);
}

void pipeline_print_gpu_batch_summary(int total_files, int succeeded,
                                      double total_wall_s,
                                      long long sum_file_batch_ms,
                                      int num_workers)
{
    const int       files          = g_batch.files.load();
    const long long pipeline_sum   = g_batch.pipeline_wall_ms.load();
    const long long file_total_sum = g_batch.file_total_ms.load();
    const long long dicom_sum      = g_batch.dicom_decode_ms.load();
    const long long load_ms        = g_batch.load_ms.load();
    const long long gpu_ms         = g_batch.gpu_ms.load();
    const long long deflate_ms     = g_batch.deflate_ms.load();
    const long long assemble_ms    = g_batch.assemble_ms.load();
    const long long write_ms_v     = g_batch.write_ms.load();
    const long long queue_wait_sum = g_batch.queue_wait_ms.load();
    const long long h2d_us         = g_batch.gpu_h2d_us.load();
    const long long d2h_wasted_us  = g_batch.gpu_d2h_us.load();
    const long long lz77_us        = g_batch.deflate_lz77_us.load();
    const long long encode_us      = g_batch.deflate_encode_us.load();
    const long long sync_us        = g_batch.deflate_sync_us.load();
    const long long crc_us         = g_batch.asm_crc_us.load();
    const long long asm_d2h_us_v   = g_batch.asm_d2h_us.load();
    const long long comp_bytes     = g_batch.compressed_bytes.load();
    const long long raw_bytes      = g_batch.raw_bytes.load();
    const long long lz_matches     = g_batch.lz77_matches.load();
    const long long lz_matched_b   = g_batch.lz77_matched_bytes.load();
    const long long lz_total_pos   = g_batch.total_positions.load();

    const double total_wall_ms     = total_wall_s * 1000.0;
    const double fps               = (total_wall_s > 0) ? (double)total_files / total_wall_s : 0.0;

    // True per-file averages — what each file actually took end-to-end.
    // file_total_ms covers encode_*_to_png() entry → return (DICOM decode + pipeline).
    // pipeline_wall_ms is run_one_modern() only (excludes DICOM decode).
    const double avg_file_total_ms = (files > 0) ? (double)file_total_sum / files : 0.0;
    const double avg_dicom_ms      = (files > 0) ? (double)dicom_sum      / files : 0.0;
    const double avg_pipeline_ms   = (files > 0) ? (double)pipeline_sum   / files : 0.0;
    const double avg_queue_wait_ms = (files > 0) ? (double)queue_wait_sum / files : 0.0;

    // Stage averages (concurrent, do not sum to wall time)
    const double avg_load_ms     = (files > 0) ? (double)load_ms    / files : 0.0;
    const double avg_gpu_ms      = (files > 0) ? (double)gpu_ms     / files : 0.0;
    const double avg_deflate_ms  = (files > 0) ? (double)deflate_ms / files : 0.0;
    const double avg_assemble_ms = (files > 0) ? (double)assemble_ms/ files : 0.0;
    const double avg_write_ms    = (files > 0) ? (double)write_ms_v / files : 0.0;

    const double avg_h2d_ms      = (files > 0) ? (double)(h2d_us / 1000LL)        / files : 0.0;
    const double avg_d2h_wst_ms  = (files > 0) ? (double)(d2h_wasted_us / 1000LL) / files : 0.0;
    const double avg_lz77_ms     = (files > 0) ? (double)(lz77_us / 1000LL)       / files : 0.0;
    const double avg_encode_ms   = (files > 0) ? (double)(encode_us / 1000LL)     / files : 0.0;
    const double avg_sync_ms     = (files > 0) ? (double)(sync_us / 1000LL)       / files : 0.0;
    const double avg_crc_ms      = (files > 0) ? (double)(crc_us / 1000LL)        / files : 0.0;
    const double avg_asmd2h_ms   = (files > 0) ? (double)(asm_d2h_us_v / 1000LL)  / files : 0.0;

    const double total_raw_gb    = (double)raw_bytes  / (1024.0*1024.0*1024.0);
    const double total_comp_gb   = (double)comp_bytes / (1024.0*1024.0*1024.0);
    const double overall_ratio   = (raw_bytes > 0) ? (double)comp_bytes / (double)raw_bytes : 0.0;

    const double lz77_coverage   = (lz_total_pos > 0)
        ? 100.0 * (double)lz_matched_b / (double)lz_total_pos : 0.0;
    const double avg_match_len   = (lz_matches > 0)
        ? (double)lz_matched_b / (double)lz_matches : 0.0;

    for (int i = 0; i < 60; i++) fputc('=', stdout); fputc('\n', stdout);
    fprintf(stdout, "GPU BATCH PERFORMANCE SUMMARY\n");
    for (int i = 0; i < 60; i++) fputc('=', stdout); fputc('\n', stdout);
    fprintf(stdout, "Files               : %d total  |  %d OK  |  %d failed\n",
            total_files, succeeded, total_files - succeeded);
    fprintf(stdout, "GPU images tracked  : %d\n", files);
    fprintf(stdout, "Batch wall time     : %.3f s\n", total_wall_s);
    fprintf(stdout, "Throughput          : %.2f files/s\n", fps);
    fprintf(stdout, "Avg pipeline time   : %.1f ms  (run_one_modern only; excludes DICOM decode)\n",
            avg_pipeline_ms);
    fprintf(stdout, "Avg file total      : %.1f ms  (DICOM decode + pipeline)\n",
            avg_file_total_ms);
    fprintf(stdout, "\nStage Averages (per file, concurrent -- do not sum to wall time):\n");
    fprintf(stdout, "  Loader            : %.1f ms\n",  avg_load_ms);
    fprintf(stdout, "  GPU Filter        : %.1f ms\n",  avg_gpu_ms);
    fprintf(stdout, "  GPU Deflate       : %.1f ms\n",  avg_deflate_ms);
    fprintf(stdout, "  PNG Assembly      : %.1f ms\n",  avg_assemble_ms);
    fprintf(stdout, "  Writer            : %.1f ms\n",  avg_write_ms);
    fprintf(stdout, "\nTransfer Averages (per file):\n");
    fprintf(stdout, "  H2D (raw image)   : %.1f ms\n",  avg_h2d_ms);
    fprintf(stdout, "  D2H (filter) [!]  : %.1f ms  (WASTED -- unused in modern path)\n",
            avg_d2h_wst_ms);
    fprintf(stdout, "  Assembly D2H      : %.1f ms  (PNG chunks → host)\n", avg_asmd2h_ms);
    fprintf(stdout, "\nDeflate Sub-phase Averages (per file):\n");
    fprintf(stdout, "  LZ77 kernel       : %.1f ms\n",  avg_lz77_ms);
    fprintf(stdout, "  Encode kernels    : %.1f ms\n",  avg_encode_ms);
    fprintf(stdout, "  Stream sync stall : %.1f ms\n",  avg_sync_ms);
    fprintf(stdout, "  Assembly CRC      : %.1f ms\n",  avg_crc_ms);
    fprintf(stdout, "\nCompression Totals:\n");
    fprintf(stdout, "  Raw (filtered)    : %.3f GB\n",  total_raw_gb);
    fprintf(stdout, "  Compressed        : %.3f GB\n",  total_comp_gb);
    fprintf(stdout, "  Overall ratio     : %.3f\n",     overall_ratio);
    if (lz_total_pos > 0) {
        fprintf(stdout, "  LZ77 coverage     : %.1f%%  (%lld matches, avg len %.1f)\n",
                lz77_coverage, (long long)lz_matches, avg_match_len);
    }
    fprintf(stdout, "\nKey Bottlenecks (sum across all files):\n");
    fprintf(stdout, "  Wasted filter D2H : %.1f ms total (%.1f ms/file)\n",
            (double)(d2h_wasted_us / 1000LL), avg_d2h_wst_ms);
    fprintf(stdout, "  LZ77 (3.1%% util)  : %.1f ms total (%.1f ms/file)\n",
            (double)(lz77_us / 1000LL), avg_lz77_ms);
    fprintf(stdout, "  CRC generation    : %.1f ms total (%.1f ms/file)\n",
            (double)(crc_us / 1000LL), avg_crc_ms);
    fprintf(stdout, "  Stream sync       : %.1f ms total (%.1f ms/file)\n",
            (double)(sync_us / 1000LL), avg_sync_ms);

    // =========================================================================
    // TIMING CONSISTENCY REPORT
    // Verifies that per-file timings reconcile with the observed batch wall time.
    // All numbers below use sum_file_batch_ms (measured by batch_processor.cpp
    // around each process_one_file() call) as the ground truth file total, which
    // is independent of the GPU batch accumulator.
    // =========================================================================
    {
        const double sum_file_ms     = (double)sum_file_batch_ms;
        const double sum_internal_ms = (double)file_total_sum;    // encode_*_to_png scope
        const double expected_par_ms = (num_workers > 0)
            ? sum_file_ms / (double)num_workers : 0.0;
        const double timing_error_ms = total_wall_ms - expected_par_ms;
        const double timing_error_pct= (expected_par_ms > 0)
            ? 100.0 * timing_error_ms / expected_par_ms : 0.0;
        const double worker_util     = (num_workers > 0 && total_wall_ms > 0)
            ? 100.0 * sum_file_ms / ((double)num_workers * total_wall_ms) : 0.0;

        // Scope gap: difference between batch_processor's view of a file and
        // the GPU pipeline's view.  Represents GPU context alloc, batch overhead,
        // and any unmeasured phases.
        const double scope_gap_ms    = (files > 0)
            ? (sum_file_ms - sum_internal_ms) / files : 0.0;
        const double avg_batch_ms    = (total_files > 0)
            ? sum_file_ms / (double)total_files : 0.0;

        fputc('\n', stdout);
        for (int i = 0; i < 50; i++) fputc('=', stdout); fputc('\n', stdout);
        fprintf(stdout, "TIMING CONSISTENCY REPORT\n");
        for (int i = 0; i < 50; i++) fputc('=', stdout); fputc('\n', stdout);
        fprintf(stdout, "Batch Wall Time        : %.0f ms\n", total_wall_ms);
        fprintf(stdout, "Workers                : %d\n", num_workers);
        fprintf(stdout, "Files Processed        : %d\n", total_files);
        fputc('\n', stdout);
        fprintf(stdout, "Avg File Total (batch) : %.0f ms  [batch_processor scope]\n",
                avg_batch_ms);
        fprintf(stdout, "Avg DICOM Decode       : %.0f ms  [open + load_frame]\n",
                avg_dicom_ms);
        fprintf(stdout, "Avg Pipeline           : %.0f ms  [run_one_modern scope]\n",
                avg_pipeline_ms);
        fprintf(stdout, "Avg Queue Wait         : %.0f ms  [inter-stage stalls inside pipeline]\n",
                avg_queue_wait_ms);
        fprintf(stdout, "Scope Gap (per file)   : %.0f ms  [batch overhead outside encode_*_to_png]\n",
                scope_gap_ms);
        fputc('\n', stdout);
        fprintf(stdout, "Sum File Total         : %.0f ms\n", sum_file_ms);
        fprintf(stdout, "Expected Parallel Time : %.0f ms  (%.0f ms / %d workers)\n",
                expected_par_ms, sum_file_ms, num_workers);
        fprintf(stdout, "Timing Error           : %+.0f ms  (%+.1f%%)  "
                "[batch_wall - expected; load imbalance + scheduling]\n",
                timing_error_ms, timing_error_pct);
        fprintf(stdout, "Worker Utilization     : %.1f%%  (%.0f / (%d × %.0f))\n",
                worker_util, sum_file_ms, num_workers, total_wall_ms);

        // Validation verdict
        const double abs_err_pct = (timing_error_pct < 0)
            ? -timing_error_pct : timing_error_pct;
        fputc('\n', stdout);
        if (abs_err_pct <= 10.0)
            fprintf(stdout, "Validation: PASS  (timing error %.1f%% <= 10%% threshold)\n",
                    abs_err_pct);
        else if (abs_err_pct <= 25.0)
            fprintf(stdout, "Validation: WARN  (timing error %.1f%% > 10%%; "
                    "acceptable if load is uneven)\n", abs_err_pct);
        else
            fprintf(stdout, "Validation: FAIL  (timing error %.1f%% > 25%%; "
                    "likely a measurement scope bug)\n", abs_err_pct);
    }
    for (int i = 0; i < 60; i++) fputc('=', stdout); fputc('\n', stdout);
}

#else
// Stubs so batch_processor.cpp links regardless of GPU_PNG_MODERN_DEFLATE.
void pipeline_reset_gpu_batch_stats() {}
void pipeline_record_file_times(long long, long long) {}
void pipeline_print_gpu_batch_summary(int, int, double, long long, int) {}
#endif  // GPU_PNG_MODERN_DEFLATE

// ---------------------------------------------------------------------------
// Per-worker persistent GPU context cache (Phase 2).
//
// The modern GPU pipeline creates 4 types of GPU contexts per image, each
// with multiple cudaMalloc/cudaMallocHost/cudaStreamCreate calls. For a
// 1424-file batch with pool_n=3 streams, this is thousands of slow driver
// calls. This cache keeps contexts alive between images on the same worker
// thread: on dimension match, contexts are simply reset (fast); only on a
// dimension change are they destroyed and recreated.
//
// Thread-local lifetime: created lazily on first use, destroyed when the
// worker thread exits (which is safe — CUDA context is still alive when
// ThreadPool joins its threads before the primary context is released).
// ---------------------------------------------------------------------------
#if defined(GPU_PNG_MODERN_DEFLATE)
struct ModernContextCache {
    int    width = 0, bpp = 0, strip_height = 0, pool_n = 0;
    bool   use_lz77   = false;
    size_t total_filt = 0, max_chunk = 0, strip_bytes = 0;

    std::vector<GpuFilterContext*> filter_pool;
    GpuDeflateContext*     gdef      = nullptr;
    GpuAdler32Context*     gadler    = nullptr;
    GpuPngAssembleContext* passemble = nullptr;
    uint8_t*               strip_buf = nullptr;

    bool matches(int w, int b, int sh, int pn, bool lz77,
                 size_t tf, size_t mc, size_t sb) const {
        return gdef != nullptr
            && width == w && bpp == b && strip_height == sh && pool_n == pn
            && use_lz77 == lz77 && total_filt == tf && max_chunk == mc
            && strip_bytes == sb;
    }

    void destroy_all() {
        gpu_filter_free_pinned(strip_buf); strip_buf = nullptr;
        for (auto* c : filter_pool) gpu_filter_destroy(c);
        filter_pool.clear();
        gpu_deflate_destroy(gdef);           gdef      = nullptr;
        gpu_adler32_destroy(gadler);         gadler    = nullptr;
        gpu_png_assemble_destroy(passemble); passemble = nullptr;
        width = bpp = strip_height = pool_n = 0;
        total_filt = max_chunk = strip_bytes = 0;
    }

    ~ModernContextCache() { destroy_all(); }
};

static thread_local ModernContextCache tl_modern_cache;
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
            // Stream pool size (Requirement 2): explicit --gpu-streams wins;
            // otherwise auto-default 8 for the modern path. Clamped to >= 3
            // so the filter->compress queue capacity (N-2) is always >= 1 --
            // see modern_filter_stage's file-level buffer-reuse-safety comment.
            int pool_size = (cfg.gpu_streams > 0) ? cfg.gpu_streams : 8;
            pool_size = std::max(pool_size, 3);

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
            const size_t strip_bytes = (size_t)cfg.strip_height * (size_t)info.width * info.bpp;

            // Reuse per-worker GPU contexts across images (Phase 2).
            // Contexts persist in the thread-local cache until dimensions change
            // or the worker thread exits; run_one_modern resets state at its start.
            ModernContextCache& cc = tl_modern_cache;
            if (!cc.matches(info.width, info.bpp, cfg.strip_height, pool_size,
                            cfg.use_gpu_lz77, total_filtered, max_chunk_data_bytes,
                            strip_bytes)) {
                cc.destroy_all();

                cc.strip_buf = gpu_filter_alloc_pinned(strip_bytes);
                if (!cc.strip_buf) {
                    fprintf(stderr, "gpu_filter_alloc_pinned failed\n");
                    return false;
                }

                cc.filter_pool.reserve(pool_size);
                for (int i = 0; i < pool_size; i++) {
                    GpuFilterContext* ctx = gpu_filter_create(
                        info.width, info.bpp, cfg.strip_height,
                        /*device_output_only=*/true);
                    if (!ctx) {
                        fprintf(stderr, "gpu_filter_create failed (pool slot %d)\n", i);
                        cc.destroy_all();
                        return false;
                    }
                    cc.filter_pool.push_back(ctx);
                }
                cc.gdef = gpu_deflate_create(
                    cfg.strip_height, (int)row_bytes, max_total_bits, cfg.use_gpu_lz77);
                cc.gadler    = gpu_adler32_create(total_filtered);
                cc.passemble = gpu_png_assemble_create(max_chunk_data_bytes);

                cc.width = info.width; cc.bpp = info.bpp;
                cc.strip_height = cfg.strip_height; cc.pool_n = pool_size;
                cc.use_lz77 = cfg.use_gpu_lz77;
                cc.total_filt = total_filtered;
                cc.max_chunk  = max_chunk_data_bytes;
                cc.strip_bytes = strip_bytes;
            }

            return run_one_modern(cfg, output_path, source_label, src,
                                  cc.filter_pool, cc.gdef, cc.gadler, cc.passemble,
                                  cc.strip_buf, cfg.verbose, cfg.verbose);
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
    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::milliseconds;
    auto t_file_start = Clock::now();

    TiffSource src;
    if (!src.open(input_path)) {
        fprintf(stderr, "Cannot open TIFF: %s\n", input_path);
        return false;
    }
    bool ok = run_pipeline(cfg, output_path, "TIFF", src);

    auto t_file_end = Clock::now();
#if defined(GPU_PNG_MODERN_DEFLATE)
    if (cfg.use_gpu_deflate) {
        long long total_ms = std::chrono::duration_cast<Ms>(t_file_end - t_file_start).count();
        pipeline_record_file_times(0LL, total_ms);
    }
#endif
    return ok;
}

bool encode_raw_to_png(const char* input_path,
                       const char* output_path,
                       const PipelineConfig& cfg)
{
    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::milliseconds;
    auto t_file_start = Clock::now();

    RawSource src;
    if (!src.open(input_path)) {
        fprintf(stderr, "Cannot open RAW: %s\n", input_path);
        return false;
    }
    bool ok = run_pipeline(cfg, output_path, "RAW", src);

    auto t_file_end = Clock::now();
#if defined(GPU_PNG_MODERN_DEFLATE)
    if (cfg.use_gpu_deflate) {
        long long total_ms = std::chrono::duration_cast<Ms>(t_file_end - t_file_start).count();
        pipeline_record_file_times(0LL, total_ms);
    }
#endif
    return ok;
}

bool encode_dicom_to_png(const char* input_path,
                         const char* output_path,
                         const PipelineConfig& cfg)
{
    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::milliseconds;
    auto t_file_start = Clock::now();

    DicomSource src;
    auto t_decode_start = Clock::now();
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
    auto t_decode_end = Clock::now();

    bool ok = run_pipeline(cfg, output_path, "DICOM", src);

    auto t_file_end = Clock::now();
#if defined(GPU_PNG_MODERN_DEFLATE)
    if (cfg.use_gpu_deflate) {
        long long dicom_ms = std::chrono::duration_cast<Ms>(t_decode_end - t_decode_start).count();
        long long total_ms = std::chrono::duration_cast<Ms>(t_file_end   - t_file_start  ).count();
        pipeline_record_file_times(dicom_ms, total_ms);
    }
#endif
    return ok;
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

// ===========================================================================
// BMP export path (Phase D1)
//
// CPU-only: no GPU, no compression. Pipeline is:
//   open source → stream strips → per-row pixel transform → write BMP
//
// This is the fastest export path. Throughput is limited by disk I/O and
// DICOM decode, not compute. All pixel transforms are done inline.
// ===========================================================================
#include "bmp_writer.h"

// ---------------------------------------------------------------------------
// 16-bit LE DICOM → 8-bit grayscale using DicomPixelParams window/level.
// norm_scale / norm_offset are used when apply_window=0 (full bit-depth range).
// ---------------------------------------------------------------------------
static void dicom16_row_to_gray8(const uint8_t* src_le, uint8_t* dst, int width,
                                  const DicomPixelParams& p,
                                  float norm_scale, float norm_offset)
{
    const int shift = p.high_bit - p.bits_stored + 1;
    const int mask  = (p.bits_stored < 32) ? ((1 << p.bits_stored) - 1) : -1;

    for (int x = 0; x < width; x++) {
        uint16_t raw_u = (uint16_t)((int)src_le[x * 2] | ((int)src_le[x * 2 + 1] << 8));
        raw_u = (uint16_t)((raw_u >> shift) & mask);

        float val;
        if (p.pixel_rep == 1 && p.bits_stored < 16) {
            // Sign-extend to int32 for narrow signed types
            int sign_bit = 1 << (p.bits_stored - 1);
            int32_t s = (int32_t)(uint32_t)raw_u;
            if (s & sign_bit) s |= ~((int32_t)mask);
            val = (float)s;
        } else if (p.pixel_rep == 1) {
            val = (float)(int16_t)raw_u;
        } else {
            val = (float)raw_u;
        }

        if (p.apply_rescale)
            val = val * p.rescale_slope + p.rescale_intercept;

        float norm;
        if (p.apply_window) {
            float lo = p.window_center - p.window_width * 0.5f;
            float hi = p.window_center + p.window_width * 0.5f;
            norm = (hi > lo) ? (val - lo) / (hi - lo) : 0.f;
        } else {
            norm = val * norm_scale + norm_offset;
        }

        int v = (int)(norm * 255.f + 0.5f);
        dst[x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
}

// ---------------------------------------------------------------------------
// Unified any-source → BMP.
//
// DICOM (dicom_params() != nullptr): 16-bit LE with window/level → 8-bit gray.
// TIFF/RAW (dicom_params() == nullptr): 16-bit BE big-endian → take MSB.
// 8-bit inputs: pass through (bmp_write_row handles RGB→BGR for colour).
// ---------------------------------------------------------------------------
static bool run_bmp_from_source(const char* output_path, ImageSource& src)
{
    const ImageInfo&        info  = src.info();
    const DicomPixelParams* p_ptr = src.dicom_params();  // null for TIFF/RAW
    const int out_channels = (info.channels == 1) ? 1 : 3;

    // Precompute full-range normalisation constants for DICOM 16-bit without window.
    float norm_scale = 1.f, norm_offset = 0.f;
    if (p_ptr && info.bits_per_sample == 16 && !p_ptr->apply_window) {
        if (p_ptr->pixel_rep == 1) {
            float half  = (float)(1u << (p_ptr->bits_stored - 1));
            float range = (float)(1u << p_ptr->bits_stored);
            norm_scale  = 1.f / range;
            norm_offset = half / range;
        } else {
            norm_scale  = 1.f / (float)((1u << p_ptr->bits_stored) - 1u);
        }
    }

    BmpWriter bmp;
    if (!bmp_open(bmp, output_path, (int)info.width, (int)info.height, out_channels)) {
        fprintf(stderr, "Cannot create BMP: %s\n", output_path);
        return false;
    }

    const int    strip_h = 64;
    const size_t row_raw = (size_t)info.width * (size_t)info.bpp;
    std::vector<uint8_t> strip_buf((size_t)strip_h * row_raw);
    std::vector<uint8_t> row8((size_t)info.width * (size_t)out_channels);

    bool ok = true;
    for (int rows_done = 0; rows_done < (int)info.height && ok; ) {
        int got = src.read_strip(strip_buf.data(), strip_h);
        if (got <= 0) { ok = false; break; }

        for (int r = 0; r < got && ok; r++) {
            const uint8_t* row_src = strip_buf.data() + (size_t)r * row_raw;

            if (p_ptr && info.bits_per_sample == 16 && info.channels == 1) {
                // DICOM 16-bit LE grayscale → 8-bit with window/level
                dicom16_row_to_gray8(row_src, row8.data(), info.width, *p_ptr,
                                     norm_scale, norm_offset);
                ok = bmp_write_row(bmp, row8.data());
            } else if (info.bits_per_sample == 16) {
                // Non-DICOM 16-bit BE (TIFF/RAW): take MSB of each sample
                for (int x = 0; x < info.width; x++)
                    for (int c = 0; c < out_channels; c++)
                        row8[(size_t)x * out_channels + c] =
                            row_src[(size_t)(x * info.channels + c) * 2];
                ok = bmp_write_row(bmp, row8.data());
            } else {
                // 8-bit: pass through (bmp_write_row does RGB→BGR for colour)
                ok = bmp_write_row(bmp, row_src);
            }
        }
        rows_done += got;
    }

    bmp_close(bmp);
    return ok;
}

// ---------------------------------------------------------------------------
// Public BMP entry points
// ---------------------------------------------------------------------------
bool encode_dicom_to_bmp(const char* input_path,
                         const char* output_path,
                         const PipelineConfig& cfg)
{
    DicomSource src;
    if (!src.open(input_path)) {
        fprintf(stderr, "Cannot open DICOM: %s\n", input_path);
        return false;
    }
    const int frame = (cfg.frame >= 0) ? cfg.frame : 0;
    if (frame >= src.num_frames()) {
        fprintf(stderr, "DICOM: requested frame %d but file has %d frame(s)\n",
                frame, src.num_frames());
        return false;
    }
    if (!src.load_frame(frame)) return false;
    return run_bmp_from_source(output_path, src);
}

bool encode_tiff_to_bmp(const char* input_path,
                        const char* output_path,
                        const PipelineConfig& /*cfg*/)
{
    TiffSource src;
    if (!src.open(input_path)) {
        fprintf(stderr, "Cannot open TIFF: %s\n", input_path);
        return false;
    }
    return run_bmp_from_source(output_path, src);
}

bool encode_raw_to_bmp(const char* input_path,
                       const char* output_path,
                       const PipelineConfig& /*cfg*/)
{
    RawSource src;
    if (!src.open(input_path)) {
        fprintf(stderr, "Cannot open RAW: %s\n", input_path);
        return false;
    }
    return run_bmp_from_source(output_path, src);
}

// ---------------------------------------------------------------------------
// Phase D2: source-based encode (bypasses file-open step).
// Used by the batch scheduler after pre-decoding a DICOM frame on a decode
// thread. The caller is responsible for calling pipeline_record_file_times()
// with the external decode time when use_gpu_deflate is set.
// ---------------------------------------------------------------------------
bool encode_source_to_png(ImageSource& src, const char* output_path,
                          const PipelineConfig& cfg)
{
    return run_pipeline(cfg, output_path, "DICOM", src);
}

bool encode_source_to_bmp(ImageSource& src, const char* output_path)
{
    return run_bmp_from_source(output_path, src);
}
