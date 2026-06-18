// batch_processor.cpp
#include "batch_processor.h"
#include "file_type.h"
#include "dicom_loader.h"
#include "thread_pool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <filesystem>
#include <future>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::milliseconds;

// Encode a single file, dispatching by detected type. DICOM files with more
// than one frame are written to output_dir/<stem>/slice_NNNN.png, reusing
// encode_dicom_all_frames_to_png()'s existing naming exactly; everything
// else (TIFF, RAW, single-frame DICOM) writes output_dir/<stem>.png.
static bool process_one_file(const fs::path& file, const fs::path& output_dir,
                             const PipelineConfig& cfg)
{
    const std::string path_str = file.string();
    const char*        path    = path_str.c_str();
    const std::string  stem    = file.stem().string();

    if (is_dicom_file(path)) {
        DicomSource probe;
        if (!probe.open(path)) {
            fprintf(stderr, "[batch] Cannot open DICOM: %s\n", path_str.c_str());
            return false;
        }
        if (probe.num_frames() > 1) {
            fs::path sub = output_dir / stem;
            std::error_code ec;
            fs::create_directories(sub, ec);
            if (ec) {
                fprintf(stderr, "[batch] Cannot create '%s': %s\n",
                        sub.string().c_str(), ec.message().c_str());
                return false;
            }
            return encode_dicom_all_frames_to_png(path, sub.string().c_str(), cfg);
        }
        fs::path outp = output_dir / (stem + ".png");
        return encode_dicom_to_png(path, outp.string().c_str(), cfg);
    }

    if (is_tiff_file(path)) {
        fs::path outp = output_dir / (stem + ".png");
        return encode_tiff_to_png(path, outp.string().c_str(), cfg);
    }

    if (is_raw_file(path)) {
        fs::path outp = output_dir / (stem + ".png");
        return encode_raw_to_png(path, outp.string().c_str(), cfg);
    }

    return false;  // unsupported -- caller already filtered the file list
}

// Per-file result returned from each worker future.
struct FileResult {
    std::string name;
    bool        ok;
    long long   ms;        // wall time from file start to file done, milliseconds
    int         worker_id; // stable 0-based index of the worker thread
};

// Thread-local worker index: assigned once per OS thread the first time that
// thread picks up a task.  Stable for the pool lifetime.
static std::atomic<int>  s_next_worker_id{0};
static thread_local int  tl_worker_id = -1;

bool encode_folder_to_png(const char*           input_dir,
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
        fprintf(stderr, "Cannot read input folder '%s': %s\n",
                input_dir, ec.message().c_str());
        return false;
    }

    if (files.empty()) {
        fprintf(stdout, "No supported files found in '%s'\n", input_dir);
        return true;
    }

    // Sort for deterministic progress-line ordering in single-worker mode
    // (multi-worker completion order is still non-deterministic by design).
    std::sort(files.begin(), files.end());

    int workers = batch_cfg.num_workers;
    if (workers <= 0) {
        int hw = (int)std::thread::hardware_concurrency();
        if (hw <= 0) hw = 4;
        workers = std::min(hw, std::min((int)files.size(), 4));
        workers = std::max(workers, 1);
    }

    const int total = (int)files.size();
    fprintf(stdout, "Batch: %d file(s), %d worker(s)%s\n",
            total, workers,
            batch_cfg.batch_verbose ? "  [--batch-verbose: per-file times shown]" : "");

    if (cfg.use_gpu_deflate)
        pipeline_reset_gpu_batch_stats();

    // Reset worker-ID counter so IDs are consistent across repeated batch runs.
    s_next_worker_id.store(0);

    const fs::path    outdir(output_dir);
    std::atomic<int>  done_count{0};
    std::mutex        print_mtx;

    ThreadPool pool(workers);
    std::vector<std::future<FileResult>> futs;
    futs.reserve(files.size());

    auto batch_start = Clock::now();

    for (auto& f : files) {
        futs.push_back(pool.submit([&, f]() -> FileResult {
            // Assign a stable 0-based ID the first time this OS thread runs a task.
            if (tl_worker_id < 0) tl_worker_id = s_next_worker_id.fetch_add(1);
            const int wid = tl_worker_id;

            auto t0  = Clock::now();
            bool ok  = process_one_file(f, outdir, cfg);
            auto t1  = Clock::now();
            long long ms = std::chrono::duration_cast<Ms>(t1 - t0).count();

            int n = ++done_count;
            {
                std::lock_guard<std::mutex> lk(print_mtx);
                if (batch_cfg.batch_verbose) {
                    fprintf(stdout, "[%d/%d] W%-2d  %-40s  %-6s  (%lld ms)\n",
                            n, total, wid,
                            f.filename().string().c_str(),
                            ok ? "OK" : "FAILED",
                            ms);
                } else {
                    fprintf(stdout, "[%d/%d] %s  %s\n",
                            n, total,
                            f.filename().string().c_str(),
                            ok ? "OK" : "FAILED");
                }
            }
            return FileResult{f.filename().string(), ok, ms, wid};
        }));
    }

    // Collect results in submission order (futures preserve input ordering).
    std::vector<FileResult> results;
    results.reserve(futs.size());
    int succeeded = 0;
    for (auto& fut : futs) {
        FileResult r = fut.get();
        if (r.ok) succeeded++;
        results.push_back(std::move(r));
    }

    auto batch_end  = Clock::now();
    long long batch_ms = std::chrono::duration_cast<Ms>(batch_end - batch_start).count();

    // Aggregate per-file timing stats
    long long sum_ms = 0;
    long long min_ms = LLONG_MAX;
    long long max_ms = 0;
    for (auto& r : results) {
        sum_ms += r.ms;
        if (r.ms < min_ms) min_ms = r.ms;
        if (r.ms > max_ms) max_ms = r.ms;
    }
    if (results.empty()) { min_ms = 0; max_ms = 0; }

    const long long avg_ms = results.empty() ? 0LL : sum_ms / (long long)results.size();
    const double fps = (batch_ms > 0) ? (double)total / ((double)batch_ms / 1000.0) : 0.0;
    const int    failed = total - succeeded;
    const bool   all_ok = (failed == 0);

    fprintf(stdout, "\n----- Batch Summary ");
    // Pad the header rule to a fixed width
    for (int i = 0; i < 41; i++) fputc('-', stdout);
    fputc('\n', stdout);

    fprintf(stdout, "Files       : %d total  |  %d OK  |  %d failed\n",
            total, succeeded, failed);
    fprintf(stdout, "Wall time   : %.1f s\n", (double)batch_ms / 1000.0);
    fprintf(stdout, "Per file    : avg %lld ms  |  min %lld ms  |  max %lld ms\n",
            avg_ms, min_ms, max_ms);
    fprintf(stdout, "Throughput  : %.2f files/s", fps);
    if (workers > 1)
        fprintf(stdout, "  (%d workers: per-file times may overlap)", workers);
    fputc('\n', stdout);

    // List failed files so the user knows exactly which ones to recheck
    if (failed > 0) {
        fprintf(stdout, "Failed files:\n");
        for (auto& r : results) {
            if (!r.ok)
                fprintf(stdout, "  FAILED  %s  (%lld ms)\n", r.name.c_str(), r.ms);
        }
    }

    if (cfg.use_gpu_deflate)
        pipeline_print_gpu_batch_summary(total, succeeded, (double)batch_ms / 1000.0,
                                         sum_ms, workers);

    return all_ok;
}
