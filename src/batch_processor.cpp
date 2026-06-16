// batch_processor.cpp
#include "batch_processor.h"
#include "file_type.h"
#include "dicom_loader.h"
#include "thread_pool.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <future>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

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
        // Open just to read frame count; pixel data is not decoded here.
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

    int workers = batch_cfg.num_workers;
    if (workers <= 0) {
        int hw = (int)std::thread::hardware_concurrency();
        if (hw <= 0) hw = 4;
        workers = std::min(hw, std::min((int)files.size(), 4));
        workers = std::max(workers, 1);
    }

    fprintf(stdout, "Batch: %zu file(s), %d worker(s)\n", files.size(), workers);

    const fs::path outdir(output_dir);
    const int      total = (int)files.size();
    std::atomic<int> done_count{0};
    std::mutex       print_mtx;

    ThreadPool pool(workers);
    std::vector<std::future<bool>> futs;
    futs.reserve(files.size());

    for (auto& f : files) {
        futs.push_back(pool.submit([&, f]() -> bool {
            bool ok = process_one_file(f, outdir, cfg);
            int  n  = ++done_count;
            {
                std::lock_guard<std::mutex> lk(print_mtx);
                fprintf(stdout, "[%d/%d] %s  %s\n",
                        n, total, f.filename().string().c_str(),
                        ok ? "OK" : "FAILED");
            }
            return ok;
        }));
    }

    bool all_ok    = true;
    int  succeeded = 0;
    for (auto& fut : futs) {
        if (fut.get()) succeeded++;
        else           all_ok = false;
    }

    fprintf(stdout, "Batch finished: %d/%d succeeded\n", succeeded, total);
    return all_ok;
}
