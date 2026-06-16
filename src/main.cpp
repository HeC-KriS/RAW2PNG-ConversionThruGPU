// main.cpp
// CLI entry point.
// Usage:
//   gpu_png_encoder <input> <output.png> [options]      single file
//   gpu_png_encoder <input_folder> <output_folder>       batch mode
//
// Options:
//   --strip-height N   rows per GPU strip        (default: auto -- 256 legacy,
//                                                  1024 modern w/ --gpu-deflate)
//   --threads N        CPU DEFLATE threads        (default 6)
//   --level N          zlib compression level     (default 3, range 0-9)
//   --verbose          print per-stage timing with wall-clock timestamps
//   --batch-workers N  concurrent files in folder mode (default: auto)
//   --bench-deflate [N]  run standalone deflate benchmark on N bytes of
//                        synthetic data (default 2400000); no input file needed

#include "pipeline.h"
#include "parallel_deflate.h"
#include "file_type.h"
#include "batch_processor.h"
#include "gpu_capability.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <string>
#include <filesystem>

static void usage(const char* prog)
{
    fprintf(stderr,
        "Usage: %s <input> <output> [options]\n"
        "  Single file:\n"
        "    <input>  : DICOM (.dcm .dicom .dic .ima or DICM magic), TIFF (.tif .tiff),\n"
        "               RAW (.cr2 .nef .dng .arw .raf .orf .rw2 .raw)\n"
        "    <output> : a .png file (single frame), or a folder with --all\n"
        "  Folder batch mode:\n"
        "    <input>  : a folder containing any mix of the formats above\n"
        "    <output> : a folder; one .png per input file is written there\n"
        "               (multi-frame DICOM gets its own <stem>/ subfolder)\n"
        "  Options:\n"
        "  --all                <output> is a folder; every DICOM frame is written as\n"
        "                       slice_0000.png, slice_0001.png, ...   (alias: -all)\n"
        "  --frame N            export only DICOM frame N (0-based)\n"
        "  --strip-height N     rows per GPU strip   (default: auto -- 256 legacy,\n"
        "                                            1024 modern w/ --gpu-deflate)\n"
        "  --threads N          DEFLATE threads      (default 6)\n"
        "  --level N            zlib level 0-9       (default 3)\n"
        "  --verbose            print timing report with wall-clock timestamps\n"
        "  --batch-workers N    concurrent files in folder mode (default: auto)\n"
        "  --gpu-streams N      concurrent CUDA streams: --all multi-frame export,\n"
        "                       and the filter-stage pool size for --gpu-deflate\n"
        "                       (default: auto -- 1 legacy GPU, 4-8 modern GPU)\n"
        "  --gpu-deflate        opt in to the custom GPU deflate encoder on a modern\n"
        "                       GPU (single-file mode only); requires --strip-height\n"
        "                       <= 1024; UNVERIFIED on real hardware -- decode-compare\n"
        "                       against the default CPU path before trusting output\n"
        "                       produced with this flag\n"
        "  --bench-deflate [N]  standalone deflate benchmark on N bytes\n"
        "                       (default 2400000); no input file needed\n",
        prog);
}

int main(int argc, char* argv[])
{
    // --bench-deflate is a standalone mode; no input/output file is needed.
    // Quick-scan argv before the argc < 3 check so it can be the only argument.
    {
        size_t bench_strip   = 0;
        int    bench_threads = 6;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--bench-deflate") == 0) {
                bench_strip = 2400000;  // ~2.3 MB: typical 256-row strip for 4728px 16-bit
                if (i + 1 < argc && isdigit((unsigned char)argv[i + 1][0]))
                    bench_strip = (size_t)strtoul(argv[++i], nullptr, 10);
            } else if ((strcmp(argv[i], "--threads") == 0) && i + 1 < argc) {
                bench_threads = atoi(argv[++i]);
            }
        }
        if (bench_strip > 0) {
            if (bench_threads < 1) bench_threads = 1;
            bench_deflate(bench_strip, bench_threads);
            return 0;
        }
    }

    if (argc < 3) { usage(argv[0]); return 1; }

    const char* input  = argv[1];
    const char* output = argv[2];

    PipelineConfig cfg;
    BatchConfig    batch_cfg;
    bool all_frames = false;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--strip-height") == 0 && i + 1 < argc)
            cfg.strip_height = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            cfg.deflate_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--level") == 0 && i + 1 < argc)
            cfg.deflate_level = atoi(argv[++i]);
        else if (strcmp(argv[i], "--frame") == 0 && i + 1 < argc)
            cfg.frame = atoi(argv[++i]);
        else if (strcmp(argv[i], "--all") == 0 || strcmp(argv[i], "-all") == 0)
            all_frames = true;
        else if (strcmp(argv[i], "--verbose") == 0)
            cfg.verbose = true;
        else if (strcmp(argv[i], "--batch-workers") == 0 && i + 1 < argc)
            batch_cfg.num_workers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gpu-streams") == 0 && i + 1 < argc)
            cfg.gpu_streams = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gpu-deflate") == 0)
            cfg.use_gpu_deflate = true;
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    // Clamp level
    if (cfg.deflate_level < 0) cfg.deflate_level = 0;
    if (cfg.deflate_level > 9) cfg.deflate_level = 9;
    if (cfg.deflate_threads < 1) cfg.deflate_threads = 1;
    if (cfg.strip_height    < 0) cfg.strip_height    = 0;  // negative -> auto

    // GPU capability is needed for strip-height auto-selection regardless of
    // --verbose, so query it once here and reuse it for both.
    GpuCapability gpu_cap = detect_gpu_capability();
    const bool modern_deflate = cfg.use_gpu_deflate && gpu_cap.mode == GpuPipelineMode::Modern;

    bool strip_height_auto = (cfg.strip_height == 0);
    if (strip_height_auto) {
        // See PipelineConfig::strip_height: 1024 is both the device-scan
        // hard ceiling and empirically far faster for the modern path;
        // 256 preserves the existing legacy default unchanged.
        cfg.strip_height = modern_deflate ? 1024 : 256;
    }

    if (cfg.verbose) {
        fprintf(stdout, "%s\n", describe_gpu_capability(gpu_cap).c_str());
        if (gpu_cap.available && gpu_cap.mode == GpuPipelineMode::Modern) {
            fprintf(stdout, cfg.use_gpu_deflate
                ? "Modern GPU detected; using the custom GPU deflate encoder "
                  "(--gpu-deflate, UNVERIFIED on real hardware).\n"
                : "Modern GPU detected; using the proven CPU-deflate path. "
                  "Pass --gpu-deflate to opt in to the unverified GPU deflate encoder.\n");
        }
        fprintf(stdout, "strip_height = %d%s\n", cfg.strip_height,
                strip_height_auto ? " (auto-selected)" : " (explicit)");
    }

    // Folder batch mode: input is a directory.
    std::error_code fs_ec;
    if (std::filesystem::is_directory(input, fs_ec)) {
        bool ok = encode_folder_to_png(input, output, cfg, batch_cfg);
        if (!ok) {
            fprintf(stderr, "Batch encoding finished with errors.\n");
            return 1;
        }
        return 0;
    }

    const bool is_tiff = is_tiff_file(input);
    const bool is_raw  = is_raw_file(input);

    // --all only applies to DICOM (TIFF/RAW are always single-image here).
    if (all_frames && (is_tiff || is_raw)) {
        fprintf(stderr, "Note: --all applies only to DICOM; ignoring for this input.\n");
        all_frames = false;
    }

    // Dispatch: try DICOM first (header sniff covers files with no/unknown extension),
    // then TIFF, then RAW camera.
    bool ok = false;
    if (is_tiff) {
        ok = encode_tiff_to_png(input, output, cfg);
    } else if (is_raw) {
        ok = encode_raw_to_png(input, output, cfg);
    } else if (is_dicom_file(input)) {
        // Covers: .dcm .dicom .dic .ima  +  numeric names  +  no-extension DICOM
        // The sniff_dicom_magic check reads 132 bytes and verifies the "DICM"
        // preamble — this is the only reliable way to detect DICOM without an
        // extension (very common on PACS systems and DICOM CDs).
        if (all_frames) ok = encode_dicom_all_frames_to_png(input, output, cfg);
        else            ok = encode_dicom_to_png(input, output, cfg);
    } else {
        // Unknown extension: try DICOM first, then RAW, then TIFF.
        if (all_frames) {
            ok = encode_dicom_all_frames_to_png(input, output, cfg);
        } else {
            ok = encode_dicom_to_png(input, output, cfg);
            if (!ok) ok = encode_raw_to_png(input, output, cfg);
            if (!ok) ok = encode_tiff_to_png(input, output, cfg);
        }
    }

    if (!ok) {
        fprintf(stderr, "Encoding failed.\n");
        return 1;
    }

    if (cfg.verbose && !all_frames)
        fprintf(stdout, "Done → %s\n", output);
    return 0;
}
