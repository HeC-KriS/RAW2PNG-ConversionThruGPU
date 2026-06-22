#pragma once
// batch_processor.h
// Folder batch mode: scans a directory (top level only) for supported image
// files -- DICOM, TIFF, RAW -- and encodes each to the format requested by
// cfg.format (PNG or BMP), writing to an output directory.
// Output filename: <stem>.<ext>  where ext = png or bmp per cfg.format.
//
// Multi-frame DICOM files (PNG mode only) are written to a per-file subfolder
// output_dir/<stem>/slice_0000.png, slice_0001.png, ...
//
// Files are processed concurrently by a worker thread pool. Single-file
// encode_* entry points are self-contained (own GPU context, own deflate
// ThreadPool) so no shared mutable state is needed for thread safety.

#include "pipeline.h"

struct BatchConfig {
    int  num_workers   = 0;     // 0 = auto: min(hardware_concurrency, file_count, 4)
    bool batch_verbose = false; // append per-file wall time to each progress line;
                                // the aggregate summary always prints regardless
};

// Encode every supported file in input_dir to output_dir (created if needed).
// Output format is determined by cfg.format. Returns true only if every file
// succeeded. Prints per-file progress and a final summary to stdout.
bool encode_folder(const char*           input_dir,
                   const char*           output_dir,
                   const PipelineConfig& cfg,
                   const BatchConfig&    batch_cfg);

// ---------------------------------------------------------------------------
// Decode-only benchmark (Phase D2 evidence gathering).
//
// Opens every supported file in input_dir, decodes pixel data into memory,
// and discards the result — no PNG/BMP write, no encoder allocation.
// Measures header-parse time and pixel-decode time separately for DICOM.
// Prints a detailed report including worker utilization and the theoretical
// decode ceiling, which quantifies how much headroom (if any) is left for
// the encode stage.
// ---------------------------------------------------------------------------
bool benchmark_decode_folder(const char*           input_dir,
                              const PipelineConfig& cfg,
                              const BatchConfig&    batch_cfg);

// Single-file variant: decode one file and print the timing report to stdout.
// Useful for quick per-file profiling without batch overhead.
bool benchmark_decode_file(const char*           input_path,
                            const PipelineConfig& cfg);
