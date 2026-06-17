#pragma once
// batch_processor.h
// Folder batch mode: scans a directory (top level only) for supported image
// files -- DICOM, TIFF, RAW -- and encodes each to PNG in an output
// directory, preserving the source filename (extension swapped for .png).
//
// Multi-frame DICOM files are written to a per-file subfolder
// output_dir/<stem>/slice_0000.png, slice_0001.png, ... reusing the existing
// tested encode_dicom_all_frames_to_png() naming exactly, rather than
// duplicating that pipeline logic with a different naming scheme.
//
// Files are processed concurrently by a worker thread pool. Each worker
// calls the existing single-file encode_*_to_png() entry points, which are
// already self-contained (own GpuFilterContext, own deflate ThreadPool, own
// PngWriter) -- so no shared mutable state needs to be added for this to be
// thread-safe. The CUDA runtime API is thread-safe across host threads
// sharing the default per-process device context; concurrent workers still
// serialize actual kernel execution on a single physical GPU, but correctness
// is unaffected.

#include "pipeline.h"

struct BatchConfig {
    int  num_workers   = 0;     // 0 = auto: min(hardware_concurrency, file_count, 4)
    bool batch_verbose = false; // append per-file wall time to each progress line;
                                // the aggregate summary always prints regardless
};

// Encode every supported file in input_dir to output_dir (created if needed).
// Returns true only if every file succeeded. Prints per-file progress and a
// final summary to stdout.
bool encode_folder_to_png(const char*           input_dir,
                          const char*           output_dir,
                          const PipelineConfig& cfg,
                          const BatchConfig&    batch_cfg);
