#pragma once
#include <cstdint>

struct PipelineConfig {
    // Rows per GPU strip. 0 = auto: resolved once, right after CLI parsing,
    // to 1024 for the modern GPU-deflate path (empirically far faster than
    // small strips -- 32 rows ~1076 ms vs 1024 rows ~181 ms), or 256
    // otherwise (legacy CPU-deflate default, unchanged). There is no hard
    // ceiling anymore -- the prefix scan is multi-block. Strip heights up
    // to ~1M rows are supported. An explicit --strip-height always wins.
    int  strip_height     = 0;
    int  deflate_threads  = 6;    // CPU threads for parallel DEFLATE
    int  deflate_level    = 3;    // zlib level (1=fast, 9=best ratio)
    int  frame            = -1;   // DICOM frame to export: -1 = first frame
    bool verbose          = false;
    // Concurrent CUDA streams. 0 = auto: 1 on the legacy pipeline
    // (SM < 7.0), 4-8 on the modern pipeline (SM >= 7.0). Used both for
    // multi-frame DICOM export (one worker/stream per concurrently-handled
    // frame, clamped to the frame count) and, when use_gpu_deflate is set,
    // for the single-image modern pipeline's GpuFilterContext pool size
    // (Requirement 2 of the RTX 5050 pipeline-overlap PRD). See
    // detect_gpu_capability().
    int  gpu_streams      = 0;
    // Opt in to the custom GPU deflate encoder (src/gpu_deflate_backend.cu)
    // on a modern (SM >= 7.0) GPU. Defaults to false (proven CPU path) even
    // when a modern GPU is detected, because that encoder has not been run
    // on real hardware yet -- this is the validation switch, not a
    // performance toggle. Verify GPU output decodes identically to the CPU
    // path's before relying on this. Single-image encode_*_to_png() only;
    // multi-frame --all always uses the CPU path regardless of this flag.
    bool use_gpu_deflate  = false;
    // Opt in to the GPU LZ77 match finder inside the GPU deflate encoder.
    // Only effective when use_gpu_deflate is also true. Adds ~200 MB VRAM
    // for the LZ77 token arrays (at strip_height=1024, 17043px RGB-16 image).
    // Default: false. Enable with --gpu-lz77; disable with --no-gpu-lz77.
    bool use_gpu_lz77     = false;
    // When true and use_gpu_lz77 is true, dump the first 20 matches from the
    // first strip to stdout after encoding (requires --verbose). Useful to
    // confirm the match finder is producing valid back-references.
    bool gpu_lz77_debug   = false;
};

bool encode_tiff_to_png(const char* input_path,
                        const char* output_path,
                        const PipelineConfig& cfg);

bool encode_raw_to_png(const char* input_path,
                       const char* output_path,
                       const PipelineConfig& cfg);

// Encode a single DICOM frame to PNG.
// cfg.frame selects the frame (-1 = first). GPU-accelerated pixel transforms
// (bit-depth, sign, rescale, window/level) are applied before PNG filtering.
bool encode_dicom_to_png(const char* input_path,
                         const char* output_path,
                         const PipelineConfig& cfg);

// Encode every frame of a multi-frame DICOM to output_dir/slice_NNNN.png.
// Creates output_dir if needed.
//
// Frames are distributed across cfg.gpu_streams worker threads (see field
// comment above); each worker owns one GpuFilterContext (and therefore one
// CUDA stream) and one deflate ThreadPool for its entire lifetime, so frames
// it processes share that context/pool the way the original single-worker
// design did -- only the worker count changes. On the legacy pipeline this
// reduces to exactly one worker (today's sequential behavior, unchanged).
bool encode_dicom_all_frames_to_png(const char* input_path,
                                    const char* output_dir,
                                    const PipelineConfig& cfg);
