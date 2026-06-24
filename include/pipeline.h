#pragma once
#include <cstdint>

struct ImageSource;  // forward declaration — full definition in image_source.h

enum class ExportFormat { PNG, BMP, JLS };

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
    // Export format: PNG (default), BMP (fast uncompressed), or JLS (JPEG-LS lossless).
    ExportFormat format   = ExportFormat::PNG;
    // Use GPU JPEG-LS pipeline (G1–G5 kernels) instead of CPU CharLS.
    // Only effective when format == ExportFormat::JLS.
    bool use_gpu_jls      = false;
    // JLS-C1: use ROW_CARRY context propagation in G2 instead of ROW_RESET.
    // Only effective when use_gpu_jls is also true.
    // Rows execute sequentially in stream order; produces spec-compliant
    // context carry and materially smaller output. Enable with --gpu-jls-carry.
    bool use_gpu_jls_carry  = false;
    // JLS-C1 v3: GPU for G1 only; CPU does G2 + full ISO run-mode coding.
    // Produces spec-compliant run-mode → smaller output than ROW_RESET.
    // Enable with --gpu-jls-cpu-rm. Mutually exclusive with --gpu-jls-carry.
    bool use_gpu_jls_cpu_rm = false;
    // GPU BMP: move Window/Level, Rescale, Normalization, and 16→8 conversion
    // to GPU (bmp_dicom_to_gray8_kernel). CPU retains DICOM decode + BMP write.
    // Enable with --gpu-bmp. Only effective for 16-bit grayscale DICOM sources.
    bool use_gpu_bmp = false;
};

// ---- PNG encode (lossless, compressed) ------------------------------------
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

// Batch telemetry — call reset before the batch, print_batch_summary after.
// Always active for all formats (BMP, JLS CPU, JLS GPU, PNG CPU, PNG GPU).
void pipeline_reset_gpu_batch_stats();

struct FileTelemetry {
    // Input I/O breakdown (from decode_dicom_to_frame)
    long long file_open_us    = 0;
    long long file_read_us    = 0;
    long long dicom_parse_us  = 0;
    long long pixel_decode_us = 0;
    // Processing breakdown (from encode_predecoded_dicom)
    long long encode_us       = 0;  // CPU encode: CharLS, BMP convert, PNG filter+compress
    long long write_us        = 0;  // output file write (separated where possible)
    // GPU JLS per-phase (µs) — only populated for use_gpu_jls paths
    long long jls_h2d_us      = 0;  // H2D pixel copy
    long long jls_g1_us       = 0;  // G1 LOCO-I kernel
    long long jls_g2_us       = 0;  // G2 Golomb-k (ROW_RESET) or D2H residuals (CPU_RM)
    long long jls_g3_us       = 0;  // G3 bit-lengths (ROW_RESET only)
    long long jls_g4_us       = 0;  // G4 prefix scan (ROW_RESET only)
    long long jls_g5_us       = 0;  // G5 bit-emit (ROW_RESET only)
    long long jls_d2h_us      = 0;  // D2H bitstream (ROW_RESET only)
    long long jls_cpu_wrap_us = 0;  // byte-stuff+write (ROW_RESET) or encode+write (CPU_RM)
    // GPU BMP per-phase (µs) — only populated when use_gpu_bmp is true
    long long bmp_h2d_us      = 0;  // H2D: pixels to device
    long long bmp_kernel_us   = 0;  // window/level + 16→8 conversion kernel
    long long bmp_d2h_us      = 0;  // D2H: 8-bit output to host
    long long total_ms        = 0;
};

// Record per-file lifecycle timings into the global batch accumulator.
// Call once per file from the encode worker (all formats).
void pipeline_record_file_times(const FileTelemetry& t);

// Print the full input/processing/output telemetry breakdown for the completed batch.
// fmt / use_gpu_deflate / use_gpu_jls / use_gpu_jls_cpu_rm determine which
// format-specific processing section is shown.
void pipeline_print_batch_summary(int total_files, int succeeded,
                                  double total_wall_s,
                                  long long sum_file_batch_ms,
                                  int num_workers,
                                  ExportFormat fmt,
                                  bool use_gpu_deflate,
                                  bool use_gpu_jls,
                                  bool use_gpu_jls_cpu_rm,
                                  bool use_gpu_bmp);

// ---- BMP encode (lossless, uncompressed, fastest export path) -------------
// Converts all inputs to 8-bit grayscale (1-channel) or 24-bit BGR (3-channel).
// For 16-bit DICOM: applies window/level from DicomPixelParams → 8-bit.
// For 16-bit TIFF/RAW: takes the high 8 bits of each channel (no windowing).
bool encode_dicom_to_bmp(const char* input_path,
                         const char* output_path,
                         const PipelineConfig& cfg);

bool encode_tiff_to_bmp(const char* input_path,
                        const char* output_path,
                        const PipelineConfig& cfg);

bool encode_raw_to_bmp(const char* input_path,
                       const char* output_path,
                       const PipelineConfig& cfg);

// ---- Phase D2: source-based encode (batch scheduler decode-ahead path) ----
// Encode directly from an already-decoded ImageSource, bypassing file-open.
// encode_source_to_png: caller must call pipeline_record_file_times() for GPU stats.
// encode_source_to_bmp: stateless, no GPU stats.
bool encode_source_to_png(ImageSource& src,
                          const char*  output_path,
                          const PipelineConfig& cfg);

bool encode_source_to_bmp(ImageSource& src,
                          const char*  output_path);
