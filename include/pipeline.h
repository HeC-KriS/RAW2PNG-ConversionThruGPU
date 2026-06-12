#pragma once
#include <cstdint>

struct PipelineConfig {
    int  strip_height     = 256;  // rows per GPU strip
    int  deflate_threads  = 6;    // CPU threads for parallel DEFLATE
    int  deflate_level    = 3;    // zlib level (1=fast, 9=best ratio)
    int  frame            = -1;   // DICOM frame to export: -1 = first frame
    bool verbose          = false;
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
// Creates output_dir if needed. GPU context and thread pool are built once and
// reused across all frames.
bool encode_dicom_all_frames_to_png(const char* input_path,
                                    const char* output_dir,
                                    const PipelineConfig& cfg);
