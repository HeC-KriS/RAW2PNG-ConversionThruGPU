#pragma once
#include <cstdint>

struct PipelineConfig {
    int  strip_height     = 64;   // rows per GPU strip
    int  deflate_threads  = 6;    // CPU threads for parallel DEFLATE
    int  deflate_level    = 3;    // zlib level (1=fast, 9=best ratio)
    bool verbose          = false;
};

bool encode_tiff_to_png(const char* input_path,
                        const char* output_path,
                        const PipelineConfig& cfg);

bool encode_raw_to_png(const char* input_path,
                       const char* output_path,
                       const PipelineConfig& cfg);

// Encode an uncompressed DICOM file to PNG.
// GPU-accelerated pixel transforms (bit-depth, sign, rescale, window/level)
// are applied on the fly before PNG filtering.
bool encode_dicom_to_png(const char* input_path,
                         const char* output_path,
                         const PipelineConfig& cfg);
