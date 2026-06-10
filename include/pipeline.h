#pragma once
#include <cstdint>

struct PipelineConfig {
    int  strip_height     = 64;   // rows per GPU strip
    int  deflate_threads  = 6;    // CPU threads for parallel DEFLATE
    int  deflate_level    = 3;    // zlib level (1=fast, 9=best ratio)
    bool verbose          = false;
};

// Encode a TIFF or BigTIFF file to PNG using the GPU filter pipeline.
bool encode_tiff_to_png(const char* input_path,
                        const char* output_path,
                        const PipelineConfig& cfg);

// Encode a RAW camera file (CR2/NEF/DNG/ARW…) to PNG using LibRaw.
bool encode_raw_to_png(const char* input_path,
                       const char* output_path,
                       const PipelineConfig& cfg);
