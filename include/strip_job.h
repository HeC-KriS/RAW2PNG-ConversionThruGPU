#pragma once
// strip_job.h
// Plain data structs that flow between pipeline stages.
// Using std::vector so ownership transfers cleanly via move.

#include <cstddef>
#include <cstdint>
#include <vector>

// Loader → GPU
struct StripJob {
    int  strip_index = 0;
    int  actual_rows = 0;  // may be < strip_height for last strip
    bool is_last     = false;
    std::vector<uint8_t> data;  // [actual_rows × width × bpp] raw pixel bytes,
                                 // 16-bit samples already byte-swapped to big-endian
};

// GPU → Deflate
struct FilteredJob {
    int  strip_index = 0;
    int  actual_rows = 0;
    bool is_last     = false;
    std::vector<uint8_t> data;  // [actual_rows × (1 + width×bpp)]
                                 // each row: [filter_byte | filtered_row_bytes]
};

// Deflate → Writer
struct CompressedJob {
    int           strip_index  = 0;
    bool          is_last      = false;
    std::vector<uint8_t> data;         // raw DEFLATE bytes for this strip
    unsigned long strip_adler  = 1;    // adler32 of the uncompressed input
    size_t        input_size   = 0;    // bytes of uncompressed input (for adler32_combine)
};
