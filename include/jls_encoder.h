#pragma once
#include <cstddef>
#include <cstdint>

// Encode raw 8-bit grayscale pixels to JPEG-LS (lossless).
// On success: *out_buf points to the compressed data (allocated with new[]);
//             caller must delete[] *out_buf.  Returns bytes written (> 0).
// On failure: *out_buf = nullptr, returns 0.
size_t jls_encode_gray8(const uint8_t* pixels, int width, int height,
                        uint8_t** out_buf);
