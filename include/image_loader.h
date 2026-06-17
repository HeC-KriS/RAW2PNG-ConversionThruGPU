#pragma once
#include <cstdint>
#include <cstddef>
#include <string>

// Describes the decoded image so the pipeline can allocate correctly.
struct ImageInfo {
    uint32_t width        = 0;
    uint32_t height       = 0;
    int      channels     = 0;  // 1 (gray) or 3 (RGB)
    int      bits_per_sample = 0;  // 8 or 16
    int      bpp          = 0;  // channels * bits_per_sample/8
};

// TIFF loader
// Reads one strip of scanlines at a time.  The caller must call
// tiff_open / tiff_read_strip (in order, strip 0 upward) / tiff_close.
// Output samples are in big-endian byte order (required by PNG spec for
// 16-bit).  8-bit images are left as-is.
struct TiffReader;
TiffReader* tiff_open(const char* path, ImageInfo& info_out);
// Returns the number of rows actually read (may be < strip_height for last).
int         tiff_read_strip(TiffReader* r, uint8_t* out, int strip_height);
void        tiff_close(TiffReader* r);

// RAW loader (LibRaw)
// Decodes the whole image into memory, then vends strips.
// Output is demosaiced RGB, 16-bit big-endian or 8-bit depending on the RAW.
struct RawReader;
RawReader*  raw_open(const char* path, ImageInfo& info_out);
int         raw_read_strip(RawReader* r, uint8_t* out, int strip_height);
void        raw_close(RawReader* r);
