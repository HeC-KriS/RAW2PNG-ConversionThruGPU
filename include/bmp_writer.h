#pragma once
#include <cstdint>
#include <cstdio>

// Streaming BMP writer — write one row at a time, top-to-bottom.
// Uses negative biHeight in the header (GDI+ / Windows convention) so rows
// are stored top-to-bottom with no flip required.
//
// Supported modes:
//   channels = 1  →  8-bit grayscale BMP (256-entry grayscale palette)
//   channels = 3  →  24-bit BGR BMP (bmp_write_row converts RGB→BGR)
struct BmpWriter {
    FILE* fp       = nullptr;
    int   width    = 0;
    int   stride   = 0;   // padded row width in bytes (multiple of 4)
    int   height   = 0;
    int   channels = 0;   // 1 or 3
    int   written  = 0;   // rows written so far
};

// Open (create/truncate) a BMP file and write the file header + info header
// + palette (grayscale only). Returns false on error.
bool bmp_open(BmpWriter& w, const char* path, int width, int height, int channels);

// Write one scanline.
//   channels=1: `src` is `width` grayscale bytes.
//   channels=3: `src` is `width*3` RGB bytes; converted to BGR internally.
// Returns false on I/O error.
bool bmp_write_row(BmpWriter& w, const uint8_t* src);

// Flush and close the BMP file.
void bmp_close(BmpWriter& w);
