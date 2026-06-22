// bmp_writer.cpp
// Streaming 8-bit grayscale and 24-bit BGR BMP writer.
//
// BMP layout written:
//   BITMAPFILEHEADER   14 bytes
//   BITMAPINFOHEADER   40 bytes
//   [Color table       1024 bytes  (grayscale only, 256 × 4)]
//   Pixel data         stride × height bytes  (stride = width padded to 4)
//
// biHeight is written as –height (negative) so rows are stored top-to-bottom,
// matching the source order from all our image loaders. This avoids an in-place
// row-reversal and is supported by every modern viewer.

#include "bmp_writer.h"
#include <algorithm>
#include <cstring>

// ---------------------------------------------------------------------------
// Little-endian helpers (BMP is always LE regardless of host byte order)
// ---------------------------------------------------------------------------
static void put_u16(uint8_t* p, uint16_t v)
{
    p[0] = (uint8_t)(v      );
    p[1] = (uint8_t)(v >>  8);
}
static void put_u32(uint8_t* p, uint32_t v)
{
    p[0] = (uint8_t)(v      );
    p[1] = (uint8_t)(v >>  8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}
static void put_i32(uint8_t* p, int32_t v) { put_u32(p, (uint32_t)v); }

// ---------------------------------------------------------------------------
bool bmp_open(BmpWriter& w, const char* path, int width, int height, int channels)
{
    if (channels != 1 && channels != 3) return false;

    w.width    = width;
    w.channels = channels;
    w.height   = height;
    w.written  = 0;

    // Row stride padded to 4-byte boundary
    w.stride = (width * channels + 3) & ~3;

    w.fp = fopen(path, "wb");
    if (!w.fp) return false;

    const uint32_t palette_bytes = (channels == 1) ? 256u * 4u : 0u;
    const uint32_t pixel_offset  = 14u + 40u + palette_bytes;
    const uint32_t pixel_bytes   = (uint32_t)w.stride * (uint32_t)height;
    const uint32_t file_size     = pixel_offset + pixel_bytes;

    // Buffer the entire file in one stdio allocation so all per-row fwrite calls
    // accumulate in memory and flush as a single OS write on fclose.
    // Cap at 4 MB for very large images; minimum 64 KB to cover BMP headers.
    const uint32_t buf_sz = std::max(65536u, std::min(file_size, 4u * 1024u * 1024u));
    setvbuf(w.fp, nullptr, _IOFBF, (size_t)buf_sz);

    // BITMAPFILEHEADER (14 bytes)
    uint8_t fhdr[14] = {};
    fhdr[0] = 'B'; fhdr[1] = 'M';
    put_u32(fhdr + 2,  file_size);
    // bytes 6-9: reserved = 0
    put_u32(fhdr + 10, pixel_offset);
    if (fwrite(fhdr, 1, 14, w.fp) != 14) { bmp_close(w); return false; }

    // BITMAPINFOHEADER (40 bytes)
    uint8_t info[40] = {};
    put_u32(info +  0, 40u);                 // biSize
    put_i32(info +  4, (int32_t)width);      // biWidth
    put_i32(info +  8, -(int32_t)height);    // biHeight (negative = top-to-bottom)
    put_u16(info + 12, 1u);                  // biPlanes
    put_u16(info + 14, (uint16_t)(channels == 1 ? 8u : 24u));  // biBitCount
    // biCompression = 0 (BI_RGB), biSizeImage = 0, pels = 0, etc. all zero
    put_u32(info + 32, (uint32_t)(channels == 1 ? 256u : 0u));  // biClrUsed
    if (fwrite(info, 1, 40, w.fp) != 40) { bmp_close(w); return false; }

    // Grayscale palette: 256 entries, B=G=R=i, reserved=0
    if (channels == 1) {
        uint8_t pal[256 * 4];
        for (int i = 0; i < 256; i++) {
            pal[i * 4 + 0] = (uint8_t)i;
            pal[i * 4 + 1] = (uint8_t)i;
            pal[i * 4 + 2] = (uint8_t)i;
            pal[i * 4 + 3] = 0;
        }
        if (fwrite(pal, 1, sizeof(pal), w.fp) != sizeof(pal)) {
            bmp_close(w); return false;
        }
    }

    return true;
}

bool bmp_write_row(BmpWriter& w, const uint8_t* src)
{
    if (!w.fp || w.written >= w.height) return false;

    if (w.channels == 1) {
        // Grayscale: write width bytes then pad to stride
        if (fwrite(src, 1, (size_t)w.width, w.fp) != (size_t)w.width) return false;
    } else {
        // RGB→BGR: write in-place converted triplets.
        // Process in chunks using a small stack buffer to avoid per-row malloc.
        uint8_t buf[1024 * 3];
        int x = 0;
        while (x < w.width) {
            int chunk = w.width - x;
            if (chunk > 1024) chunk = 1024;
            for (int i = 0; i < chunk; i++) {
                buf[i * 3 + 0] = src[(x + i) * 3 + 2];  // B
                buf[i * 3 + 1] = src[(x + i) * 3 + 1];  // G
                buf[i * 3 + 2] = src[(x + i) * 3 + 0];  // R
            }
            if (fwrite(buf, 1, (size_t)chunk * 3, w.fp) != (size_t)chunk * 3)
                return false;
            x += chunk;
        }
    }

    // Padding to 4-byte boundary
    int pad = w.stride - w.width * w.channels;
    if (pad > 0) {
        const uint8_t zeros[3] = {};
        if (fwrite(zeros, 1, (size_t)pad, w.fp) != (size_t)pad) return false;
    }

    w.written++;
    return true;
}

void bmp_close(BmpWriter& w)
{
    if (w.fp) { fclose(w.fp); w.fp = nullptr; }
    w.written = 0;
}
