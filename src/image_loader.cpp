// image_loader.cpp
// Strip-based loaders for TIFF (via libtiff) and RAW (via LibRaw).
//
// TIFF: uses TIFFReadEncodedStrip() instead of TIFFReadScanline().
// With RowsPerStrip=5 and 11710 rows this reduces API round-trips from
// 11 710 scanline calls to ~2 342 strip calls, cutting function-call and
// strip-lookup overhead.  libtiff still decompresses each strip once and
// caches it internally; boundary native-strips that span two pipeline strips
// benefit from that cache on the second read.
//
// Both loaders byte-swap 16-bit samples to big-endian as required by PNG.

#include "image_loader.h"

#include <tiffio.h>
#include <libraw/libraw.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------
static void byteswap16_inplace(uint8_t* buf, size_t nbytes)
{
    for (size_t i = 0; i + 1 < nbytes; i += 2) {
        const uint8_t lo = buf[i];
        buf[i]     = buf[i + 1];
        buf[i + 1] = lo;
    }
}

// ---------------------------------------------------------------------------
// TIFF reader  (TIFFReadEncodedStrip-based)
// ---------------------------------------------------------------------------
struct TiffReader {
    TIFF*             tif                  = nullptr;
    uint32_t          next_row             = 0;
    uint32_t          rows_per_native      = 1;  // TIFFTAG_ROWSPERSTRIP
    tsize_t           native_strip_bytes   = 0;  // TIFFStripSize()
    std::vector<uint8_t> native_buf;             // decode buffer (one strip)
    ImageInfo         info;
};

TiffReader* tiff_open(const char* path, ImageInfo& info_out)
{
    TIFF* tif = TIFFOpen(path, "r");
    if (!tif) return nullptr;

    uint32_t w = 0, h = 0;
    uint16_t spp = 1, bps = 8;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH,       &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH,      &h);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL,  &spp);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE,    &bps);

    if (!w || !h) { TIFFClose(tif); return nullptr; }
    if (spp != 1 && spp != 3 && spp != 4) { TIFFClose(tif); return nullptr; }
    if (bps != 8 && bps != 16)             { TIFFClose(tif); return nullptr; }

    uint32_t rps = h;  // default: whole image in one strip
    TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &rps);

    const tsize_t strip_bytes = TIFFStripSize(tif);

    TiffReader* r         = new TiffReader();
    r->tif                = tif;
    r->next_row           = 0;
    r->rows_per_native    = rps;
    r->native_strip_bytes = strip_bytes;
    r->native_buf.resize((size_t)strip_bytes);

    r->info.width           = w;
    r->info.height          = h;
    r->info.channels        = (int)spp;
    r->info.bits_per_sample = (int)bps;
    r->info.bpp             = (int)spp * (int)(bps / 8);

    info_out = r->info;
    return r;
}

int tiff_read_strip(TiffReader* r, uint8_t* out, int strip_height)
{
    if (!r || r->next_row >= r->info.height) return 0;

    const int rows_left = (int)(r->info.height - r->next_row);
    const int n_rows    = std::min(strip_height, rows_left);
    const size_t row_b  = (size_t)r->info.width * r->info.bpp;

    int rows_done = 0;
    while (rows_done < n_rows) {
        const uint32_t cur_row        = r->next_row + rows_done;
        const uint32_t native_idx     = cur_row / r->rows_per_native;
        const uint32_t first_in_native= native_idx * r->rows_per_native;
        const uint32_t offset         = cur_row - first_in_native;

        // Rows present in this native strip
        const uint32_t rows_in_native = std::min(
            r->rows_per_native,
            r->info.height - first_in_native);

        // How many of those rows we can use for this pipeline strip
        const int avail   = (int)(rows_in_native - offset);
        const int to_copy = std::min(avail, n_rows - rows_done);

        tsize_t got = TIFFReadEncodedStrip(r->tif, native_idx,
                                           r->native_buf.data(),
                                           r->native_strip_bytes);
        if (got < 0) {
            fprintf(stderr, "TIFFReadEncodedStrip failed at native strip %u\n",
                    native_idx);
            break;
        }

        for (int i = 0; i < to_copy; i++) {
            memcpy(out + (rows_done + i) * row_b,
                   r->native_buf.data() + (offset + i) * row_b,
                   row_b);
        }
        rows_done += to_copy;
    }

    // PNG 16-bit requires big-endian; libtiff returns host (little-endian) order
    if (r->info.bits_per_sample == 16)
        byteswap16_inplace(out, (size_t)n_rows * row_b);

    r->next_row += n_rows;
    return n_rows;
}

void tiff_close(TiffReader* r)
{
    if (!r) return;
    if (r->tif) TIFFClose(r->tif);
    delete r;
}

// ---------------------------------------------------------------------------
// RAW reader (LibRaw)
// Decodes the whole image on open(), then vends it in strips via memcpy.
// ---------------------------------------------------------------------------
struct RawReader {
    uint8_t*  buf      = nullptr;
    size_t    buf_size = 0;
    uint32_t  next_row = 0;
    ImageInfo info;
};

RawReader* raw_open(const char* path, ImageInfo& info_out)
{
    LibRaw raw;
    raw.imgdata.params.output_bps     = 16;
    raw.imgdata.params.no_auto_bright = 1;
    raw.imgdata.params.use_auto_wb    = 0;
    raw.imgdata.params.use_camera_wb  = 1;

    if (raw.open_file(path) != LIBRAW_SUCCESS) return nullptr;
    if (raw.unpack()        != LIBRAW_SUCCESS) return nullptr;
    if (raw.dcraw_process() != LIBRAW_SUCCESS) return nullptr;

    int errcode = 0;
    libraw_processed_image_t* img = raw.dcraw_make_mem_image(&errcode);
    if (!img || errcode != LIBRAW_SUCCESS) return nullptr;

    const uint32_t w  = img->width;
    const uint32_t h  = img->height;
    const int      ch = img->colors;
    const int      bps= img->bits;
    const size_t   row_b  = (size_t)w * ch * (bps / 8);
    const size_t   total  = (size_t)h * row_b;

    RawReader* r  = new RawReader();
    r->buf        = new uint8_t[total];
    r->buf_size   = total;
    r->next_row   = 0;

    memcpy(r->buf, img->data, total);
    LibRaw::dcraw_clear_mem(img);

    if (bps == 16) byteswap16_inplace(r->buf, total);

    r->info.width           = w;
    r->info.height          = h;
    r->info.channels        = ch;
    r->info.bits_per_sample = bps;
    r->info.bpp             = ch * (bps / 8);

    info_out = r->info;
    return r;
}

int raw_read_strip(RawReader* r, uint8_t* out, int strip_height)
{
    if (!r || r->next_row >= r->info.height) return 0;

    const int rows_left = (int)(r->info.height - r->next_row);
    const int n_rows    = std::min(strip_height, rows_left);
    const size_t row_b  = (size_t)r->info.width * r->info.bpp;

    memcpy(out, r->buf + r->next_row * row_b, (size_t)n_rows * row_b);
    r->next_row += n_rows;
    return n_rows;
}

void raw_close(RawReader* r)
{
    if (!r) return;
    delete[] r->buf;
    delete r;
}
