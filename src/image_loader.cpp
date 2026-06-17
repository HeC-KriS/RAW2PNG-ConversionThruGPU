// image_loader.cpp
// Strip-based loaders for TIFF (via libtiff) and RAW (via LibRaw).
//
// TIFF loader handles:
//   - Strip-organised TIFFs (TIFFTAG_ROWSPERSTRIP): uses TIFFReadEncodedStrip
//     with a one-strip decode cache so each compressed native strip is decoded
//     at most once even when the pipeline strip height is smaller.
//   - Tile-organised TIFFs (TIFFIsTiled): uses TIFFReadScanline which internally
//     manages a per-tile decode cache.
//   - PHOTOMETRIC_MINISWHITE: byte-inverts output so that white=max (PNG convention).
//   - PLANARCONFIG_SEPARATE (multi-channel): rejected with a conversion hint.
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
// TIFF reader
// ---------------------------------------------------------------------------
struct TiffReader {
    TIFF*             tif                = nullptr;
    uint32_t          next_row           = 0;
    ImageInfo         info;

    // Strip-based fields
    uint32_t          rows_per_native    = 1;
    tsize_t           native_strip_bytes = 0;
    std::vector<uint8_t> native_buf;       // decode buffer; also used as scanline buf
    int               last_native_idx    = -1;  // -1 = no cached strip

    // Tiled TIFF flag
    bool              is_tiled           = false;

    // Photometric for MINISWHITE inversion
    uint16_t          photometric        = 1;  // 1 = MINISBLACK (normal)
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

    // Reject separate-plane TIFFs: strip reader assumes interleaved data.
    uint16_t planar = PLANARCONFIG_CONTIG;
    TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &planar);
    if (planar != PLANARCONFIG_CONTIG && spp > 1) {
        fprintf(stderr,
            "TIFF: separate-plane (planar) format is not supported in '%s'.\n"
            "      Convert to interleaved with:\n"
            "        tiffcp -c none -p contig \"%s\" output.tif\n",
            path, path);
        TIFFClose(tif);
        return nullptr;
    }

    // Photometric: detect MINISWHITE so we can invert the output.
    uint16_t photometric = PHOTOMETRIC_MINISBLACK;
    TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photometric);

    TiffReader* r       = new TiffReader();
    r->tif              = tif;
    r->next_row         = 0;
    r->photometric      = photometric;
    r->is_tiled         = (TIFFIsTiled(tif) != 0);

    r->info.width           = w;
    r->info.height          = h;
    r->info.channels        = (int)spp;
    r->info.bits_per_sample = (int)bps;
    r->info.bpp             = (int)spp * (int)(bps / 8);

    if (r->is_tiled) {
        // For tiled TIFFs, use TIFFReadScanline which manages a tile decode cache.
        // Allocate a scratch buffer of one scanline.
        const tsize_t scan_bytes = TIFFScanlineSize(tif);
        r->native_buf.resize((size_t)scan_bytes);
    } else {
        uint32_t rps = h;
        TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &rps);
        r->rows_per_native    = rps;
        r->native_strip_bytes = TIFFStripSize(tif);
        r->native_buf.resize((size_t)r->native_strip_bytes);
        r->last_native_idx    = -1;
    }

    info_out = r->info;
    return r;
}

int tiff_read_strip(TiffReader* r, uint8_t* out, int strip_height)
{
    if (!r || r->next_row >= r->info.height) return 0;

    const int    rows_left = (int)(r->info.height - r->next_row);
    const int    n_rows    = std::min(strip_height, rows_left);
    const size_t row_b     = (size_t)r->info.width * r->info.bpp;

    if (r->is_tiled) {
        // Tiled TIFF: read scanline by scanline.
        // libtiff internally decodes each tile once and caches it, so sequential
        // access costs one tile decode per tile (not one per scanline).
        for (int i = 0; i < n_rows; i++) {
            uint32_t y = r->next_row + (uint32_t)i;
            if (TIFFReadScanline(r->tif, r->native_buf.data(), y, 0) < 0) {
                fprintf(stderr, "TIFFReadScanline failed at row %u\n", y);
                if (i == 0) { r->next_row += 0; return 0; }
                // Partial success: byteswap what we have and return
                if (r->info.bits_per_sample == 16)
                    byteswap16_inplace(out, (size_t)i * row_b);
                if (r->photometric == PHOTOMETRIC_MINISWHITE) {
                    for (size_t k = 0; k < (size_t)i * row_b; k++)
                        out[k] ^= 0xFF;
                }
                r->next_row += (uint32_t)i;
                return i;
            }
            memcpy(out + (size_t)i * row_b, r->native_buf.data(), row_b);
        }
    } else {
        // Strip-based TIFF: decode each native strip once, serve pipeline strips from cache.
        int rows_done = 0;
        while (rows_done < n_rows) {
            const uint32_t cur_row         = r->next_row + (uint32_t)rows_done;
            const uint32_t native_idx      = cur_row / r->rows_per_native;
            const uint32_t first_in_native = native_idx * r->rows_per_native;
            const uint32_t offset          = cur_row - first_in_native;

            const uint32_t rows_in_native = std::min(
                r->rows_per_native,
                r->info.height - first_in_native);
            const int avail   = (int)(rows_in_native - offset);
            const int to_copy = std::min(avail, n_rows - rows_done);

            // Decode only if this native strip is not already cached.
            // This avoids re-decompressing when strip_height < rows_per_native.
            if ((int)native_idx != r->last_native_idx) {
                tsize_t got = TIFFReadEncodedStrip(r->tif, native_idx,
                                                   r->native_buf.data(),
                                                   r->native_strip_bytes);
                if (got < 0) {
                    fprintf(stderr, "TIFFReadEncodedStrip failed at native strip %u\n",
                            native_idx);
                    break;
                }
                r->last_native_idx = (int)native_idx;
            }

            for (int i = 0; i < to_copy; i++) {
                memcpy(out + (size_t)(rows_done + i) * row_b,
                       r->native_buf.data() + (size_t)(offset + i) * row_b,
                       row_b);
            }
            rows_done += to_copy;
        }
        if (rows_done < n_rows) {
            // Partial decode: byteswap what we have and return
            if (r->info.bits_per_sample == 16)
                byteswap16_inplace(out, (size_t)rows_done * row_b);
            if (r->photometric == PHOTOMETRIC_MINISWHITE) {
                for (size_t k = 0; k < (size_t)rows_done * row_b; k++)
                    out[k] ^= 0xFF;
            }
            r->next_row += (uint32_t)rows_done;
            return rows_done;
        }
    }

    // PNG requires big-endian 16-bit samples; libtiff returns host (LE) order
    if (r->info.bits_per_sample == 16)
        byteswap16_inplace(out, (size_t)n_rows * row_b);

    // PHOTOMETRIC_MINISWHITE: 0=white, max=black — invert to get 0=black, max=white.
    // Byte-level NOT is endian-neutral: ~(hi*256+lo) = (~hi)*256 + (~lo).
    if (r->photometric == PHOTOMETRIC_MINISWHITE) {
        const size_t total = (size_t)n_rows * row_b;
        for (size_t k = 0; k < total; k++)
            out[k] ^= 0xFF;
    }

    r->next_row += (uint32_t)n_rows;
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
