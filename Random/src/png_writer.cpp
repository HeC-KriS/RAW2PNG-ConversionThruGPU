// png_writer.cpp
// Minimal PNG file writer that bypasses libpng's built-in DEFLATE so the
// caller can supply raw DEFLATE bytes produced by the parallel compressor.
//
// PNG chunk layout:  [4-byte length BE] [4-byte type] [data] [4-byte CRC32]
// CRC32 covers the type field + data.
// IDAT data is the raw zlib stream: zlib_header + deflate_blocks + adler32.

#include "png_writer.h"

#include <zlib.h>        // for crc32()
#include <cstring>
#include <cassert>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void write_u32_be(FILE* f, uint32_t v)
{
    uint8_t b[4] = {
        (uint8_t)(v >> 24), (uint8_t)(v >> 16),
        (uint8_t)(v >>  8), (uint8_t)(v & 0xFF) };
    fwrite(b, 1, 4, f);
}

static const uint8_t PNG_SIG[8] = { 137,80,78,71,13,10,26,10 };

// ---------------------------------------------------------------------------
// PngWriter
// ---------------------------------------------------------------------------
PngWriter::~PngWriter() { close(); }

bool PngWriter::open(const char* path,
                     uint32_t width, uint32_t height,
                     uint8_t bit_depth, PngColorType color_type)
{
    file_ = fopen(path, "wb");
    if (!file_) return false;

    // Use a 512 KB stdio buffer to batch OS writes.  The IDAT buffer is 32 KB
    // so a 56 MB compressed file would otherwise produce ~1700 fwrite calls;
    // this reduces them to ~107 at no cost to IDAT chunk semantics.
    setvbuf(file_, nullptr, _IOFBF, 512 * 1024);

    // Signature
    fwrite(PNG_SIG, 1, 8, file_);

    // IHDR (13 bytes)
    uint8_t ihdr[13];
    ihdr[ 0] = (uint8_t)(width  >> 24);
    ihdr[ 1] = (uint8_t)(width  >> 16);
    ihdr[ 2] = (uint8_t)(width  >>  8);
    ihdr[ 3] = (uint8_t)(width       );
    ihdr[ 4] = (uint8_t)(height >> 24);
    ihdr[ 5] = (uint8_t)(height >> 16);
    ihdr[ 6] = (uint8_t)(height >>  8);
    ihdr[ 7] = (uint8_t)(height       );
    ihdr[ 8] = bit_depth;
    ihdr[ 9] = (uint8_t)color_type;
    ihdr[10] = 0;  // compression method (always 0 = deflate)
    ihdr[11] = 0;  // filter method (always 0)
    ihdr[12] = 0;  // interlace method (0 = no interlace)
    write_chunk("IHDR", ihdr, 13);

    idat_pos_ = 0;
    return true;
}

void PngWriter::begin_idat()
{
    idat_pos_ = 0;
}

void PngWriter::write_idat_bytes(const uint8_t* data, size_t size)
{
    while (size > 0) {
        const size_t room = sizeof(idat_buf_) - idat_pos_;
        const size_t take = (size < room) ? size : room;
        memcpy(idat_buf_ + idat_pos_, data, take);
        idat_pos_ += take;
        data      += take;
        size      -= take;
        if (idat_pos_ == sizeof(idat_buf_))
            flush_idat_buf();
    }
}

void PngWriter::end_idat()
{
    if (idat_pos_ > 0)
        flush_idat_buf();
}

void PngWriter::close()
{
    if (!file_) return;
    // IEND
    write_chunk("IEND", nullptr, 0);
    fclose(file_);
    file_ = nullptr;
}

// ---------------------------------------------------------------------------
// Private
// ---------------------------------------------------------------------------
void PngWriter::flush_idat_buf()
{
    if (idat_pos_ == 0) return;
    write_chunk("IDAT", idat_buf_, (uint32_t)idat_pos_);
    idat_pos_ = 0;
}

void PngWriter::write_chunk(const char type[4], const uint8_t* data, uint32_t len)
{
    write_u32_be(file_, len);
    fwrite(type, 1, 4, file_);
    if (len > 0 && data) fwrite(data, 1, len, file_);

    // CRC32 over type + data
    uLong crc = crc32(0L, nullptr, 0);
    crc = crc32(crc, reinterpret_cast<const Bytef*>(type), 4);
    if (len > 0 && data)
        crc = crc32(crc, data, len);
    write_u32_be(file_, (uint32_t)crc);
}
