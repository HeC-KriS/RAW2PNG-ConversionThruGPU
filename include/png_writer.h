#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

// PNG color-type constants (subset used by this encoder).
enum class PngColorType : uint8_t {
    Gray  = 0,
    RGB   = 2,
    GrayA = 4,
    RGBA  = 6,
};

// Writes a PNG file manually without routing IDAT data through libpng's
// built-in zlib compressor.  Caller is responsible for supplying the raw
// DEFLATE bytes (zlib stream = header + deflate blocks + adler32 trailer).
//
// Usage:
//   PngWriter w;
//   w.open(path, width, height, 16, PngColorType::RGB);
//   w.begin_idat();
//   // Write zlib header first (2 bytes):
//   w.write_idat_bytes(header, 2);
//   // For each strip:
//   w.write_idat_bytes(deflate_data, deflate_size);
//   // After last strip write zlib adler32 trailer (4 bytes):
//   w.write_idat_bytes(trailer, 4);
//   w.end_idat();
//   w.close();
class PngWriter {
public:
    PngWriter() = default;
    ~PngWriter();

    bool open(const char* path, uint32_t width, uint32_t height,
              uint8_t bit_depth, PngColorType color_type);

    void begin_idat();

    // Write arbitrary bytes into the current IDAT chunk stream.
    // Internally splits into ≤32 KB PNG chunks.
    void write_idat_bytes(const uint8_t* data, size_t size);

    void end_idat();
    void close();

    bool is_open() const { return file_ != nullptr; }

private:
    void write_chunk(const char type[4], const uint8_t* data, uint32_t len);
    void flush_idat_buf();

    FILE*    file_      = nullptr;
    uint8_t  idat_buf_[32768];
    size_t   idat_pos_  = 0;
};
