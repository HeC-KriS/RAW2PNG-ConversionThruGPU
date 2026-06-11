#pragma once
#include "image_source.h"
#include <vector>
#include <cstdint>

// DICOM image source. Loads the entire pixel dataset into RAM on open(),
// then vends strips via memcpy — same design as RawReader.
//
// Supports only uncompressed transfer syntaxes (Implicit/Explicit LE).
// For compressed DICOMs, use a DICOM→uncompressed converter first.
struct DicomSource : ImageSource {
    ImageInfo            info_;
    DicomPixelParams     params_;
    std::vector<uint8_t> pixel_data_;  // raw bytes (little-endian for 16-bit)
    int                  next_row_ = 0;

    // Returns true on success. Prints error to stderr on failure.
    bool open(const char* path);

    const ImageInfo&          info()         const override { return info_;    }
    const DicomPixelParams*   dicom_params() const override { return &params_; }
    int read_strip(uint8_t* out, int max_rows) override;
};
