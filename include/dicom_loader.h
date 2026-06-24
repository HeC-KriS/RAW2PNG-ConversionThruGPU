#pragma once
#include "image_source.h"
#include <vector>
#include <cstdint>
#include <memory>

// Opaque implementation holding the live DCMTK DcmFileFormat and per-syntax
// decode state. Defined only in dicom_loader.cpp so translation units that
// include this header (e.g. pipeline.cpp, which is compiled into the CUDA
// target) never pull in DCMTK headers or its /Zc:__cplusplus interface flags.
struct DicomFileImpl;

// DICOM image source. open() parses headers and metadata (including the frame
// count) but does NOT decode pixels. Call load_frame(i) to materialise one
// frame's pixel bytes into pixel_data_, then read it via read_strip().
//
// Supports uncompressed (Implicit/Explicit VR LE), JPEG, JPEG-LS, RLE
// (via DCMTK) and JPEG2000 (via OpenJPEG). Multi-frame DICOM is supported for
// all of these.
struct DicomSource : ImageSource {
    ImageInfo            info_;
    DicomPixelParams     params_;
    std::vector<uint8_t> pixel_data_;   // current frame's raw bytes (LE for 16-bit)
    int                  next_row_   = 0;
    int                  num_frames_ = 1;
    int                  cur_frame_  = -1;  // which frame is in pixel_data_ (-1 = none)

    std::unique_ptr<DicomFileImpl> impl_;

    DicomSource();
    ~DicomSource() override;

    // Parse headers, transfer syntax, and NumberOfFrames. No pixel decode yet.
    bool open(const char* path);

    // Parse DICOM directly from memory buffer
    bool open_memory(const uint8_t* data, size_t size, const char* path = "<memory>");

    // Decode/extract frame `index` (0-based) into pixel_data_ and reset the
    // strip cursor. Returns false on error.
    bool load_frame(int index);

    const ImageInfo&        info()         const override { return info_;       }
    const DicomPixelParams* dicom_params() const override { return &params_;    }
    int                     num_frames()   const override { return num_frames_; }
    int read_strip(uint8_t* out, int max_rows) override;
    
    long long get_last_read_us() const;
};
