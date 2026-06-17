#pragma once
#include "image_loader.h"
#include "dicom_params.h"

// Abstract strip-based image source used by run_pipeline.
// Implementations: TiffSource / RawSource (pipeline.cpp), DicomSource (dicom_loader.h).
struct ImageSource {
    virtual ~ImageSource() = default;
    virtual const ImageInfo& info() const = 0;
    // Read up to max_rows rows into out. Returns actual rows read; 0 = EOF; <0 = error.
    virtual int read_strip(uint8_t* out, int max_rows) = 0;
    // Returns non-null only for DICOM sources; drives the GPU preprocessing kernel.
    virtual const DicomPixelParams* dicom_params() const { return nullptr; }
    // Number of frames in the source. 1 for single-image formats (TIFF/RAW);
    // DICOM may report more via NumberOfFrames (0028,0008).
    virtual int num_frames() const { return 1; }
};
