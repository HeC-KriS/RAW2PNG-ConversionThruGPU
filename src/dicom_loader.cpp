// dicom_loader.cpp
// Loads an uncompressed DICOM file into RAM and vends strips.
//
// Supports Implicit VR Little Endian (1.2.840.10008.1.2) and
// Explicit VR Little Endian (1.2.840.10008.1.2.1), which covers the
// overwhelming majority of DICOM files encountered in practice.
// Explicit VR Big Endian and all compressed transfer syntaxes are rejected
// with a clear error message.
//
// The raw pixel bytes (little-endian for 16-bit) are stored in pixel_data_.
// The GPU preprocessing kernel (dicom_preprocess_kernel in gpu_filter.cu)
// handles bit-depth normalisation, sign correction, rescale, and window/level
// on the fly during strip processing, so no CPU-side transforms are done here.

#include "dicom_loader.h"

#include <dcmtk/config/osconfig.h>
#include <dcmtk/dcmdata/dctk.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

// Transfer syntax UIDs we accept (uncompressed little-endian only)
static bool is_uncompressed_le(const OFString& tsuid)
{
    return tsuid == "1.2.840.10008.1.2"    // Implicit VR LE (default)
        || tsuid == "1.2.840.10008.1.2.1"  // Explicit VR LE
        || tsuid.empty();                  // no meta header → assume default
}

bool DicomSource::open(const char* path)
{
    DcmFileFormat file_format;
    OFCondition status = file_format.loadFile(path);
    if (status.bad()) {
        fprintf(stderr, "DICOM: cannot load '%s': %s\n", path, status.text());
        return false;
    }

    // Check transfer syntax before touching pixel data
    OFString tsuid;
    if (file_format.getMetaInfo())
        file_format.getMetaInfo()->findAndGetOFString(DCM_TransferSyntaxUID, tsuid);

    if (!is_uncompressed_le(tsuid)) {
        fprintf(stderr,
            "DICOM: unsupported transfer syntax '%s' in '%s'\n"
            "       Only uncompressed little-endian (1.2.840.10008.1.2[.1]) is supported.\n"
            "       Convert with dcmdjpeg / gdcmconv first.\n",
            tsuid.c_str(), path);
        return false;
    }

    DcmDataset* ds = file_format.getDataset();
    if (!ds) {
        fprintf(stderr, "DICOM: no dataset in '%s'\n", path);
        return false;
    }

    // ---- Image geometry tags ----
    Uint16 rows = 0, cols = 0;
    Uint16 bits_alloc = 16, bits_stored = 16, pixel_rep = 0, samples = 1;
    Uint16 high_bit = 0;

    ds->findAndGetUint16(DCM_Rows,                rows);
    ds->findAndGetUint16(DCM_Columns,             cols);
    ds->findAndGetUint16(DCM_BitsAllocated,       bits_alloc);
    ds->findAndGetUint16(DCM_BitsStored,          bits_stored);
    ds->findAndGetUint16(DCM_PixelRepresentation, pixel_rep);
    ds->findAndGetUint16(DCM_SamplesPerPixel,     samples);

    // HighBit (0028,0102) determines pixel bit alignment within the allocated word.
    // Most scanners are right-aligned (HighBit = BitsStored - 1).
    // Default to right-aligned so absent tags don't corrupt values.
    if (ds->findAndGetUint16(DCM_HighBit, high_bit).bad())
        high_bit = (Uint16)(bits_stored - 1);  // safe default: right-aligned

    if (rows == 0 || cols == 0) {
        fprintf(stderr, "DICOM: invalid dimensions %ux%u in '%s'\n",
                (unsigned)cols, (unsigned)rows, path);
        return false;
    }
    if (bits_alloc != 8 && bits_alloc != 16) {
        fprintf(stderr, "DICOM: unsupported BitsAllocated=%u in '%s'\n",
                (unsigned)bits_alloc, path);
        return false;
    }
    if (samples < 1 || samples > 4) {
        fprintf(stderr, "DICOM: unsupported SamplesPerPixel=%u in '%s'\n",
                (unsigned)samples, path);
        return false;
    }

    // Fill ImageInfo
    info_.width           = (uint32_t)cols;
    info_.height          = (uint32_t)rows;
    info_.channels        = (int)samples;
    info_.bits_per_sample = (int)bits_alloc;
    info_.bpp             = (int)samples * (int)(bits_alloc / 8);

    // ---- DicomPixelParams ----
    params_ = DicomPixelParams{};
    params_.bits_allocated = (int)bits_alloc;
    params_.bits_stored    = (int)bits_stored;
    params_.high_bit       = (int)high_bit;   // must be passed to GPU kernel
    params_.pixel_rep      = (int)pixel_rep;

    Float64 slope = 1.0, intercept = 0.0;
    bool has_slope     = ds->findAndGetFloat64(DCM_RescaleSlope,     slope).good();
    bool has_intercept = ds->findAndGetFloat64(DCM_RescaleIntercept, intercept).good();
    if (has_slope || has_intercept) {
        // Only activate if at least one value differs from identity
        if (std::fabs(slope - 1.0) > 1e-9 || std::fabs(intercept) > 1e-9) {
            params_.apply_rescale     = 1;
            params_.rescale_slope     = (float)slope;
            params_.rescale_intercept = (float)intercept;
        }
    }

    Float64 wc = 0.0, ww = 0.0;
    if (ds->findAndGetFloat64(DCM_WindowCenter, wc).good() &&
        ds->findAndGetFloat64(DCM_WindowWidth,  ww).good() && ww > 0.0)
    {
        params_.apply_window  = 1;
        params_.window_center = (float)wc;
        params_.window_width  = (float)ww;
    }

    // ---- Extract raw pixel bytes ----
    DcmElement* pixel_elem = nullptr;
    if (ds->findAndGetElement(DCM_PixelData, pixel_elem).bad() || !pixel_elem) {
        fprintf(stderr, "DICOM: pixel data element not found in '%s'\n", path);
        return false;
    }
    Uint8* px_ptr = nullptr;
    if (pixel_elem->getUint8Array(px_ptr).bad() || !px_ptr) {
        fprintf(stderr, "DICOM: cannot access pixel bytes in '%s'\n", path);
        return false;
    }
    const size_t expected =
        (size_t)rows * cols * samples * (bits_alloc / 8u);
    const unsigned long px_len = pixel_elem->getLength();
    if ((size_t)px_len < expected) {
        fprintf(stderr,
            "DICOM: pixel data too short — got %lu bytes, expected %zu in '%s'\n",
            (unsigned long)px_len, expected, path);
        return false;
    }

    pixel_data_.assign(px_ptr, px_ptr + expected);
    next_row_ = 0;
    return true;
}

int DicomSource::read_strip(uint8_t* out, int max_rows)
{
    const int h = (int)info_.height;
    if (next_row_ >= h) return 0;
    int rows = std::min(max_rows, h - next_row_);
    const size_t row_bytes = (size_t)info_.width * info_.bpp;
    std::memcpy(out,
                pixel_data_.data() + (size_t)next_row_ * row_bytes,
                (size_t)rows * row_bytes);
    next_row_ += rows;
    return rows;
}
