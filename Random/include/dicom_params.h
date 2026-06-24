#pragma once

// DICOM pixel transform parameters passed from DicomSource to the GPU kernel.
// All non-DICOM sources return nullptr from ImageSource::dicom_params();
// a nullptr pointer causes the GPU preprocessing kernel to be skipped.
struct DicomPixelParams {
    int   bits_allocated    = 16;  // BitsAllocated tag (0028,0100)
    int   bits_stored       = 16;  // BitsStored tag    (0028,0101)
    // HighBit (0028,0102): position of the MSB of the stored value.
    // Right-aligned (most DICOM): high_bit = bits_stored - 1  → shift = 0.
    // Left-aligned  (rare):       high_bit = bits_allocated-1 → shift = bits_allocated-bits_stored.
    // Correct shift to right-align: high_bit - bits_stored + 1.
    int   high_bit          = 15;  // default to left-aligned (safe fallback, overwritten from file)
    int   pixel_rep         = 0;   // PixelRepresentation (0028,0103): 0=unsigned, 1=signed
    int   apply_rescale     = 0;   // 1 if RescaleSlope/Intercept are non-trivial
    float rescale_slope     = 1.f; // (0028,1053) m:  display = stored * m + b
    float rescale_intercept = 0.f; // (0028,1052) b
    int   apply_window      = 0;   // 1 if WindowCenter/Width are present
    float window_center     = 32768.f; // (0028,1050)
    float window_width      = 65536.f; // (0028,1051)
};
