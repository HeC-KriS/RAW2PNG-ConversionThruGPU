#pragma once
// file_type.h
// Shared input-format detection, used by both the single-file CLI path
// (main.cpp) and the folder batch processor (batch_processor.cpp) so the
// two never drift out of sync.

bool ends_with_ci(const char* s, const char* suffix);

bool is_tiff_file(const char* path);
bool is_raw_file(const char* path);

// DICOM files are identified by one of:
//   1. A known DICOM file extension.
//   2. The 4-byte magic "DICM" at byte offset 128 (Part 10 files with preamble).
// Real-world DICOM rarely has a consistent extension (PACS systems use
// numeric filenames; CD-ROMs use no extension at all), so header sniffing is
// the only robust detection method.
bool is_dicom_file(const char* path);
