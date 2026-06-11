// main.cpp
// CLI entry point.
// Usage:
//   gpu_png_encoder <input> <output.png> [options]
//
// Options:
//   --strip-height N   rows per GPU strip        (default 64)
//   --threads N        CPU DEFLATE threads        (default 6)
//   --level N          zlib compression level     (default 3, range 1-9)
//   --verbose          print per-stage timing with wall-clock timestamps

#include "pipeline.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>

static bool ends_with(const char* s, const char* suffix)
{
    const size_t sl = strlen(s), tl = strlen(suffix);
    if (sl < tl) return false;
    for (size_t i = 0; i < tl; i++)
        if (tolower((unsigned char)s[sl - tl + i]) != tolower((unsigned char)suffix[i]))
            return false;
    return true;
}

// DICOM files are identified by one of:
//   1. A known DICOM file extension.
//   2. The 4-byte magic "DICM" at byte offset 128 (Part 10 files with preamble).
//
// Real-world DICOM rarely has a consistent extension:
//   .dcm  .dicom  .dic  .ima  numbers-only names  no extension at all
// (PACS systems often use numeric filenames; CD-ROMs use no extension.)
// Header sniffing is the only robust detection method.
static bool has_dicom_extension(const char* path)
{
    return ends_with(path, ".dcm")
        || ends_with(path, ".dicom")
        || ends_with(path, ".dic")
        || ends_with(path, ".ima");   // Siemens
}

static bool sniff_dicom_magic(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    // Seek to byte 128 and check for "DICM" preamble (Part 10 DICOM)
    if (fseek(f, 128, SEEK_SET) != 0) { fclose(f); return false; }
    char magic[4] = {};
    size_t n = fread(magic, 1, 4, f);
    fclose(f);
    return n == 4
        && magic[0] == 'D' && magic[1] == 'I'
        && magic[2] == 'C' && magic[3] == 'M';
}

static bool is_dicom_file(const char* path)
{
    return has_dicom_extension(path) || sniff_dicom_magic(path);
}

static void usage(const char* prog)
{
    fprintf(stderr,
        "Usage: %s <input> <output.png> [options]\n"
        "  Input formats:\n"
        "    DICOM  : .dcm .dicom .dic .ima  (or any file with DICM magic at byte 128)\n"
        "    TIFF   : .tif .tiff\n"
        "    RAW    : .cr2 .nef .dng .arw .raf .orf .rw2 .raw\n"
        "  --strip-height N  rows per GPU strip  (default 64)\n"
        "  --threads N       DEFLATE threads      (default 6)\n"
        "  --level N         zlib level 1-9       (default 3)\n"
        "  --verbose         print timing report with wall-clock timestamps\n",
        prog);
}

int main(int argc, char* argv[])
{
    if (argc < 3) { usage(argv[0]); return 1; }

    const char* input  = argv[1];
    const char* output = argv[2];

    PipelineConfig cfg;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--strip-height") == 0 && i + 1 < argc)
            cfg.strip_height = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            cfg.deflate_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--level") == 0 && i + 1 < argc)
            cfg.deflate_level = atoi(argv[++i]);
        else if (strcmp(argv[i], "--verbose") == 0)
            cfg.verbose = true;
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    // Clamp level
    if (cfg.deflate_level < 1) cfg.deflate_level = 1;
    if (cfg.deflate_level > 9) cfg.deflate_level = 9;
    if (cfg.deflate_threads < 1) cfg.deflate_threads = 1;
    if (cfg.strip_height    < 1) cfg.strip_height    = 1;

    // Dispatch: try DICOM first (header sniff covers files with no/unknown extension),
    // then TIFF, then RAW camera.
    bool ok = false;
    if (ends_with(input, ".tif") || ends_with(input, ".tiff")) {
        ok = encode_tiff_to_png(input, output, cfg);
    } else if (ends_with(input, ".cr2") || ends_with(input, ".nef") ||
               ends_with(input, ".dng") || ends_with(input, ".arw") ||
               ends_with(input, ".raw") || ends_with(input, ".raf") ||
               ends_with(input, ".orf") || ends_with(input, ".rw2")) {
        ok = encode_raw_to_png(input, output, cfg);
    } else if (is_dicom_file(input)) {
        // Covers: .dcm .dicom .dic .ima  +  numeric names  +  no-extension DICOM
        // The sniff_dicom_magic check reads 132 bytes and verifies the "DICM"
        // preamble — this is the only reliable way to detect DICOM without an
        // extension (very common on PACS systems and DICOM CDs).
        ok = encode_dicom_to_png(input, output, cfg);
    } else {
        // Unknown extension: try TIFF first (most common large-image format),
        // then RAW camera, then DICOM without preamble (rare legacy ACR-NEMA).
        ok = encode_tiff_to_png(input, output, cfg);
        if (!ok) ok = encode_raw_to_png(input, output, cfg);
        if (!ok) ok = encode_dicom_to_png(input, output, cfg);
    }

    if (!ok) {
        fprintf(stderr, "Encoding failed.\n");
        return 1;
    }

    if (cfg.verbose)
        fprintf(stdout, "Done → %s\n", output);
    return 0;
}
