// main.cpp
// CLI entry point.
// Usage:
//   gpu_png_encoder <input> <output.png> [options]
//
// Options:
//   --strip-height N   rows per GPU strip        (default 64)
//   --threads N        CPU DEFLATE threads        (default 6)
//   --level N          zlib compression level     (default 3, range 1-9)
//   --verbose          print per-stage timing

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

static void usage(const char* prog)
{
    fprintf(stderr,
        "Usage: %s <input> <output.png> [options]\n"
        "  --strip-height N  rows per GPU strip  (default 64)\n"
        "  --threads N       DEFLATE threads      (default 6)\n"
        "  --level N         zlib level 1-9       (default 3)\n"
        "  --verbose\n",
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

    // Dispatch by extension
    bool ok = false;
    if (ends_with(input, ".tif") || ends_with(input, ".tiff"))
        ok = encode_tiff_to_png(input, output, cfg);
    else if (ends_with(input, ".cr2") || ends_with(input, ".nef") ||
             ends_with(input, ".dng") || ends_with(input, ".arw") ||
             ends_with(input, ".raw") || ends_with(input, ".raf") ||
             ends_with(input, ".orf") || ends_with(input, ".rw2"))
        ok = encode_raw_to_png(input, output, cfg);
    else {
        // Try TIFF first, fall back to RAW
        ok = encode_tiff_to_png(input, output, cfg);
        if (!ok) ok = encode_raw_to_png(input, output, cfg);
    }

    if (!ok) {
        fprintf(stderr, "Encoding failed.\n");
        return 1;
    }

    if (cfg.verbose)
        fprintf(stdout, "Done → %s\n", output);
    return 0;
}
