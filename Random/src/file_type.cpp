// file_type.cpp
#include "file_type.h"

#include <cstdio>
#include <cstring>
#include <cctype>

bool ends_with_ci(const char* s, const char* suffix)
{
    const size_t sl = strlen(s), tl = strlen(suffix);
    if (sl < tl) return false;
    for (size_t i = 0; i < tl; i++)
        if (tolower((unsigned char)s[sl - tl + i]) != tolower((unsigned char)suffix[i]))
            return false;
    return true;
}

bool is_tiff_file(const char* path)
{
    return ends_with_ci(path, ".tif") || ends_with_ci(path, ".tiff");
}

bool is_raw_file(const char* path)
{
    return ends_with_ci(path, ".cr2") || ends_with_ci(path, ".nef") ||
           ends_with_ci(path, ".dng") || ends_with_ci(path, ".arw") ||
           ends_with_ci(path, ".raw") || ends_with_ci(path, ".raf") ||
           ends_with_ci(path, ".orf") || ends_with_ci(path, ".rw2");
}

static bool has_dicom_extension(const char* path)
{
    return ends_with_ci(path, ".dcm")
        || ends_with_ci(path, ".dicom")
        || ends_with_ci(path, ".dic")
        || ends_with_ci(path, ".ima");   // Siemens
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

bool is_dicom_file(const char* path)
{
    return has_dicom_extension(path) || sniff_dicom_magic(path);
}
