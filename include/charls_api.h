#pragma once
// Raw CharLS v1.0 C API — matches dcmjpls/libcharls/pubtypes.h + intrface.h
// as bundled in DCMTK 3.7.0.  Link against dcmtkcharls.lib / dcmtkcharls.dll.

#include <cstddef>

typedef unsigned char BYTE;

enum JLS_ERROR {
    JLS_OK                          = 0,
    JLS_InvalidJlsParameters        = 1,
    JLS_ParameterValueNotSupported  = 2,
    JLS_UncompressedBufferTooSmall  = 3,
    JLS_CompressedBufferTooSmall    = 4,
    JLS_InvalidCompressedData       = 5,
    JLS_TooMuchCompressedData       = 6,
    JLS_ImageTypeNotSupported       = 7,
    JLS_UnsupportedBitDepthForTransform = 8,
    JLS_UnsupportedColorTransform   = 9,
    JLS_MemoryAllocationFailure     = 10
};

enum charls_interleave { ILV_NONE = 0, ILV_LINE = 1, ILV_SAMPLE = 2 };

struct JlsCustomParameters { int MAXVAL, T1, T2, T3, RESET; };
struct JfifParameters {
    int   Ver;
    char  units;
    int   XDensity, YDensity;
    short Xthumb, Ythumb;
    void* pdataThumbnail;
};

struct JlsParameters {
    int               width;
    int               height;
    int               bitspersample;
    int               bytesperline;
    int               components;
    int               allowedlossyerror;
    charls_interleave ilv;
    int               colorTransform;
    char              outputBgr;
    JlsCustomParameters custom;
    JfifParameters      jfif;
};

extern "C" {
    // dst: pre-allocated buffer; CharLS reallocates via new[] if too small.
    // Caller must free *dst with delete[] after use.
    JLS_ERROR JpegLsEncode(BYTE**       dst,
                           size_t*      dstLen,
                           size_t*      bytesWritten,
                           const void*  src,
                           size_t       srcLen,
                           JlsParameters* params);

    JLS_ERROR JpegLsDecode(void*         dst,
                           size_t        dstLen,
                           const void*   src,
                           size_t        srcLen,
                           JlsParameters* params);
}
