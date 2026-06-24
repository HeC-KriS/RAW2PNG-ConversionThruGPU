#include "jls_encoder.h"
#include "charls_api.h"
#include <cstring>

size_t jls_encode_gray8(const uint8_t* pixels, int width, int height,
                        uint8_t** out_buf)
{
    JlsParameters p;
    memset(&p, 0, sizeof(p));
    p.width             = width;
    p.height            = height;
    p.bitspersample     = 8;
    p.bytesperline      = width;
    p.components        = 1;
    p.allowedlossyerror = 0;       // lossless
    p.ilv               = ILV_NONE;

    const size_t src_size = (size_t)width * height;
    size_t buf_size = src_size + 1024;   // CharLS reallocates via new[] if too small
    BYTE* buf = new BYTE[buf_size];
    size_t written = 0;

    const JLS_ERROR err = JpegLsEncode(&buf, &buf_size, &written,
                                        pixels, src_size, &p);
    if (err != JLS_OK) {
        delete[] buf;
        *out_buf = nullptr;
        return 0;
    }
    *out_buf = reinterpret_cast<uint8_t*>(buf);
    return written;
}
