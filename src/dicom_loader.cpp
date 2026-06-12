// dicom_loader.cpp
// Loads a DICOM file (any transfer syntax) into RAM and vends pixel strips.
//
// Multi-frame:
//   open() parses headers and NumberOfFrames (0028,0008) but does not decode
//   pixels. load_frame(i) materialises one frame into pixel_data_. The live
//   DcmFileFormat is kept in DicomFileImpl so frames can be decoded on demand
//   without re-reading the file.
//
// JPEG2000 decode strategy (dcmjp2k is NOT used):
//   DCMTK parses the DICOM structure and extracts the raw J2K codestream bytes
//   from the encapsulated pixel sequence. OpenJPEG decodes those bytes directly.
//   This approach works regardless of whether DCMTK was compiled with OpenJPEG.
//
// Other compressed transfer syntaxes (JPEG, JPEG-LS, RLE):
//   DCMTK's registered codecs handle decompression via chooseRepresentation().
//
// Supported transfer syntaxes:
//   1.2.840.10008.1.2      Implicit VR LE (uncompressed)
//   1.2.840.10008.1.2.1    Explicit VR LE (uncompressed)
//   1.2.840.10008.1.2.4.50 JPEG Baseline
//   1.2.840.10008.1.2.4.57 JPEG Lossless, First-Order Prediction
//   1.2.840.10008.1.2.4.70 JPEG Lossless (default selection)
//   1.2.840.10008.1.2.4.80 JPEG-LS Lossless
//   1.2.840.10008.1.2.4.81 JPEG-LS Near-Lossless
//   1.2.840.10008.1.2.4.90 JPEG 2000 Lossless Only   ← OpenJPEG
//   1.2.840.10008.1.2.4.91 JPEG 2000                 ← OpenJPEG
//   1.2.840.10008.1.2.5    RLE Lossless

#include "dicom_loader.h"

#include <dcmtk/config/osconfig.h>

#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcxfer.h>

#include <dcmtk/dcmdata/dcpixel.h>
#include <dcmtk/dcmdata/dcpixseq.h>
#include <dcmtk/dcmdata/dcpxitem.h>

#include <dcmtk/dcmjpeg/djdecode.h>
#include <dcmtk/dcmjpls/djdecode.h>
#include <dcmtk/dcmdata/dcrledrg.h>

#include <openjpeg.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <memory>

// ---------------------------------------------------------------------------
// DCMTK codec registration (one-time, thread-safe)
// Only for the non-JPEG2000 codecs; JPEG2000 is handled by OpenJPEG directly.
// ---------------------------------------------------------------------------
static void register_dcmtk_codecs()
{
    DJDecoderRegistration::registerCodecs();     // JPEG (baseline/lossless)
    DJLSDecoderRegistration::registerCodecs();   // JPEG-LS
    DcmRLEDecoderRegistration::registerCodecs(); // RLE
}
static std::once_flag s_codecs_once;

// ---------------------------------------------------------------------------
// Transfer syntax predicates
// ---------------------------------------------------------------------------
static bool is_uncompressed_le(const OFString& uid)
{
    return uid == "1.2.840.10008.1.2"    // Implicit VR LE
        || uid == "1.2.840.10008.1.2.1"  // Explicit VR LE
        || uid.empty();                  // no meta header → default
}

static bool is_jpeg2000(const OFString& uid)
{
    return uid == "1.2.840.10008.1.2.4.90"   // JPEG 2000 Lossless Only
        || uid == "1.2.840.10008.1.2.4.91";  // JPEG 2000 (lossy/lossless)
}

// ---------------------------------------------------------------------------
// Opaque per-file decode state (keeps the DcmFileFormat alive across frames)
// ---------------------------------------------------------------------------
enum DicomKind { KIND_UNCOMPRESSED = 0, KIND_J2K = 1, KIND_OTHER = 2 };

struct DicomFileImpl {
    DcmFileFormat     file_format;
    DcmDataset*       ds            = nullptr;
    OFString          tsuid;
    int               kind          = KIND_UNCOMPRESSED;
    bool              decompressed  = false;  // KIND_OTHER: chooseRepresentation done
    DcmPixelSequence* pix_seq       = nullptr;// KIND_J2K: cached encapsulated sequence
};

// ---------------------------------------------------------------------------
// OpenJPEG memory-stream helpers (callback-based; works with any OPJ 2.x)
// ---------------------------------------------------------------------------
struct OjpMemStream {
    const uint8_t* data;
    OPJ_SIZE_T     size;
    OPJ_SIZE_T     pos;
};

static OPJ_SIZE_T ojp_read(void* buf, OPJ_SIZE_T nb, void* ud)
{
    auto* m = static_cast<OjpMemStream*>(ud);
    if (m->pos >= m->size) return static_cast<OPJ_SIZE_T>(-1);
    OPJ_SIZE_T avail = m->size - m->pos;
    OPJ_SIZE_T take  = (nb < avail) ? nb : avail;
    memcpy(buf, m->data + m->pos, take);
    m->pos += take;
    return take;
}

static OPJ_OFF_T ojp_skip(OPJ_OFF_T nb, void* ud)
{
    auto* m = static_cast<OjpMemStream*>(ud);
    if (nb < 0) return -1;
    OPJ_SIZE_T avail = m->size - m->pos;
    OPJ_SIZE_T take  = ((OPJ_SIZE_T)nb < avail) ? (OPJ_SIZE_T)nb : avail;
    m->pos += take;
    return (OPJ_OFF_T)take;
}

static OPJ_BOOL ojp_seek(OPJ_OFF_T offset, void* ud)
{
    auto* m = static_cast<OjpMemStream*>(ud);
    if (offset < 0 || (OPJ_SIZE_T)offset > m->size) return OPJ_FALSE;
    m->pos = (OPJ_SIZE_T)offset;
    return OPJ_TRUE;
}

static opj_stream_t* make_opj_stream(const uint8_t* data, OPJ_SIZE_T size)
{
    opj_stream_t* s = opj_stream_create(size, OPJ_TRUE);
    if (!s) return nullptr;
    auto* m = new OjpMemStream{data, size, 0};
    opj_stream_set_user_data(s, m,
        [](void* p){ delete static_cast<OjpMemStream*>(p); });
    opj_stream_set_user_data_length(s, size);
    opj_stream_set_read_function(s, ojp_read);
    opj_stream_set_skip_function(s, ojp_skip);
    opj_stream_set_seek_function(s, ojp_seek);
    return s;
}

// ---------------------------------------------------------------------------
// JPEG2000 decode
//
// Decodes one J2K codestream (or JP2 container) into a flat little-endian
// byte buffer identical in layout to what an uncompressed DICOM would contain.
// Components are interleaved: R G B R G B … (or Y Y Y … for grayscale).
//
// The GPU kernel (dicom_preprocess_kernel) processes this buffer exactly as it
// does for uncompressed DICOM — the DicomPixelParams (bits_stored, high_bit,
// pixel_rep, rescale, window) from the DICOM tags remain valid.
// ---------------------------------------------------------------------------
static bool decode_jpeg2000(
    const uint8_t*        j2k_data,
    size_t                j2k_size,
    int                   exp_rows,
    int                   exp_cols,
    int                   exp_comps,
    int                   bits_alloc,
    std::vector<uint8_t>& out_pixels)
{
    if (!j2k_data || j2k_size < 4) {
        fprintf(stderr, "DICOM J2K: codestream too short (%zu bytes)\n", j2k_size);
        return false;
    }

    // DICOM stores J2K codestream (SOC = 0xFF4F) or, rarely, JP2 file (starts with
    // 0x0000000C 6A502020 in the JP2 signature box).
    OPJ_CODEC_FORMAT fmt = OPJ_CODEC_J2K;
    if (j2k_data[0] == 0x00 && j2k_data[1] == 0x00 &&
        j2k_data[2] == 0x00 && j2k_data[3] == 0x0C)
        fmt = OPJ_CODEC_JP2;

    opj_codec_t* codec = opj_create_decompress(fmt);
    if (!codec) {
        fprintf(stderr, "DICOM J2K: opj_create_decompress failed\n");
        return false;
    }

    // Suppress OpenJPEG's own stderr output; we print our own errors.
    opj_set_error_handler(  codec,
        [](const char* m, void*){ fprintf(stderr, "OpenJPEG error: %s", m); }, nullptr);
    opj_set_warning_handler(codec,
        [](const char*, void*){}, nullptr);
    opj_set_info_handler(   codec,
        [](const char*, void*){}, nullptr);

    opj_dparameters_t params;
    opj_set_default_decoder_parameters(&params);
    if (!opj_setup_decoder(codec, &params)) {
        fprintf(stderr, "DICOM J2K: opj_setup_decoder failed\n");
        opj_destroy_codec(codec);
        return false;
    }

    opj_stream_t* stream = make_opj_stream(j2k_data, (OPJ_SIZE_T)j2k_size);
    if (!stream) {
        opj_destroy_codec(codec);
        return false;
    }

    opj_image_t* img = nullptr;
    bool ok = opj_read_header(stream, codec, &img) != 0;
    if (ok) ok = (opj_decode(codec, stream, img) != 0);
    if (ok) ok = (opj_end_decompress(codec, stream) != 0);

    opj_stream_destroy(stream);
    opj_destroy_codec(codec);

    if (!ok || !img) {
        if (img) opj_image_destroy(img);
        fprintf(stderr, "DICOM J2K: decode failed (J2K codestream malformed?)\n");
        return false;
    }

    // Validate against DICOM tag dimensions
    const int dec_w = (int)(img->x1 - img->x0);
    const int dec_h = (int)(img->y1 - img->y0);
    if (dec_w != exp_cols || dec_h != exp_rows || (int)img->numcomps != exp_comps) {
        fprintf(stderr,
            "DICOM J2K: decoded %dx%d (%u comps) does not match "
            "DICOM tags %dx%d (%d comps)\n",
            dec_w, dec_h, img->numcomps,
            exp_cols, exp_rows, exp_comps);
        opj_image_destroy(img);
        return false;
    }

    // All components must be full-resolution (no chroma subsampling)
    for (int c = 0; c < exp_comps; c++) {
        if ((int)img->comps[c].w != exp_cols ||
            (int)img->comps[c].h != exp_rows) {
            fprintf(stderr,
                "DICOM J2K: component %d has unexpected dimensions "
                "%ux%u (subsampling not supported)\n",
                c, img->comps[c].w, img->comps[c].h);
            opj_image_destroy(img);
            return false;
        }
    }

    // Convert OpenJPEG's int32 arrays to little-endian packed bytes.
    //
    // OpenJPEG returns values right-aligned in int32:
    //   unsigned prec=N : 0 … 2^N-1
    //   signed   prec=N : -2^(N-1) … 2^(N-1)-1
    //
    // Storing as LE uint16 (or uint8) preserves the two's-complement bit pattern
    // so the existing dicom_preprocess_kernel handles sign extension and masking
    // identically to what it does for uncompressed DICOM pixels.
    const int  bps = (bits_alloc == 16) ? 2 : 1;
    const size_t total = (size_t)exp_rows * exp_cols * exp_comps * bps;
    out_pixels.resize(total);

    for (int y = 0; y < exp_rows; y++) {
        for (int x = 0; x < exp_cols; x++) {
            const size_t pi = (size_t)y * exp_cols + x;
            for (int c = 0; c < exp_comps; c++) {
                const int32_t v = img->comps[c].data[pi];
                const size_t  d = (pi * exp_comps + c) * bps;
                if (bps == 2) {
                    // LE: LSB first
                    out_pixels[d]     = (uint8_t)( v        & 0xFF);
                    out_pixels[d + 1] = (uint8_t)((v >> 8)  & 0xFF);
                } else {
                    out_pixels[d] = (uint8_t)(v & 0xFF);
                }
            }
        }
    }

    opj_image_destroy(img);
    return true;
}

// ---------------------------------------------------------------------------
// DicomSource ctor/dtor (defined here where DicomFileImpl is complete)
// ---------------------------------------------------------------------------
DicomSource::DicomSource() : impl_(new DicomFileImpl()) {}
DicomSource::~DicomSource() = default;

// ---------------------------------------------------------------------------
// DicomSource::open — parse metadata + frame count; no pixel decode
// ---------------------------------------------------------------------------
bool DicomSource::open(const char* path)
{
    // Register DCMTK codecs for JPEG / JPEG-LS / RLE (not JPEG2000 — we bypass
    // DCMTK for that and call OpenJPEG directly).
    std::call_once(s_codecs_once, register_dcmtk_codecs);

    DicomFileImpl& F = *impl_;

    OFCondition status = F.file_format.loadFile(path);
    if (status.bad()) {
        fprintf(stderr, "DICOM: cannot load '%s': %s\n", path, status.text());
        return false;
    }

    if (F.file_format.getMetaInfo())
        F.file_format.getMetaInfo()->findAndGetOFString(DCM_TransferSyntaxUID, F.tsuid);

    F.ds = F.file_format.getDataset();
    if (!F.ds) {
        fprintf(stderr, "DICOM: no dataset in '%s'\n", path);
        return false;
    }
    DcmDataset* ds = F.ds;

    // ---- Read image geometry and pixel attribute tags ----
    Uint16 rows = 0, cols = 0;
    Uint16 bits_alloc = 16, bits_stored = 16, pixel_rep = 0, samples = 1;
    Uint16 high_bit = 0;

    ds->findAndGetUint16(DCM_Rows,                rows);
    ds->findAndGetUint16(DCM_Columns,             cols);
    ds->findAndGetUint16(DCM_BitsAllocated,       bits_alloc);
    ds->findAndGetUint16(DCM_BitsStored,          bits_stored);
    ds->findAndGetUint16(DCM_PixelRepresentation, pixel_rep);
    ds->findAndGetUint16(DCM_SamplesPerPixel,     samples);

    if (ds->findAndGetUint16(DCM_HighBit, high_bit).bad())
        high_bit = (Uint16)(bits_stored - 1);  // right-aligned default

    // ---- NumberOfFrames (0028,0008), VR = IS ----
    // Absent → single frame. Try integer accessor first, fall back to string.
    {
        Sint32 nf = 0;
        if (ds->findAndGetSint32(DCM_NumberOfFrames, nf).good() && nf > 0) {
            num_frames_ = (int)nf;
        } else {
            OFString nfs;
            if (ds->findAndGetOFString(DCM_NumberOfFrames, nfs).good() && !nfs.empty()) {
                int v = atoi(nfs.c_str());
                num_frames_ = (v > 0) ? v : 1;
            } else {
                num_frames_ = 1;
            }
        }
    }

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

    // Fill DicomPixelParams
    params_ = DicomPixelParams{};
    params_.bits_allocated = (int)bits_alloc;
    params_.bits_stored    = (int)bits_stored;
    params_.high_bit       = (int)high_bit;
    params_.pixel_rep      = (int)pixel_rep;

    Float64 slope = 1.0, intercept = 0.0;
    bool has_slope     = ds->findAndGetFloat64(DCM_RescaleSlope,     slope).good();
    bool has_intercept = ds->findAndGetFloat64(DCM_RescaleIntercept, intercept).good();
    if (has_slope || has_intercept) {
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

    // ---- Classify transfer syntax and cache decode handles ----
    if (is_jpeg2000(F.tsuid)) {
        F.kind = KIND_J2K;

        DcmElement* pixElem = nullptr;
        if (ds->findAndGetElement(DCM_PixelData, pixElem).bad() || !pixElem) {
            fprintf(stderr, "DICOM J2K: pixel data element not found in '%s'\n", path);
            return false;
        }
        DcmPixelData* pixData = dynamic_cast<DcmPixelData*>(pixElem);
        if (!pixData) {
            fprintf(stderr, "DICOM J2K: (7FE0,0010) is not a DcmPixelData in '%s'\n", path);
            return false;
        }

        const E_TransferSyntax xferEnum = (F.tsuid == "1.2.840.10008.1.2.4.90")
                                        ? EXS_JPEG2000LosslessOnly
                                        : EXS_JPEG2000;

        // getEncapsulatedRepresentation returns the stored DcmPixelSequence
        // without invoking a codec — works without dcmjp2k.
        const DcmRepresentationParameter* repParam = nullptr;
        if (pixData->getEncapsulatedRepresentation(xferEnum, repParam, F.pix_seq).bad()
            || !F.pix_seq)
        {
            fprintf(stderr,
                "DICOM J2K: cannot read encapsulated pixel sequence from '%s'\n"
                "           Transfer syntax: %s\n", path, F.tsuid.c_str());
            return false;
        }

        // Encapsulated layout: item 0 = Basic Offset Table, items 1..N = frames
        // (assumes one fragment per frame, the near-universal case).
        const unsigned long n_items = F.pix_seq->card();
        if (n_items < (unsigned long)num_frames_ + 1) {
            fprintf(stderr,
                "DICOM J2K: pixel sequence has %lu items but %d frame(s) expected.\n"
                "           Multi-fragment frames are not supported.\n",
                n_items, num_frames_);
            return false;
        }
    }
    else if (is_uncompressed_le(F.tsuid)) {
        F.kind = KIND_UNCOMPRESSED;
    }
    else {
        F.kind = KIND_OTHER;   // JPEG / JPEG-LS / RLE — decompressed lazily in load_frame
    }

    cur_frame_ = -1;
    next_row_  = 0;
    return true;
}

// ---------------------------------------------------------------------------
// DicomSource::load_frame — materialise one frame's pixels into pixel_data_
// ---------------------------------------------------------------------------
bool DicomSource::load_frame(int index)
{
    if (index < 0 || index >= num_frames_) {
        fprintf(stderr, "DICOM: frame index %d out of range [0,%d)\n",
                index, num_frames_);
        return false;
    }

    DicomFileImpl& F = *impl_;
    const size_t frame_bytes =
        (size_t)info_.height * info_.width * info_.channels *
        (size_t)(info_.bits_per_sample / 8);

    if (F.kind == KIND_J2K) {
        // Frame `index` lives in pixel-sequence item index+1 (item 0 = BOT)
        DcmPixelItem* item = nullptr;
        if (F.pix_seq->getItem(item, (unsigned long)(index + 1)).bad() || !item) {
            fprintf(stderr, "DICOM J2K: cannot get frame item %d\n", index);
            return false;
        }
        Uint8* j2kBytes = nullptr;
        if (item->getUint8Array(j2kBytes).bad() || !j2kBytes) {
            fprintf(stderr, "DICOM J2K: cannot read codestream for frame %d\n", index);
            return false;
        }
        const size_t j2kLen = (size_t)item->getLength();
        if (!decode_jpeg2000(j2kBytes, j2kLen,
                             (int)info_.height, (int)info_.width,
                             info_.channels, info_.bits_per_sample,
                             pixel_data_))
        {
            fprintf(stderr, "DICOM J2K: decode failed for frame %d\n", index);
            return false;
        }
    }
    else {
        // KIND_OTHER: decompress all frames once into the dataset (Explicit VR LE)
        if (F.kind == KIND_OTHER && !F.decompressed) {
            OFCondition decomp =
                F.ds->chooseRepresentation(EXS_LittleEndianExplicit, nullptr);
            if (decomp.bad()) {
                fprintf(stderr,
                    "DICOM: cannot decompress '%s'\n"
                    "       Transfer syntax : %s\n"
                    "       DCMTK error     : %s\n",
                    "(file)", F.tsuid.c_str(), decomp.text());
                return false;
            }
            if (!F.ds->canWriteXfer(EXS_LittleEndianExplicit)) {
                fprintf(stderr, "DICOM: decompressed representation unavailable\n");
                return false;
            }
            F.decompressed = true;
        }

        // Uncompressed (or now-decompressed): all frames concatenated in PixelData
        DcmElement* pixel_elem = nullptr;
        if (F.ds->findAndGetElement(DCM_PixelData, pixel_elem).bad() || !pixel_elem) {
            fprintf(stderr, "DICOM: pixel data element not found\n");
            return false;
        }
        Uint8* px_ptr = nullptr;
        if (pixel_elem->getUint8Array(px_ptr).bad() || !px_ptr) {
            fprintf(stderr, "DICOM: cannot access pixel bytes\n");
            return false;
        }
        const size_t total  = (size_t)pixel_elem->getLength();
        const size_t offset = (size_t)index * frame_bytes;
        if (offset + frame_bytes > total) {
            fprintf(stderr,
                "DICOM: frame %d out of range — need %zu bytes, pixel data has %zu\n",
                index, offset + frame_bytes, total);
            return false;
        }
        pixel_data_.assign(px_ptr + offset, px_ptr + offset + frame_bytes);
    }

    cur_frame_ = index;
    next_row_  = 0;
    return true;
}

// ---------------------------------------------------------------------------
// DicomSource::read_strip
// ---------------------------------------------------------------------------
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
