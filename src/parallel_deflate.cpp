// parallel_deflate.cpp
// Uses a persistent ThreadPool to compress strip chunks in parallel.
// Each chunk uses raw DEFLATE (windowBits = -15) and outputs byte-aligned
// blocks.  Non-terminal chunks end with Z_FULL_FLUSH (BFINAL=0); the absolute
// last chunk ends with Z_FINISH (BFINAL=1).
//
// CORRECTNESS INVARIANT: only the very last chunk of the very last strip may
// have BFINAL=1.  Every other chunk must end with Z_FULL_FLUSH so the
// concatenated DEFLATE stream contains exactly one terminal block and a PNG
// decoder does not stop prematurely.  libdeflate is NOT suitable here because
// it always outputs BFINAL=1 and has no Z_FULL_FLUSH equivalent.
//
// Adler-32 is computed per chunk and combined with adler32_combine() so that
// the caller can append the correct zlib trailer after all strips.

#include "parallel_deflate.h"

#include <zlib.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>
#include <future>

// Smallest chunk we'll bother creating a separate task for.
static const size_t MIN_CHUNK = 4096;

// ---------------------------------------------------------------------------
// Internal per-chunk work unit
// ---------------------------------------------------------------------------
struct Chunk {
    std::vector<uint8_t> deflate_data;
    unsigned long        adler_val   = 1;
    size_t               input_size  = 0;
};

static void compress_chunk(
    const uint8_t* input, size_t input_size,
    int level, int flush_mode,
    Chunk& out)
{
    out.input_size = input_size;
    if (input_size == 0) {
        out.deflate_data.clear();
        out.adler_val = 1;
        return;
    }

    z_stream zs;
    zs.zalloc = Z_NULL;
    zs.zfree  = Z_NULL;
    zs.opaque = Z_NULL;

    // windowBits = -15  →  raw DEFLATE, no zlib envelope
    int ret = deflateInit2(&zs, level, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY);
    assert(ret == Z_OK);

    const uLong bound = deflateBound(&zs, (uLong)input_size) + 64;
    out.deflate_data.resize((size_t)bound);

    zs.next_in   = const_cast<Bytef*>(input);
    zs.avail_in  = (uInt)input_size;
    zs.next_out  = out.deflate_data.data();
    zs.avail_out = (uInt)bound;

    ret = deflate(&zs, flush_mode);
    // Z_STREAM_END for Z_FINISH; Z_OK / Z_BUF_ERROR acceptable for Z_FULL_FLUSH
    assert(ret == Z_STREAM_END || ret == Z_OK || ret == Z_BUF_ERROR);

    out.deflate_data.resize((size_t)zs.total_out);
    out.adler_val = adler32(1L, input, (uInt)input_size);

    deflateEnd(&zs);
}

// ---------------------------------------------------------------------------
// Public: deflate_strip
// ---------------------------------------------------------------------------
DeflateResult deflate_strip(
    const uint8_t*               input,
    size_t                       input_size,
    const ParallelDeflateConfig& cfg,
    ThreadPool&                  pool,
    bool                         is_last)
{
    DeflateResult result;
    result.input_size  = input_size;
    result.strip_adler = 1;

    if (input_size == 0) return result;

    // How many threads to actually use (avoid tiny chunks)
    int nt = cfg.num_threads;
    if ((size_t)nt > input_size / MIN_CHUNK)
        nt = (int)std::max<size_t>(1, input_size / MIN_CHUNK);
    nt = std::max(1, nt);

    const size_t chunk_size = (input_size + (size_t)nt - 1) / (size_t)nt;

    // Find the index of the last non-empty chunk (to apply Z_FINISH there)
    int last_active = 0;
    for (int i = nt - 1; i >= 0; i--) {
        if ((size_t)i * chunk_size < input_size) { last_active = i; break; }
    }

    // Submit chunk tasks to the persistent pool
    std::vector<std::future<Chunk>> futs;
    futs.reserve(nt);

    for (int i = 0; i < nt; i++) {
        const size_t start = (size_t)i * chunk_size;
        if (start >= input_size) continue;

        const size_t end  = std::min(start + chunk_size, input_size);
        const bool   last = is_last && (i == last_active);
        const int    mode = last ? Z_FINISH : Z_FULL_FLUSH;
        const int    lvl  = cfg.zlib_level;

        futs.push_back(pool.submit([start, end, input, lvl, mode]() -> Chunk {
            Chunk c;
            compress_chunk(input + start, end - start, lvl, mode, c);
            return c;
        }));
    }

    // Collect results in submission order; combine Adler-32 as we go
    unsigned long running = 1;
    bool          started = false;

    result.data.reserve(input_size);  // generous pre-allocation

    for (auto& fut : futs) {
        Chunk c = fut.get();
        if (c.input_size == 0) continue;

        if (!started) {
            running = c.adler_val;
            started = true;
        } else {
            running = adler32_combine(running, c.adler_val, (z_off_t)c.input_size);
        }

        result.data.insert(result.data.end(),
                           c.deflate_data.begin(), c.deflate_data.end());
    }

    result.strip_adler = started ? running : 1UL;
    return result;
}

// ---------------------------------------------------------------------------
// accum_adler – call in strip order from the writer stage
// ---------------------------------------------------------------------------
void accum_adler(ParallelDeflateState& state, const DeflateResult& dr)
{
    if (dr.input_size == 0) return;
    if (!state.started) {
        state.running_adler = dr.strip_adler;
        state.started       = true;
    } else {
        state.running_adler = adler32_combine(
            state.running_adler,
            dr.strip_adler,
            (z_off_t)dr.input_size);
    }
}

// ---------------------------------------------------------------------------
// zlib stream envelope helpers
// ---------------------------------------------------------------------------
void zlib_header(int level, uint8_t out[2])
{
    // CMF = 0x78: CM=8 (deflate), CINFO=7 (32 KB window)
    // FLG chosen so (CMF*256+FLG)%31==0 and FDICT bit clear.
    const uint8_t CMF = 0x78;
    uint8_t FLG;
    if      (level == 1)               FLG = 0x01;  // FLEVEL=0 (fastest)
    else if (level >= 2 && level <= 5) FLG = 0x5E;  // FLEVEL=1 (fast)
    else if (level >= 6 && level <= 7) FLG = 0x9C;  // FLEVEL=2 (default)
    else                               FLG = 0xDA;  // FLEVEL=3 (maximum)

    assert(((uint32_t)CMF * 256 + FLG) % 31 == 0);
    out[0] = CMF;
    out[1] = FLG;
}

void zlib_trailer(const ParallelDeflateState& state, uint8_t out[4])
{
    const unsigned long a = state.running_adler;
    out[0] = (uint8_t)((a >> 24) & 0xFF);
    out[1] = (uint8_t)((a >> 16) & 0xFF);
    out[2] = (uint8_t)((a >>  8) & 0xFF);
    out[3] = (uint8_t)( a        & 0xFF);
}
