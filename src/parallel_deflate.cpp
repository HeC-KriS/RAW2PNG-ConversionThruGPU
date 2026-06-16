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
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include <future>

static const size_t MIN_CHUNK = 4096;

// ---------------------------------------------------------------------------
// Thread-local z_stream — eliminates per-chunk deflateInit2/deflateEnd.
//
// Background: each deflateInit2 allocates and zeroes ~256 KB of workspace
// (32 KB window + hash tables).  With 6 threads × 24 strips = 144 calls per
// image, that is ~36 MB of redundant zeroing.  Keeping one stream per worker
// thread and calling deflateReset() instead reuses the workspace without
// freeing/reallocating it.  zlib docs: "deflateReset is equivalent to
// deflateEnd followed by deflateInit, but does not free and reallocate the
// internal compression state."
// ---------------------------------------------------------------------------
struct ZStreamTLS {
    z_stream zs{};
    int      level  = -1;
    bool     inited = false;

    ~ZStreamTLS() { if (inited) deflateEnd(&zs); }

    // Return a ready-to-use z_stream.  Calls deflateReset() (fast path) when
    // the level matches; otherwise deflateEnd + deflateInit2 (level change).
    z_stream& acquire(int lvl) {
        if (inited && level == lvl) {
            deflateReset(&zs);
        } else {
            if (inited) deflateEnd(&zs);
            zs = z_stream{};
            deflateInit2(&zs, lvl, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY);
            level  = lvl;
            inited = true;
        }
        return zs;
    }
};
static thread_local ZStreamTLS tl_zs;

// ---------------------------------------------------------------------------
// Internal per-chunk work unit
// ---------------------------------------------------------------------------
struct Chunk {
    std::vector<uint8_t> deflate_data;
    unsigned long        adler_val    = 1;
    size_t               input_size   = 0;
    long long            init_us      = 0;  // time in acquire() (μs)
    long long            compress_us  = 0;  // time in deflate() (μs)
};

static void compress_chunk(
    const uint8_t* input, size_t input_size,
    int level, int flush_mode,
    Chunk& out)
{
    using Clock = std::chrono::high_resolution_clock;

    out.input_size = input_size;
    if (input_size == 0) {
        out.deflate_data.clear();
        out.adler_val = 1;
        return;
    }

    auto t0 = Clock::now();
    z_stream& zs = tl_zs.acquire(level);
    auto t1 = Clock::now();
    out.init_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    const uLong bound = deflateBound(&zs, (uLong)input_size) + 64;
    out.deflate_data.resize((size_t)bound);

    zs.next_in   = const_cast<Bytef*>(input);
    zs.avail_in  = (uInt)input_size;
    zs.next_out  = out.deflate_data.data();
    zs.avail_out = (uInt)bound;

    auto t2 = Clock::now();
    int ret = deflate(&zs, flush_mode);
    auto t3 = Clock::now();
    out.compress_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    // Z_STREAM_END for Z_FINISH; Z_OK / Z_BUF_ERROR acceptable for Z_FULL_FLUSH
    assert(ret == Z_STREAM_END || ret == Z_OK || ret == Z_BUF_ERROR);

    out.deflate_data.resize((size_t)zs.total_out);
    out.adler_val = adler32(1L, input, (uInt)input_size);
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
    result.init_us     = 0;
    result.compress_us = 0;

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

        result.init_us     += c.init_us;
        result.compress_us += c.compress_us;

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

unsigned long adler32_of(const uint8_t* data, size_t len)
{
    return adler32(1L, data, (uInt)len);
}

unsigned long crc32_of(const uint8_t* data, size_t len)
{
    return crc32(crc32(0L, nullptr, 0), data, (uInt)len);
}

// ---------------------------------------------------------------------------
// bench_deflate – standalone throughput benchmark
//
// Compresses strip_bytes of synthetic data at levels 0, 1, and 3 with
// num_threads parallel chunks, prints a formatted table to stdout.
//
// The synthetic pattern (Fibonacci hash of index) produces high-entropy data
// that is resistant to LZ77 matching — a worst case for compression speed.
// Real PNG-filtered medical images are more compressible; actual throughput
// for your data will be higher.  The ratio column shows real compression for
// the synthetic pattern; the ms/strip and Est-56MB columns scale linearly.
// ---------------------------------------------------------------------------
void bench_deflate(size_t strip_bytes, int num_threads)
{
    const int WARMUP = 3;
    const int ITERS  = 15;
    const int LEVELS[] = {0, 1, 3};

    // Synthetic strip: Fibonacci-hash per-byte pattern (high entropy).
    std::vector<uint8_t> input(strip_bytes);
    for (size_t i = 0; i < strip_bytes; i++)
        input[i] = (uint8_t)((i * 0x9E3779B9u) >> 24);

    ThreadPool pool(num_threads);

    printf("\nDeflate Benchmark\n");
    printf("  strip      : %zu bytes (%.2f MB)\n",
           strip_bytes, (double)strip_bytes / (1024.0 * 1024.0));
    printf("  threads    : %d chunks/strip\n", num_threads);
    printf("  iterations : %d (+ %d warmup)\n", ITERS, WARMUP);
#ifdef ZLIBNG_VERSION
    printf("  backend    : zlib-ng %s\n", ZLIBNG_VERSION);
#else
    printf("  backend    : zlib %s\n", ZLIB_VERSION);
#endif
    printf("\n");
    printf("  Lvl  ms/strip  MB/s (input)  Ratio   Init-sum   Est 56 MB\n");
    printf("  ---  --------  ------------  ------  ---------  ---------\n");

    for (int lvl : LEVELS) {
        ParallelDeflateConfig cfg;
        cfg.num_threads = num_threads;
        cfg.zlib_level  = lvl;

        // Warmup — populates thread-local z_streams in all worker threads
        for (int w = 0; w < WARMUP; w++)
            deflate_strip(input.data(), strip_bytes, cfg, pool, false);

        double total_wall_ms  = 0.0;
        double total_init_us  = 0.0;
        size_t last_comp_size = 0;

        for (int it = 0; it < ITERS; it++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            DeflateResult dr = deflate_strip(input.data(), strip_bytes, cfg, pool, false);
            auto t1 = std::chrono::high_resolution_clock::now();

            double ms = (double)std::chrono::duration_cast<std::chrono::microseconds>(
                t1 - t0).count() / 1000.0;
            total_wall_ms  += ms;
            total_init_us  += (double)dr.init_us;
            last_comp_size  = dr.data.size();
        }

        double mean_ms    = total_wall_ms / ITERS;
        double mbps       = (double)strip_bytes / (1024.0 * 1024.0) / (mean_ms / 1000.0);
        double ratio      = 100.0 * (double)last_comp_size / (double)strip_bytes;
        // init_us is summed across all parallel chunks; wall init = sum / threads
        double init_sum_us = total_init_us / ITERS;
        double est_56mb   = (56.0 * 1024.0 * 1024.0) /
                            (double)strip_bytes * mean_ms;

        printf("   %2d  %6.1f ms  %8.1f MB/s  %5.1f%%  %6.0f us  %7.0f ms\n",
               lvl, mean_ms, mbps, ratio, init_sum_us, est_56mb);
    }

    printf("  ---  --------  ------------  ------  ---------  ---------\n");
    printf("\n  Init-sum = acquire()/deflateReset() cost summed across %d parallel chunks.\n",
           num_threads);
    printf("  Wall init overhead = Init-sum / %d (chunks run in parallel).\n",
           num_threads);
    printf("\n  Current GPU wall time for this image: ~130 ms\n");
    printf("  To reduce deflate below 130 ms:\n");
    printf("    vcpkg install zlib-ng[core,zlib-compat]\n");
    printf("    cmake --fresh  (find_package(ZLIB) resolves to zlib-ng automatically)\n");
    printf("    Expected speedup: 1.8-2.5x  (level 1 target: ~60-80 ms)\n");
}
