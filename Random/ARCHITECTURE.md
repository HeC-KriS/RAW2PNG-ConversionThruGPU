# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

All dependencies are managed via vcpkg. Configure and build from the repo root:

```powershell
# Modern GPU (RTX 5050, SM >= 7.0) — requires CMake >= 3.24
cmake -B build
cmake --build build --config Release

# Legacy GPU (GT 710, SM 3.5, CUDA 11.8) — disables GPU deflate/CRC32/assembly
cmake -B build -DLEGACY_GPU=ON
cmake --build build --config Release
```

There is no test suite. Validation is done by comparing output PNGs against a reference decoder after running the binary.

```powershell
# Standalone deflate throughput benchmark (no input file needed)
.\build\Release\gpu_png_encoder.exe --bench-deflate 2400000
```

## Two Build Targets / Two Pipelines

This is the single most important thing to understand. The `LEGACY_GPU` CMake flag controls which source files are compiled, and the compiled binary selects a path at runtime via `detect_gpu_capability()`.

| | LEGACY_GPU=ON (GT 710, SM 3.5) | LEGACY_GPU=OFF (RTX 5050, SM >= 7.0) |
|---|---|---|
| GPU deflate | Not compiled (no symbol) | Compiled; opt-in via `--gpu-deflate` |
| GPU CRC32 / Adler32 | Not compiled | Compiled |
| GPU PNG assembly | Not compiled | Compiled |
| CMake arch | `35` | `native` |
| Preprocessor gate | `GPU_PNG_MODERN_DEFLATE` not defined | `GPU_PNG_MODERN_DEFLATE=1` defined |

`src/pipeline.cpp` is the only file that `#ifdef GPU_PNG_MODERN_DEFLATE` around the modern path — that preprocessor symbol must stay in sync with the SOURCES list in CMakeLists.txt.

## Pipeline Architecture

### Strip-Based Flow (Legacy and Modern)

Both pipelines process images in horizontal strips (`strip_height` rows at a time). This bounds VRAM usage for arbitrarily large images.

**Legacy pipeline** — 4-stage producer/consumer with three bounded queues:
```
Loader → [Queue A: StripJob] → GPU Filter → [Queue B: FilteredJob]
       → CPU Deflate (ThreadPool) → [Queue C: CompressedJob] → PNG Writer
```
These queues, jobs, and the `ThreadPool` are defined in `include/strip_job.h`, `include/bounded_queue.h`, and `include/thread_pool.h`. The whole pipeline runs inside `run_pipeline()` in `src/pipeline.cpp`.

**Modern pipeline** (`run_one_modern()` in `src/pipeline.cpp`) — GPU-resident, no CPU deflate:
```
Loader → GPU Filter (pool of N streams) → GPU Deflate → GPU CRC32/Adler32
       → GPU PNG Assembly → H2D copy → PNG file write
```
The modern path has a `GpuFilterContext` pool of size `cfg.gpu_streams` (default 4–8) to overlap H2D upload of the next strip with GPU processing of the current one.

### ImageSource Abstraction (`include/image_source.h`)

All format-specific loaders implement the `ImageSource` interface:
- `read_strip(buf, max_rows)` — returns rows read, 0 = EOF, <0 = error
- `dicom_params()` — returns non-null only for DICOM; drives the DICOM pixel preprocess kernel
- `num_frames()` — >1 for multi-frame DICOM

Implementations: `TiffSource`, `RawSource` (in `src/pipeline.cpp`) and `DicomSource` (in `src/dicom_loader.cpp` / `include/dicom_loader.h`). All format dispatch happens in `src/main.cpp` → `src/pipeline.cpp` and `src/batch_processor.cpp`.

### DICOM is isolated in its own static library

`dicom_lib` (just `src/dicom_loader.cpp`) links DCMTK and OpenJPEG as `PRIVATE`. This prevents DCMTK's `/Zc:__cplusplus` flag and its other compile options from leaking into the CUDA compilation units of the main executable.

## GPU Deflate Encoder (`src/gpu_deflate_backend.cu`)

Custom Fixed Huffman DEFLATE (RFC 1951 §3.2.6). **Not** a wrapper around nvCOMP (rejected: nvCOMP always emits `BFINAL=1` with no bit-precise end position, which breaks the concatenated-stream IDAT design).

Two sub-paths inside the GPU deflate encoder, selected at context-create time:

- **Literal-only** (`use_lz77=false`): every filtered byte → one Fixed Huffman literal code. Correct, fast, ~57 MB output for a 17043×11710 RGB-16 image.
- **LZ77 path** (`use_lz77=true`, `--gpu-lz77`): `lz77_row_kernel` runs one CUDA block per row. Thread 0 of each block does a sequential left-to-right scan using a `__shared__ uint32_t s_hash[4096]` (12-bit, 16 KB static — **not** dynamic; the 3rd kernel launch arg must be `0`). LZ4-style: one slot per bucket, most-recent position wins, which guarantees all candidates are back-references. Adds ~200 MB VRAM for the token arrays.

The prefix scan (kernel B) is a 3-pass multi-block scan (partial → block totals → offset add), removing the old 1024-row strip limit. Max strip height is ~1M rows.

### DEFLATE Correctness Invariant

**Never use `libdeflate` as the CPU deflate backend.** libdeflate always emits `BFINAL=1` and has no `Z_FULL_FLUSH` equivalent. The parallel CPU deflate path (`src/parallel_deflate.cpp`) splits each strip into N chunks compressed with raw DEFLATE (`windowBits = -15`), using `Z_FULL_FLUSH` on non-terminal chunks and `Z_FINISH` on the last. The chunks concatenate into a valid single DEFLATE bitstream. This invariant is what makes incremental strip-by-strip PNG IDAT generation correct.

## Known Compilation Issue

`src/batch_processor.cpp` calls `pipeline_reset_gpu_batch_stats()` (line 132) and `pipeline_print_gpu_batch_summary(...)` (line 232–233). Both are **declared** in `include/pipeline.h` but **not yet implemented** in `src/pipeline.cpp`. This causes a linker error in the current state. Either implement both functions in `pipeline.cpp` or remove the calls from `batch_processor.cpp` to unblock a build.

## Key CLI Flags

| Flag | Effect |
|---|---|
| `--gpu-deflate` | Opt in to GPU deflate encoder (modern GPU only, unverified on hardware) |
| `--gpu-lz77` | Enable LZ77 match finder inside GPU deflate (+200 MB VRAM) |
| `--gpu-lz77-debug` | With `--verbose`: print LZ77 match stats and sample back-references |
| `--batch-verbose` | Show per-file wall time and worker ID on each progress line |
| `--bench-deflate [N]` | Standalone CPU deflate benchmark, no input file |
| `--all` | Export every DICOM frame as `slice_NNNN.png` |
| `--gpu-streams N` | CUDA stream pool size for multi-frame DICOM and GPU-deflate filter pool |
