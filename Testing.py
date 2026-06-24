import subprocess

# ==========================================================
# SCENARIO — pick exactly one, comment out the rest
# ==========================================================
#
# SCENARIO 1: BMP CPU  (fastest write, CPU LUT, uncompressed reference)
#   Throughput: ~887 fps (4 workers)   Size: 100% of raw pixels
#
# SCENARIO 2: BMP GPU  (GPU window/level + 16→8 kernel, CPU BMP write)
#   Throughput: ~855 fps (4 workers, -4% vs CPU at 4W; +3% vs CPU at 1W)
#   Benefit: frees encode thread from LUT; scales better with large images.
#   Requires: --gpu-bmp flag. Best for mammograms (>> 512×512).
#
# SCENARIO 3: JPEG-LS — CPU CharLS  (best throughput + best compression)
#   Throughput: ~887 fps (4 workers)   Size: ~17% of raw (6× compression)
#   Standard lossless JPEG-LS via DCMTK CharLS. Bit-exact reconstruction.
#
# SCENARIO 4: JPEG-LS — GPU pipeline (G1–G5 CUDA encoder)
#   Throughput: ~530 fps (1 worker, +30% vs CPU) / ~820 fps (6 workers)
#   Size: ~40% of raw (2.6× compression) — larger than CPU due to row-context reset
#   Requires: --gpu-jls flag. Batch mode only.
#
# SCENARIO 5: PNG — CPU deflate  (baseline PNG, no GPU)
#   Throughput: ~474 fps (4 workers)   Size: ~33% of raw
#
# SCENARIO 6: PNG — GPU deflate + GPU LZ77  (best PNG throughput)
#   Throughput: ~505 fps (4 workers)   Size: ~29% of raw (75% LZ77 coverage)
#   Requires modern GPU (RTX / SM >= 7.0). Adds ~200 MB VRAM for LZ77 tables.
#
# SCENARIO 7: Decode-only benchmark  (no output written)
#   Measures raw DICOM ingest ceiling (DCMTK + LibRaw + libtiff).
#   Useful for isolating whether the bottleneck is decode vs encode.
#
SCENARIO = 2  # <- change this number to switch scenarios

# ==========================================================
# PATHS
# ==========================================================

EXE         = r"D:\Projects\gpu-optimize\build\gpu_png_encoder.exe"
INPUT_PATH  = r"C:\Users\Krish\Downloads\CT_ABDPELVIS"   # input folder (DICOM)
OUTPUT_PATH = r"C:\Windows\Temp\testing_out"               # output folder

# ==========================================================
# WORKER COUNT
# Arch D decode-ahead pipeline: N decode threads + N encode threads running concurrently.
# Sweet spot: 4–6 for this CT dataset (decode-limited above 6).
# Use 1 to isolate single-file GPU performance without decode overlap.
# ==========================================================
BATCH_WORKERS = 4

# ==========================================================
# VERBOSITY
# BATCH_VERBOSE: print per-file timing on each progress line
# VERBOSE:       print per-stage GPU/CPU timing for single-file mode
# ==========================================================
BATCH_VERBOSE = False
VERBOSE       = False

# ==========================================================
# DICOM FRAME SELECTION (single-file mode only)
# EXPORT_ALL:   write every DICOM frame to its own file (--all)
# FRAME_NUMBER: export only frame N, 0-based (--frame N)
# ==========================================================
EXPORT_ALL   = False
FRAME_NUMBER = None     # e.g. 0 for first frame

# ==========================================================
# DECODE-ONLY benchmark (overrides scenario; no files written)
# ==========================================================
BENCHMARK_DECODE = False


# ----------------------------------------------------------
# Scenario definitions — do not edit below this line
# ----------------------------------------------------------

_scenarios = {
    1: {
        # Uncompressed BMP, CPU path. Fastest path overall, no compression work.
        # CPU LUT (SSE2): ~0.08 ms/file for 512×512. Use as throughput ceiling.
        "format":      "bmp",
        "strip_height": None,
        "threads":      None,
        "level":        None,
        "gpu_deflate":  False,
        "gpu_lz77":     False,
        "gpu_jls":      False,
        "gpu_bmp":      False,
        "gpu_streams":  None,
        "batch_size":   None,
    },
    2: {
        # Uncompressed BMP, GPU window/level path.
        # bmp_dicom_to_gray8_kernel: H2D -> W/L -> 16->8 -> D2H -> CPU fwrite.
        # GPU overhead ~0.50 ms/file vs CPU LUT ~0.08 ms/file; CPU write unchanged.
        # Wins at 1–2 workers; CPU is faster at 4+ workers for 512×512 CT.
        # Better choice for large images (mammograms) where GPU W/L saves ~6 ms/file.
        "format":      "bmp",
        "strip_height": None,
        "threads":      None,
        "level":        None,
        "gpu_deflate":  False,
        "gpu_lz77":     False,
        "gpu_jls":      False,
        "gpu_bmp":      True,   # enables --gpu-bmp flag
        "gpu_streams":  None,
        "batch_size":   None,
    },
    3: {
        # CPU CharLS JPEG-LS. Best overall: near-BMP speed, 6× compression.
        # Standard lossless JPEG-LS (ISO 14495-1). Decodable by any JPEG-LS decoder.
        "format":      "jls",
        "strip_height": None,
        "threads":      None,
        "level":        None,
        "gpu_deflate":  False,
        "gpu_lz77":     False,
        "gpu_jls":      False,  # CPU path (no --gpu-jls flag)
        "gpu_bmp":      False,
        "gpu_streams":  None,
        "batch_size":   None,
    },
    4: {
        # GPU JPEG-LS (G1–G5 CUDA pipeline). Non-standard: row-context reset for
        # row-level parallelism. +30% faster than CPU at 1 worker; decode-limited at 6.
        # Output is 2.3× larger than CPU CharLS due to row-context reset.
        "format":      "jls",
        "strip_height": None,
        "threads":      None,
        "level":        None,
        "gpu_deflate":  False,
        "gpu_lz77":     False,
        "gpu_jls":      True,   # enables --gpu-jls flag
        "gpu_bmp":      False,
        "gpu_streams":  None,
        "batch_size":   None,
    },
    5: {
        # PNG with CPU deflate only. Baseline PNG; no GPU involvement.
        # Useful for comparing against GPU deflate.
        "format":      "png",
        "strip_height": 256,    # legacy default; no effect without --gpu-deflate
        "threads":      6,      # CPU deflate threads
        "level":        3,      # zlib level (3 = good speed/ratio trade-off)
        "gpu_deflate":  False,
        "gpu_lz77":     False,
        "gpu_jls":      False,
        "gpu_bmp":      False,
        "gpu_streams":  None,
        "batch_size":   None,
    },
    6: {
        # PNG with GPU deflate + GPU LZ77. Best PNG throughput.
        # --gpu-lz77 adds 200 MB VRAM; improves ratio AND throughput vs literal-only.
        # Requires RTX / SM >= 7.0. Falls back to CPU deflate on legacy GPUs.
        "format":      "png",
        "strip_height": 1024,   # larger strips reduce launch overhead on modern GPU
        "threads":      6,
        "level":        1,      # GPU deflate ignores level; CPU fallback uses it
        "gpu_deflate":  True,
        "gpu_lz77":     True,
        "gpu_jls":      False,
        "gpu_bmp":      False,
        "gpu_streams":  None,   # auto-selected (4–8 streams on RTX)
        "batch_size":   None,
    },
    7: {
        # Decode-only mode. No output files. Measures raw DICOM read ceiling.
        # Same pipeline as BMP but the encode step is skipped.
        "format":      "bmp",   # format is irrelevant when BENCHMARK_DECODE=True
        "strip_height": None,
        "threads":      None,
        "level":        None,
        "gpu_deflate":  False,
        "gpu_lz77":     False,
        "gpu_jls":      False,
        "gpu_bmp":      False,
        "gpu_streams":  None,
        "batch_size":   None,
    },
}

cfg = _scenarios[SCENARIO]

# ==========================================================
# BUILD COMMAND
# ==========================================================

cmd = [EXE, INPUT_PATH, OUTPUT_PATH]

cmd.extend(["--format", cfg["format"]])
cmd.extend(["--batch-workers", str(BATCH_WORKERS)])

if cfg["strip_height"] is not None:
    cmd.extend(["--strip-height", str(cfg["strip_height"])])

if cfg["threads"] is not None:
    cmd.extend(["--threads", str(cfg["threads"])])

if cfg["level"] is not None:
    cmd.extend(["--level", str(cfg["level"])])

if cfg["gpu_deflate"]:
    cmd.append("--gpu-deflate")

if cfg["gpu_lz77"]:
    cmd.append("--gpu-lz77")

if cfg["gpu_jls"]:
    cmd.append("--gpu-jls")

if cfg["gpu_bmp"]:
    cmd.append("--gpu-bmp")

if cfg["gpu_streams"] is not None:
    cmd.extend(["--gpu-streams", str(cfg["gpu_streams"])])

if cfg["batch_size"] is not None:
    cmd.extend(["--batch-size", str(cfg["batch_size"])])

if EXPORT_ALL:
    cmd.append("--all")

if FRAME_NUMBER is not None:
    cmd.extend(["--frame", str(FRAME_NUMBER)])

if BENCHMARK_DECODE:
    cmd.append("--benchmark-decode")

if VERBOSE:
    cmd.append("--verbose")

if BATCH_VERBOSE:
    cmd.append("--batch-verbose")

# ==========================================================
# RUN
# ==========================================================

scenario_names = {
    1: "BMP CPU  (uncompressed, CPU LUT)",
    2: "BMP GPU  (H2D -> W/L kernel -> D2H -> CPU write)",
    3: "JPEG-LS CPU  (CharLS)",
    4: "JPEG-LS GPU  (G1-G5 CUDA pipeline)",
    5: "PNG CPU  (zlib deflate)",
    6: "PNG GPU  (custom deflate + LZ77)",
    7: "Decode-only benchmark",
}

print(f"\nScenario {SCENARIO}: {scenario_names[SCENARIO]}")
print("\nRunning:\n")
print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))
print()

result = subprocess.run(cmd)

print("\nExit code:", result.returncode)
