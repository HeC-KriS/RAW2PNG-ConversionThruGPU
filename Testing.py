import subprocess

# ==========================================================
# CONFIG
# ==========================================================
# Folder mode examples
EXE = r"D:\Projects\gpu-optimize\build\gpu_png_encoder.exe"

# INPUT_PATH = r"C:\Users\Krish\Downloads\mega_image.dcm"
# OUTPUT_PATH = r"C:\Users\Krish\Downloads\images_out\something.png"


INPUT_PATH = r"C:\Users\Krish\Downloads\CT_ABDPELVIS"
OUTPUT_PATH = r"C:\Users\Krish\Downloads\images_out"

# ==========================================================
# PERFORMANCE OPTIONS
# ==========================================================

STRIP_HEIGHT = 1024
THREADS = 8
LEVEL = 1

GPU_DEFLATE = 1
GPU_LZ77 = 1

# GPU_STREAMS: ignored in batch mode; only matters for single-file runs
GPU_STREAMS = None
BATCH_WORKERS = 6  # Phase 4 PNG optimal: 539 files/sec

# Export format: "png" (lossless, compressed) or "bmp" (fastest, uncompressed 8-bit gray)
# BMP peak: ~800 files/sec at 6W (1.48x faster than PNG for CT datasets)


VERBOSE = False
BATCH_VERBOSE = False
FORMAT = "bmp"            # "png" or "bmp"
BENCHMARK_DECODE = False  # True → --benchmark-decode (no output written, measures DCMTK ceiling)
# ==========================================================
# DICOM OPTIONS
# ==========================================================

EXPORT_ALL = False
FRAME_NUMBER = None

# ==========================================================
# BUILD COMMAND
# ==========================================================

cmd = [EXE, INPUT_PATH, OUTPUT_PATH]

cmd.extend([
    "--strip-height", str(STRIP_HEIGHT),
    "--threads", str(THREADS),
    "--level", str(LEVEL),
    "--format", FORMAT,
])

if GPU_DEFLATE:
    cmd.append("--gpu-deflate")

if GPU_LZ77:
    cmd.append("--gpu-lz77")

if GPU_STREAMS:
    cmd.extend([
        "--gpu-streams",
        str(GPU_STREAMS)
    ])

if BATCH_WORKERS:
    cmd.extend([
        "--batch-workers",
        str(BATCH_WORKERS)
    ])

if EXPORT_ALL:
    cmd.append("--all")

if FRAME_NUMBER is not None:
    cmd.extend([
        "--frame",
        str(FRAME_NUMBER)
    ])

if BENCHMARK_DECODE:
    cmd.append("--benchmark-decode")

if VERBOSE:
    cmd.append("--verbose")

if BATCH_VERBOSE:
    cmd.append("--batch-verbose")

# ==========================================================
# RUN
# ==========================================================

print("\nRunning:\n")
print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))
print()

result = subprocess.run(cmd)

print("\nExit code:", result.returncode)