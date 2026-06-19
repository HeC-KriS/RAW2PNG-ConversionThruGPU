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
THREADS = 5
LEVEL = 1

GPU_DEFLATE = 1
GPU_LZ77 = 1

# GPU_STREAMS is ignored in batch mode; set to None or use for single-file runs
GPU_STREAMS = None
BATCH_WORKERS = 6  # Phase 4 optimal: 539 files/sec at 2639ms for CT_CHEST

VERBOSE = False
BATCH_VERBOSE = False

# ==========================================================W
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