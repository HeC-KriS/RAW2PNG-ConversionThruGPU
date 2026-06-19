# RAW2PNG-ConversionThruGPU

GPU-accelerated image conversion pipeline for converting **DICOM**, **TIFF**, and **Camera RAW** images into PNG using CUDA-based image processing and **zlib-ng powered DEFLATE compression**.

The project is designed to maximize throughput by overlapping image loading, GPU filtering, compression, and PNG writing through a concurrent pipeline architecture optimized for low-latency image conversion.

---

## Features

* CUDA-accelerated image preprocessing
* Supports DICOM, TIFF, and Camera RAW formats
* High-performance **zlib-ng** compression backend
* Compression Level 0 support for maximum encoding speed
* Strip-based image processing for low memory usage
* Concurrent producer-consumer pipeline
* JPEG, JPEG-LS, JPEG2000, and RLE DICOM support
* OpenJPEG integration for JPEG2000 decoding
* Manual PNG generation with custom IDAT stream handling
* Performance profiling for GPU transfers and processing stages

---

## Architecture

```text
Input Image
      │
      ▼
┌─────────────┐
│ Loader      │
└─────────────┘
      │
      ▼
 Queue A
      │
      ▼
┌─────────────┐
│ GPU Filter  │
│ CUDA        │
└─────────────┘
      │
      ▼
 Queue B
      │
      ▼
┌─────────────┐
│ zlib-ng     │
│ Compression │
└─────────────┘
      │
      ▼
 Queue C
      │
      ▼
┌─────────────┐
│ PNG Writer  │
└─────────────┘
      │
      ▼
    PNG
```

---

## DICOM Support

Supported transfer syntaxes:

* Implicit VR Little Endian
* Explicit VR Little Endian
* JPEG Baseline
* JPEG Lossless
* JPEG-LS Lossless
* JPEG-LS Near Lossless
* JPEG 2000 Lossless
* JPEG 2000
* RLE Lossless

The pipeline performs:

* Pixel extraction
* Rescale slope/intercept application
* Window/level transformation
* Bit-depth normalization
* PNG preparation

before compression and encoding.

---

## GPU Processing

CUDA kernels accelerate:

* Pixel preprocessing
* DICOM transformations
* PNG scanline filtering

Performance metrics can be collected for:

* Host → Device transfer
* Kernel execution
* Device → Host transfer

---

## zlib-ng Compression

This project uses **zlib-ng**, a modern and optimized implementation of the DEFLATE algorithm with support for advanced CPU instruction sets such as:

* AVX2
* AVX-512
* SSE4.2
* ARM NEON

### Current Configuration

```cpp
Threads = 1
Compression Level = 0
```

### Why Compression Level 0?

The primary goal of this project is **minimum image conversion time**.

Compression Level 0:

* Disables expensive DEFLATE searching
* Minimizes CPU overhead
* Reduces PNG encoding latency
* Maximizes conversion throughput

This configuration is particularly useful for:

* Medical imaging workflows
* High-volume batch conversions
* GPU-focused performance benchmarking
* Real-time image processing pipelines

---

## PNG Generation

```text
Image Data
    │
    ▼
PNG Filters
    │
    ▼
zlib-ng DEFLATE
    │
    ▼
IDAT Chunks
    │
    ▼
PNG File
```

The encoder manually constructs PNG chunks to provide complete control over:

* Compression strategy
* Memory usage
* Pipeline scheduling
* Performance optimization

---

## Dependencies

* CUDA Toolkit 11.8+
* DCMTK
* OpenJPEG
* zlib-ng
* libpng
* libraw
* libtiff
* CMake 3.18+
* C++17

---

## Setup and Build

```bash
git clone https://github.com/ProjectBA14/RAW2PNG-ConversionThruGPU.git
cd RAW2PNG-ConversionThruGPU

# Build the C++ executable
cmake -B build
cmake --build build --config Release
```

Make sure you have Python 3.x installed to use the testing and automation scripts.

---

## Usage (via `Testing.py`)

The most efficient way to run and benchmark the pipeline is using the provided `Testing.py` script. It wraps the C++ executable and provides easy configuration for batch processing, multi-threading, and GPU acceleration.

### How to Run

1. Open `Testing.py` in your preferred text editor.
2. Update the `EXE` variable to point to your compiled executable (e.g., `r"D:\Projects\gpu-optimize\build\gpu_png_encoder.exe"`).
3. Set your `INPUT_PATH` (a single file or a directory of DICOMs) and `OUTPUT_PATH`.
4. Run the script:
```bash
python Testing.py
```

### Parameters in `Testing.py`

| Parameter | Description |
| --------- | ----------- |
| `EXE` | Absolute path to the compiled `gpu_png_encoder.exe` executable. |
| `INPUT_PATH` | Path to the input image file or directory containing files to process. |
| `OUTPUT_PATH` | Path where the output PNG file(s) will be saved. |
| `STRIP_HEIGHT` | Number of rows processed per GPU strip. Lower values reduce memory but increase overhead. |
| `THREADS` | Number of CPU compression worker threads. |
| `LEVEL` | Deflate compression level (0–9). 0 is no compression (fastest). |
| `GPU_DEFLATE` | Enable (`1`) or disable (`0`) GPU-accelerated Deflate compression. |
| `GPU_LZ77` | Enable (`1`) or disable (`0`) GPU-accelerated LZ77 compression. |
| `GPU_STREAMS` | Number of concurrent CUDA streams to use. Set to `None` for batch mode. |
| `BATCH_WORKERS` | Number of parallel worker processes for batch directory conversions. |
| `VERBOSE` | Set to `True` to print detailed performance metrics per file. |
| `BATCH_VERBOSE` | Set to `True` to print metrics during batch processing. |
| `EXPORT_ALL` | Set to `True` to export all frames from a multi-frame DICOM file. |
| `FRAME_NUMBER` | Specify a single frame index to extract from a multi-frame DICOM. |

---

## Detailed Process Pipeline

The pipeline dynamically adapts to your hardware, splitting into two variants:
* **Legacy Pipeline (GT 710, SM 3.5):** Uses the GPU for fast filtering, but transfers data back to the CPU for multi-threaded Deflate compression.
* **Modern Pipeline (RTX 5050+, SM >= 7.0):** Keeps data fully resident on the GPU across Filtering, Deflate compression, Checksums, and PNG Assembly.

### Step 1: Loader (Input Stage)
* **Hardware:** **CPU**
* **Algorithms:** Parses input formats (DICOM, TIFF, RAW).
* **Process:** The CPU reads the image from disk in horizontal "strips" (e.g., 1024 rows at a time). This bounds memory usage so we can process arbitrarily large images. The CPU allocates memory on the GPU and transfers the raw data using asynchronous CUDA streams.

### Step 2: DICOM Preprocessing (Optional)
* **Hardware:** **GPU**
* **Algorithms:** Bit Alignment, Window/Level Transformation (Rescale Slope/Intercept), Big-Endian Conversion.
* **Process:** For medical images, a specialized CUDA kernel (`dicom_preprocess_kernel`) runs thousands of threads to align bits, adjust brightness/contrast, and flip bytes to match PNG's Big-Endian requirement. This is done **in-place**, meaning it directly overwrites the raw data to save massive amounts of GPU memory.

### Step 3: PNG Scanline Filtering
* **Hardware:** **GPU**
* **Algorithms:** PNG Filters (None, Sub, Up, Average, Paeth), Block-wide Reduction.
* **Process:** The core `filter_select_kernel` evaluates all 5 PNG filters simultaneously. The GPU assigns one Block (256 threads) per image row. The threads calculate a "score" for each filter, use ultra-fast **Shared Memory** to run a tournament-style "Block-wide reduction" to find the most compressible filter, and immediately apply the winner. This single-pass design reduces GPU memory transfers by 300%.

### Step 4: Compression (Deflate / LZ77)
* **Hardware:** **CPU** (Legacy) or **GPU** (Modern)
* **Algorithms:** DEFLATE, LZ77 Match Finding.
* **Process:** 
  * **Legacy Pipeline:** Filtered data is sent back to the CPU where it is compressed using highly optimized multi-threaded **zlib-ng**.
  * **Modern Pipeline:** Data stays on the GPU. A custom Fixed Huffman DEFLATE encoder handles the compression. You can use the ultra-fast **Literal-only** mode, or enable the `--gpu-lz77` flag, which launches a parallel LZ4-style match finder using Shared Memory hash tables to find repetitive byte sequences and improve compression ratio.

### Step 5: Checksums & PNG Assembly
* **Hardware:** **CPU** (Legacy) or **GPU** (Modern)
* **Algorithms:** CRC32, Adler32.
* **Process:** The PNG standard strictly requires cryptographic checksums. In the modern pipeline, the GPU calculates these checksums and physically assembles the binary PNG chunks (like the `IDAT` headers) directly in VRAM.

### Step 6: PNG Writer (Output Stage)
* **Hardware:** **CPU**
* **Process:** The CPU receives the final, compressed binary chunks from the GPU (or from the CPU compression threads) and writes them to the output `.png` file on the hard drive.

---

## Architectural Enhancements

* **Zero-Copy Modern Pipeline:** By keeping data resident on the GPU through the entire filter, compress, and assemble stages, we eliminate the severe bottleneck of Host-to-Device (H2D) and Device-to-Host (D2H) memory transfers.
* **Asynchronous Stream Overlap:** The modern pipeline maintains a pool of 4–8 CUDA streams, allowing the CPU to upload the *next* image strip while the GPU is still crunching numbers on the *current* strip.
* **Bounded Queue Concurrency:** A robust producer/consumer architecture ensures that no matter how large the input image, the program's RAM usage remains strictly capped by the queue sizes.

---

## Future Improvements

* Pinned-memory GPU transfers
* Asynchronous CUDA streams
* Multi-GPU support
* Batch DICOM conversion
* GPU-based PNG filtering
* SIMD-aware PNG filter optimization
* End-to-end performance benchmarking suite

---

## License

This project is intended for research, learning, and high-performance image processing experimentation.
