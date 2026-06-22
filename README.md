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

## Build

```bash
git clone https://github.com/ProjectBA14/RAW2PNG-ConversionThruGPU.git
cd RAW2PNG-ConversionThruGPU

cmake -B build
cmake --build build --config Release
```

---

## Usage

```bash
gpu_png_encoder input.dcm output.png
```

### Options

```bash
gpu_png_encoder input.dcm output.png --strip-height 128
gpu_png_encoder input.dcm output.png --threads 1
gpu_png_encoder input.dcm output.png --level 0
gpu_png_encoder input.dcm output.png --verbose
```

| Option           | Description                          |
| ---------------- | ------------------------------------ |
| --strip-height N | Rows processed per GPU strip         |
| --threads N      | Number of compression worker threads |
| --level N        | Compression level (0–9)              |
| --verbose        | Print performance metrics            |

---

## Performance-Oriented Design

The project focuses on minimizing end-to-end conversion time through:

* GPU-accelerated pixel processing
* zlib-ng optimized compression
* Compression Level 0 encoding
* Strip-based streaming
* Bounded memory queues
* Concurrent pipeline execution
* Low-overhead PNG generation

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
