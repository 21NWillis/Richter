# Richter
**High-Performance GPU Seismic Wave Propagation Library**

A CUDA-optimized 3D Acoustic Wave Propagation engine built for Reverse Time Migration (RTM) and Full Waveform Inversion (FWI) workloads. Named after Charles F. Richter — because when it comes to seismic compute, magnitude matters.

Main Kernel is called Hello.cu, because its like a wave. Get it?

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Python API  (NumPy ↔ PyBind11)                 │
├─────────────────────────────────────────────────┤
│  C++ Orchestrator  (model.cpp)                  │
│  Memory management, time-stepping, dispatch     │
├──────────┬──────────┬──────────┬────────────────┤
│  Naive   │  SHMem   │ RegRot   │  Triton (TBD) │
│  Kernel  │  2.5D    │ Sliding  │               │
│          │  Tiling  │ Window   │               │
└──────────┴──────────┴──────────┴────────────────┘
         CUDA Computational Backend
```

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build . --config Release
```

## Running Tests

```bash
./build/runtests
```

## Benchmarking

```bash
# Default 256^3 grid
./build/benchmark

# Custom grid size and peak bandwidth (GB/s)
./build/benchmark 512 448.0
```

## Benchmark Results

Grid: 512^3 | GPU: RTX 3070

| Implementation | GPts/s | Effective BW | % Peak BW |
|----------------|--------|-------------|-----------|
| Devito (Auto-tuned) | — | — | — |
| Naive Kernel | 4.47 | 71.6 GB/s | 15.9% |
| SHMem 2.5D Tiling | — | — | — |
| RegRot Sliding Window | — | — | — |
| OpenAI Triton | — | — | — |

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Infrastructure & Correctness (Naive kernel, wavelet, validation) | In Progress |
| 2 | Optimization (Shared memory, register rotation) | Planned |
| 3 | Data & Benchmarking (SEG-Y, roofline analysis) | Planned |
| 4 | Polish (Triton experiment, Python API, documentation) | Planned |

## Key Metrics

| Metric | Description |
|--------|-------------|
| **GPts/s** | Giga-points per second — primary throughput metric |
| **Effective Bandwidth** | Actual bytes moved vs. hardware peak (GB/s) |
| **% Peak BW** | How close to the theoretical memory bandwidth ceiling |

## Tech Stack

- **C++17 / CUDA 12** — Core compute
- **PyBind11** — Python interface
- **OpenAI Triton** — Experimental kernel (planned)
- **CMake** — Build system
