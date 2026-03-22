# cfireants

Pure C port of [FireANTs](https://github.com/rohitrango/FireANTs) — GPU-accelerated medical image registration using adaptive Riemannian optimization. Provides rigid, affine, and diffeomorphic deformable registration via three GPU backends:

- **CUDA** — production quality, exceeds Python on all validation datasets
- **WebGPU** — portable via wgpu-native (Vulkan on Linux, Metal on macOS)
- **Metal** — native macOS/Apple Silicon, 7x faster than WebGPU on same hardware

Based on FireANTs commit [`0d13a3f`](https://github.com/rohitrango/FireANTs/tree/0d13a3f).

## Building

Requires CMake >= 3.18, a C11 compiler, and zlib.

```bash
mkdir -p build && cd build

# CUDA backend (default)
cmake .. -DCFIREANTS_CUDA=ON
make -j$(nproc)

# CUDA + WebGPU (requires wgpu-native in third_party/wgpu/)
cmake .. -DCFIREANTS_CUDA=ON -DCFIREANTS_WEBGPU=ON
make -j$(nproc)
```

## Quick Start

Run the full validation pipeline (Moments → Rigid → Affine → SyN) from the repo root:

```bash
# CUDA (FFT downsample, default)
build/test_validate_all --dataset small

# CUDA (trilinear downsample, GPU-native)
build/test_validate_all --dataset small --trilinear

# WebGPU
build/test_validate_webgpu --dataset small
build/test_validate_webgpu --dataset small --trilinear
```

## Validation Results

Full pipeline: Moments → Rigid → Affine → SyN/Greedy.

### CUDA Backend

| Dataset | Python SyN | C SyN | C Greedy | Python Time | C SyN Time | C Greedy Time |
|---------|-----------|-------|----------|-------------|------------|---------------|
| small   | 0.9450    | **0.9614** | 0.9512   | 4.0s        | 3.0s       | **2.7s**      |
| medium  | 0.9443    | **0.9554** | 0.9434   | 4.9s        | 5.5s       | **2.4s**      |
| large   | 0.8961    | **0.9176** | 0.9022   | 17.1s       | 17.9s      | **13.7s**     |

(NCC After — higher is better. Bold = best in row.)

### Downsample Modes

Both CUDA and WebGPU support two downsampling modes for multi-scale registration:

- **FFT** (default) — FFT-based downsample matching Python. CUDA uses cuFFT on GPU; WebGPU uses kissfft on CPU.
- **Trilinear** (`--trilinear`) — Gaussian blur + trilinear resize, fully GPU-native on both backends. Enables like-for-like comparison between CUDA and WebGPU.

### CUDA vs WebGPU — Trilinear Mode (small dataset, NVIDIA GB10 Vulkan)

Like-for-like comparison using `--trilinear` on the same hardware. Both use MI loss for rigid/affine, CC loss for SyN.

| Metric | CUDA | WebGPU | Ratio |
|--------|------|--------|-------|
| NCC Before | 0.5957 | 0.5957 | — |
| **NCC After** | **0.9644** | **0.9643** | **1.000** |
| Local NCC Loss | -0.6680 | -0.6692 | 1.002 |
| Total Time | 7.5s | 14.2s | 1.9x |
| Moments | 0.2s | 0.3s | 1.5x |
| Rigid (MI) | 2.8s | 0.9s | 0.3x |
| Affine (MI) | 3.0s | 1.1s | 0.4x |
| SyN (CC) | 1.4s | 11.9s | 8.5x |
| Peak RAM | 140 MB | 445 MB | 3.2x |

**WebGPU matches CUDA accuracy** (NCC 0.9643 vs 0.9644). MI loss drives rigid/affine. Fused CC loss (matching CUDA `fused_cc.cu`) drives SyN. WebGPU MI is faster than CUDA MI (CUDA downloads full images for max-finding each iteration). SyN is slower due to per-iteration dispatch overhead.

### All Datasets — CUDA Trilinear (NVIDIA GB10)

| Dataset | NCC Before | NCC After | Local NCC | Time | Peak RAM |
|---------|-----------|-----------|-----------|------|----------|
| small (2mm full-head) | 0.5957 | **0.9644** | -0.6680 | 7.5s | 140 MB |
| medium (1mm brain) | 0.5753 | **0.9536** | -0.8753 | 18.2s | 295 MB |
| large (1mm full-head) | 0.7254 | **0.9213** | -0.3784 | 55.1s | 369 MB |

### All Datasets — Trilinear Mode (NVIDIA GB10 Vulkan)

| Dataset | CUDA NCC | WebGPU NCC | CUDA Time | WebGPU Time | CUDA RAM | WebGPU RAM |
|---------|----------|------------|-----------|-------------|----------|------------|
| small (2mm full-head) | 0.9644 | **0.9647** | 7.5s | 13.7s | 140 MB | 485 MB |
| medium (1mm brain) | 0.9536 | **0.9541** | 18.2s | 90.5s | 295 MB | 854 MB |
| large (1mm full-head) | 0.9213 | **0.9190** | 55.1s | 93.2s | 369 MB | 870 MB |

**WebGPU matches CUDA accuracy on all datasets** (within 0.3%). Both backends use identical registration parameters: MI loss for full-head rigid/affine, CC for brain-extracted rigid/affine, fused CC for SyN deformable.

### CUDA: FFT vs Trilinear (small dataset)

| Metric | FFT (default) | Trilinear |
|--------|--------------|-----------|
| NCC After | 0.9614 | 0.9644 |
| Total Time | 7.6s | 7.5s |
| Peak RAM | 147 MB | 140 MB |

Trilinear matches or exceeds FFT accuracy. All measurements on NVIDIA GB10 (unified memory).

### All Datasets — Trilinear Mode (Apple M4 Pro, Metal backend via wgpu-native)

| Dataset | WebGPU NCC | Metal NCC | WebGPU Time | Metal Time | Metal RAM |
|---------|------------|-----------|-------------|------------|-----------|
| small (2mm full-head) | 0.9642 | **0.9638** | 60.1s | 8.5s | 391 MB |
| medium (1mm brain) | 0.9541 | **0.9542** | 131.7s | 20.8s | 3023 MB |
| large (1mm full-head) | 0.9191 | **0.9198** | 88.5s | 44.0s | 3259 MB |

**Metal matches WebGPU accuracy on all datasets** (within 0.1%) and runs **2–7x faster** on the same Apple M4 Pro hardware. WebGPU runs via wgpu-native's Metal backend, so both use the same GPU — the speed difference comes from Metal's native API with unified memory (zero-copy) vs WebGPU's abstraction overhead. Both backends use identical registration parameters: MI loss for full-head rigid/affine, CC for brain-extracted, fused CC for SyN.

See [CLAUDE.md](CLAUDE.md) for architecture details and known issues. See [validate/README.md](validate/README.md) for detailed CUDA benchmarks.

## Why C?

- No Python/PyTorch runtime dependency
- Predictable ~900 MB GPU memory regardless of image size (CUDA); ~400 MB (Metal small)
- Explicit forward+backward (no autograd overhead)
- Single static library (`libcfireants.a` + backend libs)
- Three GPU backends: CUDA (production), WebGPU (portable), Metal (macOS native)

## License

See the [FireANTs](https://github.com/rohitrango/FireANTs) repository for license information.
