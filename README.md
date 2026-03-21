# cfireants

Pure C port of [FireANTs](https://github.com/rohitrango/FireANTs) — GPU-accelerated medical image registration using adaptive Riemannian optimization. Provides rigid, affine, and diffeomorphic deformable registration via two GPU backends:

- **CUDA** — production quality, exceeds Python on all validation datasets
- **WebGPU** — portable via wgpu-native/Vulkan, in active development

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
# CUDA
build/test_validate_all --dataset small

# WebGPU
build/test_validate_webgpu --dataset small
```

## Validation Results

Measured on NVIDIA RTX 4090. Full pipeline: Moments → Rigid → Affine → SyN/Greedy.

### CUDA Backend

| Dataset | Python SyN | C SyN | C Greedy | Python Time | C SyN Time | C Greedy Time |
|---------|-----------|-------|----------|-------------|------------|---------------|
| small   | 0.9450    | **0.9614** | 0.9512   | 4.0s        | 3.0s       | **2.7s**      |
| medium  | 0.9443    | **0.9554** | 0.9434   | 4.9s        | 5.5s       | **2.4s**      |
| large   | 0.8961    | **0.9176** | 0.9022   | 17.1s       | 17.9s      | **13.7s**     |

(NCC After — higher is better. Bold = best in row.)

### WebGPU Backend (CC-only mode, small dataset)

| Stage | WebGPU | CUDA |
|-------|--------|------|
| Rigid | 1.1s | 1.0s |
| Affine | 1.4s | 1.4s |
| SyN | ~11s | 0.4s |

| Metric | WebGPU (CC) | CUDA (MI+CC) |
|--------|-------------|--------------|
| NCC Before | 0.5957 | 0.5957 |
| NCC After | 0.8139 | 0.9614 |

Rigid and affine match CUDA speed. NCC quality gap is from using CC loss instead of MI for linear stages (MI GPU shader in progress). See [CLAUDE.md](CLAUDE.md) for the detailed plan.

See [validate/README.md](validate/README.md) for detailed CUDA benchmarks.

## Why C?

- No Python/PyTorch runtime dependency
- Predictable ~900 MB GPU memory regardless of image size
- Explicit forward+backward (no autograd overhead)
- Single static library (`libcfireants.a` + `libcfireants_cuda.a`)
- WebGPU backend enables portability beyond NVIDIA GPUs

## License

See the [FireANTs](https://github.com/rohitrango/FireANTs) repository for license information.
