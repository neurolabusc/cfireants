# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

cfireants is a pure C port of [FireANTs](https://github.com/rohitrango/FireANTs) (commit `0d13a3f`), a GPU-accelerated medical image registration library using adaptive Riemannian optimization. It provides rigid, affine, and diffeomorphic deformable registration with two GPU backends:

- **CUDA** — production quality, matches or exceeds Python on all validation datasets
- **WebGPU** — functional, portable via wgpu-native/Vulkan. In active development.

The goal is a faithful C reproduction of the Python data flow across both backends.

## Build

Requires CMake >= 3.18, a C11 compiler, and zlib. CUDA and WebGPU backends are optional.

```bash
mkdir -p build && cd build

# CUDA only (default)
cmake .. -DCFIREANTS_CUDA=ON
make -j$(nproc)

# CUDA + WebGPU
cmake .. -DCFIREANTS_CUDA=ON -DCFIREANTS_WEBGPU=ON
make -j$(nproc)

# WebGPU requires wgpu-native in third_party/wgpu/
# Download from https://github.com/gfx-rs/wgpu-native/releases
```

Set `-DCMAKE_CUDA_ARCHITECTURES=89` (or your GPU arch) if auto-detection fails.

## Tests

All test binaries are in `build/`. Validation tests use relative paths and **must run from the repo root**.

```bash
# CUDA tests
build/test_nifti_load --all            # NIfTI I/O
build/test_phase2                      # grid sample, CC loss, blur
build/test_phase3                      # moments registration
build/test_phase4                      # rigid registration
build/test_phase5                      # affine registration
build/test_phase6                      # greedy deformable (CPU, small only)
build/test_cuda_backend                # CUDA kernel validation
build/test_gpu_pipeline                # Full GPU pipeline (small, MI+CC)
build/test_validate_all                # SyN, all datasets
build/test_validate_all --dataset small
build/test_validate_greedy             # Greedy, all datasets

# WebGPU tests
build/test_webgpu_basic                # Element-wise ops, H2D/D2H
build/test_fft_fallback                # kissfft vs cuFFT comparison
build/test_webgpu_linear               # Moments + Rigid + Affine pipeline
build/test_validate_webgpu --dataset small  # Full pipeline validation
```

## Architecture

### Registration Pipeline

Full pipeline: Moments (CPU+GPU) → Rigid (GPU) → Affine (GPU) → SyN or Greedy (GPU).

Registration stages in `src/registration/`:
- **moments.c** — center-of-mass alignment with GPU orientation candidate search
- **rigid.c** — quaternion-based rotation + translation (3D)
- **affine.c** — full affine (12-DOF in 3D)
- **greedy.c** — non-linear deformable via compositive displacement fields
- **syn.c** — symmetric diffeomorphic (SyN) registration

Each stage uses multi-scale optimization with configurable loss functions (CC or MI) and Adam optimizer.

### Backend Abstraction

`backend_ops_t` in `include/cfireants/backend.h` provides 18 function pointers covering tensor ops, grid sampling, losses, blur, Adam, and interpolation. Three backends:

- **CPU** (`backend/cpu/`) — reference implementations, stubs for compute-intensive ops
- **CUDA** (`backend/cuda/`) — production GPU backend with fused kernels
- **WebGPU** (`backend/webgpu/`) — portable GPU backend via wgpu-native

The CUDA and WebGPU backends also have **fused registration loops** that bypass `backend_ops_t` for performance — these call GPU kernels directly within the optimization loop, minimizing CPU↔GPU transfers.

### Source Layout

```
include/cfireants/     Public headers (tensor, backend, image, registration, etc.)
src/                   Core C (tensor, nifti_io, image, interpolator, losses, utils)
src/registration/      Registration algorithms (moments, rigid, affine, greedy, syn)
backend/cuda/          CUDA kernels + fused registration loops
backend/webgpu/        WebGPU shaders + dispatch + fused loops + FFT fallback
third_party/kissfft/   BSD-3 FFT library for WebGPU CPU fallback
third_party/wgpu/      wgpu-native static library (Vulkan backend)
tests/                 Test programs
validate/              Validation datasets and benchmarks
```

### Key Design Decisions

**No autograd.** Each differentiable operation (grid_sample, CC loss, MI loss) has explicit forward + backward implementations. The backward chain: `loss_grad → grid_sample_bwd → affine_grid_bwd → parameter_grad`.

**Loss functions — MI vs CC:**
- Default (matching Python): MI for rigid/affine (robust for cross-modal and full-head images), CC for deformable.
- Fast/CC-only mode: CC for all stages via `loss_type=LOSS_CC`. ~2.5x faster but less robust for images with different intensity ranges (e.g., full-head with scalp).
- MI uses Gaussian Parzen windowing with `num_bins=32` (matching Python defaults).

**CRITICAL porting principle:** Both CUDA and WebGPU backends must faithfully replicate the exact Python data flow — same downsampling, same multi-scale warp resize, same Gaussian kernel construction, same optimizer state management.

## CUDA Backend

Production quality. Full pipeline on GPU with fused kernels.

**Key files:** `backend/cuda/linear_gpu.cu` (rigid/affine loops), `greedy_gpu.cu`, `syn_gpu.cu`, `downsample_fft.cu` (cuFFT), `mi_loss.cu` (shared-memory histogram), `fused_cc.cu`, `fused_compose.cu`, `fused_blur.cu`, `warp_inverse.cu`.

See `validate/README.md` for benchmark tables. Summary: exceeds Python quality on all datasets, comparable speed, ~900 MB GPU memory (vs Python's 800-6300 MB).

## WebGPU Backend

Functional pipeline via wgpu-native (Vulkan backend on Linux). In active development.

**Key files:**
- `backend/webgpu/webgpu_context.h/.c` — device init, pipeline cache, batched dispatch
- `backend/webgpu/webgpu_kernels.h/.c` — GPU kernel dispatch (grid_sample, CC/MI loss, etc.)
- `backend/webgpu/webgpu_kernels_gpu.c` — GPU-native ops (compose, blur, Adam, affine_bwd, max_l2)
- `backend/webgpu/linear_webgpu.c` — rigid/affine registration loops
- `backend/webgpu/greedy_webgpu.c` — greedy deformable loop
- `backend/webgpu/syn_webgpu.c` — SyN loop + CPU warp inversion
- `backend/webgpu/fft_cpu_fallback.c` — kissfft-based FFT downsample
- `backend/webgpu/shaders/*.wgsl` — WGSL compute shaders

### WebGPU Current Metrics (small dataset, RTX 4090 via Vulkan)

**CC-only mode (all stages use CC loss):**

| Stage | WebGPU | CUDA | Ratio |
|-------|--------|------|-------|
| Rigid | 1.1s | 1.0s | 1.1x |
| Affine | 1.4s | 1.4s | 1.0x |
| SyN | ~11s | 0.4s | ~28x |
| Total | ~14s | 3.0s | ~4.7x |

| Metric | WebGPU (CC) | CUDA (MI+CC) |
|--------|-------------|--------------|
| NCC Before | 0.5957 | 0.5957 |
| NCC After | 0.8139 | 0.9614 |
| Local NCC Loss | -0.3275 | -0.6511 |

**MI mode (MI for rigid/affine, CC for deformable):**
- Rigid+Affine: ~80-500s (MI uses CPU fallback — GPU CAS atomics too slow)
- Quality: expected to match CUDA more closely, but not yet validated at speed

### WebGPU Performance Analysis

**What's fast (matches CUDA):**
- Grid sampling fwd/bwd — WGSL compute shaders
- Affine grid generation — WGSL
- CC loss (box filter + NCC + gradient) — GPU-native with batched dispatch
- Adam optimizer — WGSL
- Affine grid backward (reduction to 12 values) — WGSL with CPU final sum
- Compositive warp update — WGSL
- Displacement field blur — WGSL (3-pass separable conv1d)

**What's slow:**
- **MI loss** — CPU fallback. GPU CAS atomics on global memory are ~100x slower than CUDA's shared-memory atomics. See "MI Loss Optimization" below.
- **SyN per-iteration overhead** — CC loss reads scalar loss value back to CPU every 10 iterations + max_l2_norm readback every 5 calls. Each readback forces a pipeline flush.
- **FFT downsampling** — CPU fallback via kissfft. Only runs at scale transitions (2-3 per stage), ~30ms total for small dataset. Not a bottleneck.
- **Warp inversion** — CPU fallback (550 iterations of fixed-point). Runs once for SyN evaluation.

### WebGPU Known Issues

- Segfault on exit during wgpu-native cleanup — cosmetic, does not affect results
- MI loss GPU shader compiles and runs but too slow for production (global CAS atomics)
- SyN warp field resize between scales uses zero-init instead of interpolation
- Pipeline cache stores string literal pointers — names must be static/literal

## Plan to Proceed

### Priority 1: MI Loss GPU Performance

The MI loss is the critical remaining bottleneck. CUDA uses shared-memory histogram accumulation (workgroup-local). WebGPU approach:

1. **Workgroup-local histogram**: Each workgroup (256 threads) accumulates a local 32×32 histogram in `var<workgroup>` shared memory using `atomicAdd` on `atomic<u32>` (workgroup atomics are fast). Total shared memory: 32×32×4 = 4KB (within WebGPU's 16KB minimum).
2. **Global merge**: After the local histogram is complete, one thread per workgroup does a single CAS-based merge to the global histogram. This reduces global atomics from O(N × bins²) to O(workgroups × bins²).
3. **Gradient pass**: Already working as a separate WGSL shader (binding layout fixed).

This would bring MI from ~80s to ~2-5s for rigid+affine, making the MI+CC pipeline viable at ~15-20s total.

### Priority 2: SyN Dispatch Overhead

SyN is 11s vs CUDA's 0.4s. The overhead is from per-iteration sync points:
- CC loss reads scalar loss → forces flush+poll (mitigated: only every 10 iters)
- max_l2_norm reads partial sums → forces flush+poll (mitigated: cached every 5 calls)
- Each dispatch creates a new compute pass (even in batch mode, changing pipelines requires a new pass)

Approaches:
- **Fused CC loss**: combine box filter + NCC + gradient into fewer dispatches (matching CUDA's `fused_cc_loss`)
- **Deferred loss readback**: compute loss but defer reading until next iteration (pipeline the readback)
- **Skip convergence checking**: run fixed iterations without loss monitoring

### Priority 3: FFT on GPU

Currently uses kissfft CPU fallback. Three options (see `memory/project_webgpu_fft_challenge.md`):
- **Spatial blur + trilinear resize** — already implemented as WGSL, different numerical path but valid. Simplest.
- **Stockham FFT in WGSL** — radix-2 with zero-padding for non-power-of-2. Correct but 2-3x memory overhead.
- **VkFFT via Vulkan** — production quality, handles all sizes, but ties to Vulkan (not pure WGSL).

The CPU fallback is not a bottleneck (~30ms total for small dataset) so this is low priority.

### Priority 4: Validation Parity

Once MI is fast enough, run the full validation suite (small/medium/large) and document WebGPU vs CUDA in `validate/webgpu/README.md`. The medium dataset (brain-extracted) should work well with CC-only since both images have similar intensity ranges.

## Precision Analysis

### CUDA vs Python
- MI loss matches to 1e-13, affine_grid backward to 7e-6, FFT downsample to 0.001
- Individual operations identical to float32 precision
- Rigid/affine diverge by ~0.01 rotation / ~0.3mm translation due to float32 accumulation across 350+ iterations

### WebGPU vs CUDA
- FFT downsample (kissfft vs cuFFT): max diff 0.01 on real images (validated on all sizes)
- Grid sampling, affine grid, CC loss: expected identical to float32 (same algorithms)
- MI loss: CPU fallback uses same code as CPU backend, different from CUDA MI (different reduction order)
- Overall NCC After: 0.81 (WebGPU CC-only) vs 0.96 (CUDA MI+CC) — gap is from loss function choice, not numerical precision

## Dependencies

- **Core**: CMake >= 3.18, C11 compiler, zlib
- **CUDA backend**: CUDA toolkit, cufft
- **WebGPU backend**: wgpu-native (v27.0.4.0+, pre-downloaded in `third_party/wgpu/`)
- **FFT fallback**: kissfft (BSD-3, vendored in `third_party/kissfft/`)
- Optional: zstd (for zstd-compressed NIfTI)

## Known Issues (CUDA)

- ~100 `cudaMalloc` calls lack error checking — needs a `CUDA_CHECK()` macro
- Duplicated kernels across files: permute (`dhw3↔3dhw`), `max_l2_norm`, `make_gpu_gauss` should be consolidated
- No `_default()` functions for `rigid_opts_t`, `affine_opts_t`, `greedy_opts_t`, `syn_opts_t`
- `beta2` differs between WarpAdam (0.99, for SyN/Greedy) and torch.optim.Adam (0.999, for rigid/affine) — intentional, matching Python defaults
