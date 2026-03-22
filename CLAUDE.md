# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

cfireants is a pure C port of [FireANTs](https://github.com/rohitrango/FireANTs) (commit `0d13a3f`), a GPU-accelerated medical image registration library using adaptive Riemannian optimization. It provides rigid, affine, and diffeomorphic deformable registration with two GPU backends:

- **CUDA** — production quality, matches or exceeds Python on all validation datasets
- **WebGPU** — functional, portable via wgpu-native (Vulkan on Linux, Metal on macOS). In active development.

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

# WebGPU only (macOS or Linux without CUDA)
cmake .. -DCFIREANTS_CUDA=OFF -DCFIREANTS_WEBGPU=ON
make -j$(nproc)

# WebGPU requires wgpu-native v27.0.4.0+ in third_party/wgpu/
# Download from https://github.com/gfx-rs/wgpu-native/releases
# On macOS: uses Metal backend; on Linux: uses Vulkan backend
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
build/test_validate_all --dataset small --trilinear  # GPU-native downsample
build/test_validate_greedy             # Greedy, all datasets

# WebGPU tests
build/test_webgpu_basic                # Element-wise ops, H2D/D2H
build/test_fft_fallback                # kissfft vs cuFFT comparison
build/test_webgpu_linear               # Moments + Rigid + Affine pipeline
build/test_validate_webgpu --dataset small              # Full pipeline validation
build/test_validate_webgpu --dataset small --trilinear  # GPU-native downsample
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
third_party/wgpu/      wgpu-native static library (Vulkan on Linux, Metal on macOS)
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

Functional pipeline via wgpu-native (Vulkan on Linux, Metal on macOS). Compatible with wgpu-native v27.0.4.0 (pre-StringView webgpu.h API). In active development.

**Key files:**
- `backend/webgpu/webgpu_context.h/.c` — device init, pipeline cache, batched dispatch
- `backend/webgpu/webgpu_kernels.h/.c` — GPU kernel dispatch (grid_sample, CC/MI loss, etc.)
- `backend/webgpu/webgpu_kernels_gpu.c` — GPU-native ops (compose, blur, Adam, affine_bwd, max_l2)
- `backend/webgpu/linear_webgpu.c` — rigid/affine registration loops
- `backend/webgpu/greedy_webgpu.c` — greedy deformable loop
- `backend/webgpu/syn_webgpu.c` — SyN loop + CPU warp inversion
- `backend/webgpu/fft_cpu_fallback.c` — kissfft-based FFT downsample
- `backend/webgpu/shaders/*.wgsl` — WGSL compute shaders (MI histogram, MI gradient, affine_bwd, blur_dhw3, blur_image, compose, CC loss, warp_ops, reduction, resize, reduce_max)

### WebGPU Current Metrics (trilinear mode, NVIDIA GB10 Vulkan)

| Dataset | CUDA NCC | WebGPU NCC | CUDA Time | WebGPU Time |
|---------|----------|------------|-----------|-------------|
| small (2mm full-head) | 0.9644 | **0.9647** | 7.5s | 13.7s |
| medium (1mm brain) | 0.9536 | **0.9541** | 18.2s | 90.5s |
| large (1mm full-head) | 0.9213 | **0.9190** | 55.1s | 93.2s |

**WebGPU matches CUDA accuracy on all datasets** (within 0.3%). MI histogram uses fixed-point u32 atomicAdd with dynamic FP_SCALE (up to 4096, auto-scaled for volume size to prevent u32 overflow). WebGPU MI is faster than CUDA MI because CUDA downloads full images for max-finding (known inefficiency in `mi_loss.cu`). SyN is slower due to per-iteration dispatch overhead.

### WebGPU Performance Analysis

**What's fast (matches CUDA):**
- Grid sampling fwd/bwd — WGSL compute shaders
- Affine grid generation — WGSL
- CC loss (box filter + NCC + gradient) — GPU-native with batched dispatch
- MI loss histogram — GPU workgroup-local with fixed-point atomicAdd (Metal-compatible)
- MI loss gradient — GPU with correct softmax derivative
- Adam optimizer — WGSL
- Affine grid backward (reduction to 12 values) — WGSL with CPU final sum
- Compositive warp update — WGSL
- Displacement field blur — WGSL (3-pass separable conv1d, `blur_dhw3.wgsl`)
- Image volume blur — WGSL (3-pass separable conv1d, `blur_image.wgsl`); used for rigid moving blur and trilinear downsample

**What's slow:**
- **SyN per-iteration overhead** — CC loss reads scalar loss value back to CPU every 10 iterations + max_l2_norm readback every 5 calls. Each readback forces a pipeline flush.
- **FFT downsampling** — CPU fallback via kissfft when using default FFT mode. Only runs at scale transitions (2-3 per stage), ~30ms total for small dataset. Not a bottleneck. Avoidable via `--trilinear` mode (GPU-native Gaussian blur + trilinear resize).
- **Warp inversion** — CPU fallback (550 iterations of fixed-point). Runs once for SyN evaluation.

### WebGPU Known Issues

- **Dispatch limit >65535** — Fixed. All shaders now use 2D dispatch via `wgpu_dispatch_dims()` and `@builtin(num_workgroups)` to compute flat thread indices, supporting volumes with >16M voxels.
- **Metal shader restrictions** — naga (wgpu's shader compiler) rejects variable indexing of local `array<T, N>` on Metal. Workaround: use `vec3/vec4` (support variable indexing) or helper functions with `if/else` chains. Affects `affine_grid_bwd.wgsl` and `blur_dhw3.wgsl` (both fixed).
- ~~MI gradient weakness~~ — **Fixed.** Dynamic FP_SCALE (up to 4096, auto-scaled for volume size). MI drives effective rigid/affine convergence on all datasets.
- ~~SyN warp field resize~~ — **Fixed.** Now uses trilinear interpolation (matching CUDA).
- ~~Dispatch limit >65535~~ — **Fixed.** 2D dispatch + auto-split in `wgpu_dispatch`.
- Segfault on exit during wgpu-native cleanup — cosmetic, does not affect results
- Pipeline cache stores string literal pointers — names must be static/literal

## Plan to Proceed

### Priority 1: MI Gradient Effectiveness — DONE

MI loss drives effective rigid/affine convergence on all datasets. Dynamic FP_SCALE (up to 4096, auto-scaled for volume size) prevents u32 overflow. WebGPU MI is ~2.5x faster than CUDA MI (CUDA downloads full images for max-finding each iteration).

### Priority 2: SyN CC Gradient — DONE

Fused CC loss ported from CUDA (`fused_cc.cu`) to WebGPU. Uses 5 separate intermediate buffers with box filter, `bwd_modify` shader with kernel_volume scaling, and CPU-side final gradient computation. Produces gradients matching CUDA within float32 precision.

**Critical bugs fixed during port:**
- Missing `wgpu_flush()` before CC/MI loss computation (forward batch not submitted)
- Race condition in `wgpu_blur_downsample` (buffer released before GPU work completed)
- `wgpu_begin_batch()` before backward pass caused stale gradient reads

### Priority 3: SyN Dispatch Overhead (remaining optimization)

SyN is ~11s vs CUDA's 1.5s on small dataset. The overhead is from per-iteration sync points:
- CC loss reads scalar loss → forces flush+poll (mitigated: only every 10 iters)
- max_l2_norm reads partial sums → forces flush+poll
- Fused CC does CPU-side gradient computation (step 6) — could be moved to GPU shader

Approaches for future optimization:
- **GPU-side step 6**: move final gradient computation (`gini_a*J - gini_b*I + gini_mu`) to WGSL shader
- **Deferred loss readback**: pipeline the readback to overlap with next iteration
- **Reduce per-iteration buffer allocations** in fused CC (currently allocates/releases 5+1 buffers per call)

### Priority 4: FFT on GPU — DONE (via trilinear alternative)

`DOWNSAMPLE_TRILINEAR` mode available via `--trilinear` flag. Trilinear matches or exceeds FFT accuracy on all datasets. kissfft CPU fallback remains as `DOWNSAMPLE_FFT` for Python-matching fidelity.

### Priority 5: Validation Parity — DONE

Full validation completed on all three datasets. WebGPU matches CUDA within 0.3% NCC on all datasets (small, medium, large).

### Future Optimizations

**Performance (SyN ~8x slower than CUDA):**
- Fused CC step 6 runs on CPU (5 GPU readbacks per iteration). Move to GPU using `fused_cc_bwd_grads.wgsl` (shader exists but is unused).
- Fused CC `bwd_modify` pack/unpack: rewrite shader to accept 5 separate buffers, eliminating 2 GPU copies per backward pass.
- Reduce per-iteration buffer allocations in `wgpu_fused_cc_loss` (currently allocates/releases 5+1 buffers per call).
- Deferred loss readback: pipeline to overlap with next iteration.

**Dead code to clean up:**
- `box_filter_channel_inplace` in `webgpu_kernels.c` — unused static function from earlier fused CC design.
- `shaders/fused_cc.wgsl`, `shaders/fused_cc_bwd_modify.wgsl`, `shaders/fused_cc_bwd_grads.wgsl` — external shader files never loaded. Inline WGSL strings are used instead. Can be deleted or wired in to replace CPU step 6.

**Code duplication to consolidate:**
- Warp field resize (CPU permute + GPU trilinear + CPU permute) duplicated 3x across `syn_webgpu.c` and `greedy_webgpu.c`. Extract shared `wgpu_resize_disp_dhw3()` helper.
- `make_gauss_buf` (syn) / `make_gpu_gauss` (greedy) — identical Gaussian kernel builders.

## Precision Analysis

### CUDA vs Python
- MI loss matches to 1e-13, affine_grid backward to 7e-6, FFT downsample to 0.001
- Individual operations identical to float32 precision
- Rigid/affine diverge by ~0.01 rotation / ~0.3mm translation due to float32 accumulation across 350+ iterations

### WebGPU vs CUDA
- **Validated on all three datasets.** WebGPU matches CUDA NCC within 0.3% (small: 0.9647 vs 0.9644, medium: 0.9541 vs 0.9536, large: 0.9190 vs 0.9213).
- FFT downsample (kissfft vs cuFFT): max diff 0.01. Trilinear mode (`--trilinear`) uses identical GPU path on both backends.
- Grid sampling, affine grid, fused CC loss: identical to float32 (same algorithms, same `kv` scaling).
- MI loss: GPU histogram with dynamic FP_SCALE (up to 4096, auto-scaled for volume size). Matches CPU MI to 5e-5 (loss) and 2e-9 (gradient). Falls back to CPU when num_bins != 32.

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
