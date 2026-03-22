# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

cfireants is a pure C port of [FireANTs](https://github.com/rohitrango/FireANTs) (commit `0d13a3f`), a GPU-accelerated medical image registration library using adaptive Riemannian optimization. It provides rigid, affine, and diffeomorphic deformable registration with three GPU backends:

- **CUDA** — production quality, matches or exceeds Python on all validation datasets
- **WebGPU** — portable via wgpu-native (Vulkan on Linux, Metal on macOS). Matches CUDA within 0.3% NCC.
- **Metal** — native macOS/Apple Silicon. Matches WebGPU within 0.07% NCC, runs 7x faster on same hardware.

The goal is a faithful C reproduction of the Python data flow across all backends.

## Build

Requires CMake >= 3.18, a C11 compiler, and zlib. All GPU backends are optional.

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

# Metal only (macOS/Apple Silicon)
cmake .. -DCFIREANTS_CUDA=OFF -DCFIREANTS_METAL=ON
make -j$(nproc)

# WebGPU + Metal (macOS — both on same hardware for comparison)
cmake .. -DCFIREANTS_CUDA=OFF -DCFIREANTS_WEBGPU=ON -DCFIREANTS_METAL=ON
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
build/test_validate_webgpu --dataset small              # Full pipeline validation (SyN)
build/test_validate_webgpu --dataset small --trilinear  # GPU-native downsample
build/test_validate_webgpu --dataset small --trilinear --greedy  # Greedy (faster, ~1% less NCC)

# Metal tests
build/test_metal_backend                               # Element-wise ops, grid sample
build/test_metal_linear                                # Moments + Rigid + Affine pipeline
build/test_validate_metal --dataset small               # Full pipeline (MI+CC, FFT, SyN)
build/test_validate_metal --dataset small --trilinear   # Full pipeline (MI+CC, trilinear, SyN)
build/test_validate_metal --dataset small --trilinear --greedy  # Greedy deformable
build/test_cpu_vs_metal                                    # CPU vs Metal stage-by-stage comparison
build/test_cpu_vs_metal --syn                              # Include SyN comparison (slow)
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

`backend_ops_t` in `include/cfireants/backend.h` provides 18 function pointers covering tensor ops, grid sampling, losses, blur, Adam, and interpolation. Four backends (CPU + three GPU):

- **CPU** (`backend/cpu/`) — reference implementations, stubs for compute-intensive ops
- **CUDA** (`backend/cuda/`) — production GPU backend with fused kernels
- **WebGPU** (`backend/webgpu/`) — portable GPU backend via wgpu-native
- **Metal** (`backend/metal/`) — native macOS/Apple Silicon GPU backend

All three GPU backends have **fused registration loops** that bypass `backend_ops_t` for performance — calling GPU kernels directly within the optimization loop, minimizing CPU↔GPU transfers.

### Source Layout

```
include/cfireants/     Public headers (tensor, backend, image, registration, etc.)
src/                   Core C (tensor, nifti_io, image, interpolator, losses, utils)
src/registration/      Registration algorithms (moments, rigid, affine, greedy, syn)
backend/cuda/          CUDA kernels + fused registration loops
backend/webgpu/        WebGPU shaders + dispatch + fused loops + FFT fallback
backend/metal/         Metal shaders + dispatch + fused loops + MPSGraph FFT
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

### Priority 3: SyN Dispatch Overhead — INVESTIGATED, ARCHITECTURAL LIMIT

SyN is ~58s (WebGPU/Metal backend) vs ~8s (native Metal) on small dataset, M4 Pro.

**Root cause: wgpu-native per-dispatch overhead.** Every `wgpu_dispatch()` in non-batch mode creates a command encoder, begins a compute pass, dispatches, ends the pass, finishes, submits, and polls. SyN does ~30 dispatches per iteration × 350 iterations = ~10,500 GPU round-trips. At ~0.1ms each overhead, this accounts for the majority of the gap vs native Metal.

**Optimization attempts (all reverted — no benefit on unified memory):**
- **GPU step 6 shader**: Replaced 5 CPU readbacks with 3 GPU dispatches. *Slower* — on M4 Pro unified memory, `wgpu_read_buffer` after a flush is essentially a pointer read (~0), while each new `wgpu_dispatch` costs ~0.1ms submit+poll overhead.
- **Batch backward dispatches**: Wrapping grid_sample_bwd in `wgpu_begin_batch/flush`. *9s slower* — batch mode creates separate compute passes per pipeline change, adding overhead.
- **Remove warp_adam poll**: Skipping `wgpuDevicePoll` after buffer copies. *12s slower* — command buffers pile up in wgpu-native's internal queue.
- **3D texture trilinear resize**: Hardware texture sampling for downsample resize step. *No improvement* — resize only runs ~6 times total (scale transitions), not a bottleneck.

**Conclusion:** The WebGPU dispatch model is fundamentally mismatched with unified memory GPUs. On discrete GPUs (NVIDIA Vulkan), the CPU→GPU latency is hidden by pipelining. On Apple Silicon unified memory, native Metal dispatch (synchronous, zero-copy) is 7-8x faster. WebGPU optimizations would need to restructure the entire dispatch model (e.g., recording all dispatches into a single command buffer per iteration) which is beyond the current architecture.

### Priority 4: FFT on GPU — DONE (via trilinear alternative)

`DOWNSAMPLE_TRILINEAR` mode available via `--trilinear` flag. Trilinear matches or exceeds FFT accuracy on all datasets. kissfft CPU fallback remains as `DOWNSAMPLE_FFT` for Python-matching fidelity.

### Priority 5: Validation Parity — DONE

Full validation completed on all three datasets. WebGPU matches CUDA within 0.3% NCC on all datasets (small, medium, large).

### Future Optimizations

**Trilinear downsample via 3D textures:**
- Implemented for Metal (`trilinear_resize_texture` in grid_sample.metal). Uses `texture3d<float>` with `constexpr sampler(filter::linear)` for hardware interpolation. Accuracy identical to compute shader.
- **No speed benefit** — the `replaceRegion` copy from buffer to texture costs more than the hardware interpolation saves. Resize only runs ~6 times per pipeline (scale transitions), so compute shader is already fast enough.
- WebGPU: not implemented. Would require `Float32Filterable` device feature (requested optionally) and substantial bind group plumbing. Similar copy overhead expected via `wgpuQueueWriteTexture`.
- 3D textures would only help if the inner-loop `grid_sample_3d_fwd` used texture sampling (runs every iteration), but grid_sample requires arbitrary coordinates from the affine/warp grid, which is already what texture sampling does — the compute shader approach is equivalent.

**WebGPU SyN performance (architectural limit):**
- The per-dispatch overhead (~0.1ms × ~10,500 dispatches per SyN run) is the dominant cost. This is an inherent property of wgpu-native's Metal backend, not fixable by shader optimization.
- A major restructuring (single command buffer per iteration, persistent compute passes) could help but would require rewriting the dispatch model. Consider this if WebGPU performance on Apple Silicon is critical.
- On discrete GPUs (NVIDIA Vulkan), the overhead should be lower since CPU→GPU latency is hidden by pipelining. The NVIDIA GB10 numbers (13.7s total) confirm this.

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

### Metal vs WebGPU (small dataset, trilinear, Apple M4 Pro)
- Metal matches WebGPU within 0.1% global NCC (0.9636 vs 0.9642).
- Rigid/affine matrices agree within ~0.005 rotation / ~0.3mm translation (expected float32 precision).
- Metal runs 2–8x faster than WebGPU on same hardware (7.2s vs 60s on small, 36s vs 89s on large) — Metal uses batched command buffers with zero-copy unified memory, while WebGPU has per-dispatch overhead.
- **Important metric note:** Use global NCC (Pearson correlation) for cross-backend comparison, not local NCC (kernel=5). Local NCC gives ~0.55-0.70, global NCC gives ~0.95-0.96. The old Metal metrics (NCC After 0.6771) used local NCC and were not comparable to CUDA/WebGPU global NCC.

## Metal Backend

Native Metal backend for macOS/Apple Silicon. Uses unified memory (MTLResourceStorageModeShared) for zero-copy CPU↔GPU access, MPSGraph for 3D FFT, and custom Metal compute shaders.

**Key files:**
- `backend/metal/metal_context.h/.m` — device init, pipeline cache, buffer tracking, dispatch
- `backend/metal/metal_kernels.h/.m` — kernel dispatch wrappers for all GPU operations
- `backend/metal/metal_fft.m` — MPSGraph FFT downsample + blur_downsample (trilinear)
- `backend/metal/linear_metal.m` — rigid/affine fused loops
- `backend/metal/greedy_metal.m` — greedy deformable loop
- `backend/metal/syn_metal.m` — SyN loop + warp inverse
- `backend/metal/shaders/*.metal` — 10 Metal Shading Language compute shader files

### Metal Current Metrics (small dataset, Apple M4 Pro)

| Config | NCC After (global) | Local NCC Loss | Time |
|--------|-------------------|----------------|------|
| Metal MI+trilinear | **0.9636** | -0.6664 | 7.2s |
| Metal CC+trilinear | 0.9536 | -0.5892 | 7.1s |
| WebGPU MI+trilinear | **0.9642** | -0.6686 | 59.9s |

### Metal Design Decisions

**Downsample modes:** Both FFT (MPSGraph) and trilinear (Gaussian blur + GPU resize) available via `opts.downsample_mode`. Trilinear uses existing `metal_conv1d_axis` (separable 1D blur) + `metal_trilinear_resize` (compute shader).

**Unified memory advantage.** All tensor data lives in `MTLResourceStorageModeShared` buffers. The CPU can read/write GPU buffer contents directly after `waitUntilCompleted`. No explicit H2D/D2H transfers needed.

**MI loss.** Uses CAS-based `atomic_uint` in threadgroup memory (Metal doesn't support `atomic<float>` in threadgroup), then native `device atomic<float>` for global merge. MI drives effective rigid/affine convergence on all tested datasets.

### Metal Known Issues

- `newLibraryWithFile:` deprecated warning (should use `newLibraryWithURL:`)
- Warp inverse runs 550 sequential Metal dispatches (one per iteration) — could be batched
- Pipeline cache stores string literal pointers — names must be static/literal
- Segfault on exit during Metal cleanup — cosmetic, does not affect results

## Dependencies

- **Core**: CMake >= 3.18, C11 compiler, zlib
- **CUDA backend**: CUDA toolkit, cufft
- **WebGPU backend**: wgpu-native (v27.0.4.0+, pre-downloaded in `third_party/wgpu/`)
- **Metal backend**: macOS 14.0+, Xcode with Metal 3.0, MetalPerformanceShadersGraph
- **FFT fallback**: kissfft (BSD-3, vendored in `third_party/kissfft/`) — used by WebGPU only
- Optional: zstd (for zstd-compressed NIfTI)

## Known Issues (CUDA)

- ~100 `cudaMalloc` calls lack error checking — needs a `CUDA_CHECK()` macro
- Duplicated kernels across files: permute (`dhw3↔3dhw`), `max_l2_norm`, `make_gpu_gauss` should be consolidated
- No `_default()` functions for `rigid_opts_t`, `affine_opts_t`, `greedy_opts_t`, `syn_opts_t`
- `beta2` differs between WarpAdam (0.99, for SyN/Greedy) and torch.optim.Adam (0.999, for rigid/affine) — intentional, matching Python defaults
