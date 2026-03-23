# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

Pure C port of [FireANTs](https://github.com/rohitrango/FireANTs) (commit `0d13a3f`). GPU-accelerated medical image registration: rigid, affine, SyN, and greedy deformable. Four backends: CUDA, Metal, WebGPU, CPU.

## Build

```bash
mkdir -p build && cd build
cmake .. -DCFIREANTS_METAL=ON    # or -DCFIREANTS_CUDA=ON / -DCFIREANTS_WEBGPU=ON
make -j8
```

Tests must run from repo root. WebGPU needs wgpu-native v27+ in `third_party/wgpu/`.

## Source Layout

```
src/main.c             CLI tool (cfireants_reg, ANTs-style arguments)
src/registration/      Moments, rigid, affine, greedy, syn (CPU implementations)
src/                   Core: tensor, nifti_io, image, interpolator, losses, utils
include/cfireants/     Public headers
backend/cuda/          CUDA fused kernels + registration loops
backend/webgpu/        WGSL shaders + wgpu-native dispatch
backend/metal/         Metal shaders + batched dispatch + MPSGraph FFT
tests/                 Validation and unit test programs
validate/              Datasets, benchmarks (see validate/README.md)
```

## Key Design Decisions

**No autograd.** Explicit forward + backward for each op. Chain: `loss → grid_sample_bwd → affine_grid_bwd → param_grad`.

**Loss functions:** MI for rigid/affine (robust for full-head), CC for deformable. MI uses Gaussian Parzen windowing, 32 bins.

**WarpAdam optimizer.** Greedy/SyN use compositive updates with beta2=0.99, gradient normalization by max L2 norm, half_res scaling. Rigid/affine use standard Adam with beta2=0.999.

**Downsample modes:** FFT (default, matches Python) or trilinear (GPU-native, `--trilinear`). Trilinear uses Gaussian blur + trilinear resize. Both produce equivalent accuracy.

**Fused CC loss.** Two CC implementations: `cc_loss.cu` (used by rigid/affine, 8 box-filter passes) and `fused_cc.cu` (used by SyN/greedy, 10 passes, computes both pred and target gradients). Using the wrong one causes gradient scaling issues.

## Non-Obvious Gotchas

- **NCC metric types.** Use global NCC (Pearson correlation) for cross-backend comparison, not local NCC (kernel=5). Local NCC gives ~0.55-0.70, global gives ~0.95-0.96. They are not comparable.

- **Image coordinate convention.** Internally uses SimpleITK/LPS convention. Output NIfTI must clone the fixed image's header via `image_save_like()` to preserve the original sform/qform orientation. Using `image_save()` produces LPS-convention headers that mismatch the input.

- **WebGPU batch hazards.** `wgpu_begin_batch()` does NOT guarantee execution order between dispatches that share buffers. The greedy compositive update (scale → compose → blur → copy) was broken by batching — required sequential dispatches. Metal batching IS ordered within a command buffer.

- **Metal `atomic<float>` limitation.** Not available in threadgroup memory on Metal Shading Language. MI histogram uses CAS-based `atomic_uint` in threadgroup, then `device atomic<float>` for global merge.

- **naga shader restrictions.** wgpu's Metal shader compiler rejects variable indexing of `array<T, N>`. Workaround: use `vec3/vec4` or `if/else` chains.

- **Skullstrip output.** When `--skullstrip` is used, `-o` is the output filename (not a prefix). Uses `image_skullstrip_save()` which re-loads the original NIfTI at native datatype.

- **Verbosity.** Global `cfireants_verbose` (0=silent, 1=summary, 2=debug). Default 0 for CLI, 2 for test programs. Check before fprintf in registration code.

## Backend-Specific Notes

**CUDA:** Production quality. Key files: `linear_gpu.cu`, `greedy_gpu.cu`, `syn_gpu.cu`, `downsample_fft.cu`, `mi_loss.cu`, `fused_cc.cu`. ~100 `cudaMalloc` calls lack error checking.

**Metal:** Uses batched command buffers (`metal_begin_batch/flush_batch`) to reduce dispatch overhead. GPU WarpAdam shaders (`warp_adam_moments`, `warp_adam_direction`) in `elementwise.metal`. 3D texture trilinear resize implemented but no speed benefit over compute shader.

**WebGPU:** Per-dispatch overhead ~0.1ms on Apple Silicon unified memory. SyN ~58s vs Metal ~7s on small dataset. Architectural limit of wgpu-native, not fixable by shader optimization. On discrete GPUs (NVIDIA Vulkan), overhead is hidden by pipelining.

**CPU:** Reference implementation matching GPU pipeline: blur+downsample, WarpAdam with compositive updates, warp inversion for SyN evaluation. `cpu_fused_cc_loss()` matches Metal/CUDA fused_cc gradient formula exactly.

## Dependencies

- **Core**: CMake >= 3.18, C11 compiler, zlib
- **CUDA**: CUDA toolkit, cufft
- **WebGPU**: wgpu-native v27.0.4.0+ in `third_party/wgpu/`
- **Metal**: macOS 14.0+, Xcode with Metal 3.0, MetalPerformanceShadersGraph
- Optional: zstd (for zstd-compressed NIfTI)

## Known Issues

- CUDA: ~100 unchecked `cudaMalloc` calls, duplicated kernels (permute, max_l2_norm, make_gpu_gauss)
- Metal: `newLibraryWithFile:` deprecated (should use `newLibraryWithURL:`)
- WebGPU: segfault on exit during wgpu-native cleanup (cosmetic)
- Pipeline caches store string literal pointers — names must be static/literal
- Code duplication: alloc/free helpers 4x across Metal files, gauss kernel 3x, dataset_t in tests
