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

**Fused CC loss.** Two CC implementations: `cc_loss.cu` (used by rigid/affine, 8 box-filter passes) and `fused_cc.cu` (used by SyN/greedy, 10 passes, computes both pred and target gradients). The fused CC backward multiplies `grad_output_val` by `kernel_volume` to compensate for the mean-based box filter adjoint (Python uses sum-based separable filtering whose adjoint preserves magnitude).

**Moments identity candidates.** `--try-identity` enables identity+COM and pure-identity candidates in moments (off by default to match Python). Useful when sforms already align images.

## Non-Obvious Gotchas

- **NCC metric types.** Use global NCC (Pearson correlation) for cross-backend comparison, not local NCC (kernel=5). Local NCC gives ~0.55-0.70, global gives ~0.95-0.96. They are not comparable.

- **Image coordinate convention.** Internally uses SimpleITK/LPS convention. Output NIfTI must clone the fixed image's header via `image_save_like()` to preserve the original sform/qform orientation. Using `image_save()` produces LPS-convention headers that mismatch the input.

- **WebGPU batch hazards.** `wgpu_begin_batch()` does NOT guarantee execution order between dispatches that share buffers. The greedy compositive update (scale → compose → blur → copy) was broken by batching — required sequential dispatches. Metal batching IS ordered within a command buffer.

- **Metal `atomic<float>` limitation.** Not available in threadgroup memory on Metal Shading Language. MI histogram uses CAS-based `atomic_uint` in threadgroup, then `device atomic<float>` for global merge.

- **naga shader restrictions.** wgpu's Metal shader compiler rejects variable indexing of `array<T, N>`. Workaround: use `vec3/vec4` or `if/else` chains.

- **Skullstrip output.** When `--skullstrip` is used, `-o` is the skull-stripped output. Uses `image_skullstrip_save()` which re-loads the original NIfTI at native datatype.

- **Verbosity.** Global `cfireants_verbose` (0=silent, 1=summary, 2=debug). Default 0 for CLI, 2 for test programs. Check before fprintf in registration code.

## Backend-Specific Notes

**CUDA:** Production quality. Key files: `linear_gpu.cu`, `greedy_gpu.cu`, `syn_gpu.cu`, `downsample_fft.cu`, `mi_loss.cu`, `fused_cc.cu`, `warp_inverse.cu`. ~100 `cudaMalloc` calls lack error checking.

**Metal:** Uses batched command buffers (`metal_begin_batch/flush_batch`) to reduce dispatch overhead. GPU WarpAdam shaders (`warp_adam_moments`, `warp_adam_direction`) in `elementwise.metal`. 3D texture trilinear resize implemented but no speed benefit over compute shader.

**WebGPU:** Per-dispatch overhead ~0.1ms on Apple Silicon unified memory. SyN ~58s vs Metal ~7s on small dataset. Architectural limit of wgpu-native, not fixable by shader optimization. On discrete GPUs (NVIDIA Vulkan), overhead is hidden by pipelining.

**CPU:** Reference implementation matching GPU pipeline: blur+downsample, WarpAdam with compositive updates, warp inversion for SyN evaluation. `cpu_fused_cc_loss()` matches Metal/CUDA fused_cc gradient formula exactly.

## Dependencies

- **Core**: CMake >= 3.18, C11 compiler, zlib
- **CUDA**: CUDA toolkit, cufft
- **WebGPU**: wgpu-native v27.0.4.0+ in `third_party/wgpu/`
- **Metal**: macOS 14.0+, Xcode with Metal 3.0, MetalPerformanceShadersGraph
- Optional: zstd (for zstd-compressed NIfTI)

## Cross-Platform Validation Plan

### Current accuracy (March 2026)

Global NCC (Pearson correlation) between warped moving and fixed image:

| Dataset | Python CUDA | C CUDA | C WebGPU | CUDA-WebGPU gap |
|---------|-------------|--------|----------|-----------------|
| small (2mm head) | 0.9450 | **0.9623** | **0.9632** | 0.08% |
| medium (1mm brain) | 0.9443 | **0.9585** | **0.9587** | 0.02% |
| large (1mm head) | 0.8961 | **0.9216** | **0.9211** | 0.05% |

### Per-stage metrics (C CUDA, current build)

**Small** (`-f validate/small/MNI152_T1_2mm.nii.gz -m validate/small/T1_head_2mm.nii.gz`):
```
Moments:  local NCC -0.2250  (SVD candidate 6 wins, COM shift ~28mm z)
Rigid:    local NCC -0.2413  (MI, 3 scales 4x2x1, 200/100/50 iters)
Affine:   local NCC -0.3137  (MI, 3 scales)
SyN s4:   CC -0.671 → -0.864  (200 iters, 22x27x22)
SyN s2:   CC -0.585 → -0.789  (100 iters, 45x54x45)
SyN s1:   CC -0.440 → -0.620  (50 iters, 91x109x91)
Eval:     local NCC -0.6600 (kernel=9)  → global NCC 0.9623
```

**Medium** (`-f validate/medium/MNI152_T1_1mm_brain.nii.gz -m validate/medium/t1_brain.nii.gz`, CC for all stages):
```
Moments:  local NCC -0.7531
Rigid:    local NCC -0.7182  (CC, 3 scales)
Affine:   local NCC -0.7714  (CC, 3 scales)
SyN eval: local NCC -0.8592  → global NCC 0.9562
```

**Large** (`-f validate/large/MNI152_T1_1mm.nii.gz -m validate/large/chris_t1.nii.gz`):
```
Moments:  local NCC -0.1315  (SVD candidate 6 wins, COM shift ~10mm z)
Rigid:    local NCC -0.1119  (MI, 4 scales 8x4x2x1, 200/200/100/50 iters)
Affine:   local NCC -0.1719  (MI, 4 scales)
SyN s4:   CC -0.360 → -0.610  (200 iters, 45x54x45)
SyN s2:   CC -0.346 → -0.485  (100 iters, 91x109x91)
SyN s1:   CC -0.284 → -0.349  (50 iters, 182x218x182)
Eval:     local NCC -0.3892 (kernel=9)  → global NCC 0.9216
```

### What has been fixed

- **Fused CC gradient magnitude**: backward pass was missing `kernel_volume` scaling factor — the mean-based box filter adjoint introduced 1/kv, making gradients 125x too small. Fixed by multiplying `grad_output_val` by `kernel_volume` in all 4 backends.
- **Moments identity candidates**: identity+COM and pure-identity candidates caused large dataset to pick wrong starting orientation. Now opt-in via `--try-identity` (off by default to match Python).
- **Warp inverse parity**: CUDA switched from Adam-based IC optimization to fixed-point iteration (`inv = -interp(u, id+inv)`) matching WebGPU/CPU, giving <0.1% cross-backend gap.
- **CPU composition bug**: fixed (ux was used for all 3 channels in `syn_evaluate`)

### What has been ruled out

- **beta2**: 0.99 vs 0.999 has negligible effect (Adam adapts)
- **FFT vs trilinear downsample**: FFT slightly better, neither closes gap
- **Iteration count**: 200/200/200 for SyN is no better (oscillation at scale 1)
- **Shrink factors**: 3-scale vs 4-scale identical for large

### Convergence plan

**Phase 1: CUDA vs Python** — DONE (March 2026)
C CUDA exceeds Python on all 3 datasets by 1.5-2.5%. The remaining gap is due to Python's non-separable 3D F.conv3d for CC box filter vs C's 3 separable 1D passes — mathematically identical but different float accumulation order compounds over iterations.

**Phase 2: WebGPU vs CUDA** — DONE (March 2026)
WebGPU matches CUDA within 0.1% on all datasets. Residual differences from cuFFT vs kissfft CPU fallback and CUDA vs Vulkan shader float precision.

**Phase 3: Metal convergence** — TODO
1. Compare Metal output to CUDA/WebGPU reference
2. Metal-specific areas to check: FFT downsample (MPSGraph), fused CC shader, grid_sample, blur kernel

### Debug metrics mode (TODO)

Add `--metrics <file.json>` flag that saves per-stage diagnostics:

```json
{
  "moments": {"ncc": -0.134, "rotation": "pure_identity", "translation": [0, 0, 0]},
  "stages": [
    {
      "type": "Rigid", "ncc": -0.126,
      "affine": [[0.92, -0.02, -0.005, 0.88], [0.01, 1.01, 0.07, -1.36], [0.009, -0.06, 0.94, -6.43]],
      "per_scale": [
        {"scale": 8, "loss_start": -0.003, "loss_end": -0.004, "iters": 200},
        {"scale": 4, "loss_start": -0.004, "loss_end": -0.004, "iters": 200}
      ]
    },
    {
      "type": "SyN", "ncc": -0.263, "global_ncc": 0.851,
      "per_scale": [
        {"scale": 4, "loss_start": -0.359, "loss_end": -0.635, "iters": 200,
         "gradmax_first": 1.23, "gradmax_last": 0.87, "disp_max_l2": 0.045},
        {"scale": 2, "loss_start": -0.361, "loss_end": -0.498, "iters": 100},
        {"scale": 1, "loss_start": -0.293, "loss_end": -0.354, "iters": 50}
      ]
    }
  ]
}
```

This enables regression testing: if a code change moves any metric by more than a threshold, the test fails. It also enables cross-backend comparison by diffing the JSON files.

### Commands for CUDA validation

**Python reference** (from repo root, requires `pip install fireants`):
```bash
python validate/run_validation.py --dataset small
python validate/run_validation.py --dataset medium
python validate/run_validation.py --dataset large
# Outputs: validate/{small,medium,large}/output/{warped.nii.gz,metrics.json}
```

**C CUDA**:
```bash
# Small
cfireants_reg -f validate/small/MNI152_T1_2mm.nii.gz -m validate/small/T1_head_2mm.nii.gz \
  -v 2 -o test/small_syn.nii.gz

# Medium
cfireants_reg -f validate/medium/MNI152_T1_1mm_brain.nii.gz -m validate/medium/t1_brain.nii.gz \
  --transform 'Rigid[0.003]' --metric 'CC[5]' --convergence '[200x100x50,1e-6,10]' --shrink-factors 4x2x1 \
  --transform 'Affine[0.001]' --metric 'CC[5]' --convergence '[200x100x50,1e-6,10]' --shrink-factors 4x2x1 \
  --transform 'SyN[0.1,0.5,1.0]' --metric 'CC[5]' --convergence '[200x100x50,1e-6,10]' --shrink-factors 4x2x1 \
  -v 2 -o test/medium_syn.nii.gz

# Large
cfireants_reg -f validate/large/MNI152_T1_1mm.nii.gz -m validate/large/chris_t1.nii.gz \
  --transform 'Rigid[0.003]' --metric 'MI[32]' --convergence '[200x200x100x50,1e-6,10]' --shrink-factors 8x4x2x1 \
  --transform 'Affine[0.001]' --metric 'MI[32]' --convergence '[200x200x100x50,1e-6,10]' --shrink-factors 8x4x2x1 \
  --transform 'SyN[0.1,0.5,1.0]' --metric 'CC[5]' --convergence '[200x100x50,1e-6,10]' --shrink-factors 4x2x1 \
  -v 2 -o test/wchris_t1.nii.gz
```

**C WebGPU** (same commands with `--backend webgpu`)

## Known Issues

- CUDA: ~100 unchecked `cudaMalloc` calls, duplicated kernels (permute, max_l2_norm, make_gpu_gauss)
- Metal: `newLibraryWithFile:` deprecated (should use `newLibraryWithURL:`)
- WebGPU: segfault on exit during wgpu-native cleanup (cosmetic)
- Pipeline caches store string literal pointers — names must be static/literal
- Code duplication: alloc/free helpers 4x across Metal files, gauss kernel 3x, dataset_t in tests
