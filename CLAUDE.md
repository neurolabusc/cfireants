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
backend/cuda/          CUDA kernels + registration loops
  cuda_common.cu       Shared helpers (vec_add, vec_scale, permute, max_l2_norm, make_gpu_gauss)
  fused_cc.cu          Fused CC loss + backward (used by SyN/greedy)
  cc_loss.cu           Regular CC loss (used by rigid/affine)
  warp_inverse.cu      Fixed-point warp inversion
backend/webgpu/        WGSL shaders + wgpu-native dispatch
backend/metal/         Metal shaders + batched dispatch + MPSGraph FFT
tests/                 Validation and unit test programs
validate/              Input datasets + Python reference outputs
```

## Key Design Decisions

**No autograd.** Explicit forward + backward for each op. Chain: `loss → grid_sample_bwd → affine_grid_bwd → param_grad`.

**Loss functions:** MI for rigid/affine (robust for full-head), CC for deformable. MI uses Gaussian Parzen windowing, 32 bins.

**WarpAdam optimizer.** Greedy/SyN use compositive updates with beta2=0.99, gradient normalization by max L2 norm, half_res scaling. Rigid/affine use standard Adam with beta2=0.999. Adam step counter (`step_t`) resets to 0 at each scale transition (matching Python's per-scale WarpAdam instantiation).

**Downsample modes:** FFT (default, matches Python) or trilinear (GPU-native, `--trilinear`). Trilinear uses Gaussian blur + trilinear resize. Both produce equivalent accuracy.

**Moments identity candidates.** When SVD eigenvalues are degenerate (condition ratio < 1.3), identity+COM and z-axis jiggle candidates are automatically evaluated alongside SVD candidates. `--try-identity` forces evaluation even when eigenvalues are well-separated.

## Critical Implementation Detail: Fused CC Gradient Scaling

The fused CC backward (`fused_cc.cu`, `losses.c:cpu_fused_cc_loss`, WebGPU/Metal equivalents) requires a `1/ks²` correction on the final gradient. This is the single most impactful correctness detail in the codebase.

**Why:** The NCC forward uses `A = kv * (mean_IJ - mean_I * mean_J)` where `kv = ks³` absorbs the mean-based box filter scaling. In the backward, the box filter adjoint introduces `1/kv` (one factor of `1/ks` per axis). The interaction between the `kv` in A,B,C and the `1/kv` from the adjoint leaves the gradient `ks²` too large. Dividing the final gradient by `ks²` after step 6 matches Python's `cc.py` autograd output exactly.

**Where it's applied:** AFTER the box filter adjoint (step 5) and final gradient assembly (step 6), NOT in `grad_output_val`. Applying it to `grad_output_val` changes the gradient multipliers before the adjoint, producing wrong results.

- CUDA: `fcc_scale_kernel(grad_pred, 1.0f/(ks*ks), spatial)` after step 6
- CPU: `gp[i] = (...) * inv_ks2` in step 6 loop
- WebGPU: `h_gp[i] = (...) * inv_ks2` in step 6 loop
- Metal: `metal_tensor_scale(grad_pred, inv_ks2, spatial)` after step 6

**Verification:** On identical images, C's CC loss matches Python to 0.5%. C's fused CC gradient (`d(CC)/d(pred)`) matches Python to 3%. Without the `1/ks²` fix, the gradient is `ks² ≈ 25x` too large, causing SyN to over-deform (scalp artifacts on full-head images).

## Non-Obvious Gotchas

- **Fused CC vs regular CC.** `fused_cc.cu` (SyN/greedy, computes both pred and target gradients) and `cc_loss.cu` (rigid/affine, pred gradient only) use different backward algorithms. The `1/ks²` correction applies only to fused CC. The regular CC backward already produces correct gradients.

- **NCC metric types.** Use global NCC (Pearson correlation) for cross-backend comparison, not local NCC (kernel=5). Local NCC gives ~0.55-0.70, global gives ~0.95-0.96. They are not comparable.

- **Image coordinate convention.** Internally uses SimpleITK/LPS convention. Output NIfTI must clone the fixed image's header via `image_save_like()` to preserve the original sform/qform orientation. Using `image_save()` produces LPS-convention headers that mismatch the input.

- **WebGPU batch hazards.** `wgpu_begin_batch()` does NOT guarantee execution order between dispatches that share buffers. The greedy compositive update (scale → compose → blur → copy) was broken by batching — required sequential dispatches. Metal batching IS ordered within a command buffer.

- **Metal `atomic<float>` limitation.** Not available in threadgroup memory on Metal Shading Language. MI histogram uses CAS-based `atomic_uint` in threadgroup, then `device atomic<float>` for global merge.

- **naga shader restrictions.** wgpu's Metal shader compiler rejects variable indexing of `array<T, N>`. Workaround: use `vec3/vec4` or `if/else` chains.

- **Skullstrip output.** When `--skullstrip` is used, `-o` is the skull-stripped output. Uses `image_skullstrip_save()` which re-loads the original NIfTI at native datatype.

- **Verbosity.** Global `cfireants_verbose` (0=silent, 1=summary, 2=debug). Default 0 for CLI, 2 for test programs. Check before fprintf in registration code.

## Backend-Specific Notes

**CUDA:** Production quality. Key files: `cuda_common.cu` (shared kernels), `linear_gpu.cu`, `greedy_gpu.cu`, `syn_gpu.cu`, `downsample_fft.cu`, `mi_loss.cu`, `fused_cc.cu`, `warp_inverse.cu`. Shared kernels (`cuda_vec_add`, `cuda_vec_scale`, `cuda_permute_*`, `cuda_max_l2_norm`, `cuda_make_gpu_gauss`) are in `cuda_common.cu` with C-callable wrappers declared in `kernels.h`.

**Metal:** Uses batched command buffers (`metal_begin_batch/flush_batch`) to reduce dispatch overhead. GPU WarpAdam shaders (`warp_adam_moments`, `warp_adam_direction`) in `elementwise.metal`. The `1/ks²` gradient fix uses `metal_tensor_scale` after step 6.

**WebGPU:** Per-dispatch overhead ~0.1ms on Apple Silicon. SyN ~58s vs Metal ~7s on small dataset. On discrete GPUs (NVIDIA Vulkan), overhead is hidden by pipelining. FFT downsampling uses kissfft CPU fallback (no GPU FFT in WGSL). The `1/ks²` gradient fix is applied in the CPU step 6 loop.

**CPU:** Reference implementation in `src/registration/`. Shared `cpu_warp_inverse()` in `src/utils.c` used by both CPU and WebGPU backends.

## Dependencies

- **Core**: CMake >= 3.18, C11 compiler, zlib
- **CUDA**: CUDA toolkit, cufft
- **WebGPU**: wgpu-native v27.0.4.0+ in `third_party/wgpu/`
- **Metal**: macOS 14.0+, Xcode with Metal 3.0, MetalPerformanceShadersGraph
- Optional: zstd (for zstd-compressed NIfTI)

## Cross-Platform Validation

### Current accuracy (March 2026)

Global NCC (Pearson correlation) between warped moving and fixed image:

| Dataset | Python CUDA | C CUDA | C WebGPU | C-Python gap |
|---------|-------------|--------|----------|--------------|
| small (2mm head) | 0.9450 | 0.9533 | 0.9548 | +0.8% |
| medium (1mm brain) | 0.9443 | 0.9469 | 0.9465 | +0.3% |
| large (1mm head) | 0.8961 | 0.8966 | 0.8886 | +0.05% |

C matches Python within 0.8% on all datasets. The remaining gap comes from Python using non-separable 3D `F.conv3d` for CC box filtering vs C's 3 separable 1D passes — mathematically identical but different float32 accumulation order compounds over SyN iterations.

### Validation commands

**Python reference** (requires `pip install fireants` and fused_ops):
```bash
PYTHONPATH=/path/to/FireANTs/fused_ops/src:$PYTHONPATH python validate/run_validation.py --dataset small --save-reference
```

**C backends** (from repo root):
```bash
# Small (default params: MI for rigid/affine, CC for SyN)
cfireants_reg -f validate/small/MNI152_T1_2mm.nii.gz -m validate/small/T1_head_2mm.nii.gz \
  -v 2 -o test/small_syn.nii.gz

# Medium (CC for all stages)
cfireants_reg -f validate/medium/MNI152_T1_1mm_brain.nii.gz -m validate/medium/t1_brain.nii.gz \
  --transform 'Rigid[0.003]' --metric 'CC[5]' --convergence '[200x100x50,1e-6,10]' --shrink-factors 4x2x1 \
  --transform 'Affine[0.001]' --metric 'CC[5]' --convergence '[200x100x50,1e-6,10]' --shrink-factors 4x2x1 \
  --transform 'SyN[0.1,0.5,1.0]' --metric 'CC[5]' --convergence '[200x100x50,1e-6,10]' --shrink-factors 4x2x1 \
  -v 2 -o test/medium_syn.nii.gz

# Large (MI for rigid/affine, CC for SyN, 4 scales for linear)
cfireants_reg -f validate/large/MNI152_T1_1mm.nii.gz -m validate/large/chris_t1.nii.gz \
  --transform 'Rigid[0.003]' --metric 'MI[32]' --convergence '[200x200x100x50,1e-6,10]' --shrink-factors 8x4x2x1 \
  --transform 'Affine[0.001]' --metric 'MI[32]' --convergence '[200x200x100x50,1e-6,10]' --shrink-factors 8x4x2x1 \
  --transform 'SyN[0.1,0.5,1.0]' --metric 'CC[5]' --convergence '[200x100x50,1e-6,10]' --shrink-factors 4x2x1 \
  -v 2 -o test/large_syn.nii.gz

# Add --backend webgpu or --backend metal for other backends
# Add --init-affine <file> to override the physical affine (test SyN with a known affine)
```

### Checklist for Metal convergence

When bringing Metal to parity with CUDA/WebGPU, verify these in order:

1. **Moments**: Should match exactly (no GPU-specific code). Check SVD candidate selection and COM values.

2. **Fused CC gradient (1/ks² fix)**: Verify `metal_fused_cc_loss` applies `metal_tensor_scale(grad_pred, 1.0f/(ks*ks), spatial)` AFTER step 6 (fcc_bwd_grads dispatch), not inside `grad_output_val`. This is the most common source of quality divergence.

3. **Adam step_t reset**: Verify `fwd_step = 0; rev_step = 0;` when optimizer state is freed at scale transitions in both `syn_metal.m` and `greedy_metal.m`.

4. **Warp inverse**: Should use fixed-point iteration (`inv = -interp(u, id+inv)`, 550 iters). NOT Adam-based IC optimization.

5. **Box filter precision**: Metal's `box_filter_axis` shader should accumulate then scale (not multiply per-element during accumulation) to match CUDA/WebGPU precision.

6. **Per-stage NCC**: Run with `-v 2` and compare per-stage local NCC values against CUDA reference. They should agree within 1%.

7. **Global NCC**: Compare final warped image against Python reference in `validate/reference/`. Should be within 1% of Python.

### Validate directory structure

```
validate/
  run_validation.py    Python reference pipeline (FireANTs)
  reference/           Python reference outputs (warped.nii.gz + metrics.json per dataset)
  small/               Input pair: MNI152_T1_2mm.nii.gz + T1_head_2mm.nii.gz
  medium/              Input pair: MNI152_T1_1mm_brain.nii.gz + t1_brain.nii.gz
  large/               Input pair: MNI152_T1_1mm.nii.gz + chris_t1.nii.gz
  skulllstrip/         Skull-strip mask test data
```

Output directories (`validate/*/output/`) are gitignored — regenerated by running registration.

## Known Issues

- CUDA: ~100 unchecked `cudaMalloc` calls
- Metal: `newLibraryWithFile:` deprecated (should use `newLibraryWithURL:`)
- WebGPU: segfault on exit during wgpu-native cleanup (cosmetic)
- Pipeline caches store string literal pointers — names must be static/literal
- Code duplication: alloc/free helpers 4x across Metal files, dataset_t in tests
