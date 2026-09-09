# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

Pure C port of [FireANTs](https://github.com/rohitrango/FireANTs) (commit `0d13a3f`). GPU-accelerated medical image registration: rigid, affine, SyN, and greedy deformable. Four backends: CUDA, Metal, WebGPU, CPU. Also builds to WebAssembly and ships as an npm package in `js/`; the reference consumer is [edgefire](https://github.com/rordenlab/edgefire).

## Build

```bash
mkdir -p build && cd build
cmake .. -DCFIREANTS_METAL=ON    # or -DCFIREANTS_CUDA=ON / -DCFIREANTS_WEBGPU=ON
make -j8
```

Options: `CFIREANTS_THREADS` (pthread CPU pool, default ON), `CFIREANTS_ZLIB` (gzip NIfTI,
default ON; the WASM build turns it OFF and handles gzip in TypeScript).

Thread count comes from the online processor count, overridden by `CFIREANTS_NUM_THREADS`
or by `--threads N`. Both exist because the env var does not work under Emscripten.

`make macos-release` (root `Makefile`) builds a notarized Apple Silicon `.pkg` via
`scripts/package_macos.sh`. It configures its own `build-macos/` with
`CFIREANTS_EMBED_METALLIB=ON`, `CFIREANTS_ZSTD=OFF` (Homebrew libzstd is not on
the target machine) and a 14.0 deployment target, then asserts the linked binary
has no dependency outside `/usr/lib` and `/System`. See README.md.

Tests must run from repo root (dataset paths); `ctest --test-dir build` does this for you
(six tests: `test_phase2..6`, `test_backend_parity`; the parity test returns 77 = skip when
no GPU initialises). `make macos-release` and `.github/workflows/release-npm.yml` are the
two release paths: the `.pkg` is built locally because it needs signing certificates, the
npm tarball is attached to a GitHub release by CI on any `v*` tag. WebGPU needs wgpu-native v27+ in
`third_party/wgpu/`; its shaders are compiled into the binary, so the binary itself no
longer cares about the working directory.

## Source Layout

```
src/main.c             CLI tool (cfireants_reg, ANTs-style arguments)
src/registration/      Moments, rigid, affine, greedy, syn (CPU implementations)
src/                   Core: tensor, nifti_io, image, interpolator, losses, utils, threading
include/cfireants/     Public headers (incl. threading.h)
backend/cuda/          CUDA kernels + registration loops
  cuda_common.cu       Shared helpers (vec_add, vec_scale, permute, max_l2_norm, make_gpu_gauss)
  fused_cc.cu          Fused CC loss + backward (used by CUDA SyN/greedy)
  cc_loss.cu           Regular CC loss (used by rigid/affine)
  warp_inverse.cu      Fixed-point warp inversion
backend/webgpu/        WGSL shaders + wgpu-native dispatch
  shaders/*.wgsl       Shader source of record
  embedded_shaders.h   Table type; the .c is generated into build/generated/ by CMake
backend/metal/         Metal shaders + batched dispatch + MPSGraph FFT
scripts/               embed_shaders.py (WGSL), embed_metallib.py
js/                    WASM build script, TypeScript wrapper, npm package
tests/                 Validation and unit test programs
validate/              Input datasets + Python reference outputs
```

## Key Design Decisions

**No autograd.** Explicit forward + backward for each op. Chain: `loss → grid_sample_bwd → affine_grid_bwd → param_grad`.

**Loss functions:** MI for rigid/affine (robust for full-head), CC for deformable. MI uses Gaussian Parzen windowing, 32 bins. On data that is brain-extracted on both sides, CC at *every* stage is both better and faster (edgefire measures 0.9616 vs 0.9410 and 2.2x faster), so callers with stripped data should override the default.

**WarpAdam optimizer.** Greedy/SyN use compositive updates with beta2=0.99, gradient normalization by max L2 norm, half_res scaling. Rigid/affine use standard Adam with beta2=0.999. Adam step counter (`step_t`) resets to 0 at each scale transition (matching Python's per-scale WarpAdam instantiation).

**Downsample modes:** trilinear is the default on every backend (Gaussian blur + trilinear resize) and `DOWNSAMPLE_TRILINEAR` is 0, so a zero-initialised options struct gets the same pyramid as the CLI. FFT (matching the Python default) is the explicit GPU-only `--fft` option. There is no CPU FFT downsample: all four CPU stages (`rigid.c`, `affine.c`, `greedy.c`, `syn.c`) call `cpu_blur_downsample` unconditionally. Medium dataset, Metal: FFT 0.9457 vs trilinear 0.9454 global NCC — equivalent.

**Shrink factors are defined on the fixed image.** `moving_pyramid_size()` in `src/utils.c` converts a fixed-grid shrink factor into the moving image's own factor via the spacing ratio, so a 2mm moving against a 1mm fixed is not driven to 8mm at scale 4. Applied in CPU rigid/affine/greedy and in WebGPU linear/greedy; **deliberately not yet in Metal or CUDA** (see Known Issues). SyN never downsamples the moving image, so it is correctly exempt everywhere.

**Moments orientation.** Defaults to proper rotations only (`--orientation rot`); `both` also admits mirror-image candidates and used to be the default.

The improper (negative-determinant) candidates only ever compensated for a *voxel storage-order* mismatch, not for anatomy. Losslessly reorienting the moving image into the fixed image's axis order swaps the outcome: proper-only goes 0.9343 → 0.9573, proper-plus-improper goes 0.9572 → 0.9340. A mirrored transform is never anatomically right. The intended direction is to reorient at load and delete the improper candidate set; `--orientation` survives only because removing it before reorientation lands would leave mismatched data with no recourse.

**Moments identity candidates.** When SVD eigenvalues are degenerate (condition ratio < 1.3), identity+COM and z-axis jiggle candidates are automatically evaluated alongside SVD candidates. `--try-identity` forces evaluation even when eigenvalues are well-separated.

**Threaded CPU.** `src/threading.c` is a bounded pthread pool, restartable after
`cfireants_cleanup()` (a mutex-guarded `started` flag, not `pthread_once`), driven through
`cfireants_parallel_for(count, min_items_per_thread, fn, ctx)`; small jobs and
`-DCFIREANTS_THREADS=OFF` builds run synchronously. Used by the interpolator, losses,
utils and the registration stages. No per-thread copies of image data. Small 2mm pair on
14 cores: 59.3s single-threaded, 9.1s threaded.

## Critical Implementation Detail: Fused CC Gradient Scaling

The fused CC backward (`fused_cc.cu`, `losses.c:cpu_fused_cc_loss`, WebGPU/Metal equivalents) requires a `1/ks²` correction on the final gradient. This is the single most impactful correctness detail in the codebase.

**Why:** The NCC forward uses `A = kv * (mean_IJ - mean_I * mean_J)` where `kv = ks³` absorbs the mean-based box filter scaling. In the backward, the box filter adjoint introduces `1/kv` (one factor of `1/ks` per axis). The interaction between the `kv` in A,B,C and the `1/kv` from the adjoint leaves the gradient `ks²` too large. Dividing the final gradient by `ks²` after step 6 matches Python's `cc.py` autograd output exactly.

**Where it's applied:** AFTER the box filter adjoint (step 5) and final gradient assembly (step 6), NOT in `grad_output_val`. Applying it to `grad_output_val` changes the gradient multipliers before the adjoint, producing wrong results.

- CUDA: `fcc_scale_kernel(grad_pred, 1.0f/(ks*ks), spatial)` after step 6
- CPU: `gp[i] = (...) * inv_ks2` in step 6 loop
- WebGPU: `Params.inv_ks2` inside `fused_cc_bwd_grads.wgsl` (the step 6 work is on the GPU now, not in a CPU loop)
- Metal: `metal_tensor_scale(grad_pred, inv_ks2, spatial)` after step 6

**Verification:** On identical images, C's CC loss matches Python to 0.5%. C's fused CC gradient (`d(CC)/d(pred)`) matches Python to 3%. Without the `1/ks²` fix, the gradient is `ks² ≈ 25x` too large, causing SyN to over-deform (scalp artifacts on full-head images).

## Non-Obvious Gotchas

- **Fused CC vs regular CC.** SyN uses fused CC because it needs both pred and target gradients. Rigid, affine, and the CPU/WebGPU/Metal Greedy paths use regular CC because they need only the pred gradient. CUDA Greedy still uses fused CC. The two paths have different backward algorithms; the `1/ks²` correction applies only to fused CC. The regular CC backward already produces correct gradients. Do not substitute fused CC into a one-sided Greedy path: even though max-L2 normalization hides uniform gradient scaling, its different gradient direction measurably hurts convergence.

- **NCC metric types.** Use global NCC (Pearson correlation) for cross-backend comparison, not local NCC (kernel=5). Local NCC gives ~0.55-0.70, global gives ~0.95-0.96. They are not comparable.

- **Image coordinate convention.** Internally uses SimpleITK/LPS convention. Output NIfTI must clone the fixed image's header via `image_save_like()` to preserve the original sform/qform orientation. Using `image_save()` produces LPS-convention headers that mismatch the input.

- **`image_save_like()` must zero `cal_min`/`cal_max`.** It clones the template header, and cloning the template's display window onto warped data (MNI152's 3000..8000 against subject intensities) renders the output *blank* in any viewer that honours it. Fixed in `src/image.c`; do not reintroduce it by copying more header fields.

- **Browser WebGPU enforces baseline limits; wgpu-native does not.** `maxStorageBuffersPerShaderStage` is 8 in the WebGPU baseline. Two CC entry points bound 9, which validates natively but is silently invalid in a browser: the pipeline never runs and the loss reads uninitialised memory, surfacing as plausible-looking numbers rather than an exception. **Count storage bindings per entry point against 8, not against what the adapter offers.** The fix was packing three gradient buffers into one `grad_sources` with offsets. Separately, the baseline `maxStorageBufferBindingSize` of 128 MiB is too small for a 1mm template (needs 144,420,640 bytes), so `js/src/run.ts` requests the adapter's own size limits — sizes are raised, counts are deliberately left at baseline.

- **WebGPU batching is ordered now.** `wgpu_begin_batch()` keeps every dispatch in one compute pass, which the spec orders. It used to open a pass per dispatch, and the greedy compositive update (scale → compose → blur → copy) broke under that; `CFIREANTS_WGPU_NO_BATCH` still exists for bisecting. Metal batching is ordered within a command buffer.

- **Metal `atomic<float>` limitation.** Not available in threadgroup memory on Metal Shading Language. MI histogram uses CAS-based `atomic_uint` in threadgroup, then `device atomic<float>` for global merge.

- **Never use 64-bit integer division in a Metal kernel.** It is emulated and catastrophically slow: it made SyN *slower* than the pre-optimisation baseline (10.9s vs 8.0s). 32-bit index arithmetic fixed it.

- **`metal_flush_batch()`'s `waitUntilCompleted` is load-bearing.** Removing it to let the CPU run ahead made SyN nondeterministic (stage NCC -0.8151 became -0.7822/-0.7811 across runs). Do not "optimise" it away.

- **Metal dispatches must be padded to whole threadgroups.** `dispatchThreads` leaves the final group partial, while every reduction kernel reads all 256 shared-memory slots.

- **naga shader restrictions.** wgpu's Metal shader compiler rejects variable indexing of `array<T, N>`. Workaround: use `vec3/vec4` or `if/else` chains.

- **Skullstrip output.** When `--skullstrip` is used, `-o` is the skull-stripped output. Uses `image_skullstrip_save()` which re-loads the original NIfTI at native datatype.

- **Verbosity.** Global `cfireants_verbose` (0=silent, 1=summary, 2=debug). Default 0 for CLI, 2 for test programs. Check before fprintf in registration code.

## Backend-Specific Notes

**CUDA:** Production quality. Key files: `cuda_common.cu` (shared kernels), `linear_gpu.cu`, `greedy_gpu.cu`, `syn_gpu.cu`, `downsample_fft.cu`, `mi_loss.cu`, `fused_cc.cu`, `warp_inverse.cu`. Shared kernels (`cuda_vec_add`, `cuda_vec_scale`, `cuda_permute_*`, `cuda_max_l2_norm`, `cuda_make_gpu_gauss`) are in `cuda_common.cu` with C-callable wrappers declared in `kernels.h`.

**Metal:** Uses batched command buffers (`metal_begin_batch/flush_batch`) to reduce dispatch
overhead. GPU WarpAdam shaders (`warp_adam_moments`, `warp_adam_direction`) in
`elementwise.metal`. The `1/ks²` gradient fix uses `metal_tensor_scale` after step 6.
The fused CC box filter filters all five channels in one dispatch per axis
(`box_filter_axis_packed` + `copy_f32` in `shaders/grid_sample.metal`). The fused-CC
scratch buffers in `syn_metal.m` are sized `5L * spatial` for this — a requirement, not an
incidental change. Metal Greedy uses regular CC and does not allocate fused-CC workspace.

**WebGPU:** GPU-resident. The fused CC forward, `bwd_modify` and `bwd_grads` all live in
WGSL (`fused_cc_fwd.wgsl`, `fused_cc_box.wgsl`, `fused_cc_bwd_*.wgsl`), and MI gained
`mi_prepare`/`mi_max`. The fused CC used to perform 17 full-volume readbacks per call
against 8 dispatches; it now performs one small partial-sums reduction for the loss scalar.
`--fft` downsampling and warp inversion still fall back to the CPU, which the pthread pool
parallelises. Regular CC (rigid/affine/greedy) takes a per-scale `wgpu_cc_workspace_t`; a
NULL workspace allocates for one call. Convergence is checked every iteration, as on CPU. The `1/ks²` gradient fix lives in `fused_cc_bwd_grads.wgsl`. Greedy's
gradient normalisation is device-side (`wgpu_normalize_l2_buf` with a persistent state
buffer), so there is no per-iteration readback. It also runs in browsers, via
`js/wasm/cfireants-gpu`.

**Shaders are compiled in, and only compiled in.** Every live shader has exactly one
source, its `.wgsl` file (27 files, 30 entry points, ≤7 storage bindings each).
`scripts/embed_shaders.py` generates `build/generated/embedded_shaders.c`, and CMake
regenerates it whenever a
`.wgsl` changes. `shader_loader.h` used to read `backend/webgpu/shaders/<name>` from disk
first when run natively, so the shader tested from the repo root was a different copy
from the one shipped; that path is gone. Do not reintroduce a runtime file lookup.

**CPU:** Reference implementation in `src/registration/`. Shared `cpu_warp_inverse()` in `src/utils.c` used by both CPU and WebGPU backends. Parallelised by the pthread pool.

## WASM / npm build

`js/build-wasm.sh` produces three Emscripten modules into `js/wasm/`:

| Variant | Needs | Browser, small 2mm pair |
|---------|-------|-------------------------|
| `cfireants-mt` | SharedArrayBuffer + cross-origin isolation | 13.6s (14 threads) |
| `cfireants` | nothing | 84.9s (1 thread) |
| `cfireants-gpu` | `navigator.gpu`; **no isolation required** | 15.7s |

CPU output is byte-identical between the two CPU variants and matches the native CPU
backend to 0.99994 correlation. The npm package in `js/` is 220 kB packed / 631 kB
unpacked.

- **Two CPU variants exist because `-pthread` requires COOP/COEP.** Without cross-origin isolation a `-pthread` module *hangs* rather than failing, so the JS wrapper must pick the variant from `crossOriginIsolated` before loading.
- **No zlib.** Built with `-DCFIREANTS_ZLIB=OFF`; the C reads and writes plain `.nii`. Gzip is handled in the TypeScript wrapper via `DecompressionStream`. The zlib and pigz code was already behind `HAVE_ZLIB`/`PIGZ`, so nothing needed removing.
- **`-sPROXY_TO_PTHREAD` makes `callMain` return early.** Callers must await `Module.onExit` before reading output, or they read a file that does not exist yet. It is also why `--threads` exists: the env var is materialised before JS can set it.
- **The GPU build has a fixed 1 GiB heap.** `--use-port=emdawnwebgpu` is incompatible with `ALLOW_MEMORY_GROWTH` (a growable heap is a resizable `ArrayBuffer`, and shader creation then fails reading its own WGSL). The value was picked, not measured; measure before changing it.
- **No bespoke C API.** The wrapper drives the existing `main(argc, argv)` through `callMain` with MEMFS files.
- **Peak memory is the whole process footprint** — both images, every pyramid level, the
  displacement fields, Adam moments and CC intermediates. Quote it as a total; the numbers
  below are peak resident set for the entire run, not one allocation.

  What the *fixed* image drives is how that total **scales**, because the displacement
  fields, Adam moments and CC intermediates are all allocated in the fixed image's grid at
  each scale. The moving image contributes its own storage and its pyramid, which is why
  swapping a 1mm moving volume for a 256³ one moves the total far less than doing the same
  to the fixed image.

  | Fixed | Moving | Stage | Peak total |
  |-------|--------|-------|-----------|
  | 1mm template | 1mm brain | Greedy | 1.38 GB |
  | 1mm template | 1mm brain | SyN | ~2.0 GB |
  | 1mm template | 256³ | SyN | 1.92 GB |
  | 256³ | 1mm brain | SyN | 3.88 GB |

  The last row sits against the wasm32 4 GB ceiling with no headroom. Keep the fixed image
  at template size and feed large volumes as moving.

## Dependencies

- **Core**: CMake >= 3.18, C11 compiler; zlib optional (`CFIREANTS_ZLIB`, on by default)
- **CUDA**: CUDA toolkit, cufft
- **WebGPU**: wgpu-native v27.0.4.0+ in `third_party/wgpu/`
- **Metal**: macOS 14.0+, Xcode with Metal 3.0, MetalPerformanceShadersGraph
- **WASM**: emscripten on PATH
- Optional: zstd (for zstd-compressed NIfTI)

## Cross-Platform Validation

### Accuracy (last measured September 2026)

Global NCC (Pearson correlation) between warped moving and fixed image:

| Dataset | Python CUDA | C CUDA | C WebGPU | C-Python gap |
|---------|-------------|--------|----------|--------------|
| small (2mm head) | 0.9450 | 0.9533 | 0.9548 (`--orientation both`) | +0.8% |
| medium (1mm brain) | 0.9443 | 0.9469 | 0.9465 | +0.3% |
| large (1mm head) | 0.8961 | 0.8966 | 0.8886 | +0.05% |

C matches Python within 0.8% on all datasets. The remaining gap comes from Python using non-separable 3D `F.conv3d` for CC box filtering vs C's 3 separable 1D passes — mathematically identical but different float32 accumulation order compounds over SyN iterations.

**Small-dataset numbers above need `--orientation both`.** With the current `rot` default
the small pair lands at 0.934–0.935 on every backend ("Moments orientation" above explains
why; reorienting at load is the fix). Medium is unaffected: last measured 0.9456 CPU /
0.9454 Metal / 0.9452 WebGPU, all within 0.05% of each other.

**Small-dataset results are not deterministic.** The MI histogram uses float atomics; two
identical runs gave 0.9539 and 0.9533. Use the medium (CC-only) path as the regression
target when a number has to be compared exactly — Metal is byte-identical run to run there.

Effect of the fixed-vs-moving shrink factor fix, 2mm moving against a 1mm fixed:
0.7617 global / 0.2461 within brain before, 0.9657 / 0.7093 after.

### Native timing, Apple Silicon (14 cores)

| Dataset | Metal | WebGPU | CPU (threaded) |
|---------|-------|--------|----------------|
| small (2mm head) | 7.2s | 3.2s | 9.1s |
| medium (1mm brain, CC everywhere) | 17.0s | 18.1s | 38.6s |

WebGPU timings are after the per-scale CC workspace and single-pass batching landed
(6.4s / 30.0s before); measured at load 5–7, so indicative only.

**Benchmarking discipline.** This machine drifts badly under Spotlight indexing and thermal
load, and single-run comparisons were misleading repeatedly during development. Any timing
claim needs an *interleaved* A/B of the two binaries and a check that load average is below
about 2.

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
# Add --threads 1 for a reproducible single-threaded CPU run
# Add --init-affine <file> to override the physical affine (test SyN with a known affine)
```

### Checklist for Metal convergence

When bringing Metal to parity with CUDA/WebGPU, verify these in order:

1. **Moments**: Should match exactly (no GPU-specific code). Check SVD candidate selection and COM values.

2. **Fused CC gradient (1/ks² fix)**: Verify `metal_fused_cc_loss` applies `metal_tensor_scale(grad_pred, 1.0f/(ks*ks), spatial)` AFTER step 6 (fcc_bwd_grads dispatch), not inside `grad_output_val`. This is the most common source of quality divergence.

3. **Adam step_t reset**: Verify `fwd_step = 0; rev_step = 0;` when optimizer state is freed at scale transitions in both `syn_metal.m` and `greedy_metal.m`.

4. **Warp inverse**: Should use fixed-point iteration (`inv = -interp(u, id+inv)`, 550 iters). NOT Adam-based IC optimization.

5. **Box filter precision**: Metal's `box_filter_axis` / `box_filter_axis_packed` shaders should accumulate then scale (not multiply per-element during accumulation) to match CUDA/WebGPU precision.

6. **Per-stage NCC**: Run with `-v 2` and compare per-stage local NCC values against CUDA reference. They should agree within 1%.

7. **Global NCC**: Compare final warped image against Python reference in `validate/reference/`. Should be within 1% of Python.

8. **Mismatched spacing**: Metal applies `moving_pyramid_size()` and its FFT crop is bounds-checked, so a 2mm moving against a 1mm fixed matches CPU/WebGPU. Only CUDA still lacks it.

Datasets, per-dataset parameter choices and expected numbers are in
`validate/README.md`; `validate/run_validation.py` is the Python reference pipeline and
`validate/reference/` its saved outputs. `validate/*/output/` is gitignored.

## Known Issues

Ordered by how much damage they can do.

- **GPU failures are sticky and fatal on Metal and WebGPU.** Both contexts keep a sticky
  fatal flag set on allocation, pipeline, command-buffer, map and readback failure (and
  WebGPU's uncaptured-error callback); every stage entry point clears it on entry, checks
  it per iteration and per scale, and returns −1 with its partial result safe to free.
  WebGPU tracks every live buffer (`WGPU_MAX_BUFFERS` 512) so an abort releases what stage
  cleanup did not reach; Metal `CFRelease`s its table on cleanup. A stage abort tears the
  context down, so a library caller must re-init before the next stage. `main.c` routes
  every post-init failure through one cleanup path and exits 0 only after the save
  returned 0; the JS runner races the run against `GPUDevice.lost` and destroys the device
  exactly once. Remaining: CUDA's ~100 unchecked `cudaMalloc`s; `image_save_like()`
  ignores `nifti_image_write`'s result; the native WebGPU map wait is an untimed spin, so a
  lost device ends in a wgpu-native panic rather than −1.

- **`moving_pyramid_size()` is not applied in CUDA.** Sites: `linear_gpu.cu` (two), `greedy_gpu.cu` (one). Metal was adopted once the FFT crop was bounds-checked; CUDA is unverifiable on this machine (no toolkit), so it was left alone. The change is mechanical and identical to the Metal one. SyN is correctly exempt everywhere.

- **CUDA greedy does not export `greedy_result_t.disp`.** CPU, Metal and WebGPU return an owned displacement field at the last-scale grid (parity test: displacement correlation 0.99998 Metal, 1.00000 WebGPU); CUDA still returns the canonical empty tensor. Every entry point calls `greedy_result_init()` first; use `greedy_result_has_displacement()` before reading, and `tensor_free()` is safe on the empty result.

- **Metal regular CC still churns small buffers.** The 11 volume-sized per-call buffers are gone (per-scale `metal_cc_workspace_t`), but `metal_box_filter_axis` still allocates, registers, syncs and unregisters a tiny kernel buffer per axis: 24 small buffers and 24 `metal_sync`s per regular-CC call. Hoist the kernel into the workspace.

- **1 mm on browser WebGPU needs a capable adapter.** `js/src/run.ts` preflights the run: it reads the stage list from the argv it will pass, charges 5 channels for SyN and 3 otherwise (`syn_webgpu.c` vs `webgpu_kernels.c` `grad_sources`), checks that workspace against `maxStorageBufferBindingSize` and the largest upload (`max(n_fixed, n_moving)*4`) against `maxBufferSize`, and throws naming the byte count and the limit that failed. So a 1mm Greedy job (86.7 MB) passes a baseline adapter; a 1mm SyN job (144.4 MB) does not. Tiling the five-channel CC workspace is the open fix.

- **npm redistribution is not cleared.** The package is not on the registry. `release-npm.yml` builds and smoke-tests the tarball on every push/PR, but attaches it to a GitHub release only when the repository variable `CFIREANTS_NPM_REDISTRIBUTION_APPROVED` is `true` (`gh variable set CFIREANTS_NPM_REDISTRIBUTION_APPROVED --body true`); without it the `attach` job fails loudly. A release tarball is still redistribution under FireANTs License section 4(a) — get it confirmed in writing before setting the variable.

- CUDA: ~100 unchecked `cudaMalloc` calls
- Metal: `newLibraryWithFile:` deprecated (should use `newLibraryWithURL:`); `metal_dispatch`'s `buffer_sizes` parameter is never read and has already silently rotted at one call site
- WebGPU: `js/src/run.ts` `factories` cache retains a rejected import promise forever; `smoke-webgpu.mjs` relies on puppeteer's default 180 s `protocolTimeout`, shorter than its own 300 s budget
- Pipeline caches store string literal pointers — names must be static/literal
- Code duplication: alloc/free helpers 4x across Metal files, dataset_t in tests
