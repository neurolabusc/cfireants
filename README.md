# cfireants

Pure C port of [FireANTs](https://github.com/rohitrango/FireANTs) — GPU-accelerated medical image registration with rigid, affine, and diffeomorphic deformable alignment.

Three GPU backends plus CPU reference:

- **CUDA** — production quality (NVIDIA)
- **Metal** — native macOS/Apple Silicon
- **WebGPU** — portable via wgpu-native (Vulkan/Metal)
- **CPU** — reference implementation, no GPU required

All backends produce equivalent registration accuracy. See [validate/README.md](validate/README.md) for benchmarks.

## Building

Requires CMake >= 3.18, a C11 compiler, and zlib. Each GPU backend is optional.

```bash
mkdir -p build && cd build

cmake .. -DCFIREANTS_CUDA=ON                                  # CUDA
cmake .. -DCFIREANTS_METAL=ON                                 # Metal (macOS)
cmake .. -DCFIREANTS_WEBGPU=ON                                # WebGPU
cmake .. -DCFIREANTS_CUDA=OFF -DCFIREANTS_METAL=OFF           # CPU only

make -j$(nproc)
```

WebGPU requires [wgpu-native](https://github.com/gfx-rs/wgpu-native/releases) v27+ in `third_party/wgpu/`.

## Usage

The `cfireants_reg` tool uses ANTs-style command-line arguments:

```bash
# Default: Moments → Rigid (MI) → Affine (MI) → SyN (CC)
cfireants_reg -f fixed.nii.gz -m moving.nii.gz -o output_

# Affine only (no deformable)
cfireants_reg -f fixed.nii.gz -m moving.nii.gz --affine -o output_

# Rigid only
cfireants_reg -f fixed.nii.gz -m moving.nii.gz --rigid -o output_

# Greedy deformable (faster than SyN, ~1% lower accuracy)
cfireants_reg -f fixed.nii.gz -m moving.nii.gz --greedy -o output_

# Trilinear downsample (GPU-native, no FFT dependency)
cfireants_reg -f fixed.nii.gz -m moving.nii.gz --trilinear -o output_

# Choose backend explicitly
cfireants_reg -f fixed.nii.gz -m moving.nii.gz --backend metal -o output_
cfireants_reg -f fixed.nii.gz -m moving.nii.gz --backend cpu -o output_
```

### Custom stages

Each registration stage is specified with `--transform`, `--metric`, `--convergence`, and `--shrink-factors`. Stages execute in order:

```bash
cfireants_reg -f fixed.nii.gz -m moving.nii.gz \
  --transform 'Rigid[0.003]' --metric 'MI[32]' \
    --convergence '[200x100x50,1e-6,10]' --shrink-factors 4x2x1 \
  --transform 'Affine[0.001]' --metric 'MI[32]' \
    --convergence '[200x100x50,1e-6,10]' --shrink-factors 4x2x1 \
  --transform 'SyN[0.1,0.5,1.0]' --metric 'CC[5]' \
    --convergence '[200x100x50,1e-6,10]' --shrink-factors 4x2x1 \
  -o output_
```

### Arguments

| Argument | Description |
|----------|-------------|
| `-f, --fixed <file>` | Fixed (stationary) NIfTI image |
| `-m, --moving <file>` | Moving image to register |
| `-o, --output <prefix>` | Output prefix (default: `output_`) |
| `-w, --warped <file>` | Explicit warped output path |
| `--backend <name>` | `cpu`, `metal`, `webgpu`, `cuda` (default: best available) |
| `--trilinear` | Use GPU-native trilinear downsample instead of FFT |
| `--moments` / `--no-moments` | Enable/disable center-of-mass initialization |
| `--rigid` | Preset: Rigid only |
| `--affine` | Preset: Rigid + Affine |
| `--syn` | Preset: Rigid + Affine + SyN (default) |
| `--greedy` | Preset: Rigid + Affine + Greedy |
| `-v [level]` | Verbosity: 0=silent (default), 1=summary, 2=per-iteration |
| `--version` | Print version and exit |
| `--skullstrip <mask>` | Brain mask in template space — warps to subject, applies threshold |

### Per-stage options

| Argument | Description |
|----------|-------------|
| `--transform Type[params]` | `Rigid[lr]`, `Affine[lr]`, `SyN[lr,warp_sigma,grad_sigma]`, `Greedy[lr,warp_sigma,grad_sigma]` |
| `--metric Type[param]` | `MI[bins]` or `CC[kernel_size]` |
| `--convergence [iters,tol,win]` | Iterations per level (e.g., `200x100x50`), tolerance, window |
| `--shrink-factors NxNx...` | Downsample factors per level (e.g., `4x2x1`) |
| `--smoothing-sigmas NxNx...` | Blur sigmas per level (reserved for future use) |

### Output

- `<prefix>Warped.nii.gz` — Moving image resampled into fixed space (registration mode)
- `-o <file>` — Skull-stripped fixed image in native datatype (when `--skullstrip` used)

### Skull stripping

Register a template to a subject, then warp a brain mask to strip non-brain tissue:

```bash
cfireants_reg -f subject.nii.gz -m template.nii.gz --affine --trilinear \
  --skullstrip brain_mask.nii.gz -o brain_extracted.nii.gz
```

When `--skullstrip` is used, `-o` is the output filename (not a prefix). The mask (in template space) is warped into subject space, thresholded at 0.5, and applied — voxels outside the mask are set to the darkest intensity. Output preserves the native datatype (UINT16 in → UINT16 out).

## Validation

Run all validation tests from the repo root (requires datasets in `validate/`):

```bash
# Registration — small dataset (2mm full-head, MI+SyN)
cfireants_reg \
  -f validate/small/MNI152_T1_2mm.nii.gz \
  -m validate/small/T1_head_2mm.nii.gz \
  --trilinear -o ./test/small_syn_

# Registration — small dataset, greedy (faster)
cfireants_reg \
  -f validate/small/MNI152_T1_2mm.nii.gz \
  -m validate/small/T1_head_2mm.nii.gz \
  --greedy --trilinear -o ./test/small_greedy_

# Registration — medium dataset (1mm brain-extracted, CC throughout)
cfireants_reg \
  -f validate/medium/MNI152_T1_1mm_brain.nii.gz \
  -m validate/medium/t1_brain.nii.gz \
  --transform 'Rigid[0.003]' --metric 'CC[5]' --convergence '[200x100x50,1e-6,10]' --shrink-factors 4x2x1 \
  --transform 'Affine[0.001]' --metric 'CC[5]' --convergence '[200x100x50,1e-6,10]' --shrink-factors 4x2x1 \
  --transform 'SyN[0.1,0.5,1.0]' --metric 'CC[5]' --convergence '[200x100x50,1e-6,10]' --shrink-factors 4x2x1 \
  --trilinear -o ./test/medium_syn_

# Registration — large dataset (1mm full-head, MI+SyN, 4 scales for linear)
cfireants_reg \
  -f validate/large/MNI152_T1_1mm.nii.gz \
  -m validate/large/chris_t1.nii.gz \
  --transform 'Rigid[0.003]' --metric 'MI[32]' --convergence '[200x200x100x50,1e-6,10]' --shrink-factors 8x4x2x1 \
  --transform 'Affine[0.001]' --metric 'MI[32]' --convergence '[200x200x100x50,1e-6,10]' --shrink-factors 8x4x2x1 \
  --transform 'SyN[0.1,0.5,1.0]' --metric 'CC[5]' --convergence '[200x100x50,1e-6,10]' --shrink-factors 4x2x1 \
  --trilinear -o ./test/large_syn_

# Skull stripping — warp MNI brain mask to subject space
cfireants_reg \
  -f validate/skulllstrip/T1_head_2mm.nii.gz \
  -m validate/skulllstrip/MNI152_T1_2mm.nii.gz \
  --affine --trilinear \
  --skullstrip validate/skulllstrip/mniMask.nii.gz \
  -o ./test/bT1_head_2mm.nii.gz
```

See [validate/README.md](validate/README.md) for expected NCC values, timing, and memory usage.

## Why C?

- No Python/PyTorch runtime dependency
- Predictable GPU memory (~900 MB CUDA, ~400 MB Metal on small dataset)
- Explicit forward+backward (no autograd overhead)
- Single static library per backend

## License

See the [FireANTs](https://github.com/rohitrango/FireANTs) repository for license information.
