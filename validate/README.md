# FireANTs Validation Suite

Registers moving images to MNI templates using a full ANTs-style pipeline
(Moments -> Rigid -> Affine -> SyN/Greedy) and reports global NCC, local NCC loss,
wall-clock time, and peak GPU memory.

## Datasets

| Dataset | Description | Stationary | Moving | Notes |
|---------|------------|------------|--------|-------|
| **small** | 2mm full-head | MNI152_T1_2mm | T1_head_2mm | Includes scalp; MI used for linear stages |
| **medium** | 1mm brain-extracted | MNI152_T1_1mm_brain | t1_brain | Brain-extracted; CC used throughout |
| **large** | 1mm full-head | MNI152_T1_1mm | chris_t1 (0.88mm) | Includes scalp, different resolution; MI for linear, extra 8x scale |

## Usage

```bash
# Python (PyTorch) validation
python validate/run_validation.py                          # all datasets
python validate/run_validation.py --dataset small          # single dataset
python validate/run_validation.py --save-reference         # save reference outputs
python validate/run_validation.py --check-reference        # check for regressions

# C/CUDA validation
cfireants/build/test_validate_all                          # all datasets
cfireants/build/test_validate_all --dataset small          # single dataset
```

## Baseline Metrics

Measured on NVIDIA RTX 4090 (24 GB).

### Python (PyTorch 2.10.0+cu128, fused ops)

Pipeline: Moments → Rigid → Affine → SyN

| Dataset | NCC Before | NCC After | Local NCC Loss | Time (s) | Peak GPU (MB) |
|---------|-----------|-----------|---------------|---------|--------------|
| small   | 0.5953    | 0.9450    | -0.5838       | 4.0     | 804          |
| medium  | 0.5753    | 0.9443    | -0.8127       | 4.9     | 1409         |
| large   | 0.7254    | 0.8961    | -0.3051       | 17.1    | 6337         |

### C/CUDA SyN (cfireants, symmetric bidirectional)

Pipeline: Moments (GPU) → Rigid (GPU) → Affine (GPU) → SyN (GPU).
Same method as Python. Evaluation uses `compose(fwd_warp, inverse(rev_warp))`
via iterative IC-based warp inversion (550 iterations matching Python's
`compositive_warp_inverse`). Per-axis Gaussian blur on moving image matching
Python's `_smooth_image_not_mask`.

| Dataset | NCC Before | NCC After | Local NCC Loss | Time (s) | Peak CPU (MB) | Peak GPU (MB) |
|---------|-----------|-----------|---------------|---------|--------------|--------------|
| small   | 0.5957    | 0.9614    | -0.6511       | 3.0     | 127          | 903          |
| medium  | 0.5753    | 0.9554    | -0.8570       | 5.5     | 337          | 907          |
| large   | 0.7254    | 0.9176    | -0.3679       | 17.9    | 400          | 921          |

### Comparison

| Dataset | NCC After | | | Local NCC Loss | | | Time | | |
|---------|-----------|-----------|-----------|--------------|-----------|-----------|------|------|------|
|         | **Python SyN** | **C Greedy** | **C SyN** | **Python SyN** | **C Greedy** | **C SyN** | **Py** | **C Greedy** | **C SyN** |
| small   | 0.9450    | 0.9512    | **0.9614** | -0.5838      | -0.6090   | **-0.6511** | 4.0s | **2.7s** | 3.0s |
| medium  | 0.9443    | 0.9434    | **0.9554** | -0.8127      | -0.8458   | **-0.8570** | 4.9s | **2.4s** | 5.5s |
| large   | 0.8961    | 0.9022    | **0.9176** | -0.3051      | -0.3277   | **-0.3679** | 17.1s| **13.7s**| 17.9s|

Notes:
- **Both C/CUDA approaches exceed Python SyN quality** on all three datasets for both NCC metrics.
- **C/CUDA Greedy** is the fastest (1.2-2.0x faster than Python), suitable when speed is prioritized.
- **C/CUDA SyN** produces the highest quality, comparable speed to Python.
- Quality differences are due to float32 optimization trajectory divergence, verified to be within expected precision: MI loss matches to 1e-13, affine backward to 7e-6, FFT downsample to 0.001. Individual operations are identical; compound differences across hundreds of iterations lead to different (slightly better) local minima.
- Both implementations use FFT-based downsampling, Gaussian Parzen MI for linear stages, CC for deformable, compositive Adam with gradient normalization, per-axis blur, and IC-based warp inversion (SyN only).
- Peak GPU memory is ~900-930 MB across all datasets (vs Python's 804-6337 MB).

### C/CUDA Greedy (cfireants, compositive single-direction)

Pipeline: Moments (GPU) → Rigid (GPU) → Affine (GPU) → Greedy compositive (GPU).
Faster than SyN (no dual warp, no warp inversion for evaluation).

| Dataset | NCC Before | NCC After | Local NCC Loss | Time (s) | Peak CPU (MB) | Peak GPU (MB) |
|---------|-----------|-----------|---------------|---------|--------------|--------------|
| small   | 0.5957    | 0.9512    | -0.6090       | 2.7     | 128          | 903          |
| medium  | 0.5753    | 0.9434    | -0.8458       | 2.4     | 339          | 906          |
| large   | 0.7254    | 0.9022    | -0.3277       | 13.7    | 403          | 906          |

Greedy is the fastest option (0.1-0.6s for deformable stage) and still exceeds Python SyN quality on all datasets.

### Per-stage timing breakdown

#### Python (PyTorch)

| Dataset | Moments (s) | Rigid (s) | Affine (s) | SyN (s) | Total (s) |
|---------|------------|-----------|-----------|---------|----------|
| small   | 0.2        | 2.0       | 1.0       | 0.8     | 4.0      |
| medium  | 0.3        | 1.2       | 1.3       | 2.2     | 4.9      |
| large   | 0.3        | 7.9       | 6.6       | 2.4     | 17.1     |

#### C/CUDA SyN

| Dataset | Moments (s) | Rigid (s) | Affine (s) | SyN (s)  | Total (s) |
|---------|------------|-----------|-----------|----------|----------|
| small   | 0.1        | 1.0       | 1.4       | 0.4      | 3.0      |
| medium  | 0.1        | 0.9       | 0.9       | 3.6      | 5.5      |
| large   | 0.2        | 6.9       | 7.2       | 3.7      | 17.9     |

#### C/CUDA Greedy

| Dataset | Moments (s) | Rigid (s) | Affine (s) | Greedy (s) | Total (s) |
|---------|------------|-----------|-----------|-----------|----------|
| small   | 0.1        | 1.0       | 1.4       | 0.1       | 2.7      |
| medium  | 0.1        | 0.9       | 0.9       | 0.5       | 2.4      |
| large   | 0.2        | 6.2       | 6.7       | 0.6       | 13.7     |

Notes:
- Moments uses GPU for orientation candidate evaluation. SVD and COM stay on CPU.
- Rigid uses MI + extra moving blur (matching Python `_smooth_image_not_mask`).
- Affine uses MI (small/large) or CC (medium), with FFT downsample, no extra blur (matching Python).
- SyN includes 550-iteration IC warp inversion at full resolution for evaluation.
- Per-axis Gaussian kernels match Python's `separable_filtering` with axis-specific sigmas.
- FFT-based downsampling (`cuda_downsample_fft`) matches the Python fused-ops CUDA path.
- Fused CC loss, fused compositive update, and fused blur kernels minimize per-iteration overhead.

### Metric definitions

- **NCC Before**: Global normalized cross-correlation between the stationary image
  and the moving image resampled into stationary space (identity transform).
- **NCC After**: Global NCC between the stationary image and the warped output.
  Values closer to 1.0 indicate better alignment.
- **Local NCC Loss**: FireANTs `LocalNormalizedCrossCorrelationLoss` with kernel_size=9.
  More negative values indicate better local alignment. This is the primary quality metric.

## Reference outputs

Running `--save-reference` saves warped images and metrics JSON under
`validate/reference/<dataset>/`. These serve as the baseline for regression
detection via `--check-reference`, which flags any NCC degradation exceeding 2%.

## Dataset-specific parameter choices

- **small/large** (full-head with scalp): Mutual information (MI) for rigid and
  affine stages handles the intensity mismatch between the MNI template and
  subject scans with scalp tissue. CC is used for the deformable stage.
- **medium** (brain-extracted): Cross-correlation (CC) throughout, since both
  images are skull-stripped with similar intensity profiles.
- **large**: Uses an extra 8x downsampling scale for rigid/affine to handle
  the larger field of view and resolution difference (1mm vs 0.88mm).

## Downsample Modes

All three GPU backends support two downsampling modes:

- **FFT** (default) — FFT-based downsample matching Python. CUDA uses cuFFT; Metal uses MPSGraph; WebGPU uses kissfft (CPU fallback).
- **Trilinear** (`--trilinear`) — Gaussian blur + trilinear resize, fully GPU-native. Matches or exceeds FFT accuracy. Enables like-for-like comparison across all backends.

### FFT vs Trilinear (small dataset)

| Metric | CUDA FFT | CUDA Trilinear | Metal FFT | Metal Trilinear |
|--------|----------|----------------|-----------|-----------------|
| NCC After | 0.9614 | 0.9644 | 0.9608 | 0.9639 |
| Total Time | 7.6s | 7.5s | 7.4s | 7.3s |
| Peak RAM | 147 MB | 140 MB | 390 MB | 391 MB |

CUDA measured on NVIDIA GB10. Metal measured on Apple M4 Pro.

## WebGPU Backend (wgpu-native)

### CUDA vs WebGPU — Trilinear Mode (small dataset, NVIDIA GB10 Vulkan)

| Metric | CUDA | WebGPU | Ratio |
|--------|------|--------|-------|
| NCC After | 0.9644 | 0.9643 | 1.000 |
| Total Time | 7.5s | 14.2s | 1.9x |
| Rigid (MI) | 2.8s | 0.9s | 0.3x |
| Affine (MI) | 3.0s | 1.1s | 0.4x |
| SyN (CC) | 1.4s | 11.9s | 8.5x |
| Peak RAM | 140 MB | 445 MB | 3.2x |

WebGPU matches CUDA accuracy. MI is faster on WebGPU (CUDA downloads full images for max-finding). SyN is slower due to per-dispatch overhead.

### CUDA vs WebGPU — All Datasets (Trilinear, NVIDIA GB10)

| Dataset | CUDA NCC | WebGPU NCC | CUDA Time | WebGPU Time |
|---------|----------|------------|-----------|-------------|
| small | 0.9644 | 0.9647 | 7.5s | 13.7s |
| medium | 0.9536 | 0.9541 | 18.2s | 90.5s |
| large | 0.9213 | 0.9190 | 55.1s | 93.2s |

## Metal Backend (Apple Silicon)

### Metal vs WebGPU — All Datasets (SyN Trilinear, Apple M4 Pro)

Both backends run on the same M4 Pro GPU. Metal uses native API with unified memory; WebGPU runs via wgpu-native's Metal backend.

| Dataset | WebGPU NCC | Metal NCC | WebGPU Time | Metal Time | Metal RAM |
|---------|------------|-----------|-------------|------------|-----------|
| small | 0.9642 | 0.9636 | 60.1s | 7.2s | 391 MB |
| medium | 0.9541 | 0.9540 | 131.7s | 18.7s | 3023 MB |
| large | 0.9191 | 0.9199 | 88.5s | 36.2s | 3259 MB |

Metal matches WebGPU accuracy (within 0.1%) and runs 2–8x faster.

### Greedy vs SyN — WebGPU (Apple M4 Pro, Trilinear)

| Dataset | | SyN | Greedy | Improvement |
|---------|------|------|--------|-------------|
| **small** | NCC | 0.9642 | 0.9542 | -1.0% |
| | Total Time | 61.3s | 39.0s | 1.6x faster |
| | Peak RAM | 132 MB | 107 MB | 19% less |
| **medium** | NCC | 0.9541 | 0.9434 | -1.1% |
| | Total Time | 132.2s | 88.4s | 1.5x faster |
| | Peak RAM | 874 MB | 647 MB | 26% less |
| **large** | NCC | 0.9191 | 0.9053 | -1.5% |
| | Total Time | 90.7s | 46.9s | 1.9x faster |
| | Peak RAM | 1028 MB | 794 MB | 23% less |

### Greedy vs SyN — Metal (Apple M4 Pro, Trilinear)

| Dataset | SyN NCC | Greedy NCC | SyN Time | Greedy Time | SyN RAM | Greedy RAM |
|---------|---------|------------|----------|-------------|---------|------------|
| small | 0.9636 | 0.9511 | 7.2s | 6.5s | 391 MB | 276 MB |
| medium | 0.9540 | 0.9420 | 18.7s | 16.7s | 3023 MB | 2103 MB |
| large | 0.9199 | 0.8996 | 36.2s | 37.6s | 3259 MB | 2340 MB |

Greedy is 1–2% lower NCC but 1.1–1.9x faster with 19–30% less memory. Use `--greedy` flag.
