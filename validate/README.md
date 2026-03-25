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

# C/CUDA validation (run from repo root)
build/test_validate_all                                    # all datasets (SyN)

# Metal (macOS)
build/test_validate_metal                                  # all datasets (SyN)
build/test_validate_metal --dataset small --greedy         # Greedy (faster)

# WebGPU
build/test_validate_webgpu                                 # all datasets (SyN)
build/test_validate_webgpu --dataset small --greedy        # Greedy

# CLI tool (any backend)
build/cfireants_reg -f validate/small/MNI152_T1_2mm.nii.gz -m validate/small/T1_head_2mm.nii.gz \
  -v 2 -o test/small_syn.nii.gz --backend cuda
```

## Python vs CUDA vs WebGPU — All Datasets (NVIDIA RTX 4090)

Pipeline: Moments → Rigid → Affine → SyN. FFT downsampling.

| Dataset | Python NCC | CUDA NCC | WebGPU NCC | Python Time | CUDA Time | WebGPU Time |
|---------|-----------|---------|-----------|------------|----------|------------|
| small   | 0.9450    | 0.9530  | 0.9548    | 3.9s       | 2.7s     | 16.1s      |
| medium  | 0.9443    | 0.9469  | 0.9465    | 3.8s       | 3.2s     | 127.6s     |
| large   | 0.8961    | 0.8972  | 0.8886    | 16.1s      | 15.6s    | 127.7s     |

### Peak memory (RTX 4090)

| Dataset | Python GPU | CUDA GPU | CUDA CPU | WebGPU CPU |
|---------|-----------|---------|---------|-----------|
| small   | 804 MB    | 1093 MB | 130 MB  | 468 MB    |
| medium  | 1574 MB   | 1097 MB | 341 MB  | 872 MB    |
| large   | 7825 MB   | 1111 MB | 404 MB  | 901 MB    |

Notes:
- Python GPU memory is PyTorch CUDA peak (`torch.cuda.max_memory_allocated`), includes autograd graph overhead.
- CUDA GPU memory is approximate (total - free at peak). CUDA CPU is Linux VmPeak.
- WebGPU runs on Vulkan via wgpu-native. Reports CPU RSS only (wgpu manages GPU memory internally).
- CUDA uses ~1 GB GPU regardless of dataset size (pre-allocated workspace). Python scales with image size.

## Metal — All Datasets (Apple M4 Pro, SyN Trilinear)

Metal uses native API with unified memory on Apple Silicon.

| Dataset | Python NCC | Metal NCC | Metal Time | Metal RAM |
|---------|------------|-----------|------------|-----------|
| small   | 0.9450     | 0.9574    | 7.4s       | 391 MB    |
| medium  | 0.9443     | 0.9454    | 19.9s      | 2955 MB   |
| large   | 0.8961     | 0.9022    | 44.2s      | 3123 MB   |

## Greedy vs SyN

Greedy compositive (single-direction) is faster than SyN (no dual warp, no warp inversion) at 1–2% lower NCC.

### Greedy vs SyN — Metal (Apple M4 Pro, Trilinear)

| Dataset | SyN NCC | Greedy NCC | SyN Time | Greedy Time | SyN RAM | Greedy RAM |
|---------|---------|------------|----------|-------------|---------|------------|
| small   | 0.9574  | 0.9417     | 7.4s     | 6.5s        | 391 MB  | 276 MB     |
| medium  | 0.9454  | 0.9345     | 19.9s    | 16.3s       | 2955 MB | 2036 MB    |
| large   | 0.9022  | 0.8836     | 44.2s    | 39.6s       | 3123 MB | 2203 MB    |

Greedy uses 19–30% less memory. Use `--greedy` flag.

## FFT vs Trilinear Downsampling

| Metric | CUDA FFT | CUDA Trilinear | Metal FFT | Metal Trilinear |
|--------|----------|----------------|-----------|-----------------|
| NCC After (small) | 0.9533 | 0.9548 | 0.9532 | 0.9574 |
| Total Time | 3.0s | 3.0s | 7.6s | 7.4s |

Both modes produce equivalent accuracy. Trilinear is fully GPU-native and enables like-for-like cross-backend comparison (no FFT library dependency).

## Dataset-specific parameter choices

- **small/large** (full-head with scalp): Mutual information (MI) for rigid and
  affine stages handles the intensity mismatch between the MNI template and
  subject scans with scalp tissue. CC is used for the deformable stage.
- **medium** (brain-extracted): Cross-correlation (CC) throughout, since both
  images are skull-stripped with similar intensity profiles.
- **large**: Uses an extra 8x downsampling scale for rigid/affine to handle
  the larger field of view and resolution difference (1mm vs 0.88mm).

## Metric definitions

- **NCC**: Global normalized cross-correlation (Pearson) between the stationary image and the warped output. Values closer to 1.0 indicate better alignment.
- **Local NCC Loss**: FireANTs `LocalNormalizedCrossCorrelationLoss` with kernel_size=9. More negative = better local alignment.

## Reference outputs

Running `--save-reference` saves warped images and metrics JSON under
`validate/reference/<dataset>/`. These serve as the baseline for regression
detection via `--check-reference`, which flags any NCC degradation exceeding 2%.
