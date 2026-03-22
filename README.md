# cfireants

Pure C port of [FireANTs](https://github.com/rohitrango/FireANTs) — GPU-accelerated medical image registration with rigid, affine, and diffeomorphic deformable alignment.

Three GPU backends:

- **CUDA** — production quality (NVIDIA)
- **Metal** — native macOS/Apple Silicon
- **WebGPU** — portable via wgpu-native (Vulkan/Metal)

All three produce equivalent registration accuracy. See [validate/README.md](validate/README.md) for benchmarks.

## Building

Requires CMake >= 3.18, a C11 compiler, and zlib. Each backend is optional.

```bash
mkdir -p build && cd build

cmake .. -DCFIREANTS_CUDA=ON                                  # CUDA
cmake .. -DCFIREANTS_METAL=ON                                 # Metal (macOS)
cmake .. -DCFIREANTS_WEBGPU=ON                                # WebGPU
cmake .. -DCFIREANTS_WEBGPU=ON -DCFIREANTS_METAL=ON           # WebGPU + Metal

make -j$(nproc)
```

WebGPU requires [wgpu-native](https://github.com/gfx-rs/wgpu-native/releases) v27+ in `third_party/wgpu/`.

## Usage

Run the full pipeline (Moments → Rigid → Affine → SyN/Greedy) from the repo root:

```bash
build/test_validate_all --dataset small                       # CUDA SyN
build/test_validate_all --dataset small --trilinear           # CUDA trilinear downsample
build/test_validate_metal --dataset small --trilinear         # Metal SyN
build/test_validate_metal --dataset small --trilinear --greedy  # Metal Greedy (faster)
build/test_validate_webgpu --dataset small --trilinear        # WebGPU SyN
```

Datasets: `small` (2mm full-head), `medium` (1mm brain-extracted), `large` (1mm full-head). Omit `--dataset` to run all three.

## Why C?

- No Python/PyTorch runtime dependency
- Predictable GPU memory (~900 MB CUDA, ~400 MB Metal on small dataset)
- Explicit forward+backward (no autograd overhead)
- Single static library per backend

## License

See the [FireANTs](https://github.com/rohitrango/FireANTs) repository for license information.
