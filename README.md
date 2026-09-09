# cfireants

Pure C port of [FireANTs](https://github.com/rohitrango/FireANTs) — GPU-accelerated medical image registration with rigid, affine, and diffeomorphic deformable alignment.

Three GPU backends plus CPU reference:

- **CUDA** — production quality (NVIDIA)
- **Metal** — native macOS/Apple Silicon
- **WebGPU** — portable via wgpu-native (Vulkan/Metal), and in browsers via WebAssembly
- **CPU** — reference implementation, no GPU required, multi-threaded

All backends produce equivalent registration accuracy. On an Apple Silicon laptop the small
2 mm pair takes 7.2 s on Metal, 6.4 s on WebGPU and 9.1 s on 14 CPU threads; on the medium
1 mm brain pair Metal leads at 17.0 s against WebGPU's 30.0 s and the CPU's 38.6 s. See
[validate/README.md](validate/README.md) for accuracy benchmarks.

## Building

Requires CMake >= 3.18 and a C11 compiler. Each GPU backend is optional, and so is zlib.

Every GPU backend is off by default, so a plain `cmake ..` gives a CPU-only build.

```bash
mkdir -p build && cd build

cmake ..                                                      # CPU only
cmake .. -DCFIREANTS_CUDA=ON                                  # CUDA
cmake .. -DCFIREANTS_METAL=ON                                 # Metal (macOS)
cmake .. -DCFIREANTS_WEBGPU=ON                                # WebGPU
cmake .. -DCFIREANTS_THREADS=OFF                              # Disable CPU threading
cmake .. -DCFIREANTS_ZLIB=OFF                                 # No zlib: plain .nii only

make -j8
```

CPU operations use a bounded pthread worker pool by default. The pool uses the
number of online processors and does not make per-thread copies of image data.
On 14 cores the small 2 mm pair drops from 59.3 s single-threaded to 9.1 s.
Limit CPU use, or force a reproducible single-thread run, with either
`CFIREANTS_NUM_THREADS` or `--threads` (the flag exists because the environment
variable cannot be set under Emscripten):

```bash
CFIREANTS_NUM_THREADS=4 cfireants_reg ... --backend cpu
cfireants_reg ... --backend cpu --threads 1
```

### WebAssembly / npm

```bash
js/build-wasm.sh     # requires emscripten on PATH
```

Builds threaded and single-threaded CPU modules plus a browser WebGPU module into
`js/wasm/`. The wrapper in `js/` selects the CPU build automatically, or the WebGPU
build when passed `backend: 'webgpu'`, and handles gzip with `DecompressionStream`
(the wasm modules link no zlib). See `js/README.md`.
[edgefire](https://github.com/rordenlab/edgefire) is a browser app built on the package.

WebGPU requires [wgpu-native](https://github.com/gfx-rs/wgpu-native/releases) v27+ in `third_party/wgpu/`.

## Usage

The `cfireants_reg` tool uses ANTs-style command-line arguments:

```bash
# Default: Moments → Rigid (MI) → Affine (MI) → SyN (CC)
cfireants_reg -f fixed.nii.gz -m moving.nii.gz -o warped.nii.gz

# Affine only (no deformable)
cfireants_reg -f fixed.nii.gz -m moving.nii.gz --affine -o warped.nii.gz

# Rigid only
cfireants_reg -f fixed.nii.gz -m moving.nii.gz --rigid -o warped.nii.gz

# Greedy deformable (faster than SyN, ~1% lower accuracy)
cfireants_reg -f fixed.nii.gz -m moving.nii.gz --greedy -o warped.nii.gz

# Explicitly select the shared CPU/GPU pyramid (also the default)
cfireants_reg -f fixed.nii.gz -m moving.nii.gz --trilinear -o warped.nii.gz

# Opt into FFT downsampling on a GPU backend
cfireants_reg -f fixed.nii.gz -m moving.nii.gz --backend metal --fft -o warped.nii.gz

# Choose backend explicitly
cfireants_reg -f fixed.nii.gz -m moving.nii.gz --backend metal -o warped.nii.gz
cfireants_reg -f fixed.nii.gz -m moving.nii.gz --backend cpu -o warped.nii.gz
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
  -o warped.nii.gz
```

### Arguments

| Argument | Description |
|----------|-------------|
| `-f, --fixed <file>` | Fixed (stationary) NIfTI image |
| `-m, --moving <file>` | Moving image to register |
| `-o, --output <file>` | Output NIfTI filename (default: `output.nii.gz`) |
| `--backend <name>` | `cpu`, `metal`, `webgpu`, `cuda` (default: best available) |
| `--trilinear` | Use blur + trilinear downsampling (default and shared by every backend) |
| `--fft` | Use FFT downsampling on a GPU backend; CPU reports that it will remain trilinear |
| `--threads <n>` | Cap CPU worker threads (default: all cores) |
| `--moments` / `--no-moments` | Enable/disable center-of-mass initialization |
| `--orientation rot\|antirot\|both` | Moments candidates (default `rot`; `both` also admits mirror images) |
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

- `-o <file>` — Moving image resampled into fixed space (registration mode), or skull-stripped fixed image (when `--skullstrip` used)

### Skull stripping

Register a template to a subject, then warp a brain mask to strip non-brain tissue:

```bash
cfireants_reg -f subject.nii.gz -m template.nii.gz --affine --trilinear \
  --skullstrip brain_mask.nii.gz -o brain_extracted.nii.gz
```

The mask (in template space) is warped into subject space, thresholded at 0.5, and applied — voxels outside the mask are set to the darkest intensity. Output preserves the native datatype (UINT16 in → UINT16 out).

## Notarized macOS installer

`make macos-release` builds an Apple Silicon `.pkg` that installs `cfireants_reg`
into `/usr/local/bin`. The executable is self-contained: Metal shaders are
embedded (`CFIREANTS_EMBED_METALLIB`), threading is pthreads from libSystem, and
zstd is switched off because Homebrew's `libzstd` is not present on a user's
machine. Deployment target is macOS 14.0, which MPSGraph's FFT requires.

Two **different** certificates from the same Apple Developer account are needed:

| Certificate | Signs |
|---|---|
| Developer ID **Application** | the executable |
| Developer ID **Installer** | the `.pkg` |

Installer certificates do not appear under `security find-identity -p codesigning`,
so list them without that filter:

```bash
security find-identity -v
```

Store notarization credentials once. This prompts securely for an
*app-specific* password (appleid.apple.com → Sign-In and Security →
App-Specific Passwords), not your Apple ID password:

```bash
make macos-notary-profile APPLE_ID='you@example.com' TEAM_ID='ABCDE12345'
```

Then build, sign, notarize, staple, and Gatekeeper-check in one command:

```bash
make macos-release \
  MACOS_SIGN_IDENTITY='Developer ID Application: Your Name (ABCDE12345)' \
  MACOS_INSTALLER_IDENTITY='Developer ID Installer: Your Name (ABCDE12345)'
```

The artifact is `dist/cfireants-<version>-macos-arm64.pkg`; the version comes
from `CFIREANTS_VERSION` in `src/main.c` and `NOTARY_PROFILE` defaults to
`cfireants-notary`. For a local packaging test that cannot be notarized, the
output is deliberately named `cfireants-<version>-macos-arm64-unsigned.pkg`:

```bash
make macos-pkg-adhoc
make macos-verify-adhoc
```

`scripts/verify_macos_pkg.sh` expands the finished package **without installing
it** and checks that the executable is correctly signed, carries no foreign
dependency, and actually runs.

Publishing a GitHub release whose tag starts with `v` runs
`.github/workflows/release-npm.yml`, which verifies that the tag, C version,
package manifest and lockfile agree, runs native CTest, builds all three wasm
modules, smoke-tests both CPU variants from the packed tarball, and compares a
short packaged browser WebGPU Rigid→Affine→SyN run with CPU. It produces a
SHA-256 file. Public attachment is blocked until the repository variable
`CFIREANTS_NPM_REDISTRIBUTION_APPROVED=true` records that the redistribution
review described in `js/README.md` is complete. Release assets are never
overwritten under an existing version. `make macos-release` writes the
notarized `.pkg` and a sibling `.pkg.sha256`; upload both to the same release.

## Validation

The unit and parity tests are registered with CTest and run from the repo root
automatically; the backend parity test is recorded as a skip when no GPU initialises:

```bash
ctest --test-dir build --output-on-failure
```

Run the validation registrations from the repo root (requires datasets in `validate/`):

```bash
# Registration — small dataset (2mm full-head, MI+SyN)
cfireants_reg \
  -f validate/small/MNI152_T1_2mm.nii.gz \
  -m validate/small/T1_head_2mm.nii.gz \
  -o ./test/small_syn.nii.gz

# Registration — small dataset, greedy (faster)
cfireants_reg \
  -f validate/small/MNI152_T1_2mm.nii.gz \
  -m validate/small/T1_head_2mm.nii.gz \
  --greedy -o ./test/small_greedy.nii.gz

# Registration — medium dataset (1mm brain-extracted, CC throughout)
cfireants_reg \
  -f validate/medium/MNI152_T1_1mm_brain.nii.gz \
  -m validate/medium/t1_brain.nii.gz \
  --transform 'Rigid[0.003]' --metric 'CC[5]' --convergence '[200x100x50,1e-6,10]' --shrink-factors 4x2x1 \
  --transform 'Affine[0.001]' --metric 'CC[5]' --convergence '[200x100x50,1e-6,10]' --shrink-factors 4x2x1 \
  --transform 'SyN[0.1,0.5,1.0]' --metric 'CC[5]' --convergence '[200x100x50,1e-6,10]' --shrink-factors 4x2x1 \
  -o ./test/medium_syn.nii.gz

# Registration — large dataset (1mm full-head, MI+SyN, 4 scales for linear)
cfireants_reg \
  -f validate/large/MNI152_T1_1mm.nii.gz \
  -m validate/large/chris_t1.nii.gz \
  --transform 'Rigid[0.003]' --metric 'MI[32]' --convergence '[200x200x100x50,1e-6,10]' --shrink-factors 8x4x2x1 \
  --transform 'Affine[0.001]' --metric 'MI[32]' --convergence '[200x200x100x50,1e-6,10]' --shrink-factors 8x4x2x1 \
  --transform 'SyN[0.1,0.5,1.0]' --metric 'CC[5]' --convergence '[200x100x50,1e-6,10]' --shrink-factors 4x2x1 \
  -o ./test/wchris_t1.nii.gz

# Skull stripping — warp MNI brain mask to subject space
cfireants_reg \
  -f validate/skulllstrip/T1_head_2mm.nii.gz \
  -m validate/skulllstrip/MNI152_T1_2mm.nii.gz \
  --affine \
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

FireANTs License version 1.0 — see [`LICENSE`](LICENSE). This is a derivative work: no
FireANTs source is carried over, and it is not endorsed by the FireANTs authors. If you
publish work using it, cite the FireANTs paper.
