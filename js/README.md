# cfireants

Medical image registration in the browser: rigid, affine, SyN and greedy
deformable alignment on a threaded CPU or WebGPU backend. Two NIfTI
`ArrayBuffer`s go in, the warped moving image comes out.

> **This is a modified derivative work.** The module is compiled from
> `cfireants`, a pure C reimplementation of
> [FireANTs](https://github.com/rohitrango/FireANTs) — **the files have been
> changed**, none of the original Python is carried over, and this package is
> not endorsed by the FireANTs authors. It is distributed under the FireANTs
> License v1.0 (`LICENSE`), whose section 4 requires the copyright notices,
> license text and bibliography to travel with it; see
> [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md), which carries the
> citations. **If you publish work using this, cite the FireANTs paper.**
>
> **Public package attachment is gated pending redistribution review.** Section
> 4(a) enumerates the permitted forms of redistribution, and an npm package is
> neither the original repository nor a wheel on a package manager named in the
> project's documentation. 4(a)(iii) (a modified fork retaining notices and
> bibliography) is the clause this package relies on. The release workflow will
> build and test the tarball, but will not attach it publicly until the repository
> variable `CFIREANTS_NPM_REDISTRIBUTION_APPROVED` is explicitly set to `true`
> after written clearance or a reviewed interpretation has been recorded.

## Installing

The package is not on the npm registry. Once redistribution approval has been
recorded, each [GitHub release](https://github.com/neurolabusc/cfireants/releases)
carries the immutable tarball and SHA-256 file produced by
`.github/workflows/release-npm.yml`:

```sh
npm install https://github.com/neurolabusc/cfireants/releases/download/v0.1.20260909/cfireants-0.1.20260909.tgz
```

npm records the URL and integrity hash in the lockfile, so installs are pinned.
The workflow refuses to replace an existing asset: corrected bits require a new
version and tag.

```js
import { register } from 'cfireants'

const { image, elapsedMs, variant } = await register(
  await fixedFile.arrayBuffer(),   // .nii or .nii.gz
  await movingFile.arrayBuffer(),
)
// image is gzipped if the moving input was
```

## CPU threads and WebGPU

Three wasm builds ship. The two CPU variants are chosen automatically; WebGPU
is selected explicitly with `backend: 'webgpu'`:

| variant | needs | small 2 mm pair | quality |
| --- | --- | --- | --- |
| `mt` | a **cross-origin isolated** page | **13.6 s** | 0.9555 |
| `st` | nothing | **84.9 s** | 0.9555 |
| `gpu` | `navigator.gpu` — **no isolation** | **15.7 s** | 0.9557 |

Chrome, 14 cores; quality is global NCC against the template. **WebGPU is not
currently the fast one.** At this size Asyncify suspension and the parts still
on the CPU (notably SyN warp inversion) cost more than the GPU saves — the
same run is 6.4 s natively. What it buys is that it needs **no cross-origin
isolation**: on a host that cannot send COOP/COEP the comparison is not 15.7 s
against 13.6 s but 15.7 s against 84.9 s.

A 1 mm brain single-threaded takes several minutes, which is why the `st`
timeout default is 30 minutes against `mt`'s 15.

The threaded build is `-pthread`, so it imports shared memory, and a page that
is not cross-origin isolated **cannot instantiate it — and HANGS rather than
failing.** Emscripten's pool spawn throws
`DataCloneError: SharedArrayBuffer transfer requires self.crossOriginIsolated`
inside a promise nobody holds, so the factory neither resolves nor rejects. This
package therefore asks the question *before* it fetches anything, and quietly
takes `st` when the answer is no — a page on GitHub Pages, which cannot set
these headers, still works. `result.variant` says which one ran.

To get `mt`, serve the **top-level document** with:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Putting them on the `.wasm` response instead does nothing: isolation is a
property of the document. Note the cost, which is the usual reason a deployment
turns isolation back off: under `require-corp` every cross-origin subresource
the page loads — images, fonts, analytics, embeds — must itself send
`Cross-Origin-Resource-Policy` or proper CORS, or be blocked.

Ask before you promise the user a fast run:

```js
import { checkSupport } from 'cfireants'

const { threaded, variant, threads, reasons } = checkSupport()
if (!threaded) console.warn(`falling back to ${variant}: ${reasons.join('; ')}`)
```

WebGPU is asked about separately, because the only honest answer needs
`requestAdapter()` and is therefore async — `navigator.gpu` exists in browsers
that then hand back no adapter:

```js
import { checkWebgpuSupport, register } from 'cfireants'

if (await checkWebgpuSupport()) await register(fixed, moving, { backend: 'webgpu' })
```

`backend: 'webgpu'` never falls back. Where there is no device it throws
`CfireantsError` with `code: 'no-webgpu'`, so a caller who asked for the GPU
finds out rather than being handed a CPU result labelled as one. The gpu build
is single-threaded for the work it does not put on the device, so `threads` is
meaningless there and passing it is refused; `result.variant` is `'gpu'`.

## API

```ts
register(
  fixed: ArrayBuffer | ArrayBufferView,
  moving: ArrayBuffer | ArrayBufferView,
  options?: RegisterOptions,
): Promise<RegisterResult>

checkSupport(): SupportReport
checkWebgpuSupport(): Promise<boolean>
```

`RegisterResult` is `{ image: ArrayBuffer, elapsedMs: number, variant: 'mt' |
'st' | 'gpu', threads: number, log: string }` — `elapsedMs` is time inside the module,
`log` is everything it wrote to stdout and stderr.

`SupportReport` is `{ supported, threaded, variant, threads, reasons }`:
`supported` is false only where there is no WebAssembly at all, `threaded` says
whether `mt` is usable, and `reasons` says why it is not.

| option | default | meaning |
| --- | --- | --- |
| `backend` | `'cpu'` | `'cpu'` or `'webgpu'`; WebGPU is explicit and never silently falls back |
| `transform` | `'syn'` | preset stage sequence: `'rigid'`, `'affine'`, `'syn'`, `'greedy'` |
| `stages` | — | explicit stages, in order; overrides `transform` |
| `verbose` | `0` | 0 silent, 1 summary, 2 per-iteration. `onLog` only sees what this emits |
| `gzip` | matches the moving input | gzip the returned buffer |
| `worker` | `true` where `Worker` exists | run in a Web Worker. The only real cancellation. Node has none, so it defaults to false there |
| `assetPath` | this module's directory | where `cfireants.js`/`cfireants.wasm` are served from |
| `threads` | `true` | CPU only: `false` forces `st`; a positive integer caps the pool via `--threads` |
| `timeoutMs` | `900000` (`mt`/`gpu`) / `1800000` (`st`) | give up waiting. SyN is minutes, not seconds |
| `onLog` | — | receives the module's output lines as they arrive |
| `signal` | — | `AbortSignal`; under `worker: true` it terminates the module |
| `args` | — | appended to argv verbatim, after everything else |

### Stages

Stages are the CLI's own strings, passed through unparsed. **There is no shell**,
so write them plain — `SyN[0.1,0.5,1.0]`, never `'SyN[0.1,0.5,1.0]'` with the
quotes inside the string.

```js
await register(fixed, moving, {
  verbose: 2,
  onLog: (line) => console.log(line),
  stages: [
    { transform: 'Rigid[0.003]',      metric: 'MI[32]', convergence: '[200x100x50,1e-6,10]', shrinkFactors: '4x2x1' },
    { transform: 'Affine[0.001]',     metric: 'MI[32]', convergence: '[200x100x50,1e-6,10]', shrinkFactors: '4x2x1' },
    { transform: 'SyN[0.1,0.5,1.0]',  metric: 'CC[5]',  convergence: '[200x100x50,1e-6,10]', shrinkFactors: '4x2x1' },
  ],
})
```

`metric` defaults to MI for the linear stages and CC for the deformable ones,
matching the executable. If both images are brain-extracted, pass `CC[5]` at
every stage instead: on such a pair it is both more accurate and faster. At `verbose: 2` the module writes progress to stderr,
which `onLog` receives line by line:

```
  SyN scale 2: 96x114x96
    iter 50/100 loss=-0.789890
```

Parse it if you want a progress bar; the raw lines are also joined into
`result.log`.

### gzip

The wasm module links **no zlib** — the C reads and writes plain `.nii` only —
so all compression happens in TypeScript, via `DecompressionStream` and
`CompressionStream`. Inputs are sniffed by magic bytes, not by filename, so a
`.nii` that is really gzipped still works. An environment without those streams
gets a `no-compression-streams` refusal rather than a corrupt volume.

### Threads

The threaded pool sizes itself to `navigator.hardwareConcurrency`. A positive
integer caps it through the CLI's `--threads` flag — 14 cores is 13.6 s on the
small 2 mm pair, `threads: 4` is 29.8 s — and still selects the threaded build,
since the single-threaded one cannot honour any value but 1. `threads: false`
forces the single-threaded build even on an isolated page, which is worth having
when you would rather leave the cores to the page.

Node is exempt from the isolation rule: it has SharedArrayBuffer, needs no
headers to use it, and runs the threaded build at full width.

### Bundlers

The emscripten glue is loaded by URL at run time and locates its own `.wasm`
through its own `import.meta.url`, so the two files must stay **adjacent and
unhashed**. Vite and Rollup will neither rewrite that URL nor emit the assets.
Copy `cfireants-mt.js`, `cfireants-mt.wasm`, `cfireants.js`, `cfireants.wasm`,
`cfireants-gpu.js`, `cfireants-gpu.wasm`, and `worker.js` out of this package's
`dist/` into somewhere served, and point `assetPath` at it:

```js
await register(fixed, moving, { assetPath: '/cfireants/' })
```

`assetPath` must be same-origin: it feeds a dynamic `import()` and a
`new Worker()`, either of which would otherwise run a third party's code with
access to the patient's volumes.

## Image size and memory

Peak memory tracks the **fixed** image, not the moving one. SyN allocates its
displacement fields, optimiser moments and correlation intermediates in the fixed
image's grid, so an oversized moving volume costs only its own storage.

| Fixed | Moving | Peak | Wall, 14 threads |
|-------|--------|------|------------------|
| 182x218x182, 1mm template | 182x218x182 | 1.80 GB | 46.7s |
| 182x218x182, 1mm template | 256x256x256 | 1.92 GB | 56.3s |
| 256x256x256 | 182x218x182 | 3.88 GB | 185.1s |

wasm32 caps the heap at 4 GB and a browser tab has less to give than node, so the
last row has no headroom and a larger fixed image will fail. Register **to** a
template at 1mm and pass the large volume as `moving`.

The **gpu** build is tighter still: its heap is a fixed 1 GiB, because
emscripten's WebGPU port cannot be combined with `ALLOW_MEMORY_GROWTH`. It also
needs an adapter whose `maxStorageBufferBindingSize` fits the largest single
binding: SyN's five-channel correlation workspace (144 MB for a 1mm template),
or three channels for Greedy and the linear stages (87 MB, inside the 128 MiB
baseline). The wrapper requests the adapter's own reported size limits, so a
capable GPU gets them.

Before starting, the wrapper reads dim[1..3] out of both NIfTI headers, works
out which stages will run, and checks the workspace against
`maxStorageBufferBindingSize` and the largest upload against `maxBufferSize`.
If either fails it throws, naming the bytes needed and the limit that refused.
That is the whole of it: **a baseline adapter fails fast with a message you can
act on, it does not gain the ability to run 1 mm SyN.** Tiling the workspace is
the fix that would, and it is not written. Use Greedy, the CPU backend, or a
smaller fixed image on such a device.

## Errors

Every refusal throws a `CfireantsError` with a `code` you can branch on:

| code | when |
| --- | --- |
| `bad-input` | not an ArrayBuffer/typed array, or empty |
| `no-compression-streams` | gzip needed but `DecompressionStream`/`CompressionStream` is absent |
| `unsupported-environment` | no WebAssembly. A page without cross-origin isolation is **not** an error — CPU silently gets the single-threaded build |
| `no-webgpu` | `backend: 'webgpu'` was requested and there is no `navigator.gpu`, no adapter, or no device. Never a silent downgrade to CPU |
| `unsupported-option` | a bad option, a cross-origin `assetPath`, or `threads` on a build that cannot honour it |
| `timeout` | the module did not initialise, or did not finish, in `timeoutMs` |
| `registration-failed` | the module exited non-zero, or wrote no output |
| `aborted` | the `AbortSignal` fired |

`error.log` carries the module's own output when the failure came from inside
it. There is no fallback behind any of these: a registration that silently
downgraded would return a plausible volume that is wrong.

## Building

```sh
npm install
npm run checkVersion
npm run build      # build the wasm, stage it into dist/, emit types, bundle
npm run typecheck
npm run smokePackage -- . ../validate/small
npm run smokeWebgpu -- .  # Chrome: packaged CPU/WebGPU Rigid→Affine→SyN parity
```

`npm run build` runs `build-wasm.sh` first, which needs emscripten on `PATH`
(6.0.2 is what CI pins) and builds from the C sources in the parent repository. `scripts/copy-wasm.mjs` then stages
all six wasm artifacts out of `js/wasm/` into `dist/`; set `CFIREANTS_WASM_DIR`
if they are written somewhere else.

`src/main.c`, `package.json`, both root version fields in `package-lock.json`,
and the release tag must carry the same version. CI checks this instead of
rewriting release metadata. Pull requests and pushes run native CTest, build
and smoke-test both CPU variants from the packed tarball, then run a short
browser CPU/WebGPU parity check through the packed glue and wasm assets.
Release attachment is a separate, write-scoped job behind the redistribution
gate.
