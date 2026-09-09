import { CfireantsError } from './error.js'
import type { Variant } from './types.js'

/*
 * Loading and driving the wasm module.
 *
 * The emscripten glue is NOT bundled into this package's own JavaScript. It is
 * copied into dist/ beside it and imported at run time by URL, because the glue
 * locates its .wasm through `import.meta.url`, which a bundler would rewrite to
 * the bundle's own location.
 */

/*
 * A registration is not a stalled module; it is a correct one doing multi-scale
 * SyN on a CPU. Minutes is normal -- the small 2 mm validation pair is 13.6 s
 * threaded and 84.9 s single-threaded on 14 cores, and a 1 mm brain
 * single-threaded is several minutes -- so the budgets are generous and a
 * caller who wants a tighter one passes timeoutMs.
 */
export const TIMEOUT_MS: Record<Variant, number> = {
  mt: 900_000,
  st: 1_800_000,
  gpu: 900_000,
}

type EmscriptenFactory = (config: Record<string, unknown>) => Promise<EmscriptenModule>

interface EmscriptenModule {
  callMain: (args: string[]) => void
  FS: {
    writeFile: (path: string, data: Uint8Array) => void
    readFile: (path: string) => Uint8Array
    analyzePath: (path: string) => { exists: boolean }
  }
}

const factories = new Map<string, Promise<EmscriptenFactory>>()

/**
 * Where the emscripten glue, its .wasm, and the worker script are served from.
 *
 * Defaults to this module's own directory, which is right for an unbundled ES
 * module. `assetPath` exists because a bundler cannot follow the import: the
 * URL is computed, so Vite and Rollup neither rewrite it nor emit the assets,
 * and the glue additionally finds its own `.wasm` through its own
 * `import.meta.url` -- so the pair must stay adjacent and unhashed.
 */
export function assetUrl(file: string, assetPath?: string): string {
  if (!assetPath) return new URL(file, import.meta.url).href
  const base = assetPath.endsWith('/') ? assetPath : `${assetPath}/`
  const here = typeof location !== 'undefined' ? location.href : 'file:///'
  const url = new URL(base + file, here)
  /*
   * Same-origin only, and one implementation of the check rather than two: this
   * feeds a dynamic import() here and `new Worker()` in index.ts. A cross-origin
   * value would execute a third party's script in the host page and hand it the
   * patient's volumes. An opaque origin (file://, a sandboxed iframe) is refused
   * outright rather than compared -- there `location.origin` is the STRING
   * "null", which equals the origin of any data: URL.
   */
  if (typeof location !== 'undefined') {
    if (location.origin === 'null')
      throw new CfireantsError('unsupported-option',
        'assetPath cannot be used from an opaque origin (file:// or a sandboxed ' +
        'iframe); serve the page over http(s)')
    if (url.origin !== location.origin)
      throw new CfireantsError('unsupported-option',
        `assetPath must be same-origin; ${url.origin} is not ${location.origin}`)
  }
  return url.href
}

/** `cfireants-mt.js` / `cfireants.js`, matching what build-wasm.sh stages. */
export function moduleFile(variant: Variant): string {
  if (variant === 'mt') return 'cfireants-mt.js'
  if (variant === 'gpu') return 'cfireants-gpu.js'
  return 'cfireants.js'
}

interface WebGPUAdapterLike {
  limits: Record<string, unknown>
  requestDevice: (descriptor?: { requiredLimits?: Record<string, number> })
    => Promise<WebGPUDeviceLike>
}

interface WebGPUDeviceLike {
  limits?: Record<string, unknown>
  lost?: Promise<{ reason?: string, message?: string }>
  destroy?: () => void
  addEventListener?: (
    type: 'uncapturederror',
    listener: (event: { error?: { message?: string } }) => void,
  ) => void
  pushErrorScope?: (filter: 'validation') => void
  popErrorScope?: () => Promise<{ message?: string } | null>
}

interface WebGPUProviderLike {
  requestAdapter: (options?: { powerPreference?: 'high-performance' }) =>
    Promise<WebGPUAdapterLike | null>
}

function webgpuProvider(): WebGPUProviderLike | undefined {
  if (typeof navigator === 'undefined') return undefined
  return (navigator as unknown as { gpu?: WebGPUProviderLike }).gpu
}

function loadFactory(variant: Variant, assetPath?: string): Promise<EmscriptenFactory> {
  const url = assetUrl(moduleFile(variant), assetPath)
  let pending = factories.get(url)
  if (!pending) {
    // Not statically analysable on purpose, so bundlers leave it alone.
    pending = import(/* @vite-ignore */ url).then((m) => m.default as EmscriptenFactory)
    factories.set(url, pending)
  }
  return pending
}

export interface RunRequest {
  variant: Variant
  fixed: Uint8Array
  moving: Uint8Array
  /** argv after the input and output paths, which this function owns. */
  args: string[]
  assetPath?: string
  timeoutMs?: number
  onLog?: (line: string) => void
  signal?: AbortSignal
}

export interface RunResult {
  image: Uint8Array
  elapsedMs: number
  log: string
}

/**
 * Run one registration to completion.
 *
 * The subtlety is the exit. The threaded build is `-sPROXY_TO_PTHREAD`, so
 * `main` runs on a worker and `callMain` RETURNS the moment it is handed over
 * -- returning 0, exactly as the single-threaded module returns 0 after `main`
 * has finished. A harness that trusted that value would read `/out.nii` before
 * it exists and get ENOENT. `onExit` is awaited instead, for BOTH builds: it is
 * required for `mt` and harmless for `st`, and branching on the variant here
 * would be a second place for the two to drift apart.
 */

/*
 * Voxel count from a NIfTI-1/2 header, or 0 if it does not look like one.
 * NIfTI-1 is a 348-byte header with dim[] at offset 40 as int16; NIfTI-2 is 540
 * bytes with dim[] at offset 16 as int64. The size field at offset 0 identifies
 * which, and doubles as the endianness probe.
 */
function niftiVoxelCount(bytes: Uint8Array): number {
  if (bytes.length < 348) return 0
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  for (const little of [true, false]) {
    const sizeof = dv.getInt32(0, little)
    if (sizeof === 348) {
      let n = 1
      for (let a = 0; a < 3; a++) n *= dv.getInt16(42 + a * 2, little)
      return n > 0 ? n : 0
    }
    if (sizeof === 540 && bytes.length >= 540) {
      let n = 1
      for (let a = 0; a < 3; a++) n *= Number(dv.getBigInt64(24 + a * 8, little))
      return n > 0 ? n : 0
    }
  }
  return 0
}

/*
 * Largest single storage binding and largest single buffer the C will create,
 * checked against the device's limits. The binding is the fixed-grid CC
 * workspace: fused CC (SyN, syn_webgpu.c) packs five float channels into one
 * range, regular CC (rigid/affine/greedy, webgpu_kernels.c `grad_sources`)
 * packs three. Either image is also uploaded whole, which must fit
 * maxBufferSize. The conformant baseline is 128 MiB for both, which a
 * 182x218x182 template exceeds at 144,420,640 bytes for SyN but not at
 * 86,652,384 for greedy. Checked up front because otherwise the run fails
 * deep inside bind-group creation, after the user has waited, naming a buffer
 * the user has never heard of.
 */
function webgpuLimitError(
  fixed: Uint8Array, moving: Uint8Array, args: string[],
  limits: { maxStorageBufferBindingSize?: unknown, maxBufferSize?: unknown },
): string | null {
  // main.c defaults to the SyN preset when no stage is named.
  const stageArgs = ['--transform', '--rigid', '--affine', '--greedy', '--syn']
  const usesSyn = args.includes('--syn') || args.some((a) => a.startsWith('SyN')) ||
    !args.some((a) => stageArgs.includes(a))
  const nFixed = niftiVoxelCount(fixed)
  const nMoving = niftiVoxelCount(moving)
  if (!nFixed) return null
  const workspace = nFixed * (usesSyn ? 5 : 3) * 4
  const image = Math.max(nFixed, nMoving) * 4
  const binding = Number(limits.maxStorageBufferBindingSize ?? 0)
  const buffer = Number(limits.maxBufferSize ?? 0)
  const tail = 'Use a lower-resolution template, or run on the CPU backend.'
  if (binding && workspace > binding)
    return `this GPU allows ${binding} bytes per storage binding, but the fixed image ` +
      `needs ${workspace}. ${tail}`
  if (buffer && Math.max(workspace, image) > buffer)
    return `this GPU allows ${buffer} bytes per buffer, but the run needs ` +
      `${Math.max(workspace, image)}. ${tail}`
  return null
}

export async function run(request: RunRequest): Promise<RunResult> {
  const factory = await loadFactory(request.variant, request.assetPath)
  const lines: string[] = []
  const record = (line: string) => {
    lines.push(line)
    request.onLog?.(line)
  }
  const log = () => lines.join('\n')

  let settle: (code: number) => void
  const exited = new Promise<number>((resolve) => { settle = resolve })
  const limit = request.timeoutMs ?? TIMEOUT_MS[request.variant]
  let gpuDevice: WebGPUDeviceLike | undefined
  let gpuUncapturedError: string | undefined
  let gpuLost: Promise<never> | undefined

  const aborted: Promise<never> | undefined = request.signal && new Promise((_, reject) => {
    const fail = () => reject(new CfireantsError('aborted', 'the registration was aborted', log()))
    if (request.signal!.aborted) fail()
    else request.signal!.addEventListener('abort', fail, { once: true })
  })
  // An abort that arrives after the work is done rejects a promise no race is
  // still watching; this keeps that from surfacing as an unhandled rejection.
  aborted?.catch(() => {})

  const race = async <T>(work: Promise<T>, phase: string): Promise<T> => {
    let timer: ReturnType<typeof setTimeout> | undefined
    const expire = new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new CfireantsError('timeout',
          `the cfireants ${request.variant} module did not ${phase} within ${limit} ms`,
        log())), limit)
    })
    try {
      const contenders: Promise<unknown>[] = [work, expire]
      if (aborted) contenders.push(aborted)
      if (gpuLost) contenders.push(gpuLost)
      return await Promise.race(contenders) as T
    } finally {
      clearTimeout(timer)
    }
  }

  const config: Record<string, unknown> = {
    noInitialRun: true,
    // The C takes the program name from argv[0]; without this the module would
    // identify itself as whatever host script loaded it.
    thisProgram: 'cfireants_reg',
    print: record,
    printErr: record,
    onExit: (code: number) => settle(code),
  }

  if (request.variant === 'gpu') {
    const gpu = webgpuProvider()
    if (!gpu)
      throw new CfireantsError('no-webgpu',
        'the WebGPU backend was requested but navigator.gpu is unavailable')
    /* Wrapped the same way the device request is: a rejection here used to
     * escape as a raw browser error while the device path produced a
     * CfireantsError, so callers matching on `code` saw one but not the other. */
    const adapter = await race(
      gpu.requestAdapter({ powerPreference: 'high-performance' }),
      'acquire a WebGPU adapter',
    ).catch((e) => {
      if (e instanceof CfireantsError) throw e
      throw new CfireantsError('no-webgpu',
        `the WebGPU backend was requested but no adapter could be acquired: ${
          e instanceof Error ? e.message : String(e)}`)
    })
    if (!adapter)
      throw new CfireantsError('no-webgpu',
        'the WebGPU backend was requested but no compatible adapter was found')
    /*
     * Ask for the adapter's own buffer-size limits. Requesting a value the
     * adapter already reports can never fail, so this stays portable: a baseline
     * device yields baseline limits, a capable one yields more. The defaults are
     * not enough -- a 182x218x182 template needs a 144,420,640-byte binding for
     * the packed five-channel CC workspace, above the 134,217,728-byte default,
     * and the run dies at bind-group creation.
     *
     * Only the size limits are raised. Binding *counts* stay at the baseline
     * deliberately: the CC gradient shader is packed below the eight-storage-
     * buffer ceiling so it runs on any conforming device, and quietly relying on
     * a higher count would reintroduce the bug that made every loss garbage.
     *
     * The device is created here because the module consumes it through
     * `preinitializedWebGPUDevice`, which keeps the C-side API synchronous.
     */
    const sizeLimits: Record<string, number> = {}
    for (const key of ['maxStorageBufferBindingSize', 'maxBufferSize'] as const) {
      const value = adapter.limits?.[key]
      if (typeof value === 'number') sizeLimits[key] = value
    }
    gpuDevice = await race(
      adapter.requestDevice({ requiredLimits: sizeLimits }),
      'acquire a WebGPU device',
    ).catch((e) => {
      if (e instanceof CfireantsError) throw e
      throw new CfireantsError('no-webgpu',
        `the WebGPU backend was requested but the device could not be created: ${
          e instanceof Error ? e.message : String(e)}`)
    })
    if (gpuDevice.lost) {
      gpuLost = gpuDevice.lost.then((info) => {
        throw new CfireantsError('registration-failed',
          `the WebGPU device was lost${info.reason ? ` (${info.reason})` : ''}: ` +
          `${info.message ?? 'no reason reported'}`, log())
      })
      // Deliberate destruction after the result has been copied also resolves
      // `lost`; no active race observes it then.
      gpuLost.catch(() => {})
    }
    const limitProblem = webgpuLimitError(
      request.fixed, request.moving, request.args, gpuDevice.limits ?? {})
    if (limitProblem) {
      gpuDevice.destroy?.()
      throw new CfireantsError('no-webgpu',
        `the WebGPU backend cannot run this image: ${limitProblem}`)
    }
    gpuDevice.addEventListener?.('uncapturederror', (event) => {
      gpuUncapturedError = event.error?.message ?? 'unknown WebGPU error'
      record(`WebGPU uncaptured error: ${gpuUncapturedError}`)
    })
    gpuDevice.pushErrorScope?.('validation')
    config.preinitializedWebGPUDevice = gpuDevice
  }

  /*
   * The factory is RACED, not awaited. The `mt` module imports SHARED memory,
   * and on a page that is not cross-origin isolated emscripten's pool spawn
   * fails inside a promise nobody holds --
   *   DataCloneError: SharedArrayBuffer transfer requires self.crossOriginIsolated
   * -- so factory() neither resolves nor rejects and a bare await hangs for
   * good. index.ts picks `st` for such a page before we get here, which makes
   * this the second line rather than the first.
   */
  try {
    const module = await race(factory(config), 'finish initialising')

    // Uncompressed and named `.nii`: the C links no zlib, which is why gzip.ts exists.
    module.FS.writeFile('/fixed.nii', request.fixed)
    module.FS.writeFile('/moving.nii', request.moving)

    const started = performance.now()
    try {
      module.callMain(['-f', '/fixed.nii', '-m', '/moving.nii', '-o', '/out.nii',
        '--backend', request.variant === 'gpu' ? 'webgpu' : 'cpu', ...request.args])
    } catch (e) {
      // ExitStatus IS the normal exit under EXIT_RUNTIME and onExit has almost
      // certainly fired already; settling again is idempotent. Anything without a
      // numeric status is a genuine fault.
      if (e && typeof (e as { status?: unknown }).status === 'number') {
        settle!((e as { status: number }).status)
      } else {
        record(`threw: ${e instanceof Error ? e.stack ?? e.message : String(e)}`)
        settle!(-1)
      }
    }
    /*
     * The `mt` build is timeable because PROXY_TO_PTHREAD moved main() onto a
     * worker, leaving this thread free to run the timer; `st` runs main() right
     * here and no timer can fire until it returns, so the limit only bounds its
     * initialisation. Either way this STOPS WAITING, it does not stop the module.
     * `worker: true` is the only real cancellation, which is why it is the default.
     */
    const code = await race(exited, 'finish')
    const elapsedMs = performance.now() - started

    const validationError = gpuDevice?.popErrorScope
      ? await race(gpuDevice.popErrorScope(), 'check WebGPU validation')
      : null
    if (validationError || gpuUncapturedError) {
      const message = validationError?.message ?? gpuUncapturedError
      throw new CfireantsError('registration-failed',
        `the WebGPU backend reported an error: ${message}`, log())
    }

    if (code !== 0)
      throw new CfireantsError('registration-failed',
        `the cfireants ${request.variant} module exited with status ${code}`, log())
    if (!module.FS.analyzePath('/out.nii').exists)
      throw new CfireantsError('registration-failed',
        `the cfireants ${request.variant} module exited cleanly but wrote no output`, log())

    return { image: module.FS.readFile('/out.nii'), elapsedMs, log: log() }
  } finally {
    gpuDevice?.destroy?.()
  }
}
