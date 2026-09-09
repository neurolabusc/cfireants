import { CfireantsError } from './error.js'
import { gunzip, gzip, isGzip } from './gzip.js'
import { assetUrl, TIMEOUT_MS, run } from './run.js'
import type {
  Backend, RegisterOptions, RegisterResult, Stage, SupportReport, Variant,
} from './types.js'

export { CfireantsError } from './error.js'
export type { CfireantsErrorCode } from './error.js'
export type {
  Backend, Transform, Variant, Stage, RegisterOptions, RegisterResult, SupportReport,
} from './types.js'

const TRANSFORMS = ['rigid', 'affine', 'syn', 'greedy'] as const

/**
 * Whether this is a non-browser host, where the isolation rule does not apply.
 *
 * `crossOriginIsolated` is a browser concept: node has SharedArrayBuffer, needs
 * no headers to use it, and genuinely runs the threaded build at full width.
 * Applying the browser rule there cost a measured 6x for nothing.
 */
function isNodeHost(): boolean {
  const proc = (globalThis as { process?: { versions?: { node?: string } } }).process
  return !!proc?.versions?.node
    && typeof window === 'undefined' && typeof document === 'undefined'
}

/** What the module's thread pool will size itself to when unconstrained. */
function coreCount(): number {
  return typeof navigator !== 'undefined' && navigator.hardwareConcurrency
    ? navigator.hardwareConcurrency
    : 1
}

/**
 * Which build can run here, and which one `register()` would pick.
 *
 * Two builds are shipped because the threaded one is `-pthread`: it imports
 * SHARED memory, and a page that is not cross-origin isolated cannot
 * instantiate it. Worse, that failure is not an exception -- emscripten's pool
 * spawn rejects a promise nobody holds and initialisation simply never
 * completes -- so the question must be asked BEFORE the module is fetched, and
 * it is a question about response headers rather than about the machine.
 *
 * That rule is a BROWSER rule, and only browsers are held to it -- see
 * isNodeHost(). Within a browser it is `!== true`, not `=== false`: the
 * property is absent rather than false on hosts that never had it, and
 * tolerating absence is what would pick `mt` on a page that then hangs.
 */
export function checkSupport(): SupportReport {
  const reasons: string[] = []
  if (typeof SharedArrayBuffer === 'undefined')
    reasons.push('this environment has no SharedArrayBuffer')
  else if (!isNodeHost() && globalThis.crossOriginIsolated !== true)
    reasons.push('this page is not cross-origin isolated; serve the top-level ' +
      'document with Cross-Origin-Opener-Policy: same-origin and ' +
      'Cross-Origin-Embedder-Policy: require-corp')
  const threaded = reasons.length === 0
  return {
    supported: typeof WebAssembly !== 'undefined',
    threaded,
    variant: threaded ? 'mt' : 'st',
    threads: threaded ? coreCount() : 1,
    reasons,
  }
}

/**
 * Whether `backend: 'webgpu'` can run here.
 *
 * Separate from checkSupport(), and async, because the only honest answer needs
 * `requestAdapter()`: `navigator.gpu` exists in browsers that then hand back no
 * adapter at all. Breaking checkSupport()'s synchronous signature to fold this
 * in would make every caller await a question about CPU threads.
 *
 * This is a probe, not a reservation -- register() requests its own device --
 * so calling it is optional; `backend: 'webgpu'` throws `no-webgpu` regardless.
 */
export async function checkWebgpuSupport(): Promise<boolean> {
  const gpu = (globalThis.navigator as { gpu?: { requestAdapter: () => Promise<unknown> } })?.gpu
  if (!gpu) return false
  try {
    return await gpu.requestAdapter() !== null
  } catch {
    return false
  }
}

/**
 * WebGPU is explicit and never falls back. For CPU, `threads: false` forces
 * `st`; otherwise the threaded build is used wherever it can be. The CPU
 * fallback to `st` is silent so a page without COOP/COEP still works.
 */
function chooseVariant(
  backend: Backend | undefined,
  threads: RegisterOptions['threads'],
  threaded: boolean,
): Variant {
  if (backend === 'webgpu') return 'gpu'
  return threads === false || !threaded ? 'st' : 'mt'
}

/**
 * The numeric form of `threads` is a pool cap, passed as `--threads`.
 *
 * Through argv rather than through `CFIREANTS_NUM_THREADS`, which does not work
 * here: under `-sPROXY_TO_PTHREAD` the environment is materialised on the
 * proxied worker before JS can mutate `Module.ENV`, so setting it was silently
 * ignored and the pool still came up at full width.
 */
function threadCap(threads: RegisterOptions['threads']): number | undefined {
  if (typeof threads !== 'number') return undefined
  if (!Number.isInteger(threads) || threads < 1)
    throw new CfireantsError('unsupported-option',
      `threads must be true, false, or a positive integer, got ${threads}`)
  return threads
}

function buildArgs(options: RegisterOptions): string[] {
  // The CPU implements this pyramid and every GPU backend supports it. Pass it
  // explicitly so changing only `backend` never changes registration math.
  const args: string[] = ['--trilinear']

  if (options.backend !== undefined && options.backend !== 'cpu' && options.backend !== 'webgpu')
    throw new CfireantsError('unsupported-option',
      `backend must be 'cpu' or 'webgpu', got ${String(options.backend)}`)
  if (options.backend === 'webgpu' && options.threads !== undefined)
    throw new CfireantsError('unsupported-option',
      'threads applies only to the CPU backend; omit it when backend is webgpu')
  if (options.args?.includes('--backend'))
    throw new CfireantsError('unsupported-option',
      'pass backend: \'webgpu\' instead of putting --backend in args')

  if (options.stages) {
    if (options.stages.length === 0)
      throw new CfireantsError('unsupported-option', 'stages was empty; omit it to use a preset')
    for (const stage of options.stages) {
      const s: Stage = stage
      if (!s.transform)
        throw new CfireantsError('unsupported-option', 'every stage needs a transform')
      args.push('--transform', s.transform)
      if (s.metric) args.push('--metric', s.metric)
      if (s.convergence) args.push('--convergence', s.convergence)
      if (s.shrinkFactors) args.push('--shrink-factors', s.shrinkFactors)
    }
  } else {
    const preset = options.transform ?? 'syn'
    if (!TRANSFORMS.includes(preset))
      throw new CfireantsError('unsupported-option',
        `unknown transform '${preset}'; expected one of ${TRANSFORMS.join(', ')}`)
    args.push(`--${preset}`)
  }

  if (options.verbose !== undefined) args.push('-v', String(options.verbose))
  // Accepted by both builds; `st` parses it and stays at 1 regardless.
  const cap = threadCap(options.threads)
  if (cap !== undefined) args.push('--threads', String(cap))
  if (options.args) args.push(...options.args)
  return args
}

/**
 * Run one registration in a Web Worker.
 *
 * Created per call and terminated afterwards. That is deliberate rather than
 * lazy: it releases the module's heap and its thread pool, and it makes
 * `timeoutMs` and `signal` REAL cancellation -- terminate() actually stops the
 * work, where an in-thread timeout can only stop waiting for it while every
 * core stays pinned.
 */
async function runInWorker(
  fixed: Uint8Array,
  moving: Uint8Array,
  options: RegisterOptions,
): Promise<RegisterResult> {
  if (typeof Worker === 'undefined')
    throw new CfireantsError('unsupported-option',
      'this environment has no Worker; call with worker: false')

  const url = assetUrl('worker.js', options.assetPath)
  const worker = new Worker(url, { type: 'module' })
  const { onLog, signal, worker: _w, ...rest } = options
  void _w

  /*
   * Resolve assetPath HERE and send the absolute URL. Inside the worker
   * `location.href` is the WORKER SCRIPT's own URL -- which already lives under
   * assetPath -- so a relative value would be applied a second time.
   */
  if (rest.assetPath !== undefined) rest.assetPath = new URL('./', url).href

  let timer: ReturnType<typeof setTimeout> | undefined
  let onAbort: (() => void) | undefined
  try {
    return await new Promise<RegisterResult>((resolve, reject) => {
      if (signal) {
        // terminate() first: this is the one path that actually stops the pool.
        onAbort = () => {
          worker.terminate()
          reject(new CfireantsError('aborted', 'the registration was aborted'))
        }
        if (signal.aborted) onAbort()
        else signal.addEventListener('abort', onAbort, { once: true })
      }
      worker.onmessage = (event) => {
        const message = event.data
        if (message.type === 'log') { onLog?.(message.line); return }
        if (message.type === 'done') { resolve(message.result); return }
        // CfireantsError does not survive structured clone, so its shape is sent
        // explicitly and rebuilt here; losing the `code` would turn every
        // refusal into a generic failure.
        reject(new CfireantsError(message.code, message.message, message.log))
      }
      worker.onerror = (event) => reject(new CfireantsError('registration-failed',
        `the registration worker failed to start or threw: ${event.message ?? 'unknown'}`))
      // Without this a message that fails to deserialize settles nothing and the
      // call waits out the whole timeout.
      worker.onmessageerror = () => reject(new CfireantsError('registration-failed',
        'the registration worker sent a message that could not be deserialized'))
      const limit = options.timeoutMs
        ?? TIMEOUT_MS[chooseVariant(options.backend, options.threads, checkSupport().threaded)]
      timer = setTimeout(() => reject(new CfireantsError('timeout',
        `the registration did not finish within ${limit} ms`)), limit)
      // The inputs are COPIED, not transferred: transferring would neuter the
      // caller's buffers, and a caller registering the same fixed image twice
      // would get an empty one with no diagnostic.
      worker.postMessage({ fixed, moving, options: rest })
    })
  } finally {
    clearTimeout(timer)
    if (onAbort) signal!.removeEventListener('abort', onAbort)
    worker.terminate()
  }
}

function asBytes(input: ArrayBuffer | ArrayBufferView, name: string): Uint8Array {
  if (input instanceof ArrayBuffer) return new Uint8Array(input)
  if (ArrayBuffer.isView(input))
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength)
  throw new CfireantsError('bad-input', `${name} must be an ArrayBuffer or a typed array`)
}

/**
 * Register `moving` onto `fixed` and return the warped moving image.
 *
 * Both arguments are NIfTI file images, gzipped or not; the result is gzipped
 * if the moving input was, unless `gzip` says otherwise. Everything the
 * registration itself does -- moments, rigid, affine, SyN or greedy, at every
 * scale -- happens inside the wasm module. This function owns only what the
 * module deliberately does not: gzip, choosing the build, argv, and the
 * decision to refuse.
 *
 * On a cross-origin isolated page this runs the threaded build; anywhere else
 * it silently runs the single-threaded one, which is roughly 6x slower.
 * `result.variant` says which, and `checkSupport().reasons` says why.
 *
 * @example
 * const { image } = await register(await fixed.arrayBuffer(), await moving.arrayBuffer())
 */


export async function register(
  fixed: ArrayBuffer | ArrayBufferView,
  moving: ArrayBuffer | ArrayBufferView,
  options: RegisterOptions = {},
): Promise<RegisterResult> {
  const rawFixed = asBytes(fixed, 'fixed')
  const rawMoving = asBytes(moving, 'moving')
  if (rawFixed.byteLength === 0 || rawMoving.byteLength === 0)
    throw new CfireantsError('bad-input', 'fixed and moving must both be non-empty')

  // Built here even for the worker path, so a bad option is refused before a
  // worker and a thread pool are spawned for it.
  const args = buildArgs(options)

  const support = checkSupport()
  if (!support.supported)
    throw new CfireantsError('unsupported-environment',
      'this environment has no WebAssembly')

  // Defaulted to whether there IS a Worker, not to true: node has none, and a
  // default that threw there would make `worker: false` mandatory boilerplate.
  if (options.worker ?? typeof Worker !== 'undefined')
    return runInWorker(rawFixed, rawMoving, options)

  const variant = chooseVariant(options.backend, options.threads, support.threaded)
  const cap = threadCap(options.threads)
  const compressed = isGzip(rawMoving)
  const wantGzip = options.gzip ?? compressed
  const result = await run({
    variant,
    fixed: isGzip(rawFixed) ? await gunzip(rawFixed) : rawFixed,
    moving: compressed ? await gunzip(rawMoving) : rawMoving,
    args,
    assetPath: options.assetPath,
    timeoutMs: options.timeoutMs,
    onLog: options.onLog,
    signal: options.signal,
  })

  const out = wantGzip ? await gzip(result.image) : result.image
  return {
    // Copy out of the wasm heap: FS.readFile hands back a view whose backing
    // buffer the module may reuse or free.
    image: out.slice().buffer as ArrayBuffer,
    elapsedMs: result.elapsedMs,
    variant,
    threads: variant === 'mt' ? cap ?? support.threads : 1,
    log: result.log,
  }
}
