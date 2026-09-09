/**
 * Which wasm build ran. `mt` and `st` are the automatically selected CPU
 * builds; `gpu` is the browser WebGPU build selected by `backend`.
 */
export type Variant = 'mt' | 'st' | 'gpu'

/** Execution backend. CPU chooses the best threaded wasm variant automatically. */
export type Backend = 'cpu' | 'webgpu'

/** The four presets `cfireants_reg` offers. Each is a fixed stage sequence. */
export type Transform = 'rigid' | 'affine' | 'syn' | 'greedy'

/**
 * One registration stage, as the CLI spells it.
 *
 * The fields are passed to `--transform`, `--metric`, `--convergence` and
 * `--shrink-factors` VERBATIM. They are strings rather than parsed structures
 * because the C already owns that grammar and mirroring it here would be a
 * second parser to keep in step for no gain. Note there is no shell involved:
 * write `SyN[0.1,0.5,1.0]`, never `'SyN[0.1,0.5,1.0]'`.
 */
export interface Stage {
  /** `Rigid[lr]`, `Affine[lr]`, `SyN[lr,warp_sigma,grad_sigma]`, `Greedy[...]`. */
  transform: string
  /** `MI[bins]` or `CC[kernel]`. Defaults to MI for linear, CC for deformable. */
  metric?: string
  /** `[200x100x50,1e-6,10]` -- iterations per level, tolerance, window. */
  convergence?: string
  /** `4x2x1` -- downsample factor per level. */
  shrinkFactors?: string
}

export interface RegisterOptions {
  /** Registration backend. Defaults to `cpu`; `webgpu` requires browser WebGPU. */
  backend?: Backend
  /** Preset stage sequence. Default `'syn'` (moments + rigid + affine + SyN). */
  transform?: Transform
  /** Explicit stages, in order. Overrides `transform` entirely. */
  stages?: Stage[]
  /** 0 silent (default), 1 summary, 2 per-iteration. `onLog` is only fed what this emits. */
  verbose?: 0 | 1 | 2
  /** gzip the returned buffer. Defaults to whatever the MOVING input was. */
  gzip?: boolean
  /**
   * Run in a Web Worker. Defaults to true wherever `Worker` exists, which is
   * the opposite of the usual default and deliberate: a registration is tens of
   * seconds to minutes, and the worker is also the only real cancellation --
   * terminate() ends the thread pool, where an in-thread timeout can only stop
   * waiting for it. Node has no global `Worker`, so it defaults to false there.
   */
  worker?: boolean
  /**
   * Base URL the cfireants glue and wasm artifacts are served from. Defaults
   * to this module's own directory. Set it when a bundler has moved things;
   * must be same-origin, because it feeds a dynamic `import()` and a Worker.
   */
  assetPath?: string
  /**
   * CPU threading, which also selects the CPU wasm build. Omit for WebGPU.
   *
   * Omitted or `true`: use the threaded build where the page is cross-origin
   * isolated, and the single-threaded one where it is not. That fallback is
   * SILENT -- `result.variant` says which ran -- because a page served from
   * GitHub Pages cannot set COOP/COEP and would otherwise get nothing.
   *
   * `false` forces the single-threaded build even on an isolated page. A
   * positive integer caps the pool via the `--threads` flag and still selects
   * the threaded build, since the single-threaded one cannot honour any value
   * but 1.
   */
  threads?: boolean | number
  /**
   * Give up after this many ms. Defaults to 900000 for `mt`/`gpu` and 1800000
   * for `st`, which is roughly 6x slower: SyN on a 1 mm brain is several
   * minutes there.
   */
  timeoutMs?: number
  /** Receives the module's output lines as they are produced. */
  onLog?: (line: string) => void
  /** Stops waiting, and under `worker: true` terminates the module for real. */
  signal?: AbortSignal
  /** Appended to argv verbatim, after everything else. Escape hatch for flags this API omits. */
  args?: string[]
}

export interface RegisterResult {
  /** The warped moving image, as a NIfTI file image. gzipped per `gzip`. */
  image: ArrayBuffer
  /** Milliseconds spent inside the wasm module, excluding fetch and gzip. */
  elapsedMs: number
  /** Which build ran. `gpu` proves WebGPU was selected rather than a CPU fallback. */
  variant: Variant
  /** Threads the module was allowed: 1 for `st`/`gpu`, else the cap or hardwareConcurrency. */
  threads: number
  /** Everything the module wrote to stdout and stderr, joined by newlines. */
  log: string
}

export interface SupportReport {
  /** Whether `register()` can run at all. False only without WebAssembly. */
  supported: boolean
  /** Whether the threaded build is usable here. */
  threaded: boolean
  /** Which build `register()` would choose right now, absent a `threads` option. */
  variant: Variant
  /** Threads that variant would use. */
  threads: number
  /** Why threading is unavailable, in the caller's terms. Empty when it is. */
  reasons: string[]
}
