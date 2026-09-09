/*
 * The worker entry point: one registration, off the calling thread.
 *
 * Bundled STANDALONE rather than importing ./index.js at run time. A host that
 * bundles the package (Vite, Rollup) rewrites its own copy of index.js, and a
 * worker reaching for a relative './index.js' would then find nothing; carrying
 * the code costs a few tens of KB and removes the failure mode entirely.
 *
 * Nested workers are fine and this topology depends on it: emscripten spawns
 * its pthread pool from inside this worker.
 */

import { CfireantsError } from './error.js'
import { register } from './index.js'
import type { RegisterOptions, RegisterResult } from './types.js'

/** Everything the main thread sends. `options` is already structured-clonable. */
export interface WorkerRequest {
  fixed: Uint8Array
  moving: Uint8Array
  options: Omit<RegisterOptions, 'onLog' | 'signal' | 'worker'>
}

export type WorkerResponse =
  | { type: 'log'; line: string }
  | { type: 'done'; result: RegisterResult }
  | { type: 'error'; code: string; message: string; log?: string }

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const post = (message: WorkerResponse, transfer: Transferable[] = []) =>
    (self as unknown as Worker).postMessage(message, transfer)
  try {
    const result = await register(event.data.fixed, event.data.moving, {
      ...event.data.options,
      // worker: false explicitly -- the option defaults to TRUE, so inheriting
      // it here would spawn workers forever.
      worker: false,
      // Logs cannot be a callback across the boundary, so they are streamed.
      onLog: (line) => post({ type: 'log', line }),
    })
    // The image is transferred, not copied: this side is done with it.
    post({ type: 'done', result }, [result.image])
  } catch (error) {
    const e = error as CfireantsError
    post({
      type: 'error',
      code: e?.code ?? 'registration-failed',
      message: e?.message ?? String(error),
      log: e?.log,
    })
  }
}
