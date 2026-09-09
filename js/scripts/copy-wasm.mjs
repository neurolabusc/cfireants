#!/usr/bin/env node
// Stage the emscripten artifacts into dist/ beside the bundled index.js.
//
// They are copied rather than bundled: the glue finds its .wasm through
// import.meta.url, which a bundler would rewrite to the bundle's own location,
// so the pair has to stay adjacent and unhashed.

import { copyFileSync, existsSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const js = join(here, '..')
const to = join(js, 'dist')
// CPU builds: `-mt` is the -pthread one and the bare pair is single-threaded.
// The GPU pair is selected explicitly with `backend: 'webgpu'`.
const files = [
  'cfireants-mt.js', 'cfireants-mt.wasm',
  'cfireants.js', 'cfireants.wasm',
  'cfireants-gpu.js', 'cfireants-gpu.wasm',
]

// build-wasm.sh owns where it writes; CFIREANTS_WASM_DIR is the override when
// that is somewhere else.
const candidates = process.env.CFIREANTS_WASM_DIR
  ? [process.env.CFIREANTS_WASM_DIR]
  : [join(js, 'wasm')]

const from = candidates.find((dir) => files.every((f) => existsSync(join(dir, f))))
if (!from) {
  console.error(`copy-wasm: no directory holds all of ${files.join(', ')}; looked in:`)
  for (const dir of candidates) console.error(`  ${dir}`)
  console.error('copy-wasm: run `npm run makeWasm`, or set CFIREANTS_WASM_DIR')
  process.exit(1)
}

mkdirSync(to, { recursive: true })
for (const f of files) copyFileSync(join(from, f), join(to, f))
console.log(`copy-wasm: staged ${files.join(', ')} from ${from}`)
