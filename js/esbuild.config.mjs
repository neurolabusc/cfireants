import { build } from 'esbuild'

// The emscripten glue is deliberately NOT in this bundle; scripts/copy-wasm.mjs
// stages it into dist/ and run.ts imports it by URL at run time. `platform:
// neutral` keeps esbuild from injecting node or browser shims into what is a
// plain ES module.
const common = {
  bundle: true,
  format: 'esm',
  platform: 'neutral',
  target: ['es2022'],
  sourcemap: true,
  legalComments: 'inline',
}

await build({ ...common, entryPoints: ['src/index.ts'], outfile: 'dist/index.js' })

// The worker is a SEPARATE, self-contained bundle rather than a module that
// imports ./index.js at run time: a host that bundles the package rewrites its
// own copy of index.js, and a worker reaching for a relative './index.js' would
// then find nothing. It is loaded by URL, so it must not be part of index's
// bundle either.
await build({ ...common, entryPoints: ['src/worker.ts'], outfile: 'dist/worker.js' })

console.log('esbuild: wrote dist/index.js and dist/worker.js')
