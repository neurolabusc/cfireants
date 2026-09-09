#!/usr/bin/env node

import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const packageRoot = resolve(process.argv[2] ?? '.')
const fixtureRoot = resolve(process.argv[3] ?? '../validate/small')
const entry = pathToFileURL(resolve(packageRoot, 'dist/index.js')).href
const { register } = await import(entry)
const fixed = await readFile(resolve(fixtureRoot, 'MNI152_T1_2mm.nii.gz'))
const moving = await readFile(resolve(fixtureRoot, 'T1_head_2mm.nii.gz'))

const stage = [{
  transform: 'Rigid[0.003]',
  metric: 'MI[32]',
  convergence: '[1,1e-6,1]',
  shrinkFactors: '1',
}]

for (const threads of [false, 2]) {
  const result = await register(fixed, moving, {
    stages: stage,
    threads,
    worker: false,
    gzip: false,
  })
  const bytes = new Uint8Array(result.image)
  const header = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  if (bytes.byteLength <= 352 || header.getInt32(0, true) !== 348)
    throw new Error(`invalid NIfTI output from ${result.variant} package variant`)
  const expected = threads === false ? 'st' : 'mt'
  if (result.variant !== expected)
    throw new Error(`requested ${expected}, package selected ${result.variant}`)
  console.log(`smoke: ${result.variant}, ${bytes.byteLength} bytes, ${Math.round(result.elapsedMs)} ms`)
}
