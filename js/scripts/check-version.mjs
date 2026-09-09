#!/usr/bin/env node

import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const js = join(here, '..')
const root = join(js, '..')
const pkg = JSON.parse(readFileSync(join(js, 'package.json'), 'utf8'))
const lock = JSON.parse(readFileSync(join(js, 'package-lock.json'), 'utf8'))
const main = readFileSync(join(root, 'src', 'main.c'), 'utf8')
const cVersion = main.match(/^#define CFIREANTS_VERSION "([^"]+)"/m)?.[1]
const versions = new Map([
  ['src/main.c', cVersion],
  ['js/package.json', pkg.version],
  ['js/package-lock.json', lock.version],
  ['js/package-lock.json root package', lock.packages?.['']?.version],
])

const expected = pkg.version
for (const [source, version] of versions) {
  if (!version || version !== expected) {
    console.error(`version mismatch: ${source} has ${version ?? 'no version'}, expected ${expected}`)
    process.exit(1)
  }
}

const tag = process.env.RELEASE_TAG
if (tag) {
  if (!tag.startsWith('v') || tag.slice(1) !== expected) {
    console.error(`release tag ${tag} does not match source version v${expected}`)
    process.exit(1)
  }
}

console.log(`version: ${expected}${tag ? ` (${tag})` : ''}`)
