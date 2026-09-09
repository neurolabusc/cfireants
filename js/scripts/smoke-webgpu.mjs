#!/usr/bin/env node

import { execFileSync } from 'node:child_process'
import { createServer } from 'node:http'
import { readFile, stat } from 'node:fs/promises'
import { extname, resolve, sep } from 'node:path'
import puppeteer from 'puppeteer-core'

const packageRoot = resolve(process.argv[2] ?? '.')

function chromeExecutable() {
  const candidates = [
    process.env.CHROME_PATH,
    process.platform === 'darwin'
      ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
      : undefined,
    process.platform === 'darwin'
      ? '/Applications/Chromium.app/Contents/MacOS/Chromium'
      : undefined,
  ].filter(Boolean)
  for (const candidate of candidates) {
    try {
      execFileSync('test', ['-x', candidate])
      return candidate
    } catch { /* try PATH next */ }
  }
  for (const name of ['google-chrome', 'chromium', 'chromium-browser']) {
    try {
      return execFileSync('which', [name], { encoding: 'utf8' }).trim()
    } catch { /* try the next browser name */ }
  }
  throw new Error('Chrome/Chromium was not found; set CHROME_PATH')
}

function makeNifti(shiftX) {
  const [nx, ny, nz] = [24, 20, 16]
  const voxels = nx * ny * nz
  const bytes = new Uint8Array(352 + voxels * 4)
  const view = new DataView(bytes.buffer)
  view.setInt32(0, 348, true)
  view.setInt16(40, 3, true)
  view.setInt16(42, nx, true)
  view.setInt16(44, ny, true)
  view.setInt16(46, nz, true)
  view.setInt16(48, 1, true)
  view.setInt16(70, 16, true) // float32
  view.setInt16(72, 32, true)
  view.setFloat32(76, 1, true)
  view.setFloat32(80, 1, true)
  view.setFloat32(84, 1, true)
  view.setFloat32(88, 1, true)
  view.setFloat32(108, 352, true)
  view.setInt16(254, 1, true) // sform_code
  view.setFloat32(280, 1, true)
  view.setFloat32(300, 1, true)
  view.setFloat32(320, 1, true)
  bytes.set([0x6e, 0x2b, 0x31, 0], 344) // n+1\0

  const cx = (nx - 1) / 2 + shiftX
  const cy = (ny - 1) / 2 - 0.35 * shiftX
  const cz = (nz - 1) / 2 + 0.2 * shiftX
  let i = 0
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++, i++) {
        const dx = (x - cx) / 5.2
        const dy = (y - cy) / 4.1
        const dz = (z - cz) / 3.4
        const main = Math.exp(-0.5 * (dx * dx + dy * dy + dz * dz))
        const lobe = 0.37 * Math.exp(-0.5 * (
          ((x - cx + 4.2) / 2.1) ** 2 +
          ((y - cy - 2.7) / 2.8) ** 2 +
          ((z - cz + 1.4) / 2.0) ** 2))
        const texture = 0.025 * Math.sin(x * 0.61 + y * 0.23 + z * 0.37)
        view.setFloat32(352 + i * 4, 800 * (main + lobe + texture), true)
      }
    }
  }
  return bytes
}

const fixed = makeNifti(0)
const moving = makeNifti(1.15)
const contentTypes = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
}

const server = createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname
    if (pathname === '/') {
      response.writeHead(200, { 'content-type': contentTypes['.html'] })
      response.end('<!doctype html><meta charset="utf-8"><title>cfireants smoke</title>')
      return
    }
    if (pathname === '/favicon.ico') {
      response.writeHead(204)
      response.end()
      return
    }
    if (pathname === '/fixed.nii' || pathname === '/moving.nii') {
      const body = pathname === '/fixed.nii' ? fixed : moving
      response.writeHead(200, { 'content-type': 'application/octet-stream' })
      response.end(body)
      return
    }
    if (!pathname.startsWith('/package/')) throw new Error('not found')
    const file = resolve(packageRoot, pathname.slice('/package/'.length))
    if (file !== packageRoot && !file.startsWith(packageRoot + sep))
      throw new Error('path traversal')
    const info = await stat(file)
    if (!info.isFile()) throw new Error('not a file')
    response.writeHead(200, {
      'content-type': contentTypes[extname(file)] ?? 'application/octet-stream',
      'cache-control': 'no-store',
    })
    response.end(await readFile(file))
  } catch {
    response.writeHead(404)
    response.end('not found')
  }
})

await new Promise((resolveListen, reject) => {
  server.once('error', reject)
  server.listen(0, '127.0.0.1', resolveListen)
})
const address = server.address()
if (!address || typeof address === 'string') throw new Error('HTTP server did not bind')

// Twenty minutes: the CI job allows thirty. Real GPUs finish in seconds.
const RUN_BUDGET_MS = 1_200_000
let browser
try {
  browser = await puppeteer.launch({
    executablePath: chromeExecutable(),
    headless: true,
    // The whole CPU+GPU run is one evaluate() call; puppeteer's default 180 s
    // protocol timeout is shorter than SwiftShader needs on a 2-core runner.
    protocolTimeout: RUN_BUDGET_MS,
    args: process.platform === 'linux' ? [
      '--no-sandbox',
      '--enable-unsafe-webgpu',
      '--enable-unsafe-swiftshader',
      '--use-angle=vulkan',
      '--use-vulkan=swiftshader',
      '--enable-features=Vulkan',
      '--disable-vulkan-surface',
    ] : ['--enable-unsafe-webgpu'],
  })
  const page = await browser.newPage()
  page.setDefaultTimeout(RUN_BUDGET_MS)
  page.on('console', (message) => console.log(`browser: ${message.text()}`))
  page.on('pageerror', (error) => console.error(`browser error: ${error.message}`))
  await page.goto(`http://127.0.0.1:${address.port}/`, { waitUntil: 'load' })

  const result = await page.evaluate(async (RUN_BUDGET_MS) => {
    const { register } = await import('/package/dist/index.js')
    if (!navigator.gpu) throw new Error('navigator.gpu is unavailable')
    console.log('smoke: requesting adapter')
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) throw new Error('requestAdapter returned null')
    console.log(`smoke: adapter ${adapter.info?.device || adapter.info?.description || '?'}`)

    const [fixedBuffer, movingBuffer] = await Promise.all([
      fetch('/fixed.nii').then((response) => response.arrayBuffer()),
      fetch('/moving.nii').then((response) => response.arrayBuffer()),
    ])
    const stages = [
      { transform: 'Rigid[0.003]', metric: 'MI[16]',
        convergence: '[2,1e-6,1]', shrinkFactors: '1' },
      { transform: 'Affine[0.001]', metric: 'MI[16]',
        convergence: '[2,1e-6,1]', shrinkFactors: '1' },
      { transform: 'SyN[0.1,0.5,1.0]', metric: 'CC[3]',
        convergence: '[2,1e-6,1]', shrinkFactors: '1' },
    ]
    // Exercise the package's default deployment path too: worker bootstrap,
    // dynamic module import, colocated WASM discovery, and WebGPU acquisition
    // all happen inside the packaged worker rather than this test page.
    const common = { stages, worker: true, gzip: false, timeoutMs: RUN_BUDGET_MS,
      onLog: (line) => console.log(line) }
    console.log('smoke: cpu run')
    const cpu = await register(fixedBuffer, movingBuffer, {
      ...common, backend: 'cpu', threads: false,
    })
    console.log('smoke: webgpu run')
    const gpu = await register(fixedBuffer, movingBuffer, {
      ...common, backend: 'webgpu',
    })

    const voxels = (image) => {
      const header = new DataView(image)
      if (header.getInt32(0, true) !== 348 || header.getInt16(70, true) !== 16)
        throw new Error('smoke output is not little-endian float32 NIfTI-1')
      const offset = header.getFloat32(108, true)
      return new Float32Array(image, offset)
    }
    const a = voxels(cpu.image)
    const b = voxels(gpu.image)
    if (a.length !== b.length) throw new Error('CPU/WebGPU output sizes differ')
    let sumA = 0, sumB = 0, minA = Infinity, maxA = -Infinity
    for (let i = 0; i < a.length; i++) {
      sumA += a[i]; sumB += b[i]
      minA = Math.min(minA, a[i]); maxA = Math.max(maxA, a[i])
    }
    const meanA = sumA / a.length
    const meanB = sumB / b.length
    let aa = 0, bb = 0, ab = 0, squareError = 0
    for (let i = 0; i < a.length; i++) {
      const da = a[i] - meanA
      const db = b[i] - meanB
      aa += da * da; bb += db * db; ab += da * db
      squareError += (a[i] - b[i]) ** 2
    }
    return {
      cpuVariant: cpu.variant,
      gpuVariant: gpu.variant,
      correlation: ab / Math.sqrt(aa * bb),
      normalizedRmse: Math.sqrt(squareError / a.length) / Math.max(maxA - minA, 1e-6),
      cpuMs: cpu.elapsedMs,
      gpuMs: gpu.elapsedMs,
      adapter: adapter.info?.device || adapter.info?.description || 'WebGPU adapter',
    }
  }, RUN_BUDGET_MS)

  if (result.cpuVariant !== 'st' || result.gpuVariant !== 'gpu')
    throw new Error(`wrong variants: CPU=${result.cpuVariant}, WebGPU=${result.gpuVariant}`)
  if (!(result.correlation >= 0.995) || !(result.normalizedRmse <= 0.03))
    throw new Error(`CPU/WebGPU browser parity failed: ${JSON.stringify(result)}`)
  console.log(
    `browser smoke: ${result.adapter}; corr=${result.correlation.toFixed(6)}, ` +
    `nRMSE=${result.normalizedRmse.toFixed(6)}, ` +
    `CPU=${Math.round(result.cpuMs)}ms, WebGPU=${Math.round(result.gpuMs)}ms`,
  )
} finally {
  await browser?.close()
  await new Promise((resolveClose) => server.close(resolveClose))
}
