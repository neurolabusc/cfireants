import { CfireantsError } from './error.js'

/*
 * gzip lives here and nowhere else.
 *
 * The wasm module links no zlib: the C reads and writes plain `.nii` only, so
 * this file is the whole of the compression contract, using the platform's own
 * streams rather than a bundled inflate.
 */

export function isGzip(bytes: Uint8Array): boolean {
  // NIfTI-1's own first word is the header size, 348, so there is no ambiguity.
  return bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b
}

async function through(bytes: Uint8Array, stream: TransformStream<Uint8Array, Uint8Array>) {
  // A fresh Uint8Array, because a view into the wasm heap (or any
  // SharedArrayBuffer, which is what a -pthread module's heap is) is not
  // something Response will accept.
  const copy = new Uint8Array(bytes)
  const body = new Blob([copy as BlobPart]).stream().pipeThrough(stream)
  return new Uint8Array(await new Response(body).arrayBuffer())
}

export async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream === 'undefined')
    throw new CfireantsError('no-compression-streams',
      'the input is gzipped but this environment has no DecompressionStream; ' +
      'decompress it before calling, or pass an uncompressed NIfTI')
  return through(bytes, new DecompressionStream('gzip') as unknown as
    TransformStream<Uint8Array, Uint8Array>)
}

export async function gzip(bytes: Uint8Array): Promise<Uint8Array> {
  if (typeof CompressionStream === 'undefined')
    throw new CfireantsError('no-compression-streams',
      'gzip output was requested but this environment has no CompressionStream; ' +
      'pass gzip: false')
  return through(bytes, new CompressionStream('gzip') as unknown as
    TransformStream<Uint8Array, Uint8Array>)
}
