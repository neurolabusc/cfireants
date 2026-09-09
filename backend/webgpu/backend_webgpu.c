/*
 * backend_webgpu.c - WebGPU buffer management for tensor_t
 *
 * WebGPU buffers are stored as opaque handles in tensor_t.data (cast to void*).
 * We maintain a side table mapping buffer handles to WGPUBuffer objects since
 * WebGPU buffers are typedef'd pointers (compatible with void*).
 */

#include "webgpu_context.h"
#include "cfireants/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- Buffer tracking ---
 * We store the WGPUBuffer directly in tensor_t.data since WGPUBuffer
 * is already a pointer type (WGPUBufferImpl*). For H2D/D2H we use
 * the queue write/read helpers.
 */

int webgpu_tensor_alloc(tensor_t *t, size_t nbytes) {
    WGPUBuffer buf = wgpu_create_buffer(nbytes,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
        "tensor");
    if (!buf) return -1;

    /* Zero-initialize */
    void *zeros = calloc(1, nbytes);
    if (!zeros) {
        fprintf(stderr, "webgpu_tensor_alloc: host zero buffer allocation failed\n");
        wgpu_record_fatal_error("tensor initialization");
        wgpu_release_buffer(buf);
        return -1;
    }
    wgpu_write_buffer(buf, 0, zeros, nbytes);
    free(zeros);

    t->data = (void *)buf;
    return 0;
}

void webgpu_tensor_free(WGPUBuffer buf) {
    if (buf) wgpu_release_buffer(buf);
}

int webgpu_memcpy_h2d(WGPUBuffer dst, const void *src, size_t nbytes) {
    wgpu_write_buffer(dst, 0, src, nbytes);
    return wgpu_had_fatal_error() ? -1 : 0;
}

int webgpu_memcpy_d2h(void *dst, WGPUBuffer src, size_t nbytes) {
    wgpu_read_buffer(src, 0, dst, nbytes);
    return wgpu_had_fatal_error() ? -1 : 0;
}

int webgpu_memcpy_d2d(WGPUBuffer dst, WGPUBuffer src, size_t nbytes) {
    size_t aligned = (nbytes + 3) & ~(size_t)3;
    wgpu_copy_buffer(src, dst, aligned);
    return wgpu_had_fatal_error() ? -1 : 0;
}
