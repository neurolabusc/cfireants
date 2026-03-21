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
    if (zeros) {
        wgpu_write_buffer(buf, 0, zeros, nbytes);
        free(zeros);
    }

    t->data = (void *)buf;
    return 0;
}

void webgpu_tensor_free(WGPUBuffer buf) {
    if (buf) wgpuBufferRelease(buf);
}

int webgpu_memcpy_h2d(WGPUBuffer dst, const void *src, size_t nbytes) {
    wgpu_write_buffer(dst, 0, src, nbytes);
    return 0;
}

int webgpu_memcpy_d2h(void *dst, WGPUBuffer src, size_t nbytes) {
    wgpu_read_buffer(src, 0, dst, nbytes);
    return 0;
}

int webgpu_memcpy_d2d(WGPUBuffer dst, WGPUBuffer src, size_t nbytes) {
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(g_wgpu.device, NULL);
    size_t aligned = (nbytes + 3) & ~(size_t)3;
    wgpuCommandEncoderCopyBufferToBuffer(encoder, src, 0, dst, 0, aligned);
    WGPUCommandBuffer cmdbuf = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(g_wgpu.queue, 1, &cmdbuf);
    wgpuCommandBufferRelease(cmdbuf);
    wgpuCommandEncoderRelease(encoder);
    wgpuDevicePoll(g_wgpu.device, 1, NULL);
    return 0;
}
