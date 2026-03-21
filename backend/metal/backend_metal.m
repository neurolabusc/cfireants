/*
 * backend_metal.m - Metal memory management for cfireants
 *
 * Provides Metal tensor allocation/free and memory transfer functions.
 * Uses MTLStorageModeShared buffers for unified CPU/GPU memory access.
 * Called from tensor.c via forward declarations.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

#include "cfireants/tensor.h"
#include "metal_context.h"

/* --- Metal memory management (called from tensor.c) --- */

int metal_tensor_alloc(tensor_t *t, size_t nbytes) {
    @autoreleasepool {
        if (!g_metal.device) {
            fprintf(stderr, "metal_tensor_alloc: Metal context not initialized\n");
            return -1;
        }

        /* Allocate a shared buffer (CPU + GPU accessible) */
        id<MTLBuffer> buf = [g_metal.device newBufferWithLength:nbytes
                                                        options:MTLResourceStorageModeShared];
        if (!buf) {
            fprintf(stderr, "metal_tensor_alloc: failed to allocate %zu bytes\n", nbytes);
            return -1;
        }

        /* Zero-fill */
        memset(buf.contents, 0, nbytes);

        /* Store the CPU-accessible pointer in tensor data */
        t->data = buf.contents;

        /* Register in buffer tracking table.
         * The context struct holds a strong reference (ARC) via the
         * id<MTLBuffer> member. We pass as __bridge void* — the struct
         * assignment in metal_register_buffer retains it. */
        metal_register_buffer(t->data, (__bridge void *)buf, nbytes);

        return 0;
    }
}

void metal_tensor_free(void *ptr) {
    if (!ptr) return;

    @autoreleasepool {
        /* Unregister from buffer table.
         * The struct holds a strong reference (id<MTLBuffer>), so setting
         * it to nil in metal_unregister_buffer releases the MTLBuffer. */
        void *buf_ptr = metal_buffer_from_ptr(ptr);
        if (buf_ptr) {
            metal_unregister_buffer(ptr);
        } else {
            fprintf(stderr, "metal_tensor_free: ptr %p not found in buffer table\n", ptr);
        }
    }
}

int metal_memcpy_h2d(void *dst, const void *src, size_t nbytes) {
    /* With shared memory, dst is already CPU-accessible.
     * Just memcpy the data — it will be visible to GPU on next dispatch. */
    memcpy(dst, src, nbytes);
    return 0;
}

int metal_memcpy_d2h(void *dst, const void *src, size_t nbytes) {
    /* Ensure GPU work is complete before reading shared memory */
    metal_sync();
    memcpy(dst, src, nbytes);
    return 0;
}

int metal_memcpy_d2d(void *dst, const void *src, size_t nbytes) {
    /* Both pointers are shared memory, CPU-accessible */
    memcpy(dst, src, nbytes);
    return 0;
}

int metal_memset(void *ptr, int value, size_t nbytes) {
    memset(ptr, value, nbytes);
    return 0;
}
