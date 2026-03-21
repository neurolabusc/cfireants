/*
 * metal_context.m - Metal context initialization and dispatch helpers
 *
 * Manages the Metal device, command queue, shader library, pipeline cache,
 * and buffer tracking table. All GPU dispatches go through metal_dispatch().
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

#include "metal_context.h"

/* Global Metal context */
metal_context_t g_metal = {0};

/* --- Context lifecycle --- */

int metal_context_init(void) {
    @autoreleasepool {
        /* Create device */
        g_metal.device = MTLCreateSystemDefaultDevice();
        if (!g_metal.device) {
            fprintf(stderr, "metal_context_init: no Metal device found\n");
            return -1;
        }

        fprintf(stderr, "Metal backend: %s\n",
                [[g_metal.device name] UTF8String]);

        /* Create command queue */
        g_metal.queue = [g_metal.device newCommandQueue];
        if (!g_metal.queue) {
            fprintf(stderr, "metal_context_init: failed to create command queue\n");
            return -1;
        }

        /* Load shader library.
         * Try paths in order:
         *   1. build/backend/metal/cfireants.metallib (in-tree build)
         *   2. cfireants.metallib (next to executable)
         *   3. Default library (compiled into app bundle)
         */
        NSError *error = nil;
        NSArray<NSString *> *paths = @[
            @"build/cfireants.metallib",
            @"cfireants.metallib",
#ifdef METAL_LIBRARY_PATH
            @METAL_LIBRARY_PATH,
#endif
        ];

        g_metal.library = nil;
        for (NSString *path in paths) {
            g_metal.library = [g_metal.device newLibraryWithFile:path error:&error];
            if (g_metal.library) {
                fprintf(stderr, "Metal shaders: loaded from %s\n", [path UTF8String]);
                break;
            }
        }

        if (!g_metal.library) {
            /* Try default library (compiled into app bundle) */
            g_metal.library = [g_metal.device newDefaultLibrary];
            if (g_metal.library) {
                fprintf(stderr, "Metal shaders: loaded default library\n");
            }
        }

        if (!g_metal.library) {
            fprintf(stderr, "metal_context_init: warning: no shader library found "
                    "(kernels will fail to load)\n");
            /* Don't fail — context is still usable for buffer management.
             * Kernel dispatches will fail when pipelines can't be created. */
        }

        g_metal.n_pipelines = 0;
        g_metal.n_buffers = 0;
    }
    return 0;
}

void metal_context_cleanup(void) {
    @autoreleasepool {
        /* Release pipeline states */
        for (int i = 0; i < g_metal.n_pipelines; i++) {
            g_metal.pipelines[i].pipeline = nil;
            g_metal.pipelines[i].name = NULL;
        }
        g_metal.n_pipelines = 0;

        /* Release tracked buffers */
        for (int i = 0; i < g_metal.n_buffers; i++) {
            g_metal.buffers[i].buffer = nil;
            g_metal.buffers[i].cpu_ptr = NULL;
            g_metal.buffers[i].size = 0;
        }
        g_metal.n_buffers = 0;

        g_metal.library = nil;
        g_metal.queue = nil;
        g_metal.device = nil;
    }
}

/* --- Pipeline cache --- */

void *metal_get_pipeline(const char *function_name) {
    /* Check cache first */
    for (int i = 0; i < g_metal.n_pipelines; i++) {
        if (strcmp(g_metal.pipelines[i].name, function_name) == 0) {
            return (__bridge void *)g_metal.pipelines[i].pipeline;
        }
    }

    if (!g_metal.library) {
        fprintf(stderr, "metal_get_pipeline: no shader library loaded\n");
        return NULL;
    }

    if (g_metal.n_pipelines >= MTL_MAX_PIPELINES) {
        fprintf(stderr, "metal_get_pipeline: pipeline cache full (%d)\n",
                MTL_MAX_PIPELINES);
        return NULL;
    }

    @autoreleasepool {
        NSString *name = [NSString stringWithUTF8String:function_name];
        id<MTLFunction> func = [g_metal.library newFunctionWithName:name];
        if (!func) {
            fprintf(stderr, "metal_get_pipeline: function '%s' not found in library\n",
                    function_name);
            return NULL;
        }

        NSError *error = nil;
        id<MTLComputePipelineState> pso =
            [g_metal.device newComputePipelineStateWithFunction:func error:&error];
        if (!pso) {
            fprintf(stderr, "metal_get_pipeline: failed to create pipeline for '%s': %s\n",
                    function_name,
                    error ? [[error localizedDescription] UTF8String] : "unknown error");
            return NULL;
        }

        int idx = g_metal.n_pipelines++;
        g_metal.pipelines[idx].pipeline = pso;
        g_metal.pipelines[idx].name = function_name; /* must be a string literal */

        return (__bridge void *)pso;
    }
}

/* --- Buffer management --- */

/* Buffer table uses CFBridgingRetain/Release for manual refcount management.
   ARC cannot reliably manage id<> types inside C struct arrays. */

void metal_register_buffer(void *cpu_ptr, void *mtl_buffer, size_t size) {
    if (g_metal.n_buffers >= MTL_MAX_BUFFERS) {
        fprintf(stderr, "metal_register_buffer: buffer table full (%d)\n",
                MTL_MAX_BUFFERS);
        return;
    }
    /* Check for duplicate registration */
    for (int i = 0; i < g_metal.n_buffers; i++) {
        if (g_metal.buffers[i].cpu_ptr == cpu_ptr) {
            /* Release old, retain new */
            if (g_metal.buffers[i].buffer) CFRelease(g_metal.buffers[i].buffer);
            g_metal.buffers[i].buffer = CFBridgingRetain((__bridge id)mtl_buffer);
            g_metal.buffers[i].size = size;
            return;
        }
    }
    int idx = g_metal.n_buffers++;
    g_metal.buffers[idx].cpu_ptr = cpu_ptr;
    g_metal.buffers[idx].buffer = CFBridgingRetain((__bridge id)mtl_buffer);
    g_metal.buffers[idx].size = size;
}

void metal_unregister_buffer(void *cpu_ptr) {
    for (int i = 0; i < g_metal.n_buffers; i++) {
        if (g_metal.buffers[i].cpu_ptr == cpu_ptr) {
            /* Release the retained buffer */
            if (g_metal.buffers[i].buffer) CFRelease(g_metal.buffers[i].buffer);

            /* Swap with last entry to keep table compact */
            g_metal.n_buffers--;
            if (i < g_metal.n_buffers) {
                g_metal.buffers[i] = g_metal.buffers[g_metal.n_buffers];
            }
            g_metal.buffers[g_metal.n_buffers].cpu_ptr = NULL;
            g_metal.buffers[g_metal.n_buffers].buffer = NULL;
            g_metal.buffers[g_metal.n_buffers].size = 0;
            return;
        }
    }
    fprintf(stderr, "metal_unregister_buffer: ptr %p not found\n", cpu_ptr);
}

/* Look up the MTLBuffer containing cpu_ptr. Returns the buffer ref and
   sets *offset to the byte offset from the buffer start to cpu_ptr.
   Supports sub-buffer pointers (e.g., channel offsets within a larger buffer). */
static void *metal_buffer_from_ptr_offset(const void *cpu_ptr, size_t *offset) {
    for (int i = 0; i < g_metal.n_buffers; i++) {
        const char *base = (const char *)g_metal.buffers[i].cpu_ptr;
        const char *end = base + g_metal.buffers[i].size;
        if ((const char *)cpu_ptr >= base && (const char *)cpu_ptr < end) {
            *offset = (size_t)((const char *)cpu_ptr - base);
            return g_metal.buffers[i].buffer;
        }
    }
    *offset = 0;
    return NULL;
}

void *metal_buffer_from_ptr(const void *cpu_ptr) {
    size_t offset;
    return metal_buffer_from_ptr_offset(cpu_ptr, &offset);
}

size_t metal_buffer_size(const void *cpu_ptr) {
    for (int i = 0; i < g_metal.n_buffers; i++) {
        if (g_metal.buffers[i].cpu_ptr == cpu_ptr) {
            return g_metal.buffers[i].size;
        }
    }
    return 0;
}

/* --- Dispatch helpers --- */

void metal_dispatch(void *pipeline,
                    const void **buffer_ptrs, const size_t *buffer_sizes, int n_bufs,
                    const void *params, size_t params_size,
                    uint32_t total_threads) {
    @autoreleasepool {
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipeline;

        id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];

        /* Bind data buffers (supports sub-buffer offsets for pointer arithmetic) */
        for (int i = 0; i < n_bufs; i++) {
            size_t offset = 0;
            void *ref = metal_buffer_from_ptr_offset(buffer_ptrs[i], &offset);
            if (!ref) {
                fprintf(stderr, "metal_dispatch: buffer %d (ptr=%p) not found in table\n",
                        i, buffer_ptrs[i]);
                [enc endEncoding];
                return;
            }
            id<MTLBuffer> buf = (__bridge id<MTLBuffer>)ref;
            [enc setBuffer:buf offset:offset atIndex:i];
        }

        /* Bind params as a small buffer at index n_bufs */
        if (params && params_size > 0) {
            [enc setBytes:params length:params_size atIndex:n_bufs];
        }

        /* Dispatch threads */
        NSUInteger threadgroupSize = pso.maxTotalThreadsPerThreadgroup;
        if (threadgroupSize > MTL_THREADGROUP_SIZE) {
            threadgroupSize = MTL_THREADGROUP_SIZE;
        }

        MTLSize threads = MTLSizeMake(total_threads, 1, 1);
        MTLSize tgSize = MTLSizeMake(threadgroupSize, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:tgSize];

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void metal_dispatch_no_params(void *pipeline,
                              const void **buffer_ptrs, const size_t *buffer_sizes, int n_bufs,
                              uint32_t total_threads) {
    metal_dispatch(pipeline, buffer_ptrs, buffer_sizes, n_bufs,
                   NULL, 0, total_threads);
}

void metal_sync(void) {
    @autoreleasepool {
        /* Submit an empty command buffer and wait for it to complete.
         * This ensures all prior work on the queue has finished and
         * shared-memory buffers are CPU-coherent. */
        id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}
