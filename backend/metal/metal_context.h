/*
 * metal_context.h - Metal device/queue management and pipeline cache
 *
 * Provides a global Metal context (device, queue, pipeline cache) and
 * helpers for dispatching compute kernels. Includable from both
 * Objective-C (.m) and plain C (.c) files.
 */

#ifndef CFIREANTS_METAL_CONTEXT_H
#define CFIREANTS_METAL_CONTEXT_H

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#else
/* Opaque types for C headers */
typedef void *MTLDeviceRef;
typedef void *MTLCommandQueueRef;
typedef void *MTLLibraryRef;
typedef void *MTLComputePipelineStateRef;
typedef void *MTLBufferRef;
#endif

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MTL_MAX_PIPELINES 128
#define MTL_MAX_BUFFERS 4096
#define MTL_THREADGROUP_SIZE 256

typedef struct {
#ifdef __OBJC__
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;
#else
    MTLDeviceRef device;
    MTLCommandQueueRef queue;
    MTLLibraryRef library;
#endif

    struct {
#ifdef __OBJC__
        id<MTLComputePipelineState> pipeline;
#else
        MTLComputePipelineStateRef pipeline;
#endif
        const char *name;
    } pipelines[MTL_MAX_PIPELINES];
    int n_pipelines;

    /* Buffer tracking: map cpu_ptr -> MTLBuffer for encoder setBuffer calls.
       Buffers stored as CFTypeRef (void*) with manual retain/release to avoid
       ARC issues with id<> types in C struct arrays. */
    struct {
        void *cpu_ptr;
        void *buffer;  /* CFTypeRef to MTLBuffer — manually retained */
        size_t size;
    } buffers[MTL_MAX_BUFFERS];
    int n_buffers;
} metal_context_t;

extern metal_context_t g_metal;

/* Initialize Metal device, queue, and shader library. Returns 0 on success. */
int metal_context_init(void);

/* Cleanup all Metal resources */
void metal_context_cleanup(void);

/* --- Pipeline cache --- */
/* Get or create a compute pipeline for a named kernel function.
   The function must exist in the pre-compiled metallib. */
void *metal_get_pipeline(const char *function_name);

/* --- Buffer management --- */
/* Register a newly created MTLBuffer in the tracking table.
   cpu_ptr = buffer.contents. Called by metal_tensor_alloc. */
void metal_register_buffer(void *cpu_ptr, void *mtl_buffer, size_t size);

/* Unregister a buffer. Called by metal_tensor_free. */
void metal_unregister_buffer(void *cpu_ptr);

/* Look up the MTLBuffer for a given cpu_ptr (from tensor_t.data).
   Returns the MTLBuffer (as void* for C compatibility), or NULL if not found. */
void *metal_buffer_from_ptr(const void *cpu_ptr);

/* Look up buffer size for a given cpu_ptr. Returns 0 if not found. */
size_t metal_buffer_size(const void *cpu_ptr);

/* --- Dispatch helpers --- */
/* Dispatch a compute kernel with the given pipeline, buffers, and thread count.
   Submits immediately and waits for completion.
   buffer_ptrs: array of cpu_ptr values (looked up in buffer table)
   buffer_sizes: array of buffer sizes (for validation, may be NULL)
   n_bufs: number of buffer bindings
   params: small struct to copy into a temporary buffer at binding n_bufs
   params_size: size of params struct
   total_threads: total number of threads to launch */
void metal_dispatch(void *pipeline,
                    const void **buffer_ptrs, const size_t *buffer_sizes, int n_bufs,
                    const void *params, size_t params_size,
                    uint32_t total_threads);

/* Dispatch without params (just buffers) */
void metal_dispatch_no_params(void *pipeline,
                              const void **buffer_ptrs, const size_t *buffer_sizes, int n_bufs,
                              uint32_t total_threads);

/* Synchronize: wait for all GPU work to complete.
   After this call, all shared-memory buffers are CPU-readable. */
void metal_sync(void);

/* Compute ceil(n / d) */
static inline uint32_t metal_div_ceil(uint32_t n, uint32_t d) {
    return (n + d - 1) / d;
}

#ifdef __cplusplus
}
#endif

#endif /* CFIREANTS_METAL_CONTEXT_H */
