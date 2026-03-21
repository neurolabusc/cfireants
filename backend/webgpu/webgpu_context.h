/*
 * webgpu_context.h - WebGPU device/queue management and shader compilation
 *
 * Provides a global WebGPU context (device, queue, pipeline cache) and
 * helpers for creating compute pipelines from WGSL source.
 */

#ifndef CFIREANTS_WEBGPU_CONTEXT_H
#define CFIREANTS_WEBGPU_CONTEXT_H

#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of cached compute pipelines */
#define WGPU_MAX_PIPELINES 128

/* Standard workgroup size (matches CUDA BLOCK_SIZE) */
#define WGPU_WORKGROUP_SIZE 256

typedef struct {
    WGPUInstance   instance;
    WGPUAdapter    adapter;
    WGPUDevice     device;
    WGPUQueue      queue;

    /* Pipeline cache: compiled compute pipelines keyed by name */
    struct {
        WGPUComputePipeline pipeline;
        WGPUBindGroupLayout layout;
        const char *name;
    } pipelines[WGPU_MAX_PIPELINES];
    int n_pipelines;

    /* Reusable staging buffer for readback (grown as needed) */
    WGPUBuffer staging_buf;
    size_t     staging_size;

    /* Batch mode: accumulate dispatches into a single command encoder */
    WGPUCommandEncoder  batch_encoder;
    WGPUComputePassEncoder batch_pass;
    int batch_active;      /* 1 = dispatches are being batched */
    int batch_dispatches;  /* number of dispatches in current batch */
} wgpu_context_t;

/* Global context */
extern wgpu_context_t g_wgpu;

/* Initialize the WebGPU context (headless compute, Vulkan backend).
   Returns 0 on success, -1 on error. */
int wgpu_context_init(void);

/* Cleanup all resources */
void wgpu_context_cleanup(void);

/* --- Shader and pipeline helpers --- */

/* Compile a WGSL source string into a shader module.
   Returns NULL on failure. Caller must release. */
WGPUShaderModule wgpu_create_shader(const char *wgsl_source, const char *label);

/* Get or create a compute pipeline from WGSL source + entry point.
   Pipelines are cached by name. Returns NULL on failure. */
WGPUComputePipeline wgpu_get_pipeline(const char *name,
                                       const char *wgsl_source,
                                       const char *entry_point);

/* Get the bind group layout for a cached pipeline (group 0).
   Must call wgpu_get_pipeline first. Returns NULL if not found. */
WGPUBindGroupLayout wgpu_get_bind_group_layout(const char *name);

/* --- Buffer helpers --- */

/* Create a storage buffer with given usage flags. */
WGPUBuffer wgpu_create_buffer(size_t size, WGPUBufferUsage usage, const char *label);

/* Create a storage buffer initialized with data (uses mappedAtCreation). */
WGPUBuffer wgpu_create_buffer_init(const void *data, size_t size,
                                    WGPUBufferUsage usage, const char *label);

/* Ensure the staging buffer is at least `size` bytes. */
void wgpu_ensure_staging(size_t size);

/* --- Dispatch helpers --- */

/* Submit a compute dispatch. In batch mode, appends to the current batch.
   Outside batch mode, creates+submits+polls immediately (legacy behavior). */
void wgpu_dispatch(WGPUComputePipeline pipeline,
                   WGPUBindGroup bind_group,
                   uint32_t workgroup_count_x,
                   uint32_t workgroup_count_y,
                   uint32_t workgroup_count_z);

/* Batch mode: accumulate dispatches without submitting.
   wgpu_flush() or wgpu_read_buffer() will submit+poll the batch. */
void wgpu_begin_batch(void);

/* Flush the current batch: end pass, finish encoder, submit, poll.
   No-op if no batch is active. */
void wgpu_flush(void);

/* Read `size` bytes from a GPU buffer into `dst` (host memory).
   Auto-flushes any pending batch before reading. */
void wgpu_read_buffer(WGPUBuffer src, size_t offset, void *dst, size_t size);

/* Write `size` bytes from `src` (host memory) into a GPU buffer.
   Auto-flushes any pending batch before writing. */
void wgpu_write_buffer(WGPUBuffer dst, size_t offset, const void *src, size_t size);

/* Copy between GPU buffers. In batch mode, appends to batch. */
void wgpu_copy_buffer(WGPUBuffer src, WGPUBuffer dst, size_t size);

/* Compute ceil(n / d) as uint32_t */
static inline uint32_t wgpu_div_ceil(uint32_t n, uint32_t d) {
    return (n + d - 1) / d;
}

/* Create a uniform buffer initialized with data (convenience for params structs) */
static inline WGPUBuffer wgpu_make_params(const void *data, size_t size) {
    return wgpu_create_buffer_init(data, size,
        WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "params");
}

#ifdef __cplusplus
}
#endif

#endif /* CFIREANTS_WEBGPU_CONTEXT_H */
