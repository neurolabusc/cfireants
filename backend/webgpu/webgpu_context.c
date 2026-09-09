/*
 * webgpu_context.c - WebGPU device initialization and helpers
 *
 * Headless compute setup using wgpu-native.
 * Asynchronous requests are completed by pumping the instance event queue.
 *
 * Compatible with wgpu-native v27.0.4.0+ (webgpu.h StringView API).
 */

#include "webgpu_context.h"
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Global context */
wgpu_context_t g_wgpu = {0};

/* Helper: convert C string to WGPUStringView */
#define WGPU_STR(s) ((WGPUStringView){ .data = (s), .length = WGPU_STRLEN })

/* --- Callbacks (v27 API: WGPUStringView message, dual userdata) --- */

#ifndef __EMSCRIPTEN__
static void on_adapter_request(WGPURequestAdapterStatus status,
                               WGPUAdapter adapter,
                               WGPUStringView message,
                               void *userdata1,
                               void *userdata2)
{
    if (status == WGPURequestAdapterStatus_Success) {
        *(WGPUAdapter *)userdata1 = adapter;
    } else {
        fprintf(stderr, "wgpu: adapter request failed (status=%d): %.*s\n",
                status, (int)message.length, message.data ? message.data : "");
    }
    if (userdata2) *(int *)userdata2 = 1;
}
#endif

#ifndef __EMSCRIPTEN__
static void on_device_request(WGPURequestDeviceStatus status,
                              WGPUDevice device,
                              WGPUStringView message,
                              void *userdata1,
                              void *userdata2)
{
    if (status == WGPURequestDeviceStatus_Success) {
        *(WGPUDevice *)userdata1 = device;
    } else {
        fprintf(stderr, "wgpu: device request failed (status=%d): %.*s\n",
                status, (int)message.length, message.data ? message.data : "");
    }
    if (userdata2) *(int *)userdata2 = 1;
}
#endif

/* Sticky: a readback that cannot deliver data leaves zeros behind, which are
 * indistinguishable from a valid result. Rather than thread a status through 23
 * call sites, record it here and have each registration entry point check once
 * before returning. */
static int g_wgpu_fatal = 0;
int wgpu_had_fatal_error(void) { return g_wgpu_fatal; }
void wgpu_clear_fatal_error(void) { g_wgpu_fatal = 0; }
void wgpu_record_fatal_error(const char *operation) {
    if (!g_wgpu_fatal)
        fprintf(stderr, "wgpu: %s failed\n", operation ? operation : "GPU operation");
    g_wgpu_fatal = 1;
}

#ifndef __EMSCRIPTEN__
static void on_device_error(WGPUDevice const *device,
                            WGPUErrorType type,
                            WGPUStringView message,
                            void *userdata1,
                            void *userdata2)
{
    (void)device;
    (void)userdata1;
    (void)userdata2;
    fprintf(stderr, "wgpu device error (type=%d): %.*s\n",
            type, (int)message.length, message.data ? message.data : "");
    /* A validation error means the dispatch never ran; its outputs are garbage. */
    g_wgpu_fatal = 1;
}
#endif

static void on_buffer_map(WGPUMapAsyncStatus status,
                          WGPUStringView message,
                          void *userdata1,
                          void *userdata2)
{
    (void)userdata2;
    if (status != WGPUMapAsyncStatus_Success) {
        fprintf(stderr, "wgpu: buffer map failed (status=%d): %.*s\n",
                status, (int)message.length,
                message.data ? message.data : "");
    }
    if (userdata1)
        *(int *)userdata1 = (status == WGPUMapAsyncStatus_Success) ? 0 : -1;
}

/* --- Context lifecycle --- */

int wgpu_context_init(void) {
    wgpu_clear_fatal_error();
#ifdef __EMSCRIPTEN__
    /*
     * In a browser the adapter and device are obtained through promises, which a
     * synchronous C entry point cannot await. JS creates the device and hands it
     * over via Module.preinitializedWebGPUDevice before main() runs, exactly as
     * @brainchop/mindgrab does, which keeps this whole backend synchronous and
     * limits ASYNCIFY to the poll/readback paths.
     */
    g_wgpu.instance = NULL;
    g_wgpu.adapter = NULL;
    g_wgpu.device = emscripten_webgpu_get_device();
    if (!g_wgpu.device) {
        fprintf(stderr, "wgpu: no device supplied; set Module.preinitializedWebGPUDevice\n");
        return -1;
    }
    /* The device comes from JS, so its uncaptured-error callback is installed
     * there; WebGPU errors are asynchronous and a failed dispatch otherwise
     * returns a blank volume while reporting success. */
#else

    /* Create instance */
    WGPUInstanceExtras extras = {
        .chain = { .sType = (WGPUSType)WGPUSType_InstanceExtras },
        .backends = WGPUInstanceBackend_Metal | WGPUInstanceBackend_Vulkan,
    };
    WGPUInstanceDescriptor inst_desc = {
        .nextInChain = (const WGPUChainedStruct *)&extras,
    };
    g_wgpu.instance = wgpuCreateInstance(&inst_desc);
    if (!g_wgpu.instance) {
        fprintf(stderr, "wgpu: failed to create instance\n");
        return -1;
    }

    /* Request adapter (headless, high performance) */
    WGPURequestAdapterOptions adapter_opts = {
        .powerPreference = WGPUPowerPreference_HighPerformance,
    };
    int adapter_done = 0;
    WGPURequestAdapterCallbackInfo adapter_cb = {
        .mode = WGPUCallbackMode_AllowProcessEvents,
        .callback = on_adapter_request,
        .userdata1 = &g_wgpu.adapter,
        .userdata2 = &adapter_done,
    };
    (void)wgpuInstanceRequestAdapter(g_wgpu.instance, &adapter_opts, adapter_cb);
    while (!adapter_done) wgpuInstanceProcessEvents(g_wgpu.instance);
    if (!g_wgpu.adapter) {
        fprintf(stderr, "wgpu: no adapter found\n");
        wgpu_context_cleanup();
        return -1;
    }

    /* Print adapter info */
    WGPUAdapterInfo info = {0};
    wgpuAdapterGetInfo(g_wgpu.adapter, &info);
    fprintf(stderr, "WebGPU backend: %.*s (%.*s)\n",
            (int)info.device.length, info.device.data ? info.device.data : "unknown",
            (int)info.description.length, info.description.data ? info.description.data : "");
    wgpuAdapterInfoFreeMembers(info);

    /* Request device — use adapter limits as base, then override what we need */
    WGPULimits supported_limits = {0};
    if (wgpuAdapterGetLimits(g_wgpu.adapter, &supported_limits) !=
        WGPUStatus_Success) {
        fprintf(stderr, "wgpu: failed to query adapter limits\n");
        wgpu_context_cleanup();
        return -1;
    }

    /* Request what this adapter actually supports. Individual allocations then
     * fail cleanly if a job exceeds it; inventing 512 MiB requirements made a
     * smaller but otherwise usable adapter reject device creation outright. */
    WGPULimits required = supported_limits;

    /* Request float32-filterable if supported (optional, for texture trilinear) */
    WGPUFeatureName features[] = { WGPUFeatureName_Float32Filterable };
    int n_features = 0;
    if (wgpuAdapterHasFeature(g_wgpu.adapter, WGPUFeatureName_Float32Filterable))
        n_features = 1;

    WGPUDeviceDescriptor device_desc = {
        .requiredFeatureCount = n_features,
        .requiredFeatures = n_features ? features : NULL,
        .requiredLimits = &required,
        .uncapturedErrorCallbackInfo = (WGPUUncapturedErrorCallbackInfo){
            .callback = on_device_error,
            .userdata1 = NULL,
            .userdata2 = NULL,
        },
    };
    int device_done = 0;
    WGPURequestDeviceCallbackInfo device_cb = {
        .mode = WGPUCallbackMode_AllowProcessEvents,
        .callback = on_device_request,
        .userdata1 = &g_wgpu.device,
        .userdata2 = &device_done,
    };
    (void)wgpuAdapterRequestDevice(g_wgpu.adapter, &device_desc, device_cb);
    while (!device_done) wgpuInstanceProcessEvents(g_wgpu.instance);
    if (!g_wgpu.device) {
        fprintf(stderr, "wgpu: failed to create device\n");
        wgpu_context_cleanup();
        return -1;
    }

#endif
    g_wgpu.queue = wgpuDeviceGetQueue(g_wgpu.device);
    if (!g_wgpu.queue) {
        wgpu_record_fatal_error("device queue creation");
        wgpu_context_cleanup();
        return -1;
    }
    g_wgpu.n_pipelines = 0;
    g_wgpu.n_buffers = 0;
    g_wgpu.staging_buf = NULL;
    g_wgpu.staging_size = 0;
    g_wgpu.mi_state_buf = NULL;
    g_wgpu.mi_coeff_buf = NULL;

    return 0;
}

void wgpu_context_cleanup(void) {
    /* Release an encoder retained by batching before dropping its device. */
    if (g_wgpu.batch_active) wgpu_flush();
    for (int i = 0; i < g_wgpu.n_pipelines; i++) {
        if (g_wgpu.pipelines[i].pipeline)
            wgpuComputePipelineRelease(g_wgpu.pipelines[i].pipeline);
        if (g_wgpu.pipelines[i].layout)
            wgpuBindGroupLayoutRelease(g_wgpu.pipelines[i].layout);
    }
    /* A failure can jump out of a registration stage before its local buffer
     * handles are released. Every helper-created buffer is tracked here, so
     * context teardown remains a complete failure cleanup. */
    for (int i = 0; i < g_wgpu.n_buffers; i++)
        if (g_wgpu.buffers[i]) wgpuBufferRelease(g_wgpu.buffers[i]);
    if (g_wgpu.queue) wgpuQueueRelease(g_wgpu.queue);
    if (g_wgpu.device) wgpuDeviceRelease(g_wgpu.device);
    if (g_wgpu.adapter) wgpuAdapterRelease(g_wgpu.adapter);
    if (g_wgpu.instance) wgpuInstanceRelease(g_wgpu.instance);
    memset(&g_wgpu, 0, sizeof(g_wgpu));
}

/* --- Shader helpers --- */

WGPUShaderModule wgpu_create_shader(const char *wgsl_source, const char *label) {
    if (!g_wgpu.device || !wgsl_source) {
        wgpu_record_fatal_error("missing WebGPU device or shader source");
        return NULL;
    }
    WGPUShaderSourceWGSL wgsl = {
        .chain = { .sType = WGPUSType_ShaderSourceWGSL },
        .code = { .data = wgsl_source, .length = WGPU_STRLEN },
    };
    WGPUShaderModuleDescriptor desc = {
        .nextInChain = (WGPUChainedStruct *)&wgsl,
        .label = WGPU_STR(label),
    };
    return wgpuDeviceCreateShaderModule(g_wgpu.device, &desc);
}

WGPUComputePipeline wgpu_get_pipeline(const char *name,
                                       const char *wgsl_source,
                                       const char *entry_point) {
    if (g_wgpu_fatal) return NULL;
    if (!name || !wgsl_source || !entry_point) {
        wgpu_record_fatal_error("missing WebGPU pipeline metadata");
        return NULL;
    }
    /* Check cache */
    for (int i = 0; i < g_wgpu.n_pipelines; i++) {
        if (strcmp(g_wgpu.pipelines[i].name, name) == 0)
            return g_wgpu.pipelines[i].pipeline;
    }

    /* Compile */
    WGPUShaderModule shader = wgpu_create_shader(wgsl_source, name);
    if (!shader) {
        fprintf(stderr, "wgpu: failed to compile shader '%s'\n", name);
        g_wgpu_fatal = 1;
        return NULL;
    }

    WGPUComputePipelineDescriptor desc = {
        .label = WGPU_STR(name),
        .compute = {
            .module = shader,
            .entryPoint = WGPU_STR(entry_point),
        },
    };
    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(g_wgpu.device, &desc);
    wgpuShaderModuleRelease(shader);

    if (!pipeline) {
        fprintf(stderr, "wgpu: failed to create pipeline '%s'\n", name);
        g_wgpu_fatal = 1;
        return NULL;
    }

    /* Cache */
    if (g_wgpu.n_pipelines >= WGPU_MAX_PIPELINES) {
        fprintf(stderr, "wgpu: pipeline cache is full (%d)\n", WGPU_MAX_PIPELINES);
        g_wgpu_fatal = 1;
        wgpuComputePipelineRelease(pipeline);
        return NULL;
    }
    int idx = g_wgpu.n_pipelines++;
    g_wgpu.pipelines[idx].pipeline = pipeline;
    g_wgpu.pipelines[idx].layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    g_wgpu.pipelines[idx].name = name;  /* must be a string literal or static */
    if (!g_wgpu.pipelines[idx].layout) {
        fprintf(stderr, "wgpu: failed to get bind-group layout for '%s'\n", name);
        g_wgpu_fatal = 1;
        g_wgpu.n_pipelines--;
        wgpuComputePipelineRelease(pipeline);
        return NULL;
    }

    return pipeline;
}

WGPUBindGroupLayout wgpu_get_bind_group_layout(const char *name) {
    for (int i = 0; i < g_wgpu.n_pipelines; i++) {
        if (strcmp(g_wgpu.pipelines[i].name, name) == 0)
            return g_wgpu.pipelines[i].layout;
    }
    fprintf(stderr, "wgpu: pipeline '%s' has no cached bind-group layout\n", name);
    g_wgpu_fatal = 1;
    return NULL;
}

/* --- Buffer helpers --- */

static int track_buffer(WGPUBuffer buffer) {
    if (!buffer) return -1;
    if (g_wgpu.n_buffers >= WGPU_MAX_BUFFERS) {
        wgpu_record_fatal_error("WebGPU live-buffer table capacity");
        wgpuBufferRelease(buffer);
        return -1;
    }
    g_wgpu.buffers[g_wgpu.n_buffers++] = buffer;
    return 0;
}

static int untrack_buffer(WGPUBuffer buffer) {
    for (int i = 0; i < g_wgpu.n_buffers; i++) {
        if (g_wgpu.buffers[i] == buffer) {
            g_wgpu.n_buffers--;
            g_wgpu.buffers[i] = g_wgpu.buffers[g_wgpu.n_buffers];
            g_wgpu.buffers[g_wgpu.n_buffers] = NULL;
            return 1;
        }
    }
    return 0;
}

WGPUBuffer wgpu_create_buffer(size_t size, WGPUBufferUsage usage, const char *label) {
    if (g_wgpu_fatal) return NULL;
    if (!g_wgpu.device || size == 0 || size > SIZE_MAX - 3) {
        wgpu_record_fatal_error("invalid WebGPU buffer allocation");
        return NULL;
    }
    /* WebGPU requires buffer sizes to be multiples of 4 */
    size = (size + 3) & ~(size_t)3;
    WGPUBufferDescriptor desc = {
        .label = WGPU_STR(label),
        .usage = usage,
        .size = size,
        .mappedAtCreation = 0,
    };
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(g_wgpu.device, &desc);
    if (!buffer) {
        fprintf(stderr, "wgpu: could not allocate buffer '%s' (%zu bytes)\n",
                label ? label : "unnamed", size);
        g_wgpu_fatal = 1;
    } else if (track_buffer(buffer) != 0) {
        return NULL;
    }
    return buffer;
}

WGPUBuffer wgpu_create_buffer_init(const void *data, size_t size,
                                    WGPUBufferUsage usage, const char *label) {
    if (g_wgpu_fatal) return NULL;
    if (!g_wgpu.device || !data || size == 0 || size > SIZE_MAX - 3) {
        wgpu_record_fatal_error("invalid initialized WebGPU buffer allocation");
        return NULL;
    }
    size_t aligned = (size + 3) & ~(size_t)3;
    WGPUBufferDescriptor desc = {
        .label = WGPU_STR(label),
        .usage = usage,
        .size = aligned,
        .mappedAtCreation = 1,
    };
    WGPUBuffer buf = wgpuDeviceCreateBuffer(g_wgpu.device, &desc);
    if (!buf) {
        fprintf(stderr, "wgpu: could not allocate buffer '%s' (%zu bytes)\n",
                label ? label : "unnamed", aligned);
        g_wgpu_fatal = 1;
        return NULL;
    }
    void *mapped = wgpuBufferGetMappedRange(buf, 0, aligned);
    if (!mapped) {
        fprintf(stderr, "wgpu: initial mapping for buffer '%s' failed\n",
                label ? label : "unnamed");
        g_wgpu_fatal = 1;
        wgpuBufferRelease(buf);
        return NULL;
    }
    memcpy(mapped, data, size);
    if (aligned > size) memset((char*)mapped + size, 0, aligned - size);
    wgpuBufferUnmap(buf);
    if (track_buffer(buf) != 0) return NULL;
    return buf;
}

WGPUBindGroup wgpu_create_bind_group(const WGPUBindGroupDescriptor *desc,
                                     const char *label) {
    if (g_wgpu_fatal) return NULL;
    if (!g_wgpu.device || !desc || !desc->layout) {
        wgpu_record_fatal_error("invalid WebGPU bind-group descriptor");
        return NULL;
    }
    WGPUBindGroup group = wgpuDeviceCreateBindGroup(g_wgpu.device, desc);
    if (!group) {
        fprintf(stderr, "wgpu: could not create bind group '%s'\n",
                label ? label : "unnamed");
        g_wgpu_fatal = 1;
    }
    return group;
}

void wgpu_release_bind_group(WGPUBindGroup group) {
    if (group) wgpuBindGroupRelease(group);
}

void wgpu_release_buffer(WGPUBuffer buffer) {
    if (buffer && untrack_buffer(buffer)) {
        wgpuBufferRelease(buffer);
    }
}

void wgpu_ensure_staging(size_t size) {
    size = (size + 3) & ~(size_t)3;
    if (g_wgpu.staging_buf && g_wgpu.staging_size >= size) return;
    if (g_wgpu.staging_buf) wgpu_release_buffer(g_wgpu.staging_buf);
    g_wgpu.staging_buf = wgpu_create_buffer(size,
        WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, "staging");
    /* Only record the size if the buffer actually exists. Recording it
     * unconditionally wedged the backend permanently: the early-return above
     * then compared against memory that was never allocated, and every later
     * readback silently returned zeros. */
    g_wgpu.staging_size = g_wgpu.staging_buf ? size : 0;
}

static WGPUCommandEncoder create_command_encoder(const char *operation) {
    if (g_wgpu_fatal || !g_wgpu.device) return NULL;
    WGPUCommandEncoder encoder =
        wgpuDeviceCreateCommandEncoder(g_wgpu.device, NULL);
    if (!encoder) wgpu_record_fatal_error(operation);
    return encoder;
}

static WGPUComputePassEncoder begin_compute_pass(
    WGPUCommandEncoder encoder, const char *operation)
{
    if (g_wgpu_fatal || !encoder) return NULL;
    WGPUComputePassEncoder pass =
        wgpuCommandEncoderBeginComputePass(encoder, NULL);
    if (!pass) wgpu_record_fatal_error(operation);
    return pass;
}

static WGPUCommandBuffer finish_command_encoder(
    WGPUCommandEncoder encoder, const char *operation)
{
    if (g_wgpu_fatal || !encoder) return NULL;
    WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, NULL);
    if (!command) wgpu_record_fatal_error(operation);
    return command;
}

/* --- Batch mode dispatch --- */

void wgpu_begin_batch(void) {
    if (g_wgpu_fatal) return;
#ifdef CFIREANTS_WGPU_NO_BATCH
    /* Batching packs many dispatches into one compute pass. Dispatches that
     * share a buffer then rely on the implementation inserting a barrier
     * between them, which this codebase already documents as not guaranteed
     * (the greedy compositive update was broken by it natively). Diagnostic
     * switch: submit every dispatch separately, which is always ordered. */
    return;
#endif
    if (g_wgpu.batch_active) return;  /* already in batch */
    g_wgpu.batch_encoder = create_command_encoder(
        "batch command encoder creation");
    if (!g_wgpu.batch_encoder) return;
    g_wgpu.batch_pass = begin_compute_pass(
        g_wgpu.batch_encoder, "batch compute-pass creation");
    if (!g_wgpu.batch_pass) {
        wgpuCommandEncoderRelease(g_wgpu.batch_encoder);
        g_wgpu.batch_encoder = NULL;
        return;
    }
    g_wgpu.batch_active = 1;
    g_wgpu.batch_dispatches = 0;
}

void wgpu_flush(void) {
    if (g_wgpu_fatal && !g_wgpu.batch_active) return;
    if (!g_wgpu.batch_active) return;

    wgpuComputePassEncoderEnd(g_wgpu.batch_pass);
    wgpuComputePassEncoderRelease(g_wgpu.batch_pass);
    g_wgpu.batch_pass = NULL;

    WGPUCommandBuffer cmdbuf =
        finish_command_encoder(g_wgpu.batch_encoder,
                               "batch command-buffer creation");
    if (!cmdbuf) {
        wgpuCommandEncoderRelease(g_wgpu.batch_encoder);
        g_wgpu.batch_encoder = NULL;
        g_wgpu.batch_active = 0;
        g_wgpu.batch_dispatches = 0;
        return;
    }
    wgpuQueueSubmit(g_wgpu.queue, 1, &cmdbuf);
    wgpuCommandBufferRelease(cmdbuf);
    wgpuCommandEncoderRelease(g_wgpu.batch_encoder);
    g_wgpu.batch_encoder = NULL;

    wgpuDevicePoll(g_wgpu.device, 1, NULL);
    g_wgpu.batch_active = 0;
    g_wgpu.batch_dispatches = 0;
}

void wgpu_dispatch(WGPUComputePipeline pipeline,
                   WGPUBindGroup bind_group,
                   uint32_t wx, uint32_t wy, uint32_t wz) {
    if (g_wgpu_fatal || !pipeline || !bind_group) {
        if (!g_wgpu_fatal) wgpu_record_fatal_error("compute dispatch setup");
        return;
    }
    /* Auto-split oversized dimensions to stay within 65535 limit */
    if (wx > 65535 && wy == 1) {
        wy = (wx + 65534) / 65535;
        wx = (wx + wy - 1) / wy;
    }
    if (g_wgpu.batch_active) {
        /* Keep compute work in one pass until a transfer requires a pass
         * boundary. Dispatches are ordered, and avoiding hundreds of retained
         * pass encoders cuts both CPU overhead and peak unified memory. */
        wgpuComputePassEncoderSetPipeline(g_wgpu.batch_pass, pipeline);
        wgpuComputePassEncoderSetBindGroup(g_wgpu.batch_pass, 0, bind_group, 0, NULL);
        wgpuComputePassEncoderDispatchWorkgroups(g_wgpu.batch_pass, wx, wy, wz);
        g_wgpu.batch_dispatches++;
        return;
    }

    /* Non-batch mode: immediate submit + poll (legacy) */
    WGPUCommandEncoder encoder = create_command_encoder(
        "dispatch command encoder creation");
    if (!encoder) return;
    WGPUComputePassEncoder pass = begin_compute_pass(
        encoder, "dispatch compute-pass creation");
    if (!pass) {
        wgpuCommandEncoderRelease(encoder);
        return;
    }
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, wx, wy, wz);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    WGPUCommandBuffer cmdbuf = finish_command_encoder(
        encoder, "dispatch command-buffer creation");
    if (!cmdbuf) {
        wgpuCommandEncoderRelease(encoder);
        return;
    }
    wgpuQueueSubmit(g_wgpu.queue, 1, &cmdbuf);
    wgpuCommandBufferRelease(cmdbuf);
    wgpuCommandEncoderRelease(encoder);

    wgpuDevicePoll(g_wgpu.device, 1, NULL);
}

void wgpu_copy_buffer_range(WGPUBuffer src, size_t src_offset,
                            WGPUBuffer dst, size_t dst_offset, size_t size) {
    if (g_wgpu_fatal || !src || !dst) {
        if (!g_wgpu_fatal) wgpu_record_fatal_error("buffer copy setup");
        return;
    }
    if (g_wgpu.batch_active) {
        /* Must end compute pass for buffer copy, then restart */
        wgpuComputePassEncoderEnd(g_wgpu.batch_pass);
        wgpuComputePassEncoderRelease(g_wgpu.batch_pass);
        wgpuCommandEncoderCopyBufferToBuffer(g_wgpu.batch_encoder,
                                             src, src_offset,
                                             dst, dst_offset, size);
        g_wgpu.batch_pass = begin_compute_pass(
            g_wgpu.batch_encoder, "copy batch compute-pass creation");
        if (!g_wgpu.batch_pass) {
            wgpuCommandEncoderRelease(g_wgpu.batch_encoder);
            g_wgpu.batch_encoder = NULL;
            g_wgpu.batch_active = 0;
            g_wgpu.batch_dispatches = 0;
            return;
        }
        g_wgpu.batch_dispatches = 0;  /* reset since we started a new pass */
        return;
    }

    WGPUCommandEncoder enc = create_command_encoder(
        "copy command encoder creation");
    if (!enc) return;
    wgpuCommandEncoderCopyBufferToBuffer(enc, src, src_offset,
                                         dst, dst_offset, size);
    WGPUCommandBuffer cmd = finish_command_encoder(
        enc, "copy command-buffer creation");
    if (!cmd) {
        wgpuCommandEncoderRelease(enc);
        return;
    }
    wgpuQueueSubmit(g_wgpu.queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    wgpuDevicePoll(g_wgpu.device, 1, NULL);
}

void wgpu_copy_buffer(WGPUBuffer src, WGPUBuffer dst, size_t size) {
    wgpu_copy_buffer_range(src, 0, dst, 0, size);
}

void wgpu_read_buffer(WGPUBuffer src, size_t offset, void *dst, size_t size) {
    if (!dst || size == 0 || size > SIZE_MAX - 3) {
        wgpu_record_fatal_error("invalid buffer read");
        return;
    }
    size_t aligned = (size + 3) & ~(size_t)3;
    /* A failed map must never expose uninitialised host memory as a GPU result. */
    memset(dst, 0, size);
    if (g_wgpu_fatal || !src) {
        if (!g_wgpu_fatal) wgpu_record_fatal_error("buffer read setup");
        return;
    }
    wgpu_ensure_staging(aligned);
    if (!g_wgpu.staging_buf) { g_wgpu_fatal = 1; return; }

    /* Append the staging copy to pending compute work. This preserves ordering
     * while avoiding the separate submit+wait previously done by wgpu_flush. */
    WGPUCommandEncoder encoder;
#ifdef __EMSCRIPTEN__
    /* Submit pending browser work before starting the staging copy. Keeping
     * this transfer on a fresh encoder gives it an explicit queue-ordering
     * boundary and avoids retaining a compute pass across an async map. */
    wgpu_flush();
#endif
    if (g_wgpu.batch_active) {
        wgpuComputePassEncoderEnd(g_wgpu.batch_pass);
        wgpuComputePassEncoderRelease(g_wgpu.batch_pass);
        g_wgpu.batch_pass = NULL;
        encoder = g_wgpu.batch_encoder;
        g_wgpu.batch_encoder = NULL;
        g_wgpu.batch_active = 0;
        g_wgpu.batch_dispatches = 0;
    } else {
        encoder = create_command_encoder("read command encoder creation");
    }
    if (!encoder) return;
    wgpuCommandEncoderCopyBufferToBuffer(encoder, src, offset,
                                         g_wgpu.staging_buf, 0, aligned);
    WGPUCommandBuffer cmdbuf = finish_command_encoder(
        encoder, "read command-buffer creation");
    if (!cmdbuf) {
        wgpuCommandEncoderRelease(encoder);
        return;
    }
    wgpuQueueSubmit(g_wgpu.queue, 1, &cmdbuf);
    wgpuCommandBufferRelease(cmdbuf);
    wgpuCommandEncoderRelease(encoder);

    /* Map staging for read */
    /* 1 = pending, 0 = success, -1 = failure.  Pending and failure must be
     * distinct or a failed map spins until the timeout and is then unmapped as
     * though it were merely late. */
    int map_status = 1;
    WGPUBufferMapCallbackInfo map_cb = {
#ifdef __EMSCRIPTEN__
        /* WaitAnyOnly callbacks fire only inside wgpuInstanceWaitAny, and the
         * browser path has no instance to wait on. Spontaneous delivery lets the
         * callback arrive on the event loop, which is what emscripten_sleep turns. */
        .mode = WGPUCallbackMode_AllowSpontaneous,
#else
        .mode = WGPUCallbackMode_AllowProcessEvents,
#endif
        .callback = on_buffer_map,
        .userdata1 = &map_status,
        .userdata2 = NULL,
    };
    (void)wgpuBufferMapAsync(
        g_wgpu.staging_buf, WGPUMapMode_Read, 0, aligned, map_cb);
#ifdef __EMSCRIPTEN__
    /* Mapping genuinely takes several event-loop turns here, where the native
     * poll blocks until it is done. Yielding once left map_status at -1: the
     * copy was skipped, leaving the destination uninitialised, and the unmap of
     * a still-pending map poisoned the staging buffer for the next read. */
    for (int spins = 0; map_status == 1 && spins < 100000; spins++)
        emscripten_sleep(1);
    if (map_status == 1)
        fprintf(stderr, "wgpu: buffer map timed out\n");
#else
    while (map_status == 1) wgpuInstanceProcessEvents(g_wgpu.instance);
#endif

    if (map_status != 0) g_wgpu_fatal = 1;
    else {
        /* A MAP_READ buffer is read-only.  emdawnwebgpu correctly rejects the
         * mutable accessor (while wgpu-native historically accepted it), which
         * used to leave dst untouched and made browser losses look like random
         * 1e22 host-heap values. */
        const void *mapped = wgpuBufferGetConstMappedRange(
            g_wgpu.staging_buf, 0, aligned);
        if (mapped) {
            memcpy(dst, mapped, size);
        } else {
            fprintf(stderr, "wgpu: mapped read range is unavailable\n");
            g_wgpu_fatal = 1;
        }
    }
    wgpuBufferUnmap(g_wgpu.staging_buf);
}

void wgpu_write_buffer(WGPUBuffer dst, size_t offset, const void *src, size_t size) {
    if (g_wgpu_fatal || !dst || !src || size == 0) {
        if (!g_wgpu_fatal) wgpu_record_fatal_error("buffer write setup");
        return;
    }
    /* Auto-flush before write to ensure ordering */
    wgpu_flush();
    wgpuQueueWriteBuffer(g_wgpu.queue, dst, offset, src, size);
}
