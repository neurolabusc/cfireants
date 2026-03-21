/*
 * webgpu_context.c - WebGPU device initialization and helpers
 *
 * Headless compute setup using wgpu-native.
 * All operations are synchronous via wgpuDevicePoll(device, 1).
 *
 * Compatible with wgpu-native v27.0.4.0 (webgpu.h pre-StringView API).
 */

#include "webgpu_context.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Global context */
wgpu_context_t g_wgpu = {0};

/* --- Callbacks (v27 API: const char* message, single void* userdata) --- */

static void on_adapter_request(WGPURequestAdapterStatus status,
                               WGPUAdapter adapter,
                               char const *message,
                               void *userdata)
{
    (void)message;
    if (status == WGPURequestAdapterStatus_Success) {
        *(WGPUAdapter *)userdata = adapter;
    } else {
        fprintf(stderr, "wgpu: adapter request failed (status=%d): %s\n",
                status, message ? message : "");
    }
}

static void on_device_request(WGPURequestDeviceStatus status,
                              WGPUDevice device,
                              char const *message,
                              void *userdata)
{
    (void)message;
    if (status == WGPURequestDeviceStatus_Success) {
        *(WGPUDevice *)userdata = device;
    } else {
        fprintf(stderr, "wgpu: device request failed (status=%d): %s\n",
                status, message ? message : "");
    }
}

static void on_device_error(WGPUErrorType type, char const *message,
                            void *userdata)
{
    (void)userdata;
    fprintf(stderr, "wgpu device error (type=%d): %s\n",
            type, message ? message : "");
}

static void on_buffer_map(WGPUBufferMapAsyncStatus status, void *userdata)
{
    if (userdata)
        *(int *)userdata = (status == WGPUBufferMapAsyncStatus_Success) ? 0 : -1;
}

/* --- Context lifecycle --- */

int wgpu_context_init(void) {
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
    wgpuInstanceRequestAdapter(g_wgpu.instance, &adapter_opts,
                               on_adapter_request, &g_wgpu.adapter);
    if (!g_wgpu.adapter) {
        fprintf(stderr, "wgpu: no adapter found\n");
        return -1;
    }

    /* Print adapter info */
    WGPUAdapterInfo info = {0};
    wgpuAdapterGetInfo(g_wgpu.adapter, &info);
    fprintf(stderr, "WebGPU backend: %s (%s)\n",
            info.device ? info.device : "unknown",
            info.description ? info.description : "");
    wgpuAdapterInfoFreeMembers(info);

    /* Request device — use adapter limits as base, then override what we need */
    WGPUSupportedLimits supported_limits = {0};
    wgpuAdapterGetLimits(g_wgpu.adapter, &supported_limits);

    WGPULimits required = supported_limits.limits;
    /* Ensure minimums for our compute workloads */
    if (required.maxStorageBufferBindingSize < 512ULL * 1024 * 1024)
        required.maxStorageBufferBindingSize = 512ULL * 1024 * 1024;
    if (required.maxBufferSize < 512ULL * 1024 * 1024)
        required.maxBufferSize = 512ULL * 1024 * 1024;

    WGPURequiredLimits req_limits = { .limits = required };
    WGPUDeviceDescriptor device_desc = {
        .requiredLimits = &req_limits,
        .uncapturedErrorCallbackInfo = (WGPUUncapturedErrorCallbackInfo){
            .callback = on_device_error,
        },
    };
    wgpuAdapterRequestDevice(g_wgpu.adapter, &device_desc,
                             on_device_request, &g_wgpu.device);
    if (!g_wgpu.device) {
        fprintf(stderr, "wgpu: failed to create device\n");
        return -1;
    }

    g_wgpu.queue = wgpuDeviceGetQueue(g_wgpu.device);
    g_wgpu.n_pipelines = 0;
    g_wgpu.staging_buf = NULL;
    g_wgpu.staging_size = 0;

    return 0;
}

void wgpu_context_cleanup(void) {
    for (int i = 0; i < g_wgpu.n_pipelines; i++) {
        if (g_wgpu.pipelines[i].pipeline)
            wgpuComputePipelineRelease(g_wgpu.pipelines[i].pipeline);
        if (g_wgpu.pipelines[i].layout)
            wgpuBindGroupLayoutRelease(g_wgpu.pipelines[i].layout);
    }
    if (g_wgpu.staging_buf) wgpuBufferRelease(g_wgpu.staging_buf);
    if (g_wgpu.queue) wgpuQueueRelease(g_wgpu.queue);
    if (g_wgpu.device) wgpuDeviceRelease(g_wgpu.device);
    if (g_wgpu.adapter) wgpuAdapterRelease(g_wgpu.adapter);
    if (g_wgpu.instance) wgpuInstanceRelease(g_wgpu.instance);
    memset(&g_wgpu, 0, sizeof(g_wgpu));
}

/* --- Shader helpers --- */

WGPUShaderModule wgpu_create_shader(const char *wgsl_source, const char *label) {
    WGPUShaderModuleWGSLDescriptor wgsl = {
        .chain = { .sType = WGPUSType_ShaderModuleWGSLDescriptor },
        .code = wgsl_source,
    };
    WGPUShaderModuleDescriptor desc = {
        .nextInChain = (const WGPUChainedStruct *)&wgsl,
        .label = label,
    };
    return wgpuDeviceCreateShaderModule(g_wgpu.device, &desc);
}

WGPUComputePipeline wgpu_get_pipeline(const char *name,
                                       const char *wgsl_source,
                                       const char *entry_point) {
    /* Check cache */
    for (int i = 0; i < g_wgpu.n_pipelines; i++) {
        if (strcmp(g_wgpu.pipelines[i].name, name) == 0)
            return g_wgpu.pipelines[i].pipeline;
    }

    /* Compile */
    WGPUShaderModule shader = wgpu_create_shader(wgsl_source, name);
    if (!shader) {
        fprintf(stderr, "wgpu: failed to compile shader '%s'\n", name);
        return NULL;
    }

    WGPUComputePipelineDescriptor desc = {
        .label = name,
        .compute = {
            .module = shader,
            .entryPoint = entry_point,
        },
    };
    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(g_wgpu.device, &desc);
    wgpuShaderModuleRelease(shader);

    if (!pipeline) {
        fprintf(stderr, "wgpu: failed to create pipeline '%s'\n", name);
        return NULL;
    }

    /* Cache */
    if (g_wgpu.n_pipelines < WGPU_MAX_PIPELINES) {
        int idx = g_wgpu.n_pipelines++;
        g_wgpu.pipelines[idx].pipeline = pipeline;
        g_wgpu.pipelines[idx].layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        g_wgpu.pipelines[idx].name = name;  /* must be a string literal or static */
    }

    return pipeline;
}

WGPUBindGroupLayout wgpu_get_bind_group_layout(const char *name) {
    for (int i = 0; i < g_wgpu.n_pipelines; i++) {
        if (strcmp(g_wgpu.pipelines[i].name, name) == 0)
            return g_wgpu.pipelines[i].layout;
    }
    return NULL;
}

/* --- Buffer helpers --- */

WGPUBuffer wgpu_create_buffer(size_t size, WGPUBufferUsage usage, const char *label) {
    /* WebGPU requires buffer sizes to be multiples of 4 */
    size = (size + 3) & ~(size_t)3;
    WGPUBufferDescriptor desc = {
        .label = label,
        .usage = usage,
        .size = size,
        .mappedAtCreation = 0,
    };
    return wgpuDeviceCreateBuffer(g_wgpu.device, &desc);
}

WGPUBuffer wgpu_create_buffer_init(const void *data, size_t size,
                                    WGPUBufferUsage usage, const char *label) {
    size_t aligned = (size + 3) & ~(size_t)3;
    WGPUBufferDescriptor desc = {
        .label = label,
        .usage = usage,
        .size = aligned,
        .mappedAtCreation = 1,
    };
    WGPUBuffer buf = wgpuDeviceCreateBuffer(g_wgpu.device, &desc);
    if (buf) {
        void *mapped = wgpuBufferGetMappedRange(buf, 0, aligned);
        if (mapped) {
            memcpy(mapped, data, size);
            if (aligned > size) memset((char*)mapped + size, 0, aligned - size);
        }
        wgpuBufferUnmap(buf);
    }
    return buf;
}

void wgpu_ensure_staging(size_t size) {
    size = (size + 3) & ~(size_t)3;
    if (g_wgpu.staging_buf && g_wgpu.staging_size >= size) return;
    if (g_wgpu.staging_buf) wgpuBufferRelease(g_wgpu.staging_buf);
    g_wgpu.staging_buf = wgpu_create_buffer(size,
        WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, "staging");
    g_wgpu.staging_size = size;
}

/* --- Batch mode dispatch --- */

void wgpu_begin_batch(void) {
    if (g_wgpu.batch_active) return;  /* already in batch */
    g_wgpu.batch_encoder = wgpuDeviceCreateCommandEncoder(g_wgpu.device, NULL);
    g_wgpu.batch_pass = wgpuCommandEncoderBeginComputePass(g_wgpu.batch_encoder, NULL);
    g_wgpu.batch_active = 1;
    g_wgpu.batch_dispatches = 0;
}

void wgpu_flush(void) {
    if (!g_wgpu.batch_active) return;

    wgpuComputePassEncoderEnd(g_wgpu.batch_pass);
    wgpuComputePassEncoderRelease(g_wgpu.batch_pass);
    g_wgpu.batch_pass = NULL;

    WGPUCommandBuffer cmdbuf = wgpuCommandEncoderFinish(g_wgpu.batch_encoder, NULL);
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
    if (g_wgpu.batch_active) {
        /* Append to current batch — each dispatch gets its own compute pass
         * (required when bind groups change between dispatches) */
        if (g_wgpu.batch_dispatches > 0) {
            /* End current pass and start a new one for the new pipeline/bindings */
            wgpuComputePassEncoderEnd(g_wgpu.batch_pass);
            wgpuComputePassEncoderRelease(g_wgpu.batch_pass);
            g_wgpu.batch_pass = wgpuCommandEncoderBeginComputePass(g_wgpu.batch_encoder, NULL);
        }
        wgpuComputePassEncoderSetPipeline(g_wgpu.batch_pass, pipeline);
        wgpuComputePassEncoderSetBindGroup(g_wgpu.batch_pass, 0, bind_group, 0, NULL);
        wgpuComputePassEncoderDispatchWorkgroups(g_wgpu.batch_pass, wx, wy, wz);
        g_wgpu.batch_dispatches++;
        return;
    }

    /* Non-batch mode: immediate submit + poll (legacy) */
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(g_wgpu.device, NULL);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, wx, wy, wz);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    WGPUCommandBuffer cmdbuf = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(g_wgpu.queue, 1, &cmdbuf);
    wgpuCommandBufferRelease(cmdbuf);
    wgpuCommandEncoderRelease(encoder);

    wgpuDevicePoll(g_wgpu.device, 1, NULL);
}

void wgpu_copy_buffer(WGPUBuffer src, WGPUBuffer dst, size_t size) {
    if (g_wgpu.batch_active) {
        /* Must end compute pass for buffer copy, then restart */
        wgpuComputePassEncoderEnd(g_wgpu.batch_pass);
        wgpuComputePassEncoderRelease(g_wgpu.batch_pass);
        wgpuCommandEncoderCopyBufferToBuffer(g_wgpu.batch_encoder, src, 0, dst, 0, size);
        g_wgpu.batch_pass = wgpuCommandEncoderBeginComputePass(g_wgpu.batch_encoder, NULL);
        g_wgpu.batch_dispatches = 0;  /* reset since we started a new pass */
        return;
    }

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_wgpu.device, NULL);
    wgpuCommandEncoderCopyBufferToBuffer(enc, src, 0, dst, 0, size);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g_wgpu.queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    wgpuDevicePoll(g_wgpu.device, 1, NULL);
}

void wgpu_read_buffer(WGPUBuffer src, size_t offset, void *dst, size_t size) {
    /* Auto-flush any pending batch */
    wgpu_flush();

    size_t aligned = (size + 3) & ~(size_t)3;
    wgpu_ensure_staging(aligned);

    /* GPU copy: src -> staging */
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(g_wgpu.device, NULL);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, src, offset,
                                          g_wgpu.staging_buf, 0, aligned);
    WGPUCommandBuffer cmdbuf = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(g_wgpu.queue, 1, &cmdbuf);
    wgpuCommandBufferRelease(cmdbuf);
    wgpuCommandEncoderRelease(encoder);

    /* Map staging for read */
    int map_status = -1;
    wgpuBufferMapAsync(g_wgpu.staging_buf, WGPUMapMode_Read, 0, aligned,
                       on_buffer_map, &map_status);
    wgpuDevicePoll(g_wgpu.device, 1, NULL);

    if (map_status == 0) {
        const void *mapped = wgpuBufferGetMappedRange(g_wgpu.staging_buf, 0, aligned);
        if (mapped) memcpy(dst, mapped, size);
    }
    wgpuBufferUnmap(g_wgpu.staging_buf);
}

void wgpu_write_buffer(WGPUBuffer dst, size_t offset, const void *src, size_t size) {
    /* Auto-flush before write to ensure ordering */
    wgpu_flush();
    wgpuQueueWriteBuffer(g_wgpu.queue, dst, offset, src, size);
}
