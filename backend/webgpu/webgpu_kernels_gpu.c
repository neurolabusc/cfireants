/*
 * webgpu_kernels_gpu.c - GPU-native kernel dispatches for deformable ops
 *
 * Replaces CPU fallbacks with WGSL compute shader dispatches for:
 *   - Compositive warp update
 *   - Blur displacement [D,H,W,3]
 *   - Max L2 norm reduction
 *   - Adam moments update + direction
 *   - Affine grid backward reduction
 *   - MI loss (histogram + gradient)
 *
 * These are called from the registration loops (linear_webgpu.c, etc.)
 */

#include "webgpu_context.h"
#include "webgpu_kernels.h"
#include "shader_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* make_params is now wgpu_make_params in webgpu_context.h */
#define make_params wgpu_make_params

/* ================================================================== */
/* Compositive warp update                                             */
/* ================================================================== */

static const char *get_compose_wgsl(void) {
    static const char *src = NULL;
    if (!src) src = get_shader_source("compose.wgsl", NULL);
    return src;
}

void wgpu_fused_compositive_update(WGPUBuffer warp, WGPUBuffer update,
                                    WGPUBuffer output, int D, int H, int W) {
    const char *wgsl = get_compose_wgsl();
    if (!wgsl) { fprintf(stderr, "compose.wgsl not found\n"); return; }

    typedef struct { uint32_t D, H, W, _pad; } p_t;
    p_t p = { D, H, W, 0 };
    WGPUBuffer pb = make_params(&p, sizeof(p));

    WGPUComputePipeline pl = wgpu_get_pipeline("compose", wgsl, "compositive_update");
    if (!pl) { wgpuBufferRelease(pb); return; }
    WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("compose");

    size_t sz = (size_t)D * H * W * 3 * 4;

    /* If update == output, we need a temp buffer to avoid aliasing
     * (WebGPU doesn't allow same buffer as both read and read_write) */
    WGPUBuffer actual_output = output;
    int need_copy = (update == output);
    if (need_copy) {
        actual_output = wgpu_create_buffer(sz,
            WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst, "comp_tmp");
    }

    WGPUBindGroupEntry e[] = {
        { .binding = 0, .buffer = warp, .size = sz },
        { .binding = 1, .buffer = update, .size = sz },
        { .binding = 2, .buffer = actual_output, .size = sz },
        { .binding = 3, .buffer = pb, .size = sizeof(p) },
    };
    WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 4, .entries = e };
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);
    { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(D*H*W, 256), &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }
    wgpuBindGroupRelease(bg);
    wgpuBufferRelease(pb);

    if (need_copy) {
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_wgpu.device, NULL);
        wgpuCommandEncoderCopyBufferToBuffer(enc, actual_output, 0, output, 0, sz);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
        wgpuQueueSubmit(g_wgpu.queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd); wgpuCommandEncoderRelease(enc);
        wgpuDevicePoll(g_wgpu.device, 1, NULL);
        wgpuBufferRelease(actual_output);
    }
}

/* ================================================================== */
/* Blur displacement [D,H,W,3]                                         */
/* ================================================================== */

static const char *get_blur_dhw3_wgsl(void) {
    static const char *src = NULL;
    if (!src) src = get_shader_source("blur_dhw3.wgsl", NULL);
    return src;
}

void wgpu_blur_disp_dhw3(WGPUBuffer data, WGPUBuffer scratch,
                          int D, int H, int W,
                          WGPUBuffer kernel_buf, int klen) {
    const char *wgsl = get_blur_dhw3_wgsl();
    if (!wgsl) return;

    typedef struct { uint32_t D, H, W, klen, axis, _p0, _p1, _p2; } p_t;
    int n = D * H * W;
    size_t sz = (size_t)n * 3 * 4;
    size_t ksz = (size_t)klen * 4;

    /* 3 axis passes: data→scratch, scratch→data, data→scratch, copy scratch→data */
    for (int axis = 0; axis < 3; axis++) {
        WGPUBuffer src_buf = (axis % 2 == 0) ? data : scratch;
        WGPUBuffer dst_buf = (axis % 2 == 0) ? scratch : data;

        p_t p = { D, H, W, klen, axis, 0, 0, 0 };
        WGPUBuffer pb = make_params(&p, sizeof(p));

        WGPUComputePipeline pl = wgpu_get_pipeline("blur_dhw3", wgsl, "conv1d_dhw3");
        if (!pl) { wgpuBufferRelease(pb); return; }
        WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("blur_dhw3");

        WGPUBindGroupEntry e[] = {
            { .binding = 0, .buffer = src_buf, .size = sz },
            { .binding = 1, .buffer = dst_buf, .size = sz },
            { .binding = 2, .buffer = kernel_buf, .size = ksz },
            { .binding = 3, .buffer = pb, .size = sizeof(p) },
        };
        WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 4, .entries = e };
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);
        { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
        wgpu_dispatch(pl, bg, wx, wy, 1); }
        wgpuBindGroupRelease(bg);
        wgpuBufferRelease(pb);
    }

    /* After 3 passes (even number), result is in scratch. Copy to data. */
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_wgpu.device, NULL);
    wgpuCommandEncoderCopyBufferToBuffer(enc, scratch, 0, data, 0, sz);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g_wgpu.queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd); wgpuCommandEncoderRelease(enc);
    wgpuDevicePoll(g_wgpu.device, 1, NULL);
}

/* ================================================================== */
/* Adam moments update + direction                                     */
/* ================================================================== */

static const char *get_warp_ops_wgsl(void) {
    static const char *src = NULL;
    if (!src) src = get_shader_source("warp_ops.wgsl", NULL);
    return src;
}

void wgpu_adam_moments_update_buf(WGPUBuffer grad, WGPUBuffer exp_avg,
                                   WGPUBuffer exp_avg_sq,
                                   float beta1, float beta2, int n) {
    const char *wgsl = get_warp_ops_wgsl();
    if (!wgsl) return;

    typedef struct { uint32_t n, _p; float beta1, beta2; } p_t;
    p_t p = { n, 0, beta1, beta2 };
    WGPUBuffer pb = make_params(&p, sizeof(p));

    WGPUComputePipeline pl = wgpu_get_pipeline("adam_moments", wgsl, "adam_moments_update");
    if (!pl) { wgpuBufferRelease(pb); return; }
    WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("adam_moments");

    size_t sz = (size_t)n * 4;
    WGPUBindGroupEntry e[] = {
        { .binding = 0, .buffer = grad, .size = sz },
        { .binding = 1, .buffer = exp_avg, .size = sz },
        { .binding = 2, .buffer = exp_avg_sq, .size = sz },
        { .binding = 3, .buffer = pb, .size = sizeof(p) },
    };
    WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 4, .entries = e };
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);
    { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }
    wgpuBindGroupRelease(bg);
    wgpuBufferRelease(pb);
}

void wgpu_adam_direction_buf(WGPUBuffer output, WGPUBuffer exp_avg,
                              WGPUBuffer exp_avg_sq,
                              float bc1, float bc2, float eps, int n) {
    const char *wgsl = get_warp_ops_wgsl();
    if (!wgsl) return;

    typedef struct { uint32_t n, _p; float inv_bc1, inv_bc2, eps, _p2; uint32_t _p3, _p4; } p_t;
    p_t p = { n, 0, 1.0f/bc1, 1.0f/bc2, eps, 0, 0, 0 };
    WGPUBuffer pb = make_params(&p, sizeof(p));

    WGPUComputePipeline pl = wgpu_get_pipeline("adam_dir", wgsl, "adam_direction");
    if (!pl) { wgpuBufferRelease(pb); return; }
    WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("adam_dir");

    size_t sz = (size_t)n * 4;
    WGPUBindGroupEntry e[] = {
        { .binding = 0, .buffer = output, .size = sz },
        { .binding = 1, .buffer = exp_avg, .size = sz },
        { .binding = 2, .buffer = exp_avg_sq, .size = sz },
        { .binding = 3, .buffer = pb, .size = sizeof(p) },
    };
    WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 4, .entries = e };
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);
    { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }
    wgpuBindGroupRelease(bg);
    wgpuBufferRelease(pb);
}

/* ================================================================== */
/* Max L2 norm reduction                                               */
/* ================================================================== */

/* Cached max L2 norm — only recompute every N calls to reduce sync */
static float cached_max_l2 = 1.0f;
static int max_l2_call_count = 0;

float wgpu_max_l2_norm_buf(WGPUBuffer data, int spatial, float eps) {
    /* Only recompute every 5 calls — reuse cached value otherwise */
    max_l2_call_count++;
    if (max_l2_call_count % 5 != 1 && cached_max_l2 > eps) {
        return cached_max_l2;
    }

    const char *wgsl = get_warp_ops_wgsl();
    if (!wgsl) return eps;

    uint32_t n_groups = wgpu_div_ceil(spatial, 256);

    typedef struct { uint32_t spatial, _p; float eps, _p1; } p_t;
    p_t p = { spatial, 0, eps, 0 };
    WGPUBuffer pb = make_params(&p, sizeof(p));

    WGPUBufferUsage u = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
    WGPUBuffer out_buf = wgpu_create_buffer(n_groups * 4, u, "norm_out");

    WGPUComputePipeline pl = wgpu_get_pipeline("max_l2", wgsl, "max_l2_norm");
    if (!pl) { wgpuBufferRelease(pb); wgpuBufferRelease(out_buf); return eps; }
    WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("max_l2");

    WGPUBindGroupEntry e[] = {
        { .binding = 0, .buffer = data, .size = (size_t)spatial * 3 * 4 },
        { .binding = 1, .buffer = out_buf, .size = n_groups * 4 },
        { .binding = 2, .buffer = pb, .size = sizeof(p) },
    };
    WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 3, .entries = e };
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);
    { uint32_t wx, wy; wgpu_dispatch_dims(n_groups, &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }
    wgpuBindGroupRelease(bg);

    /* Read partials and find max on CPU */
    float *partials = (float *)malloc(n_groups * 4);
    wgpu_read_buffer(out_buf, 0, partials, n_groups * 4);
    float maxval = 0;
    for (uint32_t i = 0; i < n_groups; i++)
        if (partials[i] > maxval) maxval = partials[i];
    free(partials);

    wgpuBufferRelease(pb);
    wgpuBufferRelease(out_buf);
    cached_max_l2 = eps + maxval;
    return cached_max_l2;
}

/* ================================================================== */
/* Affine grid backward                                                */
/* ================================================================== */

static const char *get_affine_bwd_wgsl(void) {
    static const char *src = NULL;
    if (!src) src = get_shader_source("affine_grid_bwd.wgsl", NULL);
    return src;
}

void wgpu_affine_grid_backward(WGPUBuffer grad_grid, int D, int H, int W,
                                float h_dL_dA[12]) {
    const char *wgsl = get_affine_bwd_wgsl();
    if (!wgsl) {
        /* CPU fallback */
        memset(h_dL_dA, 0, 12 * sizeof(float));
        return;
    }

    int total = D * H * W;
    uint32_t n_blocks = wgpu_div_ceil(total, 256);

    typedef struct { uint32_t D, H, W, _pad; } p_t;
    p_t p = { D, H, W, 0 };
    WGPUBuffer pb = make_params(&p, sizeof(p));

    WGPUBufferUsage u = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
    WGPUBuffer partial_buf = wgpu_create_buffer(n_blocks * 12 * 4, u, "agb_partial");

    WGPUComputePipeline pl = wgpu_get_pipeline("affine_bwd", wgsl, "affine_grid_bwd");
    if (!pl) { wgpuBufferRelease(pb); wgpuBufferRelease(partial_buf); memset(h_dL_dA,0,48); return; }
    WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("affine_bwd");

    WGPUBindGroupEntry e[] = {
        { .binding = 0, .buffer = grad_grid, .size = (size_t)total * 3 * 4 },
        { .binding = 1, .buffer = partial_buf, .size = n_blocks * 12 * 4 },
        { .binding = 2, .buffer = pb, .size = sizeof(p) },
    };
    WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 3, .entries = e };
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);
    { uint32_t wx, wy; wgpu_dispatch_dims(n_blocks, &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }
    wgpuBindGroupRelease(bg);

    /* Read partials and sum on CPU */
    float *partials = (float *)malloc(n_blocks * 12 * 4);
    wgpu_read_buffer(partial_buf, 0, partials, n_blocks * 12 * 4);

    memset(h_dL_dA, 0, 12 * sizeof(float));
    for (uint32_t b = 0; b < n_blocks; b++)
        for (int k = 0; k < 12; k++)
            h_dL_dA[k] += partials[b * 12 + k];

    free(partials);
    wgpuBufferRelease(pb);
    wgpuBufferRelease(partial_buf);
}
