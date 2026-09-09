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
    if (!pl) { wgpu_release_buffer(pb); return; }
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
    WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
    { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(D*H*W, 256), &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }
    wgpu_release_bind_group(bg);
    wgpu_release_buffer(pb);

    if (need_copy) {
        wgpu_copy_buffer(actual_output, output, sz);
        wgpu_release_buffer(actual_output);
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
        if (!pl) { wgpu_release_buffer(pb); return; }
        WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("blur_dhw3");

        WGPUBindGroupEntry e[] = {
            { .binding = 0, .buffer = src_buf, .size = sz },
            { .binding = 1, .buffer = dst_buf, .size = sz },
            { .binding = 2, .buffer = kernel_buf, .size = ksz },
            { .binding = 3, .buffer = pb, .size = sizeof(p) },
        };
        WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 4, .entries = e };
        WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
        { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
        wgpu_dispatch(pl, bg, wx, wy, 1); }
        wgpu_release_bind_group(bg);
        wgpu_release_buffer(pb);
    }

    /* After 3 passes the result is in scratch. Preserve outer batching. */
    wgpu_copy_buffer(scratch, data, sz);
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
    if (!pl) { wgpu_release_buffer(pb); return; }
    WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("adam_moments");

    size_t sz = (size_t)n * 4;
    WGPUBindGroupEntry e[] = {
        { .binding = 0, .buffer = grad, .size = sz },
        { .binding = 1, .buffer = exp_avg, .size = sz },
        { .binding = 2, .buffer = exp_avg_sq, .size = sz },
        { .binding = 3, .buffer = pb, .size = sizeof(p) },
    };
    WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 4, .entries = e };
    WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
    { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }
    wgpu_release_bind_group(bg);
    wgpu_release_buffer(pb);
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
    if (!pl) { wgpu_release_buffer(pb); return; }
    WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("adam_dir");

    size_t sz = (size_t)n * 4;
    WGPUBindGroupEntry e[] = {
        { .binding = 0, .buffer = output, .size = sz },
        { .binding = 1, .buffer = exp_avg, .size = sz },
        { .binding = 2, .buffer = exp_avg_sq, .size = sz },
        { .binding = 3, .buffer = pb, .size = sizeof(p) },
    };
    WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 4, .entries = e };
    WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
    { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }
    wgpu_release_bind_group(bg);
    wgpu_release_buffer(pb);
}

/* ================================================================== */
/* Max L2 norm reduction                                               */
/* ================================================================== */

float wgpu_max_l2_norm_buf(WGPUBuffer data, int spatial, float eps) {
    /* Computed every call, deliberately. This used to reuse a cached value on
     * four calls out of five to avoid the readback, but the cache was a file
     * static: it was stale within a scale, and survived scale transitions and
     * whole registrations. Greedy normalises its update by this maximum on every
     * iteration, and the CPU backend recomputes it every step, so caching made
     * the GPU and CPU paths disagree by construction. */
    int caller_batching = g_wgpu.batch_active;
    const char *wgsl = get_warp_ops_wgsl();
    if (!wgsl) return eps;

    uint32_t n_groups = wgpu_div_ceil(spatial, 256);

    typedef struct { uint32_t spatial, _p; float eps, _p1; } p_t;
    p_t p = { spatial, 0, eps, 0 };
    WGPUBuffer pb = make_params(&p, sizeof(p));

    WGPUBufferUsage u = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
    WGPUBuffer out_buf = wgpu_create_buffer(n_groups * 4, u, "norm_out");

    WGPUComputePipeline pl = wgpu_get_pipeline("max_l2", wgsl, "max_l2_norm");
    if (!pl) { wgpu_release_buffer(pb); wgpu_release_buffer(out_buf); return eps; }
    WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("max_l2");

    WGPUBindGroupEntry e[] = {
        { .binding = 0, .buffer = data, .size = (size_t)spatial * 3 * 4 },
        { .binding = 1, .buffer = out_buf, .size = n_groups * 4 },
        { .binding = 2, .buffer = pb, .size = sizeof(p) },
    };
    WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 3, .entries = e };
    WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
    { uint32_t wx, wy; wgpu_dispatch_dims(n_groups, &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }
    wgpu_release_bind_group(bg);

    /* Read partials and find max on CPU */
    float *partials = (float *)malloc(n_groups * 4);
    if (!partials) {
        wgpu_record_fatal_error("max-L2 host reduction allocation");
        wgpu_release_buffer(pb);
        wgpu_release_buffer(out_buf);
        return eps;
    }
    wgpu_read_buffer(out_buf, 0, partials, n_groups * 4);
    float maxval = 0;
    if (!wgpu_had_fatal_error())
        for (uint32_t i = 0; i < n_groups; i++)
            if (partials[i] > maxval) maxval = partials[i];
    free(partials);

    wgpu_release_buffer(pb);
    wgpu_release_buffer(out_buf);
    if (caller_batching) wgpu_begin_batch();
    return maxval > eps ? maxval : eps;
}

void wgpu_normalize_l2_buf(WGPUBuffer data, int spatial, float eps,
                           float factor, WGPUBuffer max_state) {
    static const char *wgsl = NULL;
    if (!wgsl) wgsl = get_shader_source("normalize_l2.wgsl", NULL);
    if (!wgsl) return;

    typedef struct {
        uint32_t spatial, n;
        float eps, factor;
    } params_t;
    params_t p = {
        (uint32_t)spatial, (uint32_t)(3 * spatial), eps, factor,
    };
    WGPUBuffer pb = make_params(&p, sizeof(p));

    WGPUComputePipeline clear_pl =
        wgpu_get_pipeline("normalize_l2_clear", wgsl, "clear_max");
    if (clear_pl) {
        WGPUBindGroupEntry entry = {
            .binding = 1, .buffer = max_state, .size = sizeof(uint32_t),
        };
        WGPUBindGroupDescriptor desc = {
            .layout = wgpu_get_bind_group_layout("normalize_l2_clear"),
            .entryCount = 1,
            .entries = &entry,
        };
        WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
        wgpu_dispatch(clear_pl, bg, 1, 1, 1);
        wgpu_release_bind_group(bg);
    }

    size_t data_size = (size_t)spatial * 3 * sizeof(float);
    WGPUComputePipeline max_pl =
        wgpu_get_pipeline("normalize_l2_max", wgsl, "reduce_max_l2");
    if (max_pl) {
        WGPUBindGroupEntry entries[] = {
            { .binding = 0, .buffer = data, .size = data_size },
            { .binding = 1, .buffer = max_state, .size = sizeof(uint32_t) },
            { .binding = 2, .buffer = pb, .size = sizeof(p) },
        };
        WGPUBindGroupDescriptor desc = {
            .layout = wgpu_get_bind_group_layout("normalize_l2_max"),
            .entryCount = 3,
            .entries = entries,
        };
        WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
        uint32_t wx, wy;
        wgpu_dispatch_dims(wgpu_div_ceil((uint32_t)spatial, 256), &wx, &wy);
        wgpu_dispatch(max_pl, bg, wx, wy, 1);
        wgpu_release_bind_group(bg);
    }

    WGPUComputePipeline scale_pl =
        wgpu_get_pipeline("normalize_l2_scale", wgsl, "apply_scale");
    if (scale_pl) {
        WGPUBindGroupEntry entries[] = {
            { .binding = 0, .buffer = data, .size = data_size },
            { .binding = 1, .buffer = max_state, .size = sizeof(uint32_t) },
            { .binding = 2, .buffer = pb, .size = sizeof(p) },
        };
        WGPUBindGroupDescriptor desc = {
            .layout = wgpu_get_bind_group_layout("normalize_l2_scale"),
            .entryCount = 3,
            .entries = entries,
        };
        WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
        uint32_t wx, wy;
        wgpu_dispatch_dims(
            wgpu_div_ceil((uint32_t)(3 * spatial), 256), &wx, &wy);
        wgpu_dispatch(scale_pl, bg, wx, wy, 1);
        wgpu_release_bind_group(bg);
    }
    wgpu_release_buffer(pb);
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
    if (!pl) { wgpu_release_buffer(pb); wgpu_release_buffer(partial_buf); memset(h_dL_dA,0,48); return; }
    WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("affine_bwd");

    WGPUBindGroupEntry e[] = {
        { .binding = 0, .buffer = grad_grid, .size = (size_t)total * 3 * 4 },
        { .binding = 1, .buffer = partial_buf, .size = n_blocks * 12 * 4 },
        { .binding = 2, .buffer = pb, .size = sizeof(p) },
    };
    WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 3, .entries = e };
    WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
    { uint32_t wx, wy; wgpu_dispatch_dims(n_blocks, &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }
    wgpu_release_bind_group(bg);

    /* Read partials and sum on CPU */
    float *partials = (float *)malloc(n_blocks * 12 * 4);
    if (!partials) {
        wgpu_record_fatal_error("affine-gradient host reduction allocation");
        memset(h_dL_dA, 0, 12 * sizeof(float));
        wgpu_release_buffer(pb);
        wgpu_release_buffer(partial_buf);
        return;
    }
    wgpu_read_buffer(partial_buf, 0, partials, n_blocks * 12 * 4);

    memset(h_dL_dA, 0, 12 * sizeof(float));
    if (!wgpu_had_fatal_error())
        for (uint32_t b = 0; b < n_blocks; b++)
            for (int k = 0; k < 12; k++)
                h_dL_dA[k] += partials[b * 12 + k];

    free(partials);
    wgpu_release_buffer(pb);
    wgpu_release_buffer(partial_buf);
}
