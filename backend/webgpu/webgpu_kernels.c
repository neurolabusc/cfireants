/*
 * webgpu_kernels.c - WebGPU kernel dispatch implementations
 *
 * Each function creates a bind group, dispatches a WGSL compute shader,
 * and synchronously waits for completion. Mirrors the CUDA kernel wrappers.
 */

#include "webgpu_kernels.h"
#include "webgpu_context.h"
#include "shader_loader.h"
#include "cfireants/registration.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* make_params is now wgpu_make_params in webgpu_context.h */
#define make_params wgpu_make_params

/* Helper: dispatch with auto bind group creation */
static void dispatch_1buf(const char *name, const char *wgsl, const char *entry,
                          WGPUBuffer b0, size_t s0,
                          WGPUBuffer params, size_t ps,
                          uint32_t groups) {
    WGPUComputePipeline pl = wgpu_get_pipeline(name, wgsl, entry);
    if (!pl) return;
    WGPUBindGroupLayout lay = wgpu_get_bind_group_layout(name);
    WGPUBindGroupEntry entries[] = {
        { .binding = 0, .buffer = b0, .size = s0 },
        { .binding = 1, .buffer = params, .size = ps },
    };
    WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 2, .entries = entries };
    WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
    uint32_t wx, wy;
    wgpu_dispatch_dims(groups, &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1);
    wgpu_release_bind_group(bg);
}

/* ================================================================== */
/* Grid sampling                                                       */
/* ================================================================== */

typedef struct {
    uint32_t B, C, iD, iH, iW, oD, oH, oW;
} gs_params_t;

void wgpu_grid_sample_3d_fwd(
    WGPUBuffer input, WGPUBuffer grid, WGPUBuffer output,
    int B, int C, int iD, int iH, int iW, int oD, int oH, int oW)
{
    gs_params_t p = { B, C, iD, iH, iW, oD, oH, oW };
    WGPUBuffer pb = make_params(&p, sizeof(p));

    WGPUComputePipeline pl = wgpu_get_pipeline(
        "gs_fwd", get_shader_source("grid_sample.wgsl", NULL),
        "grid_sample_fwd");
    if (!pl) { wgpu_release_buffer(pb); return; }
    WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("gs_fwd");

    size_t in_sz = (size_t)B * C * iD * iH * iW * 4;
    size_t grid_sz = (size_t)B * oD * oH * oW * 3 * 4;
    size_t out_sz = (size_t)B * C * oD * oH * oW * 4;

    WGPUBindGroupEntry entries[] = {
        { .binding = 0, .buffer = input, .size = in_sz },
        { .binding = 1, .buffer = grid, .size = grid_sz },
        { .binding = 2, .buffer = output, .size = out_sz },
        { .binding = 3, .buffer = pb, .size = sizeof(p) },
    };
    WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 4, .entries = entries };
    WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);

    uint32_t total = B * oD * oH * oW;
    { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(total, 256), &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }

    wgpu_release_bind_group(bg);
    wgpu_release_buffer(pb);
}

void wgpu_grid_sample_3d_bwd(
    WGPUBuffer grad_output, WGPUBuffer input, WGPUBuffer grid,
    WGPUBuffer grad_grid,
    int B, int C, int iD, int iH, int iW, int oD, int oH, int oW)
{
    gs_params_t p = { B, C, iD, iH, iW, oD, oH, oW };
    WGPUBuffer pb = make_params(&p, sizeof(p));

    WGPUComputePipeline pl = wgpu_get_pipeline(
        "gs_bwd", get_shader_source("grid_sample_bwd.wgsl", NULL),
        "grid_sample_bwd");
    if (!pl) { wgpu_release_buffer(pb); return; }
    WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("gs_bwd");

    size_t go_sz = (size_t)B * C * oD * oH * oW * 4;
    size_t in_sz = (size_t)B * C * iD * iH * iW * 4;
    size_t grid_sz = (size_t)B * oD * oH * oW * 3 * 4;
    size_t gg_sz = grid_sz;

    WGPUBindGroupEntry entries[] = {
        { .binding = 0, .buffer = grad_output, .size = go_sz },
        { .binding = 1, .buffer = input, .size = in_sz },
        { .binding = 2, .buffer = grid, .size = grid_sz },
        { .binding = 3, .buffer = grad_grid, .size = gg_sz },
        { .binding = 4, .buffer = pb, .size = sizeof(p) },
    };
    WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 5, .entries = entries };
    WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);

    uint32_t total = B * oD * oH * oW;
    { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(total, 256), &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }

    wgpu_release_bind_group(bg);
    wgpu_release_buffer(pb);
}

/* ================================================================== */
/* Affine grid                                                         */
/* ================================================================== */

typedef struct { uint32_t B, D, H, W; } ag_params_t;

void wgpu_affine_grid_3d(WGPUBuffer affine, WGPUBuffer grid,
                          int B, int D, int H, int W) {
    ag_params_t p = { B, D, H, W };
    WGPUBuffer pb = make_params(&p, sizeof(p));

    WGPUComputePipeline pl = wgpu_get_pipeline(
        "affine_grid", get_shader_source("affine_grid.wgsl", NULL),
        "affine_grid");
    if (!pl) { wgpu_release_buffer(pb); return; }
    WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("affine_grid");

    WGPUBindGroupEntry entries[] = {
        { .binding = 0, .buffer = affine, .size = (size_t)B * 12 * 4 },
        { .binding = 1, .buffer = grid, .size = (size_t)B * D * H * W * 3 * 4 },
        { .binding = 2, .buffer = pb, .size = sizeof(p) },
    };
    WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 3, .entries = entries };
    WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);

    { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(B * D * H * W, 256), &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }
    wgpu_release_bind_group(bg);
    wgpu_release_buffer(pb);
}

/* ================================================================== */
/* Trilinear resize                                                    */
/* ================================================================== */

typedef struct {
    uint32_t B, C, iD, iH, iW, oD, oH, oW, align_corners, _pad;
} resize_params_t;

void wgpu_trilinear_resize(
    WGPUBuffer input, WGPUBuffer output,
    int B, int C, int iD, int iH, int iW,
    int oD, int oH, int oW, int align_corners)
{
    resize_params_t p = { B, C, iD, iH, iW, oD, oH, oW, align_corners, 0 };
    WGPUBuffer pb = make_params(&p, sizeof(p));

    WGPUComputePipeline pl = wgpu_get_pipeline(
        "resize", get_shader_source("resize.wgsl", NULL),
        "trilinear_resize");
    if (!pl) { wgpu_release_buffer(pb); return; }
    WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("resize");

    size_t in_sz = (size_t)B * C * iD * iH * iW * 4;
    size_t out_sz = (size_t)B * C * oD * oH * oW * 4;

    WGPUBindGroupEntry entries[] = {
        { .binding = 0, .buffer = input, .size = in_sz },
        { .binding = 1, .buffer = output, .size = out_sz },
        { .binding = 2, .buffer = pb, .size = sizeof(p) },
    };
    WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 3, .entries = entries };
    WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);

    uint32_t total = B * C * oD * oH * oW;
    { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(total, 256), &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }
    wgpu_release_bind_group(bg);
    wgpu_release_buffer(pb);
}

/* ================================================================== */
/* Element-wise ops on raw buffers                                     */
/* ================================================================== */

typedef struct { uint32_t n; uint32_t pad0; float value; float pad1; } ew_params_t;

void wgpu_tensor_fill_buf(WGPUBuffer buf, float value, int n) {
    ew_params_t p = { .n = n, .value = value };
    WGPUBuffer pb = make_params(&p, sizeof(p));
    dispatch_1buf("fill", get_shader_source("elementwise.wgsl", NULL),
                  "fill", buf, (size_t)n * 4, pb, sizeof(p),
                  wgpu_div_ceil(n, 256));
    wgpu_release_buffer(pb);
}

void wgpu_tensor_scale_buf(WGPUBuffer buf, float alpha, int n) {
    ew_params_t p = { .n = n, .value = alpha };
    WGPUBuffer pb = make_params(&p, sizeof(p));
    dispatch_1buf("scale", get_shader_source("elementwise.wgsl", NULL),
                  "scale", buf, (size_t)n * 4, pb, sizeof(p),
                  wgpu_div_ceil(n, 256));
    wgpu_release_buffer(pb);
}

void wgpu_tensor_add_buf(WGPUBuffer a, WGPUBuffer b, int n) {
    /* a += b, using axpy with alpha=1 */
    ew_params_t p = { .n = n, .value = 1.0f };
    WGPUBuffer pb = make_params(&p, sizeof(p));

    WGPUComputePipeline pl = wgpu_get_pipeline(
        "axpy", get_shader_source("axpy.wgsl", NULL), "axpy");
    if (!pl) { wgpu_release_buffer(pb); return; }
    WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("axpy");
    WGPUBindGroupEntry entries[] = {
        { .binding = 0, .buffer = a, .size = (size_t)n * 4 },
        { .binding = 1, .buffer = pb, .size = sizeof(p) },
        { .binding = 2, .buffer = b, .size = (size_t)n * 4 },
    };
    WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 3, .entries = entries };
    WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
    { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }
    wgpu_release_bind_group(bg);
    wgpu_release_buffer(pb);
}

/* ================================================================== */
/* Adam                                                                */
/* ================================================================== */

typedef struct {
    uint32_t n, step;
    float lr, beta1, beta2, eps;
    uint32_t _pad0, _pad1;
} adam_params_t;

void wgpu_adam_step(
    WGPUBuffer param, WGPUBuffer grad,
    WGPUBuffer exp_avg, WGPUBuffer exp_avg_sq,
    float lr, float beta1, float beta2, float eps,
    int step, int n)
{
    adam_params_t p = { n, step, lr, beta1, beta2, eps, 0, 0 };
    WGPUBuffer pb = make_params(&p, sizeof(p));

    WGPUComputePipeline pl = wgpu_get_pipeline(
        "adam_step", get_shader_source("adam.wgsl", NULL), "adam_step");
    if (!pl) { wgpu_release_buffer(pb); return; }
    WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("adam_step");

    size_t sz = (size_t)n * 4;
    WGPUBindGroupEntry entries[] = {
        { .binding = 0, .buffer = param, .size = sz },
        { .binding = 1, .buffer = grad, .size = sz },
        { .binding = 2, .buffer = exp_avg, .size = sz },
        { .binding = 3, .buffer = exp_avg_sq, .size = sz },
        { .binding = 4, .buffer = pb, .size = sizeof(p) },
    };
    WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 5, .entries = entries };
    WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);

    { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }
    wgpu_release_bind_group(bg);
    wgpu_release_buffer(pb);
}

/* ================================================================== */
/* CC Loss (multi-step dispatch)                                       */
/* ================================================================== */

/* Helper: separable box filter (3 dispatches along D, H, W) */
static void box_filter_3d(WGPUBuffer in_buf, WGPUBuffer out_buf,
                           WGPUBuffer tmp_buf,
                           int D, int H, int W, int ks) {
    typedef struct { uint32_t D, H, W, ks, axis, _p0, _p1, _p2; } box_p_t;
    int n = D * H * W;
    size_t sz = (size_t)n * 4;

    for (int axis = 0; axis < 3; axis++) {
        WGPUBuffer src = (axis == 0) ? in_buf : (axis == 1) ? tmp_buf : out_buf;
        WGPUBuffer dst = (axis == 0) ? tmp_buf : (axis == 1) ? out_buf : tmp_buf;

        box_p_t p = { D, H, W, ks, axis, 0, 0, 0 };
        WGPUBuffer pb = make_params(&p, sizeof(p));

        WGPUComputePipeline pl = wgpu_get_pipeline(
            "box_filter", get_shader_source("box_filter.wgsl", NULL),
            "box_filter");
        if (!pl) { wgpu_release_buffer(pb); continue; }
        WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("box_filter");
        WGPUBindGroupEntry entries[] = {
            { .binding = 0, .buffer = src, .size = sz },
            { .binding = 1, .buffer = dst, .size = sz },
            { .binding = 2, .buffer = pb, .size = sizeof(p) },
        };
        WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 3, .entries = entries };
        WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
        { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
        wgpu_dispatch(pl, bg, wx, wy, 1); }
        wgpu_release_bind_group(bg);
        wgpu_release_buffer(pb);
    }

    /* After 3 passes the result is in tmp_buf. Keep the copy in the active
     * command buffer when the caller is batching. */
    wgpu_copy_buffer(tmp_buf, out_buf, (size_t)n * 4);
}

/* ================================================================== */
/* Fused CC Loss (matching CUDA fused_cc.cu exactly)                   */
/* ================================================================== */

/* Box-filter one channel of the intermediates buffer in-place.
 * Channel occupies interm[ch*spatial .. (ch+1)*spatial-1]. */
static const char *get_fused_cc_shader(const char *filename) {
    static const char *create_src = NULL;
    static const char *box_src = NULL;
    static const char *fwd_src = NULL;
    static const char *modify_src = NULL;
    static const char *grads_src = NULL;
    const char **slot = NULL;

    if (strcmp(filename, "fused_cc.wgsl") == 0) slot = &create_src;
    else if (strcmp(filename, "fused_cc_box.wgsl") == 0) slot = &box_src;
    else if (strcmp(filename, "fused_cc_fwd.wgsl") == 0) slot = &fwd_src;
    else if (strcmp(filename, "fused_cc_bwd_modify.wgsl") == 0) slot = &modify_src;
    else if (strcmp(filename, "fused_cc_bwd_grads.wgsl") == 0) slot = &grads_src;
    if (!slot) return NULL;
    if (!*slot) *slot = get_shader_source(filename, NULL);
    return *slot;
}

static int fcc_dispatch_packed_box(WGPUBuffer input, WGPUBuffer output,
                                   int D, int H, int W, int ks, int axis) {
    const char *wgsl = get_fused_cc_shader("fused_cc_box.wgsl");
    if (!wgsl) return -1;

    typedef struct {
        uint32_t D, H, W, ks, axis, channels, _p0, _p1;
    } params_t;
    params_t p = { (uint32_t)D, (uint32_t)H, (uint32_t)W,
                   (uint32_t)ks, (uint32_t)axis, 5, 0, 0 };
    WGPUBuffer pb = make_params(&p, sizeof(p));
    WGPUComputePipeline pl =
        wgpu_get_pipeline("fcc_box_packed", wgsl, "box_filter_packed");
    if (!pl) {
        wgpu_release_buffer(pb);
        return -1;
    }

    size_t size = (size_t)5 * D * H * W * sizeof(float);
    WGPUBindGroupEntry entries[] = {
        { .binding = 0, .buffer = input, .size = size },
        { .binding = 1, .buffer = output, .size = size },
        { .binding = 2, .buffer = pb, .size = sizeof(p) },
    };
    WGPUBindGroupDescriptor desc = {
        .layout = wgpu_get_bind_group_layout("fcc_box_packed"),
        .entryCount = 3,
        .entries = entries,
    };
    WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
    uint32_t groups = wgpu_div_ceil((uint32_t)(5 * D * H * W), 256);
    uint32_t wx, wy;
    wgpu_dispatch_dims(groups, &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1);
    wgpu_release_bind_group(bg);
    wgpu_release_buffer(pb);
    return 0;
}

void wgpu_fused_cc_loss(
    WGPUBuffer pred, WGPUBuffer target,
    WGPUBuffer grad_pred, WGPUBuffer grad_target,
    int D, int H, int W, int ks,
    float *h_loss_out,
    WGPUBuffer interm, WGPUBuffer scratch)
{
    const uint32_t n = (uint32_t)(D * H * W);
    const size_t size = (size_t)n * sizeof(float);
    const size_t workspace_size = 5 * size;
    const uint32_t groups = wgpu_div_ceil(n, 256);
    const int caller_batching = g_wgpu.batch_active;

    if (!caller_batching) wgpu_begin_batch();

    /* Pack [I, J, I^2, J^2, IJ] in one pass. */
    {
        const char *wgsl = get_fused_cc_shader("fused_cc.wgsl");
        typedef struct { uint32_t n, _p0, _p1, _p2; } params_t;
        params_t p = { n, 0, 0, 0 };
        WGPUBuffer pb = make_params(&p, sizeof(p));
        WGPUComputePipeline pl =
            wgpu_get_pipeline("fcc_create", wgsl, "create_intermediates");
        if (!pl) {
            wgpu_release_buffer(pb);
            if (!caller_batching) wgpu_flush();
            return;
        }
        WGPUBindGroupEntry entries[] = {
            { .binding = 0, .buffer = pred, .size = size },
            { .binding = 1, .buffer = target, .size = size },
            { .binding = 2, .buffer = interm, .size = workspace_size },
            { .binding = 3, .buffer = pb, .size = sizeof(p) },
        };
        WGPUBindGroupDescriptor desc = {
            .layout = wgpu_get_bind_group_layout("fcc_create"),
            .entryCount = 4,
            .entries = entries,
        };
        WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
        uint32_t wx, wy;
        wgpu_dispatch_dims(groups, &wx, &wy);
        wgpu_dispatch(pl, bg, wx, wy, 1);
        wgpu_release_bind_group(bg);
        wgpu_release_buffer(pb);
    }

    /* Filter all five channels together. The filtered values end in scratch. */
    fcc_dispatch_packed_box(interm, scratch, D, H, W, ks, 0);
    fcc_dispatch_packed_box(scratch, interm, D, H, W, ks, 1);
    fcc_dispatch_packed_box(interm, scratch, D, H, W, ks, 2);

    if (h_loss_out) {
        const char *wgsl = get_fused_cc_shader("fused_cc_fwd.wgsl");
        typedef struct {
            uint32_t n, kernel_volume;
            float nr, dr;
        } params_t;
        params_t p = { n, (uint32_t)(ks * ks * ks), 1e-5f, 1e-5f };
        WGPUBuffer pb = make_params(&p, sizeof(p));
        WGPUComputePipeline pl = wgpu_get_pipeline("fcc_fwd", wgsl, "fcc_fwd");
        if (pl) {
            WGPUBindGroupEntry entries[] = {
                { .binding = 0, .buffer = scratch, .size = workspace_size },
                { .binding = 1, .buffer = interm, .size = workspace_size },
                { .binding = 2, .buffer = pb, .size = sizeof(p) },
            };
            WGPUBindGroupDescriptor desc = {
                .layout = wgpu_get_bind_group_layout("fcc_fwd"),
                .entryCount = 3,
                .entries = entries,
            };
            WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
            uint32_t wx, wy;
            wgpu_dispatch_dims(groups, &wx, &wy);
            wgpu_dispatch(pl, bg, wx, wy, 1);
            wgpu_release_bind_group(bg);

            float *partials = (float *)malloc((size_t)groups * sizeof(float));
            if (!partials) {
                wgpu_record_fatal_error("fused CC host reduction allocation");
                *h_loss_out = 0.0f;
            } else {
                wgpu_read_buffer(interm, 0, partials,
                                 (size_t)groups * sizeof(float));
                double sum = 0.0;
                if (!wgpu_had_fatal_error())
                    for (uint32_t i = 0; i < groups; i++) sum += partials[i];
                free(partials);
                *h_loss_out = wgpu_had_fatal_error() ? 0.0f
                                                      : -(float)(sum / n);
            }
        }
        wgpu_release_buffer(pb);

        /* Readback flushes the caller's batch; continue encoding afterward. */
        if (grad_pred || caller_batching) wgpu_begin_batch();
    }

    if (grad_pred) {
        /* Convert the filtered values into gradient multipliers in place. */
        {
            const char *wgsl =
                get_fused_cc_shader("fused_cc_bwd_modify.wgsl");
            typedef struct {
                uint32_t n, kernel_volume;
                float nr, dr, grad_output;
                uint32_t compute_grad_target, _p0, _p1;
            } params_t;
            params_t p = {
                n, (uint32_t)(ks * ks * ks), 1e-5f, 1e-5f,
                -1.0f / (float)n, grad_target ? 1u : 0u, 0, 0,
            };
            WGPUBuffer pb = make_params(&p, sizeof(p));
            WGPUComputePipeline pl =
                wgpu_get_pipeline("fcc_bwd_modify", wgsl, "bwd_modify");
            if (pl) {
                WGPUBindGroupEntry entries[] = {
                    { .binding = 0, .buffer = scratch, .size = workspace_size },
                    { .binding = 1, .buffer = pb, .size = sizeof(p) },
                };
                WGPUBindGroupDescriptor desc = {
                    .layout = wgpu_get_bind_group_layout("fcc_bwd_modify"),
                    .entryCount = 2,
                    .entries = entries,
                };
                WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
                uint32_t wx, wy;
                wgpu_dispatch_dims(groups, &wx, &wy);
                wgpu_dispatch(pl, bg, wx, wy, 1);
                wgpu_release_bind_group(bg);
            }
            wgpu_release_buffer(pb);
        }

        /* The box filter is self-adjoint. Filter all multipliers together. */
        fcc_dispatch_packed_box(scratch, interm, D, H, W, ks, 0);
        fcc_dispatch_packed_box(interm, scratch, D, H, W, ks, 1);
        fcc_dispatch_packed_box(scratch, interm, D, H, W, ks, 2);

        /* Assemble both image gradients and apply the required 1/ks^2 fix. */
        {
            const char *wgsl =
                get_fused_cc_shader("fused_cc_bwd_grads.wgsl");
            typedef struct {
                uint32_t n, compute_grad_target;
                float inv_ks2;
                uint32_t _p;
            } params_t;
            params_t p = {
                n, grad_target ? 1u : 0u,
                1.0f / (float)(ks * ks), 0,
            };
            WGPUBuffer pb = make_params(&p, sizeof(p));
            WGPUComputePipeline pl =
                wgpu_get_pipeline("fcc_bwd_grads", wgsl, "bwd_grads");
            if (pl) {
                WGPUBindGroupEntry entries[] = {
                    { .binding = 0, .buffer = interm, .size = workspace_size },
                    { .binding = 1, .buffer = pred, .size = size },
                    { .binding = 2, .buffer = target, .size = size },
                    { .binding = 3, .buffer = grad_pred, .size = size },
                    { .binding = 4,
                      .buffer = grad_target ? grad_target : grad_pred,
                      .size = size },
                    { .binding = 5, .buffer = pb, .size = sizeof(p) },
                };
                WGPUBindGroupDescriptor desc = {
                    .layout = wgpu_get_bind_group_layout("fcc_bwd_grads"),
                    .entryCount = 6,
                    .entries = entries,
                };
                WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
                uint32_t wx, wy;
                wgpu_dispatch_dims(groups, &wx, &wy);
                wgpu_dispatch(pl, bg, wx, wy, 1);
                wgpu_release_bind_group(bg);
            }
            wgpu_release_buffer(pb);
        }
    }

    if (!caller_batching) wgpu_flush();
}

int wgpu_cc_workspace_init(wgpu_cc_workspace_t *ws, int n, int with_gradient) {
    if (!ws || n <= 0) return -1;
    memset(ws, 0, sizeof(*ws));
    ws->voxels = n;
    ws->has_gradient = with_gradient != 0;
    size_t sz = (size_t)n * sizeof(float);
    size_t partial_sz = (size_t)wgpu_div_ceil((uint32_t)n, 256) * sizeof(float);
    WGPUBufferUsage u = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc |
                        WGPUBufferUsage_CopyDst;
    ws->p_sum = wgpu_create_buffer(sz, u, "cc_p_sum");
    ws->t_sum = wgpu_create_buffer(sz, u, "cc_t_sum");
    ws->p2_sum = wgpu_create_buffer(sz, u, "cc_p2_sum");
    ws->t2_sum = wgpu_create_buffer(sz, u, "cc_t2_sum");
    ws->tp_sum = wgpu_create_buffer(sz, u, "cc_tp_sum");
    ws->work = wgpu_create_buffer(sz, u, "cc_work");
    ws->tmp = wgpu_create_buffer(sz, u, "cc_tmp");
    ws->ncc = wgpu_create_buffer(partial_sz, u, "cc_ncc");
    ws->grad_sources = wgpu_create_buffer(with_gradient ? 3 * sz : 16, u,
                                           "cc_grad_sources");
    if (with_gradient) {
        ws->src_p = wgpu_create_buffer(sz, u, "cc_src_p");
        ws->src_p2 = wgpu_create_buffer(sz, u, "cc_src_p2");
        ws->src_tp = wgpu_create_buffer(sz, u, "cc_src_tp");
        ws->adj_p = wgpu_create_buffer(sz, u, "cc_adj_p");
        ws->adj_p2 = wgpu_create_buffer(sz, u, "cc_adj_p2");
        ws->adj_tp = wgpu_create_buffer(sz, u, "cc_adj_tp");
    }
    if (wgpu_had_fatal_error()) {
        wgpu_cc_workspace_cleanup(ws);
        return -1;
    }
    return 0;
}

void wgpu_cc_workspace_cleanup(wgpu_cc_workspace_t *ws) {
    if (!ws) return;
    wgpu_release_buffer(ws->p_sum); wgpu_release_buffer(ws->t_sum);
    wgpu_release_buffer(ws->p2_sum); wgpu_release_buffer(ws->t2_sum);
    wgpu_release_buffer(ws->tp_sum); wgpu_release_buffer(ws->work);
    wgpu_release_buffer(ws->tmp); wgpu_release_buffer(ws->ncc);
    wgpu_release_buffer(ws->grad_sources);
    wgpu_release_buffer(ws->src_p); wgpu_release_buffer(ws->src_p2);
    wgpu_release_buffer(ws->src_tp); wgpu_release_buffer(ws->adj_p);
    wgpu_release_buffer(ws->adj_p2); wgpu_release_buffer(ws->adj_tp);
    memset(ws, 0, sizeof(*ws));
}

void wgpu_cc_loss_3d_raw(
    WGPUBuffer pred, WGPUBuffer target,
    WGPUBuffer grad_pred,
    int D, int H, int W, int ks,
    float *h_loss_out,
    wgpu_cc_workspace_t *workspace)
{
    int n = D * H * W;
    size_t sz = (size_t)n * 4;
    int compute_grad = (grad_pred != NULL) ? 1 : 0;
    wgpu_cc_workspace_t local_workspace;
    int owns_workspace = 0;
    if (!workspace) {
        if (wgpu_cc_workspace_init(&local_workspace, n, compute_grad) != 0) return;
        workspace = &local_workspace;
        owns_workspace = 1;
    } else if (workspace->voxels < n ||
               (compute_grad && !workspace->has_gradient)) {
        wgpu_record_fatal_error("CC workspace capacity");
        return;
    }
    int caller_batching = g_wgpu.batch_active;
    if (!caller_batching) wgpu_begin_batch();

    WGPUBuffer p_sum=workspace->p_sum, t_sum=workspace->t_sum;
    WGPUBuffer p2_sum=workspace->p2_sum, t2_sum=workspace->t2_sum;
    WGPUBuffer tp_sum=workspace->tp_sum, work=workspace->work;
    WGPUBuffer tmp=workspace->tmp;

    /* Step 1: Multiply intermediates */
    typedef struct { uint32_t n, _p0, _p1, _p2; } mul_p_t;
    mul_p_t mp = { n, 0, 0, 0 };
    WGPUBuffer mpb = make_params(&mp, sizeof(mp));

    WGPUComputePipeline mul_pl = wgpu_get_pipeline(
        "multiply", get_shader_source("cc_loss.wgsl", NULL), "multiply");
    WGPUBindGroupLayout mul_lay = wgpu_get_bind_group_layout("multiply");

    /* P*P -> work, box_filter -> p2_sum */
    {
        WGPUBindGroupEntry e[] = {
            { .binding = 0, .buffer = pred, .size = sz },
            { .binding = 1, .buffer = pred, .size = sz },
            { .binding = 2, .buffer = work, .size = sz },
            { .binding = 3, .buffer = mpb, .size = sizeof(mp) },
        };
        WGPUBindGroupDescriptor d = { .layout = mul_lay, .entryCount = 4, .entries = e };
        WGPUBindGroup bg = wgpu_create_bind_group(&d, NULL);
        { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
        wgpu_dispatch(mul_pl, bg, wx, wy, 1); }
        wgpu_release_bind_group(bg);
    }
    box_filter_3d(work, p2_sum, tmp, D, H, W, ks);

    /* T*T -> work, box_filter -> t2_sum */
    {
        WGPUBindGroupEntry e[] = {
            { .binding = 0, .buffer = target, .size = sz },
            { .binding = 1, .buffer = target, .size = sz },
            { .binding = 2, .buffer = work, .size = sz },
            { .binding = 3, .buffer = mpb, .size = sizeof(mp) },
        };
        WGPUBindGroupDescriptor d = { .layout = mul_lay, .entryCount = 4, .entries = e };
        WGPUBindGroup bg = wgpu_create_bind_group(&d, NULL);
        { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
        wgpu_dispatch(mul_pl, bg, wx, wy, 1); }
        wgpu_release_bind_group(bg);
    }
    box_filter_3d(work, t2_sum, tmp, D, H, W, ks);

    /* P*T -> work, box_filter -> tp_sum */
    {
        WGPUBindGroupEntry e[] = {
            { .binding = 0, .buffer = pred, .size = sz },
            { .binding = 1, .buffer = target, .size = sz },
            { .binding = 2, .buffer = work, .size = sz },
            { .binding = 3, .buffer = mpb, .size = sizeof(mp) },
        };
        WGPUBindGroupDescriptor d = { .layout = mul_lay, .entryCount = 4, .entries = e };
        WGPUBindGroup bg = wgpu_create_bind_group(&d, NULL);
        { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
        wgpu_dispatch(mul_pl, bg, wx, wy, 1); }
        wgpu_release_bind_group(bg);
    }
    box_filter_3d(work, tp_sum, tmp, D, H, W, ks);

    /* Box filter pred and target */
    box_filter_3d(pred, p_sum, tmp, D, H, W, ks);
    box_filter_3d(target, t_sum, tmp, D, H, W, ks);

    wgpu_release_buffer(mpb);

    /* Step 2: NCC + gradient source terms — on GPU */
    uint32_t n_groups = wgpu_div_ceil((uint32_t)n, 256);
    size_t partial_sz = (size_t)n_groups * sizeof(float);
    WGPUBuffer ncc_buf = workspace->ncc;
    /* Keep the three gradient source arrays in one binding.  Together with the
     * five filtered inputs and partial-sum output this is seven storage buffers,
     * within WebGPU's baseline maxStorageBuffersPerShaderStage of eight. */
    size_t grad_sources_sz = compute_grad ? 3 * sz : 16;
    WGPUBuffer grad_sources = workspace->grad_sources;

    /* NCC kernel: 5 sums → ncc_out + one packed gradient-source buffer. */
    {
        typedef struct { uint32_t n, cg; float nr, dr; } np_t;
        np_t np = { n, compute_grad, 1e-5f, 1e-5f };
        WGPUBuffer npb = make_params(&np, sizeof(np));

        WGPUComputePipeline pl = wgpu_get_pipeline(
            "ncc_grad", get_shader_source("cc_ncc_grad.wgsl", NULL),
            "ncc_grad");
        if (pl) {
            WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("ncc_grad");
            WGPUBindGroupEntry e[] = {
                { .binding = 0, .buffer = p_sum, .size = sz },
                { .binding = 1, .buffer = t_sum, .size = sz },
                { .binding = 2, .buffer = p2_sum, .size = sz },
                { .binding = 3, .buffer = t2_sum, .size = sz },
                { .binding = 4, .buffer = tp_sum, .size = sz },
                { .binding = 5, .buffer = ncc_buf, .size = partial_sz },
                { .binding = 6, .buffer = grad_sources, .size = grad_sources_sz },
                { .binding = 7, .buffer = npb, .size = sizeof(np) },
            };
            WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 8, .entries = e };
            WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
            { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
            wgpu_dispatch(pl, bg, wx, wy, 1); }
            wgpu_release_bind_group(bg);
        }
        wgpu_release_buffer(npb);
    }

    /* Read only one partial sum per workgroup, not the full volume. */
    if (h_loss_out) {
        float *h_ncc = (float*)malloc(partial_sz);
        if (!h_ncc) {
            wgpu_record_fatal_error("CC host reduction allocation");
            if (owns_workspace) wgpu_cc_workspace_cleanup(workspace);
            return;
        }
        wgpu_read_buffer(ncc_buf, 0, h_ncc, partial_sz);
        if (wgpu_had_fatal_error()) {
            free(h_ncc);
            *h_loss_out = 0.0f;
            if (owns_workspace) wgpu_cc_workspace_cleanup(workspace);
            return;
        }
        double ncc_sum = 0;
        for (uint32_t i = 0; i < n_groups; i++) ncc_sum += h_ncc[i];
        *h_loss_out = -(float)(ncc_sum / n);
        free(h_ncc);
        if (compute_grad || caller_batching) wgpu_begin_batch();
    }

    /* Step 3: Gradient — adjoint box filter + combine on GPU */
    if (compute_grad) {
        WGPUBuffer src_p=workspace->src_p, src_p2=workspace->src_p2;
        WGPUBuffer src_tp=workspace->src_tp, adj_p=workspace->adj_p;
        WGPUBuffer adj_p2=workspace->adj_p2, adj_tp=workspace->adj_tp;

        /* Split the packed shader output for the existing box-filter path.
         * Copies remain in the active command stream, ordered after ncc_grad. */
        wgpu_copy_buffer_range(grad_sources, 0,      src_p,  0, sz);
        wgpu_copy_buffer_range(grad_sources, sz,     src_p2, 0, sz);
        wgpu_copy_buffer_range(grad_sources, 2 * sz, src_tp, 0, sz);

        box_filter_3d(src_p,  adj_p,  tmp, D, H, W, ks);
        box_filter_3d(src_p2, adj_p2, tmp, D, H, W, ks);
        box_filter_3d(src_tp, adj_tp, tmp, D, H, W, ks);

        /* Combine gradient on GPU:
         * grad = -inv_count * (adj_p + 2*P*adj_p2 + T*adj_tp) */
        {
            typedef struct { uint32_t n, _p; float ic, _p2; } cp_t;
            cp_t cp = { n, 0, 1.0f / n, 0 };
            WGPUBuffer cpb = make_params(&cp, sizeof(cp));

            WGPUComputePipeline pl = wgpu_get_pipeline(
                "cc_combine", get_shader_source("cc_combine.wgsl", NULL),
                "combine");
            if (pl) {
                WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("cc_combine");
                WGPUBindGroupEntry e[] = {
                    { .binding = 0, .buffer = adj_p, .size = sz },
                    { .binding = 1, .buffer = adj_p2, .size = sz },
                    { .binding = 2, .buffer = adj_tp, .size = sz },
                    { .binding = 3, .buffer = pred, .size = sz },
                    { .binding = 4, .buffer = target, .size = sz },
                    { .binding = 5, .buffer = grad_pred, .size = sz },
                    { .binding = 6, .buffer = cpb, .size = sizeof(cp) },
                };
                WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 7, .entries = e };
                WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
                { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
                wgpu_dispatch(pl, bg, wx, wy, 1); }
                wgpu_release_bind_group(bg);
            }
            wgpu_release_buffer(cpb);
        }

    }

    if (!caller_batching) wgpu_flush();
    if (owns_workspace) wgpu_cc_workspace_cleanup(workspace);
}

/* ================================================================== */
/* MI Loss — workgroup-local histogram + GPU gradient                  */
/* ================================================================== */

/* MI loss — fully GPU-resident histogram, preparation, and gradient
 *
 * Strategy matching CUDA mi_loss.cu:
 *   1. Find max(pred, target) for normalization
 *   2. GPU histogram: workgroup-local accumulation with fixed-point u32
 *      atomicAdd (fast), then integer atomicAdd merge to global (no CAS)
 *   3. Normalize the histogram and prepare gradient coefficients on GPU
 *   4. Evaluate the gradient on GPU; read back only the scalar loss
 */
#include "cfireants/losses.h"

static const char *get_mi_hist_wgsl(void) {
    static const char *src = NULL;
    if (!src) src = get_shader_source("mi_histogram.wgsl", NULL);
    return src;
}

static const char *get_mi_grad_wgsl(void) {
    static const char *src = NULL;
    if (!src) src = get_shader_source("mi_gradient.wgsl", NULL);
    return src;
}

void wgpu_mi_loss_3d_raw(
    WGPUBuffer pred, WGPUBuffer target,
    WGPUBuffer grad_pred,
    int D, int H, int W,
    int num_bins, float *h_loss_out)
{
    const int n = D * H * W;
    const size_t size = (size_t)n * sizeof(float);
    const int nb = num_bins;

    /* The workgroup arrays and packed layout are specialized for 32 bins. */
    if (nb != 32) {
        fprintf(stderr, "wgpu_mi_loss: GPU path requires num_bins=32 (got %d), "
                "using CPU fallback\n", nb);
        int was_batching = g_wgpu.batch_active;
        int shape[5] = {1, 1, D, H, W};
        tensor_t tp, tt, tg;
        tensor_alloc(&tp, 5, shape, DTYPE_FLOAT32, DEVICE_CPU);
        tensor_alloc(&tt, 5, shape, DTYPE_FLOAT32, DEVICE_CPU);
        wgpu_read_buffer(pred, 0, tp.data, size);
        wgpu_read_buffer(target, 0, tt.data, size);
        tensor_t *gptr = NULL;
        if (grad_pred) {
            tensor_alloc(&tg, 5, shape, DTYPE_FLOAT32, DEVICE_CPU);
            gptr = &tg;
        }
        cpu_mi_loss_3d(&tp, &tt, nb, h_loss_out, gptr);
        if (grad_pred) wgpu_write_buffer(grad_pred, 0, tg.data, size);
        tensor_free(&tp);
        tensor_free(&tt);
        if (gptr) tensor_free(&tg);
        if (was_batching) wgpu_begin_batch();
        return;
    }

    const int caller_batching = g_wgpu.batch_active;
    if (!caller_batching) wgpu_begin_batch();

    const uint32_t n_groups = wgpu_div_ceil((uint32_t)n, 256);
    const size_t joint_count = (size_t)nb * nb;
    const size_t state_count = 1 + joint_count + 2 * nb;
    const size_t coeff_count = joint_count + nb + 1;
    const size_t state_size = state_count * sizeof(uint32_t);
    const size_t coeff_size = coeff_count * sizeof(float);
    WGPUBufferUsage storage_usage =
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc |
        WGPUBufferUsage_CopyDst;
    if (!g_wgpu.mi_state_buf)
        g_wgpu.mi_state_buf =
            wgpu_create_buffer(state_size, storage_usage, "mi_state");
    if (!g_wgpu.mi_coeff_buf)
        g_wgpu.mi_coeff_buf =
            wgpu_create_buffer(coeff_size, storage_usage, "mi_coeff");
    WGPUBuffer state = g_wgpu.mi_state_buf;
    WGPUBuffer coeff = g_wgpu.mi_coeff_buf;

    /* Clear the packed state and reduce max(pred,target) to state[0]. */
    {
        static const char *wgsl = NULL;
        if (!wgsl) wgsl = get_shader_source("mi_max.wgsl", NULL);
        typedef struct { uint32_t n, _p0, _p1, _p2; } params_t;
        params_t p = { (uint32_t)n, 0, 0, 0 };
        WGPUBuffer pb = make_params(&p, sizeof(p));

        WGPUComputePipeline clear_pl =
            wgpu_get_pipeline("mi_clear_state", wgsl, "clear_state");
        if (clear_pl) {
            WGPUBindGroupEntry entry = {
                .binding = 2, .buffer = state, .size = state_size,
            };
            WGPUBindGroupDescriptor desc = {
                .layout = wgpu_get_bind_group_layout("mi_clear_state"),
                .entryCount = 1,
                .entries = &entry,
            };
            WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
            wgpu_dispatch(clear_pl, bg, 1, 1, 1);
            wgpu_release_bind_group(bg);
        }

        WGPUComputePipeline max_pl =
            wgpu_get_pipeline("mi_reduce_max", wgsl, "reduce_max");
        if (max_pl) {
            WGPUBindGroupEntry entries[] = {
                { .binding = 0, .buffer = pred, .size = size },
                { .binding = 1, .buffer = target, .size = size },
                { .binding = 2, .buffer = state, .size = state_size },
                { .binding = 3, .buffer = pb, .size = sizeof(p) },
            };
            WGPUBindGroupDescriptor desc = {
                .layout = wgpu_get_bind_group_layout("mi_reduce_max"),
                .entryCount = 4,
                .entries = entries,
            };
            WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
            uint32_t wx, wy;
            wgpu_dispatch_dims(n_groups, &wx, &wy);
            wgpu_dispatch(max_pl, bg, wx, wy, 1);
            wgpu_release_bind_group(bg);
        }
        wgpu_release_buffer(pb);
    }

    const float bin_spacing = 1.0f / (float)nb;
    const float sigma = bin_spacing * 0.5f;
    const float preterm = 1.0f / (2.0f * sigma * sigma);
    const float nr = 1e-7f;
    const float dr = 1e-7f;

    /* Accumulate the fixed-point histogram directly after the max reduction. */
    {
        float fp_scale =
            (float)((uint64_t)3221225472ULL / (uint64_t)n);
        if (fp_scale > 4096.0f) fp_scale = 4096.0f;
        if (fp_scale < 1.0f) fp_scale = 1.0f;

        typedef struct {
            uint32_t n, num_bins;
            float inv_maxval, preterm, fp_scale;
            uint32_t _p0, _p1, _p2;
        } params_t;
        params_t p = {
            (uint32_t)n, (uint32_t)nb, 0.0f, preterm,
            fp_scale, 0, 0, 0,
        };
        WGPUBuffer pb = make_params(&p, sizeof(p));
        const char *wgsl = get_mi_hist_wgsl();
        WGPUComputePipeline pl =
            wgpu_get_pipeline("mi_hist_local", wgsl, "histogram");
        if (pl) {
            WGPUBindGroupEntry entries[] = {
                { .binding = 0, .buffer = pred, .size = size },
                { .binding = 1, .buffer = target, .size = size },
                { .binding = 2, .buffer = state, .size = state_size },
                { .binding = 3, .buffer = pb, .size = sizeof(p) },
            };
            WGPUBindGroupDescriptor desc = {
                .layout = wgpu_get_bind_group_layout("mi_hist_local"),
                .entryCount = 4,
                .entries = entries,
            };
            WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
            uint32_t wx, wy;
            wgpu_dispatch_dims(n_groups, &wx, &wy);
            wgpu_dispatch(pl, bg, wx, wy, 1);
            wgpu_release_bind_group(bg);
        }
        wgpu_release_buffer(pb);
    }

    /* Normalize the histogram, prepare gradient coefficients, and reduce MI
     * in one fixed-size workgroup. */
    {
        static const char *wgsl = NULL;
        if (!wgsl) wgsl = get_shader_source("mi_prepare.wgsl", NULL);
        typedef struct { float nr, dr; uint32_t _p0, _p1; } params_t;
        params_t p = { nr, dr, 0, 0 };
        WGPUBuffer pb = make_params(&p, sizeof(p));
        WGPUComputePipeline pl =
            wgpu_get_pipeline("mi_prepare", wgsl, "prepare");
        if (pl) {
            WGPUBindGroupEntry entries[] = {
                { .binding = 0, .buffer = state, .size = state_size },
                { .binding = 1, .buffer = coeff, .size = coeff_size },
                { .binding = 2, .buffer = pb, .size = sizeof(p) },
            };
            WGPUBindGroupDescriptor desc = {
                .layout = wgpu_get_bind_group_layout("mi_prepare"),
                .entryCount = 3,
                .entries = entries,
            };
            WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
            wgpu_dispatch(pl, bg, 1, 1, 1);
            wgpu_release_bind_group(bg);
        }
        wgpu_release_buffer(pb);
    }

    if (grad_pred) {
        typedef struct {
            uint32_t n, num_bins;
            float inv_maxval, preterm;
            float inv_n, nr, dr;
            uint32_t _pad;
        } params_t;
        params_t p = {
            (uint32_t)n, (uint32_t)nb, 0.0f, preterm,
            1.0f / (float)n, nr, dr, 0,
        };
        WGPUBuffer pb = make_params(&p, sizeof(p));
        const char *wgsl = get_mi_grad_wgsl();
        WGPUComputePipeline pl =
            wgpu_get_pipeline("mi_grad_v4", wgsl, "mi_gradient");
        if (pl) {
            WGPUBindGroupEntry entries[] = {
                { .binding = 0, .buffer = pred, .size = size },
                { .binding = 1, .buffer = target, .size = size },
                { .binding = 2, .buffer = coeff,
                  .size = coeff_size - sizeof(float) },
                { .binding = 3, .buffer = state, .size = state_size },
                { .binding = 4, .buffer = grad_pred, .size = size },
                { .binding = 5, .buffer = pb, .size = sizeof(p) },
            };
            WGPUBindGroupDescriptor desc = {
                .layout = wgpu_get_bind_group_layout("mi_grad_v4"),
                .entryCount = 6,
                .entries = entries,
            };
            WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
            uint32_t wx, wy;
            wgpu_dispatch_dims(n_groups, &wx, &wy);
            wgpu_dispatch(pl, bg, wx, wy, 1);
            wgpu_release_bind_group(bg);
        }
        wgpu_release_buffer(pb);
    }

    if (h_loss_out) {
        wgpu_read_buffer(coeff, (joint_count + nb) * sizeof(float),
                         h_loss_out, sizeof(float));
        if (caller_batching) wgpu_begin_batch();
    } else if (!caller_batching) {
        wgpu_flush();
    }

}

/* ================================================================== */
/* Gaussian blur — CPU fallback                                        */
/* ================================================================== */

/* Forward declaration from utils.c */
#include "cfireants/utils.h"

void wgpu_gaussian_blur_3d_raw(
    WGPUBuffer inout, int B, int C, int D, int H, int W,
    const float *sigmas, int truncated)
{
    /* CPU fallback: download, blur per-axis on CPU, upload back.
     * sigmas[3] gives per-axis sigma. We call cpu_gaussian_blur_3d
     * which takes a single isotropic sigma, so we approximate by
     * calling it 3 times (once per axis) — or just use an average.
     * For now, use max sigma as isotropic approximation.
     * This is only called via backend_ops_t, not the fused loops. */
    size_t total_sz = (size_t)B * C * D * H * W * sizeof(float);
    int shape[5] = {B, C, D, H, W};

    float max_sigma = sigmas[0];
    if (sigmas[1] > max_sigma) max_sigma = sigmas[1];
    if (sigmas[2] > max_sigma) max_sigma = sigmas[2];

    tensor_t t;
    tensor_alloc(&t, 5, shape, DTYPE_FLOAT32, DEVICE_CPU);
    wgpu_read_buffer(inout, 0, t.data, total_sz);

    cpu_gaussian_blur_3d(&t, &t, max_sigma, (float)truncated);

    wgpu_write_buffer(inout, 0, t.data, total_sz);
    tensor_free(&t);
}

/* ================================================================== */
/* Blur + trilinear downsample (GPU-native, no FFT)                    */
/* ================================================================== */

/* Build 1D Gaussian kernel on CPU, upload to GPU buffer */
static WGPUBuffer build_gauss_kernel_buf(float sigma, float truncated, int *klen_out) {
    if (sigma <= 0) {
        *klen_out = 1;
        float one = 1.0f;
        return wgpu_create_buffer_init(&one, sizeof(float),
            WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "gk1");
    }
    int tail = (int)(truncated * sigma + 0.5f);
    int klen = 2 * tail + 1;
    float *h = (float *)malloc(klen * sizeof(float));
    if (!h) {
        *klen_out = -1;
        wgpu_record_fatal_error("Gaussian-kernel host allocation");
        return NULL;
    }
    float inv = 1.0f / (sigma * sqrtf(2.0f));
    float sum = 0;
    for (int i = 0; i < klen; i++) {
        float x = (float)(i - tail);
        h[i] = 0.5f * (erff((x+0.5f)*inv) - erff((x-0.5f)*inv));
        sum += h[i];
    }
    for (int i = 0; i < klen; i++) h[i] /= sum;
    WGPUBuffer buf = wgpu_create_buffer_init(h, klen * sizeof(float),
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "gk");
    free(h);
    *klen_out = klen;
    return buf;
}

static const char *get_blur_image_wgsl(void) {
    static const char *src = NULL;
    if (!src) src = get_shader_source("blur_image.wgsl", NULL);
    return src;
}

/* GPU separable 3-axis blur on a single [D,H,W] volume */
static void wgpu_blur_volume_gpu(WGPUBuffer data, WGPUBuffer scratch,
                                  int D, int H, int W,
                                  float sigma_d, float sigma_h, float sigma_w) {
    const char *wgsl = get_blur_image_wgsl();
    if (!wgsl) {
        fprintf(stderr, "wgpu_blur_volume_gpu: failed to load blur_image.wgsl\n");
        return;
    }

    int n = D * H * W;
    size_t sz = (size_t)n * sizeof(float);

    typedef struct { uint32_t D, H, W, klen, axis, _p0, _p1, _p2; } blur_p_t;

    float sigmas[3] = { sigma_d, sigma_h, sigma_w };

    for (int axis = 0; axis < 3; axis++) {
        int klen;
        WGPUBuffer kern_buf = build_gauss_kernel_buf(sigmas[axis], 2.0f, &klen);

        WGPUBuffer src_buf = (axis % 2 == 0) ? data : scratch;
        WGPUBuffer dst_buf = (axis % 2 == 0) ? scratch : data;

        blur_p_t p = { (uint32_t)D, (uint32_t)H, (uint32_t)W,
                        (uint32_t)klen, (uint32_t)axis, 0, 0, 0 };
        WGPUBuffer pb = make_params(&p, sizeof(p));

        WGPUComputePipeline pl = wgpu_get_pipeline("blur_image", wgsl, "conv1d_image");
        if (!pl) { wgpu_release_buffer(pb); wgpu_release_buffer(kern_buf); return; }
        WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("blur_image");

        WGPUBindGroupEntry e[] = {
            { .binding = 0, .buffer = src_buf, .size = sz },
            { .binding = 1, .buffer = dst_buf, .size = sz },
            { .binding = 2, .buffer = kern_buf, .size = (size_t)klen * sizeof(float) },
            { .binding = 3, .buffer = pb, .size = sizeof(p) },
        };
        WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 4, .entries = e };
        WGPUBindGroup bg = wgpu_create_bind_group(&desc, NULL);
        { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
        wgpu_dispatch(pl, bg, wx, wy, 1); }
        wgpu_release_bind_group(bg);
        wgpu_release_buffer(pb);
        wgpu_release_buffer(kern_buf);
    }

    /* After 3 passes: if result is in scratch (odd number wouldn't happen with 3 passes,
       but 3 passes leaves result in scratch), copy back to data */
    /* axis 0: data→scratch, axis 1: scratch→data, axis 2: data→scratch
       So result is in scratch. Copy scratch→data. */
    wgpu_copy_buffer(scratch, data, sz);
}

void wgpu_blur_volume(WGPUBuffer data, int D, int H, int W,
                       float sigma_d, float sigma_h, float sigma_w) {
    size_t sz = (size_t)D * H * W * sizeof(float);
    WGPUBufferUsage usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    WGPUBuffer scratch = wgpu_create_buffer(sz, usage, "blur_vol_scr");
    wgpu_blur_volume_gpu(data, scratch, D, H, W, sigma_d, sigma_h, sigma_w);
    wgpu_release_buffer(scratch);
}

void wgpu_blur_downsample(
    WGPUBuffer input, WGPUBuffer output,
    int B, int C, int iD, int iH, int iW, int oD, int oH, int oW)
{
    /* Registration always uses B=1,C=1. Assert rather than silently misbehave. */
    if (B * C != 1) {
        fprintf(stderr, "wgpu_blur_downsample: only B*C=1 supported (got %d*%d)\n", B, C);
        return;
    }

    float sigma_d = 0.5f * (float)iD / (float)oD;
    float sigma_h = 0.5f * (float)iH / (float)oH;
    float sigma_w = 0.5f * (float)iW / (float)oW;

    size_t in_sz = (size_t)iD * iH * iW * sizeof(float);
    WGPUBufferUsage usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;

    /* Copy input → temp buffer, blur in-place, then trilinear resize → output */
    WGPUBuffer blurred = wgpu_create_buffer(in_sz, usage, "blurred");
    wgpu_copy_buffer(input, blurred, in_sz);
    wgpu_blur_volume(blurred, iD, iH, iW, sigma_d, sigma_h, sigma_w);
    wgpuDevicePoll(g_wgpu.device, 1, NULL);  /* Ensure blur completes before resize */
    wgpu_trilinear_resize(blurred, output, B, C, iD, iH, iW, oD, oH, oW, 1);
    wgpuDevicePoll(g_wgpu.device, 1, NULL);  /* Ensure resize completes before release */
    wgpu_release_buffer(blurred);
}

/* ================================================================== */
/* Shared downsample helper (mode-selecting wrapper)                    */
/* ================================================================== */

WGPUBuffer wgpu_downsample_image(WGPUBuffer src, int iD, int iH, int iW,
                                  int oD, int oH, int oW, int mode) {
    size_t out_sz = (size_t)oD * oH * oW * sizeof(float);
    WGPUBufferUsage u = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    WGPUBuffer out = wgpu_create_buffer(out_sz, u, "ds");

    if (mode == DOWNSAMPLE_TRILINEAR) {
        wgpu_blur_downsample(src, out, 1, 1, iD, iH, iW, oD, oH, oW);
    } else {
        /* FFT on CPU: download, FFT, upload */
        size_t in_sz = (size_t)iD * iH * iW * sizeof(float);
        float *h_in = (float *)malloc(in_sz);
        float *h_out = (float *)malloc(out_sz);
        if (!h_in || !h_out) {
            free(h_in); free(h_out);
            fprintf(stderr, "wgpu_downsample_image: malloc failed\n");
            wgpu_release_buffer(out);
            return NULL;
        }
        wgpu_read_buffer(src, 0, h_in, in_sz);
        webgpu_downsample_fft(h_in, h_out, 1, 1, iD, iH, iW, oD, oH, oW);
        wgpu_write_buffer(out, 0, h_out, out_sz);
        free(h_in);
        free(h_out);
    }
    return out;
}
