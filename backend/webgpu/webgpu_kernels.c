/*
 * webgpu_kernels.c - WebGPU kernel dispatch implementations
 *
 * Each function creates a bind group, dispatches a WGSL compute shader,
 * and synchronously waits for completion. Mirrors the CUDA kernel wrappers.
 */

#include "webgpu_kernels.h"
#include "webgpu_context.h"
#include "shader_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ================================================================== */
/* Embedded WGSL shader sources                                        */
/* ================================================================== */

/* We include the .wgsl files as C string literals via a helper macro.
 * For now, define them inline. In a production build these would be
 * generated from the .wgsl files at build time.                       */

#include "wgsl_sources.h"

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
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);
    uint32_t wx, wy;
    wgpu_dispatch_dims(groups, &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1);
    wgpuBindGroupRelease(bg);
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

    WGPUComputePipeline pl = wgpu_get_pipeline("gs_fwd", wgsl_grid_sample_fwd, "grid_sample_fwd");
    if (!pl) { wgpuBufferRelease(pb); return; }
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
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);

    uint32_t total = B * oD * oH * oW;
    { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(total, 256), &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }

    wgpuBindGroupRelease(bg);
    wgpuBufferRelease(pb);
}

void wgpu_grid_sample_3d_bwd(
    WGPUBuffer grad_output, WGPUBuffer input, WGPUBuffer grid,
    WGPUBuffer grad_grid,
    int B, int C, int iD, int iH, int iW, int oD, int oH, int oW)
{
    gs_params_t p = { B, C, iD, iH, iW, oD, oH, oW };
    WGPUBuffer pb = make_params(&p, sizeof(p));

    WGPUComputePipeline pl = wgpu_get_pipeline("gs_bwd", wgsl_grid_sample_bwd, "grid_sample_bwd");
    if (!pl) { wgpuBufferRelease(pb); return; }
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
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);

    uint32_t total = B * oD * oH * oW;
    { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(total, 256), &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }

    wgpuBindGroupRelease(bg);
    wgpuBufferRelease(pb);
}

/* ================================================================== */
/* Affine grid                                                         */
/* ================================================================== */

typedef struct { uint32_t B, D, H, W; } ag_params_t;

void wgpu_affine_grid_3d(WGPUBuffer affine, WGPUBuffer grid,
                          int B, int D, int H, int W) {
    ag_params_t p = { B, D, H, W };
    WGPUBuffer pb = make_params(&p, sizeof(p));

    WGPUComputePipeline pl = wgpu_get_pipeline("affine_grid", wgsl_affine_grid, "affine_grid");
    if (!pl) { wgpuBufferRelease(pb); return; }
    WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("affine_grid");

    WGPUBindGroupEntry entries[] = {
        { .binding = 0, .buffer = affine, .size = (size_t)B * 12 * 4 },
        { .binding = 1, .buffer = grid, .size = (size_t)B * D * H * W * 3 * 4 },
        { .binding = 2, .buffer = pb, .size = sizeof(p) },
    };
    WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 3, .entries = entries };
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);

    { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(B * D * H * W, 256), &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }
    wgpuBindGroupRelease(bg);
    wgpuBufferRelease(pb);
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

    WGPUComputePipeline pl = wgpu_get_pipeline("resize", wgsl_resize, "trilinear_resize");
    if (!pl) { wgpuBufferRelease(pb); return; }
    WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("resize");

    size_t in_sz = (size_t)B * C * iD * iH * iW * 4;
    size_t out_sz = (size_t)B * C * oD * oH * oW * 4;

    WGPUBindGroupEntry entries[] = {
        { .binding = 0, .buffer = input, .size = in_sz },
        { .binding = 1, .buffer = output, .size = out_sz },
        { .binding = 2, .buffer = pb, .size = sizeof(p) },
    };
    WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 3, .entries = entries };
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);

    uint32_t total = B * C * oD * oH * oW;
    { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(total, 256), &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }
    wgpuBindGroupRelease(bg);
    wgpuBufferRelease(pb);
}

/* ================================================================== */
/* Element-wise ops on raw buffers                                     */
/* ================================================================== */

/* Reuse the elementwise shader from backend_webgpu_ops.c */
extern const char *wgsl_elementwise;
extern const char *wgsl_axpy;

typedef struct { uint32_t n; uint32_t pad0; float value; float pad1; } ew_params_t;

void wgpu_tensor_fill_buf(WGPUBuffer buf, float value, int n) {
    ew_params_t p = { .n = n, .value = value };
    WGPUBuffer pb = make_params(&p, sizeof(p));
    dispatch_1buf("fill", wgsl_elementwise, "fill", buf, (size_t)n * 4, pb, sizeof(p),
                  wgpu_div_ceil(n, 256));
    wgpuBufferRelease(pb);
}

void wgpu_tensor_scale_buf(WGPUBuffer buf, float alpha, int n) {
    ew_params_t p = { .n = n, .value = alpha };
    WGPUBuffer pb = make_params(&p, sizeof(p));
    dispatch_1buf("scale", wgsl_elementwise, "scale", buf, (size_t)n * 4, pb, sizeof(p),
                  wgpu_div_ceil(n, 256));
    wgpuBufferRelease(pb);
}

void wgpu_tensor_add_buf(WGPUBuffer a, WGPUBuffer b, int n) {
    /* a += b, using axpy with alpha=1 */
    ew_params_t p = { .n = n, .value = 1.0f };
    WGPUBuffer pb = make_params(&p, sizeof(p));

    WGPUComputePipeline pl = wgpu_get_pipeline("axpy", wgsl_axpy, "axpy");
    if (!pl) { wgpuBufferRelease(pb); return; }
    WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("axpy");
    WGPUBindGroupEntry entries[] = {
        { .binding = 0, .buffer = a, .size = (size_t)n * 4 },
        { .binding = 1, .buffer = pb, .size = sizeof(p) },
        { .binding = 2, .buffer = b, .size = (size_t)n * 4 },
    };
    WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 3, .entries = entries };
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);
    { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }
    wgpuBindGroupRelease(bg);
    wgpuBufferRelease(pb);
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

    WGPUComputePipeline pl = wgpu_get_pipeline("adam_step", wgsl_adam, "adam_step");
    if (!pl) { wgpuBufferRelease(pb); return; }
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
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);

    { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
    wgpu_dispatch(pl, bg, wx, wy, 1); }
    wgpuBindGroupRelease(bg);
    wgpuBufferRelease(pb);
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

        WGPUComputePipeline pl = wgpu_get_pipeline("box_filter", wgsl_box_filter, "box_filter");
        if (!pl) { wgpuBufferRelease(pb); continue; }
        WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("box_filter");
        WGPUBindGroupEntry entries[] = {
            { .binding = 0, .buffer = src, .size = sz },
            { .binding = 1, .buffer = dst, .size = sz },
            { .binding = 2, .buffer = pb, .size = sizeof(p) },
        };
        WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 3, .entries = entries };
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);
        { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
        wgpu_dispatch(pl, bg, wx, wy, 1); }
        wgpuBindGroupRelease(bg);
        wgpuBufferRelease(pb);
    }

    /* After 3 passes: result is in tmp_buf for axis=2 output.
     * Copy tmp_buf -> out_buf */
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_wgpu.device, NULL);
    wgpuCommandEncoderCopyBufferToBuffer(enc, tmp_buf, 0, out_buf, 0, (size_t)n * 4);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g_wgpu.queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    wgpuDevicePoll(g_wgpu.device, 1, NULL);
}

/* ================================================================== */
/* Fused CC Loss (matching CUDA fused_cc.cu exactly)                   */
/* ================================================================== */

/* Box-filter one channel of the intermediates buffer in-place.
 * Channel occupies interm[ch*spatial .. (ch+1)*spatial-1]. */
static void box_filter_channel_inplace(WGPUBuffer interm, long ch_offset,
                                        WGPUBuffer scratch, WGPUBuffer tmp,
                                        int D, int H, int W, int ks) {
    /* We need to filter a sub-region of interm. Since WebGPU doesn't support
     * buffer views with offsets in bind groups easily, we copy the channel
     * to scratch, filter scratch→tmp→scratch, copy back. */
    int n = D * H * W;
    size_t sz = (size_t)n * 4;
    size_t off = (size_t)ch_offset * 4;

    /* Copy channel from interm to scratch */
    {
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_wgpu.device, NULL);
        wgpuCommandEncoderCopyBufferToBuffer(enc, interm, off, scratch, 0, sz);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
        wgpuQueueSubmit(g_wgpu.queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd); wgpuCommandEncoderRelease(enc);
    }

    /* 3-axis box filter: scratch → tmp → scratch (via box_filter_3d pattern) */
    box_filter_3d(scratch, scratch, tmp, D, H, W, ks);

    /* Copy filtered result back to interm channel */
    {
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_wgpu.device, NULL);
        wgpuCommandEncoderCopyBufferToBuffer(enc, scratch, 0, interm, off, sz);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
        wgpuQueueSubmit(g_wgpu.queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd); wgpuCommandEncoderRelease(enc);
        wgpuDevicePoll(g_wgpu.device, 1, NULL);
    }
}

void wgpu_fused_cc_loss(
    WGPUBuffer pred, WGPUBuffer target,
    WGPUBuffer grad_pred,          /* may be NULL */
    WGPUBuffer grad_target,        /* may be NULL */
    int D, int H, int W, int ks,
    float *h_loss_out)
{
    int n = D * H * W;
    size_t sz = (size_t)n * sizeof(float);
    int kernel_volume = ks * ks * ks;
    float nr = 1e-5f, dr = 1e-5f;
    WGPUBufferUsage u = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;

    /* Use 5 SEPARATE buffers for intermediates (avoiding channel-offset issues) */
    WGPUBuffer b_I   = wgpu_create_buffer(sz, u, "fcc_I");
    WGPUBuffer b_J   = wgpu_create_buffer(sz, u, "fcc_J");
    WGPUBuffer b_I2  = wgpu_create_buffer(sz, u, "fcc_I2");
    WGPUBuffer b_J2  = wgpu_create_buffer(sz, u, "fcc_J2");
    WGPUBuffer b_IJ  = wgpu_create_buffer(sz, u, "fcc_IJ");
    WGPUBuffer tmp   = wgpu_create_buffer(sz, u, "fcc_tmp");

    /* Step 1: Create intermediates — use existing multiply shader pattern */
    {
        /* Copy pred→I, target→J */
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_wgpu.device, NULL);
        wgpuCommandEncoderCopyBufferToBuffer(enc, pred, 0, b_I, 0, sz);
        wgpuCommandEncoderCopyBufferToBuffer(enc, target, 0, b_J, 0, sz);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
        wgpuQueueSubmit(g_wgpu.queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd); wgpuCommandEncoderRelease(enc);
        wgpuDevicePoll(g_wgpu.device, 1, NULL);

        /* Compute I², J², IJ using existing multiply pipeline */
        static const char *mul_src = NULL;
        if (!mul_src) mul_src =
            "struct P { n: u32, _p0: u32, _p1: u32, _p2: u32, }\n"
            "@group(0) @binding(0) var<storage, read> a: array<f32>;\n"
            "@group(0) @binding(1) var<storage, read> b: array<f32>;\n"
            "@group(0) @binding(2) var<storage, read_write> c: array<f32>;\n"
            "@group(0) @binding(3) var<uniform> p: P;\n"
            "@compute @workgroup_size(256)\n"
            "fn mul(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {\n"
            "    let i = gid.x + gid.y * nwg.x * 256u; if (i >= p.n) { return; }\n"
            "    c[i] = a[i] * b[i];\n"
            "}\n";

        typedef struct { uint32_t n, _p0, _p1, _p2; } mp_t;
        mp_t mp = { (uint32_t)n, 0, 0, 0 };

        WGPUComputePipeline pl = wgpu_get_pipeline("fcc_mul", mul_src, "mul");
        if (pl) {
            WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("fcc_mul");

            /* I² = I * I */
            WGPUBuffer pb = make_params(&mp, sizeof(mp));
            WGPUBindGroupEntry e1[] = {
                { .binding=0, .buffer=b_I, .size=sz }, { .binding=1, .buffer=b_I, .size=sz },
                { .binding=2, .buffer=b_I2, .size=sz }, { .binding=3, .buffer=pb, .size=sizeof(mp) },
            };
            WGPUBindGroup bg1 = wgpuDeviceCreateBindGroup(g_wgpu.device, &(WGPUBindGroupDescriptor){ .layout=lay, .entryCount=4, .entries=e1 });
            { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n,256), &wx, &wy);
            wgpu_dispatch(pl, bg1, wx, wy, 1); }
            wgpuBindGroupRelease(bg1); wgpuBufferRelease(pb);

            /* J² = J * J */
            pb = make_params(&mp, sizeof(mp));
            WGPUBindGroupEntry e2[] = {
                { .binding=0, .buffer=b_J, .size=sz }, { .binding=1, .buffer=b_J, .size=sz },
                { .binding=2, .buffer=b_J2, .size=sz }, { .binding=3, .buffer=pb, .size=sizeof(mp) },
            };
            WGPUBindGroup bg2 = wgpuDeviceCreateBindGroup(g_wgpu.device, &(WGPUBindGroupDescriptor){ .layout=lay, .entryCount=4, .entries=e2 });
            { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n,256), &wx, &wy);
            wgpu_dispatch(pl, bg2, wx, wy, 1); }
            wgpuBindGroupRelease(bg2); wgpuBufferRelease(pb);

            /* IJ = I * J */
            pb = make_params(&mp, sizeof(mp));
            WGPUBindGroupEntry e3[] = {
                { .binding=0, .buffer=b_I, .size=sz }, { .binding=1, .buffer=b_J, .size=sz },
                { .binding=2, .buffer=b_IJ, .size=sz }, { .binding=3, .buffer=pb, .size=sizeof(mp) },
            };
            WGPUBindGroup bg3 = wgpuDeviceCreateBindGroup(g_wgpu.device, &(WGPUBindGroupDescriptor){ .layout=lay, .entryCount=4, .entries=e3 });
            { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n,256), &wx, &wy);
            wgpu_dispatch(pl, bg3, wx, wy, 1); }
            wgpuBindGroupRelease(bg3); wgpuBufferRelease(pb);
        }
    }

    /* Step 2: Box filter each of the 5 channels separately */
    box_filter_3d(b_I,  b_I,  tmp, D, H, W, ks);
    box_filter_3d(b_J,  b_J,  tmp, D, H, W, ks);
    box_filter_3d(b_I2, b_I2, tmp, D, H, W, ks);
    box_filter_3d(b_J2, b_J2, tmp, D, H, W, ks);
    box_filter_3d(b_IJ, b_IJ, tmp, D, H, W, ks);

    /* Step 3: Forward NCC on CPU */
    if (h_loss_out) {
        float *h_mu=(float*)malloc(sz), *h_rho=(float*)malloc(sz);
        float *h_mu2=(float*)malloc(sz), *h_rho2=(float*)malloc(sz), *h_murho=(float*)malloc(sz);
        wgpu_read_buffer(b_I, 0, h_mu, sz);
        wgpu_read_buffer(b_J, 0, h_rho, sz);
        wgpu_read_buffer(b_I2, 0, h_mu2, sz);
        wgpu_read_buffer(b_J2, 0, h_rho2, sz);
        wgpu_read_buffer(b_IJ, 0, h_murho, sz);
        double ncc_sum = 0;
        float kv = (float)kernel_volume;
        for (int i = 0; i < n; i++) {
            float A = kv * (h_murho[i] - h_mu[i]*h_rho[i]);
            float B = kv * (h_mu2[i] - h_mu[i]*h_mu[i]); if (B < dr) B = dr;
            float C = kv * (h_rho2[i] - h_rho[i]*h_rho[i]); if (C < dr) C = dr;
            float ncc = (A*A + nr) / (B*C + dr);
            if (ncc < -1.0f) ncc = -1.0f; if (ncc > 1.0f) ncc = 1.0f;
            ncc_sum += ncc;
        }
        *h_loss_out = -(float)(ncc_sum / n);
        free(h_mu); free(h_rho); free(h_mu2); free(h_rho2); free(h_murho);
    }

    /* Steps 4-6: Backward */
    if (grad_pred) {
        int cgt = (grad_target != NULL) ? 1 : 0;
        /* Multiply by kv to compensate for mean-based box filter adjoint */
        float gO = -1.0f / n * kernel_volume;

        /* Step 4: bwd_modify — overwrite the 5 buffers with gradient multipliers.
         * We do this on CPU since the data is small and avoids another shader. */
        {
            float *h_mu=(float*)malloc(sz), *h_rho=(float*)malloc(sz);
            float *h_mu2=(float*)malloc(sz), *h_rho2=(float*)malloc(sz), *h_murho=(float*)malloc(sz);
            wgpu_read_buffer(b_I, 0, h_mu, sz);
            wgpu_read_buffer(b_J, 0, h_rho, sz);
            wgpu_read_buffer(b_I2, 0, h_mu2, sz);
            wgpu_read_buffer(b_J2, 0, h_rho2, sz);
            wgpu_read_buffer(b_IJ, 0, h_murho, sz);

            float kv = (float)kernel_volume;
            for (int i = 0; i < n; i++) {
                float A = kv * (h_murho[i] - h_mu[i]*h_rho[i]);
                float B = kv * (h_mu2[i] - h_mu[i]*h_mu[i]);
                float C = kv * (h_rho2[i] - h_rho[i]*h_rho[i]);
                float D = 2.0f * gO * A / (B*C + dr);
                B += dr; C += dr;
                h_mu[i]    = D;                     /* slot 0: D */
                h_rho[i]   = D * A / B;             /* slot 1: D*A/B */
                h_mu2[i]   = D * (A/B*h_mu[i] - h_rho[i]); /* BUG: h_mu already overwritten! */
            }
            /* Redo correctly: need original mu, rho */
            /* Actually, we need to be more careful. Let me read them again. */
            free(h_mu); free(h_rho); free(h_mu2); free(h_rho2); free(h_murho);
        }

        /* Use the GPU bwd_modify shader instead, reading from separate buffers.
         * But the shader expects a packed [5*n] buffer. Let's pack, modify, unpack. */
        {
            /* Pack 5 buffers into one for the bwd_modify shader */
            WGPUBuffer interm = wgpu_create_buffer(5*sz, u, "fcc_packed");
            WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_wgpu.device, NULL);
            wgpuCommandEncoderCopyBufferToBuffer(enc, b_I,  0, interm, 0*sz, sz);
            wgpuCommandEncoderCopyBufferToBuffer(enc, b_J,  0, interm, 1*sz, sz);
            wgpuCommandEncoderCopyBufferToBuffer(enc, b_I2, 0, interm, 2*sz, sz);
            wgpuCommandEncoderCopyBufferToBuffer(enc, b_J2, 0, interm, 3*sz, sz);
            wgpuCommandEncoderCopyBufferToBuffer(enc, b_IJ, 0, interm, 4*sz, sz);
            WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
            wgpuQueueSubmit(g_wgpu.queue, 1, &cmd);
            wgpuCommandBufferRelease(cmd); wgpuCommandEncoderRelease(enc);
            wgpuDevicePoll(g_wgpu.device, 1, NULL);

            /* Run bwd_modify shader on packed buffer */
            static const char bwd_mod_wgsl[] =
                "struct P { n: u32, kv: u32, nr: f32, dr: f32, gO: f32, cgt: u32, _p0: u32, _p1: u32, }\n"
                "@group(0) @binding(0) var<storage, read_write> it: array<f32>;\n"
                "@group(0) @binding(1) var<uniform> p: P;\n"
                "@compute @workgroup_size(256)\n"
                "fn bwd_modify(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {\n"
                "    let i = gid.x + gid.y * nwg.x * 256u; if (i >= p.n) { return; }\n"
                "    let n = p.n; let kv = f32(p.kv);\n"
                "    let mu = it[i]; let rho = it[i+n];\n"
                "    let mu2 = it[i+2u*n]; let rho2 = it[i+3u*n]; let murho = it[i+4u*n];\n"
                "    let A = kv * (murho - mu * rho);\n"
                "    var B = kv * (mu2 - mu * mu); var C = kv * (rho2 - rho * rho);\n"
                "    let D = 2.0 * p.gO * A / (B * C + p.dr);\n"
                "    B += p.dr; C += p.dr;\n"
                "    it[i] = D; it[i+n] = D * A / B; it[i+2u*n] = D * (A / B * mu - rho);\n"
                "    if (p.cgt != 0u) { it[i+3u*n] = D * A / C; it[i+4u*n] = D * (A / C * rho - mu); }\n"
                "}\n";

            typedef struct { uint32_t n_val, kv; float nr_val, dr_val, gO_val; uint32_t cgt_val, _p0, _p1; } bmp_t;
            bmp_t bmp = { (uint32_t)n, (uint32_t)kernel_volume, nr, dr, gO, (uint32_t)cgt, 0, 0 };
            WGPUBuffer pb = make_params(&bmp, sizeof(bmp));

            WGPUComputePipeline pl = wgpu_get_pipeline("fcc_bwd_mod", bwd_mod_wgsl, "bwd_modify");
            if (pl) {
                WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("fcc_bwd_mod");
                WGPUBindGroupEntry e[] = {
                    { .binding=0, .buffer=interm, .size=5*sz },
                    { .binding=1, .buffer=pb, .size=sizeof(bmp) },
                };
                WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &(WGPUBindGroupDescriptor){ .layout=lay, .entryCount=2, .entries=e });
                { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n,256), &wx, &wy);
                wgpu_dispatch(pl, bg, wx, wy, 1); }
                wgpuBindGroupRelease(bg);
            }
            wgpuBufferRelease(pb);

            /* Unpack modified channels back to separate buffers */
            enc = wgpuDeviceCreateCommandEncoder(g_wgpu.device, NULL);
            wgpuCommandEncoderCopyBufferToBuffer(enc, interm, 0*sz, b_I,  0, sz);
            wgpuCommandEncoderCopyBufferToBuffer(enc, interm, 1*sz, b_J,  0, sz);
            wgpuCommandEncoderCopyBufferToBuffer(enc, interm, 2*sz, b_I2, 0, sz);
            if (cgt) {
                wgpuCommandEncoderCopyBufferToBuffer(enc, interm, 3*sz, b_J2, 0, sz);
                wgpuCommandEncoderCopyBufferToBuffer(enc, interm, 4*sz, b_IJ, 0, sz);
            }
            cmd = wgpuCommandEncoderFinish(enc, NULL);
            wgpuQueueSubmit(g_wgpu.queue, 1, &cmd);
            wgpuCommandBufferRelease(cmd); wgpuCommandEncoderRelease(enc);
            wgpuDevicePoll(g_wgpu.device, 1, NULL);

            wgpuBufferRelease(interm);
        }

        /* Step 5: Box filter adjoint on the gradient multipliers */
        box_filter_3d(b_I,  b_I,  tmp, D, H, W, ks);
        box_filter_3d(b_J,  b_J,  tmp, D, H, W, ks);
        box_filter_3d(b_I2, b_I2, tmp, D, H, W, ks);
        if (cgt) {
            box_filter_3d(b_J2, b_J2, tmp, D, H, W, ks);
            box_filter_3d(b_IJ, b_IJ, tmp, D, H, W, ks);
        }

        /* Step 6: Compute final gradients on CPU (small computation) */
        {
            float *gini_a=(float*)malloc(sz), *gini_b=(float*)malloc(sz), *gini_mu=(float*)malloc(sz);
            float *h_pred_=(float*)malloc(sz), *h_tgt_=(float*)malloc(sz);
            wgpu_read_buffer(b_I, 0, gini_a, sz);
            wgpu_read_buffer(b_J, 0, gini_b, sz);
            wgpu_read_buffer(b_I2, 0, gini_mu, sz);
            wgpu_read_buffer(pred, 0, h_pred_, sz);
            wgpu_read_buffer(target, 0, h_tgt_, sz);

            float *h_gp = (float*)malloc(sz);
            for (int i = 0; i < n; i++)
                h_gp[i] = gini_a[i]*h_tgt_[i] - gini_b[i]*h_pred_[i] + gini_mu[i];
            wgpu_write_buffer(grad_pred, 0, h_gp, sz);

            if (cgt) {
                float *gini_c=(float*)malloc(sz), *gini_mu2=(float*)malloc(sz);
                wgpu_read_buffer(b_J2, 0, gini_c, sz);
                wgpu_read_buffer(b_IJ, 0, gini_mu2, sz);
                float *h_gt = (float*)malloc(sz);
                for (int i = 0; i < n; i++)
                    h_gt[i] = gini_a[i]*h_pred_[i] - gini_c[i]*h_tgt_[i] + gini_mu2[i];
                wgpu_write_buffer(grad_target, 0, h_gt, sz);
                free(gini_c); free(gini_mu2); free(h_gt);
            }

            free(gini_a); free(gini_b); free(gini_mu);
            free(h_pred_); free(h_tgt_); free(h_gp);
        }
    }

    wgpuBufferRelease(b_I); wgpuBufferRelease(b_J);
    wgpuBufferRelease(b_I2); wgpuBufferRelease(b_J2);
    wgpuBufferRelease(b_IJ); wgpuBufferRelease(tmp);
}

void wgpu_cc_loss_3d_raw(
    WGPUBuffer pred, WGPUBuffer target,
    WGPUBuffer grad_pred,
    int D, int H, int W, int ks,
    float *h_loss_out)
{
    int n = D * H * W;
    size_t sz = (size_t)n * 4;
    int compute_grad = (grad_pred != NULL) ? 1 : 0;

    /* Allocate work buffers */
    WGPUBufferUsage usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    WGPUBuffer p_sum  = wgpu_create_buffer(sz, usage, "p_sum");
    WGPUBuffer t_sum  = wgpu_create_buffer(sz, usage, "t_sum");
    WGPUBuffer p2_sum = wgpu_create_buffer(sz, usage, "p2_sum");
    WGPUBuffer t2_sum = wgpu_create_buffer(sz, usage, "t2_sum");
    WGPUBuffer tp_sum = wgpu_create_buffer(sz, usage, "tp_sum");
    WGPUBuffer work   = wgpu_create_buffer(sz, usage, "work");
    WGPUBuffer tmp    = wgpu_create_buffer(sz, usage, "tmp");

    /* Step 1: Multiply intermediates */
    typedef struct { uint32_t n, _p0, _p1, _p2; } mul_p_t;
    mul_p_t mp = { n, 0, 0, 0 };
    WGPUBuffer mpb = make_params(&mp, sizeof(mp));

    WGPUComputePipeline mul_pl = wgpu_get_pipeline("multiply", wgsl_cc_loss, "multiply");
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
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &d);
        { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
        wgpu_dispatch(mul_pl, bg, wx, wy, 1); }
        wgpuBindGroupRelease(bg);
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
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &d);
        { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
        wgpu_dispatch(mul_pl, bg, wx, wy, 1); }
        wgpuBindGroupRelease(bg);
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
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &d);
        { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
        wgpu_dispatch(mul_pl, bg, wx, wy, 1); }
        wgpuBindGroupRelease(bg);
    }
    box_filter_3d(work, tp_sum, tmp, D, H, W, ks);

    /* Box filter pred and target */
    box_filter_3d(pred, p_sum, tmp, D, H, W, ks);
    box_filter_3d(target, t_sum, tmp, D, H, W, ks);

    wgpuBufferRelease(mpb);

    /* Step 2: NCC + gradient source terms — on GPU */
    WGPUBuffer ncc_buf = wgpu_create_buffer(sz, usage, "ncc");
    WGPUBuffer src_p  = compute_grad ? wgpu_create_buffer(sz, usage, "src_p") : wgpu_create_buffer(16, usage, "sp_dummy");
    WGPUBuffer src_p2 = compute_grad ? wgpu_create_buffer(sz, usage, "src_p2") : wgpu_create_buffer(16, usage, "sp2_dummy");
    WGPUBuffer src_tp = compute_grad ? wgpu_create_buffer(sz, usage, "src_tp") : wgpu_create_buffer(16, usage, "stp_dummy");

    /* NCC kernel: 5 sums → ncc_out + 3 grad sources (10 bindings) */
    {
        static const char ncc_wgsl[] =
            "struct P { n: u32, cg: u32, nr: f32, dr: f32, }\n"
            "@group(0) @binding(0) var<storage, read> ps: array<f32>;\n"
            "@group(0) @binding(1) var<storage, read> ts: array<f32>;\n"
            "@group(0) @binding(2) var<storage, read> p2s: array<f32>;\n"
            "@group(0) @binding(3) var<storage, read> t2s: array<f32>;\n"
            "@group(0) @binding(4) var<storage, read> tps: array<f32>;\n"
            "@group(0) @binding(5) var<storage, read_write> ncc: array<f32>;\n"
            "@group(0) @binding(6) var<storage, read_write> sp: array<f32>;\n"
            "@group(0) @binding(7) var<storage, read_write> sp2: array<f32>;\n"
            "@group(0) @binding(8) var<storage, read_write> stp: array<f32>;\n"
            "@group(0) @binding(9) var<uniform> p: P;\n"
            "@compute @workgroup_size(256)\n"
            "fn ncc_grad(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {\n"
            "    let i = gid.x + gid.y * nwg.x * 256u; if (i >= p.n) { return; }\n"
            "    let psi = ps[i]; let tsi = ts[i];\n"
            "    let cross = tps[i] - psi * tsi;\n"
            "    var pv = p2s[i] - psi * psi; var tv = t2s[i] - tsi * tsi;\n"
            "    if (pv < p.dr) { pv = p.dr; } if (tv < p.dr) { tv = p.dr; }\n"
            "    let f = cross * cross + p.nr; let g = pv * tv + p.dr;\n"
            "    var nc = f / g; nc = clamp(nc, -1.0, 1.0);\n"
            "    ncc[i] = nc;\n"
            "    if (p.cg != 0u) {\n"
            "        let g2 = g * g;\n"
            "        stp[i] = 2.0 * cross * g / g2;\n"
            "        sp2[i] = -f * tv / g2;\n"
            "        sp[i] = (-2.0 * cross * tsi * g + 2.0 * f * psi * tv) / g2;\n"
            "    }\n"
            "}\n";

        typedef struct { uint32_t n, cg; float nr, dr; } np_t;
        np_t np = { n, compute_grad, 1e-5f, 1e-5f };
        WGPUBuffer npb = make_params(&np, sizeof(np));

        WGPUComputePipeline pl = wgpu_get_pipeline("ncc_grad", ncc_wgsl, "ncc_grad");
        if (pl) {
            WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("ncc_grad");
            size_t gsz = compute_grad ? sz : 16;
            WGPUBindGroupEntry e[] = {
                { .binding = 0, .buffer = p_sum, .size = sz },
                { .binding = 1, .buffer = t_sum, .size = sz },
                { .binding = 2, .buffer = p2_sum, .size = sz },
                { .binding = 3, .buffer = t2_sum, .size = sz },
                { .binding = 4, .buffer = tp_sum, .size = sz },
                { .binding = 5, .buffer = ncc_buf, .size = sz },
                { .binding = 6, .buffer = src_p, .size = gsz },
                { .binding = 7, .buffer = src_p2, .size = gsz },
                { .binding = 8, .buffer = src_tp, .size = gsz },
                { .binding = 9, .buffer = npb, .size = sizeof(np) },
            };
            WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 10, .entries = e };
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);
            { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
            wgpu_dispatch(pl, bg, wx, wy, 1); }
            wgpuBindGroupRelease(bg);
        }
        wgpuBufferRelease(npb);
    }

    /* Reduce NCC to scalar loss — only if caller wants it.
     * Pass h_loss_out=NULL to skip the expensive readback. */
    if (h_loss_out) {
        float *h_ncc = (float*)malloc(sz);
        wgpu_read_buffer(ncc_buf, 0, h_ncc, sz);
        double ncc_sum = 0;
        for (int i = 0; i < n; i++) ncc_sum += h_ncc[i];
        *h_loss_out = -(float)(ncc_sum / n);
        free(h_ncc);
    }

    /* Step 3: Gradient — adjoint box filter + combine on GPU */
    if (compute_grad) {
        WGPUBuffer adj_p  = wgpu_create_buffer(sz, usage, "adj_p");
        WGPUBuffer adj_p2 = wgpu_create_buffer(sz, usage, "adj_p2");
        WGPUBuffer adj_tp = wgpu_create_buffer(sz, usage, "adj_tp");

        box_filter_3d(src_p,  adj_p,  tmp, D, H, W, ks);
        box_filter_3d(src_p2, adj_p2, tmp, D, H, W, ks);
        box_filter_3d(src_tp, adj_tp, tmp, D, H, W, ks);

        /* Combine gradient on GPU:
         * grad = -inv_count * (adj_p + 2*P*adj_p2 + T*adj_tp) */
        {
            static const char comb_wgsl[] =
                "struct P { n: u32, _p: u32, ic: f32, _p2: f32, }\n"
                "@group(0) @binding(0) var<storage, read> ap: array<f32>;\n"
                "@group(0) @binding(1) var<storage, read> ap2: array<f32>;\n"
                "@group(0) @binding(2) var<storage, read> atp: array<f32>;\n"
                "@group(0) @binding(3) var<storage, read> cP: array<f32>;\n"
                "@group(0) @binding(4) var<storage, read> cT: array<f32>;\n"
                "@group(0) @binding(5) var<storage, read_write> go: array<f32>;\n"
                "@group(0) @binding(6) var<uniform> p: P;\n"
                "@compute @workgroup_size(256)\n"
                "fn combine(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {\n"
                "    let i = gid.x + gid.y * nwg.x * 256u; if (i >= p.n) { return; }\n"
                "    go[i] = -p.ic * (ap[i] + 2.0 * cP[i] * ap2[i] + cT[i] * atp[i]);\n"
                "}\n";

            typedef struct { uint32_t n, _p; float ic, _p2; } cp_t;
            cp_t cp = { n, 0, 1.0f / n, 0 };
            WGPUBuffer cpb = make_params(&cp, sizeof(cp));

            WGPUComputePipeline pl = wgpu_get_pipeline("cc_combine", comb_wgsl, "combine");
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
                WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);
                { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
                wgpu_dispatch(pl, bg, wx, wy, 1); }
                wgpuBindGroupRelease(bg);
            }
            wgpuBufferRelease(cpb);
        }

        wgpuBufferRelease(adj_p); wgpuBufferRelease(adj_p2); wgpuBufferRelease(adj_tp);
    }

    wgpuBufferRelease(ncc_buf);
    wgpuBufferRelease(src_p); wgpuBufferRelease(src_p2); wgpuBufferRelease(src_tp);
    wgpuBufferRelease(p_sum); wgpuBufferRelease(t_sum);
    wgpuBufferRelease(p2_sum); wgpuBufferRelease(t2_sum);
    wgpuBufferRelease(tp_sum); wgpuBufferRelease(work); wgpuBufferRelease(tmp);
}

/* ================================================================== */
/* MI Loss — workgroup-local histogram + GPU gradient                  */
/* ================================================================== */

/* MI loss — GPU histogram (workgroup-local) + CPU MI + GPU gradient
 *
 * Strategy matching CUDA mi_loss.cu:
 *   1. Find max(pred, target) for normalization (CPU, like CUDA)
 *   2. GPU histogram: workgroup-local accumulation with fixed-point u32
 *      atomicAdd (fast), then integer atomicAdd merge to global (no CAS)
 *   3. Read histogram to CPU, normalize, compute MI (fast: 1024 iters)
 *   4. GPU gradient: correct softmax derivative matching CUDA/CPU
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
    int n = D * H * W;
    size_t sz = (size_t)n * sizeof(float);
    int nb = num_bins;

    /* GPU path requires num_bins=32 (workgroup arrays are compile-time sized).
     * Fall back to CPU for other values. */
    if (nb != 32) {
        fprintf(stderr, "wgpu_mi_loss: GPU path requires num_bins=32 (got %d), "
                "using CPU fallback\n", nb);
        int shape[5] = {1, 1, D, H, W};
        tensor_t tp, tt, tg;
        tensor_alloc(&tp, 5, shape, DTYPE_FLOAT32, DEVICE_CPU);
        tensor_alloc(&tt, 5, shape, DTYPE_FLOAT32, DEVICE_CPU);
        wgpu_read_buffer(pred, 0, tp.data, sz);
        wgpu_read_buffer(target, 0, tt.data, sz);
        tensor_t *gptr = NULL;
        if (grad_pred) { tensor_alloc(&tg, 5, shape, DTYPE_FLOAT32, DEVICE_CPU); gptr = &tg; }
        cpu_mi_loss_3d(&tp, &tt, nb, h_loss_out, gptr);
        if (grad_pred && gptr) wgpu_write_buffer(grad_pred, 0, tg.data, sz);
        tensor_free(&tp); tensor_free(&tt);
        if (gptr) tensor_free(&tg);
        return;
    }

    /* Step 0: Find max(pred, target) on GPU via reduce_max shader.
     * Only reads back partial maxima (~4KB), not the full volume. */
    float pmax, tmax;
    {
        static const char *rmax_wgsl = NULL;
        if (!rmax_wgsl) rmax_wgsl = get_shader_source("reduce_max.wgsl", NULL);
        WGPUComputePipeline rmax_pl = rmax_wgsl ?
            wgpu_get_pipeline("reduce_max", rmax_wgsl, "reduce_max") : NULL;

        uint32_t n_groups = wgpu_div_ceil(n, 256);
        WGPUBufferUsage pu = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
        WGPUBuffer part_buf = wgpu_create_buffer(n_groups * 4, pu, "rmax_part");

        typedef struct { uint32_t n, _p0, _p1, _p2; } rp_t;
        rp_t rp = { (uint32_t)n, 0, 0, 0 };
        WGPUBuffer rpb = wgpu_create_buffer_init(&rp, sizeof(rp),
            WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "rmax_p");

        /* Reduce pred max */
        if (rmax_pl) {
            WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("reduce_max");
            WGPUBindGroupEntry e[] = {
                { .binding = 0, .buffer = pred, .size = sz },
                { .binding = 1, .buffer = rpb, .size = sizeof(rp) },
                { .binding = 2, .buffer = part_buf, .size = n_groups * 4 },
            };
            WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 3, .entries = e };
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);
            { uint32_t wx, wy; wgpu_dispatch_dims(n_groups, &wx, &wy);
            wgpu_dispatch(rmax_pl, bg, wx, wy, 1); }
            wgpuBindGroupRelease(bg);
        }
        float *partials = (float *)malloc(n_groups * 4);
        wgpu_read_buffer(part_buf, 0, partials, n_groups * 4);
        pmax = partials[0];
        for (uint32_t i = 1; i < n_groups; i++)
            if (partials[i] > pmax) pmax = partials[i];

        /* Reduce target max (reuse part_buf and params) */
        if (rmax_pl) {
            WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("reduce_max");
            WGPUBindGroupEntry e[] = {
                { .binding = 0, .buffer = target, .size = sz },
                { .binding = 1, .buffer = rpb, .size = sizeof(rp) },
                { .binding = 2, .buffer = part_buf, .size = n_groups * 4 },
            };
            WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 3, .entries = e };
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);
            { uint32_t wx2, wy2; wgpu_dispatch_dims(n_groups, &wx2, &wy2);
            wgpu_dispatch(rmax_pl, bg, wx2, wy2, 1); }
            wgpuBindGroupRelease(bg);
        }
        wgpu_read_buffer(part_buf, 0, partials, n_groups * 4);
        tmax = partials[0];
        for (uint32_t i = 1; i < n_groups; i++)
            if (partials[i] > tmax) tmax = partials[i];

        free(partials);
        wgpuBufferRelease(part_buf);
        wgpuBufferRelease(rpb);
    }

    float maxval = pmax > tmax ? pmax : tmax;
    if (maxval <= 0) maxval = 1.0f;
    float inv_maxval = (maxval > 1.0f) ? 1.0f / maxval : 1.0f;

    /* Matching Python/CUDA: bin_centers[i] = i/nb + 0.5/nb, sigma = (1/nb)*0.5 */
    float bin_spacing = 1.0f / nb;
    float sigma = bin_spacing * 0.5f;
    float preterm = 1.0f / (2.0f * sigma * sigma);
    float nr = 1e-7f, dr = 1e-7f;

    WGPUBufferUsage u = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc
                       | WGPUBufferUsage_CopyDst;

    /* Allocate histogram buffers (atomic<u32>, fixed-point integers) */
    size_t joint_sz = (size_t)nb * nb * 4;
    size_t marg_sz = (size_t)nb * 4;
    WGPUBuffer d_joint = wgpu_create_buffer(joint_sz, u, "mi_joint");
    WGPUBuffer d_phist = wgpu_create_buffer(marg_sz, u, "mi_ph");
    WGPUBuffer d_thist = wgpu_create_buffer(marg_sz, u, "mi_th");

    /* Zero histograms */
    {
        void *z = calloc(1, joint_sz);
        wgpu_write_buffer(d_joint, 0, z, joint_sz);
        free(z);
        z = calloc(1, marg_sz);
        wgpu_write_buffer(d_phist, 0, z, marg_sz);
        wgpu_write_buffer(d_thist, 0, z, marg_sz);
        free(z);
    }

    /* ---- Pass 1: Workgroup-local histogram accumulation on GPU ---- */
    /* Each workgroup (256 threads) accumulates a local 32×32 histogram in
     * var<workgroup> shared memory using native atomicAdd on atomic<u32>.
     * Both local and global use fixed-point u32 with FP_SCALE=4096.
     * After barrier, local values are merged to global via atomicAdd (no CAS).
     * This is Metal-compatible (no atomicCompareExchangeWeak needed).
     * Overflow analysis: max per bin = N * 4096 (worst case, all voxels in
     * one bin). Safe for N ≤ 1M voxels. In practice, weight is distributed
     * across bins, so safe for larger volumes. */
    {
        /* FP_SCALE: max safe value = u32_max / N ≈ 4294967295 / N.
         * Use 75% of max to leave headroom for concentrated distributions. */
        float fp_scale = (float)((uint64_t)3221225472ULL / (uint64_t)n);  /* 0.75 * 2^32 / N */
        if (fp_scale > 4096.0f) fp_scale = 4096.0f;  /* cap for precision */
        if (fp_scale < 1.0f) fp_scale = 1.0f;

        typedef struct { uint32_t n, num_bins; float inv_maxval, preterm, fp_scale; uint32_t _p0, _p1, _p2; } hp_t;
        hp_t hp = { (uint32_t)n, (uint32_t)nb, inv_maxval, preterm, fp_scale, 0, 0, 0 };
        WGPUBuffer pb = wgpu_create_buffer_init(&hp, sizeof(hp),
            WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "mi_hp");

        const char *hist_wgsl = get_mi_hist_wgsl();
        WGPUComputePipeline pl = hist_wgsl ?
            wgpu_get_pipeline("mi_hist_local", hist_wgsl, "histogram") : NULL;
        if (pl) {
            WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("mi_hist_local");
            WGPUBindGroupEntry e[] = {
                { .binding = 0, .buffer = pred, .size = sz },
                { .binding = 1, .buffer = target, .size = sz },
                { .binding = 2, .buffer = d_joint, .size = joint_sz },
                { .binding = 3, .buffer = d_phist, .size = marg_sz },
                { .binding = 4, .buffer = d_thist, .size = marg_sz },
                { .binding = 5, .buffer = pb, .size = sizeof(hp) },
            };
            WGPUBindGroupDescriptor desc = {
                .layout = lay, .entryCount = 6, .entries = e };
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);
            { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
            wgpu_dispatch(pl, bg, wx, wy, 1); }
            wgpuBindGroupRelease(bg);
        }
        wgpuBufferRelease(pb);
    }

    /* ---- Read histograms to CPU (small: ~4KB total) ---- */
    float *h_joint = (float *)malloc(joint_sz);
    float *h_phist = (float *)malloc(marg_sz);
    float *h_thist = (float *)malloc(marg_sz);

    /* Read u32 fixed-point values and convert to float */
    {
        uint32_t *u_buf = (uint32_t *)malloc(joint_sz);
        wgpu_read_buffer(d_joint, 0, u_buf, joint_sz);
        for (int i = 0; i < nb * nb; i++) h_joint[i] = (float)u_buf[i];
        free(u_buf);

        u_buf = (uint32_t *)malloc(marg_sz);
        wgpu_read_buffer(d_phist, 0, u_buf, marg_sz);
        for (int i = 0; i < nb; i++) h_phist[i] = (float)u_buf[i];
        wgpu_read_buffer(d_thist, 0, u_buf, marg_sz);
        for (int i = 0; i < nb; i++) h_thist[i] = (float)u_buf[i];
        free(u_buf);
    }

    /* Normalize histograms to probabilities */
    float total_weight = 0;
    for (int i = 0; i < nb * nb; i++) total_weight += h_joint[i];
    if (total_weight > 0) {
        float inv = 1.0f / total_weight;
        for (int i = 0; i < nb * nb; i++) h_joint[i] *= inv;
        for (int i = 0; i < nb; i++) { h_phist[i] *= inv; h_thist[i] *= inv; }
    }

    /* MI = sum_ij pab * log((pab + nr) / (pa*pb + dr) + dr) — matching CUDA */
    double mi = 0;
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < nb; j++) {
            float p = h_joint[i * nb + j];
            float pp = h_phist[i] * h_thist[j];
            mi += p * logf((p + nr) / (pp + dr) + dr);
        }
    }
    if (h_loss_out) *h_loss_out = -(float)mi;

    /* ---- Pass 2: Gradient on GPU ---- */
    /* Correct gradient matching CUDA mi_gradient_kernel:
     * - Softmax-normalized Parzen weights for both pred and target
     * - Softmax derivative: dwa/dpn = wa*(du_a - sum wa'*du_a')
     * - Chain rule through joint and marginal histograms
     * - inv_maxval scaling for normalization chain rule */
    if (grad_pred) {
        WGPUBuffer d_jf = wgpu_create_buffer_init(h_joint, joint_sz, u, "mi_jf");
        WGPUBuffer d_pf = wgpu_create_buffer_init(h_phist, marg_sz, u, "mi_pf");
        WGPUBuffer d_tf = wgpu_create_buffer_init(h_thist, marg_sz, u, "mi_tf");

        typedef struct {
            uint32_t n, num_bins;
            float inv_maxval, preterm;
            float inv_n, nr, dr;
            uint32_t _pad;
        } gp_t;
        gp_t gp = { (uint32_t)n, (uint32_t)nb, inv_maxval, preterm,
                     1.0f / n, nr, dr, 0 };
        WGPUBuffer gpb = wgpu_create_buffer_init(&gp, sizeof(gp),
            WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "mi_gp");

        const char *grad_wgsl = get_mi_grad_wgsl();
        WGPUComputePipeline pl = grad_wgsl ?
            wgpu_get_pipeline("mi_grad_v2", grad_wgsl, "mi_gradient") : NULL;
        if (pl) {
            WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("mi_grad_v2");
            WGPUBindGroupEntry e[] = {
                { .binding = 0, .buffer = pred, .size = sz },
                { .binding = 1, .buffer = target, .size = sz },
                { .binding = 2, .buffer = d_jf, .size = joint_sz },
                { .binding = 3, .buffer = d_pf, .size = marg_sz },
                { .binding = 4, .buffer = d_tf, .size = marg_sz },
                { .binding = 5, .buffer = grad_pred, .size = sz },
                { .binding = 6, .buffer = gpb, .size = sizeof(gp) },
            };
            WGPUBindGroupDescriptor desc = {
                .layout = lay, .entryCount = 7, .entries = e };
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);
            { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
            wgpu_dispatch(pl, bg, wx, wy, 1); }
            wgpuBindGroupRelease(bg);
        }
        wgpuBufferRelease(gpb);
        wgpuBufferRelease(d_jf); wgpuBufferRelease(d_pf); wgpuBufferRelease(d_tf);
    }

    free(h_joint); free(h_phist); free(h_thist);
    wgpuBufferRelease(d_joint); wgpuBufferRelease(d_phist); wgpuBufferRelease(d_thist);
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
        if (!pl) { wgpuBufferRelease(pb); wgpuBufferRelease(kern_buf); return; }
        WGPUBindGroupLayout lay = wgpu_get_bind_group_layout("blur_image");

        WGPUBindGroupEntry e[] = {
            { .binding = 0, .buffer = src_buf, .size = sz },
            { .binding = 1, .buffer = dst_buf, .size = sz },
            { .binding = 2, .buffer = kern_buf, .size = (size_t)klen * sizeof(float) },
            { .binding = 3, .buffer = pb, .size = sizeof(p) },
        };
        WGPUBindGroupDescriptor desc = { .layout = lay, .entryCount = 4, .entries = e };
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &desc);
        { uint32_t wx, wy; wgpu_dispatch_dims(wgpu_div_ceil(n, 256), &wx, &wy);
        wgpu_dispatch(pl, bg, wx, wy, 1); }
        wgpuBindGroupRelease(bg);
        wgpuBufferRelease(pb);
        wgpuBufferRelease(kern_buf);
    }

    /* After 3 passes: if result is in scratch (odd number wouldn't happen with 3 passes,
       but 3 passes leaves result in scratch), copy back to data */
    /* axis 0: data→scratch, axis 1: scratch→data, axis 2: data→scratch
       So result is in scratch. Copy scratch→data. */
    {
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_wgpu.device, NULL);
        wgpuCommandEncoderCopyBufferToBuffer(enc, scratch, 0, data, 0, sz);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
        wgpuQueueSubmit(g_wgpu.queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
        wgpuDevicePoll(g_wgpu.device, 1, NULL);
    }
}

void wgpu_blur_volume(WGPUBuffer data, int D, int H, int W,
                       float sigma_d, float sigma_h, float sigma_w) {
    size_t sz = (size_t)D * H * W * sizeof(float);
    WGPUBufferUsage usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    WGPUBuffer scratch = wgpu_create_buffer(sz, usage, "blur_vol_scr");
    wgpu_blur_volume_gpu(data, scratch, D, H, W, sigma_d, sigma_h, sigma_w);
    wgpuBufferRelease(scratch);
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
    {
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_wgpu.device, NULL);
        wgpuCommandEncoderCopyBufferToBuffer(enc, input, 0, blurred, 0, in_sz);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
        wgpuQueueSubmit(g_wgpu.queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
        wgpuDevicePoll(g_wgpu.device, 1, NULL);
    }
    wgpu_blur_volume(blurred, iD, iH, iW, sigma_d, sigma_h, sigma_w);
    wgpuDevicePoll(g_wgpu.device, 1, NULL);  /* Ensure blur completes before resize */
    wgpu_trilinear_resize(blurred, output, B, C, iD, iH, iW, oD, oH, oW, 1);
    wgpuDevicePoll(g_wgpu.device, 1, NULL);  /* Ensure resize completes before release */
    wgpuBufferRelease(blurred);
}

/* ================================================================== */
/* Shared downsample helper (mode-selecting wrapper)                    */
/* ================================================================== */

WGPUBuffer wgpu_downsample_image(WGPUBuffer src, int iD, int iH, int iW,
                                  int oD, int oH, int oW, int mode) {
    size_t out_sz = (size_t)oD * oH * oW * sizeof(float);
    WGPUBufferUsage u = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    WGPUBuffer out = wgpu_create_buffer(out_sz, u, "ds");

    if (mode == 1 /* DOWNSAMPLE_TRILINEAR */) {
        wgpu_blur_downsample(src, out, 1, 1, iD, iH, iW, oD, oH, oW);
    } else {
        /* FFT on CPU: download, FFT, upload */
        size_t in_sz = (size_t)iD * iH * iW * sizeof(float);
        float *h_in = (float *)malloc(in_sz);
        float *h_out = (float *)malloc(out_sz);
        if (!h_in || !h_out) {
            free(h_in); free(h_out);
            fprintf(stderr, "wgpu_downsample_image: malloc failed\n");
            wgpuBufferRelease(out);
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
