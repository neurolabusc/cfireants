/*
 * backend_webgpu_ops.c - WebGPU backend_ops_t implementation
 *
 * Wraps WGSL compute shaders with the tensor_t API defined in backend.h.
 * Also provides cfireants_init_webgpu().
 */

#include "webgpu_context.h"
#include "cfireants/backend.h"
#include "cfireants/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Forward declarations from backend_webgpu.c */
int  webgpu_tensor_alloc(tensor_t *t, size_t nbytes);
void webgpu_tensor_free(WGPUBuffer buf);
int  webgpu_memcpy_h2d(WGPUBuffer dst, const void *src, size_t nbytes);
int  webgpu_memcpy_d2h(void *dst, WGPUBuffer src, size_t nbytes);
int  webgpu_memcpy_d2d(WGPUBuffer dst, WGPUBuffer src, size_t nbytes);

/* --- Embedded WGSL shaders --- */

static const char elementwise_wgsl[] =
    "struct Params {\n"
    "    n: u32,\n"
    "    _pad0: u32,\n"
    "    value: f32,\n"
    "    _pad1: f32,\n"
    "}\n"
    "\n"
    "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n"
    "@group(0) @binding(1) var<uniform> params: Params;\n"
    "\n"
    "@compute @workgroup_size(256)\n"
    "fn fill(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
    "    let i = gid.x;\n"
    "    if (i >= params.n) { return; }\n"
    "    data[i] = params.value;\n"
    "}\n"
    "\n"
    "@compute @workgroup_size(256)\n"
    "fn scale(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
    "    let i = gid.x;\n"
    "    if (i >= params.n) { return; }\n"
    "    data[i] = data[i] * params.value;\n"
    "}\n";

/* axpy needs a separate shader since it has different bindings */
static const char axpy_wgsl[] =
    "struct Params {\n"
    "    n: u32,\n"
    "    _pad0: u32,\n"
    "    value: f32,\n"
    "    _pad1: f32,\n"
    "}\n"
    "\n"
    "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n"
    "@group(0) @binding(1) var<uniform> params: Params;\n"
    "@group(0) @binding(2) var<storage, read> x_data: array<f32>;\n"
    "\n"
    "@compute @workgroup_size(256)\n"
    "fn axpy(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
    "    let i = gid.x;\n"
    "    if (i >= params.n) { return; }\n"
    "    data[i] = data[i] + params.value * x_data[i];\n"
    "}\n";

static const char reduction_wgsl[] =
    "struct Params {\n"
    "    n: u32,\n"
    "    _pad0: u32,\n"
    "    _pad1: u32,\n"
    "    _pad2: u32,\n"
    "}\n"
    "\n"
    "@group(0) @binding(0) var<storage, read> input: array<f32>;\n"
    "@group(0) @binding(1) var<uniform> params: Params;\n"
    "@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n"
    "\n"
    "var<workgroup> shared_data: array<f32, 256>;\n"
    "\n"
    "@compute @workgroup_size(256)\n"
    "fn reduce_sum(@builtin(global_invocation_id) gid: vec3<u32>,\n"
    "              @builtin(local_invocation_id) lid: vec3<u32>,\n"
    "              @builtin(workgroup_id) wid: vec3<u32>) {\n"
    "    let i = gid.x;\n"
    "    let tid = lid.x;\n"
    "    if (i < params.n) {\n"
    "        shared_data[tid] = input[i];\n"
    "    } else {\n"
    "        shared_data[tid] = 0.0;\n"
    "    }\n"
    "    workgroupBarrier();\n"
    "    for (var s = 128u; s > 0u; s = s >> 1u) {\n"
    "        if (tid < s) {\n"
    "            shared_data[tid] = shared_data[tid] + shared_data[tid + s];\n"
    "        }\n"
    "        workgroupBarrier();\n"
    "    }\n"
    "    if (tid == 0u) {\n"
    "        output[wid.x] = shared_data[0];\n"
    "    }\n"
    "}\n";

/* --- Helper: create uniform buffer with params --- */

typedef struct { uint32_t n; uint32_t pad0; float value; float pad1; } elem_params_t;
typedef struct { uint32_t n; uint32_t pad0; uint32_t pad1; uint32_t pad2; } reduce_params_t;

static WGPUBuffer create_params_buf(const void *data, size_t size) {
    return wgpu_create_buffer_init(data, size,
        WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "params");
}

/* --- Tensor ops --- */

static int wgpu_tensor_alloc_op(tensor_t *t, int ndim, const int *shape, int dtype) {
    return tensor_alloc(t, ndim, shape, dtype, DEVICE_WEBGPU);
}

static void wgpu_tensor_free_op(tensor_t *t) {
    tensor_free(t);
}

static int wgpu_tensor_to_device(tensor_t *dst, const tensor_t *src) {
    return tensor_copy(dst, src);
}

static int wgpu_tensor_to_host(tensor_t *dst, const tensor_t *src) {
    return tensor_copy(dst, src);
}

static int wgpu_tensor_fill(tensor_t *t, float value) {
    WGPUComputePipeline pipeline = wgpu_get_pipeline("fill", elementwise_wgsl, "fill");
    if (!pipeline) return -1;

    elem_params_t p = { .n = (uint32_t)t->numel, .value = value };
    WGPUBuffer params_buf = create_params_buf(&p, sizeof(p));

    WGPUBindGroupLayout layout = wgpu_get_bind_group_layout("fill");
    WGPUBindGroupEntry entries[] = {
        { .binding = 0, .buffer = (WGPUBuffer)t->data, .size = t->numel * sizeof(float) },
        { .binding = 1, .buffer = params_buf, .size = sizeof(p) },
    };
    WGPUBindGroupDescriptor bg_desc = {
        .layout = layout,
        .entryCount = 2,
        .entries = entries,
    };
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &bg_desc);

    wgpu_dispatch(pipeline, bg, wgpu_div_ceil((uint32_t)t->numel, WGPU_WORKGROUP_SIZE), 1, 1);

    wgpuBindGroupRelease(bg);
    wgpuBufferRelease(params_buf);
    return 0;
}

static int wgpu_tensor_scale(tensor_t *t, float alpha) {
    WGPUComputePipeline pipeline = wgpu_get_pipeline("scale", elementwise_wgsl, "scale");
    if (!pipeline) return -1;

    elem_params_t p = { .n = (uint32_t)t->numel, .value = alpha };
    WGPUBuffer params_buf = create_params_buf(&p, sizeof(p));

    WGPUBindGroupLayout layout = wgpu_get_bind_group_layout("scale");
    WGPUBindGroupEntry entries[] = {
        { .binding = 0, .buffer = (WGPUBuffer)t->data, .size = t->numel * sizeof(float) },
        { .binding = 1, .buffer = params_buf, .size = sizeof(p) },
    };
    WGPUBindGroupDescriptor bg_desc = {
        .layout = layout,
        .entryCount = 2,
        .entries = entries,
    };
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &bg_desc);

    wgpu_dispatch(pipeline, bg, wgpu_div_ceil((uint32_t)t->numel, WGPU_WORKGROUP_SIZE), 1, 1);

    wgpuBindGroupRelease(bg);
    wgpuBufferRelease(params_buf);
    return 0;
}

static int wgpu_tensor_axpy(tensor_t *y, float alpha, const tensor_t *x) {
    WGPUComputePipeline pipeline = wgpu_get_pipeline("axpy", axpy_wgsl, "axpy");
    if (!pipeline) return -1;

    elem_params_t p = { .n = (uint32_t)y->numel, .value = alpha };
    WGPUBuffer params_buf = create_params_buf(&p, sizeof(p));

    WGPUBindGroupLayout layout = wgpu_get_bind_group_layout("axpy");
    WGPUBindGroupEntry entries[] = {
        { .binding = 0, .buffer = (WGPUBuffer)y->data, .size = y->numel * sizeof(float) },
        { .binding = 1, .buffer = params_buf, .size = sizeof(p) },
        { .binding = 2, .buffer = (WGPUBuffer)x->data, .size = x->numel * sizeof(float) },
    };
    WGPUBindGroupDescriptor bg_desc = {
        .layout = layout,
        .entryCount = 3,
        .entries = entries,
    };
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &bg_desc);

    wgpu_dispatch(pipeline, bg, wgpu_div_ceil((uint32_t)y->numel, WGPU_WORKGROUP_SIZE), 1, 1);

    wgpuBindGroupRelease(bg);
    wgpuBufferRelease(params_buf);
    return 0;
}

static float wgpu_tensor_sum(const tensor_t *t) {
    /* Single-pass reduction: each workgroup produces a partial sum.
       Final reduction on CPU (matching CUDA pattern). */
    uint32_t n = (uint32_t)t->numel;
    uint32_t n_groups = wgpu_div_ceil(n, WGPU_WORKGROUP_SIZE);

    WGPUComputePipeline pipeline = wgpu_get_pipeline("reduce_sum", reduction_wgsl, "reduce_sum");
    if (!pipeline) {
        /* Fallback: copy to CPU and sum there */
        float *h = (float *)malloc(n * sizeof(float));
        wgpu_read_buffer((WGPUBuffer)t->data, 0, h, n * sizeof(float));
        double s = 0;
        for (uint32_t i = 0; i < n; i++) s += h[i];
        free(h);
        return (float)s;
    }

    reduce_params_t p = { .n = n };
    WGPUBuffer params_buf = create_params_buf(&p, sizeof(p));
    WGPUBuffer out_buf = wgpu_create_buffer(n_groups * sizeof(float),
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc, "reduce_out");

    WGPUBindGroupLayout layout = wgpu_get_bind_group_layout("reduce_sum");
    WGPUBindGroupEntry entries[] = {
        { .binding = 0, .buffer = (WGPUBuffer)t->data, .size = n * sizeof(float) },
        { .binding = 1, .buffer = params_buf, .size = sizeof(p) },
        { .binding = 2, .buffer = out_buf, .size = n_groups * sizeof(float) },
    };
    WGPUBindGroupDescriptor bg_desc = {
        .layout = layout,
        .entryCount = 3,
        .entries = entries,
    };
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_wgpu.device, &bg_desc);

    wgpu_dispatch(pipeline, bg, n_groups, 1, 1);

    /* Read partials and sum on CPU */
    float *partials = (float *)malloc(n_groups * sizeof(float));
    wgpu_read_buffer(out_buf, 0, partials, n_groups * sizeof(float));
    double s = 0;
    for (uint32_t i = 0; i < n_groups; i++) s += partials[i];

    free(partials);
    wgpuBindGroupRelease(bg);
    wgpuBufferRelease(params_buf);
    wgpuBufferRelease(out_buf);
    return (float)s;
}

static float wgpu_tensor_mean(const tensor_t *t) {
    if (t->numel == 0) return 0;
    return wgpu_tensor_sum(t) / (float)t->numel;
}

/* --- Implementations using webgpu_kernels --- */

#include "webgpu_kernels.h"

static int wgpu_grid_sample_3d_fwd_op(
    const tensor_t *input, const tensor_t *grid, tensor_t *output,
    int mode, int padding, int align_corners)
{
    (void)mode; (void)padding; (void)align_corners;
    int B = input->shape[0], C = input->shape[1];
    int iD = input->shape[2], iH = input->shape[3], iW = input->shape[4];
    int oD = grid->shape[1], oH = grid->shape[2], oW = grid->shape[3];

    int shape[5] = {B, C, oD, oH, oW};
    if (tensor_alloc(output, 5, shape, DTYPE_FLOAT32, DEVICE_WEBGPU) != 0) return -1;

    wgpu_grid_sample_3d_fwd(
        (WGPUBuffer)input->data, (WGPUBuffer)grid->data, (WGPUBuffer)output->data,
        B, C, iD, iH, iW, oD, oH, oW);
    return 0;
}

static int wgpu_grid_sample_3d_bwd_op(
    const tensor_t *grad_output, const tensor_t *input, const tensor_t *grid,
    tensor_t *grad_input, tensor_t *grad_grid,
    int mode, int padding, int align_corners)
{
    (void)grad_input; (void)mode; (void)padding; (void)align_corners;
    int B = input->shape[0], C = input->shape[1];
    int iD = input->shape[2], iH = input->shape[3], iW = input->shape[4];
    int oD = grid->shape[1], oH = grid->shape[2], oW = grid->shape[3];

    int gg_shape[5] = {B, oD, oH, oW, 3};
    if (tensor_alloc(grad_grid, 5, gg_shape, DTYPE_FLOAT32, DEVICE_WEBGPU) != 0) return -1;

    wgpu_grid_sample_3d_bwd(
        (WGPUBuffer)grad_output->data, (WGPUBuffer)input->data,
        (WGPUBuffer)grid->data, (WGPUBuffer)grad_grid->data,
        B, C, iD, iH, iW, oD, oH, oW);
    return 0;
}

static int wgpu_cc_loss_3d_op(
    const tensor_t *pred, const tensor_t *target, int ks,
    float *loss_out, tensor_t *grad_pred)
{
    int D = pred->shape[2], H = pred->shape[3], W = pred->shape[4];
    WGPUBuffer grad_buf = NULL;
    if (grad_pred) {
        int shape[5] = {1, 1, D, H, W};
        if (tensor_alloc(grad_pred, 5, shape, DTYPE_FLOAT32, DEVICE_WEBGPU) != 0) return -1;
        grad_buf = (WGPUBuffer)grad_pred->data;
    }
    wgpu_cc_loss_3d_raw(
        (WGPUBuffer)pred->data, (WGPUBuffer)target->data,
        grad_buf, D, H, W, ks, loss_out);
    return 0;
}

static int wgpu_mi_loss_3d_op(
    const tensor_t *pred, const tensor_t *target, int bins,
    float *loss_out, tensor_t *grad_pred)
{
    int D = pred->shape[2], H = pred->shape[3], W = pred->shape[4];
    WGPUBuffer grad_buf = NULL;
    if (grad_pred) {
        int shape[5] = {1, 1, D, H, W};
        if (tensor_alloc(grad_pred, 5, shape, DTYPE_FLOAT32, DEVICE_WEBGPU) != 0) return -1;
        grad_buf = (WGPUBuffer)grad_pred->data;
    }
    wgpu_mi_loss_3d_raw(
        (WGPUBuffer)pred->data, (WGPUBuffer)target->data,
        grad_buf, D, H, W, bins, loss_out);
    return 0;
}

static int wgpu_gaussian_blur_3d_op(
    tensor_t *inout, const float *sigmas, int truncated)
{
    int B = inout->shape[0], C = inout->shape[1];
    int D = inout->shape[2], H = inout->shape[3], W = inout->shape[4];
    wgpu_gaussian_blur_3d_raw((WGPUBuffer)inout->data, B, C, D, H, W, sigmas, truncated);
    return 0;
}

static int wgpu_adam_update_op(
    tensor_t *param, const tensor_t *grad,
    tensor_t *exp_avg, tensor_t *exp_avg_sq,
    float lr, float beta1, float beta2, float eps, int step)
{
    wgpu_adam_step(
        (WGPUBuffer)param->data, (WGPUBuffer)grad->data,
        (WGPUBuffer)exp_avg->data, (WGPUBuffer)exp_avg_sq->data,
        lr, beta1, beta2, eps, step, (int)param->numel);
    return 0;
}

static int wgpu_interpolate_3d_op(
    const tensor_t *input, tensor_t *output, int mode, int align_corners)
{
    (void)mode;
    int B = input->shape[0], C = input->shape[1];
    int iD = input->shape[2], iH = input->shape[3], iW = input->shape[4];
    int oD = output->shape[2], oH = output->shape[3], oW = output->shape[4];
    wgpu_trilinear_resize(
        (WGPUBuffer)input->data, (WGPUBuffer)output->data,
        B, C, iD, iH, iW, oD, oH, oW, align_corners);
    return 0;
}

static int wgpu_stub(void) {
    fprintf(stderr, "webgpu backend: operation not yet implemented\n");
    return -1;
}

#define WGPU_STUB(name, ...) static int name(__VA_ARGS__) { return wgpu_stub(); }

WGPU_STUB(wgpu_warp_compose_3d_fwd,
    const tensor_t *displacement, const tensor_t *affine, tensor_t *output, int align_corners)
WGPU_STUB(wgpu_matmul,
    const tensor_t *a, const tensor_t *b, tensor_t *c)

/* --- Backend instance --- */

static backend_ops_t webgpu_backend = {
    .tensor_alloc     = wgpu_tensor_alloc_op,
    .tensor_free      = wgpu_tensor_free_op,
    .tensor_to_device = wgpu_tensor_to_device,
    .tensor_to_host   = wgpu_tensor_to_host,
    .tensor_fill      = wgpu_tensor_fill,
    .tensor_scale     = wgpu_tensor_scale,
    .tensor_axpy      = wgpu_tensor_axpy,
    .tensor_sum       = wgpu_tensor_sum,
    .tensor_mean      = wgpu_tensor_mean,
    .grid_sample_3d_fwd  = wgpu_grid_sample_3d_fwd_op,
    .grid_sample_3d_bwd  = wgpu_grid_sample_3d_bwd_op,
    .warp_compose_3d_fwd = wgpu_warp_compose_3d_fwd,
    .cc_loss_3d       = wgpu_cc_loss_3d_op,
    .mi_loss_3d       = wgpu_mi_loss_3d_op,
    .gaussian_blur_3d = wgpu_gaussian_blur_3d_op,
    .adam_update       = wgpu_adam_update_op,
    .matmul           = wgpu_matmul,
    .interpolate_3d   = wgpu_interpolate_3d_op,
};

int cfireants_init_webgpu(void) {
    if (wgpu_context_init() != 0) {
        fprintf(stderr, "cfireants_init_webgpu: failed to initialize WebGPU\n");
        return -1;
    }
    g_backend = &webgpu_backend;
    return 0;
}
