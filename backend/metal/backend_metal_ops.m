/*
 * backend_metal_ops.m - Metal backend_ops_t implementation
 *
 * Wraps Metal kernel dispatches with the tensor_t API defined in backend.h.
 * Provides cfireants_init_metal() which sets up the Metal context and
 * installs the Metal backend as g_backend.
 *
 * Phase 1: fill, scale, axpy, sum, mean are functional.
 * Other operations return -1 (stub) until their shaders are implemented.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>

#include "cfireants/backend.h"
#include "cfireants/tensor.h"
#include "metal_context.h"
#include "metal_kernels.h"

/* --- Tensor ops --- */

static int mtl_tensor_alloc(tensor_t *t, int ndim, const int *shape, int dtype) {
    return tensor_alloc(t, ndim, shape, dtype, DEVICE_METAL);
}

static void mtl_tensor_free(tensor_t *t) {
    tensor_free(t);
}

static int mtl_tensor_to_device(tensor_t *dst, const tensor_t *src) {
    return tensor_copy(dst, src); /* handles H2D, D2D */
}

static int mtl_tensor_to_host(tensor_t *dst, const tensor_t *src) {
    return tensor_copy(dst, src); /* handles D2H */
}

/* --- Element-wise ops --- */

static int mtl_tensor_fill(tensor_t *t, float value) {
    metal_tensor_fill((float *)t->data, value, (int)t->numel);
    return 0;
}

static int mtl_tensor_scale(tensor_t *t, float alpha) {
    metal_tensor_scale((float *)t->data, alpha, (int)t->numel);
    return 0;
}

static int mtl_tensor_axpy(tensor_t *y, float alpha, const tensor_t *x) {
    metal_tensor_axpy((float *)y->data, alpha, (const float *)x->data, (int)y->numel);
    return 0;
}

/* --- Reductions --- */

static float mtl_tensor_sum(const tensor_t *t) {
    return metal_tensor_sum((const float *)t->data, (int)t->numel);
}

static float mtl_tensor_mean(const tensor_t *t) {
    return metal_tensor_mean((const float *)t->data, (int)t->numel);
}

/* --- Phase 2: Grid sampling, affine grid, resize --- */

static int mtl_grid_sample_3d_fwd(
    const tensor_t *input, const tensor_t *grid, tensor_t *output,
    int mode, int padding, int align_corners)
{
    /* input: [B, C, D, H, W], grid: [B, oD, oH, oW, 3], output: [B, C, oD, oH, oW] */
    int B = input->shape[0], C = input->shape[1];
    int iD = input->shape[2], iH = input->shape[3], iW = input->shape[4];
    int oD = grid->shape[1], oH = grid->shape[2], oW = grid->shape[3];

    metal_grid_sample_3d_fwd((const float *)input->data, (const float *)grid->data,
                              (float *)output->data, B, C, iD, iH, iW, oD, oH, oW);
    return 0;
}

static int mtl_grid_sample_3d_bwd(
    const tensor_t *grad_output, const tensor_t *input, const tensor_t *grid,
    tensor_t *grad_input, tensor_t *grad_grid,
    int mode, int padding, int align_corners)
{
    /* grad_output: [B, C, oD, oH, oW], input: [B, C, D, H, W], grid: [B, oD, oH, oW, 3] */
    int B = input->shape[0], C = input->shape[1];
    int iD = input->shape[2], iH = input->shape[3], iW = input->shape[4];
    int oD = grid->shape[1], oH = grid->shape[2], oW = grid->shape[3];

    metal_grid_sample_3d_bwd((const float *)grad_output->data, (const float *)input->data,
                              (const float *)grid->data, (float *)grad_grid->data,
                              B, C, iD, iH, iW, oD, oH, oW);
    return 0;
}

static int mtl_interpolate_3d(
    const tensor_t *input, tensor_t *output, int mode, int align_corners)
{
    /* input: [B, C, iD, iH, iW], output: [B, C, oD, oH, oW] */
    int B = input->shape[0], C = input->shape[1];
    int iD = input->shape[2], iH = input->shape[3], iW = input->shape[4];
    int oD = output->shape[2], oH = output->shape[3], oW = output->shape[4];

    metal_trilinear_resize((const float *)input->data, (float *)output->data,
                            B, C, iD, iH, iW, oD, oH, oW, align_corners);
    return 0;
}

/* --- Stubs for unimplemented operations --- */

static int mtl_stub(void) {
    fprintf(stderr, "metal backend: operation not yet implemented\n");
    return -1;
}

#define MTL_STUB(name, ...) static int name(__VA_ARGS__) { return mtl_stub(); }

MTL_STUB(mtl_warp_compose_3d_fwd,
    const tensor_t *displacement, const tensor_t *affine, tensor_t *output, int align_corners)
static int mtl_cc_loss_3d(
    const tensor_t *pred, const tensor_t *target, int ks, float *loss_out, tensor_t *grad_pred)
{
    /* pred/target: [B, C, D, H, W] — we handle B=1, C=1 */
    int D = pred->shape[2], H = pred->shape[3], W = pred->shape[4];
    float *gp = grad_pred ? (float *)grad_pred->data : NULL;
    metal_cc_loss_3d((const float *)pred->data, (const float *)target->data,
                      gp, D, H, W, ks, loss_out);
    return 0;
}
static int mtl_mi_loss_3d(const tensor_t *pred, const tensor_t *target,
                           int bins, float *loss_out, tensor_t *grad_pred) {
    int D = pred->shape[2], H = pred->shape[3], W = pred->shape[4];
    float *gp = grad_pred ? (float *)grad_pred->data : NULL;
    metal_mi_loss_3d((const float *)pred->data, (const float *)target->data,
                      gp, D, H, W, bins, loss_out);
    return 0;
}
MTL_STUB(mtl_gaussian_blur_3d,
    tensor_t *inout, const float *sigmas, int truncated)

static int mtl_adam_update(
    tensor_t *param, const tensor_t *grad,
    tensor_t *exp_avg, tensor_t *exp_avg_sq,
    float lr, float beta1, float beta2, float eps, int step)
{
    fprintf(stderr, "metal backend: adam_update not yet implemented\n");
    return -1;
}

MTL_STUB(mtl_matmul,
    const tensor_t *a, const tensor_t *b, tensor_t *c)

/* --- Backend instance --- */

static backend_ops_t metal_backend = {
    .tensor_alloc     = mtl_tensor_alloc,
    .tensor_free      = mtl_tensor_free,
    .tensor_to_device = mtl_tensor_to_device,
    .tensor_to_host   = mtl_tensor_to_host,
    .tensor_fill      = mtl_tensor_fill,
    .tensor_scale     = mtl_tensor_scale,
    .tensor_axpy      = mtl_tensor_axpy,
    .tensor_sum       = mtl_tensor_sum,
    .tensor_mean      = mtl_tensor_mean,
    .grid_sample_3d_fwd  = mtl_grid_sample_3d_fwd,
    .grid_sample_3d_bwd  = mtl_grid_sample_3d_bwd,
    .warp_compose_3d_fwd = mtl_warp_compose_3d_fwd,
    .cc_loss_3d       = mtl_cc_loss_3d,
    .mi_loss_3d       = mtl_mi_loss_3d,
    .gaussian_blur_3d = mtl_gaussian_blur_3d,
    .adam_update       = mtl_adam_update,
    .matmul           = mtl_matmul,
    .interpolate_3d   = mtl_interpolate_3d,
};

int cfireants_init_metal(void) {
    if (metal_context_init() != 0) {
        fprintf(stderr, "cfireants_init_metal: failed to initialize Metal context\n");
        return -1;
    }

    if (cfireants_verbose >= 1) {
        @autoreleasepool {
            fprintf(stderr, "Metal backend: %s (%.0f MB)\n",
                    [[g_metal.device name] UTF8String],
                    [g_metal.device recommendedMaxWorkingSetSize] / (1024.0 * 1024.0));
        }
    }

    g_backend = &metal_backend;
    return 0;
}
