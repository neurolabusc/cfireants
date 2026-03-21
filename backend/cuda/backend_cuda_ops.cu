/*
 * backend_cuda_ops.cu - CUDA backend_ops_t implementation
 *
 * Wraps the raw CUDA kernels (in grid_sample.cu) with the tensor_t API
 * defined in backend.h. Also provides cfireants_init_cuda().
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

extern "C" {

#include "cfireants/backend.h"
#include "cfireants/tensor.h"
#include "cfireants/interpolator.h"
#include "cfireants/losses.h"
#include "cfireants/utils.h"
#include "kernels.h"

/* --- Tensor ops --- */

static int gpu_tensor_alloc(tensor_t *t, int ndim, const int *shape, int dtype) {
    return tensor_alloc(t, ndim, shape, dtype, DEVICE_CUDA);
}

static void gpu_tensor_free(tensor_t *t) {
    tensor_free(t);
}

static int gpu_tensor_to_device(tensor_t *dst, const tensor_t *src) {
    return tensor_copy(dst, src); /* handles H2D, D2D */
}

static int gpu_tensor_to_host(tensor_t *dst, const tensor_t *src) {
    return tensor_copy(dst, src); /* handles D2H */
}

static int gpu_tensor_fill(tensor_t *t, float value) {
    cuda_tensor_fill((float*)t->data, value, (int)t->numel);
    return 0;
}

static int gpu_tensor_scale(tensor_t *t, float alpha) {
    cuda_tensor_scale((float*)t->data, alpha, (int)t->numel);
    return 0;
}

static int gpu_tensor_axpy(tensor_t *y, float alpha, const tensor_t *x) {
    /* y += alpha * x — need a fused kernel, for now use scale+add */
    /* TODO: write a proper axpy kernel */
    if (alpha == 1.0f) {
        cuda_tensor_add((float*)y->data, (const float*)x->data, (int)y->numel);
    } else {
        /* Not efficient but works */
        tensor_t tmp;
        tensor_alloc(&tmp, x->ndim, x->shape, x->dtype, DEVICE_CUDA);
        tensor_copy(&tmp, x);
        cuda_tensor_scale((float*)tmp.data, alpha, (int)tmp.numel);
        cuda_tensor_add((float*)y->data, (const float*)tmp.data, (int)y->numel);
        tensor_free(&tmp);
    }
    return 0;
}

static float gpu_tensor_sum(const tensor_t *t) {
    /* Copy to CPU and sum there (small overhead, only used for loss) */
    float *h = (float*)malloc(t->numel * sizeof(float));
    cudaMemcpy(h, t->data, t->numel * sizeof(float), cudaMemcpyDeviceToHost);
    double s = 0;
    for (size_t i = 0; i < t->numel; i++) s += h[i];
    free(h);
    return (float)s;
}

static float gpu_tensor_mean(const tensor_t *t) {
    if (t->numel == 0) return 0;
    return gpu_tensor_sum(t) / (float)t->numel;
}

/* --- Grid sample --- */

static int gpu_grid_sample_3d_fwd(
    const tensor_t *input, const tensor_t *grid, tensor_t *output,
    int mode, int padding, int align_corners)
{
    int B = input->shape[0], C = input->shape[1];
    int iD = input->shape[2], iH = input->shape[3], iW = input->shape[4];
    int oD = grid->shape[1], oH = grid->shape[2], oW = grid->shape[3];

    int shape[5] = {B, C, oD, oH, oW};
    if (tensor_alloc(output, 5, shape, DTYPE_FLOAT32, DEVICE_CUDA) != 0) return -1;

    cuda_grid_sample_3d_fwd(
        (const float*)input->data, (const float*)grid->data, (float*)output->data,
        B, C, iD, iH, iW, oD, oH, oW);
    return 0;
}

static int gpu_grid_sample_3d_bwd(
    const tensor_t *grad_output, const tensor_t *input, const tensor_t *grid,
    tensor_t *grad_input, tensor_t *grad_grid,
    int mode, int padding, int align_corners)
{
    int B = input->shape[0], C = input->shape[1];
    int iD = input->shape[2], iH = input->shape[3], iW = input->shape[4];
    int oD = grid->shape[1], oH = grid->shape[2], oW = grid->shape[3];

    int gg_shape[5] = {B, oD, oH, oW, 3};
    if (tensor_alloc(grad_grid, 5, gg_shape, DTYPE_FLOAT32, DEVICE_CUDA) != 0) return -1;

    cuda_grid_sample_3d_bwd(
        (const float*)grad_output->data, (const float*)input->data,
        (const float*)grid->data, (float*)grad_grid->data,
        B, C, iD, iH, iW, oD, oH, oW);
    return 0;
}

/* --- Stubs for operations that still use CPU for now --- */

static int gpu_stub(void) {
    fprintf(stderr, "cuda backend: operation not yet on GPU, falling back to CPU\n");
    return -1;
}

#define GPU_STUB(name, ...) static int name(__VA_ARGS__) { return gpu_stub(); }

GPU_STUB(gpu_warp_compose_3d_fwd,
    const tensor_t *displacement, const tensor_t *affine, tensor_t *output, int align_corners)
GPU_STUB(gpu_cc_loss_3d,
    const tensor_t *pred, const tensor_t *target, int ks, float *loss_out, tensor_t *grad_pred)
GPU_STUB(gpu_mi_loss_3d,
    const tensor_t *pred, const tensor_t *target, int bins, float *loss_out, tensor_t *grad_pred)
GPU_STUB(gpu_gaussian_blur_3d,
    tensor_t *inout, const float *sigmas, int truncated)

static int gpu_adam_update(
    tensor_t *param, const tensor_t *grad,
    tensor_t *exp_avg, tensor_t *exp_avg_sq,
    float lr, float beta1, float beta2, float eps, int step)
{
    cuda_adam_step(
        (float*)param->data, (const float*)grad->data,
        (float*)exp_avg->data, (float*)exp_avg_sq->data,
        lr, beta1, beta2, eps, step, (int)param->numel);
    return 0;
}

GPU_STUB(gpu_matmul,
    const tensor_t *a, const tensor_t *b, tensor_t *c)

static int gpu_interpolate_3d(
    const tensor_t *input, tensor_t *output, int mode, int align_corners)
{
    int B = input->shape[0], C = input->shape[1];
    int iD = input->shape[2], iH = input->shape[3], iW = input->shape[4];
    int oD = output->shape[2], oH = output->shape[3], oW = output->shape[4];
    cuda_trilinear_resize(
        (const float*)input->data, (float*)output->data,
        B, C, iD, iH, iW, oD, oH, oW, align_corners);
    return 0;
}

/* --- Backend instance --- */

static backend_ops_t cuda_backend = {
    .tensor_alloc     = gpu_tensor_alloc,
    .tensor_free      = gpu_tensor_free,
    .tensor_to_device = gpu_tensor_to_device,
    .tensor_to_host   = gpu_tensor_to_host,
    .tensor_fill      = gpu_tensor_fill,
    .tensor_scale     = gpu_tensor_scale,
    .tensor_axpy      = gpu_tensor_axpy,
    .tensor_sum       = gpu_tensor_sum,
    .tensor_mean      = gpu_tensor_mean,
    .grid_sample_3d_fwd  = gpu_grid_sample_3d_fwd,
    .grid_sample_3d_bwd  = gpu_grid_sample_3d_bwd,
    .warp_compose_3d_fwd = gpu_warp_compose_3d_fwd,
    .cc_loss_3d       = gpu_cc_loss_3d,
    .mi_loss_3d       = gpu_mi_loss_3d,
    .gaussian_blur_3d = gpu_gaussian_blur_3d,
    .adam_update       = gpu_adam_update,
    .matmul           = gpu_matmul,
    .interpolate_3d   = gpu_interpolate_3d,
};

int cfireants_init_cuda(void) {
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count == 0) {
        fprintf(stderr, "cfireants_init_cuda: no CUDA devices found\n");
        return -1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "CUDA backend: %s (%.0f MB)\n", prop.name,
            prop.totalGlobalMem / (1024.0 * 1024.0));
    g_backend = &cuda_backend;
    return 0;
}

} /* extern "C" */
