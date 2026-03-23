/*
 * backend_cpu.c - CPU reference backend for cfireants
 *
 * Provides basic CPU implementations of backend operations.
 * GPU-accelerated operations will be added in the CUDA backend.
 */

#include "cfireants/backend.h"

/* Global verbosity: default 2 for backward compat with test programs */
int cfireants_verbose = 2;

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- CPU tensor ops --- */

static int cpu_tensor_alloc(tensor_t *t, int ndim, const int *shape, int dtype) {
    return tensor_alloc(t, ndim, shape, dtype, DEVICE_CPU);
}

static void cpu_tensor_free(tensor_t *t) {
    tensor_free(t);
}

static int cpu_tensor_to_device(tensor_t *dst, const tensor_t *src) {
    /* CPU→CPU copy */
    return tensor_copy(dst, src);
}

static int cpu_tensor_to_host(tensor_t *dst, const tensor_t *src) {
    return tensor_copy(dst, src);
}

static int cpu_tensor_fill(tensor_t *t, float value) {
    if (t->dtype != DTYPE_FLOAT32) return -1;
    tensor_fill_f32(t, value);
    return 0;
}

static int cpu_tensor_scale(tensor_t *t, float alpha) {
    if (t->dtype != DTYPE_FLOAT32) return -1;
    float *data = tensor_data_f32(t);
    for (size_t i = 0; i < t->numel; i++)
        data[i] *= alpha;
    return 0;
}

static int cpu_tensor_axpy(tensor_t *y, float alpha, const tensor_t *x) {
    if (y->dtype != DTYPE_FLOAT32 || x->dtype != DTYPE_FLOAT32) return -1;
    if (y->numel != x->numel) return -1;
    float *yd = tensor_data_f32(y);
    const float *xd = tensor_data_f32(x);
    for (size_t i = 0; i < y->numel; i++)
        yd[i] += alpha * xd[i];
    return 0;
}

static float cpu_tensor_sum(const tensor_t *t) {
    if (t->dtype != DTYPE_FLOAT32) return 0.0f;
    const float *data = tensor_data_f32(t);
    double sum = 0.0;
    for (size_t i = 0; i < t->numel; i++)
        sum += data[i];
    return (float)sum;
}

static float cpu_tensor_mean(const tensor_t *t) {
    if (t->numel == 0) return 0.0f;
    return cpu_tensor_sum(t) / (float)t->numel;
}

/* --- Stubs for operations not yet implemented on CPU --- */

static int cpu_not_implemented(void) {
    fprintf(stderr, "cpu backend: operation not yet implemented\n");
    return -1;
}

/* Use a macro to create stub functions with the right signature */
#define STUB_FN(name, ...) \
    static int name(__VA_ARGS__) { return cpu_not_implemented(); }

STUB_FN(cpu_grid_sample_3d_fwd,
    const tensor_t *input, const tensor_t *grid, tensor_t *output,
    int mode, int padding, int align_corners)

STUB_FN(cpu_grid_sample_3d_bwd,
    const tensor_t *grad_output, const tensor_t *input, const tensor_t *grid,
    tensor_t *grad_input, tensor_t *grad_grid,
    int mode, int padding, int align_corners)

STUB_FN(cpu_warp_compose_3d_fwd,
    const tensor_t *displacement, const tensor_t *affine,
    tensor_t *output, int align_corners)

STUB_FN(cpu_cc_loss_3d,
    const tensor_t *pred, const tensor_t *target, int kernel_size,
    float *loss_out, tensor_t *grad_pred)

STUB_FN(cpu_mi_loss_3d,
    const tensor_t *pred, const tensor_t *target, int num_bins,
    float *loss_out, tensor_t *grad_pred)

STUB_FN(cpu_gaussian_blur_3d,
    tensor_t *inout, const float *sigmas, int truncated)

STUB_FN(cpu_adam_update,
    tensor_t *param, const tensor_t *grad, tensor_t *exp_avg,
    tensor_t *exp_avg_sq, float lr, float beta1, float beta2, float eps, int step)

STUB_FN(cpu_matmul,
    const tensor_t *a, const tensor_t *b, tensor_t *c)

STUB_FN(cpu_interpolate_3d,
    const tensor_t *input, tensor_t *output, int mode, int align_corners)

/* --- Backend instance --- */

static backend_ops_t cpu_backend = {
    .tensor_alloc     = cpu_tensor_alloc,
    .tensor_free      = cpu_tensor_free,
    .tensor_to_device = cpu_tensor_to_device,
    .tensor_to_host   = cpu_tensor_to_host,
    .tensor_fill      = cpu_tensor_fill,
    .tensor_scale     = cpu_tensor_scale,
    .tensor_axpy      = cpu_tensor_axpy,
    .tensor_sum       = cpu_tensor_sum,
    .tensor_mean      = cpu_tensor_mean,
    .grid_sample_3d_fwd  = cpu_grid_sample_3d_fwd,
    .grid_sample_3d_bwd  = cpu_grid_sample_3d_bwd,
    .warp_compose_3d_fwd = cpu_warp_compose_3d_fwd,
    .cc_loss_3d       = cpu_cc_loss_3d,
    .mi_loss_3d       = cpu_mi_loss_3d,
    .gaussian_blur_3d = cpu_gaussian_blur_3d,
    .adam_update       = cpu_adam_update,
    .matmul           = cpu_matmul,
    .interpolate_3d   = cpu_interpolate_3d,
};

/* Global backend pointer */
backend_ops_t *g_backend = NULL;

int cfireants_init_cpu(void) {
    g_backend = &cpu_backend;
    return 0;
}

void cfireants_cleanup(void) {
    g_backend = NULL;
}
