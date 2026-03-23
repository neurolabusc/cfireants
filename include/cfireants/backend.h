/*
 * backend.h - Compute backend dispatch for cfireants
 *
 * All GPU/compute operations go through function pointers in backend_ops_t.
 * At init time, the appropriate backend (CPU, CUDA, or future WebGPU)
 * populates the global g_backend pointer.
 *
 * Registration code calls g_backend->op() and is backend-agnostic.
 */

#ifndef CFIREANTS_BACKEND_H
#define CFIREANTS_BACKEND_H

#include "cfireants/tensor.h"

/* Global verbosity level: 0=silent, 1=summary, 2=per-iteration debug.
 * Set by the CLI tool; registration functions should check before printing. */
extern int cfireants_verbose;

#ifdef __cplusplus
extern "C" {
#endif

/* Interpolation modes */
#define INTERP_NEAREST  0
#define INTERP_BILINEAR 1

/* Padding modes */
#define PAD_ZEROS   0
#define PAD_BORDER  1

typedef struct {
    /* --- Tensor memory management --- */
    int  (*tensor_alloc)(tensor_t *t, int ndim, const int *shape, int dtype);
    void (*tensor_free)(tensor_t *t);

    /* Copy between host and device (or device↔device) */
    int (*tensor_to_device)(tensor_t *dst, const tensor_t *src);
    int (*tensor_to_host)(tensor_t *dst, const tensor_t *src);

    /* --- Element-wise operations --- */
    int (*tensor_fill)(tensor_t *t, float value);
    int (*tensor_scale)(tensor_t *t, float alpha);             /* t *= alpha */
    int (*tensor_axpy)(tensor_t *y, float alpha, const tensor_t *x); /* y += alpha*x */

    /* --- Reductions --- */
    float (*tensor_sum)(const tensor_t *t);
    float (*tensor_mean)(const tensor_t *t);

    /* --- Grid sampling (3D) --- */
    int (*grid_sample_3d_fwd)(
        const tensor_t *input,      /* [B, C, D, H, W] */
        const tensor_t *grid,       /* [B, D, H, W, 3] */
        tensor_t *output,           /* [B, C, D, H, W] */
        int mode, int padding, int align_corners);

    int (*grid_sample_3d_bwd)(
        const tensor_t *grad_output,/* [B, C, D, H, W] */
        const tensor_t *input,      /* [B, C, D, H, W] */
        const tensor_t *grid,       /* [B, D, H, W, 3] */
        tensor_t *grad_input,       /* [B, C, D, H, W] or NULL */
        tensor_t *grad_grid,        /* [B, D, H, W, 3] */
        int mode, int padding, int align_corners);

    /* --- Warp composition (3D): output = sample(displacement, affine*coords + grid) --- */
    int (*warp_compose_3d_fwd)(
        const tensor_t *displacement,/* [B, D, H, W, 3] */
        const tensor_t *affine,      /* [B, 3, 4] */
        tensor_t *output,            /* [B, D, H, W, 3] */
        int align_corners);

    /* --- Losses --- */
    /* CC loss: computes loss value and gradient w.r.t. prediction */
    int (*cc_loss_3d)(
        const tensor_t *pred,       /* [B, C, D, H, W] */
        const tensor_t *target,     /* [B, C, D, H, W] */
        int kernel_size,
        float *loss_out,
        tensor_t *grad_pred);       /* [B, C, D, H, W] or NULL */

    /* MI loss: computes loss value and gradient w.r.t. prediction */
    int (*mi_loss_3d)(
        const tensor_t *pred,
        const tensor_t *target,
        int num_bins,
        float *loss_out,
        tensor_t *grad_pred);

    /* --- Gaussian blur (separable, in-place) --- */
    int (*gaussian_blur_3d)(
        tensor_t *inout,            /* [B, C, D, H, W] */
        const float *sigmas,        /* sigma per spatial dim (3 values) */
        int truncated);             /* truncation in units of sigma */

    /* --- Optimizer --- */
    int (*adam_update)(
        tensor_t *param,
        const tensor_t *grad,
        tensor_t *exp_avg,
        tensor_t *exp_avg_sq,
        float lr, float beta1, float beta2, float eps, int step);

    /* --- Linear algebra --- */
    int (*matmul)(const tensor_t *a, const tensor_t *b, tensor_t *c);

    /* --- Image resampling (trilinear interpolation for multi-scale) --- */
    int (*interpolate_3d)(
        const tensor_t *input,      /* [B, C, D, H, W] */
        tensor_t *output,           /* [B, C, D', H', W'] */
        int mode,                   /* INTERP_* */
        int align_corners);

} backend_ops_t;

/* Global backend (set by cfireants_init_*) */
extern backend_ops_t *g_backend;

/* Initialize backends. Returns 0 on success. */
int cfireants_init_cpu(void);

#ifdef CFIREANTS_HAS_CUDA
int cfireants_init_cuda(void);
#endif

#ifdef CFIREANTS_HAS_WEBGPU
int cfireants_init_webgpu(void);
#endif

#ifdef CFIREANTS_HAS_METAL
int cfireants_init_metal(void);
#endif

/* Cleanup */
void cfireants_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* CFIREANTS_BACKEND_H */
