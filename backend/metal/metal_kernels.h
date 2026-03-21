/*
 * metal_kernels.h - Metal kernel dispatch wrapper declarations
 *
 * C-linkage function declarations for all Metal compute kernel dispatches.
 * Each function prepares buffers/params and calls metal_dispatch().
 *
 * Phase 1: element-wise and reduction ops.
 * Future phases will add grid_sample, losses, blur, Adam, etc.
 */

#ifndef CFIREANTS_METAL_KERNELS_H
#define CFIREANTS_METAL_KERNELS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- Phase 1: Element-wise operations --- */

/* Fill buffer with a constant value */
void metal_tensor_fill(float *data, float value, int n);

/* Scale buffer: data[i] *= alpha */
void metal_tensor_scale(float *data, float alpha, int n);

/* AXPY: y[i] += alpha * x[i] */
void metal_tensor_axpy(float *y, float alpha, const float *x, int n);

/* Sum all elements. Returns scalar result. */
float metal_tensor_sum(const float *data, int n);

/* Mean of all elements. Returns scalar result. */
float metal_tensor_mean(const float *data, int n);

/* --- Phase 2: Grid sampling, affine grid, resize --- */
void metal_grid_sample_3d_fwd(const float *input, const float *grid, float *output,
                               int B, int C, int iD, int iH, int iW,
                               int oD, int oH, int oW);
void metal_grid_sample_3d_bwd(const float *grad_output, const float *input,
                               const float *grid, float *grad_grid,
                               int B, int C, int iD, int iH, int iW,
                               int oD, int oH, int oW);
void metal_affine_grid_3d(const float *affine, float *grid,
                           int B, int D, int H, int W);
void metal_trilinear_resize(const float *input, float *output,
                             int B, int C, int iD, int iH, int iW,
                             int oD, int oH, int oW, int align_corners);
void metal_conv1d_axis(const float *in, float *out,
                        int D, int H, int W,
                        const float *kernel, int klen, int axis);
void metal_box_filter_axis(const float *in, float *out,
                            int D, int H, int W, int ks, int axis, float scale);

/* --- Phase 3: CC loss + Gaussian blur --- */

/* Fused CC loss: workspace-based, matching cuda_fused_cc_loss() */
void metal_fused_cc_loss(
    const float *pred, const float *target,
    float *grad_pred, float *grad_target,
    int D, int H, int W, int ks,
    float *h_loss_out,
    float *interm, float *scratch);

/* Separable Gaussian blur on [D,H,W,3] displacement field (in-place) */
void metal_blur_disp_dhw3(float *data, float *scratch,
                           int D, int H, int W,
                           const float *kernel_data, int klen);

/* CC loss via backend_ops_t (allocates workspace internally) */
void metal_cc_loss_3d(const float *pred, const float *target,
                       float *grad_pred,
                       int D, int H, int W, int ks, float *h_loss_out);

/* --- Phase 4: FFT downsample --- */

/* FFT-based downsampling (CPU on shared memory via kissfft) */
void metal_downsample_fft(const float *input, float *output,
                           int B, int C, int iD, int iH, int iW,
                           int oD, int oH, int oW);

/* --- Future phases (declarations for reference) --- */

/* void metal_tensor_add(float *y, const float *x, int n); */
/* void metal_mi_loss_3d(...); */
/* void metal_adam_step(...); */
/* void metal_affine_grid_bwd(...); */
/* void metal_compose_displacement(...); */
/* void metal_warp_inverse(...); */

#ifdef __cplusplus
}
#endif

#endif /* CFIREANTS_METAL_KERNELS_H */
