/*
 * webgpu_kernels.h - WebGPU kernel dispatch functions
 *
 * These are the WebGPU equivalents of the CUDA kernel functions in kernels.h.
 * Called by the registration loops (linear_webgpu.c, greedy_webgpu.c, etc.)
 * and by the backend_ops_t implementations.
 *
 * All functions operate on WGPUBuffer handles directly.
 */

#ifndef CFIREANTS_WEBGPU_KERNELS_H
#define CFIREANTS_WEBGPU_KERNELS_H

#include "webgpu_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --- Grid sampling --- */

void wgpu_grid_sample_3d_fwd(
    WGPUBuffer input, WGPUBuffer grid, WGPUBuffer output,
    int B, int C, int iD, int iH, int iW, int oD, int oH, int oW);

void wgpu_grid_sample_3d_bwd(
    WGPUBuffer grad_output, WGPUBuffer input, WGPUBuffer grid,
    WGPUBuffer grad_grid,
    int B, int C, int iD, int iH, int iW, int oD, int oH, int oW);

/* --- Affine grid generation --- */

void wgpu_affine_grid_3d(WGPUBuffer affine, WGPUBuffer grid,
                          int B, int D, int H, int W);

/* --- Trilinear resize --- */

void wgpu_trilinear_resize(
    WGPUBuffer input, WGPUBuffer output,
    int B, int C, int iD, int iH, int iW,
    int oD, int oH, int oW, int align_corners);

/* --- Element-wise ops --- */

void wgpu_tensor_fill_buf(WGPUBuffer buf, float value, int n);
void wgpu_tensor_scale_buf(WGPUBuffer buf, float alpha, int n);
void wgpu_tensor_add_buf(WGPUBuffer a, WGPUBuffer b, int n);

/* --- Adam optimizer --- */

void wgpu_adam_step(
    WGPUBuffer param, WGPUBuffer grad,
    WGPUBuffer exp_avg, WGPUBuffer exp_avg_sq,
    float lr, float beta1, float beta2, float eps,
    int step, int n);

/* --- CC loss --- */

void wgpu_cc_loss_3d_raw(
    WGPUBuffer pred, WGPUBuffer target,
    WGPUBuffer grad_pred,   /* may be NULL if no gradient needed */
    int D, int H, int W, int ks,
    float *h_loss_out);

/* --- MI loss --- */

void wgpu_mi_loss_3d_raw(
    WGPUBuffer pred, WGPUBuffer target,
    WGPUBuffer grad_pred,
    int D, int H, int W,
    int num_bins, float *h_loss_out);

/* --- FFT downsample (CPU fallback) --- */

void webgpu_downsample_fft(
    const float *input, float *output,
    int B, int C, int iD, int iH, int iW, int oD, int oH, int oW);

/* --- Separable Gaussian blur --- */

void wgpu_gaussian_blur_3d_raw(
    WGPUBuffer inout, int B, int C, int D, int H, int W,
    const float *sigmas, int truncated);

/* --- Compositive warp update --- */

void wgpu_fused_compositive_update(
    WGPUBuffer warp, WGPUBuffer update, WGPUBuffer output,
    int D, int H, int W);

/* --- Blur displacement field [D,H,W,3] --- */

void wgpu_blur_disp_dhw3(WGPUBuffer data, WGPUBuffer scratch,
                          int D, int H, int W,
                          WGPUBuffer kernel, int klen);

/* --- Adam moments + direction (for WarpAdam) --- */

void wgpu_adam_moments_update_buf(WGPUBuffer grad, WGPUBuffer exp_avg,
                                   WGPUBuffer exp_avg_sq,
                                   float beta1, float beta2, int n);

void wgpu_adam_direction_buf(WGPUBuffer output, WGPUBuffer exp_avg,
                              WGPUBuffer exp_avg_sq,
                              float bc1, float bc2, float eps, int n);

/* --- Max L2 norm reduction --- */

float wgpu_max_l2_norm_buf(WGPUBuffer data, int spatial, float eps);

/* --- Affine grid backward --- */

void wgpu_affine_grid_backward(WGPUBuffer grad_grid, int D, int H, int W,
                                float h_dL_dA[12]);

/* --- Warp inverse --- */

void wgpu_warp_inverse(WGPUBuffer u, WGPUBuffer inv_u,
                        int D, int H, int W, int n_iters);

#ifdef __cplusplus
}
#endif

#endif /* CFIREANTS_WEBGPU_KERNELS_H */
