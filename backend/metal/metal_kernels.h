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

/* --- Phase 6: MI loss --- */

/* MI loss matching cuda_mi_loss_3d exactly:
 * threadgroup atomic<float> histogram, softmax Parzen weights, proper gradient */
void metal_mi_loss_3d(const float *pred, const float *target,
                       float *grad_pred,
                       int D, int H, int W,
                       int num_bins, float *h_loss_out);

/* --- Phase 7: Deformable registration ops --- */

/* Vector add: out[i] = a[i] + b[i] */
void metal_vec_add(float *out, const float *a, const float *b, int n);

/* Fused compositive warp update:
 * output[x] = update[x] + interp(warp, identity + update[x])
 * warp, update, output are [D,H,W,3] displacement fields */
void metal_fused_compositive_update(const float *warp, const float *update,
                                     float *output, int D, int H, int W);

/* Iterative warp inversion via fixed-point iteration.
 * Computes inv_u such that compose(u, inv_u) ~ identity.
 * n_iters: number of iterations (0 = default 550) */
void metal_warp_inverse(const float *u, float *inv_u, int D, int H, int W, int n_iters);

/* Max L2 norm across [D,H,W,3] displacement field.
 * Returns eps + max(sqrt(dx^2+dy^2+dz^2)) with eps=1e-8 */
float metal_max_l2_norm(const float *data, int spatial);

/* Permute [D,H,W,3] <-> [3,D,H,W] */
void metal_permute_3dhw_dhw3(const float *in, float *out, int D, int H, int W);
void metal_permute_dhw3_3dhw(const float *in, float *out, int D, int H, int W);

#ifdef __cplusplus
}
#endif

#endif /* CFIREANTS_METAL_KERNELS_H */
