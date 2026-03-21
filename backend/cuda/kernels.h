/*
 * kernels.h - CUDA kernel declarations for cfireants
 *
 * All kernels operate on raw float pointers and dimension parameters,
 * independent of the tensor_t abstraction. The backend_cuda.cu file
 * provides the tensor_t wrapper functions.
 */

#ifndef CFIREANTS_CUDA_KERNELS_H
#define CFIREANTS_CUDA_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

/* --- Common constants --- */

#define CUDA_BLOCK_SIZE 256
#define MIN_IMG_SIZE 8
#define GAUSS_TRUNCATE 2.0f
#define EVAL_CC_KS 9
#define CC_SMOOTH_NR 1e-5f
#define CC_SMOOTH_DR 1e-5f

/* --- Grid sample 3D (bilinear, zeros padding, align_corners=True) --- */

void cuda_grid_sample_3d_fwd(
    const float *input,   /* [B, C, iD, iH, iW] */
    const float *grid,    /* [B, oD, oH, oW, 3] */
    float *output,        /* [B, C, oD, oH, oW] */
    int B, int C,
    int iD, int iH, int iW,
    int oD, int oH, int oW);

void cuda_grid_sample_3d_bwd(
    const float *grad_output,  /* [B, C, oD, oH, oW] */
    const float *input,        /* [B, C, iD, iH, iW] */
    const float *grid,         /* [B, oD, oH, oW, 3] */
    float *grad_grid,          /* [B, oD, oH, oW, 3] */
    int B, int C,
    int iD, int iH, int iW,
    int oD, int oH, int oW);

/* --- Affine grid 3D (align_corners=True) --- */

void cuda_affine_grid_3d(
    const float *affine,  /* [B, 3, 4] */
    float *grid,          /* [B, D, H, W, 3] */
    int B, int D, int H, int W);

/* --- Element-wise tensor operations --- */

void cuda_tensor_add(float *a, const float *b, int n);         /* a += b */
void cuda_tensor_fill(float *data, float val, int n);
void cuda_tensor_scale(float *data, float alpha, int n);        /* data *= alpha */

/* --- CC loss (rectangular kernel, unsigned, mean reduction) --- */

/* Separable box filter along one axis */
void cuda_box_filter_axis(const float *in, float *out,
                          int D, int H, int W, int ks, int axis,
                          float scale);

/* CC loss forward: compute loss and per-voxel gradient source terms */
void cuda_cc_loss_fwd(
    const float *pred, const float *target,
    int D, int H, int W, int ks,
    float *loss_out,          /* scalar on device */
    float *grad_pred,         /* [D, H, W] or NULL */
    float *p_sum, float *t_sum, float *p2_sum, float *t2_sum, float *tp_sum, /* work buffers */
    float *src_p, float *src_p2, float *src_tp);  /* gradient source terms */

/* --- Gaussian blur 3D (separable, zero-padded) --- */

void cuda_conv1d_axis(const float *in, float *out,
                      int D, int H, int W,
                      const float *kernel, int klen, int axis);

/* --- Adam update --- */

void cuda_adam_step(
    float *param, const float *grad,
    float *exp_avg, float *exp_avg_sq,
    float lr, float beta1, float beta2, float eps,
    int step, int n);

/* Adam moments update only (no param update) */
void cuda_adam_moments_update(
    const float *grad, float *exp_avg, float *exp_avg_sq,
    float beta1, float beta2, int n);

/* Adam direction: output = m_hat / (sqrt(v_hat) + eps) */
void cuda_adam_direction(
    float *output, const float *exp_avg, const float *exp_avg_sq,
    float bc1, float bc2, float eps, int n);

/* --- Downsample (matching Python GPU path) --- */

/* FFT-based downsample matching Python downsample_fft():
 * fftn → fftshift → crop → Gaussian window → ifftshift → ifftn → clamp
 * This is the CUDA fused ops path used by Python when FFO is available. */
void cuda_downsample_fft(
    const float *input, float *output,
    int B, int C,
    int iD, int iH, int iW,
    int oD, int oH, int oW);

/* Fallback: Gaussian blur then trilinear resize (non-FFO Python path) */
void cuda_blur_downsample(
    const float *input, float *output,
    int B, int C,
    int iD, int iH, int iW,
    int oD, int oH, int oW);

/* --- Fused blur for [D,H,W,3] displacement fields --- */

/* In-place separable Gaussian blur on [D,H,W,3] data.
 * scratch: pre-allocated [D*H*W*3] floats.
 * d_kernel: 1D Gaussian kernel on GPU. */
void cuda_blur_disp_dhw3(float *data, float *scratch,
                          int D, int H, int W,
                          const float *d_kernel, int klen);

/* --- Warp inversion (fixed-point iteration) --- */

/* Compute inverse of displacement field: (id+u) ∘ (id+inv_u) ≈ identity
 * u: [D,H,W,3] on GPU, inv_u: [D,H,W,3] on GPU (pre-allocated) */
void cuda_warp_inverse(const float *u, float *inv_u, int D, int H, int W, int n_iters);

/* --- Fused compositive warp update --- */

/* output = update + interp(warp, identity + update)
 * All tensors are [D, H, W, 3]. Output can alias update. */
void cuda_fused_compositive_update(
    const float *warp, const float *update, float *output,
    int D, int H, int W);

/* --- Fused CC loss (pre-allocated workspace, no per-call malloc) --- */

/* Workspace-based fused CC loss.
 * interm: pre-allocated [5*D*H*W] floats on GPU
 * scratch: pre-allocated [D*H*W] floats on GPU */
void cuda_fused_cc_loss(
    const float *pred, const float *target,
    float *grad_pred,          /* [D*H*W] or NULL */
    float *grad_target_out,    /* [D*H*W] or NULL */
    int D, int H, int W, int ks,
    float *h_loss_out,         /* host pointer, or NULL */
    float *interm,             /* workspace: 5*D*H*W */
    float *scratch);           /* workspace: D*H*W */

/* --- MI loss (Gaussian Parzen windowing) --- */

void cuda_mi_loss_3d(
    const float *pred, const float *target,
    float *grad_pred,  /* NULL if no gradient */
    int D, int H, int W,
    int num_bins, float *h_loss_out);

/* --- Trilinear resize --- */

void cuda_trilinear_resize(
    const float *input, float *output,
    int B, int C,
    int iD, int iH, int iW,
    int oD, int oH, int oW,
    int align_corners);

#ifdef __cplusplus
}
#endif

#endif /* CFIREANTS_CUDA_KERNELS_H */
