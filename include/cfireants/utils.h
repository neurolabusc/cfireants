/*
 * utils.h - Image processing utilities
 */

#ifndef CFIREANTS_UTILS_H
#define CFIREANTS_UTILS_H

#include "cfireants/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Build 1D Gaussian kernel using erf integration.
 * Caller must free(*kernel_out). */
int make_gaussian_kernel(float sigma, float truncated,
                          float **kernel_out, int *len_out);

/* Separable 3D Gaussian blur (CPU, in-place on a copy).
 *   input:  [B, C, D, H, W]
 *   output: [B, C, D, H, W] (allocated by this function)
 *   sigma:  standard deviation (same for all dims)
 *   truncated: kernel extent in units of sigma (e.g. 4.0)
 * Returns 0 on success. */
int cpu_gaussian_blur_3d(const tensor_t *input, tensor_t *output,
                         float sigma, float truncated);

/* Adam optimizer step (CPU).
 * Updates param in-place: param -= lr * corrected_update
 *   param, grad, exp_avg, exp_avg_sq: same shape, float32
 *   lr, beta1, beta2, eps: Adam hyperparameters
 *   step: 1-based step count (for bias correction)
 * Returns 0 on success. */
int cpu_adam_step(tensor_t *param, const tensor_t *grad,
                 tensor_t *exp_avg, tensor_t *exp_avg_sq,
                 float lr, float beta1, float beta2, float eps, int step);

/* Per-axis Gaussian blur (CPU).
 *   data: [D, H, W] single-channel volume (modified in-place)
 *   sigma_d/h/w: per-axis standard deviations
 *   truncated: kernel extent in units of sigma */
void cpu_blur_volume(float *data, int D, int H, int W,
                      float sigma_d, float sigma_h, float sigma_w,
                      float truncated);

/* Gaussian blur + trilinear resize (CPU, matching GPU blur_downsample).
 *   input:  [B, C, iD, iH, iW] (B*C must be 1)
 *   output: [1, 1, oD, oH, oW] (pre-allocated)
 *   Anti-aliasing: sigma = 0.5 * in_dim / out_dim per axis */
void cpu_blur_downsample(const float *input, float *output,
                          int iD, int iH, int iW,
                          int oD, int oH, int oW);

/* Trilinear interpolation for resizing (CPU).
 *   input:  [B, C, D, H, W]
 *   output: [B, C, D', H', W'] (pre-allocated with target shape)
 *   align_corners: if true, corner pixels are aligned
 * Returns 0 on success. */
int cpu_trilinear_resize(const tensor_t *input, tensor_t *output,
                         int align_corners);

#ifdef __cplusplus
}
#endif

#endif /* CFIREANTS_UTILS_H */
