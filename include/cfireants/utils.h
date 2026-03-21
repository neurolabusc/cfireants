/*
 * utils.h - Image processing utilities
 */

#ifndef CFIREANTS_UTILS_H
#define CFIREANTS_UTILS_H

#include "cfireants/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

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
