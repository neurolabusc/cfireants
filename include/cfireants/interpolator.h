/*
 * interpolator.h - Grid sampling and affine grid generation
 */

#ifndef CFIREANTS_INTERPOLATOR_H
#define CFIREANTS_INTERPOLATOR_H

#include "cfireants/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Generate a sampling grid from an affine matrix (align_corners=True).
 *   affine: [B, 3, 4]
 *   output: [B, D, H, W, 3] with coordinates in [-1, 1]
 * The output shape (D, H, W) is taken from out_shape[]. */
int affine_grid_3d(const tensor_t *affine, const int out_shape[3],
                   tensor_t *output);

/* 3D bilinear grid sampling (CPU reference implementation).
 *   input:  [B, C, D, H, W]
 *   grid:   [B, D_out, H_out, W_out, 3]  (coordinates in [-1, 1])
 *   output: [B, C, D_out, H_out, W_out]
 *   align_corners: if true, -1 maps to 0 and +1 maps to size-1 */
int cpu_grid_sample_3d_forward(const tensor_t *input, const tensor_t *grid,
                               tensor_t *output, int align_corners);

/* Backward pass: compute gradient of loss w.r.t. grid coordinates.
 *   grad_output: [B, C, D, H, W]  (dL/d(output))
 *   input:       [B, C, D, H, W]  (original input)
 *   grid:        [B, D, H, W, 3]  (sampling coordinates)
 *   grad_grid:   [B, D, H, W, 3]  (output: dL/d(grid))
 *   align_corners: matching forward pass */
int cpu_grid_sample_3d_backward(const tensor_t *grad_output,
                                const tensor_t *input,
                                const tensor_t *grid,
                                tensor_t *grad_grid,
                                int align_corners);

#ifdef __cplusplus
}
#endif

#endif /* CFIREANTS_INTERPOLATOR_H */
