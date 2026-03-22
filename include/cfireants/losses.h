/*
 * losses.h - Similarity metrics for image registration
 */

#ifndef CFIREANTS_LOSSES_H
#define CFIREANTS_LOSSES_H

#include "cfireants/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Local normalized cross-correlation loss (CPU).
 * Matches fireants LocalNormalizedCrossCorrelationLoss with
 * kernel_type='rectangular', unsigned=True, reduction='mean'.
 *
 *   pred:      [B, C, D, H, W]
 *   target:    [B, C, D, H, W]
 *   kernel_size: odd integer (e.g., 3, 5, 7)
 *   loss_out:  scalar output (negative NCC, lower = better)
 *   grad_pred: [B, C, D, H, W] gradient w.r.t. pred (or NULL to skip)
 *
 * Returns 0 on success. */
int cpu_cc_loss_3d(const tensor_t *pred, const tensor_t *target,
                   int kernel_size, float *loss_out, tensor_t *grad_pred);

/* CC loss with both pred AND target gradients (for SyN) */
int cpu_cc_loss_3d_both(const tensor_t *pred, const tensor_t *target,
                         int kernel_size, float *loss_out,
                         tensor_t *grad_pred, tensor_t *grad_target);

/* Fused CC loss matching CUDA fused_cc.cu / Metal fcc_* exactly.
 * Uses kernel_volume (kv=ks³) scaling in the gradient formula.
 * Computes both pred and target gradients in one pass. */
int cpu_fused_cc_loss(const tensor_t *pred, const tensor_t *target,
                       int kernel_size, float *loss_out,
                       tensor_t *grad_pred, tensor_t *grad_target);

/* Global mutual information loss (CPU).
 * Matches fireants GlobalMutualInformationLoss with
 * kernel_type='gaussian', reduction='mean'.
 *
 * Images are normalized to [0,1] internally.
 *
 *   pred:      [B, C, D, H, W]
 *   target:    [B, C, D, H, W]
 *   num_bins:  histogram bins (e.g. 32)
 *   loss_out:  scalar output (negative MI)
 *   grad_pred: [B, C, D, H, W] gradient w.r.t. pred (or NULL)
 *
 * Returns 0 on success. */
int cpu_mi_loss_3d(const tensor_t *pred, const tensor_t *target,
                   int num_bins, float *loss_out, tensor_t *grad_pred);

#ifdef __cplusplus
}
#endif

#endif /* CFIREANTS_LOSSES_H */
