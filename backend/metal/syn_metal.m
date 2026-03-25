/*
 * syn_metal.m - Metal-accelerated SyN (symmetric normalization) registration
 *
 * Faithful translation of backend/cuda/syn_gpu.cu for the Metal backend.
 * Uses Metal shared-memory buffers (MTLResourceStorageModeShared) so all
 * GPU data is CPU-accessible — no explicit H2D/D2H copies needed.
 *
 * SyN maintains TWO warp fields (forward and reverse):
 *   1. Forward: warp moving with affine + fwd_warp -> moved
 *   2. Reverse: warp fixed with identity + rev_warp -> fixed_warped
 *   3. CC loss between moved and fixed_warped (gradients for both)
 *   4. Backward through both grid samples
 *   5. Smooth both gradients
 *   6. WarpAdam on both warp fields (compositive update)
 *   7. Blur both warps
 *
 * Key difference from Greedy:
 *   - Moving image is NOT downsampled, only smoothed (Python _smooth_image_not_mask)
 *   - Fixed IS downsampled (FFT downsample)
 *   - Loss is between two warped images, not warped vs original
 *   - Two independent warp fields and optimizers
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "metal_context.h"
#include "metal_kernels.h"
#include "cfireants/tensor.h"
#include "cfireants/image.h"
#include "cfireants/registration.h"
#include "cfireants/utils.h"

/* ------------------------------------------------------------------ */
/* Metal buffer allocation helper (same as linear_metal.m)             */
/* ------------------------------------------------------------------ */

static float *syn_metal_alloc_buf(size_t bytes, id<MTLBuffer> *out_buf) {
    id<MTLBuffer> buf = [g_metal.device newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
    if (!buf) return NULL;
    float *ptr = (float *)buf.contents;
    metal_register_buffer(ptr, (__bridge void *)buf, bytes);
    *out_buf = buf;
    return ptr;
}

static void syn_metal_free_buf(float *ptr, id<MTLBuffer> buf) {
    if (ptr) metal_unregister_buffer(ptr);
    (void)buf;
}

/* ------------------------------------------------------------------ */
/* Build Gaussian kernel (matching Python separable_filtering)         */
/* ------------------------------------------------------------------ */

static float *syn_make_gauss_kernel(float sigma, int *out_klen) {
    if (sigma <= 0) { *out_klen = 0; return NULL; }
    int tail = (int)(2.0f * sigma + 0.5f);
    int klen = 2 * tail + 1;
    float *h_k = (float *)malloc(klen * sizeof(float));
    float inv = 1.0f / (sigma * sqrtf(2.0f));
    float ksum = 0;
    for (int i = 0; i < klen; i++) {
        float x = (float)(i - tail);
        h_k[i] = 0.5f * (erff((x+0.5f)*inv) - erff((x-0.5f)*inv));
        ksum += h_k[i];
    }
    for (int i = 0; i < klen; i++) h_k[i] /= ksum;
    *out_klen = klen;
    return h_k;
}

/* ------------------------------------------------------------------ */
/* WarpAdam step for one warp field (matching Python WarpAdam.step)    */
/* ------------------------------------------------------------------ */

static void syn_warp_adam_step(
    float *d_warp,           /* [D,H,W,3] displacement (replaced) */
    const float *d_grad,     /* [D,H,W,3] smoothed gradient */
    float *d_exp_avg,        /* [D,H,W,3] first moment */
    float *d_exp_avg_sq,     /* [D,H,W,3] second moment */
    float *d_adam_dir,       /* [D,H,W,3] scratch for adam direction */
    float *d_scratch,        /* [D,H,W,3] scratch for blur */
    int D, int H, int W,
    int *step_t,
    float lr, float beta1, float beta2, float eps,
    float *d_warp_kernel, int warp_klen)
{
    long spatial = (long)D * H * W;
    long n3 = spatial * 3;

    (*step_t)++;
    float bc1 = 1.0f - powf(beta1, (float)*step_t);
    float bc2 = 1.0f - powf(beta2, (float)*step_t);

    /* Update moments and compute direction on GPU */
    metal_warp_adam_moments(d_grad, d_exp_avg, d_exp_avg_sq, beta1, beta2, (int)n3);
    metal_warp_adam_direction(d_adam_dir, d_exp_avg, d_exp_avg_sq, bc1, bc2, eps, (int)n3);

    /* Normalize: gradmax = eps + max(||adam_dir||_2), clamp min=1 */
    float gradmax = metal_max_l2_norm(d_adam_dir, (int)spatial);
    if (gradmax < 1.0f) gradmax = 1.0f;
    float half_res = 1.0f / (float)((D > H ? (D > W ? D : W) : (H > W ? H : W)) - 1);
    float scale = half_res / gradmax * (-lr);

    metal_tensor_scale(d_adam_dir, scale, (int)n3);

    /* Fused compositive update: adam_dir = adam_dir + interp(warp, identity + adam_dir) */
    metal_fused_compositive_update(d_warp, d_adam_dir, d_adam_dir, D, H, W);

    /* Smooth result */
    if (warp_klen > 0)
        metal_blur_disp_dhw3(d_adam_dir, d_scratch, D, H, W, d_warp_kernel, warp_klen);

    /* warp = adam_dir */
    metal_sync();
    memcpy(d_warp, d_adam_dir, n3 * sizeof(float));
}

/* ------------------------------------------------------------------ */
/* Main GPU SyN registration                                           */
/* ------------------------------------------------------------------ */

int syn_register_metal(const image_t *fixed, const image_t *moving,
                        const float init_affine_44[4][4],
                        syn_opts_t opts, syn_result_t *result)
{
    @autoreleasepool {

    memset(result, 0, sizeof(syn_result_t));
    memcpy(result->affine_44, init_affine_44, 16 * sizeof(float));

    int fD = fixed->data.shape[2], fH = fixed->data.shape[3], fW = fixed->data.shape[4];
    int mD = moving->data.shape[2], mH = moving->data.shape[3], mW = moving->data.shape[4];
    long fSpatial = (long)fD * fH * fW;
    long mSpatial = (long)mD * mH * mW;

    /* Build combined affine [1,3,4] for forward warp */
    mat44d phys_d, tmp_m, combined;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            phys_d.m[i][j] = init_affine_44[i][j];
    mat44d_mul(&tmp_m, &phys_d, &fixed->meta.torch2phy);
    mat44d_mul(&combined, &moving->meta.phy2torch, &tmp_m);
    float h_aff[12];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            h_aff[i*4+j] = (float)combined.m[i][j];

    /* Identity affine for reverse warp */
    float h_id[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};

    /* Upload images */
    id<MTLBuffer> fixed_buf, moving_buf;
    float *d_fixed = syn_metal_alloc_buf(fSpatial * sizeof(float), &fixed_buf);
    float *d_moving = syn_metal_alloc_buf(mSpatial * sizeof(float), &moving_buf);
    if (!d_fixed || !d_moving) {
        fprintf(stderr, "syn_register_metal: buffer allocation failed\n");
        if (d_fixed) syn_metal_free_buf(d_fixed, fixed_buf);
        if (d_moving) syn_metal_free_buf(d_moving, moving_buf);
        return -1;
    }
    memcpy(d_fixed, fixed->data.data, fSpatial * sizeof(float));
    memcpy(d_moving, moving->data.data, mSpatial * sizeof(float));

    /* Gaussian kernels */
    int grad_klen = 0, warp_klen = 0;
    float *h_grad_kernel = syn_make_gauss_kernel(opts.smooth_grad_sigma, &grad_klen);
    float *h_warp_kernel = syn_make_gauss_kernel(opts.smooth_warp_sigma, &warp_klen);

    id<MTLBuffer> grad_kern_buf = nil, warp_kern_buf = nil;
    float *d_grad_kernel = NULL, *d_warp_kernel = NULL;
    if (grad_klen > 0) {
        d_grad_kernel = syn_metal_alloc_buf(grad_klen * sizeof(float), &grad_kern_buf);
        memcpy(d_grad_kernel, h_grad_kernel, grad_klen * sizeof(float));
    }
    if (warp_klen > 0) {
        d_warp_kernel = syn_metal_alloc_buf(warp_klen * sizeof(float), &warp_kern_buf);
        memcpy(d_warp_kernel, h_warp_kernel, warp_klen * sizeof(float));
    }
    free(h_grad_kernel);
    free(h_warp_kernel);

    /* Warp fields and optimizer state */
    id<MTLBuffer> fwd_warp_buf = nil, rev_warp_buf = nil;
    id<MTLBuffer> fwd_m_buf = nil, fwd_v_buf = nil, rev_m_buf = nil, rev_v_buf = nil;
    float *d_fwd_warp = NULL, *d_rev_warp = NULL;
    float *d_fwd_m = NULL, *d_fwd_v = NULL, *d_rev_m = NULL, *d_rev_v = NULL;
    int fwd_step = 0, rev_step = 0;
    int prev_dD = 0, prev_dH = 0, prev_dW = 0;
    float beta1 = 0.9f, beta2 = 0.99f, eps = 1e-8f;

    for (int si = 0; si < opts.n_scales; si++) {
        int scale = opts.scales[si], iters = opts.iterations[si];

        int dD = (scale > 1) ? fD/scale : fD;
        int dH = (scale > 1) ? fH/scale : fH;
        int dW = (scale > 1) ? fW/scale : fW;
        if (dD < 8) dD = 8; if (dH < 8) dH = 8; if (dW < 8) dW = 8;
        if (scale == 1) { dD = fD; dH = fH; dW = fW; }

        long spatial = (long)dD * dH * dW;
        long n3 = spatial * 3;

        /* Downsample fixed, smooth moving WITHOUT downsampling (matching Python) */
        id<MTLBuffer> fdown_buf = nil, mblur_buf = nil;
        float *d_fixed_down, *d_moving_blur;
        int moving_blur_owned = 0;
        if (scale > 1) {
            d_fixed_down = syn_metal_alloc_buf(spatial * sizeof(float), &fdown_buf);
            if (opts.downsample_mode == DOWNSAMPLE_TRILINEAR)
                metal_blur_downsample(d_fixed, d_fixed_down, 1, 1, fD, fH, fW, dD, dH, dW);
            else
                metal_downsample_fft(d_fixed, d_fixed_down, 1, 1, fD, fH, fW, dD, dH, dW);

            /* Python SyN: moving_image_blur = self._smooth_image_not_mask(moving_arrays, gaussians)
             * Applies per-axis Gaussian blur at FULL resolution without downsampling.
             * sigmas[i] = 0.5 * fixed_size[i] / size_down[i] */
            d_moving_blur = syn_metal_alloc_buf(mSpatial * sizeof(float), &mblur_buf);
            moving_blur_owned = 1;

            metal_sync();
            memcpy(d_moving_blur, d_moving, mSpatial * sizeof(float));

            metal_blur_volume(d_moving_blur, mD, mH, mW,
                              0.5f * (float)fD / (float)dD,
                              0.5f * (float)fH / (float)dH,
                              0.5f * (float)fW / (float)dW);
        } else {
            d_fixed_down = d_fixed;
            d_moving_blur = d_moving;
        }

        /* Resize/init warp fields */
        if (d_fwd_warp == NULL) {
            d_fwd_warp = syn_metal_alloc_buf(n3 * sizeof(float), &fwd_warp_buf);
            d_rev_warp = syn_metal_alloc_buf(n3 * sizeof(float), &rev_warp_buf);
            memset(d_fwd_warp, 0, n3 * sizeof(float));
            memset(d_rev_warp, 0, n3 * sizeof(float));
        } else if (prev_dD != dD || prev_dH != dH || prev_dW != dW) {
            /* Resize via permute+trilinear (same as CUDA) */
            int ps = prev_dD * prev_dH * prev_dW;

            id<MTLBuffer> t1_buf, t2_buf;
            float *d_t1 = syn_metal_alloc_buf(ps * 3 * sizeof(float), &t1_buf);
            float *d_t2 = syn_metal_alloc_buf(spatial * 3 * sizeof(float), &t2_buf);

            /* Resize fwd */
            metal_permute_dhw3_3dhw(d_fwd_warp, d_t1, prev_dD, prev_dH, prev_dW);
            metal_trilinear_resize(d_t1, d_t2, 1, 3, prev_dD, prev_dH, prev_dW, dD, dH, dW, 1);
            syn_metal_free_buf(d_fwd_warp, fwd_warp_buf);
            d_fwd_warp = syn_metal_alloc_buf(n3 * sizeof(float), &fwd_warp_buf);
            metal_permute_3dhw_dhw3(d_t2, d_fwd_warp, dD, dH, dW);

            /* Resize rev */
            metal_permute_dhw3_3dhw(d_rev_warp, d_t1, prev_dD, prev_dH, prev_dW);
            metal_trilinear_resize(d_t1, d_t2, 1, 3, prev_dD, prev_dH, prev_dW, dD, dH, dW, 1);
            syn_metal_free_buf(d_rev_warp, rev_warp_buf);
            d_rev_warp = syn_metal_alloc_buf(n3 * sizeof(float), &rev_warp_buf);
            metal_permute_3dhw_dhw3(d_t2, d_rev_warp, dD, dH, dW);

            syn_metal_free_buf(d_t1, t1_buf);
            syn_metal_free_buf(d_t2, t2_buf);

            /* Reset optimizer state (matching Python: new WarpAdam per scale) */
            if (d_fwd_m) { syn_metal_free_buf(d_fwd_m, fwd_m_buf); d_fwd_m = NULL; }
            if (d_fwd_v) { syn_metal_free_buf(d_fwd_v, fwd_v_buf); d_fwd_v = NULL; }
            if (d_rev_m) { syn_metal_free_buf(d_rev_m, rev_m_buf); d_rev_m = NULL; }
            if (d_rev_v) { syn_metal_free_buf(d_rev_v, rev_v_buf); d_rev_v = NULL; }
            fwd_step = 0; rev_step = 0;
        }

        if (!d_fwd_m) {
            d_fwd_m = syn_metal_alloc_buf(n3 * sizeof(float), &fwd_m_buf);
            d_fwd_v = syn_metal_alloc_buf(n3 * sizeof(float), &fwd_v_buf);
            d_rev_m = syn_metal_alloc_buf(n3 * sizeof(float), &rev_m_buf);
            d_rev_v = syn_metal_alloc_buf(n3 * sizeof(float), &rev_v_buf);
            memset(d_fwd_m, 0, n3 * sizeof(float));
            memset(d_fwd_v, 0, n3 * sizeof(float));
            memset(d_rev_m, 0, n3 * sizeof(float));
            memset(d_rev_v, 0, n3 * sizeof(float));
        }
        prev_dD = dD; prev_dH = dH; prev_dW = dW;

        /* Grids */
        id<MTLBuffer> fwd_aff_buf, rev_aff_buf, fwd_base_buf, rev_base_buf;
        float *d_fwd_aff = syn_metal_alloc_buf(12 * sizeof(float), &fwd_aff_buf);
        float *d_rev_aff = syn_metal_alloc_buf(12 * sizeof(float), &rev_aff_buf);
        memcpy(d_fwd_aff, h_aff, 12 * sizeof(float));
        memcpy(d_rev_aff, h_id, 12 * sizeof(float));

        float *d_fwd_base = syn_metal_alloc_buf(n3 * sizeof(float), &fwd_base_buf);
        float *d_rev_base = syn_metal_alloc_buf(n3 * sizeof(float), &rev_base_buf);
        metal_affine_grid_3d(d_fwd_aff, d_fwd_base, 1, dD, dH, dW);
        metal_affine_grid_3d(d_rev_aff, d_rev_base, 1, dD, dH, dW);

        /* Scratch buffers */
        id<MTLBuffer> sg_fwd_buf, sg_rev_buf, moved_buf, fwarped_buf;
        id<MTLBuffer> gmoved_buf, gfwarped_buf, gfwd_buf, grev_buf;
        id<MTLBuffer> adam_dir_fwd_buf, adam_dir_rev_buf;
        id<MTLBuffer> s1_buf, s2_buf;
        id<MTLBuffer> cc_interm_buf, cc_scratch_buf;
        float *d_sg_fwd = syn_metal_alloc_buf(n3 * sizeof(float), &sg_fwd_buf);
        float *d_sg_rev = syn_metal_alloc_buf(n3 * sizeof(float), &sg_rev_buf);
        float *d_moved = syn_metal_alloc_buf(spatial * sizeof(float), &moved_buf);
        float *d_fwarped = syn_metal_alloc_buf(spatial * sizeof(float), &fwarped_buf);
        float *d_grad_moved = syn_metal_alloc_buf(spatial * sizeof(float), &gmoved_buf);
        float *d_grad_fwarped = syn_metal_alloc_buf(spatial * sizeof(float), &gfwarped_buf);
        float *d_grad_fwd = syn_metal_alloc_buf(n3 * sizeof(float), &gfwd_buf);
        float *d_grad_rev = syn_metal_alloc_buf(n3 * sizeof(float), &grev_buf);
        float *d_adam_dir_fwd = syn_metal_alloc_buf(n3 * sizeof(float), &adam_dir_fwd_buf);
        float *d_adam_dir_rev = syn_metal_alloc_buf(n3 * sizeof(float), &adam_dir_rev_buf);
        float *d_s1 = syn_metal_alloc_buf(n3 * sizeof(float), &s1_buf);
        float *d_s2 = syn_metal_alloc_buf(n3 * sizeof(float), &s2_buf);
        float *d_cc_interm = syn_metal_alloc_buf(5L * spatial * sizeof(float), &cc_interm_buf);
        float *d_cc_scratch = syn_metal_alloc_buf(spatial * sizeof(float), &cc_scratch_buf);

        if (cfireants_verbose >= 2) fprintf(stderr, "  SyN Metal scale %d: [%d,%d,%d] x %d iters\n", scale, dD, dH, dW, iters);

        float prev_loss = 1e30f;
        int converge_count = 0;

        for (int it = 0; it < iters; it++) {
            /* 1. sampling grids = base + warp */
            metal_vec_add(d_sg_fwd, d_fwd_base, d_fwd_warp, (int)n3);
            metal_vec_add(d_sg_rev, d_rev_base, d_rev_warp, (int)n3);

            /* 2. moved = grid_sample(moving_blur, fwd_sampling_grid) */
            metal_grid_sample_3d_fwd(d_moving_blur, d_sg_fwd, d_moved,
                                      1, 1, mD, mH, mW, dD, dH, dW);

            /* 3. fixed_warped = grid_sample(fixed_down, rev_sampling_grid) */
            metal_grid_sample_3d_fwd(d_fixed_down, d_sg_rev, d_fwarped,
                                      1, 1, dD, dH, dW, dD, dH, dW);

            /* 4. Fused CC loss — computes both pred and target gradients */
            float loss;
            metal_fused_cc_loss(d_moved, d_fwarped,
                               d_grad_moved, d_grad_fwarped,
                               dD, dH, dW, opts.cc_kernel_size,
                               &loss, d_cc_interm, d_cc_scratch);

            /* 5. Backward for fwd_warp: dL/d(fwd_sg) */
            metal_grid_sample_3d_bwd(d_grad_moved, d_moving_blur, d_sg_fwd,
                                      d_grad_fwd, 1, 1, mD, mH, mW, dD, dH, dW);

            /* 6. Backward for rev_warp using target gradient */
            metal_grid_sample_3d_bwd(d_grad_fwarped, d_fixed_down, d_sg_rev,
                                      d_grad_rev, 1, 1, dD, dH, dW, dD, dH, dW);

            /* 7. Smooth gradients (hook) */
            if (grad_klen > 0) {
                metal_blur_disp_dhw3(d_grad_fwd, d_s1, dD, dH, dW, d_grad_kernel, grad_klen);
                metal_blur_disp_dhw3(d_grad_rev, d_s2, dD, dH, dW, d_grad_kernel, grad_klen);
            }

            /* 8. WarpAdam step for both warps */
            syn_warp_adam_step(d_fwd_warp, d_grad_fwd, d_fwd_m, d_fwd_v,
                               d_adam_dir_fwd, d_s1, dD, dH, dW, &fwd_step,
                               opts.lr, beta1, beta2, eps,
                               d_warp_kernel, warp_klen);
            syn_warp_adam_step(d_rev_warp, d_grad_rev, d_rev_m, d_rev_v,
                               d_adam_dir_rev, d_s2, dD, dH, dW, &rev_step,
                               opts.lr, beta1, beta2, eps,
                               d_warp_kernel, warp_klen);

            if (it % 50 == 0 || it == iters - 1)
                if (cfireants_verbose >= 2) fprintf(stderr, "    iter %d/%d loss=%.6f\n", it, iters, loss);
            if (fabsf(loss - prev_loss) < opts.tolerance) {
                converge_count++;
                if (converge_count >= opts.max_tolerance_iters) {
                    if (cfireants_verbose >= 2) fprintf(stderr, "    Converged at iter %d\n", it);
                    break;
                }
            } else converge_count = 0;
            prev_loss = loss;
        }

        /* Cleanup per-scale buffers */
        syn_metal_free_buf(d_fwd_aff, fwd_aff_buf);
        syn_metal_free_buf(d_rev_aff, rev_aff_buf);
        syn_metal_free_buf(d_fwd_base, fwd_base_buf);
        syn_metal_free_buf(d_rev_base, rev_base_buf);
        syn_metal_free_buf(d_sg_fwd, sg_fwd_buf);
        syn_metal_free_buf(d_sg_rev, sg_rev_buf);
        syn_metal_free_buf(d_moved, moved_buf);
        syn_metal_free_buf(d_fwarped, fwarped_buf);
        syn_metal_free_buf(d_grad_moved, gmoved_buf);
        syn_metal_free_buf(d_grad_fwarped, gfwarped_buf);
        syn_metal_free_buf(d_grad_fwd, gfwd_buf);
        syn_metal_free_buf(d_grad_rev, grev_buf);
        syn_metal_free_buf(d_adam_dir_fwd, adam_dir_fwd_buf);
        syn_metal_free_buf(d_adam_dir_rev, adam_dir_rev_buf);
        syn_metal_free_buf(d_s1, s1_buf);
        syn_metal_free_buf(d_s2, s2_buf);
        syn_metal_free_buf(d_cc_interm, cc_interm_buf);
        syn_metal_free_buf(d_cc_scratch, cc_scratch_buf);
        if (scale > 1) syn_metal_free_buf(d_fixed_down, fdown_buf);
        if (moving_blur_owned) syn_metal_free_buf(d_moving_blur, mblur_buf);
    }

    /* ---- Evaluate: compose fwd_warp with inverse(rev_warp) ---- */
    /*
     * Matching Python SyN.get_warp_parameters():
     *   1. Resize fwd_warp and rev_warp to full resolution
     *   2. Compute inv_rev = inverse(rev_warp) via fixed-point iteration
     *   3. Compose: composed = inv_rev + interp(fwd_warp, identity + inv_rev)
     *   4. Final grid = affine_grid(combined_affine) + composed
     *   5. Sample moving image
     */
    {
        long n3 = fSpatial * 3;

        /* Resize warps to full resolution if needed */
        id<MTLBuffer> fwd_full_buf = nil, rev_full_buf = nil;
        float *d_fwd_full, *d_rev_full;
        int fwd_full_owned = 0, rev_full_owned = 0;

        if (prev_dD == fD && prev_dH == fH && prev_dW == fW) {
            d_fwd_full = d_fwd_warp;
            d_rev_full = d_rev_warp;
        } else {
            int ps = prev_dD * prev_dH * prev_dW;

            id<MTLBuffer> t1_buf, t2_buf;
            float *d_t1 = syn_metal_alloc_buf(ps * 3 * sizeof(float), &t1_buf);
            float *d_t2 = syn_metal_alloc_buf(fSpatial * 3 * sizeof(float), &t2_buf);

            /* Resize fwd */
            metal_permute_dhw3_3dhw(d_fwd_warp, d_t1, prev_dD, prev_dH, prev_dW);
            metal_trilinear_resize(d_t1, d_t2, 1, 3, prev_dD, prev_dH, prev_dW, fD, fH, fW, 1);
            d_fwd_full = syn_metal_alloc_buf(n3 * sizeof(float), &fwd_full_buf);
            metal_permute_3dhw_dhw3(d_t2, d_fwd_full, fD, fH, fW);
            fwd_full_owned = 1;

            /* Resize rev */
            metal_permute_dhw3_3dhw(d_rev_warp, d_t1, prev_dD, prev_dH, prev_dW);
            metal_trilinear_resize(d_t1, d_t2, 1, 3, prev_dD, prev_dH, prev_dW, fD, fH, fW, 1);
            d_rev_full = syn_metal_alloc_buf(n3 * sizeof(float), &rev_full_buf);
            metal_permute_3dhw_dhw3(d_t2, d_rev_full, fD, fH, fW);
            rev_full_owned = 1;

            syn_metal_free_buf(d_t1, t1_buf);
            syn_metal_free_buf(d_t2, t2_buf);
        }

        /* Compute inverse of rev_warp via iterative fixed-point */
        id<MTLBuffer> inv_rev_buf;
        float *d_inv_rev = syn_metal_alloc_buf(n3 * sizeof(float), &inv_rev_buf);
        metal_warp_inverse(d_rev_full, d_inv_rev, fD, fH, fW, 0);

        /* Compose: composed = inv_rev + interp(fwd_warp, identity + inv_rev) */
        id<MTLBuffer> composed_buf;
        float *d_composed = syn_metal_alloc_buf(n3 * sizeof(float), &composed_buf);
        metal_sync();
        memcpy(d_composed, d_inv_rev, n3 * sizeof(float));
        metal_fused_compositive_update(d_fwd_full, d_composed, d_composed, fD, fH, fW);

        /* Build base grid from affine */
        id<MTLBuffer> eval_aff_buf, eval_base_buf, eval_sg_buf, eval_moved_buf;
        float *d_eval_aff = syn_metal_alloc_buf(12 * sizeof(float), &eval_aff_buf);
        float *d_eval_base = syn_metal_alloc_buf(n3 * sizeof(float), &eval_base_buf);
        float *d_eval_sg = syn_metal_alloc_buf(n3 * sizeof(float), &eval_sg_buf);
        float *d_eval_moved = syn_metal_alloc_buf(fSpatial * sizeof(float), &eval_moved_buf);

        memcpy(d_eval_aff, h_aff, 12 * sizeof(float));
        metal_affine_grid_3d(d_eval_aff, d_eval_base, 1, fD, fH, fW);

        /* Final sampling grid = base + composed */
        metal_vec_add(d_eval_sg, d_eval_base, d_composed, (int)n3);

        /* Sample moving image */
        metal_grid_sample_3d_fwd(d_moving, d_eval_sg, d_eval_moved,
                                  1, 1, mD, mH, mW, fD, fH, fW);
        metal_cc_loss_3d(d_eval_moved, d_fixed, NULL, fD, fH, fW, 9, &result->ncc_loss);

        /* Download warped image */
        metal_sync();
        int shape[5] = {1, 1, fD, fH, fW};
        tensor_alloc(&result->moved, 5, shape, DTYPE_FLOAT32, DEVICE_CPU);
        memcpy(result->moved.data, d_eval_moved, fSpatial * sizeof(float));

        if (fwd_full_owned) syn_metal_free_buf(d_fwd_full, fwd_full_buf);
        if (rev_full_owned) syn_metal_free_buf(d_rev_full, rev_full_buf);
        syn_metal_free_buf(d_inv_rev, inv_rev_buf);
        syn_metal_free_buf(d_composed, composed_buf);
        syn_metal_free_buf(d_eval_aff, eval_aff_buf);
        syn_metal_free_buf(d_eval_base, eval_base_buf);
        syn_metal_free_buf(d_eval_sg, eval_sg_buf);
        syn_metal_free_buf(d_eval_moved, eval_moved_buf);
    }

    /* Cleanup */
    if (d_fwd_warp) syn_metal_free_buf(d_fwd_warp, fwd_warp_buf);
    if (d_rev_warp) syn_metal_free_buf(d_rev_warp, rev_warp_buf);
    if (d_fwd_m) syn_metal_free_buf(d_fwd_m, fwd_m_buf);
    if (d_fwd_v) syn_metal_free_buf(d_fwd_v, fwd_v_buf);
    if (d_rev_m) syn_metal_free_buf(d_rev_m, rev_m_buf);
    if (d_rev_v) syn_metal_free_buf(d_rev_v, rev_v_buf);
    if (d_grad_kernel) syn_metal_free_buf(d_grad_kernel, grad_kern_buf);
    if (d_warp_kernel) syn_metal_free_buf(d_warp_kernel, warp_kern_buf);
    syn_metal_free_buf(d_fixed, fixed_buf);
    syn_metal_free_buf(d_moving, moving_buf);

    } /* @autoreleasepool */
    return 0;
}
