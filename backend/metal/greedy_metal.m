/*
 * greedy_metal.m - Metal-accelerated greedy deformable registration
 *
 * Faithful translation of backend/cuda/greedy_gpu.cu for the Metal backend.
 * Uses Metal shared-memory buffers (MTLResourceStorageModeShared) so all
 * GPU data is CPU-accessible — no explicit H2D/D2H copies needed.
 *
 * Per-iteration loop (matching Python GreedyRegistration.optimize()):
 *   1. Compose: sampling_grid = affine_base_grid + warp
 *   2. moved = grid_sample(moving, sampling_grid)
 *   3. CC loss forward + backward -> grad_moved
 *   4. grid_sample backward -> grad_grid (dL/d(warp))
 *   5. Smooth gradient (Gaussian blur)
 *   6. WarpAdam: update moments, compute direction
 *   7. Normalize by max_l2_norm, scale by -lr * half_resolution
 *   8. Compositive update: warp_new = scaled_dir + interp(warp, identity + scaled_dir)
 *   9. Blur warp field (regularization)
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

static float *greedy_metal_alloc_buf(size_t bytes, id<MTLBuffer> *out_buf) {
    id<MTLBuffer> buf = [g_metal.device newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
    if (!buf) return NULL;
    float *ptr = (float *)buf.contents;
    metal_register_buffer(ptr, (__bridge void *)buf, bytes);
    *out_buf = buf;
    return ptr;
}

static void greedy_metal_free_buf(float *ptr, id<MTLBuffer> buf) {
    if (ptr) metal_unregister_buffer(ptr);
    (void)buf;
}

/* ------------------------------------------------------------------ */
/* Build Gaussian kernel (matching Python separable_filtering)         */
/* ------------------------------------------------------------------ */

static float *greedy_make_gauss_kernel(float sigma, int *out_klen) {
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
/* Main GPU greedy registration                                        */
/* ------------------------------------------------------------------ */

int greedy_register_metal(const image_t *fixed, const image_t *moving,
                           const float init_affine_44[4][4],
                           greedy_opts_t opts, greedy_result_t *result)
{
    @autoreleasepool {

    memset(result, 0, sizeof(greedy_result_t));
    memcpy(result->affine_44, init_affine_44, 16 * sizeof(float));

    int fD = fixed->data.shape[2], fH = fixed->data.shape[3], fW = fixed->data.shape[4];
    int mD = moving->data.shape[2], mH = moving->data.shape[3], mW = moving->data.shape[4];
    long fSpatial = (long)fD * fH * fW;
    long mSpatial = (long)mD * mH * mW;

    /* ---- Build combined torch-space affine [1,3,4] ---- */
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

    /* ---- Allocate full-resolution images in Metal shared buffers ---- */
    id<MTLBuffer> fixed_buf, moving_buf;
    float *d_fixed_full = greedy_metal_alloc_buf(fSpatial * sizeof(float), &fixed_buf);
    float *d_moving_full = greedy_metal_alloc_buf(mSpatial * sizeof(float), &moving_buf);
    if (!d_fixed_full || !d_moving_full) {
        fprintf(stderr, "greedy_register_metal: buffer allocation failed\n");
        if (d_fixed_full) greedy_metal_free_buf(d_fixed_full, fixed_buf);
        if (d_moving_full) greedy_metal_free_buf(d_moving_full, moving_buf);
        return -1;
    }
    memcpy(d_fixed_full, fixed->data.data, fSpatial * sizeof(float));
    memcpy(d_moving_full, moving->data.data, mSpatial * sizeof(float));

    /* ---- Build Gaussian kernels for gradient and warp smoothing ---- */
    int grad_klen = 0, warp_klen = 0;
    float *h_grad_kernel = greedy_make_gauss_kernel(opts.smooth_grad_sigma, &grad_klen);
    float *h_warp_kernel = greedy_make_gauss_kernel(opts.smooth_warp_sigma, &warp_klen);

    /* Allocate Metal buffers for kernels */
    id<MTLBuffer> grad_kern_buf = nil, warp_kern_buf = nil;
    float *d_grad_kernel = NULL, *d_warp_kernel = NULL;
    if (grad_klen > 0) {
        d_grad_kernel = greedy_metal_alloc_buf(grad_klen * sizeof(float), &grad_kern_buf);
        memcpy(d_grad_kernel, h_grad_kernel, grad_klen * sizeof(float));
    }
    if (warp_klen > 0) {
        d_warp_kernel = greedy_metal_alloc_buf(warp_klen * sizeof(float), &warp_kern_buf);
        memcpy(d_warp_kernel, h_warp_kernel, warp_klen * sizeof(float));
    }
    free(h_grad_kernel);
    free(h_warp_kernel);

    /* ---- Warp field and optimizer state ---- */
    id<MTLBuffer> warp_buf = nil, exp_avg_buf = nil, exp_avg_sq_buf = nil;
    float *d_warp = NULL, *d_exp_avg = NULL, *d_exp_avg_sq = NULL;
    int step_t = 0;  /* Adam step counter (continues across scales) */
    int prev_dD = 0, prev_dH = 0, prev_dW = 0;
    float beta1 = 0.9f, beta2 = 0.99f, eps = 1e-8f;

    /* ---- Multi-scale loop ---- */
    for (int si = 0; si < opts.n_scales; si++) {
        int scale = opts.scales[si];
        int iters = opts.iterations[si];

        /* Compute downsampled sizes (matching Python MIN_IMG_SIZE=8) */
        int dD = (scale > 1) ? fD/scale : fD;
        int dH = (scale > 1) ? fH/scale : fH;
        int dW = (scale > 1) ? fW/scale : fW;
        if (dD < 8) dD = 8; if (dH < 8) dH = 8; if (dW < 8) dW = 8;
        if (scale == 1) { dD = fD; dH = fH; dW = fW; }

        int mdD = (scale > 1) ? mD/scale : mD;
        int mdH = (scale > 1) ? mH/scale : mH;
        int mdW = (scale > 1) ? mW/scale : mW;
        if (mdD < 8) mdD = 8; if (mdH < 8) mdH = 8; if (mdW < 8) mdW = 8;
        if (scale == 1) { mdD = mD; mdH = mH; mdW = mW; }

        long spatial = (long)dD * dH * dW;
        long n3 = spatial * 3;
        long mSpatialDown = (long)mdD * mdH * mdW;

        /* ---- Downsample images ---- */
        id<MTLBuffer> fdown_buf = nil, mdown_buf = nil;
        float *d_fixed_down, *d_moving_down;
        if (scale > 1) {
            d_fixed_down = greedy_metal_alloc_buf(spatial * sizeof(float), &fdown_buf);
            d_moving_down = greedy_metal_alloc_buf(mSpatialDown * sizeof(float), &mdown_buf);
            if (opts.downsample_mode == DOWNSAMPLE_TRILINEAR) {
                metal_blur_downsample(d_fixed_full, d_fixed_down, 1, 1, fD, fH, fW, dD, dH, dW);
                metal_blur_downsample(d_moving_full, d_moving_down, 1, 1, mD, mH, mW, mdD, mdH, mdW);
            } else {
                metal_downsample_fft(d_fixed_full, d_fixed_down, 1, 1, fD, fH, fW, dD, dH, dW);
                metal_downsample_fft(d_moving_full, d_moving_down, 1, 1, mD, mH, mW, mdD, mdH, mdW);
            }
        } else {
            d_fixed_down = d_fixed_full;
            d_moving_down = d_moving_full;
            mdD = mD; mdH = mH; mdW = mW;
        }

        /* ---- Resize warp field (matching Python set_size) ---- */
        if (d_warp == NULL) {
            /* First scale: init to zeros */
            d_warp = greedy_metal_alloc_buf(n3 * sizeof(float), &warp_buf);
            memset(d_warp, 0, n3 * sizeof(float));
        } else if (prev_dD != dD || prev_dH != dH || prev_dW != dW) {
            /* Resize via permute+trilinear (same as CUDA) */
            int prev_spatial = prev_dD * prev_dH * prev_dW;

            id<MTLBuffer> tmp_3dhw_buf, resized_3dhw_buf;
            float *d_tmp_3dhw = greedy_metal_alloc_buf(prev_spatial * 3 * sizeof(float), &tmp_3dhw_buf);
            float *d_resized_3dhw = greedy_metal_alloc_buf(spatial * 3 * sizeof(float), &resized_3dhw_buf);

            metal_permute_dhw3_3dhw(d_warp, d_tmp_3dhw, prev_dD, prev_dH, prev_dW);
            metal_trilinear_resize(d_tmp_3dhw, d_resized_3dhw,
                                    1, 3, prev_dD, prev_dH, prev_dW, dD, dH, dW, 1);

            greedy_metal_free_buf(d_tmp_3dhw, tmp_3dhw_buf);
            greedy_metal_free_buf(d_warp, warp_buf);

            d_warp = greedy_metal_alloc_buf(n3 * sizeof(float), &warp_buf);
            metal_permute_3dhw_dhw3(d_resized_3dhw, d_warp, dD, dH, dW);

            greedy_metal_free_buf(d_resized_3dhw, resized_3dhw_buf);

            /* Reset optimizer state */
            if (d_exp_avg) { greedy_metal_free_buf(d_exp_avg, exp_avg_buf); d_exp_avg = NULL; }
            if (d_exp_avg_sq) { greedy_metal_free_buf(d_exp_avg_sq, exp_avg_sq_buf); d_exp_avg_sq = NULL; }
        }

        /* Allocate optimizer state if needed */
        if (d_exp_avg == NULL) {
            d_exp_avg = greedy_metal_alloc_buf(n3 * sizeof(float), &exp_avg_buf);
            d_exp_avg_sq = greedy_metal_alloc_buf(n3 * sizeof(float), &exp_avg_sq_buf);
            memset(d_exp_avg, 0, n3 * sizeof(float));
            memset(d_exp_avg_sq, 0, n3 * sizeof(float));
        }

        prev_dD = dD; prev_dH = dH; prev_dW = dW;

        float half_resolution = 1.0f / (float)(
            (dD > dH ? (dD > dW ? dD : dW) : (dH > dW ? dH : dW)) - 1);

        fprintf(stderr, "  Greedy Metal scale %d: fixed[%d,%d,%d] moving[%d,%d,%d] x %d iters\n",
                scale, dD, dH, dW, mdD, mdH, mdW, iters);

        /* Generate affine base grid for forward pass */
        id<MTLBuffer> aff_buf, base_grid_buf;
        float *d_aff = greedy_metal_alloc_buf(12 * sizeof(float), &aff_buf);
        memcpy(d_aff, h_aff, 12 * sizeof(float));
        float *d_base_grid = greedy_metal_alloc_buf(n3 * sizeof(float), &base_grid_buf);
        metal_affine_grid_3d(d_aff, d_base_grid, 1, dD, dH, dW);

        /* Scratch buffers */
        id<MTLBuffer> sg_buf, moved_buf, gmoved_buf, ggrid_buf, adam_dir_buf;
        id<MTLBuffer> scratch_buf, scratch2_buf;
        id<MTLBuffer> cc_interm_buf, cc_scratch_buf;
        float *d_sampling_grid = greedy_metal_alloc_buf(n3 * sizeof(float), &sg_buf);
        float *d_moved = greedy_metal_alloc_buf(spatial * sizeof(float), &moved_buf);
        float *d_grad_moved = greedy_metal_alloc_buf(spatial * sizeof(float), &gmoved_buf);
        float *d_grad_grid = greedy_metal_alloc_buf(n3 * sizeof(float), &ggrid_buf);
        float *d_adam_dir = greedy_metal_alloc_buf(n3 * sizeof(float), &adam_dir_buf);
        float *d_scratch = greedy_metal_alloc_buf(spatial * sizeof(float), &scratch_buf);
        float *d_scratch2 = greedy_metal_alloc_buf(n3 * sizeof(float), &scratch2_buf);
        float *d_cc_interm = greedy_metal_alloc_buf(5L * spatial * sizeof(float), &cc_interm_buf);
        float *d_cc_scratch = greedy_metal_alloc_buf(spatial * sizeof(float), &cc_scratch_buf);

        float prev_loss = 1e30f;
        int converge_count = 0;

        for (int it = 0; it < iters; it++) {
            /* Step 2: sampling_grid = affine_base_grid + warp */
            metal_vec_add(d_sampling_grid, d_base_grid, d_warp, (int)n3);

            /* Step 3: moved = grid_sample(moving_down, sampling_grid) */
            metal_grid_sample_3d_fwd(d_moving_down, d_sampling_grid, d_moved,
                                      1, 1, mdD, mdH, mdW, dD, dH, dW);

            /* Step 4a: Fused CC loss + gradient w.r.t. moved */
            float loss;
            metal_fused_cc_loss(d_moved, d_fixed_down,
                               d_grad_moved, NULL,
                               dD, dH, dW, opts.cc_kernel_size,
                               &loss, d_cc_interm, d_cc_scratch);

            /* Step 4b: grid_sample backward -> dL/d(warp) */
            metal_grid_sample_3d_bwd(d_grad_moved, d_moving_down, d_sampling_grid,
                                      d_grad_grid, 1, 1, mdD, mdH, mdW, dD, dH, dW);

            /* Step 4c: Gradient smoothing hook (separable Gaussian blur) */
            if (grad_klen > 0)
                metal_blur_disp_dhw3(d_grad_grid, d_scratch2, dD, dH, dW,
                                      d_grad_kernel, grad_klen);

            /* Step 5: WarpAdam.step() — compositive update */
            /* 5a-b. Update moments and compute direction on GPU */
            step_t++;
            float bc1 = 1.0f - powf(beta1, (float)step_t);
            float bc2 = 1.0f - powf(beta2, (float)step_t);

            metal_warp_adam_moments(d_grad_grid, d_exp_avg, d_exp_avg_sq,
                                    beta1, beta2, (int)n3);
            metal_warp_adam_direction(d_adam_dir, d_exp_avg, d_exp_avg_sq,
                                      bc1, bc2, eps, (int)n3);

            /* 5c. Normalize: gradmax = eps + max(||adam_dir||_2 per voxel) */
            float gradmax = metal_max_l2_norm(d_adam_dir, (int)spatial);
            if (gradmax < 1.0f) gradmax = 1.0f;
            float scale_factor = half_resolution / gradmax * (-opts.lr);

            /* 5d. adam_dir *= scale on GPU */
            metal_tensor_scale(d_adam_dir, scale_factor, (int)n3);

            /* 5e. Fused compositive update: adam_dir = adam_dir + interp(warp, identity + adam_dir) */
            metal_fused_compositive_update(d_warp, d_adam_dir, d_adam_dir, dD, dH, dW);

            /* 5f. Smooth result if warp smoothing requested */
            if (warp_klen > 0)
                metal_blur_disp_dhw3(d_adam_dir, d_scratch2, dD, dH, dW,
                                      d_warp_kernel, warp_klen);

            /* 5g. warp = adam_dir (unified memory — just memcpy) */
            metal_sync();
            memcpy(d_warp, d_adam_dir, n3 * sizeof(float));

            if (it % 50 == 0 || it == iters - 1)
                fprintf(stderr, "    iter %d/%d loss=%.6f\n", it, iters, loss);

            if (fabsf(loss - prev_loss) < opts.tolerance) {
                converge_count++;
                if (converge_count >= opts.max_tolerance_iters) {
                    fprintf(stderr, "    Converged at iter %d\n", it);
                    break;
                }
            } else { converge_count = 0; }
            prev_loss = loss;
        }

        /* Cleanup per-scale buffers */
        greedy_metal_free_buf(d_aff, aff_buf);
        greedy_metal_free_buf(d_base_grid, base_grid_buf);
        greedy_metal_free_buf(d_sampling_grid, sg_buf);
        greedy_metal_free_buf(d_moved, moved_buf);
        greedy_metal_free_buf(d_grad_moved, gmoved_buf);
        greedy_metal_free_buf(d_grad_grid, ggrid_buf);
        greedy_metal_free_buf(d_adam_dir, adam_dir_buf);
        greedy_metal_free_buf(d_scratch, scratch_buf);
        greedy_metal_free_buf(d_scratch2, scratch2_buf);
        greedy_metal_free_buf(d_cc_interm, cc_interm_buf);
        greedy_metal_free_buf(d_cc_scratch, cc_scratch_buf);
        if (scale > 1) {
            greedy_metal_free_buf(d_fixed_down, fdown_buf);
            greedy_metal_free_buf(d_moving_down, mdown_buf);
        }
    }

    /* ---- Evaluate at full resolution ---- */
    {
        long n3 = fSpatial * 3;

        id<MTLBuffer> eval_aff_buf, eval_base_buf, eval_sg_buf, eval_moved_buf;
        float *d_eval_aff = greedy_metal_alloc_buf(12 * sizeof(float), &eval_aff_buf);
        float *d_eval_base = greedy_metal_alloc_buf(n3 * sizeof(float), &eval_base_buf);
        float *d_eval_sg = greedy_metal_alloc_buf(n3 * sizeof(float), &eval_sg_buf);
        float *d_eval_moved = greedy_metal_alloc_buf(fSpatial * sizeof(float), &eval_moved_buf);

        memcpy(d_eval_aff, h_aff, 12 * sizeof(float));
        metal_affine_grid_3d(d_eval_aff, d_eval_base, 1, fD, fH, fW);

        /* If warp exists and matches full res, add it; otherwise use base grid */
        if (d_warp && prev_dD == fD && prev_dH == fH && prev_dW == fW) {
            metal_vec_add(d_eval_sg, d_eval_base, d_warp, (int)n3);
        } else {
            metal_sync();
            memcpy(d_eval_sg, d_eval_base, n3 * sizeof(float));
        }

        metal_grid_sample_3d_fwd(d_moving_full, d_eval_sg, d_eval_moved,
                                  1, 1, mD, mH, mW, fD, fH, fW);
        metal_cc_loss_3d(d_eval_moved, d_fixed_full, NULL, fD, fH, fW, 9, &result->ncc_loss);

        /* Download warped image to CPU */
        metal_sync();
        int shape[5] = {1, 1, fD, fH, fW};
        tensor_alloc(&result->moved, 5, shape, DTYPE_FLOAT32, DEVICE_CPU);
        memcpy(result->moved.data, d_eval_moved, fSpatial * sizeof(float));

        greedy_metal_free_buf(d_eval_aff, eval_aff_buf);
        greedy_metal_free_buf(d_eval_base, eval_base_buf);
        greedy_metal_free_buf(d_eval_sg, eval_sg_buf);
        greedy_metal_free_buf(d_eval_moved, eval_moved_buf);
    }

    /* Cleanup */
    if (d_warp) greedy_metal_free_buf(d_warp, warp_buf);
    if (d_exp_avg) greedy_metal_free_buf(d_exp_avg, exp_avg_buf);
    if (d_exp_avg_sq) greedy_metal_free_buf(d_exp_avg_sq, exp_avg_sq_buf);
    if (d_grad_kernel) greedy_metal_free_buf(d_grad_kernel, grad_kern_buf);
    if (d_warp_kernel) greedy_metal_free_buf(d_warp_kernel, warp_kern_buf);
    greedy_metal_free_buf(d_fixed_full, fixed_buf);
    greedy_metal_free_buf(d_moving_full, moving_buf);

    } /* @autoreleasepool */
    return 0;
}
