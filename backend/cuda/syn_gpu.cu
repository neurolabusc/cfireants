/*
 * syn_gpu.cu - GPU SyN (symmetric normalization) registration
 *
 * Faithful clone of fireants SyNRegistration.optimize():
 *   Two warp fields: fwd_warp (moving→midpoint) and rev_warp (fixed→midpoint)
 *   Per iteration:
 *     1. moved = grid_sample(moving_blur, affine_grid + fwd_warp)
 *     2. fixed_warped = grid_sample(fixed_down, identity_grid + rev_warp)
 *     3. loss = CC(moved, fixed_warped)
 *     4. Backward through both grid_samples
 *     5. Smooth gradients (hooks on both warps)
 *     6. WarpAdam step on both warps (compositive update)
 *
 * Key difference from Greedy:
 *   - Moving image is NOT downsampled, only smoothed (Python: _smooth_image_not_mask)
 *   - Fixed IS downsampled (FFT downsample)
 *   - Loss is between two warped images, not warped vs original
 *   - Two independent warp fields and optimizers
 *
 * For evaluation: forward warp only (full warp composition with inverse
 * would require iterative inversion, deferred for now).
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {

#include "cfireants/tensor.h"
#include "cfireants/image.h"
#include "cfireants/registration.h"
#include "cfireants/utils.h"
#include "kernels.h"

void cuda_cc_loss_3d(const float *pred, const float *target,
                     float *grad_pred, int D, int H, int W, int ks,
                     float *h_loss_out);

void cuda_fused_cc_loss(const float *pred, const float *target,
                        float *grad_pred, float *grad_target_out,
                        int D, int H, int W, int ks,
                        float *h_loss_out, float *interm, float *scratch);

} /* extern "C" */

#define BLK 256

/* Shared helpers from cuda_common.cu (declared in kernels.h) */

/* ------------------------------------------------------------------ */
/* WarpAdam step for one warp field (matching Python WarpAdam.step)    */
/* ------------------------------------------------------------------ */
static void warp_adam_step(
    float *d_warp,           /* [D,H,W,3] displacement (modified in-place) */
    const float *d_grad,     /* [D,H,W,3] smoothed gradient */
    float *d_exp_avg,        /* [D,H,W,3] first moment */
    float *d_exp_avg_sq,     /* [D,H,W,3] second moment */
    const float *d_identity_grid,  /* [D,H,W,3] identity grid */
    int D, int H, int W,
    int *step_t,
    float lr, float beta1, float beta2, float eps,
    float *d_warp_kernel, int warp_klen,
    float *d_scratch, float *d_scratch2, float *d_scratch3)
{
    long spatial = D * H * W;
    long n3 = spatial * 3;
    int blocks = (n3 + BLK - 1) / BLK;

    (*step_t)++;
    float bc1 = 1.0f - powf(beta1, (float)*step_t);
    float bc2 = 1.0f - powf(beta2, (float)*step_t);

    /* Update moments */
    cuda_adam_moments_update(d_grad, d_exp_avg, d_exp_avg_sq, beta1, beta2, n3);

    /* Compute Adam direction */
    float *d_adam_dir;
    cudaMalloc(&d_adam_dir, n3 * sizeof(float));
    cuda_adam_direction(d_adam_dir, d_exp_avg, d_exp_avg_sq, bc1, bc2, eps, n3);

    /* Normalize: gradmax = eps + max(||adam_dir||_2), clamp min=1 */
    float gradmax = cuda_max_l2_norm(d_adam_dir, spatial, eps);
    if (gradmax < 1.0f) gradmax = 1.0f;
    float half_res = 1.0f / (float)((D > H ? (D > W ? D : W) : (H > W ? H : W)) - 1);
    float scale = half_res / gradmax * (-lr);
    cuda_vec_scale(d_adam_dir, scale, n3);

    /* Fused compositive update: adam_dir = adam_dir + interp(warp, identity + adam_dir)
     * Single kernel — no permute overhead */
    cuda_fused_compositive_update(d_warp, d_adam_dir, d_adam_dir, D, H, W);

    /* Smooth result */
    if (warp_klen > 0)
        cuda_blur_disp_dhw3(d_adam_dir, d_scratch, D, H, W, d_warp_kernel, warp_klen);

    /* warp = adam_dir */
    cudaMemcpy(d_warp, d_adam_dir, n3 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(d_adam_dir);
}

/* ------------------------------------------------------------------ */
/* Main GPU SyN registration                                           */
/* ------------------------------------------------------------------ */

extern "C" {

int syn_register_gpu(const image_t *fixed, const image_t *moving,
                     const float init_affine_44[4][4],
                     syn_opts_t opts, syn_result_t *result)
{
    memset(result, 0, sizeof(syn_result_t));
    memcpy(result->affine_44, init_affine_44, 16 * sizeof(float));

    int fD=fixed->data.shape[2], fH=fixed->data.shape[3], fW=fixed->data.shape[4];
    int mD=moving->data.shape[2], mH=moving->data.shape[3], mW=moving->data.shape[4];
    long fSpatial=(long)fD*fH*fW, mSpatial=(long)mD*mH*mW;

    /* Build combined affine [1,3,4] for forward warp */
    mat44d phys_d, tmp_m, combined;
    for (int i=0;i<4;i++) for (int j=0;j<4;j++) phys_d.m[i][j]=init_affine_44[i][j];
    mat44d_mul(&tmp_m, &phys_d, &fixed->meta.torch2phy);
    mat44d_mul(&combined, &moving->meta.phy2torch, &tmp_m);
    float h_aff[12];
    for (int i=0;i<3;i++) for (int j=0;j<4;j++) h_aff[i*4+j]=(float)combined.m[i][j];

    float *d_aff; cudaMalloc(&d_aff, 12*sizeof(float));
    cudaMemcpy(d_aff, h_aff, 12*sizeof(float), cudaMemcpyHostToDevice);

    /* Identity affine for reverse warp */
    float h_id[12]={1,0,0,0, 0,1,0,0, 0,0,1,0};
    float *d_id_aff; cudaMalloc(&d_id_aff, 12*sizeof(float));
    cudaMemcpy(d_id_aff, h_id, 12*sizeof(float), cudaMemcpyHostToDevice);

    /* Upload images */
    float *d_fixed, *d_moving;
    cudaMalloc(&d_fixed, fSpatial*sizeof(float));
    cudaMalloc(&d_moving, mSpatial*sizeof(float));
    cudaMemcpy(d_fixed, fixed->data.data, fSpatial*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_moving, moving->data.data, mSpatial*sizeof(float), cudaMemcpyHostToDevice);

    /* Gaussian kernels */
    int grad_klen=0, warp_klen=0;
    float *d_grad_kernel = cuda_make_gpu_gauss(opts.smooth_grad_sigma, 2.0f, &grad_klen);
    float *d_warp_kernel = cuda_make_gpu_gauss(opts.smooth_warp_sigma, 2.0f, &warp_klen);

    /* Warp fields and optimizer state */
    float *d_fwd_warp=NULL, *d_rev_warp=NULL;
    float *d_fwd_m=NULL, *d_fwd_v=NULL, *d_rev_m=NULL, *d_rev_v=NULL;
    int fwd_step=0, rev_step=0;
    int prev_dD=0, prev_dH=0, prev_dW=0;
    float beta1=0.9f, beta2=0.99f, eps=1e-8f;

    for (int si=0; si<opts.n_scales; si++) {
        int scale=opts.scales[si], iters=opts.iterations[si];

        int dD=(scale>1)?fD/scale:fD, dH=(scale>1)?fH/scale:fH, dW=(scale>1)?fW/scale:fW;
        if(dD<8)dD=8;if(dH<8)dH=8;if(dW<8)dW=8;if(scale==1){dD=fD;dH=fH;dW=fW;}

        long spatial=(long)dD*dH*dW, n3=spatial*3;

        /* Downsample fixed (FFT), smooth moving WITHOUT downsampling (matching Python) */
        float *d_fixed_down, *d_moving_blur;
        int moving_blur_owned = 0;
        if (scale > 1) {
            cudaMalloc(&d_fixed_down, spatial*sizeof(float));
            if (opts.downsample_mode == DOWNSAMPLE_TRILINEAR)
                cuda_blur_downsample(d_fixed, d_fixed_down, 1,1,fD,fH,fW,dD,dH,dW);
            else
                cuda_downsample_fft(d_fixed, d_fixed_down, 1,1,fD,fH,fW,dD,dH,dW);

            /* Python: moving_image_blur = self._smooth_image_not_mask(moving_arrays, gaussians)
             * Applies Gaussian blur at full resolution without downsampling.
             * sigma per axis = 0.5 * (original_size / downsampled_size) */
            cudaMalloc(&d_moving_blur, mSpatial*sizeof(float));
            moving_blur_owned = 1;

            /* Python SyN: moving_image_blur = self._smooth_image_not_mask(moving_arrays, gaussians)
             * Applies per-axis Gaussian blur at FULL resolution without downsampling.
             * sigmas[i] = 0.5 * fixed_size[i] / size_down[i] */
            float sigmas[3] = {
                0.5f * (float)fD / (float)dD,
                0.5f * (float)fH / (float)dH,
                0.5f * (float)fW / (float)dW
            };

            cudaMemcpy(d_moving_blur, d_moving, mSpatial*sizeof(float), cudaMemcpyDeviceToDevice);
            float *d_blur_scratch;
            cudaMalloc(&d_blur_scratch, mSpatial*sizeof(float));

            for (int axis = 0; axis < 3; axis++) {
                int dim_sizes[3] = {mD, mH, mW};
                float sigma = sigmas[axis];
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
                float *d_k;
                cudaMalloc(&d_k, klen * sizeof(float));
                cudaMemcpy(d_k, h_k, klen * sizeof(float), cudaMemcpyHostToDevice);
                free(h_k);
                cuda_conv1d_axis(d_moving_blur, d_blur_scratch, mD, mH, mW, d_k, klen, axis);
                cudaMemcpy(d_moving_blur, d_blur_scratch, mSpatial*sizeof(float), cudaMemcpyDeviceToDevice);
                cudaFree(d_k);
            }
            cudaFree(d_blur_scratch);
        } else {
            d_fixed_down = d_fixed;
            d_moving_blur = d_moving;
        }

        /* Resize/init warp fields */
        if (d_fwd_warp == NULL) {
            cudaMalloc(&d_fwd_warp, n3*sizeof(float)); cudaMemset(d_fwd_warp, 0, n3*sizeof(float));
            cudaMalloc(&d_rev_warp, n3*sizeof(float)); cudaMemset(d_rev_warp, 0, n3*sizeof(float));
        } else if (prev_dD!=dD || prev_dH!=dH || prev_dW!=dW) {
            /* Resize via permute+trilinear (same as greedy) */
            int ps=prev_dD*prev_dH*prev_dW;
            float *t1,*t2; int pb=(ps+BLK-1)/BLK, nb=(spatial+BLK-1)/BLK;
            /* Resize fwd */
            cudaMalloc(&t1,ps*3*sizeof(float)); cudaMalloc(&t2,spatial*3*sizeof(float));
            cuda_permute_dhw3_to_3dhw(d_fwd_warp,t1,ps);
            cuda_trilinear_resize(t1,t2,1,3,prev_dD,prev_dH,prev_dW,dD,dH,dW,1);
            cudaFree(d_fwd_warp); cudaMalloc(&d_fwd_warp,n3*sizeof(float));
            cuda_permute_3dhw_to_dhw3(t2,d_fwd_warp,spatial);
            /* Resize rev */
            cuda_permute_dhw3_to_3dhw(d_rev_warp,t1,ps);
            cuda_trilinear_resize(t1,t2,1,3,prev_dD,prev_dH,prev_dW,dD,dH,dW,1);
            cudaFree(d_rev_warp); cudaMalloc(&d_rev_warp,n3*sizeof(float));
            cuda_permute_3dhw_to_dhw3(t2,d_rev_warp,spatial);
            cudaFree(t1); cudaFree(t2);
            /* Reset optimizer state */
            if(d_fwd_m) cudaFree(d_fwd_m); if(d_fwd_v) cudaFree(d_fwd_v);
            if(d_rev_m) cudaFree(d_rev_m); if(d_rev_v) cudaFree(d_rev_v);
            d_fwd_m=d_fwd_v=d_rev_m=d_rev_v=NULL;
        }
        if (!d_fwd_m) {
            cudaMalloc(&d_fwd_m,n3*sizeof(float)); cudaMemset(d_fwd_m,0,n3*sizeof(float));
            cudaMalloc(&d_fwd_v,n3*sizeof(float)); cudaMemset(d_fwd_v,0,n3*sizeof(float));
            cudaMalloc(&d_rev_m,n3*sizeof(float)); cudaMemset(d_rev_m,0,n3*sizeof(float));
            cudaMalloc(&d_rev_v,n3*sizeof(float)); cudaMemset(d_rev_v,0,n3*sizeof(float));
        }
        prev_dD=dD; prev_dH=dH; prev_dW=dW;

        /* Grids */
        float *d_fwd_base, *d_rev_base, *d_identity_grid;
        cudaMalloc(&d_fwd_base, n3*sizeof(float));
        cudaMalloc(&d_rev_base, n3*sizeof(float));
        cudaMalloc(&d_identity_grid, n3*sizeof(float));
        cuda_affine_grid_3d(d_aff, d_fwd_base, 1, dD, dH, dW);
        cuda_affine_grid_3d(d_id_aff, d_rev_base, 1, dD, dH, dW);
        cudaMemcpy(d_identity_grid, d_rev_base, n3*sizeof(float), cudaMemcpyDeviceToDevice);

        /* Scratch — pre-allocate all per-iteration buffers */
        float *d_s1,*d_s2,*d_s3,*d_sg_fwd,*d_sg_rev,*d_moved,*d_fwarped;
        float *d_grad_moved,*d_grad_fwarped,*d_grad_fwd,*d_grad_rev;
        float *d_cc_interm, *d_cc_scratch; /* fused CC workspace */
        cudaMalloc(&d_s1,spatial*sizeof(float));
        cudaMalloc(&d_s2,n3*sizeof(float)); cudaMalloc(&d_s3,n3*sizeof(float));
        cudaMalloc(&d_sg_fwd,n3*sizeof(float)); cudaMalloc(&d_sg_rev,n3*sizeof(float));
        cudaMalloc(&d_moved,spatial*sizeof(float)); cudaMalloc(&d_fwarped,spatial*sizeof(float));
        cudaMalloc(&d_grad_moved,spatial*sizeof(float));
        cudaMalloc(&d_grad_fwarped,spatial*sizeof(float));
        cudaMalloc(&d_grad_fwd,n3*sizeof(float)); cudaMalloc(&d_grad_rev,n3*sizeof(float));
        cudaMalloc(&d_cc_interm, 5L*spatial*sizeof(float));
        cudaMalloc(&d_cc_scratch, spatial*sizeof(float));

        int blocks_n = (n3+BLK-1)/BLK;

        fprintf(stderr, "  SyN GPU scale %d: [%d,%d,%d] x %d iters\n", scale, dD, dH, dW, iters);

        float prev_loss=1e30f; int converge_count=0;

        for (int it=0; it<iters; it++) {
            /* 1. sampling grids = base + warp */
            cuda_vec_add(d_sg_fwd, d_fwd_base, d_fwd_warp, n3);
            cuda_vec_add(d_sg_rev, d_rev_base, d_rev_warp, n3);

            /* 2. moved = grid_sample(moving_blur, fwd_sampling_grid) */
            cuda_grid_sample_3d_fwd(d_moving_blur, d_sg_fwd, d_moved,
                                     1,1,mD,mH,mW,dD,dH,dW);

            /* 3. fixed_warped = grid_sample(fixed_down, rev_sampling_grid) */
            cuda_grid_sample_3d_fwd(d_fixed_down, d_sg_rev, d_fwarped,
                                     1,1,dD,dH,dW,dD,dH,dW);

            /* 4. Fused CC loss — computes both pred and target gradients in one call */
            float loss;
            cuda_fused_cc_loss(d_moved, d_fwarped,
                              d_grad_moved, d_grad_fwarped,
                              dD, dH, dW, opts.cc_kernel_size,
                              &loss, d_cc_interm, d_cc_scratch);

            /* 5. Backward for fwd_warp: dL/d(fwd_sg) */
            cuda_grid_sample_3d_bwd(d_grad_moved, d_moving_blur, d_sg_fwd,
                                    d_grad_fwd, 1,1,mD,mH,mW,dD,dH,dW);

            /* 6. Backward for rev_warp using target gradient */
            cuda_grid_sample_3d_bwd(d_grad_fwarped, d_fixed_down, d_sg_rev,
                                    d_grad_rev, 1,1,dD,dH,dW,dD,dH,dW);

            /* 7. Smooth gradients (hook) */
            if (grad_klen > 0) {
                cuda_blur_disp_dhw3(d_grad_fwd, d_s2, dD, dH, dW, d_grad_kernel, grad_klen);
                cuda_blur_disp_dhw3(d_grad_rev, d_s3, dD, dH, dW, d_grad_kernel, grad_klen);
            }

            /* 8. WarpAdam step for both warps */
            warp_adam_step(d_fwd_warp, d_grad_fwd, d_fwd_m, d_fwd_v,
                          d_identity_grid, dD, dH, dW, &fwd_step,
                          opts.lr, beta1, beta2, eps,
                          d_warp_kernel, warp_klen, d_s2, d_s3, d_sg_fwd);
            warp_adam_step(d_rev_warp, d_grad_rev, d_rev_m, d_rev_v,
                          d_identity_grid, dD, dH, dW, &rev_step,
                          opts.lr, beta1, beta2, eps,
                          d_warp_kernel, warp_klen, d_s2, d_s3, d_sg_rev);

            if (it%50==0 || it==iters-1)
                fprintf(stderr, "    iter %d/%d loss=%.6f\n", it, iters, loss);
            if (fabsf(loss-prev_loss)<opts.tolerance) {
                converge_count++; if(converge_count>=opts.max_tolerance_iters) break;
            } else converge_count=0;
            prev_loss=loss;
        }

        cudaFree(d_fwd_base); cudaFree(d_rev_base); cudaFree(d_identity_grid);
        cudaFree(d_s1); cudaFree(d_s2); cudaFree(d_s3);
        cudaFree(d_sg_fwd); cudaFree(d_sg_rev);
        cudaFree(d_moved); cudaFree(d_fwarped);
        cudaFree(d_grad_moved); cudaFree(d_grad_fwarped);
        cudaFree(d_grad_fwd); cudaFree(d_grad_rev);
        cudaFree(d_cc_interm); cudaFree(d_cc_scratch);
        if (scale>1) cudaFree(d_fixed_down);
        if (moving_blur_owned) cudaFree(d_moving_blur);
    }

    /* Evaluate: compose fwd_warp with inverse(rev_warp)
     * Matching Python SyN.get_warp_parameters():
     *   1. Resize fwd_warp and rev_warp to full resolution
     *   2. Compute inv_rev = inverse(rev_warp) via IC optimization
     *   3. Compose: composed = compose(fwd_warp, inv_rev) = inv_rev + interp(fwd, identity + inv_rev)
     *   4. Final grid = affine_grid(combined_affine) + composed
     *   5. Sample moving image */
    {
        long n3 = fSpatial * 3;

        /* Helper: resize a [D,H,W,3] field from (sD,sH,sW) to (tD,tH,tW) */
        #define RESIZE_DISP(src, dst, sD, sH, sW, tD, tH, tW) do { \
            long _ps=(long)(sD)*(sH)*(sW), _ts=(long)(tD)*(tH)*(tW); \
            float *_t1, *_t2; \
            cudaMalloc(&_t1, _ps*3*sizeof(float)); \
            cudaMalloc(&_t2, _ts*3*sizeof(float)); \
            int _pb=(_ps+BLK-1)/BLK, _tb=(_ts+BLK-1)/BLK; \
            cuda_permute_dhw3_to_3dhw(src, _t1, _ps); \
            cuda_trilinear_resize(_t1, _t2, 1, 3, sD, sH, sW, tD, tH, tW, 1); \
            cudaMalloc(&dst, _ts*3*sizeof(float)); \
            cuda_permute_3dhw_to_dhw3(_t2, dst, _ts); \
            cudaFree(_t1); cudaFree(_t2); \
        } while(0)

        /* Resize warps to full resolution */
        float *d_fwd_full, *d_rev_full;
        if (prev_dD==fD && prev_dH==fH && prev_dW==fW) {
            d_fwd_full = d_fwd_warp;
            d_rev_full = d_rev_warp;
        } else {
            RESIZE_DISP(d_fwd_warp, d_fwd_full, prev_dD, prev_dH, prev_dW, fD, fH, fW);
            RESIZE_DISP(d_rev_warp, d_rev_full, prev_dD, prev_dH, prev_dW, fD, fH, fW);
        }

        /* Compute inverse of rev_warp via fixed-point iteration
         * (matching WebGPU/CPU for cross-backend parity) */
        float *d_inv_rev;
        cudaMalloc(&d_inv_rev, n3 * sizeof(float));
        cuda_warp_inverse_fixedpoint(d_rev_full, d_inv_rev, fD, fH, fW, 550);

        /* Compose: composed = inv_rev + interp(fwd_warp, identity + inv_rev)
         * Using our fused compositive update kernel */
        float *d_composed;
        cudaMalloc(&d_composed, n3 * sizeof(float));
        cudaMemcpy(d_composed, d_inv_rev, n3 * sizeof(float), cudaMemcpyDeviceToDevice);
        cuda_fused_compositive_update(d_fwd_full, d_composed, d_composed, fD, fH, fW);

        /* Build base grid from affine */
        float *d_base;
        cudaMalloc(&d_base, n3 * sizeof(float));
        cuda_affine_grid_3d(d_aff, d_base, 1, fD, fH, fW);

        /* Final sampling grid = base + composed */
        float *d_sg;
        cudaMalloc(&d_sg, n3 * sizeof(float));
        int blocks = (n3 + BLK - 1) / BLK;
        cuda_vec_add(d_sg, d_base, d_composed, n3);

        /* Sample moving image */
        float *d_moved_full;
        cudaMalloc(&d_moved_full, fSpatial * sizeof(float));
        cuda_grid_sample_3d_fwd(d_moving, d_sg, d_moved_full, 1, 1, mD, mH, mW, fD, fH, fW);
        cuda_cc_loss_3d(d_moved_full, d_fixed, NULL, fD, fH, fW, 9, &result->ncc_loss);

        /* Download warped image */
        int shape[5] = {1, 1, fD, fH, fW};
        tensor_alloc(&result->moved, 5, shape, DTYPE_FLOAT32, DEVICE_CPU);
        cudaMemcpy(result->moved.data, d_moved_full, fSpatial * sizeof(float), cudaMemcpyDeviceToHost);

        if (d_fwd_full != d_fwd_warp) cudaFree(d_fwd_full);
        if (d_rev_full != d_rev_warp) cudaFree(d_rev_full);
        cudaFree(d_inv_rev); cudaFree(d_composed);
        cudaFree(d_base); cudaFree(d_sg); cudaFree(d_moved_full);

        #undef RESIZE_DISP
    }

    /* Cleanup */
    if(d_fwd_warp)cudaFree(d_fwd_warp); if(d_rev_warp)cudaFree(d_rev_warp);
    if(d_fwd_m)cudaFree(d_fwd_m); if(d_fwd_v)cudaFree(d_fwd_v);
    if(d_rev_m)cudaFree(d_rev_m); if(d_rev_v)cudaFree(d_rev_v);
    if(d_grad_kernel)cudaFree(d_grad_kernel); if(d_warp_kernel)cudaFree(d_warp_kernel);
    cudaFree(d_aff); cudaFree(d_id_aff);
    cudaFree(d_fixed); cudaFree(d_moving);

    return 0;
}

} /* extern "C" */
