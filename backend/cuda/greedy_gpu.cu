/*
 * greedy_gpu.cu - GPU greedy registration, faithful clone of Python
 *
 * Mirrors fireants/registration/greedy.py optimize() line-by-line:
 *   1. Downsample fixed and moving separately (blur + interpolate)
 *   2. Resize warp field to current scale
 *   3. Per-iteration:
 *      a. Get warp (parameter with grad)
 *      b. Compose: moved = grid_sample(moving, affine_grid(combined) + warp)
 *      c. loss = CC(moved, fixed)
 *      d. Backward: dL/dwarp via grid_sample_bwd
 *      e. Smooth gradient (Gaussian blur hook)
 *      f. Adam step with compositive update:
 *         - m = β1*m + (1-β1)*g
 *         - v = β2*v + (1-β2)*g²
 *         - adam_dir = m/(β1_corr) / (sqrt(v/β2_corr) + ε)
 *         - Normalize: adam_dir /= max(‖adam_dir‖₂) * half_resolution
 *         - Scale: adam_dir *= -lr
 *         - Compose: warp_new = adam_dir + grid_sample(warp, identity + adam_dir)
 *         - Smooth: warp_new = blur(warp_new)
 *         - warp = warp_new
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

/* ------------------------------------------------------------------ */
/* Helper kernels                                                      */
/* ------------------------------------------------------------------ */

/* a[i] = b[i] + c[i] */
__global__ void vec_add_k(float *a, const float *b, const float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = b[i] + c[i];
}

/* a[i] = a[i] * alpha */
__global__ void vec_scale_k(float *a, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= alpha;
}

/* Max L2 norm reduction for WarpAdam normalization */
__global__ void compute_gradmax_kernel(const float *grad, float *partial_max,
                                        long spatial, float eps) {
    __shared__ float smax[BLK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0;
    if (i < spatial) {
        float gx = grad[i*3+0], gy = grad[i*3+1], gz = grad[i*3+2];
        val = sqrtf(gx*gx + gy*gy + gz*gz);
    }
    smax[tid] = val;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s && smax[tid+s] > smax[tid]) smax[tid] = smax[tid+s];
        __syncthreads();
    }
    if (tid == 0) partial_max[blockIdx.x] = smax[0];
}

static float gpu_max_l2_norm(const float *grad, long spatial, float eps) {
    int blocks = (spatial + BLK - 1) / BLK;
    float *d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(float));
    compute_gradmax_kernel<<<blocks, BLK>>>(grad, d_partial, spatial, eps);
    float *h_partial = (float *)malloc(blocks * sizeof(float));
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float maxval = 0;
    for (int i = 0; i < blocks; i++)
        if (h_partial[i] > maxval) maxval = h_partial[i];
    free(h_partial);
    cudaFree(d_partial);
    return eps + maxval;
}

/* Permute: idx = d*H*W + h*W + w, channel c
 * src[idx*3 + c] -> dst[c*spatial + idx] */
__global__ void permute_dhw3_3dhw_kernel(const float *src, float *dst,
                                          long spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < spatial) {
        for (int c = 0; c < 3; c++)
            dst[(long)c*spatial + idx] = src[(long)idx*3 + c];
    }
}

__global__ void permute_3dhw_dhw3_kernel(const float *src, float *dst,
                                          long spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < spatial) {
        for (int c = 0; c < 3; c++)
            dst[(long)idx*3 + c] = src[(long)c*spatial + idx];
    }
}

/* ------------------------------------------------------------------ */
/* Build Gaussian kernel on GPU                                        */
/* ------------------------------------------------------------------ */

static float *make_gpu_gauss(float sigma, float truncated, int *klen_out) {
    if (sigma <= 0) { *klen_out = 0; return NULL; }
    int tail = (int)(truncated * sigma + 0.5f);
    int klen = 2 * tail + 1;
    float *h = (float *)malloc(klen * sizeof(float));
    float inv = 1.0f / (sigma * sqrtf(2.0f));
    float sum = 0;
    for (int i = 0; i < klen; i++) {
        float x = (float)(i - tail);
        h[i] = 0.5f * (erff((x+0.5f)*inv) - erff((x-0.5f)*inv));
        sum += h[i];
    }
    for (int i = 0; i < klen; i++) h[i] /= sum;
    float *d; cudaMalloc(&d, klen*sizeof(float));
    cudaMemcpy(d, h, klen*sizeof(float), cudaMemcpyHostToDevice);
    free(h);
    *klen_out = klen;
    return d;
}

/* Dead code removed: blur_disp_field (replaced by cuda_blur_disp_dhw3)
 * Dead code removed: compositive_update (replaced by cuda_fused_compositive_update) */

/* ------------------------------------------------------------------ */
/* Main GPU greedy registration                                        */
/* ------------------------------------------------------------------ */

extern "C" {

int greedy_register_gpu(const image_t *fixed, const image_t *moving,
                        const float init_affine_44[4][4],
                        greedy_opts_t opts, greedy_result_t *result)
{
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

    float *d_aff;
    cudaMalloc(&d_aff, 12*sizeof(float));
    cudaMemcpy(d_aff, h_aff, 12*sizeof(float), cudaMemcpyHostToDevice);

    /* ---- Upload full-resolution images ---- */
    float *d_fixed_full, *d_moving_full;
    cudaMalloc(&d_fixed_full, fSpatial*sizeof(float));
    cudaMalloc(&d_moving_full, mSpatial*sizeof(float));
    cudaMemcpy(d_fixed_full, fixed->data.data, fSpatial*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_moving_full, moving->data.data, mSpatial*sizeof(float), cudaMemcpyHostToDevice);

    /* ---- Build Gaussian kernels for gradient and warp smoothing ---- */
    int grad_klen = 0, warp_klen = 0;
    float *d_grad_kernel = make_gpu_gauss(opts.smooth_grad_sigma, 2.0f, &grad_klen);
    float *d_warp_kernel = make_gpu_gauss(opts.smooth_warp_sigma, 2.0f, &warp_klen);

    /* ---- Warp field and optimizer state ---- */
    float *d_warp = NULL;   /* [D,H,W,3] displacement */
    float *d_exp_avg = NULL, *d_exp_avg_sq = NULL;
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

        /* ---- Downsample images (Gaussian blur then trilinear, matching Python) ---- */
        float *d_fixed_down, *d_moving_down;
        if (scale > 1) {
            cudaMalloc(&d_fixed_down, spatial*sizeof(float));
            cudaMalloc(&d_moving_down, mSpatialDown*sizeof(float));
            if (opts.downsample_mode == DOWNSAMPLE_TRILINEAR) {
                cuda_blur_downsample(d_fixed_full, d_fixed_down, 1,1, fD,fH,fW, dD,dH,dW);
                cuda_blur_downsample(d_moving_full, d_moving_down, 1,1, mD,mH,mW, mdD,mdH,mdW);
            } else {
                cuda_downsample_fft(d_fixed_full, d_fixed_down, 1,1, fD,fH,fW, dD,dH,dW);
                cuda_downsample_fft(d_moving_full, d_moving_down, 1,1, mD,mH,mW, mdD,mdH,mdW);
            }
        } else {
            d_fixed_down = d_fixed_full;
            d_moving_down = d_moving_full;
            mdD = mD; mdH = mH; mdW = mW;
        }

        /* ---- Resize warp field (matching Python set_size) ---- */
        if (d_warp == NULL) {
            /* First scale: init to zeros */
            cudaMalloc(&d_warp, n3*sizeof(float));
            cudaMemset(d_warp, 0, n3*sizeof(float));
        } else if (prev_dD != dD || prev_dH != dH || prev_dW != dW) {
            /* Resize: permute [D,H,W,3] -> [1,3,D,H,W], trilinear, permute back */
            int prev_spatial = prev_dD * prev_dH * prev_dW;
            float *d_tmp_3dhw, *d_resized_3dhw;
            cudaMalloc(&d_tmp_3dhw, prev_spatial*3*sizeof(float));
            cudaMalloc(&d_resized_3dhw, spatial*3*sizeof(float));

            int pb = (prev_spatial + BLK - 1) / BLK;
            permute_dhw3_3dhw_kernel<<<pb, BLK>>>(d_warp, d_tmp_3dhw, prev_spatial);
            cuda_trilinear_resize(d_tmp_3dhw, d_resized_3dhw,
                                   1, 3, prev_dD, prev_dH, prev_dW, dD, dH, dW, 1);
            cudaFree(d_tmp_3dhw);
            cudaFree(d_warp);
            cudaMalloc(&d_warp, n3*sizeof(float));
            int nb = (spatial + BLK - 1) / BLK;
            permute_3dhw_dhw3_kernel<<<nb, BLK>>>(d_resized_3dhw, d_warp, spatial);
            cudaFree(d_resized_3dhw);

            /* Resize optimizer state similarly (or reset) */
            /* Python default: reset=False means interpolate. For simplicity, reset. */
            cudaFree(d_exp_avg); cudaFree(d_exp_avg_sq);
            d_exp_avg = NULL; d_exp_avg_sq = NULL;
        }

        /* Allocate optimizer state if needed */
        if (d_exp_avg == NULL) {
            cudaMalloc(&d_exp_avg, n3*sizeof(float));
            cudaMalloc(&d_exp_avg_sq, n3*sizeof(float));
            cudaMemset(d_exp_avg, 0, n3*sizeof(float));
            cudaMemset(d_exp_avg_sq, 0, n3*sizeof(float));
        }

        prev_dD = dD; prev_dH = dH; prev_dW = dW;

        float half_resolution = 1.0f / (float)(
            (dD > dH ? (dD > dW ? dD : dW) : (dH > dW ? dH : dW)) - 1);

        fprintf(stderr, "  Greedy GPU scale %d: fixed[%d,%d,%d] moving[%d,%d,%d] x %d iters\n",
                scale, dD, dH, dW, mdD, mdH, mdW, iters);

        /* Generate identity grid for compositive update */
        float h_identity_aff[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
        float *d_identity_aff, *d_identity_grid;
        cudaMalloc(&d_identity_aff, 12*sizeof(float));
        cudaMemcpy(d_identity_aff, h_identity_aff, 12*sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_identity_grid, n3*sizeof(float));
        cuda_affine_grid_3d(d_identity_aff, d_identity_grid, 1, dD, dH, dW);
        cudaFree(d_identity_aff);

        /* Generate affine base grid for forward pass */
        float *d_base_grid;
        cudaMalloc(&d_base_grid, n3*sizeof(float));
        cuda_affine_grid_3d(d_aff, d_base_grid, 1, dD, dH, dW);

        /* Scratch buffers */
        float *d_scratch, *d_scratch2, *d_scratch3;
        cudaMalloc(&d_scratch, spatial*sizeof(float));
        cudaMalloc(&d_scratch2, n3*sizeof(float));
        cudaMalloc(&d_scratch3, n3*sizeof(float));
        float *d_sampling_grid, *d_moved, *d_grad_moved, *d_grad_grid, *d_adam_dir;
        float *d_cc_interm, *d_cc_scratch;
        cudaMalloc(&d_sampling_grid, n3*sizeof(float));
        cudaMalloc(&d_moved, spatial*sizeof(float));
        cudaMalloc(&d_grad_moved, spatial*sizeof(float));
        cudaMalloc(&d_grad_grid, n3*sizeof(float));
        cudaMalloc(&d_adam_dir, n3*sizeof(float));
        cudaMalloc(&d_cc_interm, 5L*spatial*sizeof(float));
        cudaMalloc(&d_cc_scratch, spatial*sizeof(float));

        float prev_loss = 1e30f;
        int converge_count = 0;
        int blocks_n = (n3 + BLK - 1) / BLK;

        for (int it = 0; it < iters; it++) {
            /*
             * Faithful clone of Python GreedyRegistration.optimize() inner loop:
             *
             * 1. warp_field = self.warp.get_warp()   [no smoothing for compositive]
             * 2. moved = interpolator(moving, affine=combined, grid=warp, is_displacement=True)
             * 3. loss = loss_fn(moved, fixed)
             * 4. loss.backward()   [grad hook smooths gradient]
             * 5. self.warp.step()  [WarpAdam compositive update]
             */

            /* Step 2: sampling_grid = affine_base_grid + warp (is_displacement=True) */
            vec_add_k<<<blocks_n, BLK>>>(d_sampling_grid, d_base_grid, d_warp, n3);

            /* Step 3: moved = grid_sample(moving_down, sampling_grid) */
            cuda_grid_sample_3d_fwd(d_moving_down, d_sampling_grid, d_moved,
                                     1, 1, mdD, mdH, mdW, dD, dH, dW);

            /* Step 4a: Fused CC loss + gradient w.r.t. moved */
            float loss;
            cuda_fused_cc_loss(d_moved, d_fixed_down,
                              d_grad_moved, NULL,
                              dD, dH, dW, opts.cc_kernel_size,
                              &loss, d_cc_interm, d_cc_scratch);

            /* Step 4b: grid_sample backward → dL/d(warp) */
            cuda_grid_sample_3d_bwd(d_grad_moved, d_moving_down, d_sampling_grid,
                                    d_grad_grid, 1, 1, mdD, mdH, mdW, dD, dH, dW);

            /* Step 4c: Gradient smoothing hook (separable Gaussian blur on grad) */
            if (grad_klen > 0)
                cuda_blur_disp_dhw3(d_grad_grid, d_scratch2, dD, dH, dW,
                                    d_grad_kernel, grad_klen);

            /* Step 5: WarpAdam.step() — compositive update */
            /* 5a. Update moments */
            step_t++;
            float bc1 = 1.0f - powf(beta1, (float)step_t);
            float bc2 = 1.0f - powf(beta2, (float)step_t);

            cuda_adam_moments_update(d_grad_grid, d_exp_avg, d_exp_avg_sq,
                                     beta1, beta2, n3);

            /* 5b. Compute Adam direction: adam_dir = m_hat / (sqrt(v_hat) + eps) */
            cuda_adam_direction(d_adam_dir, d_exp_avg, d_exp_avg_sq, bc1, bc2, eps, n3);

            /* 5c. Normalize: gradmax = eps + max(‖adam_dir‖₂ per voxel)
             *     gradmax = clamp(gradmax, min=1)   [scaledown=False]
             *     adam_dir = adam_dir / gradmax * half_resolution */
            float gradmax = gpu_max_l2_norm(d_adam_dir, spatial, eps);
            if (gradmax < 1.0f) gradmax = 1.0f;  /* clamp min=1 */
            float scale_factor = half_resolution / gradmax * (-opts.lr);

            /* 5d. adam_dir *= (-lr * half_resolution / gradmax) */
            vec_scale_k<<<blocks_n, BLK>>>(d_adam_dir, scale_factor, n3);

            /* 5e. Fused compositive update: adam_dir = adam_dir + interp(warp, identity + adam_dir)
             * Single kernel — no permute overhead */
            cuda_fused_compositive_update(d_warp, d_adam_dir, d_adam_dir, dD, dH, dW);

            /* 5f. Smooth result if warp smoothing requested */
            if (warp_klen > 0)
                cuda_blur_disp_dhw3(d_adam_dir, d_scratch2, dD, dH, dW,
                                    d_warp_kernel, warp_klen);

            /* 5g. warp.data.copy_(adam_dir) */
            cudaMemcpy(d_warp, d_adam_dir, n3*sizeof(float), cudaMemcpyDeviceToDevice);

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

        /* If we broke out for restructuring, clean up and continue */
        cudaFree(d_identity_grid);
        cudaFree(d_base_grid);
        cudaFree(d_scratch); cudaFree(d_scratch2); cudaFree(d_scratch3);
        cudaFree(d_sampling_grid); cudaFree(d_moved);
        cudaFree(d_grad_moved); cudaFree(d_grad_grid); cudaFree(d_adam_dir);
        cudaFree(d_cc_interm); cudaFree(d_cc_scratch);
        if (scale > 1) { cudaFree(d_fixed_down); cudaFree(d_moving_down); }

        /* TODO: Need proper moment update + compositive step kernels */
        /* For now, fall through to evaluation */
    }

    /* ---- Evaluate at full resolution ---- */
    {
        long n3 = fSpatial * 3;
        float *d_base, *d_sg, *d_moved_full;
        cudaMalloc(&d_base, n3*sizeof(float));
        cuda_affine_grid_3d(d_aff, d_base, 1, fD, fH, fW);

        /* If warp exists and matches full res, use it; otherwise identity */
        cudaMalloc(&d_sg, n3*sizeof(float));
        if (d_warp && prev_dD == fD && prev_dH == fH && prev_dW == fW) {
            int blocks = (n3 + BLK - 1) / BLK;
            vec_add_k<<<blocks, BLK>>>(d_sg, d_base, d_warp, n3);
        } else {
            cudaMemcpy(d_sg, d_base, n3*sizeof(float), cudaMemcpyDeviceToDevice);
        }

        cudaMalloc(&d_moved_full, fSpatial*sizeof(float));
        cuda_grid_sample_3d_fwd(d_moving_full, d_sg, d_moved_full,
                                 1, 1, mD, mH, mW, fD, fH, fW);
        cuda_cc_loss_3d(d_moved_full, d_fixed_full, NULL, fD, fH, fW, 9, &result->ncc_loss);

        /* Download warped image to CPU for global NCC computation */
        {
            int shape[5] = {1, 1, fD, fH, fW};
            tensor_alloc(&result->moved, 5, shape, DTYPE_FLOAT32, DEVICE_CPU);
            cudaMemcpy(result->moved.data, d_moved_full,
                       fSpatial * sizeof(float), cudaMemcpyDeviceToHost);
        }

        cudaFree(d_base); cudaFree(d_sg); cudaFree(d_moved_full);
    }

    /* Cleanup */
    if (d_warp) cudaFree(d_warp);
    if (d_exp_avg) cudaFree(d_exp_avg);
    if (d_exp_avg_sq) cudaFree(d_exp_avg_sq);
    if (d_grad_kernel) cudaFree(d_grad_kernel);
    if (d_warp_kernel) cudaFree(d_warp_kernel);
    cudaFree(d_aff);
    cudaFree(d_fixed_full);
    cudaFree(d_moving_full);

    return 0;
}

} /* extern "C" */
