/*
 * linear_gpu.cu - GPU-accelerated rigid and affine registration
 *
 * Faithful clone of Python rigid.py/affine.py optimize() loops.
 * Images stay on GPU. Only scalar loss and small parameter gradients
 * (7 for rigid, 12 for affine) come back to CPU for the Adam update.
 *
 * Also includes GPU moments evaluation (CC comparison for orientation).
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern "C" {

#include "cfireants/tensor.h"
#include "cfireants/image.h"
#include "cfireants/registration.h"
#include "cfireants/utils.h"
#include "kernels.h"

void cuda_cc_loss_3d(const float *pred, const float *target,
                     float *grad_pred, int D, int H, int W, int ks,
                     float *h_loss_out);

void cuda_mi_loss_3d(const float *pred, const float *target,
                     float *grad_pred, int D, int H, int W,
                     int num_bins, float *h_loss_out);

/* ------------------------------------------------------------------ */
/* GPU affine grid backward: dL/dA from dL/dgrid                       */
/* Sum over all grid points: dL/dA[i][j] = Σ dL/dgrid[p][i] * coord[p][j] */
/* ------------------------------------------------------------------ */

__global__ void affine_grid_bwd_kernel(
    const float * __restrict__ grad_grid,  /* [D, H, W, 3] */
    float * __restrict__ partial_dA,       /* [blocks, 12] */
    int D, int H, int W)
{
    __shared__ float sdata[256 * 12];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = D * H * W;

    /* Initialize shared memory */
    for (int k = 0; k < 12; k++) sdata[tid * 12 + k] = 0;

    if (idx < total) {
        int w = idx % W;
        int tmp = idx / W;
        int h = tmp % H;
        int d = tmp / H;

        float nz = (D > 1) ? (2.0f * d / (D - 1) - 1.0f) : 0.0f;
        float ny = (H > 1) ? (2.0f * h / (H - 1) - 1.0f) : 0.0f;
        float nx = (W > 1) ? (2.0f * w / (W - 1) - 1.0f) : 0.0f;
        float coord[4] = {nx, ny, nz, 1.0f};

        const float *gg = grad_grid + idx * 3;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                sdata[tid * 12 + i * 4 + j] = gg[i] * coord[j];
    }
    __syncthreads();

    /* Reduce within block */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int k = 0; k < 12; k++)
                sdata[tid * 12 + k] += sdata[(tid + s) * 12 + k];
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int k = 0; k < 12; k++)
            partial_dA[blockIdx.x * 12 + k] = sdata[k];
    }
}

static void gpu_affine_grid_backward(const float *d_grad_grid, int D, int H, int W,
                                      float h_dL_dA[12]) {
    int total = D * H * W;
    int blocks = (total + 255) / 256;

    float *d_partial;
    cudaMalloc(&d_partial, blocks * 12 * sizeof(float));
    affine_grid_bwd_kernel<<<blocks, 256>>>(d_grad_grid, d_partial, D, H, W);

    /* Final reduction on CPU */
    float *h_partial = (float *)malloc(blocks * 12 * sizeof(float));
    cudaMemcpy(h_partial, d_partial, blocks * 12 * sizeof(float), cudaMemcpyDeviceToHost);

    memset(h_dL_dA, 0, 12 * sizeof(float));
    for (int b = 0; b < blocks; b++)
        for (int k = 0; k < 12; k++)
            h_dL_dA[k] += h_partial[b * 12 + k];

    free(h_partial);
    cudaFree(d_partial);
}

/* ------------------------------------------------------------------ */
/* Quaternion utilities (same as rigid.c, on CPU)                      */
/* ------------------------------------------------------------------ */

static void quat_to_rotmat(const float q[4], float R[3][3]) {
    float norm = sqrtf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    if (norm < 1e-8f) norm = 1e-8f;
    float w=q[0]/norm, x=q[1]/norm, y=q[2]/norm, z=q[3]/norm;
    R[0][0]=1-2*(y*y+z*z); R[0][1]=2*(x*y-w*z);   R[0][2]=2*(x*z+w*y);
    R[1][0]=2*(x*y+w*z);   R[1][1]=1-2*(x*x+z*z); R[1][2]=2*(y*z-w*x);
    R[2][0]=2*(x*z-w*y);   R[2][1]=2*(y*z+w*x);   R[2][2]=1-2*(x*x+y*y);
}

static void quat_rotmat_jacobian(const float q[4], float dR_dq[4][3][3]) {
    float norm = sqrtf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    if (norm < 1e-8f) norm = 1e-8f;
    float inv_n = 1.0f / norm;
    float w=q[0]*inv_n, x=q[1]*inv_n, y=q[2]*inv_n, z=q[3]*inv_n;
    /* dR/dw */ dR_dq[0][0][0]=0;           dR_dq[0][0][1]=-2*z*inv_n;  dR_dq[0][0][2]=2*y*inv_n;
                dR_dq[0][1][0]=2*z*inv_n;   dR_dq[0][1][1]=0;           dR_dq[0][1][2]=-2*x*inv_n;
                dR_dq[0][2][0]=-2*y*inv_n;  dR_dq[0][2][1]=2*x*inv_n;  dR_dq[0][2][2]=0;
    /* dR/dx */ dR_dq[1][0][0]=0;           dR_dq[1][0][1]=2*y*inv_n;   dR_dq[1][0][2]=2*z*inv_n;
                dR_dq[1][1][0]=2*y*inv_n;   dR_dq[1][1][1]=-4*x*inv_n; dR_dq[1][1][2]=-2*w*inv_n;
                dR_dq[1][2][0]=2*z*inv_n;   dR_dq[1][2][1]=2*w*inv_n;  dR_dq[1][2][2]=-4*x*inv_n;
    /* dR/dy */ dR_dq[2][0][0]=-4*y*inv_n;  dR_dq[2][0][1]=2*x*inv_n;  dR_dq[2][0][2]=2*w*inv_n;
                dR_dq[2][1][0]=2*x*inv_n;   dR_dq[2][1][1]=0;          dR_dq[2][1][2]=2*z*inv_n;
                dR_dq[2][2][0]=-2*w*inv_n;  dR_dq[2][2][1]=2*z*inv_n;  dR_dq[2][2][2]=-4*y*inv_n;
    /* dR/dz */ dR_dq[3][0][0]=-4*z*inv_n;  dR_dq[3][0][1]=-2*w*inv_n; dR_dq[3][0][2]=2*x*inv_n;
                dR_dq[3][1][0]=2*w*inv_n;   dR_dq[3][1][1]=-4*z*inv_n; dR_dq[3][1][2]=2*y*inv_n;
                dR_dq[3][2][0]=2*x*inv_n;   dR_dq[3][2][1]=2*y*inv_n;  dR_dq[3][2][2]=0;
}

/* ------------------------------------------------------------------ */
/* GPU rigid registration                                              */
/* ------------------------------------------------------------------ */

int rigid_register_gpu(const image_t *fixed, const image_t *moving,
                       const moments_result_t *moments_init,
                       rigid_opts_t opts, rigid_result_t *result)
{
    memset(result, 0, sizeof(rigid_result_t));

    int fD=fixed->data.shape[2], fH=fixed->data.shape[3], fW=fixed->data.shape[4];
    int mD=moving->data.shape[2], mH=moving->data.shape[3], mW=moving->data.shape[4];

    /* Upload images to GPU */
    float *d_fixed, *d_moving;
    cudaMalloc(&d_fixed, (size_t)fD*fH*fW*sizeof(float));
    cudaMalloc(&d_moving, (size_t)mD*mH*mW*sizeof(float));
    cudaMemcpy(d_fixed, fixed->data.data, (size_t)fD*fH*fW*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_moving, moving->data.data, (size_t)mD*mH*mW*sizeof(float), cudaMemcpyHostToDevice);

    /* Initialize parameters on CPU */
    float quat[4] = {1, 0, 0, 0};
    float transl[3] = {0, 0, 0};
    float moment[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    float center[3];
    for (int i = 0; i < 3; i++)
        center[i] = (float)fixed->meta.torch2phy.m[i][3];

    if (moments_init) {
        memcpy(moment, moments_init->Rf, sizeof(moment));
        /* Convert to around_center translation */
        for (int i = 0; i < 3; i++) {
            transl[i] = moments_init->tf[i] - center[i];
            for (int j = 0; j < 3; j++)
                transl[i] += moment[i][j] * center[j];
        }
    }

    /* Adam state for 7 parameters (4 quat + 3 transl) */
    float adam_m[7]={0}, adam_v[7]={0};
    int adam_step = 0;
    float lr = opts.lr, beta1=0.9f, beta2=0.999f, eps=1e-8f;

    /* Precompute coordinate transforms */
    const mat44d *fixed_t2p = &fixed->meta.torch2phy;
    const mat44d *moving_p2t = &moving->meta.phy2torch;

    for (int si = 0; si < opts.n_scales; si++) {
        int scale = opts.scales[si];
        int iters = opts.iterations[si];

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
        long mSDown = mdD * mdH * mdW;

        /* Downsample on GPU (matching Python rigid.optimize):
         * fixed: downsample_fft (or blur+trilinear)
         * moving: downsample + extra Gaussian blur (_smooth_image_not_mask) */
        float *d_fdown, *d_mdown;
        if (scale > 1) {
            cudaMalloc(&d_fdown, spatial*sizeof(float));
            cudaMalloc(&d_mdown, mSDown*sizeof(float));

            if (opts.downsample_mode == DOWNSAMPLE_TRILINEAR) {
                cuda_blur_downsample(d_fixed, d_fdown, 1,1, fD,fH,fW, dD,dH,dW);
                cuda_blur_downsample(d_moving, d_mdown, 1,1, mD,mH,mW, mdD,mdH,mdW);
            } else {
                cuda_downsample_fft(d_fixed, d_fdown, 1,1, fD,fH,fW, dD,dH,dW);
                cuda_downsample_fft(d_moving, d_mdown, 1,1, mD,mH,mW, mdD,mdH,mdW);
            }

            /* Extra Gaussian blur on moving (matching Python rigid line 345:
             * moving_image_blur = self._smooth_image_not_mask(moving_image_blur, gaussians))
             * Python uses per-axis sigmas: sigma[i] = 0.5 * fixed_size[i] / size_down[i]
             * with separable_filtering applying a different kernel per axis */
            {
                float sigmas[3] = {
                    0.5f * (float)fD / (float)dD,
                    0.5f * (float)fH / (float)dH,
                    0.5f * (float)fW / (float)dW
                };
                float *d_blur_scratch;
                cudaMalloc(&d_blur_scratch, mSDown * sizeof(float));

                /* Build and apply per-axis kernels (matching Python separable_filtering) */
                for (int axis = 0; axis < 3; axis++) {
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

                    cuda_conv1d_axis(d_mdown, d_blur_scratch, mdD, mdH, mdW, d_k, klen, axis);
                    cudaMemcpy(d_mdown, d_blur_scratch, mSDown * sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaFree(d_k);
                }
                cudaFree(d_blur_scratch);
            }
        } else {
            d_fdown = d_fixed; d_mdown = d_moving;
            mdD = mD; mdH = mH; mdW = mW;
        }

        /* GPU buffers for iteration */
        float *d_aff, *d_grid, *d_moved, *d_grad_moved, *d_grad_grid;
        cudaMalloc(&d_aff, 12*sizeof(float));
        cudaMalloc(&d_grid, n3*sizeof(float));
        cudaMalloc(&d_moved, spatial*sizeof(float));
        cudaMalloc(&d_grad_moved, spatial*sizeof(float));
        cudaMalloc(&d_grad_grid, n3*sizeof(float));

        /* Reset Adam per scale */
        adam_step = 0;
        memset(adam_m, 0, sizeof(adam_m));
        memset(adam_v, 0, sizeof(adam_v));

        fprintf(stderr, "  Rigid GPU scale %d: [%d,%d,%d] x %d iters\n",
                scale, dD, dH, dW, iters);

        float prev_loss = 1e30f;
        int converge_count = 0;

        for (int it = 0; it < iters; it++) {
            /* Build rigid matrix on CPU */
            float R[3][3];
            quat_to_rotmat(quat, R);
            float RM[3][3];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) {
                    RM[i][j] = 0;
                    for (int k = 0; k < 3; k++)
                        RM[i][j] += R[i][k] * moment[k][j];
                }

            /* Build physical 4x4 */
            float rigid44[4][4] = {{0}};
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) rigid44[i][j] = RM[i][j];
                rigid44[i][3] = transl[i] + center[i];
                for (int j = 0; j < 3; j++) rigid44[i][3] -= RM[i][j] * center[j];
            }
            rigid44[3][3] = 1.0f;

            /* Combined torch-space affine */
            mat44d phys_d, tmp_m, comb;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) phys_d.m[i][j] = rigid44[i][j];
            mat44d_mul(&tmp_m, &phys_d, fixed_t2p);
            mat44d_mul(&comb, moving_p2t, &tmp_m);

            float h_aff[12];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    h_aff[i*4+j] = (float)comb.m[i][j];

            /* Upload affine, generate grid, sample, loss — all on GPU */
            cudaMemcpy(d_aff, h_aff, 12*sizeof(float), cudaMemcpyHostToDevice);
            cuda_affine_grid_3d(d_aff, d_grid, 1, dD, dH, dW);
            cuda_grid_sample_3d_fwd(d_mdown, d_grid, d_moved,
                                     1, 1, mdD, mdH, mdW, dD, dH, dW);

            float loss;
            if (opts.loss_type == LOSS_MI) {
                int nbins = opts.mi_num_bins > 0 ? opts.mi_num_bins : 32;
                cuda_mi_loss_3d(d_moved, d_fdown, d_grad_moved, dD, dH, dW, nbins, &loss);
            } else {
                cuda_cc_loss_3d(d_moved, d_fdown, d_grad_moved, dD, dH, dW,
                               opts.cc_kernel_size, &loss);
            }

            cuda_grid_sample_3d_bwd(d_grad_moved, d_mdown, d_grid,
                                    d_grad_grid, 1, 1, mdD, mdH, mdW, dD, dH, dW);

            /* Affine grid backward: dL/dA [12 values] — GPU reduction, result on CPU */
            float dL_dA_comb[12];
            gpu_affine_grid_backward(d_grad_grid, dD, dH, dW, dL_dA_comb);

            /* Chain rule: dL/d(phys) = p2t^T @ dL/d(comb) @ t2p^T */
            mat44d dL_comb_44 = {{{0}}};
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    dL_comb_44.m[i][j] = dL_dA_comb[i*4+j];

            mat44d p2t_T, t2p_T, tmp1, dL_dphys;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) {
                    p2t_T.m[i][j] = moving_p2t->m[j][i];
                    t2p_T.m[i][j] = fixed_t2p->m[j][i];
                }
            mat44d_mul(&tmp1, &p2t_T, &dL_comb_44);
            mat44d_mul(&dL_dphys, &tmp1, &t2p_T);

            /* dL/d(RM) and dL/dt */
            float dL_dRM[3][3], dL_dt[3];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++)
                    dL_dRM[i][j] = (float)dL_dphys.m[i][j] - (float)dL_dphys.m[i][3] * center[j];
                dL_dt[i] = (float)dL_dphys.m[i][3];
            }

            /* dL/dR = dL/dRM @ moment^T */
            float dL_dR[3][3];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) {
                    dL_dR[i][j] = 0;
                    for (int k = 0; k < 3; k++)
                        dL_dR[i][j] += dL_dRM[i][k] * moment[j][k];
                }

            /* dL/dquat */
            float dR_dq[4][3][3];
            quat_rotmat_jacobian(quat, dR_dq);
            float dL_dq[4] = {0};
            for (int k = 0; k < 4; k++)
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        dL_dq[k] += dL_dR[i][j] * dR_dq[k][i][j];

            /* Pack gradient: [q0,q1,q2,q3, tx,ty,tz] */
            float grad7[7];
            for (int k = 0; k < 4; k++) grad7[k] = dL_dq[k];
            for (int k = 0; k < 3; k++) grad7[4+k] = dL_dt[k];

            /* Adam update on CPU (only 7 params) */
            adam_step++;
            float bc1 = 1.0f - powf(beta1, (float)adam_step);
            float bc2 = 1.0f - powf(beta2, (float)adam_step);
            float step_size = lr / bc1;

            float params7[7];
            for (int k = 0; k < 4; k++) params7[k] = quat[k];
            for (int k = 0; k < 3; k++) params7[4+k] = transl[k];

            for (int k = 0; k < 7; k++) {
                adam_m[k] = beta1*adam_m[k] + (1-beta1)*grad7[k];
                adam_v[k] = beta2*adam_v[k] + (1-beta2)*grad7[k]*grad7[k];
                float denom = sqrtf(adam_v[k]/bc2) + eps;
                params7[k] -= step_size * adam_m[k] / denom;
            }

            for (int k = 0; k < 4; k++) quat[k] = params7[k];
            for (int k = 0; k < 3; k++) transl[k] = params7[4+k];

            if (it % 50 == 0 || it == iters - 1)
                fprintf(stderr, "    iter %d/%d loss=%.6f\n", it, iters, loss);

            if (fabsf(loss - prev_loss) < opts.tolerance) {
                converge_count++;
                if (converge_count >= opts.max_tolerance_iters) break;
            } else { converge_count = 0; }
            prev_loss = loss;
        }

        cudaFree(d_aff); cudaFree(d_grid);
        cudaFree(d_moved); cudaFree(d_grad_moved); cudaFree(d_grad_grid);
        if (scale > 1) { cudaFree(d_fdown); cudaFree(d_mdown); }
    }

    /* Extract final rigid matrix */
    float R[3][3], RM[3][3];
    quat_to_rotmat(quat, R);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            RM[i][j] = 0;
            for (int k = 0; k < 3; k++) RM[i][j] += R[i][k] * moment[k][j];
        }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) result->rigid_mat[i][j] = RM[i][j];
        result->rigid_mat[i][3] = transl[i] + center[i];
        for (int j = 0; j < 3; j++) result->rigid_mat[i][3] -= RM[i][j] * center[j];
    }

    /* Evaluate NCC on GPU at full resolution */
    {
        float *d_aff2, *d_grid2, *d_moved2;
        cudaMalloc(&d_aff2, 12*sizeof(float));
        cudaMalloc(&d_grid2, (size_t)fD*fH*fW*3*sizeof(float));
        cudaMalloc(&d_moved2, (size_t)fD*fH*fW*sizeof(float));

        float phys44[4][4] = {{0}};
        for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) phys44[i][j] = result->rigid_mat[i][j];
        phys44[3][3] = 1.0f;
        mat44d pd, tm, cb;
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) pd.m[i][j] = phys44[i][j];
        mat44d_mul(&tm, &pd, fixed_t2p);
        mat44d_mul(&cb, moving_p2t, &tm);
        float ha[12]; for (int i=0;i<3;i++) for (int j=0;j<4;j++) ha[i*4+j]=(float)cb.m[i][j];
        cudaMemcpy(d_aff2, ha, 12*sizeof(float), cudaMemcpyHostToDevice);
        cuda_affine_grid_3d(d_aff2, d_grid2, 1, fD, fH, fW);
        cuda_grid_sample_3d_fwd(d_moving, d_grid2, d_moved2, 1,1, mD,mH,mW, fD,fH,fW);
        cuda_cc_loss_3d(d_moved2, d_fixed, NULL, fD, fH, fW, 9, &result->ncc_loss);
        cudaFree(d_aff2); cudaFree(d_grid2); cudaFree(d_moved2);
    }

    cudaFree(d_fixed); cudaFree(d_moving);
    return 0;
}

/* ------------------------------------------------------------------ */
/* GPU affine registration                                             */
/* ------------------------------------------------------------------ */

int affine_register_gpu(const image_t *fixed, const image_t *moving,
                        const float init_rigid_34[3][4],
                        affine_opts_t opts, affine_result_t *result)
{
    memset(result, 0, sizeof(affine_result_t));

    int fD=fixed->data.shape[2], fH=fixed->data.shape[3], fW=fixed->data.shape[4];
    int mD=moving->data.shape[2], mH=moving->data.shape[3], mW=moving->data.shape[4];

    float *d_fixed, *d_moving;
    cudaMalloc(&d_fixed, (size_t)fD*fH*fW*sizeof(float));
    cudaMalloc(&d_moving, (size_t)mD*mH*mW*sizeof(float));
    cudaMemcpy(d_fixed, fixed->data.data, (size_t)fD*fH*fW*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_moving, moving->data.data, (size_t)mD*mH*mW*sizeof(float), cudaMemcpyHostToDevice);

    /* Initialize affine params [3][4] on CPU */
    float A[3][4];
    float center[3];
    for (int i = 0; i < 3; i++)
        center[i] = (float)fixed->meta.torch2phy.m[i][3];

    if (init_rigid_34) memcpy(A, init_rigid_34, 12*sizeof(float));
    else { memset(A, 0, sizeof(A)); A[0][0]=A[1][1]=A[2][2]=1; }

    /* Convert to around_center: t' = t - c + A*c */
    for (int i = 0; i < 3; i++) {
        A[i][3] = A[i][3] - center[i];
        for (int j = 0; j < 3; j++) A[i][3] += A[i][j] * center[j];
    }

    float adam_m[12]={0}, adam_v[12]={0};
    int adam_step = 0;
    float lr=opts.lr, beta1=0.9f, beta2=0.999f, eps=1e-8f;

    const mat44d *fixed_t2p = &fixed->meta.torch2phy;
    const mat44d *moving_p2t = &moving->meta.phy2torch;

    for (int si = 0; si < opts.n_scales; si++) {
        int scale = opts.scales[si], iters = opts.iterations[si];
        int dD=(scale>1)?fD/scale:fD, dH=(scale>1)?fH/scale:fH, dW=(scale>1)?fW/scale:fW;
        if(dD<8)dD=8;if(dH<8)dH=8;if(dW<8)dW=8;if(scale==1){dD=fD;dH=fH;dW=fW;}
        int mdD=(scale>1)?mD/scale:mD, mdH=(scale>1)?mH/scale:mH, mdW=(scale>1)?mW/scale:mW;
        if(mdD<8)mdD=8;if(mdH<8)mdH=8;if(mdW<8)mdW=8;if(scale==1){mdD=mD;mdH=mH;mdW=mW;}

        long spatial=(long)dD*dH*dW, n3=spatial*3, mSD=mdD*mdH*mdW;

        /* Downsample on GPU (matching Python affine.optimize):
         * Both images: downsample (NO extra blur on moving, unlike rigid) */
        float *d_fdown, *d_mdown;
        if (scale > 1) {
            cudaMalloc(&d_fdown, spatial*sizeof(float));
            cudaMalloc(&d_mdown, mSD*sizeof(float));
            if (opts.downsample_mode == DOWNSAMPLE_TRILINEAR) {
                cuda_blur_downsample(d_fixed, d_fdown, 1,1,fD,fH,fW,dD,dH,dW);
                cuda_blur_downsample(d_moving, d_mdown, 1,1,mD,mH,mW,mdD,mdH,mdW);
            } else {
                cuda_downsample_fft(d_fixed, d_fdown, 1,1,fD,fH,fW,dD,dH,dW);
                cuda_downsample_fft(d_moving, d_mdown, 1,1,mD,mH,mW,mdD,mdH,mdW);
            }
        } else { d_fdown=d_fixed; d_mdown=d_moving; mdD=mD;mdH=mH;mdW=mW; }

        float *d_aff, *d_grid, *d_moved, *d_grad_moved, *d_grad_grid;
        cudaMalloc(&d_aff, 12*sizeof(float));
        cudaMalloc(&d_grid, n3*sizeof(float));
        cudaMalloc(&d_moved, spatial*sizeof(float));
        cudaMalloc(&d_grad_moved, spatial*sizeof(float));
        cudaMalloc(&d_grad_grid, n3*sizeof(float));

        adam_step=0; memset(adam_m,0,sizeof(adam_m)); memset(adam_v,0,sizeof(adam_v));

        fprintf(stderr, "  Affine GPU scale %d: [%d,%d,%d] x %d iters\n", scale,dD,dH,dW,iters);
        float prev_loss=1e30f; int converge_count=0;

        for (int it = 0; it < iters; it++) {
            /* Build physical affine from around_center params */
            float phys44[4][4] = {{0}};
            for (int i=0;i<3;i++) {
                for (int j=0;j<3;j++) phys44[i][j] = A[i][j];
                phys44[i][3] = A[i][3] + center[i];
                for (int j=0;j<3;j++) phys44[i][3] -= A[i][j]*center[j];
            }
            phys44[3][3] = 1;

            mat44d pd,tm,cb;
            for(int i=0;i<4;i++) for(int j=0;j<4;j++) pd.m[i][j]=phys44[i][j];
            mat44d_mul(&tm, &pd, fixed_t2p);
            mat44d_mul(&cb, moving_p2t, &tm);
            float ha[12]; for(int i=0;i<3;i++) for(int j=0;j<4;j++) ha[i*4+j]=(float)cb.m[i][j];

            cudaMemcpy(d_aff, ha, 12*sizeof(float), cudaMemcpyHostToDevice);
            cuda_affine_grid_3d(d_aff, d_grid, 1, dD, dH, dW);
            cuda_grid_sample_3d_fwd(d_mdown, d_grid, d_moved, 1,1,mdD,mdH,mdW,dD,dH,dW);

            float loss;
            if (opts.loss_type == LOSS_MI) {
                int nbins = opts.mi_num_bins > 0 ? opts.mi_num_bins : 32;
                cuda_mi_loss_3d(d_moved, d_fdown, d_grad_moved, dD,dH,dW, nbins, &loss);
            } else {
                cuda_cc_loss_3d(d_moved, d_fdown, d_grad_moved, dD,dH,dW, opts.cc_kernel_size, &loss);
            }

            cuda_grid_sample_3d_bwd(d_grad_moved, d_mdown, d_grid,
                                    d_grad_grid, 1,1,mdD,mdH,mdW,dD,dH,dW);

            float dL_dA_comb[12];
            gpu_affine_grid_backward(d_grad_grid, dD, dH, dW, dL_dA_comb);

            /* Chain rule to physical space */
            mat44d dL44={{{0}}};
            for(int i=0;i<3;i++) for(int j=0;j<4;j++) dL44.m[i][j]=dL_dA_comb[i*4+j];
            mat44d p2tT,t2pT,t1,dLp;
            for(int i=0;i<4;i++) for(int j=0;j<4;j++){p2tT.m[i][j]=moving_p2t->m[j][i]; t2pT.m[i][j]=fixed_t2p->m[j][i];}
            mat44d_mul(&t1,&p2tT,&dL44); mat44d_mul(&dLp,&t1,&t2pT);

            /* dL/d(A_param) accounting for around_center */
            float dL_dA_param[12];
            for(int i=0;i<3;i++){
                for(int j=0;j<3;j++)
                    dL_dA_param[i*4+j] = (float)dLp.m[i][j] - (float)dLp.m[i][3]*center[j];
                dL_dA_param[i*4+3] = (float)dLp.m[i][3];
            }

            /* Adam on 12 params */
            adam_step++;
            float bc1=1-powf(beta1,(float)adam_step), bc2=1-powf(beta2,(float)adam_step);
            float ss=lr/bc1;
            float *params = &A[0][0];
            for(int k=0;k<12;k++){
                adam_m[k]=beta1*adam_m[k]+(1-beta1)*dL_dA_param[k];
                adam_v[k]=beta2*adam_v[k]+(1-beta2)*dL_dA_param[k]*dL_dA_param[k];
                params[k] -= ss * adam_m[k] / (sqrtf(adam_v[k]/bc2)+eps);
            }

            if(it%50==0||it==iters-1) fprintf(stderr,"    iter %d/%d loss=%.6f\n",it,iters,loss);
            if(fabsf(loss-prev_loss)<opts.tolerance){converge_count++;if(converge_count>=opts.max_tolerance_iters)break;}
            else converge_count=0;
            prev_loss=loss;
        }

        cudaFree(d_aff);cudaFree(d_grid);cudaFree(d_moved);cudaFree(d_grad_moved);cudaFree(d_grad_grid);
        if(scale>1){cudaFree(d_fdown);cudaFree(d_mdown);}
    }

    /* Extract physical affine */
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++) result->affine_mat[i][j]=A[i][j];
        result->affine_mat[i][3]=A[i][3]+center[i];
        for(int j=0;j<3;j++) result->affine_mat[i][3]-=A[i][j]*center[j];
    }

    /* Evaluate NCC at full res */
    {
        float *d_a,*d_g,*d_m2; float p44[4][4]={{0}};
        for(int i=0;i<3;i++) for(int j=0;j<4;j++) p44[i][j]=result->affine_mat[i][j];
        p44[3][3]=1;
        mat44d pd,tm,cb;
        for(int i=0;i<4;i++) for(int j=0;j<4;j++) pd.m[i][j]=p44[i][j];
        mat44d_mul(&tm,&pd,fixed_t2p); mat44d_mul(&cb,moving_p2t,&tm);
        float ha[12]; for(int i=0;i<3;i++) for(int j=0;j<4;j++) ha[i*4+j]=(float)cb.m[i][j];
        cudaMalloc(&d_a,12*sizeof(float)); cudaMalloc(&d_g,(size_t)fD*fH*fW*3*sizeof(float));
        cudaMalloc(&d_m2,(size_t)fD*fH*fW*sizeof(float));
        cudaMemcpy(d_a,ha,12*sizeof(float),cudaMemcpyHostToDevice);
        cuda_affine_grid_3d(d_a,d_g,1,fD,fH,fW);
        cuda_grid_sample_3d_fwd(d_moving,d_g,d_m2,1,1,mD,mH,mW,fD,fH,fW);
        cuda_cc_loss_3d(d_m2,d_fixed,NULL,fD,fH,fW,9,&result->ncc_loss);
        cudaFree(d_a);cudaFree(d_g);cudaFree(d_m2);
    }

    cudaFree(d_fixed); cudaFree(d_moving);
    return 0;
}

} /* extern "C" */
