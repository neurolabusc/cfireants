/*
 * linear_metal.m - Metal-accelerated rigid and affine registration
 *
 * Faithful translation of backend/cuda/linear_gpu.cu for the Metal backend.
 * Uses Metal shared-memory buffers (MTLResourceStorageModeShared) so all
 * GPU data is CPU-accessible — no explicit H2D/D2H copies needed.
 *
 * Images stay on GPU. Only scalar loss and small parameter gradients
 * (7 for rigid, 12 for affine) come back to CPU for the Adam update.
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
/* Metal buffer allocation helper                                      */
/* ------------------------------------------------------------------ */

/* Allocate a Metal shared-memory buffer and register it for dispatch.
   Returns the CPU-accessible pointer (buf.contents). */
static float *metal_alloc_buf(size_t bytes, id<MTLBuffer> *out_buf) {
    id<MTLBuffer> buf = [g_metal.device newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
    if (!buf) return NULL;
    float *ptr = (float *)buf.contents;
    metal_register_buffer(ptr, (__bridge void *)buf, bytes);
    *out_buf = buf;
    return ptr;
}

/* Free a Metal buffer and unregister it. */
static void metal_free_buf(float *ptr, id<MTLBuffer> buf) {
    if (ptr) metal_unregister_buffer(ptr);
    /* ARC releases the MTLBuffer when buf goes out of scope */
    (void)buf;
}

/* ------------------------------------------------------------------ */
/* Affine grid backward: GPU reduction + CPU final sum                 */
/* ------------------------------------------------------------------ */

/* Dispatches the affine_grid_bwd Metal shader to compute partial sums,
   then sums them on CPU to produce the final 12-element gradient.

   grad_grid: GPU buffer pointer [D*H*W*3]
   D, H, W:  grid dimensions
   h_dL_dA:  output 12-element gradient on CPU */
static void metal_affine_grid_backward(const float *d_grad_grid, int D, int H, int W,
                                        float h_dL_dA[12]) {
    int total = D * H * W;
    int blocks = (total + 255) / 256;

    /* Allocate partial sums buffer on GPU */
    id<MTLBuffer> partial_buf;
    float *d_partial = metal_alloc_buf((size_t)blocks * 12 * sizeof(float), &partial_buf);
    if (!d_partial) {
        fprintf(stderr, "metal_affine_grid_backward: failed to allocate partial buffer\n");
        memset(h_dL_dA, 0, 12 * sizeof(float));
        return;
    }

    /* Dispatch the affine_grid_bwd kernel */
    struct { uint32_t D, H, W, _pad; } params = {
        (uint32_t)D, (uint32_t)H, (uint32_t)W, 0
    };
    void *pipeline = metal_get_pipeline("affine_grid_bwd");
    if (!pipeline) {
        fprintf(stderr, "metal_affine_grid_backward: failed to get pipeline\n");
        memset(h_dL_dA, 0, 12 * sizeof(float));
        metal_free_buf(d_partial, partial_buf);
        return;
    }

    const void *bufs[2] = { d_grad_grid, d_partial };
    metal_dispatch(pipeline, bufs, NULL, 2, &params, sizeof(params), (uint32_t)total);

    /* GPU work is complete after metal_dispatch (it waits). Read partial sums. */
    memset(h_dL_dA, 0, 12 * sizeof(float));
    for (int b = 0; b < blocks; b++)
        for (int k = 0; k < 12; k++)
            h_dL_dA[k] += d_partial[b * 12 + k];

    metal_free_buf(d_partial, partial_buf);
}

/* ------------------------------------------------------------------ */
/* Quaternion utilities (same as rigid.c / linear_gpu.cu, on CPU)      */
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
/* ------------------------------------------------------------------ */
/* GPU rigid registration                                              */
/* ------------------------------------------------------------------ */

int rigid_register_metal(const image_t *fixed, const image_t *moving,
                          const moments_result_t *moments_init,
                          rigid_opts_t opts, rigid_result_t *result)
{
    @autoreleasepool {

    memset(result, 0, sizeof(rigid_result_t));

    int fD=fixed->data.shape[2], fH=fixed->data.shape[3], fW=fixed->data.shape[4];
    int mD=moving->data.shape[2], mH=moving->data.shape[3], mW=moving->data.shape[4];

    /* Allocate images in Metal shared-memory buffers */
    id<MTLBuffer> fixed_buf, moving_buf;
    float *d_fixed = metal_alloc_buf((size_t)fD*fH*fW*sizeof(float), &fixed_buf);
    float *d_moving = metal_alloc_buf((size_t)mD*mH*mW*sizeof(float), &moving_buf);
    if (!d_fixed || !d_moving) {
        fprintf(stderr, "rigid_register_metal: buffer allocation failed\n");
        if (d_fixed)  metal_free_buf(d_fixed, fixed_buf);
        if (d_moving) metal_free_buf(d_moving, moving_buf);
        return -1;
    }

    /* Copy image data to shared buffers (no cudaMemcpy needed — just memcpy) */
    memcpy(d_fixed, fixed->data.data, (size_t)fD*fH*fW*sizeof(float));
    memcpy(d_moving, moving->data.data, (size_t)mD*mH*mW*sizeof(float));

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
        long mSDown = (long)mdD * mdH * mdW;

        /* Downsample on GPU (matching Python rigid.optimize):
         * fixed: downsample (FFT or trilinear)
         * moving: downsample + extra Gaussian blur (_smooth_image_not_mask) */
        id<MTLBuffer> fdown_buf = nil, mdown_buf = nil;
        float *d_fdown, *d_mdown;
        if (scale > 1) {
            d_fdown = metal_alloc_buf(spatial*sizeof(float), &fdown_buf);
            d_mdown = metal_alloc_buf(mSDown*sizeof(float), &mdown_buf);
            if (opts.downsample_mode == DOWNSAMPLE_TRILINEAR) {
                metal_blur_downsample(d_fixed, d_fdown, 1, 1, fD, fH, fW, dD, dH, dW);
                metal_blur_downsample(d_moving, d_mdown, 1, 1, mD, mH, mW, mdD, mdH, mdW);
            } else {
                metal_downsample_fft(d_fixed, d_fdown, 1, 1, fD, fH, fW, dD, dH, dW);
                metal_downsample_fft(d_moving, d_mdown, 1, 1, mD, mH, mW, mdD, mdH, mdW);
            }

            /* Extra Gaussian blur on moving (matching Python rigid line 345) */
            metal_blur_volume(d_mdown, mdD, mdH, mdW,
                              0.5f * (float)fD / (float)dD,
                              0.5f * (float)fH / (float)dH,
                              0.5f * (float)fW / (float)dW);
        } else {
            d_fdown = d_fixed;
            d_mdown = d_moving;
            mdD = mD; mdH = mH; mdW = mW;
        }

        /* GPU buffers for iteration */
        id<MTLBuffer> aff_buf, grid_buf, moved_buf, gmoved_buf, ggrid_buf;
        float *d_aff       = metal_alloc_buf(12*sizeof(float), &aff_buf);
        float *d_grid      = metal_alloc_buf(n3*sizeof(float), &grid_buf);
        float *d_moved     = metal_alloc_buf(spatial*sizeof(float), &moved_buf);
        float *d_grad_moved= metal_alloc_buf(spatial*sizeof(float), &gmoved_buf);
        float *d_grad_grid = metal_alloc_buf(n3*sizeof(float), &ggrid_buf);

        /* Reset Adam per scale */
        adam_step = 0;
        memset(adam_m, 0, sizeof(adam_m));
        memset(adam_v, 0, sizeof(adam_v));

        fprintf(stderr, "  Rigid Metal scale %d: [%d,%d,%d] x %d iters\n",
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

            /* Write affine to shared buffer (no cudaMemcpy — just memcpy) */
            memcpy(d_aff, h_aff, 12*sizeof(float));

            /* Generate grid, sample, loss — all on GPU */
            metal_affine_grid_3d(d_aff, d_grid, 1, dD, dH, dW);
            metal_grid_sample_3d_fwd(d_mdown, d_grid, d_moved,
                                      1, 1, mdD, mdH, mdW, dD, dH, dW);

            float loss;
            if (opts.loss_type == LOSS_MI) {
                int nbins = opts.mi_num_bins > 0 ? opts.mi_num_bins : 32;
                metal_mi_loss_3d(d_moved, d_fdown, d_grad_moved,
                                  dD, dH, dW, nbins, &loss);
            } else {
                metal_cc_loss_3d(d_moved, d_fdown, d_grad_moved, dD, dH, dW,
                                  opts.cc_kernel_size, &loss);
            }

            metal_grid_sample_3d_bwd(d_grad_moved, d_mdown, d_grid,
                                      d_grad_grid, 1, 1, mdD, mdH, mdW, dD, dH, dW);

            /* Affine grid backward: dL/dA [12 values] — GPU reduction, result on CPU */
            float dL_dA_comb[12];
            metal_affine_grid_backward(d_grad_grid, dD, dH, dW, dL_dA_comb);

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

        metal_free_buf(d_aff, aff_buf);
        metal_free_buf(d_grid, grid_buf);
        metal_free_buf(d_moved, moved_buf);
        metal_free_buf(d_grad_moved, gmoved_buf);
        metal_free_buf(d_grad_grid, ggrid_buf);
        if (scale > 1) {
            metal_free_buf(d_fdown, fdown_buf);
            metal_free_buf(d_mdown, mdown_buf);
        }
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
        id<MTLBuffer> aff2_buf, grid2_buf, moved2_buf;
        float *d_aff2  = metal_alloc_buf(12*sizeof(float), &aff2_buf);
        float *d_grid2 = metal_alloc_buf((size_t)fD*fH*fW*3*sizeof(float), &grid2_buf);
        float *d_moved2= metal_alloc_buf((size_t)fD*fH*fW*sizeof(float), &moved2_buf);

        float phys44[4][4] = {{0}};
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++) phys44[i][j] = result->rigid_mat[i][j];
        phys44[3][3] = 1.0f;
        mat44d pd, tm, cb;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) pd.m[i][j] = phys44[i][j];
        mat44d_mul(&tm, &pd, fixed_t2p);
        mat44d_mul(&cb, moving_p2t, &tm);
        float ha[12];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++) ha[i*4+j] = (float)cb.m[i][j];

        memcpy(d_aff2, ha, 12*sizeof(float));
        metal_affine_grid_3d(d_aff2, d_grid2, 1, fD, fH, fW);
        metal_grid_sample_3d_fwd(d_moving, d_grid2, d_moved2, 1, 1, mD, mH, mW, fD, fH, fW);
        metal_cc_loss_3d(d_moved2, d_fixed, NULL, fD, fH, fW, 9, &result->ncc_loss);

        metal_free_buf(d_aff2, aff2_buf);
        metal_free_buf(d_grid2, grid2_buf);
        metal_free_buf(d_moved2, moved2_buf);
    }

    metal_free_buf(d_fixed, fixed_buf);
    metal_free_buf(d_moving, moving_buf);

    } /* @autoreleasepool */
    return 0;
}

/* ------------------------------------------------------------------ */
/* GPU affine registration                                             */
/* ------------------------------------------------------------------ */

int affine_register_metal(const image_t *fixed, const image_t *moving,
                           const float init_rigid_34[3][4],
                           affine_opts_t opts, affine_result_t *result)
{
    @autoreleasepool {

    memset(result, 0, sizeof(affine_result_t));

    int fD=fixed->data.shape[2], fH=fixed->data.shape[3], fW=fixed->data.shape[4];
    int mD=moving->data.shape[2], mH=moving->data.shape[3], mW=moving->data.shape[4];

    /* Allocate images in Metal shared-memory buffers */
    id<MTLBuffer> fixed_buf, moving_buf;
    float *d_fixed = metal_alloc_buf((size_t)fD*fH*fW*sizeof(float), &fixed_buf);
    float *d_moving = metal_alloc_buf((size_t)mD*mH*mW*sizeof(float), &moving_buf);
    if (!d_fixed || !d_moving) {
        fprintf(stderr, "affine_register_metal: buffer allocation failed\n");
        if (d_fixed)  metal_free_buf(d_fixed, fixed_buf);
        if (d_moving) metal_free_buf(d_moving, moving_buf);
        return -1;
    }

    memcpy(d_fixed, fixed->data.data, (size_t)fD*fH*fW*sizeof(float));
    memcpy(d_moving, moving->data.data, (size_t)mD*mH*mW*sizeof(float));

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
        if(dD<8)dD=8; if(dH<8)dH=8; if(dW<8)dW=8; if(scale==1){dD=fD;dH=fH;dW=fW;}
        int mdD=(scale>1)?mD/scale:mD, mdH=(scale>1)?mH/scale:mH, mdW=(scale>1)?mW/scale:mW;
        if(mdD<8)mdD=8; if(mdH<8)mdH=8; if(mdW<8)mdW=8; if(scale==1){mdD=mD;mdH=mH;mdW=mW;}

        long spatial=(long)dD*dH*dW, n3=spatial*3, mSD=(long)mdD*mdH*mdW;

        /* Downsample on GPU (matching Python affine.optimize):
         * Both images: downsample (NO extra blur on moving, unlike rigid) */
        id<MTLBuffer> fdown_buf = nil, mdown_buf = nil;
        float *d_fdown, *d_mdown;
        if (scale > 1) {
            d_fdown = metal_alloc_buf(spatial*sizeof(float), &fdown_buf);
            d_mdown = metal_alloc_buf(mSD*sizeof(float), &mdown_buf);
            if (opts.downsample_mode == DOWNSAMPLE_TRILINEAR) {
                metal_blur_downsample(d_fixed, d_fdown, 1, 1, fD, fH, fW, dD, dH, dW);
                metal_blur_downsample(d_moving, d_mdown, 1, 1, mD, mH, mW, mdD, mdH, mdW);
            } else {
                metal_downsample_fft(d_fixed, d_fdown, 1, 1, fD, fH, fW, dD, dH, dW);
                metal_downsample_fft(d_moving, d_mdown, 1, 1, mD, mH, mW, mdD, mdH, mdW);
            }
        } else {
            d_fdown = d_fixed; d_mdown = d_moving;
            mdD = mD; mdH = mH; mdW = mW;
        }

        id<MTLBuffer> aff_buf, grid_buf, moved_buf, gmoved_buf, ggrid_buf;
        float *d_aff       = metal_alloc_buf(12*sizeof(float), &aff_buf);
        float *d_grid      = metal_alloc_buf(n3*sizeof(float), &grid_buf);
        float *d_moved     = metal_alloc_buf(spatial*sizeof(float), &moved_buf);
        float *d_grad_moved= metal_alloc_buf(spatial*sizeof(float), &gmoved_buf);
        float *d_grad_grid = metal_alloc_buf(n3*sizeof(float), &ggrid_buf);

        adam_step = 0;
        memset(adam_m, 0, sizeof(adam_m));
        memset(adam_v, 0, sizeof(adam_v));

        fprintf(stderr, "  Affine Metal scale %d: [%d,%d,%d] x %d iters\n", scale, dD, dH, dW, iters);
        float prev_loss = 1e30f;
        int converge_count = 0;

        for (int it = 0; it < iters; it++) {
            /* Build physical affine from around_center params */
            float phys44[4][4] = {{0}};
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) phys44[i][j] = A[i][j];
                phys44[i][3] = A[i][3] + center[i];
                for (int j = 0; j < 3; j++) phys44[i][3] -= A[i][j]*center[j];
            }
            phys44[3][3] = 1;

            mat44d pd, tm, cb;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) pd.m[i][j] = phys44[i][j];
            mat44d_mul(&tm, &pd, fixed_t2p);
            mat44d_mul(&cb, moving_p2t, &tm);
            float ha[12];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++) ha[i*4+j] = (float)cb.m[i][j];

            /* Write affine to shared buffer */
            memcpy(d_aff, ha, 12*sizeof(float));

            metal_affine_grid_3d(d_aff, d_grid, 1, dD, dH, dW);
            metal_grid_sample_3d_fwd(d_mdown, d_grid, d_moved, 1, 1, mdD, mdH, mdW, dD, dH, dW);

            float loss;
            if (opts.loss_type == LOSS_MI) {
                int nbins = opts.mi_num_bins > 0 ? opts.mi_num_bins : 32;
                metal_mi_loss_3d(d_moved, d_fdown, d_grad_moved,
                                  dD, dH, dW, nbins, &loss);
            } else {
                metal_cc_loss_3d(d_moved, d_fdown, d_grad_moved, dD, dH, dW,
                                  opts.cc_kernel_size, &loss);
            }

            metal_grid_sample_3d_bwd(d_grad_moved, d_mdown, d_grid,
                                      d_grad_grid, 1, 1, mdD, mdH, mdW, dD, dH, dW);

            float dL_dA_comb[12];
            metal_affine_grid_backward(d_grad_grid, dD, dH, dW, dL_dA_comb);

            /* Chain rule to physical space */
            mat44d dL44 = {{{0}}};
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++) dL44.m[i][j] = dL_dA_comb[i*4+j];
            mat44d p2tT, t2pT, t1, dLp;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) {
                    p2tT.m[i][j] = moving_p2t->m[j][i];
                    t2pT.m[i][j] = fixed_t2p->m[j][i];
                }
            mat44d_mul(&t1, &p2tT, &dL44);
            mat44d_mul(&dLp, &t1, &t2pT);

            /* dL/d(A_param) accounting for around_center */
            float dL_dA_param[12];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++)
                    dL_dA_param[i*4+j] = (float)dLp.m[i][j] - (float)dLp.m[i][3]*center[j];
                dL_dA_param[i*4+3] = (float)dLp.m[i][3];
            }

            /* Adam on 12 params */
            adam_step++;
            float bc1 = 1-powf(beta1, (float)adam_step);
            float bc2 = 1-powf(beta2, (float)adam_step);
            float ss = lr/bc1;
            float *params = &A[0][0];
            for (int k = 0; k < 12; k++) {
                adam_m[k] = beta1*adam_m[k] + (1-beta1)*dL_dA_param[k];
                adam_v[k] = beta2*adam_v[k] + (1-beta2)*dL_dA_param[k]*dL_dA_param[k];
                params[k] -= ss * adam_m[k] / (sqrtf(adam_v[k]/bc2)+eps);
            }

            if (it % 50 == 0 || it == iters - 1)
                fprintf(stderr, "    iter %d/%d loss=%.6f\n", it, iters, loss);

            if (fabsf(loss - prev_loss) < opts.tolerance) {
                converge_count++;
                if (converge_count >= opts.max_tolerance_iters) break;
            } else converge_count = 0;
            prev_loss = loss;
        }

        metal_free_buf(d_aff, aff_buf);
        metal_free_buf(d_grid, grid_buf);
        metal_free_buf(d_moved, moved_buf);
        metal_free_buf(d_grad_moved, gmoved_buf);
        metal_free_buf(d_grad_grid, ggrid_buf);
        if (scale > 1) {
            metal_free_buf(d_fdown, fdown_buf);
            metal_free_buf(d_mdown, mdown_buf);
        }
    }

    /* Extract physical affine */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) result->affine_mat[i][j] = A[i][j];
        result->affine_mat[i][3] = A[i][3] + center[i];
        for (int j = 0; j < 3; j++) result->affine_mat[i][3] -= A[i][j]*center[j];
    }

    /* Evaluate NCC at full resolution */
    {
        id<MTLBuffer> a_buf, g_buf, m2_buf;
        float *d_a  = metal_alloc_buf(12*sizeof(float), &a_buf);
        float *d_g  = metal_alloc_buf((size_t)fD*fH*fW*3*sizeof(float), &g_buf);
        float *d_m2 = metal_alloc_buf((size_t)fD*fH*fW*sizeof(float), &m2_buf);

        float p44[4][4] = {{0}};
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++) p44[i][j] = result->affine_mat[i][j];
        p44[3][3] = 1;
        mat44d pd, tm, cb;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) pd.m[i][j] = p44[i][j];
        mat44d_mul(&tm, &pd, fixed_t2p);
        mat44d_mul(&cb, moving_p2t, &tm);
        float ha[12];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++) ha[i*4+j] = (float)cb.m[i][j];

        memcpy(d_a, ha, 12*sizeof(float));
        metal_affine_grid_3d(d_a, d_g, 1, fD, fH, fW);
        metal_grid_sample_3d_fwd(d_moving, d_g, d_m2, 1, 1, mD, mH, mW, fD, fH, fW);
        metal_cc_loss_3d(d_m2, d_fixed, NULL, fD, fH, fW, 9, &result->ncc_loss);

        metal_free_buf(d_a, a_buf);
        metal_free_buf(d_g, g_buf);
        metal_free_buf(d_m2, m2_buf);
    }

    metal_free_buf(d_fixed, fixed_buf);
    metal_free_buf(d_moving, moving_buf);

    } /* @autoreleasepool */
    return 0;
}
