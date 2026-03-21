/*
 * linear_webgpu.c - WebGPU rigid and affine registration
 *
 * Faithful clone of linear_gpu.cu. Images stay on WebGPU device.
 * Only scalar loss and small parameter gradients (7 or 12) come back
 * to CPU for the Adam update.
 *
 * FFT downsample uses the kissfft CPU fallback.
 */

#include "webgpu_context.h"
#include "webgpu_kernels.h"
#include "cfireants/tensor.h"
#include "cfireants/image.h"
#include "cfireants/registration.h"
#include "cfireants/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Quaternion utilities (same as linear_gpu.cu, on CPU)                */
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
    dR_dq[0][0][0]=0;          dR_dq[0][0][1]=-2*z*inv_n; dR_dq[0][0][2]=2*y*inv_n;
    dR_dq[0][1][0]=2*z*inv_n;  dR_dq[0][1][1]=0;          dR_dq[0][1][2]=-2*x*inv_n;
    dR_dq[0][2][0]=-2*y*inv_n; dR_dq[0][2][1]=2*x*inv_n;  dR_dq[0][2][2]=0;
    dR_dq[1][0][0]=0;          dR_dq[1][0][1]=2*y*inv_n;   dR_dq[1][0][2]=2*z*inv_n;
    dR_dq[1][1][0]=2*y*inv_n;  dR_dq[1][1][1]=-4*x*inv_n; dR_dq[1][1][2]=-2*w*inv_n;
    dR_dq[1][2][0]=2*z*inv_n;  dR_dq[1][2][1]=2*w*inv_n;  dR_dq[1][2][2]=-4*x*inv_n;
    dR_dq[2][0][0]=-4*y*inv_n; dR_dq[2][0][1]=2*x*inv_n;  dR_dq[2][0][2]=2*w*inv_n;
    dR_dq[2][1][0]=2*x*inv_n;  dR_dq[2][1][1]=0;          dR_dq[2][1][2]=2*z*inv_n;
    dR_dq[2][2][0]=-2*w*inv_n; dR_dq[2][2][1]=2*z*inv_n;  dR_dq[2][2][2]=-4*y*inv_n;
    dR_dq[3][0][0]=-4*z*inv_n; dR_dq[3][0][1]=-2*w*inv_n; dR_dq[3][0][2]=2*x*inv_n;
    dR_dq[3][1][0]=2*w*inv_n;  dR_dq[3][1][1]=-4*z*inv_n; dR_dq[3][1][2]=2*y*inv_n;
    dR_dq[3][2][0]=2*x*inv_n;  dR_dq[3][2][1]=2*y*inv_n;  dR_dq[3][2][2]=0;
}

/* ------------------------------------------------------------------ */
/* Affine grid backward on CPU (download grad_grid, reduce to 12)      */
/* ------------------------------------------------------------------ */

static void cpu_affine_grid_backward(const float *h_grad_grid, int D, int H, int W,
                                      float h_dL_dA[12]) {
    memset(h_dL_dA, 0, 12 * sizeof(float));
    for (int d = 0; d < D; d++) {
        float nz = (D > 1) ? (2.0f * d / (D - 1) - 1.0f) : 0.0f;
        for (int h = 0; h < H; h++) {
            float ny = (H > 1) ? (2.0f * h / (H - 1) - 1.0f) : 0.0f;
            for (int w = 0; w < W; w++) {
                float nx = (W > 1) ? (2.0f * w / (W - 1) - 1.0f) : 0.0f;
                int idx = ((d * H + h) * W + w) * 3;
                float coord[4] = {nx, ny, nz, 1.0f};
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 4; j++)
                        h_dL_dA[i * 4 + j] += h_grad_grid[idx + i] * coord[j];
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Helper: downsample image on CPU (download, FFT, upload)             */
/* ------------------------------------------------------------------ */

static WGPUBuffer downsample_to_webgpu(WGPUBuffer src_buf, int iD, int iH, int iW,
                                        int oD, int oH, int oW) {
    size_t in_sz = (size_t)iD * iH * iW * sizeof(float);
    size_t out_sz = (size_t)oD * oH * oW * sizeof(float);

    fprintf(stderr, "    downsample_to_webgpu: [%d,%d,%d]->[%d,%d,%d] in_sz=%zu out_sz=%zu\n",
            iD, iH, iW, oD, oH, oW, in_sz, out_sz);
    float *h_in = (float *)malloc(in_sz);
    float *h_out = (float *)malloc(out_sz);
    if (!h_in || !h_out) { fprintf(stderr, "    downsample: malloc failed!\n"); return NULL; }
    fprintf(stderr, "    downsample: reading buffer...\n");
    wgpu_read_buffer(src_buf, 0, h_in, in_sz);
    fprintf(stderr, "    downsample: read done, running FFT...\n");

    webgpu_downsample_fft(h_in, h_out, 1, 1, iD, iH, iW, oD, oH, oW);

    WGPUBuffer out_buf = wgpu_create_buffer(out_sz,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst, "ds");
    wgpu_write_buffer(out_buf, 0, h_out, out_sz);

    free(h_in);
    free(h_out);
    return out_buf;
}

/* Helper: apply per-axis Gaussian blur on CPU */
static void blur_moving_cpu(WGPUBuffer buf, int D, int H, int W,
                             int fD, int fH, int fW, int dD, int dH, int dW) {
    size_t sz = (size_t)D * H * W * sizeof(float);
    float *h = (float *)malloc(sz);
    wgpu_read_buffer(buf, 0, h, sz);

    /* Per-axis blur matching Python _smooth_image_not_mask */
    float sigmas[3] = {
        0.5f * (float)fD / (float)dD,
        0.5f * (float)fH / (float)dH,
        0.5f * (float)fW / (float)dW
    };

    /* Simple separable Gaussian blur on CPU */
    for (int axis = 0; axis < 3; axis++) {
        float sigma = sigmas[axis];
        int tail = (int)(2.0f * sigma + 0.5f);
        int klen = 2 * tail + 1;
        float *kern = (float *)malloc(klen * sizeof(float));
        float inv = 1.0f / (sigma * sqrtf(2.0f));
        float ksum = 0;
        for (int i = 0; i < klen; i++) {
            float x = (float)(i - tail);
            kern[i] = 0.5f * (erff((x+0.5f)*inv) - erff((x-0.5f)*inv));
            ksum += kern[i];
        }
        for (int i = 0; i < klen; i++) kern[i] /= ksum;

        float *tmp = (float *)malloc(sz);
        int r = klen / 2;
        int dims[3] = {D, H, W};
        int strides[3] = {H * W, W, 1};

        for (int i = 0; i < D * H * W; i++) {
            int coords[3] = { i / (H * W), (i / W) % H, i % W };
            float sum = 0;
            for (int k = 0; k < klen; k++) {
                int c = coords[axis] + k - r;
                if (c >= 0 && c < dims[axis]) {
                    int src_idx = i + (c - coords[axis]) * strides[axis];
                    sum += h[src_idx] * kern[k];
                }
            }
            tmp[i] = sum;
        }
        memcpy(h, tmp, sz);
        free(tmp);
        free(kern);
    }

    wgpu_write_buffer(buf, 0, h, sz);
    free(h);
}

/* ------------------------------------------------------------------ */
/* GPU rigid registration (WebGPU)                                     */
/* ------------------------------------------------------------------ */

int rigid_register_webgpu(const image_t *fixed, const image_t *moving,
                           const moments_result_t *moments_init,
                           rigid_opts_t opts, rigid_result_t *result)
{
    memset(result, 0, sizeof(rigid_result_t));

    int fD=fixed->data.shape[2], fH=fixed->data.shape[3], fW=fixed->data.shape[4];
    int mD=moving->data.shape[2], mH=moving->data.shape[3], mW=moving->data.shape[4];

    /* Upload images to WebGPU */
    fprintf(stderr, "  rigid_register_webgpu: uploading images [%d,%d,%d] + [%d,%d,%d]\n",
            fD, fH, fW, mD, mH, mW);
    WGPUBufferUsage usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    WGPUBuffer d_fixed = wgpu_create_buffer((size_t)fD*fH*fW*4, usage, "fixed");
    WGPUBuffer d_moving = wgpu_create_buffer((size_t)mD*mH*mW*4, usage, "moving");
    fprintf(stderr, "  rigid: buffers created, writing data...\n");
    wgpu_write_buffer(d_fixed, 0, fixed->data.data, (size_t)fD*fH*fW*4);
    wgpu_write_buffer(d_moving, 0, moving->data.data, (size_t)mD*mH*mW*4);
    fprintf(stderr, "  rigid: data uploaded\n");

    /* Initialize parameters on CPU (identical to CUDA) */
    float quat[4] = {1, 0, 0, 0};
    float transl[3] = {0, 0, 0};
    float moment[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    float center[3];
    for (int i = 0; i < 3; i++)
        center[i] = (float)fixed->meta.torch2phy.m[i][3];

    if (moments_init) {
        memcpy(moment, moments_init->Rf, sizeof(moment));
        for (int i = 0; i < 3; i++) {
            transl[i] = moments_init->tf[i] - center[i];
            for (int j = 0; j < 3; j++)
                transl[i] += moment[i][j] * center[j];
        }
    }

    float adam_m[7]={0}, adam_v[7]={0};
    int adam_step = 0;
    float lr = opts.lr, beta1=0.9f, beta2=0.999f, eps=1e-8f;

    const mat44d *fixed_t2p = &fixed->meta.torch2phy;
    const mat44d *moving_p2t = &moving->meta.phy2torch;

    fprintf(stderr, "  rigid: parameter init done, entering scale loop (n_scales=%d)\n", opts.n_scales);
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

        /* Downsample */
        fprintf(stderr, "  rigid: scale %d, downsampling...\n", scale);
        WGPUBuffer d_fdown, d_mdown;
        if (scale > 1) {
            d_fdown = downsample_to_webgpu(d_fixed, fD, fH, fW, dD, dH, dW);
            d_mdown = downsample_to_webgpu(d_moving, mD, mH, mW, mdD, mdH, mdW);
            /* Extra blur on moving for rigid (matching Python) */
            blur_moving_cpu(d_mdown, mdD, mdH, mdW, fD, fH, fW, dD, dH, dW);
        } else {
            d_fdown = d_fixed; d_mdown = d_moving;
            mdD = mD; mdH = mH; mdW = mW;
        }

        /* Allocate iteration buffers */
        WGPUBuffer d_aff = wgpu_create_buffer(12*4, usage, "aff");
        WGPUBuffer d_grid = wgpu_create_buffer(n3*4, usage, "grid");
        WGPUBuffer d_moved = wgpu_create_buffer(spatial*4, usage, "moved");
        WGPUBuffer d_grad_moved = wgpu_create_buffer(spatial*4, usage, "gmoved");
        WGPUBuffer d_grad_grid = wgpu_create_buffer(n3*4, usage, "ggrad");

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

            float rigid44[4][4] = {{0}};
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) rigid44[i][j] = RM[i][j];
                rigid44[i][3] = transl[i] + center[i];
                for (int j = 0; j < 3; j++) rigid44[i][3] -= RM[i][j] * center[j];
            }
            rigid44[3][3] = 1.0f;

            mat44d phys_d, tmp_m, comb;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) phys_d.m[i][j] = rigid44[i][j];
            mat44d_mul(&tmp_m, &phys_d, fixed_t2p);
            mat44d_mul(&comb, moving_p2t, &tmp_m);

            float h_aff[12];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    h_aff[i*4+j] = (float)comb.m[i][j];

            /* Upload affine, run forward pass on GPU (batched) */
            if (it == 0) fprintf(stderr, "  rigid: first iteration, dispatching shaders...\n");
            wgpu_write_buffer(d_aff, 0, h_aff, 12*4);
            wgpu_begin_batch();
            wgpu_affine_grid_3d(d_aff, d_grid, 1, dD, dH, dW);
            wgpu_grid_sample_3d_fwd(d_mdown, d_grid, d_moved,
                                     1, 1, mdD, mdH, mdW, dD, dH, dW);

            float loss;
            if (opts.loss_type == LOSS_MI) {
                int nbins = opts.mi_num_bins > 0 ? opts.mi_num_bins : 32;
                wgpu_mi_loss_3d_raw(d_moved, d_fdown, d_grad_moved,
                                     dD, dH, dW, nbins, &loss);
            } else {
                wgpu_cc_loss_3d_raw(d_moved, d_fdown, d_grad_moved,
                                     dD, dH, dW, opts.cc_kernel_size, &loss);
            }

            wgpu_begin_batch();
            wgpu_grid_sample_3d_bwd(d_grad_moved, d_mdown, d_grid,
                                     d_grad_grid, 1, 1, mdD, mdH, mdW, dD, dH, dW);
            /* affine_grid_backward does wgpu_read_buffer which auto-flushes */
            float dL_dA_comb[12];
            wgpu_affine_grid_backward(d_grad_grid, dD, dH, dW, dL_dA_comb);

            /* Chain rule: dL/d(phys) */
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

            float dL_dRM[3][3], dL_dt[3];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++)
                    dL_dRM[i][j] = (float)dL_dphys.m[i][j] - (float)dL_dphys.m[i][3] * center[j];
                dL_dt[i] = (float)dL_dphys.m[i][3];
            }

            float dL_dR[3][3];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) {
                    dL_dR[i][j] = 0;
                    for (int k = 0; k < 3; k++)
                        dL_dR[i][j] += dL_dRM[i][k] * moment[j][k];
                }

            float dR_dq[4][3][3];
            quat_rotmat_jacobian(quat, dR_dq);
            float dL_dq[4] = {0};
            for (int k = 0; k < 4; k++)
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        dL_dq[k] += dL_dR[i][j] * dR_dq[k][i][j];

            float grad7[7];
            for (int k = 0; k < 4; k++) grad7[k] = dL_dq[k];
            for (int k = 0; k < 3; k++) grad7[4+k] = dL_dt[k];

            /* Adam on CPU */
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

        wgpuBufferRelease(d_aff); wgpuBufferRelease(d_grid);
        wgpuBufferRelease(d_moved); wgpuBufferRelease(d_grad_moved);
        wgpuBufferRelease(d_grad_grid);
        if (scale > 1) { wgpuBufferRelease(d_fdown); wgpuBufferRelease(d_mdown); }
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

    /* Evaluate NCC at full resolution */
    {
        WGPUBuffer d_aff2 = wgpu_create_buffer(12*4, usage, "a2");
        WGPUBuffer d_grid2 = wgpu_create_buffer((size_t)fD*fH*fW*3*4, usage, "g2");
        WGPUBuffer d_moved2 = wgpu_create_buffer((size_t)fD*fH*fW*4, usage, "m2");

        float phys44[4][4] = {{0}};
        for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) phys44[i][j] = result->rigid_mat[i][j];
        phys44[3][3] = 1.0f;
        mat44d pd, tm, cb;
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) pd.m[i][j] = phys44[i][j];
        mat44d_mul(&tm, &pd, fixed_t2p);
        mat44d_mul(&cb, moving_p2t, &tm);
        float ha[12]; for (int i=0;i<3;i++) for (int j=0;j<4;j++) ha[i*4+j]=(float)cb.m[i][j];
        wgpu_write_buffer(d_aff2, 0, ha, 12*4);
        wgpu_affine_grid_3d(d_aff2, d_grid2, 1, fD, fH, fW);
        wgpu_grid_sample_3d_fwd(d_moving, d_grid2, d_moved2, 1,1, mD,mH,mW, fD,fH,fW);
        wgpu_cc_loss_3d_raw(d_moved2, d_fixed, NULL, fD, fH, fW, 9, &result->ncc_loss);
        wgpuBufferRelease(d_aff2); wgpuBufferRelease(d_grid2); wgpuBufferRelease(d_moved2);
    }

    wgpuBufferRelease(d_fixed); wgpuBufferRelease(d_moving);
    return 0;
}

/* ------------------------------------------------------------------ */
/* GPU affine registration (WebGPU)                                    */
/* ------------------------------------------------------------------ */

int affine_register_webgpu(const image_t *fixed, const image_t *moving,
                            const float init_rigid_34[3][4],
                            affine_opts_t opts, affine_result_t *result)
{
    memset(result, 0, sizeof(affine_result_t));

    int fD=fixed->data.shape[2], fH=fixed->data.shape[3], fW=fixed->data.shape[4];
    int mD=moving->data.shape[2], mH=moving->data.shape[3], mW=moving->data.shape[4];

    WGPUBufferUsage usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    WGPUBuffer d_fixed = wgpu_create_buffer((size_t)fD*fH*fW*4, usage, "fixed");
    WGPUBuffer d_moving = wgpu_create_buffer((size_t)mD*mH*mW*4, usage, "moving");
    wgpu_write_buffer(d_fixed, 0, fixed->data.data, (size_t)fD*fH*fW*4);
    wgpu_write_buffer(d_moving, 0, moving->data.data, (size_t)mD*mH*mW*4);

    float A[3][4];
    float center[3];
    for (int i = 0; i < 3; i++)
        center[i] = (float)fixed->meta.torch2phy.m[i][3];

    if (init_rigid_34) memcpy(A, init_rigid_34, 12*sizeof(float));
    else { memset(A, 0, sizeof(A)); A[0][0]=A[1][1]=A[2][2]=1; }

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

        long spatial=(long)dD*dH*dW, n3=spatial*3, mSD=(long)mdD*mdH*mdW;

        WGPUBuffer d_fdown, d_mdown;
        if (scale > 1) {
            d_fdown = downsample_to_webgpu(d_fixed, fD, fH, fW, dD, dH, dW);
            d_mdown = downsample_to_webgpu(d_moving, mD, mH, mW, mdD, mdH, mdW);
            /* Affine: NO extra blur on moving (unlike rigid) */
        } else { d_fdown=d_fixed; d_mdown=d_moving; mdD=mD;mdH=mH;mdW=mW; }

        WGPUBuffer d_aff = wgpu_create_buffer(12*4, usage, "aff");
        WGPUBuffer d_grid = wgpu_create_buffer(n3*4, usage, "grid");
        WGPUBuffer d_moved = wgpu_create_buffer(spatial*4, usage, "moved");
        WGPUBuffer d_grad_moved = wgpu_create_buffer(spatial*4, usage, "gmov");
        WGPUBuffer d_grad_grid = wgpu_create_buffer(n3*4, usage, "ggrid");

        adam_step=0; memset(adam_m,0,sizeof(adam_m)); memset(adam_v,0,sizeof(adam_v));

        fprintf(stderr, "  Affine GPU scale %d: [%d,%d,%d] x %d iters\n", scale,dD,dH,dW,iters);
        float prev_loss=1e30f; int converge_count=0;

        for (int it = 0; it < iters; it++) {
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

            wgpu_write_buffer(d_aff, 0, ha, 12*4);
            wgpu_affine_grid_3d(d_aff, d_grid, 1, dD, dH, dW);
            wgpu_grid_sample_3d_fwd(d_mdown, d_grid, d_moved, 1,1,mdD,mdH,mdW,dD,dH,dW);

            float loss;
            if (opts.loss_type == LOSS_MI) {
                int nbins = opts.mi_num_bins > 0 ? opts.mi_num_bins : 32;
                wgpu_mi_loss_3d_raw(d_moved, d_fdown, d_grad_moved, dD,dH,dW, nbins, &loss);
            } else {
                wgpu_cc_loss_3d_raw(d_moved, d_fdown, d_grad_moved, dD,dH,dW, opts.cc_kernel_size, &loss);
            }

            wgpu_grid_sample_3d_bwd(d_grad_moved, d_mdown, d_grid,
                                     d_grad_grid, 1,1,mdD,mdH,mdW,dD,dH,dW);

            float *h_gg = (float *)malloc(n3 * 4);
            wgpu_read_buffer(d_grad_grid, 0, h_gg, n3 * 4);
            float dL_dA_comb[12];
            cpu_affine_grid_backward(h_gg, dD, dH, dW, dL_dA_comb);
            free(h_gg);

            mat44d dL44={{{0}}};
            for(int i=0;i<3;i++) for(int j=0;j<4;j++) dL44.m[i][j]=dL_dA_comb[i*4+j];
            mat44d p2tT,t2pT,t1,dLp;
            for(int i=0;i<4;i++) for(int j=0;j<4;j++){p2tT.m[i][j]=moving_p2t->m[j][i]; t2pT.m[i][j]=fixed_t2p->m[j][i];}
            mat44d_mul(&t1,&p2tT,&dL44); mat44d_mul(&dLp,&t1,&t2pT);

            float dL_dA_param[12];
            for(int i=0;i<3;i++){
                for(int j=0;j<3;j++)
                    dL_dA_param[i*4+j] = (float)dLp.m[i][j] - (float)dLp.m[i][3]*center[j];
                dL_dA_param[i*4+3] = (float)dLp.m[i][3];
            }

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

        wgpuBufferRelease(d_aff);wgpuBufferRelease(d_grid);
        wgpuBufferRelease(d_moved);wgpuBufferRelease(d_grad_moved);wgpuBufferRelease(d_grad_grid);
        if(scale>1){wgpuBufferRelease(d_fdown);wgpuBufferRelease(d_mdown);}
    }

    /* Extract physical affine */
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++) result->affine_mat[i][j]=A[i][j];
        result->affine_mat[i][3]=A[i][3]+center[i];
        for(int j=0;j<3;j++) result->affine_mat[i][3]-=A[i][j]*center[j];
    }

    /* Evaluate NCC at full res */
    {
        WGPUBuffer d_a = wgpu_create_buffer(12*4, usage, "a2");
        WGPUBuffer d_g = wgpu_create_buffer((size_t)fD*fH*fW*3*4, usage, "g2");
        WGPUBuffer d_m2 = wgpu_create_buffer((size_t)fD*fH*fW*4, usage, "m2");
        float p44[4][4]={{0}};
        for(int i=0;i<3;i++) for(int j=0;j<4;j++) p44[i][j]=result->affine_mat[i][j];
        p44[3][3]=1;
        mat44d pd,tm,cb;
        for(int i=0;i<4;i++) for(int j=0;j<4;j++) pd.m[i][j]=p44[i][j];
        mat44d_mul(&tm,&pd,fixed_t2p); mat44d_mul(&cb,moving_p2t,&tm);
        float ha[12]; for(int i=0;i<3;i++) for(int j=0;j<4;j++) ha[i*4+j]=(float)cb.m[i][j];
        wgpu_write_buffer(d_a, 0, ha, 12*4);
        wgpu_affine_grid_3d(d_a,d_g,1,fD,fH,fW);
        wgpu_grid_sample_3d_fwd(d_moving,d_g,d_m2,1,1,mD,mH,mW,fD,fH,fW);
        wgpu_cc_loss_3d_raw(d_m2,d_fixed,NULL,fD,fH,fW,9,&result->ncc_loss);
        wgpuBufferRelease(d_a);wgpuBufferRelease(d_g);wgpuBufferRelease(d_m2);
    }

    wgpuBufferRelease(d_fixed); wgpuBufferRelease(d_moving);
    return 0;
}
