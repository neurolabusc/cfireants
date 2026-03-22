/*
 * syn.c - Symmetric Normalization (SyN) deformable registration
 *
 * SyN optimizes two displacement fields (matching GPU pipeline):
 *   fwd_warp: maps fixed → midpoint (applied to moving image via affine)
 *   rev_warp: maps fixed → midpoint (applied to fixed image directly)
 *
 * Per iteration (WarpAdam with compositive update):
 *   1. sampling_grids = base + warp (no pre-smoothing)
 *   2. moved = grid_sample(moving_blur, fwd_sg)
 *   3. fixed_warped = grid_sample(fixed_down, rev_sg)
 *   4. fused CC loss → gradients for both pred and target
 *   5. grid_sample_backward for both warps
 *   6. Smooth gradients
 *   7. WarpAdam: moments, direction, normalize, scale
 *   8. Compositive update + smooth → new warp
 */

#include "cfireants/registration.h"
#include "cfireants/interpolator.h"
#include "cfireants/losses.h"
#include "cfireants/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Helper: permute [1,D,H,W,3] <-> [1,3,D,H,W] */
static void permute_dhw3_to_3dhw(const float *src, float *dst,
                                  int D, int H, int W) {
    for (int d = 0; d < D; d++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                for (int c = 0; c < 3; c++)
                    dst[((size_t)c*D+d)*H*W + h*W + w] =
                        src[((size_t)d*H+h)*W*3 + w*3 + c];
}

static void permute_3dhw_to_dhw3(const float *src, float *dst,
                                  int D, int H, int W) {
    for (int d = 0; d < D; d++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                for (int c = 0; c < 3; c++)
                    dst[((size_t)d*H+h)*W*3 + w*3 + c] =
                        src[((size_t)c*D+d)*H*W + h*W + w];
}

/* Resize a [1,D,H,W,3] displacement field via trilinear interpolation */
static int resize_disp(const tensor_t *in, tensor_t *out, int nD, int nH, int nW) {
    int oD = in->shape[1], oH = in->shape[2], oW = in->shape[3];
    tensor_t img;
    int is[5] = {1, 3, oD, oH, oW};
    tensor_alloc(&img, 5, is, DTYPE_FLOAT32, DEVICE_CPU);
    permute_dhw3_to_3dhw(tensor_data_f32(in), tensor_data_f32(&img), oD, oH, oW);
    tensor_t resized;
    int rs[5] = {1, 3, nD, nH, nW};
    tensor_alloc(&resized, 5, rs, DTYPE_FLOAT32, DEVICE_CPU);
    cpu_trilinear_resize(&img, &resized, 1);
    tensor_free(&img);
    int os[5] = {1, nD, nH, nW, 3};
    tensor_alloc(out, 5, os, DTYPE_FLOAT32, DEVICE_CPU);
    permute_3dhw_to_dhw3(tensor_data_f32(&resized), tensor_data_f32(out), nD, nH, nW);
    tensor_free(&resized);
    return 0;
}

/* Separable Gaussian blur on [D,H,W,3] interleaved displacement field (in-place) */
static void blur_disp_dhw3(float *data, int D, int H, int W, float sigma) {
    if (sigma <= 0) return;
    long spatial = (long)D * H * W;
    size_t n3 = (size_t)spatial * 3;
    float *scratch = (float *)malloc(n3 * sizeof(float));
    float *kern = NULL; int klen = 0;
    make_gaussian_kernel(sigma, 2.0f, &kern, &klen);
    int r = klen / 2;

    /* Axis 0 (D): data → scratch */
    for (long s = 0; s < spatial; s++) {
        int w = s % W, h = (int)((s / W) % H), d = (int)(s / ((long)H * W));
        float s0=0,s1=0,s2=0;
        for (int k = 0; k < klen; k++) {
            int dd = d + k - r;
            if (dd >= 0 && dd < D) {
                long si = ((long)dd*H+h)*W*3+w*3;
                s0+=data[si]*kern[k]; s1+=data[si+1]*kern[k]; s2+=data[si+2]*kern[k];
            }
        }
        long di=s*3; scratch[di]=s0; scratch[di+1]=s1; scratch[di+2]=s2;
    }
    /* Axis 1 (H): scratch → data */
    for (long s = 0; s < spatial; s++) {
        int w = s % W, h = (int)((s / W) % H), d = (int)(s / ((long)H * W));
        float s0=0,s1=0,s2=0;
        for (int k = 0; k < klen; k++) {
            int hh = h + k - r;
            if (hh >= 0 && hh < H) {
                long si = ((long)d*H+hh)*W*3+w*3;
                s0+=scratch[si]*kern[k]; s1+=scratch[si+1]*kern[k]; s2+=scratch[si+2]*kern[k];
            }
        }
        long di=s*3; data[di]=s0; data[di+1]=s1; data[di+2]=s2;
    }
    /* Axis 2 (W): data → scratch */
    for (long s = 0; s < spatial; s++) {
        int w = s % W, h = (int)((s / W) % H), d = (int)(s / ((long)H * W));
        float s0=0,s1=0,s2=0;
        for (int k = 0; k < klen; k++) {
            int ww = w + k - r;
            if (ww >= 0 && ww < W) {
                long si = ((long)d*H+h)*W*3+ww*3;
                s0+=data[si]*kern[k]; s1+=data[si+1]*kern[k]; s2+=data[si+2]*kern[k];
            }
        }
        long di=s*3; scratch[di]=s0; scratch[di+1]=s1; scratch[di+2]=s2;
    }
    memcpy(data, scratch, n3 * sizeof(float));
    free(scratch); free(kern);
}

/* Compositive warp update: out[x] = update[x] + interp(warp, identity + update[x]) */
static void compositive_update(const float *warp, float *update,
                                int D, int H, int W) {
    long spatial = (long)D * H * W;
    for (long s = 0; s < spatial; s++) {
        int w = s % W, h = (int)((s / W) % H), d = (int)(s / ((long)H * W));
        float nx = (W > 1) ? 2.0f * w / (W - 1) - 1.0f : 0.0f;
        float ny = (H > 1) ? 2.0f * h / (H - 1) - 1.0f : 0.0f;
        float nz = (D > 1) ? 2.0f * d / (D - 1) - 1.0f : 0.0f;
        float sx = nx + update[s*3], sy = ny + update[s*3+1], sz = nz + update[s*3+2];
        float ix = (sx + 1.0f) * 0.5f * (W - 1);
        float iy = (sy + 1.0f) * 0.5f * (H - 1);
        float iz = (sz + 1.0f) * 0.5f * (D - 1);
        int x0 = (int)floorf(ix), y0 = (int)floorf(iy), z0 = (int)floorf(iz);
        float fx = ix - x0, fy = iy - y0, fz = iz - z0;
        #define WI(dd,hh,ww,c) ((dd)>=0&&(dd)<D&&(hh)>=0&&(hh)<H&&(ww)>=0&&(ww)<W?\
            warp[((long)(dd)*H+(hh))*W*3+(ww)*3+(c)]:0.0f)
        float wt[8] = {(1-fx)*(1-fy)*(1-fz),fx*(1-fy)*(1-fz),(1-fx)*fy*(1-fz),fx*fy*(1-fz),
                        (1-fx)*(1-fy)*fz,fx*(1-fy)*fz,(1-fx)*fy*fz,fx*fy*fz};
        int dz[8]={z0,z0,z0,z0,z0+1,z0+1,z0+1,z0+1};
        int dy[8]={y0,y0,y0+1,y0+1,y0,y0,y0+1,y0+1};
        int dx[8]={x0,x0+1,x0,x0+1,x0,x0+1,x0,x0+1};
        for (int c = 0; c < 3; c++) {
            float val = 0;
            for (int k = 0; k < 8; k++) val += wt[k] * WI(dz[k],dy[k],dx[k],c);
            update[s*3+c] += val;
        }
        #undef WI
    }
}

/* WarpAdam step for one displacement field (matching GPU syn_warp_adam_step) */
static void warp_adam_step(float *warp, const float *grad, float *exp_avg, float *exp_avg_sq,
                            float *adam_dir, int D, int H, int W,
                            int *step_t, float lr, float beta1, float beta2, float eps,
                            float smooth_grad_sigma, float smooth_warp_sigma) {
    long spatial = (long)D * H * W;
    size_t n3 = (size_t)spatial * 3;

    (*step_t)++;
    float bc1 = 1.0f - powf(beta1, (float)*step_t);
    float bc2 = 1.0f - powf(beta2, (float)*step_t);

    /* Moments update + direction */
    for (size_t i = 0; i < n3; i++) {
        float g = grad[i];
        exp_avg[i] = beta1 * exp_avg[i] + (1.0f - beta1) * g;
        exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g * g;
        adam_dir[i] = (exp_avg[i] / bc1) / (sqrtf(exp_avg_sq[i] / bc2) + eps);
    }

    /* Normalize by max L2 norm */
    float gradmax = eps;
    for (long s = 0; s < spatial; s++) {
        float dx = adam_dir[s*3], dy = adam_dir[s*3+1], dz = adam_dir[s*3+2];
        float l2 = sqrtf(dx*dx + dy*dy + dz*dz);
        if (l2 > gradmax) gradmax = l2;
    }
    if (gradmax < 1.0f) gradmax = 1.0f;
    float half_res = 1.0f / (float)((D > H ? (D > W ? D : W) : (H > W ? H : W)) - 1);
    float sf = half_res / gradmax * (-lr);
    for (size_t i = 0; i < n3; i++)
        adam_dir[i] *= sf;

    /* Compositive update */
    compositive_update(warp, adam_dir, D, H, W);

    /* Smooth result */
    if (smooth_warp_sigma > 0)
        blur_disp_dhw3(adam_dir, D, H, W, smooth_warp_sigma);

    /* warp = adam_dir */
    memcpy(warp, adam_dir, n3 * sizeof(float));
}

int syn_register(const image_t *fixed, const image_t *moving,
                 const float init_affine_44[4][4],
                 syn_opts_t opts, syn_result_t *result) {
    memset(result, 0, sizeof(syn_result_t));
    memcpy(result->affine_44, init_affine_44, 16 * sizeof(float));

    int fD = fixed->data.shape[2], fH = fixed->data.shape[3], fW = fixed->data.shape[4];
    int mD = moving->data.shape[2], mH = moving->data.shape[3], mW = moving->data.shape[4];

    /* Combined torch-space affine for forward warp */
    mat44d phys_d, tmp, combined;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            phys_d.m[i][j] = init_affine_44[i][j];
    mat44d_mul(&tmp, &phys_d, &fixed->meta.torch2phy);
    mat44d_mul(&combined, &moving->meta.phy2torch, &tmp);

    tensor_t combined_aff;
    {
        int s[3] = {1, 3, 4};
        tensor_alloc(&combined_aff, 3, s, DTYPE_FLOAT32, DEVICE_CPU);
        float *d = tensor_data_f32(&combined_aff);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                d[i*4+j] = (float)combined.m[i][j];
    }

    tensor_t identity_aff;
    {
        int s[3] = {1, 3, 4};
        tensor_alloc(&identity_aff, 3, s, DTYPE_FLOAT32, DEVICE_CPU);
        float *d = tensor_data_f32(&identity_aff);
        memset(d, 0, 12 * sizeof(float));
        d[0] = d[5] = d[10] = 1.0f;
    }

    /* Displacement fields */
    tensor_t fwd_disp, rev_disp;
    tensor_init(&fwd_disp); tensor_init(&rev_disp);

    /* WarpAdam states */
    tensor_t fwd_m, fwd_v, rev_m, rev_v;
    tensor_init(&fwd_m); tensor_init(&fwd_v);
    tensor_init(&rev_m); tensor_init(&rev_v);
    int fwd_step = 0, rev_step = 0;
    float beta1 = 0.9f, beta2 = 0.99f, eps = 1e-8f;

    for (int si = 0; si < opts.n_scales; si++) {
        int scale = opts.scales[si];
        int iters = opts.iterations[si];

        int dD = (scale > 1) ? fD/scale : fD;
        int dH = (scale > 1) ? fH/scale : fH;
        int dW = (scale > 1) ? fW/scale : fW;
        if (dD < 8) dD = 8; if (dH < 8) dH = 8; if (dW < 8) dW = 8;
        if (scale == 1) { dD = fD; dH = fH; dW = fW; }

        /* SyN: moving is NOT downsampled — it stays at full resolution.
         * Instead, moving is blurred at full res (matching Python _smooth_image_not_mask) */
        tensor_t fixed_down, moving_blur;
        int mbD = mD, mbH = mH, mbW = mW;  /* moving_blur dimensions = full moving */

        if (scale > 1) {
            int fds[5] = {1, 1, dD, dH, dW};
            tensor_alloc(&fixed_down, 5, fds, DTYPE_FLOAT32, DEVICE_CPU);
            cpu_blur_downsample(tensor_data_f32(&fixed->data),
                                 tensor_data_f32(&fixed_down),
                                 fD, fH, fW, dD, dH, dW);

            /* Blur moving at full resolution (matching Python/CUDA/Metal SyN) */
            int mbs[5] = {1, 1, mD, mH, mW};
            tensor_alloc(&moving_blur, 5, mbs, DTYPE_FLOAT32, DEVICE_CPU);
            memcpy(tensor_data_f32(&moving_blur), tensor_data_f32(&moving->data),
                   (size_t)mD*mH*mW*sizeof(float));
            cpu_blur_volume(tensor_data_f32(&moving_blur), mD, mH, mW,
                             0.5f * (float)fD / (float)dD,
                             0.5f * (float)fH / (float)dH,
                             0.5f * (float)fW / (float)dW,
                             2.0f);
        } else {
            tensor_view(&fixed_down, &fixed->data);
            tensor_view(&moving_blur, &moving->data);
        }

        long spatial = (long)dD * dH * dW;
        size_t n3 = (size_t)spatial * 3;

        /* Resize or allocate displacement fields */
        if (fwd_disp.data == NULL) {
            int ds[5] = {1, dD, dH, dW, 3};
            tensor_alloc(&fwd_disp, 5, ds, DTYPE_FLOAT32, DEVICE_CPU);
            tensor_alloc(&rev_disp, 5, ds, DTYPE_FLOAT32, DEVICE_CPU);
        } else if (fwd_disp.shape[1] != dD || fwd_disp.shape[2] != dH || fwd_disp.shape[3] != dW) {
            tensor_t new_fwd, new_rev;
            resize_disp(&fwd_disp, &new_fwd, dD, dH, dW);
            resize_disp(&rev_disp, &new_rev, dD, dH, dW);
            tensor_free(&fwd_disp); tensor_free(&rev_disp);
            fwd_disp = new_fwd; rev_disp = new_rev;
        }

        /* Reset WarpAdam state */
        tensor_free(&fwd_m); tensor_free(&fwd_v);
        tensor_free(&rev_m); tensor_free(&rev_v);
        {
            int as[5] = {1, dD, dH, dW, 3};
            tensor_alloc(&fwd_m, 5, as, DTYPE_FLOAT32, DEVICE_CPU);
            tensor_alloc(&fwd_v, 5, as, DTYPE_FLOAT32, DEVICE_CPU);
            tensor_alloc(&rev_m, 5, as, DTYPE_FLOAT32, DEVICE_CPU);
            tensor_alloc(&rev_v, 5, as, DTYPE_FLOAT32, DEVICE_CPU);
        }
        fwd_step = 0; rev_step = 0;

        fprintf(stderr, "  SyN scale %d: fixed[%d,%d,%d] moving_blur[%d,%d,%d] x %d iters\n",
                scale, dD, dH, dW, mbD, mbH, mbW, iters);

        /* Generate base grids */
        int gs[3] = {dD, dH, dW};
        tensor_t fwd_base, rev_base;
        affine_grid_3d(&combined_aff, gs, &fwd_base);
        affine_grid_3d(&identity_aff, gs, &rev_base);

        float *adam_dir_fwd = (float *)calloc(n3, sizeof(float));
        float *adam_dir_rev = (float *)calloc(n3, sizeof(float));

        float prev_loss = 1e30f;
        int converge_count = 0;

        for (int it = 0; it < iters; it++) {
            /* 1. sampling_grids = base + warp (no pre-smoothing) */
            tensor_t fwd_sg, rev_sg;
            {
                int sgs[5] = {1, dD, dH, dW, 3};
                tensor_alloc(&fwd_sg, 5, sgs, DTYPE_FLOAT32, DEVICE_CPU);
                tensor_alloc(&rev_sg, 5, sgs, DTYPE_FLOAT32, DEVICE_CPU);
                float *f = tensor_data_f32(&fwd_sg), *r = tensor_data_f32(&rev_sg);
                const float *fb = tensor_data_f32(&fwd_base), *rb = tensor_data_f32(&rev_base);
                const float *fd = tensor_data_f32(&fwd_disp), *rd = tensor_data_f32(&rev_disp);
                for (size_t i = 0; i < n3; i++) {
                    f[i] = fb[i] + fd[i];
                    r[i] = rb[i] + rd[i];
                }
            }

            /* 2-3. Sample images */
            tensor_t moved, fixed_warped;
            cpu_grid_sample_3d_forward(&moving_blur, &fwd_sg, &moved, 1);
            cpu_grid_sample_3d_forward(&fixed_down, &rev_sg, &fixed_warped, 1);

            /* 4. Fused CC loss — both pred and target gradients in one call */
            float loss;
            tensor_t grad_moved, grad_fixed_warped;
            cpu_cc_loss_3d_both(&moved, &fixed_warped, opts.cc_kernel_size, &loss,
                                 &grad_moved, &grad_fixed_warped);

            /* 5. Backward through grid_sample */
            tensor_t grad_fwd_grid, grad_rev_grid;
            cpu_grid_sample_3d_backward(&grad_moved, &moving_blur, &fwd_sg, &grad_fwd_grid, 1);
            cpu_grid_sample_3d_backward(&grad_fixed_warped, &fixed_down, &rev_sg, &grad_rev_grid, 1);

            /* 6. Smooth gradients */
            float *gfwd = tensor_data_f32(&grad_fwd_grid);
            float *grev = tensor_data_f32(&grad_rev_grid);
            if (opts.smooth_grad_sigma > 0) {
                blur_disp_dhw3(gfwd, dD, dH, dW, opts.smooth_grad_sigma);
                blur_disp_dhw3(grev, dD, dH, dW, opts.smooth_grad_sigma);
            }

            /* 7-8. WarpAdam step for both warps */
            warp_adam_step(tensor_data_f32(&fwd_disp), gfwd,
                            tensor_data_f32(&fwd_m), tensor_data_f32(&fwd_v),
                            adam_dir_fwd, dD, dH, dW,
                            &fwd_step, opts.lr, beta1, beta2, eps,
                            opts.smooth_grad_sigma, opts.smooth_warp_sigma);
            warp_adam_step(tensor_data_f32(&rev_disp), grev,
                            tensor_data_f32(&rev_m), tensor_data_f32(&rev_v),
                            adam_dir_rev, dD, dH, dW,
                            &rev_step, opts.lr, beta1, beta2, eps,
                            opts.smooth_grad_sigma, opts.smooth_warp_sigma);

            /* Cleanup iteration tensors (before potential break) */
            tensor_free(&fwd_sg); tensor_free(&rev_sg);
            tensor_free(&moved); tensor_free(&fixed_warped);
            tensor_free(&grad_moved); tensor_free(&grad_fixed_warped);
            tensor_free(&grad_fwd_grid); tensor_free(&grad_rev_grid);

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

        free(adam_dir_fwd); free(adam_dir_rev);
        tensor_free(&fwd_base); tensor_free(&rev_base);
        if (scale > 1) { tensor_free(&fixed_down); tensor_free(&moving_blur); }
    }

    /* Free WarpAdam state */
    tensor_free(&fwd_m); tensor_free(&fwd_v);
    tensor_free(&rev_m); tensor_free(&rev_v);

    /* Store results */
    result->fwd_disp = fwd_disp;
    result->rev_disp = rev_disp;

    /* Evaluate NCC at full resolution */
    {
        tensor_t moved_eval;
        syn_evaluate(fixed, moving, result, &moved_eval);
        result->moved = moved_eval;
        cpu_cc_loss_3d(&moved_eval, &fixed->data, 9, &result->ncc_loss, NULL);
    }

    tensor_free(&combined_aff);
    tensor_free(&identity_aff);

    return 0;
}

/* CPU warp inversion via fixed-point iteration (matching GPU) */
static void cpu_warp_inverse(const float *u, float *inv, int D, int H, int W, int n_iters) {
    long n3 = (long)D * H * W * 3;
    memset(inv, 0, n3 * sizeof(float));
    float *tmp_buf = (float *)malloc(n3 * sizeof(float));
    for (int iter = 0; iter < n_iters; iter++) {
        for (int d = 0; d < D; d++) {
            float nz = (D > 1) ? 2.0f*d/(D-1)-1.0f : 0.0f;
            for (int h = 0; h < H; h++) {
                float ny = (H > 1) ? 2.0f*h/(H-1)-1.0f : 0.0f;
                for (int w = 0; w < W; w++) {
                    float nx = (W > 1) ? 2.0f*w/(W-1)-1.0f : 0.0f;
                    long idx = ((long)d*H+h)*W*3+w*3;
                    float sx = nx+inv[idx], sy = ny+inv[idx+1], sz = nz+inv[idx+2];
                    float ix = (sx+1)*0.5f*(W-1), iy = (sy+1)*0.5f*(H-1), iz = (sz+1)*0.5f*(D-1);
                    int x0=(int)floorf(ix), y0=(int)floorf(iy), z0=(int)floorf(iz);
                    float fx=ix-x0, fy=iy-y0, fz=iz-z0;
                    #define UV(dd,hh,ww,c) ((dd)>=0&&(dd)<D&&(hh)>=0&&(hh)<H&&(ww)>=0&&(ww)<W?\
                        u[((long)(dd)*H+(hh))*W*3+(ww)*3+(c)]:0.0f)
                    float wt[8]={(1-fx)*(1-fy)*(1-fz),fx*(1-fy)*(1-fz),(1-fx)*fy*(1-fz),fx*fy*(1-fz),
                                 (1-fx)*(1-fy)*fz,fx*(1-fy)*fz,(1-fx)*fy*fz,fx*fy*fz};
                    int dz[8]={z0,z0,z0,z0,z0+1,z0+1,z0+1,z0+1};
                    int dy[8]={y0,y0,y0+1,y0+1,y0,y0,y0+1,y0+1};
                    int dx[8]={x0,x0+1,x0,x0+1,x0,x0+1,x0,x0+1};
                    float v[3]={0,0,0};
                    for(int k=0;k<8;k++) for(int c=0;c<3;c++) v[c]+=wt[k]*UV(dz[k],dy[k],dx[k],c);
                    #undef UV
                    tmp_buf[idx]=-v[0]; tmp_buf[idx+1]=-v[1]; tmp_buf[idx+2]=-v[2];
                }
            }
        }
        memcpy(inv, tmp_buf, n3*sizeof(float));
    }
    free(tmp_buf);
}

int syn_evaluate(const image_t *fixed, const image_t *moving,
                 const syn_result_t *result, tensor_t *output) {
    int fD = fixed->data.shape[2], fH = fixed->data.shape[3], fW = fixed->data.shape[4];
    long fSp = (long)fD * fH * fW;
    size_t n3 = (size_t)fSp * 3;

    /* Build combined affine */
    mat44d phys_d, tmp2, comb;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            phys_d.m[i][j] = result->affine_44[i][j];
    mat44d_mul(&tmp2, &phys_d, &fixed->meta.torch2phy);
    mat44d_mul(&comb, &moving->meta.phy2torch, &tmp2);
    tensor_t combined_aff;
    {
        int s[3] = {1, 3, 4};
        tensor_alloc(&combined_aff, 3, s, DTYPE_FLOAT32, DEVICE_CPU);
        float *d = tensor_data_f32(&combined_aff);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                d[i*4+j] = (float)comb.m[i][j];
    }

    int gs[3] = {fD, fH, fW};
    tensor_t base_grid;
    affine_grid_3d(&combined_aff, gs, &base_grid);

    /* Resize warps to full resolution if needed */
    tensor_t full_fwd, full_rev;
    if (result->fwd_disp.shape[1] != fD || result->fwd_disp.shape[2] != fH || result->fwd_disp.shape[3] != fW) {
        resize_disp(&result->fwd_disp, &full_fwd, fD, fH, fW);
        resize_disp(&result->rev_disp, &full_rev, fD, fH, fW);
    } else {
        tensor_view(&full_fwd, &result->fwd_disp);
        tensor_view(&full_rev, &result->rev_disp);
    }

    /* Invert reverse warp (550 iterations of fixed-point, matching GPU) */
    fprintf(stderr, "  Warp inverse: 550 iters at [%d,%d,%d]\n", fD, fH, fW);
    float *inv_rev = (float *)malloc(n3 * sizeof(float));
    cpu_warp_inverse(tensor_data_f32(&full_rev), inv_rev, fD, fH, fW, 550);

    /* Compose: composed = inv_rev + interp(fwd, identity + inv_rev) */
    float *composed = (float *)malloc(n3 * sizeof(float));
    const float *h_fwd = tensor_data_f32(&full_fwd);
    for (long s = 0; s < fSp; s++) {
        int w = s % fW, h = (int)((s / fW) % fH), d = (int)(s / ((long)fH * fW));
        float nx = (fW>1) ? 2.0f*w/(fW-1)-1.0f : 0;
        float ny = (fH>1) ? 2.0f*h/(fH-1)-1.0f : 0;
        float nz = (fD>1) ? 2.0f*d/(fD-1)-1.0f : 0;
        float ux=inv_rev[s*3], uy=inv_rev[s*3+1], uz=inv_rev[s*3+2];
        float sx=nx+ux, sy=ny+uy, sz=nz+uz;
        float ix=(sx+1)*0.5f*(fW-1), iy=(sy+1)*0.5f*(fH-1), iz=(sz+1)*0.5f*(fD-1);
        int x0=(int)floorf(ix), y0=(int)floorf(iy), z0=(int)floorf(iz);
        float fx=ix-x0, fy=iy-y0, fz=iz-z0;
        #define FW(dd,hh,ww,c) ((dd)>=0&&(dd)<fD&&(hh)>=0&&(hh)<fH&&(ww)>=0&&(ww)<fW?\
            h_fwd[((long)(dd)*fH+(hh))*fW*3+(ww)*3+(c)]:0.0f)
        float wt[8]={(1-fx)*(1-fy)*(1-fz),fx*(1-fy)*(1-fz),(1-fx)*fy*(1-fz),fx*fy*(1-fz),
                     (1-fx)*(1-fy)*fz,fx*(1-fy)*fz,(1-fx)*fy*fz,fx*fy*fz};
        int dz[8]={z0,z0,z0,z0,z0+1,z0+1,z0+1,z0+1};
        int dy[8]={y0,y0,y0+1,y0+1,y0,y0,y0+1,y0+1};
        int dx[8]={x0,x0+1,x0,x0+1,x0,x0+1,x0,x0+1};
        for (int c=0;c<3;c++) {
            float v=0; for(int k=0;k<8;k++) v+=wt[k]*FW(dz[k],dy[k],dx[k],c);
            composed[s*3+c] = ux + v; /* inv_rev + interp(fwd, ...) */
            if (c==0) composed[s*3+c] = ux + v;
        }
        #undef FW
    }
    free(inv_rev);

    /* sampling_grid = base + composed */
    tensor_t sg;
    {
        int sgs[5] = {1, fD, fH, fW, 3};
        tensor_alloc(&sg, 5, sgs, DTYPE_FLOAT32, DEVICE_CPU);
        float *sd = tensor_data_f32(&sg);
        const float *b = tensor_data_f32(&base_grid);
        for (size_t i = 0; i < n3; i++) sd[i] = b[i] + composed[i];
    }
    free(composed);

    cpu_grid_sample_3d_forward(&moving->data, &sg, output, 1);

    tensor_free(&sg);
    if (full_fwd.owns_data) tensor_free(&full_fwd);
    if (full_rev.owns_data) tensor_free(&full_rev);
    tensor_free(&base_grid);
    tensor_free(&combined_aff);
    return 0;
}
