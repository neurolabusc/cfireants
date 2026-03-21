/*
 * syn.c - Symmetric Normalization (SyN) deformable registration
 *
 * SyN optimizes two displacement fields:
 *   fwd_warp: maps fixed → midpoint (applied to moving image via affine)
 *   rev_warp: maps fixed → midpoint (applied to fixed image directly)
 *
 * Per iteration:
 *   moved = grid_sample(moving, affine_grid + fwd_disp)
 *   fixed_warped = grid_sample(fixed, identity_grid + rev_disp)
 *   loss = CC(moved, fixed_warped)
 *   Backprop to both fwd_disp and rev_disp
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

/* Smooth a [1,D,H,W,3] displacement field via Gaussian blur */
static int smooth_disp(const tensor_t *in, tensor_t *out, int D, int H, int W, float sigma) {
    tensor_t img;
    int is[5] = {1, 3, D, H, W};
    tensor_alloc(&img, 5, is, DTYPE_FLOAT32, DEVICE_CPU);
    permute_dhw3_to_3dhw(tensor_data_f32(in), tensor_data_f32(&img), D, H, W);
    tensor_t blurred;
    cpu_gaussian_blur_3d(&img, &blurred, sigma, 2.0f);
    tensor_free(&img);
    int os[5] = {1, D, H, W, 3};
    tensor_alloc(out, 5, os, DTYPE_FLOAT32, DEVICE_CPU);
    permute_3dhw_to_dhw3(tensor_data_f32(&blurred), tensor_data_f32(out), D, H, W);
    tensor_free(&blurred);
    return 0;
}

/* Resize a [1,D,H,W,3] displacement field */
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

int syn_register(const image_t *fixed, const image_t *moving,
                 const float init_affine_44[4][4],
                 syn_opts_t opts, syn_result_t *result) {
    memset(result, 0, sizeof(syn_result_t));
    memcpy(result->affine_44, init_affine_44, 16 * sizeof(float));

    int fD = fixed->data.shape[2], fH = fixed->data.shape[3], fW = fixed->data.shape[4];

    /* Combined torch-space affine for forward warp */
    mat44d phys_d, tmp, combined;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            phys_d.m[i][j] = init_affine_44[i][j];
    mat44d_mul(&tmp, &phys_d, &fixed->meta.torch2phy);
    mat44d_mul(&combined, &moving->meta.phy2torch, &tmp);

    tensor_t combined_aff; /* [1, 3, 4] */
    {
        int s[3] = {1, 3, 4};
        tensor_alloc(&combined_aff, 3, s, DTYPE_FLOAT32, DEVICE_CPU);
        float *d = tensor_data_f32(&combined_aff);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                d[i*4+j] = (float)combined.m[i][j];
    }

    /* Identity affine for reverse warp (fixed space → fixed space) */
    tensor_t identity_aff;
    {
        int s[3] = {1, 3, 4};
        tensor_alloc(&identity_aff, 3, s, DTYPE_FLOAT32, DEVICE_CPU);
        float *d = tensor_data_f32(&identity_aff);
        memset(d, 0, 12 * sizeof(float));
        d[0] = d[5] = d[10] = 1.0f; /* eye(3,4) */
    }

    /* Displacement fields */
    tensor_t fwd_disp, rev_disp;
    tensor_init(&fwd_disp);
    tensor_init(&rev_disp);

    /* Adam states */
    tensor_t fwd_m, fwd_v, rev_m, rev_v;
    tensor_init(&fwd_m); tensor_init(&fwd_v);
    tensor_init(&rev_m); tensor_init(&rev_v);

    for (int si = 0; si < opts.n_scales; si++) {
        int scale = opts.scales[si];
        int iters = opts.iterations[si];

        int dD = fD/scale, dH = fH/scale, dW = fW/scale;
        if (dD < 8) dD = 8; if (dH < 8) dH = 8; if (dW < 8) dW = 8;

        /* Downsample images */
        tensor_t fixed_down, moving_down;
        if (scale > 1) {
            int ds[5] = {1, 1, dD, dH, dW};
            tensor_alloc(&fixed_down, 5, ds, DTYPE_FLOAT32, DEVICE_CPU);
            tensor_alloc(&moving_down, 5, ds, DTYPE_FLOAT32, DEVICE_CPU);
            cpu_trilinear_resize(&fixed->data, &fixed_down, 1);
            cpu_trilinear_resize(&moving->data, &moving_down, 1);
        } else {
            dD = fD; dH = fH; dW = fW;
            tensor_view(&fixed_down, &fixed->data);
            tensor_view(&moving_down, &moving->data);
        }

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

        /* Reset Adam */
        tensor_free(&fwd_m); tensor_free(&fwd_v);
        tensor_free(&rev_m); tensor_free(&rev_v);
        {
            int as[5] = {1, dD, dH, dW, 3};
            tensor_alloc(&fwd_m, 5, as, DTYPE_FLOAT32, DEVICE_CPU);
            tensor_alloc(&fwd_v, 5, as, DTYPE_FLOAT32, DEVICE_CPU);
            tensor_alloc(&rev_m, 5, as, DTYPE_FLOAT32, DEVICE_CPU);
            tensor_alloc(&rev_v, 5, as, DTYPE_FLOAT32, DEVICE_CPU);
        }
        int adam_step = 0;

        fprintf(stderr, "  SyN scale %d: [%d,%d,%d] x %d iters\n", scale, dD, dH, dW, iters);

        /* Generate base grids */
        int gs[3] = {dD, dH, dW};
        tensor_t fwd_base, rev_base;
        affine_grid_3d(&combined_aff, gs, &fwd_base);
        affine_grid_3d(&identity_aff, gs, &rev_base);

        size_t nvox3 = (size_t)dD * dH * dW * 3;
        float prev_loss = 1e30f;
        int converge_count = 0;

        for (int it = 0; it < iters; it++) {
            /* Smooth displacements */
            tensor_t fwd_smooth, rev_smooth;
            if (opts.smooth_warp_sigma > 0) {
                smooth_disp(&fwd_disp, &fwd_smooth, dD, dH, dW, opts.smooth_warp_sigma);
                smooth_disp(&rev_disp, &rev_smooth, dD, dH, dW, opts.smooth_warp_sigma);
            } else {
                tensor_view(&fwd_smooth, &fwd_disp);
                tensor_view(&rev_smooth, &rev_disp);
            }

            /* Build sampling grids: base + disp */
            tensor_t fwd_sg, rev_sg;
            {
                int sgs[5] = {1, dD, dH, dW, 3};
                tensor_alloc(&fwd_sg, 5, sgs, DTYPE_FLOAT32, DEVICE_CPU);
                tensor_alloc(&rev_sg, 5, sgs, DTYPE_FLOAT32, DEVICE_CPU);
                float *f = tensor_data_f32(&fwd_sg), *r = tensor_data_f32(&rev_sg);
                const float *fb = tensor_data_f32(&fwd_base), *rb = tensor_data_f32(&rev_base);
                const float *fd = tensor_data_f32(&fwd_smooth), *rd = tensor_data_f32(&rev_smooth);
                for (size_t i = 0; i < nvox3; i++) {
                    f[i] = fb[i] + fd[i];
                    r[i] = rb[i] + rd[i];
                }
            }

            /* Sample images */
            tensor_t moved, fixed_warped;
            cpu_grid_sample_3d_forward(&moving_down, &fwd_sg, &moved, 1);
            cpu_grid_sample_3d_forward(&fixed_down, &rev_sg, &fixed_warped, 1);

            /* CC loss between warped images at midpoint */
            float loss;
            tensor_t grad_moved;
            cpu_cc_loss_3d(&moved, &fixed_warped, opts.cc_kernel_size, &loss, &grad_moved);

            /* Backward: dL/d(fwd_sg) via moved */
            tensor_t grad_fwd_grid;
            cpu_grid_sample_3d_backward(&grad_moved, &moving_down, &fwd_sg, &grad_fwd_grid, 1);

            /* Backward: dL/d(rev_sg) via fixed_warped.
             * CC loss grad w.r.t. target (fixed_warped) — need to compute this.
             * Since CC is symmetric in practice, we recompute with swapped args. */
            tensor_t grad_fixed_warped;
            cpu_cc_loss_3d(&fixed_warped, &moved, opts.cc_kernel_size, NULL, &grad_fixed_warped);

            tensor_t grad_rev_grid;
            cpu_grid_sample_3d_backward(&grad_fixed_warped, &fixed_down, &rev_sg, &grad_rev_grid, 1);

            /* Smooth gradients */
            tensor_t fwd_grad_smooth, rev_grad_smooth;
            if (opts.smooth_grad_sigma > 0) {
                smooth_disp(&grad_fwd_grid, &fwd_grad_smooth, dD, dH, dW, opts.smooth_grad_sigma);
                smooth_disp(&grad_rev_grid, &rev_grad_smooth, dD, dH, dW, opts.smooth_grad_sigma);
            } else {
                tensor_view(&fwd_grad_smooth, &grad_fwd_grid);
                tensor_view(&rev_grad_smooth, &grad_rev_grid);
            }

            /* Adam update both displacement fields */
            adam_step++;
            cpu_adam_step(&fwd_disp, &fwd_grad_smooth, &fwd_m, &fwd_v,
                          opts.lr, 0.9f, 0.999f, 1e-8f, adam_step);
            cpu_adam_step(&rev_disp, &rev_grad_smooth, &rev_m, &rev_v,
                          opts.lr, 0.9f, 0.999f, 1e-8f, adam_step);

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

            /* Cleanup */
            if (opts.smooth_warp_sigma > 0) { tensor_free(&fwd_smooth); tensor_free(&rev_smooth); }
            tensor_free(&fwd_sg); tensor_free(&rev_sg);
            tensor_free(&moved); tensor_free(&fixed_warped);
            tensor_free(&grad_moved); tensor_free(&grad_fixed_warped);
            tensor_free(&grad_fwd_grid); tensor_free(&grad_rev_grid);
            if (opts.smooth_grad_sigma > 0) { tensor_free(&fwd_grad_smooth); tensor_free(&rev_grad_smooth); }
        }

        tensor_free(&fwd_base); tensor_free(&rev_base);
        if (scale > 1) { tensor_free(&fixed_down); tensor_free(&moving_down); }
    }

    /* Store results */
    result->fwd_disp = fwd_disp;
    result->rev_disp = rev_disp;

    /* Evaluate NCC at full resolution using forward warp only */
    syn_evaluate(fixed, moving, result, &(tensor_t){0});
    /* Actually let's compute NCC properly */
    {
        tensor_t moved;
        syn_evaluate(fixed, moving, result, &moved);
        cpu_cc_loss_3d(&moved, &fixed->data, 9, &result->ncc_loss, NULL);
        tensor_free(&moved);
    }

    tensor_free(&combined_aff);
    tensor_free(&identity_aff);
    tensor_free(&fwd_m); tensor_free(&fwd_v);
    tensor_free(&rev_m); tensor_free(&rev_v);

    return 0;
}

int syn_evaluate(const image_t *fixed, const image_t *moving,
                 const syn_result_t *result, tensor_t *output) {
    int fD = fixed->data.shape[2], fH = fixed->data.shape[3], fW = fixed->data.shape[4];

    /* Build combined affine */
    mat44d phys_d, tmp2, combined;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            phys_d.m[i][j] = result->affine_44[i][j];
    mat44d_mul(&tmp2, &phys_d, &fixed->meta.torch2phy);
    mat44d_mul(&combined, &moving->meta.phy2torch, &tmp2);
    tensor_t combined_aff;
    {
        int s[3] = {1, 3, 4};
        tensor_alloc(&combined_aff, 3, s, DTYPE_FLOAT32, DEVICE_CPU);
        float *d = tensor_data_f32(&combined_aff);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                d[i*4+j] = (float)combined.m[i][j];
    }

    /* Generate base grid */
    int gs[3] = {fD, fH, fW};
    tensor_t base_grid;
    affine_grid_3d(&combined_aff, gs, &base_grid);

    /* Resize fwd_disp to full resolution if needed */
    tensor_t full_fwd;
    if (result->fwd_disp.shape[1] != fD || result->fwd_disp.shape[2] != fH || result->fwd_disp.shape[3] != fW) {
        resize_disp(&result->fwd_disp, &full_fwd, fD, fH, fW);
    } else {
        tensor_view(&full_fwd, &result->fwd_disp);
    }

    /* sampling_grid = base + fwd_disp */
    size_t nvox3 = (size_t)fD * fH * fW * 3;
    tensor_t sg;
    {
        int sgs[5] = {1, fD, fH, fW, 3};
        tensor_alloc(&sg, 5, sgs, DTYPE_FLOAT32, DEVICE_CPU);
        float *s = tensor_data_f32(&sg);
        const float *b = tensor_data_f32(&base_grid);
        const float *f = tensor_data_f32(&full_fwd);
        for (size_t i = 0; i < nvox3; i++) s[i] = b[i] + f[i];
    }

    cpu_grid_sample_3d_forward(&moving->data, &sg, output, 1);

    tensor_free(&sg);
    if (full_fwd.owns_data) tensor_free(&full_fwd);
    tensor_free(&base_grid);
    tensor_free(&combined_aff);
    return 0;
}
