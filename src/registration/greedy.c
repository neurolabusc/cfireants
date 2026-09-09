/*
 * greedy.c - Greedy deformable registration
 *
 * Optimizes a dense displacement field [1, D, H, W, 3] that is added
 * to the affine-warped grid before sampling.
 *
 * Per iteration:
 *   1. Optionally smooth displacement field (Gaussian blur)
 *   2. base_grid = affine_grid(combined_affine)
 *   3. sampling_grid = base_grid + displacement
 *   4. moved = grid_sample(moving, sampling_grid)
 *   5. loss, dL/dmoved = CC_loss(moved, fixed)
 *   6. dL/dsampling_grid = grid_sample_backward(dL/dmoved, ...)
 *   7. dL/ddisplacement = dL/dsampling_grid (identity since grid = base + disp)
 *   8. Smooth gradient (Gaussian blur)
 *   9. Adam update on displacement
 */

#include "cfireants/registration.h"
#include "cfireants/interpolator.h"
#include "cfireants/losses.h"
#include "cfireants/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Build the combined torch-space affine [1, 3, 4] from a physical 4x4 affine */
static void build_combined_affine(const float phys44[4][4],
                                  const mat44d *fixed_t2p,
                                  const mat44d *moving_p2t,
                                  tensor_t *aff_tensor) {
    mat44d phys_d, tmp, combined;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            phys_d.m[i][j] = phys44[i][j];
    mat44d_mul(&tmp, &phys_d, fixed_t2p);
    mat44d_mul(&combined, moving_p2t, &tmp);

    int shape[3] = {1, 3, 4};
    tensor_alloc(aff_tensor, 3, shape, DTYPE_FLOAT32, DEVICE_CPU);
    float *data = tensor_data_f32(aff_tensor);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            data[i * 4 + j] = (float)combined.m[i][j];
}

int greedy_register(const image_t *fixed, const image_t *moving,
                    const float init_affine_44[4][4],
                    greedy_opts_t opts, greedy_result_t *result) {
    if (greedy_result_init(result) != 0) return -1;

    int fD = fixed->data.shape[2], fH = fixed->data.shape[3], fW = fixed->data.shape[4];

    /* Build combined torch-space affine */
    tensor_t combined_aff;
    build_combined_affine(init_affine_44, &fixed->meta.torch2phy,
                          &moving->meta.phy2torch, &combined_aff);

    /* Displacement field: starts as zeros, resized per scale */
    tensor_t disp; /* [1, D, H, W, 3] */
    tensor_init(&disp);

    /* Adam state for displacement */
    tensor_t adam_m, adam_v;
    tensor_init(&adam_m);
    tensor_init(&adam_v);

    int mD = moving->data.shape[2], mH = moving->data.shape[3], mW = moving->data.shape[4];

    for (int si = 0; si < opts.n_scales; si++) {
        int scale = opts.scales[si];
        int iters = opts.iterations[si];

        int dD = (scale > 1) ? fD / scale : fD;
        int dH = (scale > 1) ? fH / scale : fH;
        int dW = (scale > 1) ? fW / scale : fW;
        if (dD < 8) dD = 8; if (dH < 8) dH = 8; if (dW < 8) dW = 8;
        if (scale == 1) { dD = fD; dH = fH; dW = fW; }

        int mdD, mdH, mdW;
        moving_pyramid_size(scale, fixed->meta.spacing, moving->meta.spacing,
                            mD, mH, mW, &mdD, &mdH, &mdW);
        if (scale == 1) { mdD = mD; mdH = mH; mdW = mW; }

        /* Downsample images (matching GPU pipeline) */
        tensor_t fixed_down, moving_down;
        if (scale > 1) {
            int fds[5] = {1, 1, dD, dH, dW};
            int mds[5] = {1, 1, mdD, mdH, mdW};
            tensor_alloc(&fixed_down, 5, fds, DTYPE_FLOAT32, DEVICE_CPU);
            tensor_alloc(&moving_down, 5, mds, DTYPE_FLOAT32, DEVICE_CPU);
            cpu_blur_downsample(tensor_data_f32(&fixed->data),
                                 tensor_data_f32(&fixed_down),
                                 fD, fH, fW, dD, dH, dW);
            cpu_blur_downsample(tensor_data_f32(&moving->data),
                                 tensor_data_f32(&moving_down),
                                 mD, mH, mW, mdD, mdH, mdW);
        } else {
            tensor_view(&fixed_down, &fixed->data);
            tensor_view(&moving_down, &moving->data);
            mdD = mD; mdH = mH; mdW = mW;
        }

        /* Resize displacement field to current scale */
        if (disp.data == NULL) {
            /* First scale: allocate zeros */
            int disp_shape[5] = {1, dD, dH, dW, 3};
            tensor_alloc(&disp, 5, disp_shape, DTYPE_FLOAT32, DEVICE_CPU);
        } else if (disp.shape[1] != dD || disp.shape[2] != dH || disp.shape[3] != dW) {
            /* Resize from previous scale via trilinear interpolation.
             * Displacement is [1, D, H, W, 3] — we need to treat each
             * coordinate channel separately, reshaped as [1, 3, D, H, W]. */
            tensor_t disp_img;
            int img_shape[5] = {1, 3, disp.shape[1], disp.shape[2], disp.shape[3]};
            tensor_alloc(&disp_img, 5, img_shape, DTYPE_FLOAT32, DEVICE_CPU);
            /* Permute [1,D,H,W,3] -> [1,3,D,H,W] */
            float *src = tensor_data_f32(&disp);
            float *dst = tensor_data_f32(&disp_img);
            int oD = disp.shape[1], oH = disp.shape[2], oW = disp.shape[3];
            for (int d = 0; d < oD; d++)
                for (int h = 0; h < oH; h++)
                    for (int w = 0; w < oW; w++)
                        for (int c = 0; c < 3; c++)
                            dst[((size_t)c*oD + d)*oH*oW + h*oW + w] =
                                src[((size_t)d*oH + h)*oW*3 + w*3 + c];

            tensor_t disp_resized_img;
            int new_img_shape[5] = {1, 3, dD, dH, dW};
            tensor_alloc(&disp_resized_img, 5, new_img_shape, DTYPE_FLOAT32, DEVICE_CPU);
            cpu_trilinear_resize(&disp_img, &disp_resized_img, 1);
            tensor_free(&disp_img);

            /* Permute back [1,3,D,H,W] -> [1,D,H,W,3] */
            tensor_free(&disp);
            int new_disp_shape[5] = {1, dD, dH, dW, 3};
            tensor_alloc(&disp, 5, new_disp_shape, DTYPE_FLOAT32, DEVICE_CPU);
            float *s2 = tensor_data_f32(&disp_resized_img);
            float *d2 = tensor_data_f32(&disp);
            for (int d = 0; d < dD; d++)
                for (int h = 0; h < dH; h++)
                    for (int w = 0; w < dW; w++)
                        for (int c = 0; c < 3; c++)
                            d2[((size_t)d*dH + h)*dW*3 + w*3 + c] =
                                s2[((size_t)c*dD + d)*dH*dW + h*dW + w];
            tensor_free(&disp_resized_img);
        }

        /* Reset WarpAdam state for new scale */
        tensor_free(&adam_m);
        tensor_free(&adam_v);
        {
            int as[5] = {1, dD, dH, dW, 3};
            tensor_alloc(&adam_m, 5, as, DTYPE_FLOAT32, DEVICE_CPU);
            tensor_alloc(&adam_v, 5, as, DTYPE_FLOAT32, DEVICE_CPU);
        }
        int adam_step_t = 0;
        float beta1 = 0.9f, beta2 = 0.99f, eps = 1e-8f;

        if (cfireants_verbose >= 2) fprintf(stderr, "  Greedy scale %d: fixed[%d,%d,%d] moving[%d,%d,%d] x %d iters\n",
                scale, dD, dH, dW, mdD, mdH, mdW, iters);

        /* Generate base grid from affine (stays constant within scale) */
        int grid_shape[3] = {dD, dH, dW};
        tensor_t base_grid;
        affine_grid_3d(&combined_aff, grid_shape, &base_grid);

        float prev_loss = 1e30f;
        int converge_count = 0;
        size_t nvox3 = (size_t)dD * dH * dW * 3;

        /* Scratch for blur in [D,H,W,3] layout */
        float *adam_dir = (float *)calloc(nvox3, sizeof(float));

        for (int it = 0; it < iters; it++) {
            /* 1. sampling_grid = base_grid + warp (no pre-smoothing) */
            tensor_t sampling_grid;
            {
                int sg[5] = {1, dD, dH, dW, 3};
                tensor_alloc(&sampling_grid, 5, sg, DTYPE_FLOAT32, DEVICE_CPU);
                float *sg_d = tensor_data_f32(&sampling_grid);
                const float *bg = tensor_data_f32(&base_grid);
                const float *dp = tensor_data_f32(&disp);
                for (size_t i = 0; i < nvox3; i++)
                    sg_d[i] = bg[i] + dp[i];
            }

            /* 2. moved = grid_sample(moving, sampling_grid) */
            tensor_t moved;
            cpu_grid_sample_3d_forward(&moving_down, &sampling_grid, &moved, 1);

            /* 3. CC loss + gradient */
            float loss;
            tensor_t grad_moved;
            cpu_cc_loss_3d(&moved, &fixed_down, opts.cc_kernel_size, &loss, &grad_moved);

            /* 4. Backward through grid_sample */
            tensor_t grad_grid;
            cpu_grid_sample_3d_backward(&grad_moved, &moving_down, &sampling_grid,
                                        &grad_grid, 1);

            /* 5-10. Smooth gradient, WarpAdam compositive update, smooth warp */
            float *grad_dhw3 = tensor_data_f32(&grad_grid);
            cpu_blur_disp_dhw3(grad_dhw3, dD, dH, dW, opts.smooth_grad_sigma);
            cpu_warp_adam_step(tensor_data_f32(&disp), grad_dhw3,
                               tensor_data_f32(&adam_m), tensor_data_f32(&adam_v),
                               adam_dir, dD, dH, dW, &adam_step_t,
                               opts.lr, beta1, beta2, eps, opts.smooth_warp_sigma);

            if (it % 50 == 0 || it == iters - 1)
                if (cfireants_verbose >= 2) fprintf(stderr, "    iter %d/%d loss=%.6f\n", it, iters, loss);

            /* Cleanup iteration tensors (before convergence break) */
            tensor_free(&sampling_grid);
            tensor_free(&moved);
            tensor_free(&grad_moved);
            tensor_free(&grad_grid);

            /* Convergence check */
            if (fabsf(loss - prev_loss) < opts.tolerance) {
                converge_count++;
                if (converge_count >= opts.max_tolerance_iters) {
                    if (cfireants_verbose >= 2) fprintf(stderr, "    Converged at iter %d\n", it);
                    break;
                }
            } else {
                converge_count = 0;
            }
            prev_loss = loss;
        }

        free(adam_dir);
        tensor_free(&base_grid);
        if (scale > 1) {
            tensor_free(&fixed_down);
            tensor_free(&moving_down);
        }
    }

    /* Free Adam state from last scale */
    tensor_free(&adam_m);
    tensor_free(&adam_v);

    /* Store final displacement for evaluation */
    result->disp = disp; /* transfer ownership */

    /* Copy affine for evaluation */
    memcpy(result->affine_44, init_affine_44, 16 * sizeof(float));

    /* Evaluate NCC at full resolution */
    {
        int grid_shape[3] = {fD, fH, fW};
        tensor_t full_base_grid;
        affine_grid_3d(&combined_aff, grid_shape, &full_base_grid);

        /* Resize displacement to full resolution */
        tensor_t full_disp;
        if (disp.shape[1] != fD || disp.shape[2] != fH || disp.shape[3] != fW) {
            /* Need to resize */
            tensor_t disp_img;
            int is[5] = {1, 3, disp.shape[1], disp.shape[2], disp.shape[3]};
            tensor_alloc(&disp_img, 5, is, DTYPE_FLOAT32, DEVICE_CPU);
            float *s = tensor_data_f32(&disp), *d = tensor_data_f32(&disp_img);
            int oD=disp.shape[1], oH=disp.shape[2], oW=disp.shape[3];
            for (int dd=0;dd<oD;dd++) for (int h=0;h<oH;h++) for (int w=0;w<oW;w++) for (int c=0;c<3;c++)
                d[((size_t)c*oD+dd)*oH*oW+h*oW+w] = s[((size_t)dd*oH+h)*oW*3+w*3+c];
            tensor_t resized;
            int rs[5] = {1, 3, fD, fH, fW};
            tensor_alloc(&resized, 5, rs, DTYPE_FLOAT32, DEVICE_CPU);
            cpu_trilinear_resize(&disp_img, &resized, 1);
            tensor_free(&disp_img);
            int fds[5] = {1, fD, fH, fW, 3};
            tensor_alloc(&full_disp, 5, fds, DTYPE_FLOAT32, DEVICE_CPU);
            float *s2=tensor_data_f32(&resized), *d2=tensor_data_f32(&full_disp);
            for (int dd=0;dd<fD;dd++) for (int h=0;h<fH;h++) for (int w=0;w<fW;w++) for (int c=0;c<3;c++)
                d2[((size_t)dd*fH+h)*fW*3+w*3+c] = s2[((size_t)c*fD+dd)*fH*fW+h*fW+w];
            tensor_free(&resized);
        } else {
            tensor_view(&full_disp, &disp);
        }

        /* sampling_grid = base + disp */
        tensor_t full_sg;
        size_t n3 = (size_t)fD*fH*fW*3;
        int sgs[5] = {1, fD, fH, fW, 3};
        tensor_alloc(&full_sg, 5, sgs, DTYPE_FLOAT32, DEVICE_CPU);
        float *fsg = tensor_data_f32(&full_sg);
        const float *fbg = tensor_data_f32(&full_base_grid);
        const float *fdp = tensor_data_f32(&full_disp);
        for (size_t i = 0; i < n3; i++) fsg[i] = fbg[i] + fdp[i];

        cpu_grid_sample_3d_forward(&moving->data, &full_sg, &result->moved, 1);
        cpu_cc_loss_3d(&result->moved, &fixed->data, 9, &result->ncc_loss, NULL);

        tensor_free(&full_sg);
        if (full_disp.owns_data) tensor_free(&full_disp);
        tensor_free(&full_base_grid);
    }

    tensor_free(&combined_aff);
    /* Don't free adam_m/adam_v — they were freed per-scale */

    return 0;
}
