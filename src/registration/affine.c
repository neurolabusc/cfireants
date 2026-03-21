/*
 * affine.c - Affine registration (12 DOF)
 *
 * Directly optimizes the [3, 4] affine matrix via gradient descent.
 * Uses the same around_center trick as Python:
 *   y = A(x - c) + c + t'  =>  t = t' + c - A*c
 *
 * The backward pass is simpler than rigid since dL/d(affine_param) =
 * dL/d(combined_affine) propagated through the p2t/t2p chain, without
 * the quaternion Jacobian.
 */

#include "cfireants/registration.h"
#include "cfireants/interpolator.h"
#include "cfireants/losses.h"
#include "cfireants/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    float A[3][4];          /* learnable affine (possibly in around_center form) */
    float center[3];        /* image center in physical space */
    int around_center;
    /* Adam state for 12 parameters */
    float m[12], v[12];
    int step;
} affine_state_t;

/* Get the true physical-space affine [4][4] from state */
static void get_physical_affine(const affine_state_t *s, float mat[4][4]) {
    memset(mat, 0, 16 * sizeof(float));
    mat[3][3] = 1.0f;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            mat[i][j] = s->A[i][j];

    if (s->around_center) {
        /* t = t' + c - A*c */
        for (int i = 0; i < 3; i++) {
            mat[i][3] = s->A[i][3] + s->center[i];
            for (int j = 0; j < 3; j++)
                mat[i][3] -= s->A[i][j] * s->center[j];
        }
    } else {
        for (int i = 0; i < 3; i++)
            mat[i][3] = s->A[i][3];
    }
}

static void build_torch_affine_aff(const affine_state_t *s,
                                    const mat44d *fixed_t2p,
                                    const mat44d *moving_p2t,
                                    tensor_t *aff_tensor) {
    float phys44[4][4];
    get_physical_affine(s, phys44);

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

/* One optimization step */
static float affine_step(affine_state_t *s,
                         const tensor_t *fixed_down,
                         const tensor_t *moving_blur,
                         const mat44d *fixed_t2p,
                         const mat44d *moving_p2t,
                         int loss_type, int cc_kernel_size, int mi_num_bins,
                         float lr, float beta1, float beta2, float eps) {
    int D = fixed_down->shape[2], H = fixed_down->shape[3], W = fixed_down->shape[4];

    /* Forward */
    tensor_t aff_tensor;
    build_torch_affine_aff(s, fixed_t2p, moving_p2t, &aff_tensor);

    int out_shape[3] = {D, H, W};
    tensor_t grid;
    affine_grid_3d(&aff_tensor, out_shape, &grid);

    tensor_t moved;
    cpu_grid_sample_3d_forward(moving_blur, &grid, &moved, 1);

    float loss;
    tensor_t grad_moved;
    if (loss_type == LOSS_MI)
        cpu_mi_loss_3d(&moved, fixed_down, mi_num_bins, &loss, &grad_moved);
    else
        cpu_cc_loss_3d(&moved, fixed_down, cc_kernel_size, &loss, &grad_moved);

    /* Backward through grid_sample */
    tensor_t grad_grid;
    cpu_grid_sample_3d_backward(&grad_moved, moving_blur, &grid, &grad_grid, 1);

    /* Backward through affine_grid: dL/dA_combined [3, 4] */
    float dL_dA_comb[3][4];
    memset(dL_dA_comb, 0, sizeof(dL_dA_comb));
    const float *gg = tensor_data_f32(&grad_grid);
    for (int d = 0; d < D; d++) {
        float nz = (D > 1) ? (2.0f * d / (D - 1) - 1.0f) : 0.0f;
        for (int h = 0; h < H; h++) {
            float ny = (H > 1) ? (2.0f * h / (H - 1) - 1.0f) : 0.0f;
            for (int w = 0; w < W; w++) {
                float nx = (W > 1) ? (2.0f * w / (W - 1) - 1.0f) : 0.0f;
                float coord[4] = {nx, ny, nz, 1.0f};
                size_t idx = (((size_t)d * H + h) * W + w) * 3;
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 4; j++)
                        dL_dA_comb[i][j] += gg[idx + i] * coord[j];
            }
        }
    }

    /* Chain rule: combined = p2t @ phys_aff @ t2p
     * dL/d(phys_aff) = p2t^T @ dL/d(combined) @ t2p^T (for the [3,4] part)
     *
     * We need the full 4x4 version */
    mat44d dL_comb_44;
    memset(&dL_comb_44, 0, sizeof(mat44d));
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            dL_comb_44.m[i][j] = dL_dA_comb[i][j];

    mat44d p2t_T, t2p_T, tmp1, dL_dphys;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            p2t_T.m[i][j] = moving_p2t->m[j][i];
            t2p_T.m[i][j] = fixed_t2p->m[j][i];
        }
    mat44d_mul(&tmp1, &p2t_T, &dL_comb_44);
    mat44d_mul(&dL_dphys, &tmp1, &t2p_T);

    /* dL/d(phys_aff) → dL/d(A_param) accounting for around_center.
     * If around_center: phys_aff = [A | A*c_param_transl]
     *   where t_phys = t' + c - A*c, so:
     *   dL/dA[i][j] = dL/dphys_A[i][j] - dL/dt_phys[i] * c[j]
     *   dL/dt'[i] = dL/dt_phys[i]
     */
    float dL_dA_param[12]; /* row-major [3][4] */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            dL_dA_param[i * 4 + j] = (float)dL_dphys.m[i][j];
            if (s->around_center)
                dL_dA_param[i * 4 + j] -= (float)dL_dphys.m[i][3] * s->center[j];
        }
        dL_dA_param[i * 4 + 3] = (float)dL_dphys.m[i][3];
    }

    /* Adam update */
    s->step++;
    float bc1 = 1.0f - powf(beta1, (float)s->step);
    float bc2 = 1.0f - powf(beta2, (float)s->step);
    float step_size = lr / bc1;

    float *params = &s->A[0][0];
    for (int k = 0; k < 12; k++) {
        s->m[k] = beta1 * s->m[k] + (1 - beta1) * dL_dA_param[k];
        s->v[k] = beta2 * s->v[k] + (1 - beta2) * dL_dA_param[k] * dL_dA_param[k];
        float denom = sqrtf(s->v[k] / bc2) + eps;
        params[k] -= step_size * s->m[k] / denom;
    }

    tensor_free(&aff_tensor);
    tensor_free(&grid);
    tensor_free(&moved);
    tensor_free(&grad_moved);
    tensor_free(&grad_grid);

    return loss;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

int affine_register(const image_t *fixed, const image_t *moving,
                    const float init_rigid_34[3][4],
                    affine_opts_t opts, affine_result_t *result) {
    memset(result, 0, sizeof(affine_result_t));

    affine_state_t state;
    memset(&state, 0, sizeof(state));
    state.around_center = 1;

    /* Center = fixed image's physical center */
    for (int i = 0; i < 3; i++)
        state.center[i] = (float)fixed->meta.torch2phy.m[i][3];

    /* Initialize from rigid result (or identity) */
    if (init_rigid_34) {
        memcpy(state.A, init_rigid_34, 12 * sizeof(float));
    } else {
        state.A[0][0] = state.A[1][1] = state.A[2][2] = 1.0f;
    }

    /* Convert to around_center: t' = t - c + A*c */
    if (state.around_center) {
        for (int i = 0; i < 3; i++) {
            state.A[i][3] = state.A[i][3] - state.center[i];
            for (int j = 0; j < 3; j++)
                state.A[i][3] += state.A[i][j] * state.center[j];
        }
    }

    int fD = fixed->data.shape[2], fH = fixed->data.shape[3], fW = fixed->data.shape[4];

    for (int si = 0; si < opts.n_scales; si++) {
        int scale = opts.scales[si];
        int iters = opts.iterations[si];

        tensor_t fixed_down, moving_down;
        if (scale > 1) {
            int dD = fD/scale, dH = fH/scale, dW = fW/scale;
            if (dD < 8) dD = 8; if (dH < 8) dH = 8; if (dW < 8) dW = 8;
            int ds[5] = {1, 1, dD, dH, dW};
            tensor_alloc(&fixed_down, 5, ds, DTYPE_FLOAT32, DEVICE_CPU);
            tensor_alloc(&moving_down, 5, ds, DTYPE_FLOAT32, DEVICE_CPU);
            cpu_trilinear_resize(&fixed->data, &fixed_down, 1);
            cpu_trilinear_resize(&moving->data, &moving_down, 1);
        } else {
            tensor_view(&fixed_down, &fixed->data);
            tensor_view(&moving_down, &moving->data);
        }

        fprintf(stderr, "  Affine scale %d: [%d,%d,%d] x %d iters\n",
                scale, fixed_down.shape[2], fixed_down.shape[3], fixed_down.shape[4], iters);

        state.step = 0;
        memset(state.m, 0, sizeof(state.m));
        memset(state.v, 0, sizeof(state.v));

        float prev_loss = 1e30f;
        int converge_count = 0;

        for (int i = 0; i < iters; i++) {
            float loss = affine_step(&state, &fixed_down, &moving_down,
                                     &fixed->meta.torch2phy, &moving->meta.phy2torch,
                                     opts.loss_type, opts.cc_kernel_size,
                                     opts.mi_num_bins > 0 ? opts.mi_num_bins : 32,
                                     opts.lr, 0.9f, 0.999f, 1e-8f);

            if (i % 50 == 0 || i == iters - 1)
                fprintf(stderr, "    iter %d/%d loss=%.6f\n", i, iters, loss);

            if (fabsf(loss - prev_loss) < opts.tolerance) {
                converge_count++;
                if (converge_count >= opts.max_tolerance_iters) {
                    fprintf(stderr, "    Converged at iter %d\n", i);
                    break;
                }
            } else {
                converge_count = 0;
            }
            prev_loss = loss;
        }

        if (scale > 1) {
            tensor_free(&fixed_down);
            tensor_free(&moving_down);
        }
    }

    /* Extract physical-space affine */
    float phys44[4][4];
    get_physical_affine(&state, phys44);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            result->affine_mat[i][j] = phys44[i][j];

    /* Evaluate NCC */
    tensor_t moved;
    apply_affine_transform(fixed, moving, result->affine_mat, &moved);
    cpu_cc_loss_3d(&moved, &fixed->data, 9, &result->ncc_loss, NULL);
    tensor_free(&moved);

    return 0;
}
