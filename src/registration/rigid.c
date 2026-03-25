/*
 * rigid.c - Rigid registration with quaternion parameterization
 *
 * Implements gradient-based optimization of rotation (quaternion) and
 * translation parameters. The backward pass is computed explicitly:
 *
 * Forward:  quat → R → A = p2t @ [sR*moment|t] @ t2p → grid → moved → loss
 * Backward: dL/dmoved → dL/dgrid → dL/dA → dL/dquat, dL/dtransl
 */

#include "cfireants/registration.h"
#include "cfireants/interpolator.h"
#include "cfireants/losses.h"
#include "cfireants/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Quaternion utilities                                                 */
/* ------------------------------------------------------------------ */

/* Quaternion to rotation matrix (3x3).
 * q = (w, x, y, z), normalized internally. */
static void quat_to_rotmat(const float q[4], float R[3][3]) {
    float norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (norm < 1e-8f) norm = 1e-8f;
    float w = q[0]/norm, x = q[1]/norm, y = q[2]/norm, z = q[3]/norm;

    R[0][0] = 1 - 2*(y*y + z*z);
    R[0][1] = 2*(x*y - w*z);
    R[0][2] = 2*(x*z + w*y);
    R[1][0] = 2*(x*y + w*z);
    R[1][1] = 1 - 2*(x*x + z*z);
    R[1][2] = 2*(y*z - w*x);
    R[2][0] = 2*(x*z - w*y);
    R[2][1] = 2*(y*z + w*x);
    R[2][2] = 1 - 2*(x*x + y*y);
}

/* Compute dR/dq[k] for each quaternion component k=0..3.
 * dR_dq[k] is a 3x3 matrix. */
static void quat_rotmat_jacobian(const float q[4], float dR_dq[4][3][3]) {
    float norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (norm < 1e-8f) norm = 1e-8f;
    float inv_n = 1.0f / norm;
    float w = q[0]*inv_n, x = q[1]*inv_n, y = q[2]*inv_n, z = q[3]*inv_n;
    /* For each q component, dR/dq_k = dR/d(normalized_q) * d(normalized_q)/dq_k
     * d(q_normalized)/dq_k = (I - q_hat*q_hat^T) / norm, where q_hat = q/norm
     * This is complex. For simplicity, use the direct formula assuming unit quaternion
     * (norm ≈ 1 during optimization since Adam keeps it near 1). */

    /* dR/dw */
    dR_dq[0][0][0] = 0;           dR_dq[0][0][1] = -2*z*inv_n;  dR_dq[0][0][2] = 2*y*inv_n;
    dR_dq[0][1][0] = 2*z*inv_n;   dR_dq[0][1][1] = 0;           dR_dq[0][1][2] = -2*x*inv_n;
    dR_dq[0][2][0] = -2*y*inv_n;  dR_dq[0][2][1] = 2*x*inv_n;   dR_dq[0][2][2] = 0;

    /* dR/dx */
    dR_dq[1][0][0] = 0;           dR_dq[1][0][1] = 2*y*inv_n;   dR_dq[1][0][2] = 2*z*inv_n;
    dR_dq[1][1][0] = 2*y*inv_n;   dR_dq[1][1][1] = -4*x*inv_n;  dR_dq[1][1][2] = -2*w*inv_n;
    dR_dq[1][2][0] = 2*z*inv_n;   dR_dq[1][2][1] = 2*w*inv_n;   dR_dq[1][2][2] = -4*x*inv_n;

    /* dR/dy */
    dR_dq[2][0][0] = -4*y*inv_n;  dR_dq[2][0][1] = 2*x*inv_n;   dR_dq[2][0][2] = 2*w*inv_n;
    dR_dq[2][1][0] = 2*x*inv_n;   dR_dq[2][1][1] = 0;           dR_dq[2][1][2] = 2*z*inv_n;
    dR_dq[2][2][0] = -2*w*inv_n;  dR_dq[2][2][1] = 2*z*inv_n;   dR_dq[2][2][2] = -4*y*inv_n;

    /* dR/dz */
    dR_dq[3][0][0] = -4*z*inv_n;  dR_dq[3][0][1] = -2*w*inv_n;  dR_dq[3][0][2] = 2*x*inv_n;
    dR_dq[3][1][0] = 2*w*inv_n;   dR_dq[3][1][1] = -4*z*inv_n;  dR_dq[3][1][2] = 2*y*inv_n;
    dR_dq[3][2][0] = 2*x*inv_n;   dR_dq[3][2][1] = 2*y*inv_n;   dR_dq[3][2][2] = 0;
}

/* ------------------------------------------------------------------ */
/* Affine grid backward: compute dL/dA from dL/dgrid                   */
/* ------------------------------------------------------------------ */

/* dL/dA[i][j] = sum over all grid points of dL/dgrid[p][i] * coord[p][j]
 * where coord includes the homogeneous 1 for j=3.
 * A is [3, 4], grid is [D, H, W, 3], coord is normalized [-1, 1].
 */
static void affine_grid_backward(const tensor_t *grad_grid,
                                 int D, int H, int W,
                                 float dL_dA[3][4]) {
    memset(dL_dA, 0, 12 * sizeof(float));
    const float *gg = tensor_data_f32(grad_grid);

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
                        dL_dA[i][j] += gg[idx + i] * coord[j];
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Rigid registration                                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    float quat[4];          /* w, x, y, z */
    float transl[3];        /* translation (physical, possibly centered) */
    float moment[3][3];     /* initial rotation from moments */
    float center[3];        /* image center in physical space */
    int around_center;
    /* Adam state */
    float m_q[4], v_q[4];  /* first/second moments for quaternion */
    float m_t[3], v_t[3];  /* first/second moments for translation */
    int step;
} rigid_state_t;

/* Build the full rigid matrix [4, 4] from current state */
static void build_rigid_matrix(const rigid_state_t *s, float mat[4][4]) {
    float R[3][3];
    quat_to_rotmat(s->quat, R);

    /* Apply moment: R_final = R @ moment */
    float RM[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            RM[i][j] = 0;
            for (int k = 0; k < 3; k++)
                RM[i][j] += R[i][k] * s->moment[k][j];
        }

    memset(mat, 0, 16 * sizeof(float));
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            mat[i][j] = RM[i][j];
    mat[3][3] = 1.0f;

    /* Translation: if around_center, convert t' back to t */
    float t[3];
    if (s->around_center) {
        for (int i = 0; i < 3; i++) {
            t[i] = s->transl[i] + s->center[i];
            for (int j = 0; j < 3; j++)
                t[i] -= RM[i][j] * s->center[j];
        }
    } else {
        for (int i = 0; i < 3; i++)
            t[i] = s->transl[i];
    }
    for (int i = 0; i < 3; i++)
        mat[i][3] = t[i];
}

/* Build combined torch-space affine [3, 4] tensor */
static void build_torch_affine(const rigid_state_t *s,
                                const mat44d *fixed_t2p,
                                const mat44d *moving_p2t,
                                tensor_t *aff_tensor) {
    float rigid44[4][4];
    build_rigid_matrix(s, rigid44);

    /* combined = moving_p2t @ rigid44 @ fixed_t2p */
    mat44d rigid_d, tmp, combined;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            rigid_d.m[i][j] = rigid44[i][j];

    mat44d_mul(&tmp, &rigid_d, fixed_t2p);
    mat44d_mul(&combined, moving_p2t, &tmp);

    int shape[3] = {1, 3, 4};
    tensor_alloc(aff_tensor, 3, shape, DTYPE_FLOAT32, DEVICE_CPU);
    float *data = tensor_data_f32(aff_tensor);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            data[i * 4 + j] = (float)combined.m[i][j];
}

/* One optimization step: forward + backward + Adam update */
static float rigid_step(rigid_state_t *s,
                        const tensor_t *fixed_down,
                        const tensor_t *moving_blur,
                        const mat44d *fixed_t2p,
                        const mat44d *moving_p2t,
                        int loss_type, int cc_kernel_size, int mi_num_bins,
                        float lr, float beta1, float beta2, float eps) {
    int D = fixed_down->shape[2], H = fixed_down->shape[3], W = fixed_down->shape[4];

    /* Forward: build affine, generate grid, sample, compute loss */
    tensor_t aff_tensor;
    build_torch_affine(s, fixed_t2p, moving_p2t, &aff_tensor);

    int out_shape[3] = {D, H, W};
    tensor_t grid;
    affine_grid_3d(&aff_tensor, out_shape, &grid);

    tensor_t moved;
    cpu_grid_sample_3d_forward(moving_blur, &grid, &moved, 1);

    /* Loss forward + gradient w.r.t. moved image */
    float loss;
    tensor_t grad_moved;
    if (loss_type == LOSS_MI) {
        cpu_mi_loss_3d(&moved, fixed_down, mi_num_bins, &loss, &grad_moved);
    } else {
        cpu_cc_loss_3d(&moved, fixed_down, cc_kernel_size, &loss, &grad_moved);
    }

    /* Backward through grid_sample: dL/dgrid */
    tensor_t grad_grid;
    cpu_grid_sample_3d_backward(&grad_moved, moving_blur, &grid, &grad_grid, 1);

    /* Backward through affine_grid: dL/dA [3, 4] */
    float dL_dA[3][4];
    affine_grid_backward(&grad_grid, D, H, W, dL_dA);

    /* Now chain rule: dL/dA is the gradient of loss w.r.t. the combined
     * torch-space affine. We need dL/d(rigid44) and then dL/d(quat), dL/d(transl).
     *
     * combined = moving_p2t @ rigid44 @ fixed_t2p
     * dL/d(rigid44) = moving_p2t^T @ dL/d(combined) @ fixed_t2p^T
     *
     * But this is for the full 4x4. Since we only have dL/dA for [3,4],
     * we pad dL/dA to [4,4] with zeros in the last row.
     */
    mat44d dL_dA_44;
    memset(&dL_dA_44, 0, sizeof(mat44d));
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            dL_dA_44.m[i][j] = dL_dA[i][j];

    /* dL/d(rigid44) = p2t^T @ dL/dA_44 @ t2p^T */
    mat44d p2t_T, t2p_T, tmp1, dL_drigid;
    /* Transpose p2t and t2p */
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            p2t_T.m[i][j] = moving_p2t->m[j][i];
            t2p_T.m[i][j] = fixed_t2p->m[j][i];
        }
    mat44d_mul(&tmp1, &p2t_T, &dL_dA_44);
    mat44d_mul(&dL_drigid, &tmp1, &t2p_T);

    /* Extract dL/d(RM) [3,3] and dL/dt [3] from dL/drigid */
    float dL_dRM[3][3], dL_dt[3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            dL_dRM[i][j] = (float)dL_drigid.m[i][j];
        dL_dt[i] = (float)dL_drigid.m[i][3];
    }

    /* If around_center, dL/d(transl') = dL/dt (direct, since t = t' + c - RM*c) */
    float dL_dtransl[3];
    for (int i = 0; i < 3; i++)
        dL_dtransl[i] = dL_dt[i];

    /* Also, if around_center, dL/d(RM) gets an extra term from the translation:
     * t = t' + c - RM*c, so dt/dRM = -c^T (outer product contribution)
     * dL/dRM += dL/dt @ (-c)^T */
    if (s->around_center) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                dL_dRM[i][j] -= dL_dt[i] * s->center[j];
    }

    /* dL/dR: since RM = R @ moment, dL/dR = dL/dRM @ moment^T */
    float dL_dR[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            dL_dR[i][j] = 0;
            for (int k = 0; k < 3; k++)
                dL_dR[i][j] += dL_dRM[i][k] * s->moment[j][k]; /* moment^T */
        }

    /* dL/dquat: chain through quaternion → R */
    float dR_dq[4][3][3];
    quat_rotmat_jacobian(s->quat, dR_dq);
    float dL_dq[4] = {0};
    for (int k = 0; k < 4; k++)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                dL_dq[k] += dL_dR[i][j] * dR_dq[k][i][j];

    /* Adam update for quaternion */
    s->step++;
    float bc1 = 1.0f - powf(beta1, (float)s->step);
    float bc2 = 1.0f - powf(beta2, (float)s->step);
    float step_size = lr / bc1;

    for (int k = 0; k < 4; k++) {
        s->m_q[k] = beta1 * s->m_q[k] + (1 - beta1) * dL_dq[k];
        s->v_q[k] = beta2 * s->v_q[k] + (1 - beta2) * dL_dq[k] * dL_dq[k];
        float denom = sqrtf(s->v_q[k] / bc2) + eps;
        s->quat[k] -= step_size * s->m_q[k] / denom;
    }

    /* Adam update for translation */
    for (int k = 0; k < 3; k++) {
        s->m_t[k] = beta1 * s->m_t[k] + (1 - beta1) * dL_dtransl[k];
        s->v_t[k] = beta2 * s->v_t[k] + (1 - beta2) * dL_dtransl[k] * dL_dtransl[k];
        float denom = sqrtf(s->v_t[k] / bc2) + eps;
        s->transl[k] -= step_size * s->m_t[k] / denom;
    }

    /* Cleanup */
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

int rigid_register(const image_t *fixed, const image_t *moving,
                   const moments_result_t *moments_init,
                   rigid_opts_t opts, rigid_result_t *result) {
    memset(result, 0, sizeof(rigid_result_t));

    /* Initialize state */
    rigid_state_t state;
    memset(&state, 0, sizeof(state));
    state.quat[0] = 1.0f; /* identity quaternion */
    state.around_center = 1;

    /* Copy moment initialization */
    if (moments_init) {
        memcpy(state.moment, moments_init->Rf, sizeof(state.moment));
        memcpy(state.transl, moments_init->tf, sizeof(state.transl));
    } else {
        state.moment[0][0] = state.moment[1][1] = state.moment[2][2] = 1.0f;
    }

    /* Center = fixed image center in physical space (torch2phy[:3, 3]) */
    for (int i = 0; i < 3; i++)
        state.center[i] = (float)fixed->meta.torch2phy.m[i][3];

    /* If around_center, convert initial translation:
     * t' = t - center + RM*center, where RM = I*moment initially */
    if (state.around_center && moments_init) {
        float RM[3][3];
        memcpy(RM, state.moment, sizeof(RM));
        for (int i = 0; i < 3; i++) {
            state.transl[i] = moments_init->tf[i] - state.center[i];
            for (int j = 0; j < 3; j++)
                state.transl[i] += RM[i][j] * state.center[j];
        }
    }

    /* Multi-scale optimization */
    int n_scales = opts.n_scales;
    const int *scales = opts.scales;
    const int *iterations = opts.iterations;

    int fD = fixed->data.shape[2], fH = fixed->data.shape[3], fW = fixed->data.shape[4];
    int mD = moving->data.shape[2], mH = moving->data.shape[3], mW = moving->data.shape[4];

    for (int si = 0; si < n_scales; si++) {
        int scale = scales[si];
        int iters = iterations[si];

        /* Downsample fixed and moving images (matching GPU pipeline):
         * - Fixed and moving downsampled independently (may have different sizes)
         * - Anti-aliasing Gaussian blur before trilinear resize
         * - Extra Gaussian blur on moving (Python _smooth_image_not_mask) */
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
            /* Extra Gaussian blur on moving (matching Python rigid._smooth_image_not_mask) */
            cpu_blur_volume(tensor_data_f32(&moving_down), mdD, mdH, mdW,
                             0.5f * (float)fD / (float)dD,
                             0.5f * (float)fH / (float)dH,
                             0.5f * (float)fW / (float)dW,
                             2.0f);
        } else {
            tensor_view(&fixed_down, &fixed->data);
            tensor_view(&moving_down, &moving->data);
            mdD = mD; mdH = mH; mdW = mW;
        }

        fprintf(stderr, "  Scale %d: fixed[%d,%d,%d] moving[%d,%d,%d] x %d iters\n",
                scale, dD, dH, dW, mdD, mdH, mdW, iters);

        /* Reset Adam step counter for each scale */
        state.step = 0;
        memset(state.m_q, 0, sizeof(state.m_q));
        memset(state.v_q, 0, sizeof(state.v_q));
        memset(state.m_t, 0, sizeof(state.m_t));
        memset(state.v_t, 0, sizeof(state.v_t));

        float prev_loss = 1e30f;
        int converge_count = 0;

        for (int i = 0; i < iters; i++) {
            float loss = rigid_step(&state, &fixed_down, &moving_down,
                                    &fixed->meta.torch2phy, &moving->meta.phy2torch,
                                    opts.loss_type, opts.cc_kernel_size,
                                    opts.mi_num_bins > 0 ? opts.mi_num_bins : 32,
                                    opts.lr, 0.9f, 0.999f, 1e-8f);

            if (i % 50 == 0 || i == iters - 1)
                fprintf(stderr, "    iter %d/%d loss=%.6f\n", i, iters, loss);

            /* Convergence check */
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

    /* Extract final result */
    float rigid44[4][4];
    build_rigid_matrix(&state, rigid44);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            result->rigid_mat[i][j] = rigid44[i][j];
        result->rigid_mat[i][3] = rigid44[i][3];
    }

    /* Evaluate NCC on full-resolution */
    tensor_t moved;
    apply_affine_transform(fixed, moving, result->rigid_mat, &moved);
    cpu_cc_loss_3d(&moved, &fixed->data, 9, &result->ncc_loss, NULL);
    tensor_free(&moved);

    return 0;
}
