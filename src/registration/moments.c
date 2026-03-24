/*
 * moments.c - Moments-based rigid registration
 *
 * Matches fireants MomentsRegistration:
 * 1. Compute center-of-mass for fixed and moving images in physical space
 * 2. Compute second-order moment matrices (inertia tensors)
 * 3. SVD of moment matrices to extract principal axes
 * 4. Test orientation candidates to resolve sign ambiguity
 * 5. Output rotation Rf and translation tf in physical space
 */

#include "cfireants/registration.h"
#include "cfireants/backend.h"
#include "cfireants/interpolator.h"
#include "cfireants/losses.h"
#include "cfireants/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* 3x3 SVD via Jacobi iteration                                       */
/* ------------------------------------------------------------------ */

static void mat3_identity(float m[3][3]) {
    memset(m, 0, 9 * sizeof(float));
    m[0][0] = m[1][1] = m[2][2] = 1.0f;
}

static void mat3_transpose(float out[3][3], const float in[3][3]) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            out[i][j] = in[j][i];
}

static void mat3_mul(float c[3][3], const float a[3][3], const float b[3][3]) {
    float tmp[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            tmp[i][j] = 0;
            for (int k = 0; k < 3; k++)
                tmp[i][j] += a[i][k] * b[k][j];
        }
    memcpy(c, tmp, 9 * sizeof(float));
}

static float mat3_det(const float m[3][3]) {
    return m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
         - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
         + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
}

/* Jacobi rotation for symmetric 3x3 eigendecomposition.
 * Returns eigenvalues in S[3] (descending) and eigenvectors as columns of V[3][3].
 * Input A is destroyed. */
static void symmetric_3x3_svd(float A[3][3], float U[3][3], float S[3], float Vt[3][3]) {
    /*
     * For symmetric M, SVD(M) gives M = U * diag(S) * Vt where U = V * sign.
     * We compute eigendecomposition M = V * diag(lambda) * Vt, then
     * S = |lambda|, and U columns get sign-corrected.
     *
     * Use Jacobi iteration for the eigendecomposition.
     */
    float V[3][3];
    mat3_identity(V);

    /* Jacobi iteration */
    for (int iter = 0; iter < 50; iter++) {
        /* Find largest off-diagonal element */
        int p = 0, q = 1;
        float maxval = fabsf(A[0][1]);
        if (fabsf(A[0][2]) > maxval) { p = 0; q = 2; maxval = fabsf(A[0][2]); }
        if (fabsf(A[1][2]) > maxval) { p = 1; q = 2; maxval = fabsf(A[1][2]); }

        if (maxval < 1e-10f) break;

        /* Compute Jacobi rotation */
        float app = A[p][p], aqq = A[q][q], apq = A[p][q];
        float tau = (aqq - app) / (2.0f * apq);
        float t;
        if (tau >= 0)
            t = 1.0f / (tau + sqrtf(1.0f + tau * tau));
        else
            t = -1.0f / (-tau + sqrtf(1.0f + tau * tau));

        float c = 1.0f / sqrtf(1.0f + t * t);
        float s = t * c;

        /* Update A */
        float new_pp = app - t * apq;
        float new_qq = aqq + t * apq;
        A[p][p] = new_pp;
        A[q][q] = new_qq;
        A[p][q] = A[q][p] = 0.0f;

        for (int r = 0; r < 3; r++) {
            if (r == p || r == q) continue;
            float arp = A[r][p], arq = A[r][q];
            A[r][p] = A[p][r] = c * arp - s * arq;
            A[r][q] = A[q][r] = s * arp + c * arq;
        }

        /* Update V */
        for (int r = 0; r < 3; r++) {
            float vrp = V[r][p], vrq = V[r][q];
            V[r][p] = c * vrp - s * vrq;
            V[r][q] = s * vrp + c * vrq;
        }
    }

    /* Eigenvalues are on diagonal of A */
    float eig[3] = {A[0][0], A[1][1], A[2][2]};

    /* Sort by descending |eigenvalue| */
    int order[3] = {0, 1, 2};
    for (int i = 0; i < 2; i++)
        for (int j = i + 1; j < 3; j++)
            if (fabsf(eig[order[j]]) > fabsf(eig[order[i]])) {
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }

    /* For symmetric matrix SVD: S = |lambda|, U = V * sign(lambda) */
    for (int i = 0; i < 3; i++) {
        int idx = order[i];
        S[i] = fabsf(eig[idx]);
        float sign = (eig[idx] >= 0) ? 1.0f : -1.0f;
        for (int r = 0; r < 3; r++) {
            U[r][i] = V[r][idx] * sign;
            Vt[i][r] = V[r][idx]; /* Vt = V^T, columns of V become rows of Vt */
        }
    }
}

/* ------------------------------------------------------------------ */
/* Moments registration core                                           */
/* ------------------------------------------------------------------ */

/* Generate physical-space coordinates for each voxel.
 * Returns a flat array [nvox * 3] of (x, y, z) physical coordinates.
 * Uses the image's torch2phy to map from normalized to physical. */
static float *make_physical_coords(const image_meta_t *meta, int D, int H, int W) {
    size_t nvox = (size_t)D * H * W;
    float *coords = (float *)malloc(nvox * 3 * sizeof(float));
    if (!coords) return NULL;

    /* Build affine from identity grid to physical space.
     * identity grid maps voxel (d,h,w) to normalized coord in [-1,1],
     * then torch2phy maps to physical. Combined: physical = torch2phy @ norm_coord */
    const mat44d *t2p = &meta->torch2phy;

    for (int d = 0; d < D; d++) {
        float nz = (D > 1) ? (2.0f * d / (D - 1) - 1.0f) : 0.0f;
        for (int h = 0; h < H; h++) {
            float ny = (H > 1) ? (2.0f * h / (H - 1) - 1.0f) : 0.0f;
            for (int w = 0; w < W; w++) {
                float nx = (W > 1) ? (2.0f * w / (W - 1) - 1.0f) : 0.0f;

                /* Apply torch2phy: physical = R @ [nx, ny, nz]^T + t */
                float px = (float)(t2p->m[0][0]*nx + t2p->m[0][1]*ny + t2p->m[0][2]*nz + t2p->m[0][3]);
                float py = (float)(t2p->m[1][0]*nx + t2p->m[1][1]*ny + t2p->m[1][2]*nz + t2p->m[1][3]);
                float pz = (float)(t2p->m[2][0]*nx + t2p->m[2][1]*ny + t2p->m[2][2]*nz + t2p->m[2][3]);

                size_t idx = ((size_t)d * H + h) * W + w;
                coords[idx * 3 + 0] = px;
                coords[idx * 3 + 1] = py;
                coords[idx * 3 + 2] = pz;
            }
        }
    }
    return coords;
}

/* Compute intensity-weighted center of mass in physical coordinates */
static void compute_com(const float *data, const float *coords,
                        size_t nvox, float com[3]) {
    double sx = 0, sy = 0, sz = 0, sw = 0;
    for (size_t i = 0; i < nvox; i++) {
        float w = data[i];
        sx += w * coords[i * 3 + 0];
        sy += w * coords[i * 3 + 1];
        sz += w * coords[i * 3 + 2];
        sw += w;
    }
    if (sw > 1e-12) {
        com[0] = (float)(sx / sw);
        com[1] = (float)(sy / sw);
        com[2] = (float)(sz / sw);
    } else {
        com[0] = com[1] = com[2] = 0.0f;
    }
}

/* Compute 3x3 second-order moment matrix (inertia tensor) */
static void compute_moment_matrix(const float *data, const float *coords,
                                  size_t nvox, const float com[3],
                                  float M[3][3]) {
    double m[3][3] = {{0}};
    double sw = 0;

    for (size_t i = 0; i < nvox; i++) {
        float w = data[i];
        float dx = coords[i*3+0] - com[0];
        float dy = coords[i*3+1] - com[1];
        float dz = coords[i*3+2] - com[2];
        float xyz[3] = {dx, dy, dz};

        for (int a = 0; a < 3; a++)
            for (int b = a; b < 3; b++)
                m[a][b] += w * xyz[a] * xyz[b];
        sw += w;
    }

    for (int a = 0; a < 3; a++)
        for (int b = a; b < 3; b++) {
            M[a][b] = (float)(m[a][b] / sw);
            M[b][a] = M[a][b];
        }
}


int apply_affine_transform(const image_t *fixed, const image_t *moving,
                           const float affine_34[3][4], tensor_t *output) {
    /* Build 4x4 physical-space affine */
    mat44d phys_aff;
    mat44d_identity(&phys_aff);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            phys_aff.m[i][j] = affine_34[i][j];

    /* Combined torch-space affine:
     * combined = moving_p2t @ phys_aff @ fixed_t2p */
    mat44d tmp, combined;
    mat44d_mul(&tmp, &phys_aff, &fixed->meta.torch2phy);
    mat44d_mul(&combined, &moving->meta.phy2torch, &tmp);

    /* Extract top 3 rows as [1, 3, 4] tensor */
    tensor_t aff_tensor;
    int aff_shape[3] = {1, 3, 4};
    tensor_alloc_cpu_f32(&aff_tensor, 3, aff_shape);
    float *aff_data = tensor_data_f32(&aff_tensor);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            aff_data[i * 4 + j] = (float)combined.m[i][j];

    /* Generate grid and sample */
    int D = fixed->data.shape[2], H = fixed->data.shape[3], W = fixed->data.shape[4];
    int out_shape[3] = {D, H, W};
    tensor_t grid;
    affine_grid_3d(&aff_tensor, out_shape, &grid);
    tensor_free(&aff_tensor);

    cpu_grid_sample_3d_forward(&moving->data, &grid, output, 1);
    tensor_free(&grid);
    return 0;
}

/* GPU candidate evaluation (defined in moments_gpu.cu when CUDA available) */
#ifdef CFIREANTS_HAS_CUDA
extern float moments_eval_candidate_gpu(const image_t *fixed, const image_t *moving,
                                         const float aff[3][4], int cc_ks,
                                         void *gpu_state);
extern void *moments_gpu_init(const image_t *fixed, const image_t *moving);
extern void moments_gpu_free(void *gpu_state);
#endif

/* Evaluate CC loss for a candidate physical-space affine [3][4] */
static float eval_candidate_cc(const image_t *fixed, const image_t *moving,
                                const float aff[3][4], int cc_ks,
                                void *gpu_state) {
#ifdef CFIREANTS_HAS_CUDA
    if (gpu_state)
        return moments_eval_candidate_gpu(fixed, moving, aff, cc_ks, gpu_state);
#endif
    (void)gpu_state;
    tensor_t moved;
    apply_affine_transform(fixed, moving, aff, &moved);

    float loss;
    cpu_cc_loss_3d(&moved, &fixed->data, cc_ks, &loss, NULL);
    tensor_free(&moved);
    return loss;
}

int moments_register(const image_t *fixed, const image_t *moving,
                     moments_opts_t opts, moments_result_t *result) {
    memset(result, 0, sizeof(moments_result_t));

    int D_f = fixed->data.shape[2], H_f = fixed->data.shape[3], W_f = fixed->data.shape[4];
    int D_m = moving->data.shape[2], H_m = moving->data.shape[3], W_m = moving->data.shape[4];
    size_t nvox_f = (size_t)D_f * H_f * W_f;
    size_t nvox_m = (size_t)D_m * H_m * W_m;

    /* Get image data (skip batch and channel dims) */
    const float *fixed_data = tensor_data_f32(&fixed->data);   /* [1,1,D,H,W] */
    const float *moving_data = tensor_data_f32(&moving->data);

    /* Generate physical coordinates */
    float *coords_f = make_physical_coords(&fixed->meta, D_f, H_f, W_f);
    float *coords_m = make_physical_coords(&moving->meta, D_m, H_m, W_m);

    /* Compute centers of mass */
    float com_f[3], com_m[3];
    compute_com(fixed_data, coords_f, nvox_f, com_f);
    compute_com(moving_data, coords_m, nvox_m, com_m);

    if (cfireants_verbose >= 2) fprintf(stderr, "  COM fixed:  [%.4f, %.4f, %.4f]\n", com_f[0], com_f[1], com_f[2]);
    if (cfireants_verbose >= 2) fprintf(stderr, "  COM moving: [%.4f, %.4f, %.4f]\n", com_m[0], com_m[1], com_m[2]);

    if (opts.moments == 1) {
        /* Translation only */
        mat3_identity(result->Rf);
        for (int i = 0; i < 3; i++)
            result->tf[i] = -com_f[i] + com_m[i];
    } else {
        /* Second-order moments */
        float M_f[3][3], M_m[3][3];
        compute_moment_matrix(fixed_data, coords_f, nvox_f, com_f, M_f);
        compute_moment_matrix(moving_data, coords_m, nvox_m, com_m, M_m);

        /* SVD of moment matrices */
        float U_f[3][3], S_f[3], Vt_f[3][3];
        float U_m[3][3], S_m[3], Vt_m[3][3];
        symmetric_3x3_svd(M_f, U_f, S_f, Vt_f);
        symmetric_3x3_svd(M_m, U_m, S_m, Vt_m);

        if (cfireants_verbose >= 2) {
            fprintf(stderr, "  S_f: [%.2f, %.2f, %.2f]  S_m: [%.2f, %.2f, %.2f]\n",
                    S_f[0], S_f[1], S_f[2], S_m[0], S_m[1], S_m[2]);
        }

        /* Transpose U_m for the formula R = (U_f @ D @ U_m^T)^T */
        float U_m_t[3][3];
        mat3_transpose(U_m_t, U_m);

        /* Compute det(U_f @ U_m^T) and apply correction */
        float UfUmt[3][3];
        mat3_mul(UfUmt, U_f, U_m_t);
        float det = mat3_det(UfUmt);

        /* detmat: identity with last diagonal = det */
        float detmat_base[3][3];
        mat3_identity(detmat_base);
        detmat_base[2][2] = (det >= 0) ? 1.0f : -1.0f;

        /* U_f_corrected = U_f @ detmat_base */
        float U_f_corr[3][3];
        mat3_mul(U_f_corr, U_f, detmat_base);

        /* Generate orientation candidates */
        /* For 3D:
         * rot candidates: 4 matrices with det=+1 (one axis positive, rest negative)
         * antirot: 4 with det=-1
         * "both" tests all 8 */
        float candidates[8][3][3];
        int n_cand = 0;

        /* rot: det=+1 */
        float rot[4][3][3];
        for (int c = 0; c < 4; c++) {
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    rot[c][i][j] = (i == j) ? -1.0f : 0.0f;
        }
        rot[0][0][0] = 1.0f;
        rot[1][1][1] = 1.0f;
        rot[2][2][2] = 1.0f;
        /* rot[3] = all positive (negate the all-negative) */
        for (int i = 0; i < 3; i++) rot[3][i][i] = 1.0f;

        /* antirot: det=-1 */
        float antirot[4][3][3];
        for (int c = 0; c < 4; c++) {
            mat3_identity(antirot[c]);
        }
        antirot[0][0][0] = -1.0f;
        antirot[1][1][1] = -1.0f;
        antirot[2][2][2] = -1.0f;
        for (int i = 0; i < 3; i++) antirot[3][i][i] = -1.0f;

        if (opts.orientation == 0 || opts.orientation == 2) { /* rot or both */
            for (int c = 0; c < 4; c++)
                memcpy(candidates[n_cand++], rot[c], 9 * sizeof(float));
        }
        if (opts.orientation == 1 || opts.orientation == 2) { /* antirot or both */
            for (int c = 0; c < 4; c++)
                memcpy(candidates[n_cand++], antirot[c], 9 * sizeof(float));
        }

        /* Evaluate each candidate (GPU-accelerated if available) */
        void *gpu_state = NULL;
#ifdef CFIREANTS_HAS_CUDA
        gpu_state = moments_gpu_init(fixed, moving);
#endif
        float best_loss = 1e30f;
        int best_idx = 0;
        float best_R[3][3];
        mat3_identity(best_R);

        for (int c = 0; c < n_cand; c++) {
            /* R = (U_f_corr @ candidate @ I @ U_m^T)^T */
            float tmp1[3][3], R_cand[3][3];
            mat3_mul(tmp1, U_f_corr, candidates[c]);
            mat3_mul(R_cand, tmp1, U_m_t);
            /* Transpose: R = R_cand^T */
            float Rt[3][3];
            mat3_transpose(Rt, R_cand);

            /* Build affine for this candidate */
            float aff[3][4];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++)
                    aff[i][j] = Rt[i][j];
                aff[i][3] = com_m[i];
                for (int j = 0; j < 3; j++)
                    aff[i][3] -= Rt[i][j] * com_f[j];
            }

            float loss = eval_candidate_cc(fixed, moving, aff, opts.cc_kernel_size, gpu_state);
            if (cfireants_verbose >= 2) fprintf(stderr, "  Candidate %d: CC loss = %.6f\n", c, loss);

            if (loss < best_loss) {
                best_loss = loss;
                best_idx = c;
                memcpy(best_R, Rt, sizeof(best_R));
            }
        }

        /* Optionally evaluate Identity+COM and Pure identity candidates.
         * Not in Python FireANTs — useful when sforms already align images. */
        int best_extra = 0; /* 0=SVD, 1=COM identity, 2=pure identity */
        if (opts.try_identity) {
            /* Identity + COM translation */
            float com_aff[3][4];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++)
                    com_aff[i][j] = (i == j) ? 1.0f : 0.0f;
                com_aff[i][3] = com_m[i] - com_f[i];
            }
            float com_loss = eval_candidate_cc(fixed, moving, com_aff, opts.cc_kernel_size, gpu_state);
            if (cfireants_verbose >= 2) fprintf(stderr, "  Identity+COM candidate: CC loss = %.6f\n", com_loss);
            if (com_loss < best_loss) {
                best_loss = com_loss;
                best_idx = -1;
                best_extra = 1;
                mat3_identity(best_R);
            }

            /* Pure identity (no translation) — best when sforms already align */
            float pure_aff[3][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0}};
            float pure_loss = eval_candidate_cc(fixed, moving, pure_aff, opts.cc_kernel_size, gpu_state);
            if (cfireants_verbose >= 2) fprintf(stderr, "  Pure identity candidate: CC loss = %.6f\n", pure_loss);
            if (pure_loss < best_loss) {
                best_loss = pure_loss;
                best_idx = -2;
                best_extra = 2;
                mat3_identity(best_R);
            }
        }

#ifdef CFIREANTS_HAS_CUDA
        if (gpu_state) moments_gpu_free(gpu_state);
#endif
        if (cfireants_verbose >= 2) {
            if (best_extra == 1)
                fprintf(stderr, "  Best: identity+COM (loss=%.6f)\n", best_loss);
            else if (best_extra == 2)
                fprintf(stderr, "  Best: pure identity (loss=%.6f)\n", best_loss);
            else
                fprintf(stderr, "  Best: SVD candidate %d (loss=%.6f)\n", best_idx, best_loss);
        }

        /* Use best rotation */
        memcpy(result->Rf, best_R, sizeof(result->Rf));

        /* Translation depends on which candidate won */
        if (best_extra == 2) {
            /* Pure identity: no translation */
            result->tf[0] = result->tf[1] = result->tf[2] = 0.0f;
        } else {
            /* SVD or COM identity: tf = com_m - Rf @ com_f */
            for (int i = 0; i < 3; i++) {
                result->tf[i] = com_m[i];
                for (int j = 0; j < 3; j++)
                    result->tf[i] -= result->Rf[i][j] * com_f[j];
            }
        }

        result->ncc_loss = best_loss;

        if (cfireants_verbose >= 2) {
            fprintf(stderr, "  Moments Rf:\n");
            for (int i = 0; i < 3; i++)
                fprintf(stderr, "    [%8.4f %8.4f %8.4f]\n",
                        result->Rf[i][0], result->Rf[i][1], result->Rf[i][2]);
            fprintf(stderr, "  Moments tf: [%.4f, %.4f, %.4f]\n",
                    result->tf[0], result->tf[1], result->tf[2]);
        }
    }

    /* Build combined affine [3, 4] = [Rf | tf] */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            result->affine[i][j] = result->Rf[i][j];
        result->affine[i][3] = result->tf[i];
    }

    free(coords_f);
    free(coords_m);
    return 0;
}
