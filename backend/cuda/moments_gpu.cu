/*
 * moments_gpu.cu - GPU-accelerated moments orientation candidate evaluation
 *
 * Provides GPU versions of the candidate CC evaluation used by moments.c.
 * Images are uploaded once and reused for all 4-8 candidates.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {

#include "cfireants/tensor.h"
#include "cfireants/image.h"
#include "kernels.h"

void cuda_cc_loss_3d(const float *pred, const float *target, float *grad_pred,
                     int D, int H, int W, int ks, float *h_loss_out);

/* GPU state: holds uploaded images and reusable scratch buffers */
typedef struct {
    float *d_fixed;
    float *d_moving;
    float *d_grid;
    float *d_moved;
    int fD, fH, fW;
    int mD, mH, mW;
} moments_gpu_state_t;

void *moments_gpu_init(const image_t *fixed, const image_t *moving) {
    moments_gpu_state_t *s = (moments_gpu_state_t *)malloc(sizeof(moments_gpu_state_t));
    s->fD = fixed->data.shape[2]; s->fH = fixed->data.shape[3]; s->fW = fixed->data.shape[4];
    s->mD = moving->data.shape[2]; s->mH = moving->data.shape[3]; s->mW = moving->data.shape[4];
    size_t fN = (size_t)s->fD * s->fH * s->fW;
    size_t mN = (size_t)s->mD * s->mH * s->mW;

    cudaMalloc(&s->d_fixed, fN * sizeof(float));
    cudaMalloc(&s->d_moving, mN * sizeof(float));
    cudaMalloc(&s->d_grid, fN * 3 * sizeof(float));
    cudaMalloc(&s->d_moved, fN * sizeof(float));

    cudaMemcpy(s->d_fixed, fixed->data.data, fN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(s->d_moving, moving->data.data, mN * sizeof(float), cudaMemcpyHostToDevice);

    return s;
}

void moments_gpu_free(void *gpu_state) {
    moments_gpu_state_t *s = (moments_gpu_state_t *)gpu_state;
    cudaFree(s->d_fixed);
    cudaFree(s->d_moving);
    cudaFree(s->d_grid);
    cudaFree(s->d_moved);
    free(s);
}

float moments_eval_candidate_gpu(const image_t *fixed, const image_t *moving,
                                  const float aff34[3][4], int cc_ks,
                                  void *gpu_state) {
    moments_gpu_state_t *s = (moments_gpu_state_t *)gpu_state;

    /* Build combined torch-space affine on CPU */
    mat44d phys_aff;
    mat44d_identity(&phys_aff);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            phys_aff.m[i][j] = aff34[i][j];

    mat44d tmp, combined;
    mat44d_mul(&tmp, &phys_aff, &fixed->meta.torch2phy);
    mat44d_mul(&combined, &moving->meta.phy2torch, &tmp);

    float h_aff[12];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            h_aff[i*4+j] = (float)combined.m[i][j];

    /* Upload 12-float affine */
    float *d_aff;
    cudaMalloc(&d_aff, 12 * sizeof(float));
    cudaMemcpy(d_aff, h_aff, 12 * sizeof(float), cudaMemcpyHostToDevice);

    /* Generate grid, sample, CC — all on GPU using pre-uploaded images */
    cuda_affine_grid_3d(d_aff, s->d_grid, 1, s->fD, s->fH, s->fW);
    cuda_grid_sample_3d_fwd(s->d_moving, s->d_grid, s->d_moved,
                             1, 1, s->mD, s->mH, s->mW, s->fD, s->fH, s->fW);

    float loss;
    cuda_cc_loss_3d(s->d_moved, s->d_fixed, NULL, s->fD, s->fH, s->fW, cc_ks, &loss);

    cudaFree(d_aff);
    return loss;
}

} /* extern "C" */
