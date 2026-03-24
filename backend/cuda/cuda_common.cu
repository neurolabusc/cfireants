/*
 * cuda_common.cu - Shared CUDA helper kernels
 *
 * Kernels used by both greedy_gpu.cu and syn_gpu.cu.
 * Eliminates duplication of vec_add, vec_scale, permute, max_l2_norm,
 * and make_gpu_gauss across CUDA registration files.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define COMMON_BLK 512

/* ------------------------------------------------------------------ */
/* Elementwise kernels                                                 */
/* ------------------------------------------------------------------ */

__global__ void cuda_vec_add_k(float *a, const float *b, const float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = b[i] + c[i];
}

__global__ void cuda_vec_scale_k(float *a, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= alpha;
}

/* ------------------------------------------------------------------ */
/* Permute: DHW3 <-> 3DHW                                              */
/* ------------------------------------------------------------------ */

__global__ void cuda_permute_dhw3_to_3dhw_k(const float *src, float *dst, long spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < spatial)
        for (int c = 0; c < 3; c++)
            dst[(long)c * spatial + idx] = src[(long)idx * 3 + c];
}

__global__ void cuda_permute_3dhw_to_dhw3_k(const float *src, float *dst, long spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < spatial)
        for (int c = 0; c < 3; c++)
            dst[(long)idx * 3 + c] = src[(long)c * spatial + idx];
}

/* ------------------------------------------------------------------ */
/* Max L2 norm reduction (for WarpAdam gradient normalization)         */
/* ------------------------------------------------------------------ */

__global__ void cuda_max_l2_norm_k(const float *grad, float *partial_max, long spatial) {
    __shared__ float smax[COMMON_BLK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0;
    if (i < spatial) {
        float gx = grad[i * 3], gy = grad[i * 3 + 1], gz = grad[i * 3 + 2];
        val = sqrtf(gx * gx + gy * gy + gz * gz);
    }
    smax[tid] = val;
    __syncthreads();
    for (int s = COMMON_BLK / 2; s > 0; s >>= 1) {
        if (tid < s && smax[tid + s] > smax[tid]) smax[tid] = smax[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_max[blockIdx.x] = smax[0];
}

extern "C" {

/* C-callable wrappers for kernels (kernels can't be extern across .cu files) */
void cuda_vec_add(float *a, const float *b, const float *c, int n) {
    int blocks = (n + COMMON_BLK - 1) / COMMON_BLK;
    cuda_vec_add_k<<<blocks, COMMON_BLK>>>(a, b, c, n);
}

void cuda_vec_scale(float *a, float alpha, int n) {
    int blocks = (n + COMMON_BLK - 1) / COMMON_BLK;
    cuda_vec_scale_k<<<blocks, COMMON_BLK>>>(a, alpha, n);
}

void cuda_permute_dhw3_to_3dhw(const float *src, float *dst, long spatial) {
    int blocks = (spatial + COMMON_BLK - 1) / COMMON_BLK;
    cuda_permute_dhw3_to_3dhw_k<<<blocks, COMMON_BLK>>>(src, dst, spatial);
}

void cuda_permute_3dhw_to_dhw3(const float *src, float *dst, long spatial) {
    int blocks = (spatial + COMMON_BLK - 1) / COMMON_BLK;
    cuda_permute_3dhw_to_dhw3_k<<<blocks, COMMON_BLK>>>(src, dst, spatial);
}

float cuda_max_l2_norm(const float *grad, long spatial, float eps) {
    int blocks = (spatial + COMMON_BLK - 1) / COMMON_BLK;
    float *d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(float));
    cuda_max_l2_norm_k<<<blocks, COMMON_BLK>>>(grad, d_partial, spatial);
    float *h = (float *)malloc(blocks * sizeof(float));
    cudaMemcpy(h, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float maxval = 0;
    for (int i = 0; i < blocks; i++)
        if (h[i] > maxval) maxval = h[i];
    free(h);
    cudaFree(d_partial);
    return eps + maxval;
}

/* ------------------------------------------------------------------ */
/* Gaussian kernel builder (erf approximation, GPU-allocated)          */
/* ------------------------------------------------------------------ */

float *cuda_make_gpu_gauss(float sigma, float truncated, int *klen_out) {
    if (sigma <= 0) { *klen_out = 0; return NULL; }
    int tail = (int)(truncated * sigma + 0.5f);
    int klen = 2 * tail + 1;
    float *h = (float *)malloc(klen * sizeof(float));
    float inv = 1.0f / (sigma * sqrtf(2.0f));
    float sum = 0;
    for (int i = 0; i < klen; i++) {
        float x = (float)(i - tail);
        h[i] = 0.5f * (erff((x + 0.5f) * inv) - erff((x - 0.5f) * inv));
        sum += h[i];
    }
    for (int i = 0; i < klen; i++) h[i] /= sum;
    float *d;
    cudaMalloc(&d, klen * sizeof(float));
    cudaMemcpy(d, h, klen * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
    *klen_out = klen;
    return d;
}

} /* extern "C" */
