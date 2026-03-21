/*
 * cc_loss.cu - CUDA CC loss with gradient (rectangular kernel, unsigned)
 *
 * Computes local NCC using separable box filtering on GPU.
 * Formula:
 *   cross = box(P*T) - box(P)*box(T)
 *   p_var = max(box(P^2) - box(P)^2, eps)
 *   t_var = max(box(T^2) - box(T)^2, eps)
 *   ncc = (cross^2 + nr) / (p_var * t_var + dr)
 *   loss = -mean(ncc)
 *   grad_P = box_adj(dncc/dbox(P)) + 2*P*box_adj(dncc/dbox(P^2)) + T*box_adj(dncc/dbox(PT))
 */

#include "kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define BLOCK 256

/* ------------------------------------------------------------------ */
/* Separable box filter: 3 passes along D, H, W                       */
/* ------------------------------------------------------------------ */

static void separable_box_filter_gpu(const float *in, float *out,
                                     int D, int H, int W, int ks,
                                     float *tmp) {
    float scale = 1.0f / ks;
    /* D axis: in -> tmp */
    cuda_box_filter_axis(in, tmp, D, H, W, ks, 0, scale);
    /* H axis: tmp -> out */
    cuda_box_filter_axis(tmp, out, D, H, W, ks, 1, scale);
    /* W axis: out -> tmp, copy back */
    cuda_box_filter_axis(out, tmp, D, H, W, ks, 2, scale);
    cudaMemcpy(out, tmp, (size_t)D*H*W*sizeof(float), cudaMemcpyDeviceToDevice);
}

/* ------------------------------------------------------------------ */
/* Element-wise kernels for CC                                         */
/* ------------------------------------------------------------------ */

__global__ void multiply_kernel(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

__global__ void cc_ncc_and_grad_sources_kernel(
    const float *p_sum, const float *t_sum,
    const float *p2_sum, const float *t2_sum, const float *tp_sum,
    const float *P, const float *T,
    float *ncc_out,        /* per-voxel NCC (for loss reduction) */
    float *src_p,          /* gradient source: dncc/d(p_sum) */
    float *src_p2,         /* gradient source: dncc/d(p2_sum) */
    float *src_tp,         /* gradient source: dncc/d(tp_sum) */
    int n, float nr, float dr, int compute_grad)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float ps = p_sum[i], ts = t_sum[i];
    float cross = tp_sum[i] - ps * ts;
    float p_var = p2_sum[i] - ps * ps;
    float t_var = t2_sum[i] - ts * ts;
    if (p_var < dr) p_var = dr;
    if (t_var < dr) t_var = dr;

    float f = cross * cross + nr;
    float g = p_var * t_var + dr;
    float ncc = f / g;
    if (ncc > 1.0f) ncc = 1.0f;
    if (ncc < -1.0f) ncc = -1.0f;
    ncc_out[i] = ncc;

    if (compute_grad) {
        float g2 = g * g;
        src_tp[i] = 2.0f * cross * g / g2;
        src_p2[i] = -f * t_var / g2;
        src_p[i] = (-2.0f * cross * ts * g + 2.0f * f * ps * t_var) / g2;
    }
}

__global__ void cc_combine_grad_kernel(
    const float *adj_p, const float *adj_p2, const float *adj_tp,
    const float *P, const float *T,
    float *grad_out, int n, float inv_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_out[i] = -inv_count * (adj_p[i] + 2.0f*P[i]*adj_p2[i] + T[i]*adj_tp[i]);
}

/* ------------------------------------------------------------------ */
/* Sum reduction (simple two-pass for moderate sizes)                  */
/* ------------------------------------------------------------------ */

__global__ void partial_sum_kernel(const float *data, float *partial, int n) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? data[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

static float gpu_sum_reduce(const float *data, int n) {
    int blocks = (n + BLOCK - 1) / BLOCK;
    float *d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(float));
    partial_sum_kernel<<<blocks, BLOCK>>>(data, d_partial, n);

    /* Second pass if needed */
    float *h_partial = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < blocks; i++) sum += h_partial[i];
    free(h_partial);
    cudaFree(d_partial);
    return (float)sum;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

extern "C" {

void cuda_cc_loss_3d(
    const float *pred, const float *target,
    float *grad_pred,          /* NULL if no gradient needed */
    int D, int H, int W, int ks,
    float *h_loss_out)         /* host pointer for scalar loss */
{
    int n = D * H * W;
    int blocks = (n + BLOCK - 1) / BLOCK;
    float nr = 1e-5f, dr = 1e-5f;

    /* Allocate work buffers on GPU */
    float *p_sum, *t_sum, *p2_sum, *t2_sum, *tp_sum, *work, *tmp;
    cudaMalloc(&p_sum,  n*sizeof(float));
    cudaMalloc(&t_sum,  n*sizeof(float));
    cudaMalloc(&p2_sum, n*sizeof(float));
    cudaMalloc(&t2_sum, n*sizeof(float));
    cudaMalloc(&tp_sum, n*sizeof(float));
    cudaMalloc(&work,   n*sizeof(float));
    cudaMalloc(&tmp,    n*sizeof(float));

    /* Box filter the 5 intermediates */
    separable_box_filter_gpu(pred, p_sum, D, H, W, ks, tmp);
    separable_box_filter_gpu(target, t_sum, D, H, W, ks, tmp);

    multiply_kernel<<<blocks, BLOCK>>>(pred, pred, work, n);
    separable_box_filter_gpu(work, p2_sum, D, H, W, ks, tmp);

    multiply_kernel<<<blocks, BLOCK>>>(target, target, work, n);
    separable_box_filter_gpu(work, t2_sum, D, H, W, ks, tmp);

    multiply_kernel<<<blocks, BLOCK>>>(pred, target, work, n);
    separable_box_filter_gpu(work, tp_sum, D, H, W, ks, tmp);

    /* Allocate gradient source terms */
    float *ncc_buf, *src_p = NULL, *src_p2 = NULL, *src_tp = NULL;
    cudaMalloc(&ncc_buf, n*sizeof(float));
    int compute_grad = (grad_pred != NULL);
    if (compute_grad) {
        cudaMalloc(&src_p,  n*sizeof(float));
        cudaMalloc(&src_p2, n*sizeof(float));
        cudaMalloc(&src_tp, n*sizeof(float));
    }

    /* Compute NCC and gradient source terms */
    cc_ncc_and_grad_sources_kernel<<<blocks, BLOCK>>>(
        p_sum, t_sum, p2_sum, t2_sum, tp_sum,
        pred, target, ncc_buf,
        src_p, src_p2, src_tp,
        n, nr, dr, compute_grad);

    /* Reduce NCC to scalar loss */
    float ncc_sum = gpu_sum_reduce(ncc_buf, n);
    if (h_loss_out)
        *h_loss_out = -(ncc_sum / n);

    /* Compute gradient if requested */
    if (compute_grad) {
        float *adj_p, *adj_p2, *adj_tp;
        cudaMalloc(&adj_p,  n*sizeof(float));
        cudaMalloc(&adj_p2, n*sizeof(float));
        cudaMalloc(&adj_tp, n*sizeof(float));

        separable_box_filter_gpu(src_p,  adj_p,  D, H, W, ks, tmp);
        separable_box_filter_gpu(src_p2, adj_p2, D, H, W, ks, tmp);
        separable_box_filter_gpu(src_tp, adj_tp, D, H, W, ks, tmp);

        float inv_count = 1.0f / n;
        cc_combine_grad_kernel<<<blocks, BLOCK>>>(
            adj_p, adj_p2, adj_tp, pred, target, grad_pred, n, inv_count);

        cudaFree(adj_p); cudaFree(adj_p2); cudaFree(adj_tp);
        cudaFree(src_p); cudaFree(src_p2); cudaFree(src_tp);
    }

    cudaFree(ncc_buf);
    cudaFree(p_sum); cudaFree(t_sum);
    cudaFree(p2_sum); cudaFree(t2_sum); cudaFree(tp_sum);
    cudaFree(work); cudaFree(tmp);
}

} /* extern "C" */
