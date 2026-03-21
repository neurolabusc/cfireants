/*
 * mi_loss.cu - GPU Mutual Information loss (Gaussian Parzen windowing)
 *
 * Faithful clone of fireants GlobalMutualInformationLoss:
 *   1. Normalize images to [0,1] (divide by max)
 *   2. Gaussian Parzen weights: w[n][b] = exp(-preterm * (img[n] - center[b])^2) / sum
 *   3. Joint histogram: pab[a][b] = (1/N) * sum_n wa[n][a] * wb[n][b]
 *   4. Marginals: pa[a] = (1/N) * sum_n wa[n][a], same for pb
 *   5. MI = sum_ab pab * log((pab + nr) / (pa*pb + dr) + dr)
 *   6. loss = -MI (negated for minimization)
 *
 * Gradient: dL/d(pred[n]) flows through the Parzen weights wa.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define MI_BLOCK 256
#define MAX_BINS 64  /* max supported num_bins */

/* ------------------------------------------------------------------ */
/* Step 1: Compute Parzen weights and accumulate histogram             */
/* ------------------------------------------------------------------ */

/* Each thread processes one voxel: compute its Parzen weights for pred
 * and target, then atomically add to the joint histogram pab[a][b]
 * and marginals pa[a], pb[b]. */
/* Shared-memory histogram accumulation.
 * Each block accumulates into a local shared histogram, then atomicAdd to global. */
__global__ void mi_histogram_kernel(
    const float * __restrict__ pred,
    const float * __restrict__ target,
    float * __restrict__ d_pab,    /* [num_bins * num_bins] */
    float * __restrict__ d_pa,     /* [num_bins] */
    float * __restrict__ d_pb,     /* [num_bins] */
    int N, int num_bins,
    float preterm, const float * __restrict__ d_bin_centers,
    float inv_maxval_p, float inv_maxval_t)
{
    /* Shared memory: pa_local[num_bins] + pb_local[num_bins] + pab_local[num_bins*num_bins] */
    extern __shared__ float smem[];
    float *s_pa = smem;
    float *s_pb = smem + num_bins;
    float *s_pab = smem + 2 * num_bins;
    int hist_size = num_bins * num_bins + 2 * num_bins;

    /* Initialize shared histogram to zero */
    for (int i = threadIdx.x; i < hist_size; i += blockDim.x)
        smem[i] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float pn = pred[idx] * inv_maxval_p;
        float tn = target[idx] * inv_maxval_t;
        if (pn < 0) pn = 0; if (pn > 1) pn = 1;
        if (tn < 0) tn = 0; if (tn > 1) tn = 1;

        float wa[MAX_BINS], wb[MAX_BINS];
        float sum_wa = 0, sum_wb = 0;

        for (int b = 0; b < num_bins; b++) {
            float dp = pn - d_bin_centers[b];
            float dt = tn - d_bin_centers[b];
            wa[b] = expf(-preterm * dp * dp);
            wb[b] = expf(-preterm * dt * dt);
            sum_wa += wa[b];
            sum_wb += wb[b];
        }

        float inv_wa = 1.0f / sum_wa;
        float inv_wb = 1.0f / sum_wb;
        for (int b = 0; b < num_bins; b++) { wa[b] *= inv_wa; wb[b] *= inv_wb; }

        /* Accumulate into shared memory (block-local atomic) */
        float inv_N = 1.0f / N;
        for (int a = 0; a < num_bins; a++) {
            atomicAdd(&s_pa[a], wa[a] * inv_N);
            atomicAdd(&s_pb[a], wb[a] * inv_N);
            for (int b = 0; b < num_bins; b++)
                atomicAdd(&s_pab[a * num_bins + b], wa[a] * wb[b] * inv_N);
        }
    }
    __syncthreads();

    /* Write shared histogram to global (one atomic per element) */
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&d_pa[i], s_pa[i]);
        atomicAdd(&d_pb[i], s_pb[i]);
    }
    for (int i = threadIdx.x; i < num_bins * num_bins; i += blockDim.x)
        atomicAdd(&d_pab[i], s_pab[i]);
}

/* ------------------------------------------------------------------ */
/* Step 2: Compute MI from histogram                                   */
/* ------------------------------------------------------------------ */

__global__ void mi_compute_kernel(
    const float * __restrict__ pab,
    const float * __restrict__ pa,
    const float * __restrict__ pb,
    float * __restrict__ mi_partial,  /* [blocks] partial sums */
    int num_bins, float nr, float dr)
{
    __shared__ float sdata[MI_BLOCK];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_bins * num_bins;

    float val = 0;
    if (idx < total) {
        int a = idx / num_bins;
        int b = idx % num_bins;
        float p = pab[idx];
        float pp = pa[a] * pb[b];
        val = p * logf((p + nr) / (pp + dr) + dr);
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) mi_partial[blockIdx.x] = sdata[0];
}

/* ------------------------------------------------------------------ */
/* Step 3: Compute gradient d(MI)/d(pred[n])                           */
/* ------------------------------------------------------------------ */

/* For each voxel n:
 * dMI/d(pred[n]) = sum_a dMI/d(wa[n][a]) * d(wa[n][a])/d(pred[n])
 *
 * dMI/d(wa[n][a]) = (1/N) * sum_b wb[n][b] * dMI/d(pab[a][b])
 *                 + (1/N) * dMI/d(pa[a])
 *
 * dMI/d(pab[a][b]) = log((pab+nr)/(papb+dr)+dr) + pab/(pab+nr)
 * dMI/d(pa[a]) = -sum_b pab[a][b] * pb[b] / (pa[a]*pb[b] + dr)
 *
 * d(wa[n][a])/d(pred[n]) = wa[n][a] * (-2*preterm*(pn - center[a]) * inv_max)
 *   - wa[n][a] * sum_a' wa[n][a'] * (-2*preterm*(pn - center[a']) * inv_max)
 *   (softmax-style derivative)
 */
__global__ void mi_gradient_kernel(
    const float * __restrict__ pred,
    const float * __restrict__ target,
    const float * __restrict__ pab,
    const float * __restrict__ pa,
    const float * __restrict__ pb,
    float * __restrict__ grad_pred,
    int N, int num_bins,
    float preterm, const float * __restrict__ d_bin_centers,
    float inv_maxval_p, float inv_maxval_t,
    float nr, float dr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float pn = pred[idx] * inv_maxval_p;
    float tn = target[idx] * inv_maxval_t;
    if (pn < 0) pn = 0; if (pn > 1) pn = 1;
    if (tn < 0) tn = 0; if (tn > 1) tn = 1;

    /* Recompute Parzen weights */
    float wa[MAX_BINS], wb[MAX_BINS];
    float sum_wa = 0, sum_wb = 0;
    for (int b = 0; b < num_bins; b++) {
        float dp = pn - d_bin_centers[b];
        float dt = tn - d_bin_centers[b];
        wa[b] = expf(-preterm * dp * dp);
        wb[b] = expf(-preterm * dt * dt);
        sum_wa += wa[b];
        sum_wb += wb[b];
    }
    float inv_swa = 1.0f / sum_wa;
    float inv_swb = 1.0f / sum_wb;
    for (int b = 0; b < num_bins; b++) { wa[b] *= inv_swa; wb[b] *= inv_swb; }

    /* Precompute dMI/d(pab[a][b]) and dMI/d(pa[a]) */
    /* These are the same for all voxels (histogram is global) */
    float inv_N = 1.0f / N;

    /* Compute dMI/d(pred[n]) via chain rule */
    /* d(wa[a])/d(pn) = wa[a] * (du_a - sum_a' wa[a']*du_a')
     * where du_a = -2*preterm*(pn - center[a]) */
    float weighted_du = 0;
    for (int a = 0; a < num_bins; a++)
        weighted_du += wa[a] * (-2.0f * preterm * (pn - d_bin_centers[a]));

    float dmi_dpn = 0;
    for (int a = 0; a < num_bins; a++) {
        float du_a = -2.0f * preterm * (pn - d_bin_centers[a]);
        float dwa_dpn = wa[a] * (du_a - weighted_du);

        /* Through joint histogram */
        for (int b = 0; b < num_bins; b++) {
            float p = pab[a * num_bins + b];
            float pp = pa[a] * pb[b];
            float dmi_dpab = logf((p + nr) / (pp + dr) + dr) + p / (p + nr);
            dmi_dpn += dmi_dpab * inv_N * wb[b] * dwa_dpn;
        }

        /* Through marginal pa */
        float dmi_dpa = 0;
        for (int b = 0; b < num_bins; b++) {
            float pp = pa[a] * pb[b];
            dmi_dpa -= pab[a * num_bins + b] * pb[b] / (pp + dr);
        }
        dmi_dpn += dmi_dpa * inv_N * dwa_dpn;
    }

    /* Chain rule: d(pn)/d(pred[n]) = inv_maxval_p */
    /* Negate for loss = -MI */
    grad_pred[idx] = -dmi_dpn * inv_maxval_p;
}

/* ------------------------------------------------------------------ */
/* Host wrapper                                                        */
/* ------------------------------------------------------------------ */

extern "C" {

void cuda_mi_loss_3d(
    const float *pred, const float *target,
    float *grad_pred,          /* NULL if no gradient needed */
    int D, int H, int W,
    int num_bins, float *h_loss_out)
{
    int N = D * H * W;
    float nr = 1e-7f, dr = 1e-7f;

    if (num_bins > MAX_BINS) {
        fprintf(stderr, "cuda_mi_loss: num_bins=%d exceeds MAX_BINS=%d\n", num_bins, MAX_BINS);
        if (h_loss_out) *h_loss_out = 0;
        return;
    }

    /* Compute max values on GPU (download two floats) */
    /* For simplicity, use thrust-style reduction or just download and compute on CPU */
    float *h_pred = (float *)malloc(N * sizeof(float));
    float *h_target = (float *)malloc(N * sizeof(float));
    cudaMemcpy(h_pred, pred, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_target, target, N * sizeof(float), cudaMemcpyDeviceToHost);

    float pmax = h_pred[0], tmax = h_target[0];
    for (int i = 1; i < N; i++) {
        if (h_pred[i] > pmax) pmax = h_pred[i];
        if (h_target[i] > tmax) tmax = h_target[i];
    }
    free(h_pred); free(h_target);

    float maxval = (pmax > tmax) ? pmax : tmax;
    float inv_maxval_p = (maxval > 1.0f) ? 1.0f / maxval : 1.0f;
    float inv_maxval_t = inv_maxval_p; /* same maxval for both, matching Python */

    /* Build bin centers on device */
    float h_bins[MAX_BINS];
    for (int i = 0; i < num_bins; i++)
        h_bins[i] = (float)i / num_bins + 0.5f / num_bins;

    float bin_spacing = 1.0f / num_bins;
    float sigma = bin_spacing * 0.5f; /* sigma_ratio = 1.0 */
    float preterm = 1.0f / (2.0f * sigma * sigma);

    float *d_bins;
    cudaMalloc(&d_bins, num_bins * sizeof(float));
    cudaMemcpy(d_bins, h_bins, num_bins * sizeof(float), cudaMemcpyHostToDevice);

    /* Allocate histogram on device */
    float *d_pab, *d_pa, *d_pb;
    cudaMalloc(&d_pab, num_bins * num_bins * sizeof(float));
    cudaMalloc(&d_pa, num_bins * sizeof(float));
    cudaMalloc(&d_pb, num_bins * sizeof(float));
    cudaMemset(d_pab, 0, num_bins * num_bins * sizeof(float));
    cudaMemset(d_pa, 0, num_bins * sizeof(float));
    cudaMemset(d_pb, 0, num_bins * sizeof(float));

    /* Step 1: Accumulate histogram */
    int blocks = (N + MI_BLOCK - 1) / MI_BLOCK;
    int smem_size = (num_bins * num_bins + 2 * num_bins) * sizeof(float);
    mi_histogram_kernel<<<blocks, MI_BLOCK, smem_size>>>(
        pred, target, d_pab, d_pa, d_pb,
        N, num_bins, preterm, d_bins,
        inv_maxval_p, inv_maxval_t);

    /* Step 2: Compute MI */
    if (h_loss_out) {
        int hist_total = num_bins * num_bins;
        int mi_blocks = (hist_total + MI_BLOCK - 1) / MI_BLOCK;
        float *d_mi_partial;
        cudaMalloc(&d_mi_partial, mi_blocks * sizeof(float));
        mi_compute_kernel<<<mi_blocks, MI_BLOCK>>>(
            d_pab, d_pa, d_pb, d_mi_partial, num_bins, nr, dr);

        float *h_partial = (float *)malloc(mi_blocks * sizeof(float));
        cudaMemcpy(h_partial, d_mi_partial, mi_blocks * sizeof(float), cudaMemcpyDeviceToHost);
        double mi = 0;
        for (int i = 0; i < mi_blocks; i++) mi += h_partial[i];
        free(h_partial);
        cudaFree(d_mi_partial);

        *h_loss_out = -(float)mi;
    }

    /* Step 3: Gradient */
    if (grad_pred) {
        mi_gradient_kernel<<<blocks, MI_BLOCK>>>(
            pred, target, d_pab, d_pa, d_pb, grad_pred,
            N, num_bins, preterm, d_bins,
            inv_maxval_p, inv_maxval_t, nr, dr);
    }

    cudaFree(d_bins);
    cudaFree(d_pab); cudaFree(d_pa); cudaFree(d_pb);
}

} /* extern "C" */
