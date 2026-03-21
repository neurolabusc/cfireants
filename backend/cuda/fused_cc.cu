/*
 * fused_cc.cu - Fused CC loss adapted from fused_ops/CrossCorrelation.cu
 *
 * Operates on pre-allocated intermediates buffer to avoid per-call malloc/free.
 * Flow:
 *   1. create_intermediates: pack [I, J, I², J², IJ] into [B, 5C, D, H, W]
 *   2. box_filter: separable convolution on intermediates (all 5 channels)
 *   3. fwd_interm: compute NCC from filtered means → scalar loss
 *   4. bwd_modify_interm: compute gradient multipliers into intermediates
 *   5. box_filter: adjoint (same separable conv) on modified intermediates
 *   6. bwd_compute_grads: final per-voxel gradient
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define FCC_BLOCK 512

extern "C" {

/* From grid_sample.cu */
void cuda_conv1d_axis(const float *in, float *out,
                      int D, int H, int W,
                      const float *kernel, int klen, int axis);
void cuda_box_filter_axis(const float *in, float *out,
                          int D, int H, int W, int ks, int axis, float scale);

/* ------------------------------------------------------------------ */
/* Kernel 1: Create intermediates [B, 5C, D, H, W]                    */
/* Pack: [I, J, I², J², IJ]                                           */
/* ------------------------------------------------------------------ */

__global__ void fcc_create_intermediates_kernel(
    const float * __restrict__ input, const float * __restrict__ target,
    float * __restrict__ interm,
    long n_elements, long spatial)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    float I = input[i];
    float J = target[i];

    /* Intermediates layout: 5 blocks of spatial, channel c maps to 5*c offsets */
    /* For single channel (C=1): interm has 5 spatial-sized blocks */
    interm[i]                = I;
    interm[i + spatial]      = J;
    interm[i + 2 * spatial]  = I * I;
    interm[i + 3 * spatial]  = J * J;
    interm[i + 4 * spatial]  = I * J;
}

/* ------------------------------------------------------------------ */
/* Kernel 2: Forward NCC from filtered intermediates                   */
/* Computes per-voxel NCC, returns sum for reduction                   */
/* ------------------------------------------------------------------ */

__global__ void fcc_fwd_kernel(
    const float * __restrict__ interm,
    float * __restrict__ partial_sum,
    long spatial, int kernel_volume, float nr, float dr)
{
    __shared__ float sdata[FCC_BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0;
    if (i < spatial) {
        float mu    = interm[i];
        float rho   = interm[i + spatial];
        float mu2   = interm[i + 2 * spatial];
        float rho2  = interm[i + 3 * spatial];
        float murho = interm[i + 4 * spatial];

        float kv = (float)kernel_volume;
        float A = kv * (murho - mu * rho);
        float B = fmaxf(kv * (mu2 - mu * mu), dr);
        float C = fmaxf(kv * (rho2 - rho * rho), dr);

        float ncc = (A * A + nr) / (B * C + dr);
        if (ncc < -1.0f) ncc = -1.0f;
        if (ncc > 1.0f) ncc = 1.0f;
        val = ncc;
    }

    sdata[tid] = val;
    __syncthreads();
    for (int s = FCC_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_sum[blockIdx.x] = sdata[0];
}

/* ------------------------------------------------------------------ */
/* Kernel 3: Backward — modify intermediates with gradient multipliers */
/* Overwrites intermediates[0..2] (and optionally [3..4] for target)   */
/* ------------------------------------------------------------------ */

__global__ void fcc_bwd_modify_kernel(
    float * __restrict__ interm,
    const float * __restrict__ input, const float * __restrict__ target,
    long spatial, int kernel_volume, float nr, float dr,
    float grad_output_val, int compute_grad_target)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= spatial) return;

    float mu    = interm[i];
    float rho   = interm[i + spatial];
    float mu2   = interm[i + 2 * spatial];
    float rho2  = interm[i + 3 * spatial];
    float murho = interm[i + 4 * spatial];

    float kv = (float)kernel_volume;
    float A = kv * (murho - mu * rho);
    float B = kv * (mu2 - mu * mu);
    float C = kv * (rho2 - rho * rho);

    float gO = grad_output_val;
    float D = 2.0f * gO * A / (B * C + dr);

    B += dr;
    C += dr;

    /* Write gradient multipliers back into intermediates */
    interm[i]               = D;                          /* D */
    interm[i + spatial]     = D * A / B;                  /* D*A/B */
    interm[i + 2 * spatial] = D * (A / B * mu - rho);    /* D*(A/B*mu - rho) */

    if (compute_grad_target) {
        interm[i + 3 * spatial] = D * A / C;              /* D*A/C */
        interm[i + 4 * spatial] = D * (A / C * rho - mu); /* D*(A/C*rho - mu) */
    }
}

/* ------------------------------------------------------------------ */
/* Kernel 4: Backward — compute final gradients from filtered interm   */
/* ------------------------------------------------------------------ */

__global__ void fcc_bwd_grads_kernel(
    const float * __restrict__ interm,
    const float * __restrict__ input, const float * __restrict__ target,
    float * __restrict__ grad_input, float * __restrict__ grad_target,
    long spatial)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= spatial) return;

    float I = input[i];
    float J = target[i];

    float gini_a  = interm[i];
    float gini_b  = interm[i + spatial];
    float gini_mu = interm[i + 2 * spatial];

    grad_input[i] = gini_a * J - gini_b * I + gini_mu;

    if (grad_target) {
        float gini_c   = interm[i + 3 * spatial];
        float gini_mu2 = interm[i + 4 * spatial];
        grad_target[i] = gini_a * I - gini_c * J + gini_mu2;
    }
}

/* ------------------------------------------------------------------ */
/* Separable box filter on all 5 channels of intermediates             */
/* ------------------------------------------------------------------ */

static void box_filter_intermediates(float *interm, long spatial, int D, int H, int W,
                                      int ks, float *scratch) {
    float scale = 1.0f / ks;
    /* Filter each of the 5 channels */
    for (int ch = 0; ch < 5; ch++) {
        float *data = interm + (long)ch * spatial;
        /* 3 separable passes: D, H, W */
        cuda_box_filter_axis(data, scratch, D, H, W, ks, 0, scale);
        cuda_box_filter_axis(scratch, data, D, H, W, ks, 1, scale);
        cuda_box_filter_axis(data, scratch, D, H, W, ks, 2, scale);
        cudaMemcpy(data, scratch, spatial * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

/* ------------------------------------------------------------------ */
/* Public API: Fused CC loss (pre-allocated workspace version)         */
/* ------------------------------------------------------------------ */

/*
 * Workspace-based fused CC loss.
 *
 * interm: pre-allocated [5 * D * H * W] floats on GPU
 * scratch: pre-allocated [D * H * W] floats on GPU
 *
 * Avoids all per-call malloc/free.
 */
void cuda_fused_cc_loss(
    const float *pred, const float *target,
    float *grad_pred,          /* [D*H*W] or NULL */
    float *grad_target_out,    /* [D*H*W] or NULL */
    int D, int H, int W, int ks,
    float *h_loss_out,         /* host pointer for scalar loss, or NULL */
    float *interm,             /* workspace: 5*D*H*W floats */
    float *scratch)            /* workspace: D*H*W floats */
{
    long spatial = (long)D * H * W;
    int blocks = (spatial + FCC_BLOCK - 1) / FCC_BLOCK;
    float nr = 1e-5f, dr = 1e-5f;
    int kernel_volume = ks * ks * ks;

    /* Step 1: Create intermediates */
    fcc_create_intermediates_kernel<<<blocks, FCC_BLOCK>>>(
        pred, target, interm, spatial, spatial);

    /* Step 2: Box filter intermediates (separable, in-place) */
    box_filter_intermediates(interm, spatial, D, H, W, ks, scratch);

    /* Step 3: Forward NCC */
    if (h_loss_out) {
        float *d_partial;
        cudaMalloc(&d_partial, blocks * sizeof(float));
        fcc_fwd_kernel<<<blocks, FCC_BLOCK>>>(interm, d_partial, spatial, kernel_volume, nr, dr);

        float *h_partial = (float *)malloc(blocks * sizeof(float));
        cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
        double sum = 0;
        for (int i = 0; i < blocks; i++) sum += h_partial[i];
        *h_loss_out = -(float)(sum / spatial);
        free(h_partial);
        cudaFree(d_partial);
    }

    /* Steps 4-6: Backward (if gradient requested) */
    if (grad_pred) {
        int compute_grad_target = (grad_target_out != NULL);
        float grad_output_val = -1.0f / spatial;  /* d(-mean(ncc))/d(ncc) = -1/N */

        /* Step 4: Modify intermediates with gradient multipliers */
        fcc_bwd_modify_kernel<<<blocks, FCC_BLOCK>>>(
            interm, pred, target, spatial, kernel_volume, nr, dr,
            grad_output_val, compute_grad_target);

        /* Step 5: Box filter adjoint (same operation, since box filter is self-adjoint) */
        box_filter_intermediates(interm, spatial, D, H, W, ks, scratch);

        /* Step 6: Compute final gradients */
        fcc_bwd_grads_kernel<<<blocks, FCC_BLOCK>>>(
            interm, pred, target, grad_pred, grad_target_out, spatial);
    }
}

} /* extern "C" */
