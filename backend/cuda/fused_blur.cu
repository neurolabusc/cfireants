/*
 * fused_blur.cu - Separable Gaussian blur for [D,H,W,3] displacement fields
 *
 * Operates directly on interleaved [D,H,W,3] layout without permuting.
 * Three passes (D, H, W axes), each handling all 3 channels.
 * Uses ping-pong between input and scratch buffer.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define FB_BLOCK 256

/* ------------------------------------------------------------------ */
/* Separable 1D convolution along each axis for [D,H,W,3] data        */
/* Each thread processes one spatial voxel (all 3 channels)            */
/* ------------------------------------------------------------------ */

__global__ void blur_dhw3_axis0_kernel(
    const float * __restrict__ in,
    float * __restrict__ out,
    int D, int H, int W,
    const float * __restrict__ kernel, int klen)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long spatial = (long)D * H * W;
    if (idx >= spatial) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int d = idx / (H * W);
    int r = klen / 2;

    float sum0 = 0, sum1 = 0, sum2 = 0;
    for (int k = 0; k < klen; k++) {
        int dd = d + k - r;
        if (dd >= 0 && dd < D) {
            int src = ((dd * H + h) * W + w) * 3;
            float kv = kernel[k];
            sum0 += in[src + 0] * kv;
            sum1 += in[src + 1] * kv;
            sum2 += in[src + 2] * kv;
        }
    }
    int dst = idx * 3;
    out[dst + 0] = sum0;
    out[dst + 1] = sum1;
    out[dst + 2] = sum2;
}

__global__ void blur_dhw3_axis1_kernel(
    const float * __restrict__ in,
    float * __restrict__ out,
    int D, int H, int W,
    const float * __restrict__ kernel, int klen)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long spatial = (long)D * H * W;
    if (idx >= spatial) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int d = idx / (H * W);
    int r = klen / 2;

    float sum0 = 0, sum1 = 0, sum2 = 0;
    for (int k = 0; k < klen; k++) {
        int hh = h + k - r;
        if (hh >= 0 && hh < H) {
            int src = ((d * H + hh) * W + w) * 3;
            float kv = kernel[k];
            sum0 += in[src + 0] * kv;
            sum1 += in[src + 1] * kv;
            sum2 += in[src + 2] * kv;
        }
    }
    int dst = idx * 3;
    out[dst + 0] = sum0;
    out[dst + 1] = sum1;
    out[dst + 2] = sum2;
}

__global__ void blur_dhw3_axis2_kernel(
    const float * __restrict__ in,
    float * __restrict__ out,
    int D, int H, int W,
    const float * __restrict__ kernel, int klen)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long spatial = (long)D * H * W;
    if (idx >= spatial) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int d = idx / (H * W);
    int r = klen / 2;

    float sum0 = 0, sum1 = 0, sum2 = 0;
    for (int k = 0; k < klen; k++) {
        int ww = w + k - r;
        if (ww >= 0 && ww < W) {
            int src = ((d * H + h) * W + ww) * 3;
            float kv = kernel[k];
            sum0 += in[src + 0] * kv;
            sum1 += in[src + 1] * kv;
            sum2 += in[src + 2] * kv;
        }
    }
    int dst = idx * 3;
    out[dst + 0] = sum0;
    out[dst + 1] = sum1;
    out[dst + 2] = sum2;
}

extern "C" {

/*
 * In-place separable Gaussian blur on [D,H,W,3] data.
 * Uses pre-allocated scratch buffer of same size.
 * d_kernel: 1D Gaussian kernel on GPU, length klen.
 *
 * Three passes: D→scratch, scratch→data (H), data→scratch (W), copy back.
 * Total: 3 kernel launches + 1 memcpy (vs 14 launches + 3 memcpys before).
 */
void cuda_blur_disp_dhw3(
    float *data,              /* [D,H,W,3] in-place */
    float *scratch,           /* [D,H,W,3] temp buffer */
    int D, int H, int W,
    const float *d_kernel, int klen)
{
    if (klen <= 0) return;
    long spatial = (long)D * H * W;
    int blocks = (spatial + FB_BLOCK - 1) / FB_BLOCK;

    /* Pass 1: D axis, data → scratch */
    blur_dhw3_axis0_kernel<<<blocks, FB_BLOCK>>>(data, scratch, D, H, W, d_kernel, klen);
    /* Pass 2: H axis, scratch → data */
    blur_dhw3_axis1_kernel<<<blocks, FB_BLOCK>>>(scratch, data, D, H, W, d_kernel, klen);
    /* Pass 3: W axis, data → scratch */
    blur_dhw3_axis2_kernel<<<blocks, FB_BLOCK>>>(data, scratch, D, H, W, d_kernel, klen);
    /* Copy result back */
    cudaMemcpy(data, scratch, (size_t)spatial * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
}

} /* extern "C" */
