/*
 * fused_compose.cu - Fused warp composition for compositive update
 *
 * Adapted from fused_ops/FusedGridComposer.cu.
 *
 * Computes: output[x] = input[(identity + grid)[x]]
 * where input and output are [D, H, W, 3] displacement fields
 * and grid is the update direction [D, H, W, 3].
 *
 * This replaces the permute→grid_sample→permute chain in the
 * compositive warp update.
 *
 * Combined with the add (output += grid), this gives the full
 * compositive update: warp_new = grid + interp(warp, identity + grid)
 */

#include <cuda_runtime.h>
#include <math.h>

#define FC_BLOCK 512

/* Unnormalize: [-1,1] → [0, size-1] (align_corners=True) */
__device__ __forceinline__ float fc_unnorm(float x, int size) {
    return (x + 1.0f) * 0.5f * (size - 1);
}

/*
 * Fused compositive warp update kernel.
 *
 * For each output voxel at (d, h, w):
 *   1. Compute identity grid coordinate: norm_coord = [-1,1] mapped from (d,h,w)
 *   2. Add update direction: sample_coord = norm_coord + update[d,h,w,:]
 *   3. Unnormalize to input space
 *   4. Trilinear sample warp field at sample_coord
 *   5. output[d,h,w,:] = update[d,h,w,:] + sampled_warp[:]
 *
 * This is the full compositive update in a single kernel:
 *   warp_new = v + interp(warp, identity + v)
 */
__global__ void fused_compositive_update_kernel(
    const float * __restrict__ warp,     /* [D, H, W, 3] current warp */
    const float * __restrict__ update,   /* [D, H, W, 3] Adam update direction */
    float * __restrict__ output,         /* [D, H, W, 3] result */
    int D, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = D * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int tmp = idx / W;
    int h = tmp % H;
    int d = tmp / H;

    /* Identity grid coordinates in [-1, 1] */
    float nx = (W > 1) ? (2.0f * w / (W - 1) - 1.0f) : 0.0f;
    float ny = (H > 1) ? (2.0f * h / (H - 1) - 1.0f) : 0.0f;
    float nz = (D > 1) ? (2.0f * d / (D - 1) - 1.0f) : 0.0f;

    /* Read update direction */
    int base = idx * 3;
    float vx = update[base + 0];
    float vy = update[base + 1];
    float vz = update[base + 2];

    /* Sample coordinate = identity + update */
    float sx = nx + vx;
    float sy = ny + vy;
    float sz = nz + vz;

    /* Unnormalize to input pixel coordinates */
    float ix = fc_unnorm(sx, W);
    float iy = fc_unnorm(sy, H);
    float iz = fc_unnorm(sz, D);

    /* Floor indices */
    int ix0 = __float2int_rd(ix);
    int iy0 = __float2int_rd(iy);
    int iz0 = __float2int_rd(iz);
    int ix1 = ix0 + 1, iy1 = iy0 + 1, iz1 = iz0 + 1;

    float fx = ix - ix0, fy = iy - iy0, fz = iz - iz0;

    /* Trilinear weights */
    float w000 = (1-fx)*(1-fy)*(1-fz);
    float w001 = fx*(1-fy)*(1-fz);
    float w010 = (1-fx)*fy*(1-fz);
    float w011 = fx*fy*(1-fz);
    float w100 = (1-fx)*(1-fy)*fz;
    float w101 = fx*(1-fy)*fz;
    float w110 = (1-fx)*fy*fz;
    float w111 = fx*fy*fz;

    /* Sample warp field at (iz, iy, ix) — 3 channels interleaved */
    #define WARP_AT(d, h, w, c) \
        (((d)>=0 && (d)<D && (h)>=0 && (h)<H && (w)>=0 && (w)<W) \
         ? warp[((d)*H + (h))*W*3 + (w)*3 + (c)] : 0.0f)

    #pragma unroll
    for (int c = 0; c < 3; c++) {
        float sampled = w000 * WARP_AT(iz0, iy0, ix0, c)
                      + w001 * WARP_AT(iz0, iy0, ix1, c)
                      + w010 * WARP_AT(iz0, iy1, ix0, c)
                      + w011 * WARP_AT(iz0, iy1, ix1, c)
                      + w100 * WARP_AT(iz1, iy0, ix0, c)
                      + w101 * WARP_AT(iz1, iy0, ix1, c)
                      + w110 * WARP_AT(iz1, iy1, ix0, c)
                      + w111 * WARP_AT(iz1, iy1, ix1, c);

        /* output = update + sampled_warp */
        output[base + c] = update[base + c] + sampled;
    }
    #undef WARP_AT
}

extern "C" {

/*
 * Fused compositive warp update: output = update + interp(warp, identity + update)
 *
 * Replaces: permute→grid_sample→permute→add
 * All tensors are [D, H, W, 3] layout.
 * Output can alias update (in-place operation).
 */
void cuda_fused_compositive_update(
    const float *warp,      /* [D, H, W, 3] current warp field */
    const float *update,    /* [D, H, W, 3] Adam update direction */
    float *output,          /* [D, H, W, 3] result (can be same as update) */
    int D, int H, int W)
{
    int total = D * H * W;
    int blocks = (total + FC_BLOCK - 1) / FC_BLOCK;
    fused_compositive_update_kernel<<<blocks, FC_BLOCK>>>(
        warp, update, output, D, H, W);
}

} /* extern "C" */
