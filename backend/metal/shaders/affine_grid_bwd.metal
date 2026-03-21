/*
 * affine_grid_bwd.metal - Reduce grad_grid [D,H,W,3] to dL/dA [12]
 *
 * Each workgroup produces a partial sum of 12 values.
 * Final reduction done on CPU (same pattern as CUDA).
 *
 * Faithful translation of backend/cuda/linear_gpu.cu affine_grid_bwd_kernel
 * and backend/webgpu/shaders/affine_grid_bwd.wgsl.
 */

#include <metal_stdlib>
using namespace metal;

struct AGBParams {
    uint D, H, W, _pad;
};

/*
 * affine_grid_bwd kernel
 *
 * For each grid point, compute outer product grad_grid[i] * coord[j]
 * and accumulate via threadgroup reduction to produce partial sums.
 *
 * grad_grid: [D*H*W*3] flat float array
 * partial:   [n_threadgroups * 12] output partial sums
 */
kernel void affine_grid_bwd(
    device const float *grad_grid [[buffer(0)]],
    device       float *partial   [[buffer(1)]],
    constant AGBParams &p         [[buffer(2)]],
    uint tid   [[thread_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint gid   [[threadgroup_position_in_grid]])
{
    /* Shared memory: 256 threads * 12 values = 3072 floats */
    threadgroup float sdata[256 * 12];

    uint total = p.D * p.H * p.W;

    /* Initialize shared memory */
    for (uint k = 0; k < 12; k++) {
        sdata[lid * 12 + k] = 0.0f;
    }

    if (tid < total) {
        uint w = tid % p.W;
        uint tmp = tid / p.W;
        uint h = tmp % p.H;
        uint d = tmp / p.H;

        float nz = (p.D > 1) ? (2.0f * float(d) / float(p.D - 1) - 1.0f) : 0.0f;
        float ny = (p.H > 1) ? (2.0f * float(h) / float(p.H - 1) - 1.0f) : 0.0f;
        float nx = (p.W > 1) ? (2.0f * float(w) / float(p.W - 1) - 1.0f) : 0.0f;

        /* Use float4 to avoid variable indexing into arrays */
        float4 coord = float4(nx, ny, nz, 1.0f);

        uint gi = tid * 3;
        float3 gg = float3(grad_grid[gi], grad_grid[gi + 1], grad_grid[gi + 2]);

        /* Unrolled outer product: sdata[lid*12 + i*4 + j] = gg[i] * coord[j] */
        uint base = lid * 12;
        sdata[base + 0]  = gg[0] * coord[0];
        sdata[base + 1]  = gg[0] * coord[1];
        sdata[base + 2]  = gg[0] * coord[2];
        sdata[base + 3]  = gg[0] * coord[3];
        sdata[base + 4]  = gg[1] * coord[0];
        sdata[base + 5]  = gg[1] * coord[1];
        sdata[base + 6]  = gg[1] * coord[2];
        sdata[base + 7]  = gg[1] * coord[3];
        sdata[base + 8]  = gg[2] * coord[0];
        sdata[base + 9]  = gg[2] * coord[1];
        sdata[base + 10] = gg[2] * coord[2];
        sdata[base + 11] = gg[2] * coord[3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Tree reduction within threadgroup */
    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s) {
            for (uint k = 0; k < 12; k++) {
                sdata[lid * 12 + k] += sdata[(lid + s) * 12 + k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* Thread 0 writes the partial sum for this threadgroup */
    if (lid == 0) {
        for (uint k = 0; k < 12; k++) {
            partial[gid * 12 + k] = sdata[k];
        }
    }
}
