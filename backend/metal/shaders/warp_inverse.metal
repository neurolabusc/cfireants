/*
 * warp_inverse.metal - Warp field inversion kernels for Metal
 *
 * Translated from warp_inverse.cu.
 *
 * Provides two kernels for iterative fixed-point warp inversion:
 *   negate_field     - negate a [D,H,W,3] displacement field
 *   warp_inverse_iter - one iteration: out[x] = -interp(u, identity + inv_u[x])
 *
 * The iterative scheme: for n_iters iterations, inv_u = -interp(u, identity + inv_u)
 * progressively refines the inverse displacement field.
 */

#include <metal_stdlib>
using namespace metal;

struct WarpParams {
    uint D, H, W, _pad;
};

// ----------------------------------------------------------------
// negate_field: out[i] = -in[i]
// One thread per element (D*H*W*3 total threads)
// ----------------------------------------------------------------

kernel void negate_field(
    const device float *in   [[buffer(0)]],
    device float *out        [[buffer(1)]],
    constant WarpParams &p   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = p.D * p.H * p.W * 3;
    if (gid >= n) return;
    out[gid] = -in[gid];
}

// ----------------------------------------------------------------
// warp_inverse_iter: one iteration of fixed-point inversion
//   out[x] = -interp(u, identity + inv_u[x])
//
// One thread per spatial voxel (D*H*W total threads).
// All tensors are [D,H,W,3] layout.
// ----------------------------------------------------------------

/* Unnormalize: [-1,1] -> [0, size-1] (align_corners=True) */
static inline float unnorm(float x, int size) {
    return (x + 1.0f) * 0.5f * (size - 1);
}

/* Safe warp fetch with zeros padding */
static inline float warp_at(const device float *warp,
                             int d, int h, int w, int c,
                             int D, int H, int W) {
    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W)
        return warp[((d * H + h) * W + w) * 3 + c];
    return 0.0f;
}

kernel void warp_inverse_iter(
    const device float *u      [[buffer(0)]],  // [D,H,W,3] forward warp to invert
    const device float *inv_u  [[buffer(1)]],  // [D,H,W,3] current inverse estimate
    device float *output       [[buffer(2)]],  // [D,H,W,3] updated inverse estimate
    constant WarpParams &p     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = p.D * p.H * p.W;
    if (gid >= total) return;

    int W = (int)p.W;
    int H = (int)p.H;
    int D = (int)p.D;

    int w = (int)(gid % p.W);
    int tmp = (int)(gid / p.W);
    int h = tmp % H;
    int d = tmp / H;

    // Identity grid coordinates in [-1, 1]
    float nx = (W > 1) ? (2.0f * w / (W - 1) - 1.0f) : 0.0f;
    float ny = (H > 1) ? (2.0f * h / (H - 1) - 1.0f) : 0.0f;
    float nz = (D > 1) ? (2.0f * d / (D - 1) - 1.0f) : 0.0f;

    // Read current inverse estimate at this voxel
    int base = (int)gid * 3;
    float inv_x = inv_u[base + 0];
    float inv_y = inv_u[base + 1];
    float inv_z = inv_u[base + 2];

    // Sample coordinate = identity + inv_u
    float sx = nx + inv_x;
    float sy = ny + inv_y;
    float sz = nz + inv_z;

    // Unnormalize to input pixel coordinates
    float ix = unnorm(sx, W);
    float iy = unnorm(sy, H);
    float iz = unnorm(sz, D);

    // Floor indices
    int ix0 = (int)floor(ix);
    int iy0 = (int)floor(iy);
    int iz0 = (int)floor(iz);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;
    int iz1 = iz0 + 1;

    float fx = ix - (float)ix0;
    float fy = iy - (float)iy0;
    float fz = iz - (float)iz0;

    // Trilinear weights
    float w000 = (1 - fx) * (1 - fy) * (1 - fz);
    float w001 = fx * (1 - fy) * (1 - fz);
    float w010 = (1 - fx) * fy * (1 - fz);
    float w011 = fx * fy * (1 - fz);
    float w100 = (1 - fx) * (1 - fy) * fz;
    float w101 = fx * (1 - fy) * fz;
    float w110 = (1 - fx) * fy * fz;
    float w111 = fx * fy * fz;

    // Sample u at (iz, iy, ix) for all 3 channels, negate result
    for (int c = 0; c < 3; c++) {
        float sampled = w000 * warp_at(u, iz0, iy0, ix0, c, D, H, W)
                      + w001 * warp_at(u, iz0, iy0, ix1, c, D, H, W)
                      + w010 * warp_at(u, iz0, iy1, ix0, c, D, H, W)
                      + w011 * warp_at(u, iz0, iy1, ix1, c, D, H, W)
                      + w100 * warp_at(u, iz1, iy0, ix0, c, D, H, W)
                      + w101 * warp_at(u, iz1, iy0, ix1, c, D, H, W)
                      + w110 * warp_at(u, iz1, iy1, ix0, c, D, H, W)
                      + w111 * warp_at(u, iz1, iy1, ix1, c, D, H, W);

        // output = -sampled_u (negate for inverse)
        output[base + c] = -sampled;
    }
}
