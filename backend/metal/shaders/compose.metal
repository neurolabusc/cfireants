/*
 * compose.metal - Fused compositive warp update for Metal
 *
 * Translated from fused_compose.cu.
 *
 * Computes: output[x] = update[x] + interp(warp, identity + update[x])
 * where warp, update, output are [D, H, W, 3] displacement fields.
 *
 * This replaces the permute->grid_sample->permute->add chain in the
 * compositive warp update.
 */

#include <metal_stdlib>
using namespace metal;

struct ComposeParams {
    uint D, H, W, _pad;
};

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

kernel void fused_compositive_update(
    const device float *warp     [[buffer(0)]],  // [D,H,W,3] current warp
    const device float *update   [[buffer(1)]],  // [D,H,W,3] Adam update direction
    device float *output         [[buffer(2)]],  // [D,H,W,3] result
    constant ComposeParams &p    [[buffer(3)]],
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

    // Read update direction
    int base = (int)gid * 3;
    float vx = update[base + 0];
    float vy = update[base + 1];
    float vz = update[base + 2];

    // Sample coordinate = identity + update
    float sx = nx + vx;
    float sy = ny + vy;
    float sz = nz + vz;

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

    // Sample warp field at (iz, iy, ix) for all 3 channels and add update
    for (int c = 0; c < 3; c++) {
        float sampled = w000 * warp_at(warp, iz0, iy0, ix0, c, D, H, W)
                      + w001 * warp_at(warp, iz0, iy0, ix1, c, D, H, W)
                      + w010 * warp_at(warp, iz0, iy1, ix0, c, D, H, W)
                      + w011 * warp_at(warp, iz0, iy1, ix1, c, D, H, W)
                      + w100 * warp_at(warp, iz1, iy0, ix0, c, D, H, W)
                      + w101 * warp_at(warp, iz1, iy0, ix1, c, D, H, W)
                      + w110 * warp_at(warp, iz1, iy1, ix0, c, D, H, W)
                      + w111 * warp_at(warp, iz1, iy1, ix1, c, D, H, W);

        // output = update + sampled_warp
        output[base + c] = update[base + c] + sampled;
    }
}
