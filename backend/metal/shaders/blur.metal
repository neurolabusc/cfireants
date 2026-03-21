/*
 * blur.metal - Separable Gaussian blur for [D,H,W,3] displacement fields
 *
 * 1:1 translation of fused_blur.cu CUDA kernels.
 * Three separate kernel functions (one per axis) to avoid variable
 * indexing issues on Metal. Each thread processes one spatial voxel,
 * all 3 channels. Uses ping-pong between in and out buffers.
 */

#include <metal_stdlib>
using namespace metal;

struct BlurParams {
    uint D;
    uint H;
    uint W;
    uint klen;
};

/* ------------------------------------------------------------------ */
/* Axis 0: convolve along D (depth) axis                               */
/* ------------------------------------------------------------------ */

kernel void blur_dhw3_axis0(
    const device float *in          [[buffer(0)]],
    device float       *out         [[buffer(1)]],
    constant float     *kern        [[buffer(2)]],
    constant BlurParams &p          [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint spatial = p.D * p.H * p.W;
    if (gid >= spatial) return;

    uint w = gid % p.W;
    uint h = (gid / p.W) % p.H;
    uint d = gid / (p.H * p.W);
    int r = int(p.klen) / 2;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
    for (int k = 0; k < int(p.klen); k++) {
        int dd = int(d) + k - r;
        if (dd >= 0 && dd < int(p.D)) {
            uint src = ((uint(dd) * p.H + h) * p.W + w) * 3;
            float kv = kern[k];
            sum0 += in[src + 0] * kv;
            sum1 += in[src + 1] * kv;
            sum2 += in[src + 2] * kv;
        }
    }
    uint dst = gid * 3;
    out[dst + 0] = sum0;
    out[dst + 1] = sum1;
    out[dst + 2] = sum2;
}

/* ------------------------------------------------------------------ */
/* Axis 1: convolve along H (height) axis                              */
/* ------------------------------------------------------------------ */

kernel void blur_dhw3_axis1(
    const device float *in          [[buffer(0)]],
    device float       *out         [[buffer(1)]],
    constant float     *kern        [[buffer(2)]],
    constant BlurParams &p          [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint spatial = p.D * p.H * p.W;
    if (gid >= spatial) return;

    uint w = gid % p.W;
    uint h = (gid / p.W) % p.H;
    uint d = gid / (p.H * p.W);
    int r = int(p.klen) / 2;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
    for (int k = 0; k < int(p.klen); k++) {
        int hh = int(h) + k - r;
        if (hh >= 0 && hh < int(p.H)) {
            uint src = ((d * p.H + uint(hh)) * p.W + w) * 3;
            float kv = kern[k];
            sum0 += in[src + 0] * kv;
            sum1 += in[src + 1] * kv;
            sum2 += in[src + 2] * kv;
        }
    }
    uint dst = gid * 3;
    out[dst + 0] = sum0;
    out[dst + 1] = sum1;
    out[dst + 2] = sum2;
}

/* ------------------------------------------------------------------ */
/* Axis 2: convolve along W (width) axis                               */
/* ------------------------------------------------------------------ */

kernel void blur_dhw3_axis2(
    const device float *in          [[buffer(0)]],
    device float       *out         [[buffer(1)]],
    constant float     *kern        [[buffer(2)]],
    constant BlurParams &p          [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint spatial = p.D * p.H * p.W;
    if (gid >= spatial) return;

    uint w = gid % p.W;
    uint h = (gid / p.W) % p.H;
    uint d = gid / (p.H * p.W);
    int r = int(p.klen) / 2;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
    for (int k = 0; k < int(p.klen); k++) {
        int ww = int(w) + k - r;
        if (ww >= 0 && ww < int(p.W)) {
            uint src = ((d * p.H + h) * p.W + uint(ww)) * 3;
            float kv = kern[k];
            sum0 += in[src + 0] * kv;
            sum1 += in[src + 1] * kv;
            sum2 += in[src + 2] * kv;
        }
    }
    uint dst = gid * 3;
    out[dst + 0] = sum0;
    out[dst + 1] = sum1;
    out[dst + 2] = sum2;
}
