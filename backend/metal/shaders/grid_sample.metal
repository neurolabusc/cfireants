/*
 * grid_sample.metal - Metal compute shaders for grid sampling, affine grid,
 *                     trilinear resize, and separable convolution.
 *
 * Faithful translation of backend/cuda/grid_sample.cu.
 * Float32-only, 3D-only, align_corners=true, zeros padding.
 */

#include <metal_stdlib>
using namespace metal;

/* ------------------------------------------------------------------ */
/* Parameter structs                                                    */
/* ------------------------------------------------------------------ */

struct GridSampleParams {
    uint B, C, iD, iH, iW, oD, oH, oW;
};

struct AffineGridParams {
    uint B, D, H, W;
};

struct ResizeParams {
    uint B, C, iD, iH, iW, oD, oH, oW;
    uint align_corners;
    uint _pad;
};

struct Conv1dParams {
    uint D, H, W;
    uint klen;
    uint axis;
    uint _pad0, _pad1, _pad2;
};

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

/* Unnormalize: [-1,1] -> [0, size-1] (align_corners=true) */
static inline float unnorm(float x, int size) {
    return (x + 1.0f) * 0.5f * float(size - 1);
}

/* ------------------------------------------------------------------ */
/* Forward kernel: trilinear interpolation                              */
/* One thread per output voxel in B*oD*oH*oW                           */
/* Input:  [B, C, iD, iH, iW] contiguous float                         */
/* Grid:   [B, oD, oH, oW, 3] interleaved (x=W, y=H, z=D)            */
/* Output: [B, C, oD, oH, oW] contiguous float                         */
/* ------------------------------------------------------------------ */

kernel void grid_sample_3d_fwd(
    device const float *input   [[buffer(0)]],
    device const float *grid    [[buffer(1)]],
    device       float *output  [[buffer(2)]],
    constant GridSampleParams &p [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    long total = long(p.B) * p.oD * p.oH * p.oW;
    if (long(tid) >= total) return;

    int idx = int(tid);
    int w = idx % int(p.oW);
    int tmp = idx / int(p.oW);
    int h = tmp % int(p.oH);
    tmp /= int(p.oH);
    int d = tmp % int(p.oD);
    int b = tmp / int(p.oD);

    /* Read grid coordinates */
    long gidx = long(idx) * 3;
    float gx = grid[gidx + 0]; /* x -> W */
    float gy = grid[gidx + 1]; /* y -> H */
    float gz = grid[gidx + 2]; /* z -> D */

    float ix = unnorm(gx, int(p.iW));
    float iy = unnorm(gy, int(p.iH));
    float iz = unnorm(gz, int(p.iD));

    int ix0 = int(floor(ix));
    int iy0 = int(floor(iy));
    int iz0 = int(floor(iz));
    int ix1 = ix0 + 1, iy1 = iy0 + 1, iz1 = iz0 + 1;

    float fx = ix - float(ix0), fy = iy - float(iy0), fz = iz - float(iz0);

    /* Trilinear weights */
    float w000 = (1 - fx) * (1 - fy) * (1 - fz);
    float w001 = fx * (1 - fy) * (1 - fz);
    float w010 = (1 - fx) * fy * (1 - fz);
    float w011 = fx * fy * (1 - fz);
    float w100 = (1 - fx) * (1 - fy) * fz;
    float w101 = fx * (1 - fy) * fz;
    float w110 = (1 - fx) * fy * fz;
    float w111 = fx * fy * fz;

    int iD = int(p.iD), iH = int(p.iH), iW = int(p.iW);
    int C = int(p.C);

    long out_base = ((long(b) * C) * long(p.oD) + d) * long(p.oH) * p.oW + long(h) * p.oW + w;
    long out_stride = long(p.oD) * p.oH * p.oW;

    for (int c = 0; c < C; c++) {
        long inp_base = (long(b) * C + c) * long(iD) * iH * iW;

        /* Zeros padding: out-of-bounds samples return 0 */
        float v000 = (iz0 >= 0 && iz0 < iD && iy0 >= 0 && iy0 < iH && ix0 >= 0 && ix0 < iW)
                     ? input[inp_base + long(iz0) * iH * iW + iy0 * iW + ix0] : 0.0f;
        float v001 = (iz0 >= 0 && iz0 < iD && iy0 >= 0 && iy0 < iH && ix1 >= 0 && ix1 < iW)
                     ? input[inp_base + long(iz0) * iH * iW + iy0 * iW + ix1] : 0.0f;
        float v010 = (iz0 >= 0 && iz0 < iD && iy1 >= 0 && iy1 < iH && ix0 >= 0 && ix0 < iW)
                     ? input[inp_base + long(iz0) * iH * iW + iy1 * iW + ix0] : 0.0f;
        float v011 = (iz0 >= 0 && iz0 < iD && iy1 >= 0 && iy1 < iH && ix1 >= 0 && ix1 < iW)
                     ? input[inp_base + long(iz0) * iH * iW + iy1 * iW + ix1] : 0.0f;
        float v100 = (iz1 >= 0 && iz1 < iD && iy0 >= 0 && iy0 < iH && ix0 >= 0 && ix0 < iW)
                     ? input[inp_base + long(iz1) * iH * iW + iy0 * iW + ix0] : 0.0f;
        float v101 = (iz1 >= 0 && iz1 < iD && iy0 >= 0 && iy0 < iH && ix1 >= 0 && ix1 < iW)
                     ? input[inp_base + long(iz1) * iH * iW + iy0 * iW + ix1] : 0.0f;
        float v110 = (iz1 >= 0 && iz1 < iD && iy1 >= 0 && iy1 < iH && ix0 >= 0 && ix0 < iW)
                     ? input[inp_base + long(iz1) * iH * iW + iy1 * iW + ix0] : 0.0f;
        float v111 = (iz1 >= 0 && iz1 < iD && iy1 >= 0 && iy1 < iH && ix1 >= 0 && ix1 < iW)
                     ? input[inp_base + long(iz1) * iH * iW + iy1 * iW + ix1] : 0.0f;

        float val = w000 * v000 + w001 * v001 + w010 * v010 + w011 * v011
                  + w100 * v100 + w101 * v101 + w110 * v110 + w111 * v111;
        output[out_base + long(c) * out_stride] = val;
    }
}

/* ------------------------------------------------------------------ */
/* Backward kernel: gradient w.r.t. grid only                          */
/* Same thread mapping as forward (one per output voxel B*oD*oH*oW)    */
/* ------------------------------------------------------------------ */

kernel void grid_sample_3d_bwd(
    device const float *grad_output [[buffer(0)]],
    device const float *input       [[buffer(1)]],
    device const float *grid        [[buffer(2)]],
    device       float *grad_grid   [[buffer(3)]],
    constant GridSampleParams &p    [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    long total = long(p.B) * p.oD * p.oH * p.oW;
    if (long(tid) >= total) return;

    int idx = int(tid);
    int w = idx % int(p.oW);
    int tmp = idx / int(p.oW);
    int h = tmp % int(p.oH);
    tmp /= int(p.oH);
    int d = tmp % int(p.oD);
    int b = tmp / int(p.oD);

    long gidx = long(idx) * 3;
    float gx = grid[gidx + 0];
    float gy = grid[gidx + 1];
    float gz = grid[gidx + 2];

    int iD = int(p.iD), iH = int(p.iH), iW = int(p.iW);
    int C = int(p.C);

    float ix = unnorm(gx, iW);
    float iy = unnorm(gy, iH);
    float iz = unnorm(gz, iD);

    int ix0 = int(floor(ix));
    int iy0 = int(floor(iy));
    int iz0 = int(floor(iz));
    int ix1 = ix0 + 1, iy1 = iy0 + 1, iz1 = iz0 + 1;

    float fx = ix - float(ix0), fy = iy - float(iy0), fz = iz - float(iz0);

    float dgx = 0.0f, dgy = 0.0f, dgz = 0.0f;

    for (int c = 0; c < C; c++) {
        long go_idx = (long(b) * C + c) * long(p.oD) * p.oH * p.oW + long(d) * p.oH * p.oW + long(h) * p.oW + w;
        float go = grad_output[go_idx];

        long inp_base = (long(b) * C + c) * long(iD) * iH * iW;

        /* Fetch input values with zeros padding */
        float i000 = (iz0 >= 0 && iz0 < iD && iy0 >= 0 && iy0 < iH && ix0 >= 0 && ix0 < iW)
                     ? input[inp_base + long(iz0) * iH * iW + iy0 * iW + ix0] : 0.0f;
        float i001 = (iz0 >= 0 && iz0 < iD && iy0 >= 0 && iy0 < iH && ix1 >= 0 && ix1 < iW)
                     ? input[inp_base + long(iz0) * iH * iW + iy0 * iW + ix1] : 0.0f;
        float i010 = (iz0 >= 0 && iz0 < iD && iy1 >= 0 && iy1 < iH && ix0 >= 0 && ix0 < iW)
                     ? input[inp_base + long(iz0) * iH * iW + iy1 * iW + ix0] : 0.0f;
        float i011 = (iz0 >= 0 && iz0 < iD && iy1 >= 0 && iy1 < iH && ix1 >= 0 && ix1 < iW)
                     ? input[inp_base + long(iz0) * iH * iW + iy1 * iW + ix1] : 0.0f;
        float i100 = (iz1 >= 0 && iz1 < iD && iy0 >= 0 && iy0 < iH && ix0 >= 0 && ix0 < iW)
                     ? input[inp_base + long(iz1) * iH * iW + iy0 * iW + ix0] : 0.0f;
        float i101 = (iz1 >= 0 && iz1 < iD && iy0 >= 0 && iy0 < iH && ix1 >= 0 && ix1 < iW)
                     ? input[inp_base + long(iz1) * iH * iW + iy0 * iW + ix1] : 0.0f;
        float i110 = (iz1 >= 0 && iz1 < iD && iy1 >= 0 && iy1 < iH && ix0 >= 0 && ix0 < iW)
                     ? input[inp_base + long(iz1) * iH * iW + iy1 * iW + ix0] : 0.0f;
        float i111 = (iz1 >= 0 && iz1 < iD && iy1 >= 0 && iy1 < iH && ix1 >= 0 && ix1 < iW)
                     ? input[inp_base + long(iz1) * iH * iW + iy1 * iW + ix1] : 0.0f;

        /* d(val)/d(fx) */
        float dval_dfx = -(1 - fy) * (1 - fz) * i000 + (1 - fy) * (1 - fz) * i001
                        - fy * (1 - fz) * i010 + fy * (1 - fz) * i011
                        - (1 - fy) * fz * i100 + (1 - fy) * fz * i101
                        - fy * fz * i110 + fy * fz * i111;

        /* d(val)/d(fy) */
        float dval_dfy = -(1 - fx) * (1 - fz) * i000 - fx * (1 - fz) * i001
                        + (1 - fx) * (1 - fz) * i010 + fx * (1 - fz) * i011
                        - (1 - fx) * fz * i100 - fx * fz * i101
                        + (1 - fx) * fz * i110 + fx * fz * i111;

        /* d(val)/d(fz) */
        float dval_dfz = -(1 - fx) * (1 - fy) * i000 - fx * (1 - fy) * i001
                        - (1 - fx) * fy * i010 - fx * fy * i011
                        + (1 - fx) * (1 - fy) * i100 + fx * (1 - fy) * i101
                        + (1 - fx) * fy * i110 + fx * fy * i111;

        dgx += go * dval_dfx;
        dgy += go * dval_dfy;
        dgz += go * dval_dfz;
    }

    /* Chain rule: d(unnorm)/d(norm) = 0.5 * (size - 1) */
    float mx = float(iW - 1) * 0.5f;
    float my = float(iH - 1) * 0.5f;
    float mz = float(iD - 1) * 0.5f;

    grad_grid[gidx + 0] = dgx * mx;
    grad_grid[gidx + 1] = dgy * my;
    grad_grid[gidx + 2] = dgz * mz;
}

/* ------------------------------------------------------------------ */
/* Affine grid generation                                              */
/* One thread per voxel in B*D*H*W                                     */
/* Normalized coords: nz = 2*d/(D-1)-1, ny = 2*h/(H-1)-1, etc.       */
/* Output: grid[idx*3+{0,1,2}] = affine transform of (nx, ny, nz)     */
/* ------------------------------------------------------------------ */

kernel void affine_grid_3d(
    device const float *affine  [[buffer(0)]],
    device       float *grid    [[buffer(1)]],
    constant AffineGridParams &p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    long total = long(p.B) * p.D * p.H * p.W;
    if (long(tid) >= total) return;

    int idx = int(tid);
    int w = idx % int(p.W);
    int tmp = idx / int(p.W);
    int h = tmp % int(p.H);
    tmp /= int(p.H);
    int d = tmp % int(p.D);
    int b = tmp / int(p.D);

    float nz = (p.D > 1) ? (2.0f * d / float(p.D - 1) - 1.0f) : 0.0f;
    float ny = (p.H > 1) ? (2.0f * h / float(p.H - 1) - 1.0f) : 0.0f;
    float nx = (p.W > 1) ? (2.0f * w / float(p.W - 1) - 1.0f) : 0.0f;

    device const float *A = affine + b * 12;
    float ox = A[0] * nx + A[1] * ny + A[2] * nz + A[3];
    float oy = A[4] * nx + A[5] * ny + A[6] * nz + A[7];
    float oz = A[8] * nx + A[9] * ny + A[10] * nz + A[11];

    long gidx = long(idx) * 3;
    grid[gidx + 0] = ox;
    grid[gidx + 1] = oy;
    grid[gidx + 2] = oz;
}

/* ------------------------------------------------------------------ */
/* Trilinear resize                                                    */
/* One thread per output element in B*C*oD*oH*oW                       */
/* Supports align_corners=true and =false, clamped boundary handling   */
/* ------------------------------------------------------------------ */

kernel void trilinear_resize(
    device const float *input   [[buffer(0)]],
    device       float *output  [[buffer(1)]],
    constant ResizeParams &p    [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    long total = long(p.B) * p.C * p.oD * p.oH * p.oW;
    if (long(tid) >= total) return;

    int idx = int(tid);
    int ow = idx % int(p.oW);
    int tmp = idx / int(p.oW);
    int oh = tmp % int(p.oH);
    tmp /= int(p.oH);
    int od = tmp % int(p.oD);
    tmp /= int(p.oD);
    int c = tmp % int(p.C);
    int b = tmp / int(p.C);

    int iD = int(p.iD), iH = int(p.iH), iW = int(p.iW);
    int oD = int(p.oD), oH = int(p.oH), oW = int(p.oW);
    bool ac = (p.align_corners != 0);

    float sd, sh, sw;
    if (ac && oD > 1) sd = float(od) * float(iD - 1) / float(oD - 1);
    else              sd = (float(od) + 0.5f) * float(iD) / float(oD) - 0.5f;
    if (ac && oH > 1) sh = float(oh) * float(iH - 1) / float(oH - 1);
    else              sh = (float(oh) + 0.5f) * float(iH) / float(oH) - 0.5f;
    if (ac && oW > 1) sw = float(ow) * float(iW - 1) / float(oW - 1);
    else              sw = (float(ow) + 0.5f) * float(iW) / float(oW) - 0.5f;

    int d0 = int(floor(sd)), h0 = int(floor(sh)), w0 = int(floor(sw));
    int d1 = d0 + 1, h1 = h0 + 1, w1 = w0 + 1;
    float fd = sd - float(d0), fh = sh - float(h0), fw = sw - float(w0);

    /* Clamped boundary handling */
    if (d0 < 0) { d0 = 0; fd = 0.0f; } if (d1 >= iD) d1 = iD - 1;
    if (h0 < 0) { h0 = 0; fh = 0.0f; } if (h1 >= iH) h1 = iH - 1;
    if (w0 < 0) { w0 = 0; fw = 0.0f; } if (w1 >= iW) w1 = iW - 1;

    long src_base = (long(b) * p.C + c) * long(iD) * iH * iW;

    float v000 = input[src_base + long(d0) * iH * iW + h0 * iW + w0];
    float v001 = input[src_base + long(d0) * iH * iW + h0 * iW + w1];
    float v010 = input[src_base + long(d0) * iH * iW + h1 * iW + w0];
    float v011 = input[src_base + long(d0) * iH * iW + h1 * iW + w1];
    float v100 = input[src_base + long(d1) * iH * iW + h0 * iW + w0];
    float v101 = input[src_base + long(d1) * iH * iW + h0 * iW + w1];
    float v110 = input[src_base + long(d1) * iH * iW + h1 * iW + w0];
    float v111 = input[src_base + long(d1) * iH * iW + h1 * iW + w1];

    float val = (1 - fd) * (1 - fh) * (1 - fw) * v000 + (1 - fd) * (1 - fh) * fw * v001
              + (1 - fd) * fh * (1 - fw) * v010 + (1 - fd) * fh * fw * v011
              + fd * (1 - fh) * (1 - fw) * v100 + fd * (1 - fh) * fw * v101
              + fd * fh * (1 - fw) * v110 + fd * fh * fw * v111;

    output[idx] = val;
}

/* ------------------------------------------------------------------ */
/* Separable 1D convolution along one axis (zero-padded)               */
/* One thread per voxel in D*H*W                                       */
/* axis: 0=D, 1=H, 2=W                                                */
/* ------------------------------------------------------------------ */

kernel void conv1d_axis(
    device const float *input   [[buffer(0)]],
    device       float *output  [[buffer(1)]],
    device const float *kern    [[buffer(2)]],
    constant Conv1dParams &p    [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    long total = long(p.D) * p.H * p.W;
    if (long(tid) >= total) return;

    int idx = int(tid);
    int w = idx % int(p.W);
    int tmp = idx / int(p.W);
    int h = tmp % int(p.H);
    int d = tmp / int(p.H);

    int D = int(p.D), H = int(p.H), W = int(p.W);
    int klen = int(p.klen);
    int r = klen / 2;

    float sum = 0.0f;

    if (p.axis == 2) {
        /* Convolve along W */
        for (int k = 0; k < klen; k++) {
            int ww = w + k - r;
            if (ww >= 0 && ww < W)
                sum += input[long(d) * H * W + h * W + ww] * kern[k];
        }
    } else if (p.axis == 1) {
        /* Convolve along H */
        for (int k = 0; k < klen; k++) {
            int hh = h + k - r;
            if (hh >= 0 && hh < H)
                sum += input[long(d) * H * W + hh * W + w] * kern[k];
        }
    } else {
        /* Convolve along D (axis == 0) */
        for (int k = 0; k < klen; k++) {
            int dd = d + k - r;
            if (dd >= 0 && dd < D)
                sum += input[long(dd) * H * W + h * W + w] * kern[k];
        }
    }

    output[idx] = sum;
}

/* ------------------------------------------------------------------ */
/* Box filter along one axis                                           */
/* Same as conv1d but with uniform weights (scale parameter)           */
/* The kernel buffer contains scale replicated klen times              */
/* ------------------------------------------------------------------ */

kernel void box_filter_axis(
    device const float *input   [[buffer(0)]],
    device       float *output  [[buffer(1)]],
    constant Conv1dParams &p    [[buffer(2)]],
    constant float &scale       [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    long total = long(p.D) * p.H * p.W;
    if (long(tid) >= total) return;

    int idx = int(tid);
    int w = idx % int(p.W);
    int tmp = idx / int(p.W);
    int h = tmp % int(p.H);
    int d = tmp / int(p.H);

    int D = int(p.D), H = int(p.H), W = int(p.W);
    int klen = int(p.klen);
    int r = klen / 2;

    float sum = 0.0f;

    if (p.axis == 2) {
        /* Convolve along W */
        for (int k = 0; k < klen; k++) {
            int ww = w + k - r;
            if (ww >= 0 && ww < W)
                sum += input[long(d) * H * W + h * W + ww];
        }
    } else if (p.axis == 1) {
        /* Convolve along H */
        for (int k = 0; k < klen; k++) {
            int hh = h + k - r;
            if (hh >= 0 && hh < H)
                sum += input[long(d) * H * W + hh * W + w];
        }
    } else {
        /* Convolve along D (axis == 0) */
        for (int k = 0; k < klen; k++) {
            int dd = d + k - r;
            if (dd >= 0 && dd < D)
                sum += input[long(dd) * H * W + h * W + w];
        }
    }

    output[idx] = sum * scale;
}
