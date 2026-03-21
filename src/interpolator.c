/*
 * interpolator.c - CPU grid sampling and affine grid generation
 *
 * Matches PyTorch's F.grid_sample with mode='bilinear', padding_mode='zeros',
 * align_corners=True, and F.affine_grid with align_corners=True.
 */

#include "cfireants/interpolator.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Convert normalized coordinate to unnormalized.
 * align_corners=True: -1 -> 0, +1 -> size-1 */
static inline float unnormalize(float coord, int size) {
    return ((coord + 1.0f) * 0.5f) * (size - 1);
}

/* Clip to [0, size-1] */
static inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

int affine_grid_3d(const tensor_t *affine, const int out_shape[3],
                   tensor_t *output) {
    /* affine: [B, 3, 4], output: [B, D, H, W, 3] */
    int B = affine->shape[0];
    int D = out_shape[0], H = out_shape[1], W = out_shape[2];

    int shape[5] = {B, D, H, W, 3};
    if (tensor_alloc(output, 5, shape, DTYPE_FLOAT32, DEVICE_CPU) != 0)
        return -1;

    const float *aff = tensor_data_f32(affine);
    float *out = tensor_data_f32(output);

    for (int b = 0; b < B; b++) {
        const float *A = aff + b * 12; /* 3x4 row-major */
        for (int d = 0; d < D; d++) {
            /* Normalized coordinates: maps index to [-1, 1] */
            float z = (D > 1) ? (2.0f * d / (D - 1) - 1.0f) : 0.0f;
            for (int h = 0; h < H; h++) {
                float y = (H > 1) ? (2.0f * h / (H - 1) - 1.0f) : 0.0f;
                for (int w = 0; w < W; w++) {
                    float x = (W > 1) ? (2.0f * w / (W - 1) - 1.0f) : 0.0f;

                    /* Apply affine: out = A @ [x, y, z, 1]^T */
                    float ox = A[0]*x + A[1]*y + A[2]*z  + A[3];
                    float oy = A[4]*x + A[5]*y + A[6]*z  + A[7];
                    float oz = A[8]*x + A[9]*y + A[10]*z + A[11];

                    size_t idx = (((size_t)b*D + d)*H + h)*W + w;
                    out[idx * 3 + 0] = ox;
                    out[idx * 3 + 1] = oy;
                    out[idx * 3 + 2] = oz;
                }
            }
        }
    }
    return 0;
}

int cpu_grid_sample_3d_forward(const tensor_t *input, const tensor_t *grid,
                               tensor_t *output, int align_corners) {
    /*
     * input:  [B, C, iD, iH, iW]
     * grid:   [B, oD, oH, oW, 3]  (x, y, z coordinates in [-1,1])
     * output: [B, C, oD, oH, oW]
     *
     * PyTorch grid_sample convention:
     *   grid[..., 0] = x -> indexes dim W (last spatial dim)
     *   grid[..., 1] = y -> indexes dim H
     *   grid[..., 2] = z -> indexes dim D (first spatial dim)
     */
    int B  = input->shape[0], C  = input->shape[1];
    int iD = input->shape[2], iH = input->shape[3], iW = input->shape[4];
    int oD = grid->shape[1],  oH = grid->shape[2],  oW = grid->shape[3];

    int out_shape[5] = {B, C, oD, oH, oW};
    if (tensor_alloc(output, 5, out_shape, DTYPE_FLOAT32, DEVICE_CPU) != 0)
        return -1;

    const float *inp = tensor_data_f32(input);
    const float *grd = tensor_data_f32(grid);
    float *out = tensor_data_f32(output);

    size_t inp_sB = (size_t)C * iD * iH * iW;
    size_t inp_sC = (size_t)iD * iH * iW;
    size_t inp_sD = (size_t)iH * iW;
    size_t inp_sH = (size_t)iW;

    size_t out_sB = (size_t)C * oD * oH * oW;
    size_t out_sC = (size_t)oD * oH * oW;
    size_t out_sD = (size_t)oH * oW;
    size_t out_sH = (size_t)oW;

    size_t grd_sB = (size_t)oD * oH * oW * 3;
    size_t grd_sD = (size_t)oH * oW * 3;
    size_t grd_sH = (size_t)oW * 3;

    for (int b = 0; b < B; b++) {
        for (int od = 0; od < oD; od++) {
            for (int oh = 0; oh < oH; oh++) {
                for (int ow = 0; ow < oW; ow++) {
                    size_t gidx = b*grd_sB + od*grd_sD + oh*grd_sH + ow*3;
                    float gx = grd[gidx + 0]; /* x -> W */
                    float gy = grd[gidx + 1]; /* y -> H */
                    float gz = grd[gidx + 2]; /* z -> D */

                    /* Unnormalize to input coordinates */
                    float ix = unnormalize(gx, iW);
                    float iy = unnormalize(gy, iH);
                    float iz = unnormalize(gz, iD);

                    /* Floor indices */
                    int ix0 = (int)floorf(ix);
                    int iy0 = (int)floorf(iy);
                    int iz0 = (int)floorf(iz);
                    int ix1 = ix0 + 1;
                    int iy1 = iy0 + 1;
                    int iz1 = iz0 + 1;

                    /* Fractional parts */
                    float fx = ix - ix0;
                    float fy = iy - iy0;
                    float fz = iz - iz0;

                    /* Trilinear weights */
                    float w000 = (1-fx)*(1-fy)*(1-fz);
                    float w001 = fx    *(1-fy)*(1-fz);
                    float w010 = (1-fx)*fy    *(1-fz);
                    float w011 = fx    *fy    *(1-fz);
                    float w100 = (1-fx)*(1-fy)*fz;
                    float w101 = fx    *(1-fy)*fz;
                    float w110 = (1-fx)*fy    *fz;
                    float w111 = fx    *fy    *fz;

                    /* Boundary check flags (zeros padding) */
                    int v000 = (iz0>=0 && iz0<iD && iy0>=0 && iy0<iH && ix0>=0 && ix0<iW);
                    int v001 = (iz0>=0 && iz0<iD && iy0>=0 && iy0<iH && ix1>=0 && ix1<iW);
                    int v010 = (iz0>=0 && iz0<iD && iy1>=0 && iy1<iH && ix0>=0 && ix0<iW);
                    int v011 = (iz0>=0 && iz0<iD && iy1>=0 && iy1<iH && ix1>=0 && ix1<iW);
                    int v100 = (iz1>=0 && iz1<iD && iy0>=0 && iy0<iH && ix0>=0 && ix0<iW);
                    int v101 = (iz1>=0 && iz1<iD && iy0>=0 && iy0<iH && ix1>=0 && ix1<iW);
                    int v110 = (iz1>=0 && iz1<iD && iy1>=0 && iy1<iH && ix0>=0 && ix0<iW);
                    int v111 = (iz1>=0 && iz1<iD && iy1>=0 && iy1<iH && ix1>=0 && ix1<iW);

                    for (int c = 0; c < C; c++) {
                        const float *inp_c = inp + b*inp_sB + c*inp_sC;
                        float val = 0.0f;

                        if (v000) val += w000 * inp_c[iz0*inp_sD + iy0*inp_sH + ix0];
                        if (v001) val += w001 * inp_c[iz0*inp_sD + iy0*inp_sH + ix1];
                        if (v010) val += w010 * inp_c[iz0*inp_sD + iy1*inp_sH + ix0];
                        if (v011) val += w011 * inp_c[iz0*inp_sD + iy1*inp_sH + ix1];
                        if (v100) val += w100 * inp_c[iz1*inp_sD + iy0*inp_sH + ix0];
                        if (v101) val += w101 * inp_c[iz1*inp_sD + iy0*inp_sH + ix1];
                        if (v110) val += w110 * inp_c[iz1*inp_sD + iy1*inp_sH + ix0];
                        if (v111) val += w111 * inp_c[iz1*inp_sD + iy1*inp_sH + ix1];

                        out[b*out_sB + c*out_sC + od*out_sD + oh*out_sH + ow] = val;
                    }
                }
            }
        }
    }
    return 0;
}

int cpu_grid_sample_3d_backward(const tensor_t *grad_output,
                                const tensor_t *input,
                                const tensor_t *grid,
                                tensor_t *grad_grid,
                                int align_corners) {
    /*
     * Compute dL/d(grid) given dL/d(output).
     * For bilinear interpolation, the gradient w.r.t. each grid coordinate
     * is the sum over channels of (dL/dout * d(interp)/d(coord)).
     */
    int B  = input->shape[0], C  = input->shape[1];
    int iD = input->shape[2], iH = input->shape[3], iW = input->shape[4];
    int oD = grid->shape[1],  oH = grid->shape[2],  oW = grid->shape[3];

    int gg_shape[5] = {B, oD, oH, oW, 3};
    if (tensor_alloc(grad_grid, 5, gg_shape, DTYPE_FLOAT32, DEVICE_CPU) != 0)
        return -1;

    const float *inp = tensor_data_f32(input);
    const float *grd = tensor_data_f32(grid);
    const float *go  = tensor_data_f32(grad_output);
    float *gg = tensor_data_f32(grad_grid);

    size_t inp_sC = (size_t)iD * iH * iW;
    size_t inp_sD = (size_t)iH * iW;
    size_t inp_sH = (size_t)iW;

    size_t out_sC = (size_t)oD * oH * oW;
    size_t out_sD = (size_t)oH * oW;
    size_t out_sH = (size_t)oW;

    /* Unnormalization factors: d(unnorm)/d(norm) */
    float mult_x = (iW - 1) * 0.5f;
    float mult_y = (iH - 1) * 0.5f;
    float mult_z = (iD - 1) * 0.5f;

    for (int b = 0; b < B; b++) {
        for (int od = 0; od < oD; od++) {
            for (int oh = 0; oh < oH; oh++) {
                for (int ow = 0; ow < oW; ow++) {
                    size_t gidx = ((((size_t)b*oD + od)*oH + oh)*oW + ow) * 3;
                    float gx = grd[gidx + 0];
                    float gy = grd[gidx + 1];
                    float gz = grd[gidx + 2];

                    float ix = unnormalize(gx, iW);
                    float iy = unnormalize(gy, iH);
                    float iz = unnormalize(gz, iD);

                    int ix0 = (int)floorf(ix);
                    int iy0 = (int)floorf(iy);
                    int iz0 = (int)floorf(iz);
                    int ix1 = ix0 + 1;
                    int iy1 = iy0 + 1;
                    int iz1 = iz0 + 1;

                    float fx = ix - ix0;
                    float fy = iy - iy0;
                    float fz = iz - iz0;

                    /* Accumulate gradient over channels */
                    float dgx = 0.0f, dgy = 0.0f, dgz = 0.0f;

                    for (int c = 0; c < C; c++) {
                        size_t out_idx = ((size_t)b*C + c)*out_sC + od*out_sD + oh*out_sH + ow;
                        float go_val = go[out_idx];
                        const float *inp_c = inp + ((size_t)b*C + c)*inp_sC;

                        /* Helper to get input value with bounds check */
                        #define INP(d, h, w) \
                            ((d)>=0 && (d)<iD && (h)>=0 && (h)<iH && (w)>=0 && (w)<iW \
                             ? inp_c[(d)*inp_sD + (h)*inp_sH + (w)] : 0.0f)

                        /* d(interp)/d(ix) partial derivatives */
                        float dval_dfx =
                            -(1-fy)*(1-fz)*INP(iz0,iy0,ix0) + (1-fy)*(1-fz)*INP(iz0,iy0,ix1)
                            -fy    *(1-fz)*INP(iz0,iy1,ix0) + fy    *(1-fz)*INP(iz0,iy1,ix1)
                            -(1-fy)*fz    *INP(iz1,iy0,ix0) + (1-fy)*fz    *INP(iz1,iy0,ix1)
                            -fy    *fz    *INP(iz1,iy1,ix0) + fy    *fz    *INP(iz1,iy1,ix1);

                        float dval_dfy =
                            -(1-fx)*(1-fz)*INP(iz0,iy0,ix0) - fx*(1-fz)*INP(iz0,iy0,ix1)
                            +(1-fx)*(1-fz)*INP(iz0,iy1,ix0) + fx*(1-fz)*INP(iz0,iy1,ix1)
                            -(1-fx)*fz    *INP(iz1,iy0,ix0) - fx*fz    *INP(iz1,iy0,ix1)
                            +(1-fx)*fz    *INP(iz1,iy1,ix0) + fx*fz    *INP(iz1,iy1,ix1);

                        float dval_dfz =
                            -(1-fx)*(1-fy)*INP(iz0,iy0,ix0) - fx*(1-fy)*INP(iz0,iy0,ix1)
                            -(1-fx)*fy    *INP(iz0,iy1,ix0) - fx*fy    *INP(iz0,iy1,ix1)
                            +(1-fx)*(1-fy)*INP(iz1,iy0,ix0) + fx*(1-fy)*INP(iz1,iy0,ix1)
                            +(1-fx)*fy    *INP(iz1,iy1,ix0) + fx*fy    *INP(iz1,iy1,ix1);

                        #undef INP

                        dgx += go_val * dval_dfx;
                        dgy += go_val * dval_dfy;
                        dgz += go_val * dval_dfz;
                    }

                    /* Chain rule: d(unnorm)/d(norm) */
                    gg[gidx + 0] = dgx * mult_x;
                    gg[gidx + 1] = dgy * mult_y;
                    gg[gidx + 2] = dgz * mult_z;
                }
            }
        }
    }
    return 0;
}
