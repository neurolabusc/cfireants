/*
 * interpolator.c - CPU grid sampling and affine grid generation
 *
 * Matches PyTorch's F.grid_sample with mode='bilinear', padding_mode='zeros',
 * align_corners=True, and F.affine_grid with align_corners=True.
 */

#include "cfireants/interpolator.h"
#include "cfireants/threading.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Convert normalized coordinate to unnormalized.
 * align_corners=True: -1 -> 0, +1 -> size-1 */
static inline float unnormalize(float coord, int size) {
    return ((coord + 1.0f) * 0.5f) * (size - 1);
}

typedef struct {
    const float *inp, *grd;
    float *out;
    int C, iD, iH, iW, oD, oH, oW;
    size_t inp_sB, inp_sC, inp_sD, inp_sH;
    size_t out_sB, out_sC, out_sD, out_sH;
    size_t grd_sB, grd_sD, grd_sH;
} grid_forward_context_t;

static void grid_forward_range(size_t begin, size_t end, void *opaque) {
    grid_forward_context_t *ctx = (grid_forward_context_t *)opaque;
    for (size_t slice = begin; slice < end; slice++) {
        int b = (int)(slice / (size_t)ctx->oD);
        int od = (int)(slice % (size_t)ctx->oD);
        for (int oh = 0; oh < ctx->oH; oh++) {
            for (int ow = 0; ow < ctx->oW; ow++) {
                size_t gidx = b*ctx->grd_sB + od*ctx->grd_sD + oh*ctx->grd_sH + ow*3;
                float ix = unnormalize(ctx->grd[gidx], ctx->iW);
                float iy = unnormalize(ctx->grd[gidx + 1], ctx->iH);
                float iz = unnormalize(ctx->grd[gidx + 2], ctx->iD);
                int ix0 = (int)floorf(ix), iy0 = (int)floorf(iy), iz0 = (int)floorf(iz);
                int ix1 = ix0 + 1, iy1 = iy0 + 1, iz1 = iz0 + 1;
                float fx = ix - ix0, fy = iy - iy0, fz = iz - iz0;
                float w000 = (1-fx)*(1-fy)*(1-fz), w001 = fx*(1-fy)*(1-fz);
                float w010 = (1-fx)*fy*(1-fz), w011 = fx*fy*(1-fz);
                float w100 = (1-fx)*(1-fy)*fz, w101 = fx*(1-fy)*fz;
                float w110 = (1-fx)*fy*fz, w111 = fx*fy*fz;
                int v000 = (iz0>=0 && iz0<ctx->iD && iy0>=0 && iy0<ctx->iH && ix0>=0 && ix0<ctx->iW);
                int v001 = (iz0>=0 && iz0<ctx->iD && iy0>=0 && iy0<ctx->iH && ix1>=0 && ix1<ctx->iW);
                int v010 = (iz0>=0 && iz0<ctx->iD && iy1>=0 && iy1<ctx->iH && ix0>=0 && ix0<ctx->iW);
                int v011 = (iz0>=0 && iz0<ctx->iD && iy1>=0 && iy1<ctx->iH && ix1>=0 && ix1<ctx->iW);
                int v100 = (iz1>=0 && iz1<ctx->iD && iy0>=0 && iy0<ctx->iH && ix0>=0 && ix0<ctx->iW);
                int v101 = (iz1>=0 && iz1<ctx->iD && iy0>=0 && iy0<ctx->iH && ix1>=0 && ix1<ctx->iW);
                int v110 = (iz1>=0 && iz1<ctx->iD && iy1>=0 && iy1<ctx->iH && ix0>=0 && ix0<ctx->iW);
                int v111 = (iz1>=0 && iz1<ctx->iD && iy1>=0 && iy1<ctx->iH && ix1>=0 && ix1<ctx->iW);
                for (int c = 0; c < ctx->C; c++) {
                    const float *p = ctx->inp + b*ctx->inp_sB + c*ctx->inp_sC;
                    float val = 0.0f;
                    if (v000) val += w000 * p[iz0*ctx->inp_sD + iy0*ctx->inp_sH + ix0];
                    if (v001) val += w001 * p[iz0*ctx->inp_sD + iy0*ctx->inp_sH + ix1];
                    if (v010) val += w010 * p[iz0*ctx->inp_sD + iy1*ctx->inp_sH + ix0];
                    if (v011) val += w011 * p[iz0*ctx->inp_sD + iy1*ctx->inp_sH + ix1];
                    if (v100) val += w100 * p[iz1*ctx->inp_sD + iy0*ctx->inp_sH + ix0];
                    if (v101) val += w101 * p[iz1*ctx->inp_sD + iy0*ctx->inp_sH + ix1];
                    if (v110) val += w110 * p[iz1*ctx->inp_sD + iy1*ctx->inp_sH + ix0];
                    if (v111) val += w111 * p[iz1*ctx->inp_sD + iy1*ctx->inp_sH + ix1];
                    ctx->out[b*ctx->out_sB + c*ctx->out_sC + od*ctx->out_sD + oh*ctx->out_sH + ow] = val;
                }
            }
        }
    }
}

typedef struct {
    const float *inp, *grd, *go;
    float *gg;
    int C, iD, iH, iW, oD, oH, oW;
    size_t inp_sC, inp_sD, inp_sH;
    size_t out_sC, out_sD, out_sH;
    float mult_x, mult_y, mult_z;
} grid_backward_context_t;

static void grid_backward_range(size_t begin, size_t end, void *opaque) {
    grid_backward_context_t *ctx = (grid_backward_context_t *)opaque;
    int C = ctx->C, iD = ctx->iD, iH = ctx->iH, iW = ctx->iW;
    for (size_t slice = begin; slice < end; slice++) {
        int b = (int)(slice / (size_t)ctx->oD);
        int od = (int)(slice % (size_t)ctx->oD);
        for (int oh = 0; oh < ctx->oH; oh++) {
            for (int ow = 0; ow < ctx->oW; ow++) {
                size_t gidx = ((((size_t)b*ctx->oD + od)*ctx->oH + oh)*ctx->oW + ow) * 3;
                float ix = unnormalize(ctx->grd[gidx], iW);
                float iy = unnormalize(ctx->grd[gidx + 1], iH);
                float iz = unnormalize(ctx->grd[gidx + 2], iD);
                int ix0 = (int)floorf(ix), iy0 = (int)floorf(iy), iz0 = (int)floorf(iz);
                int ix1 = ix0 + 1, iy1 = iy0 + 1, iz1 = iz0 + 1;
                float fx = ix - ix0, fy = iy - iy0, fz = iz - iz0;
                float dgx = 0.0f, dgy = 0.0f, dgz = 0.0f;

                for (int c = 0; c < C; c++) {
                    size_t out_idx = ((size_t)b*C + c)*ctx->out_sC
                                   + od*ctx->out_sD + oh*ctx->out_sH + ow;
                    float go_val = ctx->go[out_idx];
                    const float *p = ctx->inp + ((size_t)b*C + c)*ctx->inp_sC;
                    #define INP(d, h, w) \
                        ((d)>=0 && (d)<iD && (h)>=0 && (h)<iH && (w)>=0 && (w)<iW \
                         ? p[(d)*ctx->inp_sD + (h)*ctx->inp_sH + (w)] : 0.0f)
                    float dval_dfx =
                        -(1-fy)*(1-fz)*INP(iz0,iy0,ix0) + (1-fy)*(1-fz)*INP(iz0,iy0,ix1)
                        -fy*(1-fz)*INP(iz0,iy1,ix0) + fy*(1-fz)*INP(iz0,iy1,ix1)
                        -(1-fy)*fz*INP(iz1,iy0,ix0) + (1-fy)*fz*INP(iz1,iy0,ix1)
                        -fy*fz*INP(iz1,iy1,ix0) + fy*fz*INP(iz1,iy1,ix1);
                    float dval_dfy =
                        -(1-fx)*(1-fz)*INP(iz0,iy0,ix0) - fx*(1-fz)*INP(iz0,iy0,ix1)
                        +(1-fx)*(1-fz)*INP(iz0,iy1,ix0) + fx*(1-fz)*INP(iz0,iy1,ix1)
                        -(1-fx)*fz*INP(iz1,iy0,ix0) - fx*fz*INP(iz1,iy0,ix1)
                        +(1-fx)*fz*INP(iz1,iy1,ix0) + fx*fz*INP(iz1,iy1,ix1);
                    float dval_dfz =
                        -(1-fx)*(1-fy)*INP(iz0,iy0,ix0) - fx*(1-fy)*INP(iz0,iy0,ix1)
                        -(1-fx)*fy*INP(iz0,iy1,ix0) - fx*fy*INP(iz0,iy1,ix1)
                        +(1-fx)*(1-fy)*INP(iz1,iy0,ix0) + fx*(1-fy)*INP(iz1,iy0,ix1)
                        +(1-fx)*fy*INP(iz1,iy1,ix0) + fx*fy*INP(iz1,iy1,ix1);
                    #undef INP
                    dgx += go_val * dval_dfx;
                    dgy += go_val * dval_dfy;
                    dgz += go_val * dval_dfz;
                }
                ctx->gg[gidx] = dgx * ctx->mult_x;
                ctx->gg[gidx + 1] = dgy * ctx->mult_y;
                ctx->gg[gidx + 2] = dgz * ctx->mult_z;
            }
        }
    }
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

    grid_forward_context_t context = {
        inp, grd, out, C, iD, iH, iW, oD, oH, oW,
        inp_sB, inp_sC, inp_sD, inp_sH,
        out_sB, out_sC, out_sD, out_sH,
        grd_sB, grd_sD, grd_sH
    };
    cfireants_parallel_for((size_t)B * oD, 2, grid_forward_range, &context);
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

    grid_backward_context_t context = {
        inp, grd, go, gg, C, iD, iH, iW, oD, oH, oW,
        inp_sC, inp_sD, inp_sH, out_sC, out_sD, out_sH,
        mult_x, mult_y, mult_z
    };
    cfireants_parallel_for((size_t)B * oD, 2, grid_backward_range, &context);
    return 0;
}
