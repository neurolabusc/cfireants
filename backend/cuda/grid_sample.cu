/*
 * grid_sample.cu - CUDA grid sampling (bilinear, zeros padding, align_corners=True)
 *
 * Adapted from fused_ops/FusedGridSampler.cu (FireANTs license).
 * Simplified to float32-only, 3D-only, bilinear-only.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define BLOCK_SIZE 256

/* Forward-declare the extern "C" wrapper functions at the bottom */

/* Unnormalize: [-1,1] → [0, size-1] (align_corners=True) */
__device__ __forceinline__ float unnorm(float x, int size) {
    return (x + 1.0f) * 0.5f * (size - 1);
}

/* ------------------------------------------------------------------ */
/* Forward kernel                                                      */
/* ------------------------------------------------------------------ */

__global__ void grid_sample_3d_fwd_kernel(
    const float * __restrict__ input,
    const float * __restrict__ grid,
    float * __restrict__ output,
    int B, int C, int iD, int iH, int iW,
    int oD, int oH, int oW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * oD * oH * oW;
    if (idx >= total) return;

    int w = idx % oW;
    int tmp = idx / oW;
    int h = tmp % oH;
    tmp /= oH;
    int d = tmp % oD;
    int b = tmp / oD;

    /* Read grid coordinates */
    int gidx = idx * 3;
    float gx = grid[gidx + 0]; /* x -> W */
    float gy = grid[gidx + 1]; /* y -> H */
    float gz = grid[gidx + 2]; /* z -> D */

    float ix = unnorm(gx, iW);
    float iy = unnorm(gy, iH);
    float iz = unnorm(gz, iD);

    int ix0 = __float2int_rd(ix); /* floor */
    int iy0 = __float2int_rd(iy);
    int iz0 = __float2int_rd(iz);
    int ix1 = ix0 + 1, iy1 = iy0 + 1, iz1 = iz0 + 1;

    float fx = ix - ix0, fy = iy - iy0, fz = iz - iz0;

    /* Weights */
    float w000 = (1-fx)*(1-fy)*(1-fz);
    float w001 = fx*(1-fy)*(1-fz);
    float w010 = (1-fx)*fy*(1-fz);
    float w011 = fx*fy*(1-fz);
    float w100 = (1-fx)*(1-fy)*fz;
    float w101 = fx*(1-fy)*fz;
    float w110 = (1-fx)*fy*fz;
    float w111 = fx*fy*fz;

    /* Bounds check (zeros padding) */
    #define VALID(d,h,w) ((d)>=0 && (d)<iD && (h)>=0 && (h)<iH && (w)>=0 && (w)<iW)
    #define INP(b,c,d,h,w) (VALID(d,h,w) ? input[((((long)(b)*C+(c))*(long)iD+(d))*(long)iH+(h))*(long)iW+(w)] : 0.0f)

    long out_base = (((long)b * C) * (long)oD + d) * (long)oH * oW + (long)h * oW + w;
    long out_stride = (long)oD * oH * oW;

    for (int c = 0; c < C; c++) {
        float val = w000 * INP(b,c,iz0,iy0,ix0)
                  + w001 * INP(b,c,iz0,iy0,ix1)
                  + w010 * INP(b,c,iz0,iy1,ix0)
                  + w011 * INP(b,c,iz0,iy1,ix1)
                  + w100 * INP(b,c,iz1,iy0,ix0)
                  + w101 * INP(b,c,iz1,iy0,ix1)
                  + w110 * INP(b,c,iz1,iy1,ix0)
                  + w111 * INP(b,c,iz1,iy1,ix1);
        output[out_base + (long)c * out_stride] = val;
    }
    #undef VALID
    #undef INP
}

extern "C" {

void cuda_grid_sample_3d_fwd(
    const float *input, const float *grid, float *output,
    int B, int C, int iD, int iH, int iW, int oD, int oH, int oW)
{
    int total = B * oD * oH * oW;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid_sample_3d_fwd_kernel<<<blocks, BLOCK_SIZE>>>(
        input, grid, output, B, C, iD, iH, iW, oD, oH, oW);
}

/* ------------------------------------------------------------------ */
/* Backward kernel (gradient w.r.t. grid only)                         */
/* ------------------------------------------------------------------ */

__global__ void grid_sample_3d_bwd_kernel(
    const float * __restrict__ grad_output,
    const float * __restrict__ input,
    const float * __restrict__ grid,
    float * __restrict__ grad_grid,
    int B, int C, int iD, int iH, int iW,
    int oD, int oH, int oW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * oD * oH * oW;
    if (idx >= total) return;

    int w = idx % oW;
    int tmp = idx / oW;
    int h = tmp % oH;
    tmp /= oH;
    int d = tmp % oD;
    int b = tmp / oD;

    int gidx = idx * 3;
    float gx = grid[gidx + 0];
    float gy = grid[gidx + 1];
    float gz = grid[gidx + 2];

    float ix = unnorm(gx, iW);
    float iy = unnorm(gy, iH);
    float iz = unnorm(gz, iD);

    int ix0 = __float2int_rd(ix);
    int iy0 = __float2int_rd(iy);
    int iz0 = __float2int_rd(iz);
    int ix1 = ix0 + 1, iy1 = iy0 + 1, iz1 = iz0 + 1;

    float fx = ix - ix0, fy = iy - iy0, fz = iz - iz0;

    #define VALID(d,h,w) ((d)>=0 && (d)<iD && (h)>=0 && (h)<iH && (w)>=0 && (w)<iW)
    #define INP(b,c,d,h,w) (VALID(d,h,w) ? input[((((long)(b)*C+(c))*(long)iD+(d))*(long)iH+(h))*(long)iW+(w)] : 0.0f)

    long go_stride = (long)oD * oH * oW;
    float dgx = 0, dgy = 0, dgz = 0;

    for (int c = 0; c < C; c++) {
        long go_idx = (((long)b * C + c) * (long)oD + d) * (long)oH * oW + (long)h * oW + w;
        float go = grad_output[go_idx];

        float dval_dfx =
            -(1-fy)*(1-fz)*INP(b,c,iz0,iy0,ix0) + (1-fy)*(1-fz)*INP(b,c,iz0,iy0,ix1)
            -fy*(1-fz)*INP(b,c,iz0,iy1,ix0) + fy*(1-fz)*INP(b,c,iz0,iy1,ix1)
            -(1-fy)*fz*INP(b,c,iz1,iy0,ix0) + (1-fy)*fz*INP(b,c,iz1,iy0,ix1)
            -fy*fz*INP(b,c,iz1,iy1,ix0) + fy*fz*INP(b,c,iz1,iy1,ix1);

        float dval_dfy =
            -(1-fx)*(1-fz)*INP(b,c,iz0,iy0,ix0) - fx*(1-fz)*INP(b,c,iz0,iy0,ix1)
            +(1-fx)*(1-fz)*INP(b,c,iz0,iy1,ix0) + fx*(1-fz)*INP(b,c,iz0,iy1,ix1)
            -(1-fx)*fz*INP(b,c,iz1,iy0,ix0) - fx*fz*INP(b,c,iz1,iy0,ix1)
            +(1-fx)*fz*INP(b,c,iz1,iy1,ix0) + fx*fz*INP(b,c,iz1,iy1,ix1);

        float dval_dfz =
            -(1-fx)*(1-fy)*INP(b,c,iz0,iy0,ix0) - fx*(1-fy)*INP(b,c,iz0,iy0,ix1)
            -(1-fx)*fy*INP(b,c,iz0,iy1,ix0) - fx*fy*INP(b,c,iz0,iy1,ix1)
            +(1-fx)*(1-fy)*INP(b,c,iz1,iy0,ix0) + fx*(1-fy)*INP(b,c,iz1,iy0,ix1)
            +(1-fx)*fy*INP(b,c,iz1,iy1,ix0) + fx*fy*INP(b,c,iz1,iy1,ix1);

        dgx += go * dval_dfx;
        dgy += go * dval_dfy;
        dgz += go * dval_dfz;
    }

    /* Chain rule: d(unnorm)/d(norm) */
    float mx = (iW - 1) * 0.5f;
    float my = (iH - 1) * 0.5f;
    float mz = (iD - 1) * 0.5f;

    grad_grid[gidx + 0] = dgx * mx;
    grad_grid[gidx + 1] = dgy * my;
    grad_grid[gidx + 2] = dgz * mz;

    #undef VALID
    #undef INP
}

void cuda_grid_sample_3d_bwd(
    const float *grad_output, const float *input, const float *grid,
    float *grad_grid,
    int B, int C, int iD, int iH, int iW, int oD, int oH, int oW)
{
    int total = B * oD * oH * oW;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid_sample_3d_bwd_kernel<<<blocks, BLOCK_SIZE>>>(
        grad_output, input, grid, grad_grid,
        B, C, iD, iH, iW, oD, oH, oW);
}

/* ------------------------------------------------------------------ */
/* Affine grid generation                                              */
/* ------------------------------------------------------------------ */

__global__ void affine_grid_3d_kernel(
    const float * __restrict__ affine,
    float * __restrict__ grid,
    int B, int D, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * D * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int tmp = idx / W;
    int h = tmp % H;
    tmp /= H;
    int d = tmp % D;
    int b = tmp / D;

    float nz = (D > 1) ? (2.0f * d / (D - 1) - 1.0f) : 0.0f;
    float ny = (H > 1) ? (2.0f * h / (H - 1) - 1.0f) : 0.0f;
    float nx = (W > 1) ? (2.0f * w / (W - 1) - 1.0f) : 0.0f;

    const float *A = affine + b * 12;
    float ox = A[0]*nx + A[1]*ny + A[2]*nz + A[3];
    float oy = A[4]*nx + A[5]*ny + A[6]*nz + A[7];
    float oz = A[8]*nx + A[9]*ny + A[10]*nz + A[11];

    int gidx = idx * 3;
    grid[gidx + 0] = ox;
    grid[gidx + 1] = oy;
    grid[gidx + 2] = oz;
}

void cuda_affine_grid_3d(const float *affine, float *grid,
                         int B, int D, int H, int W)
{
    int total = B * D * H * W;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    affine_grid_3d_kernel<<<blocks, BLOCK_SIZE>>>(affine, grid, B, D, H, W);
}

/* ------------------------------------------------------------------ */
/* Element-wise ops                                                    */
/* ------------------------------------------------------------------ */

__global__ void tensor_add_kernel(float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void tensor_fill_kernel(float *data, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = val;
}

__global__ void tensor_scale_kernel(float *data, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= alpha;
}

void cuda_tensor_add(float *a, const float *b, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    tensor_add_kernel<<<blocks, BLOCK_SIZE>>>(a, b, n);
}

void cuda_tensor_fill(float *data, float val, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    tensor_fill_kernel<<<blocks, BLOCK_SIZE>>>(data, val, n);
}

void cuda_tensor_scale(float *data, float alpha, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    tensor_scale_kernel<<<blocks, BLOCK_SIZE>>>(data, alpha, n);
}

/* ------------------------------------------------------------------ */
/* Adam moments update (separate from param update, for compositive)   */
/* ------------------------------------------------------------------ */

__global__ void adam_moments_kernel(
    const float *grad, float *exp_avg, float *exp_avg_sq,
    float beta1, float beta2, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    exp_avg[i] = beta1 * exp_avg[i] + (1.0f - beta1) * g;
    exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g * g;
}

void cuda_adam_moments_update(const float *grad, float *exp_avg, float *exp_avg_sq,
                              float beta1, float beta2, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    adam_moments_kernel<<<blocks, BLOCK_SIZE>>>(grad, exp_avg, exp_avg_sq, beta1, beta2, n);
}

__global__ void adam_dir_kernel(
    float *output, const float *exp_avg, const float *exp_avg_sq,
    float inv_bc1, float inv_bc2, float eps, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float m_hat = exp_avg[i] * inv_bc1;
    float v_hat = exp_avg_sq[i] * inv_bc2;
    output[i] = m_hat / (sqrtf(v_hat) + eps);
}

void cuda_adam_direction(float *output, const float *exp_avg, const float *exp_avg_sq,
                         float bc1, float bc2, float eps, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    adam_dir_kernel<<<blocks, BLOCK_SIZE>>>(output, exp_avg, exp_avg_sq,
                                            1.0f/bc1, 1.0f/bc2, eps, n);
}

/* ------------------------------------------------------------------ */
/* Adam update (standard, for rigid/affine)                            */
/* ------------------------------------------------------------------ */

__global__ void adam_step_kernel(
    float *param, const float *grad,
    float *exp_avg, float *exp_avg_sq,
    float step_size, float beta1, float beta2, float eps,
    float bc2_sqrt_inv, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grad[i];
    float m = beta1 * exp_avg[i] + (1 - beta1) * g;
    float v = beta2 * exp_avg_sq[i] + (1 - beta2) * g * g;
    exp_avg[i] = m;
    exp_avg_sq[i] = v;

    float denom = sqrtf(v * bc2_sqrt_inv) + eps;
    param[i] -= step_size * m / denom;
}

void cuda_adam_step(
    float *param, const float *grad,
    float *exp_avg, float *exp_avg_sq,
    float lr, float beta1, float beta2, float eps,
    int step, int n)
{
    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);
    float step_size = lr / bc1;
    float bc2_sqrt_inv = 1.0f / bc2; /* v/bc2 inside sqrt */

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    adam_step_kernel<<<blocks, BLOCK_SIZE>>>(
        param, grad, exp_avg, exp_avg_sq,
        step_size, beta1, beta2, eps, bc2_sqrt_inv, n);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "cuda_adam_step: %s (blocks=%d, n=%d)\n", cudaGetErrorString(err), blocks, n);
}

/* ------------------------------------------------------------------ */
/* Trilinear resize                                                    */
/* ------------------------------------------------------------------ */

__global__ void trilinear_resize_kernel(
    const float * __restrict__ input, float * __restrict__ output,
    int B, int C, int iD, int iH, int iW, int oD, int oH, int oW,
    int align_corners)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * oD * oH * oW;
    if (idx >= total) return;

    int ow = idx % oW;
    int tmp = idx / oW;
    int oh = tmp % oH;
    tmp /= oH;
    int od = tmp % oD;
    tmp /= oD;
    int c = tmp % C;
    int b = tmp / C;

    float sd, sh, sw;
    if (align_corners && oD > 1) sd = (float)od * (iD - 1) / (oD - 1);
    else sd = ((float)od + 0.5f) * iD / oD - 0.5f;
    if (align_corners && oH > 1) sh = (float)oh * (iH - 1) / (oH - 1);
    else sh = ((float)oh + 0.5f) * iH / oH - 0.5f;
    if (align_corners && oW > 1) sw = (float)ow * (iW - 1) / (oW - 1);
    else sw = ((float)ow + 0.5f) * iW / oW - 0.5f;

    int d0 = __float2int_rd(sd), h0 = __float2int_rd(sh), w0 = __float2int_rd(sw);
    int d1 = d0+1, h1 = h0+1, w1 = w0+1;
    float fd = sd-d0, fh = sh-h0, fw = sw-w0;
    if (d0 < 0) { d0 = 0; fd = 0; } if (d1 >= iD) d1 = iD-1;
    if (h0 < 0) { h0 = 0; fh = 0; } if (h1 >= iH) h1 = iH-1;
    if (w0 < 0) { w0 = 0; fw = 0; } if (w1 >= iW) w1 = iW-1;

    const float *src = input + ((long)b*C+c)*(long)iD*iH*iW;
    #define S(d,h,w) src[(long)(d)*iH*iW+(h)*iW+(w)]

    float val = (1-fd)*(1-fh)*(1-fw)*S(d0,h0,w0) + (1-fd)*(1-fh)*fw*S(d0,h0,w1)
              + (1-fd)*fh*(1-fw)*S(d0,h1,w0) + (1-fd)*fh*fw*S(d0,h1,w1)
              + fd*(1-fh)*(1-fw)*S(d1,h0,w0) + fd*(1-fh)*fw*S(d1,h0,w1)
              + fd*fh*(1-fw)*S(d1,h1,w0) + fd*fh*fw*S(d1,h1,w1);
    #undef S

    output[idx] = val;
}

void cuda_trilinear_resize(
    const float *input, float *output,
    int B, int C, int iD, int iH, int iW,
    int oD, int oH, int oW, int align_corners)
{
    int total = B * C * oD * oH * oW;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    trilinear_resize_kernel<<<blocks, BLOCK_SIZE>>>(
        input, output, B, C, iD, iH, iW, oD, oH, oW, align_corners);
}

/* ------------------------------------------------------------------ */
/* Separable 1D convolution along one axis (zero-padded)               */
/* ------------------------------------------------------------------ */

__global__ void conv1d_axis_kernel(
    const float * __restrict__ in, float * __restrict__ out,
    int D, int H, int W,
    const float * __restrict__ kernel, int klen, int axis)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = D * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int tmp = idx / W;
    int h = tmp % H;
    int d = tmp / H;
    int r = klen / 2;

    float sum = 0;
    if (axis == 2) { /* W */
        for (int k = 0; k < klen; k++) {
            int ww = w + k - r;
            if (ww >= 0 && ww < W)
                sum += in[(long)d*H*W + h*W + ww] * kernel[k];
        }
    } else if (axis == 1) { /* H */
        for (int k = 0; k < klen; k++) {
            int hh = h + k - r;
            if (hh >= 0 && hh < H)
                sum += in[(long)d*H*W + hh*W + w] * kernel[k];
        }
    } else { /* D */
        for (int k = 0; k < klen; k++) {
            int dd = d + k - r;
            if (dd >= 0 && dd < D)
                sum += in[(long)dd*H*W + h*W + w] * kernel[k];
        }
    }
    out[idx] = sum;
}

void cuda_conv1d_axis(const float *in, float *out,
                      int D, int H, int W,
                      const float *kernel, int klen, int axis)
{
    int total = D * H * W;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    conv1d_axis_kernel<<<blocks, BLOCK_SIZE>>>(in, out, D, H, W, kernel, klen, axis);
}

/* Box filter along one axis */
void cuda_box_filter_axis(const float *in, float *out,
                          int D, int H, int W, int ks, int axis,
                          float scale)
{
    /* Build uniform kernel on device */
    float *d_kernel;
    cudaMalloc(&d_kernel, ks * sizeof(float));
    float *h_kernel = (float *)malloc(ks * sizeof(float));
    for (int i = 0; i < ks; i++) h_kernel[i] = scale;
    cudaMemcpy(d_kernel, h_kernel, ks * sizeof(float), cudaMemcpyHostToDevice);
    free(h_kernel);

    cuda_conv1d_axis(in, out, D, H, W, d_kernel, ks, axis);
    cudaFree(d_kernel);
}

/* ------------------------------------------------------------------ */
/* Blur-then-downsample                                                */
/* ------------------------------------------------------------------ */

/* Build 1D Gaussian kernel (erf approx), allocate on device */
static float *build_gpu_gauss_kernel(float sigma, float truncated, int *klen_out) {
    if (sigma <= 0) { *klen_out = 1; float one = 1.0f; float *d; cudaMalloc(&d, sizeof(float)); cudaMemcpy(d, &one, sizeof(float), cudaMemcpyHostToDevice); return d; }
    int tail = (int)(truncated * sigma + 0.5f);
    int klen = 2 * tail + 1;
    float *h = (float *)malloc(klen * sizeof(float));
    float inv = 1.0f / (sigma * sqrtf(2.0f));
    float sum = 0;
    for (int i = 0; i < klen; i++) {
        float x = (float)(i - tail);
        h[i] = 0.5f * (erff((x+0.5f)*inv) - erff((x-0.5f)*inv));
        sum += h[i];
    }
    for (int i = 0; i < klen; i++) h[i] /= sum;
    float *d; cudaMalloc(&d, klen * sizeof(float));
    cudaMemcpy(d, h, klen * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
    *klen_out = klen;
    return d;
}

void cuda_blur_downsample(
    const float *input, float *output,
    int B, int C, int iD, int iH, int iW, int oD, int oH, int oW)
{
    /* Sigma per axis: 0.5 * (input_size / output_size), matching Python */
    float sigma_d = 0.5f * (float)iD / (float)oD;
    float sigma_h = 0.5f * (float)iH / (float)oH;
    float sigma_w = 0.5f * (float)iW / (float)oW;

    /* Build per-axis Gaussian kernels */
    int klen_d, klen_h, klen_w;
    float *dk_d = build_gpu_gauss_kernel(sigma_d, 2.0f, &klen_d);
    float *dk_h = build_gpu_gauss_kernel(sigma_h, 2.0f, &klen_h);
    float *dk_w = build_gpu_gauss_kernel(sigma_w, 2.0f, &klen_w);

    long spatial_in = (long)iD * iH * iW;

    /* Blur each (B,C) slice: separable 3-pass */
    float *d_tmp;
    cudaMalloc(&d_tmp, (size_t)spatial_in * sizeof(float));

    float *d_blurred;
    cudaMalloc(&d_blurred, (size_t)B * C * spatial_in * sizeof(float));

    for (int bc = 0; bc < B * C; bc++) {
        const float *src = input + (size_t)bc * spatial_in;
        float *dst = d_blurred + (size_t)bc * spatial_in;

        /* D axis: src -> d_tmp */
        cuda_conv1d_axis(src, d_tmp, iD, iH, iW, dk_d, klen_d, 0);
        /* H axis: d_tmp -> dst */
        cuda_conv1d_axis(d_tmp, dst, iD, iH, iW, dk_h, klen_h, 1);
        /* W axis: dst -> d_tmp, copy back */
        cuda_conv1d_axis(dst, d_tmp, iD, iH, iW, dk_w, klen_w, 2);
        cudaMemcpy(dst, d_tmp, spatial_in * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_tmp);

    /* Trilinear resize */
    cuda_trilinear_resize(d_blurred, output, B, C, iD, iH, iW, oD, oH, oW, 1);

    cudaFree(d_blurred);
    cudaFree(dk_d); cudaFree(dk_h); cudaFree(dk_w);
}

} /* extern "C" */
