/*
 * downsample_fft.cu - FFT-based downsampling matching Python downsample_fft
 *
 * Faithful clone of fireants/utils/imageutils.py downsample_fft():
 *   1. fftn(image, dim=spatial)
 *   2. fftshift
 *   3. Crop to [target_size + 2*padding] centered region
 *   4. Apply Gaussian window: exp(-0.5 * (freq/sigma)^2), sigma = dim/4
 *   5. Scale by prod(target/source)
 *   6. Remove padding
 *   7. ifftshift + ifftn
 *   8. Clamp to original range
 *
 * Uses cuFFT for the forward/inverse transforms.
 */

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

extern "C" void cuda_tensor_scale(float *data, float alpha, int n);

#define BLK 256

/* ------------------------------------------------------------------ */
/* Gaussian blur in frequency domain (clone of fused_ops kernel)       */
/* ------------------------------------------------------------------ */

__global__ void gaussian_blur_fft3_kernel(
    cufftComplex *im_fft,
    int D, int H, int W,
    int zs, int ys, int xs,
    float multiplier,
    long n_elements)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    int w = i % W;
    int h = (i / W) % H;
    int d = (i / ((long)H * W)) % D;

    float sigma_z = (float)D / 4.0f;
    float sigma_y = (float)H / 4.0f;
    float sigma_x = (float)W / 4.0f;

    float z_freq = (float)(zs + d) / sigma_z;
    float y_freq = (float)(ys + h) / sigma_y;
    float x_freq = (float)(xs + w) / sigma_x;

    float exp_val = expf(-0.5f * (z_freq*z_freq + y_freq*y_freq + x_freq*x_freq));
    float scale = exp_val * multiplier;

    im_fft[i].x *= scale;
    im_fft[i].y *= scale;
}

/* ------------------------------------------------------------------ */
/* fftshift / ifftshift for 3D complex data                            */
/* Applied per (B,C) slice on the spatial dims [D,H,W]                 */
/* ------------------------------------------------------------------ */

__global__ void fftshift_3d_kernel(
    const cufftComplex *in, cufftComplex *out,
    int D, int H, int W, int inverse)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)D * H * W;
    if (i >= total) return;

    int w = i % W;
    int h = (i / W) % H;
    int d = i / ((long)H * W);

    /* fftshift: out[i] = in[(i + ceil(N/2)) % N]  (moves DC from 0 to N//2)
     * ifftshift: out[i] = in[(i + N//2) % N]     (moves DC from N//2 back to 0)
     *
     * For even N: ceil(N/2) = N/2, so both are the same.
     * For odd N: ceil(N/2) = (N+1)/2, and N//2 = (N-1)/2. */
    int sd, sh, sw;
    if (inverse) {
        sd = (d + D/2) % D;         /* shift by N//2 = floor(N/2) */
        sh = (h + H/2) % H;
        sw = (w + W/2) % W;
    } else {
        sd = (d + (D+1)/2) % D;     /* shift by ceil(N/2) */
        sh = (h + (H+1)/2) % H;
        sw = (w + (W+1)/2) % W;
    }
    long src_idx = ((long)sd * H + sh) * W + sw;
    out[i] = in[src_idx];
}

/* ------------------------------------------------------------------ */
/* Real-to-complex and complex-to-real helpers                         */
/* ------------------------------------------------------------------ */

__global__ void real_to_complex_kernel(const float *real, cufftComplex *cmplx, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { cmplx[i].x = real[i]; cmplx[i].y = 0; }
}

__global__ void complex_to_real_kernel(const cufftComplex *cmplx, float *real, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) real[i] = cmplx[i].x;
}

__global__ void clamp_kernel(float *data, float lo, float hi, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (data[i] < lo) data[i] = lo;
        if (data[i] > hi) data[i] = hi;
    }
}

/* Copy a sub-region from a 3D complex volume (crop) */
__global__ void crop_3d_kernel(
    const cufftComplex *src, cufftComplex *dst,
    int sD, int sH, int sW,   /* source dims */
    int dD, int dH, int dW,   /* dest dims */
    int d0, int h0, int w0)   /* start offsets in source */
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)dD * dH * dW;
    if (i >= total) return;

    int dw = i % dW;
    int dh = (i / dW) % dH;
    int dd = i / ((long)dH * dW);

    long src_idx = ((long)(dd + d0) * sH + (dh + h0)) * sW + (dw + w0);
    dst[i] = src[src_idx];
}

/* ------------------------------------------------------------------ */
/* Main FFT downsample function                                        */
/* ------------------------------------------------------------------ */

extern "C" {

/* Find min/max of a float array on GPU */
static void gpu_minmax(const float *data, long n, float *h_min, float *h_max) {
    /* Simple: download and scan on CPU. For large arrays, a proper reduction
     * would be faster, but this is called once per downsample. */
    float *h = (float *)malloc(n * sizeof(float));
    cudaMemcpy(h, data, n * sizeof(float), cudaMemcpyDeviceToHost);
    float lo = h[0], hi = h[0];
    for (long i = 1; i < n; i++) { if (h[i] < lo) lo = h[i]; if (h[i] > hi) hi = h[i]; }
    *h_min = lo; *h_max = hi;
    free(h);
}

void cuda_downsample_fft(
    const float *input, float *output,
    int B, int C,
    int iD, int iH, int iW,
    int oD, int oH, int oW)
{
    long spatial_in = (long)iD * iH * iW;
    long spatial_out = (long)oD * oH * oW;
    int padding = 1;

    /* Cropped dims (with padding) */
    int cD = oD + 2 * padding;
    int cH = oH + 2 * padding;
    int cW = oW + 2 * padding;
    long spatial_crop = (long)cD * cH * cW;

    /* Start indices for crop (matching Python) */
    int d0 = iD/2 - (oD/2 + padding);
    int h0 = iH/2 - (oH/2 + padding);
    int w0 = iW/2 - (oW/2 + padding);

    /* Multiplier = prod(target/source) */
    float multiplier = ((float)oD / iD) * ((float)oH / iH) * ((float)oW / iW);

    /* Frequency-domain start coords for Gaussian window */
    int zs = -(oD/2 + padding);
    int ys = -(oH/2 + padding);
    int xs = -(oW/2 + padding);

    /* Allocate complex buffers */
    cufftComplex *d_fft_in, *d_fft_shifted, *d_fft_cropped, *d_fft_unshifted;
    cudaMalloc(&d_fft_in, spatial_in * sizeof(cufftComplex));
    cudaMalloc(&d_fft_shifted, spatial_in * sizeof(cufftComplex));
    cudaMalloc(&d_fft_cropped, spatial_crop * sizeof(cufftComplex));
    cudaMalloc(&d_fft_unshifted, spatial_out * sizeof(cufftComplex));

    /* Output for trimmed (after removing padding) */
    cufftComplex *d_fft_trimmed;
    cudaMalloc(&d_fft_trimmed, spatial_out * sizeof(cufftComplex));

    /* cuFFT plans */
    cufftHandle plan_fwd, plan_inv;
    cufftPlan3d(&plan_fwd, iD, iH, iW, CUFFT_C2C);
    cufftPlan3d(&plan_inv, oD, oH, oW, CUFFT_C2C);

    long blk_in = (spatial_in + BLK - 1) / BLK;
    long blk_crop = (spatial_crop + BLK - 1) / BLK;
    long blk_out = (spatial_out + BLK - 1) / BLK;

    for (int bc = 0; bc < B * C; bc++) {
        const float *src = input + (long)bc * spatial_in;
        float *dst = output + (long)bc * spatial_out;

        /* Get min/max for clamping after IFFT */
        float vmin, vmax;
        gpu_minmax(src, spatial_in, &vmin, &vmax);

        /* Real to complex */
        real_to_complex_kernel<<<blk_in, BLK>>>(src, d_fft_in, spatial_in);

        /* Forward FFT */
        cufftExecC2C(plan_fwd, d_fft_in, d_fft_in, CUFFT_FORWARD);
        cudaDeviceSynchronize();

        /* fftshift */
        fftshift_3d_kernel<<<blk_in, BLK>>>(d_fft_in, d_fft_shifted, iD, iH, iW, 0);

        /* Crop centered region */
        crop_3d_kernel<<<blk_crop, BLK>>>(
            d_fft_shifted, d_fft_cropped,
            iD, iH, iW, cD, cH, cW, d0, h0, w0);

        /* Gaussian blur in frequency domain */
        gaussian_blur_fft3_kernel<<<blk_crop, BLK>>>(
            d_fft_cropped, cD, cH, cW, zs, ys, xs, multiplier, spatial_crop);

        /* Remove padding: crop [padding:-padding] from each dim */
        crop_3d_kernel<<<blk_out, BLK>>>(
            d_fft_cropped, d_fft_trimmed,
            cD, cH, cW, oD, oH, oW, padding, padding, padding);

        /* ifftshift */
        fftshift_3d_kernel<<<blk_out, BLK>>>(d_fft_trimmed, d_fft_unshifted, oD, oH, oW, 1);

        /* Inverse FFT */
        cufftExecC2C(plan_inv, d_fft_unshifted, d_fft_unshifted, CUFFT_INVERSE);

        /* cuFFT doesn't normalize, divide by N */
        /* Actually, the Python code doesn't divide by N because the multiplier
         * already accounts for the size ratio. But cuFFT's inverse doesn't
         * include 1/N normalization while PyTorch's does. So we need to divide. */
        /* PyTorch ifftn includes 1/N normalization. cuFFT does not. */
        float inv_N = 1.0f / (float)spatial_out;

        /* Complex to real (with 1/N scale) */
        /* We'll handle the scale in the copy kernel */
        complex_to_real_kernel<<<blk_out, BLK>>>(d_fft_unshifted, dst, spatial_out);

        /* Scale by 1/N (cuFFT normalization) */
        /* Actually: cuFFT forward doesn't normalize, inverse doesn't normalize.
         * PyTorch fftn doesn't normalize by default ("backward" norm).
         * PyTorch ifftn normalizes by 1/N ("backward" norm).
         * So we need to divide the IFFT output by N. */
        cuda_tensor_scale(dst, inv_N, (int)spatial_out);

        /* Clamp to original range */
        clamp_kernel<<<blk_out, BLK>>>(dst, vmin, vmax, spatial_out);
    }

    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);
    cudaFree(d_fft_in); cudaFree(d_fft_shifted);
    cudaFree(d_fft_cropped); cudaFree(d_fft_unshifted);
    cudaFree(d_fft_trimmed);
}

} /* extern "C" */
