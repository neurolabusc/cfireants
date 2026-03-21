/*
 * test_fft_fallback.c - Validate kissfft CPU downsample against CUDA cuFFT
 *
 * Loads a NIfTI image, downsamples with both CUDA and CPU FFT paths,
 * compares the results. This ensures the WebGPU FFT fallback matches CUDA.
 */

#include "cfireants/backend.h"
#include "cfireants/tensor.h"
#include "cfireants/image.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* CUDA FFT downsample (from downsample_fft.cu) */
#ifdef CFIREANTS_HAS_CUDA
extern int cfireants_init_cuda(void);
extern void cuda_downsample_fft(const float *input, float *output,
    int B, int C, int iD, int iH, int iW, int oD, int oH, int oW);
extern int cuda_memcpy_h2d(void *dst, const void *src, size_t nbytes);
extern int cuda_memcpy_d2h(void *dst, const void *src, size_t nbytes);
extern int cuda_malloc(void **ptr, size_t nbytes);
extern void cuda_free(void *ptr);
#endif

/* CPU FFT downsample (from fft_cpu_fallback.c) */
extern void webgpu_downsample_fft(const float *input, float *output,
    int B, int C, int iD, int iH, int iW, int oD, int oH, int oW);

static int n_pass = 0, n_fail = 0;

static void compare_arrays(const char *label,
                           const float *ref, const float *test,
                           int n, float atol, float rtol_pct) {
    float max_diff = 0, mean_diff = 0;
    int n_mismatch = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(ref[i] - test[i]);
        if (d > max_diff) max_diff = d;
        mean_diff += d;
        if (d > atol) n_mismatch++;
    }
    mean_diff /= n;

    /* Relative: max_diff as % of data range */
    float vmin = ref[0], vmax = ref[0];
    for (int i = 1; i < n; i++) {
        if (ref[i] < vmin) vmin = ref[i];
        if (ref[i] > vmax) vmax = ref[i];
    }
    float range = vmax - vmin;
    float rel_pct = (range > 0) ? 100.0f * max_diff / range : 0;

    printf("  %s: max_diff=%.6f mean_diff=%.6f rel=%.2f%% mismatches=%d/%d\n",
           label, max_diff, mean_diff, rel_pct, n_mismatch, n);

    if (rel_pct <= rtol_pct) {
        n_pass++;
        printf("  PASS: relative error %.2f%% <= %.2f%%\n", rel_pct, rtol_pct);
    } else {
        n_fail++;
        printf("  FAIL: relative error %.2f%% > %.2f%%\n", rel_pct, rtol_pct);
    }
}

/* Test with synthetic data (known pattern) */
static void test_synthetic(int iD, int iH, int iW, int oD, int oH, int oW) {
    printf("\n=== Synthetic %dx%dx%d -> %dx%dx%d ===\n", iD, iH, iW, oD, oH, oW);

    int B = 1, C = 1;
    long n_in = (long)iD * iH * iW;
    long n_out = (long)oD * oH * oW;

    float *h_in = (float *)malloc(n_in * sizeof(float));
    float *h_cpu_out = (float *)malloc(n_out * sizeof(float));

    /* Create a smooth 3D pattern (low-frequency content survives downsampling) */
    for (int d = 0; d < iD; d++) {
        for (int h = 0; h < iH; h++) {
            for (int w = 0; w < iW; w++) {
                float z = 2.0f * (float)d / iD - 1.0f;
                float y = 2.0f * (float)h / iH - 1.0f;
                float x = 2.0f * (float)w / iW - 1.0f;
                h_in[((long)d * iH + h) * iW + w] =
                    1000.0f * expf(-2.0f * (z*z + y*y + x*x));
            }
        }
    }

    /* CPU (kissfft) downsample */
    webgpu_downsample_fft(h_in, h_cpu_out, B, C, iD, iH, iW, oD, oH, oW);

#ifdef CFIREANTS_HAS_CUDA
    /* CUDA downsample */
    float *d_in, *d_out;
    cuda_malloc((void **)&d_in, n_in * sizeof(float));
    cuda_malloc((void **)&d_out, n_out * sizeof(float));
    cuda_memcpy_h2d(d_in, h_in, n_in * sizeof(float));

    cuda_downsample_fft(d_in, d_out, B, C, iD, iH, iW, oD, oH, oW);

    float *h_cuda_out = (float *)malloc(n_out * sizeof(float));
    cuda_memcpy_d2h(h_cuda_out, d_out, n_out * sizeof(float));

    /* Compare CPU vs CUDA */
    compare_arrays("CPU vs CUDA", h_cuda_out, h_cpu_out, (int)n_out, 0.1f, 1.0f);

    free(h_cuda_out);
    cuda_free(d_in);
    cuda_free(d_out);
#else
    /* No CUDA — just verify CPU output is reasonable */
    float vmin = h_cpu_out[0], vmax = h_cpu_out[0];
    for (long i = 1; i < n_out; i++) {
        if (h_cpu_out[i] < vmin) vmin = h_cpu_out[i];
        if (h_cpu_out[i] > vmax) vmax = h_cpu_out[i];
    }
    printf("  CPU output range: [%.2f, %.2f]\n", vmin, vmax);
    if (vmax > 100.0f) { n_pass++; printf("  PASS: output has expected range\n"); }
    else { n_fail++; printf("  FAIL: output range too small\n"); }
#endif

    free(h_in);
    free(h_cpu_out);
}

/* Test with actual validation image */
static void test_with_image(const char *path, int scale) {
    printf("\n=== Image downsample: %s (scale=%d) ===\n", path, scale);

    image_t img;
    if (image_load(&img, path, DEVICE_CPU) != 0) {
        printf("  SKIP: could not load %s\n", path);
        return;
    }

    int B = 1, C = 1;
    int iD = img.data.shape[2], iH = img.data.shape[3], iW = img.data.shape[4];
    int oD = iD / scale, oH = iH / scale, oW = iW / scale;
    long n_in = (long)iD * iH * iW;
    long n_out = (long)oD * oH * oW;

    printf("  %dx%dx%d -> %dx%dx%d\n", iD, iH, iW, oD, oH, oW);

    float *h_in = (float *)img.data.data;
    float *h_cpu_out = (float *)malloc(n_out * sizeof(float));

    /* CPU downsample */
    webgpu_downsample_fft(h_in, h_cpu_out, B, C, iD, iH, iW, oD, oH, oW);

#ifdef CFIREANTS_HAS_CUDA
    float *d_in, *d_out;
    cuda_malloc((void **)&d_in, n_in * sizeof(float));
    cuda_malloc((void **)&d_out, n_out * sizeof(float));
    cuda_memcpy_h2d(d_in, h_in, n_in * sizeof(float));

    cuda_downsample_fft(d_in, d_out, B, C, iD, iH, iW, oD, oH, oW);

    float *h_cuda_out = (float *)malloc(n_out * sizeof(float));
    cuda_memcpy_d2h(h_cuda_out, d_out, n_out * sizeof(float));

    compare_arrays("CPU vs CUDA", h_cuda_out, h_cpu_out, (int)n_out, 1.0f, 1.0f);

    free(h_cuda_out);
    cuda_free(d_in);
    cuda_free(d_out);
#else
    printf("  SKIP CUDA comparison (no CUDA)\n");
#endif

    free(h_cpu_out);
    image_free(&img);
}

int main(void) {
    printf("=== FFT CPU Fallback Tests ===\n");

    cfireants_init_cpu();
#ifdef CFIREANTS_HAS_CUDA
    cfireants_init_cuda();
#endif

    /* Synthetic tests — various nPoT sizes matching our validation datasets */
    test_synthetic(91, 109, 91, 45, 54, 45);    /* small: scale 2 */
    test_synthetic(91, 109, 91, 22, 27, 22);    /* small: scale 4 */
    test_synthetic(182, 218, 182, 91, 109, 91); /* medium: scale 2 */
    test_synthetic(182, 218, 182, 45, 54, 45);  /* medium: scale 4 */

    /* Real image tests (if data available) */
    test_with_image("validate/small/MNI152_T1_2mm.nii.gz", 2);
    test_with_image("validate/medium/MNI152_T1_1mm_brain.nii.gz", 2);

    printf("\n========================================\n");
    printf("%d passed, %d failed\n", n_pass, n_fail);
    printf("========================================\n");

    return n_fail > 0 ? 1 : 0;
}
