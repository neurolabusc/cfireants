/*
 * test_webgpu_basic.c - Phase 1 WebGPU backend tests
 *
 * Tests: device init, buffer alloc, fill, scale, axpy, sum, H2D/D2H round-trip.
 */

#include "cfireants/backend.h"
#include "cfireants/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef CFIREANTS_HAS_WEBGPU
extern int cfireants_init_webgpu(void);
#endif

static int n_pass = 0, n_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { n_pass++; printf("  PASS: %s\n", msg); } \
    else { n_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

#define CHECK_CLOSE(a, b, tol, msg) do { \
    float _a = (a), _b = (b); \
    if (fabsf(_a - _b) <= (tol)) { n_pass++; printf("  PASS: %s (%.6f == %.6f)\n", msg, _a, _b); } \
    else { n_fail++; printf("  FAIL: %s (%.6f != %.6f, diff=%.6e)\n", msg, _a, _b, fabsf(_a-_b)); } \
} while(0)

static void test_alloc_free(void) {
    printf("\n=== Tensor Alloc/Free ===\n");

    tensor_t t;
    int shape[] = {10, 20, 30};
    int rc = tensor_alloc(&t, 3, shape, DTYPE_FLOAT32, DEVICE_WEBGPU);
    CHECK(rc == 0, "alloc WebGPU tensor");
    CHECK(t.numel == 6000, "numel correct");
    CHECK(t.device == DEVICE_WEBGPU, "device correct");
    CHECK(t.data != NULL, "data pointer non-null");
    tensor_free(&t);
    CHECK(t.data == NULL, "data NULL after free");
}

static void test_fill(void) {
    printf("\n=== Tensor Fill ===\n");

    tensor_t gpu;
    int shape[] = {1000};
    tensor_alloc(&gpu, 1, shape, DTYPE_FLOAT32, DEVICE_WEBGPU);

    g_backend->tensor_fill(&gpu, 3.14f);

    /* Read back */
    float *host = (float *)malloc(1000 * sizeof(float));
    tensor_t cpu;
    tensor_alloc_cpu_f32(&cpu, 1, shape);
    tensor_copy(&cpu, &gpu);
    memcpy(host, cpu.data, 1000 * sizeof(float));

    CHECK_CLOSE(host[0], 3.14f, 1e-5, "fill[0]");
    CHECK_CLOSE(host[500], 3.14f, 1e-5, "fill[500]");
    CHECK_CLOSE(host[999], 3.14f, 1e-5, "fill[999]");

    free(host);
    tensor_free(&cpu);
    tensor_free(&gpu);
}

static void test_scale(void) {
    printf("\n=== Tensor Scale ===\n");

    int shape[] = {100};
    tensor_t cpu, gpu;
    tensor_alloc_cpu_f32(&cpu, 1, shape);
    for (int i = 0; i < 100; i++) ((float*)cpu.data)[i] = (float)i;

    tensor_alloc(&gpu, 1, shape, DTYPE_FLOAT32, DEVICE_WEBGPU);
    tensor_copy(&gpu, &cpu);

    g_backend->tensor_scale(&gpu, 2.5f);

    tensor_copy(&cpu, &gpu);
    float *d = (float *)cpu.data;
    CHECK_CLOSE(d[0], 0.0f, 1e-5, "scale[0]");
    CHECK_CLOSE(d[1], 2.5f, 1e-5, "scale[1]");
    CHECK_CLOSE(d[99], 247.5f, 1e-3, "scale[99]");

    tensor_free(&cpu);
    tensor_free(&gpu);
}

static void test_axpy(void) {
    printf("\n=== Tensor AXPY ===\n");

    int shape[] = {100};
    tensor_t y_cpu, x_cpu, y_gpu, x_gpu;
    tensor_alloc_cpu_f32(&y_cpu, 1, shape);
    tensor_alloc_cpu_f32(&x_cpu, 1, shape);
    for (int i = 0; i < 100; i++) {
        ((float*)y_cpu.data)[i] = (float)i;
        ((float*)x_cpu.data)[i] = 1.0f;
    }

    tensor_alloc(&y_gpu, 1, shape, DTYPE_FLOAT32, DEVICE_WEBGPU);
    tensor_alloc(&x_gpu, 1, shape, DTYPE_FLOAT32, DEVICE_WEBGPU);
    tensor_copy(&y_gpu, &y_cpu);
    tensor_copy(&x_gpu, &x_cpu);

    g_backend->tensor_axpy(&y_gpu, 10.0f, &x_gpu);

    tensor_copy(&y_cpu, &y_gpu);
    float *d = (float *)y_cpu.data;
    CHECK_CLOSE(d[0], 10.0f, 1e-5, "axpy[0] = 0 + 10*1");
    CHECK_CLOSE(d[50], 60.0f, 1e-4, "axpy[50] = 50 + 10*1");
    CHECK_CLOSE(d[99], 109.0f, 1e-3, "axpy[99] = 99 + 10*1");

    tensor_free(&y_cpu);
    tensor_free(&x_cpu);
    tensor_free(&y_gpu);
    tensor_free(&x_gpu);
}

static void test_sum(void) {
    printf("\n=== Tensor Sum ===\n");

    int shape[] = {1000};
    tensor_t cpu, gpu;
    tensor_alloc_cpu_f32(&cpu, 1, shape);
    for (int i = 0; i < 1000; i++) ((float*)cpu.data)[i] = 1.0f;

    tensor_alloc(&gpu, 1, shape, DTYPE_FLOAT32, DEVICE_WEBGPU);
    tensor_copy(&gpu, &cpu);

    float s = g_backend->tensor_sum(&gpu);
    CHECK_CLOSE(s, 1000.0f, 1e-1, "sum of 1000 ones");

    /* Test with larger array */
    tensor_free(&cpu);
    tensor_free(&gpu);

    int shape2[] = {10000};
    tensor_alloc_cpu_f32(&cpu, 1, shape2);
    for (int i = 0; i < 10000; i++) ((float*)cpu.data)[i] = 0.1f;

    tensor_alloc(&gpu, 1, shape2, DTYPE_FLOAT32, DEVICE_WEBGPU);
    tensor_copy(&gpu, &cpu);

    s = g_backend->tensor_sum(&gpu);
    CHECK_CLOSE(s, 1000.0f, 1.0f, "sum of 10000 * 0.1");

    float m = g_backend->tensor_mean(&gpu);
    CHECK_CLOSE(m, 0.1f, 1e-4, "mean of 10000 * 0.1");

    tensor_free(&cpu);
    tensor_free(&gpu);
}

static void test_h2d_d2h_roundtrip(void) {
    printf("\n=== H2D/D2H Round-trip ===\n");

    int shape[] = {256};
    tensor_t cpu_src, cpu_dst, gpu;
    tensor_alloc_cpu_f32(&cpu_src, 1, shape);
    tensor_alloc_cpu_f32(&cpu_dst, 1, shape);
    for (int i = 0; i < 256; i++) ((float*)cpu_src.data)[i] = (float)i * 0.01f;

    tensor_alloc(&gpu, 1, shape, DTYPE_FLOAT32, DEVICE_WEBGPU);

    /* CPU -> GPU -> CPU */
    tensor_copy(&gpu, &cpu_src);
    tensor_copy(&cpu_dst, &gpu);

    float *src = (float *)cpu_src.data;
    float *dst = (float *)cpu_dst.data;

    int match = 1;
    for (int i = 0; i < 256; i++) {
        if (fabsf(src[i] - dst[i]) > 1e-6f) { match = 0; break; }
    }
    CHECK(match, "round-trip 256 floats exact");

    tensor_free(&cpu_src);
    tensor_free(&cpu_dst);
    tensor_free(&gpu);
}

int main(void) {
    printf("=== WebGPU Backend Basic Tests ===\n");

#ifndef CFIREANTS_HAS_WEBGPU
    printf("WebGPU not enabled, skipping\n");
    return 0;
#else
    cfireants_init_cpu();
    if (cfireants_init_webgpu() != 0) {
        printf("Failed to initialize WebGPU backend\n");
        return 1;
    }

    test_alloc_free();
    test_fill();
    test_scale();
    test_axpy();
    test_sum();
    test_h2d_d2h_roundtrip();

    printf("\n========================================\n");
    printf("%d passed, %d failed\n", n_pass, n_fail);
    printf("========================================\n");

    return n_fail > 0 ? 1 : 0;
#endif
}
