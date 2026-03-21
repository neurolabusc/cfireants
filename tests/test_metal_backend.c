/*
 * test_metal_backend.c - Basic Metal backend tests
 *
 * Tests tensor allocation (unified memory), element-wise ops, and reductions.
 */

#include "cfireants/backend.h"
#include "cfireants/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef CFIREANTS_HAS_METAL
extern int cfireants_init_metal(void);
#endif

static int passed = 0, failed = 0;

static void check(const char *name, int cond) {
    if (cond) { printf("  PASS: %s\n", name); passed++; }
    else      { printf("  FAIL: %s\n", name); failed++; }
}

static void checkf(const char *name, float got, float expected, float tol) {
    float diff = fabsf(got - expected);
    if (diff <= tol) {
        printf("  PASS: %s (%.6f == %.6f)\n", name, got, expected);
        passed++;
    } else {
        printf("  FAIL: %s (%.6f != %.6f, diff=%.6e)\n", name, got, expected, diff);
        failed++;
    }
}

int main(void) {
    printf("=== Metal Backend Basic Tests ===\n\n");

    cfireants_init_cpu();

#ifndef CFIREANTS_HAS_METAL
    printf("Metal not enabled, skipping\n");
    return 0;
#else
    if (cfireants_init_metal() != 0) {
        printf("Failed to initialize Metal\n");
        return 1;
    }

    /* --- Tensor Alloc/Free --- */
    printf("=== Tensor Alloc/Free ===\n");
    {
        int shape[] = {1, 1, 10, 10, 10};
        tensor_t t;
        int rc = tensor_alloc(&t, 5, shape, DTYPE_FLOAT32, DEVICE_METAL);
        check("alloc Metal tensor", rc == 0);
        check("numel correct", t.numel == 1000);
        check("device correct", t.device == DEVICE_METAL);
        check("data pointer non-null", t.data != NULL);

        /* Unified memory: we can read/write data directly from CPU */
        float *fdata = (float *)t.data;
        fdata[0] = 42.0f;
        check("direct CPU write to unified memory", fdata[0] == 42.0f);

        tensor_free(&t);
        check("data NULL after free", t.data == NULL);
    }

    /* --- Tensor Fill --- */
    printf("\n=== Tensor Fill ===\n");
    {
        int shape[] = {1, 1, 10, 10, 10};
        tensor_t t;
        tensor_alloc(&t, 5, shape, DTYPE_FLOAT32, DEVICE_METAL);
        g_backend->tensor_fill(&t, 3.14f);
        float *fdata = (float *)t.data;
        checkf("fill[0]", fdata[0], 3.14f, 1e-6f);
        checkf("fill[500]", fdata[500], 3.14f, 1e-6f);
        checkf("fill[999]", fdata[999], 3.14f, 1e-6f);
        tensor_free(&t);
    }

    /* --- Tensor Scale --- */
    printf("\n=== Tensor Scale ===\n");
    {
        int shape[] = {1, 1, 1, 1, 100};
        tensor_t t;
        tensor_alloc(&t, 5, shape, DTYPE_FLOAT32, DEVICE_METAL);
        float *fdata = (float *)t.data;
        for (int i = 0; i < 100; i++) fdata[i] = (float)i;
        g_backend->tensor_scale(&t, 2.5f);
        checkf("scale[0]", fdata[0], 0.0f, 1e-6f);
        checkf("scale[1]", fdata[1], 2.5f, 1e-6f);
        checkf("scale[99]", fdata[99], 247.5f, 1e-4f);
        tensor_free(&t);
    }

    /* --- Tensor AXPY --- */
    printf("\n=== Tensor AXPY ===\n");
    {
        int shape[] = {1, 1, 1, 1, 100};
        tensor_t x, y;
        tensor_alloc(&x, 5, shape, DTYPE_FLOAT32, DEVICE_METAL);
        tensor_alloc(&y, 5, shape, DTYPE_FLOAT32, DEVICE_METAL);
        float *xd = (float *)x.data;
        float *yd = (float *)y.data;
        for (int i = 0; i < 100; i++) { xd[i] = 1.0f; yd[i] = (float)i; }
        g_backend->tensor_axpy(&y, 10.0f, &x);
        checkf("axpy[0] = 0 + 10*1", yd[0], 10.0f, 1e-5f);
        checkf("axpy[50] = 50 + 10*1", yd[50], 60.0f, 1e-4f);
        checkf("axpy[99] = 99 + 10*1", yd[99], 109.0f, 1e-4f);
        tensor_free(&x);
        tensor_free(&y);
    }

    /* --- Tensor Sum --- */
    printf("\n=== Tensor Sum ===\n");
    {
        int shape[] = {1, 1, 1, 1, 1000};
        tensor_t t;
        tensor_alloc(&t, 5, shape, DTYPE_FLOAT32, DEVICE_METAL);
        float *fdata = (float *)t.data;
        for (int i = 0; i < 1000; i++) fdata[i] = 1.0f;
        float sum = g_backend->tensor_sum(&t);
        checkf("sum of 1000 ones", sum, 1000.0f, 1e-2f);

        /* Larger test */
        int shape2[] = {1, 1, 1, 1, 10000};
        tensor_t t2;
        tensor_alloc(&t2, 5, shape2, DTYPE_FLOAT32, DEVICE_METAL);
        float *fd2 = (float *)t2.data;
        for (int i = 0; i < 10000; i++) fd2[i] = 0.1f;
        sum = g_backend->tensor_sum(&t2);
        checkf("sum of 10000 * 0.1", sum, 1000.0f, 1.0f);

        float mean = g_backend->tensor_mean(&t2);
        checkf("mean of 10000 * 0.1", mean, 0.1f, 1e-4f);

        tensor_free(&t);
        tensor_free(&t2);
    }

    /* --- H2D/D2H Round-trip (unified memory) --- */
    printf("\n=== H2D/D2H Round-trip ===\n");
    {
        int shape[] = {1, 1, 1, 1, 256};
        tensor_t cpu_src, metal_dst, cpu_back;
        tensor_alloc(&cpu_src, 5, shape, DTYPE_FLOAT32, DEVICE_CPU);
        tensor_alloc(&metal_dst, 5, shape, DTYPE_FLOAT32, DEVICE_METAL);
        tensor_alloc(&cpu_back, 5, shape, DTYPE_FLOAT32, DEVICE_CPU);

        float *src = (float *)cpu_src.data;
        for (int i = 0; i < 256; i++) src[i] = (float)i * 0.5f;

        tensor_copy(&metal_dst, &cpu_src);  /* CPU -> Metal */
        tensor_copy(&cpu_back, &metal_dst); /* Metal -> CPU */

        float *back = (float *)cpu_back.data;
        int exact = 1;
        for (int i = 0; i < 256; i++) {
            if (back[i] != src[i]) { exact = 0; break; }
        }
        check("round-trip 256 floats exact", exact);

        tensor_free(&cpu_src);
        tensor_free(&metal_dst);
        tensor_free(&cpu_back);
    }

    printf("\n========================================\n");
    printf("%d passed, %d failed\n", passed, failed);
    printf("========================================\n");
    return failed > 0 ? 1 : 0;
#endif
}
