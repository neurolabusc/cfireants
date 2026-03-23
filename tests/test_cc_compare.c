/*
 * test_cc_compare.c — Compare cpu_cc_loss_3d vs cpu_fused_cc_loss values.
 * Verifies both CC implementations agree on loss and gradient magnitude.
 */
#include "cfireants/tensor.h"
#include "cfireants/losses.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int cfireants_verbose = 2;

int main(void) {
    /* Create small test volume */
    int D = 16, H = 16, W = 16;
    int shape[5] = {1, 1, D, H, W};
    int n = D * H * W;

    tensor_t pred, target;
    tensor_alloc(&pred, 5, shape, DTYPE_FLOAT32, DEVICE_CPU);
    tensor_alloc(&target, 5, shape, DTYPE_FLOAT32, DEVICE_CPU);

    float *p = tensor_data_f32(&pred);
    float *t = tensor_data_f32(&target);

    /* Fill with brain-like data: smooth gradient + noise */
    srand(42);
    for (int i = 0; i < n; i++) {
        int z = i / (H*W), y = (i / W) % H, x = i % W;
        float base = sinf(z * 0.3f) * cosf(y * 0.2f) * sinf(x * 0.25f);
        p[i] = base + 0.1f * ((float)rand() / RAND_MAX - 0.5f);
        t[i] = base + 0.15f * ((float)rand() / RAND_MAX - 0.5f) + 0.05f;
    }

    int ks = 5;

    /* Test 1: Compare loss values */
    float loss_regular, loss_fused;
    tensor_t grad_regular, grad_fused_p, grad_fused_t;

    cpu_cc_loss_3d(&pred, &target, ks, &loss_regular, &grad_regular);
    cpu_fused_cc_loss(&pred, &target, ks, &loss_fused, &grad_fused_p, &grad_fused_t);

    printf("Loss comparison (kernel_size=%d, kv=%d):\n", ks, ks*ks*ks);
    printf("  cc_loss_3d:    %.8f\n", loss_regular);
    printf("  fused_cc_loss: %.8f\n", loss_fused);
    printf("  diff:          %.8f (%.4f%%)\n",
           loss_fused - loss_regular,
           100.0 * (loss_fused - loss_regular) / fabsf(loss_regular));

    /* Test 2: Compare gradient magnitudes */
    float *gr = tensor_data_f32(&grad_regular);
    float *gf = tensor_data_f32(&grad_fused_p);

    double sum_gr = 0, sum_gf = 0, max_gr = 0, max_gf = 0;
    double sum_diff = 0, max_diff = 0;
    for (int i = 0; i < n; i++) {
        sum_gr += gr[i] * gr[i];
        sum_gf += gf[i] * gf[i];
        if (fabsf(gr[i]) > max_gr) max_gr = fabsf(gr[i]);
        if (fabsf(gf[i]) > max_gf) max_gf = fabsf(gf[i]);
        float d = fabsf(gr[i] - gf[i]);
        sum_diff += d * d;
        if (d > max_diff) max_diff = d;
    }

    printf("\nGradient comparison (w.r.t. pred):\n");
    printf("  cc_loss_3d     RMS: %.8f  max: %.8f\n", sqrt(sum_gr/n), max_gr);
    printf("  fused_cc_loss  RMS: %.8f  max: %.8f\n", sqrt(sum_gf/n), max_gf);
    printf("  ratio (fused/regular): %.4f\n", sqrt(sum_gf/sum_gr));
    printf("  diff RMS: %.8f  max_diff: %.8f\n", sqrt(sum_diff/n), max_diff);

    /* Test 3: Check if ratio is ~1/kv */
    float expected_ratio = 1.0f / (ks * ks * ks);
    printf("\n  Expected ratio if kv scaling: 1/kv = %.6f\n", expected_ratio);
    printf("  Actual ratio:                       %.6f\n", sqrt(sum_gf/sum_gr));

    tensor_free(&pred); tensor_free(&target);
    tensor_free(&grad_regular); tensor_free(&grad_fused_p); tensor_free(&grad_fused_t);
    return 0;
}
