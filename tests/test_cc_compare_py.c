/*
 * test_cc_compare_py.c - Compare CC loss with Python on identical data.
 * Reads test/cc_test_pred.raw and test/cc_test_target.raw,
 * computes CC loss, prints result for comparison.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cfireants/tensor.h"
#include "cfireants/losses.h"

#ifdef CFIREANTS_HAS_CUDA
#include <cuda_runtime.h>
extern void cuda_fused_cc_loss(
    const float *pred, const float *target,
    float *grad_pred, float *grad_target_out,
    int D, int H, int W, int ks,
    float *h_loss_out,
    float *interm, float *scratch);
extern void cuda_cc_loss_3d(
    const float *pred, const float *target,
    float *grad_pred,
    int D, int H, int W, int ks,
    float *h_loss_out);
#endif

int main(void) {
    /* Read metadata */
    FILE *f = fopen("test/cc_test_meta.txt", "r");
    if (!f) { fprintf(stderr, "No test data. Run validate/compare_cc_loss.py first.\n"); return 1; }
    int D, H, W, ks;
    fscanf(f, "%d %d %d %d", &D, &H, &W, &ks);
    fclose(f);
    int n = D * H * W;
    printf("Grid: %dx%dx%d, kernel=%d\n", D, H, W, ks);

    /* Read data */
    float *pred = (float *)malloc(n * sizeof(float));
    float *target = (float *)malloc(n * sizeof(float));
    f = fopen("test/cc_test_pred.raw", "rb");
    fread(pred, sizeof(float), n, f); fclose(f);
    f = fopen("test/cc_test_target.raw", "rb");
    fread(target, sizeof(float), n, f); fclose(f);

    printf("Pred range: [%.4f, %.4f]\n", pred[0], pred[n-1]);
    printf("Target range: [%.4f, %.4f]\n", target[0], target[n-1]);

    /* CPU CC loss (via tensor interface) */
    {
        int shape[5] = {1, 1, D, H, W};
        tensor_t t_pred = {0}, t_target = {0};
        tensor_alloc(&t_pred, 5, shape, DTYPE_FLOAT32, DEVICE_CPU);
        tensor_alloc(&t_target, 5, shape, DTYPE_FLOAT32, DEVICE_CPU);
        memcpy(tensor_data_f32(&t_pred), pred, n * sizeof(float));
        memcpy(tensor_data_f32(&t_target), target, n * sizeof(float));

        float loss;
        tensor_t grad = {0};
        cpu_cc_loss_3d(&t_pred, &t_target, ks, &loss, &grad);
        printf("\nCPU cc_loss_3d: %.8f\n", loss);

        float *g = tensor_data_f32(&grad);
        float gmin = g[0], gmax = g[0], gsum = 0;
        for (int i = 0; i < n; i++) {
            if (g[i] < gmin) gmin = g[i];
            if (g[i] > gmax) gmax = g[i];
            gsum += g[i];
        }
        printf("CPU grad_pred: min=%.8f max=%.8f mean=%.8f\n", gmin, gmax, gsum/n);

        tensor_free(&t_pred); tensor_free(&t_target); tensor_free(&grad);
    }

    /* CPU fused CC loss */
    {
        int shape[5] = {1, 1, D, H, W};
        tensor_t t_pred = {0}, t_target = {0};
        tensor_alloc(&t_pred, 5, shape, DTYPE_FLOAT32, DEVICE_CPU);
        tensor_alloc(&t_target, 5, shape, DTYPE_FLOAT32, DEVICE_CPU);
        memcpy(tensor_data_f32(&t_pred), pred, n * sizeof(float));
        memcpy(tensor_data_f32(&t_target), target, n * sizeof(float));

        float loss;
        tensor_t gp = {0}, gt = {0};
        cpu_fused_cc_loss(&t_pred, &t_target, ks, &loss, &gp, &gt);
        printf("\nCPU fused_cc_loss: %.8f\n", loss);

        float *g = tensor_data_f32(&gp);
        float gmin = g[0], gmax = g[0], gsum = 0;
        for (int i = 0; i < n; i++) {
            if (g[i] < gmin) gmin = g[i];
            if (g[i] > gmax) gmax = g[i];
            gsum += g[i];
        }
        printf("CPU fused grad_pred: min=%.8f max=%.8f mean=%.8f\n", gmin, gmax, gsum/n);

        tensor_free(&t_pred); tensor_free(&t_target); tensor_free(&gp); tensor_free(&gt);
    }

#ifdef CFIREANTS_HAS_CUDA
    /* GPU fused CC loss */
    {
        float *d_pred, *d_target, *d_grad_pred, *d_grad_target;
        float *d_interm, *d_scratch;
        cudaMalloc(&d_pred, n * sizeof(float));
        cudaMalloc(&d_target, n * sizeof(float));
        cudaMalloc(&d_grad_pred, n * sizeof(float));
        cudaMalloc(&d_grad_target, n * sizeof(float));
        cudaMalloc(&d_interm, 5L * n * sizeof(float));
        cudaMalloc(&d_scratch, n * sizeof(float));
        cudaMemcpy(d_pred, pred, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_target, target, n * sizeof(float), cudaMemcpyHostToDevice);

        float loss;
        cuda_fused_cc_loss(d_pred, d_target, d_grad_pred, d_grad_target,
                           D, H, W, ks, &loss, d_interm, d_scratch);
        printf("\nGPU fused_cc_loss: %.8f\n", loss);

        float *h_grad = (float *)malloc(n * sizeof(float));
        cudaMemcpy(h_grad, d_grad_pred, n * sizeof(float), cudaMemcpyDeviceToHost);
        float gmin = h_grad[0], gmax = h_grad[0], gsum = 0;
        for (int i = 0; i < n; i++) {
            if (h_grad[i] < gmin) gmin = h_grad[i];
            if (h_grad[i] > gmax) gmax = h_grad[i];
            gsum += h_grad[i];
        }
        printf("GPU fused grad_pred: min=%.8f max=%.8f mean=%.8f\n", gmin, gmax, gsum/n);

        free(h_grad);
        cudaFree(d_pred); cudaFree(d_target); cudaFree(d_grad_pred);
        cudaFree(d_grad_target); cudaFree(d_interm); cudaFree(d_scratch);
    }
#endif

    free(pred); free(target);
    return 0;
}
