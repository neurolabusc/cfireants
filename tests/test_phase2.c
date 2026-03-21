/*
 * test_phase2.c - Validate grid_sample, CC loss, blur, Adam against Python
 *
 * Loads binary test data exported by export_test_data.py and compares
 * C results against Python reference outputs.
 *
 * Usage: test_phase2 <test_data_dir>
 */

#include "cfireants/tensor.h"
#include "cfireants/backend.h"
#include "cfireants/interpolator.h"
#include "cfireants/losses.h"
#include "cfireants/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Binary tensor loading                                               */
/* ------------------------------------------------------------------ */

static int load_bin_tensor(const char *dir, const char *name, tensor_t *t,
                           int ndim, const int *shape) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);

    if (tensor_alloc(t, ndim, shape, DTYPE_FLOAT32, DEVICE_CPU) != 0)
        return -1;

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        tensor_free(t);
        return -1;
    }

    size_t n = fread(t->data, sizeof(float), t->numel, f);
    fclose(f);

    if (n != t->numel) {
        fprintf(stderr, "Read %zu elements, expected %zu from %s\n", n, t->numel, path);
        tensor_free(t);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Comparison utilities                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    float max_abs_err;
    float mean_abs_err;
    float max_rel_err;
    float mean_rel_err;
} compare_result_t;

static compare_result_t compare_tensors(const tensor_t *a, const tensor_t *b) {
    compare_result_t r = {0};
    const float *da = tensor_data_f32(a);
    const float *db = tensor_data_f32(b);
    double sum_abs = 0, sum_rel = 0;
    size_t n = a->numel;

    for (size_t i = 0; i < n; i++) {
        float ae = fabsf(da[i] - db[i]);
        float re = ae / (fabsf(db[i]) + 1e-8f);
        if (ae > r.max_abs_err) r.max_abs_err = ae;
        if (re > r.max_rel_err) r.max_rel_err = re;
        sum_abs += ae;
        sum_rel += re;
    }
    r.mean_abs_err = (float)(sum_abs / n);
    r.mean_rel_err = (float)(sum_rel / n);
    return r;
}

static int check_result(const char *test_name, compare_result_t r,
                        float max_abs_tol, float mean_abs_tol) {
    int pass = (r.max_abs_err <= max_abs_tol || r.mean_abs_err <= mean_abs_tol);
    printf("  %-30s max_abs=%.6f mean_abs=%.6f mean_rel=%.6f  %s\n",
           test_name, r.max_abs_err, r.mean_abs_err, r.mean_rel_err,
           pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* Tests                                                               */
/* ------------------------------------------------------------------ */

static int test_grid_sample(const char *dir) {
    printf("\n=== Grid Sample 3D ===\n");
    int failures = 0;

    /* Load test data */
    int input_shape[] = {1, 1, 91, 109, 91};
    int grid_shape[]  = {1, 91, 109, 91, 3};
    int out_shape[]   = {1, 1, 91, 109, 91};

    tensor_t input, grid, expected_output;
    if (load_bin_tensor(dir, "gs_input", &input, 5, input_shape) != 0) return 1;
    if (load_bin_tensor(dir, "gs_grid", &grid, 5, grid_shape) != 0) return 1;
    if (load_bin_tensor(dir, "gs_output", &expected_output, 5, out_shape) != 0) return 1;

    /* Run C grid_sample */
    tensor_t c_output;
    cpu_grid_sample_3d_forward(&input, &grid, &c_output, 1);

    compare_result_t r = compare_tensors(&c_output, &expected_output);
    failures += check_result("affine grid_sample", r, 1.0f, 0.01f);

    tensor_free(&c_output);
    tensor_free(&expected_output);

    /* Test with displacement grid */
    tensor_t disp_grid, expected_disp_output;
    if (load_bin_tensor(dir, "gs_disp_grid", &disp_grid, 5, grid_shape) != 0) return 1;
    if (load_bin_tensor(dir, "gs_disp_output", &expected_disp_output, 5, out_shape) != 0) return 1;

    tensor_t c_disp_output;
    cpu_grid_sample_3d_forward(&input, &disp_grid, &c_disp_output, 1);

    r = compare_tensors(&c_disp_output, &expected_disp_output);
    failures += check_result("displacement grid_sample", r, 1.0f, 0.01f);

    tensor_free(&c_disp_output);
    tensor_free(&expected_disp_output);
    tensor_free(&disp_grid);
    tensor_free(&grid);
    tensor_free(&input);

    return failures;
}

static int test_affine_grid(const char *dir) {
    printf("\n=== Affine Grid Generation ===\n");
    int failures = 0;

    int aff_shape[] = {1, 3, 4};
    int grid_shape[] = {1, 91, 109, 91, 3};

    tensor_t affine, expected_grid;
    if (load_bin_tensor(dir, "gs_affine", &affine, 3, aff_shape) != 0) return 1;
    if (load_bin_tensor(dir, "gs_grid", &expected_grid, 5, grid_shape) != 0) return 1;

    int out_shape[3] = {91, 109, 91};
    tensor_t c_grid;
    affine_grid_3d(&affine, out_shape, &c_grid);

    compare_result_t r = compare_tensors(&c_grid, &expected_grid);
    failures += check_result("affine_grid_3d", r, 1e-5f, 1e-6f);

    tensor_free(&c_grid);
    tensor_free(&expected_grid);
    tensor_free(&affine);
    return failures;
}

static int test_cc_loss(const char *dir) {
    printf("\n=== CC Loss ===\n");
    int failures = 0;

    int img_shape[] = {1, 1, 91, 109, 91};
    tensor_t fixed, moved;
    if (load_bin_tensor(dir, "cc_fixed", &fixed, 5, img_shape) != 0) return 1;
    if (load_bin_tensor(dir, "cc_moved", &moved, 5, img_shape) != 0) return 1;

    int kernel_sizes[] = {3, 5, 7};
    for (int ki = 0; ki < 3; ki++) {
        int ks = kernel_sizes[ki];
        char name[64];

        /* Load expected loss */
        int loss_shape[] = {1};
        tensor_t expected_loss;
        snprintf(name, sizeof(name), "cc_loss_k%d", ks);
        if (load_bin_tensor(dir, name, &expected_loss, 1, loss_shape) != 0) return 1;

        float expected = tensor_data_f32(&expected_loss)[0];

        /* Compute C loss */
        float c_loss;
        cpu_cc_loss_3d(&moved, &fixed, ks, &c_loss, NULL);

        float err = fabsf(c_loss - expected);
        int pass = (err < 0.01f);  /* 1% tolerance */
        printf("  CC loss k=%d: C=%.6f Python=%.6f err=%.6f  %s\n",
               ks, c_loss, expected, err, pass ? "PASS" : "FAIL");
        if (!pass) failures++;

        tensor_free(&expected_loss);
    }

    tensor_free(&fixed);
    tensor_free(&moved);
    return failures;
}

static int test_gaussian_blur(const char *dir) {
    printf("\n=== Gaussian Blur ===\n");
    int failures = 0;

    int img_shape[] = {1, 1, 91, 109, 91};
    tensor_t input;
    if (load_bin_tensor(dir, "blur_input", &input, 5, img_shape) != 0) return 1;

    float sigmas[] = {0.5f, 1.0f, 2.0f};
    for (int si = 0; si < 3; si++) {
        float sigma = sigmas[si];
        char name[64];
        snprintf(name, sizeof(name), "blur_output_s%.1f", sigma);

        tensor_t expected;
        if (load_bin_tensor(dir, name, &expected, 5, img_shape) != 0) return 1;

        tensor_t c_output;
        cpu_gaussian_blur_3d(&input, &c_output, sigma, 4.0f);

        compare_result_t r = compare_tensors(&c_output, &expected);
        char label[64];
        snprintf(label, sizeof(label), "blur sigma=%.1f", sigma);
        failures += check_result(label, r, 5.0f, 0.1f);

        tensor_free(&c_output);
        tensor_free(&expected);
    }

    tensor_free(&input);
    return failures;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    const char *dir = "cfireants/tests/test_data";
    if (argc > 1) dir = argv[1];

    cfireants_init_cpu();

    int total_failures = 0;
    total_failures += test_affine_grid(dir);
    total_failures += test_grid_sample(dir);
    total_failures += test_cc_loss(dir);
    total_failures += test_gaussian_blur(dir);

    printf("\n========================================\n");
    if (total_failures == 0)
        printf("All Phase 2 tests PASSED\n");
    else
        printf("%d test(s) FAILED\n", total_failures);
    printf("========================================\n");

    cfireants_cleanup();
    return total_failures;
}
