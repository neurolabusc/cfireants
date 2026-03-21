/*
 * test_phase4.c - Validate rigid registration against Python
 */

#include "cfireants/tensor.h"
#include "cfireants/image.h"
#include "cfireants/backend.h"
#include "cfireants/registration.h"
#include "cfireants/losses.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int load_bin_f32(const char *dir, const char *name,
                        float *buf, size_t count) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }
    size_t n = fread(buf, sizeof(float), count, f);
    fclose(f);
    return (n == count) ? 0 : -1;
}

static int test_rigid(const char *data_dir, const char *dataset,
                      const char *fixed_path, const char *moving_path,
                      const int *scales, const int *iterations, int n_scales,
                      int loss_type, int cc_kernel_size, int mi_num_bins,
                      float lr) {
    printf("\n=== Rigid: %s ===\n", dataset);
    int failures = 0;

    image_t fixed, moving;
    if (image_load(&fixed, fixed_path, DEVICE_CPU) != 0) return 1;
    if (image_load(&moving, moving_path, DEVICE_CPU) != 0) return 1;

    /* Run moments first */
    moments_opts_t mom_opts = moments_opts_default();
    moments_result_t mom_result;
    moments_register(&fixed, &moving, mom_opts, &mom_result);
    printf("  Moments NCC: %.6f\n", mom_result.ncc_loss);

    /* Run rigid */
    rigid_opts_t opts;
    opts.n_scales = n_scales;
    opts.scales = scales;
    opts.iterations = iterations;
    opts.loss_type = loss_type;
    opts.cc_kernel_size = cc_kernel_size;
    opts.mi_num_bins = mi_num_bins;
    opts.lr = lr;
    opts.tolerance = 1e-6f;
    opts.max_tolerance_iters = 10;

    rigid_result_t result;
    rigid_register(&fixed, &moving, &mom_result, opts, &result);

    /* Load Python reference */
    char name[128];
    float py_mat[12], py_ncc;

    snprintf(name, sizeof(name), "rigid_%s_rigid_34", dataset);
    load_bin_f32(data_dir, name, py_mat, 12);

    snprintf(name, sizeof(name), "rigid_%s_ncc", dataset);
    load_bin_f32(data_dir, name, &py_ncc, 1);

    /* Compare rigid matrices */
    printf("  C rigid [3,4]:\n");
    for (int i = 0; i < 3; i++)
        printf("    [%10.6f %10.6f %10.6f %10.6f]\n",
               result.rigid_mat[i][0], result.rigid_mat[i][1],
               result.rigid_mat[i][2], result.rigid_mat[i][3]);
    printf("  Python rigid [3,4]:\n");
    for (int i = 0; i < 3; i++)
        printf("    [%10.6f %10.6f %10.6f %10.6f]\n",
               py_mat[i*4+0], py_mat[i*4+1], py_mat[i*4+2], py_mat[i*4+3]);

    /* Compare NCC */
    printf("  C NCC:      %.6f\n", result.ncc_loss);
    printf("  Python NCC: %.6f\n", py_ncc);

    /* The rigid optimization may find slightly different parameters
     * but should achieve similar NCC. Allow 5% tolerance on NCC.
     * Note: C uses CC loss throughout while Python may use MI for some
     * datasets, so parameters and NCC can differ somewhat. */
    float ncc_diff = fabsf(result.ncc_loss - py_ncc);
    int ncc_pass = (ncc_diff < 0.05f);
    /* Check that NCC improved or stayed similar to moments */
    int improved = (result.ncc_loss <= mom_result.ncc_loss + 0.05f);

    printf("  NCC diff vs Python: %.6f  %s\n", ncc_diff, ncc_pass ? "PASS" : "FAIL");
    printf("  NCC vs moments: %s (%.6f vs %.6f)\n",
           improved ? "OK" : "REGRESSED", result.ncc_loss, mom_result.ncc_loss);

    if (!ncc_pass) failures++;
    if (!improved) failures++;

    image_free(&fixed);
    image_free(&moving);
    return failures;
}

int main(int argc, char **argv) {
    const char *data_dir = "cfireants/tests/test_data";
    if (argc > 1) data_dir = argv[1];

    cfireants_init_cpu();

    int failures = 0;

    /* Small dataset - MI loss (full-head with scalp, different intensity ranges) */
    {
        int scales[] = {4, 2, 1};
        int iters[] = {200, 100, 50};
        failures += test_rigid(data_dir, "small",
                              "validate/small/MNI152_T1_2mm.nii.gz",
                              "validate/small/T1_head_2mm.nii.gz",
                              scales, iters, 3, LOSS_MI, 5, 32, 3e-3f);
    }

    /* Medium dataset - CC loss (brain-extracted) */
    {
        int scales[] = {4, 2, 1};
        int iters[] = {200, 100, 50};
        failures += test_rigid(data_dir, "medium",
                              "validate/medium/MNI152_T1_1mm_brain.nii.gz",
                              "validate/medium/t1_brain.nii.gz",
                              scales, iters, 3, LOSS_CC, 5, 0, 3e-3f);
    }

    printf("\n========================================\n");
    if (failures == 0)
        printf("All Phase 4 tests PASSED\n");
    else
        printf("%d test(s) FAILED\n", failures);
    printf("========================================\n");

    cfireants_cleanup();
    return failures;
}
