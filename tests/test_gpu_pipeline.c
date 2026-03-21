/*
 * test_gpu_pipeline.c - Full registration pipeline with GPU deformable
 *
 * Moments + Rigid + Affine on CPU, then Greedy deformable on GPU.
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
#include <time.h>

#ifdef CFIREANTS_HAS_CUDA
extern int cfireants_init_cuda(void);
#endif

static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int load_bin_f32(const char *dir, const char *name, float *buf, size_t count) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t n = fread(buf, sizeof(float), count, f);
    fclose(f);
    return (n == count) ? 0 : -1;
}

int main(int argc, char **argv) {
    const char *data_dir = "cfireants/tests/test_data";
    const char *fixed_path = "validate/small/MNI152_T1_2mm.nii.gz";
    const char *moving_path = "validate/small/T1_head_2mm.nii.gz";

    /* Use medium if --medium flag */
    int use_medium = (argc > 1 && strcmp(argv[1], "--medium") == 0);
    if (use_medium) {
        fixed_path = "validate/medium/MNI152_T1_1mm_brain.nii.gz";
        moving_path = "validate/medium/t1_brain.nii.gz";
    }

    cfireants_init_cpu();
#ifdef CFIREANTS_HAS_CUDA
    cfireants_init_cuda();
#endif

    printf("\n=== GPU Pipeline: %s ===\n", use_medium ? "medium" : "small");
    double t0 = get_time();

    image_t fixed, moving;
    image_load(&fixed, fixed_path, DEVICE_CPU);
    image_load(&moving, moving_path, DEVICE_CPU);

    /* Moments stays on CPU (fast, needs SVD) */
    double t1 = get_time();
    moments_opts_t mom_opts = moments_opts_default();
    moments_result_t mom_result;
    moments_register(&fixed, &moving, mom_opts, &mom_result);
    printf("  Moments: NCC=%.4f (%.1fs)\n", mom_result.ncc_loss, get_time()-t1);

    int scales3[] = {4, 2, 1};
    int iters_rigid[] = {200, 100, 50};
    int loss_type = use_medium ? LOSS_CC : LOSS_MI;

#ifdef CFIREANTS_HAS_CUDA
    /* GPU rigid */
    t1 = get_time();
    rigid_opts_t ropts = {.n_scales=3, .scales=scales3, .iterations=iters_rigid,
        .loss_type=loss_type, .cc_kernel_size=5, .mi_num_bins=32,
        .lr=3e-3f, .tolerance=1e-6f, .max_tolerance_iters=10};
    rigid_result_t rigid_result;
    rigid_register_gpu(&fixed, &moving, &mom_result, ropts, &rigid_result);
    printf("  Rigid:   NCC=%.4f (%.1fs) [GPU]\n", rigid_result.ncc_loss, get_time()-t1);

    /* GPU affine */
    t1 = get_time();
    affine_opts_t aopts = {.n_scales=3, .scales=scales3, .iterations=iters_rigid,
        .loss_type=loss_type, .cc_kernel_size=5, .mi_num_bins=32,
        .lr=1e-3f, .tolerance=1e-6f, .max_tolerance_iters=10};
    affine_result_t affine_result;
    affine_register_gpu(&fixed, &moving, rigid_result.rigid_mat, aopts, &affine_result);
    printf("  Affine:  NCC=%.4f (%.1fs) [GPU]\n", affine_result.ncc_loss, get_time()-t1);
#else
    /* CPU fallback */
    t1 = get_time();
    rigid_opts_t ropts = {.n_scales=3, .scales=scales3, .iterations=iters_rigid,
        .loss_type=loss_type, .cc_kernel_size=5, .mi_num_bins=32,
        .lr=3e-3f, .tolerance=1e-6f, .max_tolerance_iters=10};
    rigid_result_t rigid_result;
    rigid_register(&fixed, &moving, &mom_result, ropts, &rigid_result);
    printf("  Rigid:   NCC=%.4f (%.1fs) [CPU]\n", rigid_result.ncc_loss, get_time()-t1);

    t1 = get_time();
    affine_opts_t aopts = {.n_scales=3, .scales=scales3, .iterations=iters_rigid,
        .loss_type=loss_type, .cc_kernel_size=5, .mi_num_bins=32,
        .lr=1e-3f, .tolerance=1e-6f, .max_tolerance_iters=10};
    affine_result_t affine_result;
    affine_register(&fixed, &moving, rigid_result.rigid_mat, aopts, &affine_result);
    printf("  Affine:  NCC=%.4f (%.1fs) [CPU]\n", affine_result.ncc_loss, get_time()-t1);
#endif

    /* Build 4x4 affine for greedy init */
    float aff44[4][4];
    memset(aff44, 0, sizeof(aff44));
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            aff44[i][j] = affine_result.affine_mat[i][j];
    aff44[3][3] = 1.0f;

    /* --- GPU deformable --- */
#ifdef CFIREANTS_HAS_CUDA
    t1 = get_time();
    cfireants_init_cuda();

    int deform_iters[] = {200, 100, 50};
    greedy_opts_t gopts = {.n_scales=3, .scales=scales3, .iterations=deform_iters,
        .cc_kernel_size=5, .lr=0.1f,
        .smooth_warp_sigma=0.5f, .smooth_grad_sigma=1.0f,
        .tolerance=1e-6f, .max_tolerance_iters=10};
    greedy_result_t greedy_result;
    greedy_register_gpu(&fixed, &moving, aff44, gopts, &greedy_result);
    double gpu_time = get_time() - t1;
    printf("  Greedy GPU: NCC=%.4f (%.1fs)\n", greedy_result.ncc_loss, gpu_time);

    /* Load Python SyN reference */
    float py_ncc = 0;
    if (!use_medium)
        load_bin_f32(data_dir, "syn_small_ncc", &py_ncc, 1);

    printf("\n  Pipeline:  Moments(%.4f) -> Rigid(%.4f) -> Affine(%.4f) -> Greedy(%.4f)\n",
           mom_result.ncc_loss, rigid_result.ncc_loss,
           affine_result.ncc_loss, greedy_result.ncc_loss);
    if (!use_medium)
        printf("  Python SyN: %.4f\n", py_ncc);
    printf("  Total time: %.1fs\n", get_time() - t0);

    int pass = (greedy_result.ncc_loss < affine_result.ncc_loss);
    printf("\n  Greedy improved over affine: %s\n", pass ? "YES" : "NO");
#else
    printf("  CUDA not available, skipping GPU greedy\n");
    int pass = 0;
#endif

    printf("\n========================================\n");
    printf(pass ? "GPU pipeline test PASSED\n" : "GPU pipeline test FAILED\n");
    printf("========================================\n");

    image_free(&fixed);
    image_free(&moving);
    cfireants_cleanup();
    return pass ? 0 : 1;
}
