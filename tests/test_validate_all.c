/*
 * test_validate_all.c - Run full validation pipeline on all three datasets
 *
 * Matches validate/run_validation.py: Moments → Rigid → Affine → Greedy
 * Reports global NCC (before/after), local NCC loss, time, peak GPU memory.
 *
 * Usage: test_validate_all [--dataset small|medium|large]
 */

#include "cfireants/tensor.h"
#include "cfireants/image.h"
#include "cfireants/backend.h"
#include "cfireants/registration.h"
#include "cfireants/losses.h"
#include "cfireants/interpolator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef CFIREANTS_HAS_CUDA
#include <cuda_runtime.h>
extern int cfireants_init_cuda(void);
#endif

static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Peak RSS in MB (Linux) */
static double get_peak_rss_mb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    double peak_kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmHWM:", 6) == 0) {
            sscanf(line + 6, "%lf", &peak_kb);
            break;
        }
    }
    fclose(f);
    return peak_kb / 1024.0;
}

#ifdef CFIREANTS_HAS_CUDA
static double get_peak_gpu_mb(void) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    /* Peak = total - current_free is an approximation.
     * For accurate peak, use cudaMemPoolGetAttribute or track manually. */
    return (double)(total_mem - free_mem) / (1024.0 * 1024.0);
}
#endif

/* Global NCC: mean-subtracted Pearson correlation (matching Python compute_ncc) */
static float compute_global_ncc(const float *a, const float *b, size_t n) {
    double sum_a = 0, sum_b = 0;
    for (size_t i = 0; i < n; i++) { sum_a += a[i]; sum_b += b[i]; }
    double mean_a = sum_a / n, mean_b = sum_b / n;
    double cov = 0, var_a = 0, var_b = 0;
    for (size_t i = 0; i < n; i++) {
        double da = a[i] - mean_a, db = b[i] - mean_b;
        cov += da * db; var_a += da * da; var_b += db * db;
    }
    double denom = sqrt(var_a * var_b);
    return (denom < 1e-12) ? 0.0f : (float)(cov / denom);
}

/* Resample moving into fixed space via identity (for NCC Before) */
static void resample_identity(const image_t *fixed, const image_t *moving,
                               tensor_t *resampled) {
    float identity[3][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0}};
    apply_affine_transform(fixed, moving, identity, resampled);
}

typedef struct {
    const char *name;
    const char *description;
    const char *fixed_path;
    const char *moving_path;
    /* Rigid/Affine params */
    int rigid_n_scales;
    int rigid_scales[4];
    int rigid_iters[4];
    int rigid_loss_type;
    float rigid_lr;
    int affine_n_scales;
    int affine_scales[4];
    int affine_iters[4];
    int affine_loss_type;
    float affine_lr;
    /* Greedy params */
    int greedy_scales[3];
    int greedy_iters[3];
    int greedy_cc_ks;
    float greedy_lr;
    float greedy_smooth_warp;
    float greedy_smooth_grad;
} dataset_config_t;

static dataset_config_t DATASETS[] = {
    {
        .name = "small",
        .description = "2mm full-head MNI to subject (includes scalp)",
        .fixed_path = "validate/small/MNI152_T1_2mm.nii.gz",
        .moving_path = "validate/small/T1_head_2mm.nii.gz",
        .rigid_n_scales = 3, .rigid_scales = {4, 2, 1}, .rigid_iters = {200, 100, 50},
        .rigid_loss_type = LOSS_MI, .rigid_lr = 3e-3f,
        .affine_n_scales = 3, .affine_scales = {4, 2, 1}, .affine_iters = {200, 100, 50},
        .affine_loss_type = LOSS_MI, .affine_lr = 1e-3f,
        .greedy_scales = {4, 2, 1}, .greedy_iters = {200, 100, 50},
        .greedy_cc_ks = 5, .greedy_lr = 0.1f,
        .greedy_smooth_warp = 0.5f, .greedy_smooth_grad = 1.0f,
    },
    {
        .name = "medium",
        .description = "1mm brain-extracted MNI to subject",
        .fixed_path = "validate/medium/MNI152_T1_1mm_brain.nii.gz",
        .moving_path = "validate/medium/t1_brain.nii.gz",
        .rigid_n_scales = 3, .rigid_scales = {4, 2, 1}, .rigid_iters = {200, 100, 50},
        .rigid_loss_type = LOSS_CC, .rigid_lr = 3e-3f,
        .affine_n_scales = 3, .affine_scales = {4, 2, 1}, .affine_iters = {200, 100, 50},
        .affine_loss_type = LOSS_CC, .affine_lr = 1e-3f,
        .greedy_scales = {4, 2, 1}, .greedy_iters = {200, 100, 50},
        .greedy_cc_ks = 5, .greedy_lr = 0.1f,
        .greedy_smooth_warp = 0.5f, .greedy_smooth_grad = 1.0f,
    },
    {
        .name = "large",
        .description = "1mm full-head MNI to subject (0.88mm, includes scalp)",
        .fixed_path = "validate/large/MNI152_T1_1mm.nii.gz",
        .moving_path = "validate/large/chris_t1.nii.gz",
        .rigid_n_scales = 4, .rigid_scales = {8, 4, 2, 1}, .rigid_iters = {200, 200, 100, 50},
        .rigid_loss_type = LOSS_MI, .rigid_lr = 3e-3f,
        .affine_n_scales = 4, .affine_scales = {8, 4, 2, 1}, .affine_iters = {200, 200, 100, 50},
        .affine_loss_type = LOSS_MI, .affine_lr = 1e-3f,
        .greedy_scales = {4, 2, 1}, .greedy_iters = {200, 100, 50},
        .greedy_cc_ks = 5, .greedy_lr = 0.1f,
        .greedy_smooth_warp = 0.5f, .greedy_smooth_grad = 1.0f,
    },
};

static void run_dataset(const dataset_config_t *ds) {
    printf("\n============================================================\n");
    printf("Dataset: %s — %s\n", ds->name, ds->description);
    printf("============================================================\n");

    image_t fixed, moving;
    image_load(&fixed, ds->fixed_path, DEVICE_CPU);
    image_load(&moving, ds->moving_path, DEVICE_CPU);

    int fD = fixed.data.shape[2], fH = fixed.data.shape[3], fW = fixed.data.shape[4];
    size_t fN = (size_t)fD * fH * fW;

    /* NCC Before (resample moving into fixed space via identity) */
    tensor_t resampled;
    resample_identity(&fixed, &moving, &resampled);
    float ncc_before = compute_global_ncc(
        tensor_data_f32(&fixed.data), tensor_data_f32(&resampled), fN);
    tensor_free(&resampled);
    printf("  NCC before: %.4f\n", ncc_before);

    double t_total = get_time();

    /* Moments (CPU) */
    double t1 = get_time();
    moments_opts_t mom_opts = moments_opts_default();
    moments_result_t mom_result;
    moments_register(&fixed, &moving, mom_opts, &mom_result);
    double t_moments = get_time() - t1;

#ifdef CFIREANTS_HAS_CUDA
    /* Rigid (GPU) */
    t1 = get_time();
    rigid_opts_t ropts = {
        .n_scales = ds->rigid_n_scales, .scales = ds->rigid_scales,
        .iterations = ds->rigid_iters, .loss_type = ds->rigid_loss_type,
        .cc_kernel_size = 5, .mi_num_bins = 32, .lr = ds->rigid_lr,
        .tolerance = 1e-6f, .max_tolerance_iters = 10
    };
    rigid_result_t rigid_result;
    rigid_register_gpu(&fixed, &moving, &mom_result, ropts, &rigid_result);
    double t_rigid = get_time() - t1;

    /* Affine (GPU) */
    t1 = get_time();
    affine_opts_t aopts = {
        .n_scales = ds->affine_n_scales, .scales = ds->affine_scales,
        .iterations = ds->affine_iters, .loss_type = ds->affine_loss_type,
        .cc_kernel_size = 5, .mi_num_bins = 32, .lr = ds->affine_lr,
        .tolerance = 1e-6f, .max_tolerance_iters = 10
    };
    affine_result_t affine_result;
    affine_register_gpu(&fixed, &moving, rigid_result.rigid_mat, aopts, &affine_result);
    double t_affine = get_time() - t1;

    /* Build 4x4 affine for deformable init */
    float aff44[4][4] = {{0}};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            aff44[i][j] = affine_result.affine_mat[i][j];
    aff44[3][3] = 1.0f;

    /* Deformable stage */
    t1 = get_time();
    float local_ncc_loss = 0;
    float ncc_after = 0;
    int have_greedy = 0, have_syn = 0;
    greedy_result_t greedy_result; memset(&greedy_result, 0, sizeof(greedy_result));
    syn_result_t syn_result; memset(&syn_result, 0, sizeof(syn_result));

#ifdef USE_GREEDY
    {
        greedy_opts_t gopts = {
            .n_scales = 3, .scales = ds->greedy_scales, .iterations = ds->greedy_iters,
            .cc_kernel_size = ds->greedy_cc_ks, .lr = ds->greedy_lr,
            .smooth_warp_sigma = ds->greedy_smooth_warp,
            .smooth_grad_sigma = ds->greedy_smooth_grad,
            .tolerance = 1e-6f, .max_tolerance_iters = 10
        };
        greedy_register_gpu(&fixed, &moving, aff44, gopts, &greedy_result);
        local_ncc_loss = greedy_result.ncc_loss;
        ncc_after = compute_global_ncc(
            tensor_data_f32(&fixed.data), tensor_data_f32(&greedy_result.moved), fN);
        have_greedy = 1;
    }
#else
    {
        syn_opts_t sopts = {
            .n_scales = 3, .scales = ds->greedy_scales, .iterations = ds->greedy_iters,
            .cc_kernel_size = ds->greedy_cc_ks, .lr = ds->greedy_lr,
            .smooth_warp_sigma = ds->greedy_smooth_warp,
            .smooth_grad_sigma = ds->greedy_smooth_grad,
            .tolerance = 1e-6f, .max_tolerance_iters = 10
        };
        syn_register_gpu(&fixed, &moving, aff44, sopts, &syn_result);
        local_ncc_loss = syn_result.ncc_loss;
        ncc_after = compute_global_ncc(
            tensor_data_f32(&fixed.data), tensor_data_f32(&syn_result.moved), fN);
        have_syn = 1;
    }
#endif
    double t_syn = get_time() - t1;
#else
    double t_rigid = 0, t_affine = 0, t_syn = 0;
    float local_ncc_loss = 0, ncc_after = 0;
    int have_greedy = 0, have_syn = 0;
    greedy_result_t greedy_result; memset(&greedy_result, 0, sizeof(greedy_result));
    syn_result_t syn_result; memset(&syn_result, 0, sizeof(syn_result));
#endif

    double t_total_elapsed = get_time() - t_total;

    /* Memory stats */
    double peak_cpu_mb = get_peak_rss_mb();
    double peak_gpu_mb = 0;
#ifdef CFIREANTS_HAS_CUDA
    peak_gpu_mb = get_peak_gpu_mb();
#endif

    printf("\n  Results:\n");
    printf("    NCC Before:     %.4f\n", ncc_before);
    printf("    NCC After:      %.4f\n", ncc_after);
    printf("    Local NCC Loss: %.4f\n", local_ncc_loss);
    printf("    Time:           %.1fs\n", t_total_elapsed);
    printf("    Moments:        %.1fs\n", t_moments);
    printf("    Rigid:          %.1fs\n", t_rigid);
    printf("    Affine:         %.1fs\n", t_affine);
    printf("    SyN:            %.1fs\n", t_syn);
    printf("    Peak CPU RAM:   %.0f MB\n", peak_cpu_mb);
    printf("    Peak GPU RAM:   %.0f MB\n", peak_gpu_mb);

    /* CSV-style output for easy parsing */
    printf("\n  CSV: %s,%.4f,%.4f,%.4f,%.1f,%.1f,%.1f,%.1f,%.1f,%.0f,%.0f\n",
           ds->name, ncc_before, ncc_after, local_ncc_loss,
           t_total_elapsed, t_moments, t_rigid, t_affine, t_syn,
           peak_cpu_mb, peak_gpu_mb);

    if (have_syn) {
        tensor_free(&syn_result.moved);
        tensor_free(&syn_result.fwd_disp);
        tensor_free(&syn_result.rev_disp);
    }
    if (have_greedy) {
        tensor_free(&greedy_result.moved);
        tensor_free(&greedy_result.disp);
    }
    image_free(&fixed);
    image_free(&moving);
}

int main(int argc, char **argv) {
    cfireants_init_cpu();
#ifdef CFIREANTS_HAS_CUDA
    cfireants_init_cuda();
#endif

    const char *only_dataset = NULL;
    if (argc > 2 && strcmp(argv[1], "--dataset") == 0)
        only_dataset = argv[2];

    int n_datasets = sizeof(DATASETS) / sizeof(DATASETS[0]);
    for (int i = 0; i < n_datasets; i++) {
        if (only_dataset && strcmp(only_dataset, DATASETS[i].name) != 0)
            continue;
        run_dataset(&DATASETS[i]);
    }

    cfireants_cleanup();
    return 0;
}
