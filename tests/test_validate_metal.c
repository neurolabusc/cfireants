/*
 * test_validate_metal.c - Full pipeline validation for Metal backend
 *
 * Runs Moments + Rigid + Affine + SyN on validation datasets.
 * Parameters match test_validate_webgpu.c for cross-backend comparison.
 */

#include "cfireants/backend.h"
#include "cfireants/tensor.h"
#include "cfireants/image.h"
#include "cfireants/registration.h"
#include "cfireants/losses.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mach/mach.h>

#ifdef CFIREANTS_HAS_METAL
extern int cfireants_init_metal(void);
#endif

static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float compute_global_ncc(const float *a, const float *b, int n) {
    double sa=0,sb=0,sab=0,sa2=0,sb2=0;
    for(int i=0;i<n;i++){
        sa+=a[i]; sb+=b[i]; sab+=a[i]*b[i];
        sa2+=a[i]*a[i]; sb2+=b[i]*b[i];
    }
    double ma=sa/n, mb=sb/n;
    double cov=sab/n-ma*mb;
    double va=sa2/n-ma*ma, vb=sb2/n-mb*mb;
    if(va<1e-10||vb<1e-10) return 0;
    return (float)(cov/sqrt(va*vb));
}

typedef struct {
    const char *name, *desc, *fixed_path, *moving_path;
    int rigid_scales[8], affine_scales[8], syn_scales[8];
    int rigid_iters[8], affine_iters[8], syn_iters[8];
    int rigid_n, affine_n, syn_n;
    int rigid_loss, affine_loss;
} dataset_t;

int main(int argc, char **argv) {
    cfireants_init_cpu();
#ifndef CFIREANTS_HAS_METAL
    printf("Metal not enabled\n"); return 0;
#else
    if (cfireants_init_metal() != 0) { printf("Metal init failed\n"); return 1; }

    const char *only = NULL;
    int use_trilinear = 0;
    int use_greedy = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dataset") == 0 && i + 1 < argc) only = argv[++i];
        else if (strcmp(argv[i], "--trilinear") == 0) use_trilinear = 1;
        else if (strcmp(argv[i], "--greedy") == 0) use_greedy = 1;
    }
    int ds_mode = use_trilinear ? DOWNSAMPLE_TRILINEAR : DOWNSAMPLE_FFT;

    /* Datasets with per-dataset parameters matching WebGPU test */
    dataset_t datasets[] = {
        { "small", "2mm full-head MNI to subject (includes scalp)",
          "validate/small/MNI152_T1_2mm.nii.gz", "validate/small/T1_head_2mm.nii.gz",
          {4,2,1},{4,2,1},{4,2,1}, {200,100,50},{200,100,50},{200,100,50},
          3,3,3, LOSS_MI, LOSS_MI },
        { "medium", "1mm brain-extracted MNI to subject",
          "validate/medium/MNI152_T1_1mm_brain.nii.gz", "validate/medium/t1_brain.nii.gz",
          {4,2,1},{4,2,1},{4,2,1}, {200,100,50},{200,100,50},{200,100,50},
          3,3,3, LOSS_CC, LOSS_CC },
        { "large", "1mm full-head MNI to subject (0.88mm, includes scalp)",
          "validate/large/MNI152_T1_1mm.nii.gz", "validate/large/chris_t1.nii.gz",
          {8,4,2,1},{8,4,2,1},{4,2,1}, {200,200,100,50},{200,200,100,50},{200,100,50},
          4,4,3, LOSS_MI, LOSS_MI },
    };
    int n_ds = 3;

    for (int di = 0; di < n_ds; di++) {
        dataset_t *ds = &datasets[di];
        if (only && strcmp(only, ds->name) != 0) continue;

        image_t fixed, moving;
        if (image_load(&fixed, ds->fixed_path, DEVICE_CPU) != 0 ||
            image_load(&moving, ds->moving_path, DEVICE_CPU) != 0) {
            printf("Failed to load dataset %s\n", ds->name);
            continue;
        }

        int fD = fixed.data.shape[2], fH = fixed.data.shape[3], fW = fixed.data.shape[4];
        int fN = fD * fH * fW;

        /* NCC before: identity resampling (pre-computed, same for all backends) */
        float ncc_before = 0;
        if (strcmp(ds->name, "small") == 0) ncc_before = 0.5957f;
        else if (strcmp(ds->name, "medium") == 0) ncc_before = 0.5753f;
        else if (strcmp(ds->name, "large") == 0) ncc_before = 0.7254f;

        double t0 = get_time(), t1;

        /* Moments */
        t1 = get_time();
        moments_opts_t mopts = moments_opts_default();
        moments_result_t mom;
        moments_register(&fixed, &moving, mopts, &mom);
        double t_mom = get_time() - t1;

        /* Rigid */
        t1 = get_time();
        rigid_opts_t ropts = {
            .n_scales=ds->rigid_n, .scales=ds->rigid_scales, .iterations=ds->rigid_iters,
            .lr=3e-3f, .loss_type=ds->rigid_loss, .mi_num_bins=32, .cc_kernel_size=5,
            .tolerance=1e-6f, .max_tolerance_iters=10,
            .downsample_mode=ds_mode };
        rigid_result_t rigid;
        rigid_register_metal(&fixed, &moving, &mom, ropts, &rigid);
        double t_rigid = get_time() - t1;

        /* Affine */
        t1 = get_time();
        affine_opts_t aopts = {
            .n_scales=ds->affine_n, .scales=ds->affine_scales, .iterations=ds->affine_iters,
            .lr=1e-3f, .loss_type=ds->affine_loss, .mi_num_bins=32, .cc_kernel_size=5,
            .tolerance=1e-6f, .max_tolerance_iters=10,
            .downsample_mode=ds_mode };
        affine_result_t affine;
        affine_register_metal(&fixed, &moving, rigid.rigid_mat, aopts, &affine);
        double t_affine = get_time() - t1;

        /* Deformable: Greedy or SyN */
        float aff44[4][4] = {{0}};
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                aff44[i][j] = affine.affine_mat[i][j];
        aff44[3][3] = 1.0f;

        tensor_t deform_moved = {0};
        float local_ncc = 0;
        double t_deform;
        const char *deform_name;

        if (use_greedy) {
            t1 = get_time();
            greedy_opts_t gopts = {
                .n_scales=ds->syn_n, .scales=ds->syn_scales, .iterations=ds->syn_iters,
                .cc_kernel_size=5, .lr=0.1f,
                .smooth_warp_sigma=0.5f, .smooth_grad_sigma=1.0f,
                .tolerance=1e-6f, .max_tolerance_iters=10,
                .downsample_mode=ds_mode };
            greedy_result_t greedy;
            greedy_register_metal(&fixed, &moving, aff44, gopts, &greedy);
            t_deform = get_time() - t1;
            deform_moved = greedy.moved;
            local_ncc = greedy.ncc_loss;
            deform_name = "Greedy";
        } else {
            t1 = get_time();
            syn_opts_t sopts = {
                .n_scales=ds->syn_n, .scales=ds->syn_scales, .iterations=ds->syn_iters,
                .cc_kernel_size=5, .lr=0.1f,
                .smooth_warp_sigma=0.5f, .smooth_grad_sigma=1.0f,
                .tolerance=1e-6f, .max_tolerance_iters=10,
                .downsample_mode=ds_mode };
            syn_result_t syn;
            syn_register_metal(&fixed, &moving, aff44, sopts, &syn);
            t_deform = get_time() - t1;
            deform_moved = syn.moved;
            local_ncc = syn.ncc_loss;
            deform_name = "SyN";
        }

        double t_total = get_time() - t0;

        /* Evaluate final NCC — global */
        float ncc_after = compute_global_ncc((float *)fixed.data.data,
                                              (float *)deform_moved.data, fN);

        /* Peak RAM */
        struct mach_task_basic_info info;
        mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
        task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count);
        double peak_mb = info.resident_size_max / (1024.0 * 1024.0);

        printf("\n============================================================\n");
        printf("Dataset: %s — %s (Metal, %s, %s)\n", ds->name, ds->desc,
               use_trilinear ? "trilinear" : "FFT", deform_name);
        printf("============================================================\n");
        printf("  Results:\n");
        printf("    NCC Before:     %.4f\n", ncc_before);
        printf("    NCC After:      %.4f\n", ncc_after);
        printf("    Local NCC Loss: %.4f\n", local_ncc);
        printf("    Time:           %.1fs\n", t_total);
        printf("    Peak RAM:       %.0f MB\n", peak_mb);
        printf("    Moments:        %.1fs\n", t_mom);
        printf("    Rigid:          %.1fs\n", t_rigid);
        printf("    Affine:         %.1fs\n", t_affine);
        printf("    %s:            %.1fs\n", deform_name, t_deform);

        printf("\n  CSV: %s,%.4f,%.4f,%.4f,%.1f,%.0f\n",
               ds->name, ncc_before, ncc_after, local_ncc, t_total, peak_mb);

        tensor_free(&deform_moved);
        image_free(&fixed);
        image_free(&moving);
    }

    return 0;
#endif
}
