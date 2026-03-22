/*
 * test_validate_webgpu.c - Full validation pipeline using WebGPU backend
 *
 * Mirrors test_validate_all.c but uses WebGPU registration functions.
 * Pipeline: Moments (CPU) → Rigid (WebGPU) → Affine (WebGPU) → SyN (WebGPU)
 */

#include "cfireants/backend.h"
#include "cfireants/tensor.h"
#include "cfireants/image.h"
#include "cfireants/registration.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#ifdef CFIREANTS_HAS_WEBGPU
extern int cfireants_init_webgpu(void);

/* WebGPU kernel functions for NCC-before evaluation */
typedef struct WGPUBufferImpl* WGPUBuffer;
extern void wgpu_affine_grid_3d(WGPUBuffer affine, WGPUBuffer grid, int B, int D, int H, int W);
extern void wgpu_grid_sample_3d_fwd(WGPUBuffer input, WGPUBuffer grid, WGPUBuffer output,
    int B, int C, int iD, int iH, int iW, int oD, int oH, int oW);
extern WGPUBuffer wgpu_create_buffer(size_t size, uint64_t usage, const char *label);
extern void wgpu_write_buffer(WGPUBuffer dst, size_t offset, const void *src, size_t size);
extern void wgpu_read_buffer(WGPUBuffer src, size_t offset, void *dst, size_t size);
extern void wgpuBufferRelease(WGPUBuffer buffer);
#endif

static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double get_peak_rss_mb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256]; double peak_kb = 0;
    while (fgets(line, sizeof(line), f))
        if (strncmp(line, "VmHWM:", 6) == 0) { sscanf(line + 6, "%lf", &peak_kb); break; }
    fclose(f);
    return peak_kb / 1024.0;
}

static float compute_global_ncc(const float *a, const float *b, int n) {
    double sum_a=0,sum_b=0,sum_ab=0,sum_a2=0,sum_b2=0;
    for(int i=0;i<n;i++){
        sum_a+=a[i]; sum_b+=b[i]; sum_ab+=a[i]*b[i];
        sum_a2+=a[i]*a[i]; sum_b2+=b[i]*b[i];
    }
    double ma=sum_a/n, mb=sum_b/n;
    double cov=sum_ab/n-ma*mb;
    double va=sum_a2/n-ma*ma, vb=sum_b2/n-mb*mb;
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
#ifndef CFIREANTS_HAS_WEBGPU
    printf("WebGPU not enabled\n"); return 0;
#else
    if(cfireants_init_webgpu()!=0){printf("WebGPU init failed\n");return 1;}

    const char *only_dataset = NULL;
    int downsample_mode = DOWNSAMPLE_FFT;
    int use_greedy = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dataset") == 0 && i+1 < argc) only_dataset = argv[++i];
        else if (strcmp(argv[i], "--trilinear") == 0) downsample_mode = DOWNSAMPLE_TRILINEAR;
        else if (strcmp(argv[i], "--greedy") == 0) use_greedy = 1;
    }

    dataset_t datasets[] = {
        { "small", "2mm full-head MNI to subject (includes scalp)",
          "validate/small/MNI152_T1_2mm.nii.gz", "validate/small/T1_head_2mm.nii.gz",
          {4,2,1},{4,2,1},{4,2,1}, {200,100,50},{200,100,50},{200,100,50},
          3,3,3, LOSS_MI, LOSS_MI },  /* MI for rigid/affine (matching CUDA) */
        { "medium", "1mm brain-extracted MNI to subject",
          "validate/medium/MNI152_T1_1mm_brain.nii.gz", "validate/medium/t1_brain.nii.gz",
          {4,2,1},{4,2,1},{4,2,1}, {200,100,50},{200,100,50},{200,100,50},
          3,3,3, LOSS_CC, LOSS_CC },  /* CC for brain-extracted (same intensity range) */
        { "large", "1mm full-head MNI to subject (0.88mm, includes scalp)",
          "validate/large/MNI152_T1_1mm.nii.gz", "validate/large/chris_t1.nii.gz",
          {8,4,2,1},{8,4,2,1},{4,2,1}, {200,200,100,50},{200,200,100,50},{200,100,50},
          4,4,3, LOSS_MI, LOSS_MI },  /* MI for full-head (matching CUDA) */
    };
    int n_ds = 3;

    for(int di=0; di<n_ds; di++){
        dataset_t *ds = &datasets[di];
        if(only_dataset && strcmp(only_dataset, ds->name)!=0) continue;

        image_t fixed, moving;
        if(image_load(&fixed,ds->fixed_path,DEVICE_CPU)!=0||
           image_load(&moving,ds->moving_path,DEVICE_CPU)!=0){
            fprintf(stderr,"Failed to load %s\n",ds->name); continue;
        }

        int fD=fixed.data.shape[2],fH=fixed.data.shape[3],fW=fixed.data.shape[4];
        int mD=moving.data.shape[2],mH=moving.data.shape[3],mW=moving.data.shape[4];
        int fN=fD*fH*fW;

        /* NCC before: resample moving into fixed space with identity affine */
        fprintf(stderr, "Computing NCC before...\n");
        float ncc_before = 0;
        {
            /* Build identity physical affine */
            float id44[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
            mat44d pd, tm, cb;
            for(int i=0;i<4;i++) for(int j=0;j<4;j++) pd.m[i][j]=id44[i][j];
            mat44d_mul(&tm, &pd, &fixed.meta.torch2phy);
            mat44d_mul(&cb, &moving.meta.phy2torch, &tm);
            float ha[12]; for(int i=0;i<3;i++) for(int j=0;j<4;j++) ha[i*4+j]=(float)cb.m[i][j];

            /* Upload, grid_sample, download */
            uint64_t uu = 0x80 | 0x04 | 0x08; /* Storage|CopySrc|CopyDst */
            WGPUBuffer d_af=wgpu_create_buffer(48,uu,"nba");
            WGPUBuffer d_fi=wgpu_create_buffer((size_t)fN*4,uu,"nbf");
            WGPUBuffer d_mi=wgpu_create_buffer((size_t)mD*mH*mW*4,uu,"nbm");
            WGPUBuffer d_gr=wgpu_create_buffer((size_t)fN*3*4,uu,"nbg");
            WGPUBuffer d_mo=wgpu_create_buffer((size_t)fN*4,uu,"nbo");
            wgpu_write_buffer(d_af,0,ha,48);
            wgpu_write_buffer(d_fi,0,fixed.data.data,(size_t)fN*4);
            wgpu_write_buffer(d_mi,0,moving.data.data,(size_t)mD*mH*mW*4);
            wgpu_affine_grid_3d(d_af,d_gr,1,fD,fH,fW);
            wgpu_grid_sample_3d_fwd(d_mi,d_gr,d_mo,1,1,mD,mH,mW,fD,fH,fW);
            float *h_moved_id=(float*)malloc((size_t)fN*4);
            wgpu_read_buffer(d_mo,0,h_moved_id,(size_t)fN*4);
            ncc_before = compute_global_ncc((float*)fixed.data.data, h_moved_id, fN);
            free(h_moved_id);
            wgpuBufferRelease(d_af);wgpuBufferRelease(d_fi);wgpuBufferRelease(d_mi);
            wgpuBufferRelease(d_gr);wgpuBufferRelease(d_mo);
        }
        float *h_id_moved = NULL; /* not needed */

        double t0 = get_time();

        /* Moments */
        double t1 = get_time();
        moments_opts_t mopts = moments_opts_default();
        moments_result_t mom;
        moments_register(&fixed, &moving, mopts, &mom);
        double t_mom = get_time()-t1;

        /* Rigid */
        t1 = get_time();
        rigid_opts_t ropts = {
            .n_scales=ds->rigid_n, .scales=ds->rigid_scales, .iterations=ds->rigid_iters,
            .lr=3e-3f, .loss_type=ds->rigid_loss, .mi_num_bins=32, .cc_kernel_size=5,
            .tolerance=1e-6f, .max_tolerance_iters=10,
            .downsample_mode=downsample_mode };
        rigid_result_t rigid;
        rigid_register_webgpu(&fixed, &moving, &mom, ropts, &rigid);
        double t_rigid = get_time()-t1;
        fprintf(stderr, "  Rigid NCC: %.4f\n", rigid.ncc_loss);
        fprintf(stderr, "  Rigid mat: [%.6f %.6f %.6f %.6f; %.6f %.6f %.6f %.6f; %.6f %.6f %.6f %.6f]\n",
                rigid.rigid_mat[0][0], rigid.rigid_mat[0][1], rigid.rigid_mat[0][2], rigid.rigid_mat[0][3],
                rigid.rigid_mat[1][0], rigid.rigid_mat[1][1], rigid.rigid_mat[1][2], rigid.rigid_mat[1][3],
                rigid.rigid_mat[2][0], rigid.rigid_mat[2][1], rigid.rigid_mat[2][2], rigid.rigid_mat[2][3]);

        /* Affine */
        t1 = get_time();
        affine_opts_t aopts = {
            .n_scales=ds->affine_n, .scales=ds->affine_scales, .iterations=ds->affine_iters,
            .lr=1e-3f, .loss_type=ds->affine_loss, .mi_num_bins=32, .cc_kernel_size=5,
            .tolerance=1e-6f, .max_tolerance_iters=10,
            .downsample_mode=downsample_mode };
        affine_result_t affine;
        affine_register_webgpu(&fixed, &moving, rigid.rigid_mat, aopts, &affine);
        double t_affine = get_time()-t1;
        fprintf(stderr, "  Affine NCC: %.4f\n", affine.ncc_loss);
        fprintf(stderr, "  Affine mat: [%.6f %.6f %.6f %.6f; %.6f %.6f %.6f %.6f; %.6f %.6f %.6f %.6f]\n",
                affine.affine_mat[0][0], affine.affine_mat[0][1], affine.affine_mat[0][2], affine.affine_mat[0][3],
                affine.affine_mat[1][0], affine.affine_mat[1][1], affine.affine_mat[1][2], affine.affine_mat[1][3],
                affine.affine_mat[2][0], affine.affine_mat[2][1], affine.affine_mat[2][2], affine.affine_mat[2][3]);

        /* Build 4x4 for deformable */
        float aff44[4][4]={{0}};
        for(int i=0;i<3;i++) for(int j=0;j<4;j++) aff44[i][j]=affine.affine_mat[i][j];
        aff44[3][3]=1;

        /* Deformable: Greedy or SyN */
        tensor_t deform_moved = {0};
        float deform_ncc_loss = 0;
        double t_deform;
        const char *deform_name;

        if (use_greedy) {
            t1 = get_time();
            int gs[] = {4, 2, 1}, gi[] = {200, 100, 50};
            greedy_opts_t gopts = {
                .n_scales=3, .scales=gs, .iterations=gi,
                .cc_kernel_size=5, .lr=0.1f,
                .smooth_warp_sigma=0.5f, .smooth_grad_sigma=1.0f,
                .tolerance=1e-6f, .max_tolerance_iters=10,
                .downsample_mode=downsample_mode };
            greedy_result_t greedy;
            greedy_register_webgpu(&fixed, &moving, aff44, gopts, &greedy);
            t_deform = get_time()-t1;
            deform_moved = greedy.moved;
            deform_ncc_loss = greedy.ncc_loss;
            deform_name = "Greedy";
        } else {
            t1 = get_time();
            syn_opts_t sopts = {
                .n_scales=ds->syn_n, .scales=ds->syn_scales, .iterations=ds->syn_iters,
                .cc_kernel_size=5, .lr=0.1f,
                .smooth_warp_sigma=0.5f, .smooth_grad_sigma=1.0f,
                .tolerance=1e-6f, .max_tolerance_iters=10,
                .downsample_mode=downsample_mode };
            syn_result_t syn;
            syn_register_webgpu(&fixed, &moving, aff44, sopts, &syn);
            t_deform = get_time()-t1;
            deform_moved = syn.moved;
            deform_ncc_loss = syn.ncc_loss;
            deform_name = "SyN";
        }

        double t_total = get_time()-t0;

        float ncc_after = compute_global_ncc((float*)fixed.data.data,
                                              (float*)deform_moved.data, fN);

        printf("\n============================================================\n");
        printf("Dataset: %s — %s (WebGPU, %s, %s)\n", ds->name, ds->desc,
               downsample_mode == DOWNSAMPLE_TRILINEAR ? "trilinear" : "FFT", deform_name);
        printf("============================================================\n");
        printf("  Results:\n");
        printf("    NCC Before:     %.4f\n", ncc_before);
        printf("    NCC After:      %.4f\n", ncc_after);
        printf("    Local NCC Loss: %.4f\n", deform_ncc_loss);
        printf("    Time:           %.1fs\n", t_total);
        printf("    Moments:        %.1fs\n", t_mom);
        printf("    Rigid:          %.1fs\n", t_rigid);
        printf("    Affine:         %.1fs\n", t_affine);
        printf("    %s:            %.1fs\n", deform_name, t_deform);
        printf("    Peak RAM:       %.0f MB\n", get_peak_rss_mb());

        printf("\n  CSV: %s,%.4f,%.4f,%.4f,%.1f,%.1f,%.1f,%.1f,%.1f,%.0f\n",
               ds->name, ncc_before, ncc_after, deform_ncc_loss,
               t_total, t_mom, t_rigid, t_affine, t_deform, get_peak_rss_mb());

        tensor_free(&deform_moved);
        image_free(&fixed); image_free(&moving);
    }

    return 0;
#endif
}
