/*
 * greedy_webgpu.c - WebGPU greedy deformable registration (GPU-native)
 *
 * All per-iteration operations stay on GPU:
 *   grid_sample, CC loss, backward, Adam moments/direction,
 *   compositive update, blur — all via WGSL compute shaders.
 * Only scalar loss and Gaussian kernel come to CPU.
 */

#include "webgpu_context.h"
#include "webgpu_kernels.h"
#include "cfireants/tensor.h"
#include "cfireants/image.h"
#include "cfireants/registration.h"
#include "cfireants/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ds_buf removed — uses shared wgpu_downsample_image from webgpu_kernels.h */

/* Build Gaussian kernel and upload to GPU buffer */
static WGPUBuffer make_gpu_gauss(float sigma, float truncated, int *klen_out) {
    if (sigma <= 0) { *klen_out = 0; return NULL; }
    int tail = (int)(truncated * sigma + 0.5f);
    int klen = 2 * tail + 1;
    float *h = (float *)malloc(klen * sizeof(float));
    if (!h) { *klen_out = -1; return NULL; }
    float inv = 1.0f / (sigma * sqrtf(2.0f));
    float sum = 0;
    for (int i = 0; i < klen; i++) {
        float x = (float)(i - tail);
        h[i] = 0.5f * (erff((x+0.5f)*inv) - erff((x-0.5f)*inv));
        sum += h[i];
    }
    for (int i = 0; i < klen; i++) h[i] /= sum;
    WGPUBufferUsage u = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    WGPUBuffer buf = wgpu_create_buffer_init(h, klen*sizeof(float), u, "gauss_kern");
    free(h);
    *klen_out = klen;
    return buf;
}

static int greedy_webgpu_abort(greedy_result_t *result, const char *message) {
    fprintf(stderr, "greedy_register_webgpu: %s\n", message);
    tensor_free(&result->disp);
    tensor_free(&result->moved);
    wgpu_context_cleanup();
    return -1;
}

int greedy_register_webgpu(const image_t *fixed, const image_t *moving,
                            const float init_affine_44[4][4],
                            greedy_opts_t opts, greedy_result_t *result)
{
    wgpu_clear_fatal_error();
    if (greedy_result_init(result) != 0) return -1;
    memcpy(result->affine_44, init_affine_44, 16 * sizeof(float));

    int fD=fixed->data.shape[2], fH=fixed->data.shape[3], fW=fixed->data.shape[4];
    int mD=moving->data.shape[2], mH=moving->data.shape[3], mW=moving->data.shape[4];
    long fSp=(long)fD*fH*fW;
    int result_alloc_failed = 0;

    mat44d pd, tm, comb;
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) pd.m[i][j]=init_affine_44[i][j];
    mat44d_mul(&tm, &pd, &fixed->meta.torch2phy);
    mat44d_mul(&comb, &moving->meta.phy2torch, &tm);
    float h_aff[12]; for(int i=0;i<3;i++) for(int j=0;j<4;j++) h_aff[i*4+j]=(float)comb.m[i][j];

    WGPUBufferUsage u=WGPUBufferUsage_Storage|WGPUBufferUsage_CopySrc|WGPUBufferUsage_CopyDst;
    WGPUBuffer d_aff=wgpu_create_buffer(48,u,"aff");
    wgpu_write_buffer(d_aff, 0, h_aff, 48);

    WGPUBuffer d_ff=wgpu_create_buffer(fSp*4,u,"ff");
    WGPUBuffer d_mf=wgpu_create_buffer((long)mD*mH*mW*4,u,"mf");
    wgpu_write_buffer(d_ff, 0, fixed->data.data, fSp*4);
    wgpu_write_buffer(d_mf, 0, moving->data.data, (long)mD*mH*mW*4);

    /* Build Gaussian kernels on GPU */
    int grad_klen=0, warp_klen=0;
    WGPUBuffer d_grad_kern = make_gpu_gauss(opts.smooth_grad_sigma, 2.0f, &grad_klen);
    WGPUBuffer d_warp_kern = make_gpu_gauss(opts.smooth_warp_sigma, 2.0f, &warp_klen);
    if (grad_klen < 0 || warp_klen < 0)
        return greedy_webgpu_abort(result, "host kernel allocation failed");
    if (wgpu_had_fatal_error())
        return greedy_webgpu_abort(result, "initial GPU allocation failed");

    /* GPU-side warp + optimizer state */
    WGPUBuffer d_warp=NULL, d_ea=NULL, d_eas=NULL;
    int step_t=0;
    int prev_dD=0, prev_dH=0, prev_dW=0;
    float beta1=0.9f, beta2=0.99f, eps=1e-8f;

    for (int si=0; si<opts.n_scales; si++) {
        int scale=opts.scales[si], iters=opts.iterations[si];
        int dD=(scale>1)?fD/scale:fD, dH=(scale>1)?fH/scale:fH, dW=(scale>1)?fW/scale:fW;
        if(dD<8)dD=8;if(dH<8)dH=8;if(dW<8)dW=8;if(scale==1){dD=fD;dH=fH;dW=fW;}
        int mdD, mdH, mdW;
        moving_pyramid_size(scale, fixed->meta.spacing, moving->meta.spacing,
                            mD, mH, mW, &mdD, &mdH, &mdW);
        if(scale==1){mdD=mD;mdH=mH;mdW=mW;}

        long sp=(long)dD*dH*dW, n3=sp*3;

        WGPUBuffer d_fd, d_md;
        if(scale>1) {
            d_fd=wgpu_downsample_image(d_ff, fD,fH,fW, dD,dH,dW, opts.downsample_mode);
            d_md=wgpu_downsample_image(d_mf, mD,mH,mW, mdD,mdH,mdW, opts.downsample_mode);
        } else { d_fd=d_ff; d_md=d_mf; mdD=mD;mdH=mH;mdW=mW; }
        if (wgpu_had_fatal_error() || !d_fd || !d_md)
            return greedy_webgpu_abort(result, "pyramid allocation failed");

        /* Init/reset warp on GPU */
        if(!d_warp) {
            d_warp=wgpu_create_buffer(n3*4,u,"warp");
            if (!d_warp)
                return greedy_webgpu_abort(result, "warp allocation failed");
            wgpu_tensor_fill_buf(d_warp, 0.0f, (int)n3);
        } else if(prev_dD!=dD||prev_dH!=dH||prev_dW!=dW) {
            /* Resize warp via trilinear interpolation (matching CUDA) */
            long prev_sp=(long)prev_dD*prev_dH*prev_dW;
            size_t prev_n3_sz=(size_t)prev_sp*3*sizeof(float);
            size_t new_n3_sz=(size_t)n3*sizeof(float);
            float *h_old=(float*)malloc(prev_n3_sz);
            float *h_3dhw=(float*)malloc(prev_n3_sz);
            float *h_3dhw_new=(float*)malloc(new_n3_sz);
            float *h_new=(float*)malloc(new_n3_sz);
            if (!h_old || !h_3dhw || !h_3dhw_new || !h_new) {
                free(h_old);free(h_3dhw);free(h_3dhw_new);free(h_new);
                return greedy_webgpu_abort(result, "warp-resize host allocation failed");
            }

            wgpu_read_buffer(d_warp,0,h_old,prev_n3_sz);
            /* Permute [D,H,W,3] → [3,D,H,W] */
            for(long i=0;i<prev_sp;i++) for(int c=0;c<3;c++) h_3dhw[c*prev_sp+i]=h_old[i*3+c];
            WGPUBuffer b_src=wgpu_create_buffer_init(h_3dhw,prev_n3_sz,u,"wr_src");
            WGPUBuffer b_dst=wgpu_create_buffer(new_n3_sz,u,"wr_dst");
            if (wgpu_had_fatal_error()) {
                free(h_old);free(h_3dhw);free(h_3dhw_new);free(h_new);
                return greedy_webgpu_abort(result, "warp-resize GPU allocation failed");
            }
            wgpu_trilinear_resize(b_src,b_dst,1,3,prev_dD,prev_dH,prev_dW,dD,dH,dW,1);
            wgpu_read_buffer(b_dst,0,h_3dhw_new,new_n3_sz);
            wgpu_release_buffer(b_src);wgpu_release_buffer(b_dst);
            /* Permute [3,D,H,W] → [D,H,W,3] */
            for(long i=0;i<sp;i++) for(int c=0;c<3;c++) h_new[i*3+c]=h_3dhw_new[c*sp+i];
            wgpu_release_buffer(d_warp);
            d_warp=wgpu_create_buffer(n3*4,u,"warp");wgpu_write_buffer(d_warp,0,h_new,new_n3_sz);

            free(h_old);free(h_3dhw);free(h_3dhw_new);free(h_new);
            if(d_ea){wgpu_release_buffer(d_ea);wgpu_release_buffer(d_eas);d_ea=NULL;}
            step_t=0;
        }
        if(!d_ea) {
            d_ea=wgpu_create_buffer(n3*4,u,"ea");
            d_eas=wgpu_create_buffer(n3*4,u,"eas");
            if (!d_ea || !d_eas)
                return greedy_webgpu_abort(result, "optimizer allocation failed");
            wgpu_tensor_fill_buf(d_ea, 0.0f, (int)n3);
            wgpu_tensor_fill_buf(d_eas, 0.0f, (int)n3);
        }
        if (wgpu_had_fatal_error())
            return greedy_webgpu_abort(result, "warp or optimizer allocation failed");
        prev_dD=dD;prev_dH=dH;prev_dW=dW;

        float half_res=1.0f/(float)((dD>dH?(dD>dW?dD:dW):(dH>dW?dH:dW))-1);

        fprintf(stderr,"  Greedy GPU scale %d: fixed[%d,%d,%d] moving[%d,%d,%d] x %d iters\n",
                scale,dD,dH,dW,mdD,mdH,mdW,iters);

        /* Generate grids */
        WGPUBuffer d_base=wgpu_create_buffer(n3*4,u,"base");
        if (!d_base)
            return greedy_webgpu_abort(result, "base-grid allocation failed");
        wgpu_affine_grid_3d(d_aff, d_base, 1, dD,dH,dW);

        /* Iteration buffers — all stay on GPU */
        WGPUBuffer d_sg=wgpu_create_buffer(n3*4,u,"sg");
        WGPUBuffer d_mov=wgpu_create_buffer(sp*4,u,"mov");
        WGPUBuffer d_gmov=wgpu_create_buffer(sp*4,u,"gm");
        WGPUBuffer d_gg=wgpu_create_buffer(n3*4,u,"gg");
        WGPUBuffer d_adir=wgpu_create_buffer(n3*4,u,"ad");
        WGPUBuffer d_norm_state=wgpu_create_buffer(sizeof(uint32_t),u,"norm_max");
        WGPUBuffer d_scratch=wgpu_create_buffer(n3*4,u,"scr");
        wgpu_cc_workspace_t cc_workspace;
        if (wgpu_cc_workspace_init(&cc_workspace, (int)sp, 1) != 0)
            return greedy_webgpu_abort(result, "CC workspace allocation failed");
        if (wgpu_had_fatal_error())
            return greedy_webgpu_abort(result, "per-scale workspace allocation failed");

        float prev_loss=1e30f; int cc=0;

        for (int it=0; it<iters; it++) {
            /* Batch forward pass */
            wgpu_begin_batch();
            wgpu_copy_buffer(d_base, d_sg, n3*4);
            wgpu_tensor_add_buf(d_sg, d_warp, (int)n3);
            wgpu_grid_sample_3d_fwd(d_md, d_sg, d_mov, 1,1,mdD,mdH,mdW,dD,dH,dW);
            /* Flush forward batch before CC loss */
            wgpu_flush();
            float loss = 0;
            wgpu_cc_loss_3d_raw(d_mov, d_fd, d_gmov, dD,dH,dW,
                                opts.cc_kernel_size, &loss, &cc_workspace);
            if (wgpu_had_fatal_error())
                return greedy_webgpu_abort(result, "CC iteration failed");

            /* Backward — no batch (each dispatch flushes internally) */
            wgpu_grid_sample_3d_bwd(d_gmov, d_md, d_sg, d_gg, 1,1,mdD,mdH,mdW,dD,dH,dW);
            if(grad_klen>0) wgpu_blur_disp_dhw3(d_gg, d_scratch, dD,dH,dW, d_grad_kern, grad_klen);

            step_t++;
            float bc1=1.0f-powf(beta1,(float)step_t);
            float bc2=1.0f-powf(beta2,(float)step_t);
            wgpu_adam_moments_update_buf(d_gg, d_ea, d_eas, beta1, beta2, (int)n3);
            wgpu_adam_direction_buf(d_adir, d_ea, d_eas, bc1, bc2, eps, (int)n3);
            /* Normalise on the device, as SyN does. Reading the maximum back
             * to compute the scale on the CPU cost one blocking readback per
             * iteration, 350 for a standard schedule, each an ASYNCIFY suspend
             * in the browser. The shader clears its atomic first, so unlike the
             * cache this replaced there is no state carried between calls. */
            wgpu_normalize_l2_buf(d_adir, (int)sp, eps,
                                  half_res * (-opts.lr), d_norm_state);

            /* Compose + blur + copy (no batch — requires ordered execution) */
            wgpu_fused_compositive_update(d_warp, d_adir, d_adir, dD,dH,dW);
            if(warp_klen>0) wgpu_blur_disp_dhw3(d_adir, d_scratch, dD,dH,dW, d_warp_kern, warp_klen);
            wgpu_copy_buffer(d_adir, d_warp, n3 * 4);

            if(it%50==0||it==iters-1)
                fprintf(stderr,"    iter %d/%d loss=%.6f\n",it,iters,loss);

            if(fabsf(loss-prev_loss)<opts.tolerance){cc++;if(cc>=opts.max_tolerance_iters){fprintf(stderr,"    Converged at iter %d\n",it);break;}}
            else cc=0;
            prev_loss=loss;
        }

        wgpu_release_buffer(d_base);wgpu_release_buffer(d_sg);
        wgpu_release_buffer(d_mov);wgpu_release_buffer(d_gmov);
        wgpu_release_buffer(d_gg);wgpu_release_buffer(d_adir);wgpu_release_buffer(d_norm_state);wgpu_release_buffer(d_scratch);
        wgpu_cc_workspace_cleanup(&cc_workspace);
        if(scale>1){wgpu_release_buffer(d_fd);wgpu_release_buffer(d_md);}
    }

    /* Evaluate at full res */
    {
        long n3=fSp*3;
        WGPUBuffer d_base=wgpu_create_buffer(n3*4,u,"eb");
        WGPUBuffer d_sg=wgpu_create_buffer(n3*4,u,"esg");
        WGPUBuffer d_mov=wgpu_create_buffer(fSp*4,u,"em");

        if (wgpu_had_fatal_error())
            return greedy_webgpu_abort(result, "evaluation allocation failed");

        wgpu_affine_grid_3d(d_aff, d_base, 1, fD,fH,fW);

        wgpu_copy_buffer(d_base, d_sg, n3 * 4);

        if(d_warp && prev_dD==fD && prev_dH==fH && prev_dW==fW)
            wgpu_tensor_add_buf(d_sg, d_warp, (int)n3);

        wgpu_grid_sample_3d_fwd(d_mf, d_sg, d_mov, 1,1,mD,mH,mW,fD,fH,fW);
        wgpu_cc_loss_3d_raw(d_mov, d_ff, NULL, fD,fH,fW, 9,
                            &result->ncc_loss, NULL);

        int shape[5]={1,1,fD,fH,fW};
        if (tensor_alloc(&result->moved,5,shape,DTYPE_FLOAT32,DEVICE_CPU) != 0)
            return greedy_webgpu_abort(result, "output allocation failed");
        wgpu_read_buffer(d_mov,0,result->moved.data,fSp*4);

        wgpu_release_buffer(d_base);wgpu_release_buffer(d_sg);wgpu_release_buffer(d_mov);
    }

    /* Export the optimized warp just like the CPU backend.  Keeping this as a
     * CPU-owned tensor makes result ownership independent of the backend and
     * lets callers compare or persist the genuinely sensitive stage output. */
    if (d_warp) {
        int disp_shape[5] = {1, prev_dD, prev_dH, prev_dW, 3};
        if (tensor_alloc(&result->disp, 5, disp_shape,
                         DTYPE_FLOAT32, DEVICE_CPU) != 0) {
            result_alloc_failed = 1;
        } else {
            wgpu_read_buffer(d_warp, 0, result->disp.data,
                             (size_t)prev_dD * prev_dH * prev_dW * 3 * sizeof(float));
        }
    }

    if(d_warp) wgpu_release_buffer(d_warp);
    if(d_ea) wgpu_release_buffer(d_ea);
    if(d_eas) wgpu_release_buffer(d_eas);
    if(d_grad_kern) wgpu_release_buffer(d_grad_kern);
    if(d_warp_kern) wgpu_release_buffer(d_warp_kern);
    wgpu_release_buffer(d_aff);wgpu_release_buffer(d_ff);wgpu_release_buffer(d_mf);
    if (result_alloc_failed || wgpu_had_fatal_error()) {
        fprintf(stderr, "%s: a GPU operation failed; result is not trustworthy\n", __func__);
        tensor_free(&result->disp);
        tensor_free(&result->moved);
        return -1;
    }
    return 0;
}
