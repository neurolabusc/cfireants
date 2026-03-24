/*
 * syn_webgpu.c - WebGPU SyN registration (GPU-native)
 *
 * All per-iteration operations stay on GPU. Only scalar loss transferred.
 * Warp inversion for evaluation uses CPU fallback.
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

static WGPUBuffer make_gauss_buf(float sigma, float trunc, int *klen) {
    if(sigma<=0){*klen=0;return NULL;}
    int tail=(int)(trunc*sigma+0.5f), kl=2*tail+1;
    float *h=(float*)malloc(kl*sizeof(float));
    float inv=1.0f/(sigma*sqrtf(2.0f)),sum=0;
    for(int i=0;i<kl;i++){float x=(float)(i-tail);h[i]=0.5f*(erff((x+.5f)*inv)-erff((x-.5f)*inv));sum+=h[i];}
    for(int i=0;i<kl;i++)h[i]/=sum;
    WGPUBuffer b=wgpu_create_buffer_init(h,kl*4,WGPUBufferUsage_Storage|WGPUBufferUsage_CopyDst,"gk");
    free(h);*klen=kl;return b;
}

/* cpu_warp_inverse is now in utils.c */

/* Helper: WarpAdam step for one warp (all on GPU) */
static void warp_adam_step(WGPUBuffer d_grad, WGPUBuffer d_warp,
                            WGPUBuffer d_ea, WGPUBuffer d_eas,
                            WGPUBuffer d_adir, WGPUBuffer d_scratch,
                            int dD, int dH, int dW,
                            int *step_t, float beta1, float beta2, float eps,
                            float lr, float half_res,
                            WGPUBuffer d_grad_kern, int grad_klen,
                            WGPUBuffer d_warp_kern, int warp_klen) {
    long n3 = (long)dD*dH*dW*3;
    long sp = (long)dD*dH*dW;

    /* Blur gradient */
    if(grad_klen>0) wgpu_blur_disp_dhw3(d_grad, d_scratch, dD,dH,dW, d_grad_kern, grad_klen);

    (*step_t)++;
    float bc1=1.0f-powf(beta1,(float)*step_t);
    float bc2=1.0f-powf(beta2,(float)*step_t);

    wgpu_adam_moments_update_buf(d_grad, d_ea, d_eas, beta1, beta2, (int)n3);
    wgpu_adam_direction_buf(d_adir, d_ea, d_eas, bc1, bc2, eps, (int)n3);

    float gmax = wgpu_max_l2_norm_buf(d_adir, (int)sp, eps);
    if(gmax<1.0f) gmax=1.0f;
    float sf = half_res / gmax * (-lr);
    wgpu_tensor_scale_buf(d_adir, sf, (int)n3);

    wgpu_fused_compositive_update(d_warp, d_adir, d_adir, dD,dH,dW);

    if(warp_klen>0) wgpu_blur_disp_dhw3(d_adir, d_scratch, dD,dH,dW, d_warp_kern, warp_klen);

    /* Copy adir → warp */
    WGPUCommandEncoder e=wgpuDeviceCreateCommandEncoder(g_wgpu.device,NULL);
    wgpuCommandEncoderCopyBufferToBuffer(e,d_adir,0,d_warp,0,n3*4);
    WGPUCommandBuffer c=wgpuCommandEncoderFinish(e,NULL);
    wgpuQueueSubmit(g_wgpu.queue,1,&c);
    wgpuCommandBufferRelease(c);wgpuCommandEncoderRelease(e);
    wgpuDevicePoll(g_wgpu.device,1,NULL);
}

int syn_register_webgpu(const image_t *fixed, const image_t *moving,
                         const float init_affine_44[4][4],
                         syn_opts_t opts, syn_result_t *result)
{
    memset(result,0,sizeof(syn_result_t));
    memcpy(result->affine_44,init_affine_44,64);

    int fD=fixed->data.shape[2],fH=fixed->data.shape[3],fW=fixed->data.shape[4];
    int mD=moving->data.shape[2],mH=moving->data.shape[3],mW=moving->data.shape[4];
    long fSp=(long)fD*fH*fW;

    mat44d pd,tm,cb;
    for(int i=0;i<4;i++)for(int j=0;j<4;j++)pd.m[i][j]=init_affine_44[i][j];
    mat44d_mul(&tm,&pd,&fixed->meta.torch2phy);
    mat44d_mul(&cb,&moving->meta.phy2torch,&tm);
    float ha[12];for(int i=0;i<3;i++)for(int j=0;j<4;j++)ha[i*4+j]=(float)cb.m[i][j];

    WGPUBufferUsage u=WGPUBufferUsage_Storage|WGPUBufferUsage_CopySrc|WGPUBufferUsage_CopyDst;
    WGPUBuffer d_aff=wgpu_create_buffer(48,u,"aff");wgpu_write_buffer(d_aff,0,ha,48);
    WGPUBuffer d_ff=wgpu_create_buffer(fSp*4,u,"ff");wgpu_write_buffer(d_ff,0,fixed->data.data,fSp*4);
    WGPUBuffer d_mf=wgpu_create_buffer((long)mD*mH*mW*4,u,"mf");wgpu_write_buffer(d_mf,0,moving->data.data,(long)mD*mH*mW*4);

    int gk_len=0,wk_len=0;
    WGPUBuffer d_gk=make_gauss_buf(opts.smooth_grad_sigma,2.0f,&gk_len);
    WGPUBuffer d_wk=make_gauss_buf(opts.smooth_warp_sigma,2.0f,&wk_len);

    /* GPU-side dual warps + optimizer states */
    WGPUBuffer d_fw=NULL,d_rv=NULL;
    WGPUBuffer d_fea=NULL,d_fev=NULL,d_rea=NULL,d_rev=NULL;
    int fwd_step=0,rev_step=0;
    int prev_dD=0,prev_dH=0,prev_dW=0;
    float b1=0.9f,b2=0.99f,eps=1e-8f;

    for(int si=0;si<opts.n_scales;si++){
        int scale=opts.scales[si],iters=opts.iterations[si];
        int dD=(scale>1)?fD/scale:fD,dH=(scale>1)?fH/scale:fH,dW=(scale>1)?fW/scale:fW;
        if(dD<8)dD=8;if(dH<8)dH=8;if(dW<8)dW=8;if(scale==1){dD=fD;dH=fH;dW=fW;}
        int mdD=mD,mdH=mH,mdW=mW; /* SyN: moving not downsampled */
        long sp=(long)dD*dH*dW,n3=sp*3;

        WGPUBuffer d_fd;
        if(scale>1) {
            d_fd=wgpu_downsample_image(d_ff,fD,fH,fW,dD,dH,dW,opts.downsample_mode);
        } else d_fd=d_ff;
        /* Blur moving image on GPU (matching Python _smooth_image_not_mask) */
        WGPUBuffer d_mb;
        int moving_blur_owned = 0;
        if (scale > 1) {
            float sig_d = 0.5f * (float)fD / (float)dD;
            float sig_h = 0.5f * (float)fH / (float)dH;
            float sig_w = 0.5f * (float)fW / (float)dW;
            size_t msz = (size_t)mD*mH*mW*sizeof(float);

            /* Copy moving to new buffer, blur in-place on GPU */
            d_mb = wgpu_create_buffer(msz, u, "mblur");
            {
                WGPUCommandEncoder enc=wgpuDeviceCreateCommandEncoder(g_wgpu.device,NULL);
                wgpuCommandEncoderCopyBufferToBuffer(enc,d_mf,0,d_mb,0,msz);
                WGPUCommandBuffer cmd=wgpuCommandEncoderFinish(enc,NULL);
                wgpuQueueSubmit(g_wgpu.queue,1,&cmd);
                wgpuCommandBufferRelease(cmd);wgpuCommandEncoderRelease(enc);
                wgpuDevicePoll(g_wgpu.device,1,NULL);
            }
            wgpu_blur_volume(d_mb, mD, mH, mW, sig_d, sig_h, sig_w);
            moving_blur_owned = 1;
        } else {
            d_mb = d_mf;
        }

        /* Init/reset warps — resize via trilinear interpolation (matching CUDA) */
        if(!d_fw){
            d_fw=wgpu_create_buffer(n3*4,u,"fw");wgpu_tensor_fill_buf(d_fw,0,(int)n3);
            d_rv=wgpu_create_buffer(n3*4,u,"rv");wgpu_tensor_fill_buf(d_rv,0,(int)n3);
        } else if(prev_dD!=dD||prev_dH!=dH||prev_dW!=dW){
            /* Resize warps: permute [D,H,W,3]→[3,D,H,W], trilinear resize, permute back */
            long prev_sp=(long)prev_dD*prev_dH*prev_dW;
            size_t prev_n3_sz=(size_t)prev_sp*3*sizeof(float);
            size_t new_n3_sz=(size_t)n3*sizeof(float);
            float *h_old=(float*)malloc(prev_n3_sz);
            float *h_3dhw=(float*)malloc(prev_n3_sz);
            float *h_3dhw_new=(float*)malloc(new_n3_sz);
            float *h_new=(float*)malloc(new_n3_sz);

            /* Resize fwd warp */
            wgpu_read_buffer(d_fw,0,h_old,prev_n3_sz);
            /* Permute [D,H,W,3] → [3,D,H,W] */
            for(long i=0;i<prev_sp;i++) for(int c=0;c<3;c++) h_3dhw[c*prev_sp+i]=h_old[i*3+c];
            /* Upload permuted, resize via GPU trilinear, download */
            WGPUBuffer b_src=wgpu_create_buffer_init(h_3dhw,prev_n3_sz,u,"wr_src");
            WGPUBuffer b_dst=wgpu_create_buffer(new_n3_sz,u,"wr_dst");
            wgpu_trilinear_resize(b_src,b_dst,1,3,prev_dD,prev_dH,prev_dW,dD,dH,dW,1);
            wgpu_read_buffer(b_dst,0,h_3dhw_new,new_n3_sz);
            wgpuBufferRelease(b_src);wgpuBufferRelease(b_dst);
            /* Permute [3,D,H,W] → [D,H,W,3] */
            for(long i=0;i<sp;i++) for(int c=0;c<3;c++) h_new[i*3+c]=h_3dhw_new[c*sp+i];
            wgpuBufferRelease(d_fw);
            d_fw=wgpu_create_buffer(n3*4,u,"fw");wgpu_write_buffer(d_fw,0,h_new,new_n3_sz);

            /* Resize rev warp */
            wgpu_read_buffer(d_rv,0,h_old,prev_n3_sz);
            for(long i=0;i<prev_sp;i++) for(int c=0;c<3;c++) h_3dhw[c*prev_sp+i]=h_old[i*3+c];
            b_src=wgpu_create_buffer_init(h_3dhw,prev_n3_sz,u,"wr_src");
            b_dst=wgpu_create_buffer(new_n3_sz,u,"wr_dst");
            wgpu_trilinear_resize(b_src,b_dst,1,3,prev_dD,prev_dH,prev_dW,dD,dH,dW,1);
            wgpu_read_buffer(b_dst,0,h_3dhw_new,new_n3_sz);
            wgpuBufferRelease(b_src);wgpuBufferRelease(b_dst);
            for(long i=0;i<sp;i++) for(int c=0;c<3;c++) h_new[i*3+c]=h_3dhw_new[c*sp+i];
            wgpuBufferRelease(d_rv);
            d_rv=wgpu_create_buffer(n3*4,u,"rv");wgpu_write_buffer(d_rv,0,h_new,new_n3_sz);

            free(h_old);free(h_3dhw);free(h_3dhw_new);free(h_new);
            if(d_fea){wgpuBufferRelease(d_fea);wgpuBufferRelease(d_fev);wgpuBufferRelease(d_rea);wgpuBufferRelease(d_rev);d_fea=NULL;}
        }
        if(!d_fea){
            d_fea=wgpu_create_buffer(n3*4,u,"fea");wgpu_tensor_fill_buf(d_fea,0,(int)n3);
            d_fev=wgpu_create_buffer(n3*4,u,"fev");wgpu_tensor_fill_buf(d_fev,0,(int)n3);
            d_rea=wgpu_create_buffer(n3*4,u,"rea");wgpu_tensor_fill_buf(d_rea,0,(int)n3);
            d_rev=wgpu_create_buffer(n3*4,u,"rev2");wgpu_tensor_fill_buf(d_rev,0,(int)n3);
        }
        prev_dD=dD;prev_dH=dH;prev_dW=dW;
        float hr=1.0f/(float)((dD>dH?(dD>dW?dD:dW):(dH>dW?dH:dW))-1);

        fprintf(stderr,"  SyN GPU scale %d: [%d,%d,%d] x %d iters\n",scale,dD,dH,dW,iters);

        WGPUBuffer d_ag=wgpu_create_buffer(n3*4,u,"ag");
        WGPUBuffer d_ig=wgpu_create_buffer(n3*4,u,"ig");
        wgpu_affine_grid_3d(d_aff,d_ag,1,dD,dH,dW);
        float id[12]={1,0,0,0,0,1,0,0,0,0,1,0};
        WGPUBuffer d_ia=wgpu_create_buffer(48,u,"ia");wgpu_write_buffer(d_ia,0,id,48);
        wgpu_affine_grid_3d(d_ia,d_ig,1,dD,dH,dW);wgpuBufferRelease(d_ia);

        WGPUBuffer d_sg1=wgpu_create_buffer(n3*4,u,"s1");
        WGPUBuffer d_sg2=wgpu_create_buffer(n3*4,u,"s2");
        WGPUBuffer d_m1=wgpu_create_buffer(sp*4,u,"m1");
        WGPUBuffer d_m2=wgpu_create_buffer(sp*4,u,"m2");
        WGPUBuffer d_g1=wgpu_create_buffer(sp*4,u,"g1");
        WGPUBuffer d_g2=wgpu_create_buffer(sp*4,u,"g2");
        WGPUBuffer d_gg1=wgpu_create_buffer(n3*4,u,"gg1");
        WGPUBuffer d_gg2=wgpu_create_buffer(n3*4,u,"gg2");
        WGPUBuffer d_ad1=wgpu_create_buffer(n3*4,u,"ad1");
        WGPUBuffer d_ad2=wgpu_create_buffer(n3*4,u,"ad2");
        WGPUBuffer d_scr=wgpu_create_buffer(n3*4,u,"scr");

        float pl=1e30f;int cc=0;
        for(int it=0;it<iters;it++){
            /* Batch forward pass */
            wgpu_begin_batch();
            wgpu_copy_buffer(d_ag, d_sg1, n3*4);
            wgpu_tensor_add_buf(d_sg1,d_fw,(int)n3);
            wgpu_copy_buffer(d_ig, d_sg2, n3*4);
            wgpu_tensor_add_buf(d_sg2,d_rv,(int)n3);
            wgpu_grid_sample_3d_fwd(d_mb,d_sg1,d_m1,1,1,mdD,mdH,mdW,dD,dH,dW);
            wgpu_grid_sample_3d_fwd(d_fd,d_sg2,d_m2,1,1,dD,dH,dW,dD,dH,dW);
            /* Flush forward batch, then fused CC (matching CUDA fused_cc.cu) */
            wgpu_flush();
            float loss = 0;
            int read_loss = (it % 10 == 0 || it == iters - 1);
            wgpu_fused_cc_loss(d_m1, d_m2, d_g1, d_g2, dD, dH, dW,
                               opts.cc_kernel_size, read_loss ? &loss : NULL);

            /* Backward: grid_sample_bwd for both warps */
            wgpu_grid_sample_3d_bwd(d_g1,d_mb,d_sg1,d_gg1,1,1,mdD,mdH,mdW,dD,dH,dW);
            wgpu_grid_sample_3d_bwd(d_g2,d_fd,d_sg2,d_gg2,1,1,dD,dH,dW,dD,dH,dW);

            /* WarpAdam on both warps */
            warp_adam_step(d_gg1,d_fw,d_fea,d_fev,d_ad1,d_scr,dD,dH,dW,
                           &fwd_step,b1,b2,eps,opts.lr,hr,d_gk,gk_len,d_wk,wk_len);
            warp_adam_step(d_gg2,d_rv,d_rea,d_rev,d_ad2,d_scr,dD,dH,dW,
                           &rev_step,b1,b2,eps,opts.lr,hr,d_gk,gk_len,d_wk,wk_len);

            if(it%50==0||it==iters-1) fprintf(stderr,"    iter %d/%d loss=%.6f\n",it,iters,loss);
            if(read_loss) {
                if(fabsf(loss-pl)<opts.tolerance){cc++;if(cc>=opts.max_tolerance_iters){fprintf(stderr,"    Converged at iter %d\n",it);break;}}
                else cc=0;
                pl=loss;
            }
        }

        wgpuBufferRelease(d_ag);wgpuBufferRelease(d_ig);
        wgpuBufferRelease(d_sg1);wgpuBufferRelease(d_sg2);
        wgpuBufferRelease(d_m1);wgpuBufferRelease(d_m2);
        wgpuBufferRelease(d_g1);wgpuBufferRelease(d_g2);
        wgpuBufferRelease(d_gg1);wgpuBufferRelease(d_gg2);
        wgpuBufferRelease(d_ad1);wgpuBufferRelease(d_ad2);wgpuBufferRelease(d_scr);
        if(scale>1) wgpuBufferRelease(d_fd);
        if(moving_blur_owned) wgpuBufferRelease(d_mb);
    }

    /* Evaluate: compose fwd with inverse(rev) */
    fprintf(stderr,"  Warp inverse: 550 iters at [%d,%d,%d]\n",fD,fH,fW);
    {
        long n3=fSp*3;
        /* Download warps for CPU warp inversion */
        float *h_fw=(float*)malloc(n3*4),*h_rv2=(float*)malloc(n3*4);
        if(d_fw && prev_dD==fD && prev_dH==fH && prev_dW==fW){
            wgpu_read_buffer(d_fw,0,h_fw,n3*4);
            wgpu_read_buffer(d_rv,0,h_rv2,n3*4);
        } else { memset(h_fw,0,n3*4); memset(h_rv2,0,n3*4); }

        float *inv_rv=(float*)malloc(n3*4);
        cpu_warp_inverse(h_rv2,inv_rv,fD,fH,fW,550);

        /* Compose on CPU: composed = inv_rv + interp(fwd, identity + inv_rv) */
        float *composed=(float*)malloc(n3*4);
        /* Inline compose */
        for(int d=0;d<fD;d++){
            float nz=(fD>1)?(2.0f*d/(fD-1)-1.0f):0.0f;
            for(int h=0;h<fH;h++){
                float ny=(fH>1)?(2.0f*h/(fH-1)-1.0f):0.0f;
                for(int w=0;w<fW;w++){
                    float nx=(fW>1)?(2.0f*w/(fW-1)-1.0f):0.0f;
                    int idx=((d*fH+h)*fW+w)*3;
                    float ux=inv_rv[idx],uy=inv_rv[idx+1],uz=inv_rv[idx+2];
                    float sx=nx+ux,sy=ny+uy,sz=nz+uz;
                    float ix=(sx+1)*0.5f*(fW-1),iy=(sy+1)*0.5f*(fH-1),iz=(sz+1)*0.5f*(fD-1);
                    int x0=(int)floorf(ix),y0=(int)floorf(iy),z0=(int)floorf(iz);
                    float fx=ix-x0,fy=iy-y0,fz2=iz-z0;
                    float v[3]={0,0,0};
                    #define FW(dd,hh,ww,c) ((dd)>=0&&(dd)<fD&&(hh)>=0&&(hh)<fH&&(ww)>=0&&(ww)<fW?\
                        h_fw[((dd)*fH+(hh))*fW*3+(ww)*3+(c)]:0.0f)
                    float ws[8]={(1-fx)*(1-fy)*(1-fz2),fx*(1-fy)*(1-fz2),(1-fx)*fy*(1-fz2),fx*fy*(1-fz2),
                                 (1-fx)*(1-fy)*fz2,fx*(1-fy)*fz2,(1-fx)*fy*fz2,fx*fy*fz2};
                    int dz[8]={z0,z0,z0,z0,z0+1,z0+1,z0+1,z0+1};
                    int dy[8]={y0,y0,y0+1,y0+1,y0,y0,y0+1,y0+1};
                    int dx[8]={x0,x0+1,x0,x0+1,x0,x0+1,x0,x0+1};
                    for(int k=0;k<8;k++)for(int c=0;c<3;c++)v[c]+=ws[k]*FW(dz[k],dy[k],dx[k],c);
                    #undef FW
                    composed[idx]=ux+v[0];composed[idx+1]=uy+v[1];composed[idx+2]=uz+v[2];
                }
            }
        }

        /* Upload composed, add to affine grid, sample */
        WGPUBuffer d_base=wgpu_create_buffer(n3*4,u,"eb");
        WGPUBuffer d_cw=wgpu_create_buffer(n3*4,u,"cw");
        WGPUBuffer d_sg=wgpu_create_buffer(n3*4,u,"esg");
        WGPUBuffer d_mov=wgpu_create_buffer(fSp*4,u,"em");

        wgpu_affine_grid_3d(d_aff,d_base,1,fD,fH,fW);
        wgpu_write_buffer(d_cw,0,composed,n3*4);

        {WGPUCommandEncoder e=wgpuDeviceCreateCommandEncoder(g_wgpu.device,NULL);
         wgpuCommandEncoderCopyBufferToBuffer(e,d_base,0,d_sg,0,n3*4);
         WGPUCommandBuffer c=wgpuCommandEncoderFinish(e,NULL);
         wgpuQueueSubmit(g_wgpu.queue,1,&c);wgpuCommandBufferRelease(c);wgpuCommandEncoderRelease(e);
         wgpuDevicePoll(g_wgpu.device,1,NULL);}
        wgpu_tensor_add_buf(d_sg,d_cw,(int)n3);

        wgpu_grid_sample_3d_fwd(d_mf,d_sg,d_mov,1,1,mD,mH,mW,fD,fH,fW);
        wgpu_cc_loss_3d_raw(d_mov,d_ff,NULL,fD,fH,fW,9,&result->ncc_loss);

        int shape[5]={1,1,fD,fH,fW};
        tensor_alloc(&result->moved,5,shape,DTYPE_FLOAT32,DEVICE_CPU);
        wgpu_read_buffer(d_mov,0,result->moved.data,fSp*4);

        wgpuBufferRelease(d_base);wgpuBufferRelease(d_cw);wgpuBufferRelease(d_sg);wgpuBufferRelease(d_mov);
        free(h_fw);free(h_rv2);free(inv_rv);free(composed);
    }

    if(d_fw)wgpuBufferRelease(d_fw);
    if(d_rv)wgpuBufferRelease(d_rv);
    if(d_fea){wgpuBufferRelease(d_fea);wgpuBufferRelease(d_fev);wgpuBufferRelease(d_rea);wgpuBufferRelease(d_rev);}
    if(d_gk)wgpuBufferRelease(d_gk);
    if(d_wk)wgpuBufferRelease(d_wk);
    wgpuBufferRelease(d_aff);wgpuBufferRelease(d_ff);wgpuBufferRelease(d_mf);
    return 0;
}
