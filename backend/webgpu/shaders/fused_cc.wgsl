// fused_cc.wgsl — Create intermediates [I, J, I², J², IJ]
// Matching CUDA fused_cc.cu fcc_create_intermediates_kernel

struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, }

@group(0) @binding(0) var<storage, read> pred: array<f32>;
@group(0) @binding(1) var<storage, read> tgt: array<f32>;
@group(0) @binding(2) var<storage, read_write> interm: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(256)
fn create_intermediates(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= p.n) { return; }
    let n = p.n;
    let I = pred[i];
    let J = tgt[i];
    interm[i]         = I;
    interm[i + n]     = J;
    interm[i + 2u*n]  = I * I;
    interm[i + 3u*n]  = J * J;
    interm[i + 4u*n]  = I * J;
}
