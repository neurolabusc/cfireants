// fused_cc_bwd_grads.wgsl — Compute final per-voxel gradients
// Matching CUDA fused_cc.cu fcc_bwd_grads_kernel

struct Params { n: u32, cgt: u32, _p0: u32, _p1: u32, }

@group(0) @binding(0) var<storage, read> interm: array<f32>;
@group(0) @binding(1) var<storage, read> pred: array<f32>;
@group(0) @binding(2) var<storage, read> tgt: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_pred: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_tgt: array<f32>;
@group(0) @binding(5) var<uniform> p: Params;

@compute @workgroup_size(256)
fn bwd_grads(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= p.n) { return; }
    let n = p.n;

    let I = pred[i];
    let J = tgt[i];

    let gini_a  = interm[i];
    let gini_b  = interm[i + n];
    let gini_mu = interm[i + 2u*n];

    grad_pred[i] = gini_a * J - gini_b * I + gini_mu;

    if (p.cgt != 0u) {
        let gini_c   = interm[i + 3u*n];
        let gini_mu2 = interm[i + 4u*n];
        grad_tgt[i] = gini_a * I - gini_c * J + gini_mu2;
    }
}
