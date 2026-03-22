// fused_cc_bwd_modify.wgsl — Compute gradient multipliers from filtered intermediates
// Matching CUDA fused_cc.cu fcc_bwd_modify_kernel

struct Params { n: u32, kv: u32, nr: f32, dr: f32, gO: f32, cgt: u32, _p0: u32, _p1: u32, }

@group(0) @binding(0) var<storage, read_write> interm: array<f32>;
@group(0) @binding(1) var<storage, read> pred: array<f32>;
@group(0) @binding(2) var<storage, read> tgt: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(256)
fn bwd_modify(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= p.n) { return; }
    let n = p.n;
    let kv = f32(p.kv);
    let nr = p.nr;
    let dr = p.dr;
    let gO = p.gO;

    let mu    = interm[i];
    let rho   = interm[i + n];
    let mu2   = interm[i + 2u*n];
    let rho2  = interm[i + 3u*n];
    let murho = interm[i + 4u*n];

    let A = kv * (murho - mu * rho);
    var B = kv * (mu2 - mu * mu);
    var C = kv * (rho2 - rho * rho);

    let D = 2.0 * gO * A / (B * C + dr);

    B += dr;
    C += dr;

    interm[i]         = D;
    interm[i + n]     = D * A / B;
    interm[i + 2u*n]  = D * (A / B * mu - rho);

    if (p.cgt != 0u) {
        interm[i + 3u*n]  = D * A / C;
        interm[i + 4u*n]  = D * (A / C * rho - mu);
    }
}
