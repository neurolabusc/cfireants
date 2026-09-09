// Regular local-CC value reduction and gradient source terms.
struct P { n: u32, cg: u32, nr: f32, dr: f32, }
@group(0) @binding(0) var<storage, read> ps: array<f32>;
@group(0) @binding(1) var<storage, read> ts: array<f32>;
@group(0) @binding(2) var<storage, read> p2s: array<f32>;
@group(0) @binding(3) var<storage, read> t2s: array<f32>;
@group(0) @binding(4) var<storage, read> tps: array<f32>;
@group(0) @binding(5) var<storage, read_write> ncc: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_sources: array<f32>;
@group(0) @binding(7) var<uniform> p: P;
var<workgroup> sums: array<f32, 256>;
@compute @workgroup_size(256)
fn ncc_grad(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u; var nc = 0.0;
    if (i < p.n) {
        let psi = ps[i]; let tsi = ts[i];
        let cross = tps[i] - psi * tsi;
        var pv = p2s[i] - psi * psi; var tv = t2s[i] - tsi * tsi;
        if (pv < p.dr) { pv = p.dr; } if (tv < p.dr) { tv = p.dr; }
        let f = cross * cross + p.nr; let g = pv * tv + p.dr;
        nc = clamp(f / g, -1.0, 1.0);
        if (p.cg != 0u) {
            let g2 = g * g;
            grad_sources[i] = (-2.0 * cross * tsi * g + 2.0 * f * psi * tv) / g2;
            grad_sources[i + p.n] = -f * tv / g2;
            grad_sources[i + 2u * p.n] = 2.0 * cross * g / g2;
        }
    }
    sums[lid.x] = nc; workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) { if (lid.x < s) { sums[lid.x] += sums[lid.x+s]; } workgroupBarrier(); }
    if (lid.x == 0u) { ncc[wid.x + wid.y*nwg.x] = sums[0]; }
}
