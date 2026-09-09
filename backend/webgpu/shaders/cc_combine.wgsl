// Combine adjoint-filtered regular local-CC gradient terms.
struct P { n: u32, _p: u32, ic: f32, _p2: f32, }
@group(0) @binding(0) var<storage, read> ap: array<f32>;
@group(0) @binding(1) var<storage, read> ap2: array<f32>;
@group(0) @binding(2) var<storage, read> atp: array<f32>;
@group(0) @binding(3) var<storage, read> cP: array<f32>;
@group(0) @binding(4) var<storage, read> cT: array<f32>;
@group(0) @binding(5) var<storage, read_write> go: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
@compute @workgroup_size(256)
fn combine(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u; if (i >= p.n) { return; }
    go[i] = -p.ic * (ap[i] + 2.0 * cP[i] * ap2[i] + cT[i] * atp[i]);
}
