// Regular local-CC element-wise product.
struct MulP { n:u32, _p0:u32, _p1:u32, _p2:u32, }
@group(0) @binding(0) var<storage, read> ma: array<f32>;
@group(0) @binding(1) var<storage, read> mb: array<f32>;
@group(0) @binding(2) var<storage, read_write> mc: array<f32>;
@group(0) @binding(3) var<uniform> mp: MulP;
@compute @workgroup_size(256)
fn multiply(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u; if (i >= mp.n) { return; }
    mc[i] = ma[i] * mb[i];
}
