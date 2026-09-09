// In-place AXPY kernel. This file is the sole source embedded by CMake.
struct Params {
    n: u32,
    _pad0: u32,
    value: f32,
    _pad1: f32,
}
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> x_data: array<f32>;
@compute @workgroup_size(256)
fn axpy(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= params.n) { return; }
    data[i] = data[i] + params.value * x_data[i];
}
