// elementwise.wgsl - Element-wise tensor operations
//
// Each entry point operates on a flat array of f32 values.
// Params are passed via a uniform buffer at binding 1.

struct Params {
    n: u32,        // number of elements
    _pad0: u32,
    value: f32,    // scalar operand (fill value, scale factor, alpha)
    _pad1: f32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

// fill: data[i] = value
@compute @workgroup_size(256)
fn fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    data[i] = params.value;
}

// scale: data[i] *= value
@compute @workgroup_size(256)
fn scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    data[i] = data[i] * params.value;
}

// axpy: data[i] += alpha * x[i]
// x is at binding 2
@group(0) @binding(2) var<storage, read> x_data: array<f32>;

@compute @workgroup_size(256)
fn axpy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    data[i] = data[i] + params.value * x_data[i];
}
