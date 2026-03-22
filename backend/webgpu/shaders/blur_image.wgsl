// blur_image.wgsl - Separable convolution on [D,H,W] image (single channel)
//
// Single axis per dispatch. Ping-pong between input and output buffers.
// Kernel weights passed via storage buffer.

struct Params {
    D: u32, H: u32, W: u32,
    klen: u32,
    axis: u32,    // 0=D, 1=H, 2=W
    _p0: u32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> kernel: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

fn get_dim(D: u32, H: u32, W: u32, axis: u32) -> i32 {
    if (axis == 0u) { return i32(D); }
    if (axis == 1u) { return i32(H); }
    return i32(W);
}

fn get_stride(H: u32, W: u32, axis: u32) -> i32 {
    if (axis == 0u) { return i32(H * W); }
    if (axis == 1u) { return i32(W); }
    return 1;
}

@compute @workgroup_size(256)
fn conv1d_image(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 256u;
    let total = p.D * p.H * p.W;
    if (idx >= total) { return; }

    let w = idx % p.W;
    var tmp = idx / p.W;
    let h = tmp % p.H;
    let d = tmp / p.H;

    let r = i32(p.klen / 2u);
    var coord: i32;
    if (p.axis == 0u) { coord = i32(d); }
    else if (p.axis == 1u) { coord = i32(h); }
    else { coord = i32(w); }

    let dim = get_dim(p.D, p.H, p.W, p.axis);
    let stride = get_stride(p.H, p.W, p.axis);

    var sum = 0.0f;
    for (var k = 0; k < i32(p.klen); k = k + 1) {
        let ci = coord + k - r;
        if (ci >= 0 && ci < dim) {
            let src_idx = u32(i32(idx) + (ci - coord) * stride);
            sum += input[src_idx] * kernel[u32(k)];
        }
    }
    output[idx] = sum;
}
