// blur_dhw3.wgsl - Separable convolution on [D,H,W,3] displacement field
//
// Single axis per dispatch. Ping-pong between input and output buffers.
// Kernel weights passed via storage buffer.

struct Params {
    D: u32, H: u32, W: u32,
    klen: u32,    // kernel length
    axis: u32,    // 0=D, 1=H, 2=W
    _p0: u32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;     // [D*H*W*3]
@group(0) @binding(1) var<storage, read_write> output: array<f32>; // [D*H*W*3]
@group(0) @binding(2) var<storage, read> kernel: array<f32>;     // [klen]
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(256)
fn conv1d_dhw3(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.D * p.H * p.W;
    if (idx >= total) { return; }

    let w = idx % p.W;
    var tmp = idx / p.W;
    let h = tmp % p.H;
    let d = tmp / p.H;

    let r = i32(p.klen / 2u);
    let dims = array<u32, 3>(p.D, p.H, p.W);
    let strides = array<i32, 3>(i32(p.H * p.W), i32(p.W), 1);
    let coords = array<i32, 3>(i32(d), i32(h), i32(w));

    for (var c = 0u; c < 3u; c = c + 1u) {
        var sum = 0.0f;
        for (var k = 0; k < i32(p.klen); k = k + 1) {
            let ci = coords[p.axis] + k - r;
            if (ci >= 0 && ci < i32(dims[p.axis])) {
                let src_idx = u32(i32(idx) + (ci - coords[p.axis]) * strides[p.axis]);
                sum += input[src_idx * 3u + c] * kernel[u32(k)];
            }
        }
        output[idx * 3u + c] = sum;
    }
}
