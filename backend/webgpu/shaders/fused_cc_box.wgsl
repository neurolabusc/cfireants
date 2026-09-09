// Filter all five fused-CC channels in one dispatch.
// The C driver ping-pongs the packed [5,D,H,W] buffers for the three axes.

struct Params {
    D: u32, H: u32, W: u32, ks: u32,
    axis: u32, channels: u32, _p0: u32, _p1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn box_filter_packed(@builtin(global_invocation_id) gid: vec3<u32>,
                     @builtin(num_workgroups) nwg: vec3<u32>) {
    let flat = gid.x + gid.y * nwg.x * 256u;
    let spatial = p.D * p.H * p.W;
    if (flat >= spatial * p.channels) { return; }

    let channel = flat / spatial;
    let idx = flat - channel * spatial;
    let w = idx % p.W;
    let dh = idx / p.W;
    let h = dh % p.H;
    let d = dh / p.H;
    let base = channel * spatial;
    let radius = i32(p.ks / 2u);
    var sum = 0.0f;

    if (p.axis == 2u) {
        for (var k = -radius; k <= radius; k = k + 1) {
            let ww = i32(w) + k;
            if (ww >= 0 && ww < i32(p.W)) {
                sum += input[base + d * p.H * p.W + h * p.W + u32(ww)];
            }
        }
    } else if (p.axis == 1u) {
        for (var k = -radius; k <= radius; k = k + 1) {
            let hh = i32(h) + k;
            if (hh >= 0 && hh < i32(p.H)) {
                sum += input[base + d * p.H * p.W + u32(hh) * p.W + w];
            }
        }
    } else {
        for (var k = -radius; k <= radius; k = k + 1) {
            let dd = i32(d) + k;
            if (dd >= 0 && dd < i32(p.D)) {
                sum += input[base + u32(dd) * p.H * p.W + h * p.W + w];
            }
        }
    }
    output[flat] = sum / f32(p.ks);
}
