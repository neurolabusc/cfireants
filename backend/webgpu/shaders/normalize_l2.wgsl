// Normalize a [spatial,3] vector field by max L2 norm entirely on GPU.

struct Params {
    spatial: u32, n: u32, eps: f32, factor: f32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> max_bits: atomic<u32>;
@group(0) @binding(2) var<uniform> p: Params;

var<workgroup> maxima: array<f32, 256>;

@compute @workgroup_size(1)
fn clear_max() {
    atomicStore(&max_bits, 0u);
}

@compute @workgroup_size(256)
fn reduce_max_l2(@builtin(global_invocation_id) gid: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>,
                 @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    var value = 0.0f;
    if (i < p.spatial) {
        let x = data[3u * i];
        let y = data[3u * i + 1u];
        let z = data[3u * i + 2u];
        value = sqrt(x * x + y * y + z * z);
    }
    maxima[lid.x] = value;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) {
            maxima[lid.x] = max(maxima[lid.x], maxima[lid.x + stride]);
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        atomicMax(&max_bits, bitcast<u32>(maxima[0]));
    }
}

@compute @workgroup_size(256)
fn apply_scale(@builtin(global_invocation_id) gid: vec3<u32>,
               @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= p.n) { return; }
    let maximum = bitcast<f32>(atomicLoad(&max_bits));
    let denom = max(p.eps + maximum, 1.0);
    data[i] *= p.factor / denom;
}
