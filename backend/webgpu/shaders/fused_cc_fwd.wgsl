// Per-workgroup reduction of local normalized cross correlation.

struct Params {
    n: u32, kernel_volume: u32, nr: f32, dr: f32,
}

@group(0) @binding(0) var<storage, read> interm: array<f32>;
@group(0) @binding(1) var<storage, read_write> partial_sum: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

var<workgroup> sums: array<f32, 256>;

@compute @workgroup_size(256)
fn fcc_fwd(@builtin(global_invocation_id) gid: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(workgroup_id) wid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    var value = 0.0f;
    if (i < p.n) {
        let mu = interm[i];
        let rho = interm[i + p.n];
        let mu2 = interm[i + 2u * p.n];
        let rho2 = interm[i + 3u * p.n];
        let murho = interm[i + 4u * p.n];
        let kv = f32(p.kernel_volume);
        let a = kv * (murho - mu * rho);
        let b = max(kv * (mu2 - mu * mu), p.dr);
        let c = max(kv * (rho2 - rho * rho), p.dr);
        value = clamp((a * a + p.nr) / (b * c + p.dr), -1.0, 1.0);
    }

    sums[lid.x] = value;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) {
            sums[lid.x] += sums[lid.x + stride];
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        partial_sum[wid.x + wid.y * nwg.x] = sums[0];
    }
}
