// Clear the packed MI state and find max(pred, target) without a readback.
// Non-negative float bit patterns preserve numeric ordering for atomicMax.

const STATE_COUNT: u32 = 1089u;

struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, }

@group(0) @binding(0) var<storage, read> pred: array<f32>;
@group(0) @binding(1) var<storage, read> target_image: array<f32>;
@group(0) @binding(2) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p: Params;

var<workgroup> maxima: array<f32, 256>;

@compute @workgroup_size(256)
fn clear_state(@builtin(local_invocation_id) lid: vec3<u32>) {
    for (var i = lid.x; i < STATE_COUNT; i += 256u) {
        atomicStore(&state[i], 0u);
    }
}

@compute @workgroup_size(256)
fn reduce_max(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    var value = 0.0f;
    if (i < p.n) {
        value = max(max(pred[i], target_image[i]), 0.0);
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
        atomicMax(&state[0], bitcast<u32>(maxima[0]));
    }
}
