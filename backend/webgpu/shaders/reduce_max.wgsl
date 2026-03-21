// reduce_max.wgsl - Find maximum value in a float buffer
//
// Each workgroup reduces a chunk to a partial max, written to output.
// Host reads partials and does final max (matching reduction.wgsl pattern).

struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce_max(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let i = gid.x;
    let tid = lid.x;

    if (i < params.n) {
        shared_data[tid] = input[i];
    } else {
        shared_data[tid] = -1e38;  // -inf for max reduction
    }
    workgroupBarrier();

    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            if (shared_data[tid + s] > shared_data[tid]) {
                shared_data[tid] = shared_data[tid + s];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[wid.x] = shared_data[0];
    }
}
