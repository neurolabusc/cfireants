// reduction.wgsl - Reduction operations (sum)
//
// Two-pass reduction: each workgroup reduces a chunk, writes partial result.
// Host reads partials and does final sum (matching CUDA pattern).

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
fn reduce_sum(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let i = gid.x;
    let tid = lid.x;

    // Load element or zero
    if (i < params.n) {
        shared_data[tid] = input[i];
    } else {
        shared_data[tid] = 0.0;
    }
    workgroupBarrier();

    // Tree reduction in shared memory
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + s];
        }
        workgroupBarrier();
    }

    // Write partial sum
    if (tid == 0u) {
        output[wid.x] = shared_data[0];
    }
}
