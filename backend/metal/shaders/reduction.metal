#include <metal_stdlib>
using namespace metal;

#define THREADGROUP_SIZE 256

struct ReduceParams {
    uint n;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// Sum reduction: each threadgroup produces one partial sum
kernel void reduce_sum(const device float *input [[buffer(0)]],
                       constant ReduceParams &p [[buffer(1)]],
                       device float *output [[buffer(2)]],
                       uint gid [[thread_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float shared_data[THREADGROUP_SIZE];

    shared_data[tid] = (gid < p.n) ? input[gid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint s = THREADGROUP_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[wid] = shared_data[0];
    }
}

// Max L2 norm reduction for [D,H,W,3] displacement fields
// Each thread computes sqrt(dx^2+dy^2+dz^2) for one voxel, then reduces to max
struct MaxL2Params {
    uint spatial;  // D*H*W
    uint _pad;
    float eps;
    float _pad1;
};

kernel void max_l2_norm(const device float *data [[buffer(0)]],
                        device float *output [[buffer(1)]],
                        constant MaxL2Params &p [[buffer(2)]],
                        uint gid [[thread_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float shared_max[THREADGROUP_SIZE];

    float val = 0.0f;
    if (gid < p.spatial) {
        uint i = gid * 3;
        float dx = data[i], dy = data[i + 1], dz = data[i + 2];
        val = sqrt(dx * dx + dy * dy + dz * dz);
    }
    shared_max[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = THREADGROUP_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s && shared_max[tid + s] > shared_max[tid]) {
            shared_max[tid] = shared_max[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[wid] = shared_max[0];
    }
}

// Max reduction (find max value in buffer) - used by MI loss normalization
kernel void reduce_max(const device float *input [[buffer(0)]],
                       constant ReduceParams &p [[buffer(1)]],
                       device float *output [[buffer(2)]],
                       uint gid [[thread_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float shared_data[THREADGROUP_SIZE];

    shared_data[tid] = (gid < p.n) ? input[gid] : -1e38f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = THREADGROUP_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s && shared_data[tid + s] > shared_data[tid]) {
            shared_data[tid] = shared_data[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[wid] = shared_data[0];
    }
}
