#include <metal_stdlib>
using namespace metal;

struct PermuteParams {
    uint D;
    uint H;
    uint W;
    uint _pad;
};

// [3,D,H,W] -> [D,H,W,3]
kernel void permute_3dhw_dhw3(const device float *input [[buffer(0)]],
                               device float *output [[buffer(1)]],
                               constant PermuteParams &p [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    uint spatial = p.D * p.H * p.W;
    if (gid >= spatial) return;
    output[gid * 3 + 0] = input[gid];
    output[gid * 3 + 1] = input[spatial + gid];
    output[gid * 3 + 2] = input[2 * spatial + gid];
}

// [D,H,W,3] -> [3,D,H,W]
kernel void permute_dhw3_3dhw(const device float *input [[buffer(0)]],
                               device float *output [[buffer(1)]],
                               constant PermuteParams &p [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    uint spatial = p.D * p.H * p.W;
    if (gid >= spatial) return;
    output[gid] = input[gid * 3 + 0];
    output[spatial + gid] = input[gid * 3 + 1];
    output[2 * spatial + gid] = input[gid * 3 + 2];
}
