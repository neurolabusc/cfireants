// affine_grid_bwd.wgsl - Reduce grad_grid [D,H,W,3] to dL/dA [12]
//
// Each workgroup produces a partial sum of 12 values.
// Final reduction done on CPU (same pattern as CUDA).

struct Params { D: u32, H: u32, W: u32, _pad: u32, }

@group(0) @binding(0) var<storage, read> grad_grid: array<f32>;  // [D*H*W*3]
@group(0) @binding(1) var<storage, read_write> partial: array<f32>; // [n_blocks * 12]
@group(0) @binding(2) var<uniform> p: Params;

var<workgroup> sdata: array<f32, 3072>;  // 256 * 12

@compute @workgroup_size(256)
fn affine_grid_bwd(@builtin(global_invocation_id) gid: vec3<u32>,
                   @builtin(local_invocation_id) lid: vec3<u32>,
                   @builtin(workgroup_id) wid: vec3<u32>) {
    let idx = gid.x;
    let tid = lid.x;
    let total = p.D * p.H * p.W;

    // Initialize shared memory
    for (var k = 0u; k < 12u; k = k + 1u) { sdata[tid * 12u + k] = 0.0; }

    if (idx < total) {
        let w = idx % p.W;
        var tmp = idx / p.W;
        let h = tmp % p.H;
        let d = tmp / p.H;

        var nz: f32; var ny: f32; var nx: f32;
        if (p.D > 1u) { nz = 2.0 * f32(d) / f32(p.D - 1u) - 1.0; } else { nz = 0.0; }
        if (p.H > 1u) { ny = 2.0 * f32(h) / f32(p.H - 1u) - 1.0; } else { ny = 0.0; }
        if (p.W > 1u) { nx = 2.0 * f32(w) / f32(p.W - 1u) - 1.0; } else { nx = 0.0; }

        let coord = array<f32, 4>(nx, ny, nz, 1.0);
        let gi = idx * 3u;
        let gg = array<f32, 3>(grad_grid[gi], grad_grid[gi + 1u], grad_grid[gi + 2u]);

        for (var i = 0u; i < 3u; i = i + 1u) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                sdata[tid * 12u + i * 4u + j] = gg[i] * coord[j];
            }
        }
    }
    workgroupBarrier();

    // Tree reduction
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            for (var k = 0u; k < 12u; k = k + 1u) {
                sdata[tid * 12u + k] = sdata[tid * 12u + k] + sdata[(tid + s) * 12u + k];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        for (var k = 0u; k < 12u; k = k + 1u) {
            partial[wid.x * 12u + k] = sdata[k];
        }
    }
}
