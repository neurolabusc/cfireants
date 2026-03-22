// affine_grid.wgsl - Generate sampling grid from [B,3,4] affine matrix
//
// Mirrors cuda_affine_grid_3d: normalized coords [-1,1] -> transformed coords.

struct AffineGridParams {
    B: u32, D: u32, H: u32, W: u32,
}

@group(0) @binding(0) var<storage, read> affine: array<f32>;   // [B, 12]
@group(0) @binding(1) var<storage, read_write> grid: array<f32>; // [B, D, H, W, 3]
@group(0) @binding(2) var<uniform> p: AffineGridParams;

@compute @workgroup_size(256)
fn affine_grid(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 256u;
    let total = p.B * p.D * p.H * p.W;
    if (idx >= total) { return; }

    let w = idx % p.W;
    var tmp = idx / p.W;
    let h = tmp % p.H;
    tmp = tmp / p.H;
    let d = tmp % p.D;
    let b = tmp / p.D;

    var nz: f32;
    var ny: f32;
    var nx: f32;
    if (p.D > 1u) { nz = 2.0 * f32(d) / f32(p.D - 1u) - 1.0; } else { nz = 0.0; }
    if (p.H > 1u) { ny = 2.0 * f32(h) / f32(p.H - 1u) - 1.0; } else { ny = 0.0; }
    if (p.W > 1u) { nx = 2.0 * f32(w) / f32(p.W - 1u) - 1.0; } else { nx = 0.0; }

    let A = b * 12u;
    let ox = affine[A+0u]*nx + affine[A+1u]*ny + affine[A+2u]*nz + affine[A+3u];
    let oy = affine[A+4u]*nx + affine[A+5u]*ny + affine[A+6u]*nz + affine[A+7u];
    let oz = affine[A+8u]*nx + affine[A+9u]*ny + affine[A+10u]*nz + affine[A+11u];

    let gi = idx * 3u;
    grid[gi + 0u] = ox;
    grid[gi + 1u] = oy;
    grid[gi + 2u] = oz;
}
