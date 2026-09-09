// Generate a normalized sampling grid from a 3x4 affine matrix.
struct P { B: u32, D: u32, H: u32, W: u32, }
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> grid: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn affine_grid(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 256u;
    if (idx >= p.B*p.D*p.H*p.W) { return; }
    let w = idx%p.W; var tmp = idx/p.W;
    let h = tmp%p.H; tmp=tmp/p.H;
    let d = tmp%p.D; let b = tmp/p.D;
    var nz: f32; var ny: f32; var nx: f32;
    if (p.D>1u) { nz = 2.0*f32(d)/f32(p.D-1u)-1.0; } else { nz = 0.0; }
    if (p.H>1u) { ny = 2.0*f32(h)/f32(p.H-1u)-1.0; } else { ny = 0.0; }
    if (p.W>1u) { nx = 2.0*f32(w)/f32(p.W-1u)-1.0; } else { nx = 0.0; }
    let a = b*12u;
    grid[idx*3u] = A[a]*nx+A[a+1u]*ny+A[a+2u]*nz+A[a+3u];
    grid[idx*3u+1u] = A[a+4u]*nx+A[a+5u]*ny+A[a+6u]*nz+A[a+7u];
    grid[idx*3u+2u] = A[a+8u]*nx+A[a+9u]*ny+A[a+10u]*nz+A[a+11u];
}
