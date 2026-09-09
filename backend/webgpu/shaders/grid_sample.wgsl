// Trilinear 3D grid sampling, forward pass.
struct P { B: u32, C: u32, iD: u32, iH: u32, iW: u32, oD: u32, oH: u32, oW: u32, }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> grid: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
fn unnorm(x: f32, s: u32) -> f32 { return (x + 1.0) * 0.5 * f32(s - 1u); }
fn get_inp(b: u32, c: u32, d: i32, h: i32, w: i32) -> f32 {
    if (d < 0 || d >= i32(p.iD) || h < 0 || h >= i32(p.iH) || w < 0 || w >= i32(p.iW)) { return 0.0; }
    return input[((b * p.C + c) * p.iD + u32(d)) * p.iH * p.iW + u32(h) * p.iW + u32(w)];
}
@compute @workgroup_size(256)
fn grid_sample_fwd(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 256u;
    if (idx >= p.B * p.oD * p.oH * p.oW) { return; }
    let w = idx % p.oW; var tmp = idx / p.oW;
    let h = tmp % p.oH; tmp = tmp / p.oH;
    let d = tmp % p.oD; let b = tmp / p.oD;
    let gi = idx * 3u;
    let ix = unnorm(grid[gi], p.iW); let iy = unnorm(grid[gi+1u], p.iH); let iz = unnorm(grid[gi+2u], p.iD);
    let x0 = i32(floor(ix)); let y0 = i32(floor(iy)); let z0 = i32(floor(iz));
    let fx = ix - f32(x0); let fy = iy - f32(y0); let fz = iz - f32(z0);
    let os = p.oD * p.oH * p.oW;
    var ob = (b * p.C * p.oD + d) * p.oH * p.oW + h * p.oW + w;
    for (var c = 0u; c < p.C; c++) {
        let v = (1.0-fx)*(1.0-fy)*(1.0-fz)*get_inp(b,c,z0,y0,x0)
              + fx*(1.0-fy)*(1.0-fz)*get_inp(b,c,z0,y0,x0+1)
              + (1.0-fx)*fy*(1.0-fz)*get_inp(b,c,z0,y0+1,x0)
              + fx*fy*(1.0-fz)*get_inp(b,c,z0,y0+1,x0+1)
              + (1.0-fx)*(1.0-fy)*fz*get_inp(b,c,z0+1,y0,x0)
              + fx*(1.0-fy)*fz*get_inp(b,c,z0+1,y0,x0+1)
              + (1.0-fx)*fy*fz*get_inp(b,c,z0+1,y0+1,x0)
              + fx*fy*fz*get_inp(b,c,z0+1,y0+1,x0+1);
        output[ob + c * os] = v;
    }
}
