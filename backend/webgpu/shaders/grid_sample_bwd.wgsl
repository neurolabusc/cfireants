// Trilinear 3D grid sampling, gradient with respect to the grid.
struct P { B: u32, C: u32, iD: u32, iH: u32, iW: u32, oD: u32, oH: u32, oW: u32, }
@group(0) @binding(0) var<storage, read> go: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> grid: array<f32>;
@group(0) @binding(3) var<storage, read_write> gg: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
fn unnorm(x: f32, s: u32) -> f32 { return (x + 1.0) * 0.5 * f32(s - 1u); }
fn gi(b: u32, c: u32, d: i32, h: i32, w: i32) -> f32 {
    if (d<0||d>=i32(p.iD)||h<0||h>=i32(p.iH)||w<0||w>=i32(p.iW)){return 0.0;}
    return input[((b*p.C+c)*p.iD+u32(d))*p.iH*p.iW+u32(h)*p.iW+u32(w)];
}
@compute @workgroup_size(256)
fn grid_sample_bwd(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 256u;
    if (idx >= p.B*p.oD*p.oH*p.oW) { return; }
    let w = idx%p.oW; var tmp = idx/p.oW;
    let h = tmp%p.oH; tmp = tmp/p.oH;
    let d = tmp%p.oD; let b = tmp/p.oD;
    let gx = grid[idx*3u]; let gy = grid[idx*3u+1u]; let gz = grid[idx*3u+2u];
    let ix = unnorm(gx,p.iW); let iy = unnorm(gy,p.iH); let iz = unnorm(gz,p.iD);
    let x0=i32(floor(ix)); let y0=i32(floor(iy)); let z0=i32(floor(iz));
    let fx=ix-f32(x0); let fy=iy-f32(y0); let fz=iz-f32(z0);
    var dgx=0.0f; var dgy=0.0f; var dgz=0.0f;
    for (var c=0u; c<p.C; c++) {
        let goi = ((b*p.C+c)*p.oD+d)*p.oH*p.oW+h*p.oW+w;
        let g = go[goi];
        let dfx = -(1.0-fy)*(1.0-fz)*gi(b,c,z0,y0,x0)+(1.0-fy)*(1.0-fz)*gi(b,c,z0,y0,x0+1)
                  -fy*(1.0-fz)*gi(b,c,z0,y0+1,x0)+fy*(1.0-fz)*gi(b,c,z0,y0+1,x0+1)
                  -(1.0-fy)*fz*gi(b,c,z0+1,y0,x0)+(1.0-fy)*fz*gi(b,c,z0+1,y0,x0+1)
                  -fy*fz*gi(b,c,z0+1,y0+1,x0)+fy*fz*gi(b,c,z0+1,y0+1,x0+1);
        let dfy = -(1.0-fx)*(1.0-fz)*gi(b,c,z0,y0,x0)-fx*(1.0-fz)*gi(b,c,z0,y0,x0+1)
                  +(1.0-fx)*(1.0-fz)*gi(b,c,z0,y0+1,x0)+fx*(1.0-fz)*gi(b,c,z0,y0+1,x0+1)
                  -(1.0-fx)*fz*gi(b,c,z0+1,y0,x0)-fx*fz*gi(b,c,z0+1,y0,x0+1)
                  +(1.0-fx)*fz*gi(b,c,z0+1,y0+1,x0)+fx*fz*gi(b,c,z0+1,y0+1,x0+1);
        let dfz = -(1.0-fx)*(1.0-fy)*gi(b,c,z0,y0,x0)-fx*(1.0-fy)*gi(b,c,z0,y0,x0+1)
                  -(1.0-fx)*fy*gi(b,c,z0,y0+1,x0)-fx*fy*gi(b,c,z0,y0+1,x0+1)
                  +(1.0-fx)*(1.0-fy)*gi(b,c,z0+1,y0,x0)+fx*(1.0-fy)*gi(b,c,z0+1,y0,x0+1)
                  +(1.0-fx)*fy*gi(b,c,z0+1,y0+1,x0)+fx*fy*gi(b,c,z0+1,y0+1,x0+1);
        dgx+=g*dfx; dgy+=g*dfy; dgz+=g*dfz;
    }
    gg[idx*3u] = dgx * f32(p.iW-1u)*0.5;
    gg[idx*3u+1u] = dgy * f32(p.iH-1u)*0.5;
    gg[idx*3u+2u] = dgz * f32(p.iD-1u)*0.5;
}
