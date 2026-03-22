// compose.wgsl - Fused compositive warp update
//
// output[i] = update[i] + trilinear_sample(warp, identity_coord[i] + update[i])
// Operates on [D,H,W,3] displacement fields.

struct Params { D: u32, H: u32, W: u32, _pad: u32, }

@group(0) @binding(0) var<storage, read> warp: array<f32>;    // [D,H,W,3]
@group(0) @binding(1) var<storage, read> update: array<f32>;  // [D,H,W,3]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [D,H,W,3]
@group(0) @binding(3) var<uniform> p: Params;

fn sample_warp(nx: f32, ny: f32, nz: f32, c: u32) -> f32 {
    let ix = (nx + 1.0) * 0.5 * f32(p.W - 1u);
    let iy = (ny + 1.0) * 0.5 * f32(p.H - 1u);
    let iz = (nz + 1.0) * 0.5 * f32(p.D - 1u);
    let x0 = i32(floor(ix)); let y0 = i32(floor(iy)); let z0 = i32(floor(iz));
    let x1 = x0 + 1; let y1 = y0 + 1; let z1 = z0 + 1;
    let fx = ix - f32(x0); let fy = iy - f32(y0); let fz = iz - f32(z0);

    var val = 0.0f;
    let iD = i32(p.D); let iH = i32(p.H); let iW = i32(p.W);
    // 8 corners of trilinear interpolation
    if (z0>=0 && z0<iD && y0>=0 && y0<iH && x0>=0 && x0<iW) { val += (1.0-fx)*(1.0-fy)*(1.0-fz) * warp[(u32(z0)*p.H+u32(y0))*p.W*3u+u32(x0)*3u+c]; }
    if (z0>=0 && z0<iD && y0>=0 && y0<iH && x1>=0 && x1<iW) { val += fx*(1.0-fy)*(1.0-fz) * warp[(u32(z0)*p.H+u32(y0))*p.W*3u+u32(x1)*3u+c]; }
    if (z0>=0 && z0<iD && y1>=0 && y1<iH && x0>=0 && x0<iW) { val += (1.0-fx)*fy*(1.0-fz) * warp[(u32(z0)*p.H+u32(y1))*p.W*3u+u32(x0)*3u+c]; }
    if (z0>=0 && z0<iD && y1>=0 && y1<iH && x1>=0 && x1<iW) { val += fx*fy*(1.0-fz) * warp[(u32(z0)*p.H+u32(y1))*p.W*3u+u32(x1)*3u+c]; }
    if (z1>=0 && z1<iD && y0>=0 && y0<iH && x0>=0 && x0<iW) { val += (1.0-fx)*(1.0-fy)*fz * warp[(u32(z1)*p.H+u32(y0))*p.W*3u+u32(x0)*3u+c]; }
    if (z1>=0 && z1<iD && y0>=0 && y0<iH && x1>=0 && x1<iW) { val += fx*(1.0-fy)*fz * warp[(u32(z1)*p.H+u32(y0))*p.W*3u+u32(x1)*3u+c]; }
    if (z1>=0 && z1<iD && y1>=0 && y1<iH && x0>=0 && x0<iW) { val += (1.0-fx)*fy*fz * warp[(u32(z1)*p.H+u32(y1))*p.W*3u+u32(x0)*3u+c]; }
    if (z1>=0 && z1<iD && y1>=0 && y1<iH && x1>=0 && x1<iW) { val += fx*fy*fz * warp[(u32(z1)*p.H+u32(y1))*p.W*3u+u32(x1)*3u+c]; }
    return val;
}

@compute @workgroup_size(256)
fn compositive_update(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 256u;
    let total = p.D * p.H * p.W;
    if (idx >= total) { return; }

    let w = idx % p.W;
    var tmp = idx / p.W;
    let h = tmp % p.H;
    let d = tmp / p.H;

    var nz: f32; var ny: f32; var nx: f32;
    if (p.D > 1u) { nz = 2.0 * f32(d) / f32(p.D - 1u) - 1.0; } else { nz = 0.0; }
    if (p.H > 1u) { ny = 2.0 * f32(h) / f32(p.H - 1u) - 1.0; } else { ny = 0.0; }
    if (p.W > 1u) { nx = 2.0 * f32(w) / f32(p.W - 1u) - 1.0; } else { nx = 0.0; }

    let i3 = idx * 3u;
    let ux = update[i3]; let uy = update[i3 + 1u]; let uz = update[i3 + 2u];

    // Sample warp at (identity + update)
    output[i3]     = ux + sample_warp(nx + ux, ny + uy, nz + uz, 0u);
    output[i3 + 1u] = uy + sample_warp(nx + ux, ny + uy, nz + uz, 1u);
    output[i3 + 2u] = uz + sample_warp(nx + ux, ny + uy, nz + uz, 2u);
}
