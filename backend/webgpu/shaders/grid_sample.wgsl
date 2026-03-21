// grid_sample.wgsl - 3D bilinear grid sampling (forward + backward)
//
// Mirrors grid_sample.cu: bilinear interpolation, zeros padding, align_corners=true.
// One thread per output spatial location (B*oD*oH*oW).
// Iterates over C channels per thread.

struct GridSampleParams {
    B: u32, C: u32,
    iD: u32, iH: u32, iW: u32,
    oD: u32, oH: u32, oW: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> grid: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> p: GridSampleParams;

fn unnorm(x: f32, size: u32) -> f32 {
    return (x + 1.0) * 0.5 * f32(size - 1u);
}

fn inp(b: u32, c: u32, d: i32, h: i32, w: i32) -> f32 {
    if (d < 0 || d >= i32(p.iD) || h < 0 || h >= i32(p.iH) || w < 0 || w >= i32(p.iW)) {
        return 0.0;
    }
    let idx = ((u32(i64(b) * i64(p.C) + i64(c)) * p.iD + u32(d)) * p.iH + u32(h)) * p.iW + u32(w);
    return input[idx];
}

@compute @workgroup_size(256)
fn grid_sample_fwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.B * p.oD * p.oH * p.oW;
    if (idx >= total) { return; }

    let w = idx % p.oW;
    var tmp = idx / p.oW;
    let h = tmp % p.oH;
    tmp = tmp / p.oH;
    let d = tmp % p.oD;
    let b = tmp / p.oD;

    let gidx = idx * 3u;
    let gx = grid[gidx + 0u];
    let gy = grid[gidx + 1u];
    let gz = grid[gidx + 2u];

    let ix = unnorm(gx, p.iW);
    let iy = unnorm(gy, p.iH);
    let iz = unnorm(gz, p.iD);

    let ix0 = i32(floor(ix));
    let iy0 = i32(floor(iy));
    let iz0 = i32(floor(iz));
    let ix1 = ix0 + 1;
    let iy1 = iy0 + 1;
    let iz1 = iz0 + 1;

    let fx = ix - f32(ix0);
    let fy = iy - f32(iy0);
    let fz = iz - f32(iz0);

    let w000 = (1.0 - fx) * (1.0 - fy) * (1.0 - fz);
    let w001 = fx * (1.0 - fy) * (1.0 - fz);
    let w010 = (1.0 - fx) * fy * (1.0 - fz);
    let w011 = fx * fy * (1.0 - fz);
    let w100 = (1.0 - fx) * (1.0 - fy) * fz;
    let w101 = fx * (1.0 - fy) * fz;
    let w110 = (1.0 - fx) * fy * fz;
    let w111 = fx * fy * fz;

    let out_stride = p.oD * p.oH * p.oW;
    var out_base = (b * p.C * p.oD + d) * p.oH * p.oW + h * p.oW + w;

    for (var c = 0u; c < p.C; c = c + 1u) {
        let val = w000 * inp(b, c, iz0, iy0, ix0)
                + w001 * inp(b, c, iz0, iy0, ix1)
                + w010 * inp(b, c, iz0, iy1, ix0)
                + w011 * inp(b, c, iz0, iy1, ix1)
                + w100 * inp(b, c, iz1, iy0, ix0)
                + w101 * inp(b, c, iz1, iy0, ix1)
                + w110 * inp(b, c, iz1, iy1, ix0)
                + w111 * inp(b, c, iz1, iy1, ix1);
        output[out_base + c * out_stride] = val;
    }
}
