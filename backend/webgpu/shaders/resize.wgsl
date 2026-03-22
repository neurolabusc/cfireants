// resize.wgsl - Trilinear 3D resize
//
// Mirrors cuda_trilinear_resize: [B,C,iD,iH,iW] -> [B,C,oD,oH,oW]
// One thread per output element (B*C*oD*oH*oW).

struct ResizeParams {
    B: u32, C: u32,
    iD: u32, iH: u32, iW: u32,
    oD: u32, oH: u32, oW: u32,
    align_corners: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> p: ResizeParams;

@compute @workgroup_size(256)
fn trilinear_resize(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 256u;
    let total = p.B * p.C * p.oD * p.oH * p.oW;
    if (idx >= total) { return; }

    let ow = idx % p.oW;
    var tmp = idx / p.oW;
    let oh = tmp % p.oH;
    tmp = tmp / p.oH;
    let od = tmp % p.oD;
    tmp = tmp / p.oD;
    let c = tmp % p.C;
    let b = tmp / p.C;

    var sd: f32; var sh: f32; var sw: f32;
    if (p.align_corners != 0u && p.oD > 1u) {
        sd = f32(od) * f32(p.iD - 1u) / f32(p.oD - 1u);
    } else {
        sd = (f32(od) + 0.5) * f32(p.iD) / f32(p.oD) - 0.5;
    }
    if (p.align_corners != 0u && p.oH > 1u) {
        sh = f32(oh) * f32(p.iH - 1u) / f32(p.oH - 1u);
    } else {
        sh = (f32(oh) + 0.5) * f32(p.iH) / f32(p.oH) - 0.5;
    }
    if (p.align_corners != 0u && p.oW > 1u) {
        sw = f32(ow) * f32(p.iW - 1u) / f32(p.oW - 1u);
    } else {
        sw = (f32(ow) + 0.5) * f32(p.iW) / f32(p.oW) - 0.5;
    }

    var d0 = i32(floor(sd)); var h0 = i32(floor(sh)); var w0 = i32(floor(sw));
    var d1 = d0 + 1; var h1 = h0 + 1; var w1 = w0 + 1;
    var fd = sd - f32(d0); var fh = sh - f32(h0); var fw = sw - f32(w0);

    if (d0 < 0) { d0 = 0; fd = 0.0; } if (d1 >= i32(p.iD)) { d1 = i32(p.iD) - 1; }
    if (h0 < 0) { h0 = 0; fh = 0.0; } if (h1 >= i32(p.iH)) { h1 = i32(p.iH) - 1; }
    if (w0 < 0) { w0 = 0; fw = 0.0; } if (w1 >= i32(p.iW)) { w1 = i32(p.iW) - 1; }

    let base = (b * p.C + c) * p.iD * p.iH * p.iW;
    let s000 = input[base + u32(d0) * p.iH * p.iW + u32(h0) * p.iW + u32(w0)];
    let s001 = input[base + u32(d0) * p.iH * p.iW + u32(h0) * p.iW + u32(w1)];
    let s010 = input[base + u32(d0) * p.iH * p.iW + u32(h1) * p.iW + u32(w0)];
    let s011 = input[base + u32(d0) * p.iH * p.iW + u32(h1) * p.iW + u32(w1)];
    let s100 = input[base + u32(d1) * p.iH * p.iW + u32(h0) * p.iW + u32(w0)];
    let s101 = input[base + u32(d1) * p.iH * p.iW + u32(h0) * p.iW + u32(w1)];
    let s110 = input[base + u32(d1) * p.iH * p.iW + u32(h1) * p.iW + u32(w0)];
    let s111 = input[base + u32(d1) * p.iH * p.iW + u32(h1) * p.iW + u32(w1)];

    let val = (1.0-fd)*(1.0-fh)*(1.0-fw)*s000 + (1.0-fd)*(1.0-fh)*fw*s001
            + (1.0-fd)*fh*(1.0-fw)*s010 + (1.0-fd)*fh*fw*s011
            + fd*(1.0-fh)*(1.0-fw)*s100 + fd*(1.0-fh)*fw*s101
            + fd*fh*(1.0-fw)*s110 + fd*fh*fw*s111;

    output[idx] = val;
}
