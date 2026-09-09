// One axis of the separable box filter used by regular local CC.
struct BoxP { D:u32, H:u32, W:u32, ks:u32, axis:u32, _p0:u32, _p1:u32, _p2:u32, }
@group(0) @binding(0) var<storage, read> bi: array<f32>;
@group(0) @binding(1) var<storage, read_write> bo: array<f32>;
@group(0) @binding(2) var<uniform> bp: BoxP;
@compute @workgroup_size(256)
fn box_filter(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 256u;
    if (idx >= bp.D*bp.H*bp.W) { return; }
    let w = idx%bp.W; var tmp = idx/bp.W;
    let h = tmp%bp.H; let d = tmp/bp.H;
    let r = i32(bp.ks/2u); let sc = 1.0/f32(bp.ks);
    var sum = 0.0f;
    if (bp.axis==2u) {
        for (var k=-r; k<=r; k++) {
            let ww=i32(w)+k;
            if (ww>=0 && ww<i32(bp.W)) { sum+=bi[d*bp.H*bp.W+h*bp.W+u32(ww)]; }
        }
    } else if (bp.axis==1u) {
        for (var k=-r; k<=r; k++) {
            let hh=i32(h)+k;
            if (hh>=0 && hh<i32(bp.H)) { sum+=bi[d*bp.H*bp.W+u32(hh)*bp.W+w]; }
        }
    } else {
        for (var k=-r; k<=r; k++) {
            let dd=i32(d)+k;
            if (dd>=0 && dd<i32(bp.D)) { sum+=bi[u32(dd)*bp.H*bp.W+h*bp.W+w]; }
        }
    }
    bo[idx] = sum*sc;
}
