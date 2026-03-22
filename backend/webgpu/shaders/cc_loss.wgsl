// cc_loss.wgsl - Cross-correlation loss helper kernels
//
// CC loss requires multiple dispatch steps orchestrated from C:
//   1. multiply (P*P, T*T, P*T)
//   2. separable box filter (3 axis passes per channel)
//   3. NCC computation + gradient source terms
//   4. box filter adjoint on gradient sources
//   5. combine gradient
//
// Each step is a separate entry point.

// --- Element-wise multiply: c[i] = a[i] * b[i] ---

struct MulParams { n: u32, _p0: u32, _p1: u32, _p2: u32, }

@group(0) @binding(0) var<storage, read> mul_a: array<f32>;
@group(0) @binding(1) var<storage, read> mul_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> mul_c: array<f32>;
@group(0) @binding(3) var<uniform> mul_p: MulParams;

@compute @workgroup_size(256)
fn multiply(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= mul_p.n) { return; }
    mul_c[i] = mul_a[i] * mul_b[i];
}

// --- Separable box filter along one axis ---
// This computes: out[i] = (1/ks) * sum_{k=-r..r} in[shifted(i, k, axis)]

struct BoxParams {
    D: u32, H: u32, W: u32,
    ks: u32,     // kernel size
    axis: u32,   // 0=D, 1=H, 2=W
    _p0: u32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<storage, read> box_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> box_out: array<f32>;
@group(0) @binding(2) var<uniform> box_p: BoxParams;

@compute @workgroup_size(256)
fn box_filter(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 256u;
    let total = box_p.D * box_p.H * box_p.W;
    if (idx >= total) { return; }

    let w = idx % box_p.W;
    var tmp = idx / box_p.W;
    let h = tmp % box_p.H;
    let d = tmp / box_p.H;

    let r = i32(box_p.ks / 2u);
    let scale = 1.0 / f32(box_p.ks);
    var sum = 0.0f;

    if (box_p.axis == 2u) {
        for (var k = -r; k <= r; k = k + 1) {
            let ww = i32(w) + k;
            if (ww >= 0 && ww < i32(box_p.W)) {
                sum += box_in[d * box_p.H * box_p.W + h * box_p.W + u32(ww)];
            }
        }
    } else if (box_p.axis == 1u) {
        for (var k = -r; k <= r; k = k + 1) {
            let hh = i32(h) + k;
            if (hh >= 0 && hh < i32(box_p.H)) {
                sum += box_in[d * box_p.H * box_p.W + u32(hh) * box_p.W + w];
            }
        }
    } else {
        for (var k = -r; k <= r; k = k + 1) {
            let dd = i32(d) + k;
            if (dd >= 0 && dd < i32(box_p.D)) {
                sum += box_in[u32(dd) * box_p.H * box_p.W + h * box_p.W + w];
            }
        }
    }
    box_out[idx] = sum * scale;
}

// --- NCC + gradient source terms ---

struct NccParams {
    n: u32,
    compute_grad: u32,
    nr: f32,
    dr: f32,
}

@group(0) @binding(0) var<storage, read> p_sum: array<f32>;
@group(0) @binding(1) var<storage, read> t_sum: array<f32>;
@group(0) @binding(2) var<storage, read> p2_sum: array<f32>;
@group(0) @binding(3) var<storage, read> t2_sum: array<f32>;
@group(0) @binding(4) var<storage, read> tp_sum: array<f32>;
@group(0) @binding(5) var<storage, read_write> ncc_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> src_p: array<f32>;
@group(0) @binding(7) var<storage, read_write> src_p2: array<f32>;
@group(0) @binding(8) var<storage, read_write> src_tp: array<f32>;
@group(0) @binding(9) var<uniform> ncc_p: NccParams;

@compute @workgroup_size(256)
fn ncc_and_grad(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= ncc_p.n) { return; }

    let ps = p_sum[i]; let ts = t_sum[i];
    let cross = tp_sum[i] - ps * ts;
    var p_var = p2_sum[i] - ps * ps;
    var t_var = t2_sum[i] - ts * ts;
    if (p_var < ncc_p.dr) { p_var = ncc_p.dr; }
    if (t_var < ncc_p.dr) { t_var = ncc_p.dr; }

    let f = cross * cross + ncc_p.nr;
    let g = p_var * t_var + ncc_p.dr;
    var ncc = f / g;
    ncc = clamp(ncc, -1.0, 1.0);
    ncc_out[i] = ncc;

    if (ncc_p.compute_grad != 0u) {
        let g2 = g * g;
        src_tp[i] = 2.0 * cross * g / g2;
        src_p2[i] = -f * t_var / g2;
        src_p[i] = (-2.0 * cross * ts * g + 2.0 * f * ps * t_var) / g2;
    }
}

// --- Combine gradient: grad = -inv_count * (adj_p + 2*P*adj_p2 + T*adj_tp) ---

struct CombineParams {
    n: u32,
    _p0: u32,
    inv_count: f32,
    _p1: f32,
}

@group(0) @binding(0) var<storage, read> adj_p: array<f32>;
@group(0) @binding(1) var<storage, read> adj_p2: array<f32>;
@group(0) @binding(2) var<storage, read> adj_tp: array<f32>;
@group(0) @binding(3) var<storage, read> cc_P: array<f32>;
@group(0) @binding(4) var<storage, read> cc_T: array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_out: array<f32>;
@group(0) @binding(6) var<uniform> comb_p: CombineParams;

@compute @workgroup_size(256)
fn combine_grad(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= comb_p.n) { return; }
    grad_out[i] = -comb_p.inv_count * (adj_p[i] + 2.0 * cc_P[i] * adj_p2[i] + cc_T[i] * adj_tp[i]);
}
