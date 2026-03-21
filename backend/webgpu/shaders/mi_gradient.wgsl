// mi_gradient.wgsl - Per-voxel MI gradient matching CUDA mi_gradient_kernel
//
// Correct softmax derivative with chain rule through both joint and
// marginal histograms. Receives normalized histograms (probabilities)
// and raw pred/target values.

const NB: u32 = 32u;

struct GP { n: u32, num_bins: u32, inv_maxval: f32, preterm: f32,
            inv_n: f32, nr: f32, dr: f32, _pad: u32, }

@group(0) @binding(0) var<storage, read> g_pred: array<f32>;
@group(0) @binding(1) var<storage, read> g_target: array<f32>;
@group(0) @binding(2) var<storage, read> gjh: array<f32>;
@group(0) @binding(3) var<storage, read> gph: array<f32>;
@group(0) @binding(4) var<storage, read> gth: array<f32>;
@group(0) @binding(5) var<storage, read_write> gg: array<f32>;
@group(0) @binding(6) var<uniform> gpp: GP;

@compute @workgroup_size(256)
fn mi_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= gpp.n) { return; }
    let nb = gpp.num_bins;
    let im = gpp.inv_maxval;
    let pt = gpp.preterm;
    let inv_N = gpp.inv_n;
    let nr = gpp.nr; let dr = gpp.dr;

    let pv = clamp(g_pred[i] * im, 0.0, 1.0);
    let tv = clamp(g_target[i] * im, 0.0, 1.0);
    let inv_nb = 1.0 / f32(nb);
    let half_inv = 0.5 * inv_nb;

    // Recompute softmax-normalized Parzen weights
    var wa: array<f32, 32>;
    var wb: array<f32, 32>;
    var sa = 0.0f; var sb = 0.0f;
    for (var b = 0u; b < nb; b++) {
        let bc = f32(b) * inv_nb + half_inv;
        let dp = pv - bc; let dt = tv - bc;
        wa[b] = exp(-pt * dp * dp);
        wb[b] = exp(-pt * dt * dt);
        sa += wa[b]; sb += wb[b];
    }
    for (var b = 0u; b < nb; b++) { wa[b] /= sa; wb[b] /= sb; }

    // Softmax derivative: weighted_du = sum_a wa[a] * du_a
    var wdu = 0.0f;
    for (var a = 0u; a < nb; a++) {
        let bc = f32(a) * inv_nb + half_inv;
        wdu += wa[a] * (-2.0 * pt * (pv - bc));
    }

    // Window active pred bins
    let pcb = u32(clamp(pv * f32(nb), 0.0, f32(nb - 1u)));
    let R = 4u;
    let plo = select(pcb - R, 0u, pcb < R);
    let phi = min(pcb + R + 1u, nb);

    var dmi = 0.0f;
    for (var a = plo; a < phi; a++) {
        let bc_a = f32(a) * inv_nb + half_inv;
        let du_a = -2.0 * pt * (pv - bc_a);
        let dwa = wa[a] * (du_a - wdu);

        // Through joint histogram: d(MI)/d(pab) * d(pab)/d(wa)
        for (var b = 0u; b < nb; b++) {
            let p = gjh[a * nb + b];
            let pp = gph[a] * gth[b];
            let dm = log((p + nr) / (pp + dr) + dr) + p / (p + nr);
            dmi += dm * inv_N * wb[b] * dwa;
        }

        // Through marginal pa: d(MI)/d(pa) * d(pa)/d(wa)
        var dpa = 0.0f;
        for (var b = 0u; b < nb; b++) {
            let pp = gph[a] * gth[b];
            dpa -= gjh[a * nb + b] * gth[b] / (pp + dr);
        }
        dmi += dpa * inv_N * dwa;
    }

    // Chain rule: d(pn)/d(pred[i]) = inv_maxval, negate for loss=-MI
    gg[i] = -dmi * im;
}
