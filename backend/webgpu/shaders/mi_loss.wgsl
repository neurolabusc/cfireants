// mi_loss.wgsl - Mutual Information loss with Gaussian Parzen windowing
//
// Pass 1 (histogram): Accumulate histograms via atomicAdd on atomic<u32>
//   using bitcast(f32 → u32) and CAS loops.
// Pass 2 (gradient): Compute per-voxel gradient from normalized histograms.

// --- Pass 1: Histogram accumulation ---

struct HistParams {
    n: u32,
    num_bins: u32,
    _p0: u32, _p1: u32,
}

@group(0) @binding(0) var<storage, read> hist_pred: array<f32>;
@group(0) @binding(1) var<storage, read> hist_target: array<f32>;
@group(0) @binding(2) var<storage, read_write> joint_hist: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> pred_hist: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> target_hist: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> hp: HistParams;

@compute @workgroup_size(256)
fn histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= hp.n) { return; }

    let nb = hp.num_bins;
    let p_val = clamp(hist_pred[i], 0.0, 1.0);
    let t_val = clamp(hist_target[i], 0.0, 1.0);

    let bin_width = 1.0 / f32(nb - 1u);
    let inv_sigma = 1.0 / bin_width;

    for (var bi = 0u; bi < nb; bi = bi + 1u) {
        let bin_center = f32(bi) * bin_width;

        let p_diff = (p_val - bin_center) * inv_sigma;
        let p_weight = exp(-0.5 * p_diff * p_diff);

        // CAS-based float atomicAdd on pred_hist[bi]
        if (p_weight > 1e-10) {
            var old_p = atomicLoad(&pred_hist[bi]);
            loop {
                let new_p = bitcast<u32>(bitcast<f32>(old_p) + p_weight);
                let res_p = atomicCompareExchangeWeak(&pred_hist[bi], old_p, new_p);
                if (res_p.exchanged) { break; }
                old_p = res_p.old_value;
            }
        }

        let t_diff = (t_val - bin_center) * inv_sigma;
        let t_weight = exp(-0.5 * t_diff * t_diff);

        if (t_weight > 1e-10) {
            var old_t = atomicLoad(&target_hist[bi]);
            loop {
                let new_t = bitcast<u32>(bitcast<f32>(old_t) + t_weight);
                let res_t = atomicCompareExchangeWeak(&target_hist[bi], old_t, new_t);
                if (res_t.exchanged) { break; }
                old_t = res_t.old_value;
            }
        }

        // Joint histogram: accumulate p_weight * t_weight for each target bin
        if (p_weight > 1e-10) {
            for (var bj = 0u; bj < nb; bj = bj + 1u) {
                let bc_j = f32(bj) * bin_width;
                let td_j = (t_val - bc_j) * inv_sigma;
                let tw_j = exp(-0.5 * td_j * td_j);
                let jw = p_weight * tw_j;
                if (jw > 1e-10) {
                    var old_j = atomicLoad(&joint_hist[bi * nb + bj]);
                    loop {
                        let new_j = bitcast<u32>(bitcast<f32>(old_j) + jw);
                        let res_j = atomicCompareExchangeWeak(&joint_hist[bi * nb + bj], old_j, new_j);
                        if (res_j.exchanged) { break; }
                        old_j = res_j.old_value;
                    }
                }
            }
        }
    }
}

// --- Pass 2: Gradient per voxel ---

struct GradParams {
    n: u32,
    num_bins: u32,
    _p0: u32, _p1: u32,
}

@group(0) @binding(0) var<storage, read> g_pred: array<f32>;
@group(0) @binding(1) var<storage, read> g_target: array<f32>;
@group(0) @binding(2) var<storage, read> g_joint: array<f32>;
@group(0) @binding(3) var<storage, read> g_phist: array<f32>;
@group(0) @binding(4) var<storage, read> g_thist: array<f32>;
@group(0) @binding(5) var<storage, read_write> g_grad: array<f32>;
@group(0) @binding(6) var<uniform> gp: GradParams;

@compute @workgroup_size(256)
fn mi_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= gp.n) { return; }

    let nb = gp.num_bins;
    let p_val = clamp(g_pred[i], 0.0, 1.0);
    let bin_width = 1.0 / f32(nb - 1u);
    let inv_sigma = 1.0 / bin_width;
    let inv_sigma2 = inv_sigma * inv_sigma;

    var grad_val = 0.0f;

    for (var bi = 0u; bi < nb; bi = bi + 1u) {
        let bin_center = f32(bi) * bin_width;
        let p_diff = p_val - bin_center;
        let p_weight = exp(-0.5 * (p_diff * inv_sigma) * (p_diff * inv_sigma));
        let dp_weight = -p_diff * inv_sigma2 * p_weight;

        let p_marg = g_phist[bi];
        if (p_marg < 1e-10) { continue; }

        for (var bj = 0u; bj < nb; bj = bj + 1u) {
            let joint_val = g_joint[bi * nb + bj];
            if (joint_val < 1e-10) { continue; }

            let t_marg = g_thist[bj];
            if (t_marg < 1e-10) { continue; }

            let log_term = log(joint_val) - log(p_marg);
            grad_val += dp_weight * (log_term + 1.0);
            grad_val -= dp_weight * joint_val / p_marg;
        }
    }

    g_grad[i] = -grad_val;
}
