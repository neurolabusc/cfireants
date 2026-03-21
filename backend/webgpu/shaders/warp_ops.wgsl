// warp_ops.wgsl - WarpAdam operations for deformable registration
//
// Entry points:
//   adam_moments_update: m = β1*m + (1-β1)*g; v = β2*v + (1-β2)*g²
//   adam_direction: output = (m/bc1) / (sqrt(v/bc2) + eps)
//   max_l2_norm: per-workgroup max of sqrt(x²+y²+z²) for [spatial,3] data
//   vec_scale: data *= alpha
//   vec_add: a += b

struct MomentsParams {
    n: u32, _p: u32,
    beta1: f32, beta2: f32,
}

@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> exp_avg: array<f32>;
@group(0) @binding(2) var<storage, read_write> exp_avg_sq: array<f32>;
@group(0) @binding(3) var<uniform> mp: MomentsParams;

@compute @workgroup_size(256)
fn adam_moments_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= mp.n) { return; }
    let g = grad[i];
    exp_avg[i] = mp.beta1 * exp_avg[i] + (1.0 - mp.beta1) * g;
    exp_avg_sq[i] = mp.beta2 * exp_avg_sq[i] + (1.0 - mp.beta2) * g * g;
}

// --- Adam direction ---
struct DirParams {
    n: u32, _p: u32,
    inv_bc1: f32, inv_bc2: f32, eps: f32, _p2: f32, _p3: u32, _p4: u32,
}

@group(0) @binding(0) var<storage, read_write> dir_output: array<f32>;
@group(0) @binding(1) var<storage, read> dir_ea: array<f32>;
@group(0) @binding(2) var<storage, read> dir_eas: array<f32>;
@group(0) @binding(3) var<uniform> dp: DirParams;

@compute @workgroup_size(256)
fn adam_direction(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dp.n) { return; }
    let m_hat = dir_ea[i] * dp.inv_bc1;
    let v_hat = dir_eas[i] * dp.inv_bc2;
    dir_output[i] = m_hat / (sqrt(v_hat) + dp.eps);
}

// --- Max L2 norm reduction ---
// Each workgroup computes max(sqrt(x²+y²+z²)) over its elements
struct NormParams { spatial: u32, _p0: u32, eps: f32, _p1: f32, }

@group(0) @binding(0) var<storage, read> norm_data: array<f32>;  // [spatial*3]
@group(0) @binding(1) var<storage, read_write> norm_out: array<f32>; // [n_workgroups]
@group(0) @binding(2) var<uniform> np: NormParams;

var<workgroup> smax: array<f32, 256>;

@compute @workgroup_size(256)
fn max_l2_norm(@builtin(global_invocation_id) gid: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>,
               @builtin(workgroup_id) wid: vec3<u32>) {
    let i = gid.x;
    let tid = lid.x;
    var val = 0.0f;
    if (i < np.spatial) {
        let x = norm_data[i * 3u];
        let y = norm_data[i * 3u + 1u];
        let z = norm_data[i * 3u + 2u];
        val = sqrt(x * x + y * y + z * z);
    }
    smax[tid] = val;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s && smax[tid + s] > smax[tid]) {
            smax[tid] = smax[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) { norm_out[wid.x] = smax[0]; }
}
