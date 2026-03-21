// adam.wgsl - Adam optimizer step
//
// Mirrors cuda_adam_step: standard Adam update for rigid/affine parameters.

struct AdamParams {
    n: u32,
    step: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> exp_avg: array<f32>;
@group(0) @binding(3) var<storage, read_write> exp_avg_sq: array<f32>;
@group(0) @binding(4) var<uniform> p: AdamParams;

@compute @workgroup_size(256)
fn adam_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }

    let g = grad[i];
    let m = p.beta1 * exp_avg[i] + (1.0 - p.beta1) * g;
    let v = p.beta2 * exp_avg_sq[i] + (1.0 - p.beta2) * g * g;
    exp_avg[i] = m;
    exp_avg_sq[i] = v;

    let bc1 = 1.0 - pow(p.beta1, f32(p.step));
    let bc2 = 1.0 - pow(p.beta2, f32(p.step));
    let step_size = p.lr / bc1;
    let denom = sqrt(v / bc2) + p.eps;
    param[i] = param[i] - step_size * m / denom;
}
