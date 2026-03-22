#include <metal_stdlib>
using namespace metal;

// Params struct for element-wise ops (matches C-side struct)
struct EWParams {
    uint n;
    float value;
};

// Fill tensor with constant value
kernel void tensor_fill(device float *data [[buffer(0)]],
                        constant EWParams &p [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    data[gid] = p.value;
}

// Scale tensor: data *= alpha
kernel void tensor_scale(device float *data [[buffer(0)]],
                         constant EWParams &p [[buffer(1)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    data[gid] *= p.value;
}

// AXPY: y += alpha * x
struct AXPYParams {
    uint n;
    float alpha;
};

kernel void tensor_axpy(device float *y [[buffer(0)]],
                        const device float *x [[buffer(1)]],
                        constant AXPYParams &p [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    y[gid] += p.alpha * x[gid];
}

// Adam optimizer step (all in one kernel, matching CUDA adam_step)
struct AdamParams {
    uint n;
    int step;
    float lr;
    float beta1;
    float beta2;
    float eps;
};

kernel void adam_step(device float *param [[buffer(0)]],
                      const device float *grad [[buffer(1)]],
                      device float *exp_avg [[buffer(2)]],
                      device float *exp_avg_sq [[buffer(3)]],
                      constant AdamParams &p [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;

    float g = grad[gid];
    float m = p.beta1 * exp_avg[gid] + (1.0f - p.beta1) * g;
    float v = p.beta2 * exp_avg_sq[gid] + (1.0f - p.beta2) * g * g;
    exp_avg[gid] = m;
    exp_avg_sq[gid] = v;

    // Bias correction
    float bc1 = 1.0f - pow(p.beta1, float(p.step));
    float bc2 = 1.0f - pow(p.beta2, float(p.step));
    float m_hat = m / bc1;
    float v_hat = v / bc2;

    param[gid] -= p.lr * m_hat / (sqrt(v_hat) + p.eps);
}

// WarpAdam moments update: exp_avg/exp_avg_sq updated from gradient
struct WarpAdamMomentsParams {
    uint n;
    float beta1;
    float beta2;
    uint _pad;
};

kernel void warp_adam_moments(const device float *grad [[buffer(0)]],
                              device float *exp_avg [[buffer(1)]],
                              device float *exp_avg_sq [[buffer(2)]],
                              constant WarpAdamMomentsParams &p [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    float g = grad[gid];
    exp_avg[gid] = p.beta1 * exp_avg[gid] + (1.0f - p.beta1) * g;
    exp_avg_sq[gid] = p.beta2 * exp_avg_sq[gid] + (1.0f - p.beta2) * g * g;
}

// WarpAdam direction: output = (exp_avg/bc1) / (sqrt(exp_avg_sq/bc2) + eps)
struct WarpAdamDirParams {
    uint n;
    float bc1;
    float bc2;
    float eps;
};

kernel void warp_adam_direction(device float *output [[buffer(0)]],
                                const device float *exp_avg [[buffer(1)]],
                                const device float *exp_avg_sq [[buffer(2)]],
                                constant WarpAdamDirParams &p [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    float m_hat = exp_avg[gid] / p.bc1;
    float v_hat = exp_avg_sq[gid] / p.bc2;
    output[gid] = m_hat / (sqrt(v_hat) + p.eps);
}
