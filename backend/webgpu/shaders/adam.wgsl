// Adam optimizer update.
struct P { n:u32, step:u32, lr:f32, beta1:f32, beta2:f32, eps:f32, _p0:u32, _p1:u32, }
@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> ea: array<f32>;
@group(0) @binding(3) var<storage, read_write> eas: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(256)
fn adam_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u; if (i >= p.n) { return; }
    let g = grad[i];
    let m = p.beta1*ea[i]+(1.0-p.beta1)*g;
    let v = p.beta2*eas[i]+(1.0-p.beta2)*g*g;
    ea[i]=m; eas[i]=v;
    let bc1=1.0-pow(p.beta1,f32(p.step));
    let bc2=1.0-pow(p.beta2,f32(p.step));
    param[i] = param[i] - (p.lr/bc1)*m/(sqrt(v/bc2)+p.eps);
}
