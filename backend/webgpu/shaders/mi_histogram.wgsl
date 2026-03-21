// mi_histogram.wgsl - Workgroup-local MI histogram accumulation
//
// Each workgroup (256 threads) accumulates a local 32x32 joint histogram
// plus 2x32 marginals in var<workgroup> shared memory using fixed-point
// u32 atomicAdd (fast). After barrier, local values are merged to global
// via native atomicAdd (no CAS -- Metal-compatible).
//
// Overflow analysis: FP_SCALE=256, max per bin = N * 256 (worst case).
// Safe for N <= 16M voxels (~256^3). In practice, weights are distributed
// across bins, so safe for much larger volumes.

const NB: u32 = 32u;
const NB2: u32 = 1024u;
const WG: u32 = 256u;
const FP_SCALE: f32 = 256.0;

struct HP { n: u32, num_bins: u32, inv_maxval: f32, preterm: f32, }

@group(0) @binding(0) var<storage, read> h_pred: array<f32>;
@group(0) @binding(1) var<storage, read> h_target: array<f32>;
@group(0) @binding(2) var<storage, read_write> g_joint: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> g_pa: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> g_pb: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> hp: HP;

var<workgroup> lj: array<atomic<u32>, 1024>;
var<workgroup> la: array<atomic<u32>, 32>;
var<workgroup> lb: array<atomic<u32>, 32>;

@compute @workgroup_size(256)
fn histogram(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    for (var k = tid; k < NB2; k += WG) { atomicStore(&lj[k], 0u); }
    if (tid < NB) { atomicStore(&la[tid], 0u); atomicStore(&lb[tid], 0u); }
    workgroupBarrier();

    let i = gid.x;
    if (i < hp.n) {
        let im = hp.inv_maxval;
        let pt = hp.preterm;
        let pv = clamp(h_pred[i] * im, 0.0, 1.0);
        let tv = clamp(h_target[i] * im, 0.0, 1.0);
        let inv_nb = 1.0 / f32(NB);
        let half_inv = 0.5 * inv_nb;

        // Compute softmax-normalized Parzen weights (matching CUDA/Python)
        var wa: array<f32, 32>;
        var wb: array<f32, 32>;
        var sa = 0.0f;
        var sb = 0.0f;
        for (var b = 0u; b < NB; b++) {
            let bc = f32(b) * inv_nb + half_inv;
            let dp = pv - bc; let dt = tv - bc;
            wa[b] = exp(-pt * dp * dp);
            wb[b] = exp(-pt * dt * dt);
            sa += wa[b]; sb += wb[b];
        }
        let ia = 1.0 / sa; let ib = 1.0 / sb;
        for (var b = 0u; b < NB; b++) { wa[b] *= ia; wb[b] *= ib; }

        // Window: only accumulate bins with significant weight
        let pcb = u32(clamp(pv * f32(NB), 0.0, f32(NB - 1u)));
        let tcb = u32(clamp(tv * f32(NB), 0.0, f32(NB - 1u)));
        let R = 4u;
        let plo = select(pcb - R, 0u, pcb < R);
        let phi = min(pcb + R + 1u, NB);
        let tlo = select(tcb - R, 0u, tcb < R);
        let thi = min(tcb + R + 1u, NB);

        // Accumulate into local histogram (fixed-point atomicAdd)
        for (var a = plo; a < phi; a++) {
            let wfp = u32(wa[a] * FP_SCALE + 0.5);
            if (wfp > 0u) {
                atomicAdd(&la[a], wfp);
                for (var b = tlo; b < thi; b++) {
                    let jfp = u32(wa[a] * wb[b] * FP_SCALE + 0.5);
                    if (jfp > 0u) { atomicAdd(&lj[a * NB + b], jfp); }
                }
            }
        }
        for (var b = tlo; b < thi; b++) {
            let wfp = u32(wb[b] * FP_SCALE + 0.5);
            if (wfp > 0u) { atomicAdd(&lb[b], wfp); }
        }
    }

    workgroupBarrier();

    // Merge local -> global via native atomicAdd (no CAS, Metal-safe)
    for (var k = tid; k < NB2; k += WG) {
        let val = atomicLoad(&lj[k]);
        if (val > 0u) { atomicAdd(&g_joint[k], val); }
    }
    if (tid < NB) {
        let pav = atomicLoad(&la[tid]);
        if (pav > 0u) { atomicAdd(&g_pa[tid], pav); }
        let pbv = atomicLoad(&lb[tid]);
        if (pbv > 0u) { atomicAdd(&g_pb[tid], pbv); }
    }
}
