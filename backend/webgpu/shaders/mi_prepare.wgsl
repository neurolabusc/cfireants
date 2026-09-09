// Convert the packed fixed-point histogram into reusable MI-gradient
// coefficients and reduce the scalar MI loss. One workgroup handles the
// fixed 32x32 histogram, avoiding a CPU round trip between GPU kernels.

const NB: u32 = 32u;
const NB2: u32 = 1024u;
const JOINT_OFFSET: u32 = 1u;
const PA_OFFSET: u32 = JOINT_OFFSET + NB2;
const PB_OFFSET: u32 = PA_OFFSET + NB;
const LOSS_OFFSET: u32 = NB2 + NB;

struct Params { nr: f32, dr: f32, _p0: u32, _p1: u32, }

@group(0) @binding(0) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> coeff: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

var<workgroup> partial: array<f32, 256>;
var<workgroup> inv_total_shared: f32;

@compute @workgroup_size(256)
fn prepare(@builtin(local_invocation_id) lid: vec3<u32>) {
    var count_sum = 0.0f;
    for (var i = lid.x; i < NB2; i += 256u) {
        count_sum += f32(atomicLoad(&state[JOINT_OFFSET + i]));
    }
    partial[lid.x] = count_sum;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) {
            partial[lid.x] += partial[lid.x + stride];
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        inv_total_shared = select(0.0, 1.0 / partial[0], partial[0] > 0.0);
    }
    workgroupBarrier();

    let inv_total = inv_total_shared;
    var mi_sum = 0.0f;
    for (var i = lid.x; i < NB2; i += 256u) {
        let a = i / NB;
        let b = i - a * NB;
        let pab = f32(atomicLoad(&state[JOINT_OFFSET + i])) * inv_total;
        let pa = f32(atomicLoad(&state[PA_OFFSET + a])) * inv_total;
        let pb = f32(atomicLoad(&state[PB_OFFSET + b])) * inv_total;
        let product = pa * pb;
        let log_term = log((pab + p.nr) / (product + p.dr) + p.dr);
        mi_sum += pab * log_term;
        coeff[i] = log_term + pab / (pab + p.nr);
    }

    if (lid.x < NB) {
        let a = lid.x;
        let pa = f32(atomicLoad(&state[PA_OFFSET + a])) * inv_total;
        var marginal = 0.0f;
        for (var b = 0u; b < NB; b++) {
            let pab = f32(atomicLoad(&state[JOINT_OFFSET + a * NB + b])) * inv_total;
            let pb = f32(atomicLoad(&state[PB_OFFSET + b])) * inv_total;
            marginal -= pab * pb / (pa * pb + p.dr);
        }
        coeff[NB2 + a] = marginal;
    }

    partial[lid.x] = mi_sum;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) {
            partial[lid.x] += partial[lid.x + stride];
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        coeff[LOSS_OFFSET] = -partial[0];
    }
}
