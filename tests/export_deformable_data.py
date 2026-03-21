#!/usr/bin/env python
"""Export deformable registration (SyN) reference data for C validation.
Uses the same pipeline as validate/run_validation.py."""
import json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from fireants.io.image import Image, BatchedImages
from fireants.registration.moments import MomentsRegistration
from fireants.registration.rigid import RigidRegistration
from fireants.registration.affine import AffineRegistration
from fireants.registration.syn import SyNRegistration
from fireants.losses.cc import LocalNormalizedCrossCorrelationLoss

OUT_DIR = os.path.join(os.path.dirname(__file__), "test_data")
os.makedirs(OUT_DIR, exist_ok=True)

def save_tensor(name, t):
    t_cpu = t.detach().cpu().contiguous().float()
    data = t_cpu.numpy()
    data.tofile(os.path.join(OUT_DIR, f"{name}.bin"))
    with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
        json.dump({"shape": list(data.shape), "dtype": "float32"}, f)
    print(f"  {name}: shape={list(data.shape)}")

def export_syn(dataset_name, fixed_path, moving_path,
               mom_params, rigid_params, affine_params, syn_params):
    print(f"\n=== SyN: {dataset_name} ===")
    fixed_img = Image.load_file(fixed_path, device="cpu")
    moving_img = Image.load_file(moving_path, device="cpu")
    fb = BatchedImages([fixed_img])
    mb = BatchedImages([moving_img])

    # Moments
    mom = MomentsRegistration(scale=1.0, fixed_images=fb, moving_images=mb,
                              moments=2, orientation="both", blur=True,
                              loss_type="cc", cc_kernel_size=5)
    mom.optimize()

    # Rigid
    rig = RigidRegistration(fixed_images=fb, moving_images=mb,
                            init_moment=mom.get_rigid_moment_init(),
                            init_translation=mom.get_rigid_transl_init(),
                            **rigid_params)
    rig.optimize()

    # Affine
    aff = AffineRegistration(fixed_images=fb, moving_images=mb,
                             init_rigid=rig.get_rigid_matrix(homogenous=False),
                             **affine_params)
    aff.optimize()
    aff_mat = aff.get_affine_matrix()

    # Save affine matrix for C to initialize from
    prefix = f"syn_{dataset_name}"
    save_tensor(f"{prefix}_affine_init_44", aff_mat)

    # SyN
    syn = SyNRegistration(fixed_images=fb, moving_images=mb,
                          init_affine=aff_mat, **syn_params)
    syn.optimize()

    # Evaluate
    moved = syn.evaluate(fb, mb)
    save_tensor(f"{prefix}_moved", moved)

    loss_fn = LocalNormalizedCrossCorrelationLoss(spatial_dims=3, kernel_size=9, reduction="mean")
    ncc = loss_fn(moved, fb())
    print(f"  NCC after SyN: {ncc.item():.6f}")
    save_tensor(f"{prefix}_ncc", ncc.detach().unsqueeze(0))

if __name__ == "__main__":
    # Small dataset (matching validate/run_validation.py parameters)
    export_syn("small",
        "validate/small/MNI152_T1_2mm.nii.gz",
        "validate/small/T1_head_2mm.nii.gz",
        {},
        dict(scales=[4,2,1], iterations=[200,100,50], loss_type="mi",
             optimizer="Adam", optimizer_lr=3e-3, blur=True),
        dict(scales=[4,2,1], iterations=[200,100,50], loss_type="mi",
             optimizer="Adam", optimizer_lr=1e-3, blur=True),
        dict(scales=[4,2,1], iterations=[200,100,50], loss_type="cc",
             cc_kernel_size=5, optimizer="Adam", optimizer_lr=0.1,
             smooth_warp_sigma=0.5, smooth_grad_sigma=1.0))
    print(f"\nAll deformable data exported to {OUT_DIR}")
