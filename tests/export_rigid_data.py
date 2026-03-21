#!/usr/bin/env python
"""Export rigid registration reference data for C validation."""

import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from fireants.io.image import Image, BatchedImages
from fireants.registration.moments import MomentsRegistration
from fireants.registration.rigid import RigidRegistration
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


def export_rigid(dataset_name, fixed_path, moving_path, params):
    print(f"\n=== Rigid test data: {dataset_name} ===")

    fixed_img = Image.load_file(fixed_path, device="cpu")
    moving_img = Image.load_file(moving_path, device="cpu")
    fixed_batch = BatchedImages([fixed_img])
    moving_batch = BatchedImages([moving_img])

    # First run moments for initialization
    moments = MomentsRegistration(
        scale=1.0, fixed_images=fixed_batch, moving_images=moving_batch,
        moments=2, orientation="both", blur=True, loss_type="cc", cc_kernel_size=5,
    )
    moments.optimize()

    # Run rigid registration
    rigid = RigidRegistration(
        fixed_images=fixed_batch, moving_images=moving_batch,
        init_moment=moments.get_rigid_moment_init(),
        init_translation=moments.get_rigid_transl_init(),
        **params,
    )
    rigid.optimize()

    prefix = f"rigid_{dataset_name}"

    # Save rigid matrix [N, 4, 4]
    rigid_mat = rigid.get_rigid_matrix(homogenous=True)
    save_tensor(f"{prefix}_rigid_mat", rigid_mat)

    # Save as [N, 3, 4] for apply_affine_transform
    rigid_34 = rigid.get_rigid_matrix(homogenous=False)
    save_tensor(f"{prefix}_rigid_34", rigid_34)

    # Evaluate warped image
    moved = rigid.evaluate(fixed_batch, moving_batch)
    save_tensor(f"{prefix}_moved", moved)

    # NCC
    loss_fn = LocalNormalizedCrossCorrelationLoss(spatial_dims=3, kernel_size=9, reduction="mean")
    ncc = loss_fn(moved, fixed_batch())
    print(f"  NCC loss after rigid: {ncc.item():.6f}")
    save_tensor(f"{prefix}_ncc", ncc.detach().unsqueeze(0))

    # Also save the raw parameters for debugging
    print(f"  quaternion: {rigid.rotation.data.cpu().numpy()}")
    print(f"  translation: {rigid.transl.data.cpu().numpy()}")
    print(f"  rigid matrix:\n{rigid_34[0].detach().cpu().numpy()}")


if __name__ == "__main__":
    export_rigid("small",
                 "validate/small/MNI152_T1_2mm.nii.gz",
                 "validate/small/T1_head_2mm.nii.gz",
                 dict(scales=[4, 2, 1], iterations=[200, 100, 50],
                      loss_type="mi", optimizer="Adam", optimizer_lr=3e-3, blur=True))

    export_rigid("medium",
                 "validate/medium/MNI152_T1_1mm_brain.nii.gz",
                 "validate/medium/t1_brain.nii.gz",
                 dict(scales=[4, 2, 1], iterations=[200, 100, 50],
                      loss_type="cc", cc_kernel_size=5,
                      optimizer="Adam", optimizer_lr=3e-3, blur=True))

    print(f"\nAll rigid data exported to {OUT_DIR}")
