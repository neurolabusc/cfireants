#!/usr/bin/env python
"""Export moments registration intermediate data for C validation."""

import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from fireants.io.image import Image, BatchedImages
from fireants.registration.moments import MomentsRegistration

OUT_DIR = os.path.join(os.path.dirname(__file__), "test_data")
os.makedirs(OUT_DIR, exist_ok=True)


def save_tensor(name, t):
    t_cpu = t.detach().cpu().contiguous().float()
    data = t_cpu.numpy()
    bin_path = os.path.join(OUT_DIR, f"{name}.bin")
    json_path = os.path.join(OUT_DIR, f"{name}.json")
    data.tofile(bin_path)
    with open(json_path, "w") as f:
        json.dump({"shape": list(data.shape), "dtype": "float32"}, f)
    print(f"  {name}: shape={list(data.shape)}")


def export_moments(dataset_name, fixed_path, moving_path):
    print(f"\n=== Moments test data: {dataset_name} ===")

    fixed_img = Image.load_file(fixed_path, device="cpu")
    moving_img = Image.load_file(moving_path, device="cpu")
    fixed_batch = BatchedImages([fixed_img])
    moving_batch = BatchedImages([moving_img])

    # Run moments registration
    reg = MomentsRegistration(
        scale=1.0,
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        moments=2,
        orientation="both",
        blur=True,
        loss_type="cc",
        cc_kernel_size=5,
    )
    reg.optimize()

    # Export results
    prefix = f"mom_{dataset_name}"

    # Rotation matrix [N, 3, 3]
    Rf = reg.get_rigid_moment_init()
    save_tensor(f"{prefix}_Rf", Rf)
    print(f"  Rf det: {torch.linalg.det(Rf).item():.6f}")

    # Translation [N, 3]
    tf = reg.get_rigid_transl_init()
    save_tensor(f"{prefix}_tf", tf)

    # Full affine init [N, 3, 4]
    aff = reg.get_affine_init()
    save_tensor(f"{prefix}_affine_init", aff)

    # Coordinate transforms
    save_tensor(f"{prefix}_fixed_t2p", fixed_batch.get_torch2phy())
    save_tensor(f"{prefix}_fixed_p2t", fixed_batch.get_phy2torch())
    save_tensor(f"{prefix}_moving_t2p", moving_batch.get_torch2phy())
    save_tensor(f"{prefix}_moving_p2t", moving_batch.get_phy2torch())

    # Evaluate: warp moving to fixed space
    moved = reg.evaluate(fixed_batch, moving_batch)
    save_tensor(f"{prefix}_moved", moved)

    # Compute NCC
    from fireants.losses.cc import LocalNormalizedCrossCorrelationLoss
    loss_fn = LocalNormalizedCrossCorrelationLoss(spatial_dims=3, kernel_size=9, reduction="mean")
    ncc = loss_fn(moved, fixed_batch())
    print(f"  NCC loss after moments: {ncc.item():.6f}")
    save_tensor(f"{prefix}_ncc", ncc.detach().unsqueeze(0))

    print(f"  Rf:\n{Rf[0].numpy()}")
    print(f"  tf: {tf[0].numpy()}")


if __name__ == "__main__":
    export_moments("small",
                   "validate/small/MNI152_T1_2mm.nii.gz",
                   "validate/small/T1_head_2mm.nii.gz")
    export_moments("medium",
                   "validate/medium/MNI152_T1_1mm_brain.nii.gz",
                   "validate/medium/t1_brain.nii.gz")
    print(f"\nAll moments data exported to {OUT_DIR}")
