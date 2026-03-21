#!/usr/bin/env python
"""Export intermediate tensors from Python FireANTs for C validation.

Saves raw float32 binary files (.bin) with shape metadata (.json) that
the C test programs can load and compare against.
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from fireants.io.image import Image, BatchedImages
from fireants.losses.cc import LocalNormalizedCrossCorrelationLoss

OUT_DIR = os.path.join(os.path.dirname(__file__), "test_data")
os.makedirs(OUT_DIR, exist_ok=True)


def save_tensor(name, t):
    """Save a torch tensor as raw binary + JSON metadata."""
    t_cpu = t.detach().cpu().contiguous().float()
    data = t_cpu.numpy()
    bin_path = os.path.join(OUT_DIR, f"{name}.bin")
    json_path = os.path.join(OUT_DIR, f"{name}.json")
    data.tofile(bin_path)
    with open(json_path, "w") as f:
        json.dump({"shape": list(data.shape), "dtype": "float32"}, f)
    print(f"  {name}: shape={list(data.shape)} -> {bin_path}")


def export_grid_sample_test():
    """Export grid_sample inputs and outputs for validation."""
    print("=== Grid sample test data ===")

    # Load small MNI as the image to sample from
    img = Image.load_file("validate/small/MNI152_T1_2mm.nii.gz", device="cpu")
    batch = BatchedImages([img])
    input_tensor = batch()  # [1, 1, 91, 109, 91]
    save_tensor("gs_input", input_tensor)

    # Create an affine transform (slight rotation + translation)
    theta = torch.eye(3, 4, dtype=torch.float32).unsqueeze(0)  # [1, 3, 4]
    # Small rotation around Z axis (5 degrees)
    angle = torch.tensor(5.0 * 3.14159 / 180.0)
    theta[0, 0, 0] = torch.cos(angle)
    theta[0, 0, 1] = -torch.sin(angle)
    theta[0, 1, 0] = torch.sin(angle)
    theta[0, 1, 1] = torch.cos(angle)
    # Small translation
    theta[0, 0, 3] = 0.05
    theta[0, 1, 3] = -0.03
    theta[0, 2, 3] = 0.02

    # Generate sampling grid
    grid = F.affine_grid(theta, input_tensor.shape, align_corners=True)  # [1,91,109,91,3]
    save_tensor("gs_grid", grid)
    save_tensor("gs_affine", theta)

    # Run grid_sample
    output = F.grid_sample(input_tensor, grid, mode="bilinear",
                           padding_mode="zeros", align_corners=True)
    save_tensor("gs_output", output)

    # Also test with a displacement field added to identity
    identity_theta = torch.eye(3, 4, dtype=torch.float32).unsqueeze(0)
    identity_grid = F.affine_grid(identity_theta, input_tensor.shape, align_corners=True)

    # Small smooth displacement field
    torch.manual_seed(42)
    disp = torch.randn(1, 3, 10, 12, 10) * 0.02
    disp_upsampled = F.interpolate(disp, size=(91, 109, 91),
                                   mode="trilinear", align_corners=True)
    disp_field = disp_upsampled.permute(0, 2, 3, 4, 1)  # [1, 91, 109, 91, 3]
    warp_grid = identity_grid + disp_field
    save_tensor("gs_disp_grid", warp_grid)

    output_disp = F.grid_sample(input_tensor, warp_grid, mode="bilinear",
                                padding_mode="zeros", align_corners=True)
    save_tensor("gs_disp_output", output_disp)


def export_cc_loss_test():
    """Export CC loss inputs and outputs for validation."""
    print("\n=== CC loss test data ===")

    # Load two images
    fixed_img = Image.load_file("validate/small/MNI152_T1_2mm.nii.gz", device="cpu")
    moving_img = Image.load_file("validate/small/T1_head_2mm.nii.gz", device="cpu")

    fixed_batch = BatchedImages([fixed_img])
    moving_batch = BatchedImages([moving_img])

    fixed_t = fixed_batch()  # [1,1,91,109,91]

    # Resample moving into fixed space via identity (for size matching)
    identity = torch.eye(3, 4, dtype=torch.float32).unsqueeze(0)
    grid = F.affine_grid(identity, fixed_t.shape, align_corners=True)
    # Use the moving image's coordinate transform
    t2p_fixed = fixed_batch.get_torch2phy()
    p2t_moving = moving_batch.get_phy2torch()
    combined = torch.matmul(p2t_moving, t2p_fixed)[:, :3]  # [1, 3, 4]
    grid_phys = F.affine_grid(combined, fixed_t.shape, align_corners=True)
    moved_t = F.grid_sample(moving_batch(), grid_phys, mode="bilinear",
                            padding_mode="zeros", align_corners=True)

    save_tensor("cc_fixed", fixed_t)
    save_tensor("cc_moved", moved_t)

    # Compute CC loss with different kernel sizes
    for ks in [3, 5, 7]:
        loss_fn = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3, kernel_size=ks, reduction="mean",
        )
        moved_t_grad = moved_t.clone().requires_grad_(True)
        loss = loss_fn(moved_t_grad, fixed_t)
        loss.backward()

        save_tensor(f"cc_loss_k{ks}", loss.detach().unsqueeze(0))
        save_tensor(f"cc_grad_k{ks}", moved_t_grad.grad)
        print(f"  CC loss (k={ks}): {loss.item():.6f}")


def export_gaussian_blur_test():
    """Export Gaussian blur inputs and outputs for validation."""
    print("\n=== Gaussian blur test data ===")

    fixed_img = Image.load_file("validate/small/MNI152_T1_2mm.nii.gz", device="cpu")
    fixed_batch = BatchedImages([fixed_img])
    fixed_t = fixed_batch()

    save_tensor("blur_input", fixed_t)

    # Apply separable Gaussian blur with different sigmas
    from fireants.losses.cc import separable_filtering, gaussian_1d

    for sigma_val in [0.5, 1.0, 2.0]:
        sigma = torch.tensor([sigma_val], dtype=torch.float32)
        kernels = [gaussian_1d(sigma, truncated=4.0) for _ in range(3)]
        blurred = separable_filtering(fixed_t, kernels)
        save_tensor(f"blur_output_s{sigma_val}", blurred)
        print(f"  Blur sigma={sigma_val}: range=[{blurred.min():.2f}, {blurred.max():.2f}]")


def export_adam_test():
    """Export Adam optimizer test data."""
    print("\n=== Adam update test data ===")

    torch.manual_seed(123)
    # Simulate a warp field update
    param = torch.randn(1, 20, 24, 20, 3) * 0.01
    grad = torch.randn_like(param) * 0.001
    exp_avg = torch.zeros_like(param)
    exp_avg_sq = torch.zeros_like(param)

    save_tensor("adam_param", param)
    save_tensor("adam_grad", grad)

    # Run 3 Adam steps
    lr, beta1, beta2, eps = 0.1, 0.9, 0.999, 1e-8
    for step in range(1, 4):
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bc1 = 1 - beta1 ** step
        bc2 = 1 - beta2 ** step
        step_size = lr / bc1
        denom = (exp_avg_sq.sqrt() / (bc2 ** 0.5)).add_(eps)
        param.add_(exp_avg / denom, alpha=-step_size)
        # Generate new gradient for next step
        grad = torch.randn_like(param) * 0.001

    save_tensor("adam_param_after3", param)
    save_tensor("adam_exp_avg_after3", exp_avg)
    save_tensor("adam_exp_avg_sq_after3", exp_avg_sq)


if __name__ == "__main__":
    export_grid_sample_test()
    export_cc_loss_test()
    export_gaussian_blur_test()
    export_adam_test()
    print(f"\nAll test data exported to {OUT_DIR}")
