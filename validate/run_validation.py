#!/usr/bin/env python
"""
FireANTs validation suite.

Registers moving images to MNI templates using a full ANTs-style pipeline
(Moments -> Rigid -> Affine -> SyN) and reports NCC, wall-clock time, and
peak GPU memory for each dataset.

Usage:
    # Run all datasets
    python validate/run_validation.py

    # Run a single dataset
    python validate/run_validation.py --dataset small
    python validate/run_validation.py --dataset medium
    python validate/run_validation.py --dataset large

    # Generate reference outputs (saves warped images + metrics JSON)
    python validate/run_validation.py --save-reference

    # Compare against saved reference outputs
    python validate/run_validation.py --check-reference
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from fireants.io.image import Image, BatchedImages
from fireants.losses.cc import LocalNormalizedCrossCorrelationLoss
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration
from fireants.registration.moments import MomentsRegistration
from fireants.registration.rigid import RigidRegistration
from fireants.registration.syn import SyNRegistration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VALIDATE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = VALIDATE_DIR / "reference"

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
# Each dataset specifies its image paths and registration parameters tuned
# to the specific characteristics of the data (brain-extracted vs full-head,
# resolution, intensity range).
# ---------------------------------------------------------------------------

DATASETS = {
    "small": {
        "description": "2mm full-head MNI to subject (includes scalp)",
        "stationary": "small/MNI152_T1_2mm.nii.gz",
        "moving": "small/T1_head_2mm.nii.gz",
        "params": {
            # Full-head images with very different intensity ranges →
            # MI is more robust for linear stages
            "moments": dict(
                scale=1.0,
                moments=2,
                orientation="both",
                blur=True,
                loss_type="cc",
                cc_kernel_size=5,
            ),
            "rigid": dict(
                scales=[4, 2, 1],
                iterations=[200, 100, 50],
                loss_type="mi",
                optimizer="Adam",
                optimizer_lr=3e-3,
                blur=True,
            ),
            "affine": dict(
                scales=[4, 2, 1],
                iterations=[200, 100, 50],
                loss_type="mi",
                optimizer="Adam",
                optimizer_lr=1e-3,
                blur=True,
            ),
            "syn": dict(
                scales=[4, 2, 1],
                iterations=[200, 100, 50],
                loss_type="cc",
                cc_kernel_size=5,
                optimizer="Adam",
                optimizer_lr=0.1,
                smooth_warp_sigma=0.5,
                smooth_grad_sigma=1.0,
            ),
        },
    },
    "medium": {
        "description": "1mm brain-extracted MNI to subject",
        "stationary": "medium/MNI152_T1_1mm_brain.nii.gz",
        "moving": "medium/t1_brain.nii.gz",
        "params": {
            # Brain-extracted, single-modality → CC works well throughout
            "moments": dict(
                scale=1.0,
                moments=2,
                orientation="both",
                blur=True,
                loss_type="cc",
                cc_kernel_size=5,
            ),
            "rigid": dict(
                scales=[4, 2, 1],
                iterations=[200, 100, 50],
                loss_type="cc",
                cc_kernel_size=5,
                optimizer="Adam",
                optimizer_lr=3e-3,
                blur=True,
            ),
            "affine": dict(
                scales=[4, 2, 1],
                iterations=[200, 100, 50],
                loss_type="cc",
                cc_kernel_size=5,
                optimizer="Adam",
                optimizer_lr=1e-3,
                blur=True,
            ),
            "syn": dict(
                scales=[4, 2, 1],
                iterations=[200, 100, 50],
                loss_type="cc",
                cc_kernel_size=5,
                optimizer="Adam",
                optimizer_lr=0.1,
                smooth_warp_sigma=0.5,
                smooth_grad_sigma=1.0,
            ),
        },
    },
    "large": {
        "description": "1mm full-head MNI to subject (0.88mm, includes scalp)",
        "stationary": "large/MNI152_T1_1mm.nii.gz",
        "moving": "large/chris_t1.nii.gz",
        "params": {
            # Full-head, different resolutions, uint8 moving → MI for linear
            "moments": dict(
                scale=1.0,
                moments=2,
                orientation="both",
                blur=True,
                loss_type="cc",
                cc_kernel_size=5,
            ),
            "rigid": dict(
                scales=[8, 4, 2, 1],
                iterations=[200, 200, 100, 50],
                loss_type="mi",
                optimizer="Adam",
                optimizer_lr=3e-3,
                blur=True,
            ),
            "affine": dict(
                scales=[8, 4, 2, 1],
                iterations=[200, 200, 100, 50],
                loss_type="mi",
                optimizer="Adam",
                optimizer_lr=1e-3,
                blur=True,
            ),
            "syn": dict(
                scales=[4, 2, 1],
                iterations=[200, 100, 50],
                loss_type="cc",
                cc_kernel_size=5,
                optimizer="Adam",
                optimizer_lr=0.1,
                smooth_warp_sigma=0.5,
                smooth_grad_sigma=1.0,
            ),
        },
    },
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ncc(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Global normalized cross-correlation between two images (scalar)."""
    a = img_a.astype(np.float64).ravel()
    b = img_b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-12:
        return 0.0
    return float((a * b).sum() / denom)


def compute_local_ncc_loss(fixed_tensor: torch.Tensor, moved_tensor: torch.Tensor,
                           kernel_size: int = 9) -> float:
    """Compute FireANTs local NCC loss (lower = better match)."""
    loss_fn = LocalNormalizedCrossCorrelationLoss(
        spatial_dims=3, kernel_size=kernel_size, reduction="mean",
    )
    with torch.no_grad():
        val = loss_fn(fixed_tensor, moved_tensor)
    return float(val.cpu())


# ---------------------------------------------------------------------------
# Registration pipeline
# ---------------------------------------------------------------------------

def run_registration(dataset_name: str, device: str = "cuda") -> dict:
    """Run the full registration pipeline for one dataset.

    Returns a dict with timing, memory, NCC metrics, and output paths.
    """
    cfg = DATASETS[dataset_name]
    params = cfg["params"]

    stationary_path = str(VALIDATE_DIR / cfg["stationary"])
    moving_path = str(VALIDATE_DIR / cfg["moving"])

    output_dir = VALIDATE_DIR / dataset_name / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Dataset: %s — %s", dataset_name, cfg["description"])
    logger.info("  Stationary: %s", cfg["stationary"])
    logger.info("  Moving:     %s", cfg["moving"])
    logger.info("=" * 60)

    # Load images
    fixed_img = Image.load_file(stationary_path, device=device)
    moving_img = Image.load_file(moving_path, device=device)
    fixed_batch = BatchedImages([fixed_img])
    moving_batch = BatchedImages([moving_img])

    # Pre-registration NCC (resample moving into fixed space for comparison)
    fixed_sitk = sitk.ReadImage(stationary_path, sitk.sitkFloat32)
    moving_sitk = sitk.ReadImage(moving_path, sitk.sitkFloat32)
    moving_resampled = sitk.Resample(
        moving_sitk, fixed_sitk,
        sitk.Transform(), sitk.sitkLinear, 0.0,
        moving_sitk.GetPixelID(),
    )
    fixed_arr = sitk.GetArrayFromImage(fixed_sitk).astype(np.float32)
    moving_arr_resampled = sitk.GetArrayFromImage(moving_resampled).astype(np.float32)
    ncc_before = compute_ncc(fixed_arr, moving_arr_resampled)
    logger.info("NCC before registration: %.4f", ncc_before)

    # Reset peak memory tracking
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()

    # --- Stage 1: Moments ---
    logger.info("Stage 1/4: Moments registration")
    moments = MomentsRegistration(
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        **params["moments"],
    )
    moments.optimize()
    affine_init = moments.get_affine_init()
    t_moments = time.perf_counter()

    # --- Stage 2: Rigid ---
    logger.info("Stage 2/4: Rigid registration")
    rigid = RigidRegistration(
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        init_moment=moments.get_rigid_moment_init(),
        init_translation=moments.get_rigid_transl_init(),
        **params["rigid"],
    )
    rigid.optimize()
    t_rigid = time.perf_counter()

    # --- Stage 3: Affine ---
    logger.info("Stage 3/4: Affine registration")
    affine = AffineRegistration(
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        init_rigid=rigid.get_rigid_matrix(homogenous=False),
        **params["affine"],
    )
    affine.optimize()
    affine_matrix = affine.get_affine_matrix()
    t_affine = time.perf_counter()

    # --- Stage 4: SyN ---
    logger.info("Stage 4/4: SyN deformable registration")
    syn = SyNRegistration(
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        init_affine=affine_matrix,
        **params["syn"],
    )
    syn.optimize()
    t_syn = time.perf_counter()

    t_total = t_syn - t_start

    # Peak GPU memory
    if device == "cuda" and torch.cuda.is_available():
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        peak_mem_mb = peak_mem_bytes / (1024 * 1024)
    else:
        peak_mem_mb = 0.0

    # Evaluate and save warped image
    with torch.no_grad():
        moved_tensor = syn.evaluate(fixed_batch, moving_batch).detach()

    warped_path = str(output_dir / "warped.nii.gz")
    syn.save_moved_images(moved_tensor, warped_path)
    logger.info("Saved warped image: %s", warped_path)

    # Save transforms
    syn.save_as_ants_transforms(str(output_dir / "warp_field.nii.gz"))
    affine.save_as_ants_transforms(str(output_dir / "affine_transform.mat"))

    # Post-registration NCC (read back saved warped image for consistency)
    warped_arr = sitk.GetArrayFromImage(sitk.ReadImage(warped_path)).astype(np.float32)
    ncc_after = compute_ncc(fixed_arr, warped_arr)

    # Local NCC loss (FireANTs metric)
    fixed_tensor = fixed_batch().detach()
    local_ncc_loss = compute_local_ncc_loss(fixed_tensor, moved_tensor)

    logger.info("NCC after registration:  %.4f", ncc_after)
    logger.info("Local NCC loss:          %.4f", local_ncc_loss)
    logger.info("Total time:              %.1fs", t_total)
    logger.info("Peak GPU memory:         %.0f MB", peak_mem_mb)

    results = {
        "dataset": dataset_name,
        "description": cfg["description"],
        "ncc_before": round(ncc_before, 4),
        "ncc_after": round(ncc_after, 4),
        "local_ncc_loss": round(local_ncc_loss, 4),
        "time_seconds": round(t_total, 1),
        "time_moments_s": round(t_moments - t_start, 1),
        "time_rigid_s": round(t_rigid - t_moments, 1),
        "time_affine_s": round(t_affine - t_rigid, 1),
        "time_syn_s": round(t_syn - t_affine, 1),
        "peak_gpu_memory_mb": round(peak_mem_mb, 0),
        "warped_path": warped_path,
    }

    # Save per-dataset metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Metrics saved: %s", metrics_path)

    return results


# ---------------------------------------------------------------------------
# Reference management
# ---------------------------------------------------------------------------

def save_reference(results: dict) -> None:
    """Save results as reference outputs for future comparison."""
    name = results["dataset"]
    ref_dir = REFERENCE_DIR / name
    ref_dir.mkdir(parents=True, exist_ok=True)

    # Copy warped image to reference
    import shutil
    src_warped = results["warped_path"]
    dst_warped = str(ref_dir / "warped.nii.gz")
    shutil.copy2(src_warped, dst_warped)

    # Save reference metrics
    ref_metrics = {k: v for k, v in results.items() if k != "warped_path"}
    with open(ref_dir / "metrics.json", "w") as f:
        json.dump(ref_metrics, f, indent=2)

    logger.info("Reference saved for '%s' in %s", name, ref_dir)


def check_reference(results: dict) -> bool:
    """Compare current results against saved reference.

    Returns True if NCC is within tolerance of reference.
    """
    name = results["dataset"]
    ref_dir = REFERENCE_DIR / name
    ref_metrics_path = ref_dir / "metrics.json"

    if not ref_metrics_path.exists():
        logger.warning("No reference found for '%s' — skipping comparison", name)
        return True

    with open(ref_metrics_path) as f:
        ref = json.load(f)

    ref_ncc = ref["ncc_after"]
    cur_ncc = results["ncc_after"]
    # Allow 2% relative degradation
    tolerance = 0.02
    degraded = cur_ncc < ref_ncc * (1 - tolerance)

    logger.info("Reference comparison for '%s':", name)
    logger.info("  Reference NCC: %.4f", ref_ncc)
    logger.info("  Current NCC:   %.4f", cur_ncc)
    logger.info("  Ref time:      %.1fs  |  Current: %.1fs", ref["time_seconds"], results["time_seconds"])
    logger.info("  Ref GPU mem:   %.0f MB  |  Current: %.0f MB", ref["peak_gpu_memory_mb"], results["peak_gpu_memory_mb"])

    if degraded:
        logger.error(
            "  REGRESSION: NCC dropped from %.4f to %.4f (>%.0f%% degradation)",
            ref_ncc, cur_ncc, tolerance * 100,
        )
        return False
    else:
        logger.info("  PASS")
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FireANTs validation suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", choices=list(DATASETS.keys()),
        help="Run a single dataset (default: all)",
    )
    parser.add_argument(
        "--save-reference", action="store_true",
        help="Save outputs as reference for future comparison",
    )
    parser.add_argument(
        "--check-reference", action="store_true",
        help="Compare results against saved reference outputs",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device to run on (default: cuda)",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())
    all_results = []
    all_pass = True

    for name in datasets:
        results = run_registration(name, device=args.device)
        all_results.append(results)

        if args.save_reference:
            save_reference(results)

        if args.check_reference:
            if not check_reference(results):
                all_pass = False

    # Print summary table
    print("\n" + "=" * 78)
    print(f"{'Dataset':<10} {'NCC Before':>11} {'NCC After':>11} {'Local NCC':>11} "
          f"{'Time (s)':>9} {'GPU (MB)':>9}")
    print("-" * 78)
    for r in all_results:
        print(f"{r['dataset']:<10} {r['ncc_before']:>11.4f} {r['ncc_after']:>11.4f} "
              f"{r['local_ncc_loss']:>11.4f} {r['time_seconds']:>9.1f} "
              f"{r['peak_gpu_memory_mb']:>9.0f}")
    print("=" * 78)

    if args.check_reference and not all_pass:
        logger.error("One or more datasets regressed against reference!")
        sys.exit(1)


if __name__ == "__main__":
    main()
