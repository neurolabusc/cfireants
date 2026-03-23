# Skullstrip

This validation dataset comes from the [allineate](https://github.com/neurolabusc/allineate) project. The goal is to de-identify anatomical scans by estimating an affine transform that maps standard space to the individual's native space. This transform is then used to warp a dilated brain mask from standard space into native space, with voxels outside the mask set to the darkest intensity in the native image. The desired output is the individual's anatomical scan with non-brain tissue removed. The original mask is binary, with zero outside the brain and one inside. It is warped using trilinear interpolation and thresholded at 0.5.

The goal is to show that cfireANTS can achieve similar skull stripping to allineate while leveraging the proven ANTs affine transform and GPU acceleration. The allineate command is:

```bash
allineate MNI152_T1_2mm T1_head_2mm -cost ls -skullstrip mniMask.nii.gz ./out/T1ls_2mm_mask
```

The equivalent command for cfireants is:

```bash
cfireants_reg \
  -f T1_head_2mm.nii.gz \
  -m MNI152_T1_2mm.nii.gz \
  --affine --trilinear \
  --skullstrip mniMask.nii.gz \
  -o out/bT1_head_2mm.nii.gz
```

This produces:
- `out/bT1_head_2mm.nii.gz` — Subject image with non-brain voxels set to darkest intensity (e.g. brain extracted)

The pipeline:
1. Moments initialization (center-of-mass + orientation matching)
2. Rigid registration (MI loss, 3 scales)
3. Affine registration (MI loss, 3 scales)
4. Warp mask from MNI space to subject space using the affine transform
5. Threshold warped mask at 0.5 (trilinear interpolation produces fractional values)
6. Apply: voxels outside mask set to the darkest intensity in the subject image

Notes:
- When `--skullstrip` is used, `-o` specifies the output filename directly (not a prefix)
- Output preserves the native datatype of the input image (UINT16, INT16, FLOAT32, etc.)
- For MRI magnitude images, the darkest voxel is typically 0 (air background)
- For CT scans, the darkest voxel is typically ~-1024 (air in Hounsfield units)
- The `--affine` preset is appropriate since skull stripping doesn't need deformable registration
- The `--trilinear` flag uses GPU-native downsampling (faster, no FFT dependency)
