# Learn2Breath Dataset Inspection

Generated UTC: 2026-05-29T03:08:44.043811+00:00

## Main files

- `nifti_inventory.csv`: every discovered NIfTI file.
- `nifti_metadata.csv`: shape, spacing, affine hash, orientation, dtype.
- `ct_intensity_stats.csv`: CT HU/value distribution per image.
- `segmentation_label_stats.csv`: label IDs, voxel counts, physical volumes.
- `case_summary.csv`: per-case completeness and mismatch checks.
- `issues.csv`: duplicated/missing/mismatched files.
- `identity_lobe_dice_summary.csv`: no-registration EXP-vs-INSP lobe Dice.
- `identity_lobe_dice_per_label.csv`: identity Dice per lobe label.
- `directory_tree.csv`: folder tree summary.
- `dataset_summary.json`: compact global summary.
- `manifest.json`: full machine-readable manifest.
- `example_images/`: quicklook PNGs for selected cases.

## Critical checks

1. Check `issues.csv` first.
2. Check whether training labels exist in `case_summary.csv`.
3. Check CT shape/spacing/orientation consistency in `nifti_metadata.csv`.
4. Check no-registration difficulty using `identity_lobe_dice_summary.csv`.
5. Visually inspect PNGs in `example_images/`.

## Expected challenge assumptions

Expected CT shape: (224, 192, 224)
Expected spacing: (1.5, 1.5, 1.5)
Moving image: EXP
Fixed image: INSP
