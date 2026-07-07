SGR-FM Phase-2 training asset generation
========================================

Purpose
-------
Create the training-only assets required before C3:
  1) 400 raw training CT segmentations (200 EXP + 200 INSP)
     - anatomical support
     - five lobes
     - derived lobe interfaces
  2) 200 C1 support-guided uniGradICON initializers
     - canonical RAS DVFs by default
  3) a strict count/geometry audit

Key experimental policy
-----------------------
- No validation GT is used in this phase.
- Accepted C1 settings are NOT tuned:
    mode       = soft
    outside_hu = -1024
    fade_mm    = 7.5
    io_iterations = None
    io_sim     = lncc2
- The C1 training runner requires its fingerprint to match:
    outputs/sgr_fm/ablation_C1_support_guided_fm/run_config_resolved.json
- Frozen C0 DVF repositories remain protected and untouched.
- Training segmentation outputs are stored separately from the completed
  validation segmentation benchmark, so its manifests are preserved.
- Only canonical-RAS C1 training DVFs are saved by default to reduce disk use.

Files added
-----------
run_sgr_fm_training_assets.py
run_sgr_fm_c1_training.py
utils/sgr_fm_training_assets.py
configs/sgr_fm_phase2_training_assets.yaml
configs/sgr_fm_phase2_training_segmentation.yaml
configs/sgr_fm_phase2_c1_training.yaml

main.py is patched with:
  python main.py training-assets ...
  python main.py c1-train ...

Recommended execution
---------------------
1) Overlay patch from project root:
   unzip -o SGR_FM_phase2_training_assets_patch.zip -d .

2) One-case dry run of the full phase:
   python main.py training-assets \
     --config configs/sgr_fm_phase2_training_assets.yaml \
     --case-ids NLST_0011 \
     --dry-run

3) One-case real smoke test:
   python main.py training-assets \
     --config configs/sgr_fm_phase2_training_assets.yaml \
     --case-ids NLST_0011

4) Full 200-case run:
   python main.py training-assets \
     --config configs/sgr_fm_phase2_training_assets.yaml

Resume behavior
---------------
Both segmentation and C1 generation are resumable. Existing valid outputs are
reused unless --overwrite is explicitly provided.

Useful stage-only commands
--------------------------
Segmentation only:
  python main.py training-assets \
    --config configs/sgr_fm_phase2_training_assets.yaml \
    --stages segmentation

C1 only (after segmentation exists):
  python main.py training-assets \
    --config configs/sgr_fm_phase2_training_assets.yaml \
    --stages c1

Audit only:
  python main.py training-assets \
    --config configs/sgr_fm_phase2_training_assets.yaml \
    --stages audit

Direct C1 training runner:
  python main.py c1-train \
    --config configs/sgr_fm_phase2_c1_training.yaml

Expected full outputs
---------------------
outputs/sgr_fm/segmentation_phase1_totalsegmentator_training/
  support/training/       400 files
  lobes/training/         400 files
  interfaces/training/    400 files

outputs/sgr_fm/c1_support_guided_training/
  dvfs_canonical_ras/     200 files
  dvfs/                   0 files by default
  case_manifests/         200 JSON files

outputs/sgr_fm/phase2_training_assets/audit/
  phase2_training_asset_audit_summary.json
  phase2_training_asset_audit.json
  phase2_training_asset_audit.csv

Do not start C3 training until the final audit status is PASS.
