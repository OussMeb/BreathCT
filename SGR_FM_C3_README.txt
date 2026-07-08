SGR-FM PHASE 3 — C3 LOBE-GUIDED RESIDUAL REFINEMENT
=====================================================

PURPOSE
-------
C3 starts from the accepted C1 support-guided uniGradICON initializer and learns
only a small segmentation-guided residual correction. It is designed to improve
lobe alignment without giving back the near-zero folding achieved by C1.

The refinement training uses only Phase-2 predicted assets from the 200 training
pairs. Validation GT is never used as a network input or training target.

C3 NETWORK INPUTS (17 CHANNELS)
-------------------------------
  5  fixed INSP predicted lobe one-hot channels
  5  moving EXP predicted lobes warped by C1
  5  absolute lobe mismatch channels
  1  fixed predicted anatomical support
  1  moving predicted support warped by C1

No CT clipping, normalization, preprocessing dataset, or validation labels are
required by C3.

RESIDUAL GEOMETRY
-----------------
- Model output: unit-bounded residual SVF
- Max residual bound: 0.75 voxel on the full-resolution grid
- Scaling-and-squaring steps: 4
- Exact pull composition is used:

    final(x) = residual(x) + C1(x + residual(x))

This is intentionally not the old additive approximation C1 + residual.

TRAINING OBJECTIVE
------------------
- equal-weight five-lobe soft Dice loss
- differentiable lobe-interface/boundary Dice loss
- residual bending penalty
- residual magnitude penalty
- Jacobian barrier on the composed C1+C3 field

The first-pass weights in the YAML are fixed design defaults; they were not
selected by validation-set tuning.

INTERNAL SPLIT
--------------
The 200 training pairs are split deterministically with seed 2026:
  160 training cases
   40 internal holdout cases

The exact case lists and split SHA256 are saved to split.json.

FILES ADDED
-----------
  freeze_sgr_fm_phase2_assets.py
  prepare_sgr_fm_c3_cache.py
  train_sgr_fm_c3.py
  infer_sgr_fm_c3_validation.py
  evaluate_sgr_fm_c3_holdout.py
  models/sgr_fm_lobe_refiner.py
  utils/sgr_fm_c3.py
  configs/sgr_fm_c3_lobe_refiner.yaml
  SGR_FM_C3_README.txt

main.py gains:
  phase2-freeze
  c3-cache
  c3-train
  c3-holdout
  c3-infer

STEP 1 — OVERLAY PATCH
----------------------
cd "/home/oussama/Desktop/MICCAI FRANCE"
unzip -o SGR_FM_C3_lobe_guided_refinement_patch.zip -d .

STEP 2 — FREEZE ACCEPTED PHASE-2 ASSETS
---------------------------------------
python main.py phase2-freeze freeze \
  --project-root "/home/oussama/Desktop/MICCAI FRANCE" \
  --make-read-only

Then verify:

python main.py phase2-freeze verify \
  --manifest "/home/oussama/Desktop/MICCAI FRANCE/outputs/sgr_fm/frozen_assets/phase2_assets_freeze_manifest.json" \
  --require-read-only

Expected tracked inventory:
  support masks          400
  five-lobe masks        400
  interface maps         400
  canonical C1 DVFs      200
  total                  1400

Do not proceed unless verification status is PASS.

STEP 3 — ONE-CASE CACHE SMOKE TEST
----------------------------------
python main.py c3-cache \
  --config configs/sgr_fm_c3_lobe_refiner.yaml \
  --case-ids NLST_0011

Inspect:
  outputs/sgr_fm/c3_lobe_refinement/cache_lowres/c3_cache_summary.json

STEP 4 — BUILD FULL 200-CASE C3 CACHE
-------------------------------------
python main.py c3-cache \
  --config configs/sgr_fm_c3_lobe_refiner.yaml

Expected:
  status       PASS
  case_count   200
  failures     0

The compact cache prevents rereading ~20 GiB of full C1 DVFs every epoch.

STEP 5 — TRAIN C3
-----------------
python main.py c3-train \
  --config configs/sgr_fm_c3_lobe_refiner.yaml

Outputs:
  outputs/sgr_fm/c3_lobe_refinement/training/
    config_resolved.json
    split.json
    holdout_c1_proxy_baseline.json
    train_history.json
    c3_training_summary.json
    checkpoints/latest.pt
    checkpoints/best_dice.pt
    checkpoints/best_guarded.pt       (only when topology guard is met)
    checkpoints/epoch_XXX.pt

STEP 6 — FULL-RESOLUTION INTERNAL HOLDOUT AUDIT
------------------------------------------------
First audit the topology-guarded checkpoint on the 40 held-out training cases:

python main.py c3-holdout \
  --checkpoint outputs/sgr_fm/c3_lobe_refinement/training/checkpoints/best_guarded.pt

This uses predicted training lobes only. It does not touch the 10 validation
cases or validation GT. It reports full-resolution C1-vs-C3 proxy lobe Dice and
full-resolution folding after exact composition.

Upload these files before any validation inference:
  training/c3_training_summary.json
  training/train_history.json
  training/holdout_c1_proxy_baseline.json
  training/split.json
  training/config_resolved.json
  holdout_epoch_XXX/c3_holdout_summary.json
  holdout_epoch_XXX/c3_holdout_per_case.csv

The validation inference runner is included now so the final checkpoint can be
run without another code patch, but it should not be used until the full-resolution
internal holdout has been reviewed and the checkpoint choice is locked.

LATER — LOCKED VALIDATION INFERENCE
-----------------------------------
Example only; do not run before holdout review:

python main.py c3-infer \
  --checkpoint outputs/sgr_fm/c3_lobe_refinement/training/checkpoints/best_guarded.pt

A complete 10-case run writes challenge-grid DVFs, canonical-RAS DVFs, exact
project evaluation metrics, and C1-vs-C3 comparison files.
