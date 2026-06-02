# 🫁 Learn2Breath MICCAI 2026 — Respiratory CT Registration Project

<p align="center">
  <b>EXP → INSP chest CT deformable registration</b><br>
  Foundation-model initialization + topology-aware refinement strategy
</p>

<p align="center">
  <img alt="Status" src="https://img.shields.io/badge/status-active%20research-blue">
  <img alt="Task" src="https://img.shields.io/badge/task-3D%20deformable%20registration-purple">
  <img alt="Modality" src="https://img.shields.io/badge/modality-CT-green">
  <img alt="Best Dice" src="https://img.shields.io/badge/best%20Dice-0.96467-brightgreen">
  <img alt="Best Folding" src="https://img.shields.io/badge/folding-0.03309%25-yellowgreen">
</p>

---

## 📌 Project root

All commands assume the project root is:

```bash
/home/oussama/Desktop/MICCAI FRANCE
```

---

## 🧠 1. Challenge definition

The **MICCAI Learn2Breath 2026** task is an intra-subject respiratory chest CT registration challenge.

The goal is:

```text
Register EXP CT → INSP CT

moving image = expiratory CT scan, EXP
fixed image  = inspiratory CT scan, INSP
output       = dense displacement vector field, DVF
DVF grid     = original INSP image grid
```

The submitted DVF is used to warp the **EXP lobe segmentation** into the **INSP space**.

The evaluator then compares:

```text
warped EXP lobe segmentation
vs
INSP lobe segmentation
```

Primary metrics:

| Metric | Meaning |
|---|---|
| 🫁 Lobe Dice | Anatomical overlap after registration |
| 🧬 Mean Dice | Mean Dice over evaluated lobes |
| ⚠️ Folding % | Percentage of voxels with non-positive Jacobian determinant |
| 🏁 Composite score | Dice penalized by deformation folding |

> [!IMPORTANT]
> This is **not only an image-similarity problem**.  
> The final score rewards **anatomically correct, topology-safe deformation**.

---

## 🗂️ 2. Dataset decomposition

### Dataset summary

| Split | Cases | CT pairs | Lobe labels | Fissure labels |
|---|---:|---:|---:|---:|
| Training | 200 | 200 EXP/INSP pairs | ❌ No | ❌ No |
| Validation | 10 | 10 EXP/INSP pairs | ✅ Yes | ✅ Yes |

Confirmed global image properties:

```text
shape:   224 × 192 × 224
spacing: 1.5 × 1.5 × 1.5 mm
modality: CT
pairing: EXP/INSP from the same subject
```

### Raw data layout

```text
Learn2Breath_train_val_data/
  training/
    NLST_0011_EXP.nii.gz
    NLST_0011_INSP.nii.gz
    ...
    NLST_0299_EXP.nii.gz
    NLST_0299_INSP.nii.gz

  validation/
    ct_data/
      NLST_0001_EXP.nii.gz
      NLST_0001_INSP.nii.gz
      ...
      NLST_0010_EXP.nii.gz
      NLST_0010_INSP.nii.gz

    seg_net/
      NLST_0001_EXP_lobe.nii.gz
      NLST_0001_INSP_lobe.nii.gz
      NLST_0001_EXP_fissure.nii.gz
      NLST_0001_INSP_fissure.nii.gz
      ...
```

### Label values

Validation lobe labels:

```text
0 background
8, 16, 32, 64, 128 lobes
```

Validation fissure labels:

```text
0 background
1, 2 fissure classes
```

---

## 🧭 3. Orientation and DVF conventions

This was a critical issue.

Training scans are mostly:

```text
RAS
```

Validation scans are:

```text
LPS
```

Our current policy:

```text
model/internal processing: canonical RAS
challenge/evaluation output: original INSP grid/header
```

Current DVF convention:

```text
grid:            original INSP grid
layout:          X, Y, Z, 3
units:           voxel displacement
warp convention: pull
formula:         warped_EXP[x] = EXP[x + DVF[x]]
component order: voxel axes X, Y, Z
```

> [!WARNING]
> Identity DVF proves shape/header/zip formatting, but it does **not** prove direction or voxel/mm convention because zero displacement is invariant.

---

## 🧪 4. Preprocessing decisions

We decided:

```text
no trimming
no resizing for final pipeline
HU clip: [-1000, 600]
scale: [-1, 1]
foreground mask from rough HU threshold
```

Reason:

| Decision | Reason |
|---|---|
| ❌ No trimming | Avoid EXP/INSP spatial shift and DVF reinsertion bugs |
| ❌ No permanent resizing | Submission DVF must be on original INSP grid |
| ✅ Foreground masks | Prevent empty black slices dominating loss |
| ✅ Raw CT for uniGradICON | `unigradicon-register` expects CT-like HU intensities |

---

## 🏗️ 5. Current project organization

Current intended organization:

```text
MICCAI FRANCE/
  main.py

  configs/
    identity_eval.yaml
    unigradicon_baseline.yaml
    refiner_cached_lowspace.yaml

  models/
    icon_model.py
    residual_refiner.py
    refinement.py
    unet3d.py

  utils/
    config.py
    data.py
    dvf_io.py
    losses.py
    metrics.py
    orientation.py
    refine_io.py
    refine_losses.py
    refine_spatial.py
    spatial.py
    unigradicon_runner.py

  run_unigradicon.py
  train_refiner_cached.py
  infer_refiner_cached.py
  evaluate_validation.py
  inspect_registration_results_streamlit.py

  Learn2Breath_train_val_data/
  outputs/
```

---

## 🧩 6. Main code utilities

### `main.py`

Primary command entry point.

Example:

```bash
python main.py unigradicon ...
```

---

### `run_unigradicon.py`

Runs raw pretrained uniGradICON.

Responsibilities:

```text
- discover cases
- call official unigradicon-register CLI
- save transforms
- convert transforms to challenge DVFs
- optionally evaluate validation results
```

---

### `utils/unigradicon_runner.py`

Core uniGradICON integration.

Handles:

```text
- training/validation split discovery
- official CLI execution
- HDF5 transform reading
- physical displacement → voxel DVF conversion
- original INSP-grid DVF export
- canonical RAS DVF export
```

---

### `evaluate_validation.py`

Local validation evaluator.

Computes:

```text
- lobe Dice
- per-lobe Dice
- Jacobian determinant stats
- folding percentage
```

---

### `inspect_registration_results_streamlit.py`

Streamlit visual viewer.

Shows:

```text
- moving EXP
- fixed INSP
- warped EXP
- before/after difference images
- lobe overlays
- fissure overlays
- warped moving lobe overlays
- Jacobian determinant
- folding mask
- validation metric plots
```

Run:

```bash
streamlit run inspect_registration_results_streamlit.py
```

---

### `train_refiner_cached.py`

Low-space residual refinement training.

Uses:

```text
raw CTs on the fly
cached uniGradICON DVFs
no saved warped training volumes
reduction_factor = 2
```

---

### `infer_refiner_cached.py`

Runs full-resolution validation inference using:

```text
raw validation CTs
cached validation uniGradICON initializer DVFs
trained residual refiner checkpoint
```

---

## 🚀 7. Important execution commands

### 7.1 Project tree inspection

```bash
cd "/home/oussama/Desktop/MICCAI FRANCE"

find . -maxdepth 2 -type d | sort
find . -maxdepth 4 -type f \( -name "*.py" -o -name "*.yaml" -o -name "*.json" \) | sort
```

---

### 7.2 Identity baseline

```bash
cd "/home/oussama/Desktop/MICCAI FRANCE"

PYTORCH_NVML_BASED_CUDA_CHECK=0 \
PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
python identity_eval.py \
  --config configs/identity_eval.yaml \
  --raw-data-root "/home/oussama/Desktop/MICCAI FRANCE/Learn2Breath_train_val_data" \
  --output-dir outputs/identity_validation \
  --make-identity \
  --evaluate \
  --make-zip \
  --overwrite
```

---

### 7.3 Raw uniGradICON on validation

```bash
cd "/home/oussama/Desktop/MICCAI FRANCE"

PYTHONPATH=. \
PYTORCH_NVML_BASED_CUDA_CHECK=0 \
python main.py unigradicon \
  --config configs/unigradicon_baseline.yaml \
  --split validation \
  --output-dir outputs/unigradicon_raw_validation \
  --overwrite \
  --evaluate \
  --make-zip
```

---

### 7.4 Raw uniGradICON sanity check on training

```bash
cd "/home/oussama/Desktop/MICCAI FRANCE"

PYTHONPATH=. \
PYTORCH_NVML_BASED_CUDA_CHECK=0 \
python main.py unigradicon \
  --config configs/unigradicon_baseline.yaml \
  --split training \
  --case-ids NLST_0011 NLST_0020 NLST_0299 \
  --output-dir outputs/unigradicon_train_debug \
  --overwrite
```

---

### 7.5 Raw uniGradICON on all training cases

```bash
cd "/home/oussama/Desktop/MICCAI FRANCE"

PYTHONPATH=. \
PYTORCH_NVML_BASED_CUDA_CHECK=0 \
python main.py unigradicon \
  --config configs/unigradicon_baseline.yaml \
  --split training \
  --output-dir outputs/unigradicon_raw_training \
  --overwrite
```

If interrupted, resume only missing cases:

```bash
cd "/home/oussama/Desktop/MICCAI FRANCE"

python - <<'PY'
from pathlib import Path

raw = Path("Learn2Breath_train_val_data/training")
dvf_dir = Path("outputs/unigradicon_raw_training/dvfs")

expected = sorted(p.name.replace("_INSP.nii.gz", "") for p in raw.glob("NLST_*_INSP.nii.gz"))
done = set(p.name.replace("_DVF.nii.gz", "") for p in dvf_dir.glob("NLST_*_DVF.nii.gz"))

missing = [case for case in expected if case not in done]
Path("outputs/unigradicon_raw_training/missing_cases.txt").write_text("\n".join(missing) + "\n")

print(f"expected={len(expected)}")
print(f"done={len(done)}")
print(f"missing={len(missing)}")
print(missing[:30])
PY
```

Then:

```bash
PYTHONPATH=. \
PYTORCH_NVML_BASED_CUDA_CHECK=0 \
python main.py unigradicon \
  --config configs/unigradicon_baseline.yaml \
  --split training \
  --case-ids $(cat outputs/unigradicon_raw_training/missing_cases.txt) \
  --output-dir outputs/unigradicon_raw_training \
  --overwrite
```

---

### 7.6 Low-space refiner smoke test

```bash
cd "/home/oussama/Desktop/MICCAI FRANCE"

PYTHONPATH=. \
PYTORCH_NVML_BASED_CUDA_CHECK=0 \
PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
python train_refiner_cached.py \
  --config configs/refiner_cached_lowspace.yaml \
  --epochs 1 \
  --output-dir outputs/refiner_cached_smoke
```

---

### 7.7 Low-space refiner 100 epochs

```bash
cd "/home/oussama/Desktop/MICCAI FRANCE"

PYTHONPATH=. \
PYTORCH_NVML_BASED_CUDA_CHECK=0 \
PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
python train_refiner_cached.py \
  --config configs/refiner_cached_lowspace.yaml \
  --epochs 100 \
  --output-dir outputs/refiner_cached_100
```

---

### 7.8 Refiner validation inference

```bash
cd "/home/oussama/Desktop/MICCAI FRANCE"

PYTHONPATH=. \
PYTORCH_NVML_BASED_CUDA_CHECK=0 \
python infer_refiner_cached.py \
  --config configs/refiner_cached_lowspace.yaml \
  --checkpoint outputs/refiner_cached_100/checkpoints/latest.pt \
  --output-dir outputs/refiner_cached_100_validation_latest \
  --make-zip \
  --overwrite
```

Evaluate:

```bash
PYTHONPATH=. \
python evaluate_validation.py \
  --raw-data-root "/home/oussama/Desktop/MICCAI FRANCE/Learn2Breath_train_val_data" \
  --dvf-dir outputs/refiner_cached_100_validation_latest/dvfs \
  --output-dir outputs/refiner_cached_100_validation_latest/eval
```

---

## 📊 8. Experimental results

### 8.1 Identity baseline

| Metric | Value |
|---|---:|
| Mean lobe Dice | `0.87195` |
| Min case Dice | `0.72990` |
| Max case Dice | `0.91675` |
| Mean folding | `0.0%` |
| Max folding | `0.0%` |

Interpretation:

> Identity is surprisingly strong.  
> Any model below `0.87195` mean Dice is useless.

---

### 8.2 Raw uniGradICON validation baseline

Configuration:

```text
model:         unigradicon
io_iterations: None
io_sim:        lncc2
fixed:         INSP CT
moving:        EXP CT
```

| Metric | Value |
|---|---:|
| Mean lobe Dice | `0.96467` |
| Min case Dice | `0.91953` |
| Max case Dice | `0.98210` |
| Mean folding | `0.03309%` |
| Max folding | `0.17399%` |

This is currently the **best valid result**.

Best current submission:

```text
outputs/unigradicon_raw_validation/submission_unigradicon_raw.zip
```

---

### 8.3 Raw uniGradICON training cache

Status:

```text
training cases: 200
cached challenge DVFs: 200
cached canonical RAS DVFs: 200
failed cases: 0
```

Useful folders:

```text
outputs/unigradicon_raw_training/dvfs/
outputs/unigradicon_raw_training/dvfs_canonical_ras/
```

---

### 8.4 Residual refiner smoke test

Smoke test was valid:

| Metric | Value |
|---|---:|
| Epoch | `1` |
| Total loss | `0.35460` |
| Image loss | `0.35414` |
| Bending | `0.00904` |
| Jacobian loss | `0.0000076` |
| Folding | `0.04509%` |

No NaNs. No crash. Training loop works.

---

### 8.5 Residual refiner 100 epochs

Training behavior:

| Epoch | Image loss | Folding |
|---:|---:|---:|
| 1 | `0.35414` | `0.045%` |
| 10 | `0.32199` | `0.227%` |
| 50 | `0.29431` | `0.380%` |
| 100 | `0.27672` | `0.486%` |

Interpretation:

> The refiner improves image similarity but increases folding.  
> That is the wrong direction for the challenge metric.

---

### 8.6 Refiner epoch 100 validation result

| Method | Mean Dice | Mean folding | Max folding |
|---|---:|---:|---:|
| Raw uniGradICON | **0.96467** | **0.03309%** | **0.17399%** |
| Refiner epoch 100 | `0.95915` | `0.15736%` | `0.52034%` |

Conclusion:

> ❌ Do **not** use epoch-100 refiner output.  
> Raw uniGradICON is better.

---

### 8.7 Refiner epoch 15 validation result

| Method | Mean Dice | Mean folding | Max folding |
|---|---:|---:|---:|
| Raw uniGradICON | **0.96467** | **0.03309%** | **0.17399%** |
| Refiner epoch 15 | `0.96261` | `0.06834%` | `0.24769%` |

Conclusion:

> ❌ Epoch 15 is also worse than raw uniGradICON.

---

## 🏆 9. Current challenge-board position estimate

From the leaderboard screenshot, top methods are approximately:

| Rank | Combined score | Lobe DSC | Folding |
|---:|---:|---:|---:|
| 🥇 1 | `0.97292` | `0.97292` | `0.0%` |
| 🥈 2 | `0.97155` | `0.97300` | `0.14504%` |
| 🧪 Our raw uniGradICON | `~0.96467` | `0.96467` | `0.03309%` |

Gap to top:

```text
needed Dice gain: roughly +0.005 to +0.009
needed topology: ideally folding → 0%
```

Current position:

> We are competitive, but not top-tier yet.  
> Raw uniGradICON is strong, but it is not enough to challenge top 3.

---

## 💾 10. Disk-space policy

Keep:

```text
Learn2Breath_train_val_data/

outputs/unigradicon_raw_training/dvfs/
outputs/unigradicon_raw_training/dvfs_canonical_ras/
outputs/unigradicon_raw_training/unigradicon_summary.json

outputs/unigradicon_raw_validation/dvfs/
outputs/unigradicon_raw_validation/dvfs_canonical_ras/
outputs/unigradicon_raw_validation/submission_unigradicon_raw.zip
outputs/unigradicon_raw_validation/eval/
outputs/unigradicon_raw_validation/unigradicon_summary.json
```

Safe to delete:

```bash
cd "/home/oussama/Desktop/MICCAI FRANCE"

rm -rf outputs/unigradicon_raw_training/warped
rm -rf outputs/unigradicon_raw_training/transforms

rm -rf outputs/unigradicon_raw_validation/warped
rm -rf outputs/unigradicon_raw_validation/transforms
```

Optional delete if space is critical:

```bash
rm -rf Learn2Breath_preprocessed
```

Current refiner uses raw CTs on the fly, so preprocessed data can be regenerated later if needed.

Verify required files:

```bash
cd "/home/oussama/Desktop/MICCAI FRANCE"

echo "train canonical:"
find outputs/unigradicon_raw_training/dvfs_canonical_ras \
  -name "NLST_*_DVF_RAS_XYZC_voxel.nii.gz" | wc -l

echo "validation challenge:"
find outputs/unigradicon_raw_validation/dvfs \
  -name "NLST_*_DVF.nii.gz" | wc -l

echo "validation canonical:"
find outputs/unigradicon_raw_validation/dvfs_canonical_ras \
  -name "NLST_*_DVF_RAS_XYZC_voxel.nii.gz" | wc -l
```

Expected:

```text
train canonical:      200
validation challenge: 10
validation canonical: 10
```

---

## 🧠 11. What we learned

### ✅ Raw uniGradICON is excellent

It gives:

```text
Dice:    0.96467
folding: 0.03309%
```

This validates the foundation-registration backbone idea.

---

### ❌ CT-only residual refinement is risky

The current refiner:

```text
improves image loss
but decreases lobe Dice
and increases folding
```

So it is optimizing the wrong target for this challenge.

---

### ⚠️ Training labels are unavailable

The 200 training cases have no lobe/fissure labels.

Therefore, supervised lobe/fissure loss cannot be used for normal training unless we generate pseudo-labels or use external segmentation.

---

### 🧩 Validation labels are available

Validation labels are useful for:

```text
- evaluation
- debugging
- visual inspection
- candidate selection if challenge rules allow it
```

But using them directly for training/test-time optimization may be considered leakage depending on challenge rules.

---

## 🚧 12. Recommended next approach

The current best path is **not more of the same refiner**.

The next competitive approach should be:

```text
raw uniGradICON
→ uniGradICON variant sweep
→ topology/folding repair
→ candidate evaluation/selection
→ optional new anatomy-aware refinement
```

---

### 12.1 🧪 uniGradICON variant sweep

Current baseline:

```text
model: unigradicon
io_iterations: None
io_sim: lncc2
```

Variants to test:

| Variant | Purpose |
|---|---|
| `unigradicon + io_iterations=50 + lncc2` | official instance optimization |
| `unigradicon + io_iterations=50 + lncc` | alternate similarity |
| `unigradicon + io_iterations=50 + mind` | robust structural similarity |
| `unigradicon + intensity_conservation_loss` | CT mass/density-aware behavior |
| `multigradicon` | stronger/different pretrained model |

This is lower risk than neural refinement.

---

### 12.2 🧬 Topology repair

Goal:

```text
keep Dice high
reduce folding toward 0%
```

Safe candidates:

```text
global DVF scaling:
  u_final = alpha * u
  alpha ∈ [0.95, 1.00]

local folding repair:
  detect detJ <= 0
  smooth/dampen only around folding regions
  preserve field elsewhere
```

Why this matters:

```text
raw uniGradICON already has high Dice
main gap to top 1 is near-zero folding and small extra Dice
```

---

### 12.3 🧠 Candidate evaluator/selector

For each validation case, generate multiple candidate DVFs:

```text
raw uniGradICON
IO-refined uniGradICON
MIND variant
intensity-conservation variant
alpha-scaled variants
local-repaired variants
```

Then evaluate:

```text
lobe Dice
folding %
composite score
```

Select the best candidate per case and build:

```text
selected_submission.zip
```

> [!IMPORTANT]
> This is the most practical next step for leaderboard improvement.

---

## 🧭 13. Suggested next files to add

```text
run_registration_variants.py
repair_dvf_topology.py
compare_dvf_candidates.py
make_ensemble_submission.py
```

Suggested output structure:

```text
outputs/variant_benchmark/
  candidates/
    raw_unigradicon/
    unigradicon_io50_lncc2/
    unigradicon_io50_mind/
    unigradicon_alpha_098/
    unigradicon_repaired/
  comparison.csv
  selected_dvfs/
  selected_submission.zip
```

---

## 🟢 14. Current best submission

Use this for now:

```text
outputs/unigradicon_raw_validation/submission_unigradicon_raw.zip
```

Do **not** use:

```text
outputs/refiner_cached_100_validation_latest/submission_refined.zip
```

because it is worse.

---

## ✅ 15. Immediate next action

Recommended next implementation step:

```text
patch and run:
  1. uniGradICON variant runner
  2. DVF topology repair
  3. candidate comparison table
  4. selected final submission builder
```

Do not continue training the current refiner unless the loss is redesigned.

---

## 🧾 16. Final current status

```text
Dataset understood:               ✅
Preprocessing policy decided:      ✅
Dataloader validated:              ✅
Identity baseline:                 ✅
Raw uniGradICON validation:        ✅ strong
Raw uniGradICON training cache:    ✅ complete
Residual refiner v1:               ❌ worse than raw uniGradICON
Current best submission:           ✅ raw uniGradICON
Next direction:                    🚧 variant sweep + topology repair
```

---

<p align="center">
  <b>Current strategic position:</b><br>
  We already have a strong foundation baseline. The remaining challenge is not “make any registration work” — it is to squeeze the last 0.5–1.0 Dice points while eliminating folding.
</p>
