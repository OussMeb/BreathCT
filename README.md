# SGR-FM Project MICCAI 2026 Learn2Breath
## MICCAI 2026 Learn2Breath Task 2 — EXP→INSP Deformable Lung CT Registration

**Project:** SGR-FM (Segmentation-Guided Registration with Flow Matching)  
**Researcher:** Mebarki M. Oussama  
**Status:** Active development  
**Current local champion:** **C5b-LMIND**  
**Current local mean lobe Dice:** **0.976539817822**  
**Current local mean folding:** **1.06792425427e-05%**  
**Last updated:** 2026-07-09

---

## 1. Project objective

The goal is to register **EXP CT** (moving image) to **INSP CT** (fixed image) for the MICCAI 2026 Learn2Breath Task 2 deformable registration challenge.

The project evolved from a verified raw uniGradICON baseline into a sequence of increasingly anatomy-aware, surface-aware, consensus-aware, and image-structure-aware residual refinements.

The main design principle is:

> Keep every successful registration stage frozen, add only a small residual correction, compose transforms exactly in pull convention, enforce strict topology guards, and preserve an exact byte-level fallback to the previous champion.

The current target is to push the validation lobe Dice toward or above **0.98** while maintaining approximately zero folding.

---

## 2. Registration conventions and geometry

The project uses the following conventions throughout:

- **Direction:** EXP → INSP
- **Fixed image:** INSP CT
- **Moving image:** EXP CT
- **DVF layout internally:** `[X, Y, Z, 3]`
- **Displacement units:** voxels
- **Convention:** pull / backward sampling
- **Warp equation:** `warped_EXP[x] = EXP[x + DVF[x]]`
- **Internal working orientation:** canonical RAS
- **Final output:** restored to the original INSP grid and header
- **Transform composition:** exact pull composition, never naive displacement addition

The baseline geometry audit confirmed:

- correct EXP→INSP direction;
- correct XYZ component order;
- correct pull convention;
- correct original-grid restoration;
- correct Jacobian calculation path;
- correct canonical-RAS conversion.

---

## 3. Dataset and output assumptions

### Validation set

- 10 EXP/INSP CT pairs
- lobe segmentations available for local evaluation
- fissure segmentations available for local analysis
- nominal DVF grid shape: `224 × 192 × 224`
- spacing: approximately `1.5 mm` isotropic

### Lobe labels

| Label | Anatomy |
|---:|---|
| 8 | Left upper lobe |
| 16 | Left lower lobe |
| 32 | Right upper lobe |
| 64 | Right middle lobe |
| 128 | Right lower lobe |

### Submission format

The successful validation submission format is exactly 10 top-level NIfTI files:

```text
NLST_0001_DVF.nii.gz
NLST_0002_DVF.nii.gz
...
NLST_0010_DVF.nii.gz
```

No folder prefix is used inside the ZIP.

---

# 4. Development timeline and method summary

| Stage | Method | Mean lobe Dice | Mean folding | Decision |
|---|---|---:|---:|---|
| C0 | Raw uniGradICON baseline | 0.964670322 | 0.033095% | Baseline |
| C1 | Support-guided FM initializer | 0.964682812 | 0.000033106% | Kept |
| C2 | Segmentation-guided residual TTO | 0.971941508 | 0.000017087% | Kept |
| C2-LONG | Longer C2 TTO schedule | 0.972198141 | 0.000014951% | Kept |
| C3 | Learned lobe-guided residual refiner | 0.963666966 | — | Rejected |
| C4 | HAST hierarchical anatomical surface TTO | 0.976117844 | 0.000010679% | Kept / submitted |
| C4.1 | HAST-PCF paired-consensus fissure refinement | 0.976383730 | 0.000010679% | Local champion before C5b |
| C5a | FireANTs image-driven residual branch | — | — | Aborted after smoke |
| C5b | LMIND localized MIND residual refinement | 0.976539818 | 0.000010679% | Current local champion |

### Current progression

```text
C0          0.964670
C1          0.964683
C2          0.971942
C2-LONG     0.972198
C4          0.976118
C4.1        0.976384
C5b         0.976540   ← current local champion
```

The remaining local gap to `0.98` is:

```text
0.98 - 0.976539817822
= 0.003460182178
```

---

# 5. Detailed method descriptions

## 5.1 C0 — Raw uniGradICON baseline

### Purpose

Establish a reproducible, geometry-verified initializer using raw CT volumes only.

### Pipeline

```text
raw EXP + raw INSP
        ↓
uniGradICON
        ↓
canonical-RAS DVF
        ↓
original INSP grid restoration
```

### Result

- mean lobe Dice: **0.964670322**
- mean folding: **0.033094973%**

### Interpretation

The Dice was already strong, but topology was insufficient for the challenge objective. The baseline was therefore treated as a strong initializer rather than a final method.

---

## 5.2 Phase 1 — TotalSegmentator anatomical segmentation

### Purpose

Create anatomical guidance without relying on validation ground truth during optimization.

### Predicted classes

The five TotalSegmentator lung lobes were mapped to challenge labels:

```text
LUL → 8
LLL → 16
RUL → 32
RML → 64
RLL → 128
```

A lung/body support mask was also generated.

### Validation segmentation quality

| Metric | Score |
|---|---:|
| Mean five-lobe Dice | 0.971632 |
| Whole-lung Dice | 0.993097 |
| LUL | 0.985950 |
| LLL | 0.981340 |
| RUL | 0.973490 |
| RML | 0.934750 |
| RLL | 0.982640 |

The **RML** was identified as the weakest predicted lobe and became the main anatomical bottleneck.

Derived fissure/interface quality:

| Metric | Score |
|---|---:|
| Exact surface agreement | 0.499220 |
| F1 at 1.5 mm | 0.771710 |
| F1 at 3.0 mm | 0.868870 |
| F1 at 4.5 mm | 0.918220 |

This motivated robust narrow-band surface losses rather than exact hard fissure matching.

---

## 5.3 C1 — Support-guided FM initializer

### Purpose

Suppress non-anatomical background influence while keeping the same underlying registration backbone.

### Main idea

```text
raw EXP / INSP
      ↓
predicted support masks
      ↓
soft support-weighted CT
      ↓
same uniGradICON registration
```

### Main settings

- soft support masking
- outside intensity approximately `-1024 HU`
- fade distance approximately `7.5 mm`

### Result

- mean lobe Dice: **0.964682812**
- mean folding: **0.000033106%**

### Interpretation

Dice improved only marginally, but folding was reduced by roughly three orders of magnitude. C1 became the first topology-safe initializer.

---

## 5.4 C2 — Segmentation-guided residual TTO

### Purpose

Perform a small test-time residual optimization on top of frozen C1.

### Main idea

- zero-initialized residual SVF;
- two-stage coarse/fine optimization;
- exact pull composition;
- lobe, interface, support, LNCC, bending, magnitude, and Jacobian terms;
- candidate scale backtracking;
- exact C1 fallback.

### Key stages

```text
coarse:
reduce 2
control 8
60 iterations
lr 0.05

fine:
reduce 2
control 4
80 iterations
lr 0.02
```

### Result

- mean lobe Dice: **0.971941508**
- mean folding: **0.000017087%**
- improved: **10/10 cases**

### Interpretation

This was the first large gain after the baseline and established residual TTO as a strong strategy.

---

## 5.5 C2-LONG — Longer TTO schedule

### Purpose

Test whether more iterations of the same C2 mechanism could continue improving alignment.

### Change

Only the optimization duration was extended:

```text
coarse: 120 iterations
fine:   160 iterations
```

### Result

- mean lobe Dice: **0.972198141**
- gain over C2: **+0.000256633**
- mean folding: **0.000014951%**

### Interpretation

The branch improved, but the gain was small. This was the first clear sign of diminishing returns from extending the same mechanism.

C2-LONG was nevertheless retained as a useful secondary initializer for later consensus construction.

---

## 5.6 C3 — Learned lobe-guided residual refiner

### Purpose

Learn a residual correction from frozen training DVFs and pseudo-anatomical guidance.

### Main idea

- frozen FM DVFs;
- learned residual network;
- lobe-guided proxy losses;
- holdout monitoring.

### Outcome

- validation mean lobe Dice: approximately **0.963667**
- worse than C1/C2 branches
- only a small subset of difficult cases improved

### Interpretation

The training and holdout proxy losses improved, but the real validation target worsened. The likely issue was target mismatch between pseudo-label optimization and true evaluation anatomy.

**Decision:** rejected as primary route.

---

## 5.7 C4 — HAST: Hierarchical Anatomical Surface TTO

### Full name

**C4-HAST: Hierarchical Anatomical Surface Test-Time Optimization**

### Purpose

Introduce a genuinely new representation based on anatomical surfaces, signed distances, fissure interfaces, and hierarchical residual stages.

### Main components

- five per-lobe signed distance fields;
- explicit left and right fissure/interface definitions;
- robust narrow-band surface distances;
- structural CT gradient descriptor;
- three sequential residual SVF stages;
- exact composition after each stage;
- independent scale selection;
- topology backtracking;
- exact C2-LONG fallback.

### Interface definitions

```text
left oblique:
LUL ↔ LLL

right horizontal:
RUL ↔ RML

right oblique:
RLL ↔ (RUL | RML)
```

### Hierarchy

```text
Stage A:
coarse anatomical correction

Stage B:
lobe-boundary refinement

Stage C:
local fissure refinement
```

### Result

- mean lobe Dice: **0.976117844**
- gain over C2-LONG: **+0.003919704**
- improved: **9/10 cases**
- mean folding: **0.000010679%**

### Interpretation

C4 produced the largest gain after C2. This confirmed that introducing a new anatomical representation was more effective than simply extending optimization time.

---

## 5.8 C4.1 — HAST-PCF paired consensus refinement

### Full name

**C4.1-HAST-PCF: Paired Consensus and Confidence-weighted Fissure refinement**

### Purpose

Reduce pseudo-label dependence and focus especially on the difficult right fissure/RML region.

### Consensus sources

For each case:

```text
direct INSP lobe prediction
EXP lobes transported by C4
EXP lobes transported by C2-LONG
```

These sources were combined into confidence-aware consensus guidance.

### Main additions

- paired segmentation consensus;
- confidence-aware pseudo-label weighting;
- class-balanced SDF weighting;
- separate right-horizontal and right-oblique losses;
- localized Stage D residual;
- exact C4 fallback.

### Result

- mean lobe Dice: **0.976383730**
- gain over C4: **+0.000265886**
- improved: **10/10**
- worsened: **0/10**
- mean folding: unchanged

### Interpretation

The method worked consistently and validated the RML/right-fissure hypothesis, but the magnitude remained small.

---

## 5.9 C5a — FireANTs residual branch

### Purpose

Test whether an independent image-driven optimizer could recover residual correspondences missed by the segmentation-guided hierarchy.

### Pipeline

```text
frozen C4.1
    ↓
warp EXP by C4.1
    ↓
FireANTs residual registration
    ↓
residual conversion to project DVF convention
    ↓
exact composition with C4.1
    ↓
candidate scaling + guards
```

### Smoke cases

```text
0003
0006
0007
0009
```

### Outcome

All four smoke cases fell back to exact C4.1.

Observed pattern:

- some image metrics improved;
- predicted lobe alignment often worsened;
- no reliable candidate passed the selector.

### Interpretation

The branch demonstrated real residual image signal, but global masked CC optimization was not sufficiently aligned with the lobe objective.

**Decision:** abort C5a and move to a more structural, localized image descriptor.

---

## 5.10 C5b — LMIND localized MIND residual refinement

### Full name

**C5b-LMIND: Localized MIND Residual Refinement**

### Purpose

Retain independent image-driven information while restricting residual capacity to uncertain, structurally mismatched, or fissure-adjacent regions.

### Main innovations

1. **12-channel 3D MIND-SSC-style descriptor**
2. **paired-consensus uncertainty**
3. **MIND mismatch localization**
4. **fissure proximity weighting**
5. **soft residual gating**
6. **zero-initialized residual SVF**
7. **exact composition with C4.1**
8. **candidate backtracking**
9. **exact C4.1 fallback**

### Priority map

Conceptually:

```text
priority =
    consensus uncertainty
  + normalized MIND mismatch
  + fissure proximity
```

The residual is spatially gated:

```text
v_effective(x) = gate(x) * v(x)
```

with a small nonzero floor to avoid hard discontinuities.

### Optimization stage

```text
stage name: E_localized_MIND
reduce factor: 2
control factor: 3
iterations: 180
learning rate: 0.02
max residual: 0.75 full-resolution voxels
integration steps: 6
```

### Loss design

| Term | Relative role |
|---|---|
| MIND | primary |
| LNCC | weak auxiliary |
| structural gradient | weak auxiliary |
| confidence-aware SDF | anatomy stabilizer |
| support | weak |
| bending | regularization |
| magnitude | regularization |
| Jacobian | strong topology guard |

### Full validation result

- mean lobe Dice: **0.976539817822**
- gain over C4.1: **+0.000156087823**
- improved cases: **10/10**
- worsened cases: **0/10**
- mean folding: **1.06792425427e-05%**
- maximum folding: **0.000106792425427%**

### Interpretation

C5b is the current local champion. The gain is modest, but the result is highly consistent because **all 10 cases improved** and no case regressed.

---

# 6. Current champion: C5b per-case results

| Case | C4.1 Dice | C5b Dice | Δ Dice | C5b folding | C5b RML Dice |
|---|---:|---:|---:|---:|---:|
| 0001 | 0.982398 | 0.982510 | +0.000112 | 0.000000000% | 0.973504 |
| 0002 | 0.981991 | 0.982110 | +0.000119 | 0.000000000% | 0.969756 |
| 0003 | 0.984152 | 0.984273 | +0.000121 | 0.000000000% | 0.979295 |
| 0004 | 0.979617 | 0.979693 | +0.000076 | 0.000000000% | 0.966435 |
| 0005 | 0.981352 | 0.981424 | +0.000072 | 0.000000000% | 0.963750 |
| 0006 | 0.955196 | 0.955335 | +0.000139 | 0.000000000% | 0.874497 |
| 0007 | 0.963211 | 0.963716 | +0.000506 | 0.000000000% | 0.941030 |
| 0008 | 0.975163 | 0.975318 | +0.000155 | 0.000000000% | 0.952232 |
| 0009 | 0.983997 | 0.984112 | +0.000115 | 0.000106792% | 0.973904 |
| 0010 | 0.976762 | 0.976907 | +0.000145 | 0.000000000% | 0.951894 |

### Paired comparison summary

| Statistic | Value |
|---|---:|
| C4.1 mean Dice | 0.976383729999 |
| C5b mean Dice | 0.976539817822 |
| Mean Dice gain | +0.000156087823 |
| Improved cases | 10 |
| Worsened cases | 0 |
| Minimum paired gain | +0.000072466381 |
| Maximum paired gain | +0.000505625719 |
| Mean folding change | +0% |

---

# 7. Current C5b lobe-wise performance

| Lobe | Label | Mean Dice |
|---|---:|---:|
| Left upper lobe | 8 | 0.985919 |
| Left lower lobe | 16 | 0.980501 |
| Right upper lobe | 32 | 0.981134 |
| Right middle lobe | 64 | 0.954630 |
| Right lower lobe | 128 | 0.980515 |

The **RML remains the weakest class**, especially in `NLST_0006`.

Current `NLST_0006` values:

```text
mean lobe Dice = 0.955335
RML Dice       = 0.874497
```

This remains the largest unresolved anatomical failure.

---

# 8. Official validation-server milestones

## First topology-safe submission

Approximate official result:

| Submission | Rank | Combined | Lobe DSC | Folding |
|---|---:|---:|---:|---:|
| C1-derived submission | 20 | 0.96182 | 0.96182 | 0.0 |

## C4-HAST official submission

| Submission | Rank | Combined | Lobe DSC | Folding |
|---|---:|---:|---:|---:|
| C4-HAST | **7** | **0.97459** | **0.97459** | **0.0** |

The C4 local score was:

```text
0.976117844
```

The official validation score was:

```text
0.97459
```

This confirmed that the C4 direction transferred to the server, while also showing that local-to-server offsets should not be assumed constant.

C4.1 and C5b are currently summarized here as local results unless separately submitted.

---

# 9. Reproducibility and freezing

## Frozen DVF repositories

A freeze manifest was verified successfully:

```text
status: PASS
checked files: 220
failure count: 0
```

Repositories:

```text
training canonical RAS: 200 / 200
validation original grid: 10 / 10
validation canonical RAS: 10 / 10
```

Manifest:

```text
outputs/sgr_fm/frozen_assets/fm_dvf_freeze_manifest.json
```

## Important fingerprints

### C2-LONG

```text
fdc2c0a64374aee658b647b55ae673f50faf2d08aed79ea7e0d6dd3d17fadcd3
```

### C4

```text
79a39af64f3a4678983c56c7edf9b5796c8dd349dc078ef11b10e32d349a8c7c
```

### C4.1 configuration lock used by later branches

```text
0b7e49fee00fd2ef84db9abf3121b82d971ea7b9b01a5b3b28b5005b6cd636b2
```

The general rule is:

> Do not weaken or bypass a fingerprint mismatch. Resolve the environment or asset inconsistency first.

---

# 10. Code organization

The project is organized around one command dispatcher (`main.py`), per-method runners, reusable method utilities, configuration files, smoke tests, and isolated output directories.

```text
/home/oussama/Desktop/MICCAI FRANCE/
│
├── main.py
├── evaluate_validation.py
│
├── configs/
│   ├── sgr_fm_c4_hast.yaml
│   ├── sgr_fm_c4_1_hast_pcf.yaml
│   ├── sgr_fm_c5a_fireants.yaml
│   └── sgr_fm_c5b_lmind.yaml
│
├── utils/
│   ├── sgr_fm_c4_hast.py
│   ├── sgr_fm_c4_1_hast_pcf.py
│   ├── sgr_fm_c5_fireants.py
│   └── sgr_fm_c5b_lmind.py
│
├── run_sgr_fm_c4_hast_validation.py
├── run_sgr_fm_c4_1_hast_pcf_validation.py
├── run_sgr_fm_c5a_fireants_validation.py
├── run_sgr_fm_c5b_lmind_validation.py
│
├── smoke_sgr_fm_c4_hast_synthetic.py
├── smoke_sgr_fm_c4_1_hast_pcf_synthetic.py
├── smoke_sgr_fm_c5a_fireants_synthetic.py
├── smoke_sgr_fm_c5b_lmind_synthetic.py
│
├── freeze_and_package_c4_hast_submission.py
├── freeze_and_package_c4_1_hast_pcf_submission.py
│
├── Learn2Breath_train_val_data/
│   ├── training/
│   └── validation/
│
└── outputs/
    ├── unigradicon_raw_validation/
    │
    └── sgr_fm/
        ├── segmentation_phase1_totalsegmentator/
        │   ├── lobes/
        │   └── support/
        │
        ├── frozen_assets/
        │   ├── fm_dvf_freeze_manifest.json
        │   └── ...
        │
        ├── c2_long_tto/
        │   ├── dvfs/
        │   ├── dvfs_canonical_ras/
        │   ├── eval/
        │   └── ...
        │
        ├── c4_hast/
        │   ├── dvfs/
        │   ├── dvfs_canonical_ras/
        │   ├── case_manifests/
        │   ├── eval/
        │   └── ...
        │
        ├── c4_1_hast_pcf/
        │   ├── dvfs/
        │   ├── dvfs_canonical_ras/
        │   ├── case_manifests/
        │   ├── eval/
        │   └── ...
        │
        ├── c5a_fireants_residual/
        │   ├── dvfs/
        │   ├── dvfs_canonical_ras/
        │   ├── case_manifests/
        │   └── ...
        │
        └── c5b_lmind_residual/
            ├── dvfs/
            ├── dvfs_canonical_ras/
            ├── case_manifests/
            ├── eval/
            │   └── validation_metrics.json
            ├── comparison/
            ├── run_config_resolved.json
            └── c5b_lmind_summary.json
```

---

# 11. Responsibility of each code layer

## `main.py`

Central command dispatcher.

Known commands include:

```text
c4-hast
c4.1-hast-pcf
c5-fireants
c5-mind
```

It is responsible for routing CLI arguments to the correct experiment runner.

---

## `configs/`

Stores method-specific hyperparameters and path definitions.

Examples:

```text
configs/sgr_fm_c4_hast.yaml
configs/sgr_fm_c4_1_hast_pcf.yaml
configs/sgr_fm_c5b_lmind.yaml
```

Configuration files should remain immutable after a result is declared a champion. New experiments should use new configuration files.

---

## `utils/`

Contains reusable method logic:

- descriptor computation;
- SDF/interface construction;
- residual SVF parameterization;
- scaling-and-squaring;
- warping;
- exact pull composition;
- topology metrics;
- candidate selection;
- fingerprinting;
- fallback handling.

---

## `run_*_validation.py`

Method-specific orchestration:

- discover cases;
- load CTs and assets;
- verify fingerprints;
- run optimization;
- generate candidates;
- select accepted scale;
- write canonical and original-grid DVFs;
- save case manifests;
- optionally evaluate.

---

## `smoke_*_synthetic.py`

Synthetic safety tests for:

- warp direction;
- component order;
- descriptor behavior;
- exact composition;
- candidate rejection;
- topology guards;
- fallback byte identity;
- output writing.

These smoke tests should be run before any real validation experiment.

---

## `outputs/sgr_fm/<method>/case_manifests/`

Per-case audit logs.

Typical content:

- input paths;
- asset paths;
- baseline proxy metrics;
- priority maps;
- optimizer iterations;
- candidate scales;
- rejection reasons;
- selected candidate;
- topology;
- fallback status;
- elapsed time.

These manifests are the primary diagnostic artifacts for each experiment.

---

# 12. Typical commands

## C4-HAST

```bash
python main.py c4-hast   --config configs/sgr_fm_c4_hast.yaml
```

## C4.1-HAST-PCF

```bash
python main.py c4.1-hast-pcf   --config configs/sgr_fm_c4_1_hast_pcf.yaml
```

## C5a-FireANTs

```bash
python main.py c5-fireants   --config configs/sgr_fm_c5a_fireants.yaml
```

## C5b-LMIND dry run

```bash
python main.py c5-mind   --config configs/sgr_fm_c5b_lmind.yaml   --dry-run
```

## C5b-LMIND one case

```bash
python main.py c5-mind   --config configs/sgr_fm_c5b_lmind.yaml   --case-ids NLST_0006   --skip-evaluation
```

## C5b-LMIND full validation

```bash
python main.py c5-mind   --config configs/sgr_fm_c5b_lmind.yaml
```

---

# 13. Main experimental lessons so far

## Lesson 1 — topology can be improved without sacrificing Dice

C1 reduced folding from approximately:

```text
0.033095%
```

to:

```text
0.000033%
```

with nearly unchanged Dice.

---

## Lesson 2 — residual TTO is effective when the initializer is already strong

C2 produced a large gain over C1:

```text
+0.007259 Dice
```

---

## Lesson 3 — more iterations of the same mechanism give diminishing returns

C2-LONG over C2:

```text
+0.000257
```

---

## Lesson 4 — proxy-label learning can fail despite good proxy training curves

C3 improved training and holdout proxies but worsened true validation Dice.

---

## Lesson 5 — new anatomical representations create the largest jumps

C4 over C2-LONG:

```text
+0.003920
```

This came from SDFs, explicit interfaces, fissure bands, and hierarchical surface refinement.

---

## Lesson 6 — consensus is useful but incremental

C4.1 over C4:

```text
+0.000266
```

---

## Lesson 7 — global image-driven residual optimization is unsafe near the ceiling

C5a-FireANTs produced image improvements but repeatedly conflicted with lobe alignment and fell back on all smoke cases.

---

## Lesson 8 — localized structural image evidence is useful

C5b over C4.1:

```text
+0.000156
```

with:

```text
10/10 cases improved
0/10 worsened
mean folding unchanged
```

This validates MIND as an independent residual signal, provided it is spatially localized and strongly guarded.

---

# 14. Known limitations and cautions

## 14.1 RML remains the main bottleneck

The right middle lobe remains the weakest class overall.

The most difficult case is still:

```text
NLST_0006
```

with current RML Dice approximately:

```text
0.874497
```

---

## 14.2 Local validation is not the hidden final test

The visible validation set contains lobe annotations and is not equivalent to the hidden OOD test distribution.

Avoid assuming that every local gain transfers linearly.

---

## 14.3 Local-to-server offset is not constant

C4 local and official results differed by roughly `-0.0015`, but this should not be treated as a fixed correction.

---

## 14.4 Topology evaluators have shown frame-dependent discrepancies

Some internal canonical-stage Jacobian summaries and final challenge-style evaluator outputs have differed.

For publication-quality claims, topology should be audited independently in the final evaluation frame.

---

## 14.5 Reused smoke outputs

Some full runs reused already-computed smoke-case outputs when the configuration was identical.

This is acceptable for development efficiency, but strict reproducibility records should state when a case was reused instead of recomputed.

---

# 15. Current decision

The current local champion is:

```text
C5b-LMIND
mean lobe Dice = 0.976539817822
mean folding   = 1.06792425427e-05%
```

C5b should be frozen as the current local champion.

The next development step should not be a simple longer C5b schedule.

The strongest next direction is a new correspondence source, for example:

```text
C6-KPIR
Keypoint-guided inverse-consistent residual refinement
```

Conceptually:

```text
frozen C5b
    ↓
mutual structural keypoints
    ↓
forward/backward consistency
    ↓
sparse residual anchors
    ↓
small smooth residual SVF
    ↓
weak MIND + confidence SDF + topology
    ↓
exact C5b fallback
```

The goal is to introduce explicit sparse correspondences rather than further tuning the same segmentation/MIND family.

---

# 16. Final status snapshot

| Item | Status |
|---|---|
| Raw baseline reproduced | PASS |
| Warp direction audited | PASS |
| XYZ component order audited | PASS |
| Pull convention audited | PASS |
| Canonical-RAS conversion audited | PASS |
| Frozen FM DVF repositories | PASS |
| Phase 1 lobe segmentation | PASS |
| C1 support guidance | PASS |
| C2 TTO | PASS |
| C2-LONG | PASS |
| C3 learned refiner | REJECTED |
| C4-HAST | PASS |
| C4 official submission | Rank 7, 0.97459, 0 folding |
| C4.1-HAST-PCF | PASS |
| C5a-FireANTs | ABORTED |
| C5b-LMIND | **PASS / CURRENT LOCAL CHAMPION** |

---

## End of README

This file is intended to serve as the project-level technical summary, experiment log, result overview, and code-organization reference for the SGR-FM Learn2Breath Task 2 work.
