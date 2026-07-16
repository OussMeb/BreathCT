# SGR-FM Project — MICCAI 2026 Learn2Breath Task 2

## EXP→INSP deformable lung CT registration

**Project:** SGR-FM — Segmentation-Guided Registration with Flow Matching  
**Researcher:** Mebarki M. Oussama  
**Status:** Active development  
**Current hybrid framework:** **C10 label-aware registration with explicit GT/pseudo routing**  
**Current best validation-GT-free local result:** **C8B — 0.979183639347 mean lobe Dice**  
**C10 forced-pseudo validation result:** **0.979129733326 mean lobe Dice**  
**C10 GT-available validation result:** **0.984021391501 mean lobe Dice**  
**Current GT-oracle diagnostic upper bound:** **C8A — 0.984083315736 mean lobe Dice**  
**Last updated:** 2026-07-16

---

# 1. Executive summary

The project began from a fully audited raw uniGradICON EXP→INSP baseline and evolved through support guidance, segmentation-guided residual test-time optimization, hierarchical lobe/fissure surface refinement, localized MIND refinement, lineage retuning, RML-aware refinement, relaxed-topology semantic registration, multi-initializer selection, a validation-GT oracle diagnostic, GT-gap-calibrated pseudo-label registration, and finally an explicit hybrid label-aware deployment framework.

The current best non-oracle local result is:

```text
C8B calibrated pseudo-label registration
mean lobe Dice = 0.979183639347
mean folding   = 0.088432671647%
max folding    = 0.241414956920%
gap to 0.98    = 0.000816360653
```

The central finding is now experimentally supported:

> The registration machinery is strong, but the main remaining bottleneck is the accuracy of the automatically predicted lobe targets, especially the EXP right-middle-lobe and right-fissure geometry.

The C8A validation-GT oracle increased the mean score from C7B's `0.978546710807` to `0.984083315736`, a gain of `+0.005536604930`. This demonstrates that better anatomical targets can unlock a large improvement without replacing the entire registration framework.

C8B then transferred part of this oracle information into a pseudo-label-only per-case optimizer by applying global lobe calibration and conservative fallback rules. C8B improved C7B by `+0.000636928540`, recovering approximately `11.5%` of the measured C8A oracle gap.

C8B is therefore the current practical baseline, but it remains **validation-calibrated** because its global lobe weights were derived from the C8A validation-GT gap audit. It does not use case-specific GT labels during optimization, but it should not be described as completely independent of validation-label development.

C10 now makes the anatomical-source decision explicit:

```text
paired GT labels exist and policy permits their use
→ ground-truth-guided branch

otherwise
→ automatic-segmentation/pseudo-label branch
```

The recovered C10 evidence contains results for both intended operating conditions; the forced-pseudo mode is explicitly verified, while the main GT-branch attribution still requires its unrecovered case manifests:

```text
C10 validation, GT available     = 0.984021391501 true mean lobe Dice
C10 validation, forced pseudo    = 0.979129733326 true mean lobe Dice
label-source penalty             = 0.004891658175

C10 unlabeled training pilot     = 10 cases, no true lobe Dice
mean pseudo transport Dice       = 0.933362709933
mean folding                     = 0.227539685308%
max folding                      = 0.406932182053%
```

C10 therefore supports the hybrid architecture at the output level, pending a runner/config audit. It does not change the competition-safety rule: the GT branch is usable only if labels are genuinely exposed to the submitted method. If the hidden evaluator exposes CT pairs only, the pseudo branch is mandatory.

The ten-case unlabeled training run is a deployment pilot, not a completed 100- or 200-case robustness study and not evidence of true anatomical Dice. It nevertheless exposes a larger pseudo-consistency and topology gap than the ten validation cases, so broad stress testing is now a priority.

---

# 2. Task definition

The objective is to register the expiration CT to the inspiration CT:

```text
moving image = EXP CT
fixed image  = INSP CT
direction    = EXP → INSP
```

The submitted displacement field follows the pull/backward convention:

```text
warped_EXP[x] = EXP[x + DVF[x]]
```

The DVF is dense: every sampled EXP voxel is transformed, including lung parenchyma, vessels, bronchi, chest wall, mediastinum, and structures outside the lobes. This must not be confused with dense anatomical supervision. A lobe-label loss directly constrains lobe regions and fissure boundaries, but it does not by itself establish correct vessel, airway, lesion, or whole-thorax correspondence.

## Dataset

### Training set

- 200 complete EXP/INSP pairs.
- No distributed training lobe ground truth.
- Useful for robustness testing, pseudo-label generation, teacher-student training, and failure analysis.
- It cannot provide true training-set lobe Dice unless additional labels are created or obtained.

### Validation set

- 10 EXP/INSP pairs.
- Ground-truth lobe and fissure labels are available locally.
- These labels are used by `evaluate_validation.py`.
- Until C8A, they were not used directly inside the registration optimizer.
- C8A intentionally used them as an oracle diagnostic.
- C8B uses no case-specific GT labels in optimization, but uses global calibration statistics derived from C8A.
- C10 can resolve paired validation labels to the GT branch when the selected policy permits it.
- A separate C10 forced-pseudo run is required to test label-free behavior on the same cases.

### Hidden test set and method-input uncertainty

- The project expects 100 paired COPDGene INSP/EXP scans in the hidden test.
- The evaluator must possess hidden labels to score Dice, but this does not imply that submitted methods can read those labels.
- Until the organizers explicitly confirm the algorithm input contract, the deployable submission must be assumed to receive CT pairs only.
- C10 must record the selected `label_mode` per case; it must never silently substitute or mix GT and pseudo anatomy.
- The GT-available branch is a conditional capability, not a justification for packaging hidden or validation labels with a submission.

### Nominal image geometry

- Grid: approximately `224 × 192 × 224`.
- Spacing: approximately `1.5 mm` isotropic.
- Training is mostly RAS.
- Validation CTs are LPS in the original files.
- Internal earlier branches use canonical RAS and restore the result to the original fixed INSP grid.

## Lobe labels

| Label | Anatomy |
|---:|---|
| 8 | Left upper lobe |
| 16 | Left lower lobe |
| 32 | Right upper lobe |
| 64 | Right middle lobe |
| 128 | Right lower lobe |

## Submission archive

A validation submission contains exactly ten top-level NIfTI DVFs:

```text
NLST_0001_DVF.nii.gz
...
NLST_0010_DVF.nii.gz
```

No directory prefix should exist inside the ZIP.

---

# 3. Registration conventions and geometry audit

The baseline audit independently confirmed:

- correct EXP→INSP direction;
- correct fixed/moving order;
- correct XYZ vector-component order;
- correct pull convention;
- correct voxel-displacement interpretation;
- correct challenge-grid restoration;
- correct Jacobian evaluation path;
- correct canonical-RAS conversion for the audited baseline pipeline.

The raw uniGradICON baseline was reproduced at:

```text
mean lobe Dice = 0.964670322194
mean folding   = 0.033094972640%
```

## Important composition caveat

The statement that every branch uses exact pull composition is no longer true for the full project history.

- Earlier C2–C6 branches were designed around exact pull composition or SVF-based residual composition.
- C7A, C7A-v2, C7B, C8A, and C8B use additive residual updates, linear blends, or extrapolation in voxel-displacement space.
- These later branches produced measurable gains, but additive displacement updates are a known mathematical limitation and may be less reliable for large residuals.
- Their final `dvfs/` outputs were evaluated successfully on the challenge grid and are the authoritative outputs.
- The `dvfs_canonical_ras/` folders produced by the recent lightweight C7B/C8A/C8B patches should not be assumed to be independently verified canonical conversions. They mirror the exported field and must be re-audited before reuse as canonical assets.

## Optimization-supervision caveat

C8A starts from C7B and optimizes a residual using GT lobe Dice, residual smoothness, and residual-magnitude regularization. Its residual stages do not directly optimize CT intensity, MIND, vessel, airway, landmark, or whole-thorax correspondence.

Consequently:

- the final field warps the complete image;
- the lobe geometry is directly supervised;
- internal lung structures inherit correspondence mainly from the parent field and the smoothness assumptions;
- high lobe Dice alone does not prove equally high internal-anatomy accuracy.

The C10 metrics include NCC, HU MAE, and difference-image QC, but recorded evaluation metrics are not automatically optimization losses. The C10 code/config package must be audited before claiming that these terms are active in the objective.

---

# 4. Current result timeline

| Stage | Method | Mean lobe Dice | Mean folding | Max folding | Decision |
|---|---|---:|---:|---:|---|
| C0 | Raw uniGradICON | 0.964670322 | 0.033094973% | — | Audited baseline |
| C1 | Support-guided FM | 0.964682812 | 0.000033106% | — | Topology-safe initializer |
| C2 | Segmentation-guided residual TTO | 0.971941508 | 0.000017087% | — | Kept |
| C2-LONG | Longer C2 schedule | 0.972198141 | 0.000014951% | — | Kept |
| C3 | Learned pseudo-label refiner | 0.963666966 | — | — | Rejected |
| C4 | HAST hierarchical surface TTO | 0.976117844 | 0.000010679% | — | Kept / submitted |
| C4.1 | HAST-PCF consensus fissure refinement | 0.976383730 | 0.000010679% | — | Kept |
| C5a | FireANTs residual | — | — | — | Aborted after fallback smoke |
| C5b | LMIND localized MIND residual | 0.976539818 | 0.000010679% | 0.000106792% | Kept |
| B0 lineage | C2R-B0 propagated through C4→C4.1→C5b | 0.976660368 | 0.000005340% | 0.000053396% | Champion before C6 |
| C6 | RML-aware refinement | 0.976803528 | 0.000005340% | 0.000053396% | Kept |
| C7A | Semantic-Adam relaxed branch | 0.977190675 | 0.028185725% | 0.055030137% | Positive but unstable |
| C7A-v2 | Relaxed selector branch | 0.978496117 | 0.068465692% | 0.181344218% | Major gain |
| C7B | Multi-init blend/extrapolation selector | 0.978546711 | 0.083025771% | 0.211117946% | Small gain |
| C8A | **Validation-GT oracle diagnostic** | **0.984083316** | **0.123560972%** | **0.255693104%** | Diagnostic only |
| C8B | **GT-gap-calibrated pseudo-label TTO** | **0.979183639** | **0.088432672%** | **0.241414957%** | **Current practical champion** |
| C10-GT | Hybrid C10, GT-available validation branch | 0.984021392 | 0.119675617% | 0.254271631% | Conditional GT branch; not CT-only safe |
| C10-P | Hybrid C10, forced-pseudo validation branch | 0.979129733 | 0.087845990% | 0.238255092% | PASS; slightly below C8B |
| C10-TRAIN-10 | Hybrid C10, unlabeled training pilot | GT unavailable; pseudo proxy 0.933362710 | 0.227539685% | 0.406932182% | 10-case robustness pilot only |

## Current progression

```text
C0       0.964670
C2       0.971942
C4       0.976118
C5b      0.976540
B0       0.976660
C6       0.976804
C7A-v2   0.978496
C7B      0.978547
C8B      0.979184   ← current practical champion
C10-P    0.979130   ← forced-pseudo C10 regression run
C10-GT   0.984021   ← conditional GT-available C10 branch
C8A      0.984083   ← GT-oracle diagnostic, not hidden-test-safe
```

---

# 5. Phase 1 segmentation baseline

TotalSegmentator was used to generate five-lobe pseudo labels and lung/body support masks.

## Validation segmentation quality

| Metric | Score |
|---|---:|
| Mean five-lobe Dice | 0.971632 |
| Whole-lung Dice | 0.993097 |
| LUL | 0.985950 |
| LLL | 0.981340 |
| RUL | 0.973490 |
| RML | 0.934750 |
| RLL | 0.982640 |

Derived interface/fissure agreement:

| Metric | Score |
|---|---:|
| Exact surface agreement | 0.499220 |
| F1 at 1.5 mm | 0.771710 |
| F1 at 3.0 mm | 0.868870 |
| F1 at 4.5 mm | 0.918220 |

The RML and the right horizontal/oblique fissures were identified early as the weakest pseudo-anatomical structures. Later C8A results confirmed this diagnosis.

---

# 6. Historical methods C0–C5b

## C0 — raw uniGradICON

Purpose:

- create a reproducible raw-CT initializer;
- verify the full geometry chain;
- establish challenge-compatible DVF output.

Result:

```text
Dice    = 0.964670322
folding = 0.033094973%
```

## C1 — support-guided FM

Soft lung/body support weighting reduced non-anatomical background influence.

Result:

```text
Dice    = 0.964682812
folding = 0.000033106%
```

The main contribution was topology stabilization rather than Dice.

## C2 and C2-LONG — segmentation-guided residual TTO

A small residual field was optimized using pseudo-lobe, interface, support, CT-similarity, smoothness, magnitude, and topology terms.

Results:

```text
C2      = 0.971941508
C2-LONG = 0.972198141
```

C2 produced the first large post-baseline gain. C2-LONG showed diminishing returns from simply extending the same optimizer.

## C3 — learned residual refiner

C3 trained a learned refiner from pseudo targets and frozen initializer DVFs.

The training/holdout proxy improved, but true validation Dice fell to approximately:

```text
0.963667
```

This was the first strong evidence that better pseudo-objective performance does not guarantee better ground-truth lobe alignment.

Decision: rejected.

## C4 — HAST

C4 introduced:

- per-lobe signed-distance fields;
- explicit left and right fissure interfaces;
- narrow-band surface terms;
- hierarchical coarse/boundary/fissure stages;
- strict fallback to C2-LONG.

Result:

```text
0.976117844
```

This was the largest gain after C2 and showed that new anatomical representations were more useful than longer optimization schedules.

## C4.1 — HAST-PCF

C4.1 combined:

- direct INSP pseudo lobes;
- EXP pseudo lobes transported by C4;
- EXP pseudo lobes transported by C2-LONG;
- paired consensus;
- confidence-aware fissure refinement.

Result:

```text
0.976383730
```

All ten cases improved over C4, but the gain remained incremental.

## C5a — FireANTs residual

FireANTs was tested as an independent image-driven residual optimizer. All smoke cases fell back to C4.1 because improved image similarity did not reliably improve lobe alignment.

Decision: aborted.

## C5b — LMIND

C5b localized a MIND-style structural residual using:

- 12-channel 3D MIND descriptor;
- consensus uncertainty;
- MIND mismatch;
- fissure proximity;
- gated residual capacity;
- exact C4.1 fallback.

Result:

```text
0.976539818
```

All ten cases improved, but the mean gain was small.

---

# 7. Lineage retuning and C6

## C2R-LRT lineage sweep

A C2-LONG retuning sweep evaluated B0 and V1–V7 variants. V3 had the best immediate C2R score, but downstream propagation compressed the differences.

Strict propagation was:

```text
C2R(X)
→ C4(X)
→ C4.1(primary=C4(X), secondary=C2R(X))
→ C5b(primary=C4.1(X), secondary=C2R(X))
```

Final propagated results:

| Lineage | Final mean Dice |
|---|---:|
| B0 | 0.976660368386 |
| V3 | 0.976660273171 |
| V7 | 0.976539548868 |

B0 was selected because it was marginally best globally and had lower folding.

## C6-RMLA

C6 applied targeted RML/right-fissure refinement to the B0 champion.

Result:

```text
mean Dice       = 0.976803528478
mean RML Dice   = 0.955481732691
mean folding    = 0.000005339621%
max folding     = 0.000053396213%
```

Comparison against B0:

```text
mean gain            = +0.000143160092
RML mean gain        = +0.000565130345
RML improved cases   = 10/10
global improved      = 8/10
global worsened      = 2/10
```

C6 confirmed the importance of the RML but also showed that an RML-only strategy could not close the full leaderboard gap.

Even perfect RML correction from C6 would not reach a mean score of 0.988 because errors remain in other lobes and surfaces.

---

# 8. C7 relaxed-topology semantic branches

The project priority changed at C7:

> Prioritize Dice while keeping folding within a controlled competition-acceptable range, rather than forcing folding toward zero at all costs.

The working policy became:

```text
soft folding zone = 0.25% to 0.30%
hard rejection    = 0.50%
```

These thresholds are development choices, not official challenge guarantees.

## C7A — semantic-Adam branch

C7A introduced a label-first semantic residual branch initialized from C6.

Result:

```text
mean Dice     = 0.977190675446
mean folding  = 0.028185724843%
max folding   = 0.055030136822%
```

C7A improved six cases and worsened four. It demonstrated that a more permissive deformation regime could produce larger gains, especially on hard cases, but easy cases could be over-warped.

## C7A-v2 — relaxed selector

C7A-v2 added a non-GT case selector and exact parent fallback.

Result:

```text
mean Dice     = 0.978496116512
mean folding  = 0.068465691865%
max folding   = 0.181344217617%
```

Compared with C6:

- six cases improved;
- four cases were exact-parent fallback/equal;
- no case worsened;
- mean RML increased to approximately `0.959293`.

This was the largest practical improvement after C4.

## C7B — multi-init selector

C7B evaluated:

- C6 parent;
- C7A-v2 parent;
- linear blends;
- extrapolations beyond C7A-v2.

Result:

```text
mean Dice     = 0.978546710807
mean folding  = 0.083025771148%
max folding   = 0.211117945826%
```

Selection histogram:

```text
C6 parent       4 cases
C7A-v2 parent   4 cases
extrapolation   2 cases
```

The main extra gain came from `NLST_0006`.

### Important C7B implementation finding

The C7B case manifests show that its segmentation proxy paths resolved to files under:

```text
segmentation_phase1_totalsegmentator/interfaces/validation/
```

rather than the full five-lobe pseudo-label files.

Therefore, C7B should be interpreted as an interface-proxy candidate selector, not a clean full-lobe pseudo-label optimizer. This path-resolution behavior likely limited the magnitude of the C7B gain.

C8B explicitly excludes interface, fissure, support, and body files when locating pseudo-lobe inputs.

---

# 9. C8A — validation-GT oracle diagnostic

## Purpose

C8A deliberately used the distributed validation ground-truth lobe labels inside the optimizer to answer:

```text
If the anatomical target were correct, how high could the current registration machinery go?
```

C8A is not a hidden-test-safe method and should not be used to claim generalizable performance.

## Result

```text
parent C7B mean Dice = 0.978546710807
C8A oracle mean Dice = 0.984083315736
oracle gain           = +0.005536604930
mean folding          = 0.123560972067%
max folding           = 0.255693104200%
minimum case Dice     = 0.974084135459
maximum case Dice     = 0.988465721709
```

## Per-lobe oracle gap

| Label | Lobe | C7B | C8B | C8B−C7B | C8A oracle | C8A−C7B |
|---:|---|---:|---:|---:|---:|---:|
| 8 | LUL | 0.986750 | 0.987182 | +0.000433 | 0.988417 | +0.001668 |
| 16 | LLL | 0.981808 | 0.982455 | +0.000647 | 0.985135 | +0.003326 |
| 32 | RUL | 0.982372 | 0.982759 | +0.000388 | 0.986987 | +0.004615 |
| 64 | RML | 0.959500 | 0.960607 | +0.001107 | 0.973988 | +0.014488 |
| 128 | RLL | 0.982304 | 0.982914 | +0.000610 | 0.985890 | +0.003586 |

The RML oracle gain is approximately `+0.014488`, far larger than the gain for any other lobe.

## Hard-case findings

### NLST_0006

```text
C7B mean Dice    = 0.966176454
C8A mean Dice    = 0.974084135
gain             = +0.007907681
C7B RML          = 0.911278963
C8A RML          = 0.936691721
RML gain         = +0.025412758
```

The C8A manifest showed:

```text
pseudo INSP RML Dice vs GT ≈ 0.929878
pseudo EXP  RML Dice vs GT ≈ 0.855577
```

This confirms a major EXP RML pseudo-label error.

### NLST_0007

```text
C7B mean Dice    = 0.969538650
C8A mean Dice    = 0.977952285
gain             = +0.008413635
C7B RML          = 0.953268908
C8A RML          = 0.967591175
RML gain         = +0.014322267
```

## Folding-versus-Dice tradeoff

C8A also showed that some stronger residual candidates had better GT Dice but exceeded the chosen folding limit.

Examples:

- `NLST_0006` O2 scale `1.0`: mean Dice approximately `0.98247`, folding approximately `0.85779%`, rejected.
- `NLST_0007` O2 scale `1.0`: mean Dice approximately `0.97947`, folding approximately `0.55410%`, rejected.
- `NLST_0007` O2 scale `0.75`: mean Dice approximately `0.97929`, folding approximately `0.33281%`, accepted by the hard threshold but penalized by the soft folding score, so a lower-folding candidate was selected.

Therefore, the remaining gap is not purely segmentation. It also includes optimizer parameterization, residual composition, and the chosen topology tradeoff.

---

# 10. C8B — calibrated pseudo-label registration

## Purpose

C8B attempted to transfer the C8A oracle findings into a method that does not use validation GT labels during individual-case optimization.

It uses:

- C7B as parent;
- full pseudo INSP and EXP lobe labels;
- default RML-aware class weighting;
- global per-lobe calibration derived from the C8A gap audit;
- weighted pseudo-label Dice;
- smoothness and magnitude penalties;
- staged residual optimization;
- non-RML regression guards;
- minimum pseudo-gain requirements;
- exact-parent fallback.

## Generalization status

C8B is more deployable than C8A because it does not require hidden-test GT labels.

However:

- its global calibration weights were derived from the validation-GT C8A audit;
- its hyperparameters were selected through repeated validation experiments;
- it is therefore validation-calibrated, not an unbiased estimate of hidden-test performance.

## Result

```text
mean lobe Dice = 0.979183639347
mean folding   = 0.088432671647%
max folding    = 0.241414956920%
minimum case   = 0.968161501088
maximum case   = 0.985245954425
```

C8B internal pseudo-objective result:

```text
parent pseudo Dice = 0.988758054764
final pseudo Dice  = 0.989540140375
pseudo gain        = +0.000782085611
```

Execution modes:

```text
calibrated pseudo TTO       = 9 cases
exact-parent fallback       = 1 case
failures                    = 0
```

## C8B versus C7B and C8A

| Case | C7B | C8B | C8B−C7B | C8A oracle | C8A−C7B | C8B RML | C8A RML | C8B folding |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0001 | 0.982537 | 0.983379 | +0.000842 | 0.986278 | +0.003740 | 0.975442 | 0.983102 | 0.000577% |
| 0002 | 0.982143 | 0.982834 | +0.000690 | 0.985916 | +0.003773 | 0.970828 | 0.978760 | 0.000064% |
| 0003 | 0.984379 | 0.985246 | +0.000867 | 0.988466 | +0.004087 | 0.981128 | 0.985579 | 0.000000% |
| 0004 | 0.981596 | 0.982143 | +0.000547 | 0.985644 | +0.004048 | 0.970067 | 0.978974 | 0.212335% |
| 0005 | 0.981557 | 0.982013 | +0.000456 | 0.987100 | +0.005543 | 0.964484 | 0.979360 | 0.000000% |
| 0006 | 0.966176 | 0.968162 | +0.001985 | 0.974084 | +0.007908 | 0.917270 | 0.936692 | 0.241415% |
| 0007 | 0.969539 | 0.970057 | +0.000518 | 0.977952 | +0.008414 | 0.953547 | 0.967591 | 0.192728% |
| 0008 | 0.975865 | 0.975865 | +0.000000 | 0.983302 | +0.007437 | 0.949485 | 0.974266 | 0.139215% |
| 0009 | 0.984407 | 0.984859 | +0.000453 | 0.987449 | +0.003042 | 0.974686 | 0.981463 | 0.035701% |
| 0010 | 0.977268 | 0.977279 | +0.000011 | 0.984642 | +0.007374 | 0.949130 | 0.974093 | 0.062292% |

C8B improved nine cases and preserved one case exactly (`NLST_0008`) relative to C7B.

The largest C8B gain was:

```text
NLST_0006: +0.001985047
```

But C8A still remained substantially higher on the same case, demonstrating that weighted pseudo labels do not fully correct pseudo-label geometry.

## Oracle-gap recovery

```text
C7B → C8A oracle gap = 0.005536604930
C7B → C8B gain       = 0.000636928540
recovered fraction   = 11.50%
```

C8B recovered only about one ninth of the measured oracle gap. This is the central reason the next step should correct pseudo-label geometry rather than only reweight the same labels.

---

# 11. C10 — hybrid label-aware registration

## 11.1 Purpose and routing policy

C10 turns the GT/pseudo distinction into one explicit registration interface. The registration backend can remain common while the anatomical target source changes according to data availability and policy.

The required case-level behavior is:

```python
if (
    policy_permits_gt
    and fixed_gt_lobes_exist
    and moving_gt_lobes_exist
):
    label_mode = "ground_truth"
    fixed_labels = fixed_gt_lobes
    moving_labels = moving_gt_lobes
else:
    label_mode = "pseudo"
    fixed_labels = segment(fixed_ct)
    moving_labels = segment(moving_ct)
```

Safety requirements:

- both fixed and moving GT labels must exist before entering the GT branch;
- the selected mode must be written to each case manifest and summary;
- partial or silent GT/pseudo mixing is forbidden;
- a forced-pseudo policy must remain available for CT-only regression testing even when local GT exists;
- missing labels must trigger the pseudo branch, not a crash and not an implicit use of evaluator-only labels.

This is scientifically reasonable for deployment: label-rich cases can use the best available anatomy, while label-free cases remain fully automatic.

## 11.2 Verified C10 output metrics

The recovered C10 metrics and QC support three runs:

| Run | Cases | Anatomical source | True mean lobe Dice | Pseudo mean lobe Dice | Mean folding | Max folding | Mean NCC after |
|---|---:|---|---:|---:|---:|---:|---:|
| C10 hybrid validation | 10 | GT available; GT QC overlay; mode manifest pending | 0.984021392 | 0.984591940 | 0.119675617% | 0.254271631% | 0.729580819 |
| C10 forced-pseudo validation | 10 | pseudo, histogram `pseudo: 10` | 0.979129733 | 0.989530391 | 0.087845990% | 0.238255092% | 0.804103846 |
| C10 unlabeled training pilot | 10 | pseudo overlay | unavailable | 0.933362710 | 0.227539685% | 0.406932182% | 0.438406596 |

QC titles independently confirm `label overlay=GT` for the recovered validation hybrid examples and `label overlay=pseudo` for the recovered training examples. A QC overlay identifies what was visualized; by itself it does not prove which labels entered the optimizer. The forced-pseudo summary does prove `status: PASS`, ten cases, zero failures, and `label_mode_histogram: {"pseudo": 10}`.

The main C10 validation run is treated below as the intended GT branch because validation GT was available, the project policy selects it, and the result closely reproduces C8A. Its case-level mode histogram/manifests were not recoverable, so this attribution must be confirmed during the code-package audit.

Comparison with the preceding branches:

```text
C10 GT branch − C8A oracle       = -0.000061924235
C10 pseudo branch − C8B          = -0.000053906021
C10 GT branch − C10 pseudo       = +0.004891658175
```

C10 therefore reproduces the two established performance regimes to within approximately `6.2e-5`. It currently adds a coherent deployment interface, not a new label-free accuracy record. C8B remains the best local GT-free validation result by approximately `0.000053906`.

## 11.3 Measured label-source gap

Under the intended C10 mode attribution above, the GT-available run exceeds the forced-pseudo run on the same ten validation cases by:

```text
0.004891658175 mean lobe Dice
```

Per-lobe true validation Dice is:

| Lobe | C10 GT branch | C10 pseudo branch | GT−pseudo |
|---|---:|---:|---:|
| LUL | 0.988472086 | 0.987187481 | +0.001284606 |
| LLL | 0.985220737 | 0.982468807 | +0.002751930 |
| RUL | 0.986936355 | 0.982709578 | +0.004226777 |
| RML | 0.973576481 | 0.960410873 | +0.013165608 |
| RLL | 0.985901298 | 0.982871928 | +0.003029370 |

The RML remains the dominant source of the gap.

The proxy behavior is also informative:

- forced-pseudo C10 has much higher pseudo-label Dice than the GT branch (`0.989530` versus `0.984592`);
- forced-pseudo C10 also has higher post-warp NCC (`0.804104` versus `0.729581`) and lower post-warp HU MAE (`55.446` versus `61.903`);
- nevertheless, its true GT lobe Dice is lower by `0.004892`.

This is direct evidence that pseudo overlap and global CT similarity are useful but imperfect selection proxies. Neither should be treated as a substitute for anatomical ground truth.

## 11.4 Ten-case unlabeled training pilot

The recovered training metrics cover `NLST_0011` through `NLST_0020`. All ten cases appear in the PASS evaluation file, but no training GT lobes exist.

Aggregate proxy results:

```text
mean pseudo transport Dice  = 0.933362709933
minimum case pseudo Dice    = 0.885855074363
maximum case pseudo Dice    = 0.967220007034
mean folding                = 0.227539685308%
max folding                 = 0.406932182053%
mean NCC before             = 0.273752364615
mean NCC after              = 0.438406596285
mean NCC gain               = 0.164654231670
mean HU MAE before          = 142.527125549
mean HU MAE after           = 83.000247574
mean DVF p95 magnitude      = 14.346451759 vox
```

Per-lobe pseudo transport Dice:

| Lobe | Mean proxy Dice |
|---|---:|
| LUL | 0.976824297 |
| LLL | 0.976240544 |
| RUL | 0.948743264 |
| RML | 0.848905155 |
| RLL | 0.916100290 |

The training pilot is materially harder than the ten validation cases by the available proxies. In particular, the RML proxy falls below `0.85`, and the maximum folding approaches the development hard limit of `0.50%`.

This pilot can support claims about execution, proxy consistency, image similarity, deformation magnitude, and topology. It cannot establish true lobe Dice or predict that hidden-test Dice will exceed `0.98`.

## 11.5 Dense transformation versus anatomical supervision

C10 outputs a dense DVF, so the full EXP image is sampled under the transformation. However, this does not mean that every internal structure is directly supervised.

Five lobe masks strongly constrain:

- the outer lung/lobe shapes;
- fissure interfaces;
- regional volume transport.

They weakly constrain or do not directly identify:

- internal vessels and airway bifurcations;
- lesions and local parenchymal patterns;
- sliding motion near the pleura;
- mediastinal and chest-wall correspondence;
- structures outside the labeled lungs.

Two fields can therefore obtain almost identical lobe Dice while differing substantially in internal correspondence. C10 QC, NCC, and HU MAE help detect gross failures, but vessel/airway/landmark evaluation or a structural optimization term is required for stronger claims.

## 11.6 Proposed unified C10 objective

The final hybrid system should use the label source conditionally but keep a common multi-cue objective:

```text
L = λlabel   · Llabel
  + λimage   · LMIND/LNCC
  + λsurface · Lfissure/surface
  + λsmooth  · Lsmooth
  + λjac     · LJacobian/topology
  + λmag     · Lresidual-magnitude
```

Mode-specific behavior:

- with GT available, `Llabel` uses GT lobes;
- without GT, `Llabel` uses confidence-weighted pseudo lobes;
- CT/feature, surface, regularization, and topology terms remain active in both modes;
- uncertain pseudo boundaries, especially the EXP RML/right fissures, receive reduced or spatially calibrated label weight;
- the selector must evaluate label, structural, and topology criteria rather than maximizing pseudo Dice alone.

This objective is the next design target. The recovered metrics do not prove that all these terms are already implemented in C10.

## 11.7 Hidden-test policy scenarios

Three input contracts remain possible:

1. **Hidden labels are exposed to the algorithm.** C10 may use the GT branch if challenge rules explicitly permit it.
2. **Only CT pairs are exposed.** C10 must use the pseudo branch.
3. **Labels exist only inside the scorer.** This is operationally identical to CT-only input for the submitted method; C10 must use the pseudo branch.

Until written organizer clarification is obtained, development and packaging must assume scenarios 2 or 3. The forced-pseudo branch is therefore the submission-critical path.

## 11.8 C10 evidence-package limitation

The supplied `c10.zip` was truncated at `15,925,248` bytes, during a QC PNG, and lacked its central directory. Fourteen complete entries were recovered with size and CRC verification, including:

- validation and training JSON/CSV metrics;
- the forced-pseudo C10 summaries;
- three validation and three training QC images.

The runner, configuration, manifests, complete QC set, and code fingerprints were not present in the recoverable prefix. Therefore:

- the numerical results above are verified from recovered outputs;
- the exact implementation and command line are not yet independently audited;
- no C10 code hash should be frozen from this attachment;
- a small code/config/manifests-only archive must be supplied before the C10 implementation is declared reproducible.

---

# 12. Current scientific conclusions

## 12.1 The basic registration conventions are not the main problem

The baseline geometry was independently audited, and multiple branches consistently improved Dice. C8A showed that the same general deformation machinery can gain more than `0.0055` when given correct anatomical targets.

Therefore, the project is not primarily blocked by:

- wrong warp direction;
- wrong component order;
- wrong pull convention;
- wrong output grid;
- inability to produce useful deformation.

## 12.2 Automatic anatomical target quality is the dominant current bottleneck

Evidence:

- TotalSegmentator RML is the weakest pseudo lobe.
- C6 improved RML in all ten cases but only modestly improved the global score.
- C8A increased RML by approximately `0.01449` on average relative to C7B.
- `NLST_0006` EXP pseudo RML Dice against GT was only approximately `0.85558`.
- C8B class reweighting improved the score but recovered only approximately `11.5%` of the oracle gap.
- C10 measured a `0.004891658` mean-Dice penalty when the same validation cohort was switched from GT to pseudo anatomy.
- The C10 training pilot reduced the RML pseudo-transport proxy to approximately `0.84891`, indicating that the segmentation/registration problem is harder outside the validation cohort.

## 12.3 Registration/regularization remains a secondary bottleneck

C8A did not reach a mean of 0.988 even with GT labels.

The remaining limitations include:

- additive residual updates in recent branches;
- coarse control-grid parameterization;
- only two or three optimization stages;
- simple one-hot Dice rather than richer GT-SDF/surface terms in C8A/C8B;
- folding penalties and hard rejection;
- candidate-scale discretization;
- lack of exact residual composition in the recent lightweight branches.
- lack of a verified common label + CT/feature + surface objective in C10.

## 12.4 High validation scores do not prove hidden-test superiority

A visible validation method may use:

- stronger automatic segmentation;
- validation-label calibration;
- direct validation-GT optimization if permitted;
- more aggressive topology tradeoffs;
- case-specific selection.

It is not justified to claim that another participant is overfitting or will fail on the 100-case hidden test without their method and hidden-test results.

What can be stated is:

> Direct use of validation GT can substantially inflate visible-validation performance relative to a method that must infer anatomy automatically at test time.

## 12.5 C10 is a conditional pipeline, not a single model that “understands” both tasks

The current architecture is:

```text
paired labels available and permitted?
├── yes → GT anatomical targets
└── no  → pretrained lobe segmentation → pseudo-anatomical targets
→ pretrained registration initializer
→ common registration/refinement backend
→ case-specific semantic TTO
→ selector / fallback
→ DVF
```

It is not yet a jointly trained segmentation-registration model, and the current evidence does not establish that both branches optimize an identical multi-cue objective.

Testing on the unlabeled training pairs measures pipeline robustness, not true lobe Dice, because training GT lobes are unavailable. C10 has completed only a ten-case pilot so far.

## 12.6 Whole-image warping is not whole-image validation

The dense C10 DVF transforms every voxel, but the current primary anatomical metric observes only five lobes. Claims about vessels, bronchi, lesions, sliding surfaces, or structures outside the lungs require additional losses and evaluation targets.

---

# 13. Current limitations

## 13.1 Pseudo-label geometry

The largest limitation is not only label confidence but boundary placement.

C8B reweighted the pseudo labels but did not change their geometry. It therefore cannot correct a pseudo RML boundary that is systematically misplaced.

## 13.2 RML and right fissures

The RML remains the weakest lobe in C8B:

```text
mean C8B RML Dice = 0.960606699
```

Hard cases:

```text
NLST_0006 C8B RML = 0.917269527
NLST_0007 C8B RML = 0.953547064
NLST_0008 C8B RML = 0.949485379
NLST_0010 C8B RML = 0.949130010
```

## 13.3 Validation calibration and overfitting risk

C8B does not use validation GT during per-case optimization, but its global calibration depends on C8A validation-GT statistics.

Repeated tuning on only ten validation cases can overfit:

- class weights;
- selector thresholds;
- folding thresholds;
- candidate scales;
- stage schedules.

## 13.4 Additive displacement updates

C7A onward largely uses additive residual updates. Exact pull composition should be restored before large residuals or training-time teacher generation.

## 13.5 Canonical-output labeling

The recent `dvfs_canonical_ras/` outputs have not undergone the same independent canonical-frame audit as the original baseline and early branches.

Use final `dvfs/` plus `evaluate_validation.py` as the authoritative result.

## 13.6 Proxy metrics are not ground truth

C8B pseudo Dice increased by approximately `0.000782`, while GT Dice increased by approximately `0.000637`. The correlation is positive but imperfect.

A candidate that improves pseudo labels can still worsen true anatomy.

C10 strengthens this warning: the forced-pseudo validation branch achieved better pseudo Dice and NCC than the GT branch while producing lower true GT lobe Dice.

## 13.7 No true train-set Dice

The 200 training pairs do not include lobe ground truth. Evaluation there must rely on:

- pseudo-label transport consistency;
- image/MIND similarity;
- lung support overlap;
- folding;
- inverse consistency;
- residual magnitude;
- failure/fallback rate;
- qualitative review.

## 13.8 Deleted training/output assets

Older generated training DVFs and training-output folders were deleted for storage reasons.

Historical freeze verification passed for 220 files, but future work must not assume those generated assets still exist. They must be inventoried or regenerated before train-set experiments.

## 13.9 C10 robustness scale

Only ten unlabeled training cases have been evaluated in the recovered C10 pilot. This is insufficient to estimate the failure rate expected on a 100-case hidden cohort.

The next stress test must record:

- completion and fallback rate;
- segmentation failures and empty/missing lobes;
- folding and Jacobian quantiles;
- inverse consistency;
- NCC/MIND and HU residuals;
- pseudo-label transport consistency;
- displacement magnitudes;
- runtime and peak CPU/GPU memory;
- case-level outlier flags and QC.

## 13.10 Hidden-label input contract

The project does not yet have written confirmation that hidden labels are exposed to submitted methods. Evaluator access to labels is not equivalent to participant-method access. The GT branch must remain disabled unless the interface and rules explicitly permit it.

## 13.11 C10 reproducibility package

The C10 outputs are partially verified, but the supplied archive was truncated before the runner/config/manifests could be recovered. Exact loss terms, policy options, parent DVFs, path resolution, fallback logic, and hashes remain pending a code-only package audit.

---

# 14. Official validation milestones

## First topology-safe submission

Approximate visible result:

| Submission | Rank | Combined | Lobe DSC | Folding |
|---|---:|---:|---:|---:|
| C1-derived submission | about 20 | 0.96182 | 0.96182 | 0.0 |

## C4-HAST official submission

| Submission | Rank at the time | Combined | Lobe DSC | Folding |
|---|---:|---:|---:|---:|
| C4-HAST | 7 | 0.97459 | 0.97459 | 0.0 |

C4 local score:

```text
0.976117844
```

C4 official visible score:

```text
0.97459
```

The local-to-server difference should not be treated as a constant offset.

Later C7/C8/C10 results in this README are local unless separately packaged and submitted.

---

# 15. Reproducibility and frozen assets

## Historical freeze verification

A freeze manifest previously passed:

```text
status          = PASS
checked files   = 220
failure count   = 0
training RAS    = 200/200
validation grid = 10/10
validation RAS  = 10/10
```

Manifest path:

```text
outputs/sgr_fm/frozen_assets/fm_dvf_freeze_manifest.json
```

Because some older training/output assets were later deleted, the current filesystem must be re-audited before relying on this historical manifest.

## Important fingerprints

```text
C2-LONG
fdc2c0a64374aee658b647b55ae673f50faf2d08aed79ea7e0d6dd3d17fadcd3

B0 propagated final
d064cd195baa6b721126bf27b9ca735845a83b39a1c18660b970958414711d4b
```

Patch hashes from recent branches:

```text
C6-RMLA patch
792b6e8b4d787407ff5b68aa049d62461655a00dcdefd4a92fdeb534b9f95639

C7A semantic-Adam patch
64479da28ed5b7814f415591d85152d1a00665c42ffcae8d72961b78d86a5751

C7A-v2 selector patch
e0cb6de2928fe5084b3d70f9f7222fd01460df368cb1a65a0ac2e6dcb5975a49

C8A GT-oracle patch
cd03607d7bc7f1421ec5334cb3c24f281a5ce4c8a8de8fbd20c6a797ce0b08a2

C8B calibrated pseudo-label patch
28438fa7451f8ed02d38c4a214fb17f0f0d666d2aef785f31a640c74bc32540f
```

## C10 evidence status

Verified from the recoverable C10 output prefix:

```text
c10_hybrid_label_aware/evaluation/validation/validation_metrics.json
c10_hybrid_label_aware/evaluation/training/training_metrics.json
c10_validation_pseudo/c10_hybrid_summary.json
c10_validation_pseudo/c10_validation_summary.json
```

The C10 runner/config/manifests and a complete archive hash are still pending. Do not mark C10 frozen or submission-ready until a code-only package is audited and its parent/input fingerprints are recorded.

---

# 16. Current output directories

```text
outputs/sgr_fm/
├── segmentation_phase1_totalsegmentator/
├── c2_long_tto/
├── c4_hast/
├── c4_1_hast_pcf/
├── c5b_lmind_residual/
├── c2r_lineage_retune/
├── lineage_propagation/
├── c6_rml_aware/
├── c7a_semantic_adam/
├── c7a_v2_relaxed_selector/
├── c7b_multi_init_semantic/
├── c8a_gt_oracle/
├── c8b_pseudolabel_calibrated/
├── c10_hybrid_label_aware/
└── c10_validation_pseudo/
```

The most important current directories are:

```text
outputs/sgr_fm/c8b_pseudolabel_calibrated/
outputs/sgr_fm/c8a_gt_oracle/
outputs/sgr_fm/c10_hybrid_label_aware/
outputs/sgr_fm/c10_validation_pseudo/
outputs/sgr_fm/c7b_multi_init_semantic/
outputs/sgr_fm/c6_rml_aware/
outputs/sgr_fm/segmentation_phase1_totalsegmentator/
```

---

# 17. Recent commands

## C7B

```bash
python run_sgr_fm_c7b_multi_init_validation.py \
  --config configs/sgr_fm_c7b_multi_init.yaml
```

## C8A oracle

```bash
python run_sgr_fm_c8a_gt_oracle_validation.py \
  --config configs/sgr_fm_c8a_gt_oracle.yaml
```

## C8B calibrated pseudo-label branch

```bash
python run_sgr_fm_c8b_pseudolabel_calibrated_validation.py \
  --config configs/sgr_fm_c8b_pseudolabel_calibrated.yaml
```

C8B evaluator output:

```text
outputs/sgr_fm/c8b_pseudolabel_calibrated/eval/validation_metrics.json
```

C8B summary:

```text
outputs/sgr_fm/c8b_pseudolabel_calibrated/c8b_pseudolabel_calibrated_summary.json
```

## C10 hybrid runs

The exact C10 runner and configuration filenames were not recoverable from the truncated attachment. They must not be guessed in the reproducibility record. After the code-only package is supplied, record separate commands for:

- automatic GT-when-available routing;
- forced-pseudo validation;
- the unlabeled training stress test;
- post-run evaluation/QC.

Verified C10 output locations:

```text
outputs/sgr_fm/c10_hybrid_label_aware/evaluation/validation/validation_metrics.json
outputs/sgr_fm/c10_hybrid_label_aware/evaluation/training/training_metrics.json
outputs/sgr_fm/c10_validation_pseudo/c10_hybrid_summary.json
outputs/sgr_fm/c10_validation_pseudo/c10_validation_summary.json
```

---

# 18. Current decision and next development step

## Current branch roles

### Label-free validation champion

```text
C8B calibrated pseudo-label registration
mean Dice     = 0.979183639347
mean folding  = 0.088432671647%
max folding   = 0.241414956920%
```

C8B remains the current label-free champion. The C10 forced-pseudo regression is extremely close but lower by `0.000053906021`.

### Hybrid deployment framework

```text
C10 label-aware registration
GT-available validation Dice = 0.984021391501
forced-pseudo validation Dice = 0.979129733326
unlabeled training pilot      = 10 cases, proxy evaluation only
```

C10 is accepted as the deployment architecture because it makes label availability explicit. It is not yet frozen or submission-ready because its code/config/manifests were not recoverable from the supplied archive.

### Oracle diagnostic

```text
C8A GT-oracle
mean Dice     = 0.984083315736
mean folding  = 0.123560972067%
max folding   = 0.255693104200%
```

C8A remains the highest local diagnostic result. It is not the practical CT-only champion because it uses validation GT labels directly.

## Immediate reproducibility gate

Before changing the C10 algorithm:

1. Re-upload C10 as a small code/config/manifests-only archive without DVFs or PNG QC.
2. Audit the actual label resolver, loss terms, parent DVFs, path resolution, fallback rules, and output geometry.
3. Run synthetic smoke tests for GT, forced-pseudo, missing-one-label, missing-both-labels, and forbidden-GT policies.
4. Verify that each case records `label_mode`, label paths/hashes, parent DVF hash, policy, fallback reason, and final geometry.
5. Freeze a reproducible C10 baseline only after exact regression against the recovered metrics.

## Next accuracy branch: unified multi-cue C10

The next accuracy experiment should improve the pseudo branch while preserving the hybrid interface. Its objective should combine:

- GT labels when legitimately available, otherwise confidence-weighted pseudo labels;
- CT structural similarity using MIND and/or local NCC;
- lobe SDF and fissure/surface alignment;
- smoothness and residual-magnitude control;
- explicit Jacobian/topology control;
- exact pull-field composition for nontrivial residuals;
- parent fallback and non-RML regression guards.

Pseudo-geometry work should remain concentrated on:

- EXP RML;
- RUL–RML horizontal fissure;
- RML/RUL–RLL oblique fissure;
- hard-case right-lung boundaries;
- confidence-weighted boundary regions.

The submission-critical pseudo branch must satisfy:

```text
no case-specific validation GT at inference
no GT-based per-case candidate selection
small, confidence-driven anatomical corrections
exact-parent fallback
non-RML preservation
soft folding zone 0.25–0.30%
hard rejection 0.50%
```

Candidate mechanisms:

1. Global RML/right-fissure morphological calibration derived from the C8A gap audit.
2. Confidence maps that weaken pseudo-label loss where the segmenter is unreliable.
3. Boundary-offset correction rather than whole-lobe dilation.
4. Multiple corrected pseudo-label candidates with a non-GT selector.
5. Exact composition restoration for the residual update.
6. Structural CT/MIND evidence to reject pseudo-consistent but anatomically implausible candidates.
7. Surface-distance and internal vessel/airway proxy evaluation in addition to lobe Dice.

## Scale-up plan for unlabeled cases

The recovered C10 training run contains only ten cases. Scale-up should be staged:

```text
10-case pilot already completed
→ 25-case debugging cohort
→ 100-case deployment-matched stress test
→ all 200 training pairs if resources permit
```

The 100-case report must summarize failures, fallback, per-case proxy distributions, folding, inverse consistency, displacement, runtime, memory, and QC outliers. It must explicitly label pseudo Dice as a proxy, not true Dice.

Teacher-student training remains deferred until the pseudo branch and geometry are stronger:

```text
C10 reproducibility audit
→ unified multi-cue forced-pseudo baseline
→ validation regression without per-case GT selection
→ 100-case unlabeled stress test
→ optional 200-case completion
→ regenerate frozen teacher DVFs
→ train a lung-specific registration/refinement model
→ reapply semantic TTO
→ evaluate hidden 100-case test
```

Training directly from current C8B/C10 pseudo targets may improve speed and robustness, but it may also distill the current RML/right-fissure bias. It should not be assumed to surpass the teacher without stronger targets or additional self-supervised structural information.

---

# 19. Main lessons

1. **Geometry auditing was necessary.** The raw baseline is now trusted.
2. **Topology can be improved without losing Dice.** C1 reduced folding by orders of magnitude.
3. **Residual TTO is effective when the initializer is already strong.**
4. **Longer schedules alone give diminishing returns.**
5. **Pseudo-objective learning can fail despite good training curves.**
6. **New anatomical representations produce the largest improvements.**
7. **RML/right-fissure awareness is necessary but not sufficient.**
8. **Near-zero folding was over-constraining the search near the score ceiling.**
9. **Relaxed topology enabled the C7A-v2 jump.**
10. **Multi-init blending alone adds little once parent branches saturate.**
11. **Validation GT reveals a large anatomical-target gap.**
12. **The RML is the dominant oracle gap.**
13. **Lobe reweighting helps, but geometry correction is still required.**
14. **C8B is better than C7B without per-case GT optimization, but remains validation-calibrated.**
15. **The hidden test cannot be predicted reliably from ten validation cases alone.**
16. **Explicit GT/pseudo routing is a valid deployment architecture, but it does not make GT available where the method input contract withholds it.**
17. **The paired C10 runs estimate a label-source penalty of approximately `0.004892` mean Dice, pending confirmation of the main run's mode manifests.**
18. **A dense DVF warps the whole image; five-lobe supervision does not validate every internal structure.**
19. **Pseudo Dice and NCC can improve while true lobe Dice worsens. Multi-cue selection is required.**
20. **The ten-case unlabeled C10 pilot is encouraging for execution but exposes harder RML/topology behavior and must be scaled.**

---

# 20. Final status snapshot

| Item | Status |
|---|---|
| Raw uniGradICON reproduced | PASS |
| Warp direction audited | PASS |
| XYZ order audited | PASS |
| Pull convention audited | PASS |
| Challenge-grid restoration audited | PASS |
| Phase-1 pseudo lobes/support | PASS |
| C1 support-guided FM | PASS |
| C2 / C2-LONG | PASS |
| C3 learned refiner | REJECTED |
| C4-HAST | PASS |
| C4 official visible submission | 0.97459, rank 7 at the time |
| C4.1 | PASS |
| C5a FireANTs | ABORTED |
| C5b LMIND | PASS |
| B0 lineage propagation | PASS |
| C6 RML-aware | PASS |
| C7A | PASS, mixed per-case behavior |
| C7A-v2 | PASS, major gain |
| C7B | PASS, small gain |
| C8A GT-oracle | PASS, diagnostic only |
| C8B calibrated pseudo labels | **PASS / CURRENT PRACTICAL CHAMPION** |
| C10 GT-available validation run | PASS locally, 0.984021392; optimizer mode-manifest confirmation pending |
| C10 forced-pseudo validation branch | PASS locally, 0.979129733; slightly below C8B |
| C10 ten-case unlabeled training pilot | PASS metrics available; true Dice unavailable |
| C10 hybrid routing architecture | ACCEPTED AS CURRENT FRAMEWORK |
| C10 code/config/manifests audit | **PENDING — supplied archive truncated** |
| Current label-free gap to 0.98 | **0.000816361 using C8B** |
| Next intended branch | **Unified label + CT/MIND + surface + topology C10** |
| 100-pair train stress test | PENDING |
| Full 200-pair train stress test | OPTIONAL AFTER 100-PAIR GATE |
| Written hidden-label input clarification | PENDING |
| Hidden 100-case test | PENDING |

---

## End of README

This README is the project-level technical history, current result record, limitation audit, and development roadmap for SGR-FM on MICCAI 2026 Learn2Breath Task 2.
