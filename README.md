# SGR-FM Project — MICCAI 2026 Learn2Breath Task 2

## EXP→INSP deformable lung CT registration

**Project:** SGR-FM — Segmentation-Guided Registration with Flow Matching  
**Researcher:** Mebarki M. Oussama  
**Status:** Active development  
**Current best validation-GT-free per-case optimizer:** **C8B calibrated pseudo-label registration**  
**Current local mean lobe Dice:** **0.979183639347**  
**Current local mean folding:** **0.088432671647%**  
**Current GT-oracle diagnostic upper bound:** **0.984083315736**  
**Last updated:** 2026-07-14

---

# 1. Executive summary

The project began from a fully audited raw uniGradICON EXP→INSP baseline and evolved through support guidance, segmentation-guided residual test-time optimization, hierarchical lobe/fissure surface refinement, localized MIND refinement, lineage retuning, RML-aware refinement, relaxed-topology semantic registration, multi-initializer selection, a validation-GT oracle diagnostic, and finally GT-gap-calibrated pseudo-label registration.

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

# 11. Current scientific conclusions

## 11.1 The basic registration conventions are not the main problem

The baseline geometry was independently audited, and multiple branches consistently improved Dice. C8A showed that the same general deformation machinery can gain more than `0.0055` when given correct anatomical targets.

Therefore, the project is not primarily blocked by:

- wrong warp direction;
- wrong component order;
- wrong pull convention;
- wrong output grid;
- inability to produce useful deformation.

## 11.2 Automatic anatomical target quality is the dominant current bottleneck

Evidence:

- TotalSegmentator RML is the weakest pseudo lobe.
- C6 improved RML in all ten cases but only modestly improved the global score.
- C8A increased RML by approximately `0.01449` on average relative to C7B.
- `NLST_0006` EXP pseudo RML Dice against GT was only approximately `0.85558`.
- C8B class reweighting improved the score but recovered only approximately `11.5%` of the oracle gap.

## 11.3 Registration/regularization remains a secondary bottleneck

C8A did not reach a mean of 0.988 even with GT labels.

The remaining limitations include:

- additive residual updates in recent branches;
- coarse control-grid parameterization;
- only two or three optimization stages;
- simple one-hot Dice rather than richer GT-SDF/surface terms in C8A/C8B;
- folding penalties and hard rejection;
- candidate-scale discretization;
- lack of exact residual composition in the recent lightweight branches.

## 11.4 High validation scores do not prove hidden-test superiority

A visible validation method may use:

- stronger automatic segmentation;
- validation-label calibration;
- direct validation-GT optimization if permitted;
- more aggressive topology tradeoffs;
- case-specific selection.

It is not justified to claim that another participant is overfitting or will fail on the 100-case hidden test without their method and hidden-test results.

What can be stated is:

> Direct use of validation GT can substantially inflate visible-validation performance relative to a method that must infer anatomy automatically at test time.

## 11.5 Current method is a pipeline, not a single model that “understands” both tasks

The practical pipeline is:

```text
pretrained lobe segmentation
→ pseudo-anatomical targets
→ pretrained registration initializer
→ case-specific semantic TTO
→ selector / fallback
→ DVF
```

It is not yet a jointly trained segmentation-registration model.

Testing on the 200 training pairs will measure pipeline robustness, not true lobe Dice, because training GT lobes are unavailable.

---

# 12. Current limitations

## 12.1 Pseudo-label geometry

The largest limitation is not only label confidence but boundary placement.

C8B reweighted the pseudo labels but did not change their geometry. It therefore cannot correct a pseudo RML boundary that is systematically misplaced.

## 12.2 RML and right fissures

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

## 12.3 Validation calibration and overfitting risk

C8B does not use validation GT during per-case optimization, but its global calibration depends on C8A validation-GT statistics.

Repeated tuning on only ten validation cases can overfit:

- class weights;
- selector thresholds;
- folding thresholds;
- candidate scales;
- stage schedules.

## 12.4 Additive displacement updates

C7A onward largely uses additive residual updates. Exact pull composition should be restored before large residuals or training-time teacher generation.

## 12.5 Canonical-output labeling

The recent `dvfs_canonical_ras/` outputs have not undergone the same independent canonical-frame audit as the original baseline and early branches.

Use final `dvfs/` plus `evaluate_validation.py` as the authoritative result.

## 12.6 Proxy metrics are not ground truth

C8B pseudo Dice increased by approximately `0.000782`, while GT Dice increased by approximately `0.000637`. The correlation is positive but imperfect.

A candidate that improves pseudo labels can still worsen true anatomy.

## 12.7 No true train-set Dice

The 200 training pairs do not include lobe ground truth. Evaluation there must rely on:

- pseudo-label transport consistency;
- image/MIND similarity;
- lung support overlap;
- folding;
- inverse consistency;
- residual magnitude;
- failure/fallback rate;
- qualitative review.

## 12.8 Deleted training/output assets

Older generated training DVFs and training-output folders were deleted for storage reasons.

Historical freeze verification passed for 220 files, but future work must not assume those generated assets still exist. They must be inventoried or regenerated before train-set experiments.

---

# 13. Official validation milestones

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

Later C7/C8 results in this README are local unless separately packaged and submitted.

---

# 14. Reproducibility and frozen assets

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

---

# 15. Current output directories

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
└── c8b_pseudolabel_calibrated/
```

The most important current directories are:

```text
outputs/sgr_fm/c8b_pseudolabel_calibrated/
outputs/sgr_fm/c8a_gt_oracle/
outputs/sgr_fm/c7b_multi_init_semantic/
outputs/sgr_fm/c6_rml_aware/
outputs/sgr_fm/segmentation_phase1_totalsegmentator/
```

---

# 16. Recent commands

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

---

# 17. Current decision and next development step

## Frozen practical champion

```text
C8B calibrated pseudo-label registration
mean Dice     = 0.979183639347
mean folding  = 0.088432671647%
max folding   = 0.241414956920%
```

## Frozen oracle diagnostic

```text
C8A GT-oracle
mean Dice     = 0.984083315736
mean folding  = 0.123560972067%
max folding   = 0.255693104200%
```

C8A is not the practical champion because it uses validation GT labels directly.

## Next branch: C8C pseudo-label geometry correction

C8B changed lobe importance but did not correct lobe shape or boundary position.

The next branch should therefore modify pseudo-label geometry conservatively, particularly:

- EXP RML;
- RUL–RML horizontal fissure;
- RML/RUL–RLL oblique fissure;
- hard-case right-lung boundaries;
- confidence-weighted boundary regions.

The intended constraints are:

```text
no case-specific validation GT at inference
small, global or confidence-driven corrections
exact-parent fallback
non-RML preservation
soft folding zone 0.25–0.30%
hard rejection 0.50%
```

Potential C8C mechanisms:

1. Global RML/right-fissure morphological calibration derived from the C8A gap audit.
2. Confidence maps that weaken pseudo-label loss where the segmenter is unreliable.
3. Boundary-offset correction rather than whole-lobe dilation.
4. Multiple corrected pseudo-label candidates with a non-GT selector.
5. Exact composition restoration for the residual update.
6. Independent evaluation on the 200 training pairs using robustness proxies before any teacher-student training.

## Training on the 200 pairs

Training remains deferred until the teacher baseline is stronger and more geometrically reliable.

The correct sequence is:

```text
C8C corrected pseudo-label baseline
→ validate locally
→ stress-test on 200 training pairs
→ regenerate frozen teacher DVFs
→ train a lung-specific registration/refinement model
→ reapply semantic TTO
→ evaluate hidden 100-case test
```

Training directly from current C8B pseudo targets may improve speed and robustness, but it may also distill the current segmentation bias. It should not be assumed to surpass the teacher without stronger targets or additional self-supervised information.

---

# 18. Main lessons

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

---

# 19. Final status snapshot

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
| Current gap to 0.98 | **0.000816361** |
| Next intended branch | **C8C pseudo-label geometry correction** |
| 200-pair train stress test | PENDING |
| Hidden 100-case test | PENDING |

---

## End of README

This README is the project-level technical history, current result record, limitation audit, and development roadmap for SGR-FM on MICCAI 2026 Learn2Breath Task 2.
