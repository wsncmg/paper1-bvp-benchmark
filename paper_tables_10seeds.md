# Manuscript tables (generated, do not hand-edit)

Sources: `teachers_erm_epcon.json, teachers_gpu1.json, teachers_id.json, teachers_rest.json, teachers_supcon.json, students_epcon.json, students_erm.json, students_kdepcon.json`

Convention: mean +/- sample std over seeds (ddof=1) of each seed's fold-mean accuracy; fold-level std separately. Epoch selection uses a 10% source-internal validation split; the target test split is evaluated once.

## Coverage (completed seeds per cell, out of 10)

| Method | Cross-Location | Cross-Orientation | Cross-User | Cross-Session |
|---|---|---|---|---|
| ERM | 10/10 | 10/10 | 10/10 | 5/10 |
| SupCon | 10/10 | 10/10 | 10/10 | 5/10 |
| EPCon | 10/10 | 10/10 | 10/10 | 5/10 |
| EPCon+CORAL | 5/10 | 5/10 | 5/10 | 5/10 |
| DANN | 5/10 | 5/10 | 5/10 | 5/10 |
| GroupDRO | 10/10 | 10/10 | 10/10 | 5/10 |
| IRM | 5/10 | 5/10 | 5/10 | 5/10 |

## Table 1 -- Protocol A (in-domain) vs Protocol B (strict LODO)

| Method | In-domain (80/20) | Cross-Location | Cross-Orientation | Cross-User | Cross-Session |
|---|---|---|---|---|---|
| ERM | 75.90 +/- 2.21 | 71.61 +/- 1.91 | 62.43 +/- 1.71 | 64.01 +/- 2.16 | 30.20 +/- 1.26 |
| EPCon | 79.26 +/- 4.17 | 72.73 +/- 5.11 | 61.81 +/- 4.63 | 65.54 +/- 1.79 | 31.22 +/- 0.37 |

## Table 2 -- teacher domain generalisation

| Method | Cross-Location | Cross-Orientation | Cross-User | Cross-Session |
|---|---|---|---|---|
| ERM | 71.61 +/- 1.91 | 62.43 +/- 1.71 | 64.01 +/- 2.16 | 30.20 +/- 1.26 |
| SupCon | 74.25 +/- 2.23 | 63.14 +/- 4.49 | 65.42 +/- 1.74 | 30.35 +/- 0.89 |
| EPCon | 72.73 +/- 5.11 | 61.81 +/- 4.63 | 65.54 +/- 1.79 | 31.22 +/- 0.37 |
| EPCon+CORAL | 70.50 +/- 3.56 | 63.13 +/- 2.15 | 65.78 +/- 1.01 | 30.09 +/- 1.02 |
| DANN | 70.89 +/- 2.30 | 62.95 +/- 1.76 | 64.27 +/- 1.80 | 30.73 +/- 1.56 |
| GroupDRO | 69.95 +/- 2.91 | 61.28 +/- 1.82 | 58.79 +/- 1.73 | 29.80 +/- 1.34 |
| IRM | 68.07 +/- 5.72 | 61.75 +/- 2.14 | 65.06 +/- 2.05 | 30.58 +/- 0.78 |

_Fold-level std across all seed x fold cells:_

| Method | Cross-Location | Cross-Orientation | Cross-User | Cross-Session |
|---|---|---|---|---|
| ERM | 4.15 | 10.04 | 16.57 | 3.44 |
| SupCon | 4.15 | 13.03 | 16.32 | 3.23 |
| EPCon | 10.37 | 14.57 | 15.11 | 3.42 |
| EPCon+CORAL | 6.17 | 11.69 | 17.22 | 2.45 |
| DANN | 7.03 | 11.61 | 16.93 | 2.92 |
| GroupDRO | 5.15 | 7.73 | 15.42 | 2.81 |
| IRM | 12.11 | 11.08 | 16.78 | 2.49 |

## Paired tests vs. ERM (matched seed x fold), Holm-corrected


**Cross-Location** (n=50 matched cells)

| Method | Delta vs ERM (pp) | t | p (t) | p (Holm) | p (Wilcoxon) | p (Holm) |
|---|---|---|---|---|---|---|
| SupCon | +2.64 | 4.530 | 0.0000 | 0.0001 | 0.0000 | 0.0001 |
| EPCon | +1.12 | 0.739 | 0.4637 | 0.4637 | 0.0045 | 0.0089 |
| GroupDRO | -1.66 | -2.194 | 0.0330 | 0.0661 | 0.0649 | 0.0649 |

**Cross-Orientation** (n=50 matched cells)

| Method | Delta vs ERM (pp) | t | p (t) | p (Holm) | p (Wilcoxon) | p (Holm) |
|---|---|---|---|---|---|---|
| SupCon | +0.71 | 0.553 | 0.5825 | 1.0000 | 0.0065 | 0.0196 |
| EPCon | -0.62 | -0.514 | 0.6097 | 1.0000 | 0.4319 | 0.4319 |
| GroupDRO | -1.15 | -1.545 | 0.1288 | 0.3864 | 0.1515 | 0.3030 |

**Cross-User** (n=120 matched cells)

| Method | Delta vs ERM (pp) | t | p (t) | p (Holm) | p (Wilcoxon) | p (Holm) |
|---|---|---|---|---|---|---|
| SupCon | +1.42 | 2.445 | 0.0159 | 0.0159 | 0.0463 | 0.0463 |
| EPCon | +1.53 | 2.870 | 0.0049 | 0.0097 | 0.0149 | 0.0298 |
| GroupDRO | -5.22 | -7.537 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**Cross-Session** (n=10 matched cells)

| Method | Delta vs ERM (pp) | t | p (t) | p (Holm) | p (Wilcoxon) | p (Holm) |
|---|---|---|---|---|---|---|
| SupCon | +0.16 | 0.198 | 0.8478 | 1.0000 | 0.7695 | 1.0000 |
| EPCon | +1.02 | 1.439 | 0.1840 | 1.0000 | 0.1934 | 1.0000 |
| EPCon+CORAL | -0.11 | -0.219 | 0.8317 | 1.0000 | 1.0000 | 1.0000 |
| DANN | +0.53 | 0.919 | 0.3820 | 1.0000 | 0.2324 | 1.0000 |
| GroupDRO | -0.40 | -0.468 | 0.6511 | 1.0000 | 0.8457 | 1.0000 |
| IRM | +0.39 | 0.520 | 0.6154 | 1.0000 | 0.3750 | 1.0000 |

## Table 3 -- teacher vs. student

| Model | Cross-Location | Cross-Orientation | Cross-User | Cross-Session |
|---|---|---|---|---|
| Teacher (EPCon) | 69.55 +/- 6.88 | 62.95 +/- 4.26 | 65.77 +/- 1.69 | 31.49 +/- 1.20 |
| Teacher (ERM) | 71.70 +/- 2.06 | 62.35 +/- 2.92 | 64.22 +/- 1.65 | 31.25 +/- 1.84 |
| Student (scratch) | 58.69 +/- 1.96 | 52.73 +/- 2.37 | 53.54 +/- 1.68 | 28.39 +/- 1.94 |
| Student (KD) | 62.87 +/- 2.06 | 56.03 +/- 2.89 | 55.96 +/- 1.35 | 28.99 +/- 1.25 |
| Student (KD+EPCon) | 61.21 +/- 4.51 | 54.45 +/- 4.24 | 55.01 +/- 3.85 | 30.02 +/- 1.26 |

## Per-class accuracy (cross-location, pooled over folds, same runs)

| Class | Gesture | teacher_epcon|loc | kd_epcon|loc | Delta pp |
|---|---|---|---|---|
| 0 | Push | 85.13 | 76.07 | -9.06 |
| 1 | Pull | 81.05 | 68.08 | -12.97 |
| 2 | Slide | 78.48 | 76.66 | -1.81 |
| 3 | Up | 71.07 | 65.52 | -5.56 |
| 4 | Down | 61.54 | 41.74 | -19.80 |
| 5 | Push-Pull | 75.91 | 58.55 | -17.35 |
| - | **overall** | 75.53 | 64.44 | -11.09 |

_The unweighted mean of the per-class rows must match the overall row when class supports are balanced; a gap means the per-class and aggregate numbers came from different runs._
