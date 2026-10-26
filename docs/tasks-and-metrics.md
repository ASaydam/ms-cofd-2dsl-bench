# Tasks and metrics

This document fixes what is predicted, from what, on which cases, and how it is scored. It is written before any model
is trained, and a change to it is a new benchmark version.

The data is MS-COFD-2DSL v1.0.0 (git tag `data-v1.0.0` of the pipeline repository; `checksums.md5` of the dataset
folder has the MD5 817adefb132205f30bf633710f554dc2).

## Data

Every case is one obstacle in a 360 × 120 mm channel, solved to steady state on five levels:

| Level | Name in the files | Grid of the arrays | Convection |
|---|---|---|---|
| 2 mm | `0.25x` | 180 × 60 | second order |
| 1 mm, first order | `base_fo` | 360 × 120 | first order |
| 1 mm | `base_lu` | 360 × 120 | second order |
| 0.5 mm | `4x` | 720 × 240 | second order |
| 0.25 mm | `16x` | 1440 × 480 | second order |

Array channels: `u_x`, `u_y`, `p`, `wall_distance`, `cell_type`, `obstacle_sdf`. The labels are always the 0.25 mm
level.

## Splits

- **`train` (3,085) and `val` (385):** for fitting and model selection.
- **`test` (386):** in distribution.
- **`ood_blockage` (224), `ood_rotation` (216), `ood_position` (204):** the upper 5 % tails of the blockage ratio,
  |rotation| and the position shift.
- Test and out-of-distribution cases are for scoring only: not for training, model selection or normalisation.

## Tasks

| Task | Input | Output |
|---|---|---|
| **T1, shape to flow** | the geometry channels of the 0.25 mm grid (`wall_distance`, `cell_type`, `obstacle_sdf`) | `u_x`, `u_y`, `p` on the 0.25 mm grid |
| **T2, coarse to fine**, three tracks: T2-2mm, T2-1mm-FO, T2-1mm | the 6 channels of the coarse level (`0.25x`, `base_fo` or `base_lu`), plus the geometry channels of the 0.25 mm grid | `u_x`, `u_y`, `p` on the 0.25 mm grid |
| **T3, coarse to attributes**, the same three tracks | as T2, plus the OpenFOAM attributes of the coarse run (computed with it, at no extra cost) | the scored attributes of the 0.25 mm run (below) |

The geometry of the fine grid is allowed as an input to T2 and T3: the geometry is known exactly before anything is
solved.

## Field metrics (T1, T2)

On the fluid points of the 0.25 mm grid (`cell_type` ≠ 1), without the first 5 mm behind the inlet (its corners):

- **relative L2 error** of `u_x`, `u_y`, `p` and of the velocity (u_x, u_y together): ‖pred − label‖ / ‖label‖;
- the same in the **near band** (`0 < obstacle_sdf ≤ 8` grid points, i.e. within 2 mm of the obstacle) and in the
  **bulk** (the other fluid points).

Each split reports the median and the mean over its cases.

## Attribute metrics (T2, T3)

**Scored attributes:**
- drag Fx, with its pressure and friction parts;
- lift Fy;
- recirculation length and area;
- wake velocity (smallest u_x 20 mm behind the obstacle);
- base pressure;
- dissipation;
- mean vorticity;
- pressure drop between the sections;
- flow split.

**Reported but not scored** (flagged in the dataset: they do not converge with the mesh): peak wall shear, peak
vorticity, front (stagnation) pressure.

The error of a case is e = (prediction − label) / scale, with the scale of the attribute:
- the drag parts and lift relative to the drag of the label (decision 2026-10-18);
- the wake velocity relative to the inlet velocity (0.1 m/s);
- base pressure relative to the dynamic pressure (0.005 m²/s²);
- recirculation length relative to at least the obstacle height, and its area to at least its square;
- every other attribute relative to its label.

Every label carries its numerical uncertainty U on the same scale. U comes from the four second-order levels, with
the least-squares procedure of Eça & Hoekstra (2014), as computed by `tools/uncertainty.py` of the pipeline. T2 and
T3 both use the U of the OpenFOAM values (decision 4).

Per split and attribute:

- median and 90th percentile of |e|;
- the **within-uncertainty rate**: the share of the cases with |e| ≤ U. A model cannot be shown to be better than the
  labels, so an error below U counts as exact.
- Spearman's rank correlation of prediction and label, for drag and lift (as in AirfRANS / ML4CFD).

**In T3** the labels are the OpenFOAM attributes of the 0.25 mm run (on the solver mesh).

**In T2** the attributes are computed from the predicted arrays with the same array function as from the 0.25 mm label
arrays (`msbench/attributes.py`), and compared with those. The forces are not computable from the arrays, so T2 scores
only the array attributes: recirculation length and area, flow split, Δp inlet–outlet, dissipation, mean vorticity and
wake velocity.

## Cost

Every result states:
- the training cost (GPU-hours and the GPU);
- the inference time per case;
- the hardware.

The reference costs are the solver times of the dataset (one core, median per case):

| Level | Solver time |
|---|---|
| 2 mm | 8 s |
| 1 mm, first order | 77 s |
| 1 mm | 110 s |
| 0.5 mm | 19 min |
| 0.25 mm | 5.6 h |

Meshing, resampling and attributes add 30–40 s per case at the coarse levels.

**The equal-cost reference for T2 and T3 is "solve at 0.5 mm"**: the released 0.5 mm results, upsampled to the
0.25 mm grid (fields) or taken as they are (attributes). A learned method that is cheaper than that, and scores better,
improves on simply refining.

## Baselines

- **T2:** interpolation of the coarse level to the 0.25 mm grid (nearest, bilinear, bicubic, and bicubic with the
  obstacle filled), and the 0.5 mm solve.
- **T3:** the coarse attributes taken as they are, and the 0.5 mm attributes.
- **T1:** the mean field of the training split, and the nearest training case in the sampled parameters.
- One UNet and one FNO for each task, as reference networks (not new methods).

## Submission and evaluation

- **Fields:** `predictions/<method>/<task>/<split>.h5`, with `fields` (N, 3, 1440, 480) float32 for u_x, u_y, p and
  `case_name` (N,) in the order of the dataset file of that split.
- **Attributes (T3):** `predictions/<method>/T3-<track>/<split>.csv`, with `case_name` and one column per scored
  attribute.
- `python -m msbench.evaluate <dataset folder> <predictions folder> <task>` writes
  `results/<method>/<task>.json`.
- The leaderboard in the README is made from `results/`. A new method arrives by pull request, with its predictions or
  code and its cost.

## Decisions (2026-10-18)

1. **T1 input:** the geometry channels of the 0.25 mm grid. The geometry is known exactly at any resolution, and the
   output is at 0.25 mm.
2. **T3 inputs:** the coarse run's own OpenFOAM attributes are allowed. They come with the coarse solve at no extra
   cost, and the baseline "coarse attribute as it is" uses them.
3. **The label uncertainty is a file of the benchmark:** `references/label_uncertainty.parquet`, made by the
   pipeline's `tools/uncertainty.py` at cb3473b from the v1.0.0 files. The benchmark does not compute it itself.
4. **The uncertainty of the T2 array attributes:** the U of the same OpenFOAM attribute. It is the uncertainty of the
   0.25 mm solution. (Decided 2026-10-18; briefly changed the same day and taken back, see below.)
   - In T2 the prediction and the label are arrays on the same 1440 × 480 grid, and their attributes come from the
     same array function. Both carry a resampling loss of the same origin (the array check: dissipation 1.7 % low at
     0.25 mm). The two losses need not cancel, because an error in the prediction can change gradients and other
     nonlinear quantities differently from the label. A T2 score therefore measures agreement with the stored label
     arrays, not accuracy against the solution on the mesh.
   - The uncertainty of the array attributes over the four levels (source `array` in the uncertainty file) measures
     something else. It is how the array values converge when the solver mesh and the array grid are refined
     together, with the resampling loss changing from 13 % at 2 mm to 1.7 % at 0.25 mm for dissipation. That makes
     it a validation result for the arrays, not the uncertainty of a comparison on a fixed grid.
   - Tried and taken back: scoring T2 against the array U (dissipation 2.94 % instead of 0.49 %). Interpolating the
     1 mm solution then counted as within U for dissipation in every test case, although its dissipation is 1.7 %
     off on the solver mesh, outside the label's 0.49 %. The low within-U rate of the 0.5 mm solve for dissipation
     is a real error of that baseline: a 0.5 mm field stored at 720 × 240 and upsampled loses gradient detail.
