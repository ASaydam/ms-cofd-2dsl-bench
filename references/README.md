# References

## label_uncertainty.parquet

The numerical uncertainty U of every attribute and case of MS-COFD-2DSL v1.0.0. It comes from the four second-order
levels (2, 1, 0.5 and 0.25 mm), with the least-squares procedure of Eça & Hoekstra (2014). Both sources were made with
the pipeline's `tools/uncertainty.py` at commit 37588e4 (2026-10-19). The two sources are in the column `source`:

- **`openfoam`:** the attributes computed by OpenFOAM on the simulation mesh. 85,500 rows (4,500 cases × 19
  attributes). **This is the U that T2 and T3 are scored against** (docs/tasks-and-metrics.md, decision 4).
  - **Made with:** `python -m tools.uncertainty <dataset folder> <output>`.
  - **Inputs:** the v1.0.0 files `mesh_study.parquet` (MD5 7f7988317b54c06e33eaa0b0c7ee1db7) and the cell counts of
    `metadata.parquet`.
  - The cell counts are the same before and after the metadata rebuild of 2026-10-19. The rebuild changed only empty
    `<run>.init_from` cells: MD5 286ce8431baf13cb0490828f91ce4858 before, 43bf223f07f4e5e2081e639beffc1882 after.
  - The values are identical to those of the first version of this file (cb3473b, 2026-10-18).
- **`array`:** the same attributes computed from the arrays of every level on its own grid. 36,000 rows (4,500 cases
  × 8 attributes). **A validation result, not used for scoring.**
  - **What it measures:** how the array values converge when the solver mesh and the array grid are refined together.
    The resampling loses a different amount at each level, so this U is larger than the OpenFOAM one for gradient
    quantities: dissipation 2.9 % against 0.5 % (median).
  - **Why it is not used for scoring:** in T2 the prediction and the label are arrays on the same grid, and the
    resampling loss cancels between them (decision 4).
  - **Made with:** `python -m tools.uncertainty <dataset folder> <output> --arrays array_check`.
  - **Inputs:** `metadata.parquet` as above, and the csv files in `array_check/`.
  - Compared with the first version of 2026-10-18 (ec2ac0e, which used pandas): pandas' fast float parser had dropped
    the 17th digit of the inputs. The values changed by at most 2.4e-10 relative.

**This file:** MD5 fdd7677924a218804bb9d50ef60afd0c; 121,500 rows.

**Columns:**
- `case_name`, `attribute`, `source`;
- the values of the four levels: `16x`, `4x`, `base`, `0.25x`;
- `U` (absolute), `p`, `method`, `phi_0`;
- `U_%`: U on the attribute's scale (docs/tasks-and-metrics.md), which the metrics use;
- `U1_%`: the simple 3·|φ(0.25) − φ(0.5)|, for comparison.

## array_check/

The attributes computed from the arrays of every level, against the OpenFOAM values on the simulation mesh.
- **Made with:** `python -m tools.flow_attributes --coarse <run> --fine 16x --output <folder>` (pipeline 40b637f, on
  the per-case results of the v1.0.0 production, 2026-10-18).
- **Contents:** one file per coarse run. Each holds that run and 16x, for all 4,500 cases.
- **Used by:**
  - the `array` source above;
  - the pipeline's `validation/array_vs_openfoam.py` (Table 10 of the dataset paper).
- **MD5:**
  - `flow_attributes_0.25x_16x.csv`: 0d38ae8f7d209ab4610b2611cbd9be9e
  - `flow_attributes_base_fo_16x.csv`: 03d823b72d7265d1cea500fab1e41f33
  - `flow_attributes_base_lu_16x.csv`: a6a80739a0ce13ff504eae893b38efc4
  - `flow_attributes_4x_16x.csv`: f5fe5384e824ad3bd29575c9b5501558
