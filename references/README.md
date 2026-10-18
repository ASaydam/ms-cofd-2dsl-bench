# References

`label_uncertainty.parquet`: the numerical uncertainty U of every attribute and case of MS-COFD-2DSL v1.0.0. It comes
from the four second-order levels (2, 1, 0.5 and 0.25 mm), with the least-squares procedure of Eça & Hoekstra (2014).
It has two sources, in the column `source`:

- **`openfoam`:** the attributes computed by OpenFOAM on the simulation mesh. 85,500 rows (4,500 cases × 19
  attributes). **This is the U that T2 and T3 are scored against** (docs/tasks-and-metrics.md, decision 4).
  - **Made with:** `python -m tools.uncertainty <dataset folder> <output>` (pipeline commit cb3473b; identical with
    ec2ac0e).
  - **Inputs:** the v1.0.0 files `mesh_study.parquet` (MD5 7f7988317b54c06e33eaa0b0c7ee1db7) and `metadata.parquet`
    (MD5 286ce8431baf13cb0490828f91ce4858). Both match the dataset's `checksums.md5`.
- **`array`:** the same attributes computed from the arrays of every level on its own grid. 36,000 rows (4,500 cases
  × 8 attributes). **A validation result, not used for scoring.**
  - **What it measures:** how the array values converge when the solver mesh and the array grid are refined together.
    The resampling loses a different amount at each level, so this U is larger than the OpenFOAM one for gradient
    quantities: dissipation 2.9 % against 0.5 % (median).
  - **Why it is not used for scoring:** in T2 the prediction and the label are arrays on the same grid, and the
    resampling loss cancels between them (decision 4).
  - **Made with:** `python -m tools.uncertainty <dataset folder> <output> --arrays <folder>` (pipeline commit
    ec2ac0e).
  - **Inputs:** `metadata.parquet` as above, and the array attributes from `python -m tools.flow_attributes --coarse
    <run> --fine 16x --output <folder>` for the runs 0.25x, base_lu and 4x (pipeline 40b637f, on the v1.0.0 results,
    2026-10-18). MD5 of the three csv files:
    - `flow_attributes_0.25x_16x.csv`: 0d38ae8f7d209ab4610b2611cbd9be9e
    - `flow_attributes_base_lu_16x.csv`: a6a80739a0ce13ff504eae893b38efc4
    - `flow_attributes_4x_16x.csv`: f5fe5384e824ad3bd29575c9b5501558

**This file:** MD5 45336550b3ea8499e1f7506d17ad6911; 121,500 rows.

**Columns:**
- `case_name`, `attribute`, `source`;
- the values of the four levels: `16x`, `4x`, `base`, `0.25x`;
- `U` (absolute), `p`, `method`, `phi_0`;
- `U_%`: U on the attribute's scale (docs/tasks-and-metrics.md), which the metrics use;
- `U1_%`: the simple 3·|φ(0.25) − φ(0.5)|, for comparison.
