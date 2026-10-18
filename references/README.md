# References

`label_uncertainty.parquet`: the numerical uncertainty U of every attribute and case of MS-COFD-2DSL v1.0.0. It comes
from the four second-order levels (2, 1, 0.5 and 0.25 mm), with the least-squares procedure of Eça & Hoekstra (2014).
It has two sources, in the column `source`:

- **`openfoam`:** the attributes computed by OpenFOAM on the simulation mesh, the labels of T3. 85,500 rows (4,500 cases
  × 19 attributes).
  - **Made with:** `python -m tools.uncertainty <dataset folder> <output>` (pipeline commit cb3473b; identical with
    ec2ac0e).
  - **Inputs:** the v1.0.0 files `mesh_study.parquet` (MD5 7f7988317b54c06e33eaa0b0c7ee1db7) and `metadata.parquet`
    (MD5 286ce8431baf13cb0490828f91ce4858). Both match the dataset's `checksums.md5`.
- **`array`:** the same attributes computed from the arrays of every level on its own grid, the quantities T2 scores
  (docs/tasks-and-metrics.md, decision 4). 36,000 rows (4,500 cases × 8 attributes).
  - **Why:** the resampling to the grid loses a different amount at each level, so the array attributes converge
    differently from the OpenFOAM ones. Dissipation, for instance, has a median U of 2.9 % from the arrays and 0.5 %
    from OpenFOAM.
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
