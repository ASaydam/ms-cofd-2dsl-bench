# References

## label_uncertainty.parquet

The numerical uncertainty U of every attribute and case of MS-COFD-2DSL v1.0.0. It comes from the four second-order
levels (2, 1, 0.5 and 0.25 mm), with the least-squares procedure of Eça & Hoekstra (2014). It is the file
`label_uncertainty.parquet` of the dataset (the same MD5), written by the pipeline's `dataset/build_dataset.py --arrays`
with `tools/uncertainty.py` at commit a21e54b (2026-11-06). The two sources are in the column `source`:

- **`openfoam`:** the attributes computed by OpenFOAM on the simulation mesh. 85,500 rows (4,500 cases × 19
  attributes). **This is the U that T2 and T3 are scored against** (docs/tasks-and-metrics.md, decision 4).
  - **Made with:** `python -m tools.uncertainty <dataset folder> <output>`.
  - **Inputs:** the v1.0.0 files `mesh_study.parquet` (MD5 5b6a1e309987a073af5a6fb7bb448131) and the cell counts of
    `metadata.parquet` (MD5 ab8d0464846a5d22023dc9a76ebc4b94).
- **`array`:** the same attributes computed from the arrays of every level on its own grid. 36,000 rows (4,500 cases
  × 8 attributes). **A validation result, not used for scoring.**
  - **What it measures:** how the array values converge when the solver mesh and the array grid are refined together.
    The resampling loses a different amount at each level, so this U is larger than the OpenFOAM one for gradient
    quantities: dissipation 2.9 % against 0.5 % (median).
  - **Why it is not used for scoring:** in T2 the prediction and the label are arrays on the same grid, so both carry a
    resampling loss of the same origin. This U describes how that loss changes from level to level, not a comparison
    on one fixed grid (decision 4). The two losses need not cancel exactly, so a T2 score measures agreement with the
    stored label arrays.
  - **Made with:** `python -m tools.uncertainty <dataset folder> <output> --arrays array_check`.
  - **Inputs:** `metadata.parquet` as above, and the csv files in `array_check/`.

**This file:** MD5 b90adbb0c090b36521d6bd9d219e2ebe; 121,500 rows.

**Columns:**
- `case_name`, `attribute`, `source`;
- the values of the four levels: `16x`, `4x`, `base`, `0.25x`;
- `U` (absolute), `p`, `method`, `phi_0`;
- `U_%`: U on the attribute's scale (docs/tasks-and-metrics.md), which the metrics use;
- `U1_%`: the simple 3·|φ(0.25) − φ(0.5)|, for comparison.

## array_check/

The attributes computed from the arrays of every level, against the OpenFOAM values on the simulation mesh.
- **Made with:** `python -m tools.flow_attributes --coarse <run> --fine 16x --output <folder>` (pipeline 69c8c75, on
  the per-case results of the v1.0.0 production, 2026-11-06).
- **Contents:** one file per coarse run. Each holds that run and 16x, for all 4,500 cases.
- **Used by:**
  - the `array` source above;
  - the pipeline's `validation/array_vs_openfoam.py` (Table 10 of the dataset paper).
- **MD5:**
  - `flow_attributes_0.25x_16x.csv`: 150e4381cf29b4b386a08a73b9164060
  - `flow_attributes_base_fo_16x.csv`: 44fe5d8c4e0a42e1a18d691f74ad9a4e
  - `flow_attributes_base_lu_16x.csv`: ba7846aa0f3e1f8380258563a64fac6e
  - `flow_attributes_4x_16x.csv`: eaf606976522fcdea651047b58137249
