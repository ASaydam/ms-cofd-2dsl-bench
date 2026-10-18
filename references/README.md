# References

`label_uncertainty.parquet`: the numerical uncertainty U of every attribute and case of MS-COFD-2DSL v1.0.0. It comes
from the four second-order levels (2, 1, 0.5 and 0.25 mm), with the least-squares procedure of Eça & Hoekstra (2014).

- **Made with:** `python -m tools.uncertainty <dataset folder> <output>`, pipeline commit cb3473b (tag
  `data-v1.0.0` is the dataset).
- **Inputs:** the v1.0.0 files `mesh_study.parquet` (MD5 7f7988317b54c06e33eaa0b0c7ee1db7) and `metadata.parquet`
  (MD5 286ce8431baf13cb0490828f91ce4858). Both match the dataset's `checksums.md5`.
- **This file:** MD5 c92d4b4db74c47096438d18e907f4523; 85,500 rows (4,500 cases × 19 attributes).
- **Columns:**
  - `case_name`, `attribute`;
  - the values of the four levels: `16x`, `4x`, `base`, `0.25x`;
  - `U` (absolute), `p`, `method`, `phi_0`;
  - `U_%`: U on the attribute's scale (docs/tasks-and-metrics.md), which the metrics use;
  - `U1_%`: the simple 3·|φ(0.25) − φ(0.5)|, for comparison.
