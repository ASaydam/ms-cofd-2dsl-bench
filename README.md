# MS-COFD-2DSL benchmark

Tasks, metrics and baselines for **MS-COFD-2DSL** (Multi-Scale Central Object Flow Dataset, 2D Steady Laminar). The
dataset has 4,500 obstacles in a 360 × 120 mm channel, each solved to steady state on five levels from 2 mm to
0.25 mm, with an uncertainty for every label.

The tasks, splits and metrics are fixed in [docs/tasks-and-metrics.md](docs/tasks-and-metrics.md):

- **T1, shape to flow:** geometry → u_x, u_y, p on the 0.25 mm grid.
- **T2, coarse to fine:** a coarse solution (2 mm, 1 mm first order or 1 mm) → the 0.25 mm fields.
- **T3, coarse to attributes:** a coarse solution → drag, lift, recirculation, wake, … of the 0.25 mm solution.

Every task is scored on the in-distribution test split and on three out-of-distribution sets (blockage, rotation,
position). Errors are compared with the uncertainty of the labels, and every result states its cost.

## Use

```bash
pip install -e .
python -m msbench.evaluate <dataset folder> <predictions folder> T2-1mm \
    --uncertainty label_uncertainty.parquet --output results/<method>/T2-1mm.json
```

Predictions: `<split>.h5` with `fields` (N, 3, 1440, 480) and `case_name`, in the order of the dataset file of the split
(fields tasks), or `<split>.csv` with `case_name` and one column per scored attribute (T3).

## Reference networks

A UNet and a Fourier neural operator (`msbench/models.py`), trained and scored in one command:

```bash
pip install -e .[train]
python -m msbench.train <dataset folder> <run folder> --task T2-1mm --model unet --seed 0 \
    --uncertainty references/label_uncertainty.parquet
```

For T2 the networks learn the correction to the interpolated coarse fields: bicubic for the velocity and bicubic with
the obstacle filled for the pressure, the best interpolation of each channel. A SLURM example is in
`scripts/train.sbatch`.

## Adding a method

Open a pull request with:
- `results/<method>/<task>.json`, written by `msbench.evaluate`;
- the predictions or the code that makes them;
- the training cost (GPU-hours, GPU), the inference time per case, and the hardware.

## Leaderboard

Reference results on MS-COFD-2DSL v1.0.0 (release identifier: MD5 of `checksums.md5`
209ebc4f85e13235bbc3e6b7f2a63243). Networks: mean ± standard deviation over seeds 0–2, trained and scored at
2b8b8e7 on one A40; scores were checked to reproduce exactly from the saved checkpoints.

**Fields (T1, T2):** median relative L2 error over the cases, in %. Velocity on the test split and the three
out-of-distribution sets; pressure on the test split.

| Task | Method | Velocity, test | Blockage | Rotation | Position | Pressure, test |
|---|---|---|---|---|---|---|
| T1 | training mean | 22.84 | 29.66 | 22.18 | 29.24 | 34.40 |
| T1 | nearest training case | 10.97 | 17.04 | 11.12 | 11.87 | 19.34 |
| T1 | UNet | 0.751 ± 0.010 | 1.286 | 0.741 | 0.955 | 1.422 ± 0.008 |
| T1 | **FNO** | **0.384 ± 0.007** | **0.947** | **0.398** | **0.482** | **0.956 ± 0.028** |
| T2-2mm | interpolation | 0.98 | 1.25 | 1.02 | 0.98 | 1.89 |
| T2-2mm | FNO | 0.208 ± 0.002 | 0.262 | 0.193 | 0.221 | 0.442 ± 0.003 |
| T2-2mm | **UNet** | **0.139 ± 0.002** | **0.181** | **0.137** | **0.156** | **0.300 ± 0.007** |
| T2-1mm-FO | interpolation | 1.94 | 3.33 | 2.44 | 1.85 | 3.82 |
| T2-1mm-FO | FNO | **0.216 ± 0.001** | **0.295** | **0.218** | 0.237 | 0.425 ± 0.001 |
| T2-1mm-FO | UNet | 0.219 ± 0.002 | 0.302 | 0.220 | **0.232** | **0.407 ± 0.006** |
| T2-1mm | interpolation | 0.42 | 0.52 | 0.48 | 0.39 | 1.09 |
| T2-1mm | FNO | 0.119 ± 0.001 | 0.147 | 0.117 | 0.123 | 0.292 ± 0.004 |
| T2-1mm | **UNet** | **0.075 ± 0.000** | **0.097** | **0.072** | **0.079** | **0.187 ± 0.001** |
| T2 (any input) | solve at 0.5 mm | 0.18 | 0.23 | 0.20 | 0.18 | 0.50 |

- **Interpolation:** bicubic for the velocity and bicubic with the obstacle filled for the pressure, the best
  interpolation of each channel; the networks for T2 learn the correction to these fields.
- **Solve at 0.5 mm:** the `4x` run of the dataset against the label, the cost-equivalent alternative of refining
  the mesh.
- **Bold:** the best network in each column of a task. On T2-1mm-FO the FNO and the UNet are within about two seed
  standard deviations of each other in velocity.

**Attributes (T3):** median absolute error on the attribute's scale (docs/tasks-and-metrics.md), in %, and in
brackets the share of test cases within the label uncertainty.

| Task | Method | Drag | Lift | Recirculation length | Dissipation |
|---|---|---|---|---|---|
| T3-2mm | coarse solution as is | 1.06 (0.55) | 0.74 (0.57) | 2.68 (0.51) | 4.63 (0.00) |
| T3-1mm-FO | coarse solution as is | 3.99 (0.13) | 1.36 (0.42) | 10.35 (0.16) | 4.44 (0.00) |
| T3-1mm | coarse solution as is | 0.74 (0.78) | 0.47 (0.80) | 1.58 (0.70) | 1.64 (0.00) |
| T3 (any input) | solve at 0.5 mm | 0.32 (0.96) | 0.24 (0.94) | 0.61 (0.90) | 0.51 (0.37) |

No network has been trained for T3 yet.

**Cost:** UNet 7.8 M parameters, 1.6 GPU-hours of training, 38 ms per case; FNO 25 M parameters, 2.6 GPU-hours,
35 ms per case (A40, inference without the coarse solve). The coarse solves take 8 s (2 mm), 77 s (1 mm, first order)
and 110 s (1 mm) on one core, the 0.5 mm solve 19 min (medians). The result files of every entry will be added under
`results/` with the first public release.

## Licence

Code: MIT. The dataset itself: CC BY 4.0.
