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

For T2 the networks learn the correction to the bicubic interpolation of the coarse fields. A SLURM example is in
`scripts/train.sbatch`.

## Adding a method

Open a pull request with:
- `results/<method>/<task>.json`, written by `msbench.evaluate`;
- the predictions or the code that makes them;
- the training cost (GPU-hours, GPU), the inference time per case, and the hardware.

## Leaderboard

Not started: the reference baselines are being run.

## Licence

Code: MIT. The dataset itself: CC BY 4.0.
