"""
The reference baselines of docs/tasks-and-metrics.md that need no training, scored without writing predictions:

    python -m msbench.baselines <dataset folder> <results folder> --uncertainty <label_uncertainty.parquet>
                                [--tasks T1 T2 T3] [--splits test ood_blockage ...]

    T2  every track (2 mm, 1 mm first order, 1 mm) interpolated to the 0.25 mm grid with every method of
        msbench/interpolation.py: results/interp-<method>/T2-<track>.json; and the equal-cost reference "solve at
        0.5 mm", the 0.5 mm results interpolated (bicubic, obstacle filled): results/solve-0.5mm/T2.json
    T3  the attributes of the coarse run as they are: results/coarse-as-is/T3-<track>.json; and the attributes of the
        0.5 mm run: results/solve-0.5mm/T3.json
    T1  the mean field of the training split: results/train-mean/T1.json; the nearest training case of the same size
        and family in the sampled parameters (width, height, rotation, shift, standardised): results/nearest-case/T1.json

Every split is read once: the 0.25 mm labels and their attributes are computed once per case for all baselines.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from msbench.data import LABEL_RUN, SCORING_SPLITS, TRACKS, Dataset
from msbench.evaluate import read_uncertainty, score_attribute_values, score_predictors
from msbench.interpolation import METHODS, upsample_sample
from msbench.metrics import SCORED

PARAMETERS = ['sampled_width', 'sampled_height', 'rotation', 'shift_x', 'shift_y']


def write(results_folder, baseline, task, splits):
    path = Path(results_folder) / baseline / f"{task}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'task': task, 'baseline': baseline, 'splits': splits}, indent=1) + '\n')


def open_runs(dataset, split, runs):
    """the open h5 files of a split for every run: {run: h5 file}"""
    import h5py
    return {run: h5py.File(dataset.folder / run / f"{split}.h5") for run in runs}


def t2(dataset, results_folder, splits, uncertainty):
    runs = list(TRACKS.values()) + ['4x']
    out = {}
    for split in splits:
        files = open_runs(dataset, split, runs)
        try:
            predictors = {}
            for track, run in TRACKS.items():
                for method in METHODS:
                    def predict(i, case, label, run=run, method=method):
                        return upsample_sample(files[run]['fields'][i].astype(float), *label.shape[1:], method)
                    predictors[(f"interp-{method}", f"T2-{track}")] = predict
            predictors[('solve-0.5mm', 'T2')] = lambda i, case, label: upsample_sample(
                files['4x']['fields'][i].astype(float), *label.shape[1:], 'bicubic_filled')
            for key, result in score_predictors(dataset, split, predictors, uncertainty, with_attributes=True).items():
                out.setdefault(key, {})[split] = result
        finally:
            for h5 in files.values():
                h5.close()
        print(f"T2 {split}: scored", flush=True)
    for (baseline, task), split_results in out.items():
        write(results_folder, baseline, task, split_results)


def t3(dataset, results_folder, splits, uncertainty):
    choices = [('coarse-as-is', f"T3-{track}", run) for track, run in TRACKS.items()] + [('solve-0.5mm', 'T3', '4x')]
    meta = dataset.metadata([f"{run}.{a}" for _, _, run in choices for a in SCORED])
    for baseline, task, run in choices:
        split_results = {}
        for split in splits:
            predicted = {case: {a: float(meta[case][f"{run}.{a}"]) for a in SCORED}
                         for case in dataset.case_names(split)}
            split_results[split] = score_attribute_values(dataset, split, predicted, uncertainty)
        write(results_folder, baseline, task, split_results)
        print(f"{baseline} {task}: scored", flush=True)


def t1(dataset, results_folder, splits, uncertainty):
    import h5py

    meta = dataset.metadata(PARAMETERS + ['size', 'geometry'])
    train = h5py.File(dataset.folder / LABEL_RUN / 'train.h5')
    try:
        train_names = [c.decode() if isinstance(c, bytes) else c for c in train['case_name'][:]]
        total = None
        for i in range(len(train_names)):
            values = train['fields'][i][:3].astype(float)
            total = values if total is None else total + values
        mean = total / len(train_names)

        def parameters(case):
            return np.array([float(meta[case][p]) for p in PARAMETERS])

        table = np.stack([parameters(c) for c in train_names])
        centre, spread = table.mean(0), table.std(0, ddof=1)
        standard = (table - centre) / spread
        classes = np.array([f"{meta[c]['size']} {meta[c]['geometry']}" for c in train_names])

        def nearest(i, case, label):
            point = (parameters(case) - centre) / spread
            same = classes == f"{meta[case]['size']} {meta[case]['geometry']}"
            distance = np.where(same, np.linalg.norm(standard - point, axis=1), np.inf)
            return train['fields'][int(np.argmin(distance))][:3]

        out = {}
        for split in splits:
            predictors = {'train-mean': lambda i, case, label: mean, 'nearest-case': nearest}
            for name, result in score_predictors(dataset, split, predictors, uncertainty,
                                                 with_attributes=False).items():
                out.setdefault(name, {})[split] = result
            print(f"T1 {split}: scored", flush=True)
    finally:
        train.close()
    for name, split_results in out.items():
        write(results_folder, name, 'T1', split_results)


def main():
    parser = argparse.ArgumentParser(description="Score the reference baselines that need no training")
    parser.add_argument('dataset')
    parser.add_argument('results')
    parser.add_argument('--uncertainty', required=True)
    parser.add_argument('--tasks', nargs='*', default=['T3', 'T2', 'T1'], choices=['T1', 'T2', 'T3'])
    parser.add_argument('--splits', nargs='*', default=SCORING_SPLITS)
    args = parser.parse_args()
    dataset, uncertainty = Dataset(args.dataset), read_uncertainty(args.uncertainty)
    for task in args.tasks:
        {'T1': t1, 'T2': t2, 'T3': t3}[task](dataset, args.results, args.splits, uncertainty)
    print(f"written to {args.results}")


if __name__ == "__main__":
    main()
