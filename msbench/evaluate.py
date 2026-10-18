"""
Scores the predictions of one method for one task (docs/tasks-and-metrics.md):

    python -m msbench.evaluate <dataset folder> <predictions folder> <task> --uncertainty <label_uncertainty.parquet>
                               --output results/<method>/<task>.json [--splits test ood_blockage ...]

    task  T1, T2-2mm, T2-1mm-FO, T2-1mm (fields: <predictions folder>/<split>.h5 with fields (N, 3, 1440, 480) and
          case_name), T3-2mm, T3-1mm-FO, T3-1mm (attributes: <predictions folder>/<split>.csv with case_name and one
          column per scored attribute)

The label uncertainty is the table of the pipeline's tools/uncertainty.py (case_name, attribute, source, U_%). Both
T2 and T3 are scored against source "openfoam", the uncertainty of the 0.25 mm solution (decision 4 of
docs/tasks-and-metrics.md). Source "array", the convergence of the array attributes as mesh and grid are refined
together, is a validation result and is not used for scoring. A table without the source column gives every attribute
one U for both.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from msbench.attributes import attributes
from msbench.data import SCORING_SPLITS, Dataset
from msbench.metrics import (ARRAY_SCORED, SCORED, attribute_error, field_errors, summarise_attribute,
                             summarise_fields)

FIELD_TASKS = ['T1', 'T2-2mm', 'T2-1mm-FO', 'T2-1mm']
SCORING_SOURCE = 'openfoam'           # the U both tasks are scored against (decision 4)
ATTRIBUTE_TASKS = ['T3-2mm', 'T3-1mm-FO', 'T3-1mm']


def read_uncertainty(path):
    """{(case, attribute, source): U as a fraction on the attribute's scale}, source 'openfoam' or 'array'"""
    import pyarrow.parquet as pq
    from msbench.data import read_columns

    has_source = 'source' in pq.read_schema(path).names
    table = read_columns(path, ['case_name', 'attribute', 'U_%'] + (['source'] if has_source else []))
    sources = table['source'] if has_source else [None] * len(table['case_name'])
    uncertainty = {}
    for c, a, u, s in zip(table['case_name'], table['attribute'], table['U_%'], sources):
        for source in ([s] if s else ['openfoam', 'array']):
            uncertainty[(c, a, source)] = u / 100 if u is not None else float('nan')
    return uncertainty


def read_predicted_fields(path):
    import h5py
    h5 = h5py.File(path)
    names = [c.decode() if isinstance(c, bytes) else c for c in h5['case_name'][:]]
    return h5, names


def score_predictors(dataset, split, predictors, uncertainty, with_attributes):
    """
    field errors of every case for several predictors at once, and for T2 the errors of the array attributes; the label
    is read and its attributes computed once per case
    :param predictors: {name: function(row index, case name, label sample) -> (3, nx, ny) prediction}
    :return: {name: {'fields': ..., 'attributes': ...}}
    """
    labels = dataset.labels(split, [])
    rows = {name: [] for name in predictors}
    errors = {name: {a: ([], []) for a in ARRAY_SCORED} for name in predictors}
    for i, (case, label) in enumerate(dataset.samples('16x', split)):
        label = label.astype(float)
        true = attributes(label) if with_attributes else None
        for name, predict in predictors.items():
            prediction = np.asarray(predict(i, case, label), dtype=float)
            rows[name].append(field_errors(prediction, label, label))
            if with_attributes:
                predicted = attributes(np.concatenate([prediction, label[3:]]))
                for a in ARRAY_SCORED:
                    errors[name][a][0].append(attribute_error(a, predicted[a], true[a], labels[case]['drag'],
                                                              labels[case]['height']))
                    errors[name][a][1].append(uncertainty.get((case, a, SCORING_SOURCE), float('nan')))
    results = {}
    for name in predictors:
        results[name] = {'fields': summarise_fields(rows[name])}
        if with_attributes:
            results[name]['attributes'] = {a: summarise_attribute(*errors[name][a]) for a in ARRAY_SCORED}
    return results


def score_fields(dataset, predictions, split, uncertainty, with_attributes):
    """the predictions of one method from <predictions>/<split>.h5, in the order of the dataset file"""
    h5, names = read_predicted_fields(Path(predictions) / f"{split}.h5")

    def from_file(i, case, label):
        if names[i] != case:
            raise ValueError(f"{split}: row {i} of the predictions is {names[i]}, the dataset has {case}")
        return h5['fields'][i]

    with h5:
        return score_predictors(dataset, split, {'method': from_file}, uncertainty, with_attributes)['method']


def score_attributes(dataset, predictions, split, uncertainty):
    """errors of the predicted attributes (<predictions>/<split>.csv) against the 0.25 mm OpenFOAM attributes"""
    with open(Path(predictions) / f"{split}.csv", newline='') as file:
        predicted = {row['case_name']: row for row in csv.DictReader(file)}
    return score_attribute_values(dataset, split, predicted, uncertainty)


def score_attribute_values(dataset, split, predicted, uncertainty):
    """
    :param predicted: {case: {attribute: value}} for every case of the split
    """
    labels = dataset.labels(split, SCORED)
    missing = sorted(set(labels) - set(predicted))
    if missing:
        raise ValueError(f"{split}: no prediction for {len(missing)} cases, e.g. {missing[:3]}")
    result = {}
    for name in SCORED:
        e, u, p, t = [], [], [], []
        for case, label in labels.items():
            value = float(predicted[case][name])
            e.append(attribute_error(name, value, label[name], label['drag'], label['height']))
            u.append(uncertainty.get((case, name, SCORING_SOURCE), float('nan')))
            p.append(value)
            t.append(label[name])
        result[name] = summarise_attribute(e, u, p if name in ('Fx', 'Fy') else None, t if name in ('Fx', 'Fy') else None)
    return {'attributes': result}


def main():
    parser = argparse.ArgumentParser(description="Score the predictions of a method for a task")
    parser.add_argument('dataset')
    parser.add_argument('predictions')
    parser.add_argument('task', choices=FIELD_TASKS + ATTRIBUTE_TASKS)
    parser.add_argument('--uncertainty', required=True, help="label_uncertainty.parquet")
    parser.add_argument('--output', required=True, help="results/<method>/<task>.json")
    parser.add_argument('--splits', nargs='*', default=SCORING_SPLITS)
    args = parser.parse_args()

    dataset, uncertainty = Dataset(args.dataset), read_uncertainty(args.uncertainty)
    results = {'task': args.task, 'splits': {}}
    for split in args.splits:
        if args.task in FIELD_TASKS:
            results['splits'][split] = score_fields(dataset, args.predictions, split, uncertainty,
                                                    with_attributes=args.task.startswith('T2'))
        else:
            results['splits'][split] = score_attributes(dataset, args.predictions, split, uncertainty)
        print(f"{split}: scored", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=1) + '\n')
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
