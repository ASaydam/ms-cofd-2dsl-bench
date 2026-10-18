import csv
import json
import sys

import numpy as np
import pytest

from msbench import metrics
from msbench.metrics import (attribute_error, field_errors, regions, scale_of, spearman, summarise_attribute)

NX, NY = 145, 49           # a small grid of the 360 x 120 mm channel (2.5 mm)


def sample(seed=0, nx=NX, ny=NY):
    """a (6, nx, ny) sample: Poiseuille flow with a rectangular obstacle and a recirculation behind it"""
    from scipy.ndimage import distance_transform_edt

    NX, NY = nx, ny   # noqa: N806 (the module defaults, replaced for other grids)
    rng = np.random.default_rng(seed)
    x = np.arange(NX)[:, None] * 360 / (NX - 1) * np.ones((1, NY))
    y = np.ones((NX, 1)) * np.arange(NY)[None, :] * 120 / (NY - 1)
    obstacle = (x >= 120) & (x <= 150) & (y >= 45) & (y <= 75)
    ux = 6 * 0.1 * y * (120 - y) / 120 ** 2
    ux = np.where((x > 150) & (x < 180) & (np.abs(y - 60) < 10), -0.01, ux)      # reverse flow behind it
    ux = np.where(obstacle, 0, ux) * (1 + 0.01 * rng.standard_normal(ux.shape))
    cell_type = np.full(x.shape, 4.0)
    cell_type[obstacle] = 1
    cell_type[:, [0, -1]] = 1
    cell_type[0], cell_type[-1] = 2, 3
    sdf = np.where(obstacle, -distance_transform_edt(obstacle), distance_transform_edt(~obstacle))
    return np.stack([ux, 0.01 * ux, np.where(obstacle, 0, 1e-3 * (360 - x) / 360),
                     np.minimum(np.arange(NY), NY - np.arange(NY))[None, :] * np.ones((NX, 1)), cell_type, sdf])


def test_field_errors_and_regions():
    label = sample()
    fluid, near, bulk = regions(label)
    assert near.any() and bulk.any() and not (near & bulk).any() and np.array_equal(near | bulk, fluid)
    assert all(v == 0 for v in field_errors(label[:3], label, label).values())
    off = label[:3].copy()
    off[0] *= 1.05
    errors = field_errors(off, label, label)
    assert errors['u_x'] == pytest.approx(0.05) and errors['p'] == 0 and 0 < errors['velocity'] < 0.05


def test_attribute_scales_and_summary():
    assert scale_of('Fx_viscous', 0.1, 2.0, 30) == 2.0                      # the drag parts relative to the drag
    assert scale_of('recirculation_length_mm', 5.0, 1.0, 30) == 30          # at least the obstacle height
    assert attribute_error('Fx', 1.1, 1.0, 1.0, 30) == pytest.approx(0.1)
    summary = summarise_attribute([0.01, -0.02, 0.05], [0.02, 0.02, 0.02], [1, 2, 3], [1.1, 2.1, 2.9])
    assert summary['within_uncertainty_rate'] == pytest.approx(2 / 3) and summary['spearman'] == pytest.approx(1)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    assert set(metrics.NOT_SCORED).isdisjoint(metrics.SCORED)


def fake_dataset(folder, cases=('normal_rectangle_0001', 'normal_rectangle_0002')):
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq

    (folder / '16x').mkdir(parents=True)
    with h5py.File(folder / '16x' / 'test.h5', 'w') as h5:
        h5['fields'] = np.stack([sample(i) for i in range(len(cases))]).astype(np.float32)
        h5['case_name'] = np.array(cases, dtype=object).astype('S')
    rows = [{'case_name': c, 'split': 'test', 'bbox_height': 30.0, **{f'16x.{a}': 1.0 + i for a in metrics.SCORED}}
            for i, c in enumerate(cases)]
    pq.write_table(pa.Table.from_pylist(rows), folder / 'metadata.parquet')
    attributes = set(metrics.SCORED) | set(metrics.ARRAY_SCORED)
    pq.write_table(pa.Table.from_pylist([{'case_name': c, 'attribute': a, 'U_%': 1.0} for c in cases for a in attributes]), 
        folder / 'uncertainty.parquet')
    return cases


def test_evaluate_perfect_predictions(tmp_path, monkeypatch):
    import h5py
    from msbench import evaluate

    dataset = tmp_path / 'dataset'
    cases = fake_dataset(dataset)
    predictions = tmp_path / 'predictions'
    predictions.mkdir()
    with h5py.File(dataset / '16x' / 'test.h5') as source, h5py.File(predictions / 'test.h5', 'w') as h5:
        h5['fields'] = source['fields'][:, :3]
        h5['case_name'] = source['case_name'][:]
    with open(predictions / 'test.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['case_name'] + metrics.SCORED)
        writer.writeheader()
        writer.writerows({'case_name': c, **{a: 1.0 + i for a in metrics.SCORED}} for i, c in enumerate(cases))

    for task in ('T2-1mm', 'T3-1mm'):
        output = tmp_path / 'results' / f'{task}.json'
        monkeypatch.setattr(sys, 'argv', ['evaluate', str(dataset), str(predictions), task, '--uncertainty',
                                          str(dataset / 'uncertainty.parquet'), '--output', str(output),
                                          '--splits', 'test'])
        evaluate.main()
        result = json.loads(output.read_text())['splits']['test']
        for summary in result['attributes'].values():
            assert summary['median_abs_error'] == 0 and summary['within_uncertainty_rate'] == 1
        if task.startswith('T2'):
            assert result['fields']['velocity']['median'] == 0


def test_evaluate_refuses_another_case_order(tmp_path):
    import h5py
    from msbench import evaluate
    from msbench.data import Dataset

    dataset = tmp_path / 'dataset'
    cases = fake_dataset(dataset)
    predictions = tmp_path / 'predictions'
    predictions.mkdir()
    with h5py.File(predictions / 'test.h5', 'w') as h5:
        h5['fields'] = np.zeros((2, 3, NX, NY), np.float32)
        h5['case_name'] = np.array(cases[::-1], dtype=object).astype('S')
    with pytest.raises(ValueError, match='row 0'):
        evaluate.score_fields(Dataset(dataset), predictions, 'test', {}, with_attributes=False)
