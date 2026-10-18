import json
import sys

import numpy as np
import pytest

from msbench import metrics
from msbench.interpolation import upsample, weights
from test_benchmark import NX, NY, sample

GRIDS = {'0.25x': (37, 13), 'base_fo': (73, 25), 'base_lu': (73, 25), '4x': (109, 37), '16x': (NX, NY)}


def test_interpolation_is_exact_for_linear_fields():
    x = np.linspace(0, 1, 10)[:, None] * np.ones((1, 5))
    exact = 2 * np.linspace(0, 1, 37)[:, None] * np.ones((1, 17)) + 1
    assert np.allclose(upsample(2 * x + 1, 37, 17, 'bilinear'), exact)
    # bicubic repeats the edge values beyond the ends: exact for a linear field away from the first and last interval
    assert np.allclose(upsample(2 * x + 1, 37, 17, 'bicubic')[4:-4], exact[4:-4])
    assert np.allclose(weights(5, 9, 'nearest').sum(1), 1)


def fake_dataset(folder):
    """two splits (train with 3 cases, test with 2), every run, attributes of every run equal to the label's"""
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq

    names = {'train': [f"normal_rectangle_{i:04d}" for i in range(1, 4)],
             'test': [f"normal_rectangle_{i:04d}" for i in range(4, 6)]}
    for run, (nx, ny) in GRIDS.items():
        (folder / run).mkdir(parents=True)
        for split, cases in names.items():
            with h5py.File(folder / run / f"{split}.h5", 'w') as h5:
                h5['fields'] = np.stack([sample(int(c[-1]), nx, ny) for c in cases]).astype(np.float32)
                h5['case_name'] = np.array(cases, dtype=object).astype('S')
    rows = []
    for split, cases in names.items():
        for c in cases:
            k = int(c[-1])
            row = {'case_name': c, 'split': split, 'size': 'normal', 'geometry': 'rectangle', 'bbox_height': 30.0,
                   'sampled_width': 40.0 + k, 'sampled_height': 20.0 + k, 'rotation': 5.0 * k, 'shift_x': -k,
                   'shift_y': k}
            for run in GRIDS:
                row.update({f"{run}.{a}": 1.0 + k for a in metrics.SCORED})
            rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), folder / 'metadata.parquet')
    attributes = set(metrics.SCORED) | set(metrics.ARRAY_SCORED)
    pq.write_table(pa.Table.from_pylist([{'case_name': r['case_name'], 'attribute': a, 'U_%': 1.0}
                                         for r in rows for a in attributes]), folder / 'uncertainty.parquet')


def test_baselines_write_every_result(tmp_path, monkeypatch):
    pytest.importorskip("scipy")
    from msbench import baselines

    dataset, results = tmp_path / 'dataset', tmp_path / 'results'
    fake_dataset(dataset)
    monkeypatch.setattr(sys, 'argv', ['baselines', str(dataset), str(results), '--uncertainty',
                                      str(dataset / 'uncertainty.parquet'), '--splits', 'test'])
    baselines.main()

    expected = [f"interp-{m}/T2-{t}.json" for m in ('nearest', 'bilinear', 'bicubic', 'bicubic_filled')
                for t in ('2mm', '1mm-FO', '1mm')]
    expected += ['solve-0.5mm/T2.json', 'solve-0.5mm/T3.json', 'train-mean/T1.json', 'nearest-case/T1.json']
    expected += [f"coarse-as-is/T3-{t}.json" for t in ('2mm', '1mm-FO', '1mm')]
    assert sorted(p.relative_to(results).as_posix() for p in results.rglob('*.json')) == sorted(expected)

    t2 = json.loads((results / 'interp-bicubic' / 'T2-1mm.json').read_text())['splits']['test']
    assert 0 < t2['fields']['u_x']['median'] < 0.2 and set(t2['attributes']) == set(metrics.ARRAY_SCORED)
    finer = json.loads((results / 'solve-0.5mm' / 'T2.json').read_text())['splits']['test']
    assert finer['fields']['u_x']['median'] < t2['fields']['u_x']['median']        # a finer level is closer
    t3 = json.loads((results / 'coarse-as-is' / 'T3-1mm.json').read_text())['splits']['test']['attributes']
    assert all(s['median_abs_error'] == 0 for s in t3.values())                 # equal attributes in the fake data
    t1 = json.loads((results / 'nearest-case' / 'T1.json').read_text())['splits']['test']
    assert t1['fields']['velocity']['median'] > 0
