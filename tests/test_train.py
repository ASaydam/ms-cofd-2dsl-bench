import json
import shutil
import sys

import numpy as np
import pytest

from test_baselines import GRIDS, fake_dataset

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")


def test_models_keep_the_grid():
    from msbench.models import MODELS

    x = torch.randn(2, 6, 37, 13)
    for name, model in MODELS.items():
        assert model(6, 3, width=8, **({'modes_x': 6, 'modes_y': 4} if name == 'fno' else {}))(x).shape == (2, 3, 37, 13)


def training_dataset(folder):
    """the fake dataset of the baseline test, with a val split and the train statistics in every file"""
    import h5py

    fake_dataset(folder)
    for run in GRIDS:
        shutil.copy(folder / run / 'train.h5', folder / run / 'val.h5')
        with h5py.File(folder / run / 'train.h5') as h5:
            fields = h5['fields'][:]
        mean, std = fields.mean(axis=(0, 2, 3)), fields.std(axis=(0, 2, 3))
        for split in ('train', 'val', 'test'):
            with h5py.File(folder / run / f"{split}.h5", 'a') as h5:
                h5.attrs.update({'mean': mean, 'std': std})


@pytest.mark.parametrize('task, model', [('T1', 'unet'), ('T2-1mm', 'fno')])
def test_train_writes_results_and_continues(tmp_path, monkeypatch, task, model):
    from msbench import train
    from msbench.torch_data import TaskData

    dataset, run = tmp_path / 'dataset', tmp_path / 'run'
    training_dataset(dataset)
    data = TaskData(dataset, task, 'test')
    inputs, target, base, fluid = data[0]
    assert inputs.shape[0] == data.in_channels and target.shape == base.shape == (3, *GRIDS['16x'])
    if task.startswith('T2'):                    # the base is the interpolated coarse field, close to the label
        assert float(((base - target) ** 2 * fluid).mean()) < float((target ** 2 * fluid).mean())
    assert np.allclose(data.denormalise_fields(target.numpy()), data.files['16x']['fields'][0][:3], atol=1e-4)

    arguments = ['train', str(dataset), str(run), '--task', task, '--model', model, '--width', '8', '--modes-x', '6',
                 '--modes-y', '4', '--batch', '1', '--workers', '0', '--epochs', '2', '--splits', 'test',
                 '--uncertainty', str(dataset / 'uncertainty.parquet')]
    monkeypatch.setattr(sys, 'argv', arguments + ['--hours', '1e-9'])
    train.main()                                 # the time budget stops it after the first epoch
    results = json.loads((run / 'results' / f"{model}-s0" / f"{task}.json").read_text())
    assert results['cost']['epochs'] == 1 and results['splits']['test']['fields']['velocity']['median'] > 0
    assert ('attributes' in results['splits']['test']) == task.startswith('T2')

    monkeypatch.setattr(sys, 'argv', arguments)
    train.main()                                 # continues from last.pt with the second epoch
    log = (run / 'log.csv').read_text().splitlines()
    assert [line.split(',')[0] for line in log[1:]] == ['1', '2']
    assert json.loads((run / 'results' / f"{model}-s0" / f"{task}.json").read_text())['cost']['epochs'] == 2
