"""
PyTorch view of a task (docs/tasks-and-metrics.md) on the dataset files, one sample at a time from disk (the 0.25 mm
training split is 51 GB, more than a job's share of memory).

    T1          input: the geometry channels of the 0.25 mm grid; base: 0
    T2-<track>  input: the coarse fields interpolated to the 0.25 mm grid (bicubic, obstacle filled) and the geometry
                channels of the 0.25 mm grid; base: the interpolated fields, so a network learns the correction

Every channel is normalised with the mean and standard deviation of the training split stored in the files (the
fields with those of the 0.25 mm run). A sample is (input, target, base, fluid mask), the target and the base
normalised like the fields; the prediction of a network is base + output.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from msbench.data import LABEL_RUN, TRACKS
from msbench.interpolation import upsample_sample

GEOMETRY = slice(3, 6)
FIELDS = slice(0, 3)


def task_run(task):
    """the coarse run of a T2 task, None for T1"""
    return TRACKS[task.split('-', 1)[1]] if task.startswith('T2') else None


class TaskData(Dataset):
    def __init__(self, folder, task, split):
        import h5py
        from pathlib import Path

        self.folder, self.task, self.split = Path(folder), task, split
        self.coarse_run = task_run(task)
        with h5py.File(self.folder / LABEL_RUN / f"{split}.h5") as h5:
            self.length = len(h5['case_name'])
            self.shape = h5['fields'].shape[2:]
            self.mean = np.asarray(h5.attrs['mean'], dtype=np.float32)
            self.std = np.asarray(h5.attrs['std'], dtype=np.float32)
            self.case_names = [c.decode() if isinstance(c, bytes) else c for c in h5['case_name'][:]]
        self.std = np.where(self.std > 0, self.std, 1).astype(np.float32)
        if self.coarse_run:
            with h5py.File(self.folder / self.coarse_run / f"{split}.h5") as h5:
                coarse = [c.decode() if isinstance(c, bytes) else c for c in h5['case_name'][:]]
            if coarse != self.case_names:
                raise ValueError(f"{split}: the cases of {self.coarse_run} are not in the order of {LABEL_RUN}")
        self.files = None                      # opened in each worker process

    @property
    def in_channels(self):
        return 3 if self.task == 'T1' else 6

    def open(self):
        import h5py
        self.files = {LABEL_RUN: h5py.File(self.folder / LABEL_RUN / f"{self.split}.h5")}
        if self.coarse_run:
            self.files['coarse'] = h5py.File(self.folder / self.coarse_run / f"{self.split}.h5")

    def __len__(self):
        return self.length

    def normalise(self, values, channels):
        return (values - self.mean[channels, None, None]) / self.std[channels, None, None]

    def denormalise_fields(self, values):
        """normalised (3, nx, ny) u_x, u_y, p back to physical units"""
        return values * self.std[FIELDS, None, None] + self.mean[FIELDS, None, None]

    def __getitem__(self, i):
        if self.files is None:
            self.open()
        label = self.files[LABEL_RUN]['fields'][i].astype(np.float32)
        geometry = self.normalise(label[GEOMETRY], GEOMETRY)
        target = self.normalise(label[FIELDS], FIELDS)
        if self.coarse_run:
            coarse = self.files['coarse']['fields'][i].astype(np.float64)
            base = self.normalise(upsample_sample(coarse, *self.shape, 'bicubic_filled').astype(np.float32), FIELDS)
            inputs = np.concatenate([base, geometry])
        else:
            base = np.zeros_like(target)
            inputs = geometry
        fluid = (label[4] != 1)[None].astype(np.float32)
        return tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in (inputs, target, base, fluid))
