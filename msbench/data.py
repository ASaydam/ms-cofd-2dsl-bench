"""
Reading a MS-COFD-2DSL dataset folder (v1.0.0):

    <run>/<split>.h5     fields (N, 6, nx, ny) float32, case_name (N,)
    metadata.parquet     one row per case: split, sampled parameters, geometry, and per run <run>.<attribute>
    mesh_study.parquet   per case and attribute: the values of every level

Runs: 0.25x (2 mm), base_fo (1 mm, first order), base_lu (1 mm), 4x (0.5 mm), 16x (0.25 mm, the labels).
"""

from pathlib import Path

RUNS = ['0.25x', 'base_fo', 'base_lu', '4x', '16x']
LABEL_RUN = '16x'
TRACKS = {'2mm': '0.25x', '1mm-FO': 'base_fo', '1mm': 'base_lu'}
SPLITS = ['train', 'val', 'test', 'ood_blockage', 'ood_rotation', 'ood_position']
SCORING_SPLITS = ['test', 'ood_blockage', 'ood_rotation', 'ood_position']


class Dataset:
    def __init__(self, folder):
        self.folder = Path(folder)
        if not (self.folder / 'metadata.parquet').exists():
            raise FileNotFoundError(f"{self.folder} is not a MS-COFD-2DSL dataset folder (no metadata.parquet)")

    def metadata(self, columns=None):
        import pandas as pd
        return pd.read_parquet(self.folder / 'metadata.parquet', columns=columns)

    def case_names(self, split, run=LABEL_RUN):
        """case names of a split, in the order of its files"""
        import h5py
        with h5py.File(self.folder / run / f"{split}.h5") as h5:
            return [c.decode() if isinstance(c, bytes) else c for c in h5['case_name'][:]]

    def samples(self, run, split):
        """(case name, (6, nx, ny) sample) of every case of a split, one at a time"""
        import h5py
        with h5py.File(self.folder / run / f"{split}.h5") as h5:
            names = h5['case_name'][:]
            for i in range(len(names)):
                name = names[i].decode() if isinstance(names[i], bytes) else names[i]
                yield name, h5['fields'][i]

    def labels(self, split, attributes):
        """{case: {attribute: value of the 0.25 mm run}}, with the drag and the obstacle height (bbox_height)"""
        columns = ['case_name', 'bbox_height', f'{LABEL_RUN}.Fx'] + [f'{LABEL_RUN}.{a}' for a in attributes]
        meta = self.metadata(columns=list(dict.fromkeys(columns))).set_index('case_name')
        names = self.case_names(split)
        return {name: {'drag': float(meta.at[name, f'{LABEL_RUN}.Fx']), 'height': float(meta.at[name, 'bbox_height']),
                       **{a: float(meta.at[name, f'{LABEL_RUN}.{a}']) for a in attributes}} for name in names}
