"""
Reading a MS-COFD-2DSL dataset folder (v1.0.0):

    <run>/<split>.h5     fields (N, 6, nx, ny) float32, case_name (N,)
    metadata.parquet     one row per case: split, sampled parameters, geometry, and per run <run>.<attribute>
    mesh_study.parquet   per case and attribute: the values of every level

Runs: 0.25x (2 mm), base_fo (1 mm, first order), base_lu (1 mm), 4x (0.5 mm), 16x (0.25 mm, the labels).
Needs numpy, h5py and pyarrow only (the pipeline's container has them).
"""

from pathlib import Path

RUNS = ['0.25x', 'base_fo', 'base_lu', '4x', '16x']
LABEL_RUN = '16x'
TRACKS = {'2mm': '0.25x', '1mm-FO': 'base_fo', '1mm': 'base_lu'}
SPLITS = ['train', 'val', 'test', 'ood_blockage', 'ood_rotation', 'ood_position']
SCORING_SPLITS = ['test', 'ood_blockage', 'ood_rotation', 'ood_position']


def read_columns(path, columns=None):
    """{column: list of values} of a parquet file"""
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=columns).to_pydict()


class Dataset:
    def __init__(self, folder):
        self.folder = Path(folder)
        if not (self.folder / 'metadata.parquet').exists():
            raise FileNotFoundError(f"{self.folder} is not a MS-COFD-2DSL dataset folder (no metadata.parquet)")

    def metadata(self, columns):
        """{case: {column: value}} for the given columns of metadata.parquet"""
        columns = list(dict.fromkeys(['case_name'] + list(columns)))
        table = read_columns(self.folder / 'metadata.parquet', columns)
        return {case: {c: table[c][i] for c in columns[1:]} for i, case in enumerate(table['case_name'])}

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
        columns = ['bbox_height', f'{LABEL_RUN}.Fx'] + [f'{LABEL_RUN}.{a}' for a in attributes]
        meta = self.metadata(columns)
        return {name: {'drag': float(meta[name][f'{LABEL_RUN}.Fx']), 'height': float(meta[name]['bbox_height']),
                       **{a: float(meta[name][f'{LABEL_RUN}.{a}']) for a in attributes}}
                for name in self.case_names(split)}
