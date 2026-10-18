"""
The metrics of docs/tasks-and-metrics.md: field errors on the 0.25 mm grid, and attribute errors on the scale of
every attribute, with the within-uncertainty rate.
"""

import numpy as np

CHANNELS = {'u_x': 0, 'u_y': 1, 'p': 2}
INLET_VELOCITY = 0.1                 # m/s
DYNAMIC_PRESSURE = 0.5 * INLET_VELOCITY ** 2
INLET_SKIP_MM = 5                    # the inlet corners are left out
NEAR_BAND = 8                        # grid points of obstacle_sdf: 2 mm on the 0.25 mm grid

# scored attributes; T2 can score only the ones computable from arrays
SCORED = ['Fx', 'Fx_pressure', 'Fx_viscous', 'Fy', 'recirculation_length_mm', 'recirculation_area_mm2',
          'wake_min_u_x', 'base_pressure', 'dissipation', 'mean_vorticity', 'dp_sections', 'flow_split_upper']
ARRAY_SCORED = ['recirculation_length_mm', 'recirculation_area_mm2', 'flow_split_upper', 'dp_inlet_outlet',
                'dissipation', 'mean_vorticity', 'wake_min_u_x']
NOT_SCORED = ['max_wall_shear', 'peak_vorticity', 'front_pressure']   # do not converge with the mesh


def regions(geometry, length_mm=360):
    """
    fluid, near-band and bulk points of a sample's geometry channels (wall_distance, cell_type, obstacle_sdf)
    :param geometry: (3, nx, ny) or the full (6, nx, ny) sample
    """
    cell_type, sdf = (geometry[1], geometry[2]) if len(geometry) == 3 else (geometry[4], geometry[5])
    nx = cell_type.shape[0]
    x = np.arange(nx)[:, None] * length_mm / (nx - 1)
    fluid = (cell_type != 1) & (x > INLET_SKIP_MM)
    near = fluid & (sdf > 0) & (sdf <= NEAR_BAND)
    return fluid, near, fluid & ~near


def relative_l2(prediction, label, mask):
    norm = np.linalg.norm(label[mask])
    return float(np.linalg.norm(prediction[mask] - label[mask]) / norm) if norm > 0 else float('nan')


def field_errors(prediction, label, geometry):
    """
    relative L2 errors of one case
    :param prediction: (3, nx, ny) u_x, u_y, p
    :param label: (3, nx, ny) or (6, nx, ny), the 0.25 mm sample
    :param geometry: geometry channels of the 0.25 mm grid
    :return: {'<channel>[_near|_bulk]': error}, channels u_x, u_y, p and velocity
    """
    masks = dict(zip(('', '_near', '_bulk'), regions(geometry)))
    errors = {}
    for suffix, mask in masks.items():
        for name, channel in CHANNELS.items():
            errors[f"{name}{suffix}"] = relative_l2(prediction[channel], label[channel], mask)
        velocity = np.stack([prediction[0], prediction[1]]), np.stack([label[0], label[1]])
        errors[f"velocity{suffix}"] = relative_l2(velocity[0], velocity[1], np.stack([mask, mask]))
    return errors


def scale_of(name, label, drag, height):
    """
    the scale of an attribute's error (decision 2026-10-18: the drag parts and lift relative to the drag)
    :param label: label value; :param drag: drag Fx of the label; :param height: obstacle height in mm
    """
    if name.startswith('Fy') or name in ('Fx_pressure', 'Fx_viscous'):
        return abs(drag)
    if name == 'wake_min_u_x':
        return INLET_VELOCITY
    if name in ('front_pressure', 'base_pressure'):
        return DYNAMIC_PRESSURE
    if name == 'recirculation_length_mm':
        return max(abs(label), height)
    if name == 'recirculation_area_mm2':
        return max(abs(label), height ** 2)
    return abs(label)


def attribute_error(name, prediction, label, drag, height):
    """(prediction - label) / scale; nan when the scale is 0"""
    scale = scale_of(name, label, drag, height)
    return (prediction - label) / scale if scale > 0 else float('nan')


def spearman(a, b):
    """Spearman's rank correlation (no ties expected in continuous values)"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def summarise_attribute(errors, uncertainties, predictions=None, labels=None):
    """
    one attribute over the cases of a split
    :param errors: e of every case; :param uncertainties: U of every case on the same scale (fractions, not %)
    """
    e, u = np.abs(np.asarray(errors, float)), np.asarray(uncertainties, float)
    valid = ~np.isnan(e)
    summary = {'cases': int(valid.sum()), 'median_abs_error': float(np.median(e[valid])),
               'p90_abs_error': float(np.percentile(e[valid], 90)),
               'within_uncertainty_rate': float(np.mean(e[valid] <= u[valid]))}
    if predictions is not None:
        summary['spearman'] = spearman(np.asarray(predictions)[valid], np.asarray(labels)[valid])
    return summary


def summarise_fields(rows):
    """median and mean of every field error over the cases of a split (rows: field_errors of every case)"""
    names = rows[0].keys()
    return {name: {'median': float(np.nanmedian([r[name] for r in rows])),
                   'mean': float(np.nanmean([r[name] for r in rows]))} for name in names}
