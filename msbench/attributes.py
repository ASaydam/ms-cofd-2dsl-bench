"""
Flow attributes computed from the arrays of a sample, the same function as `tools/flow_attributes.py` of the
MS-COFD-2DSL pipeline (copied unchanged, so a prediction and its label are measured the same way):

    recirculation_length_mm   furthest reverse flow (u_x < 0) connected to the obstacle behind it, from its
                              downstream end (a separate thin reverse-flow layer along a channel wall does not count)
    recirculation_area_mm2    fluid area with u_x < 0 connected to the obstacle
    flow_split_upper          share of the flow passing above the obstacle (at its centroid)
    dp_inlet_outlet           mean pressure on the inlet minus the outlet column
    dissipation               nu * integral of |grad u|^2 over the fluid (first 5 mm left out: inlet corners)
    peak_vorticity            largest |vorticity| (not scored: singular at corners)
    mean_vorticity            |vorticity| averaged over the fluid area, first 5 mm left out
    wake_min_u_x              smallest u_x 20 mm behind the obstacle, within its y extent (wake velocity deficit)
"""

import numpy as np

NU = 1e-4                                    # m^2/s
DOMAIN = {'length': 360, 'height': 120}      # mm
ARRAY_ATTRIBUTES = ['recirculation_length_mm', 'recirculation_area_mm2', 'flow_split_upper', 'dp_inlet_outlet',
                    'dissipation', 'peak_vorticity', 'mean_vorticity', 'wake_min_u_x']


def spacing(nx, ny, domain=DOMAIN):
    """grid spacing in mm (the grid points include both ends of the domain)"""
    return domain['length'] / (nx - 1), domain['height'] / (ny - 1)


def attributes(sample, domain=DOMAIN):
    """flow attributes of one (6, nx, ny) sample, domain in mm (length, height)"""
    from scipy import ndimage

    ux, uy, p, cell_type = sample[0], sample[1], sample[2], sample[4]
    nx, ny = ux.shape
    hx, hy = spacing(nx, ny, domain)
    x = np.arange(nx)[:, None] * hx * np.ones((1, ny))
    y = np.ones((nx, 1)) * np.arange(ny)[None, :] * hy

    obstacle = np.zeros(ux.shape, bool)
    obstacle[1:-1, 1:-1] = cell_type[1:-1, 1:-1] == 1
    fluid = cell_type != 1
    x_end = x[obstacle].max()

    labels, _ = ndimage.label((ux < 0) & fluid)
    touching = labels[ndimage.binary_dilation(obstacle) & (labels > 0)]
    reverse = np.isin(labels, touching)
    behind = reverse & (x > x_end)
    result = {'recirculation_length_mm': float(x[behind].max() - x_end) if behind.any() else 0.0,
              'recirculation_area_mm2': float(reverse.sum() * hx * hy)}

    # share of the flow above the obstacle, in the column of its centroid
    i = int(round(x[obstacle].mean() / hx))
    top = y[i][obstacle[i]].max()
    column = fluid[i]
    result['flow_split_upper'] = float(ux[i][column & (y[i] > top)].sum() / ux[i][column].sum())
    result['dp_inlet_outlet'] = float(p[0, 1:-1].mean() - p[-1, 1:-1].mean())

    # gradients in 1/s (spacing in m), without the inlet corners
    dx, dy = hx / 1000, hy / 1000
    gradients = [np.gradient(u, dx, axis=0) for u in (ux, uy)] + [np.gradient(u, dy, axis=1) for u in (ux, uy)]
    region = fluid & (x > 5)
    result['dissipation'] = float(NU * sum((g[region] ** 2).sum() for g in gradients) * dx * dy)
    vorticity = gradients[1] - gradients[2]  # d(u_y)/dx - d(u_x)/dy
    result['peak_vorticity'] = float(np.abs(vorticity[region]).max())
    result['mean_vorticity'] = float(np.abs(vorticity[region]).mean())

    i_wake = min(int(round((x_end + 20) / hx)), nx - 1)
    behind_obstacle = (y[i_wake] >= y[obstacle].min()) & (y[i_wake] <= y[obstacle].max())
    result['wake_min_u_x'] = float(ux[i_wake][fluid[i_wake] & behind_obstacle].min())
    return result
