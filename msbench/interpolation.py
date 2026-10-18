"""
Interpolation of a coarse level to a finer grid, the same functions as `tools/upsample_mse.py` of the MS-COFD-2DSL
pipeline (copied unchanged). Both grids span the whole channel with points on its edges ("align corners"):

    nearest          value of the nearest coarse point
    bilinear         linear in x and y (4 coarse points)
    bicubic          cubic convolution in x and y (16 coarse points, Keys 1981 with a = -0.5)
    bicubic_filled   bicubic after the obstacle points of the coarse grid got the value of the nearest fluid point
                     (the arrays hold 0 inside the obstacle, which the other methods mix into the fluid next to it)
"""

import numpy as np

METHODS = ['nearest', 'bilinear', 'bicubic', 'bicubic_filled']


def keys_kernel(s, a=-0.5):
    """cubic convolution kernel (Keys 1981)"""
    s = np.abs(s)
    return np.where(s <= 1, (a + 2) * s ** 3 - (a + 3) * s ** 2 + 1,
                    np.where(s < 2, a * s ** 3 - 5 * a * s ** 2 + 8 * a * s - 4 * a, 0.0))


def weights(n_coarse, n_fine, method):
    """
    (n_fine, n_coarse) matrix that interpolates a line of coarse points to the fine points, both lines with points
    on their two ends; the edge values are repeated beyond the ends (bicubic)
    """
    t = np.arange(n_fine) * (n_coarse - 1) / (n_fine - 1)
    rows = np.arange(n_fine)
    w = np.zeros((n_fine, n_coarse))
    if method == 'nearest':
        w[rows, np.floor(t + 0.5).astype(int)] = 1
    elif method == 'bilinear':
        i0 = np.minimum(np.floor(t).astype(int), n_coarse - 2)
        w[rows, i0] = 1 - (t - i0)
        w[rows, i0 + 1] = t - i0
    elif method.startswith('bicubic'):
        i0 = np.floor(t).astype(int)
        for k in (-1, 0, 1, 2):
            np.add.at(w, (rows, np.clip(i0 + k, 0, n_coarse - 1)), keys_kernel(t - (i0 + k)))
    else:
        raise ValueError(f"unknown method {method}")
    return w


def obstacle_mask(cell_type):
    """obstacle points: wall type (1) except the two channel walls (first and last row)"""
    mask = cell_type == 1
    mask[:, 0] = mask[:, -1] = False
    return mask


def fill_obstacle(field, cell_type):
    """the obstacle points get the value of the nearest fluid point"""
    from scipy.ndimage import distance_transform_edt

    obstacle = obstacle_mask(cell_type)
    if not obstacle.any():
        return field
    nearest = distance_transform_edt(obstacle, return_distances=False, return_indices=True)
    return field[tuple(nearest)]


def upsample(field, nx, ny, method, cell_type=None):
    """coarse field (bx, by) at the fine grid points (nx, ny)"""
    if method == 'bicubic_filled':
        field = fill_obstacle(field, cell_type)
    bx, by = field.shape
    return weights(bx, nx, method) @ field @ weights(by, ny, method).T


def upsample_sample(sample, nx, ny, method):
    """u_x, u_y, p of a coarse (6, bx, by) sample on the fine grid: (3, nx, ny)"""
    return np.stack([upsample(sample[c], nx, ny, method, sample[4]) for c in range(3)])
