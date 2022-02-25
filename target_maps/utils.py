import numpy as np


def full_map_from_lower_triangular(map_2d: np.ndarray) -> np.ndarray:
    final_map = np.zeros_like(map_2d)
    grid_len = map_2d.shape[0]
    for i in range(grid_len):
        for j in range(grid_len):
            if j <= i:
                final_map[i, j] = map_2d[i, j]
            else:
                final_map[i, j] = map_2d[j, i]
    return final_map
