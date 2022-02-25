import numpy as np

from target_maps.utils import full_map_from_lower_triangular

GOBLIN = np.zeros([8, 8]) + 1
for i in range(8):
    for j in range(8):
        if i == j:
            GOBLIN[j, i] = 1

# eyebrows
GOBLIN[3, 0] = 0.5
GOBLIN[3, 1] = 0.5
GOBLIN[3, 2] = 0.5
GOBLIN[4, 0] = 0.5
GOBLIN[5, 0] = 0.5

# eyes
GOBLIN[5, 2] = 0.7
GOBLIN[5, 3] = 0.7
GOBLIN[6, 2] = 0.7
GOBLIN[6, 3] = 0.7

# non_face
GOBLIN[1, 0] = 0.3
GOBLIN[2, 0] = 0.3
GOBLIN[6, 0] = 0.3

GOBLIN[7, 0] = 0.3
GOBLIN[7, 1] = 0.3
GOBLIN[6, 5] = 0.3
GOBLIN[7, 5] = 0.3


GOBLIN = full_map_from_lower_triangular(GOBLIN)
