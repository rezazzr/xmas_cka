import numpy as np

from target_maps.utils import full_map_from_lower_triangular

GOBLIN = np.zeros([8, 8]) + 1
for i in range(8):
    for j in range(8):
        if i == j:
            GOBLIN[j, i] = 1

# eyebrows
GOBLIN[3, 0] = 0.3
GOBLIN[3, 1] = 0.3
GOBLIN[3, 2] = 0.3
GOBLIN[4, 0] = 0.3
GOBLIN[5, 0] = 0.3

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

SWORD = np.zeros([8, 8]) + 0.3
for i in range(8):
    for j in range(8):
        if i == j:
            SWORD[j, i] = 1

# side sword:
SWORD[1, 0] = 0.9
SWORD[2, 1] = 0.9
SWORD[3, 2] = 0.9
SWORD[4, 3] = 0.9
# handle
SWORD[5, 3] = 0.4
SWORD[5, 4] = 0.4
SWORD[6, 2] = 0.4
SWORD[6, 3] = 0.4
# base
SWORD[7, 6] = 0.8

SWORD = full_map_from_lower_triangular(SWORD)

BOW_ARROW = np.zeros([8, 8]) + 0.3
for i in range(8):
    for j in range(8):
        if i == j:
            BOW_ARROW[j, i] = 1

# arrow tip:
BOW_ARROW[1, 0] = 0.7
BOW_ARROW[2, 0] = 0.6
# bow
BOW_ARROW[7, 1] = 0.53
BOW_ARROW[6, 1] = 0.53
BOW_ARROW[7, 0] = 0.7
BOW_ARROW[5, 1] = 0.47
BOW_ARROW[4, 1] = 0.63
BOW_ARROW[3, 2] = 0.63
BOW_ARROW[4, 2] = 0.47
BOW_ARROW[7, 2] = 0.57
BOW_ARROW[7, 3] = 0.41
BOW_ARROW[6, 3] = 0.38
BOW_ARROW[6, 4] = 0.7
BOW_ARROW[5, 4] = 0.54

BOW_ARROW = full_map_from_lower_triangular(BOW_ARROW)

XMASS_TREE = np.zeros([8, 8]) + 1
for i in range(8):
    for j in range(8):
        if i == j:
            XMASS_TREE[j, i] = 1

# ornaments
XMASS_TREE[5, 2] = 0.9
XMASS_TREE[7, 4] = 0.9

# non-tree
for i in range(1, 8):
    XMASS_TREE[i, 0] = 0.3

for i in range(5, 8):
    XMASS_TREE[i, 1] = 0.3

for i in range(6, 8):
    XMASS_TREE[i, 2] = 0.3

XMASS_TREE[4, 2] = 0.3
XMASS_TREE[6, 5] = 0.3
XMASS_TREE[7, 5] = 0.3
XMASS_TREE[7, 6] = 0.3


XMASS_TREE = full_map_from_lower_triangular(XMASS_TREE)

CARROT = np.zeros([8, 8]) + 0.3
for i in range(8):
    CARROT[i, i] = 1

# side carrot
for i in range(4, 8):
    CARROT[i, i - 1] = 0.9

for i in range(5, 8):
    CARROT[i, i - 2] = 0.9

CARROT[5, 2] = 0.9
CARROT[6, 3] = 0.9

# green bits
CARROT[2, 0] = 0.7
CARROT[2, 1] = 0.7
CARROT[3, 1] = 0.7


CARROT = full_map_from_lower_triangular(CARROT)
