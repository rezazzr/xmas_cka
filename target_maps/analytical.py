import numpy as np

ALL_ONES = np.ones([8, 8])

ALL_ZEROS = np.zeros([8, 8])
# make diagonals 1
for i in range(8):
    ALL_ZEROS[i, i] = 1
