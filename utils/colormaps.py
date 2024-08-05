import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np

# Arrange viridis with black background
original_map = plt.cm.get_cmap('viridis')
color_mat = original_map(np.arange(original_map.N))
color_mat[0, 0:3] = 0  # minimum value is set to black
b_viridis = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)

# Arrange winter with black background
original_map = plt.cm.get_cmap('winter')
color_mat = original_map(np.arange(original_map.N))
color_mat[0, 0:3] = 0  # minimum value is set to black
b_winter = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)
