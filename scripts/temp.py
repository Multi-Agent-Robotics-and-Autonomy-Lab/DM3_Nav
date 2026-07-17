import skfmm
import numpy as np
from numpy import ma
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap




# # Define traversible map
# traversible = np.ones((30, 30), dtype=np.int8)
# traversible[10:20, 10:20] = 0 
# traversible[5:15, 0:10] = 0
# traversible[0:10, 5:15] = 0
# # traversible[19, 19] = 1
# # traversible[18, 18] = 1

# # traversible[2:10, 12:18] = 0 

# # Define goal
# goal_map = np.zeros((30, 30), dtype=np.int8)
# goal_map[29, 29] = 1

traversible = np.ones((3, 4), dtype=np.int8)
# traversible[1,1] = 0
# traversible[0,1] = 0
# traversible[1,0] = 0
goal_map = np.zeros((3, 4), dtype=np.int8)
goal_map[2,3] = 1


traversible_ma = ma.masked_values(traversible, 0)
traversible_ma[goal_map == 1] = 0

dd = skfmm.distance(traversible_ma, dx=1.0)
print(dd)
# print("traversible ma: ", traversible_ma)
# print("Distance field ", dd)
# print("Distance field type", type(dd))
# print("Distance field 0,0", dd[0,0])
# print("Distance field 0,0 type", type(dd[0,0]))
# dd = ma.filled(dd, np.max(dd) + 1)  # Fill masked values with large number
# print("Distance field ", dd)
# print("Distance field 0,0", dd[0,0])


# goal_mask = goal_map.astype(bool)
# goal_value = -1  # Set to a unique value not present elsewhere
# dd[goal_mask] = goal_value

# # Create a custom colormap
# base_cmap = plt.get_cmap('viridis', 256)
# newcolors = base_cmap(np.linspace(0, 1, 256))
# goal_color = np.array([1.0, 0.0, 0.0, 1.0])  # RGBA red
# newcolors = np.vstack((goal_color, newcolors))  # Insert red at the front
# new_cmap = ListedColormap(newcolors)

# # Normalize so -1 maps to index 0, rest map correctly
# from matplotlib import colors
# norm = colors.Normalize(vmin=-1, vmax=np.max(dd))

# # Plot
# plt.figure(figsize=(8, 8))
# im = plt.imshow(dd, cmap=new_cmap, norm=norm, origin='lower')
# plt.colorbar(im, label='Distance to Goal')

# plt.imshow(traversible == 0, cmap=ListedColormap([[0, 0, 0, 0], [0, 0, 0, 1]]), origin='lower')
# # plt.contour(traversible == 0, levels=[0.5], colors='black', linewidths=1)
# plt.title('Fast Marching Distance Field with Goal in Red')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.grid(False)
# plt.tight_layout()
# plt.show()