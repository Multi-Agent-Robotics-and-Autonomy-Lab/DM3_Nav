import open3d as o3d
import numpy as np
print(o3d.__version__)

points = (np.random.rand(20000, 3) - 0.5) * 10.0

# Convert numpy array to CPU Vector3dVector first
cpu_vec = o3d.utility.Vector3dVector(points)

# Then move it to CUDA vector
cuda_vec = o3d.cuda.pybind.utility.Vector3dVector(cpu_vec)

pcd_cuda = o3d.cuda.pybind.geometry.PointCloud()
pcd_cuda.points = cuda_vec

print(pcd_cuda)
print(type(pcd_cuda))
geoms = [pcd_cuda]
print("Points attribute type:", type(geoms[0].points))
print("Number of points:", len(geoms[0].points))
points_np = np.asarray(geoms[0].points)
print("Points shape:", points_np.shape)
print("Any NaNs?", np.isnan(points_np).any())
print("Any Infs?", np.isinf(points_np).any())
print("Has colors?", pcd_cuda.has_colors())
print("Has normals?", pcd_cuda.has_normals())

if pcd_cuda.has_points():
    print("First 5 points:\n", np.asarray(pcd_cuda.points)[:5])

if pcd_cuda.has_colors():
    print("Colors type:", type(pcd_cuda.colors))
    print("Number of colors:", len(pcd_cuda.colors))
    print("First 5 colors:\n", np.asarray(pcd_cuda.colors)[:5])
else:
    print("Point cloud has no colors")

pcd_cuda = o3d.cuda.pybind.io.read_point_cloud("debug.ply")
print(pcd_cuda)
print(type(pcd_cuda))
o3d.visualization.draw_geometries([pcd_cuda])
