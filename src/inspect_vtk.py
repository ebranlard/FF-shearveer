
import pyvista as pv

filename = (
    "tasks/task2_uniform_aligned/vtk_ff/input_cart_FD.Low.DisYZ001.00.vtk"
)

data = pv.read(filename)

print("Dataset:")
print(data)

print("\nDimensions:")
print(data.dimensions)

print("\nNumber of points:")
print(data.n_points)

print("\nPoint data:")
for name in data.point_data:
    print(name, data.point_data[name].shape)

print("\nVelocity components:")

Velocity = data.point_data["Velocity"]

Vx = Velocity[:, 0]
Vy = Velocity[:, 1]
Vz = Velocity[:, 2]

print("Vx:", Vx[:5])
print("Vy:", Vy[:5])
print("Vz:", Vz[:5])
