
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


def read_wake_profile(filename, U0=12.0, D=126.0, zMid=250.0):

    data = pv.read(filename)

    # Grid dimensions: (1, ny, nz)
    nx, ny, nz = data.dimensions

    # Coordinates
    points = data.points

    # Reshape using the actual VTK dimensions
    Y = points[:, 1].reshape(ny, nz)
    Z = points[:, 2].reshape(ny, nz)

    # Velocity vector
    Velocity = data.point_data["Velocity"]
    Vx = Velocity[:, 0].reshape(ny, nz)

    # Find the vertical index closest to hub height
    iz = np.argmin(np.abs(Z[0, :] - zMid))

    # Extract horizontal profile
    y = Y[:, iz] / D
    U = Vx[:, iz] / U0

    return y, U


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

files = {
    "1D": "tasks/task2_uniform_aligned/vtk_ff/input_cart_FD.Low.DisYZ001.00.vtk",
    # Add the 5D and 8D filenames once confirmed
}

fig, ax = plt.subplots(figsize=(7, 6))

for label, filename in files.items():

    y, U = read_wake_profile(filename)

    ax.plot(U, y, linewidth=2, label=label)

ax.set_xlabel(r"$U_x/U_\infty$")
ax.set_ylabel(r"$y/D$")
ax.set_title("Horizontal wake velocity profile")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
