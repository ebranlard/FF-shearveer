
import os
import re
import numpy as np
import matplotlib.pyplot as plt

from wake_cart_fd_lib import gradient_y, gradient_z


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

filename = os.path.join("debug", "continuity_debug.txt")
output_dir = os.path.join("debug", "continuity_results")
os.makedirs(output_dir, exist_ok=True)


# ------------------------------------------------------------
# Read Fortran debug file
# ------------------------------------------------------------

def read_debug_file(filename):

    with open(filename, "r") as f:
        lines = f.readlines()

    # Grid dimensions
    ny, nz = map(int, re.search(
        r"Ny, Nz\s*=\s*(\d+)\s+(\d+)", "".join(lines)
    ).groups())

    # Grid spacing
    dx, dy, dz = map(float, re.search(
        r"dx, dy, dz\s*=\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)",
        "".join(lines)
    ).groups())

    def read_field(name):

        start = next(
            i for i, line in enumerate(lines)
            if f"--- {name} ---" in line
        )

        field = np.zeros((ny, nz))

        for line in lines[start + 1:]:

            if "---" in line:
                break

            parts = line.split()

            if len(parts) == 3:
                iy, iz, value = parts
                field[int(iy)-1, int(iz)-1] = float(value)

        return field

    Vx_old = read_field("Vx_old")
    Vx_new = read_field("Vx_new")
    Vy = read_field("Vy")
    Vz = read_field("Vz")

    return Vx_old, Vx_new, Vy, Vz, dx, dy, dz


# ------------------------------------------------------------
# Load fields
# ------------------------------------------------------------

Vx_old, Vx_new, Vy, Vz, dx, dy, dz = read_debug_file(filename)

ny, nz = Vx_old.shape

print(f"Grid: {ny} x {nz}")
print(f"dx = {dx:.6f}, dy = {dy:.6f}, dz = {dz:.6f}")


# ------------------------------------------------------------
# Continuity equation
# ------------------------------------------------------------

# Source term: -dVx/dx
source = -(Vx_new - Vx_old) / dx

# Transverse divergence
dVy_dy = gradient_y(Vy, dy)
dVz_dz = gradient_z(Vz, dz)

divergence = dVy_dy + dVz_dz

# Continuity residual
residual = divergence - source


# ------------------------------------------------------------
# Error metrics
# ------------------------------------------------------------

source_rms = np.sqrt(np.mean(source**2))
divergence_rms = np.sqrt(np.mean(divergence**2))

rmse = np.sqrt(np.mean(residual**2))
max_error = np.max(np.abs(residual))
mean_error = np.mean(np.abs(residual))

relative_rmse = rmse / source_rms if source_rms > 0 else np.nan

stats = [
    "CONTINUITY CHECK",
    "----------------",
    f"Grid: {ny} x {nz}",
    f"dx = {dx:.8e}",
    f"dy = {dy:.8e}",
    f"dz = {dz:.8e}",
    "",
    f"Source RMS:        {source_rms:.6e}",
    f"Divergence RMS:    {divergence_rms:.6e}",
    f"Residual RMSE:     {rmse:.6e}",
    f"Mean abs error:    {mean_error:.6e}",
    f"Maximum abs error: {max_error:.6e}",
    f"Relative RMSE:     {relative_rmse:.6e}",
]

print("\n" + "\n".join(stats))

with open(os.path.join(output_dir, "continuity_stats.txt"), "w") as f:
    f.write("\n".join(stats) + "\n")


# ------------------------------------------------------------
# Continuity field plots
# ------------------------------------------------------------

y = (np.arange(ny) - (ny - 1)/2) * dy
z = (np.arange(nz) - (nz - 1)/2) * dz

extent = [z[0], z[-1], y[0], y[-1]]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plots = [
    (source, r"Continuity source $-\partial V_x/\partial x$"),
    (divergence, r"Transverse divergence $\partial V_y/\partial y+\partial V_z/\partial z$"),
    (residual, "Continuity residual"),
    (Vx_new, r"$V_x$ new"),
]

for ax, (field, title) in zip(axes.flat, plots):

    im = ax.imshow(
        field,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="RdBu_r"
    )

    ax.set_title(title)
    ax.set_xlabel("z")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "continuity_fields.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ------------------------------------------------------------
# Transverse velocity vector field
# ------------------------------------------------------------

Y, Z = np.meshgrid(y, z, indexing="ij")

fig, ax = plt.subplots(figsize=(8, 7))

ax.quiver(Y, Z, Vy, Vz)

ax.set_xlabel("y")
ax.set_ylabel("z")
ax.set_title("Transverse velocity field")
ax.set_aspect("equal")

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "transverse_velocity.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ------------------------------------------------------------
# Save complete
# ------------------------------------------------------------

print(f"\nAll results saved in: {output_dir}")
