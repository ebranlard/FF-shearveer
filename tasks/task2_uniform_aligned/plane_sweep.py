
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob
import os
import re

# ------------------------------------------------------------
# User inputs
# ------------------------------------------------------------

folder = '.'
filebase = 'input_cart_FD'

plane = 'XY'       # 'XY', 'XZ' or 'YZ'
nout = 1
U0 = 12.0

skip = 4
interval = 300

# ------------------------------------------------------------
# Find files for selected plane and nout
# ------------------------------------------------------------

pattern = os.path.join(
    folder,
    'tasks/task2_uniform_aligned/vtk_ff',
    f'{filebase}.Low.Dis{plane}{nout:03d}.*.vtk'
)

files = glob.glob(pattern)

if not files:
    raise FileNotFoundError(f'No files found: {pattern}')

def get_nlow(filename):
    match = re.search(r'\.(\d+)\.vtk$', filename)
    if match is None:
        raise ValueError(f'Cannot extract nlow from {filename}')
    return int(match.group(1))

files.sort(key=get_nlow)

print(f'Found {len(files)} frames for {plane} plane, nout={nout}')

# ------------------------------------------------------------
# Read all velocity fields
# ------------------------------------------------------------

frames = []

for filename in files:

    with open(filename, 'r') as f:
        lines = f.readlines()

    nx, ny, nz = map(int, lines[4].split()[1:])

    origin = np.array(
        [float(v) for v in lines[5].split()[1:]]
    )

    spacing = np.array(
        [float(v) for v in lines[6].split()[1:]]
    )

    data_start = next(
        i for i, line in enumerate(lines)
        if line.startswith('VECTORS Velocity')
    ) + 1

    velocity = np.loadtxt(lines[data_start:])
    velocity = velocity.reshape((nz, ny, nx, 3))

    x = origin[0] + np.arange(nx) * spacing[0]
    y = origin[1] + np.arange(ny) * spacing[1]
    z = origin[2] + np.arange(nz) * spacing[2]

    # --------------------------------------------------------
    # XY plane
    # --------------------------------------------------------

    if plane == 'XY':

        if nz != 1:
            raise ValueError(f'Expected nz=1 for XY, got {nz}')

        X, Y = np.meshgrid(x, y)

        U = velocity[0, :, :, 0] / U0
        V = velocity[0, :, :, 1] / U0

        xlabel = 'X [m]'
        ylabel = 'Y [m]'
        velocity_label = r'$U_x/U_\infty$'

    # --------------------------------------------------------
    # XZ plane
    # --------------------------------------------------------

    elif plane == 'XZ':

        if ny != 1:
            raise ValueError(f'Expected ny=1 for XZ, got {ny}')

        X, Z = np.meshgrid(x, z)

        U = velocity[:, 0, :, 0] / U0
        V = velocity[:, 0, :, 2] / U0

        X, Y = X, Z

        xlabel = 'X [m]'
        ylabel = 'Z [m]'
        velocity_label = r'$U_x/U_\infty$'

    # --------------------------------------------------------
    # YZ plane
    # --------------------------------------------------------

    elif plane == 'YZ':

        if nx != 1:
            raise ValueError(f'Expected nx=1 for YZ, got {nx}')

        Y, Z = np.meshgrid(y, z)

        U = velocity[:, :, 0, 1] / U0
        V = velocity[:, :, 0, 2] / U0

        X, Y = Y, Z

        xlabel = 'Y [m]'
        ylabel = 'Z [m]'
        velocity_label = r'$U_y/U_\infty$'

    else:
        raise ValueError("plane must be 'XY', 'XZ' or 'YZ'")

    # --------------------------------------------------------
    # Time information
    # --------------------------------------------------------

    nlow = get_nlow(filename)

    time_match = re.search(r'time\s*=\s*([\d.]+)', lines[1])
    time = float(time_match.group(1)) if time_match else np.nan

    frames.append({
        'nlow': nlow,
        'time': time,
        'X': X,
        'Y': Y,
        'U': U,
        'V': V
    })

    print(
        f'nlow={nlow:04d}, time={time:8.2f} s, '
        f'U range=[{U.min():.3f}, {U.max():.3f}]'
    )

# ------------------------------------------------------------
# Common color scale
# ------------------------------------------------------------

vmin = min(frame['U'].min() for frame in frames)
vmax = max(frame['U'].max() for frame in frames)

if np.isclose(vmin, vmax):
    vmin -= 1e-6
    vmax += 1e-6

levels = np.linspace(vmin, vmax, 40)

# ------------------------------------------------------------
# Initial plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 6))

frame = frames[0]

contour = ax.contourf(
    frame['X'],
    frame['Y'],
    frame['U'],
    levels=levels,
    cmap='viridis',
    extend='both'
)

cbar = fig.colorbar(contour, ax=ax)
cbar.set_label(velocity_label)

quiver = ax.quiver(
    frame['X'][::skip, ::skip],
    frame['Y'][::skip, ::skip],
    frame['U'][::skip, ::skip],
    frame['V'][::skip, ::skip],
    color='white',
    scale=30
)

title = ax.set_title('')

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_aspect('equal', adjustable='box')

# ------------------------------------------------------------
# Animation update
# ------------------------------------------------------------

def update(i):

    global contour

    frame = frames[i]

    contour.remove()

    contour = ax.contourf(
        frame['X'],
        frame['Y'],
        frame['U'],
        levels=levels,
        cmap='viridis',
        extend='both'
    )

    quiver.set_UVC(
        frame['U'][::skip, ::skip],
        frame['V'][::skip, ::skip]
    )

    title.set_text(
        f'{filebase} | {plane} | nout={nout} | '
        f'nlow={frame["nlow"]} | t={frame["time"]:.2f} s'
    )

    return contour.collections + [quiver, title]

# ------------------------------------------------------------
# Run animation
# ------------------------------------------------------------

ani = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=interval,
    blit=False,
    repeat=True
)

plt.tight_layout()
plt.show()
