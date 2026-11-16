
import os
import glob
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


# =============================================================================
# User inputs
# =============================================================================

folder = 'tasks/task2_uniform_aligned'
U0 = 12.0
D = 126.0
zMid = 250.0

DPlanes = [1, 5, 8]

filebases = {
    'Cart_FE': 'input_cart_FE',
}

# None = automatically use the latest available time index
iTime = 40


# =============================================================================
# Plot settings
# =============================================================================

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 14

colors = {
    'Cart_FE': np.array([63, 63, 153]) / 255,
    'Cart_FD': np.array([218, 124, 48]) / 255,
}


# =============================================================================
# VTK file handling
# =============================================================================

def get_time_indices(folder, filebase):

    pattern = os.path.join(
        folder, 'vtk_ff',
        filebase + '.Low.DisYZ*.vtk'
    )

    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f'No VTK files found matching: {pattern}'
        )

    indices = []

    for filename in files:
        # Example: input_cart_FE.Low.DisYZ001.03.vtk
        time_index = os.path.splitext(
            os.path.splitext(filename)[0]
        )[1][1:]

        indices.append(int(time_index))

    indices = np.sort(indices)

    return indices


def extract_plane(folder, filebase, iPlane, iTime, U0, D):

    filename = os.path.join(
        folder,
        'vtk_ff',
        f'{filebase}.Low.DisYZ{iPlane:03d}.{iTime:02d}.vtk'
    )

    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f'VTK file not found: {filename}'
        )

    data = pv.read(filename)

    # Coordinates and velocity vectors
    points = data.points
    velocity = data.point_data['Velocity']

    # FAST.Farm YZ plane coordinates
    y = np.unique(points[:, 1]) / D
    z = np.unique(points[:, 2]) / D

    # Reconstruct the plane using the actual point coordinates.
    # This avoids assuming a particular VTK array orientation.
    Y, Z = np.meshgrid(y, z, indexing='ij')

    Ux = np.full(Y.shape, np.nan)

    for point, vel in zip(points, velocity):

        iy = np.argmin(np.abs(y - point[1] / D))
        iz = np.argmin(np.abs(z - point[2] / D))

        Ux[iy, iz] = vel[0] / U0

    return Y, Z, Ux


# =============================================================================
# Extract horizontal and vertical profiles
# =============================================================================

def extract_lines(Y, Z, Ux, y0=0.0, z0=None):

    if z0 is None:
        raise ValueError('Specify z0.')

    # Equivalent to the original extractLines function
    id_y = np.argmin(np.abs(Y[:, 0] - y0))
    id_z = np.argmin(np.abs(Z[0, :] - z0))

    y = Y[:, 0]
    z = Z[0, :]

    Uh = Ux[:, id_z]
    Uv = Ux[id_y, :]

    return y, Uh, z, Uv


# =============================================================================
# Store profiles
# =============================================================================

def store_lines():

    profiles = {}

    for label, filebase in filebases.items():

        indices = get_time_indices(folder, filebase)

        current_time = indices[-1] if iTime is None else iTime

        print(f'\n{label}: time index = {current_time}')

        profiles[label] = {}

        for iPlane, Dplane in enumerate(DPlanes):

            Y, Z, Ux = extract_plane(
                folder=folder,
                filebase=filebase,
                iPlane=iPlane + 1,
                iTime=current_time,
                U0=U0,
                D=D,
            )

            y, Uh, z, Uv = extract_lines(
                Y, Z, Ux,
                y0=0.0,
                z0=zMid / D,
            )

            profiles[label][Dplane] = {
                'y': y,
                'z': z,
                'Ux': Ux,
                'Uh': Uh,
                'Uv': Uv,
            }

            print(
                f'  {Dplane}D: '
                f'Uh range = [{np.nanmin(Uh):.4f}, '
                f'{np.nanmax(Uh):.4f}]'
            )

    return profiles


# =============================================================================
# Plot wake deficit
# =============================================================================

def plot_wake_def(ax, y, profiles, Dplane):

    for label in filebases:

        Uh = profiles[label][Dplane]['Uh']

        ax.plot(
            Uh,
            y,
            '-',
            color=colors[label],
            label=label,
        )

    ax.set_ylim([-1.5, 1.5])
    ax.set_xlim([0.45, 1.1])

    ax.set_xlabel(r'$u/U_\infty$')
    ax.set_title(f'{Dplane}D')

    ax.grid(
        True,
        linestyle=':',
        color=(0.5, 0.5, 0.5),
    )

    ax.tick_params(
        direction='in',
        top=True,
        right=True,
        which='both',
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':

    profiles = store_lines()

    fig, axes = plt.subplots(
        1, 3,
        sharey=True,
        figsize=(12.8, 3.8),
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.930,
        bottom=0.133,
        hspace=0.20,
        wspace=0.14,
    )

    for iPlane, Dplane in enumerate(DPlanes):

        ax = axes[iPlane]

        # Both models share the same lateral coordinates
        y = profiles[list(filebases.keys())[0]][Dplane]['y']

        plot_wake_def(
            ax,
            y,
            profiles,
            Dplane,
        )

        if iPlane == 0:
            ax.set_ylabel(r'$y/D$')
            ax.legend(loc='upper left')

    fig._title = 'Task2Deficit'

    plt.show()
