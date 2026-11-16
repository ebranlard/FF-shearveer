"""
Compare the Python Cartesian FD wake calculation against the Fortran debug dump.

Put this file beside:
    helper_functions.py
    debug_2_plane.rtf

Run:
    python compare_wake_python_fortran.py

The script reconstructs the input state for each dumped plane, calls the
existing helper_functions.solve_Vx(), and compares its result with the
Fortran OUTPUT Vx_wake2 field.
"""

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

import wake_cart_fd_lib as hf


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------
DUMP_FILE = Path("debug/debug_dump.txt")

# The dump contains six planes. Set to None to process all of them.
PLANES_TO_PLOT = None

# Background flow reported by the Fortran diagnostics (UU, VV, WW).
# These are the m['U'], m['V'], m['W'] fields used by build_A_matrix_plane_i.
U_BACKGROUND = 12.0
V_BACKGROUND = 0.0
W_BACKGROUND = 0.0

# IMPORTANT: the dump does not contain complete xd['Vy'] and xd['Vz']
# fields. The helper's Vx equation uses them, so this first comparison
# assumes both transverse wake-velocity fields are zero.
ASSUME_TRANSVERSE_WAKE_ZERO = True

# Save figures as PNG files in this directory.
SAVE_FIGURES = True
OUTPUT_DIR = Path(__file__).with_name("wake_comparison")


# ---------------------------------------------------------------------
# Read and minimally decode the RTF dump
# ---------------------------------------------------------------------
def read_rtf_as_text(path):
    raw = path.read_text(encoding="latin-1", errors="replace")

    # Convert RTF paragraph/line/tab markers before stripping formatting.
    raw = re.sub(r"\\par[d]?\s?", "\n", raw)
    raw = re.sub(r"\\line\s?", "\n", raw)
    raw = re.sub(r"\\tab\s?", "\t", raw)

    # Remove hex character escapes and RTF control words.
    raw = re.sub(r"\\'[0-9a-fA-F]{2}", " ", raw)
    raw = re.sub(r"\\[a-zA-Z]+-?\d*\s?", " ", raw)

    # Remove escaped literal characters and RTF group braces.
    raw = raw.replace(r"\{", "{").replace(r"\}", "}").replace(r"\\", "\\")
    raw = raw.replace("{", "").replace("}", "")

    return raw


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def fortran_float(value):
    return float(value.replace("D", "E").replace("d", "e"))


def extract_scalar(block, name):
    match = re.search(
        rf"^\s*{re.escape(name)}\s*=\s*({FLOAT_RE})",
        block,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Could not find '{name} =' in plane block.")
    return fortran_float(match.group(1))


def extract_field(block, field_name, ny, nz):
    """
    Extract a Fortran field section containing rows:
        iy  iz  value
    The Fortran indices run from -39 to +39.
    """
    marker = re.compile(
        rf"^\s*---\s*{re.escape(field_name)}\s*---\s*$",
        flags=re.MULTILINE | re.IGNORECASE,
    )
    match = marker.search(block)
    if not match:
        raise ValueError(f"Could not find section '{field_name}'.")

    # A field section ends at the next dashed section heading.
    next_heading = re.search(r"^\s*---.*---\s*$", block[match.end():],
                             flags=re.MULTILINE)
    end = match.end() + next_heading.start() if next_heading else len(block)
    section = block[match.end():end]

    field = np.full((ny, nz), np.nan, dtype=float)
    n_found = 0

    for line in section.splitlines():
        cols = line.split()
        if len(cols) < 3:
            continue

        try:
            iy = int(cols[0])
            iz = int(cols[1])
            value = fortran_float(cols[2])
        except (ValueError, IndexError):
            continue

        j = iy + (ny // 2)
        k = iz + (nz // 2)
        if 0 <= j < ny and 0 <= k < nz:
            field[j, k] = value
            n_found += 1

    if n_found != ny * nz or np.isnan(field).any():
        raise ValueError(
            f"Parsed {n_found} values from '{field_name}', "
            f"but expected {ny * nz}. Check the dump section format."
        )

    return field


def split_plane_blocks(text):
    headers = list(re.finditer(
        r"^\s*PLANE\s+i\s*=\s*(\d+)\s*$",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    ))
    if not headers:
        raise ValueError("No 'PLANE i =' sections found in the dump.")

    blocks = {}
    for n, header in enumerate(headers):
        start = header.start()
        end = headers[n + 1].start() if n + 1 < len(headers) else len(text)
        plane_number = int(header.group(1))
        blocks[plane_number] = text[start:end]

    return blocks


# ---------------------------------------------------------------------
# Reconstruct the inputs expected by the existing helper functions
# ---------------------------------------------------------------------
def reconstruct_plane(plane_number, block, nr, dr):
    # Read the plane-specific grid spacings from the Fortran dump.
    dx = extract_scalar(block, "dx")
    dy = extract_scalar(block, "dy")
    dz = extract_scalar(block, "dz")

    # One plane is enough for the helper's setup; nx only sizes p['x'].
    p, m, xd = hf.setup(nr=nr, nx=1, dr=dr)
    p["dx"] = dx
    p["dy"] = dy
    p["dz"] = dz

    # The coordinate vectors are centered at zero, as in the Fortran dump.
    p["y"] = np.linspace(-(nr - 1) * dy, (nr - 1) * dy, p["ny"])
    p["z"] = np.linspace(-(nr - 1) * dz, (nr - 1) * dz, p["nz"])

    # Input state at the beginning of this Fortran update.
    xd["Vx"] = extract_field(
        block, "INPUT FIELD: Vx_wake2", p["ny"], p["nz"]
    )
    m["vt_tot"] = extract_field(
        block, "INPUT FIELD: vt_tot2", p["ny"], p["nz"]
    )

    # Fortran point diagnostics show these background components as
    # spatially uniform: (UU, VV, WW) = (12, 0, 0).
    m["U"][:, :] = U_BACKGROUND
    m["V"][:, :] = V_BACKGROUND
    m["W"][:, :] = W_BACKGROUND

    # setup() initializes xd['Vy'] and xd['Vz'] to zero. The current dump
    # does not provide full fields for these transverse wake components.
    if not ASSUME_TRANSVERSE_WAKE_ZERO:
        raise RuntimeError(
            "Provide full transverse wake fields xd['Vy'] and xd['Vz'] "
            "before running this comparison."
        )
    xd["Vy"][:, :] = 0.0
    xd["Vz"][:, :] = 0.0

    # Reference solution after the Fortran update.
    vx_fortran = extract_field(
        block, "OUTPUT Vx_wake2", p["ny"], p["nz"]
    )

    return p, m, xd, vx_fortran


# ---------------------------------------------------------------------
# Run the existing Python solver and compare with Fortran
# ---------------------------------------------------------------------
def calculate_metrics(python_field, fortran_field):
    error = python_field - fortran_field
    return {
        "MAE": np.mean(np.abs(error)),
        "RMSE": np.sqrt(np.mean(error**2)),
        "Max abs error": np.max(np.abs(error)),
    }, error


def plot_comparison(plane_number, p, vx_python, vx_fortran, error, metrics):
    # Vx_wake2 is a velocity perturbation; plot positive wake deficit as -Vx.
    deficit_python = -vx_python
    deficit_fortran = -vx_fortran

    deficit_min = min(np.min(deficit_python), np.min(deficit_fortran))
    deficit_max = max(np.max(deficit_python), np.max(deficit_fortran))
    error_limit = max(np.max(np.abs(error)), 1.0e-12)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    panels = [
        (deficit_python, "Python wake deficit", "viridis",
         deficit_min, deficit_max),
        (deficit_fortran, "Fortran wake deficit", "viridis",
         deficit_min, deficit_max),
        (error, "Error: Python − Fortran (Vx)", "coolwarm",
         -error_limit, error_limit),
    ]

    for ax, (field, title, cmap, vmin, vmax) in zip(axes, panels):
        image = ax.imshow(
            field.T,
            origin="lower",
            extent=[p["y"][0], p["y"][-1], p["z"][0], p["z"][-1]],
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("y [m]")
        ax.set_ylabel("z [m]")
        fig.colorbar(image, ax=ax, label="Velocity [m/s]")

    fig.suptitle(
        f"Plane {plane_number} | "
        f"MAE={metrics['MAE']:.4e} m/s, "
        f"RMSE={metrics['RMSE']:.4e} m/s, "
        f"Max={metrics['Max abs error']:.4e} m/s"
    )

    if SAVE_FIGURES:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = OUTPUT_DIR / f"plane_{plane_number:02d}_comparison.png"
        fig.savefig(filename, dpi=200)
        print(f"  Saved: {filename}")

    plt.show()


def main():
    if not DUMP_FILE.exists():
        raise FileNotFoundError(f"Could not find dump file: {DUMP_FILE}")

    print(f"Reading: {DUMP_FILE}")
    text = read_rtf_as_text(DUMP_FILE)

    # Global grid information from the dump.
    nr_match = re.search(r"^\s*NumRadii\s*=\s*(\d+)",
                         text, flags=re.MULTILINE)
    dr_match = re.search(r"^\s*dr\s*=\s*(" + FLOAT_RE + r")",
                         text, flags=re.MULTILINE)
    if not nr_match or not dr_match:
        raise ValueError("Could not read NumRadii and dr from dump header.")

    nr = int(nr_match.group(1))
    dr = fortran_float(dr_match.group(1))
    ny = nz = 2 * nr - 1

    print(f"Grid: {ny} x {nz}, dr={dr:g} m")
    print("NOTE: assuming xd['Vy'] = xd['Vz'] = 0 because full fields")
    print("      for transverse wake velocity are not in this dump.")
    plane_blocks = split_plane_blocks(text)

    plane_numbers = sorted(plane_blocks.keys(), reverse=True)
    if PLANES_TO_PLOT is not None:
        plane_numbers = [i for i in plane_numbers if i in PLANES_TO_PLOT]

    print(f"Planes found: {sorted(plane_blocks.keys(), reverse=True)}")

    results = []

    for plane_number in plane_numbers:
        print(f"\n--- Plane {plane_number} ---")

        p, m, xd, vx_fortran = reconstruct_plane(
            plane_number, plane_blocks[plane_number], nr, dr
        )

        print(
            f"dx={p['dx']:.8f} m, dy={p['dy']:.8f} m, "
            f"dz={p['dz']:.8f} m"
        )

        # Use the user's existing implementation, including its matrix
        # assembly and conjugate-gradient solve.
        vx_python = hf.solve_Vx(plane_number, xd, m, p)

        metrics, error = calculate_metrics(vx_python, vx_fortran)
        results.append((plane_number, metrics))

        print(f"Python Vx range:  [{vx_python.min(): .6f}, "
              f"{vx_python.max(): .6f}] m/s")
        print(f"Fortran Vx range: [{vx_fortran.min(): .6f}, "
              f"{vx_fortran.max(): .6f}] m/s")
        print(f"MAE:             {metrics['MAE']:.6e} m/s")
        print(f"RMSE:            {metrics['RMSE']:.6e} m/s")
        print(f"Max abs error:   {metrics['Max abs error']:.6e} m/s")

        plot_comparison(
            plane_number, p, vx_python, vx_fortran, error, metrics
        )

    print("\nSummary")
    print("Plane       MAE [m/s]       RMSE [m/s]      Max error [m/s]")
    for plane_number, metrics in results:
        print(
            f"{plane_number:5d}  "
            f"{metrics['MAE']:14.6e}  "
            f"{metrics['RMSE']:14.6e}  "
            f"{metrics['Max abs error']:14.6e}"
        )


if __name__ == "__main__":
    main()
