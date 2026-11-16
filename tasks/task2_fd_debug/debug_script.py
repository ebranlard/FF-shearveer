import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix


FILE = "debug/debug_dump.txt"

N = 6241
NY = 79
NZ = 79

# Grid indices run from -39 to 39
IY_MIN = -39
IZ_MIN = -39


# --------------------------------------------------
# SPLIT DEBUG FILE INTO CALLS
# --------------------------------------------------

def read_calls(path):
    calls = []
    current_call = None

    with open(path, "r") as f:
        for line in f:
            match = re.search(r"DEBUG CALL NUMBER:\s*(\d+)", line)

            if match:
                if current_call is not None:
                    calls.append(current_call)

                current_call = {
                    "number": int(match.group(1)),
                    "lines": []
                }

            elif current_call is not None:
                current_call["lines"].append(line)

        if current_call is not None:
            calls.append(current_call)

    return calls


# --------------------------------------------------
# READ ALL OCCURRENCES OF A SECTION
# --------------------------------------------------

def read_all_sections(lines, marker):
    """
    Return every occurrence of a section, not just the first one.
    Each section ends at the next '---' marker.
    """

    sections = []

    i = 0

    while i < len(lines):

        if marker in lines[i]:

            section = []
            j = i + 1

            while j < len(lines) and "---" not in lines[j]:
                section.append(lines[j])
                j += 1

            sections.append({
                "header_index": i,
                "lines": section
            })

            i = j

        else:
            i += 1

    return sections


# --------------------------------------------------
# IDENTIFY THE PLANE FOR EACH SECTION
# --------------------------------------------------

def get_plane_number(lines, header_index):
    """
    Look backwards from a section header to find the
    nearest line identifying the current plane.
    """

    for i in range(header_index - 1, max(-1, header_index - 20), -1):

        line = lines[i]

        if "plane" in line.lower():

            numbers = re.findall(r"-?\d+", line)

            if numbers:
                return int(numbers[-1])

    return None


# --------------------------------------------------
# PARSE GRID FIELD
# --------------------------------------------------

def parse_grid_field(section_lines):

    field = np.full((NY, NZ), np.nan)

    for line in section_lines:

        parts = line.split()

        if len(parts) < 3:
            continue

        try:
            iy = int(parts[0])
            iz = int(parts[1])
            value = float(parts[2].replace("D", "E").replace("d", "e"))

            py = iy - IY_MIN
            pz = iz - IZ_MIN

            if 0 <= py < NY and 0 <= pz < NZ:
                field[py, pz] = value

        except ValueError:
            continue

    return field


# --------------------------------------------------
# PARSE SPARSE MATRIX
# --------------------------------------------------

def parse_sparse_matrix(section_lines):

    rows, cols, values = [], [], []

    pattern = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+([-+0-9.EeDd]+)"
    )

    for line in section_lines:

        match = pattern.match(line)

        if match:
            row, col, value = match.groups()

            rows.append(int(row) - 1)
            cols.append(int(col) - 1)
            values.append(float(value.replace("D", "E").replace("d", "e")))

    return coo_matrix(
        (values, (rows, cols)),
        shape=(N, N)
    ).tocsr()


# --------------------------------------------------
# PARSE RHS
# --------------------------------------------------

def parse_rhs(section_lines):

    rhs = np.zeros(N)

    for line in section_lines:

        parts = line.split()

        if len(parts) < 2:
            continue

        try:
            row = int(parts[0])
            value = float(parts[1].replace("D", "E").replace("d", "e"))

            if 1 <= row <= N:
                rhs[row - 1] = value

        except ValueError:
            continue

    return rhs


# --------------------------------------------------
# FIELD STATISTICS
# --------------------------------------------------

def print_field_stats(name, field):

    valid = field[np.isfinite(field)]

    if len(valid) == 0:
        print(f"{name}: no valid values")
        return

    print(
        f"{name}: "
        f"min={valid.min():.6e}, "
        f"max={valid.max():.6e}, "
        f"nonzero={np.count_nonzero(valid)}, "
        f"missing={np.isnan(field).sum()}"
    )


# --------------------------------------------------
# READ ALL CALLS
# --------------------------------------------------

calls = read_calls(FILE)

print(f"Found {len(calls)} debug calls")


# --------------------------------------------------
# PROCESS EACH CALL
# --------------------------------------------------

for call in calls:

    call_number = call["number"]
    lines = call["lines"]

    print("\n" + "=" * 65)
    print(f"DEBUG CALL {call_number}")
    print("=" * 65)

    # --------------------------------------------------
    # READ ALL PLANES
    # --------------------------------------------------

    input_sections = read_all_sections(
        lines, "--- INPUT FIELD: Vx_wake2 ---"
    )

    vt_sections = read_all_sections(
        lines, "--- INPUT FIELD: vt_tot2 ---"
    )

    output_sections = read_all_sections(
        lines, "--- OUTPUT Vx_wake2 ---"
    )

    rhs_sections = read_all_sections(
        lines, "--- RHS BEFORE SGESV ---"
    )

    matrix_sections = read_all_sections(
        lines, "--- SPARSE MATRIX BEFORE SGESV ---"
    )

    print(f"\nInput Vx_wake2 sections: {len(input_sections)}")
    print(f"Input vt_tot2 sections:  {len(vt_sections)}")
    print(f"Output Vx_wake2 sections: {len(output_sections)}")
    print(f"RHS sections: {len(rhs_sections)}")
    print(f"Matrix sections: {len(matrix_sections)}")

    # --------------------------------------------------
    # INPUT AND OUTPUT FIELD DIAGNOSTICS
    # --------------------------------------------------

    output_fields = []

    print("\n--- VELOCITY FIELD DIAGNOSTICS ---")

    for k, section in enumerate(output_sections):

        plane = get_plane_number(lines, section["header_index"])

        if plane is None:
            plane = k

        Vx_out = parse_grid_field(section["lines"])
        output_fields.append((plane, Vx_out))

        print(f"\nPlane {plane}:")

        # Match input fields by their occurrence order.
        if k < len(input_sections):
            Vx_in = parse_grid_field(input_sections[k]["lines"])
            print_field_stats("  Input Vx_wake2", Vx_in)
        else:
            Vx_in = None
            print("  Input Vx_wake2: no matching section")

        if k < len(vt_sections):
            vt = parse_grid_field(vt_sections[k]["lines"])
            print_field_stats("  Input vt_tot2", vt)
        else:
            print("  Input vt_tot2: no matching section")

        print_field_stats("  Output Vx_wake2", Vx_out)

    # --------------------------------------------------
    # RHS DIAGNOSTICS FOR ALL PLANES
    # --------------------------------------------------

    print("\n--- RHS DIAGNOSTICS ---")

    for k, section in enumerate(rhs_sections):

        plane = get_plane_number(lines, section["header_index"])

        if plane is None:
            plane = k

        rhs = parse_rhs(section["lines"])

        print(
            f"Plane {plane}: "
            f"min={rhs.min():.6e}, "
            f"max={rhs.max():.6e}, "
            f"nonzero={np.count_nonzero(rhs)}"
        )

    # --------------------------------------------------
    # MATRIX DIAGNOSTICS
    # --------------------------------------------------

    print("\n--- MATRIX DIAGNOSTICS ---")

    matrices = []

    for k, section in enumerate(matrix_sections):

        plane = get_plane_number(lines, section["header_index"])

        if plane is None:
            plane = k

        A = parse_sparse_matrix(section["lines"])
        matrices.append((plane, A))

        print(
            f"Plane {plane}: "
            f"shape={A.shape}, "
            f"nonzeros={A.nnz}, "
            f"density={A.nnz / (N*N):.6e}"
        )

    # --------------------------------------------------
    # PLOTS: MATRIX AND RHS FOR FIRST PLANE
    # --------------------------------------------------

    if matrices:

        plane, A = matrices[0]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.spy(A, markersize=0.5)
        ax.set_title(
            f"Call {call_number}, Plane {plane}: Matrix A"
        )
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        plt.tight_layout()

    if rhs_sections:

        plane = get_plane_number(
            lines, rhs_sections[0]["header_index"]
        )

        if plane is None:
            plane = 0

        rhs = parse_rhs(rhs_sections[0]["lines"])

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(rhs)
        ax.set_title(
            f"Call {call_number}, Plane {plane}: RHS"
        )
        ax.set_xlabel("Grid point")
        ax.set_ylabel("RHS")
        ax.grid(True)
        plt.tight_layout()

    # --------------------------------------------------
    # PLOT OUTPUT VELOCITY FIELDS FOR ALL PLANES
    # --------------------------------------------------

    if output_fields:

        nplanes = len(output_fields)

        fig, axes = plt.subplots(
            1, nplanes,
            figsize=(5 * nplanes, 5),
            squeeze=False
        )

        for ax, (plane, Vx) in zip(axes.ravel(), output_fields):

            im = ax.imshow(
                Vx.T,
                origin="lower",
                aspect="equal"
            )

            ax.set_title(f"Plane {plane}")
            ax.set_xlabel("iy")
            ax.set_ylabel("iz")

            fig.colorbar(im, ax=ax, label="Vx")

        fig.suptitle(
            f"Call {call_number}: Output Vx_wake2"
        )

        plt.tight_layout()

plt.show()