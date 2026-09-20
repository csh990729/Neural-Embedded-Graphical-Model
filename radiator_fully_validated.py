
import numpy as np
import random

# =============================================================================
# Geometry parameters
# =============================================================================
L = 1.0
base_thickness = 0.2
pin_height = 0.7
pin_size = 0.1
n_pins = 5

pin_centers_x = np.linspace(pin_size/2, L - pin_size/2, n_pins)
pin_centers_y = np.linspace(pin_size/2, L - pin_size/2, n_pins)

pin_id_map = {}
for i, cx in enumerate(pin_centers_x):
    for j, cy in enumerate(pin_centers_y):
        pin_id_map[(cx, cy)] = i * n_pins + j

pin_materials = {pid: random.randint(1, 3)
                 for pid in range(n_pins*n_pins)}

# =============================================================================
def get_coarse_cell_info(i, j, k, n_coarse):

    dx = L / n_coarse

    xc = (i + 0.5) * dx
    yc = (j + 0.5) * dx
    zc = (k + 0.5) * dx

    if zc <= base_thickness:
        return (True, 0, -1)

    if zc <= base_thickness + pin_height:
        for cx in pin_centers_x:
            for cy in pin_centers_y:
                if abs(xc - cx) <= pin_size/2 and abs(yc - cy) <= pin_size/2:
                    return (True, 1, pin_id_map[(cx, cy)])

    return (False, -1, -1)

# =============================================================================
# Coarse mesh
# =============================================================================
def build_coarse_mesh(n_coarse):

    dx = L / n_coarse

    cells = []

    for i in range(n_coarse):
        for j in range(n_coarse):
            for k in range(n_coarse):

                keep, reg, pid = get_coarse_cell_info(
                    i, j, k, n_coarse
                )

                if keep:
                    cells.append((i, j, k, reg, pid))

    node_map = {}
    nodes = []

    elements = []
    cell_regions = []
    cell_pin_ids = []
    cell_ijk = []
    mat_type = []

    for (i, j, k, reg, pid) in cells:

        corners = [
            (i, j, k),
            (i+1, j, k),
            (i+1, j+1, k),
            (i, j+1, k),
            (i, j, k+1),
            (i+1, j, k+1),
            (i+1, j+1, k+1),
            (i, j+1, k+1)
        ]

        elem_nodes = []

        for corner in corners:

            if corner not in node_map:

                node_map[corner] = len(nodes)

                nodes.append((
                    corner[0]*dx,
                    corner[1]*dx,
                    corner[2]*dx
                ))

            elem_nodes.append(node_map[corner])

        elements.append(elem_nodes)

        cell_regions.append(reg)
        cell_pin_ids.append(pid)
        cell_ijk.append((i, j, k))

        if reg == 0:
            mat_type.append(0)
        else:
            mat_type.append(pin_materials[pid])

    return (
        nodes,
        elements,
        cell_regions,
        cell_pin_ids,
        cell_ijk,
        node_map,
        mat_type
    )

# =============================================================================
# Fine mesh
# =============================================================================
def build_fine_mesh(n_coarse):

    n_fine = 4 * n_coarse
    dx_fine = L / n_fine

    coarse_cells = []

    for i in range(n_coarse):
        for j in range(n_coarse):
            for k in range(n_coarse):

                keep, reg, pid = get_coarse_cell_info(
                    i, j, k, n_coarse
                )

                if keep:
                    coarse_cells.append((i, j, k, reg, pid))

    fine_node_map = {}
    fine_nodes = []

    fine_elements = []
    fine_regions = []
    fine_pin_ids = []
    fine_cell_ijk = []
    mat_type = []

    for (ci, cj, ck, reg, pid) in coarse_cells:

        i0 = ci * 4
        j0 = cj * 4
        k0 = ck * 4

        for di in range(4):
            for dj in range(4):
                for dk in range(4):

                    fi = i0 + di
                    fj = j0 + dj
                    fk = k0 + dk

                    corners = [
                        (fi, fj, fk),
                        (fi+1, fj, fk),
                        (fi+1, fj+1, fk),
                        (fi, fj+1, fk),
                        (fi, fj, fk+1),
                        (fi+1, fj, fk+1),
                        (fi+1, fj+1, fk+1),
                        (fi, fj+1, fk+1)
                    ]

                    elem_nodes = []

                    for corner in corners:

                        if corner not in fine_node_map:

                            fine_node_map[corner] = len(fine_nodes)

                            fine_nodes.append((
                                corner[0]*dx_fine,
                                corner[1]*dx_fine,
                                corner[2]*dx_fine
                            ))

                        elem_nodes.append(
                            fine_node_map[corner]
                        )

                    fine_elements.append(elem_nodes)
                    fine_regions.append(reg)
                    fine_pin_ids.append(pid)
                    fine_cell_ijk.append((fi, fj, fk))

                    if reg == 0:
                        mat_type.append(0)
                    else:
                        mat_type.append(pin_materials[pid])

    return (
        fine_nodes,
        fine_elements,
        fine_regions,
        fine_pin_ids,
        fine_cell_ijk,
        fine_node_map,
        mat_type
    )

def verify_mesh(nodes, elements):

    node_count = np.zeros(len(nodes), dtype=int)

    for conn in elements:
        node_count[np.asarray(conn)] += 1

    orphan = np.where(node_count == 0)[0]

    print("nodes      =", len(nodes))
    print("elements   =", len(elements))
    print("orphans    =", len(orphan))

    if len(orphan) > 0:
        raise RuntimeError(
            f"Mesh contains {len(orphan)} orphan nodes."
        )

# Remaining helper functions unchanged from original...
print("Corrected mesh generator loaded.")
(nodes,elements,cell_regions,cell_pin_ids,cell_ijk,node_map,mat_type) = build_coarse_mesh(10)
verify_mesh(nodes, elements)
