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

# Random material for each pin (1,2,3)
pin_materials = {pid: random.randint(1, 3) for pid in range(n_pins*n_pins)}

# =============================================================================
def get_coarse_cell_info(i, j, k, n_coarse):
    xc = (i + 0.5) * (L / n_coarse)
    yc = (j + 0.5) * (L / n_coarse)
    zc = (k + 0.5) * (L / n_coarse)
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
    cells = []          # (i,j,k,region,pin_id)
    for i in range(n_coarse):
        for j in range(n_coarse):
            for k in range(n_coarse):
                keep, reg, pid = get_coarse_cell_info(i, j, k, n_coarse)
                if keep:
                    cells.append((i, j, k, reg, pid))

    node_map = {}
    nodes = []
    for i in range(n_coarse+1):
        for j in range(n_coarse+1):
            for k in range(n_coarse+1):
                node_map[(i,j,k)] = len(nodes)
                nodes.append((i*dx, j*dx, k*dx))

    elements = []
    cell_regions = []
    cell_pin_ids = []
    cell_ijk = []
    mat_type = []
    for (i,j,k,reg,pid) in cells:
        corners = [(i,j,k), (i+1,j,k), (i+1,j+1,k), (i,j+1,k),
                   (i,j,k+1), (i+1,j,k+1), (i+1,j+1,k+1), (i,j+1,k+1)]
        elem_nodes = [node_map[c] for c in corners]
        elements.append(elem_nodes)
        cell_regions.append(reg)
        cell_pin_ids.append(pid)
        cell_ijk.append((i,j,k))
        if reg == 0:
            mat_type.append(0)
        elif reg == 1:
            mat_type.append(pin_materials[pid])

    return nodes, elements, cell_regions, cell_pin_ids, cell_ijk, node_map, mat_type

# =============================================================================
# Fine mesh by subdividing coarse cells
# =============================================================================
def build_fine_mesh(n_coarse):
    n_fine = 4 * n_coarse
    dx_fine = L / n_fine

    # Collect all kept coarse cells with their (i,j,k,region,pid)
    coarse_cells = []
    for i in range(n_coarse):
        for j in range(n_coarse):
            for k in range(n_coarse):
                keep, reg, pid = get_coarse_cell_info(i, j, k, n_coarse)
                if keep:
                    coarse_cells.append((i, j, k, reg, pid))

    fine_node_map = {}
    fine_nodes = []
    fine_elements = []
    fine_regions = []
    fine_pin_ids = []
    fine_cell_ijk = []   # (i_fine, j_fine, k_fine)
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
                    corners = [(fi, fj, fk), (fi+1, fj, fk), (fi+1, fj+1, fk), (fi, fj+1, fk),
                               (fi, fj, fk+1), (fi+1, fj, fk+1), (fi+1, fj+1, fk+1), (fi, fj+1, fk+1)]
                    node_ids = []
                    for (I,J,K) in corners:
                        key = (I,J,K)
                        if key not in fine_node_map:
                            fine_node_map[key] = len(fine_nodes)
                            fine_nodes.append((I*dx_fine, J*dx_fine, K*dx_fine))
                        node_ids.append(fine_node_map[key])
                    fine_elements.append(node_ids)
                    fine_regions.append(reg)
                    fine_pin_ids.append(pid)
                    fine_cell_ijk.append((fi, fj, fk))
                    if reg == 0:
                        mat_type.append(0)
                    elif reg == 1:
                        mat_type.append(pin_materials[pid])

    return fine_nodes, fine_elements, fine_regions, fine_pin_ids, fine_cell_ijk, fine_node_map, mat_type

# =============================================================================
# Extract bottom quads (z = 0)
# =============================================================================
def get_bottom_quads(cell_ijk, node_map, is_fine=False):
    """cell_ijk: list of (i,j,k) for each cell. For fine, k is fine index."""
    quads = []
    for idx, (i,j,k) in enumerate(cell_ijk):
        if k == 0:
            # bottom face nodes: (i,j,k), (i+1,j,k), (i+1,j+1,k), (i,j+1,k)
            n1 = node_map[(i, j, k)]
            n2 = node_map[(i+1, j, k)]
            n3 = node_map[(i+1, j+1, k)]
            n4 = node_map[(i, j+1, k)]
            quads.append([n1, n2, n3, n4])
    return quads

# =============================================================================
# Extract top face nodes for each pin
# =============================================================================
def get_pin_top_nodes(cell_ijk, cell_pin_ids, node_map):
    """Return dict: pin_id -> list of unique node IDs on top surface of that pin."""
    # Group cells by pin_id (ignore base cells, pid=-1)
    pin_cells = {}
    for idx, pid in enumerate(cell_pin_ids):
        if pid != -1:
            pin_cells.setdefault(pid, []).append(idx)

    pin_top_nodes = {}
    for pid, indices in pin_cells.items():
        # For each pin, find the maximum k among its cells
        max_k = max(cell_ijk[idx][2] for idx in indices)
        top_nodes = set()
        for idx in indices:
            i, j, k = cell_ijk[idx]
            if k == max_k:
                # top face nodes of this cell
                n5 = node_map[(i, j, k+1)]
                n6 = node_map[(i+1, j, k+1)]
                n7 = node_map[(i+1, j+1, k+1)]
                n8 = node_map[(i, j+1, k+1)]
                top_nodes.update([n5, n6, n7, n8])
        pin_top_nodes[pid] = list(top_nodes)
    return pin_top_nodes

# =============================================================================
# Write outputs
# =============================================================================
def write_outputs(prefix, nodes, elements, bottom_quads, pin_top_nodes, mat_list):
    # Nodes
    with open(f"{prefix}_nodes.txt", "w") as f:
        f.write("# NodeID X Y Z\n")
        for idx, (x,y,z) in enumerate(nodes):
            f.write(f"{idx} {x:.6f} {y:.6f} {z:.6f}\n")

    # Elements
    with open(f"{prefix}_elements.txt", "w") as f:
        f.write("# ElementID Node1 Node2 Node3 Node4 Node5 Node6 Node7 Node8\n")
        for eidx, conn in enumerate(elements):
            f.write(f"{eidx} " + " ".join(str(nid) for nid in conn) + "\n")

    # Materials
    with open(f"{prefix}_materials.txt", "w") as f:
        f.write("# MateiralID\n")
        for matID in mat_list:
            f.write(f"{matID}\n")

    # Bottom quads
    with open(f"{prefix}_bottom_quads.txt", "w") as f:
        f.write("# QuadID Node1 Node2 Node3 Node4\n")
        for qid, quad in enumerate(bottom_quads):
            f.write(f"{qid} " + " ".join(str(n) for n in quad) + "\n")

    # Pin top nodes and materials
    with open(f"{prefix}_pin_top_nodes.txt", "w") as f:
        f.write("# PinID MaterialLabel NodeList\n")
        for pid, nodes_list in pin_top_nodes.items():
            f.write(f"{pid} " + " ".join(str(n) for n in nodes_list) + "\n")

    # VTK
    with open(f"{prefix}.vtk", "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"Heat radiator mesh: {prefix}\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(nodes)} double\n")
        for x, y, z in nodes:
            f.write(f"{x} {y} {z}\n")
        f.write(f"CELLS {len(elements)} {len(elements) * 9}\n")
        for conn in elements:
            f.write("8 " + " ".join(str(nid) for nid in conn) + "\n")
        f.write(f"CELL_TYPES {len(elements)}\n")
        for _ in elements:
            f.write("12\n")
        f.write(f"CELL_DATA {len(elements)}\n")
        f.write("SCALARS region_id int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for matID in mat_list:
            f.write(f"{matID}\n")

    print(f"Written {prefix}_bottom_quads.txt and {prefix}_pin_top_nodes.txt")

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    n_coarse = int(input("Enter coarse mesh resolution n_coarse (must be integer, e.g., 10): "))
    n_fine = 4 * n_coarse

    # Coarse mesh
    coarse_nodes, coarse_elems, coarse_regions, coarse_pin_ids, coarse_ijk, coarse_node_map, mat_list = build_coarse_mesh(n_coarse)
    coarse_bottom = get_bottom_quads(coarse_ijk, coarse_node_map, is_fine=False)
    coarse_pin_top = get_pin_top_nodes(coarse_ijk, coarse_pin_ids, coarse_node_map)
    write_outputs("radiator_coarse", coarse_nodes, coarse_elems, coarse_bottom, coarse_pin_top, mat_list)

    # Fine mesh
    fine_nodes, fine_elems, fine_regions, fine_pin_ids, fine_ijk, fine_node_map, mat_list = build_fine_mesh(n_coarse)
    fine_bottom = get_bottom_quads(fine_ijk, fine_node_map, is_fine=True)
    fine_pin_top = get_pin_top_nodes(fine_ijk, fine_pin_ids, fine_node_map)
    write_outputs("radiator_fine", fine_nodes, fine_elems, fine_bottom, fine_pin_top, mat_list)

    print("\nDone. Both meshes have identical geometry (coarse cells determine shape).")
    print("Files generated:")
    print("  - radiator_coarse_nodes.txt, radiator_coarse_bottom_quads.txt, radiator_coarse_pin_top_nodes.txt")
    print("  - radiator_fine_nodes.txt, radiator_fine_bottom_quads.txt, radiator_fine_pin_top_nodes.txt")
