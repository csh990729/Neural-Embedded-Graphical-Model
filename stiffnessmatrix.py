import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import networkx as nx

# ------------------- Mesh generation -------------------
def generate_square_tri_mesh(n_cells_x, n_cells_y, Lx=1.0, Ly=1.0):
    x = np.linspace(0, Lx, n_cells_x+1)
    y = np.linspace(0, Ly, n_cells_y+1)
    nodes = np.array([[xi, yj] for yj in y for xi in x])
    Nx = n_cells_x + 1
    Ny = n_cells_y + 1
    elements = []
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            n0 = j * Nx + i
            n1 = j * Nx + (i+1)
            n2 = (j+1) * Nx + i
            n3 = (j+1) * Nx + (i+1)
            elements.append((n0, n1, n2))
            elements.append((n1, n3, n2))
    return nodes, elements

# ------------------- Elasticity stiffness matrix (plane stress, CST) -------------------
def elasticity_stiffness_matrix(nodes, elements, E=1.0, nu=0.3, thickness=1.0):
    N = len(nodes)
    K_glob = np.zeros((2*N, 2*N))
    
    D = (E / (1 - nu**2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu)/2]
    ])
    
    for tri in elements:
        i, j, k = tri
        xi, yi = nodes[i]
        xj, yj = nodes[j]
        xk, yk = nodes[k]
        area = 0.5 * abs((xj - xi)*(yk - yi) - (xk - xi)*(yj - yi))
        if area < 1e-12:
            continue
        beta = np.array([yj - yk, yk - yi, yi - yj])
        gamma = np.array([xk - xj, xi - xk, xj - xi])
        
        B = np.zeros((3, 6))
        for m in range(3):
            B[0, 2*m]   = beta[m] / (2*area)
            B[1, 2*m+1] = gamma[m] / (2*area)
            B[2, 2*m]   = gamma[m] / (2*area)
            B[2, 2*m+1] = beta[m] / (2*area)
        
        Ke = area * thickness * (B.T @ D @ B)
        dofs = []
        for node in (i, j, k):
            dofs.extend([2*node, 2*node+1])
        for a in range(6):
            for b in range(6):
                K_glob[dofs[a], dofs[b]] += Ke[a, b]
    return K_glob

# ------------------- Solve with proper elimination -------------------
def solve_elasticity(K_glob, fixed_dofs, prescribed_dofs, prescribed_vals):
    """
    Solve K u = f with given essential BCs using reduction method.
    fixed_dofs: list of DOF indices with zero displacement.
    prescribed_dofs: list of DOF indices with given non‑zero displacement.
    prescribed_vals: corresponding displacement values.
    """
    N = K_glob.shape[0]
    # Build RHS (zero for now, but we will modify for prescribed)
    f = np.zeros(N)
    
    # Identify free DOFs (all DOFs not in prescribed or fixed)
    all_prescribed = list(set(fixed_dofs + prescribed_dofs))
    free_dofs = [i for i in range(N) if i not in all_prescribed]
    
    # Partition matrices
    K_ff = K_glob[np.ix_(free_dofs, free_dofs)]
    K_fp = K_glob[np.ix_(free_dofs, all_prescribed)]
    K_pf = K_glob[np.ix_(all_prescribed, free_dofs)]
    K_pp = K_glob[np.ix_(all_prescribed, all_prescribed)]
    
    # Known displacements for prescribed DOFs
    u_p = np.zeros(len(all_prescribed))
    for i, dof in enumerate(all_prescribed):
        if dof in fixed_dofs:
            u_p[i] = 0.0
        else:
            idx = prescribed_dofs.index(dof)
            u_p[i] = prescribed_vals[idx]
    
    # RHS for free DOFs: f_f = - K_fp * u_p
    f_f = -K_fp @ u_p
    
    # Solve for free DOFs
    u_f = np.linalg.solve(K_ff, f_f)
    
    # Assemble full solution
    u = np.zeros(N)
    u[free_dofs] = u_f
    for i, dof in enumerate(all_prescribed):
        u[dof] = u_p[i]
    
    return u

# ------------------- Helper: get DOFs on left and right edges -------------------
def get_boundary_dofs(nodes, Lx, tol=1e-6):
    left_nodes = [i for i, (x, y) in enumerate(nodes) if abs(x) < tol]
    right_nodes = [i for i, (x, y) in enumerate(nodes) if abs(x - Lx) < tol]
    left_dofs = []
    for node in left_nodes:
        left_dofs.extend([2*node, 2*node+1])
    right_dofs = []
    for node in right_nodes:
        right_dofs.extend([2*node, 2*node+1])
    return left_dofs, right_dofs

# ------------------- Visualisation mesh -------------------
def plot_mesh(nodes, elements, scale=10.0, title="Mesh"):
    N = len(nodes)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Original mesh (gray)
    for tri in elements:
        tri_nodes = nodes[list(tri)]
        
        # Compute centroid (average of vertices)
        centroid_x = np.mean(tri_nodes[:, 0])
        centroid_y = np.mean(tri_nodes[:, 1])
        # Choose color based on centroid
        facecolor = None
        if centroid_x < 0.5 and centroid_y < 0.5:
            facecolor = "magenta"
        elif centroid_x >= 0.5 and centroid_y >= 0.5:
            facecolor = "magenta"
        else:
            facecolor = "firebrick"
        
        polygon = Polygon(tri_nodes, closed=True, facecolor=facecolor, 
                edgecolor="black", linewidth=1.0, alpha=0.7)

        ax.add_patch(polygon)
    
    # Plot nodes (optional)
    ax.plot(nodes[:,0], nodes[:,1], 'ko', markersize=3)
    # Adjust limits and aspect
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.show()

    
# ------------------- Visualisation of deformed shape -------------------
def plot_deformed_mesh(nodes, elements, displacements, scale=10.0, title="Deformed shape"):
    N = len(nodes)
    ux = displacements[0::2]
    uy = displacements[1::2]
    deformed_nodes = nodes + scale * np.column_stack((ux, uy))
    
    plt.figure(figsize=(8,6))
    # Original mesh (gray)
    for tri in elements:
        tri_nodes = nodes[list(tri)]
        tri_pts = np.vstack([tri_nodes, tri_nodes[0]])
        plt.plot(tri_pts[:,0], tri_pts[:,1], 'k-', linewidth=0.5, alpha=0.3)
    # Deformed mesh (red)
    for tri in elements:
        tri_nodes_def = deformed_nodes[list(tri)]
        tri_pts_def = np.vstack([tri_nodes_def, tri_nodes_def[0]])
        plt.plot(tri_pts_def[:,0], tri_pts_def[:,1], 'r-', linewidth=1.2)
    plt.plot(deformed_nodes[:,0], deformed_nodes[:,1], 'ro', markersize=3)
    for i, (x,y) in enumerate(deformed_nodes):
        plt.text(x, y, str(i), fontsize=8, ha='right', va='bottom')
    plt.title(title)
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show()

# ------------------- Compute thermal stiffness and graph (optional) -------------------
def compute_thermal_stiffness(nodes, elements):
    N = len(nodes)
    K = np.zeros((N, N))
    for tri in elements:
        i, j, k = tri
        xi, yi = nodes[i]; xj, yj = nodes[j]; xk, yk = nodes[k]
        area = 0.5 * abs((xj - xi)*(yk - yi) - (xk - xi)*(yj - yi))
        if area < 1e-12: continue
        b = np.array([yj - yk, yk - yi, yi - yj])
        c = np.array([xk - xj, xi - xk, xj - xi])
        Ke = np.zeros((3,3))
        for m in range(3):
            for n in range(3):
                Ke[m,n] = (b[m]*b[n] + c[m]*c[n]) / (4.0 * area)
        nodes_tri = [i, j, k]
        for a in range(3):
            for b_ in range(3):
                K[nodes_tri[a], nodes_tri[b_]] += Ke[a, b_]
    return K

# ------------------- Main script -------------------
if __name__ == "__main__":
    # Mesh parameters – avoid name conflict with networkx
    n_cells_x, n_cells_y = 6, 6
    Lx, Ly = 1.0, 1.0
    nodes, elements = generate_square_tri_mesh(n_cells_x, n_cells_y, Lx, Ly)
    print(f"number of nodes / elemens: {len(nodes)}, {len(elements)}")

    # Visualise shape (scale = 5 for visibility)
    plot_mesh(nodes, elements, scale=5.0)
    
    # Material properties
    E = 1.0
    nu = 0.3
    thickness = 1.0
    
    # Assemble elasticity stiffness matrix
    K_elas = elasticity_stiffness_matrix(nodes, elements, E, nu, thickness)
    print(f"Elasticity stiffness matrix shape: {K_elas.shape}")
    
    # Boundary conditions
    left_dofs, right_dofs = get_boundary_dofs(nodes, Lx)
    fixed_dofs = left_dofs
    prescribed_dofs = []
    prescribed_vals = []
    for dof in right_dofs:
        node = dof // 2
        if dof % 2 == 0:   # ux
            prescribed_dofs.append(dof)
            prescribed_vals.append(0.01)
        else:               # uy
            prescribed_dofs.append(dof)
            prescribed_vals.append(0.005)
    
    # Solve
    displacements = solve_elasticity(K_elas, fixed_dofs, prescribed_dofs, prescribed_vals)
    
    # Extract and print some results
    ux = displacements[0::2]
    uy = displacements[1::2]
    print(f"Max ux = {np.max(ux):.6f}, Max uy = {np.max(uy):.6f}")
    
##    # Visualise deformed shape (scale = 5 for visibility)
##    plot_deformed_mesh(nodes, elements, displacements, scale=5.0)
    
    # Optional: plot the graph from the binary stiffness matrix
    K_thermal = compute_thermal_stiffness(nodes, elements)
    K_binary = (np.abs(K_thermal) > 1e-12).astype(float)

    print("\nOriginal stiffness matrix (dense):")
    print(np.array2string(K_thermal, precision=4, suppress_small=True))
    print("\nNonzero → 1 matrix:")
    print(np.array2string(K_binary.astype(int), precision=0, suppress_small=True))
    
    # Build graph using networkx (now no conflict because we renamed mesh vars)
    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            if K_binary[i, j] > 0.5:
                G.add_edge(i, j)
    pos = nx.spring_layout(G, dim=2, seed=42, k=0.16, iterations=50)
    
    plt.figure(figsize=(6,5))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray',
            node_size=300, font_size=8)
    plt.title("Graph from binary stiffness matrix (spring layout)")
    plt.axis('equal')
    plt.show()
