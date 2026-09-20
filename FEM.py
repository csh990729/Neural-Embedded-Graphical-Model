import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sys import exit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# generate unstructured vtk file for visualization
def generate_vtk(coor, conn, d, fname):
    nnode = np.shape(coor)[1]
    ncell = np.shape(conn)[0]
    filename = fname+'.vtk'
    fout = open(filename, 'w')
    fout.write("# vtk DataFile Version 3.0\n")
    fout.write("fem heat conduction result\n")
    fout.write("ASCII\n")
    fout.write("\n")
    fout.write("DATASET UNSTRUCTURED_GRID\n")
    fout.write("POINTS "+str(nnode)+" float\n")
    np.savetxt(fout, coor.transpose())
    fout.write("\n")
    fout.write("CELLS "+str(ncell)+" "+str(9*ncell)+'\n')
    data = np.concatenate((8*np.ones((ncell,1),'int64'), conn), axis=1)
    np.savetxt(fout, data, '%d')
    fout.write("CELL_TYPES "+str(ncell)+'\n')
    np.savetxt(fout, 12*np.ones((ncell,1),'int64'), '%d')
    fout.write("POINT_DATA "+str(nnode)+'\n')
    fout.write("SCALARS scalars float 1\n")
    fout.write("LOOKUP_TABLE default\n")
    np.savetxt(fout,d)
    fout.close()

    
def matmodel(etype, d_):
    prop = 0
    if etype == 0:
        prop = 1.0
    elif etype == 1: # linear increase
        param1 = 1.0
        param2 = 1.0
        prop = param1 + param2*d_
    elif etype == 2: # sinusoidal
        param3 = 1.0
        param4 = np.pi
        param5 = 1.0
        prop = param3*np.sin(param4*d_) + param5
    elif etype == 3: # piecewise linear change
        param6 = -2.0
        param7 = 2.0
        param8 = 0.5
        param9 = 1.0
        if d_ <= param8:
            prop = param6*(d_-param8) + param9
        else:
            prop = param7*(d_-param8) + param9
    elif etype == 4: # discontinuous
        param10 = 1.5
        param11 = 0.5
        param12 = 0.5
        prop = param10 + param11*np.tanh(20.0*(d_-param12))
    
    return prop


# ------------------------------------------------------------
# Main solver
# ------------------------------------------------------------
# elements
data = np.loadtxt("radiator_fine_elements.txt", skiprows=1, dtype=int)
elem_num = len(data)
conn = data[:, 1:]
print("number of elements:"+str(elem_num))

# coordinates
data = np.loadtxt("radiator_fine_nodes.txt", skiprows=1)
node_num = len(data)
coor = data[:, 1:] * 10
coor = coor.transpose()
print("number of nodes:"+str(node_num))

# material
elem_type = np.loadtxt("radiator_fine_materials.txt", skiprows=1, dtype=int)

# hf BC
data = np.loadtxt("radiator_fine_bottom_quads.txt", skiprows=1, dtype=int)
hfBC = data[:, 1:]
nhfs = len(hfBC)
hfval = np.ones(nhfs) * 0.02

# temperature BC
data = np.loadtxt("radiator_fine_pin_top_nodes.txt", skiprows=1, dtype=int)
tempBC = data[:, 1:].ravel()
ntemps = len(tempBC)
tempval = np.ones(ntemps) * 0.3

node_count = np.zeros(node_num,dtype=int)

for e in range(elem_num):
    node_count[conn[e,:]] += 1

orphan_nodes = np.where(node_count == 0)[0]

print("Number of orphan nodes =", len(orphan_nodes))
print("Node IDs =", orphan_nodes)
orphan_coords = coor[:, orphan_nodes].T

print(orphan_coords)


# --------------------------------------------------------
# Newton-Raphson iteration
# --------------------------------------------------------
load_maxiter = 100
convergence_maxiter = 200
bisec_maxiter = 5

load_factor_start = 0.0
load_factor_end = 1.0
load_factor_inc = 0.1

load_factor = load_factor_start

# temperature field
d = np.ones(node_num) * 0.3

# previous converged state
d_ref = d.copy()

qpts = np.array([[ -1.0, 1.0, 1.0,-1.0,-1.0, 1.0, 1.0,-1.0],
[ -1.0,-1.0, 1.0, 1.0,-1.0,-1.0, 1.0, 1.0],
[ -1.0,-1.0,-1.0,-1.0, 1.0, 1.0, 1.0, 1.0],
[  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
qpts[0:3, :] = qpts[0:3, :]/np.sqrt(3.0)

sfun = np.zeros((8,8)) #shape function evaluated at each quadrature point
dndp = np.zeros((8,3,8)) #derivative of shape function
for i in range(0,8):
    q = qpts[:, i]
    sfun[:,i] = np.array([(1-q[0])*(1-q[1])*(1-q[2]),
        (1+q[0])*(1-q[1])*(1-q[2]),
        (1+q[0])*(1+q[1])*(1-q[2]),
        (1-q[0])*(1+q[1])*(1-q[2]),
        (1-q[0])*(1-q[1])*(1+q[2]),
        (1+q[0])*(1-q[1])*(1+q[2]),
        (1+q[0])*(1+q[1])*(1+q[2]),
        (1-q[0])*(1+q[1])*(1+q[2])])
    sfun[:,i] = 0.125*sfun[:,i]

    dndp[:,:,i] = np.array([[-(1-q[1])*(1-q[2]), -(1-q[0])*(1-q[2]), -(1-q[0])*(1-q[1])],
        [(1-q[1])*(1-q[2]), -(1+q[0])*(1-q[2]), -(1+q[0])*(1-q[1])],
        [(1+q[1])*(1-q[2]),  (1+q[0])*(1-q[2]), -(1+q[0])*(1+q[1])],
        [-(1+q[1])*(1-q[2]),  (1-q[0])*(1-q[2]), -(1-q[0])*(1+q[1])],
        [-(1-q[1])*(1+q[2]), -(1-q[0])*(1+q[2]),  (1-q[0])*(1-q[1])],
        [(1-q[1])*(1+q[2]), -(1+q[0])*(1+q[2]),  (1+q[0])*(1-q[1])],
        [(1+q[1])*(1+q[2]),  (1+q[0])*(1+q[2]),  (1+q[0])*(1+q[1])],
        [-(1+q[1])*(1+q[2]),  (1-q[0])*(1+q[2]),  (1-q[0])*(1+q[1])]])
    dndp[:,:,i] = 0.125*dndp[:,:,i]

bcqpts = np.array([[-1.0, 1.0, 1.0,-1.0],
           [-1.0,-1.0, 1.0, 1.0],
           [ 1.0, 1.0, 1.0, 1.0]])
bcqpts[0:2,:] = bcqpts[0:2,:]/np.sqrt(3.0);
bsfun = np.zeros((4,4)); # shape function for boundary
bdndp = np.zeros((4,2,4)); # derivative of shape function for boundary
for i in range(0,4) :
    q = bcqpts[:,i]
    bsfun[:,i] = np.array([(1-q[0])*(1-q[1]),
         (1+q[0])*(1-q[1]),
         (1+q[0])*(1+q[1]),
         (1-q[0])*(1+q[1])])
    bsfun[:,i] = 0.25*bsfun[:,i]
    bdndp[:,:,i] = np.array([[-(1-q[1]), -(1-q[0])],
         [(1-q[1]), -(1+q[0])],
         [(1+q[1]),  (1+q[0])],
         [-(1+q[1]),  (1-q[0])]])
    bdndp[:,:,i] = 0.25*bdndp[:,:,i]

#obtain heat flux boundary
f_ref = np.zeros(node_num)

for i in range(nhfs):

    c = hfBC[i,:]

    xe = coor[:,c]

    nv = np.cross(
        xe[:,1]-xe[:,0],
        xe[:,3]-xe[:,0]
    )

    nv = nv / np.linalg.norm(nv)

    tmp = np.sqrt(nv[0]**2 + nv[1]**2)

    Rmat = np.array([
        [nv[2]+nv[1]**2*(1-nv[2]),
         -nv[0]*nv[1]*(1-nv[2]),
         -nv[0]*tmp],

        [-nv[0]*nv[1]*(1-nv[2]),
         nv[2]+nv[0]**2*(1-nv[2]),
         -nv[1]*tmp],

        [nv[0]*tmp,
         nv[1]*tmp,
         nv[2]]
    ])

    xer = Rmat @ xe

    s = hfval[i]

    for j in range(4):

        bsfunx = bsfun[:,j]
        bdndpx = bdndp[:,:,j]

        Jacob = xer[0:2,:] @ bdndpx

        detJ = np.abs(np.linalg.det(Jacob))

        f_ref[c] += (
            bsfunx *
            s *
            detJ *
            bcqpts[2,j]
        )

# iterations
load_iter = 0
bisec_iter = 0

while True:

    load_iter += 1

    if load_iter > load_maxiter:
        raise RuntimeError(
            "Maximum load-step iterations reached."
        )

    if load_factor + load_factor_inc > load_factor_end:
        load_factor_inc = load_factor_end - load_factor

    target_load = load_factor + load_factor_inc

    convergence_flag = False

    for iiter in range(convergence_maxiter):

        print(
            f"Load={target_load:.6f}, "
            f"Iter={iiter+1}"
        )

        # --------------------------------------------------
        # Assemble stiffness
        # --------------------------------------------------

        nnz = 64 * elem_num

        I = np.zeros(nnz,dtype=np.int64)
        J = np.zeros(nnz,dtype=np.int64)
        V = np.zeros(nnz)

        nz_num = 0

        for ielem in range(elem_num):

            c = conn[ielem,:]

            xe = coor[:,c]
            de = d[c]

            c1,c2 = np.meshgrid(c,c,indexing='ij')

            cc1 = c1.ravel()
            cc2 = c2.ravel()

            nps = 64

            Ke = np.zeros((8,8))

            for iq in range(8):

                sfunx = sfun[:,iq]

                d_sample = np.dot(sfunx,de)

                kappa = matmodel(
                    elem_type[ielem],
                    d_sample
                )

                dndpx = dndp[:,:,iq]

                Jacob = xe @ dndpx

                detJ = np.linalg.det(Jacob)

                invJ = np.linalg.inv(Jacob)

                B = dndpx @ invJ

                Ke += (
                    kappa *
                    (B @ B.T) *
                    np.abs(detJ) *
                    qpts[3,iq]
                )

            I[nz_num:nz_num+nps] = cc1
            J[nz_num:nz_num+nps] = cc2
            V[nz_num:nz_num+nps] = Ke.ravel()

            nz_num += nps

        Kstiff = sparse.coo_matrix(
            (V,(I,J)),
            shape=(node_num,node_num)
        ).tolil()

        # --------------------------------------------------
        # Load vector
        # --------------------------------------------------

        f = f_ref * target_load

        f_r = f - Kstiff * d

        # --------------------------------------------------
        # Dirichlet BC
        # --------------------------------------------------

        for ibc in range(ntemps):

            indx = tempBC[ibc]
            value = tempval[ibc]

            Kstiff[indx,:] = 0.0
            Kstiff[:,indx] = 0.0

            Kstiff[indx,indx] = 1.0
            f_r[indx] = 0

        Kstiff = Kstiff.tocsr()

        # --------------------------------------------------
        # Solve
        # --------------------------------------------------

        d_delta = np.asarray(
            spsolve(Kstiff,f_r)
        ).ravel()

        # --------------------------------------------------
        # Convergence
        # --------------------------------------------------

        d += d_delta

        for ibc in range(ntemps):
            indx = tempBC[ibc]
            value = tempval[ibc]
            d[indx] = value

        err_r = np.max(np.abs(d_delta))

        print(err_r)

        if err_r < 1.0e-6:

            convergence_flag = True

            d_ref = d.copy()

            load_factor = target_load

            print(
                f"Converged. "
                f"Load factor = {load_factor}"
            )

            break

    # ------------------------------------------------------
    # Load step accepted
    # ------------------------------------------------------

    if convergence_flag:

        if abs(load_factor-load_factor_end) < 1e-10:
            break

##        load_factor_inc *= 1.1
        bisec_iter = 0

    # ------------------------------------------------------
    # Load step rejected
    # ------------------------------------------------------

    else:

        bisec_iter += 1

        if bisec_iter >= bisec_maxiter:
            raise RuntimeError(
                "Maximum bisection attempts exceeded."
            )

        load_factor_inc *= 0.5

        d = d_ref.copy()

        print(
            f"Step failed. "
            f"Reducing increment to "
            f"{load_factor_inc}"
        )

generate_vtk(coor, conn, d_ref, 'fem_result')



