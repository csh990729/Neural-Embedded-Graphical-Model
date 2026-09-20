import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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

#####CONSTRUCT FENN#############
# elements
data = np.loadtxt("radiator_coarse_elements.txt", skiprows=1, dtype=int)
elem_num = len(data)
conn = data[:, 1:]
print("number of elements:"+str(elem_num))

# coordinates
data = np.loadtxt("radiator_coarse_nodes.txt", skiprows=1)
node_num = len(data)
coor = data[:, 1:] * 10
coor = coor.transpose()
print("number of nodes:"+str(node_num))

# material
elem_type = np.loadtxt("radiator_coarse_materials.txt", skiprows=1, dtype=int)

# hf BC
data = np.loadtxt("radiator_coarse_bottom_quads.txt", skiprows=1, dtype=int)
hfBC = data[:, 1:]
nhfs = len(hfBC)
hfval = np.ones(nhfs) * 0.02

# temperature BC
data = np.loadtxt("radiator_coarse_pin_top_nodes.txt", skiprows=1, dtype=int)
tempBC = data[:, 1:].ravel()
ntemps = len(tempBC)
tempval = np.ones(ntemps) * 0.3

unique_ids = np.unique(hfBC)
bcpoints = np.concatenate((tempBC, unique_ids))


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

# # gather stiffness matrix
# K_ref = np.zeros((node_num,node_num))
# for i in range(0, elem_num) :
#     c = conn[i,:] # 1 by 8
#     xe = coor[:, c] # 3 by 8
#     for j in range(0,8) :
#         sfunx = sfun[:,j] # 8 by 1
#         dndpx = dndp[:,:,j] # 8 by 3
#         J = np.matmul(xe,dndpx) # 3 by 3
#         B = np.matmul(dndpx,np.linalg.inv(J)) # 8 by 3
#         c1, c2 = np.meshgrid(c,c)
#         K_ref[c1,c2] = K_ref[c1,c2] + np.matmul(B,np.transpose(B)*np.linalg.det(J)*qpts[3,j])

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

# obtain heat flux boundary
nload = len(bcpoints) - ntemps
all2load = np.zeros((node_num, nload))
flag = np.ones(node_num, dtype='int')
flag[bcpoints[ntemps:]] = 0
reindex = node_num*np.ones(node_num, dtype='int')
iv = 0
for i in np.arange(node_num):
    if flag[i] == 0:
        all2load[i, iv] = 1
        reindex[i] = iv
        iv += 1
load = np.zeros(nload)
for i in np.arange(nhfs):
    c = hfBC[i,:] # 1 by 4
    xe = coor[:, c]
    for j in range(0,4) :
        bsfunx = bsfun[:,j]
        bdndpx = bdndp[:,:,j]
        nv = np.cross((xe[:,1]-xe[:,0]), (xe[:,3]-xe[:,0]))
        nv = nv / np.linalg.norm(nv)
        Rmat = np.array([[nv[2]+nv[1]*nv[1]*(1-nv[2]), -nv[0]*nv[1]*(1-nv[2]), -nv[0]*np.sqrt(nv[0]*nv[0]+nv[1]*nv[1])],
            [-nv[0]*nv[1]*(1-nv[2]), nv[2]+nv[0]*nv[0]*(1-nv[2]), -nv[1]*np.sqrt(nv[0]*nv[0]+nv[1]*nv[1])],
            [nv[0]*np.sqrt(nv[0]*nv[0]+nv[1]*nv[1]), nv[1]*np.sqrt(nv[0]*nv[0]+nv[1]*nv[1]), nv[2]]]) # Rotation matrix
        xer = np.matmul(Rmat, xe) # Arbitrary plane reduced into a plane parallel to xy plane
        Jacob = np.matmul(xer[0:2,:],bdndpx)
        B = np.matmul(bdndpx,np.linalg.inv(Jacob))
        xq = np.matmul(xe,bsfunx)
        s = hfval[i]
        load[reindex[c]] = load[reindex[c]] + bsfunx*s*np.abs(np.linalg.det(Jacob))*bcqpts[2,j]

# gather internal dofs
tmpindices = bcpoints
ninternal = node_num - len(tmpindices)
all2internal = np.zeros((node_num, ninternal))
flag = np.ones(node_num, dtype='int')
flag[tmpindices] = 0
iv = 0
for i in np.arange(node_num):
    if flag[i] == 1:
        all2internal[i, iv] = 1
        iv += 1
        
# gather temperature boundaries
bc = np.zeros(node_num)
nvar = node_num - ntemps
var2all = np.zeros((nvar, node_num))
reindex = node_num*np.ones(node_num, dtype='int')
reindex[tempBC] = np.arange(ntemps)
iv = 0
for i in np.arange(node_num):
    if reindex[i] == node_num:
        var2all[iv, i] = 1
        iv += 1
    else:
        bc[i] = tempval[reindex[i]]

eindxT1 = np.where(elem_type==0)[0]; elem_numT1 = len(eindxT1)
eindxT2 = np.where(elem_type==1)[0]; elem_numT2 = len(eindxT2)
eindxT3 = np.where(elem_type==2)[0]; elem_numT3 = len(eindxT3)
eindxT4 = np.where(elem_type==3)[0]; elem_numT4 = len(eindxT4)

node2elemT1 = np.zeros((node_num, 8*elem_numT1))
for i in np.arange(elem_numT1):
    node2elemT1[conn[eindxT1[i],:], (8*i+0):(8*i+8)] = np.eye(8)
node2elemT2 = np.zeros((node_num, 8*elem_numT2))
for i in np.arange(elem_numT2):
    node2elemT2[conn[eindxT2[i],:], (8*i+0):(8*i+8)] = np.eye(8)
node2elemT3 = np.zeros((node_num, 8*elem_numT3))
for i in np.arange(elem_numT3):
    node2elemT3[conn[eindxT3[i],:], (8*i+0):(8*i+8)] = np.eye(8)
node2elemT4 = np.zeros((node_num, 8*elem_numT4))
for i in np.arange(elem_numT4):
    node2elemT4[conn[eindxT4[i],:], (8*i+0):(8*i+8)] = np.eye(8)

eindxTT = np.concatenate((eindxT1, eindxT2, eindxT3, eindxT4))
elem2node_sum = np.zeros((8*elem_num, node_num))
for i in np.arange(elem_num):
    elem2node_sum[(8*i+0):(8*i+8), conn[eindxTT[i],:]] = np.eye(8)

K0_forward = np.eye(8)*1.0/3.0
K0_forward[0, 2] = -1.0/12.0
K0_forward[0, 5] = -1.0/12.0
K0_forward[0, 6] = -1.0/12.0
K0_forward[0, 7] = -1.0/12.0
K0_forward[1, 3] = -1.0/12.0
K0_forward[1, 4] = -1.0/12.0
K0_forward[1, 6] = -1.0/12.0
K0_forward[1, 7] = -1.0/12.0
K0_forward[2, 0] = -1.0/12.0
K0_forward[2, 4] = -1.0/12.0
K0_forward[2, 5] = -1.0/12.0
K0_forward[2, 7] = -1.0/12.0
K0_forward[3, 1] = -1.0/12.0
K0_forward[3, 4] = -1.0/12.0
K0_forward[3, 5] = -1.0/12.0
K0_forward[3, 6] = -1.0/12.0
K0_forward[4, 1] = -1.0/12.0
K0_forward[4, 2] = -1.0/12.0
K0_forward[4, 3] = -1.0/12.0
K0_forward[4, 6] = -1.0/12.0
K0_forward[5, 0] = -1.0/12.0
K0_forward[5, 2] = -1.0/12.0
K0_forward[5, 3] = -1.0/12.0
K0_forward[5, 7] = -1.0/12.0
K0_forward[6, 0] = -1.0/12.0
K0_forward[6, 1] = -1.0/12.0
K0_forward[6, 3] = -1.0/12.0
K0_forward[6, 4] = -1.0/12.0
K0_forward[7, 0] = -1.0/12.0
K0_forward[7, 1] = -1.0/12.0
K0_forward[7, 2] = -1.0/12.0
K0_forward[7, 5] = -1.0/12.0

K0_backward = np.eye(8)

#####DEFINE ELEMENTS#############
class NNElement(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_num, K):
        super().__init__()
        
        self.K = torch.from_numpy(K).float()
        
        self.Li = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.Softplus()
            nn.Tanh()
        )
        
        self.Linter = torch.nn.ModuleList([])
        for i in range(hidden_num):
            self.Linter.append(nn.Linear(hidden_dim, hidden_dim))
            # self.Linter.append(nn.Softplus())
            self.Linter.append(nn.Tanh())
        
        self.Lo = nn.Linear(hidden_dim, output_dim, bias=False)
        self.Lo.weight.data = nn.Parameter(self.K, requires_grad=False)

    def forward(self, x):
        x = self.Li(x)
        for layer in self.Linter:
            x = layer(x)
        out = self.Lo(x)
        return out

class NNElement_new(nn.Module):
    def __init__(self, device, K0):
        super().__init__()

        self.K = torch.from_numpy(K0).float().to(device)

        self.L1 = nn.Linear(8, 8)
        self.L2 = nn.Tanh()
        self.L3 = nn.Linear(8, 8)
        self.L4 = nn.Tanh()

        self.L5 = nn.Linear(8, 8)
        self.L6 = nn.Tanh()
        self.L7 = nn.Linear(8, 8)
        self.L8 = nn.Tanh()

        #self.L9 = nn.Linear(16, 8)
        #self.L10 = nn.Tanh()

        #self.L11 = nn.Linear(16, 8)
        #self.L12 = nn.Tanh()

        self.Lo = nn.Linear(8, 8)

    def forward(self, x):
        y1 = self.L1(x)
        y2 = self.L2(y1)
        y3 = self.L3(y2)
        y4 = self.L4(y3)

        z0 = torch.matmul(x, self.K)
        z = torch.mul(y4, z0)
        y5 = self.L5(z)
        y6 = self.L6(y5)
        y7 = self.L7(y6)
        y8 = self.L8(y7)

        #z = torch.cat((y8,x), 1)
        #y9 = self.L9(z)
        #y10 = self.L10(y9)
        #
        #z = torch.cat((y10,x), 1)
        #y11 = self.L11(z)
        #y12 = self.L12(y11)

        out = self.Lo(y8)
        return out

class NNElement_new_new(nn.Module):
    def __init__(self, device, K0, input_dim, output_dim, hidden_dim, L1_num, L2_num):
        super().__init__()

        self.K = torch.from_numpy(K0).float().to(device)

        self.Li = nn.Linear(input_dim, hidden_dim)

        self.L1 = torch.nn.ModuleList([])
        for i in range(L1_num):
            self.L1.append(nn.Linear(hidden_dim, hidden_dim))
            self.L1.append(nn.LeakyReLU(0.01))

        self.L2 = torch.nn.ModuleList([])
        for i in range(L2_num):
            self.L2.append(nn.Linear(hidden_dim, hidden_dim))
            self.L2.append(nn.Tanh())

        self.Lo = nn.Linear(hidden_dim, output_dim)

        for m in self.L1:
            if isinstance(m, nn.Linear):
                #nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.bias.data.zero_()

        for m in self.L2:
            if isinstance(m, nn.Linear):
                #nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                m.bias.data.zero_()

    def forward(self, x):
        y = self.Li(x)

        for layer in self.L1:
            y = layer(y)

        x = torch.matmul(x, self.K)
        y = torch.mul(y, x)

        for layer in self.L2:
            y = layer(y)

        out = self.Lo(y)
        return out


prefix='./'

kT1_forward = NNElement(8, 8, 8, 2, K0_forward).to(device)
kT1_forward.load_state_dict(torch.load(prefix+'etype1/trained_matrix_d2f.pth',map_location=device))
kT1_forward.eval()
for param in kT1_forward.parameters(): param.requires_grad = False
kT1_backward = NNElement(8, 8, 8, 2, K0_backward).to(device)
kT1_backward.load_state_dict(torch.load(prefix+'etype1/trained_matrix_f2d.pth',map_location=device))
kT1_backward.eval()
for param in kT1_backward.parameters(): param.requires_grad = False
inputaveT1 = 1.0
outputaveT1 = 0.02705513178572368

kT2_forward = NNElement(8, 8, 8, 2, K0_forward).to(device)
kT2_forward.load_state_dict(torch.load(prefix+'etype2/trained_matrix_d2f.pth',map_location=device))
kT2_forward.eval()
for param in kT2_forward.parameters(): param.requires_grad = False
kT2_backward = NNElement(8, 8, 8, 2, K0_backward).to(device)
kT2_backward.load_state_dict(torch.load(prefix+'etype2/trained_matrix_f2d.pth',map_location=device))
kT2_backward.eval()
for param in kT2_backward.parameters(): param.requires_grad = False
inputaveT2 = 1.0
outputaveT2 = 0.03327096535815818

kT3_forward = NNElement_new_new(device, K0_forward, input_dim=8, output_dim=8, hidden_dim=8, L1_num=6, L2_num=3).to(device)
kT3_forward.load_state_dict(torch.load(prefix+'etype3/trained_matrix_d2f.pth',map_location=device))
kT3_forward.eval()
for param in kT3_forward.parameters(): param.requires_grad = False
kT3_backward = NNElement(8, 8, 8, 2, K0_backward).to(device)
kT3_backward.load_state_dict(torch.load(prefix+'etype3/trained_matrix_f2d.pth',map_location=device))
kT3_backward.eval()
for param in kT3_backward.parameters(): param.requires_grad = False
inputaveT3 = 1.0
outputaveT3 = 0.023693512810296576

kT4_forward = NNElement_new(device, K0_forward).to(device)
kT4_forward.load_state_dict(torch.load(prefix+'etype4/trained_matrix_d2f.pth',map_location=device))
kT4_forward.eval()
for param in kT4_forward.parameters(): param.requires_grad = False
kT4_backward = NNElement(8, 8, 8, 2, K0_backward).to(device)
kT4_backward.load_state_dict(torch.load(prefix+'etype4/trained_matrix_f2d.pth',map_location=device))
kT4_backward.eval()
for param in kT4_backward.parameters(): param.requires_grad = False
inputaveT4 = 1.0
outputaveT4 = 0.027081768129645023


class solveFENN(nn.Module):
    def __init__(self, device,\
                  kT1_forward, kT1_backward,\
                  kT2_forward, kT2_backward,\
                 kT3_forward, kT3_backward,\
                  kT4_forward, kT4_backward,\
                 node2elemT1, node2elemT2,\
                 node2elemT3, node2elemT4,\
                  inputaveT1, outputaveT1,\
                  inputaveT2, outputaveT2,\
                  inputaveT3, outputaveT3,\
                 inputaveT4, outputaveT4,\
                 var2all, elem2node_sum,\
                 all2load, all2internal,\
                 bc, load):
        
        super().__init__()
        self.kT1_forward = kT1_forward
        self.kT2_forward = kT2_forward
        self.kT3_forward = kT3_forward
        self.kT4_forward = kT4_forward
        
        self.kT1_backward = kT1_backward
        self.kT2_backward = kT2_backward
        self.kT3_backward = kT3_backward
        self.kT4_backward = kT4_backward
        
        self.node2elemT1 = torch.from_numpy(node2elemT1).float().to(device)
        self.node2elemT2 = torch.from_numpy(node2elemT2).float().to(device)
        self.node2elemT3 = torch.from_numpy(node2elemT3).float().to(device)
        self.node2elemT4 = torch.from_numpy(node2elemT4).float().to(device)
        
        self.inputaveT1 = inputaveT1
        self.inputaveT2 = inputaveT2
        self.inputaveT3 = inputaveT3
        self.inputaveT4 = inputaveT4
        
        self.outputaveT1 = outputaveT1
        self.outputaveT2 = outputaveT2
        self.outputaveT3 = outputaveT3
        self.outputaveT4 = outputaveT4
        
        self.var2all = torch.from_numpy(var2all).float().to(device)
        
        self.elem2node_sum = torch.from_numpy(elem2node_sum).float().to(device)
        
        self.all2load = torch.from_numpy(all2load).float().to(device)
        
        self.all2internal = torch.from_numpy(all2internal).float().to(device)
        
        self.bc = torch.from_numpy(bc).float().to(device)
        
        self.load = torch.from_numpy(load).float().to(device)
        
        dhalf = np.zeros((8, 4))
        dhalf[:4, :] = np.eye(4)
        self.dhalf = torch.from_numpy(dhalf).float().to(device)
        
        fhalf = np.zeros((8, 4))
        fhalf[4:, :] = np.eye(4)
        self.fhalf = torch.from_numpy(fhalf).float().to(device)

    def forward(self, x):
        # consistency, internal force == 0, surface force == load
        
        y0 = torch.matmul(x, self.var2all) + self.bc # unkown variables and bcs to all node
        
        y1_1 = torch.matmul(y0, self.node2elemT1)
        y2_1 = y1_1.view(-1, 8)
        y3_1 = self.kT1_forward(y2_1/self.inputaveT1)*self.outputaveT1
        
        y1_2 = torch.matmul(y0, self.node2elemT2)
        y2_2 = y1_2.view(-1, 8)
        y3_2 = self.kT2_forward(y2_2/self.inputaveT2)*self.outputaveT2
        
        y1_3 = torch.matmul(y0, self.node2elemT3)
        y2_3 = y1_3.view(-1, 8)
        y3_3 = self.kT3_forward(y2_3/self.inputaveT3)*self.outputaveT3
        
        y1_4 = torch.matmul(y0, self.node2elemT4)
        y2_4 = y1_4.view(-1, 8)
        y3_4 = self.kT4_forward(y2_4/self.inputaveT4)*self.outputaveT4
        
        y4 = torch.cat((y2_1, y2_2, y2_3, y2_4), 0)
        # y4 = torch.cat((y2_1, y2_2), 0)
        # y4 = y2_3
        
        y5 = torch.cat((y3_1, y3_2, y3_3, y3_4), 0)
        # y5 = torch.cat((y3_1, y3_2), 0)
        # y5 = y3_3
        
        y6 = torch.matmul(y5.view(-1), self.elem2node_sum)
        
        err_load = (torch.matmul(y6, self.all2load) - self.load)/0.027775344520955866
        
        err_internal = torch.matmul(y6, self.all2internal)/0.027775344520955866
        
        
        z1_1 = torch.cat((torch.matmul(y2_1/self.inputaveT1, self.dhalf), torch.matmul(y3_1/self.outputaveT1, self.fhalf)), 1)        
        z2_1 = self.kT1_backward(z1_1)*self.inputaveT1
        
        z1_2 = torch.cat((torch.matmul(y2_2/self.inputaveT2, self.dhalf), torch.matmul(y3_2/self.outputaveT2, self.fhalf)), 1)        
        z2_2 = self.kT2_backward(z1_2)*self.inputaveT2
        
        z1_3 = torch.cat((torch.matmul(y2_3/self.inputaveT3, self.dhalf), torch.matmul(y3_3/self.outputaveT3, self.fhalf)), 1)        
        z2_3 = self.kT3_backward(z1_3)
        
        z1_4 = torch.cat((torch.matmul(y2_4/self.inputaveT4, self.dhalf), torch.matmul(y3_4/self.outputaveT4, self.fhalf)), 1)        
        z2_4 = self.kT4_backward(z1_4)*self.inputaveT4
        
        z3 = torch.cat((z2_1, z2_2, z2_3, z2_4), 0)        
        # z3 = torch.cat((z2_1, z2_2), 0)        
        # z3 = z2_3
        
        err_consistency = (y4.view(-1) - z3.view(-1))
        
        out = torch.cat((err_load, err_internal, err_consistency), 0)
        
        return out, y6

lr0 = 1e-2 # initial
lrx = 1e-4 # last
nepochs = 15000
decayrate = np.exp(np.log(lrx/lr0)/nepochs)

x0 = np.ones(nvar)*0.3 #+ np.random.rand(nvar)*0.1
# x0 = np.array([0.32807019, 0.32807019, 0.32807019, 0.32807019])

# bc = np.array([0, 0.31692428, 0, 0.31692428, 0, 0.31692428, 0, 0.31692428])
# bc = np.array([0, 0.3, 0, 0.3, 0, 0.3, 0, 0.3])
# load = np.array([1.75e-2, 1.75e-2, 1.75e-2, 1.75e-2])

# tmpd = np.loadtxt('forward_sol.txt').reshape((-1,1))
# x0 = np.matmul(var2all, tmpd).flatten()

# heat flux and hidden variables
xin = Variable(torch.from_numpy(x0).float().to(device), requires_grad=True)
target = torch.from_numpy(np.zeros(nload + ninternal + 8*elem_num)).float().to(device)

model = solveFENN(device,\
                  kT1_forward, kT1_backward,\
                  kT2_forward, kT2_backward,\
                 kT3_forward, kT3_backward,\
                  kT4_forward, kT4_backward,\
                 node2elemT1, node2elemT2,\
                 node2elemT3, node2elemT4,\
                  inputaveT1, outputaveT1,\
                  inputaveT2, outputaveT2,\
                 inputaveT3, outputaveT3,\
                  inputaveT4, outputaveT4,\
                 var2all, elem2node_sum,\
                 all2load, all2internal,\
                 bc, load).to(device)

optimizer = optim.Adam([xin], lr=lr0)
# optimizer = optim.SGD([xin], lr=lr0, momentum=0.5)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decayrate)

loss_fn = nn.MSELoss()
##loss_fn = nn.L1Loss()

printstep = 10
loss_track = np.zeros((int(nepochs/printstep)+1, 2))
iprint = 0

for epoch in range(nepochs+1):
        y_, tmp_ = model(xin)
        loss = loss_fn(y_, target)
        if epoch%printstep == 0:
                print(epoch, loss.item())
                loss_track[iprint, 0] = epoch
                loss_track[iprint, 1] = loss.item()
                iprint += 1
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

dispvar = xin.detach().to(device).numpy()
disp = np.matmul(dispvar, var2all) + bc

generate_vtk(coor, conn, disp, prefix+'fenn_result')
