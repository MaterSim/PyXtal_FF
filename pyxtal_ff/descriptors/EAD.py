import math
import numpy as np
from ase.neighborlist import NeighborList
from .cutoff import Cutoff


class EAD:
    """EAD is an atom-centered descriptor that is inspired by Embedded Atom method (EAM).
    The EAM utilizes the orbital-dependent density components. The orbital-dependent
    component consists of a set of local atomic density descriptions.

    The functional form of EAD is consistent with:
        Zhang, Y., et. al. (2019). The Journal of Physical Chemistry Letters, 10(17), 4962-4967.

    Parameters
    ----------
    parameters: dict
        The user-defined parameters for component of local atomic density descriptions.
        i.e. {'L': 2, 'eta': [0.36], 'Rs': [1.0]}
    Rc: float
        The EAD will be calculated within this radius.
    derivative: bool
        If True, calculate the derivative of EAD.
    stress: bool
        If True, calculate the virial stress contribution of EAD.
    """
    def __init__(self, parameters, Rc=5., derivative=True, stress=False, cutoff='cosine'):
        self._type = 'EAD'
        self.dsize = int(1)
        self.parameters = {}
        
        # Set up keywords
        keywords = ['L', 'eta', 'Rs']
        for k, v in parameters.items():
            if k not in keywords:
                msg = f"{k} is not a valid key. "\
                        f"Choose from {keywords}"
                raise NotImplementedError(msg)
            else:
                if k == 'L':
                    self.dsize *= (v+1)
                    self.parameters[k] = v
                else:
                    self.dsize *= len(v)
                    self.parameters[k] = np.array(v)
        self.parameters['cutoff'] = cutoff
        
        self.Rc = Rc
        self.derivative = derivative
        self.stress = stress

    def __str__(self):
        s = "EAMD descriptor with Cutoff: {:6.3f}\n".format(self.Rc)
        for key in self.parameters.keys():
            s += "  {:s}: ".format(key)
            vals = self.parameters[key]
            if key in ['eta']:
                for val in vals:
                    s += "{:8.3f}, ".format(val)
            elif key in ['Rs']:
                for val in vals:
                    s += "{:6.3f}".format(val)
            else:
                s += "{:2d}".format(vals)
        s += "\n"

        return s

    def __repr__(self):
        return str(self)

    def load_from_dict(self, dict0):
        self.parameters = dict0["parameters"]
        self.Rc = dict0["Rc"]
        self.derivative = dict0["derivative"]
        self.stress = dict0["stress"]

    def save_dict(self):
        """
        save the model as a dictionary in json
        """
        dict = {
                "parameters": self.parameters,
                "rcut": self.rcut,
                "derivative": self.derivative,
                "stress": self.stress,
               }
        return dict
    
    def calculate(self, crystal, ids=None):
        """Calculate and return the EAD.
        
        Parameters
        ----------
        crystal: object
            ASE Structure object.
        ids: list
            A list of the centered atoms to be computed
            if None, all atoms will be considered
        
        Returns
        -------
        d: dict
            The user-defined EAD that represent the crystal.
            d = {'x': [N, d], 'dxdr': [N, m, d, 3], 'rdxdr': [N, m, d, 3, 3],
            'elements': list of elements}
        """
        self.crystal = crystal
        self.total_atoms = len(crystal) # total atoms in the unit cell
        vol = crystal.get_volume()
        
        rc = [self.Rc/2.]*self.total_atoms
        neighbors = NeighborList(rc, self_interaction=False, bothways=True, skin=0.)
        neighbors.update(crystal)

        unique_N = 0
        for i in range(self.total_atoms):
            indices, offsets = neighbors.get_neighbors(i)
            ith = 0
            if i not in indices:
                ith += 1
            unique_N += len(np.unique(indices))+ith # +1 is for i
            
        # Make numpy array here
        self.d = {'x': np.zeros([self.total_atoms, self.dsize]), 'elements': []}
        if self.derivative:
            self.d['dxdr'] = np.zeros([unique_N, self.dsize, 3])
            self.d['seq'] = np.zeros([unique_N, 2], dtype=int)
        if self.stress:
            self.d['rdxdr'] = np.zeros([unique_N, self.dsize, 3, 3])
        
        seq_count = 0

        if ids is None:
            ids = range(len(crystal))

        for i in ids: #range(self.total_atoms):
            element = crystal.get_chemical_symbols()[i]
            indices, offsets = neighbors.get_neighbors(i)
            Z = []  # atomic numbers of neighbors
            
            assert len(indices) > 0, \
            f"There's no neighbor for this structure at Rc = {self.Rc} A."

            Ri = crystal.get_positions()[i]
            total_neighbors = len(indices)

            Rj = np.zeros([total_neighbors, 3])
            IDs = np.zeros(total_neighbors, dtype=int)

            count = 0
            for j, offset in zip(indices, offsets):
                Rj[count, :] = crystal.positions[j] + np.dot(offset, crystal.get_cell())
                IDs[count] = j
                Z.append(crystal[j].number)
                count += 1
            Z = np.array(Z)

            Rij = Rj - Ri
            Dij = np.sqrt(np.sum(Rij**2, axis=1))

            d = calculate_eamd(i, self.total_atoms, Ri, Rij, Dij, Z, IDs, self.Rc, 
                               self.parameters, self.derivative, self.stress)
            
            self.d['x'][i] = d['x']
            if self.derivative:
                n_seq = len(d['seq'])
                self.d['dxdr'][seq_count:seq_count+n_seq] = d['dxdr']
                self.d['seq'][seq_count:seq_count+n_seq] = d['seq']
            if self.stress:
                self.d['rdxdr'][seq_count:seq_count+n_seq] = d['rdxdr']/vol
            
            if self.derivative:
                seq_count += n_seq
            
            self.d['elements'].append(element)

        return self.d


def calculate_eamd(i, m, ri, rij, dij, Z, IDs, Rc, parameters, derivative, stress):
    """ Calculate the EAD for a center atom i.
    
    Parameters
    ----------
    i: int
        The i-th atom center.
    m: int
        The total atoms in the crystal unit cell.
    ri: array [3]
        The position of atom i.
    rij: array [j, 3]
        The vector distances of atom i to neighbors j.
    dij: array [j]
        The array of distances of i-th center atom.
    Z: array [j]
        The atomic numbers of neighbors.
    IDs: int array [j]
        The indices of neighbors centering about atom i.
    Rc: float
        The cutoff radius.
    parameters: dict
        Rs: float array (d1)
            The shift from the center of the Gaussian-type orbitals.
        etas: float array (d2)
            The width of the Gaussian-type orbitals.
        L: int (d3)
            The total orbital angular momentum.
        cutoff: str
            The cutoff function.
    derivative:
        If True, calculate the derivative of EAD.
    stress: bool
        If True, calculate the virial stress contribution of EAD.

    Returns
    -------
    Dict of EAD descriptors with its derivative and stress contribution.
    """
    l_index = [1, 4, 10, 20]
    normalize = 1 / np.sqrt(np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 
                                      1, 2, 2, 2, 2, 2, 2, 6, 6, 6]))

    Rs = parameters['Rs']       # d1
    etas = parameters['eta']    # d2
    L = parameters['L']         # d3
    cutoff = Cutoff(parameters['cutoff'])
    d1, d2, d3, j = len(Rs), len(etas), L+1, len(dij)

    ij_list = i * np.ones([len(IDs), 2], dtype=int)
    ij_list[:, 1] = IDs

    unique_js = np.unique(IDs)
    if i not in unique_js:
        unique_js = np.append(i, unique_js)
    unique_js.sort()
    seq = i*np.ones([len(unique_js), 2], dtype=int)
    seq[:, 1] = unique_js
    uN = len(unique_js)
    _i = np.where(unique_js==i)[0][0]

    term1, d_term1, i_d_term1, j_d_term1 = get_xyz(unique_js, rij, ij_list, L, derivative=derivative) # [j, D3], [j, uN, 3, l]

    d0 = dij - Rs[:, np.newaxis]
    d02 = d0 ** 2 # [d1, j]

    fc = cutoff.calculate(dij, Rc) # [j]
    cj_cutoff = Z * fc # [j]
    term2_1 = np.exp(np.einsum('i,jk->ijk', -etas, d02)) # [d2, d1, j]
    term2 = np.einsum('ijk,k->ijk', term2_1, cj_cutoff) # [d2, d1, j]

    term = np.einsum('ij,kli->jkl', term1, term2) # [D3, d2, d1]
        
    if derivative:
        dterm0 = np.einsum('k, ijk->ijk', Z, term2_1).reshape([d1*d2, j]) # [d2*d1, j]
        dterm11 = np.einsum('ij, j->ij', dterm0, fc).reshape([d1*d2, j]) # [d2*d1, j]
        dterm1 = np.einsum('ij, jklm->jmilk', dterm11, d_term1) # [j, D3, d2*d1, 3, uN]
        i_dterm1 = np.einsum('ij, jklm->jmilk', dterm11, i_d_term1) 
        j_dterm1 = np.einsum('ij, jklm->jmilk', dterm11, j_d_term1) 
        
        dterm20 = np.einsum('ij, ki->jki', term1, dterm0) # [D3, d2*d1, j]
        dterm21 = cutoff.calculate_derivative(dij, Rc) # [j]
        _dterm22 = np.einsum('ij,j->ij', d0, fc) # [d1, j]
        dterm22 = 2 * np.einsum('i,jk->ijk', etas, _dterm22) # [d2, d1, j]
        dterm23 = (dterm21 - dterm22).reshape([d2*d1, j]) # [d2*d1, j]
        dterm24 = np.einsum('ijk, jk->ijk', dterm20, dterm23) # [D3, d2*d1, j]
        
        dRij_dRm = np.zeros([j, 3, uN])
        i_dRij_dRm = np.zeros([j, 3, uN])
        j_dRij_dRm = np.zeros([j, 3, uN])
        for mm, _m in enumerate(unique_js):
            mm_list = _m * np.ones([j, 1], dtype=int)
            dRij_dRm[:,:,mm], i_dRij_dRm[:,:,mm], j_dRij_dRm[:,:,mm] = \
                    dRij_dRm_norm(rij, np.hstack((ij_list, mm_list))) # [j, 3, uN]

        dterm2 = np.einsum('ijk, klm->kijlm', dterm24, dRij_dRm) # [j, D3, d2*d1, 3, uN]
        i_dterm2 = np.einsum('ijk, klm->kijlm', dterm24, i_dRij_dRm) # [j, D3, d2*d1, 3, uN]
        j_dterm2 = np.einsum('ijk, klm->kijlm', dterm24, j_dRij_dRm) # [j, D3, d2*d1, 3, uN]
        
        dphi_dRm = dterm1 + dterm2 # [j, D3, d2*d1, 3, uN]
        i_dphi_dRm = i_dterm1 + i_dterm2 
        j_dphi_dRm = j_dterm1 + j_dterm2 
        
        dterm = np.einsum('ij, hijkl->ijkl', term.reshape([term.shape[0], d2*d1]), dphi_dRm) # [D3, d2*d1, 3, uN]

        if stress:
            _RDXDR = np.zeros([term.shape[0], d2*d1, 3, uN, 3])  # [D3, d2*d1, 3, uN, 3]
            for count, ij in enumerate(ij_list):
                _j = np.where(unique_js==ij[1])[0][0]
                i_tmp = i_dphi_dRm[count, :, :, :]
                j_tmp = j_dphi_dRm[count, :, :, :]
                _RDXDR[:, :, :, _i, :] += np.einsum('ijk,l->ijkl', i_tmp[:,:,:,_i], ri)
                _RDXDR[:, :, :, _j, :] += np.einsum('ijk,l->ijkl', j_tmp[:,:,:,_j], rij[count]+ri)
            sterm = np.einsum('ij, ijklm->ijklm', term.reshape([term.shape[0], d2*d1]), _RDXDR)

    count = 0
    x = np.zeros([d3*d2*d1]) # [d3*d2*d1]
    dxdr, rdxdr = None, None
    if derivative:
        dxdr = np.zeros([uN, d1*d2*d3, 3]) # [uN, d3*d2*d1, 3]
    if stress:
        rdxdr = np.zeros([uN, d1*d2*d3, 3, 3])
    for l in range(L+1):
        Rc2l = Rc**(2*l)
        L_fac = math.factorial(l)
        x[count:count+d1*d2] = L_fac * np.einsum('ijk->jk', term[:l_index[l]] ** 2).ravel() / Rc2l
        if derivative:
            dxdr[:, count:count+d1*d2, :] = 2 * L_fac * np.einsum('ijkl->ljk', dterm[:l_index[l]]) / Rc2l
        if stress:
            rdxdr[:, count:count+d1*d2, :, :] = 2 * L_fac * np.einsum('ijklm->ljkm', sterm[:l_index[l]]) / Rc2l

        count += d1*d2

    return {'x': x, 'dxdr': dxdr, 'rdxdr': rdxdr, 'seq': seq}


def get_xyz(unique_js, rij, ij_list, L, derivative):
    """ (x ** l_x) * (y ** l_y) * (z ** l_z) / (l_x! * l_y! * l_z!) ** 0.5 """
    normalize = 1 / np.sqrt(np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2,     # 1 / sqrt(lx! ly! lz!)
                                      1, 2, 2, 2, 2, 2, 2, 6, 6, 6]))

    L_list = [[[0], [0]],                   # lx = 1, ly = 0, lz = 0; L = 1
              [[1], [1]],                   # lx = 0, ly = 1, lz = 0; L = 1
              [[2], [2]],                   # lx = 0, ly = 0, lz = 1; L = 1
              [[0,1], [0,1]],               # lx = 1, ly = 1, lz = 0; L = 2
              [[0,2], [0,2]],               # lx = 1, ly = 0, lz = 1; L = 2
              [[1,2], [1,2]],               # lx = 0, ly = 1, lz = 1; L = 2
              [[0], [3]],                   # lx = 2, ly = 0, lz = 0; L = 2
              [[1], [4]],                   # lx = 0, ly = 2, lz = 0; L = 2
              [[2], [5]],                   # lx = 0, ly = 0, lz = 2; L = 2
              [[0,1,2], [0,1,2]],           # lx = 1, ly = 1, lz = 1; L = 3
              [[1,2], [1,5]],               # lx = 0, ly = 1, lz = 2; L = 3
              [[1,2], [4,2]],               # lx = 0, ly = 2, lz = 1; L = 3
              [[0,2], [0,5]],               # lx = 1, ly = 0, lz = 2; L = 3
              [[0,1], [0,4]],               # lx = 1, ly = 2, lz = 0; L = 3
              [[0,1], [3,1]],               # lx = 2, ly = 1, lz = 0; L = 3
              [[0,2], [3,2]],               # lx = 2, ly = 0, lz = 1; L = 3
              [[0], [6]],                   # lx = 3, ly = 0, lz = 0; L = 3
              [[1], [7]],                   # lx = 0, ly = 3, lz = 0; L = 3
              [[2], [8]]                    # lx = 0, ly = 0, lz = 3; L = 3
              ]

    uN = len(unique_js)

    l = 1
    RIJ = np.zeros([len(rij), 9])
    dRIJ = np.zeros([len(rij), 9])

    if L == 1:
        l = 4
        RIJ[:, :3] = rij
        if derivative:
            dRIJ[:, :3] += 1
    
    elif L == 2:
        l = 10
        RIJ[:, :3] = rij
        RIJ[:, 3:6] = rij*rij
        if derivative:
            dRIJ[:, :3] += 1
            dRIJ[:, 3:6] = 2*rij

    elif L == 3:
        l = 20
        RIJ[:, :3] = rij
        RIJ[:, 3:6] = rij*rij
        RIJ[:, 6:9] = (rij*rij)*rij

        if derivative:
            dRIJ[:, :3] += 1
            dRIJ[:, 3:6] = 2*rij
            dRIJ[:, 6:9] = 3*RIJ[:, 3:6] 

    xyz = np.ones([len(rij), 3, l])
    if derivative:
        dxyz = np.zeros([len(rij), uN, 3, l])
        i_dxyz = np.zeros([len(rij), uN, 3, l])
        j_dxyz = np.zeros([len(rij), uN, 3, l])
    
    dij_dmlist, i_dij_dmlist, j_dij_dmlist = dij_dm_list(unique_js, ij_list) # [j, uN]

    for i in range(1, l):
        xyz[:, L_list[i-1][0], i] = RIJ[:, L_list[i-1][1]]
        if derivative: # [j, uN], [j, l] -> [j, uN, l]
            dxyz[:, :, L_list[i-1][0], i] = np.einsum('ij,ik->ijk', dij_dmlist, dRIJ[:, L_list[i-1][1]]) 
            i_dxyz[:, :, L_list[i-1][0], i] = np.einsum('ij,ik->ijk', i_dij_dmlist, dRIJ[:, L_list[i-1][1]])
            j_dxyz[:, :, L_list[i-1][0], i] = np.einsum('ij,ik->ijk', j_dij_dmlist, dRIJ[:, L_list[i-1][1]])

    result = xyz[:, 0, :] * xyz[:, 1, :] * xyz[:, 2, :] * normalize[:l] # [j, l]
    
    if derivative:
        d_result = np.zeros_like(dxyz) # [j, uN, 3, l]
        d_result[:, :, 0, :] = np.einsum('ijk,ik->ijk', dxyz[:, :, 0, :], xyz[:, 1, :]*xyz[:, 2, :])
        d_result[:, :, 1, :] = np.einsum('ijk,ik->ijk', dxyz[:, :, 1, :], xyz[:, 0, :]*xyz[:, 2, :])
        d_result[:, :, 2, :] = np.einsum('ijk,ik->ijk', dxyz[:, :, 2, :], xyz[:, 0, :]*xyz[:, 1, :])
        d_result = np.einsum('ijkl,l->ijkl', d_result, normalize[:l])

        i_d_result = np.zeros_like(dxyz) # [j, uN, 3, l]
        i_d_result[:, :, 0, :] = np.einsum('ijk,ik->ijk', i_dxyz[:, :, 0, :], xyz[:, 1, :]*xyz[:, 2, :])
        i_d_result[:, :, 1, :] = np.einsum('ijk,ik->ijk', i_dxyz[:, :, 1, :], xyz[:, 0, :]*xyz[:, 2, :])
        i_d_result[:, :, 2, :] = np.einsum('ijk,ik->ijk', i_dxyz[:, :, 2, :], xyz[:, 0, :]*xyz[:, 1, :])
        i_d_result = np.einsum('ijkl,l->ijkl', i_d_result, normalize[:l])

        j_d_result = np.zeros_like(dxyz) # [j, uN, 3, l]
        j_d_result[:, :, 0, :] = np.einsum('ijk,ik->ijk', j_dxyz[:, :, 0, :], xyz[:, 1, :]*xyz[:, 2, :])
        j_d_result[:, :, 1, :] = np.einsum('ijk,ik->ijk', j_dxyz[:, :, 1, :], xyz[:, 0, :]*xyz[:, 2, :])
        j_d_result[:, :, 2, :] = np.einsum('ijk,ik->ijk', j_dxyz[:, :, 2, :], xyz[:, 0, :]*xyz[:, 1, :])
        j_d_result = np.einsum('ijkl,l->ijkl', j_d_result, normalize[:l])
       
        return result, d_result, i_d_result, j_d_result
    else:
        return result, None, None, None


def dij_dm_list(unique_js, ij_list):
    """Get the sign of the derivative of x-y-z ** lx-ly-lz. 
    
    Parameters
    ----------
    uN: int
        the unique index of the atom that force is acting on.
    ij_list: list
        The list of center atom i w.r.t. the neighbors atom j.

    Returns
    -------
    result: array [j, uN]
        The signs (+ or -) for dXij_dm (YZ) * dYij_dm (XZ) * dZij_dm (XY)
    """
    uN = len(unique_js)
    result = np.zeros([len(ij_list), uN])
    i_result = np.zeros([len(ij_list), uN])
    j_result = np.zeros([len(ij_list), uN])

    ijm_list = np.zeros([len(ij_list), 3, uN], dtype=int)
    ijm_list[:, -1, :] = unique_js 
    ijm_list[:, :2, :] = np.broadcast_to(ij_list[...,None], ij_list.shape+(uN,))

    arr = (ijm_list[:, 2, :] == ijm_list[:, 0, :])
    result[arr] = -1
    i_result[arr] = -1

    arr = (ijm_list[:, 2, :] == ijm_list[:, 1, :])
    result[arr] = 1
    j_result[arr] = 1

    arr = (ijm_list[:, 0, :] == ijm_list[:, 1, :])  
    result[arr] = 0                                 

    return result, i_result, j_result # [j, uN]


def dRij_dRm_norm(Rij, ijm_list):
    """Calculate the derivative of Rij norm w. r. t. atom m. This term affects 
    only on i and j.
    
    Parameters
    ----------
    Rij : array [j, 3]
        The vector distances of atom i to atom j.
    ijm_list: array [j, 3] or [j*k, 3]
        Id list of center atom i, neighbors atom j, and atom m.
    
    Returns
    -------
    dRij_m: array [j, 3]
        The derivative of pair atoms w.r.t. atom m in x, y, z directions.
    """
    dRij_m = np.zeros([len(Rij), 3])
    i_dRij_m = np.zeros([len(Rij), 3])
    j_dRij_m = np.zeros([len(Rij), 3])
    R1ij = np.linalg.norm(Rij, axis=1).reshape([len(Rij),1])

    l1 = (ijm_list[:,2]==ijm_list[:,0])
    dRij_m[l1, :] = -Rij[l1]/R1ij[l1]
    i_dRij_m[l1, :] = -Rij[l1]/R1ij[l1]

    l2 = (ijm_list[:,2]==ijm_list[:,1])
    dRij_m[l2, :] = Rij[l2]/R1ij[l2]
    j_dRij_m[l2, :] = Rij[l2]/R1ij[l2]

    l3 = (ijm_list[:,0]==ijm_list[:,1])
    dRij_m[l3, :] = 0
    
    return dRij_m, i_dRij_m, j_dRij_m

    
if __name__ == '__main__':
    import time
    from ase.build import bulk
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

    Rc = 10
    parameters1 = {'L': 2, 'eta': [0.036, 0.071], 'Rs': [0]}

    # Test for stress
    for a in [5.0]: #, 5.4, 5.8]:
        si = bulk('Si', 'diamond', a=a, cubic=True)
        cell = si.get_cell()
        cell[0,1] += 0.1
        si.set_cell(cell)

        bp = EAD(parameters1, Rc=Rc, derivative=True, stress=True, cutoff='cosine')
        des = bp.calculate(si)
        
        print("G:", des['x'])
        print("GPrime")
        print(des['dxdr'])
        #print(des['rdxdr'][0:8, -1, :, :])
        #pprint(np.einsum('ijklm->klm', des['rdxdr']))
