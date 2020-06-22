import math
import numpy as np
from ase.neighborlist import NeighborList


class EAMD:
    """EAMD is an atom-centered descriptor that is inspired by Embedded Atom method (EAM).
    The EAM utilizes the orbital-dependent density components. The orbital-dependent
    component consists of a set of local atomic density descriptions.

    The functional form of EAMD is consistent with:
        Zhang, Y., et. al. (2019). The Journal of Physical Chemistry Letters, 10(17), 4962-4967.

    Parameters
    ----------
    parameters: dict
        The user-defined parameters for component of local atomic density descriptions.
        i.e. {'L': 2, 'eta': [0.36], 'Rs': [1.0]}
    Rc: float
        The EAMD will be calculated within this radius.
    derivative: bool
        If True, calculate the derivative of EAMD.
    stress: bool
        If True, calculate the virial stress contribution of EAMD.
    """
    def __init__(self, parameters, Rc=5., derivative=True, stress=False):
        self._type = 'EAMD'
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

        self.Rc = Rc
        self.derivative = derivative
        self.stress = stress

    
    def calculate(self, crystal):
        """Calculate and return the EAMD.
        
        Parameters
        ----------
        crystal: object
            ASE Structure object.
        
        Returns
        -------
        d: dict
            The user-defined EAMD that represent the crystal.
            d = {'x': [N, d], 'dxdr': [N, m, d, 3], 'rdxdr': [N, m, d, 3, 3],
            'elements': list of elements}
        """

        self.crystal = crystal
        self.total_atoms = len(crystal) # total atoms in the unit cell
        vol = crystal.get_volume()
        
        # Make numpy array here
        self.d = {'x': np.zeros([self.total_atoms, self.dsize]), 'elements': []}
        if self.derivative:
            self.d['dxdr'] = np.zeros([self.total_atoms, self.total_atoms, self.dsize, 3])
        if self.stress:
            self.d['rdxdr'] = np.zeros([self.total_atoms, self.total_atoms, self.dsize, 3, 3])

        rc = [self.Rc/2.]*self.total_atoms
        neighbors = NeighborList(rc, self_interaction=False, bothways=True, skin=0.)
        neighbors.update(crystal)

        for i in range(self.total_atoms):
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

            d = calculate_eamd(i, self.total_atoms, Rij, Dij, Z, IDs, self.Rc, 
                               self.parameters, self.derivative, self.stress)

            self.d['x'][i] = d['x']
            if self.derivative:
                self.d['dxdr'][i] = d['dxdr']
            if self.stress:
                self.d['rdxdr'][i] = -d['rdxdr']/vol
            
            self.d['elements'].append(element)

        return self.d


def calculate_eamd(i, m, rij, dij, Z, IDs, Rc, parameters, derivative, stress):
    """ Calculate the EAMD for a center atom i.
    
    Parameters
    ----------
    i: int
        The i-th atom center.
    m: int
        The total atoms in the crystal unit cell.
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
    derivative:
        If True, calculate the derivative of EAMD.
    stress: bool
        If True, calculate the virial stress contribution of EAMD.

    Returns
    -------
    Dict of EAMD descriptors with its derivative and stress contribution.
    """
    l_index = [1, 4, 10, 20]
    normalize = 1 / np.sqrt(np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 
                                      1, 2, 2, 2, 2, 2, 2, 6, 6, 6]))

    Rs = parameters['Rs']       # d1
    etas = parameters['eta']    # d2
    L = parameters['L']         # d3
    d1, d2, d3, j = len(Rs), len(etas), L+1, len(dij)

    ij_list = i * np.ones([len(IDs), 2], dtype=int)
    ij_list[:, 1] = IDs

    term1, d_term1 = get_xyz(m, rij, ij_list, L, derivative=derivative) # [j, D3], [j, m, 3, l]

    d0 = dij - Rs[:, np.newaxis]
    d02 = d0 ** 2 # [d1, j]

    fc = Cosine(dij, Rc) # [j]
    cj_cutoff = Z * fc # [j]
    term2_1 = np.exp(np.einsum('i,jk->ijk', -etas, d02)) # [d2, d1, j]
    term2 = np.einsum('ijk,k->ijk', term2_1, cj_cutoff) # [d2, d1, j]

    term = np.einsum('ij,kli->jkl', term1, term2) # [D3, d2, d1]
        
    if derivative:
        dterm0 = np.einsum('k, ijk->ijk', Z, term2_1).reshape([d1*d2, j]) # [d2*d1, j]
        dterm11 = np.einsum('ij, j->ij', dterm0, fc).reshape([d1*d2, j]) # [d2*d1, j]
        dterm1 = np.einsum('ij, jklm->jmilk', dterm11, d_term1) # [D3, d2*d1, 3, m]

        dterm20 = np.einsum('ij, ki->jki', term1, dterm0) # [D3, d2*d1, j]
        dterm21 = CosinePrime(dij, Rc) # [j]
        _dterm22 = np.einsum('ij,j->ij', d0, fc) # [d1, j]
        dterm22 = 2 * np.einsum('i,jk->ijk', etas, _dterm22) # [d2, d1, j]
        dterm23 = (dterm21 - dterm22).reshape([d2*d1, j]) # [d2*d1, j]
        dterm24 = np.einsum('ijk, jk->ijk', dterm20, dterm23) # [D3, d2*d1, j]
        
        dRij_dRm = np.zeros([j, 3, m])
        for mm in range(m):
            mm_list = mm * np.ones([j, 1], dtype=int)
            dRij_dRm[:,:,mm] = dRij_dRm_norm(rij, np.hstack((ij_list, mm_list))) # [j, 3, m]

        dterm2 = np.einsum('ijk, klm->kijlm', dterm24, dRij_dRm) # [j, D3, d2*d1, 3, m]
        
        dphi_dRm = dterm1 + dterm2 # [j, D3, d2*d1, 3, m]
        dterm = np.einsum('ij, hijkl->ijkl', term.reshape([term.shape[0], d2*d1]), dphi_dRm) # [D3, d2*d1, 3, m]
    
    if stress:
        _RDXDR = np.zeros([term.shape[0], d2*d1, 3, m, 3])  # [D3, d2*d1, 3, m, 3]
        for count, ij in enumerate(ij_list):
            _j = ij[1]
            tmp = dphi_dRm[count, :, :, :, _j]
            _RDXDR[:, :, :, _j, :] += np.einsum('ijk,l->ijkl', tmp, rij[count])
        
        sterm = np.einsum('ij, ijklm->ijklm', term.reshape([term.shape[0], d2*d1]), _RDXDR)
            
    count = 0
    x = np.zeros([d3*d2*d1]) # [d3*d2*d1]
    dxdr, rdxdr = None, None
    if derivative:
        dxdr = np.zeros([m, d1*d2*d3, 3]) # [m, d3*d2*d1, 3]
    if stress:
        rdxdr = np.zeros([m, d1*d2*d3, 3, 3])
    for l in range(L+1):
        Rc2l = Rc**(2*l)
        L_fac = math.factorial(l)
        x[count:count+d1*d2] = L_fac * np.einsum('ijk->jk', term[:l_index[l]] ** 2).ravel()/Rc2l
        if derivative:
            dxdr[:, count:count+d1*d2, :] = 2 * L_fac * np.einsum('ijkl->ljk', dterm[:l_index[l]])/Rc2l
        if stress:
            rdxdr[:, count:count+d1*d2, :, :] = 2 * L_fac * np.einsum('ijklm->ljkm', sterm[:l_index[l]])/Rc2l

        count += d1*d2

    return {'x': x, 'dxdr': dxdr, 'rdxdr': rdxdr}


def get_xyz(m, rij, ij_list, L, derivative):
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
        dxyz = np.zeros([len(rij), m, 3, l])
    
    dij_dmlist = dij_dm_list(m, ij_list)

    for i in range(1, l):
        xyz[:, L_list[i-1][0], i] = RIJ[:, L_list[i-1][1]]
        if derivative:
            dxyz[:, :, L_list[i-1][0], i] = np.einsum('ij,ik->ijk', dij_dmlist, dRIJ[:, L_list[i-1][1]])

    result = xyz[:, 0, :] * xyz[:, 1, :] * xyz[:, 2, :] * normalize[:l] # [j, l]
    
    if derivative:
        d_result = np.zeros_like(dxyz)
        d_result[:, :, 0, :] = np.einsum('ijk,ik->ijk', dxyz[:, :, 0, :], xyz[:, 1, :]*xyz[:, 2, :])
        d_result[:, :, 1, :] = np.einsum('ijk,ik->ijk', dxyz[:, :, 1, :], xyz[:, 0, :]*xyz[:, 2, :])
        d_result[:, :, 2, :] = np.einsum('ijk,ik->ijk', dxyz[:, :, 2, :], xyz[:, 0, :]*xyz[:, 1, :])

        d_result = np.einsum('ijkl,l->ijkl', d_result, normalize[:l])
        
        return result, d_result
    else:
        return result, None


def dij_dm_list(m, ij_list):
    """Get the sign of the derivative of x-y-z ** lx-ly-lz. 
    
    Parameters
    ----------
    m: int
        the index of the atom that force is acting on.
    ij_list: list
        The list of center atom i w.r.t. the neighbors atom j.

    Returns
    -------
    result: array [j, m]
        The signs (+ or -) for dXij_dm (XZ) * dYij_dm (XZ) * dZij_dm (XY)
    """
    result = np.zeros([len(ij_list), m])
    ijm_list = np.zeros([len(ij_list), 3, m], dtype=int)
    
    ijm_list[:, -1, :] = np.arange(m)
    ijm_list[:, :2, :] = np.broadcast_to(ij_list[...,None], ij_list.shape+(m,))

    arr1 = (ijm_list[:, 2, :] == ijm_list[:, 0, :])
    result[arr1] = -1

    arr2 = (ijm_list[:, 2, :] == ijm_list[:, 1, :])
    result[arr2] = 1

    arr3 = (ijm_list[:, 0, :] == ijm_list[:, 1, :])
    result[arr3] = 0

    return result # [j, m]


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

    R1ij = np.linalg.norm(Rij, axis=1).reshape([len(Rij),1])
    l1 = (ijm_list[:,2]==ijm_list[:,0])
    dRij_m[l1, :] = -Rij[l1]/R1ij[l1]
    l2 = (ijm_list[:,2]==ijm_list[:,1])
    dRij_m[l2, :] = Rij[l2]/R1ij[l2]
    l3 = (ijm_list[:,0]==ijm_list[:,1])
    dRij_m[l3, :] = 0
    
    return dRij_m

    
############################# Cutoff Functionals ##############################


def Cosine(dij, rc):
    ids = (dij > rc)
    result = 0.5 * (np.cos(np.pi * dij / rc) + 1.)
    result[ids] = 0.
    return result


def CosinePrime(Rij, Rc):
    # Rij is the norm
    ids = (Rij > Rc)
    result = -0.5 * np.pi / Rc * np.sin(np.pi * Rij / Rc)
    result[ids] = 0
    
    return result

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

        bp = EAMD(parameters1, Rc=Rc, derivative=True, stress=True)
        des = bp.calculate(si)
        
        print("G:", des['x'][0])
        print("GPrime")
        print(des['dxdr'][0,:,:,2])
        #print(des['rdxdr'][0:8, -1, :, :])
        #pprint(np.einsum('ijklm->klm', des['rdxdr']))

