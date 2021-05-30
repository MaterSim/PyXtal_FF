import numpy as np
from ase.neighborlist import NeighborList

factor = 14.399645306701126

class ZBL:
    """A class for calculating the Ziegler-Biersack-Littmark (ZBL) 
    screened nuclear repulsion for describing high-energy collisions between atoms.

    Source: J.F. Ziegler, J. P. Biersack and U. Littmark, 
            “The Stopping and Range of Ions in Matter,” Volume 1, Pergamon, 1985.

    Parameters
    ----------
    inner: float
        distance where switching function begins.
    outer: float
        global cutoff for ZBL interaction.
    atomic_energy: bool
        If True, atomic_energy will be recorded and returned.
    """
    def __init__(self, inner, outer, atomic_energy=False):
        self.inner = inner
        self.outer = outer
        self.atomic_energy = atomic_energy

    def calculate(self, crystal):
        """ The calculate function.
        
        Parameters
        ----------
        crystal: object
            ASE Structure object.

        Returns
        -------
        results: dict
            The energy, forces, and stress of ZBL contribution.
        """
        self.crystal = crystal
        self.total_atoms = len(crystal)
        vol = crystal.get_volume()

        rc = [(2.0+self.outer)/2.] * self.total_atoms
        neighbors = NeighborList(rc, self_interaction=False, bothways=False, skin=0.)
        #neighbors = NeighborList(rc, self_interaction=False, bothways=True, skin=0)
        neighbors.update(crystal)

        self.result = {'energy': 0, 'force': np.zeros([self.total_atoms, 3]),
                       'stress': np.zeros([3,3]),
                       'energies': None}
        if self.atomic_energy:
            self.result['energies'] = []

        # Get unique Zi and Zj
        elements = []
        for i in range(self.total_atoms):
            if crystal[i].number not in elements:
                elements.append(crystal[i].number)

        # Get A, B, and C coefficients
        ABC = {}
        for i in range(len(elements)):
            abc = get_ABC_coefficients(elements[i], elements[i], self.outer, self.inner)
            ABC[(elements[i], elements[i])] = abc
            for j in range(i+1, len(elements)):
                abc = get_ABC_coefficients(elements[i], elements[j], self.outer, self.inner)
                ABC[(elements[i], elements[j])] = abc
                ABC[(elements[j], elements[i])] = abc

        for i in range(self.total_atoms):
            element = crystal.get_chemical_symbols()[i]
            elementi = crystal[i].number
            indices, offsets = neighbors.get_neighbors(i)
            Zi, Zj, ABCij = crystal[i].number, [], []
            
            if len(indices) > 0:
                Ri = crystal.get_positions()[i]
                total_neighbors = len(indices)

                Rj = np.zeros([total_neighbors, 3])
                IDs = np.zeros(total_neighbors, dtype=int)

                count = 0
                for j, offset in zip(indices, offsets):
                    Rj[count, :] = crystal.positions[j] + np.dot(offset, crystal.get_cell())
                    IDs[count] = j
                    elementj = crystal[j].number
                    Zj.append(elementj)
                    ABCij.append(ABC[(elementi, elementj)])
                    count += 1
                Zj = np.array(Zj)
                ABCij = np.array(ABCij)

                Rij = Rj - Ri
                Dij = np.sqrt(np.sum(Rij**2, axis=1))

                energy, forces, stress = calculate_ZBL(i, Ri, Rij, Dij, Zi, Zj, self.outer, self.inner, ABCij, self.total_atoms, IDs)
                self.result['energy'] += energy
                self.result['force'] += forces
                self.result['stress'] += stress
                if self.atomic_energy:
                    self.result['energies'].append(energy)
            else:
                if self.atomic_energy:
                    self.result['energies'].append(0.)
        
        if self.atomic_energy:
            self.result['_energies'] = np.array(self.result['energies'])
        self.result['stress'] /= vol
        self.result['stress'] = self.result['stress'].ravel()[[0,4,8,1,2,5]]

        return self.result


def calculate_ZBL(i, ri, rij, dij, Zi, Zj, r_outer, r_inner, ABC, total_atoms, IDs, derivative=True):
    """Calculate the atomic ZBL energy, force, and stress.

    Parameters
    ----------
    i: int
        The i-th center atom
    rij: float array [j,3]
        The vector distances of atom i to neighbors js.
    dij: float array [j]
        The distance between atom i and neighbors js.
    Zi: int
        The atomic number of atom i.
    Zj: int array
        The atomic number of atom j.
    r_outer: float
        global cutoff for ZBL interaction.
    r_inner: float
        distance where switching function begins.
    total_atoms: int
        The total atom in the unit cell
    IDs: int array [j]
        The indices of neighbors centering about atom i.
    """
    ids1 = (dij < r_outer)

    if True not in ids1:
        return 0., np.zeros([total_atoms, 3]), np.zeros([3,3])
    else:
        _i = i 
        ij_list = i * np.ones([len(IDs), 2], dtype=int)
        ij_list[:, 1] = IDs

        energy = np.zeros([len(dij)])
        forces = np.zeros([total_atoms, 3])
        stress = np.zeros([total_atoms, len(ij_list), 3, 3])
        
        kZi = factor * Zi
        kZiZj = kZi * Zj
        dij_inv = 1 / dij
        Zi_inv = Zi ** 0.23 / 0.46850
        Zj_inv = Zj ** 0.23 / 0.46850
        a_inv = Zi_inv + Zj_inv
        x = dij * a_inv
        
        exp1 = np.exp(-3.19980 * x)
        exp2 = np.exp(-0.94229 * x)
        exp3 = np.exp(-0.40290 * x)
        exp4 = np.exp(-0.20162 * x)
        phi_ij = 0.18175 * exp1 + 0.50986 * exp2 + 0.28022 * exp3 + 0.02817 * exp4
        Eij = kZiZj * dij_inv * phi_ij 
        
        # Switching function
        if derivative:
            dSA = ABC[:,0] * (dij - r_inner) ** 2
            dSB = ABC[:,1] * (dij - r_inner) ** 3
            SA = 0.333333333 * dSA * (dij - r_inner)
            SB = 0.25 * dSB * (dij - r_inner)
            SC = ABC[:,2]
        else:
            SA = 0.333333333 * ABC[:,0] * (dij - r_inner) ** 3
            SB = 0.25 * ABC[:,1] * (dij - r_inner) ** 4
            SC = ABC[:,2]
                
        # Collecting atomic energy
        energy[ids1] += SC[ids1] + Eij[ids1]
        #ids2 = (dij <= r_outer) & (dij > r_inner)
        #ids3 = (dij <= r_outer)
        ids2 = (dij < r_outer) & (dij > r_inner)
        ids3 = (dij < r_outer)
        energy[ids2] += SA[ids2] + SB[ids2]

        # Force
        dE1_ddij = -Eij * dij_inv
        dphi_ddij = (-0.18175 * 3.19980 * exp1 - 0.50986 * 0.94229 * exp2 - \
                      0.28022 * 0.40290 * exp3 - 0.02817 * 0.20162 * exp4) * a_inv
        dE2_ddij = kZiZj * dij_inv * dphi_ddij
        dE_ddij = dE1_ddij + dE2_ddij
        dE_ddij[ids2] += dSA[ids2] + dSB[ids2]

        dRij_dRm = np.zeros([len(dij), 3, total_atoms])
        i_dRij_dRm = np.zeros([len(dij), 3, total_atoms])
        j_dRij_dRm = np.zeros([len(dij), 3, total_atoms])
        for mm in range(total_atoms):
            mm_list = mm * np.ones([len(dij), 1], dtype=int)
            dRij_dRm[:,:,mm], i_dRij_dRm[:,:,mm], j_dRij_dRm[:,:,mm] = dRij_dRm_norm(rij, np.hstack((ij_list, mm_list))) # [j,3,N]

        forces -= np.einsum('ijk,i->kj', dRij_dRm[ids3], dE_ddij[ids3])
        i_force = np.einsum('ijk,i->ikj', i_dRij_dRm, dE_ddij) # [j, N, 3]
        j_force = np.einsum('ijk,i->ikj', j_dRij_dRm, dE_ddij) # [j, N, 3]
        
        # Stress
        for count, ij in enumerate(ij_list):
            _j = ij[1]
            stress[_i, count, :, :] += np.einsum('i,j->ij', i_force[count,_i,:], ri)
            stress[_j, count, :, :] += np.einsum('i,j->ij', j_force[count,_j,:], rij[count]+ri)
        
        stress = -1 * np.einsum('ijkl->kl', stress[:, (dij < r_outer)])
            
        return np.sum(energy), forces, stress


def get_ABC_coefficients(Zi, Zj, r_outer, r_inner):
    """A function to get the switching A, B, and C coefficients."""
    kZiZj = factor * Zi * Zj
    a_inv = (Zi ** 0.23 + Zj ** 0.23) / 0.46850
    r = r_outer - r_inner
    x = r_outer * a_inv

    exp1 = np.exp(-3.19980 * x)
    exp2 = np.exp(-0.94229 * x)
    exp3 = np.exp(-0.40290 * x)
    exp4 = np.exp(-0.20162 * x)
    
    phi = 0.18175 * exp1 + 0.50986 * exp2 + 0.28022 * exp3 + 0.02817 * exp4
    phiP = (-0.18175 * 3.19980 * exp1 - 0.50986 * 0.94229 * exp2 - \
             0.28022 * 0.40290 * exp3 - 0.02817 * 0.20162 * exp4) * a_inv
    phiDP = (0.18175 * 3.19980 ** 2 * exp1 + 0.50986 * 0.94229 ** 2 * exp2 + \
             0.28022 * 0.40290 ** 2 * exp3 + 0.02817 * 0.20162 ** 2 * exp4) * (a_inv ** 2)

    E = kZiZj * (1 / r_outer) * phi
    EP = kZiZj * (-1 / r_outer ** 2 * phi + 1 / r_outer * phiP)
    EDP = kZiZj * (2 / r_outer ** 3 * phi - 2 / r_outer ** 2 * phiP + 1 / r_outer * phiDP)

    A = (-3 * EP + r * EDP) / r ** 2
    B = (2 * EP - r * EDP) / r ** 3
    C = -E + 0.5 * r * EP - (1/12) * r ** 2 * EDP
    
    return [A, B, C]


def dRij_dRm_norm(Rij, ijm_list):
    """Calculate the derivative of Rij norm w. r. t. atom m. This term affects
    only on i and j.

    Parameters
    ----------
    Rij : array [j, 3] or [j*k, 3]
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
    from ase.io import read
    np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

    t0 = time.time()
    #struc = read("MOD_NiMo.cif")
    #struc = read("MOD_NiMo_real.cif")
    for inner in [1.0, 1.5, 2.0, 2.5, 3.0]:
        print("inner==================", inner)
        for a in [2.0, 3.0, 4.0, 4.5, 5.0, 5.5]:
            struc = bulk('Si', 'diamond', a=a, cubic=True)
            zbl = ZBL(inner, 4.0)
            d = zbl.calculate(struc)
            energy = d['energy'] / len(struc)
            forces = d['force']
            stress = d['stress'] #* 1602176.6208

            #print("Energy: ", energy)
            #print("Force: ")
            #print(forces)
            print(a, "Stress: ", stress)
    t1 = time.time()
    print("\nTime: ", round(t1-t0, 6), "s")
