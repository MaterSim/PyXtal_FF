import numpy as np
from ase.neighborlist import NeighborList

# 8.9875517923e9 (Coulomb constant) * 1.602176634e-19 ^ 2 (electron charge) * 1e-10 (m to A) * 6.241509e18 (J to eV)
factor = 14.399645306701126

class ZBL:
    def __init__(self, inner, outer):
        self.inner = inner
        self.outer = outer

    def calculate(self, crystal):
        self.crystal = crystal
        self.total_atoms = len(crystal)
        vol = crystal.get_volume()

        rc = [(2.0+self.outer)/2.] * self.total_atoms
        neighbors = NeighborList(rc, self_interaction=False, bothways=False, skin=0.)
        neighbors.update(crystal)

        self.result = {'energy': 0, 'force': np.zeros([self.total_atoms, 3]),
                       'stress': np.zeros([3,3])}

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
                elementj = crystal[j].number
                Zj.append(elementj)
                ABCij.append(ABC[(elementi, elementj)])
                count += 1
            Zj = np.array(Zj)
            ABCij = np.array(ABCij)

            Rij = Rj - Ri
            Dij = np.sqrt(np.sum(Rij**2, axis=1))

            energy, forces, stress = calculate_ZBL(i, Rij, Dij, Zi, Zj, self.outer, self.inner, ABCij, self.total_atoms, IDs)
            self.result['energy'] += energy
            self.result['force'] += forces
            self.result['stress'] += stress

        self.result['stress'] /= vol

        return self.result


def calculate_ZBL(i, rij, dij, Zi, Zj, r_outer, r_inner, ABC, total_atoms, IDs, derivative=True, stress_derivative=True):
    ids1 = (dij <= r_outer)

    if True not in ids1:
        return 0., np.zeros([total_atoms, 3]), np.zeros([3,3])
    else:
        #import time
        #t0 = time.time()
        ij_list = i * np.ones([len(IDs), 2], dtype=int)
        ij_list[:, 1] = IDs
        unique_js = np.unique(IDs)
        if i not in unique_js:
            unique_js = np.append(i, unique_js)
        unique_js.sort()
        seq = i*np.ones([len(unique_js), 2], dtype=int)
        seq[:, 1] = unique_js
        uN = len(unique_js)
        #t1 = time.time()
        #print("initial", t1-t0)

        energy = np.zeros([len(dij)])
        forces = np.zeros([total_atoms, 3])
        stress = np.zeros([3,3])
        
        #t0 = time.time()
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
        ids2 = (dij <= r_outer) & (dij > r_inner)
        energy[ids2] += SA[ids2] + SB[ids2]
        #t1 = time.time()
        #print("Energy", t1-t0)


        # Force
        #t0 = time.time()
        dE1_ddij = -Eij * dij_inv
        dphi_ddij = (-0.18175 * 3.19980 * exp1 - 0.50986 * 0.94229 * exp2 - \
                      0.28022 * 0.40290 * exp3 - 0.02817 * 0.20162 * exp4) * a_inv
        dE2_ddij = kZiZj * dij_inv * dphi_ddij
        dE_ddij = dE1_ddij + dE2_ddij + dSA + dSB # [j]

        dRij_dRm = np.zeros([len(dij), 3, uN])
        for mm, _m in enumerate(unique_js):
            mm_list = _m * np.ones([len(dij), 1], dtype=int)
            dRij_dRm[:,:,mm] = dRij_dRm_norm(rij, np.hstack((ij_list, mm_list))) # [j, 3, uN]
        #force = np.einsum('ijk,i->ikj', dRij_dRm[ids2], dE_ddij[ids2]) # [j,uN,3]
        #forces -= np.einsum('ijk->jk', force)
        force = np.einsum('ijk,i->ikj', dRij_dRm, dE_ddij) # [j,uN,3]
        forces -= np.einsum('ijk->jk', force[ids2])
        #t1 = time.time()
        #print("force", t1-t0)

        # Stress
        #t0 = time.time()
        for count, ij in enumerate(ij_list):
            if dij[count] <= r_outer and dij[count] > r_inner:
                _j = np.where(unique_js==ij[1])[0][0]
                stress -= np.einsum('i,j->ij', force[count,_j,:], rij[count])
        #t1 = time.time()
        #print("stress", t1-t0)
        #import sys; sys.exit()
    
        return np.sum(energy), forces, stress


def get_ABC_coefficients(Zi, Zj, r_outer, r_inner):
    import time
    t0 =time.time()
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
    t1 = time.time()
    print(t1-t0)

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
    dRij_m: array [j, 3] or [j*k, 3]
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

if __name__ == '__main__':
    import time
    from ase.build import bulk
    from ase.io import read
    np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

    #struc = read("MOD_NiMo.cif")
    struc = read("MOD_NiMo_real.cif")
    t0 = time.time()
    #zbl = ZBL(2.0, 3.5)
    zbl = ZBL(2.0, 10.5)
    d = zbl.calculate(struc)
    energy = d['energy'] / len(struc)
    forces = d['force']
    stress = d['stress']
    t1 = time.time()

    print("Energy: ", energy)
    print("Force: ")
    print(forces)
    print("\nStress: ")
    print(stress)
    print("\nTime: ", round(t1-t0, 6), "s")
