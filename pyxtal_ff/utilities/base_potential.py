import numpy as np
from ase.neighborlist import NeighborList

# 8.9875517923e9 (Coulomb constant) * 1.602176634e-19 ^ 2 (electron charge) * 1e-10 (m to A) * 6.241509e18 (J to eV)
factor = 14.39964 #e-19

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
                       'stress': np.zeros([6])}

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

            #print(ABCij)
            #print(ABCij.shape)
            #import sys; sys.exit()



            Rij = Rj - Ri
            Dij = np.sqrt(np.sum(Rij**2, axis=1))

            d = calculate_ZBL(Rij, Dij, Zi, Zj, self.outer, self.inner, ABCij)
            self.result['energy'] += d

        return self.result


def calculate_ZBL(rij, dij, Zi, Zj, r_outer, r_inner, ABC):
    ids1 = (dij <= r_outer)
    #nids1 = np.logical_not(ids1)



    if True not in ids1:
        return 0.

    else:
        results = np.zeros([len(rij)])

        kZi = factor * Zi
        #kZi = Zi
        kZiZj = kZi * Zj
        dij_inv = 1 / dij

        Zi_inv = Zi ** 0.23 / 0.46850
        Zj_inv = Zj ** 0.23 / 0.46850
        a_inv = Zi_inv + Zj_inv

        x = dij * a_inv
        phi_ij = 0.18175 * np.exp(-3.19980 * x) + 0.50986 * np.exp(-0.94229 * x) + \
                 0.28022 * np.exp(-0.40290 * x) + 0.02817 * np.exp(-0.20162 * x) 

        Eij = kZiZj * dij_inv * phi_ij 

        # add switching function
        SA = 0.333333333 * ABC[:,0] * (dij - r_inner) ** 3
        SB = 0.25 * ABC[:,1] * (dij - r_inner) ** 4
        SC = ABC[:,2]

        results[ids1] += SC[ids1]

        ids2 = (dij <= r_outer) & (dij > r_inner)
        results[ids2] += SA[ids2] + SB[ids2]
    
        return np.sum(results) #np.sum(ZBL_ij)


def get_ABC_coefficients(Zi, Zj, r_outer, r_inner):
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
             0.28022 * 0.40290 ** 2 * exp3 + 0.02817 * 0.20162 ** 2 * exp4) * a_inv

    E = kZiZj * (1 / r_outer) * phi
    EP = kZiZj * (-1 / r_outer ** 2 * phi + 1 / r_outer * phiP)
    EDP = kZiZj * (2 / r_outer ** 3 * phi - 2 / r_outer ** 2 * phiP + 1 / r_outer * phiDP)

    A = (-3 * EP + r * EDP) / r ** 2
    B = (2 * EP - r * EDP) / r ** 3
    C = -E + 0.5 * r * EP - (1/12) * r ** 2 * EP

    return [A, B, C]


if __name__ == '__main__':
    import time
    from ase.build import bulk
    from ase.io import read
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

    struc = read("MOD_NiMo.cif")
    zbl = ZBL(2.0, 3.0)
    energy = zbl.calculate(struc)['energy'] / len(struc)

    print(energy)
