import numpy as np
import itertools
import numba
#from numerical_element import Element



################################ Gaussian Class ###############################


class BehlerParrinello:
    def __init__(self, crystal, symmetry_parameters, Rc=6.5, Rs=0., 
                 derivative=True):
        # Set up the symmetry parameters keywords. If a string are not in the
        # keyword, code will return an error.
        self.G1_keywords = []
        self.G2_keywords = ['eta',]
        self.G3_keywords = ['kappa',]
        self.G4_keywords = ['eta', 'lamBda', 'zeta',]
        self.G5_keywords = ['eta', 'lamBda', 'zeta',]
        
        # Extract useful quantities from Pymatgen Structure object.
        self.crystal = crystal
        self.num_of_cores = crystal.num_sites # Integer
        self.atom_types = crystal.symbol_set
#        self.atom_types = Element(1, crystal.symbol_set).convert() # ['Na', 'Cl']
        # element pair types, i.e. [('Na', 'Na'), ('Na', 'Cl'), ('Cl', 'Cl')]
        self.pair_atoms = list(
                          itertools.combinations_with_replacement(
                          self.atom_types, 2))
        self.cores_species = []
        self.cores_coordinates = []
        for i in range(self.num_of_cores):
            self.cores_coordinates.append(crystal.cart_coords[i])
            self.cores_species.append(crystal[i].species_string)
#        self.cores_species = Element(1, cores_species).convert()
        
        # Obtain neighbors information from Pymatgen Structure object.
        neighbors = crystal.get_all_neighbors(Rc,
                                              include_index=True,
                                              include_image=True)
        self.neighbors_species = []
        self.neighbors_coordinates = []
        self.neighbors_images = []
        self.neighbors_indexes = []
        for i in range(len(neighbors)):
            spe = []
            coords = []
            imag = []
            ind = []
            for j in range(len(neighbors[i])):
                spe.append(neighbors[i][j][0].species_string)
                coords.append(neighbors[i][j][0]._coords)
                ind.append(neighbors[i][j][2])
                imag.append(neighbors[i][j][3])
            self.neighbors_species.append(spe)
            self.neighbors_coordinates.append(coords)
            self.neighbors_images.append(imag)
            self.neighbors_indexes.append(ind)
#        self.neighbors_species = Element(2, neighbors_species).convert()
            
        self.Rc = Rc
        self.Rs = Rs
        
        # Setting up parameters for each of the symmetry function type.
        self.symmetry_parameters = symmetry_parameters
        self.G_types = [] # e.g. ['G2', 'G4']
        self.G1_parameters = None
        self.G2_parameters = None
        self.G3_parameters = None
        self.G4_parameters = None
        self.G5_parameters = None
        for key, value in self.symmetry_parameters.items():
            if key == 'G1':
                self.G1_parameters = value
                self.G_types.append(key)
            elif key == 'G2':
                self.G2_parameters = value
                self.G_types.append(key)
            elif key == 'G3':
                self.G3_parameters = value
                self.G_types.append(key)
            elif key == 'G4':
                self.G4_parameters = value
                self.G_types.append(key)
            else:
                self.G5_parameters = value
                self.G_types.append(key)
        
        # Calculate and obtain all the symmetry functions defined in the
        # symmetry_parameters.
        self.Gs = self.get_all_G(derivative)
        
    
    def get_all_G(self, derivative):
        """Calculate and obtain all the symmetry functions defined in the 
        symmetry_parameters."""
        all_G = None
                
        if self.G1_parameters is not None:
            G1 = {}
            # Check sanity
            self.G1, self.G1_prime = get_G1(derivative)
            G1['G'] = self.G1
            G1['Gprime'] = self.G1_prime
            all_G = G1
            
        if self.G2_parameters is not None:
            G2 = {}
            
            # Check sanity
            for key, value in self.G2_parameters.items():
                if key in self.G2_keywords:
                    if isinstance(value, (list, np.ndarray)):
                        etas = value
                    else:
                        etas = [value]
                else:
                    msg = f"{key} is not available. "\
                    f"Choose from {self.G2_keywords}"
                    raise NotImplementedError(msg)
            
            # Get and calculate G2
            self.G2, self.G2Prime = get_G2(self.num_of_cores,
                                           self.atom_types,
                                           self.cores_species,
                                           self.cores_coordinates,
                                           self.neighbors_species,
                                           self.neighbors_coordinates,
                                           self.neighbors_images,
                                           self.neighbors_indexes,
                                           self.Rc, self.Rs, etas, derivative)
            
            # Arrange G2 to be combined with all_G
            G2['G'] = self.G2
            G2['Gprime'] = self.G2Prime
            
            if all_G != None:
                assert len(G2['G']) == len(all_G['G']), \
                "The length of two symmetry function are different"
                assert len(G2['Gprime']) == len(all_G['Gprime']), \
                "The length of two symmetry function derivative are different"
                
                for i in range(len(all_G['G'])):
                    atom1, d1 = all_G['G'][i]
                    atom2, d2 = G2['G'][i]
                    if atom1 == atom2:
                        d = d1 + d2
                        all_G['G'][i] = (atom1, d)
                    else:
                        print("Defected!")

                for j in range(len(all_G['Gprime'])):
                    pair_atom1, dp1 = all_G['Gprime'][j]
                    pair_atom2, dp2 = G2['Gprime'][j]
                    if pair_atom1 == pair_atom2:
                        dp = dp1 + dp2
                        all_G['Gprime'][j] = (pair_atom1, dp)
                    else:
                        print("Defected!")
            else:
                all_G = G2
            
        self.all_G = all_G
        
        return self.all_G
        
        
def get_G1(derivative):
    pass

#@numba.njit(cache=True, nogil=True, fastmath=True)
def get_G2(num_of_cores, atom_types, cores_species, cores_coordinates,
           neighbors_species, neighbors_coordinates, neighbors_images, 
           neighbors_indexes, Rc, Rs, etas, derivative):
    G, Gp = [], []
    
    for i in range(num_of_cores):
        g = []
        for eta in etas:
            for atom in atom_types:
                print(neighbors_coordinates[i])
                g.append(G2(Ri=cores_coordinates[i],
                            n_coordinates=neighbors_coordinates[i],
                            n_species=neighbors_species[i],
                            atom=atom,
                            Rc=Rc,
                            Rs=Rs,
                            eta=eta,))
        G.append(g)
        
    if derivative:
        for i in range(num_of_cores):
            for q in range(3):
                gp = []
                for eta in etas:
                    for atom in atom_types:
                        gp.append(G2_prime(Ri=cores_coordinates[i],
                                           i=i,
                                           n_coordinates=neighbors_coordinates[i],
                                           n_species=neighbors_species[i],
                                           n_indexes=neighbors_indexes[i],
                                           atom=atom,
                                           Rc=Rc,
                                           Rs=Rs,
                                           eta=eta,
                                           p=i,
                                           q=q))
                Gp.append(gp)
                
                for j in range(len(neighbors_coordinates[i])):
                    if neighbors_images[i][j] == (0., 0., 0.):
                        gp = []
                        for eta in etas:
                            for atom in atom_types:
                                index = neighbors_indexes[i][j]
                                prime = G2_prime(Ri=cores_coordinates[index],
                                                 i=index,
                                                 n_coordinates=neighbors_coordinates[index],
                                                 n_species=neighbors_species[index],
                                                 n_indexes=neighbors_indexes[index],
                                                 atom=atom,
                                                 Rc=Rc,
                                                 Rs=Rs,
                                                 eta=eta,
                                                 p=i,
                                                 q=q)
                                gp.append(prime)
                        Gp.append(gp)
    
    return G, Gp

#@numba.njit(cache=True, nogil=True, fastmath=True)
@numba.njit(numba.f8(numba.f8[:], numba.f8[:,:], numba.i1[:], numba.i1, numba.f8, numba.f8, numba.f8),
            cache=True, nogil=True, fastmath=True)
def G2(Ri, n_coordinates, n_species, atom, Rc=6.5, Rs=0., eta=2):
    G2 = 0
    for j in range(len(n_coordinates)):
        species = n_species[j]
        if atom == species:
            Rj = n_coordinates[j]
            Rij = np.linalg.norm(Ri - Rj)
            G2 += np.exp(-eta * (Rij - Rs) ** 2. / Rc ** 2.) * Cosine(Rij, Rc)
    
    return G2


def G2_prime(Ri, i, n_coordinates, n_species, n_indexes, atom, 
             Rc=6.5, Rs=0.0, eta=2, p=1, q=0):
    G2p = 0
    for count in range(len(n_coordinates)):
        symbol = n_species[count]
        Rj = n_coordinates[count]
        j = n_indexes[count]
        if atom == symbol:
            dRabdRpq = dRab_dRpq(i, j, Ri, Rj, p, q)
            if dRabdRpq != 0.:
                Rij = np.linalg.norm(Rj - Ri)
                G2p += np.exp(-eta * (Rij - Rs)**2. / Rc**2.) * \
                        dRabdRpq * \
                        (-2. * eta * (Rij - Rs) * Cosine(Rij, Rc) / Rc ** 2. + 
                        CosinePrime(Rij, Rc))

    return G2p

@numba.njit(numba.f8(numba.f8, numba.f8),
            cache=True, nogil=True, fastmath=True)
def Cosine(Rij, Rc):
    if Rij > Rc:
        return 0.

    else:
        return 0.5 * (np.cos(np.pi * Rij / Rc) + 1.)

@numba.njit(numba.f8(numba.f8, numba.f8),
            cache=True, nogil=True, fastmath=True)
def CosinePrime(Rij, Rc):
    if Rij > Rc:
        return 0.
    else:
        return(-0.5 * np.pi / Rc * np.sin(np.pi * Rij / Rc))

@numba.njit(numba.f8(numba.i8, numba.i8, numba.f8[:], 
                     numba.f8[:], numba.i8, numba.i8),
            cache=True, nogil=True, fastmath=True)
def dRab_dRpq(a, b, Ra, Rb, p, q):
    Rab = np.linalg.norm(Rb - Ra)
    if p == a and a != b:  # a != b is necessary for periodic systems
        dRab_dRpq = -(Rb[q] - Ra[q]) / Rab
    elif p == b and a != b:  # a != b is necessary for periodic systems
        dRab_dRpq = (Rb[q] - Ra[q]) / Rab
    else:
        dRab_dRpq = 0

    return dRab_dRpq

# Test
from pymatgen import Structure, Lattice
crystal = Structure.from_spacegroup(225, Lattice.cubic(5.69169),
                                    ['Na', 'Cl'], [[0, 0, 0], [0, 0, 0.5]])

symmetry = {'G2': {'eta': [0.036, 0.071]}}
bp = BehlerParrinello(crystal, symmetry,)
