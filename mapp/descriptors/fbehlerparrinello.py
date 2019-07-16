import numpy as np
import itertools
import numba
from ..utilities.neighborhood import Element, Neighborhood


################################ Gaussian Class ###############################


class fBehlerParrinello:
    """A class that calculates Behler-Parrinello symmetry functions.
    
    The forms of the functions are consistent with the 
    functions presented in:
        Behler, J. (2011). The Journal of Chemical Physics, 134(7), 074106.
        
    This code is equipped with Numba. Numba is a compiler for Python that
    accelerates calculations. It is achieved by compiling to machine code for 
    execution. Consequently, this code can run at the speed of machine code.
        
    Parameters
    ----------
    crystal: object
        Pymatgen Structure object.
    symmetry_parameters: dict
        The parameters for the symmetry functions to be calculated.
        i.e. {'G2': {'eta': [0.05, 0.1]}}
    Rc: float
        The symmetry functions will be calculated within this radius.
    derivative: bool
        If True, calculate the derivatives of symmetry functions.
    
    To do:
        1. Implement G1
        2. Derivative = False
        3. Add a variety of cutoff functionals: Polynomial or Tanh.
        4. Fix warning pair_atoms
    """
    def __init__(self, crystal, symmetry_parameters, Rc=6.5, derivative=True):
        # Set up the symmetry parameters keywords. If a string are not in the
        # keyword, code will return an error.
        self.G1_keywords = []
        self.G2_keywords = ['eta', 'Rs',]
        self.G3_keywords = ['kappa',]
        self.G4_keywords = ['eta', 'lambda', 'zeta',]
        self.G5_keywords = ['eta', 'lambda', 'zeta',]
        
        # Extract useful quantities from Pymatgen Structure object.
        self.crystal = crystal
        self.num_of_cores = crystal.num_sites # Integer
        self.atom_types = Element(1, crystal.symbol_set).convert() # ['Na', 'Cl']
        # element pair types, i.e. [('Na', 'Na'), ('Na', 'Cl'), ('Cl', 'Cl')]
        self.pair_atoms = list(itertools.combinations_with_replacement(
                               self.atom_types, 2))
                            
        # Obtain atom in the unit cell info.
        cores_species = []
        self.cores_coordinates = []
        for i in range(self.num_of_cores):
            self.cores_coordinates.append(crystal.cart_coords[i])
            cores_species.append(crystal[i].species_string)
        self.c_species = cores_species  # string
        self.cores_species = Element(1, cores_species).convert()
        self.cores_coordinates = np.asarray(self.cores_coordinates)
        
        # Obtain neighbors info.
        neighbors = crystal.get_all_neighbors(Rc,
                                              include_index=True,
                                              include_image=True)
        neighbors_species = []
        neighbors_coordinates = []
        neighbors_images = []
        neighbors_indexes = []
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
            neighbors_species.append(spe)
            neighbors_coordinates.append(coords)
            neighbors_images.append(imag)
            neighbors_indexes.append(ind)
        self.n_species = neighbors_species # Species in string
        
        neighbors = Neighborhood(neighbors_coordinates, neighbors_species, 
                                 neighbors_images, neighbors_indexes)
        self.neighbors_coordinates, self.neighbors_limit = \
                                                    neighbors.get_coordinates()
        self.neighbors_species = neighbors.get_species() # Species in int
        self.neighbors_images = neighbors.get_images()
        self.neighbors_indexes = neighbors.get_indexes()
        
        self.Rc = Rc
        
        # Setting up parameters for each of the symmetry function type.
        self.symmetry_parameters = symmetry_parameters
        self.G_types = [] # ['G2', 'G4']
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
        # This is the attribute that needed to be passed in the main script
        # to the Neural Network model.
        self.Gs = self.get_all_G(derivative)
        
    
    def get_all_G(self, derivative):
        """This method calculates all the symmetry functions defined in the 
        symmetry_parameters. This method also arranges the symmetry functions
        to a desireable format for Neural Network model.
        
        Parameters
        ----------
        derivative: bool
        If True, calculate the derivatives of symmetry functions.
        (Note: this still needs to be fixed for False.)
        """
        all_G = None
                
        if self.G1_parameters is not None:
            G1 = {}
            
            self.G1, self.G1_prime = get_G1(derivative)
            G1['G'] = self.G1
            G1['Gprime'] = self.G1_prime
            all_G = G1
            
        if self.G2_parameters is not None:
            Rs = 0.
            G2 = {}
            G2['G'] = []
            G2['Gprime'] = []
            
            for key, value in self.G2_parameters.items():
                if key in self.G2_keywords:
                    if key == 'eta':
                        if isinstance(value, (list, np.ndarray)):
                            etas = np.asarray(value)
                        else:
                            etas = np.asarray([value])
                    elif key == 'Rs':
                        Rs += value
                else:
                    msg = f"{key} is not available. "\
                    f"Choose from {self.G2_keywords}"
                    raise NotImplementedError(msg)
            
            # Calculate G2
            g2 = get_G2_or_G2Prime(self.num_of_cores,
                                   self.atom_types,
                                   self.cores_species,
                                   self.cores_coordinates,
                                   self.neighbors_species,
                                   self.neighbors_coordinates,
                                   self.neighbors_images,
                                   self.neighbors_indexes,
                                   self.neighbors_limit,
                                   self.Rc, Rs, etas, 
                                   derivative=False)

            # Perhaps add the option for derivative here.
            g2Prime = get_G2_or_G2Prime(self.num_of_cores,
                                        self.atom_types,
                                        self.cores_species,
                                        self.cores_coordinates,
                                        self.neighbors_species,
                                        self.neighbors_coordinates,
                                        self.neighbors_images,
                                        self.neighbors_indexes,
                                        self.neighbors_limit,
                                        self.Rc, Rs, etas, 
                                        derivative=True)
                        
            count = 0
            for i in range(self.num_of_cores):
                G2['G'].append((self.c_species[i], g2[i]))
                for q in range(3):
                    G2['Gprime'].append(((i, self.c_species[i], i, 
                                          self.c_species[i], q),
                                        g2Prime[count]))
                    count += 1
                    for j in range(self.neighbors_limit[i]):
                        if (self.neighbors_images[i][j] == [0., 0., 0.]).all():
                            index = self.neighbors_indexes[i][j]
                            G2['Gprime'].append(((i, self.c_species[i],
                                                  index,
                                                  self.c_species[index],
                                                  q), g2Prime[count]))
                            count += 1
            
            # Arrange G2 to be combined with all_G
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
                
        if self.G3_parameters is not None:
            G3 = {}
            G3['G'] = []
            G3['Gprime'] = []
            
            for key, value in self.G3_parameters.items():
                if key in self.G3_keywords:
                    if key == 'kappa':
                        if isinstance(value, (list, np.ndarray)):
                            kappas = np.asarray(value)
                        else:
                            kappas = np.asarray([value])
                else:
                    msg = f"{key} is not available. "\
                    f"Choose from {self.G3_keywords}"
                    raise NotImplementedError(msg)
                    
            # Calculate G3
            g3 = get_G3_or_G3Prime(self.num_of_cores,
                                   self.atom_types,
                                   self.cores_species,
                                   self.cores_coordinates,
                                   self.neighbors_species,
                                   self.neighbors_coordinates,
                                   self.neighbors_images,
                                   self.neighbors_indexes,
                                   self.neighbors_limit,
                                   self.Rc, kappas, 
                                   derivative=False)
            
            g3Prime = get_G3_or_G3Prime(self.num_of_cores,
                                        self.atom_types,
                                        self.cores_species,
                                        self.cores_coordinates,
                                        self.neighbors_species,
                                        self.neighbors_coordinates,
                                        self.neighbors_images,
                                        self.neighbors_indexes,
                                        self.neighbors_limit,
                                        self.Rc, kappas,
                                        derivative=True)
            
            count = 0
            for i in range(self.num_of_cores):
                G3['G'].append((self.c_species[i], g3[i]))
                for q in range(3):
                    G3['Gprime'].append(((i, self.c_species[i], i,
                                          self.c_species[i], q),
                                        g3Prime[count]))
                    count += 1
                    for j in range(self.neighbors_limit[i]):
                        if (self.neighbors_images[i][j] == [0., 0., 0.]).all():
                            index = self.neighbors_indexes[i][j]
                            G3['Gprime'].append(((i, self.c_species[i],
                                                  index,
                                                  self.c_species[index],
                                                  q), g3Prime[count]))
                            count += 1

            # Arrange G3 to be combined with all_G
            if all_G != None:
                assert len(G3['G']) == len(all_G['G']), \
                "The length of two symmetry function are different"
                assert len(G3['Gprime']) == len(all_G['Gprime']), \
                "The length of two symmetry function derivative are different"
                
                for i in range(len(all_G['G'])):
                    atom1, d1 = all_G['G'][i]
                    atom2, d2 = G3['G'][i]
                    if atom1 == atom2:
                        d = d1 + d2
                        all_G['G'][i] = (atom1, d)
                    else:
                        print("Defected!")

                for j in range(len(all_G['Gprime'])):
                    pair_atom1, dp1 = all_G['Gprime'][j]
                    pair_atom2, dp2 = G3['Gprime'][j]
                    if pair_atom1 == pair_atom2:
                        dp = dp1 + dp2
                        all_G['Gprime'][j] = (pair_atom1, dp)
                    else:
                        print("Defected!")
            else:
                all_G = G3
                            
        if self.G4_parameters is not None:
            G4 = {}
            G4['G'] = []
            G4['Gprime'] = []
            
            for key, value in self.G4_parameters.items():
                if key in self.G4_keywords:
                    if key == 'eta':
                        if isinstance(value, (list, np.ndarray)):
                            etas = np.asarray(value)
                        else:
                            etas = np.asarray([value])
                    elif key == 'lambda':
                        if isinstance(value, (list, np.ndarray)):
                            lamBdas = np.asarray(value)
                        else:
                            lamBdas = np.asarray([value])
                    elif key == 'zeta':
                        if isinstance(value, (list, np.ndarray)):
                            zetas = np.asarray(value)
                        else:
                            zetas = np.asarray([value])
                else:
                    msg = f"{key} is not available. "\
                    f"Choose from {self.G4_keywords}"
                    raise NotImplementedError(msg)
            
            # Calculate G4
            g4 = get_G4_or_G4Prime(self.num_of_cores,
                                   self.pair_atoms,
                                   self.cores_species,
                                   self.cores_coordinates,
                                   self.neighbors_species,
                                   self.neighbors_coordinates,
                                   self.neighbors_images,
                                   self.neighbors_indexes,
                                   self.neighbors_limit,
                                   self.Rc, etas, zetas, lamBdas,
                                   derivative=False)
            g4Prime = get_G4_or_G4Prime(self.num_of_cores,
                                        self.pair_atoms,
                                        self.cores_species,
                                        self.cores_coordinates,
                                        self.neighbors_species,
                                        self.neighbors_coordinates,
                                        self.neighbors_images,
                                        self.neighbors_indexes,
                                        self.neighbors_limit,
                                        self.Rc, etas, zetas, lamBdas,
                                        derivative=True)
            
            count = 0
            for i in range(self.num_of_cores):
                G4['G'].append((self.c_species[i], g4[i]))
                for q in range(3):
                    G4['Gprime'].append(((i, self.c_species[i], i, 
                                          self.c_species[i], q),
                                        g4Prime[count]))
                    count += 1
                    for j in range(self.neighbors_limit[i]):
                        if (self.neighbors_images[i][j] == [0., 0., 0.]).all():
                            index = self.neighbors_indexes[i][j]
                            G4['Gprime'].append(((i, self.c_species[i],
                                                  index,
                                                  self.c_species[index],
                                                  q), g4Prime[count]))
                            count += 1
                            
            # Arrange G4 to be combined with all_G
            if all_G != None:
                assert len(G4['G']) == len(all_G['G']), \
                "The length of two symmetry function are different"
                assert len(G4['Gprime']) == len(all_G['Gprime']), \
                "The length of two symmetry function derivative are different"
                
                for i in range(len(all_G['G'])):
                    atom1, d1 = all_G['G'][i]
                    atom2, d2 = G4['G'][i]
                    if atom1 == atom2:
                        d = d1 + d2
                        all_G['G'][i] = (atom1, d)
                    else:
                        print("Defected!")

                for j in range(len(all_G['Gprime'])):
                    pair_atom1, dp1 = all_G['Gprime'][j]
                    pair_atom2, dp2 = G4['Gprime'][j]
                    if pair_atom1 == pair_atom2:
                        dp = dp1 + dp2
                        all_G['Gprime'][j] = (pair_atom1, dp)
                    else:
                        print("Defected!")
            else:
                all_G = G4
                
        if self.G5_parameters is not None:
            G5 = {}
            G5['G'] = []
            G5['Gprime'] = []
            
            for key, value in self.G5_parameters.items():
                if key in self.G5_keywords:
                    if key == 'eta':
                        if isinstance(value, (list, np.ndarray)):
                            etas = np.asarray(value)
                        else:
                            etas = np.asarray([value])
                    elif key == 'lambda':
                        if isinstance(value, (list, np.ndarray)):
                            lamBdas = np.asarray(value)
                        else:
                            lamBdas = np.asarray([value])
                    elif key == 'zeta':
                        if isinstance(value, (list, np.ndarray)):
                            zetas = np.asarray(value)
                        else:
                            zetas = np.asarray([value])
                else:
                    msg = f"{key} is not available. "\
                    f"Choose from {self.G5_keywords}"
                    raise NotImplementedError(msg)
            
            # Calculate G5
            g5 = get_G5_or_G5Prime(self.num_of_cores,
                                   self.pair_atoms,
                                   self.cores_species,
                                   self.cores_coordinates,
                                   self.neighbors_species,
                                   self.neighbors_coordinates,
                                   self.neighbors_images,
                                   self.neighbors_indexes,
                                   self.neighbors_limit,
                                   self.Rc, etas, zetas, lamBdas,
                                   derivative=False)
            g5Prime = get_G5_or_G5Prime(self.num_of_cores,
                                        self.pair_atoms,
                                        self.cores_species,
                                        self.cores_coordinates,
                                        self.neighbors_species,
                                        self.neighbors_coordinates,
                                        self.neighbors_images,
                                        self.neighbors_indexes,
                                        self.neighbors_limit,
                                        self.Rc, etas, zetas, lamBdas,
                                        derivative=True)
            
            count = 0
            for i in range(self.num_of_cores):
                G5['G'].append((self.c_species[i], g5[i]))
                for q in range(3):
                    G5['Gprime'].append(((i, self.c_species[i], i, 
                                          self.c_species[i], q), 
                                        g5Prime[count]))
                    count += 1
                    for j in range(self.neighbors_limit[i]):
                        if (self.neighbors_images[i][j] == [0., 0., 0.]).all():
                            index = self.neighbors_indexes[i][j]
                            G5['Gprime'].append(((i, self.c_species[i],
                                                  index, 
                                                  self.c_species[index], 
                                                  q), g5Prime[count]))
                            count += 1
                            
            # Arrange G5 to be combined with all_G
            if all_G != None:
                assert len(G5['G']) == len(all_G['G']), \
                "The length of two symmetry function are different"
                assert len(G5['Gprime']) == len(all_G['Gprime']), \
                "The length of two symmetry function derivative are different"
                
                for i in range(len(all_G['G'])):
                    atom1, d1 = all_G['G'][i]
                    atom2, d2 = G5['G'][i]
                    if atom1 == atom2:
                        d = d1 + d2
                        all_G['G'][i] = (atom1, d)
                    else:
                        print("Defected!")

                for j in range(len(all_G['Gprime'])):
                    pair_atom1, dp1 = all_G['Gprime'][j]
                    pair_atom2, dp2 = G5['Gprime'][j]
                    if pair_atom1 == pair_atom2:
                        dp = dp1 + dp2
                        all_G['Gprime'][j] = (pair_atom1, dp)
                    else:
                        print("Defected!")
            else:
                all_G = G5
                
        return all_G
        

########################### Symmetry Functions ################################


def get_G1(derivative):
    pass


@numba.njit(cache=True, nogil=True, fastmath=True)
def get_G2_or_G2Prime(num_of_cores, atom_types, cores_species, 
                      cores_coordinates, neighbors_species, 
                      neighbors_coordinates, neighbors_images, 
                      neighbors_indexes, neighbors_limit, Rc, Rs, etas, 
                      derivative):
    """A function to get G2 or the derivative of G2. 
    
    Parameters
    ----------
    num_of_cores: int
        The number of atoms in the crystal unit cell.
    atom_types: array of int
        Contains the atomic kinds in the crystal structure. These atoms have
        been translated into integer. i.e. H is represented as 1.
    cores_species: array of int
        Contains the atomic number of the core atoms in the unit cell.
    cores_coordinates: 2D array of int
        Contains the coordinates of each atoms in the unit cell.
    neighbors_species: 2D of int
        Contains the atomic number of the neighbors of the core atoms.
    neighbors_coordinates: 3D array
        Contains the coordinates of the neighbors of the core atoms.
    neighbors_images: 3D array
        Contains the images of the neighbors of the core atoms.
    neighbors_indexes: 2D array of int
        Contains the indexes of the neighbors of the core atoms.
    neighbors_limit: array of int
        Contains the number of neighbors for each core atoms.
    Rc: float
        The cutoff radius.
    Rs: float
        The shift from the center of the G2 symmetry function.
    etas: array
        The parameters of G2 symmetry function.
    derivative:
        If True, calculate the derivative of G2 instead of G2.
        
    Returns
    -------
    G: array
        The G2 or the derivative of G2.
    """
    G = []
        
    if derivative:
        for i in range(num_of_cores):
            nl = neighbors_limit[i]
            nc = neighbors_coordinates[i]
            ns = neighbors_species[i]
            ni = neighbors_indexes[i]
            for q in range(3):
                gp = []
                for eta in etas:
                    for atom in atom_types:
                        prime = G2Prime(Ri=cores_coordinates[i],
                                        i=i,
                                        n_coordinates=nc,
                                        n_species=ns,
                                        n_indexes=ni,
                                        n_limit=nl,
                                        atom=atom,
                                        Rc=Rc,
                                        Rs=Rs,
                                        eta=eta,
                                        p=i,
                                        q=q)
                        gp.append(prime)
                G.append(gp)
                
                for j in range(neighbors_limit[i]):
                    for _ in range(3):
                        if neighbors_images[i][j][_] == 0.:
                            boolean = True
                        else:
                            boolean = False
                            break
                    
                    if boolean:
                        gpp = []
                        for eta in etas:
                            for atom in atom_types:
                                index = neighbors_indexes[i][j]
                                nci = neighbors_coordinates[index]
                                nsi = neighbors_species[index]
                                nii = neighbors_indexes[index]
                                nli = neighbors_limit[index]
                                prime = G2Prime(Ri=cores_coordinates[index],
                                                i=index,
                                                n_coordinates=nci,
                                                n_species=nsi,
                                                n_indexes=nii,
                                                n_limit=nli,
                                                atom=atom,
                                                Rc=Rc,
                                                Rs=Rs,
                                                eta=eta,
                                                p=i,
                                                q=q)
                                gpp.append(prime)
                        G.append(gpp)
                        
    else:
        for i in range(num_of_cores):
            g = []
            for eta in etas:
                for atom in atom_types:
                    g.append(G2(Ri=cores_coordinates[i],
                                n_coordinates=neighbors_coordinates[i],
                                n_species=neighbors_species[i],
                                n_limit=neighbors_limit[i],
                                atom=atom,
                                Rc=Rc,
                                Rs=Rs,
                                eta=eta,))
            G.append(g)
    
    return G


@numba.njit(cache=True, nogil=True, fastmath=True)
def G2(Ri, n_coordinates, n_species, n_limit, atom, Rc=6.5, Rs=0., eta=2):
    """Calculate G2 symmetry function.
    
    G2 function is a better choice to describe the radial feature of atoms in 
    a crystal structure given a cutoff radius. One can refer to equation 6 in
    the journal paper described above.
    
    Parameters
    ----------
    Ri: array
        The coordinate(x, y, z) of the i-th core atom.
    n_coordinates: 2D array
        The coordinates of the neighbors of the i-th core atom.
    n_species: array of int
        The species of the neighbors of the i-th core atom.
    n_limit: float
        The number of neighbors of the i-th core atom.
    atom: int
        The atomic number of the "bonded" atom.
    Rc: float
        The cutoff radius.
    Rs: float
        The shift from the center of the G2 symmetry function.
    eta: float
        A parameter of G2 symmetry function.
    
    Returns
    -------
    G2: float
        G2 symmetry value.
    """
    G2 = 0.
    for j in range(n_limit):
        species = n_species[j]
        if atom == species:
            Rj = n_coordinates[j]
            Rij = np.linalg.norm(Ri - Rj)
            G2 += np.exp(-eta * (Rij - Rs) ** 2. / Rc ** 2.) * Cosine(Rij, Rc)
    
    return G2


@numba.njit(cache=True, nogil=True, fastmath=True)
def G2Prime(Ri, i, n_coordinates, n_species, n_indexes, n_limit, atom, 
             Rc=6.5, Rs=0.0, eta=2, p=1, q=0):
    """Calculate the derivative of G2 symmetry function.
    
    Parameters
    ----------
    Ri: array
        The coordinate(x, y, z) of the i-th core atoms.
    i: int
        The index of the i-th core atom.
    n_coordinates: 2D array
        The coordinates of the neighbors of the i-th core atom.
    n_species: array of int
        The species of the neighbors of the i-th core atom.
    n_limit: float
        The number of neighbors of the i-th core atom.
    atom: int
        The atomic number of the "bonded" atom.
    Rc: float
        The cutoff radius.
    Rs: float
        The shift from the center of the G2 symmetry function.
    eta: float
        A parameter of G2 symmetry function.
    p: int
        The atom that the force is acting on.
    q: int
        Direction of force.
    
    Returns
    -------
    G2p: float
        The derivative of G2 symmetry value.
    """
    G2p = 0.
    for count in range(n_limit):
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


@numba.njit(cache=True, nogil=True, fastmath=True)
def get_G3_or_G3Prime(num_of_cores, atom_types, cores_species, 
                      cores_coordinates, neighbors_species, 
                      neighbors_coordinates, neighbors_images, 
                      neighbors_indexes, neighbors_limit, Rc, kappas, 
                      derivative):
    """A function to get G3 or the derivative of G3.
    
    Parameters
    ----------
    num_of_cores: int
        The number of atoms in the crystal unit cell.
    atom_types: array of int
        Contains the atomic kinds in the crystal structure. These atoms have
        been translated into integer. i.e. H is represented as 1.
    cores_species: array of int
        Contains the atomic number of the core atoms in the unit cell.
    cores_coordinates: 2D array of int
        Contains the coordinates of each atoms in the unit cell.
    neighbors_species: 2D of int
        Contains the atomic number of the neighbors of the core atoms.
    neighbors_coordinates: 3D array
        Contains the coordinates of the neighbors of the core atoms.
    neighbors_images: 3D array
        Contains the images of the neighbors of the core atoms.
    neighbors_indexes: 2D array of int
        Contains the indexes of the neighbors of the core atoms.
    neighbors_limit: array of int
        Contains the number of neighbors for each core atoms.
    Rc: float
        The cutoff radius.
    kappas: array
        The parameters of G3 symmetry function.
    derivative:
        If True, calculate the derivative of G3 instead of G3.
        
    Returns
    -------
    G: array
        The G3 or the derivative of G3.
    """
    G = []
    
    if derivative:
        for i in range(num_of_cores):
            nl = neighbors_limit[i]
            nc = neighbors_coordinates[i]
            ns = neighbors_species[i]
            ni = neighbors_indexes[i]
            for q in range(3):
                gp = []
                for kappa in kappas:
                    for atom in atom_types:
                        prime = G3Prime(Ri=cores_coordinates[i],
                                        i=i,
                                        n_coordinates=nc,
                                        n_species=ns,
                                        n_indexes=ni,
                                        n_limit=nl,
                                        atom=atom,
                                        Rc=Rc,
                                        k=kappa,
                                        p=i,
                                        q=q)
                        gp.append(prime)
                G.append(gp)
                
                for j in range(neighbors_limit[i]):
                    for _ in range(3):
                        if neighbors_images[i][j][_] == 0.:
                            boolean = True
                        else:
                            boolean = False
                            break
                    
                    if boolean:
                        gpp = []
                        for kappa in kappas:
                            for atom in atom_types:
                                index = neighbors_indexes[i][j]
                                nci = neighbors_coordinates[index]
                                nsi = neighbors_species[index]
                                nii = neighbors_indexes[index]
                                nli = neighbors_limit[index]
                                prime = G3Prime(Ri=cores_coordinates[index],
                                                i=index,
                                                n_coordinates=nci,
                                                n_species=nsi,
                                                n_indexes=nii,
                                                n_limit=nli,
                                                atom=atom,
                                                Rc=Rc,
                                                k=kappa,
                                                p=i,
                                                q=q)
                                gpp.append(prime)
                        G.append(gpp)
                        
    else:
        for i in range(num_of_cores):
            g = []
            for kappa in kappas:
                for atom in atom_types:
                    g.append(G3(Ri=cores_coordinates[i],
                                n_coordinates=neighbors_coordinates[i],
                                n_species=neighbors_species[i],
                                n_limit=neighbors_limit[i],
                                atom=atom,
                                Rc=Rc,
                                k=kappa,))
            G.append(g)
    
    return G


@numba.njit(cache=True, nogil=True, fastmath=True)
def G3(Ri, n_coordinates, n_species, n_limit, atom, Rc=6.5, k=10):
    """Calculate G3 symmetry function.
    
    G3 function is a damped cosine functions with a period length described by
    K. For example, a Fourier series expansion a suitable description of the 
    radial atomic environment can be obtained by comibning several G3
    functions with different values for K. One can refer to equation 7 in
    the journal paper described above.
    Note: due to the distances of atoms, G3 can cancel each other out depending
    on the positive and negative value.
    
    Parameters
    ----------
    Ri: array
        The coordinate(x, y, z) of the i-th core atom.
    n_coordinates: 2D array
        The coordinates of the neighbors of the i-th core atom.
    n_species: array of int
        The species of the neighbors of the i-th core atom.
    n_limit: float
        The number of neighbors of the i-th core atom.
    atom: int
        The atomic number of the "bonded" atom.
    Rc: float
        The cutoff radius.
    k: float
        A parameter of G3 symmetry function.
    
    Returns
    -------
    G3: float
        G3 symmetry value.
    """
    G3 = 0
    for j in range(n_limit):
        if atom == n_species[j]:
            Rj = n_coordinates[j]
            Rij = np.linalg.norm(Ri - Rj)
            G3 += np.cos(k * Rij / Rc) * Cosine(Rij, Rc)
    
    return G3


@numba.njit(cache=True, nogil=True, fastmath=True)
def G3Prime(Ri, i, n_coordinates, n_species, n_indexes, n_limit, atom, 
             Rc=6.5, k=2, p=1, q=0):
    """Calculate the derivative of G3 symmetry function.
    
    Parameters
    ----------
    Ri: array
        The coordinate(x, y, z) of the i-th core atoms.
    i: int
        The index of the i-th core atom.
    n_coordinates: 2D array
        The coordinates of the neighbors of the i-th core atom.
    n_species: array of int
        The species of the neighbors of the i-th core atom.
    n_limit: float
        The number of neighbors of the i-th core atom.
    atom: int
        The atomic number of the "bonded" atom.
    Rc: float
        The cutoff radius.
    k: float
        A parameter of G3 symmetry function.
    p: int
        The atom that the force is acting on.
    q: int
        Direction of force.
    
    Returns
    -------
    G3p: float
        The derivative of G3 symmetry value.
    """
    G3p = 0
    for count in range(n_limit):
        Rj = n_coordinates[count]
        symbol = n_species[count]
        j = n_indexes[count]
        if atom == symbol:
            dRabdRpq = dRab_dRpq(i, j, Ri, Rj, p, q)
            if dRabdRpq != 0:
                Rij = np.linalg.norm(Rj - Ri)
                G3p += (np.cos(k * Rij) * CosinePrime(Rij, Rc) - \
                             k * np.sin(k * Rij) * Cosine(Rij, Rc)) * \
                             dRab_dRpq(i, j, Ri, Rj, p, q)

    return G3p


@numba.njit(cache=True, nogil=True, fastmath=True)
def get_G4_or_G4Prime(num_of_cores, pair_atoms, cores_species, 
                      cores_coordinates, neighbors_species, 
                      neighbors_coordinates, neighbors_images, 
                      neighbors_indexes, neighbors_limit, Rc, etas, zetas, 
                      lamBdas, derivative):
    """A function to get G4 or the derivative of G4.
    
    Parameters
    ----------
    num_of_cores: int
        The number of atoms in the crystal unit cell.
    pair_atoms: 2D array
        Contains the pair of atomic kinds in the crystal structure. 
        These atoms have been translated into integer.
    cores_species: array of int
        Contains the atomic number of the core atoms in the unit cell.
    cores_coordinates: 2D array of int
        Contains the coordinates of each atoms in the unit cell.
    neighbors_species: 2D of int
        Contains the atomic number of the neighbors of the core atoms.
    neighbors_coordinates: 3D array
        Contains the coordinates of the neighbors of the core atoms.
    neighbors_images: 3D array
        Contains the images of the neighbors of the core atoms.
    neighbors_indexes: 2D array of int
        Contains the indexes of the neighbors of the core atoms.
    neighbors_limit: array of int
        Contains the number of neighbors for each core atoms.
    Rc: float
        The cutoff radius.
    etas: array
        Parameters of G4 symmetry function.
    zetas: array
        Parameters of G4 symmetry function.
    lamBdas: array
        Parameters of G4 symmetry function.
    derivative:
        If True, calculate the derivative of G4 instead of G4.
        
    Returns
    -------
    G: array
        The G4 or the derivative of G4.
    """    
    G = []
    
    if derivative:
        for i in range(num_of_cores):
            nl = neighbors_limit[i]
            nc = neighbors_coordinates[i]
            ns = neighbors_species[i]
            ni = neighbors_indexes[i]
            for q in range(3):
                gp = []
                for eta in etas:
                    for zeta in zetas:
                        for lb in lamBdas:
                            for patoms in pair_atoms:
                                prime = G4Prime(Ri=cores_coordinates[i],
                                                i=i,
                                                n_coordinates=nc,
                                                n_species=ns,
                                                n_indexes=ni,
                                                n_limit=nl,
                                                pair_atoms=patoms,
                                                Rc=Rc,
                                                eta=eta,
                                                zeta=zeta,
                                                lamBda=lb,
                                                p=i,
                                                q=q)
                                gp.append(prime)
                G.append(gp)
                
                for j in range(neighbors_limit[i]):
                    for _ in range(3):
                        if neighbors_images[i][j][_] == 0.:
                            boolean = True
                        else:
                            boolean = False
                            break
                    
                    if boolean:
                        gpp = []
                        for eta in etas:
                            for zeta in zetas:
                                for lb in lamBdas:
                                    for patoms in pair_atoms:
                                        index = neighbors_indexes[i][j]
                                        cc = cores_coordinates[index]
                                        nci = neighbors_coordinates[index]
                                        nsi = neighbors_species[index]
                                        nii = neighbors_indexes[index]
                                        nli = neighbors_limit[index]
                                        prime = G4Prime(Ri=cc,
                                                        i=index,
                                                        n_coordinates=nci,
                                                        n_species=nsi,
                                                        n_indexes=nii,
                                                        n_limit=nli,
                                                        pair_atoms=patoms,
                                                        Rc=Rc,
                                                        eta=eta,
                                                        lamBda=lb,
                                                        zeta=zeta,
                                                        p=i,
                                                        q=q)
                                        gpp.append(prime)
                        G.append(gpp)
                        
    else:
        for i in range(num_of_cores):
            g = []
            for eta in etas:
                for zeta in zetas:
                    for lb in lamBdas:
                        for patoms in pair_atoms:
                            g.append(G4(Ri=cores_coordinates[i],
                                        n_coordinates=neighbors_coordinates[i],
                                        n_species=neighbors_species[i],
                                        n_limit=neighbors_limit[i],
                                        pair_atoms=patoms,
                                        Rc=Rc,
                                        eta=eta,
                                        lamBda=lb,
                                        zeta=zeta))
            G.append(g)
            
    return G


@numba.njit(cache=True, nogil=True, fastmath=True)
def G4(Ri, n_coordinates, n_species, n_limit, pair_atoms,
       Rc=6.5, eta=2, lamBda=1, zeta=1):
    """Calculate G4 symmetry function.
    
    G4 function is an angular function utilizing the cosine funtion of the
    angle theta_ijk centered at atom i. One can refer to equation 8 in
    the journal paper described above.
    
    Parameters
    ----------
    Ri: array
        The coordinate(x, y, z) of the i-th core atom.
    n_coordinates: 2D array
        The coordinates of the neighbors of the i-th core atom.
    n_species: array of int
        The species of the neighbors of the i-th core atom.
    n_limit: float
        The number of neighbors of the i-th core atom.
    pair_atoms: int
        The atomic numbers of the pair "bonded" atom.
    Rc: float
        The cutoff radius.
    eta: float
        A parameter of G4 symmetry function.
    lamBda: float
        LamBda take values from -1 to +1 shifting the maxima of the cosine
        function to 0 to 180 degree.
    zeta: float
        The angular resolution. Zeta with high values give a narrower range of
        the nonzero G4 values. Different zeta values is preferrable for
        distribution of angles centered at each reference atom. In some sense,
        zeta is illustrated as the eta value.
    
    Returns
    -------
    G4: float
        G4 symmetry value.
    """
    G4 = 0.0
    for j in range(n_limit-1):
        for k in range(j+1, n_limit):
            n1 = n_species[j]
            n2 = n_species[k]
            if (pair_atoms[0] == n1 and pair_atoms[1] == n2) or \
                (pair_atoms[1] == n1 and pair_atoms[0] == n2):
                Rj = n_coordinates[j]
                Rk = n_coordinates[k]
            
                Rij_vector = Rj - Ri
                Rij = np.linalg.norm(Rij_vector)
            
                Rik_vector = Rk - Ri
                Rik = np.linalg.norm(Rik_vector)
            
                Rjk_vector = Rk - Rj
                Rjk = np.linalg.norm(Rjk_vector)
            
                cos_ijk = np.dot(Rij_vector, Rik_vector)/ Rij / Rik
                term = (1. + lamBda * cos_ijk) ** zeta
                term *= np.exp(-eta * 
                               (Rij ** 2. + Rik ** 2. + Rjk ** 2.) /
                               Rc ** 2.)
                term *= Cosine(Rij, Rc) * Cosine(Rik, Rc) * Cosine(Rjk, Rc)
                G4 += term

    G4 *= 2. ** (1. - zeta)
    
    return G4


@numba.njit(cache=True, nogil=True, fastmath=True)
def G4Prime(Ri, i, n_coordinates, n_species, n_indexes, n_limit, pair_atoms,
             Rc=6.5, eta=2, lamBda=1, zeta=1, p=1, q=0):
    """Calculate the derivative of the G4 symmetry function.
    
    Parameters
    ----------
    Ri: array
        The coordinate(x, y, z) of the i-th core atom.
    i: int
        The index of the i-th core atom.
    n_coordinates: 2D array
        The coordinates of the neighbors of the i-th core atom.
    n_species: array of int
        The species of the neighbors of the i-th core atom.
    n_limit: float
        The number of neighbors of the i-th core atom.
    pair_atoms: int
        The atomic numbers of the pair "bonded" atom.
    Rc: float
        The cutoff radius.
    eta: float
        A parameter of G4 symmetry function.
    lamBda: float
        LamBda take values from -1 to +1 shifting the maxima of the cosine
        function to 0 to 180 degree.
    zeta: float
        The angular resolution. Zeta with high values give a narrower range of
        the nonzero G4 values. Different zeta values is preferrable for
        distribution of angles centered at each reference atom. In some sense,
        zeta is illustrated as the eta value.
    p: int
        The atom that the force is acting on.
    q: int
        Direction of force.
        
    Returns
    -------
    G4p: float
        The derivative of G4 symmetry function.        
    """
    G4p = 0
    for j in range(n_limit-1):
        for k in range(j+1, n_limit):
            n1 = n_species[j]
            n2 = n_species[k]
            if (pair_atoms[0] == n1 and pair_atoms[1] == n2) or \
                (pair_atoms[1] == n1 and pair_atoms[0] == n2):
                Rj = n_coordinates[j]
                Rk = n_coordinates[k]
                
                Rij_vector = Rj - Ri
                Rij = np.linalg.norm(Rij_vector)
                
                Rik_vector = Rk - Ri
                Rik = np.linalg.norm(Rik_vector)
                
                Rjk_vector = Rk - Rj
                Rjk = np.linalg.norm(Rjk_vector)
                
                j_index = n_indexes[j]
                k_index = n_indexes[k]
                cos_ijk = np.dot(Rij_vector, Rik_vector)/ Rij / Rik
                dcos_ijk = dcos_dRpq(i, j_index, k_index, Ri, Rj, Rk, p, q)
                
                cutoff = Cosine(Rij, Rc) * Cosine(Rik, Rc) * Cosine(Rjk, Rc)
                cutoff_Rik_Rjk = Cosine(Rik, Rc) * Cosine(Rjk, Rc)
                cutoff_Rij_Rjk = Cosine(Rij, Rc) * Cosine(Rjk, Rc)
                cutoff_Rij_Rik = Cosine(Rij, Rc) * Cosine(Rik, Rc)
                
                dRij = dRab_dRpq(i, j_index, Ri, Rj, p, q)
                dRik = dRab_dRpq(i, k_index, Ri, Rk, p, q)
                dRjk = dRab_dRpq(j_index, k_index, Rj, Rk, p, q)
                
                cutoff_Rij_derivative = CosinePrime(Rij, Rc) * dRij
                cutoff_Rik_derivative = CosinePrime(Rik, Rc) * dRik
                cutoff_Rjk_derivative = CosinePrime(Rjk, Rc) * dRjk
                
                lamBda_term = 1. + lamBda * cos_ijk
                
                first_term = lamBda * zeta * dcos_ijk
                first_term += (-2. * eta * lamBda_term / (Rc ** 2)) * \
                                (Rij * dRij + Rik * dRik + Rjk * dRjk)
                first_term *= cutoff
                
                second_term = cutoff_Rij_derivative * cutoff_Rik_Rjk + \
                                    cutoff_Rik_derivative * cutoff_Rij_Rjk + \
                                    cutoff_Rjk_derivative * cutoff_Rij_Rik
                second_term *= lamBda_term
                
                term = first_term + second_term
                term *= lamBda_term ** (zeta - 1.)
                term *= np.exp(-eta * (Rij ** 2. + Rik ** 2. + Rjk ** 2.) /
                               Rc ** 2.)
                
                G4p += term
                
    G4p *= 2. ** (1. - zeta)
                            
    return G4p


@numba.njit(cache=True, nogil=True, fastmath=True)
def get_G5_or_G5Prime(num_of_cores, pair_atoms, cores_species, 
                      cores_coordinates, neighbors_species, 
                      neighbors_coordinates, neighbors_images, 
                      neighbors_indexes, neighbors_limit, Rc, etas, zetas, 
                      lamBdas, derivative):
    """A function to get G5 or the derivative of G5.
    
    Parameters
    ----------
    num_of_cores: int
        The number of atoms in the crystal unit cell.
    pair_atoms: 2D array
        Contains the pair of atomic kinds in the crystal structure. 
        These atoms have been translated into integer.
    cores_species: array of int
        Contains the atomic number of the core atoms in the unit cell.
    cores_coordinates: 2D array of int
        Contains the coordinates of each atoms in the unit cell.
    neighbors_species: 2D of int
        Contains the atomic number of the neighbors of the core atoms.
    neighbors_coordinates: 3D array
        Contains the coordinates of the neighbors of the core atoms.
    neighbors_images: 3D array
        Contains the images of the neighbors of the core atoms.
    neighbors_indexes: 2D array of int
        Contains the indexes of the neighbors of the core atoms.
    neighbors_limit: array of int
        Contains the number of neighbors for each core atoms.
    Rc: float
        The cutoff radius.
    etas: array
        Parameters of G5 symmetry function.
    zetas: array
        Parameters of G5 symmetry function.
    lamBdas: array
        Parameters of G5 symmetry function.
    derivative:
        If True, calculate the derivative of G5 instead of G5.
        
    Returns
    -------
    G: array
        The G5 or the derivative of G5.
    """
    G = []
    
    if derivative:
        for i in range(num_of_cores):
            nl = neighbors_limit[i]
            nc = neighbors_coordinates[i]
            ns = neighbors_species[i]
            ni = neighbors_indexes[i]
            for q in range(3):
                gp = []
                for eta in etas:
                    for zeta in zetas:
                        for lb in lamBdas:
                            for patoms in pair_atoms:
                                prime = G5Prime(Ri=cores_coordinates[i],
                                                i=i,
                                                n_coordinates=nc,
                                                n_species=ns,
                                                n_indexes=ni,
                                                n_limit=nl,
                                                pair_atoms=patoms,
                                                Rc=Rc,
                                                eta=eta,
                                                zeta=zeta,
                                                lamBda=lb,
                                                p=i,
                                                q=q)
                                gp.append(prime)
                G.append(gp)
                
                for j in range(neighbors_limit[i]):
                    for _ in range(3):
                        if neighbors_images[i][j][_] == 0.:
                            boolean = True
                        else:
                            boolean = False
                            break
                    
                    if boolean:
                        gpp = []
                        for eta in etas:
                            for zeta in zetas:
                                for lb in lamBdas:
                                    for patoms in pair_atoms:
                                        index = neighbors_indexes[i][j]
                                        cc = cores_coordinates[index]
                                        nci = neighbors_coordinates[index]
                                        nsi = neighbors_species[index]
                                        nii = neighbors_indexes[index]
                                        nli = neighbors_limit[index]
                                        prime = G5Prime(Ri=cc,
                                                        i=index,
                                                        n_coordinates=nci,
                                                        n_species=nsi,
                                                        n_indexes=nii,
                                                        n_limit=nli,
                                                        pair_atoms=patoms,
                                                        Rc=Rc,
                                                        eta=eta,
                                                        lamBda=lb,
                                                        zeta=zeta,
                                                        p=i,
                                                        q=q)
                                        gpp.append(prime)
                        G.append(gpp)
                        
    else:
        for i in range(num_of_cores):
            g = []
            for eta in etas:
                for zeta in zetas:
                    for lb in lamBdas:
                        for patoms in pair_atoms:
                            g.append(G5(Ri=cores_coordinates[i],
                                        n_coordinates=neighbors_coordinates[i],
                                        n_species=neighbors_species[i],
                                        n_limit=neighbors_limit[i],
                                        pair_atoms=patoms,
                                        Rc=Rc,
                                        eta=eta,
                                        lamBda=lb,
                                        zeta=zeta))
            G.append(g)
            
    return G


@numba.njit(cache=True, nogil=True, fastmath=True)
def G5(Ri, n_coordinates, n_species, n_limit, pair_atoms,
       Rc=6.5, eta=2, lamBda=1, zeta=1):
    """Calculate G5 symmetry function.
    
    G5 function is also an angular function utilizing the cosine function of 
    the angle theta_ijk centered at atom i. The difference between G5 and G4 is 
    that G5 does not depend on the Rjk value. Hence, the G5 will generate a 
    greater value after the summation compared to G4. One can refer to 
    equation 9 in the journal paper described above.
    
    Parameters
    ----------
    Ri: array
        The coordinate(x, y, z) of the i-th core atom.
    n_coordinates: 2D array
        The coordinates of the neighbors of the i-th core atom.
    n_species: array of int
        The species of the neighbors of the i-th core atom.
    n_limit: float
        The number of neighbors of the i-th core atom.
    pair_atoms: int
        The atomic numbers of the pair "bonded" atom.
    Rc: float
        The cutoff radius.
    eta: float
        A parameter of G5 symmetry function.
    lamBda: float
        LamBda take values from -1 to +1 shifting the maxima of the cosine
        function to 0 to 180 degree.
    zeta: float
        The angular resolution. Zeta with high values give a narrower range of
        the nonzero G5 values. Different zeta values is preferrable for
        distribution of angles centered at each reference atom. In some sense,
        zeta is illustrated as the eta value.
    
    Returns
    -------
    G5: float
        G5 symmetry value.
    """
    G5 = 0.0
    for j in range(n_limit-1):
        for k in range(j+1, n_limit):
            n1 = n_species[j]
            n2 = n_species[k]
            if (pair_atoms[0] == n1 \
                and pair_atoms[1] == n2) or \
                (pair_atoms[1] == n1 and \
                 pair_atoms[0] == n2):
                Rij_vector = Ri - n_coordinates[j]
                Rij = np.linalg.norm(Rij_vector)
                Rik_vector = Ri - n_coordinates[k]
                Rik = np.linalg.norm(Rik_vector)
                cos_ijk = np.dot(Rij_vector, Rik_vector)/ Rij / Rik
                term = (1. + lamBda * cos_ijk) ** zeta
                term *= np.exp(-eta * 
                               (Rij ** 2. + Rik ** 2.) / Rc ** 2.)
                term *= Cosine(Rij, Rc) * Cosine(Rik, Rc)
                G5 += term
    G5 *= 2. ** (1. - zeta)
       
    return G5


@numba.njit(cache=True, nogil=True, fastmath=True)
def G5Prime(Ri, i, n_coordinates, n_species, n_indexes, n_limit, pair_atoms,
             Rc=6.5, eta=2, lamBda=1, zeta=1, p=1, q=0):
    """Calculate the derivative of the G5 symmetry function.
    
    Parameters
    ----------
    Ri: array
        The coordinate(x, y, z) of the i-th core atom.
    i: int
        The index of the i-th core atom.
    n_coordinates: 2D array
        The coordinates of the neighbors of the i-th core atom.
    n_species: array of int
        The species of the neighbors of the i-th core atom.
    n_limit: float
        The number of neighbors of the i-th core atom.
    pair_atoms: int
        The atomic numbers of the pair "bonded" atom.
    Rc: float
        The cutoff radius.
    eta: float
        A parameter of G5 symmetry function.
    lamBda: float
        LamBda take values from -1 to +1 shifting the maxima of the cosine
        function to 0 to 180 degree.
    zeta: float
        The angular resolution. Zeta with high values give a narrower range of
        the nonzero G5 values. Different zeta values is preferrable for
        distribution of angles centered at each reference atom. In some sense,
        zeta is illustrated as the eta value.
    p: int
        The atom that the force is acting on.
    q: int
        Direction of force.
        
    Returns
    -------
    G5p: float
        The derivative of G5 symmetry function.
    """
    G5p = 0
    for j in range(n_limit-1):
        for k in range(j+1, n_limit):
            n1 = n_species[j]
            n2 = n_species[k]
            if (pair_atoms[0] == n1 and pair_atoms[1] == n2) or \
                (pair_atoms[1] == n1 and pair_atoms[0] == n2):
                Rj = n_coordinates[j]
                Rk = n_coordinates[k]
                
                Rij_vector = Rj - Ri
                Rik_vector = Rk - Ri
                Rij = np.linalg.norm(Rij_vector)
                Rik = np.linalg.norm(Rik_vector)
                
                j_index = n_indexes[j]
                k_index = n_indexes[k]
                cos_ijk = np.dot(Rij_vector, Rik_vector) / Rij / Rik
                dcos_ijk = dcos_dRpq(i, j_index, k_index, Ri, Rj, Rk, p, q)
                
                cutoff = Cosine(Rij, Rc) * Cosine(Rik, Rc)
                cutoff_Rij_derivative = CosinePrime(Rij, Rc) * \
                                        dRab_dRpq(i, j_index, Ri, Rj, p, q)
                cutoff_Rik_derivative = CosinePrime(Rik, Rc) * \
                                        dRab_dRpq(i, k_index, Ri, Rk, p, q)

                lamBda_term = 1. + lamBda * cos_ijk
                
                first_term = -2 * eta / Rc ** 2 * lamBda_term * \
                                (Rij * dRab_dRpq(i, j_index, Ri, Rj, p, q) + 
                                 Rik * dRab_dRpq(i, k_index, Ri, Rk, p, q))
                first_term += lamBda * zeta * dcos_ijk
                first_term *= cutoff
                
                second_term = lamBda_term * \
                                (cutoff_Rij_derivative * Cosine(Rik, Rc) + 
                                 cutoff_Rik_derivative * Cosine(Rij, Rc))
                                
                term = first_term + second_term
                term *= lamBda_term ** (zeta - 1.)
                term *= np.exp(-eta * (Rij ** 2. + Rik ** 2.) /
                               Rc ** 2.)
                                
                G5p += term

    G5p *= 2. ** (1. - zeta)
        
    return G5p
    

############################ Cutoff Functionals ###############################
    

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


########################### Auxiliary Functions ###############################
        
        
@numba.njit(numba.f8(numba.i8, numba.i8, numba.f8[:], 
                     numba.f8[:], numba.i8, numba.i8),
            cache=True, nogil=True, fastmath=True)
def dRab_dRpq(a, b, Ra, Rb, p, q):
    """
    Calculate the derivative of the norm of position vector R_{ab} with
    respect to coordinate x, y, or z denoted by q of atom with index p.
    See Eq. 14c of the supplementary information of Khorshidi, Peterson,
    CPC(2016).
    Parameters
    ----------
    a : int
        Index of the first atom.
    b : int
        Index of the second atom.
    Ra : float
        Position of the first atom.
    Rb : float
        Position of the second atom.
    p : int
        Index of the atom force is acting on.
    q : int
        Direction of force. x = 0, y = 1, and z = 2.
    Returns
    -------
    the derivative of pair atoms w.r.t. one of the atom in q direction.
    """
    Rab = np.linalg.norm(Rb - Ra)
    if p == a and a != b:  # a != b is necessary for periodic systems
        dRab_dRpq = -(Rb[q] - Ra[q]) / Rab
    elif p == b and a != b:  # a != b is necessary for periodic systems
        dRab_dRpq = (Rb[q] - Ra[q]) / Rab
    else:
        dRab_dRpq = 0.

    return dRab_dRpq


@numba.njit(cache=True, nogil=True, fastmath=True)
def dcos_dRpq(a, b, c, Ra, Rb, Rc, p, q):
    """
    Calculate the derivative of cosine dot product function with respect to 
    the radius of an atom m in a particular direction l.
    
    Parameters
    ----------
    a: int
        Index of the center atom.
    b: int
        Index of the first neighbor atom.
    c: int
        Index of the second neighbor atom.
    Ra: list of floats
        Position of the center atom.
    Rb: list of floats
        Position of the first atom.
    Rc: list of floats
        Postition of the second atom.
    p: int
        Atom that is experiencing force.
    q: int
        Direction of force. x = 0, y = 1, and z = 2.
        
    Returns
    -------
    Derivative of cosine dot product w.r.t. the radius of an atom m in a 
    particular direction l.
    """
    Rab_vector = Rb - Ra
    Rac_vector = Rc - Ra
    Rab = np.linalg.norm(Rab_vector)
    Rac = np.linalg.norm(Rac_vector)
    
    f_term = 1 / (Rab * Rac) * \
            np.dot(dRab_dRpq_vector(a, b, p, q), Rac_vector)
    s_term = 1 / (Rab * Rac) * \
                    np.dot(Rab_vector, dRab_dRpq_vector(a, c, p, q))
    t_term = np.dot(Rab_vector, Rac_vector) / Rab ** 2 / Rac * \
                    dRab_dRpq(a, b, Ra, Rb, p, q)
    fo_term = np.dot(Rab_vector, Rac_vector) / Rab / Rac ** 2 * \
                    dRab_dRpq(a, c, Ra, Rc, p, q)

    return (f_term + s_term - t_term - fo_term)


@numba.njit(cache=True, nogil=True, fastmath=True)
def dRab_dRpq_vector(a, b, p, q):
    """
    Calculate the derivative of the position vector R_{ab} with
    respect to coordinate x, y, or z denoted by q of atom with index p.
    See Eq. 14d of the supplementary information of Khorshidi, Peterson,
    CPC(2016).
    Parameters
    ----------
    a : int
        Index of the first atom.
    b : int
        Index of the second atom.
    p : int
        Index of the atom force is acting on.
    q : int
        Direction of force. x = 0, y = 1, and z = 2.
    Returns
    -------
    list of float
        The derivative of the position vector R_{ab} with respect to atom 
        index p in direction of q.
    """
    dRab_dRpq_vector = np.zeros((3,))
    if (p == a) or (p == b):
        c1 = Kronecker(p, b) - Kronecker(p, a)
        dRab_dRpq_vector[0] += c1 * Kronecker(0, q)
        dRab_dRpq_vector[1] += c1 * Kronecker(1, q)
        dRab_dRpq_vector[2] += c1 * Kronecker(2, q)
        
    return dRab_dRpq_vector


@numba.njit(numba.i8(numba.i8, numba.i8),
            cache=True, nogil=True, fastmath=True)
def Kronecker(a,b):
    """
    Kronecker delta function.
    
    Parameters
    ----------
    a: int
        first index
    b: int
        second index
        
    Returns
    -------
        Return value 0 if a and b are not equal; return value 1 if a and b 
        are equal.
    """
    if a == b:
        kronecker = 1
    else:
        kronecker = 0
    return kronecker

# Test
# To run this test script you need to make sure fbehlerparrinello.py 
# know where to find neighborhood (see line 4).

#from ase.calculators.emt import EMT
#from ase.build import fcc110
#from ase import Atoms, Atom
#from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
#from ase import units
#from ase.md import VelocityVerlet
#from ase.constraints import FixAtoms
#
##from pymatgen.core.structure import Structure
#from pymatgen.io.ase import AseAtomsAdaptor
#
#def generate_data(count):
#    """Generates test or training data with a simple MD simulation."""
#    atoms = fcc110('Pt', (2, 2, 2), vacuum=7.)
#    adsorbate = Atoms([Atom('Cu', atoms[7].position + (0., 0., 2.5)),
#                       Atom('Cu', atoms[7].position + (0., 0., 5.))])
#    atoms.extend(adsorbate)
#    atoms.set_constraint(FixAtoms(indices=[0, 2]))
#    MaxwellBoltzmannDistribution(atoms, 300. * units.kB)
#    dyn = VelocityVerlet(atoms, dt=1. * units.fs)
#    atoms.set_calculator(EMT())
#    newatoms = atoms.copy()
#    newatoms.set_calculator(EMT())
#    newatoms.get_potential_energy()
#    images = [newatoms]
#    for step in range(count - 1):
#        dyn.run(50)
#        newatoms = atoms.copy()
#        newatoms.set_calculator(EMT())
#        newatoms.get_potential_energy()
#        images.append(newatoms)
#    return images
#
#crystal = AseAtomsAdaptor.get_structure(generate_data(1)[0])
#
#symmetry = {'G4': {'lambda': [1], 'zeta':[1], 'eta': [0.036, 0.071], }}
#
#bp = BehlerParrinello(crystal, symmetry, Rc=5.2)
#for z1, z2 in bp.Gs['Gprime']:
#    print(z1, ": ", z2)
