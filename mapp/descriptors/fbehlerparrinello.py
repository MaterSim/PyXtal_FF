import numpy as np
import itertools
import numba
from neighborhood import Element, Neighborhood


################################ Gaussian Class ###############################


class BehlerParrinello:
    def __init__(self, crystal, symmetry_parameters, Rc=6.5, derivative=True):
        # Set up the symmetry parameters keywords. If a string are not in the
        # keyword, code will return an error.
        self.G1_keywords = []
        self.G2_keywords = ['eta', 'Rs']
        self.G3_keywords = ['kappa',]
        self.G4_keywords = ['eta', 'lambda', 'zeta',]
        self.G5_keywords = ['eta', 'lambda', 'zeta',]
        
        # Extract useful quantities from Pymatgen Structure object.
        self.crystal = crystal
        self.num_of_cores = crystal.num_sites # Integer
        self.atom_types = Element(1, crystal.symbol_set).convert() # ['Na', 'Cl']
        # element pair types, i.e. [('Na', 'Na'), ('Na', 'Cl'), ('Cl', 'Cl')]
        self.pair_atoms = list(
                            itertools.combinations_with_replacement(
                            self.atom_types, 2))

        cores_species = []
        self.cores_coordinates = []
        for i in range(self.num_of_cores):
            self.cores_coordinates.append(crystal.cart_coords[i])
            cores_species.append(crystal[i].species_string)
        self.c_species = cores_species  # string
        self.cores_species = Element(1, cores_species).convert()
        self.cores_coordinates = np.asarray(self.cores_coordinates)
        
        # Obtain neighbors information from Pymatgen Structure object.
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
        self.n_species = neighbors_species # string
        
        neighbors = Neighborhood(neighbors_coordinates, neighbors_species, 
                                 neighbors_images, neighbors_indexes)
        self.neighbors_coordinates, self.neighbors_limit = neighbors.get_coordinates()
        self.neighbors_species = neighbors.get_species()
        self.neighbors_images = neighbors.get_images()
        self.neighbors_indexes = neighbors.get_indexes()
        
        self.Rc = Rc
        
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
            Rs = 0.
            G2 = {}
            G2['G'] = []
            G2['Gprime'] = []
            
            # Check sanity
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
            
            # Get and calculate G2
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
                                        self.Rc, self.Rs, etas, 
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
            #print(g4)
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
        
        self.all_G = all_G

        return self.all_G
        
        
def get_G1(derivative):
    pass


@numba.njit(cache=True, nogil=True, fastmath=True)
def get_G2_or_G2Prime(num_of_cores, atom_types, cores_species, 
                      cores_coordinates, neighbors_species, 
                      neighbors_coordinates, neighbors_images, 
                      neighbors_indexes, neighbors_limit, Rc, Rs, etas, 
                      derivative):
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
def get_G4_or_G4Prime(num_of_cores, pair_atoms, cores_species, 
                      cores_coordinates, neighbors_species, 
                      neighbors_coordinates, neighbors_images, 
                      neighbors_indexes, neighbors_limit, Rc, etas, zetas, 
                      lamBdas, derivative):
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
def G2(Ri, n_coordinates, n_species, n_limit, atom, Rc=6.5, Rs=0., eta=2):
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
def G4(Ri, n_coordinates, n_species, n_limit, pair_atoms,
       Rc=6.5, eta=2, lamBda=1, zeta=1):
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
                
                cos_ijk = np.dot(Rij_vector, Rik_vector)/ Rij / Rik
                dcos_ijk = dcos_dRpq(i, n_indexes[j], n_indexes[k], Ri, Rj, Rk, p, q)
                
                cutoff = Cosine(Rij, Rc) * Cosine(Rik, Rc) * Cosine(Rjk, Rc)
                cutoff_Rik_Rjk = Cosine(Rik, Rc) * Cosine(Rjk, Rc)
                cutoff_Rij_Rjk = Cosine(Rij, Rc) * Cosine(Rjk, Rc)
                cutoff_Rij_Rik = Cosine(Rij, Rc) * Cosine(Rik, Rc)
                
                dRij = dRab_dRpq(i, n_indexes[j], Ri, Rj, p, q)
                dRik = dRab_dRpq(i, n_indexes[k], Ri, Rk, p, q)
                dRjk = dRab_dRpq(n_indexes[j], n_indexes[k], Rj, Rk, p, q)
                
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
        dRab_dRpq = 0.

    return dRab_dRpq

@numba.njit(cache=True, nogil=True, fastmath=True)
def dcos_dRpq(a, b, c, Ra, Rb, Rc, p, q):
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
    if a == b:
        kronecker = 1
    else:
        kronecker = 0
    return kronecker

# Test
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
