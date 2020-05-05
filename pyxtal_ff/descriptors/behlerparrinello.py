# Numpy Version

import numpy as np
from ase.neighborlist import NeighborList
from itertools import combinations, combinations_with_replacement

class BehlerParrinello:
    """A class for calculating Behler-Parrinello symmetry functions.
    
    The forms of the functions are consistent with the 
    functions presented in:
        Behler, J. (2011). The Journal of Chemical Physics, 134(7), 074106.
        
    Parameters
    ----------
    symmetry_parameters: dict
        The user-defined parameters for the symmetry functions calculations.
        i.e. {'G2': {'eta': [0.1, 0.2,], 'Rs': [0.0, 0.1]}}
    Rc: float
        The symmetry functions will be calculated within this radius.
    derivative: bool
        If True, calculate the derivatives of symmetry functions.
    stress: bool
        If True, calculate the virial stress contribution of EAMD.
    """
    def __init__(self, symmetry_parameters, Rc=6.5, derivative=True, stress=False):
        self._type = 'BehlerParrinello'
        
        # Set up the symmetry parameters keywords. If a string are not in the
        # keyword, code will return an error.
        G2_keywords = ['Rs', 'eta']
        G4_keywords = ['Rs', 'eta', 'lambda', 'zeta']
        G5_keywords = G4_keywords
        
        # Setting up parameters for each of the symmetry function type.
        self.G2_parameters = None
        self.G4_parameters = None
        self.G5_parameters = None
        
        for key, value in symmetry_parameters.items():
            if key == 'G2':
                self.G2_parameters = {'eta': [1], 'Rs': np.array([0.])}
                for key0, value0 in value.items():
                    if key0 in G2_keywords:
                        self.G2_parameters[key0] = to_array(value0)
                    else:
                        msg = f"{key0} is not available. "\
                        f"Choose from {G2_keywords}"
                        raise NotImplementedError(msg)
                        
            elif key == 'G4':
                self.G4_parameters = {'Rs': np.array([0.]), 
                                      'eta': [1], 'zeta': [1], 'lambda': [1]}
                for key0, value0 in value.items():
                    if key0 in G4_keywords:
                        self.G4_parameters[key0] = to_array(value0)
                    else:
                        msg = f"{key0} is not available. "\
                        f"Choose from {G4_keywords}"
                        raise NotImplementedError(msg)
                        
            elif key == 'G5':
                self.G5_parameters = {'Rs': np.array([0.]), 
                                      'eta': [1], 'zeta': [1], 'lambda': [1]}
                for key0, value0 in value.items():
                    if key0 in G5_keywords:
                        self.G5_parameters[key0] = to_array(value0)
                    else:
                        msg = f"{key0} is not available. "\
                        f"Choose from {G5_keywords}"
                        raise NotImplementedError(msg)

        self.Rc = Rc
        self.derivative = derivative
        self.stress = stress
        
       
    def calculate(self, crystal, system=None):
        """The symmetry functions are obtained through this `calculate` method.
        
        Parameters
        ----------
        crystal: object
            ASE Structure object.
        system: list
            A list of the crystal structures system. 
            All elements in the list have to be integer. For example, the
            system of crystal structure is NaCl system. Then, system should be
            pass as [11, 17]
            
        Returns
        -------
        all_G: dict
            The user-defined symmetry functions that represent the crystal.
            
            Currently, there are 3 types of symmetry functions implemented. 
            Here are the order of the descriptors are printed out based on
            their symmetry parameters:
                - G2: ["element", "Rs", "eta"]
                - G4: ["pair_elements", "eta", "lambda", "zeta"]
                - G5: ["pair_elements", "eta", "lambda", "zeta"]
        """
        self.crystal = crystal
        atomic_numbers = np.array(crystal.get_atomic_numbers())
        vol = crystal.get_volume()

        if system is None:
            type_set1 = create_type_set(atomic_numbers, 1)   # l
            type_set2 = create_type_set(atomic_numbers, 2)   # l
        else:
            system = np.array(system, dtype=int)
            type_set1 = create_type_set(system, 1)   # l
            type_set2 = create_type_set(system, 2)   # l
                    
        # Initialize dict for the symmetry functions
        self.all_G = {'x':[], 
                      'elements': []}
        if self.derivative:
            self.all_G['dxdr'] = []
        if self.stress:
            self.all_G['rdxdr'] = []

        # Obtain neighbors info.
        rc = [self.Rc/2]*len(self.crystal)
        neighbors = NeighborList(rc, self_interaction=False, bothways=True, 
                                 skin=0.0)
        neighbors.update(crystal)
        
        for i in range(len(crystal)):
            element = crystal.get_chemical_symbols()[i]
            indices, offsets = neighbors.get_neighbors(i)
            
            assert len(indices)>0, \
            f"There's no neighbor for this structure at Rc = {self.Rc} A."

            Ri = crystal.get_positions()[i]
            total_neighbors = len(indices) # total number of neighbors of atom i
            
            Rj = np.zeros([total_neighbors, 3])
            Dij = np.zeros(total_neighbors)
            IDs = np.zeros(total_neighbors, dtype=int)
            jks = np.array(list(combinations(range(total_neighbors), 2)))
            
            count = 0
            for j, offset in zip(indices, offsets):
                Rj[count, :] = crystal.positions[j] + np.dot(offset, crystal.get_cell())
                Dij[count] = np.sqrt(sum((Rj[count,:] - Ri) ** 2))
                IDs[count] = j
                count += 1
                
            Rij = Rj - Ri
            
            Gi = []
            GiPrime = None

            if self.G2_parameters is not None:
                G2i = calculate_G2(Dij, IDs, atomic_numbers, type_set1, 
                                   self.Rc, self.G2_parameters)
                Gi = np.append(Gi, G2i)
                
                if self.derivative:
                    G2iP, rG2iP = calculate_G2Prime(Rij, Ri, i, IDs, atomic_numbers,
                                             type_set1, self.Rc, 
                                             self.G2_parameters)
                    if GiPrime is None:
                        GiPrime = G2iP
                        if self.stress:
                            rGiPrime = rG2iP
                    else:
                        GiPrime = np.append(GiPrime, G2iP, axis=1)
                        if self.stress:
                            rGiPrime = np.append(rGiPrime, rG2iP, axis=1)
                
            if self.G4_parameters is not None:
                G4i = calculate_G4(Rij, IDs, jks, atomic_numbers, type_set2,
                                   self.Rc, self.G4_parameters)
                Gi = np.append(Gi, G4i)
                
                if self.derivative:
                    G4iP, rG4iP = calculate_G4Prime(Rij, Ri, i, IDs, jks, atomic_numbers,
                                             type_set2, self.Rc, 
                                             self.G4_parameters)
                    if GiPrime is None:
                        GiPrime = G4iP
                        if self.stress:
                            rGiPrime = rG4iP
                    else:
                        GiPrime = np.append(GiPrime, G4iP, axis=1)
                        if self.stress:
                            rGiPrime = np.append(rGiPrime, rG4iP, axis=1)

            if self.G5_parameters is not None:
                G5i = calculate_G5(Rij, IDs, jks, atomic_numbers, type_set2,
                                   self.Rc, self.G5_parameters)
                Gi = np.append(Gi, G5i)

                if self.derivative:
                    G5iP, rG5iP = calculate_G5Prime(Rij, Ri, i, IDs, jks, atomic_numbers,
                                             type_set2, self.Rc,
                                             self.G5_parameters)
                    if GiPrime is None:
                        GiPrime = G5iP
                        if self.stress:
                            rGiPrime = rG5iP
                    else:
                        GiPrime = np.append(GiPrime, G5iP, axis=1)
                        if self.stress:
                            rGiPrime = np.append(rGiPrime, rG5iP, axis=1)
            
            self.all_G['x'].append(Gi)
            self.all_G['elements'].append(element)
            if self.derivative:
                self.all_G['dxdr'].append(GiPrime)
                if self.stress:
                    self.all_G['rdxdr'].append(rGiPrime)
        
        self.all_G['x'] = np.asarray(self.all_G['x'])
        if self.derivative:
            self.all_G['dxdr'] = np.asarray(self.all_G['dxdr'])
            if self.stress:
                self.all_G['rdxdr'] = -np.asarray(self.all_G['rdxdr'])/vol
            else:
                self.all_G['rdxdr'] = None
        return self.all_G
    
        
########################### Symmetry Functions ################################


def calculate_G2(Dij, IDs, atomic_numbers, type_set, Rc, parameters):
    """Calculate G2 symmetry function for a center atom i.
    
    G2 function describes the radial feature of atoms in a crystal structure
    given a cutoff radius. 

    Parameters
    ----------
    Dij: float array [m]
        The array of distances for a given atom.
    IDs: int array [m]
        The indices of neighbors centering about atom i.
    atomic_numbers: int array [N]
        The elemental indices for atoms within the unitcell.
        e.g. [11, 11, 11, 11, 17, 17, 17, 17]
    Rc: float
        The cutoff radius.
    parameters: dict
        Rs: float array (n1)
            The shift from the center of the G2 symmetry function.
        etas: float array (n2)
             eta parameters of G2 symmetry function.
    
    Returns
    -------
    G2: 1D float array [d = n1*n2*l]
        G2 symmetry value.
    """
    Rs = parameters['Rs']       # n1
    etas = parameters['eta']    # n2
    n1, n2, m, l = len(Rs), len(etas), len(Dij), len(type_set)

    d20 = (Dij - Rs[:, np.newaxis]) ** 2  # n1*m
    term = np.exp(np.einsum('i,jk->ijk', -etas, d20)) # n2*n1*m
    results = np.einsum('ijk,k->ijk', term, Cosine(Dij, Rc)) # n2*n1*m
    results = results.reshape([n1*n2, m])
    
    # Decompose G2 by species
    G2 = np.zeros([n1*n2*l])
    j_ids = atomic_numbers[IDs]
    for id, j_type in enumerate(type_set):
        ids = select_rows(j_ids, j_type)
        G2[id::l] = np.sum(results[:, ids], axis=1)

    return G2


def calculate_G2Prime(Rij, Ri, i, IDs, atomic_numbers, type_set, Rc, parameters):
    """Calculate the derivative of G2 symmetry function for atom i.
    
    Parameters
    ----------
    Rij: float array [m, 3]
        The vector distances of atom i to neighbors js.
    i: int
        The i-th atom center.
    IDs: int array [m]
        The indices of neighbors centering about atom i.
    atomic_numbers: int array [N]
        The elemental indices for atoms within the unitcell.
        e.g. [11, 11, 11, 11, 17, 17, 17, 17]
    Rc: float
        The cutoff radius.
    parameters: dict
        Rs: float array (n1)
            The shift from the center of the G2 symmetry function.
        etas: float array (n2)
             eta parameters of G2 symmetry function.
    
    Returns
    -------
    G2Prime: 1D float array [m, (n1*n2*l), 3]
        The derivative of G2 symmetry value at i-th atom. m is the index of the
        atom that force is acting on.
        
    """
    etas = parameters['eta']
    Rs = parameters['Rs']
    n1, n2, m, l = len(Rs), len(etas), len(Rij), len(type_set)
    js, N = IDs, len(atomic_numbers)
    
    ij_list = i * np.ones([len(js), 2], dtype=int)
    ij_list[:,1] = js
    
    R2ij = np.sum(Rij**2., axis=1)
    R1ij = np.sqrt(R2ij)
    dij = R1ij - Rs[:, np.newaxis]
    dij2 = (dij) ** 2
    term1 = np.exp(np.einsum('i,jk->ijk', -etas, dij2)) # n2*n1*m
    term21 = CosinePrime(R1ij, Rc) # m
    _term22 = np.einsum('ij,j->ij', dij, Cosine(R1ij, Rc)) # n1*m
    term22 = 2 * np.einsum('i,jk->ijk', etas, _term22) # n2*n1*m
    term2 = term21 - term22
    term_1_and_2 = term1 * term2
    term_1_and_2 = term_1_and_2.reshape([n1*n2, m])

    dRij_dRm = np.zeros([m, 3, N])
    for mm in range(N):
        mm_list = mm * np.ones([m, 1], dtype=int)
        dRij_dRm[:,:,mm] = dRij_dRm_norm(Rij, np.hstack((ij_list, mm_list)))
    
    # [d, m, 3, N]
    G2ip0 = np.einsum('ij,jkl->ijkl', term_1_and_2, dRij_dRm)

    rG2ip0 = np.zeros([n1*n2, len(ij_list), 3, N, 3])
    for mm, ij in enumerate(ij_list):
        j = ij[1]
        tmp = G2ip0[:,mm,:,:] #S,N,3 -> S,3 * 3 -> S*3*3 
        rG2ip0[:,mm,:,i,:] += np.einsum('ij,k->ijk', tmp[:,:,i], Ri)
        rG2ip0[:,mm,:,j,:] += np.einsum('ij,k->ijk', tmp[:,:,j], Rij[mm]+Ri)


    ## this is for j contributions, [d, m, 3, N] * [m, 3] -> [d, m, 3, N, 3] 
    #rG2ip0 = np.einsum('ijkl,jm->ijklm', G2ip0, Rij+Ri)
    ## Now we consider i contribution
    #ij_ids = (js != i)
    #rG2ip0[:,ij_ids,:,i,:] = np.einsum('ijk,l->ijkl', G2ip0[:,ij_ids,:,i], Ri) #[d,m,3], 3,->[d,m,3,3]
    #ii_ids = (js == i)
    #if len(js[ii_ids]) > 0:
    #    rG2ip0[:,ii_ids,:,i,:] = np.einsum('ijk,il->ijkl', G2ip0[:,ii_ids,:,i], Rij[ii_ids]) #[d,m,3], [m,3],->[d,m,3,3]
    
    # Decompose G2 Prime by species
    j_ids = atomic_numbers[js]
    G2Prime = np.zeros([N, n1*n2*l, 3]) # This is the final output
    rG2Prime = np.zeros([N, n1*n2*l, 3, 3]) # This is the final output
    
    for id, j_type in enumerate(type_set):
        ids = select_rows(j_ids, j_type)
        G2Prime[:, id::l, :] += np.einsum('ijkl->lik', G2ip0[:, ids, :, :])
        rG2Prime[:, id::l, :, :] += np.einsum('ijklm->likm', rG2ip0[:, ids, :, :, :])
        
    return G2Prime, rG2Prime


def calculate_G4(Rij, IDs, jks, atomic_numbers, type_set, Rc, parameters):
    """Calculate G4 symmetry function for a given atom i.
    
    G4 function also describes the angular feature of atoms in a crystal 
    structure given a cutoff radius.
    
    Parameters
    ----------
    Rij: array [j, 3]
        The vector distances of atom i to neighbors js.
    IDs: int array [m]
        The indices of neighbors centering about atom i.
    jks: int array [m*(m-1)/2, 2]
        The list of [j,k] pairs
    atomic_numbers: int array [N]
        The elemental indices for atoms within the unitcell.
        e.g. [11, 11, 11, 11, 17, 17, 17, 17]
    Rc: float
        The cutoff radius.
    parameters: dict
        Rs: float array (n1)
        etas: float array (n2)
        lamBdas: float array (n3)
        zetas: float array (n4)
        
    Returns
    -------
    G4: array [d = n1*n2*n3*n4*l]
        G4 symmetry value with d members.
    """
    Rs = parameters['Rs']    
    etas = parameters['eta']
    zetas = parameters['zeta']
    lamBdas = parameters['lambda']
    n1, n2, n3, n4, l = len(Rs), len(etas), len(lamBdas), len(zetas), len(type_set)
    jk = len(jks)  #m1
    
    rij = Rij[jks[:,0]] # [m1, 3]
    rik = Rij[jks[:,1]] # [m1, 3]
    rjk = rik - rij     # [m1, 3]
    R2ij0 = np.sum(rij**2., axis=1) 
    R2ik0 = np.sum(rik**2., axis=1) 
    R2jk0 = np.sum(rjk**2., axis=1) 
    R1ij0 = np.sqrt(R2ij0) # m1
    R1ik0 = np.sqrt(R2ik0) # m1
    R1jk0 = np.sqrt(R2jk0) # m1
    R2ij = R2ij0 - Rs[:, np.newaxis]**2  # n1*m1
    R2ik = R2ik0 - Rs[:, np.newaxis]**2  # n1*m1
    R2jk = R2jk0 - Rs[:, np.newaxis]**2  # n1*m1
    R1ij = R1ij0 - Rs[:, np.newaxis] # n1*m1
    R1ik = R1ik0 - Rs[:, np.newaxis] # n1*m1
    R1jk = R1jk0 - Rs[:, np.newaxis] # n1*m1
 
    powers = 2. ** (1.-zetas) #n4
    cos_ijk = np.sum(rij*rik, axis=1)/R1ij0/R1ik0 # m1 array
    term1 = 1. + np.einsum('i,j->ij', lamBdas, cos_ijk) # n3*m1

    zetas1 = zetas.repeat(n3*jk).reshape([n4, n3, jk])  # n4*n3*m1
    term2 = np.power(term1, zetas1) # n4*n3*m1
    term3 = np.exp(np.einsum('i,jk->ijk', -etas, (R2ij+R2jk+R2ik))) # n2*n1*m1
    term4 = Cosine(R1ij0, Rc) * Cosine(R1ik0, Rc) * Cosine(R1jk0, Rc) # m1
    term5 = np.einsum('ijk,lmk->ijlmk', term2, term3) #n4*n3*n2*n1*m1
    term6 = np.einsum('ijkml,l->ijkml', term5, term4) #n4*n3*n2*n1*m1
    results = np.einsum('i,ijkml->ijkml', powers, term6) #n4*n3*n2*n1*m1
    results = results.reshape([n1*n2*n3*n4, jk])

    # Decompose G4 by species
    G4 = np.zeros([n1*n2*n3*n4*l])
    jk_ids = atomic_numbers[IDs[jks]] 
    for id, jk_type in enumerate(type_set):
        ids = select_rows(jk_ids, jk_type)
        G4[id::l] = np.sum(results[:, ids], axis=1)
        
    return G4


def calculate_G5(Rij, IDs, jks, atomic_numbers, type_set, Rc, parameters):
    """Calculate G5 symmetry function for a given atom i.
    
    G5 function also describes the angular feature of atoms in a crystal 
    structure given a cutoff radius.
    
    Parameters
    ----------
    Rij: array [j, 3]
        The vector distances of atom i to neighbors js.
    IDs: int array [m]
        The indices of neighbors centering about atom i.
    jks: int array [m*(m-1)/2, 2]
        The list of [j,k] pairs
    atomic_numbers: int array [N]
        The elemental indices for atoms within the unitcell.
        e.g. [11, 11, 11, 11, 17, 17, 17, 17]
    Rc: float
        The cutoff radius.
    parameters: dict
        Rs: float array (n1)
        etas: float array (n2)
        lamBdas: float array (n3)
        zetas: float array (n4)
        
    Returns
    -------
    G5: array [d = n1*n2*n3*n4*l]
        G5 symmetry value with d members.
    """
    Rs = parameters['Rs']    
    etas = parameters['eta']
    zetas = parameters['zeta']
    lamBdas = parameters['lambda']
    n1, n2, n3, n4, l = len(Rs), len(etas), len(lamBdas), len(zetas), len(type_set)
    jk = len(jks)  #m1
    
    rij = Rij[jks[:,0]] # [m1, 3]
    rik = Rij[jks[:,1]] # [m1, 3]
    #rjk = rik - rij     # [m1, 3]
    R2ij0 = np.sum(rij**2., axis=1) 
    R2ik0 = np.sum(rik**2., axis=1) 
    
    #R2jk0 = np.sum(rjk**2., axis=1) 
    
    R1ij0 = np.sqrt(R2ij0) # m1
    R1ik0 = np.sqrt(R2ik0) # m1
    #R1jk0 = np.sqrt(R2jk0) # m1

    R2ij = R2ij0 - Rs[:, np.newaxis]**2  # n1*m1
    R2ik = R2ik0 - Rs[:, np.newaxis]**2  # n1*m1
    
    #R2jk = R2jk0 - Rs[:, np.newaxis]**2  # n1*m1
    
    R1ij = R1ij0 - Rs[:, np.newaxis] # n1*m1
    R1ik = R1ik0 - Rs[:, np.newaxis] # n1*m1
    
    #R1jk = R1jk0 - Rs[:, np.newaxis] # n1*m1
 
    powers = 2. ** (1.-zetas) #n4
    cos_ijk = np.sum(rij*rik, axis=1)/R1ij0/R1ik0 # m1 array
    term1 = 1. + np.einsum('i,j->ij', lamBdas, cos_ijk) # n3*m1

    zetas1 = zetas.repeat(n3*jk).reshape([n4, n3, jk])  # n4*n3*m1
    term2 = np.power(term1, zetas1) # n4*n3*m1
    term3 = np.exp(np.einsum('i,jk->ijk', -etas, (R2ij+R2ik))) # n2*n1*m1
    term4 = Cosine(R1ij0, Rc) * Cosine(R1ik0, Rc) #* Cosine(R1jk0, Rc) # m1
    term5 = np.einsum('ijk,lmk->ijlmk', term2, term3) #n4*n3*n2*n1*m1
    term6 = np.einsum('ijkml,l->ijkml', term5, term4) #n4*n3*n2*n1*m1
    results = np.einsum('i,ijkml->ijkml', powers, term6) #n4*n3*n2*n1*m1
    results = results.reshape([n1*n2*n3*n4, jk])

    # Decompose G5 by species
    G5 = np.zeros([n1*n2*n3*n4*l])
    jk_ids = atomic_numbers[IDs[jks]] 
    for id, jk_type in enumerate(type_set):
        ids = select_rows(jk_ids, jk_type)
        G5[id::l] = np.sum(results[:, ids], axis=1)
        
    return G5


def calculate_G4Prime(Rij, Ri, i, IDs, jks, atomic_numbers, type_set, Rc, 
                      parameters):
    """Calculate G4 symmetry function for a given atom i.
    
    G4 function also describes the angular feature of atoms in a crystal 
    structure given a cutoff radius.
    
    Parameters
    ----------
    Rij: array [j, 3]
        The vector distances of atom i to neighbors js.
    i: int
        The i-th atom center.    
    IDs: int array [j]
        The indices of neighbors centering about atom i.
    jks: int array [j*k, 2]
        The list of [j,k] pairs
    atomic_numbers: int array [j]
        The elemental indices for atoms within the unitcell.
        e.g. [11, 11, 11, 11, 17, 17, 17, 17]
    Rc: float
        The cutoff radius.
    parameters: dict
        zetas: float array (n1)
        lamBdas: float array (n2)
        etas: float array (n3)
        
    Returns

        The derivative of G4 symmetry value at i-th atom. m is the index of the
        atom that force is acting on.
    """
    Rs = parameters['Rs']    
    etas = parameters['eta']
    zetas = parameters['zeta']
    lamBdas = parameters['lambda']
    n1, n2, n3, n4, l = len(Rs), len(etas), len(lamBdas), len(zetas), len(type_set)
    N, jk = len(atomic_numbers), len(jks)
    
    ijk_list = i * np.ones([jk, 3], dtype=int)
    ijk_list[:,1] = IDs[jks[:,0]]
    ijk_list[:,2] = IDs[jks[:,1]]

    rij = Rij[jks[:,0]] # [m1, 3]
    rik = Rij[jks[:,1]] # [m1, 3]
    rjk = rik - rij     # [m1, 3]

    R2ij0 = np.sum(rij**2., axis=1) 
    R2ik0 = np.sum(rik**2., axis=1) 
    R2jk0 = np.sum(rjk**2., axis=1) 
    R1ij0 = np.sqrt(R2ij0) # m1
    R1ik0 = np.sqrt(R2ik0) # m1
    R1jk0 = np.sqrt(R2jk0) # m1
    R2ij = R2ij0 - Rs[:, np.newaxis]**2  # n1*m1
    R2ik = R2ik0 - Rs[:, np.newaxis]**2  # n1*m1
    R2jk = R2jk0 - Rs[:, np.newaxis]**2  # n1*m1
    R1ij = R1ij0 - Rs[:, np.newaxis] # n1*m1
    R1ik = R1ik0 - Rs[:, np.newaxis] # n1*m1
    R1jk = R1jk0 - Rs[:, np.newaxis] # n1*m1
 
    cos_ijk = np.sum(rij * rik, axis=1) / R1ij0/ R1ik0 # m1 array

    dfcij = CosinePrime(R1ij0, Rc)
    dfcjk = CosinePrime(R1jk0, Rc)
    dfcik = CosinePrime(R1ik0, Rc)
    fcij = Cosine(R1ij0, Rc)
    fcjk = Cosine(R1jk0, Rc)
    fcik = Cosine(R1ik0, Rc)

    powers = 2. ** (1.-zetas) #n4
    term1 = 1. + np.einsum('i,j->ij', lamBdas, cos_ijk) # n3*m1
    zetas1 = zetas.repeat(n3*jk).reshape([n4, n3, jk])  # n4*n3*m1
    term2 = np.power(term1, zetas1-1) # n4*n3*m1
    g41 = np.exp(np.einsum('i,jk->ijk', -etas, (R2ij+R2jk+R2ik))) # n2*n1*m1
    g41 = np.einsum('ijk,lmk->ijlmk', term2, g41) # n4*n3*n2**n1*m1
    g41 = np.einsum('i,ijklm->ijklm', powers, g41) # n4*n3*n2*n1*m1
    
    lamBda_zeta = np.einsum('i,j->ij', zetas, lamBdas) # n4*n3
    (dRij_dRm, dRik_dRm, dRjk_dRm) = dRijk_dRm(rij, rik, rjk, ijk_list, N) # m1*3*N
    Rijk_dRm = np.einsum('i,ijk->ijk', R1ij0, dRij_dRm) + \
               np.einsum('i,ijk->ijk', R1ik0, dRik_dRm) + \
               np.einsum('i,ijk->ijk', R1jk0, dRjk_dRm) 
    dcos = dcosijk_dRm(rij, rik, ijk_list, dRij_dRm, dRik_dRm)
    dcos = np.einsum('ij,klm->ijklm', lamBda_zeta, dcos) # n4*n3*3*m1*N
    dcos = np.broadcast_to(dcos, (n2,)+(n4,n3,jk,3,N))
    dcos = np.transpose(dcos, (1,2,0,3,4,5))
    cost = np.einsum('i,jk->jik', 2 * etas, term1) #n3*n2*m1
    cost = np.einsum('ijk,klm->ijklm', cost, Rijk_dRm) # n3*n2*m1*3*N
    cost = np.broadcast_to(cost, (n4,)+(n3,n2,jk,3,N))
    g42 = np.einsum('l,ijklmn->ijklmn', fcij*fcik*fcjk, dcos-cost) # n4*n3*n2*m*3*N
    
    g43 = np.einsum('i,ijk->ijk', dfcij*fcik*fcjk, dRij_dRm) + \
          np.einsum('i,ijk->ijk', fcij*dfcik*fcjk, dRik_dRm) + \
          np.einsum('i,ijk->ijk', fcij*fcik*dfcjk, dRjk_dRm)
    g43 = np.einsum('ij,jkl->ijkl', term1, g43) # n3*m1*3*N
    g43 = np.broadcast_to(g43, (n4, n2,)+(n3,jk,3,N))
    g43 = np.transpose(g43, (0,2,1,3,4,5))

    # [n4, n3, n2, n1, m] * [n4, n3, n2, m, 3, N] -> [n4, n3, n2, n1, m, 3, N] -> [S, m, 3, N] 
    G4ip0 = np.einsum('ijklm, ijkmno->ijklmno', g41, g42+g43,\
                      optimize='greedy').reshape([n1*n2*n3*n4, jk, 3, N])
    # [S, m, 3, N] * [m, 3] -> [S, m, 3, N, 3] 
    # partition the dxdr to each i, j, k
    rG4ip0 = np.zeros([n1*n2*n3*n4, len(ijk_list), 3, N, 3])
    for mm, ijk in enumerate(ijk_list):
        j,k = ijk[1], ijk[2]
        tmp = G4ip0[:,mm,:,:] #S,N,3 -> S,3 * 3 -> S*3*3 
        rG4ip0[:,mm,:,i,:] += np.einsum('ij,k->ijk', tmp[:,:,i], Ri)
        rG4ip0[:,mm,:,j,:] += np.einsum('ij,k->ijk', tmp[:,:,j], rij[mm]+Ri)
        rG4ip0[:,mm,:,k,:] += np.einsum('ij,k->ijk', tmp[:,:,k], rik[mm]+Ri)

    # Decompose G4 Prime by species
    G4Prime = np.zeros([N, n1*n2*n3*n4*l, 3])
    rG4Prime = np.zeros([N, n1*n2*n3*n4*l, 3, 3])

    jk_ids = atomic_numbers[IDs[jks]] 
    for id, jk_type in enumerate(type_set):
        ids = select_rows(jk_ids, jk_type)
        G4Prime[:, id::l, :] += np.einsum('ijkl->lik', G4ip0[:, ids, :, :])
        rG4Prime[:, id::l, :, :] += np.einsum('ijklm->likm', rG4ip0[:, ids, :, :, :])

    return G4Prime, rG4Prime


def calculate_G5Prime(Rij, Ri, i, IDs, jks, atomic_numbers, type_set, Rc, 
                      parameters):
    """Calculate G5 symmetry function for a given atom i.
    
    G5 function also describes the angular feature of atoms in a crystal 
    structure given a cutoff radius.
    
    Parameters
    ----------
    Rij: array [j, 3]
        The vector distances of atom i to neighbors js.
    i: int
        The i-th atom center.    
    IDs: int array [j]
        The indices of neighbors centering about atom i.
    jks: int array [j*k, 2]
        The list of [j,k] pairs
    atomic_numbers: int array [j]
        The elemental indices for atoms within the unitcell.
        e.g. [11, 11, 11, 11, 17, 17, 17, 17]
    Rc: float
        The cutoff radius.
    parameters: dict
        zetas: float array (n1)
        lamBdas: float array (n2)
        etas: float array (n3)
        
    Returns
    -------
    G5Prime: array [m, d, 3]
        The derivative of G5 symmetry value at i-th atom. m is the index of the
        atom that force is acting on.
    """
    Rs = parameters['Rs']    
    etas = parameters['eta']
    zetas = parameters['zeta']
    lamBdas = parameters['lambda']
    n1, n2, n3, n4, l = len(Rs), len(etas), len(lamBdas), len(zetas), len(type_set)
    N, jk = len(atomic_numbers), len(jks)
    
    ijk_list = i * np.ones([jk, 3], dtype=int)
    ijk_list[:,1] = IDs[jks[:,0]]
    ijk_list[:,2] = IDs[jks[:,1]]

    rij = Rij[jks[:,0]] # [m1, 3]
    rik = Rij[jks[:,1]] # [m1, 3]
    rjk = rik - rij     # [m1, 3]

    R2ij0 = np.sum(rij**2., axis=1) 
    R2ik0 = np.sum(rik**2., axis=1) 
    #R2jk0 = np.sum(rjk**2., axis=1) 
    R1ij0 = np.sqrt(R2ij0) # m1
    R1ik0 = np.sqrt(R2ik0) # m1
    #R1jk0 = np.sqrt(R2jk0) # m1
    R2ij = R2ij0 - Rs[:, np.newaxis]**2  # n1*m1
    R2ik = R2ik0 - Rs[:, np.newaxis]**2  # n1*m1
    #R2jk = R2jk0 - Rs[:, np.newaxis]**2  # n1*m1
    R1ij = R1ij0 - Rs[:, np.newaxis] # n1*m1
    R1ik = R1ik0 - Rs[:, np.newaxis] # n1*m1
    #R1jk = R1jk0 - Rs[:, np.newaxis] # n1*m1
 
    cos_ijk = np.sum(rij * rik, axis=1) / R1ij0/ R1ik0 # m1 array

    dfcij = CosinePrime(R1ij0, Rc)
    #dfcjk = CosinePrime(R1jk0, Rc)
    dfcik = CosinePrime(R1ik0, Rc)
    fcij = Cosine(R1ij0, Rc)
    #fcjk = Cosine(R1jk0, Rc)
    fcik = Cosine(R1ik0, Rc)

    powers = 2. ** (1.-zetas) #n4
    term1 = 1. + np.einsum('i,j->ij', lamBdas, cos_ijk) # n3*m1
    zetas1 = zetas.repeat(n3*jk).reshape([n4, n3, jk])  # n4*n3*m1
    term2 = np.power(term1, zetas1-1) # n4*n3*m1
    g51 = np.exp(np.einsum('i,jk->ijk', -etas, (R2ij+R2ik))) # n2*n1*m1
    g51 = np.einsum('ijk,lmk->ijlmk', term2, g51) # n4*n3*n2**n1*m1
    g51 = np.einsum('i,ijklm->ijklm', powers, g51) # n4*n3*n2*n1*m1
    
    lamBda_zeta = np.einsum('i,j->ij', zetas, lamBdas) # n4*n3
    (dRij_dRm, dRik_dRm, dRjk_dRm) = dRijk_dRm(rij, rik, rjk, ijk_list, N) # m1*3*N
    Rijk_dRm = np.einsum('i,ijk->ijk', R1ij0, dRij_dRm) + \
               np.einsum('i,ijk->ijk', R1ik0, dRik_dRm)
               #np.einsum('i,ijk->ijk', R1jk0, dRjk_dRm) 
    dcos = dcosijk_dRm(rij, rik, ijk_list, dRij_dRm, dRik_dRm)
    dcos = np.einsum('ij,klm->ijklm', lamBda_zeta, dcos) # n4*n3*3*m1*N
    dcos = np.broadcast_to(dcos, (n2,)+(n4,n3,jk,3,N))
    dcos = np.transpose(dcos, (1,2,0,3,4,5))
    cost = np.einsum('i,jk->jik', 2 * etas, term1) #n3*n2*m1
    cost = np.einsum('ijk,klm->ijklm', cost, Rijk_dRm) # n3*n2*m1*3*N
    cost = np.broadcast_to(cost, (n4,)+(n3,n2,jk,3,N))
    g52 = np.einsum('l,ijklmn->ijklmn', fcij*fcik, dcos-cost) # n4*n3*n2*m*3*N
    
    g53 = np.einsum('i,ijk->ijk', dfcij*fcik, dRij_dRm) + \
          np.einsum('i,ijk->ijk', fcij*dfcik, dRik_dRm)
    g53 = np.einsum('ij,jkl->ijkl', term1, g53) # n3*m1*3*N
    g53 = np.broadcast_to(g53, (n4, n2,)+(n3,jk,3,N))
    g53 = np.transpose(g53, (0,2,1,3,4,5))

    G5ip0 = np.einsum('ijklm, ijkmno->ijklmno', g51, g52+g53,\
                      optimize='greedy').reshape([n1*n2*n3*n4, jk, 3, N])

    # [S, m, 3, N] * [m, 3] -> [S, m, 3, N, 3] 
    # partition the dxdr to each i, j, k
    rG5ip0 = np.zeros([n1*n2*n3*n4, len(ijk_list), 3, N, 3])
    for mm, ijk in enumerate(ijk_list):
        j,k = ijk[1], ijk[2]
        tmp = G5ip0[:,mm,:,:] #S,N,3 -> S,3 * 3 -> S*3*3 
        rG5ip0[:,mm,:,i,:] += np.einsum('ij,k->ijk', tmp[:,:,i], Ri)
        rG5ip0[:,mm,:,j,:] += np.einsum('ij,k->ijk', tmp[:,:,j], rij[mm]+Ri)
        rG5ip0[:,mm,:,k,:] += np.einsum('ij,k->ijk', tmp[:,:,k], rik[mm]+Ri)

    # Decompose G4 Prime by species
    G5Prime = np.zeros([N, n1*n2*n3*n4*l, 3])
    rG5Prime = np.zeros([N, n1*n2*n3*n4*l, 3, 3])

    jk_ids = atomic_numbers[IDs[jks]] 
    for id, jk_type in enumerate(type_set):
        ids = select_rows(jk_ids, jk_type)
        G5Prime[:, id::l, :] += np.einsum('ijkl->lik', G5ip0[:, ids, :, :])
        rG5Prime[:, id::l, :, :] += np.einsum('ijklm->likm', rG5ip0[:, ids, :, :, :])

    return G5Prime, rG5Prime


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


def dcosijk_dRm(Rij, Rik, ijk_list, dRij_dRm, dRik_dRm):
    """Calculate the derivative of cosine_ijk function w. r. t. atom m.
    m must belong to one of the ijks. Otherwise, the derivative is zero.
    If the input Rij and Rik are (j*k)*3 dimensions, the output will be 
    (j*k)*3*3. The extra dimension comes from looping over {i, j, k}.
    
    Parameters
    ----------
    Rij: array [j*k, 3]
        The vector distances of atom i to neighbors js.
    Rik: array [j*k, 3]
        The vector distances of atom i to neighbors ks, where j != k.
        
    Returns
    -------
    Derivative of cosine dot product w.r.t. the radius of an atom m,
    The atom m has to be in the an array with 3 indices: i, j, and k.
    """
    m = dRik_dRm.shape[-1]
    Dij = np.linalg.norm(Rij, axis=1)
    Dik = np.linalg.norm(Rik, axis=1)
    rDijDik = 1/Dij/Dik # jk array
    Rij_Rik = np.sum(Rij*Rik,axis=1) # jk array
    ij_list = ijk_list[:,[0,1]]
    ik_list = ijk_list[:,[0,2]]
    dcos = np.zeros([len(Rij), 3, m])

    for mm in range(m):
        mm_list = mm * np.ones([len(Rij), 1], dtype=int)
        dRij_dRm_v = dRij_dRm_vector(np.append(ij_list, mm_list, axis=1))
        dRik_dRm_v = dRij_dRm_vector(np.append(ik_list, mm_list, axis=1))
        term10 = np.einsum('i,ij->ij', rDijDik, Rik) # jk*3 arrray
        dcos[:,:,mm] += np.einsum('ij,i->ij', term10, dRij_dRm_v)
        term20 = np.einsum('i,ij->ij', rDijDik, Rij) # jk*3 array
        dcos[:,:,mm] += np.einsum('ij,i->ij', term20, dRik_dRm_v)
        term30 = Rij_Rik*rDijDik/Dij # jk*3
        dcos[:,:,mm] -= np.einsum('i,ij->ij', term30, dRij_dRm[:,:,mm])
        term40 = Rij_Rik*rDijDik/Dik # jk*3
        dcos[:,:,mm] -= np.einsum('i,ij->ij', term40, dRik_dRm[:,:,mm]) # jk*3*m
        
    return dcos


def dRij_dRm_vector(ijm_list):
    """Calculate the derivative of Rij vector w. r. t. atom m.
    
    Parameters
    ----------
    ijm_list: array [i*j, 3]
        List of indices of center atom i, neighbors atom j, and atom m.
    
    Returns
    -------
    list of float
        The derivative of the position vector R_{ij} with respect to atom
        index m in x, y, z directions.
    """
    
    dRij_dRm_vector = np.zeros(len(ijm_list))
    l1 = (ijm_list[:,2]==ijm_list[:,0])
    dRij_dRm_vector[l1] = -1
    l2 = (ijm_list[:,2]==ijm_list[:,1])
    dRij_dRm_vector[l2] = 1

    return dRij_dRm_vector


def dRijk_dRm(Rij, Rik, Rjk, ijk_list, m):
    """Calculate the derivative of R_{ab} norm with respect to atom m.
    
    Parameters
    ----------
    Rij: array [j*k, 3]
        The vector distances of atom i to neighbors js.
    Rik: array [j*k, 3]
        The vector distances of atom i to neighbors ks.
    Rjk: array [j*k, 3]
        The vector distances of atom j to atom k, where j != k.
    ijk_list: array
        The combinations of atom i, neighbor j, and neighbor k indices.
    m: int
        the total number of atom in the unit cell
        
    Returns
    -------
    dR{ab}_dRm: array [j*k, 3, m]
        The derivative of R_{ab} norm with respect to atom m.
    """
    dRij_dRm = np.zeros([len(Rij), 3, m])
    dRik_dRm = np.zeros([len(Rij), 3, m])
    dRjk_dRm = np.zeros([len(Rij), 3, m])
    ij_list = ijk_list[:,[0,1]]
    ik_list = ijk_list[:,[0,2]]
    jk_list = ijk_list[:,[1,2]]
    
    for mm in range(m):
        mm_list = mm * np.ones([len(Rij), 1], dtype=int)
        dRij_dRm[:,:,mm] = dRij_dRm_norm(Rij, np.append(ij_list, mm_list, 
                                         axis=1))
        dRik_dRm[:,:,mm] = dRij_dRm_norm(Rik, np.append(ik_list, mm_list, 
                                         axis=1))
        dRjk_dRm[:,:,mm] = dRij_dRm_norm(Rjk, np.append(jk_list, mm_list,
                                         axis=1))
        
    return (dRij_dRm, dRik_dRm, dRjk_dRm)


############################# Cutoff Functionals ##############################


def Cosine(Rij, Rc):
    # Rij is the norm 
    ids = (Rij > Rc)
    result = 0.5 * (np.cos(np.pi * Rij / Rc) + 1.)
    result[ids] = 0
    
    return result


def CosinePrime(Rij, Rc):
    # Rij is the norm
    ids = (Rij > Rc)
    result = -0.5 * np.pi / Rc * np.sin(np.pi * Rij / Rc)
    result[ids] = 0
    
    return result


########################### Auxiliary Functions ###############################
    

def to_array(input):
    if isinstance(input, (list, np.ndarray)):
        res = np.asarray(input)
    else:
        res = np.asarray([input])
    return res


def create_type_set(number_set, order=1):
    types = list(set(number_set))
    return np.array(list(combinations_with_replacement(types, order)))


def select_rows(data, row_pattern):
    if len(row_pattern) == 1:
        ids = (data==row_pattern)
    elif len(row_pattern) == 2:
        a, b = row_pattern
        ids = []
        for id, d in enumerate(data):
            if a==b:
                if d[0]==a and d[1]==a:
                    ids.append(id)
            else:
                if (d[0] == a and d[1]==b) or (d[0] == b and d[1]==a):
                    ids.append(id)
    return ids 


################################### Test ######################################
    

if __name__ == '__main__':
    from ase.build import bulk
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    from ase.spacegroup import crystal
    # Set up symmetry parameters
    Rc = 5.5
    symmetry = {'G2': {'eta': [0.036, 0.071,], 'Rs': [0]},
                'G4': {'Rs': [0], 'lambda': [1, -1], 'zeta': [1,], 'eta': [0.036, 0.071]},
                'G5': {'Rs': [0], 'lambda': [1, -1], 'zeta': [1,], 'eta': [0.036, 0.071]},
               }
    
    for a in [4.2, 5.0]:
        si = crystal('Si', [(0,0,0)], spacegroup=227, cellpar=[a, a, a, 90, 90, 90])
        cell = si.get_cell()
        cell[0,1] += 0.2
        #cell[0,2] += 0.3
        si.set_cell(cell)
        print(si.get_cell())

        bp = BehlerParrinello(symmetry, Rc=Rc, derivative=True, stress=True)
        des = bp.calculate(si, system=[14])
        
        print("G:", des['x'][0])
        #print("GPrime")
        #print(des['dxdr'][0, 0, :, :])
        print("rGPrime", des['rdxdr'].shape)
        #print(np.einsum('ijklm->jklm', des['rdxdr'])[:,:1,:,:])
        print(np.einsum('ijklm->klm', des['rdxdr']))
