# Numpy Version

import numpy as np
from ase.neighborlist import NeighborList
from itertools import combinations, combinations_with_replacement

class wACSF:
    """A class for calculating weighted atom-centered symmetry functions (wACSFs).
    
    The forms of the functions are consistent with the functions presented in:
        Gastegger, M., et. al. (2018). The Journal of chemical physics, 148(24), 241709.

    Note that this code does not implement Rs in the angular wACSF.
        
    Parameters
    ----------
    symmetry_parameters: dict
        The user-defined parameters for the symmetry functions calculations.
        i.e. {'G2': {'eta': [0.1, 0.2,], 'Rs': [0.0, 0.1]}}
    Rc: float
        The symmetry functions will be calculated within this radius.
    derivative: bool
        If True, calculate the derivatives.
    stress: bool
        If True, calculate the virial stress contribution.
    """
    def __init__(self, symmetry_parameters, Rc=6.5, derivative=True, stress=False):
        self._type = 'wACSF'
        
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
        
       
    def calculate(self, crystal):
        """The symmetry functions are obtained through this `calculate` method.
        
        Parameters
        ----------
        crystal: object
            ASE Structure object.
                    
        Returns
        -------
        all_G: dict
            The user-defined symmetry functions that represent the crystal.
            
            Currently, there are 3 types of symmetry functions implemented. 
            Here are the order of the descriptors are printed out based on
            their symmetry parameters:
                - G2: ["Rs", "eta"]
                - G4: ["eta", "lambda", "zeta"]
                - G5: ["eta", "lambda", "zeta"]
        """
        self.crystal, self.total_atoms = crystal, len(crystal)
        atomic_numbers = np.array(crystal.get_atomic_numbers())
        vol = crystal.get_volume()

        # Initialize dict for the symmetry functions
        self.all_G = {'x':[], 'elements': []}
        if self.derivative:
            self.all_G['dxdr'] = []
        if self.stress:
            self.all_G['rdxdr'] = []

        # Obtain neighbors info.
        rc = [self.Rc/2]*len(self.crystal)
        neighbors = NeighborList(rc, self_interaction=False, bothways=True, skin=0.0)
        neighbors.update(crystal)
        
        for i in range(len(crystal)):
            element = crystal.get_chemical_symbols()[i]
            indices, offsets = neighbors.get_neighbors(i)
            Z = [] # atomic numbers of neighbors
            
            assert len(indices)>0, \
            f"There's no neighbor for this structure at Rc = {self.Rc} A."

            Ri = crystal.get_positions()[i]
            total_neighbors = len(indices) # total number of neighbors of atom i
            
            Rj = np.zeros([total_neighbors, 3])
            IDs = np.zeros(total_neighbors, dtype=int)
            jks = np.array(list(combinations(range(total_neighbors), 2)))
            
            count = 0
            for j, offset in zip(indices, offsets):
                Rj[count, :] = crystal.positions[j] + np.dot(offset, crystal.get_cell())
                IDs[count] = j
                Z.append(crystal[j].number)
                count += 1
            Z = np.array(Z)
                
            Rij = Rj - Ri
            Dij = np.sqrt(np.sum(Rij**2, axis=1))
            
            Gi = []
            GiPrime = None

            if self.G2_parameters is not None:
                G2i = calculate_G2(Dij, Z, self.Rc, self.G2_parameters)
                Gi = np.append(Gi, G2i)
                
                if self.derivative:
                    G2iP, rG2iP = calculate_G2Prime(Rij, Dij, Z, self.total_atoms,
                                                    i, Ri, IDs, self.Rc, self.G2_parameters)
                    if GiPrime is None:
                        GiPrime = G2iP
                        if self.stress:
                            rGiPrime = rG2iP
                    else:
                        GiPrime = np.append(GiPrime, G2iP, axis=1)
                        if self.stress:
                            rGiPrime = np.append(rGiPrime, rG2iP, axis=1)
            
            if self.G4_parameters is not None:
                G4i = calculate_G4(Rij, IDs, Z, jks, self.Rc, self.G4_parameters)
                Gi = np.append(Gi, G4i)
                
                if self.derivative:
                    G4iP, rG4iP = calculate_G4Prime(Rij, Ri, i, IDs, self.total_atoms, 
                                                    jks, Z, self.Rc, self.G4_parameters)
                    
                    if GiPrime is None:
                        GiPrime = G4iP
                        if self.stress:
                            rGiPrime = rG4iP
                    else:
                        GiPrime = np.append(GiPrime, G4iP, axis=1)
                        if self.stress:
                            rGiPrime = np.append(rGiPrime, rG4iP, axis=1)

            if self.G5_parameters is not None:
                G5i = calculate_G5(Rij, IDs, Z, jks, self.Rc, self.G4_parameters)
                Gi = np.append(Gi, G5i)

                if self.derivative:
                    G5iP, rG5iP = calculate_G5Prime(Rij, Ri, i, IDs, self.total_atoms, 
                                                    jks, Z, self.Rc, self.G4_parameters)
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


def calculate_G2(Dij, Z, Rc, parameters):
    """Calculate radial weighted symmetry function for a center atom i.
    
    Parameters
    ----------
    Dij: float array [j]
        Array of neighbors (j) distances with respect to center atom i.
    Z: float array [j]
        Array of atomic numbers of neighbors.
    Rc: float
        The cutoff radius.
    parameters: dict
        Rs: float array (d1)
            The shift from the center of the G2 symmetry function.
        etas: float array (d2)
             eta parameters of G2 symmetry function.
    
    Returns
    -------
    G2: float array [d = d1*d2]
        G2 symmetry value.
    """
    Rs = parameters['Rs']       # [d1]
    etas = parameters['eta']    # [d2]
    d1, d2, j = len(Rs), len(etas), len(Dij) 
    
    Z_cutoff = Cosine(Dij, Rc) * Z # [j]
    d20 = (Dij - Rs[:, np.newaxis]) ** 2  # [d1, j]
    term = np.exp(np.einsum('i,jk->ijk', -etas, d20)) # [d2] * [d1, j] -> [d2, d1, j]
    results = np.einsum('ijk,k->ij', term, Z_cutoff) # [d2, d1, j] * [j] -> [d2, d1]
    G2 = results.reshape([d2*d1]) # [d2*d1]
    
    return G2


def calculate_G2Prime(Rij, Dij, Z, m, i, Ri, IDs, Rc, parameters):
    """Calculate the derivative of G2 symmetry function for atom i.
    
    Parameters
    ----------
    Rij: float array [j, 3]
        The vector distances of atom i to neighbors js.
    Dij: float array [j]
        Array of neighbors (j) distances with respect to center atom i.
    Z: float array [j]
        Array of atomic numbers of neighbors.
    m: int
        The total atoms in the crystal unit cell.
    i: int
        The i-th atom center.
    IDs: int array [j]
        The indices of neighbors centering about atom i.
    Ri: float array [3]
        The x, y, and z positions of atom i.
    Rc: float
        The cutoff radius.
    parameters: dict
        Rs: float array (d1)
            The shift from the center of the G2 symmetry function.
        etas: float array (d2)
             eta parameters of G2 symmetry function.
    
    Returns
    -------
    G2Prime: float array [m, d2*d1, 3]
        The derivative of G2 symmetry value at i-th atom.
    """
    Rs = parameters['Rs']       # [d1]
    etas = parameters['eta']    # [d2]
    d1, d2, j = len(Rs), len(etas), len(Rij)
    
    ij_list = i * np.ones([len(IDs), 2], dtype=int) 
    ij_list[:, 1] = IDs # [j]

    d0 = Dij - Rs[:, np.newaxis]    # [d1, j]
    d20 = d0 ** 2                   # [d1, j]
    term1 = np.exp(np.einsum('i,jk->ijk', -etas, d20))   # [d2, d1, j]
    
    term21 = CosinePrime(Dij, Rc) * Z   # [j]
    Z_cutoff = Cosine(Dij, Rc) * Z      # [j]
    _term22 = np.einsum('ij,j->ij', d0, Z_cutoff) # [d1, j]
    term22 = 2 * np.einsum('i,jk->ijk', etas, _term22) # [d2, d1, j]
    term2 = term21 - term22     # [d2, d1, j]

    term_1_and_2 = (term1 * term2).reshape([d2*d1, j]) # [d2*d1, j] 

    dRij_dRm = np.zeros([j, 3, m])
    for mm in range(m):
        mm_list = mm * np.ones([j, 1], dtype=int)
        dRij_dRm[:, :, mm] = dRij_dRm_norm(Rij, np.hstack((ij_list, mm_list)))
    
    G2ip0 = np.einsum('ij,jkl->ijkl', term_1_and_2, dRij_dRm) # [d2*d1, j, 3, m]

    rG2ip0 = np.zeros([d1*d2, j, 3, m, 3])
    for mm, ij in enumerate(ij_list):
        _j = ij[1]
        tmp = G2ip0[:, mm, :, :] 
        rG2ip0[:,mm,:,i,:] += np.einsum('ij,k->ijk', tmp[:,:,i], Ri)
        rG2ip0[:,mm,:,_j,:] += np.einsum('ij,k->ijk', tmp[:,:,_j], Rij[mm]+Ri)

    G2Prime = np.einsum('ijkl->lik', G2ip0)     # [m, d2*d1, 3]
    rG2Prime = np.einsum('ijklm->likm', rG2ip0) # [m, d2*d1, 3, 3]

    return G2Prime, rG2Prime


def calculate_G4(Rij, IDs, Z, jks, Rc, parameters):
    """Calculate G4 symmetry function for a given atom i.
    
    Parameters
    ----------
    Rij: array [j, 3]
        The vector distances of atom i to neighbors js.
    IDs: int array [j]
        The indices of neighbors centering about atom i.
    Z: float array [j]
        Array of atomic numbers of neighbors.
    jks: int array [j*(j-1)/2, 2]
        The list of jk pairs.
    Rc: float
        The cutoff radius.
    parameters: dict
        Rs: float array (d1)
        etas: float array (d2)
        lambdas: float array (d3)
        zetas: float array (d4)
        
    Returns
    -------
    G4: array [d = d1*d2*d3*d4]
        G4 symmetry value with d members.
    """
    Rs, etas = parameters['Rs'], parameters['eta']                  # [d1], [d2]
    zetas, lambdas = parameters['zeta'], parameters['lambda']       # [d3], [d4]
    d1, d2, d3, d4 = len(Rs), len(etas), len(lambdas), len(zetas)
    jk = len(jks)  # [jk]; j*k pairs

    Zjk = Z[jks[:, 0]] * Z[jks[:, 1]] # [jk]
    
    rij = Rij[jks[:,0]] # [jk, 3]
    rik = Rij[jks[:,1]] # [jk, 3]
    rjk = rik - rij     # [jk, 3]
    R2ij0 = np.sum(rij**2., axis=1) 
    R2ik0 = np.sum(rik**2., axis=1) 
    R2jk0 = np.sum(rjk**2., axis=1) 
    R1ij0 = np.sqrt(R2ij0) # [jk]
    R1ik0 = np.sqrt(R2ik0) # [jk]
    R1jk0 = np.sqrt(R2jk0) # [jk]
    R2ij = R2ij0 - Rs[:, np.newaxis]**2  # [d1, jk]
    R2ik = R2ik0 - Rs[:, np.newaxis]**2  # [d1, jk]
    R2jk = R2jk0 - Rs[:, np.newaxis]**2  # [d1, jk]
    R1ij = R1ij0 - Rs[:, np.newaxis] # [d1, jk]
    R1ik = R1ik0 - Rs[:, np.newaxis] # [d1, jk]
    R1jk = R1jk0 - Rs[:, np.newaxis] # [d1, jk]
 
    powers = 2. ** (1.-zetas) # d4
    cos_ijk = np.sum(rij*rik, axis=1)/R1ij0/R1ik0 # jk
    term1 = 1. + np.einsum('i,j->ij', lambdas, cos_ijk) # [d3, jk]

    zetas1 = zetas.repeat(d3*jk).reshape([d4, d3, jk])  # [d4, d3, jk] 
    term2 = np.power(term1, zetas1) # [d4, d3, jk] 
    term3 = np.exp(np.einsum('i,jk->ijk', -etas, (R2ij+R2jk+R2ik))) # [d2, d1, jk] 
    term4 = Cosine(R1ij0, Rc) * Cosine(R1ik0, Rc) * Cosine(R1jk0, Rc) * Zjk # [jk]
    term5 = np.einsum('ijk,lmk->ijlmk', term2, term3) # [d4, d3, d2, d1, jk]
    term6 = np.einsum('ijkml,l->ijkml', term5, term4) # [d4, d3, d2, d1, jk]
    results = np.einsum('i,ijkml->ijkml', powers, term6) # [d4, d3, d2, d1, jk]
    results = results.reshape([d1*d2*d3*d4, jk]) # [d4*d3*d2*d1, jk]

    G4 = np.einsum('ij->i', results) # [d]
    
    return G4


def calculate_G5(Rij, IDs, Z, jks, Rc, parameters):
    """Calculate G5 symmetry function for a given atom i.
    
    Parameters
    ----------
    Rij: array [j, 3]
        The vector distances of atom i to neighbors js.
    IDs: int array [j]
        The indices of neighbors centering about atom i.
    Z: float array [j]
        Array of atomic numbers of neighbors.
    jks: int array [j*(j-1)/2, 2]
        The list of jk pairs.
    Rc: float
        The cutoff radius.
    parameters: dict
        Rs: float array (d1)
        etas: float array (d2)
        lambdas: float array (d3)
        zetas: float array (d4)
        
    Returns
    -------
    G5: array [d = d1*d2*d3*d4]
        G5 symmetry value with d members.
    """
    Rs, etas = parameters['Rs'], parameters['eta']                  # [d1], [d2]
    zetas, lambdas = parameters['zeta'], parameters['lambda']       # [d3], [d4]
    d1, d2, d3, d4 = len(Rs), len(etas), len(lambdas), len(zetas)
    jk = len(jks)  # [jk]; j*k pairs

    Zjk = Z[jks[:, 0]] * Z[jks[:, 1]] # [jk]
    
    rij = Rij[jks[:,0]] # [jk, 3]
    rik = Rij[jks[:,1]] # [jk, 3]
    R2ij0 = np.sum(rij**2., axis=1) 
    R2ik0 = np.sum(rik**2., axis=1) 
    
    R1ij0 = np.sqrt(R2ij0) # [jk]
    R1ik0 = np.sqrt(R2ik0) # [jk]

    R2ij = R2ij0 - Rs[:, np.newaxis]**2  # [d1, jk]
    R2ik = R2ik0 - Rs[:, np.newaxis]**2  # [d1, jk]
    
    R1ij = R1ij0 - Rs[:, np.newaxis] # [d1, jk]
    R1ik = R1ik0 - Rs[:, np.newaxis] # [d1, jk]
 
    powers = 2. ** (1.-zetas) # [d4]
    cos_ijk = np.sum(rij*rik, axis=1)/R1ij0/R1ik0 # [jk]
    term1 = 1. + np.einsum('i,j->ij', lambdas, cos_ijk) # [d3, jk]

    zetas1 = zetas.repeat(d3*jk).reshape([d4, d3, jk])  # [d4, d3, jk]
    term2 = np.power(term1, zetas1) # [d4, d3, jk]
    term3 = np.exp(np.einsum('i,jk->ijk', -etas, (R2ij+R2ik))) # [d2, d1, jk]
    term4 = Cosine(R1ij0, Rc) * Cosine(R1ik0, Rc) * Zjk # [jk]
    term5 = np.einsum('ijk,lmk->ijlmk', term2, term3) # [d4, d3, d2, d1, jk]
    term6 = np.einsum('ijkml,l->ijkml', term5, term4) # [d4, d3, d2, d1, jk]
    results = np.einsum('i,ijkml->ijkml', powers, term6) # [d4, d3, d2, d1, jk]
    results = results.reshape([d1*d2*d3*d4, jk]) # [d, jk]

    G5 = np.einsum('ij->i', results) # [d] 
    
    return G5


def calculate_G4Prime(Rij, Ri, i, IDs, m, jks, Z, Rc, parameters):
    """Calculate G4 symmetry function for a given atom i.
    
    Parameters
    ----------
    Rij: array [j, 3]
        The vector distances of atom i to neighbors js.
    Ri: float array [3]
        The x, y, and z positions of atom i.
    i: int
        The i-th atom center.    
    IDs: int array [j]
        The indices of neighbors centering about atom i.
    m: int
        The total atoms in the crystal unit cell.
    jks: int array [j*k, 2]
        The list of [j,k] pairs.
    Z: float array [j]
        Array of atomic numbers of neighbors.
    Rc: float
        The cutoff radius.
    parameters: dict
        Rs: float array (d1)
        etas: float array (d2)
        lambdas: float array (d3)
        zetas: float array (d4)
        
    Returns
    -------
        G4Prime: array [m, d, 3]
        The derivative of G4 symmetry value at i-th atom. m is the index of the
        atom that force is acting on.
    """
    Rs, etas = parameters['Rs'], parameters['eta']                  # [d1], [d2]
    zetas, lambdas = parameters['zeta'], parameters['lambda']       # [d3], [d4]
    d1, d2, d3, d4 = len(Rs), len(etas), len(lambdas), len(zetas)
    jk = len(jks) # [jk]; j*k pairs
    
    ijk_list = i * np.ones([jk, 3], dtype=int)
    ijk_list[:,1] = IDs[jks[:,0]]
    ijk_list[:,2] = IDs[jks[:,1]]

    Zjk = Z[jks[:, 0]] * Z[jks[:, 1]] # [jk]

    rij = Rij[jks[:,0]] # [jk, 3]
    rik = Rij[jks[:,1]] # [jk, 3]
    rjk = rik - rij     # [jk, 3]

    R2ij0 = np.sum(rij**2., axis=1) 
    R2ik0 = np.sum(rik**2., axis=1) 
    R2jk0 = np.sum(rjk**2., axis=1) 
    R1ij0 = np.sqrt(R2ij0) # jk
    R1ik0 = np.sqrt(R2ik0) # jk
    R1jk0 = np.sqrt(R2jk0) # jk
    R2ij = R2ij0 - Rs[:, np.newaxis]**2  # [d1, jk]
    R2ik = R2ik0 - Rs[:, np.newaxis]**2  # [d1, jk]
    R2jk = R2jk0 - Rs[:, np.newaxis]**2  # [d1, jk]
    R1ij = R1ij0 - Rs[:, np.newaxis] # [d1, jk]
    R1ik = R1ik0 - Rs[:, np.newaxis] # [d1, jk]
    R1jk = R1jk0 - Rs[:, np.newaxis] # [d1, jk]
 
    cos_ijk = np.sum(rij * rik, axis=1) / R1ij0/ R1ik0 # [jk]

    dfcij = CosinePrime(R1ij0, Rc) 
    dfcjk = CosinePrime(R1jk0, Rc)
    dfcik = CosinePrime(R1ik0, Rc)
    fcij = Cosine(R1ij0, Rc)
    fcjk = Cosine(R1jk0, Rc)
    fcik = Cosine(R1ik0, Rc)

    powers = 2. ** (1.-zetas) # d4
    term1 = 1. + np.einsum('i,j->ij', lambdas, cos_ijk) # [d3, jk] 
    zetas1 = zetas.repeat(d3*jk).reshape([d4, d3, jk])  # [d4, d3, jk]
    term2 = np.power(term1, zetas1-1) # [d4, d3, jk]
    g41 = np.exp(np.einsum('i,jk->ijk', -etas, (R2ij+R2jk+R2ik))) # [d2, d1, jk]
    g41 = np.einsum('ijk,lmk->ijlmk', term2, g41) # [d4, d3, d2, d1, jk]
    g41 = np.einsum('i,ijklm->ijklm', powers, g41) # [d4, d3, d2, d1, jk]
    
    lambda_zeta = np.einsum('i,j->ij', zetas, lambdas) # [d4, d3]
    (dRij_dRm, dRik_dRm, dRjk_dRm) = dRijk_dRm(rij, rik, rjk, ijk_list, m) # [jk, 3, m]
    Rijk_dRm = np.einsum('i,ijk->ijk', R1ij0, dRij_dRm) + \
               np.einsum('i,ijk->ijk', R1ik0, dRik_dRm) + \
               np.einsum('i,ijk->ijk', R1jk0, dRjk_dRm) 
    dcos = dcosijk_dRm(rij, rik, ijk_list, dRij_dRm, dRik_dRm)
    dcos = np.einsum('ij,klm->ijklm', lambda_zeta, dcos) # [d4, d3, 3, jk, m]
    dcos = np.broadcast_to(dcos, (d2,)+(d4,d3,jk,3,m))
    dcos = np.transpose(dcos, (1,2,0,3,4,5))
    cost = np.einsum('i,jk->jik', 2 * etas, term1) # [d3, d2, jk]
    cost = np.einsum('ijk,klm->ijklm', cost, Rijk_dRm) # [d3, d2, jk, 3, m]
    cost = np.broadcast_to(cost, (d4,)+(d3,d2,jk,3,m))
    g42 = np.einsum('l,ijklmn->ijklmn', fcij*fcik*fcjk*Zjk, dcos-cost) # [d4, d3, d2, jk, 3, m]
    
    g43 = np.einsum('i,ijk->ijk', dfcij*fcik*fcjk*Zjk, dRij_dRm) + \
          np.einsum('i,ijk->ijk', fcij*dfcik*fcjk*Zjk, dRik_dRm) + \
          np.einsum('i,ijk->ijk', fcij*fcik*dfcjk*Zjk, dRjk_dRm)
    g43 = np.einsum('ij,jkl->ijkl', term1, g43) # [d3, jk, 3, m]
    g43 = np.broadcast_to(g43, (d4, d2,)+(d3,jk,3,m))
    g43 = np.transpose(g43, (0,2,1,3,4,5))

    # [d4, d3, d2, d1, jk] * [d4, d3, d2, jk, 3, m] -> [d4, d3, d2, d1, jk, 3, m] -> [S, jk, 3, m] 
    G4ip0 = np.einsum('ijklm, ijkmno->ijklmno', g41, g42+g43,\
                      optimize='greedy').reshape([d1*d2*d3*d4, jk, 3, m])
    # [S, m, 3, N] * [m, 3] -> [S, m, 3, N, 3] 
    # partition the dxdr to each i, j, k
    rG4ip0 = np.zeros([d1*d2*d3*d4, len(ijk_list), 3, m, 3])
    for mm, ijk in enumerate(ijk_list):
        j,k = ijk[1], ijk[2]
        tmp = G4ip0[:,mm,:,:] #S,m,3 -> S,3 * 3 -> S*3*3 
        rG4ip0[:,mm,:,i,:] += np.einsum('ij,k->ijk', tmp[:,:,i], Ri)
        rG4ip0[:,mm,:,j,:] += np.einsum('ij,k->ijk', tmp[:,:,j], rij[mm]+Ri)
        rG4ip0[:,mm,:,k,:] += np.einsum('ij,k->ijk', tmp[:,:,k], rik[mm]+Ri)

    G4Prime = np.einsum('ijkl->lik', G4ip0)
    rG4Prime = np.einsum('ijklm->likm', rG4ip0)
    
    return G4Prime, rG4Prime


def calculate_G5Prime(Rij, Ri, i, IDs, m, jks, Z, Rc, parameters):
    """Calculate G5 symmetry function for a given atom i.
    
    Parameters
    ----------
    Rij: array [j, 3]
        The vector distances of atom i to neighbors js.
    Ri: float array [3]
        The x, y, and z positions of atom i.
    i: int
        The i-th atom center.    
    IDs: int array [j]
        The indices of neighbors centering about atom i.
    m: int
        The total atoms in the crystal unit cell.
    jks: int array [j*k, 2]
        The list of [j,k] pairs.
    Z: float array [j]
        Array of atomic numbers of neighbors.
    Rc: float
        The cutoff radius.
    parameters: dict
        Rs: float array (d1)
        etas: float array (d2)
        lambdas: float array (d3)
        zetas: float array (d4)
        
    Returns
    -------
    G5Prime: array [m, d, 3]
        The derivative of G5 symmetry value at i-th atom. m is the index of the
        atom that force is acting on.
    """
    Rs, etas = parameters['Rs'], parameters['eta']                  # [d1], [d2]
    zetas, lambdas = parameters['zeta'], parameters['lambda']       # [d3], [d4]
    d1, d2, d3, d4 = len(Rs), len(etas), len(lambdas), len(zetas)
    jk = len(jks) # [jk]; j*k pairs
    
    ijk_list = i * np.ones([jk, 3], dtype=int)
    ijk_list[:,1] = IDs[jks[:,0]]
    ijk_list[:,2] = IDs[jks[:,1]]

    Zjk = Z[jks[:, 0]] * Z[jks[:, 1]] # [jk]

    rij = Rij[jks[:,0]] # [jk, 3]
    rik = Rij[jks[:,1]] # [jk, 3]
    rjk = rik - rij     # [jk, 3]

    R2ij0 = np.sum(rij**2., axis=1) 
    R2ik0 = np.sum(rik**2., axis=1) 
    R1ij0 = np.sqrt(R2ij0) # [jk]
    R1ik0 = np.sqrt(R2ik0) # [jk]
    R2ij = R2ij0 - Rs[:, np.newaxis]**2  # [d1, jk]
    R2ik = R2ik0 - Rs[:, np.newaxis]**2  # [d1, jk]
    R1ij = R1ij0 - Rs[:, np.newaxis] # [d1, jk] 
    R1ik = R1ik0 - Rs[:, np.newaxis] # [d1, jk]
 
    cos_ijk = np.sum(rij * rik, axis=1) / R1ij0/ R1ik0 # [jk]

    dfcij = CosinePrime(R1ij0, Rc)
    dfcik = CosinePrime(R1ik0, Rc)
    fcij = Cosine(R1ij0, Rc)
    fcik = Cosine(R1ik0, Rc)

    powers = 2. ** (1.-zetas) # [d4]
    term1 = 1. + np.einsum('i,j->ij', lambdas, cos_ijk) # [d3, jk]
    zetas1 = zetas.repeat(d3*jk).reshape([d4, d3, jk])  # [d4, d3, jk]
    term2 = np.power(term1, zetas1-1) # [d4, d3, jk] 
    g51 = np.exp(np.einsum('i,jk->ijk', -etas, (R2ij+R2ik))) # [d2, d1, jk]
    g51 = np.einsum('ijk,lmk->ijlmk', term2, g51) # [d4, d3, d2, d1, jk]
    g51 = np.einsum('i,ijklm->ijklm', powers, g51) # [d4, d3, d2, d1, jk]
    
    lambda_zeta = np.einsum('i,j->ij', zetas, lambdas) # [d4, d3]
    (dRij_dRm, dRik_dRm, dRjk_dRm) = dRijk_dRm(rij, rik, rjk, ijk_list, m) # [jk, 3, m]
    Rijk_dRm = np.einsum('i,ijk->ijk', R1ij0, dRij_dRm) + \
               np.einsum('i,ijk->ijk', R1ik0, dRik_dRm)
    dcos = dcosijk_dRm(rij, rik, ijk_list, dRij_dRm, dRik_dRm)
    dcos = np.einsum('ij,klm->ijklm', lambda_zeta, dcos) # [d4, d3, 3, jk, m] 
    dcos = np.broadcast_to(dcos, (d2,)+(d4,d3,jk,3,m))
    dcos = np.transpose(dcos, (1,2,0,3,4,5))
    cost = np.einsum('i,jk->jik', 2 * etas, term1) # [d3, d2, jk] 
    cost = np.einsum('ijk,klm->ijklm', cost, Rijk_dRm) # [d3, d2, jk, 3, m]
    cost = np.broadcast_to(cost, (d4,)+(d3,d2,jk,3,m))
    g52 = np.einsum('l,ijklmn->ijklmn', fcij*fcik*Zjk, dcos-cost) # [d4, d3, d2, jk, 3, m]
    
    g53 = np.einsum('i,ijk->ijk', dfcij*fcik*Zjk, dRij_dRm) + \
          np.einsum('i,ijk->ijk', fcij*dfcik*Zjk, dRik_dRm)
    g53 = np.einsum('ij,jkl->ijkl', term1, g53) # [d3, jk, 3,m]
    g53 = np.broadcast_to(g53, (d4, d2,)+(d3,jk,3,m))
    g53 = np.transpose(g53, (0,2,1,3,4,5))

    G5ip0 = np.einsum('ijklm, ijkmno->ijklmno', g51, g52+g53,\
                      optimize='greedy').reshape([d1*d2*d3*d4, jk, 3, m])

    # [S, m, 3, N] * [m, 3] -> [S, m, 3, N, 3] 
    # partition the dxdr to each i, j, k
    rG5ip0 = np.zeros([d1*d2*d3*d4, len(ijk_list), 3, m, 3])
    for mm, ijk in enumerate(ijk_list):
        j,k = ijk[1], ijk[2]
        tmp = G5ip0[:,mm,:,:] #S,N,3 -> S,3 * 3 -> S*3*3 
        rG5ip0[:,mm,:,i,:] += np.einsum('ij,k->ijk', tmp[:,:,i], Ri)
        rG5ip0[:,mm,:,j,:] += np.einsum('ij,k->ijk', tmp[:,:,j], rij[mm]+Ri)
        rG5ip0[:,mm,:,k,:] += np.einsum('ij,k->ijk', tmp[:,:,k], rik[mm]+Ri)
    
    G5Prime = np.einsum('ijkl->lik', G5ip0)
    rG5Prime = np.einsum('ijklm->likm', rG5ip0)

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
    import time
    from ase.build import bulk
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

    # Set up symmetry parameters
    Rc = 5.5
    symmetry = {'G2': {'eta': [0.036, 0.071,], 'Rs': [0]},
                'G4': {'Rs': [0], 'lambda': [1], 'zeta': [1,], 'eta': [0.036, 0.071]},
                'G5': {'Rs': [0], 'lambda': [1], 'zeta': [1,], 'eta': [0.036, 0.071]},
               }
    
    for a in [5.0]: 
        si = bulk('Si', 'diamond', a=a, cubic=True)
        cell = si.get_cell()
        si.set_cell(cell)
        print(si.get_cell())

        bp = wACSF(symmetry, Rc=Rc, derivative=True, stress=True)
        des = bp.calculate(si)
        print("G:", des['x'][0])
        print("GPrime", des['dxdr'][0,:,0,:])
        print("GPrime", des['dxdr'][0,:,2,:])
        print("GPrime", des['dxdr'][0,:,4,:])
        print(np.einsum('ijklm->klm', des['rdxdr']))
