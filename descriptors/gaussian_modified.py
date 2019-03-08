import numpy as np
from scipy.stats import skew, kurtosis
import itertools
from pymatgen.core.structure import Structure


################################ Gaussian Class ###############################


class gaussian:
    """
    Get the all the desired symmetry functions.

    Parameters
    ----------
    crystal: object
        Pymatgen crystal structure object.
    sys_params: dict
        Dictionary of symmetry parameters.
        i.e. {'G2': {'eta': [0.05, 0.1]}}
    derivative: bool
        to calculate derivative.
    """
    def __init__(self, crystal, symmetry_parameters, derivative=False):
        self.crystal = crystal
        self.symmetry_parameters = symmetry_parameters
#        print(symmetry_parameters)
        self.derivative = derivative

        self.G1_keywords = ['Rc', 'cutoff_f']
        self.G2_keywords = ['eta', 'Rc', 'cutoff_f', 'Rs']
        self.G3_keywords = ['kappa', 'Rc', 'cutoff_f']
        self.G4_keywords = ['eta', 'lamBda', 'zeta', 'Rc', 'cutoff_f']
        self.G5_keywords = ['eta', 'lamBda', 'zeta', 'Rc', 'cutoff_f']

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
        
        
        self.G1 = []
        if self.G1_parameters is not None:
            self._check_sanity(self.G1_parameters, self.G1_keywords)

            G1 = np.asarray(self.calculate('G1', self.G1_parameters, self.derivative))
            self.G1 = self.reshaping(G1)

        self.G2 = []
        if self.G2_parameters is not None:
            self._check_sanity(self.G2_parameters, self.G2_keywords)
            G2, G2D = self.calculate('G2', self.G2_parameters, self.derivative)
            self.G2 = self.reshaping(np.asarray(G2))
            self.G2_derivative = G2D
        
        self.G3 = []
        if self.G3_parameters is not None:
            self._check_sanity(self.G3_parameters, self.G3_keywords)
            G3 = np.asarray(self.calculate('G3', self.G3_parameters, self.derivative))
            self.G3 = self.reshaping(G3)

        self.G4 = []
        if self.G4_parameters is not None:
            self._check_sanity(self.G4_parameters, self.G4_keywords)
            G4, G4D = self.calculate('G4', self.G4_parameters, self.derivative)
            self.G4 = self.reshaping(np.asarray(G4))
            self.G4_derivative = G4D

        self.G5 = []
        if self.G5_parameters is not None:
            self._check_sanity(self.G5_parameters, self.G5_keywords)
            G5, G5D = self.calculate('G5', self.G5_parameters, self.derivative)
            self.G5 = self.reshaping(np.asarray(G5))
            self.G5_derivative = G5D
    

    def calculate(self, G_type, symmetry_parameters, derivative=False):
        G, Gd = [], []
        Rc = [6.5]
        cutoff_f = ['Cosine']
        Rs = [0.]
        
        for key, value in symmetry_parameters.items():
            if key == 'Rc':
                if isinstance(value, (list, np.ndarray)):
                    Rc = value
                else:
                    Rc = [value]
            elif key == 'cutoff_f':
                if isinstance(value, (list, np.ndarray)):
                    cutoff_f = value
                else:
                    cutoff_f = [value]
            elif key == 'Rs':
                if isinstance(value, (list, np.ndarray)):
                    Rs = value
                else:
                    Rs = [value]
            elif key == 'eta':
                if isinstance(value, (list, np.ndarray)):
                    etas = value
                else:
                    etas = [value]
            elif key == 'kappa':
                if isinstance(value, (list, np.ndarray)):
                    kappas = value
                else:
                    kappas = [value]
            elif key == 'lamBda':
                if isinstance(value, (list, np.ndarray)):
                    lamBdas = value
                else:
                    lamBdas = [value]
            elif key == 'zeta':
                if isinstance(value, (list, np.ndarray)):
                    zetas = value
                else:
                    zetas = [value]
                    
        if derivative:
            core_elements = [site.species_string for site in self.crystal.sites]
            neighbors = self.crystal.get_all_neighbors(r=max(Rc),
                                                       include_index=True,
                                                       include_image=True)              

        if G_type == 'G1':
            for rc in Rc:
                for co in cutoff_f:
                    g = calculate_G1(self.crystal, co, rc)
                    G.append(g)

        elif G_type == 'G2':
            for rc in Rc:
                for rs in Rs:
                    for co in cutoff_f:
                        for eta in etas:
                            g = calculate_G2(self.crystal, co, rc, eta, rs)
                            G.append(g)

            if derivative:
                for i, neigh in enumerate(neighbors):
                    for q in range(3):
                        gd = G2_derivative(crystal=self.crystal, i=i, element=core_elements[i], ni=neigh, 
                                           cutoff_f='Cosine', Rc=6.5, 
                                           eta=etas, Rs=0.0, p=i, q=q)
                        Gd.append(gd)
                        for n in neigh:
                            if n[3] == (0.0, 0.0, 0.0):
                                gd = G2_derivative(crystal=self.crystal, i=n[2], 
                                                   element=n[0].species_string, 
                                                   ni=neighbors[n[2]], 
                                                   cutoff_f='Cosine', Rc=6.5, 
                                                   eta=etas, Rs=0.0, p=i, q=q)
                                Gd.append(gd)
                            
                            
        elif G_type == 'G3':
            for rc in Rc:
                for rs in Rs:
                    for co in cutoff_f:
                        for kappa in kappas:
                            g = calculate_G3(self.crystal, co, rc, kappa)
                            G.append(g)
                                                    

        elif G_type == 'G4':
            for rc in Rc:
                for co in cutoff_f:
                    for eta in etas:
                        for zeta in zetas:
                            for lb in lamBdas:
                                g = calculate_G4(self.crystal, co, rc, eta, lb, 
                                                 zeta)
                                G.append(g)
            if derivative:
                for i, neigh in enumerate(neighbors):
                    for q in range(3):
                        gd = G4_derivative(crystal=self.crystal, i=i, element=core_elements[i], ni=neigh, 
                                           cutoff_f='Cosine', Rc=6.5, 
                                           eta=etas, lamBda=lamBdas, zeta=zetas, p=i, q=q)
                        Gd.append(gd)
                        for n in neigh:
                            if n[3] == (0.0, 0.0, 0.0):
                                gd = G4_derivative(crystal=self.crystal, i=n[2], 
                                                   element=n[0].species_string, 
                                                   ni=neighbors[n[2]], 
                                                   cutoff_f='Cosine', Rc=6.5, 
                                                   eta=etas, lamBda=lamBdas, zeta=zetas, p=i, q=q)
                                Gd.append(gd)

        elif G_type == 'G5':
            for rc in Rc:
                for co in cutoff_f:
                    for eta in etas:
                        for zeta in zetas:
                            for lb in lamBdas:
                                g = calculate_G5(self.crystal, co, rc, eta, lb, 
                                                 zeta)
                                G.append(g)
            
            if derivative:
                for i, neigh in enumerate(neighbors):
                    for q in range(3):
                        gd = G5_derivative(crystal=self.crystal, i=i, element=core_elements[i], ni=neigh, 
                                           cutoff_f='Cosine', Rc=6.5, 
                                           eta=etas, lamBda=lamBdas, zeta=zetas, p=i, q=q)
                        Gd.append(gd)
#                        for n in neigh:
#                            if n[3] == (0.0, 0.0, 0.0):
#                                gd = G5_derivative(crystal=self.crystal, i=n[2], 
#                                                   element=n[0].species_string, 
#                                                   ni=neighbors[n[2]], 
#                                                   cutoff_f='Cosine', Rc=6.5, 
#                                                   eta=etas, lamBda=lamBdas, zeta=zetas, p=i, q=q)
#                                Gd.append(gd)
                                
        print(Gd)

        return G, Gd


    def get_parameters(self, G_type, params):
        """
        Function to get symmetry functions parameters: eta, zeta, etc.

        Returns
        -------
        Return an array of arrays of parameters.
        """
        # Need to finish G1 and G3.
        D = []
        
        if G_type == 'G2':
            elements = self.crystal.symbol_set

            for key, value in params.items():
                if key == 'eta':
                    eta = value

            for e in eta:
                for elem in elements:
                    d = {}
                    d['type'] = 'G2'
                    d['element'] = elem
                    d['eta'] = e
                    D.append(d)

        elif G_type == 'G4':
            elements = self.crystal.symbol_set
            elements = list(itertools.combinations_with_replacement(elements, 
                                                                    2))

            for key, value in params.items():
                if key == 'eta':
                    eta = value
                elif key == 'zeta':
                    zeta = value
                else:
                    lamBda = value

            for e in eta:
                for z in zeta:
                    for l in lamBda:
                        for elem in elements:
                            d = {}
                            d['type'] = 'G4'
                            d['elements'] = elem
                            d['eta'] = e
                            d['zeta'] = z
                            d['lamBda'] = l
                            D.append(d)
        
        elif G_type == 'G5':
            elements = self.crystal.symbol_set
            elements = list(itertools.combinations_with_replacement(elements, 
                                                                    2))

            for key, value in params.items():
                if key == 'eta':
                    eta = value
                elif key == 'zeta':
                    zeta = value
                else:
                    lamBda = value

            for e in eta:
                for z in zeta:
                    for l in lamBda:
                        for elem in elements:
                            d = {}
                            d['type'] = 'G5'
                            d['elements'] = elem
                            d['eta'] = e
                            d['zeta'] = z
                            d['lamBda'] = l
                            D.append(d)
        
        return D


#    def get_G_parameters(self):
#        G_parameters = []
#
#        if self.G1_parameters != []:
#            for i in self.G1_parameters:
#                G_parameters.append(i)
#        if self.G2_parameters != []:
#            for i in self.G2_parameters:
#                G_parameters.append(i)
#        if self.G3_parameters != []:
#            for i in self.G3_parameters:
#                G_parameters.append(i)
#        if self.G4_parameters != []:
#            for i in self.G4_parameters:
#                G_parameters.append(i)
#        if self.G5_parameters != []:
#            for i in self.G5_parameters:
#                G_parameters.append(i)
#
#        return G_parameters
        

    def reshaping(self, arr):
        m, n = arr.shape
        m, n = int(m * n / self.crystal.num_sites), self.crystal.num_sites
        arr = np.reshape(np.ravel(arr), (m, n))

        return arr.T


    def get_all_G(self, Gs='all'):
        if Gs == 'all':
            Gs = []

            if self.G1 != []:
                Gs = self.G1

            if self.G2 != []:
                if Gs != []:
                    Gs = np.hstack((Gs, self.G2))
                else:
                    Gs = self.G2

            if self.G3 != []:
                if Gs != []:
                    Gs = np.hstack((Gs, self.G3))
                else:
                    Gs = self.G3

            if self.G4 != []:
                if Gs != []:
                    Gs = np.hstack((Gs, self.G4))
                else:
                    Gs = self.G4

            if self.G5 != []:
                if Gs != []:
                    Gs = np.hstack((Gs, self.G5))
                else:
                    Gs = self.G5
        
        else:
            pass

        return Gs
    
    
    def _check_sanity(self, G_parameters, G_keywords):
        """
        Check if any of the parameters in the keywords.
        """        
        for key, value in G_parameters.items():
            if key in G_keywords:
                pass
            else:
                raise NotImplementedError(
                        f"Unknown parameter: {key}. "\
                        f"The available parameters are {self.G5_keywords}.")


############################# Auxiliary Functions #############################


def distance(arr):
    """
    L2 norm for cartesian coordinates
    """
    return np.linalg.norm(arr)


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
        return 1
    else:
        return 0


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
        dRab_dRpq = 0
    return dRab_dRpq


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
    if (p == a) or (p == b):
        dRab_dRpq_vector = [None, None, None]
        c1 = Kronecker(p, b) - Kronecker(p, a)
        dRab_dRpq_vector[0] = c1 * Kronecker(0, q)
        dRab_dRpq_vector[1] = c1 * Kronecker(1, q)
        dRab_dRpq_vector[2] = c1 * Kronecker(2, q)
        return dRab_dRpq_vector
    else:
        return [0, 0, 0]
    
    
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


############################## Cutoff Functional ##############################


"""
This script provides three cutoff functionals:
    1. Cosine
    2. Polynomial
    3. Hyperbolic Tangent

All cutoff functionals have an 'Rc' attribute which is the cutoff radius;
The Rc is used to calculate the neighborhood attribute. The functional will
return zero if the radius is beyond Rc.

This script is adopted from AMP:
    https://bitbucket.org/andrewpeterson/amp/src/2865e75a199a?at=master
"""


class Cosine(object):
    """
    Cutoff cosine functional suggested by Behler:
    Behler, J., & Parrinello, M. (2007). Generalized neural-network 
    representation of high-dimensional potential-energy surfaces. 
    Physical review letters, 98(14), 146401.
    (see eq. 3)
    
    Args:
        Rc(float): the cutoff radius.
    """
    def __init__(self, Rc):
        
        self.Rc = Rc


    def __call__(self, Rij):
        """
        Args:
            Rij(float): distance between pair atoms.
            
        Returns:
            The value (float) of the cutoff Cosine functional, will return zero
            if the radius is beyond the cutoff value.
        """
        if Rij > self.Rc:
            return 0.0
        else:
            return (0.5 * (np.cos(np.pi * Rij / self.Rc) + 1.))


    def derivative(self, Rij):
        """
        Calculate derivative (dF/dRij) of the Cosine functional with respect
        to Rij.
        
        Args:
            Rij(float): distance between pair atoms.
            
        Returns:
            The derivative (float) of the Cosine functional.
        """
        if Rij > self.Rc:
            return 0.0
        else:
            return (-0.5 * np.pi / self.Rc * np.sin(np.pi * Rij / self.Rc))


    def todict(self):
        return {'name': 'Cosine',
                'kwargs': {'Rc': self.Rc}}
        
        
class Polynomial(object):
    """
    Polynomial functional suggested by Khorshidi and Peterson:
    Khorshidi, A., & Peterson, A. A. (2016).
    Amp: A modular approach to machine learning in atomistic simulations. 
    Computer Physics Communications, 207, 310-324.
    (see eq. 9)

    Args:
        gamma(float): the polynomial power.
        Rc(float): the cutoff radius.
    """
    def __init__(self, Rc, gamma=4):
        self.gamma = gamma
        self.Rc = Rc


    def __call__(self, Rij):
        """
        Args:
            Rij(float): distance between pair atoms.
            
        Returns:
            The value (float) of the cutoff functional.
        """
        if Rij > self.Rc:
            return 0.0
        else:
            value = 1. + self.gamma * (Rij / self.Rc) ** (self.gamma + 1) - \
                (self.gamma + 1) * (Rij / self.Rc) ** self.gamma
            return value


    def derivative(self, Rij):
        """
        Derivative (dF/dRij) of the Polynomial functional with respect to Rij.
        
        Args:
            Rij(float): distance between pair atoms.
            
        Returns:
            The derivative (float) of the cutoff functional.
        """
        if Rij > self.Rc:
            return 0.0
        else:
            ratio = Rij / self.Rc
            value = (self.gamma * (self.gamma + 1) / self.Rc) * \
                (ratio ** self.gamma - ratio ** (self.gamma - 1))
        return value


    def todict(self):
        return {'name': 'Polynomial',
                'kwargs': {'Rc': self.Rc,
                           'gamma': self.gamma
                           }
                }
                        

class TangentH(object):
    """
    Cutoff hyperbolic Tangent functional suggested by Behler:
    Behler, J. (2015). 
    Constructing high‐dimensional neural network potentials: A tutorial review. 
    International Journal of Quantum Chemistry, 115(16), 1032-1050.
    (see eq. 7)

    Args:
        Rc(float): the cutoff radius.
    """
    def __init__(self, Rc):
        
        self.Rc = Rc


    def __call__(self, Rij):
        """
        Args:
            Rij(float): distance between pair atoms.
            
        Returns:
            The value (float) of the cutoff hyperbolic tangent functional, 
            will return zero if the radius is beyond the cutoff value.
        """
        if Rij > self.Rc:
            return 0.0
        else:
            return ((np.tanh(1.0 - (Rij / self.Rc))) ** 3)


    def derivative(self, Rij):
        """
        Calculate derivative (dF/dRij) of the hyperbolic Tangent functional 
        with respect to Rij.
        
        Args:
            Rij(float): distance between pair atoms.
            
        Returns:
            The derivative (float) of the hyberbolic tangent functional.
        """
        if Rij > self.Rc:
            return 0.0
        else:
            return (-3.0 / self.Rc * ((np.tanh(1.0 - (Rij / self.Rc))) ** 2 - \
                     (np.tanh(1.0 - (Rij / self.Rc))) ** 4))


    def todict(self):
        return {'name': 'TanH',
                'kwargs': {'Rc': self.Rc
                           }
                }


############################# Symmetry Functions ##############################
                

def calculate_G1(crystal, cutoff_f='Cosine', Rc=6.5):
    """
    Calculate G1 symmetry function.
    The most basic radial symmetry function using only the cutoff functional,
    the sum of the cutoff functionals for all neighboring atoms j inside the
    cutoff radius, Rc.
    
    One can refer to equation 8 in:
    Behler, J. (2015). Constructing high‐dimensional neural network 
    potentials: A tutorial review. 
    International Journal of Quantum Chemistry, 115(16), 1032-1050.

    Parameters
    ----------
    crystal: object
        Pymatgen crystal structure object.
    cutoff_f: str
        Cutoff functional. Default is Cosine functional.
    Rc: float
        Cutoff radius which the symmetry function will be calculated.
        Default value is 6.5 as suggested by Behler.
    """
    # Cutoff functional
    if cutoff_f == 'Cosine':
        func = Cosine(Rc=Rc)
    elif cutoff_f == 'Polynomial':
        func = Polynomial(Rc=Rc)
    elif cutoff_f == 'TangentH':
        func = TangentH(Rc=Rc)
    else:
        raise NotImplementedError('Unknown cutoff functional: %s' %cutoff_f)
     
    # Get elements in the crystal structure
    elements = crystal.symbol_set

    # Get core atoms information
    n_core = crystal.num_sites
    core_cartesians = crystal.cart_coords
    
    # Get neighbors information
    neighbors = crystal.get_all_neighbors(Rc)
    
    G1 = []
    
    for elem in elements:
        for i in range(n_core):
            G1_core = 0
            for j in range(len(neighbors[i])):
                if elem == neighbors[i][j][0].species_string:
                    Rij = np.linalg.norm(core_cartesians[i] - 
                                         neighbors[i][j][0].coords)
                    G1_core += func(Rij)
            G1.append(G1_core)
    
    return G1


def G1_derivative(crystal, cutoff_f='Cosine', Rc=6.5, p=1, q=0):
    """
    Calculate the derivative of the G1 symmetry function.
    
    Args:
        crystal: object
            Pymatgen crystal structure object
        cutoff_f: str
            Cutoff functional. Default is the cosine functional
        Rc: float
            Cutoff raidus which the symmetry function will be calculated
            Default value is 6.5 angstoms
        p : int
            Index of the atom force is acting on.
        q : int
            Direction of force. x = 0, y = 1, and z = 2.
    Returns:
        G1D: float
            The value of the derivative of the G1 symmetry function.
    """
    # Cutoff functional
    if cutoff_f == 'Cosine':
        func = Cosine(Rc=Rc)
    elif cutoff_f == 'Polynomial':
        func = Polynomial(Rc=Rc)
    elif cutoff_f == 'TangentH':
        func = TangentH(Rc=Rc)
    else:
        raise NotImplementedError('Unknown cutoff functional: %s' % cutoff_f)

    # Get core atoms information
    n_core = crystal.num_sites
    core_cartesians = crystal.cart_coords

    # Get neighbors information
    neighbors = crystal.get_all_neighbors(Rc)

    G1D = []

    for i in range(n_core):
        G1D_core = 0
        for j in range(len(neighbors[i])):
            Ri = core_cartesians[i]
            Rj = neighbors[i][j][0].coords
            Rij = np.linalg.norm(Rj - Ri)
            G1D_core += func.derivative(Rij) * \
                    dRab_dRpq(i, j, Ri, Rj, p, q)
        G1D.append(G1D_core)

    return G1D


def calculate_G2(crystal, cutoff_f='Cosine', Rc=6.5, eta=2, Rs=0.0):
    """
    Calculate G2 symmetry function.
    G2 function is a better choice to describe the radial feature of atoms in
    a crystal structure within the cutoff radius.
    
    One can refer to equation 9 in:
    Behler, J. (2015). Constructing high‐dimensional neural network 
    potentials: A tutorial review. 
    International Journal of Quantum Chemistry, 115(16), 1032-1050.

    Parameters
    ----------
    crystal: object
        Pymatgen crystal structure object.
    cutoff_f: str
        Cutoff functional. Default is Cosine functional.
    Rc: float
        Cutoff radius which the symmetry function will be calculated.
        Default value is 6.5 as suggested by Behler.
    eta: float
        The parameter of G2 symmetry function.
    Rs: float
        Determine the shift from the center of the Gaussian.
        Default value is zero.

    Returns
    -------
    G2 : an array of floats
        G2 symmetry value.
    """
    # Cutoff functional
    if cutoff_f == 'Cosine':
        func = Cosine(Rc=Rc)
    elif cutoff_f == 'Polynomial':
        func = Polynomial(Rc=Rc)
    elif cutoff_f == 'TangentH':
        func = TangentH(Rc=Rc)
    else:
        raise NotImplementedError('Unknown cutoff functional: %s' %cutoff_f)
    
    # Get elements in the crystal structure
    elements = crystal.symbol_set
    
    # Get positions of core atoms
    n_core = crystal.num_sites
    core_cartesians = crystal.cart_coords

    # Their neighbors within the cutoff radius
    neighbors = crystal.get_all_neighbors(Rc)
    
    G2 = []
    
    for elem in elements:
        for i in range(n_core):   
            G2_core = 0
            for j in range(len(neighbors[i])):
                if elem == neighbors[i][j][0].species_string:
                    Rij = np.linalg.norm(core_cartesians[i] - 
                                         neighbors[i][j][0]._coords)
                    G2_core += np.exp(-eta * Rij ** 2. / Rc ** 2.) * func(Rij)
            G2.append(G2_core)
    
    return G2


def G2_derivative(crystal, element, i, ni, cutoff_f='Cosine', Rc=6.5, eta=2, Rs=0.0, p=1, q=0):
    """
    Calculate the derivative of the G2 symmetry function.
    
    Args:
        crystal: object
            Pymatgen crystal structure object
        cutoff_f: str
            Cutoff functonal. Default cosine functional
        Rc: float
            Cutoff radius for symmetry function, defauly 6.5 angstoms
        eta: float
            The parameter for the G2 symmetry function
        Rs: float
            Determine the shift from the center of the gaussian, default= 0
    Returns:
        G2D: float
            The derivative of G2 symmetry function
    """
    # Cutoff functional
    if cutoff_f == 'Cosine':
        func = Cosine(Rc=Rc)
    elif cutoff_f == 'Polynomial':
        func = Polynomial(Rc=Rc)
    elif cutoff_f == 'TangentH':
        func = TangentH(Rc=Rc)
    else:
        raise NotImplementedError('Unknown cutoff functional: %s' %cutoff_f)
    
    # Get positions of core atoms
    core_cartesians = crystal.cart_coords
    Ri = core_cartesians[i]

    # Their neighbors within the cutoff radius
    elements = crystal.symbol_set

    G2D = []
    for e in eta:
        for elem in elements:
            g2D = 0
            for count in range(len(ni)):
                symbol = ni[count][0].species_string
                Rj = ni[count][0]._coords
                j = ni[count][2]
                if elem == symbol:
                    dRabdRpq = dRab_dRpq(i, j, Ri, Rj, p, q)
                    if dRabdRpq != 0:
                        Rij = np.linalg.norm(Rj - Ri)
                        g2D += np.exp(-e * (Rij - Rs)**2. / Rc**2.) * \
                                dRabdRpq * \
                                (-2. * e * (Rij - Rs) * func(Rij) / Rc**2. + 
                                func.derivative(Rij))
            G2D.append(g2D)

    return G2D


def calculate_G3(crystal, cutoff_f='Cosine', Rc=6.5, k=10):
    """
    Calculate G3 symmetry function.
    G3 function is a damped cosine functions with a period length described by
    K. For example, a Fourier series expansion a suitable description of the 
    radial atomic environment can be obtained by comibning several G3
    functions with different values for K.
    Note: due to the distances of atoms, G3 can cancel each other out depending
    on the positive and negative value.
    
    One can refer to equation 7 in:
    Behler, J. (2011). Atom-centered symmetry functions for constructing 
    high-dimensional neural network potentials. 
    The Journal of chemical physics, 134(7), 074106.
    
    Parameters
    ----------
    crystal: object
        Pymatgen crystal structure object.
    cutoff_f: str
        Cutoff functional. Default is Cosine functional.
    Rc: float
        Cutoff radius which the symmetry function will be calculated.
        Default value is 6.5 as suggested by Behler.
    k: float
        The Kappa value as G3 parameter.
    
    Returns
    -------
    G3: float
        G3 symmetry value
    """
    if cutoff_f == 'Cosine':
        func = Cosine(Rc=Rc)
    elif cutoff_f == 'Polynomial':
        func = Polynomial(Rc=Rc)
    elif cutoff_f == 'TangentH':
        func = TangentH(Rc=Rc)
    else:
        raise NotImplementedError('Unknown cutoff functional: %s' %cutoff_f)
    
    # Get elements in the crystal structure
    elements = crystal.symbol_set
    
    # Get core atoms information
    n_core = crystal.num_sites
    core_cartesians = crystal.cart_coords
    
    # Get neighbors information
    neighbors = crystal.get_all_neighbors(Rc)

    G3 = []

    for elem in elements:
        for i in range(n_core):
            G3_core = 0
            for j in range(len(neighbors[i])):
                if elem == neighbors[i][j][0].species_string:
                    Rij = np.linalg.norm(core_cartesians[i] - 
                                         neighbors[i][j][0]._coords)
                    G3_core += np.cos(k * Rij / Rc) * func(Rij)
            G3.append(G3_core)
    
    return G3


def G3_derivative(crystal, cutoff_f='Cosine', Rc=6.5, k=10, p=1, q=0):
    """
    Calculate derivative of the G3 symmetry function.
    
    Args:
        crystal: object
            Pymatgen crystal structure object.
        cutoff_f: str
            Cutoff functional. Default is Cosine functional.
        Rc: float
            Cutoff radius which the symmetry function will be calculated.
            Default value is 6.5 as suggested by Behler.
        k: float
            The Kappa value as G3 parameter.
    Returns:
        G3D: float
            Derivative of G3 symmetry function
    """
    if cutoff_f == 'Cosine':
        func = Cosine(Rc=Rc)
    else:
        raise NotImplementedError('Unknown cutoff functional: %s' % cutoff_f)

    # Get core atoms information
    n_core = crystal.num_sites
    core_cartesians = crystal.cart_coords

    # Get neighbors information
    neighbors = crystal.get_all_neighbors(Rc)

    G3D = []

    for i in range(n_core):
        G3D_core = 0
        for j in range(len(neighbors[i])):
            Ri = core_cartesians[i]
            Rj = neighbors[i][j][0].coords
            Rij = np.linalg.norm(Rj - Ri)
            G3D_core += (np.cos(k * Rij) * func.derivative(Rij) - \
                         k * np.sin(k * Rij) * func(Rij)) * \
                         dRab_dRpq(i, j, Ri, Rj, p, q)
        G3D.append(G3D_core)

    return G3D


def calculate_G4(crystal, cutoff_f='Cosine', Rc=6.5, eta=2, lamBda=1, zeta=1):
    """
    Calculate G4 symmetry function.
    G4 function is an angular function utilizing the cosine funtion of the
    angle theta_ijk centered at atom i.

    One can refer to equation 8 in:
    Behler, J. (2011). Atom-centered symmetry functions for constructing 
    high-dimensional neural network potentials. 
    The Journal of chemical physics, 134(7), 074106.
    
    Parameters
    ----------
    crystal: object
        Pymatgen crystal structure object.
    cutoff_f: str
        Cutoff functional. Default is Cosine functional.
    Rc: float
        Cutoff radius which the symmetry function will be calculated.
        Default value is 6.5 as suggested by Behler.
    eta: float
        The parameter of G4 symmetry function.
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
        G4 symmetry value
    """
    # Cutoff functional
    if cutoff_f == 'Cosine':
        func = Cosine(Rc=Rc)
    elif cutoff_f == 'Polynomial':
        func = Polynomial(Rc=Rc)
    elif cutoff_f == 'TangentH':
        func = TangentH(Rc=Rc)
    else:
        raise NotImplementedError('Unknown cutoff functional: %s' %cutoff_f)
    
    # Get elements in the crystal structure
    elements = crystal.symbol_set
    elements = list(itertools.combinations_with_replacement(elements, 2))

    # Get core atoms information
    n_core = crystal.num_sites
    core_cartesians = crystal.cart_coords
    
    # Get neighbors information
    neighbors = crystal.get_all_neighbors(Rc)
    
    G4 = []
    
    for elem in elements:
        for i in range(n_core):
            G4_core = 0.0
            for j in range(len(neighbors[i])-1):
                for k in range(j+1, len(neighbors[i])):
                    n1 = neighbors[i][j][0].species_string
                    n2 = neighbors[i][k][0].species_string
                    if (elem[0] == n1 and elem[1] == n2) or \
                        (elem[1] == n1 and elem[0] == n2):
                        Ri = core_cartesians[i]
                        Rj = neighbors[i][j][0].coords
                        Rk = neighbors[i][k][0].coords
                    
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
                        term *= func(Rij) * func(Rik) * func(Rjk)
                        G4_core += term
            G4_core *= 2. ** (1. - zeta)
            G4.append(G4_core)
        
    return G4


def G4_derivative(crystal, i, element, ni, cutoff_f='Cosine', 
                  Rc=6.5, eta=2, lamBda=1, zeta=1, p=1, q=0):
    """
    Calculate the derivative of the G4 symmetry function.
    
    Parameters
    ----------
    crystal: object
        Pymatgen crystal structure object.
    cutoff_f: str
        Cutoff functional. Default is Cosine functional.
    Rc: float
        Cutoff radius which the symmetry function will be calculated.
        Default value is 6.5 as suggested by Behler.
    eta: float
        The parameter of G4 symmetry function.
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
    G4D: float
        The derivative of G4 symmetry function
    """
    # Cutoff functional
    if cutoff_f == 'Cosine':
        func = Cosine(Rc=Rc)
    elif cutoff_f == 'Polynomial':
        func = Polynomial(Rc=Rc)
    elif cutoff_f == 'TangentH':
        func = TangentH(Rc=Rc)
    else:
        raise NotImplementedError('Unknown cutoff functional: %s' %cutoff_f)
        
    # Get positions of core atoms
    core_cartesians = crystal.cart_coords
    Ri = core_cartesians[i]

    # Their neighbors within the cutoff radius
    elements = crystal.symbol_set
    elements = list(itertools.combinations_with_replacement(elements, 2))
    
    counts = range(len(ni))
    
    G4D = []
    for e in eta:
        for z in zeta:
            for l in lamBda:
                for elem in elements:
                    g4D = 0
                    for j in counts:
                        for k in counts[(j+1):]:
                            n1 = ni[j][0].species_string
                            n2 = ni[k][0].species_string
                            if (elem[0] == n1 and elem[1] == n2) or \
                                (elem[1] == n1 and elem[0] == n2):
                                Rj = ni[j][0].coords
                                Rk = ni[k][0].coords
                                
                                Rij_vector = Rj - Ri
                                Rij = np.linalg.norm(Rij_vector)
                                
                                Rik_vector = Rk - Ri
                                Rik = np.linalg.norm(Rik_vector)
                                
                                Rjk_vector = Rk - Rj
                                Rjk = np.linalg.norm(Rjk_vector)
                                
                                cos_ijk = np.dot(Rij_vector, Rik_vector)/ Rij / Rik
                                dcos_ijk = dcos_dRpq(i, ni[j][2], ni[k][2], Ri, Rj, Rk, p, q)
                                
                                cutoff = func(Rij) * func(Rik) * func(Rjk)
                                cutoff_Rik_Rjk = func(Rik) * func(Rjk)
                                cutoff_Rij_Rjk = func(Rij) * func(Rjk)
                                cutoff_Rij_Rik = func(Rij) * func(Rik)
                                
                                cutoff_Rij_derivative = func.derivative(Rij) * \
                                                        dRab_dRpq(i, ni[j][2], Ri, Rj, p, q)
                                cutoff_Rik_derivative = func.derivative(Rik) * \
                                                        dRab_dRpq(i, ni[k][2], Ri, Rk, p, q)
                                cutoff_Rjk_derivative = func.derivative(Rjk) * \
                                                        dRab_dRpq(ni[j][2], ni[k][2], Rj, Rk, p, q)
                                
                                lamBda_term = 1. + l * cos_ijk
                                
                                first_term = l * z * dcos_ijk
                                first_term += (-2. * e * lamBda_term / (Rc ** 2)) * \
                                                (Rij * dRab_dRpq(i, ni[j][2], Ri, Rj, p, q) + 
                                                 Rik * dRab_dRpq(i, ni[k][2], Ri, Rk, p, q) +
                                                 Rjk * dRab_dRpq(ni[j][2], ni[k][2], Rj, Rk, p, q))
                                first_term *= cutoff
                                
                                second_term = cutoff_Rij_derivative * cutoff_Rik_Rjk + \
                                                    cutoff_Rik_derivative * cutoff_Rij_Rjk + \
                                                    cutoff_Rjk_derivative * cutoff_Rij_Rik
                                second_term *= lamBda_term
                                
                                term = first_term + second_term
                                term *= lamBda_term ** (z - 1.)
                                term *= np.exp(-e * (Rij ** 2. + Rik ** 2. + Rjk ** 2.) /
                                               Rc ** 2.)
                                
                                g4D += term
                                
                    g4D *= 2. ** (1. - z)

                    G4D.append(g4D)
                            
    return G4D
    

def calculate_G5(crystal, cutoff_f='Cosine', Rc=6.5, eta=2, lamBda=1, zeta=1):
    """
    Calculate G5 symmetry function.
    G5 function is also an angular function utilizing the cosine funtion of the
    angle theta_ijk centered at atom i. The difference between G5 and G4 is 
    that G5 does not depend on the Rjk value. Hence, the G5 will generate a 
    greater value after the summation compared to G4.

    One can refer to equation 9 in:
    Behler, J. (2011). Atom-centered symmetry functions for constructing 
    high-dimensional neural network potentials. 
    The Journal of chemical physics, 134(7), 074106.
    
    Parameters
    ----------
    crystal: object
        Pymatgen crystal structure object.
    cutoff_f: str
        Cutoff functional. Default is Cosine functional.
    Rc: float
        Cutoff radius which the symmetry function will be calculated.
        Default value is 6.5 as suggested by Behler.
    eta: float
        The parameter of G5 symmetry function.
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
    G5: float
        G5 symmetry value
    """    
    if cutoff_f == 'Cosine':
        func = Cosine(Rc=Rc)
    elif cutoff_f == 'Polynomial':
        func = Polynomial(Rc=Rc)
    elif cutoff_f == 'TangentH':
        func = TangentH(Rc=Rc)
    else:
        raise NotImplementedError('Unknown cutoff functional: %s' %cutoff_f)
    
    # Get elements in the crystal structure
    elements = crystal.symbol_set
    elements = list(itertools.combinations_with_replacement(elements, 2))
    
    #Get core atoms information
    n_core = crystal.num_sites
    core_cartesians = crystal.cart_coords
    
    # Get neighbors information
    neighbors = crystal.get_all_neighbors(Rc)
    
    G5 = []

    for elem in elements:
        for i in range(n_core):
            G5_core = 0.0
            for j in range(len(neighbors[i])-1):
                for k in range(j+1, len(neighbors[i])):
                    n1 = neighbors[i][j][0]
                    n2 = neighbors[i][k][0]
                    if (elem[0] == n1.species_string \
                        and elem[1] == n2.species_string) or \
                        (elem[1] == n1.species_string and \
                         elem[0] == n2.species_string):
                        Rij_vector = core_cartesians[i] - n1.coords
                        Rij = np.linalg.norm(Rij_vector)
                        Rik_vector = core_cartesians[i] - n2.coords
                        Rik = np.linalg.norm(Rik_vector)
                        cos_ijk = np.dot(Rij_vector, Rik_vector)/ Rij / Rik
                        term = (1. + lamBda * cos_ijk) ** zeta
                        term *= np.exp(-eta * 
                                       (Rij ** 2. + Rik ** 2.) / Rc ** 2.)
                        term *= func(Rij) * func(Rik)
                        G5_core += term
            G5_core *= 2. ** (1. - zeta)
            G5.append(G5_core)
        
    return G5


def G5_derivative(crystal, i, element, ni, cutoff_f='Cosine', 
                  Rc=6.5, eta=2, lamBda=1, zeta=1, p=1, q=0):
    """
    Calculate the derivative of the G5 symmetry function.
    
    Parameters
    ----------
    crystal: object
        Pymatgen crystal structure object.
    cutoff_f: str
        Cutoff functional. Default is Cosine functional.
    Rc: float
        Cutoff radius which the symmetry function will be calculated.
        Default value is 6.5 as suggested by Behler.
    eta: float
        The parameter of G5 symmetry function.
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
    G5D: float
        The derivative of G5 symmetry function
    """
    if cutoff_f == 'Cosine':
        func = Cosine(Rc=Rc)
    elif cutoff_f == 'Polynomial':
        func = Polynomial(Rc=Rc)
    elif cutoff_f == 'TangentH':
        func = TangentH(Rc=Rc)
    else:
        raise NotImplementedError('Unknown cutoff functional: %s' %cutoff_f)
        
    # Get positions of core atoms
    core_cartesians = crystal.cart_coords
    Ri = core_cartesians[i]

    # Their neighbors within the cutoff radius
    elements = crystal.symbol_set
    elements = list(itertools.combinations_with_replacement(elements, 2))
    
    counts = range(len(ni))
    
    G5D = []
    for e in eta:
        for z in zeta:
            for l in lamBda:
                for elem in elements:
                    g5D = 0
                    for j in counts:
                        for k in counts[(j+1):]:
                            n1 = ni[j][0].species_string
                            n2 = ni[k][0].species_string
                            if (elem[0] == n1 and elem[1] == n2) or \
                                (elem[1] == n1 and elem[0] == n2):
                                Rj = ni[j][0].coords
                                Rk = ni[k][0].coords
                                
                                Rij_vector = Rj - Ri
                                Rik_vector = Rk - Ri
                                Rij = np.linalg.norm(Rij_vector)
                                Rik = np.linalg.norm(Rik_vector)
                                
                                cos_ijk = np.dot(Rij_vector, Rik_vector) / Rij / Rik
                                dcos_ijk = dcos_dRpq(i, ni[j][2], ni[k][2], Ri, Rj, Rk, p, q)
                                
                                cutoff = func(Rij) * func(Rik)
                                cutoff_Rij_derivative = func.derivative(Rij) * \
                                                        dRab_dRpq(i, ni[j][2], Ri, Rj, p, q)
                                cutoff_Rik_derivative = func.derivative(Rik) * \
                                                        dRab_dRpq(i, ni[k][2], Ri, Rk, p, q)
                
                                lamBda_term = 1. + l * cos_ijk
                                
                                first_term = -2 * e / Rc ** 2 * lamBda_term * \
                                                (Rij * dRab_dRpq(i, ni[j][2], Ri, Rj, p, q) + 
                                                 Rik * dRab_dRpq(i, ni[k][2], Ri, Rk, p, q))
                                first_term += l * z * dcos_ijk
                                first_term *= cutoff
                                
                                second_term = lamBda_term * \
                                                (cutoff_Rij_derivative * func(Rik) + 
                                                 cutoff_Rik_derivative * func(Rij))
                                                
                                term = first_term + second_term
                                term *= lamBda_term ** (z - 1.)
                                term *= np.exp(-e * (Rij ** 2. + Rik ** 2.) /
                                               Rc ** 2.)
                                                
                                g5D += term
            
                    g5D *= 2. ** (1. - z)
                    G5D.append(g5D)
        
    return G5D


#crystal = Structure.from_file('../POSCARs/POSCAR-NaCl')
#
#sym_params = {'G2': {'eta': np.logspace(np.log10(0.05), 
#                                         np.log10(5.), num=4)}}
#                'G5': {'eta': [0.005],
#                        'zeta': [1., 4.],
#                        'lamBda': [1., -1.]}}

#gauss = gaussian(crystal, sym_params, derivative=True)
#print(gauss.G2_derivative)


#    def get_statistics(self, Gs):
#        Gs_mean = np.mean(Gs, axis=0)
#        Gs_std = np.std(Gs, axis=0)
#        Gs_skew = skew(Gs, axis=0)
#        Gs_kurtosis = kurtosis(Gs, axis=0)
#        Gs_covariance = np.cov(Gs.T)
#
#        return None
