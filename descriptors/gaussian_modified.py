import numpy as np
from scipy.stats import skew, kurtosis
import itertools


################################ Gaussian Class ###############################


class symf:
    """
    Get the all the desired symmetry functions.

    Parameters
    ----------
    crystal: object
        Pymatgen crystal structure object.
    symmetry_parameters: dict
        Dictionary of symmetry parameters.
        i.e. {'G2': {'eta': [0.05, 0.1]}}
    derivative: bool
        If True, calculate the derivatives of symmetry functions.
    """
    def __init__(self, crystal, symmetry_parameters, derivative=False):
        self.crystal = crystal
        self.symmetry_parameters = symmetry_parameters
        self.derivative = derivative

        self.G1_keywords = ['Rc', 'functional']
        self.G2_keywords = ['eta', 'Rc', 'Rs', 'functional']
        self.G3_keywords = ['kappa', 'Rc', 'functional']
        self.G4_keywords = ['eta', 'lamBda', 'zeta', 'Rc', 'functional']
        self.G5_keywords = ['eta', 'lamBda', 'zeta', 'Rc', 'functional']

        self.G_types = [] # e.g. ['G2', 'G4']

        # Setting up parameters for each of the symmetry function type.
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
        
        if self.G1_parameters is not None:
            self._check_sanity(self.G1_parameters, self.G1_keywords)
            self.G1, self.G1_prime = self.calculate('G1', self.G1_parameters, 
                                                    derivative)

        if self.G2_parameters is not None:
            self._check_sanity(self.G2_parameters, self.G2_keywords)
            self.G2, self.G2_prime = self.calculate('G2', self.G2_parameters, 
                                                    derivative)
        
        if self.G3_parameters is not None:
            self._check_sanity(self.G3_parameters, self.G3_keywords)
            self.G3, self.G3_prime = self.calculate('G3', self.G3_parameters, 
                                                    derivative)

        if self.G4_parameters is not None:
            self._check_sanity(self.G4_parameters, self.G4_keywords)
            self.G4, self.G4_prime = self.calculate('G4', self.G4_parameters, 
                                                    derivative)

        if self.G5_parameters is not None:
            self._check_sanity(self.G5_parameters, self.G5_keywords)
            self.G5, self.G5_prime = self.calculate('G5', self.G5_parameters, 
                                                    derivative)
    

    def calculate(self, G_type, symmetry_parameters, derivative=False):
        G, Gp = [], []
        Rc = 6.5
        functional = 'Cosine'
        Rs = 0.
        
        for key, value in symmetry_parameters.items():
            if key == 'Rc':
                Rc = value
            elif key == 'functional':
                functional = value
            elif key == 'Rs':
                Rs = value
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
        
        n_core = self.crystal.num_sites
        e_types = self.crystal.symbol_set # element types, i.e. ['Na', 'Cl']
        # element pair types, i.e. [('Na', 'Na'), ('Na', 'Cl'), ('Cl', 'Cl')]
        ep_types = list(itertools.combinations_with_replacement(e_types, 2))

        if derivative:
            neighbors_info = self.crystal.get_all_neighbors(r=Rc,
                                                            include_index=True,
                                                            include_image=True)


        if G_type == 'G2':
            for i in range(n_core):
                g = []
                for eta in etas:
                    for ele in e_types:
                        g.append(G2(self.crystal, 
                                    i=i,
                                    e_type=ele,
                                    functional=functional,
                                    Rc=Rc, eta=eta, Rs=Rs))
                G.append(g)

            if derivative:
                for i in range(n_core):
                    for q in range(3):
                        gp = []
                        for eta in etas:
                            for ele in e_types:
                                gp.append(G2_prime(self.crystal, 
                                                   i=i, 
                                                   e_type=ele,
                                                   ni=neighbors_info[i], 
                                                   functional='Cosine', 
                                                   Rc=Rc,
                                                   eta=eta, 
                                                   Rs=Rs,
                                                   p=i, q=q))
                        Gp.append(gp)
                        
                        for n in neighbors_info[i]:
                            if n[3] == (0.0, 0.0, 0.0):
                                gp = []
                                for eta in etas:
                                    for ele in e_types:
                                        ni = neighbors_info[n[2]]
                                        prime = G2_prime(self.crystal,
                                                         i=n[2],
                                                         e_type=ele,
                                                         ni=ni,
                                                         functional='Cosine', 
                                                         Rc=Rc, 
                                                         eta=eta, 
                                                         Rs=Rs, 
                                                         p=i, q=q)
                                        gp.append(prime)
                                Gp.append(gp)

                            
        elif G_type == 'G3':
            for i in range(n_core):
                g = []
                for kappa in kappas:
                    for ele in e_types:
                        g.append(G3(self.crystal, 
                                    i=i,
                                    e_type=ele,
                                    functional=functional,
                                    Rc=Rc, k=kappa))
                G.append(g)

            if derivative:
                for i in range(n_core):
                    for q in range(3):
                        gp = []
                        for kappa in kappas:
                            for elem in e_types:
                                gp.append(G3_prime(self.crystal, 
                                             i=i, 
                                             e_type=elem,
                                             ni=neighbors_info[i], 
                                             functional='Cosine', 
                                             Rc=Rc,
                                             kappa=kappa, 
                                             Rs=Rs,
                                             p=i, q=q))
                        Gp.append(gp)
                        
                        for n in neighbors_info[i]:
                            if n[3] == (0.0, 0.0, 0.0):
                                gp = []
                                for kappa in kappas:
                                    for elem in e_types:
                                        ni = neighbors_info[n[2]]
                                        prime = G3_prime(self.crystal,
                                                         i=n[2],
                                                         e_type=elem,
                                                         ni=ni,
                                                         functional='Cosine', 
                                                         Rc=Rc, 
                                                         kappa=kappa, 
                                                         Rs=Rs, 
                                                         p=i, q=q)
                                        gp.append(prime)
                                Gp.append(gp)
                                                                    

        elif G_type == 'G4':
            for i in range(n_core):
                g = []
                for eta in etas:
                    for zeta in zetas:
                        for lb in lamBdas:
                            for pele in ep_types:
                                g.append(G4(self.crystal, 
                                            i=i,
                                            ep_type=pele,
                                            functional=functional,
                                            Rc=Rc, eta=eta, 
                                            lamBda=lb, zeta=zeta))
                G.append(g)
             
            if derivative:
                for i in range(n_core):
                    for q in range(3):
                        gp = []
                        for eta in etas:
                            for zeta in zetas:
                                for lb in lamBdas:
                                    for pele in ep_types:
                                        ni = neighbors_info[i]
                                        gp.append(G4_prime(self.crystal, 
                                                           i=i, 
                                                           ep_type=pele, 
                                                           ni=ni, 
                                                           functional=functional, 
                                                           Rc=Rc, 
                                                           eta=eta, 
                                                           lamBda=lb, 
                                                           zeta=zeta, 
                                                           p=i, q=q))
                        Gp.append(gp)
                        
                        for n in neighbors_info[i]:
                            if n[3] == (0.0, 0.0, 0.0):
                                gp = []
                                for eta in etas:
                                    for zeta in zetas:
                                        for lb in lamBdas:
                                            for pele in ep_types:
                                                ni = neighbors_info[n[2]]
                                                prime = G4_prime(self.crystal, 
                                                                 i=n[2], 
                                                                 ep_type=pele, 
                                                                 ni=ni, 
                                                                 functional=functional,
                                                                 Rc=Rc, 
                                                                 eta=eta, 
                                                                 lamBda=lb, 
                                                                 zeta=zeta, 
                                                                 p=i, q=q)
                                                gp.append(prime)
                                Gp.append(gp)

        elif G_type == 'G5':
            for i in range(n_core):
                g = []
                for eta in etas:
                    for zeta in zetas:
                        for lb in lamBdas:
                            for pele in ep_types:
                                g.append(G5(self.crystal, 
                                            i=i,
                                            ep_type=pele,
                                            functional=functional,
                                            Rc=Rc, eta=eta, 
                                            lamBda=lb, zeta=zeta))
                G.append(g)
             
            if derivative:
                for i in range(n_core):
                    for q in range(3):
                        gp = []
                        for eta in etas:
                            for zeta in zetas:
                                for lb in lamBdas:
                                    for pele in ep_types:
                                        ni = neighbors_info[i]
                                        gp.append(G5_prime(self.crystal, 
                                                           i=i, 
                                                           ep_type=pele, 
                                                           ni=ni, 
                                                           functional=functional, 
                                                           Rc=Rc, 
                                                           eta=eta, 
                                                           lamBda=lb, 
                                                           zeta=zeta, 
                                                           p=i, q=q))
                        Gp.append(gp)
                        
                        for n in neighbors_info[i]:
                            if n[3] == (0.0, 0.0, 0.0):
                                gp = []
                                for eta in etas:
                                    for zeta in zetas:
                                        for lb in lamBdas:
                                            for pele in ep_types:
                                                ni = neighbors_info[n[2]]
                                                prime = G5_prime(self.crystal, 
                                                                 i=n[2], 
                                                                 ep_type=pele, 
                                                                 ni=ni, 
                                                                 functional=functional,
                                                                 Rc=Rc, 
                                                                 eta=eta, 
                                                                 lamBda=lb, 
                                                                 zeta=zeta, 
                                                                 p=i, q=q)
                                                gp.append(prime)
                                Gp.append(gp)

        return G, Gp


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
                           'gamma': self.gamma}}
                        

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
                'kwargs': {'Rc': self.Rc}}


############################# Symmetry Functions ##############################
                

def G1(crystal, cutoff_f='Cosine', Rc=6.5):
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
    functional: str
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


def G1_prime(crystal, cutoff_f='Cosine', Rc=6.5, p=1, q=0):
    """
    Calculate the derivative of the G1 symmetry function.
    
    Args:
        crystal: object
            Pymatgen crystal structure object
        functional: str
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
            The value of the derivative of the G1 symmetry function
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


def G2(crystal, i, e_type, functional='Cosine', Rc=6.5, eta=2, Rs=0.0):
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
    i: int
        The index of core element.
    e_type: str
        The allowed element presents in the symmetry function calculation.
    functional: str
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
    G2 : float
        G2 symmetry value.
    """
    
    # Cutoff functional
    if functional == 'Cosine':
        func = Cosine(Rc=Rc)
    elif functional == 'Polynomial':
        func = Polynomial(Rc=Rc)
    elif functional == 'TangentH':
        func = TangentH(Rc=Rc)
    else:
        raise NotImplementedError('Unknown cutoff functional: %s' %functional)
        
    # Get positions of core atoms
    Ri = crystal.cart_coords[i]

    # Their neighbors within the cutoff radius
    neighbors = crystal.get_all_neighbors(Rc)
        
    G2 = 0
    for j in range(len(neighbors[i])):
        if e_type == neighbors[i][j][0].species_string:
            Rj = neighbors[i][j][0]._coords
            Rij = np.linalg.norm(Ri - Rj)
            G2 += np.exp(-eta * Rij ** 2. / Rc ** 2.) * func(Rij)
    
    return G2


def G2_prime(crystal, i, e_type, ni, functional='Cosine', 
                  Rc=6.5, eta=2, Rs=0.0, p=1, q=0):
    """
    Calculate the derivative of the G2 symmetry function.
    
    Parameters
    ----------
    crystal: object
        Pymatgen crystal structure object.
    i: int
        The index of core element.
    e_types: str
        The allowed element presents in the symmetry function calculation.
    ni: array of neighbors information
        Neighbors information of the core element.
    functional: str
        Cutoff functional. Default is Cosine functional.
    Rc: float
        Cutoff radius which the symmetry function will be calculated.
        Default value is 6.5 as suggested by Behler.
    eta: float
        The parameter of G2 symmetry function.
    Rs: float
        Determine the shift from the center of the Gaussian.
        Default value is zero.
    p: int
        The atom that the force is acting on.
    q: int
        Direction of force.
        
    Returns
    -------
    G2p : float
        The derivative of G2 symmetry value.
    """
    
    # Cutoff functional
    if functional == 'Cosine':
        func = Cosine(Rc=Rc)
    elif functional == 'Polynomial':
        func = Polynomial(Rc=Rc)
    elif functional == 'TangentH':
        func = TangentH(Rc=Rc)
    else:
        raise NotImplementedError('Unknown cutoff functional: %s' %functional)
    
    # Get positions of core atoms
    Ri = crystal.cart_coords[i]

    G2p = 0
    for count in range(len(ni)):
        symbol = ni[count][0].species_string
        Rj = ni[count][0]._coords
        j = ni[count][2]
        if e_type == symbol:
            dRabdRpq = dRab_dRpq(i, j, Ri, Rj, p, q)
            if dRabdRpq != 0:
                Rij = np.linalg.norm(Rj - Ri)
                G2p += np.exp(-eta * (Rij - Rs)**2. / Rc**2.) * \
                        dRabdRpq * \
                        (-2. * eta * (Rij - Rs) * func(Rij) / Rc**2. + 
                        func.derivative(Rij))

    return G2p


def G3(crystal, i, e_type, functional='Cosine', Rc=6.5, k=10):
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
    i: int
        The index of core element.
    e_type: str
        The allowed element presents in the symmetry function calculation.
    functional: str
        Cutoff functional. Default is Cosine functional.
    Rc: float
        Cutoff radius which the symmetry function will be calculated.
        Default value is 6.5 as suggested by Behler.
    k: float
        The Kappa value as G3 parameter.
    
    Returns
    -------
    G3: float
        G3 symmetry value.
    """
    
    if functional == 'Cosine':
        func = Cosine(Rc=Rc)
    elif functional == 'Polynomial':
        func = Polynomial(Rc=Rc)
    elif functional == 'TangentH':
        func = TangentH(Rc=Rc)
    else:
        raise NotImplementedError('Unknown cutoff functional: %s' %functional)
    
    # Get positions of core atoms
    Ri = crystal.cart_coords[i]
    
    # Get neighbors information
    neighbors = crystal.get_all_neighbors(Rc)

    G3 = 0
    for j in range(len(neighbors[i])):
        if e_type == neighbors[i][j][0].species_string:
            Rj = neighbors[i][j][0]._coords
            Rij = np.linalg.norm(Ri - Rj)
            G3 += np.cos(k * Rij / Rc) * func(Rij)
    
    return G3


def G3_prime(crystal, i, e_type, ni, functional='Cosine', 
             Rc=6.5, k=10, p=1, q=0):
    """
    Calculate derivative of the G3 symmetry function.
    
    Parameters
    ----------
    crystal: object
        Pymatgen crystal structure object.
    i: int
        The index of core element.
    e_types: str
        The allowed element presents in the symmetry function calculation.
    ni: array of neighbors information
        Neighbors information of the core element.
    functional: str
        Cutoff functional. Default is Cosine functional.
    Rc: float
        Cutoff radius which the symmetry function will be calculated.
        Default value is 6.5 as suggested by Behler.
    k: float
        The Kappa value as G3 parameter.
    p: int
        The atom that the force is acting on.
    q: int
        Direction of force.
        
    Returns
    -------
    G3p : float
        The derivative of G3 symmetry value.
    """
    
    if functional == 'Cosine':
        func = Cosine(Rc=Rc)
    elif functional == 'Polynomial':
        func = Polynomial(Rc=Rc)
    elif functional == 'TangentH':
        func = TangentH(Rc=Rc)
    else:
        raise NotImplementedError('Unknown cutoff functional: %s' %functional)

    # Get positions of core atoms
    Ri = crystal.cart_coords[i]

    G3p = 0
    for count in range(len(ni)):
        Rj = ni[count][0]._coords
        symbol = ni[count][0].species_string
        j = ni[count][2]
        if e_type == symbol:
            dRabdRpq = dRab_dRpq(i, j, Ri, Rj, p, q)
            if dRabdRpq != 0:
                Rij = np.linalg.norm(Rj - Ri)
                G3p += (np.cos(k * Rij) * func.derivative(Rij) - \
                             k * np.sin(k * Rij) * func(Rij)) * \
                             dRab_dRpq(i, j, Ri, Rj, p, q)

    return G3p


def G4(crystal, i, ep_type, functional='Cosine', 
       Rc=6.5, eta=2, lamBda=1, zeta=1):
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
    i: int
        The index of core element.
    ep_types: str
        The allowed element pair presents in the symmetry function calculation.
    functional: str
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
        G4 symmetry value.
    """
    
    # Cutoff functional
    if functional == 'Cosine':
        func = Cosine(Rc=Rc)
    elif functional == 'Polynomial':
        func = Polynomial(Rc=Rc)
    elif functional == 'TangentH':
        func = TangentH(Rc=Rc)
    else:
        raise NotImplementedError('Unknown cutoff functional: %s' %functional)

    # Get core atoms information
    Ri = crystal.cart_coords[i]
    
    # Get neighbors information
    neighbors = crystal.get_all_neighbors(Rc)    

    G4 = 0.0
    for j in range(len(neighbors[i])-1):
        for k in range(j+1, len(neighbors[i])):
            n1 = neighbors[i][j][0].species_string
            n2 = neighbors[i][k][0].species_string
            if (ep_type[0] == n1 and ep_type[1] == n2) or \
                (ep_type[1] == n1 and ep_type[0] == n2):
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
                G4 += term
    G4 *= 2. ** (1. - zeta)
        
    return G4


def G4_prime(crystal, i, ep_type, ni, functional='Cosine', 
             Rc=6.5, eta=2, lamBda=1, zeta=1, p=1, q=0):
    """
    Calculate the derivative of the G4 symmetry function.
    
    Parameters
    ----------
    crystal: object
        Pymatgen crystal structure object.
    i: int
        The index of core element.
    ep_types: str
        The allowed element pair presents in the symmetry function calculation.
    ni: array of neighbors information
        Neighbors information of the core element.
    functional: str
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
    p: int
        The atom that the force is acting on.
    q: int
        Direction of force.

    Returns
    -------
    G4p: float
        The derivative of G4 symmetry function.
    """
    
    # Cutoff functional
    if functional == 'Cosine':
        func = Cosine(Rc=Rc)
    elif functional == 'Polynomial':
        func = Polynomial(Rc=Rc)
    elif functional == 'TangentH':
        func = TangentH(Rc=Rc)
    else:
        raise NotImplementedError('Unknown cutoff functional: %s' %functional)
        
    # Get positions of core atoms
    Ri = crystal.cart_coords[i]
    
    counts = range(len(ni))
    
    G4p = 0
    for j in counts:
        for k in counts[(j+1):]:
            n1 = ni[j][0].species_string
            n2 = ni[k][0].species_string
            if (ep_type[0] == n1 and ep_type[1] == n2) or \
                (ep_type[1] == n1 and ep_type[0] == n2):
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
                
                dRij = dRab_dRpq(i, ni[j][2], Ri, Rj, p, q)
                dRik = dRab_dRpq(i, ni[k][2], Ri, Rk, p, q)
                dRjk = dRab_dRpq(ni[j][2], ni[k][2], Rj, Rk, p, q)
                
                cutoff_Rij_derivative = func.derivative(Rij) * dRij
                cutoff_Rik_derivative = func.derivative(Rik) * dRik
                cutoff_Rjk_derivative = func.derivative(Rjk) * dRjk
                
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


def G5(crystal, i, ep_type, functional='Cosine', 
       Rc=6.5, eta=2, lamBda=1, zeta=1):
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
    i: int
        The index of core element.
    ep_types: str
        The allowed element pair presents in the symmetry function calculation.
    functional: str
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
    G5: float
        G5 symmetry value.
    """    
    
    # Cutoff functional
    if functional == 'Cosine':
        func = Cosine(Rc=Rc)
    elif functional == 'Polynomial':
        func = Polynomial(Rc=Rc)
    elif functional == 'TangentH':
        func = TangentH(Rc=Rc)
    else:
        raise NotImplementedError('Unknown cutoff functional: %s' %functional)
    
    # Get core atoms information
    Ri = crystal.cart_coords[i]
    
    # Get neighbors information
    neighbors = crystal.get_all_neighbors(Rc)

    G5 = 0.0
    for j in range(len(neighbors[i])-1):
        for k in range(j+1, len(neighbors[i])):
            n1 = neighbors[i][j][0]
            n2 = neighbors[i][k][0]
            if (ep_type[0] == n1.species_string \
                and ep_type[1] == n2.species_string) or \
                (ep_type[1] == n1.species_string and \
                 ep_type[0] == n2.species_string):
                Rij_vector = Ri - n1.coords
                Rij = np.linalg.norm(Rij_vector)
                Rik_vector = Ri - n2.coords
                Rik = np.linalg.norm(Rik_vector)
                cos_ijk = np.dot(Rij_vector, Rik_vector)/ Rij / Rik
                term = (1. + lamBda * cos_ijk) ** zeta
                term *= np.exp(-eta * 
                               (Rij ** 2. + Rik ** 2.) / Rc ** 2.)
                term *= func(Rij) * func(Rik)
                G5 += term
    G5 *= 2. ** (1. - zeta)
        
    return G5


def G5_prime(crystal, i, ep_type, ni, functional='Cosine', 
                  Rc=6.5, eta=2, lamBda=1, zeta=1, p=1, q=0):
    """
    Calculate the derivative of the G5 symmetry function.
    
    Parameters
    ----------
    crystal: object
        Pymatgen crystal structure object.
    i: int
        The index of core element.
    ep_types: str
        The allowed element pair presents in the symmetry function calculation.
    ni: array of neighbors information
        Neighbors information of the core element.
    functional: str
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
    p: int
        The atom that the force is acting on.
    q: int
        Direction of force.

    Returns
    -------
    G5p: float
        The derivative of G5 symmetry function.
    """
    if functional == 'Cosine':
        func = Cosine(Rc=Rc)
    elif functional == 'Polynomial':
        func = Polynomial(Rc=Rc)
    elif functional == 'TangentH':
        func = TangentH(Rc=Rc)
    else:
        raise NotImplementedError('Unknown cutoff functional: %s' %functional)
        
    # Get positions of core atoms
    Ri = crystal.cart_coords[i]
    
    counts = range(len(ni))
    
    G5p = 0
    for j in counts:
        for k in counts[(j+1):]:
            n1 = ni[j][0].species_string
            n2 = ni[k][0].species_string
            if (ep_type[0] == n1 and ep_type[1] == n2) or \
                (ep_type[1] == n1 and ep_type[0] == n2):
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

                lamBda_term = 1. + lamBda * cos_ijk
                
                first_term = -2 * eta / Rc ** 2 * lamBda_term * \
                                (Rij * dRab_dRpq(i, ni[j][2], Ri, Rj, p, q) + 
                                 Rik * dRab_dRpq(i, ni[k][2], Ri, Rk, p, q))
                first_term += lamBda * zeta * dcos_ijk
                first_term *= cutoff
                
                second_term = lamBda_term * \
                                (cutoff_Rij_derivative * func(Rik) + 
                                 cutoff_Rik_derivative * func(Rij))
                                
                term = first_term + second_term
                term *= lamBda_term ** (zeta - 1.)
                term *= np.exp(-eta * (Rij ** 2. + Rik ** 2.) /
                               Rc ** 2.)
                                
                G5p += term

    G5p *= 2. ** (1. - zeta)
        
    return G5p


#from pymatgen.core.structure import Structure

#crystal = Structure.from_file('../datasets/POSCARs/POSCAR-NaCl')

#sym_params = {'G2': {'eta': np.logspace(np.log10(0.05), 
#                                         np.log10(5.), num=4),
#                    'Rc': 6.5}}
#                'G5': {'eta': [0.005],
#                        'zeta': [1., 4.],
#                        'lamBda': [1., -1.]}}

#gauss = symf(crystal, sym_params, derivative=False)
#print(gauss.G2p)


#    def get_statistics(self, Gs):
#        Gs_mean = np.mean(Gs, axis=0)
#        Gs_std = np.std(Gs, axis=0)
#        Gs_skew = skew(Gs, axis=0)
#        Gs_kurtosis = kurtosis(Gs, axis=0)
#        Gs_covariance = np.cov(Gs.T)
#
#        return None

#        if G_type == 'G1':
#            for rc in Rc:
#                for co in cutoff_f:
#                    g = calculate_G1(self.crystal, co, rc)
#                    G.append(g)
