from __future__ import division
import numpy as np
import numba as nb
from ase.neighborlist import NeighborList
from optparse import OptionParser
from copy import deepcopy
from numba import prange, cuda
from pyxtal_ff.descriptors.angular_momentum import Wigner_D

class SOAP:
    '''
    A class to generate the SO3 power spectrum components
    based off of the Gaussian atomic neighbor density function
    defined in "On Representing Atomic Environments".

    args:
        nmax: int, degree of radial expansion
        lmax: int, degree of spherical harmonic expansion
        rcut: float, cutoff radius for neighbor calculation
        alpha: float, gaussian width parameter
        derivative: bool, whether to calculate the gradient of not
    '''

    def __init__(self, nmax, lmax, rcut, alpha=2.0, derivative=True, stress=False):
        # populate attributes
        self.nmax = nmax
        self.lmax = lmax
        self.rcut = rcut
        self.alpha = alpha
        self.derivative = derivative
        self.stress = stress
        return

    def __repr__(self):
        return """SO3 SOAP power spectrum calculator"""

    @property
    def nmax(self):
        return self._nmax

    @nmax.setter
    def nmax(self, nmax):
        if isinstance(nmax, int) is True:
            if nmax < 1:
                raise ValueError('nmax must be greater than or equal to 1')
            if nmax > 11:
                raise ValueError('nmax > 11 yields complex eigen values which will mess up the calculation')
            self._nmax = nmax
        else:
            raise ValueError('nmax must be an integer')

    @property
    def lmax(self):
        return self._lmax

    @lmax.setter
    def lmax(self, lmax):
        if isinstance(lmax, int) is True:
            if lmax < 0:
                raise ValueError('lmax must be greater than or equal to zero')
            elif lmax > 32:
                raise NotImplementedError('''Currently we only support Wigner-D matrices and spherical harmonics
                                          for arguments up to l=32.  If you need higher functionality, raise an issue
                                          in our Github and we will expand the set of supported functions''')
            self._lmax = lmax
        else:
            raise ValueError('lmax must be an integer')

    @property
    def rcut(self):
        return self._rcut

    @rcut.setter
    def rcut(self, rcut):
        if isinstance(rcut, float) is True or isinstance(rcut, int) is True:
            if rcut <= 0:
                raise ValueError('rcut must be greater than zero')
            self._rcut = rcut
        else:
            raise ValueError('rcut must be a float')

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if isinstance(alpha, float) is True or isinstance(alpha, int) is True:
            if alpha <= 0:
                raise ValueError('alpha must be greater than zero')
            self._alpha = alpha
        else:
            raise ValueError('alpha must be a float')

    @property
    def derivative(self):
        return self._derivative

    @derivative.setter
    def derivative(self, derivative):
        if isinstance(derivative, bool) is True:
            self._derivative = derivative
        else:
            raise ValueError('derivative must be a boolean value')

    @property
    def stress(self):
        return self._stress

    @stress.setter
    def stress(self, stress):
        if isinstance(stress, bool) is True:
            self._stress = stress
        else:
            raise ValueError('stress must be a boolean value')

    def clear_memory(self):
        '''
        Clears all memory that isn't an essential attribute for the calculator
        '''
        attrs = list(vars(self).keys())
        for attr in attrs:
            if attr not in {'_nmax', '_lmax', '_rcut', '_alpha', '_derivative', '_stress'}:
                delattr(self, attr)
        return

    def calculate(self, atoms, backend='ase'):
        '''
        Calculates the SO(3) power spectrum components of the
        smoothened atomic neighbor density function
        for given nmax, lmax, rcut, and alpha.

        Args:
            atoms: an ASE atoms object corresponding to the desired
                   atomic arrangement

            backend: string, specifies the method to compute the neighborlist
                     elements, either ASE or pymatgen
        '''
        self._atoms = atoms
        self._backend = backend

        self.build_neighbor_list()
        self.initialize_arrays()

        get_power_spectrum_components(self.center_atoms, self.neighborlist, self.neighbor_indices,
                                      self.atomic_numbers, self.nmax, self.lmax,
                                      self.rcut, self.alpha, self.derivative,
                                      self.stress, self._plist, self._dplist, self._pstress)
        vol = atoms.get_volume()

        if self.derivative:

            x = {'x':self._plist.real, 'dxdr':self._dplist.real, 'elements':list(atoms.symbols)}

            if self._stress:
                x['rdxdr'] = self._pstress.real/vol
            else:
                x['rdxdr'] = None

        else:
            x = {'x':self._plist.real, 'elements':list(atoms.symbols)}

        self.clear_memory()
        return x

    def initialize_arrays(self):
        # number of atoms in periodic arrangement
        # for a crystal this will be the number of
        # atoms in the unit cell
        # for a cluster/molecule(s) this will be the total number
        # of atoms
        ncell = len(self._atoms)
        # degree of spherical harmonic expansion
        lmax = self.lmax
        # degree of radial expansion
        nmax = self.nmax
        # number of unique power spectrum components
        # this is given by the triangular elements of
        # the radial expansion multiplied by the degree
        # of spherical harmonic expansion (including 0)
        ncoefs = nmax*(nmax+1)//2*(lmax+1)

        # to include GPU support we will need to define more arrays here
        # as no array creation methods are supported in the numba cuda package
        self._plist = np.zeros((ncell, ncoefs), dtype=np.complex128)
        self._dplist = np.zeros((ncell, ncell, ncoefs, 3), dtype=np.complex128)
        self._pstress = np.zeros((ncell, ncell, ncoefs, 3, 3), dtype=np.complex128)

        return

    def build_neighbor_list(self):
        '''
        Builds a neighborlist for the calculation of bispectrum components for
        a given ASE atoms object given in the calculate method.
        '''
        if self._backend == 'ase':
            atoms = self._atoms
            # cutoffs for each atom
            cutoffs = [self.rcut/2]*len(atoms)
            # instantiate neighborlist calculator
            nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
            # provide atoms object to neighborlist calculator
            nl.update(atoms)
            # instantiate memory for neighbor separation vectors, periodic indices, and atomic numbers
            center_atoms = np.zeros((len(atoms), 3), dtype=np.float64)
            neighbors = []
            neighbor_indices = []
            atomic_numbers = []

            max_len = 0
            for i in range(len(atoms)):
                # get center atom position
                center_atom = atoms.positions[i]
                center_atoms[i] = center_atom
                # get indices and cell offsets of each neighbor
                indices, offsets = nl.get_neighbors(i)
                # add an empty list to neighbors and atomic numbers for population
                neighbors.append([])
                atomic_numbers.append([])
                # the indices are already numpy arrays so just append as is
                neighbor_indices.append(indices)
                for j, offset in zip(indices, offsets):
                    # compute separation vector
                    pos = atoms.positions[j] + np.dot(offset, atoms.get_cell()) - center_atom
                    neighbors[i].append(pos)
                    atomic_numbers[i].append(atoms[j].number)

                if len(neighbors[i]) > max_len:
                    max_len = len(neighbors[i])

            # declare arrays to store the separation vectors, neighbor indices
            # atomic numbers of each neighbor, and the atomic numbers of each
            # site
            neighborlist = np.zeros((len(atoms), max_len, 3), dtype=np.float64)
            neighbor_inds = np.zeros((len(atoms), max_len), dtype=np.int64)
            atm_nums = np.zeros((len(atoms), max_len), dtype=np.int64)
            site_atomic_numbers = np.array(list(atoms.numbers), dtype=np.int64)

            # populate the arrays with list elements
            for i in range(len(atoms)):
                neighborlist[i, :len(neighbors[i]), :] = neighbors[i]
                neighbor_inds[i, :len(neighbors[i])] = neighbor_indices[i]
                atm_nums[i, :len(neighbors[i])] = atomic_numbers[i]


        elif self._backend == 'pymatgen':
            from pymatgen.io.ase import AseAtomsAdaptor
            struc = AseAtomsAdaptor.get_structure(self._atoms)
            neighbors = struc.get_all_neighbors(self._rcut, include_index=True)

            max_len = 0
            for i, neighlist in enumerate(neighbors):
                if len(neighlist) > max_len:
                    max_len = len(neighlist)

            center_atoms = np.zeros((len(struc), 3), dtype=np.float64)
            neighborlist = np.zeros((len(struc), max_len, 3), dtype=np.float64)
            neighbor_inds = np.zeros((len(struc), max_len), dtype=np.int64)
            atm_nums = np.zeros((len(struc), max_len), dtype=np.int64)
            site_atomic_numbers = np.zeros(len(struc), dtype=np.int64)

            for i, site in enumerate(struc):
                neighlist = neighbors[i]
                site_atomic_numbers[i] = site.specie.number
                center_atoms[i] = site.coords
                for j, neighbor in enumerate(neighlist):
                    neighborlist[i, j, :] = neighbor[0].coords - site.coords
                    neighbor_inds[i, j] = neighbor[2]
                    atm_nums[i, j] = neighbor[0].specie.number

        else: raise NotImplementedError('Specified backend not supported')

        # assign these arrays to attributes
        self.center_atoms = center_atoms
        self.neighborlist = neighborlist
        self.neighbor_indices = neighbor_inds
        self.atomic_numbers = atm_nums
        self.site_atomic_numbers = site_atomic_numbers

        return

@nb.njit(nb.c16(nb.c16, nb.c16, nb.i8, nb.i8), cache=True,
         fastmath=True, nogil=True)
def sph_harm(Ra, Rb, l, m):
    '''
    Spherical harmonics from Wigner-D functions

    args:
        Ra: complex, Cayley-Klein parameter for spherical harmonic
        Rb: complex, Cayley-Klein parameter for spherical harmonic
        l: int, index of spherical harmonic l >= 0
        m: int, index of spherical harmonic -l <= m <= l

    The spherical harmonics are a subset of Wigner-D matrices,
    and can be calculated in the same manner
    '''
    if m%2 == 0:
        return Wigner_D(Ra, Rb, 2*l, 0, -2*m).conjugate() * np.sqrt((2*l+1)/4/np.pi) * (-1.0)**(m)
    else:
        return Wigner_D(Ra, Rb, 2*l, 0, -2*m).conjugate() * np.sqrt((2*l+1)/4/np.pi) * (-1.0)**(m+1)

@nb.njit(nb.f8(nb.f8, nb.i8, nb.b1), cache=True,
         fastmath=True, nogil=True)
def modifiedSphericalBessel1(r, n, derivative):
    '''
    Modified spherical bessel functions of the first kind
    with and without first derivative

    We don't have to be careful of the singularity at x=0 as
    the Chebyshev quadrature used will never include the point 0

    To include GPU support here, we will need to shift to temp
    variables
    '''
    if derivative == False:
        if n == 0:
            return np.sinh(r)/r
        elif n == 1:
            return (r*np.cosh(r)-np.sinh(r))/r**2
        else:
            temp_arr = np.zeros((n+1), np.float64)
            temp_arr[0] = np.sinh(r)/r
            temp_arr[1] = (r*np.cosh(r)-np.sinh(r))/r**2
            for i in range(2, n+1, 1):
                temp_arr[i] = temp_arr[i-2] - (2*i-1)/r*temp_arr[i-1]
            return temp_arr[n]
    else:
        if n == 0:
            # the derivative if i0 is i1
            return (r*np.cosh(r)-np.sinh(r))/r**2

        else:
            temp_arr = np.zeros((n+2), np.float64)
            temp_arr[0] = np.sinh(r)/r
            temp_arr[1] = (r*np.cosh(r)-np.sinh(r))/r**2
            for i in range(2, n+2, 1):
                temp_arr[i] = temp_arr[i-2] - (2*i-1)/r*temp_arr[i-1]
            return (n*temp_arr[n-1] + (n+1)*temp_arr[n+1]) / (2*n+1)

'''
These next three functions are for the evaluation of the orthonormal polynomial basis on [0,rcut]
proposed in on representing chemical environments
'''
@nb.njit(nb.void(nb.i8, nb.f8[:,:]), cache=True, nogil=True, fastmath=True)
def W(nmax, arr):
    '''
    Constructs the matrix of linear combination coefficients
    from the overlap matrix S for the polynomial basis g(r)
    defined below and in On Representing Chemical Environments.
    This normalizes this basis on the interval [0, rcut]

    W = S^(-1/2)
    '''
    # first construct the overlap matrix S
    for alpha in range(1, nmax+1, 1):
        temp1 = (2*alpha+5)*(2*alpha+6)*(2*alpha+7)
        for beta in range(1, alpha+1, 1):
            temp2 = (2*beta+5)*(2*beta+6)*(2*beta+7)
            arr[alpha-1, beta-1] = np.sqrt(temp1*temp2)/(5+alpha+beta)/(6+alpha+beta)/(7+alpha+beta)
            arr[beta-1, alpha-1] = arr[alpha-1, beta-1]

    sinv = np.linalg.inv(arr)
    eigvals, V = np.linalg.eig(sinv)
    sqrtD = np.diag(np.sqrt(eigvals))
    arr[:,:] = np.dot(np.dot(V, sqrtD), np.linalg.inv(V))
    return

@nb.njit(nb.f8(nb.f8, nb.i8, nb.f8), cache=True, nogil=True, fastmath=True)
def phi(r, alpha, rcut):
    '''
    See g below
    '''
    return (rcut-r)**(alpha+2)/np.sqrt(2*rcut**(2*alpha+7)/(2*alpha+5)/(2*alpha+6)/(2*alpha+7))

@nb.njit(nb.c16(nb.f8, nb.i8, nb.i8, nb.f8, nb.f8[:,:]), cache=True,
         nogil=True, fastmath=True)
def g(r, n, nmax, rcut, w):
    '''
    Evaluate the radial basis at a given r, given the overlap matrix
    for the maximal n.
    '''
    Sum = 0.0 + 0.0j
    for alpha in range(1, nmax+1):
        Sum += w[n-1, alpha-1]*phi(r, alpha, rcut)

    return Sum

@nb.njit(nb.c16(nb.f8, nb.f8, nb.f8, nb.f8, nb.i8, nb.i8, nb.i8, nb.f8[:,:], nb.b1),
         cache=True, fastmath=True, nogil=True)
def integrand(r, ri, alpha, rcut, n, l, nmax, w, derivative):
    '''
    The integrand of the radial inner product as in
    *cite our paper later*
    '''
    if derivative == False:
        return r**2*g(r, n, nmax, rcut, w)*np.exp(-alpha*r**2)*modifiedSphericalBessel1(2*alpha*r*ri, l, False)
    else:
        return r**3*g(r, n, nmax, rcut, w)*np.exp(-alpha*r**2)*modifiedSphericalBessel1(2*alpha*r*ri, l, True)

@nb.njit(nb.c16(nb.f8, nb.f8, nb.f8, nb.i8, nb.i8, nb.i8, nb.f8[:,:], nb.b1),
         cache=True, fastmath=True, nogil=True)
def get_radial_inner_product(ri, alpha, rcut, n, l, nmax, w, derivative):
    '''
    Chebyshev-Gauss quadrature integral calculator
    for the radial inner product as in *cite our paper later*
    '''
    integral = 0.0
    # heuristic rule for how many points to include in the quadrature
    N = (n+l+1)*10
    for i in range(1, N+1, 1):
        # roots of Chebyshev polynomial of degree N
        x = np.cos((2*i-1)*np.pi/2/N)
        # transforming the root from the interval [-1,1] to the interval [0, rcut]
        xi = rcut/2*(x+1)
        integral += np.sqrt(1-x**2)*integrand(xi, ri, alpha, rcut, n, l, nmax, w, derivative)
    # the weight (pi/N) is uniform for Chebyshev-Gauss quadrature
    integral *= rcut/2*np.pi/N
    return integral

@nb.njit(nb.void(nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.i8, nb.i8, nb.f8[:,:], nb.c16[:,:]),
         cache=True, fastmath=True, nogil=True)
def compute_carray(x, y, z, ri, alpha, rcut, nmax, lmax, w, clist):
    '''
    Get expansion coefficient for one neighbor.  Then add
    to the whole expansion coefficient
    '''

    # construct Cayley-Klein parameters of the unit quaternion
    # for one neighbor for spherical harmonic
    # calculation

    # get spherical coordinates from cartesian
    theta = np.arccos(z/ri)
    phi = np.arctan2(y,x)

    # construct Cayley-Klein parameters for rotation
    # about initial y axis
    atheta = np.cos(theta/2)
    btheta = np.sin(theta/2)

    # construct Cayley-Klein parameters for rotation
    # about new z axis
    # note that bphi = 0 for rotations about z
    aphi = np.cos(phi/2) + 1j*np.sin(phi/2)

    # compose the rotations
    Ra = atheta*aphi
    Rb = btheta*aphi


    # gaussian factor for this neighbor for the inner product
    expfac = 4*np.pi*np.exp(-alpha*ri**2)

    for n in range(1, nmax+1, 1):
        i = 0
        for l in range(0, lmax+1 ,1):
            # the radial portion of this inner product cannot be calculated
            # analytically, hence we calculate the inner product numerically
            r_int = get_radial_inner_product(ri, alpha, rcut, n, l, nmax, w, False)
            for m in range(-l, l+1, 1):
                Ylm = sph_harm(Ra, Rb, l, m)
                clist[n-1, i] += r_int*Ylm*expfac
                i += 1
    return

@nb.njit(nb.void(nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.i8, nb.i8, nb.f8[:,:], nb.c16[:,:], nb.c16[:,:,:]),
         cache=True, fastmath=True, nogil=True)
def compute_carray_wD(x, y, z, ri, alpha, rcut, nmax, lmax, w, clist, dclist):
    '''
    Get expansion coefficient for one neighbor.  Then add
    to the whole expansion coefficient
    '''

    # keep x,y,z instead of array for GPU support
    rvec = np.array((x,y,z), dtype=np.float64)

    # construct Cayley-Klein parameters of the unit quaternion
    # for one neighbor for spherical harmonic
    # calculation

    # get spherical coordinates from cartesian
    theta = np.arccos(z/ri)
    phi = np.arctan2(y,x)

    # construct Cayley-Klein parameters for rotation
    # about initial y axis
    atheta = np.cos(theta/2)
    btheta = np.sin(theta/2)

    # construct Cayley-Klein parameters for rotation
    # about new z axis
    # note that bphi = 0 for rotations about z
    aphi = np.cos(phi/2) + 1j*np.sin(phi/2)

    # compose the rotations
    Ra = atheta*aphi
    Rb = btheta*aphi

    Ylms = np.zeros((lmax+2)**2, dtype=np.complex128)

    # get spherical harmonics up to l+1
    i = 0
    for l in range(0, lmax+2, 1):
        for m in range(-l, l+1, 1):
            Ylms[i] = sph_harm(Ra, Rb, l, m)
            i += 1

    dYlm = np.zeros(((lmax+1)**2,3), dtype=np.complex128)
    # get gradient of spherical harmonics using relationship
    # to covariant spherical coordinates
    # this avoids singularities at the poles that exist
    # when using polar coordinates or cartesian coordinates
    i = 1
    # start i at 1 as the gradient of Y00 = 0

    #NOTE: for GPU support get rid of the array and calculate each
    # spherical harmonic on the fly as the if conditions are built into
    # the spherical harmonic calculation
    # Also just use the dC array to store the initial dYlms
    for l in range(1, lmax+1, 1):
        ellpl1 = np.sum(np.arange(0,l+2,1)*2)
        ellm1 = np.sum(np.arange(0,l,1)*2)
        for m in range(-l, l+1, 1):
            # get indices of l+1 and l-1 spherical harmonics for m = 0

            # get the gradient of spherical harmonics with respect to
            # covariant spherical coordinates VMK 5.8.3
            xcov0 = -np.sqrt(((l+1)**2-m**2)/(2*l+1)/(2*l+3))*l*Ylms[ellpl1+m]/ri
            if abs(m) <= l-1:
                xcov0 += np.sqrt((l**2-m**2)/(2*l-1)/(2*l+1))*(l+1)*Ylms[ellm1+m]/ri

            xcovpl1 = -np.sqrt((l+m+1)*(l+m+2)/2/(2*l+1)/(2*l+3))*l*Ylms[ellpl1+m+1]/ri
            if abs(m+1) <= l-1:
                xcovpl1 -= np.sqrt((l-m-1)*(l-m)/2/(2*l-1)/(2*l+1))*(l+1)*Ylms[ellm1+m+1]/ri

            xcovm1 = -np.sqrt((l-m+1)*(l-m+2)/2/(2*l+1)/(2*l+3))*l*Ylms[ellpl1+m-1]/ri
            if abs(m-1) <= l-1:
                xcovm1 -= np.sqrt((l+m-1)*(l+m)/2/(2*l-1)/(2*l+1))*(l+1)*Ylms[ellm1+m-1]/ri

            #transform the gradient to cartesian
            dYlm[i,0] = 1/np.sqrt(2)*(xcovm1-xcovpl1)
            dYlm[i,1] = 1j/np.sqrt(2)*(xcovm1+xcovpl1)
            dYlm[i,2] = xcov0
            i += 1

    # gaussian factor for this neighbor for the inner product
    expfac = 4*np.pi*np.exp(-alpha*ri**2)
    dexpfac = -2*alpha*expfac*rvec

    for n in range(1, nmax+1, 1):
        i = 0
        for l in range(0, lmax+1 ,1):
            # the radial portion of this inner product cannot be calculated
            # analytically, hence we calculate the inner product numerically
            r_int = get_radial_inner_product(ri, alpha, rcut, n, l, nmax, w, False)
            dr_int = get_radial_inner_product(ri, alpha, rcut, n, l, nmax, w, True)*2*alpha*rvec/ri
            for m in range(-l, l+1, 1):
                clist[n-1, i] += r_int*Ylms[i]*expfac
                dclist[n-1,i,:] += r_int*Ylms[i]*dexpfac + dr_int*Ylms[i]*expfac + r_int*expfac*dYlm[i,:]
                i += 1
    return

@nb.njit(nb.void(nb.c16[:,:], nb.c16[:,:]), cache=True,
         fastmath=True, nogil=True)
def add_carraytot(clisttot, clist):
    '''
    Add the expansion coefficient array for
    one neighbor to the total
    '''
    clisttot[:,:] += clist[:,:]
    return

@nb.njit(nb.void(nb.i8, nb.i8, nb.c16[:,:], nb.c16[:]), cache=True,
         fastmath=True, nogil=True)
def compute_pi(nmax, lmax, clisttot, plist):
    '''
    Compute the power spectrum components by p(n1,n2,l) by summing over l
    There is a symmetry for interchanging n1 and n2 so we only take the unique
    elements of the power spectrum.
    '''
    i = 0
    for n1 in range(0, nmax, 1):
        for n2 in range(0, n1+1, 1):
            j = 0
            for l in range(0, lmax+1, 1):
                # normalization factor in erratum
                norm = 2*np.sqrt(2)*np.pi/np.sqrt(2*l+1)
                for m in range(-l, l+1, 1):
                    plist[i] += clisttot[n1, j] * clisttot[n2, j].conjugate()*norm
                    j += 1
                i += 1
    return

@nb.njit(nb.void(nb.i8, nb.i8, nb.c16[:,:], nb.c16[:,:,:], nb.c16[:,:]), cache=True,
         fastmath=True, nogil=True)
def compute_dpidrj(nmax, lmax, clisttot, dclist, dplist):
    '''
    Compute the power spectrum components by p(n1,n2,l) by summing over l
    There is a symmetry for interchanging n1 and n2 so we only take the unique
    elements of the power spectrum.
    '''
    i = 0
    for n1 in range(0, nmax, 1):
        for n2 in range(0, n1+1, 1):
            j = 0
            for l in range(0, lmax+1, 1):
                # normalization factor in erratum
                norm = 2*np.sqrt(2)*np.pi/np.sqrt(2*l+1)
                for m in range(-l, l+1, 1):
                    temp = dclist[n1, j] * clisttot[n2, j].conjugate()
                    temp += clisttot[n1, j] * np.conj(dclist[n2, j])
                    temp *= norm
                    dplist[i,:] += temp

                    j += 1
                i += 1
    return

@nb.njit(nb.void(nb.c16[:,:]), cache=True, fastmath=True, nogil=True)
def zero_2D_array(arr):
    '''
    zeros an arbitrary 2D array
    '''
    for i in range(0, arr.shape[0], 1):
        for j in range(0, arr.shape[1], 1):
            arr[i,j] = 0.0 + 0.0j
    return

@nb.njit(nb.void(nb.c16[:,:,:,:]), cache=True, fastmath=True, nogil=True)
def zero_4D_array(arr):
    '''
    zeros an arbitrary 2D array
    '''
    for i in range(0, arr.shape[0], 1):
        for j in range(0, arr.shape[1], 1):
            for k in range(0, arr.shape[2], 1):
                for l in range(0, arr.shape[3], 1):
                    arr[i,j,k,l] = 0.0 + 0.0j
    return

@nb.njit(nb.void(nb.f8[:,:], nb.f8[:,:,:], nb.i8[:,:], nb.i8[:,:], nb.i8, nb.i8, nb.f8, nb.f8, nb.b1, nb.b1, nb.c16[:,:], nb.c16[:,:,:,:], nb.c16[:,:,:,:,:]),
         cache=True, fastmath=True, nogil=True)
def get_power_spectrum_components(center_atoms, neighborlist, neighbor_indices, neighbor_ANs, nmax, lmax, rcut, alpha, derivative, stress, plist, dplist, pstress):
    '''
    Interface to SOAP class, this is the main work function for the power spectrum calculation.
    '''
    # get the number of sites, number of neighbors, and number of spherical harmonics
    nsites = neighborlist.shape[0]
    nneighbors = neighborlist.shape[1]
    numYlms = (lmax+1)**2

    # allocate array memory for the total inner product
    clisttot = np.zeros((nmax, numYlms), dtype=np.complex128)
    # the inner product for one neighbor
    clist = np.zeros((nmax, numYlms), dtype=np.complex128)
    # the gradient of the inner product with respect to one neighbor
    dclist = np.zeros((nneighbors, nmax, numYlms, 3), dtype=np.complex128)

    # get the overlap matrix for n max
    w = np.zeros((nmax, nmax), np.float64)
    W(nmax, w)

    if derivative == True:
        if stress == True:
            numps = nmax*(nmax+1)*(lmax+1)//2
            tempdp = np.zeros((numps, 3), dtype=np.complex128)
            Rj = np.zeros(3, dtype=np.float64)
            for site in range(nsites):
                zero_2D_array(clisttot)
                zero_4D_array(dclist)
                for neighbor in prange(nneighbors):
                    x = neighborlist[site, neighbor, 0]
                    y = neighborlist[site, neighbor, 1]
                    z = neighborlist[site, neighbor, 2]
                    r = np.sqrt(x*x + y*y + z*z)
                    if r < 10**(-8):
                        continue
                    zero_2D_array(clist)

                    compute_carray_wD(x, y, z, r, alpha, rcut, nmax, lmax, w, clist, dclist[neighbor])

                    weight = neighbor_ANs[site, neighbor]
                    clist *= weight
                    dclist[neighbor] *= weight

                    add_carraytot(clisttot, clist)

                compute_pi(nmax, lmax, clisttot, plist[site])
                Ri = center_atoms[site]
                for neighbor in prange(nneighbors):
                    zero_2D_array(tempdp)
                    compute_dpidrj(nmax, lmax, clisttot, dclist[neighbor],
                                   tempdp)

                    dplist[site, neighbor_indices[site, neighbor]] += tempdp

                    # get cartesian coordinates
                    Rj[0] = neighborlist[site, neighbor, 0] + Ri[0]
                    Rj[1] = neighborlist[site, neighbor, 1] + Ri[1]
                    Rj[2] = neighborlist[site, neighbor, 2] + Ri[2]

                    for k in range(numps):
                        pstress[site, site, k] += np.outer(Ri, tempdp[k])
                        pstress[site, neighbor_indices[site, neighbor], k] -= np.outer(Rj, tempdp[k])


            for i in range(nsites):
                for j in range(nsites):
                    if i != j:
                        dplist[i,i] -= dplist[i,j]
        else:
            for site in range(nsites):
                zero_2D_array(clisttot)
                zero_4D_array(dclist)
                for neighbor in prange(nneighbors):
                    x = neighborlist[site, neighbor, 0]
                    y = neighborlist[site, neighbor, 1]
                    z = neighborlist[site, neighbor, 2]
                    r = np.sqrt(x*x + y*y + z*z)
                    if r < 10**(-8):
                        continue
                    zero_2D_array(clist)

                    compute_carray_wD(x, y, z, r, alpha, rcut, nmax, lmax, w, clist, dclist[neighbor])

                    weight = neighbor_ANs[site, neighbor]
                    clist *= weight
                    dclist[neighbor] *= weight

                    add_carraytot(clisttot, clist)

                compute_pi(nmax, lmax, clisttot, plist[site])
                for neighbor in prange(nneighbors):
                    compute_dpidrj(nmax, lmax, clisttot, dclist[neighbor],
                                   dplist[site, neighbor_indices[site,neighbor]])

            for i in range(nsites):
                for j in range(nsites):
                    if i != j:
                        dplist[i,i] -= dplist[i,j]

    else:

        for site in range(nsites):
            zero_2D_array(clisttot)
            for neighbor in prange(nneighbors):
                x = neighborlist[site, neighbor, 0]
                y = neighborlist[site, neighbor, 1]
                z = neighborlist[site, neighbor, 2]
                r = np.sqrt(x*x + y*y + z*z)
                if r < 10**(-8):
                    continue
                zero_2D_array(clist)

                compute_carray(x, y, z, r, alpha, rcut, nmax, lmax, w, clist)

                weight = neighbor_ANs[site, neighbor]
                clist *= weight

                add_carraytot(clisttot, clist)

            compute_pi(nmax, lmax, clisttot, plist[site])
    return

if  __name__ == "__main__":
    from ase.io import read
    import time
    # ---------------------- Options ------------------------
    parser = OptionParser()
    parser.add_option("-c", "--crystal", dest="structure",
                      help="crystal from file, cif or poscar, REQUIRED",
                      metavar="crystal")

    parser.add_option("-r", "--rcut", dest="rcut", default=4.0, type=float,
                      help="cutoff for neighbor calcs, default: 2.0"
                      )

    parser.add_option("-l", "--lmax", dest="lmax", default=1, type=int,
                      help="lmax, default: 1"
                      )

    parser.add_option("-n", "--nmax", dest="nmax", default=1, type=int,
                      help="nmax, default: 1"
                      )

    parser.add_option("-a", "--alpha", dest="alpha", default=2.0, type=float,
                      help="cutoff for neighbor calcs, default: 2.0"
                      )

    parser.add_option("-s", dest="stress", default=True, 
                      action='store_true',help='derivative flag')

    parser.add_option("-f", dest="der", default=True,
                      action='store_false',help='derivative flag')

    (options, args) = parser.parse_args()

    if options.structure is None:
        from ase.build import bulk
        test = bulk('Si', 'diamond', a=5.459)
        cell = test.get_cell()
        cell[0,1] += 0.5
        test.set_cell(cell)
    else:
        test = read(options.structure, format='vasp')

    lmax = options.lmax
    nmax = options.nmax
    rcut = options.rcut
    alpha = options.alpha
    der = options.der
    stress = options.stress

    import time
    start1 = time.time()
    f = SOAP(nmax, lmax, rcut, alpha, derivative=der, stress=stress)
    x = f.calculate(test)
    start2 = time.time()
    '''
    for key, item in x.items():
        print(key, item)
        print('time elapsed: {}'.format(start2 - start1))
    '''

    print(x['rdxdr'].shape)
    print(x['rdxdr'])
    print(np.einsum('ijklm->klm', x['rdxdr']))
