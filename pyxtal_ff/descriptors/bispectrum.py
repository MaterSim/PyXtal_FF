from __future__ import division
import numpy as np
import numba as nb
from ase.neighborlist import NeighborList
from optparse import OptionParser
from copy import deepcopy
from pyxtal_ff.descriptors.angular_momentum import Wigner_D, factorial, deltacg, Wigner_D_wDerivative
from numba import prange

class SO4_Bispectrum:
    '''
    Pyxtal implementation of SO4 bispectrum component calculator.
    The difference between this implementation and the SNAP implementation
    lies exclusively with the choice of unit quaternion (removing singularities
    for rotations of 0 and 2pi) and the method of calculating Wigner-U functions

    here we use a polynomial form of the Wigner-D matrices to calculate the U-functions
    and thus the gradients can be calculated simultaneously through differentiating the
    U-functions using horner form
    '''

    def __init__(self, lmax, rcut, derivative=True, stress=False, normalize_U=False):
        # populate attributes
        self.lmax = lmax
        self.rcut = rcut
        self.derivative = derivative
        self.stress = stress
        self.normalize_U = normalize_U

    def __repr__(self):
        return """S04 bispectrum coefficient calculator for the
    atomic neighbor density function for a Wigner-D
    expansion of order {} and cutoff
    radius {}""".format(self.lmax, self.rcut)

    @property
    def lmax(self):
        return self._twol//2

    @lmax.setter
    def lmax(self, lmax):
        if isinstance(lmax, int) is True:
            if lmax < 0:
                raise ValueError('lmax must be greater than or equal to zero')
            elif lmax > 32:
                raise NotImplementedError('''Currently we only support Wigner-D matrices and spherical harmonics
                                          for arguments up to l=32.  If you need higher functionality, raise an issue
                                          in our github and we will expand the set of supported functions''')
            self._twol = 2*lmax
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

    @property
    def normalize_U(self):
        return self._norm

    @normalize_U.setter
    def normalize_U(self, normalize_U):
        if isinstance(normalize_U, bool) is True:
            self._norm = normalize_U
        else:
            raise ValueError('normalize_U must be a boolean value')

    def clear_memory(self):
        '''
        Clears all memory that isn't an essential attribute for the calculator
        '''
        attrs = list(vars(self).keys())
        for attr in attrs:
            if attr not in {'_twol', '_rcut', '_derivative', '_stress', '_norm'}:
                delattr(self, attr)
        return

    def calculate(self, atoms):
        '''
        args:
            atoms:  ASE atoms object for the corresponding structure

        returns: a dictionary with the bispectrum components, their
        gradients, and the elemental specie of each atom in the atoms
        object
        '''
        self._atoms = atoms
        vol = atoms.get_volume()
        self.build_neighbor_list()
        self.initialize_arrays()

        get_bispectrum_components(self.center_atoms,self.neighborlist, self.seq,
                                  self.atomic_numbers, self.site_atomic_numbers,
                                  self._twol, self.rcut, self._norm, self.derivative,
                                  self.stress, self._blist, self._dblist, self._bstress)

        if self.derivative is True:

            x = {'x':self._blist.real, 'dxdr':self._dblist.real,
                 'elements':list(atoms.symbols), 'seq':self.seq}

            if self.stress is True:
                x['rdxdr'] = self._bstress.real/vol
            else:
                x['rdxdr'] = None

        else:
            x = {'x':self._blist.real, 'dxdr': None, 'elements':list(atoms.symbols)}

            if self.stress is True:
                x['rdxdr'] = self._bstress.real/vol
            else:
                x['rdxdr'] = None

        self.clear_memory()
        return x

    def build_neighbor_list(self):
        '''
        Builds a neighborlist for the calculation of bispectrum components for
        a given ASE atoms object given in the calculate method.
        '''
        atoms = self._atoms
        cutoffs = [self.rcut/2]*len(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
        nl.update(atoms)
        center_atoms = np.zeros((len(atoms),3), dtype=np.float64)
        neighbors = []
        neighbor_indices = []
        atomic_numbers = []

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

        Neighbors = []
        Atomic_numbers = []
        Seq = []
        max_len = 0
        for i in range(len(atoms)):
            unique_atoms = np.unique(neighbor_indices[i])
            if i not in unique_atoms:
                at = list(unique_atoms)
                at.append(i)
                at.sort()
                unique_atoms = np.array(at)
            for j in unique_atoms:
                Seq.append([i,j])
                neigh_locs = neighbor_indices[i] == j
                temp_neighs = np.array(neighbors[i])
                temp_ANs = np.array(atomic_numbers[i])
                Neighbors.append(temp_neighs[neigh_locs])
                Atomic_numbers.append(temp_ANs[neigh_locs])
                length = sum(neigh_locs)
                if length > max_len:
                    max_len = length

        neighborlist = np.zeros((len(Neighbors), max_len, 3), dtype=np.float64)
        Seq = np.array(Seq, dtype=np.int64)
        atm_nums = np.zeros((len(Neighbors), max_len), dtype=np.int64)
        site_atomic_numbers = np.array(list(atoms.numbers), dtype=np.int64)


        for i in range(len(Neighbors)):
            neighborlist[i, :len(Neighbors[i]), :] = Neighbors[i]
            atm_nums[i, :len(Neighbors[i])] = Atomic_numbers[i]



        # assign these arrays to attributes
        self.center_atoms = center_atoms
        self.neighborlist = neighborlist
        self.seq = Seq
        self.atomic_numbers = atm_nums
        self.site_atomic_numbers = site_atomic_numbers

        return

    def initialize_arrays(self):
        '''
        Initialize the arrays to store the bispectrum components and their
        derivatives
        '''
        # number of atoms in periodic arrangement
        ncell = len(self._atoms)
        # degree of hyperspherical expansion
        lmax = self.lmax
        # number of bispectrum coefficents
        ncoefs = round(int((lmax+1)*(lmax+2)*(lmax+1.5)//3))
        # allocate memory for the bispectrum and its derivative
        self._blist = np.zeros([ncell, ncoefs], dtype=np.complex128)
        self._dblist = np.zeros([len(self.seq), ncoefs, 3], dtype=np.complex128)
        self._bstress = np.zeros([len(self.seq), ncoefs, 3, 3], dtype=np.complex128)
        return

@nb.njit(nb.void(nb.i8, nb.f8[:]), cache=True,
         fastmath=True, nogil=True)
def init_clebsch_gordan(twol, cglist):
    '''Populate an array ofpe Clebsch-Gordan coefficients needed
    for a bispectrum coefficient calculation

    Clebsch-Gordan coefficients arise in the coupling of
    angular momenta.  The method to calculate here is given in
    "Quantum Theory of Angular Momentum" D.A. Varshalovich. 8.2.1 (3)

    Parameters
    ----------
    twol:  integer
        Corresponds to the order of hyperspherical expansion
        in this software, the degree of expansion is defined and handled
        within the SO4_Bispectrum class.
    cglist:  1-D array for storing the coefficients.  The sizing of
        this array is handled in the SO4_Bispectrum class.

    Returns
    -------
    None
    '''
    ldim = twol + 1

    idxcg_count = 0
    for l1 in range(0, ldim, 1):
        for l2 in range(0, l1 + 1, 1):
            for l in range(l1 - l2, min(twol, l1 + l2) + 1, 2):
                for m1 in range(0, l1 + 1, 1):
                    aa2 = 2 * m1 - l1
                    for m2 in range(0, l2 + 1, 1):
                        bb2 = 2 * m2 - l2
                        m = (aa2 + bb2 + l) / 2

                        if (m < 0 or m > l):
                            cglist[idxcg_count] = 0.0
                            idxcg_count += 1
                            continue

                        Sum = 0.0

                        for z in range(max(0, max(-(l - l2 + aa2) // 2, -(l - l1 - bb2) // 2)), min(
                                (l1 + l2 - l) // 2, min((l1 - aa2) // 2, (l2 + bb2) // 2)) + 1, 1):

                            if z % 2 == 0:
                                ifac = 1
                            else:
                                ifac = -1

                            Sum += ifac / (factorial(z) *
                                           factorial((l1 + l2 - l) // 2 - z) *
                                           factorial((l1 - aa2) // 2 - z) *
                                           factorial((l2 + bb2) // 2 - z) *
                                           factorial((l - l2 + aa2) // 2 + z) *
                                           factorial((l - l1 - bb2) // 2 + z))

                        cc2 = 2 * m - l
                        dcg = deltacg(l1, l2, l)
                        sfaccg = np.sqrt(factorial((l1 + aa2) // 2) *
                                         factorial((l1 - aa2) // 2) *
                                         factorial((l2 + bb2) // 2) *
                                         factorial((l2 - bb2) // 2) *
                                         factorial((l + cc2) // 2) *
                                         factorial((l - cc2) // 2) *
                                         (l + 1))

                        cglist[idxcg_count] = Sum * dcg * sfaccg
                        idxcg_count += 1

    return

@nb.njit(nb.f8(nb.f8, nb.f8), cache=True, fastmath=True, nogil=True)
def compute_sfac(r, rcut):
    '''Calculates the cosine cutoff function value given in
    On Representing Chemical Environments, Batrok, et al.

    The cosine cutoff function ensures that the hyperspherical
    expansion for the calculation of bispectrum coefficients goes
    smoothly to zero for atomic neighbors tending to the cutoff
    radius.

    Parameters
    ----------
    r: float
        The magnitude of the separation vector from
        an atom in the unit cell to a particular neighbor
    rcut: float
        The cutoff radius specified in the SO4_Bispectrum class

    Returns
    -------
    cosine_cutoff:  float 0.5 * (cos(pi*r/rcut) + 1.0)
    '''
    if r > rcut:
        return 0
    else:
        rcutfac = np.pi / rcut
        return 0.5 * (np.cos(r * rcutfac) + 1.0)

@nb.njit(nb.f8(nb.f8, nb.f8), cache=True, fastmath=True, nogil=True)
def compute_dsfac(r, rcut):
    '''Calculates the derivative of the cosine cutoff for a given radii

    Parameters
    ----------
    r: float
        see compute_sfac
    rcut: float
        see compute_sfac

    Returns
    -------
    dcosine_cutoff/dr float -0.5*pi/rcut*sin(pi*r/rcut)
    '''
    if r > rcut:
        return 0

    else:
        rcutfac = np.pi / rcut
        return -0.5 * np.sin(r * rcutfac) * rcutfac

@nb.njit(nb.void(nb.i8, nb.i8[:], nb.c16[:,:]), cache=True,
         fastmath=True, nogil=True)
def addself_uarraytot(twol, idxu_block, ulisttot):
    '''Add the central atom contribution to the hyperspherical
    expansion coefficient array.

    This initializes the expansion coefficient array with a
    Kroenocker delta for ma and mb.

    Parameters
    ----------
    twol: integer
        Corresponds to the order of hyperspherical expansion
        in this software, the degree of expansion is defined
        and handled within the SO4 bispectrum class.

    idxu_block: 1-D integer array
        Each element corresponds to the first element of the
        expansion coefficient array ->(l,0,0) for each l.  This
        is handled in the SO4_Bispectrum class.

    ulisttot: 1-D complex array
        Array for storing hyperspherical expansion coefficients.
        The sizing and filling of this array is handled in the
        SO4_Bispectrum class.

    Returns
    -------
    None
    '''
    ldim = twol + 1
    for l in range(0, ldim, 1):
        llu = idxu_block[l]
        for ma in range(0, l + 1, 1):
            ulisttot[llu] = 1.0 + 0.0j
            llu += l + 2
    return

@nb.njit(nb.c16(nb.i8, nb.i8, nb.i8, nb.c16, nb.c16),
         cache=True, fastmath=True, nogil=True)
def U(l, mp, m, a, b):
    '''Computes the hyperspherical harmonic function
    value for given Cayley-Klein parameters and angular
    momentum numbers.  The hyperspherical harmonics are the
    elements of the Wigner-D matrices.

    This function is an interface to the Wigner-D matrix
    function in angular_momentum.py.  The indexing and
    generation of the Cayley-Klein parameters is handled in
    the compute_uarray functions, which are handled by
    the SO4 bispectrum class.

    Parameters
    ----------
        l:  integer
        positive integer parameter, corresponds to 2 times the
        orbital angular momentum quantum number.

        mp:  integer
        integer parameter, corresponds to the magnetic quantum
        number by the relation m' = mp - l/2

        m:  integer
        integer parameter, corresponds to the magnetic quantum
        number by the relation m = m - l/2

        a:  complex
        Cayley-Klein parameter of unit quaternion.  Corresponds
        to the diagonal elements in the corresponding element of
        SU(2) in the 2x2 matrix representation.

        b:  complex
        Cayley-Klein parameter of unit quaternion.  Corresponds
        to the off-diagonal elements in the corresponding element of
        SU(2) in the 2x2 matrix representation.


    Returns
    -------
    complex, element of the Wigner-D matrix, see
    the corresponding function in angular_momentum.py
    '''
    m = m - l/2
    mp = mp - l/2
    m = 2 * m
    mp = 2 * mp
    return Wigner_D(a, b, l, m, mp)

@nb.njit(nb.c16(nb.i8, nb.i8, nb.i8, nb.c16, nb.c16, nb.c16[:], nb.c16[:], nb.c16[:]),
         cache=True, fastmath=True, nogil=True)
def U_wD(l, mp, m, Ra, Rb, gradRa, gradRb, dU):
    '''Computes the hyperspherical harmonic function
    value for given Cayley-Klein parameters and angular
    momentum numbers.  The hyperspherical harmonics are the
    elements of the Wigner-D matrices.

    This function is an interface to the Wigner-D matrix
    function in angular_momentum.py.  The indexing and
    generation of the Cayley-Klein parameters is handled in
    the compute_uarray functions, which are handled by
    the SO4 bispectrum class.

    Parameters
    ----------
        l:  integer
        positive integer parameter, corresponds to 2 times the
        orbital angular momentum quantum number.

        mp:  integer
        integer parameter, corresponds to the magnetic quantum
        number by the relation m' = mp - l/2

        m:  integer
        integer parameter, corresponds to the magnetic quantum
        number by the relation m = m - l/2

        Ra:  complex
        Cayley-Klein parameter of unit quaternion.  Corresponds
        to the diagonal elements in the corresponding element of
        SU(2) in the 2x2 matrix representation.

        Rb:  complex
        Cayley-Klein parameter of unit quaternion.  Corresponds
        to the off-diagonal elements in the corresponding element of
        SU(2) in the 2x2 matrix representation.


    Returns
    -------
    complex, element of the Wigner-D matrix, see
    the corresponding function in angular_momentum.py
    '''
    m = m - l/2
    mp = mp - l/2
    m = 2 * m
    mp = 2 * mp
    return Wigner_D_wDerivative(Ra, Rb, l, m, mp, gradRa, gradRb, dU)

@nb.njit(nb.void(nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.i8, nb.c16[:,:], nb.i8[:]),
         cache=True, fastmath=True, nogil=True)
def compute_uarray_polynomial(x, y, z, psi, r, twol, ulist, idxu_block):
    '''Compute the Wigner-D matrix of order twol given an axis (x,y,z)
    and rotation angle 2*psi.

    This function constructs a unit quaternion representating a rotation
    of 2*psi through an axis defined by x,y,z; then populates an array of
    Wigner-D matrices of order twol for this rotation.  The Wigner-D matrices
    are calculated using a polynomial form.  See angular_momentum.Wigner_D for
    details.

    Parameters
    ----------
    x: float
    the x coordinate corresponding to the axis of rotation.

    y: float
    the y coordinate corresponding to the axis of rotation.

    z: float
    the z coordinate corresponding to the axis of rotation

    psi: float
    one half of the rotation angle

    r: float
    magnitude of the vector (x,y,z)

    twol: integer
    order of hyperspherical expansion

    ulist: 1-D complex array
    array to populate with D-matrix elements, mathematically
    this is a 3-D matrix, although we broadcast this to a 1-D
    matrix

    idxu_block: 1-D int array
    used to index ulist

    Returns
    -------
    None
    '''

    ldim = twol + 1

    # construct Cayley-Klein parameters for unit quaternion
    Ra = np.cos(psi) - 1j * np.sin(psi) / r * z
    Rb = np.sin(psi) / r * (y - x * 1j)

    # populate array with D matrix elements
    for l in range(0, ldim, 1):
        llu = idxu_block[l]
        for mb in range(0, l + 1, 1):
            for ma in range(0, l + 1, 1):
                ulist[llu,0] = U(l, mb, ma, Ra, Rb)
                llu += 1
    return

@nb.njit(nb.void(nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.i8, nb.c16[:,:], nb.c16[:,:], nb.i8[:]),
         cache=True, fastmath=True, nogil=True)
def compute_uarray_polynomial_wD(x, y, z, psi, r, twol, ulist, dulist, idxu_block):
    '''Compute the Wigner-D matrix of order twol given an axis (x,y,z)
    and rotation angle 2*psi.

    This function constructs a unit quaternion representating a rotation
    of 2*psi through an axis defined by x,y,z; then populates an array of
    Wigner-D matrices of order twol for this rotation.  The Wigner-D matrices
    are calculated using a polynomial form.  See angular_momentum.Wigner_D for
    details.

    Parameters
    ----------
    x: float
    the x coordinate corresponding to the axis of rotation.

    y: float
    the y coordinate corresponding to the axis of rotation.

    z: float
    the z coordinate corresponding to the axis of rotation

    psi: float
    one half of the rotation angle

    r: float
    magnitude of the vector (x,y,z)

    twol: integer
    order of hyperspherical expansion

    ulist: 1-D complex array
    array to populate with D-matrix elements, mathematically
    this is a 3-D matrix, although we broadcast this to a 1-D
    matrix

    idxu_block: 1-D int array
    used to index ulist

    Returns
    -------
    None
    '''

    ldim = twol + 1

    # construct Cayley-Klein parameters for unit quaternion
    cospsi = np.cos(psi)
    sinpsi = np.sin(psi)
    gradr = np.array((x,y,z), np.complex128)/r
    Ra = cospsi - 1j * sinpsi / r * z
    Rb = sinpsi / r * (y - x * 1j)

    gradRa = -sinpsi*psi/r*gradr - 1j*z*cospsi/r/r*psi*gradr + 1j*z*sinpsi/r/r*gradr
    gradRa[2] += -1j*sinpsi/r

    gradRb = cospsi/r/r*psi*(y-x*1j)*gradr - sinpsi/r/r*(y-x*1j)*gradr

    gradRb[1] += sinpsi/r
    gradRb[0] += -1j*sinpsi/r

    # populate array with D matrix elements
    for l in range(0, ldim, 1):
        llu = idxu_block[l]
        for mb in range(0, l + 1, 1):
            for ma in range(0, l + 1, 1):
                ulist[llu,0] = U_wD(l, mb, ma, Ra, Rb, gradRa, gradRb, dulist[llu,:])
                llu += 1

    return

@nb.njit(nb.void(nb.f8, nb.f8, nb.c16[:,:], nb.c16[:,:]),
         cache=True, fastmath=True, nogil=True)
def add_uarraytot(r, rcut, ulisttot, ulist):
    '''
    add the hyperspherical harmonic array for one neighbor to the
    expansion coefficient array
    '''
    sfac = compute_sfac(r, rcut)

    ulisttot += sfac * ulist
    return

@nb.njit(nb.void(nb.f8, nb.f8, nb.f8, nb.f8, nb. f8, nb.c16[:,:], nb.c16[:,:]),
         cache=True, fastmath=True, nogil=True)
def dudr(x, y, z, r, rcut, ulist, dulist):
    '''
    Compute the total derivative of the hyperspherical
    expansion coefficients.
    '''
    sfac = compute_sfac(r, rcut)
    gradr = np.zeros((len(ulist),3), np.complex128)
    gradr[:,0] = x
    gradr[:,1] = y
    gradr[:,2] = z
    gradr = gradr/r
    dsfac = compute_dsfac(r, rcut)*gradr

    dulist *= sfac
    dulist += ulist*dsfac
    return

@nb.njit(nb.void(nb.i8, nb.i8[:,:], nb.f8[:], nb.i8[:,:,:], nb.i8[:], nb.c16[:,:], nb.c16[:,:]),
         cache=True, fastmath=True, nogil=True)
def compute_zi(idxz_max, idxz, cglist, idxcg_block, idxu_block, ulisttot, zlist):
    '''
    Precompute the Kronoecker product of two rotated expansion coefficient tensors
    using Clebsch-Gordan expansion
    '''

    for llz in range(0, idxz_max, 1):
        l1 = idxz[llz, 0]
        l2 = idxz[llz, 1]
        l = idxz[llz, 2]
        ma1min = idxz[llz, 3]
        ma2max = idxz[llz, 4]
        na = idxz[llz, 5]
        mb1min = idxz[llz, 6]
        mb2max = idxz[llz, 7]
        nb = idxz[llz, 8]

        cgblock = cglist[idxcg_block[l1, l2, l]::]

        zlist[llz] = 0

        llu1 = idxu_block[l1] + (l1 + 1) * mb1min
        llu2 = idxu_block[l2] + (l2 + 1) * mb2max
        icgb = mb1min * (l2 + 1) + mb2max

        for ib in range(0, nb, 1):
            suma1 = 0

            u1 = ulisttot[llu1::]
            u2 = ulisttot[llu2::]

            ma1 = ma1min
            ma2 = ma2max
            icga = ma1min * (l2 + 1) + ma2max
            for ia in range(0, na, 1):
                suma1 += cgblock[icga] * u1[ma1,0] * u2[ma2,0]
                ma1 += 1
                ma2 -= 1
                icga += l2

            zlist[llz] += cgblock[icgb] * suma1

            llu1 += l1 + 1
            llu2 -= l2 + 1
            icgb += l2

    return


@nb.njit(nb.void(nb.i8, nb.i8[:,:], nb.i8[:,:,:], nb.i8[:], nb.c16[:,:], nb.c16[:,:], nb.c16[:]),
         cache=True, fastmath=True, nogil=True)
def compute_bi(ncoefs, idxb, idxz_block, idxu_block, ulisttot, zlist, blist):
    '''
    compute the bispectrum components from the Kronoecker product of the hermitian adjoint
    of the expansion coefficient array with the Z list (see compute Z_i for description)
    '''

    for llb in range(0, ncoefs, 1):
        l1 = idxb[llb, 0]
        l2 = idxb[llb, 1]
        l = idxb[llb, 2]

        llz = idxz_block[l1][l2][l]
        llu = idxu_block[l]

        sumzu = 0

        mb = 0
        while 2 * mb < l:
            for ma in range(0, l + 1, 1):
                sumzu += ulisttot[llu,0].conjugate() * zlist[llz,0]
                llz += 1
                llu += 1

            mb += 1

            # for l even handle middle column in a different manner

        if l % 2 == 0:
            mb = l / 2
            for ma in range(0, mb, 1):
                sumzu += ulisttot[llu,0].conjugate() * zlist[llz,0]
                llz += 1
                llu += 1

            sumzu += 0.5 * (ulisttot[llu,0].conjugate() * zlist[llz,0])

        blist[llb] = 2.0 * sumzu

    return

@nb.njit(nb.void(nb.i8, nb.i8[:,:], nb.i8[:,:,:], nb.i8[:], nb.c16[:,:], nb.c16[:,:], nb.c16[:,:]),
         cache=True, fastmath=True, nogil=True)
def compute_dbidrj(ncoefs, idxb, idxz_block, idxu_block, dulist, zlist, dblist):
    '''
    Compute the gradient of the bispectrum components
    '''

    for llb in range(0, ncoefs, 1):
        l1 = idxb[llb,0]
        l2 = idxb[llb,1]
        l = idxb[llb,2]

        llz = idxz_block[l1,l2,l]
        llu = idxu_block[l]

        sumzdu = np.zeros(3, np.complex128)

        mb = 0
        while 2*mb < l:
            for ma in range(0, l+1, 1):
                dudr = np.conj(dulist[llu])
                z = zlist[llz]
                sumzdu += dudr*z
                llu += 1
                llz += 1
            mb += 1

        if l%2 == 0:
            mb = l/2
            for ma in range(0, mb, 1):
                dudr = np.conj(dulist[llu])
                z = zlist[llz]
                sumzdu += dudr*z
                llu += 1
                llz += 1

            ma = mb
            dudr = np.conj(dulist[llu])
            z = zlist[llz]
            sumzdu += dudr*z*0.5
            llu += 1
            llz += 1

        dblist[llb] += 2.0*sumzdu

        l1fac = (l+1)/(l1+1.0)

        llz = idxz_block[l,l2,l1]
        llu = idxu_block[l1]

        sumzdu = np.zeros(3, np.complex128)

        mb = 0
        while 2*mb < l1:
            for ma in range(0, l1+1, 1):
                dudr = np.conj(dulist[llu])
                z = zlist[llz]
                sumzdu += dudr*z
                llu += 1
                llz += 1
            mb += 1

        if l1%2 == 0:
            mb = l1/2
            for ma in range(0, mb, 1):
                dudr = np.conj(dulist[llu])
                z = zlist[llz]
                sumzdu += dudr*z
                llu += 1
                llz += 1

            ma = mb
            dudr = np.conj(dulist[llu])
            z = zlist[llz]
            sumzdu += dudr*z*0.5
            llu += 1
            llz += 1

        dblist[llb] += 2.0*sumzdu*l1fac

        l2fac = (l+1)/(l2+1.0)

        llz = idxz_block[l,l1,l2]
        llu = idxu_block[l2]

        sumzdu = np.zeros(3, np.complex128)

        mb = 0
        while 2*mb < l2:
            for ma in range(0, l2+1, 1):
                dudr = np.conj(dulist[llu])
                z = zlist[llz]
                sumzdu += dudr*z
                llu += 1
                llz += 1
            mb += 1

        if l2%2 == 0:
            mb = l2/2
            for ma in range(0, mb, 1):
                dudr = np.conj(dulist[llu])
                z = zlist[llz]
                sumzdu += dudr*z
                llu += 1
                llz += 1

            ma = mb
            dudr = np.conj(dulist[llu])
            z = zlist[llz]
            sumzdu += dudr*z*0.5
            llu += 1
            llz += 1


        dblist[llb] += 2.0*sumzdu*l2fac

@nb.njit(nb.void(nb.c16[:,:]),
         cache=True, fastmath=True, nogil=True)
def zero_1d(arr):
    # zero a generic 1-d array
    for i in range(arr.shape[0]):
        arr[i] = 0
    return

@nb.njit(nb.void(nb.c16[:,:]),
         cache=True, fastmath=True, nogil=True)
def zero_2d(arr):
    # zero a generic 2d array
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i,j] = 0
    return

@nb.njit(nb.void(nb.c16[:,:,:]),
         cache=True, fastmath=True, nogil=True)
def zero_3d(arr):
    # zero a generic 3d array
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                arr[i,j,k] = 0
    return

@nb.njit(nb.void(nb.f8[:,:], nb.f8[:,:,:], nb.i8[:,:], nb.i8[:,:], nb.i8[:], nb.i8, nb.f8,
                 nb.b1, nb.b1, nb.b1, nb.c16[:,:], nb.c16[:,:,:], nb.c16[:,:,:,:]),
         cache=True, fastmath=True, nogil=True)
def get_bispectrum_components(center_atoms, neighborlist, seq, neighbor_ANs, site_ANs,
                              twolmax, rcut, norm, derivative, stress, blist, dblist, bstress):
    '''
    Calculate the bispectrum components, and their derivatives (if specified)
    for a given neighbor list.  This is the main work function.
    '''
    # build index lists, normalization array, and CG array
    ldim = twolmax + 1

    # array for indexing Clebsch-Gordan coefficients
    idxcg_block = np.zeros((ldim, ldim, ldim), dtype=np.int64)

    # variable to count the number of CG coefficients for a given twol
    idxcg_count = 0
    # populate the index list and count the number of CG coefficients
    for l1 in range(0, ldim, 1):
        for l2 in range(0, l1 + 1, 1):
            for l in range(l1 - l2, min(twolmax, l1 + l2) + 1, 2):
                idxcg_block[l1, l2, l] = idxcg_count
                for m1 in range(0, l1 + 1, 1):
                    for m2 in range(0, l2 + 1, 1):
                        idxcg_count += 1

    # allocate array for Clebsch-Gordan coefficients
    # then populate array
    cglist = np.zeros(idxcg_count, dtype=np.float64)
    init_clebsch_gordan(twolmax, cglist)

    # index list for u array
    idxu_block = np.zeros(ldim, dtype=np.int64)

    # populate the index list for u arrays and count
    # the number of u arrays
    idxu_count = 0
    for l in range(0, ldim, 1):
        idxu_block[l] = idxu_count
        for mb in range(0, l + 1, 1):
            for ma in range(0, l + 1, 1):
                idxu_count += 1

    # populate array for normalization of Wigner-D functions
    u_norm  = np.ones((idxu_count, 1), dtype=np.float64)

    # if normalization option is enabled populate array of
    # normalization factors
    if norm == True:
        idxu_count = 0
        for l in range(0, ldim, 1):
            idxu_block[l] = idxu_count
            for mb in range(0, l + 1, 1):
                for ma in range(0, l + 1, 1):
                    u_norm[idxu_count] = np.sqrt(l+1)/4/np.pi
                    idxu_count += 1


    # index list for B
    ncoefs = blist.shape[1]
    idxb = np.zeros((ncoefs, 3), dtype=np.int64)

    # populate index list with bispectrum component indices
    idxb_count = 0
    for l1 in range(0, ldim, 1):
        for l2 in range(0, l1 + 1, 1):
            for l in range(l1 - l2, min(twolmax, l1 + l2) + 1, 2):
                if l >= l1:
                    idxb[idxb_count, 0] = l1
                    idxb[idxb_count, 1] = l2
                    idxb[idxb_count, 2] = l
                    idxb_count += 1

    # index list for zlist
    idxz_count = 0
    for l1 in range(0, ldim, 1):
        for l2 in range(0, l1 + 1, 1):
            for l in range(l1 - l2, min(twolmax, l1 + l2) + 1, 2):
                mb = 0
                while 2 * mb <= l:
                    for ma in range(0, l + 1, 1):
                        idxz_count += 1
                    mb += 1

    idxz = np.zeros((idxz_count, 10), dtype=np.int64)

    idxz_block = np.zeros((ldim, ldim, ldim), dtype=np.int64)

    idxz_count = 0
    for l1 in range(0, ldim, 1):
        for l2 in range(0, l1 + 1, 1):
            for l in range(l1 - l2, min(twolmax, l1 + l2) + 1, 2):
                idxz_block[l1, l2, l] = idxz_count

                # find right beta [llb] entry
                # multiply and divide by l+1 factors
                # account for multiplicity of 1, 2, or 3

                mb = 0
                while 2 * mb <= l:
                    for ma in range(0, l + 1, 1):
                        idxz[idxz_count, 0] = l1  # l1 at 0 position
                        idxz[idxz_count, 1] = l2  # l2 at 1 position
                        idxz[idxz_count, 2] = l  # l at 2 position

                        ma1min = max(0, (2 * ma - l - l2 + l1) / 2)
                        # ma1min at 3 position
                        idxz[idxz_count, 3] = ma1min

                        ma2max = (2 * ma - l -
                                  (2 * idxz[idxz_count, 3] - l1) + l2) / 2
                        # ma2max at 4 position
                        idxz[idxz_count, 4] = ma2max

                        na = min(l1, (2 * ma - l + l2 + l1) / 2) - \
                            idxz[idxz_count, 3] + 1
                        idxz[idxz_count, 5] = na  # na at 5 position

                        mb1min = max(0, (2 * mb - l - l2 + l1) / 2)
                        # mb1min at 6 position
                        idxz[idxz_count, 6] = mb1min

                        mb2max = (2 * mb - l -
                                  (2 * idxz[idxz_count, 6] - l1) + l2) / 2
                        # mb2max at 7 position
                        idxz[idxz_count, 7] = mb2max

                        nb = min(l1, (2 * mb - l + l2 + l1) / 2) - \
                            idxz[idxz_count, 6] + 1
                        idxz[idxz_count, 8] = nb  # nb at 8 position

                        # apply to z(l1,l2,jma,mb) to unique element of
                        # y(l)

                        llu = idxu_block[l] + (l + 1) * mb + ma
                        idxz[idxz_count, 9] = llu  # llu at 9 position

                        idxz_count += 1

                    mb += 1


    # calculate bispectrum components
    npairs = neighborlist.shape[0]
    nneighbors = neighborlist.shape[1]

    ulisttot = np.zeros((idxu_count, 1), dtype=np.complex128)
    zlist = np.zeros((idxz_count, 1), dtype=np.complex128)
    dulist = np.zeros((nneighbors, idxu_count, 3), dtype=np.complex128)
    ulist = np.zeros((idxu_count, 1), dtype=np.complex128)
    tempdb = np.zeros((idxb_count, 3), dtype=np.complex128)
    Rj = np.zeros(3, dtype=np.float64)

    if derivative == True:
        if stress == True:
            isite = seq[0,0]
            nstart = 0
            nsite = 0 # 0,0
            for n in range(npairs):
                i, j = seq[n]
                # talk to Qiang about potential issue here
                # when there are neighborlists where there
                # are no i-i atom pairs
                if i == j:
                    nsite = n
                weight = neighbor_ANs[n,0]

                if i != isite:
                    zero_1d(ulist)
                    addself_uarraytot(twolmax, idxu_block, ulist)
                    ulist *= site_ANs[isite]
                    ulisttot += ulist
                    ulisttot *= u_norm
                    compute_zi(idxz_count, idxz, cglist, idxcg_block, idxu_block,
                               ulisttot, zlist)
                    compute_bi(ncoefs, idxb, idxz_block, idxu_block, ulisttot, zlist,
                               blist[isite])
                    for N in range(nstart, n, 1):
                        I, J = seq[N]
                        Ri = center_atoms[I]
                        Weight = neighbor_ANs[N,0]
                        zero_3d(dulist)
                        for neighbor in prange(nneighbors):
                            x = neighborlist[N, neighbor, 0]
                            y = neighborlist[N, neighbor, 1]
                            z = neighborlist[N, neighbor, 2]
                            r = np.sqrt(x*x + y*y + z*z)
                            if r < 10**(-8):
                                continue
                            psi = np.pi*r/rcut
                            zero_1d(ulist)
                            compute_uarray_polynomial_wD(x, y, z, psi, r, twolmax, ulist,
                                                         dulist[neighbor], idxu_block)

                            dudr(x,y,z,r,rcut,ulist,dulist[neighbor])

                            dulist[neighbor] *= Weight
                            dulist[neighbor] *= u_norm

                            zero_2d(tempdb)

                            compute_dbidrj(ncoefs, idxb, idxz_block, idxu_block, dulist[neighbor],
                                           zlist, tempdb)

                            dblist[N] += tempdb
                            if I != J:
                                dblist[nsite] -= tempdb

                            Rj[0] = x + Ri[0]
                            Rj[1] = y + Ri[1]
                            Rj[2] = z + Ri[2]

                            for k in range(idxb_count):
                                bstress[nsite,k] += np.outer(Ri, tempdb[k])
                                bstress[N,k] -= np.outer(Rj, tempdb[k])

                    isite = i
                    nstart = n
                    zero_1d(ulisttot)
                    zero_1d(zlist)
                    zero_3d(dulist)
                # end if i != isite


                for neighbor in prange(nneighbors):
                    # get components of separation vector and its magnitude
                    # this is also the axis of rotation
                    x = neighborlist[n, neighbor, 0]
                    y = neighborlist[n, neighbor, 1]
                    z = neighborlist[n, neighbor, 2]
                    r = np.sqrt(x*x + y*y + z*z)
                    if r < 10**(-8):
                        continue
                    # angle of rotation
                    psi = np.pi*r/rcut
                    # populate ulist and dulist with Wigner U functions
                    # and derivatives
                    zero_1d(ulist)
                    compute_uarray_polynomial(x, y, z, psi, r, twolmax, ulist,
                                                 idxu_block)


                    ulist *= weight

                    add_uarraytot(r, rcut, ulisttot, ulist)

            zero_1d(ulist)
            addself_uarraytot(twolmax, idxu_block, ulist)
            ulist *= site_ANs[isite]
            ulisttot += ulist
            ulisttot *= u_norm
            compute_zi(idxz_count, idxz, cglist, idxcg_block, idxu_block,
                       ulisttot, zlist)
            compute_bi(ncoefs, idxb, idxz_block, idxu_block, ulisttot, zlist,
                       blist[isite])
            for N in range(nstart, npairs, 1):
                I, J = seq[N]
                Ri = center_atoms[I]
                Weight = neighbor_ANs[N,0]
                zero_3d(dulist)
                for neighbor in prange(nneighbors):
                    x = neighborlist[N, neighbor, 0]
                    y = neighborlist[N, neighbor, 1]
                    z = neighborlist[N, neighbor, 2]
                    r = np.sqrt(x*x + y*y + z*z)
                    if r < 10**(-8):
                        continue
                    psi = np.pi*r/rcut
                    zero_1d(ulist)
                    compute_uarray_polynomial_wD(x, y, z, psi, r, twolmax, ulist,
                                                 dulist[neighbor], idxu_block)

                    dudr(x,y,z,r,rcut,ulist,dulist[neighbor])

                    dulist[neighbor] *= Weight
                    dulist[neighbor] *= u_norm

                    zero_2d(tempdb)

                    compute_dbidrj(ncoefs, idxb, idxz_block, idxu_block, dulist[neighbor],
                                   zlist, tempdb)

                    dblist[N] += tempdb
                    if I != J:
                        dblist[nsite] -= tempdb

                    Rj[0] = x + Ri[0]
                    Rj[1] = y + Ri[1]
                    Rj[2] = z + Ri[2]

                    for k in range(idxb_count):
                        bstress[nsite,k] += np.outer(Ri, tempdb[k])
                        bstress[N,k] -= np.outer(Rj, tempdb[k])


        else:
            isite = seq[0,0]
            nstart = 0
            nsite = 0 # 0,0
            for n in range(npairs):
                i, j = seq[n]
                if i == j:
                    nsite = n
                weight = neighbor_ANs[n,0]

                if i != isite:
                    zero_1d(ulist)
                    addself_uarraytot(twolmax, idxu_block, ulist)
                    ulist *= site_ANs[isite]
                    ulisttot += ulist
                    ulisttot *= u_norm
                    compute_zi(idxz_count, idxz, cglist, idxcg_block, idxu_block,
                               ulisttot, zlist)
                    compute_bi(ncoefs, idxb, idxz_block, idxu_block, ulisttot, zlist,
                               blist[isite])
                    for N in range(nstart, n, 1):
                        I, J = seq[N]
                        Weight = neighbor_ANs[N,0]
                        zero_3d(dulist)
                        for neighbor in prange(nneighbors):
                            x = neighborlist[N, neighbor, 0]
                            y = neighborlist[N, neighbor, 1]
                            z = neighborlist[N, neighbor, 2]
                            r = np.sqrt(x*x + y*y + z*z)
                            if r < 10**(-8):
                                continue
                            psi = np.pi*r/rcut
                            zero_1d(ulist)
                            compute_uarray_polynomial_wD(x, y, z, psi, r, twolmax, ulist,
                                                         dulist[neighbor], idxu_block)

                            dudr(x,y,z,r,rcut,ulist,dulist[neighbor])

                            dulist[neighbor] *= Weight
                            dulist[neighbor] *= u_norm

                            zero_2d(tempdb)

                            compute_dbidrj(ncoefs, idxb, idxz_block, idxu_block, dulist[neighbor],
                                           zlist, tempdb)

                            dblist[N] += tempdb
                            if I != J:
                                dblist[nsite] -= tempdb


                    isite = i
                    nstart = n
                    zero_1d(ulisttot)
                    zero_1d(zlist)
                    zero_3d(dulist)
                # end if i != isite


                for neighbor in prange(nneighbors):
                    # get components of separation vector and its magnitude
                    # this is also the axis of rotation
                    x = neighborlist[n, neighbor, 0]
                    y = neighborlist[n, neighbor, 1]
                    z = neighborlist[n, neighbor, 2]
                    r = np.sqrt(x*x + y*y + z*z)
                    if r < 10**(-8):
                        continue
                    # angle of rotation
                    psi = np.pi*r/rcut
                    # populate ulist and dulist with Wigner U functions
                    # and derivatives
                    zero_1d(ulist)
                    compute_uarray_polynomial(x, y, z, psi, r, twolmax, ulist,
                                                 idxu_block)


                    ulist *= weight

                    add_uarraytot(r, rcut, ulisttot, ulist)

            zero_1d(ulist)
            addself_uarraytot(twolmax, idxu_block, ulist)
            ulist *= site_ANs[isite]
            ulisttot += ulist
            ulisttot *= u_norm
            compute_zi(idxz_count, idxz, cglist, idxcg_block, idxu_block,
                       ulisttot, zlist)
            compute_bi(ncoefs, idxb, idxz_block, idxu_block, ulisttot, zlist,
                       blist[isite])
            for N in range(nstart, npairs, 1):
                I, J = seq[N]
                Weight = neighbor_ANs[N,0]
                zero_3d(dulist)
                for neighbor in prange(nneighbors):
                    x = neighborlist[N, neighbor, 0]
                    y = neighborlist[N, neighbor, 1]
                    z = neighborlist[N, neighbor, 2]
                    r = np.sqrt(x*x + y*y + z*z)
                    if r < 10**(-8):
                        continue
                    psi = np.pi*r/rcut
                    zero_1d(ulist)
                    compute_uarray_polynomial_wD(x, y, z, psi, r, twolmax, ulist,
                                                 dulist[neighbor], idxu_block)

                    dudr(x,y,z,r,rcut,ulist,dulist[neighbor])

                    dulist[neighbor] *= Weight
                    dulist[neighbor] *= u_norm

                    zero_2d(tempdb)

                    compute_dbidrj(ncoefs, idxb, idxz_block, idxu_block, dulist[neighbor],
                                   zlist, tempdb)

                    dblist[N] += tempdb
                    if I != J:
                        dblist[nsite] -= tempdb


    else:
        isite = seq[0,0]
        nstart = 0
        nsite = 0 # 0,0
        for n in range(npairs):
            i, j = seq[n]
            if i == j:
                nsite = n
            weight = neighbor_ANs[n,0]

            if i != isite:
                zero_1d(ulist)
                addself_uarraytot(twolmax, idxu_block, ulist)
                ulist *= site_ANs[isite]
                ulisttot += ulist
                ulisttot *= u_norm
                compute_zi(idxz_count, idxz, cglist, idxcg_block, idxu_block,
                           ulisttot, zlist)
                compute_bi(ncoefs, idxb, idxz_block, idxu_block, ulisttot, zlist,
                           blist[isite])

                isite = i
                nstart = n
                zero_1d(ulisttot)
                zero_1d(zlist)
            # end if i != isite


            for neighbor in prange(nneighbors):
                # get components of separation vector and its magnitude
                # this is also the axis of rotation
                x = neighborlist[n, neighbor, 0]
                y = neighborlist[n, neighbor, 1]
                z = neighborlist[n, neighbor, 2]
                r = np.sqrt(x*x + y*y + z*z)
                if r < 10**(-8):
                    continue
                # angle of rotation
                psi = np.pi*r/rcut
                # populate ulist and dulist with Wigner U functions
                # and derivatives
                zero_1d(ulist)
                compute_uarray_polynomial(x, y, z, psi, r, twolmax, ulist,
                                             idxu_block)


                ulist *= weight

                add_uarraytot(r, rcut, ulisttot, ulist)

        zero_1d(ulist)
        addself_uarraytot(twolmax, idxu_block, ulist)
        ulist *= site_ANs[isite]
        ulisttot += ulist
        ulisttot *= u_norm
        compute_zi(idxz_count, idxz, cglist, idxcg_block, idxu_block,
                   ulisttot, zlist)
        compute_bi(ncoefs, idxb, idxz_block, idxu_block, ulisttot, zlist,
                   blist[isite])

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
                      help="cutoff for neighbor calcs, default: 4.0"
                      )

    parser.add_option("-l", "--lmax", dest="lmax", default=1, type=int,
                      help="lmax, default: 1"
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
    print(test)
    lmax = options.lmax
    rcut = options.rcut
    der = options.der
    stress = options.stress

    #import time
    f = SO4_Bispectrum(lmax, rcut, derivative=False, stress=False, normalize_U=False)
    x = f.calculate(test)
    #start2 = time.time()
    #for key, item in x.items():
    #    print(key, item)
    #print('time elapsed: {}'.format(start2 - start1))
    #print(x['rdxdr'].shape)
    #print(x['rdxdr'])
    #print(np.einsum('ijklm->klm', x['rdxdr']))
    print(x['x'])
