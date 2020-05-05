from __future__ import division
import numpy as np
import os.path
import numba

#print('load the precomputed values from ', os.path.join(os.path.dirname(__file__), 'Wigner_coefficients.npy'))
_Wigner_coefficients = np.load(os.path.join(os.path.dirname(__file__), 'Wigner_coefficients.npy'))

@numba.njit(numba.i8(numba.i8, numba.i8, numba.i8),
            cache=True, fastmath=True, nogil=True)
def _Wigner_index(twoj, twomp, twom):
    return twoj*((2*twoj + 3) * twoj + 1) // 6 + (twoj + twomp)//2 * (twoj + 1) + (twoj + twom) //2

@numba.njit(numba.f8(numba.i8, numba.i8, numba.i8),
           cache=True, fastmath=True, nogil=True)
def _Wigner_coefficient(twoj, twomp, twom):
    return _Wigner_coefficients[_Wigner_index(twoj, twomp, twom)]

@numba.njit(numba.f8(numba.f8,numba.f8,numba.f8),
            cache=True, fastmath=True, nogil=True)
def Wigner_coefficient(j,mp,m):
    return _Wigner_coefficient(round(2*j), round(2*mp), round(2*m))

