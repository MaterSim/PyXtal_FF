import numpy as np
import mpmath

lmax = 32

mpmath.mp.dps = 4*lmax


_Wigner_coefficients = np.empty(((8 * lmax ** 3 + 18 * lmax ** 2 + 13 * lmax + 3) // 3,), dtype=float)
i = 0
for twol in range(0, 2*lmax + 1):
    for twomp in range(-twol, twol + 1, 2):
        for twom in range(-twol, twol + 1, 2):
            tworho_min = max(0, twomp - twom)
            _Wigner_coefficients[i] = float(mpmath.sqrt(mpmath.fac((twol + twom)//2) * mpmath.fac((twol - twom)//2)
                                                        / (mpmath.fac((twol + twomp)//2) * mpmath.fac((twol - twomp)//2)))
                                            * mpmath.fac((twol + twomp)//2)
                                            * mpmath.fac((twol - twomp)//2))
            i += 1
print(i, _Wigner_coefficients.shape)
np.save('Wigner_coefficients', _Wigner_coefficients)
