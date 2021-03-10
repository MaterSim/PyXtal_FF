import numpy as np
from optparse import OptionParser
import numba as nb
import cmath
import os.path
from pyxtal_ff.descriptors.utils import (_Wigner_coefficient as _coeff,
               Wigner_coefficient as coeff)

@nb.njit(nb.f8(nb.i8), cache=True, nogil=True, fastmath=True)
def factorial(n):

    n = int(round(n))

    fac_arr = np.array([1,
  1,
  2,
  6,
  24,
  120,
  720,
  5040,
  40320,
  362880,
  3628800,
  39916800,
  479001600,
  6227020800,
  87178291200,
  1307674368000,
  20922789888000,
  355687428096000,
  6.402373705728e+15,
  1.21645100408832e+17,
  2.43290200817664e+18,
  5.10909421717094e+19,
  1.12400072777761e+21,
  2.5852016738885e+22,
  6.20448401733239e+23,
  1.5511210043331e+25,
  4.03291461126606e+26,
  1.08888694504184e+28,
  3.04888344611714e+29,
  8.8417619937397e+30,
  2.65252859812191e+32,
  8.22283865417792e+33,
  2.63130836933694e+35,
  8.68331761881189e+36,
  2.95232799039604e+38,
  1.03331479663861e+40,
  3.71993326789901e+41,
  1.37637530912263e+43,
  5.23022617466601e+44,
  2.03978820811974e+46,
  8.15915283247898e+47,
  3.34525266131638e+49,
  1.40500611775288e+51,
  6.04152630633738e+52,
  2.65827157478845e+54,
  1.1962222086548e+56,
  5.50262215981209e+57,
  2.58623241511168e+59,
  1.24139155925361e+61,
  6.08281864034268e+62,
  3.04140932017134e+64,
  1.55111875328738e+66,
  8.06581751709439e+67,
  4.27488328406003e+69,
  2.30843697339241e+71,
  1.26964033536583e+73,
  7.10998587804863e+74,
  4.05269195048772e+76,
  2.35056133128288e+78,
  1.3868311854569e+80,
  8.32098711274139e+81,
  5.07580213877225e+83,
  3.14699732603879e+85,
  1.98260831540444e+87,
  1.26886932185884e+89,
  8.24765059208247e+90,
  5.44344939077443e+92,
  3.64711109181887e+94,
  2.48003554243683e+96,
  1.71122452428141e+98,
  1.19785716699699e+100,
  8.50478588567862e+101,
  6.12344583768861e+103,
  4.47011546151268e+105,
  3.30788544151939e+107,
  2.48091408113954e+109,
  1.88549470166605e+111,
  1.45183092028286e+113,
  1.13242811782063e+115,
  8.94618213078297e+116,
  7.15694570462638e+118,
  5.79712602074737e+120,
  4.75364333701284e+122,
  3.94552396972066e+124,
  3.31424013456535e+126,
  2.81710411438055e+128,
  2.42270953836727e+130,
  2.10775729837953e+132,
  1.85482642257398e+134,
  1.65079551609085e+136,
  1.48571596448176e+138,
  1.3520015276784e+140,
  1.24384140546413e+142,
  1.15677250708164e+144,
  1.08736615665674e+146,
  1.03299784882391e+148,
  9.91677934870949e+149,
  9.61927596824821e+151,
  9.42689044888324e+153,
  9.33262154439441e+155,
  9.33262154439441e+157,
  9.42594775983835e+159,
  9.61446671503512e+161,
  9.90290071648618e+163,
  1.02990167451456e+166,
  1.08139675824029e+168,
  1.14628056373471e+170,
  1.22652020319614e+172,
  1.32464181945183e+174,
  1.44385958320249e+176,
  1.58824554152274e+178,
  1.76295255109024e+180,
  1.97450685722107e+182,
  2.23119274865981e+184,
  2.54355973347219e+186,
  2.92509369349301e+188,
  3.3931086844519e+190,
  3.96993716080872e+192,
  4.68452584975429e+194,
  5.5745857612076e+196,
  6.68950291344912e+198,
  8.09429852527344e+200,
  9.8750442008336e+202,
  1.21463043670253e+205,
  1.50614174151114e+207,
  1.88267717688893e+209,
  2.37217324288005e+211,
  3.01266001845766e+213,
  3.8562048236258e+215,
  4.97450422247729e+217,
  6.46685548922047e+219,
  8.47158069087882e+221,
  1.118248651196e+224,
  1.48727070609069e+226,
  1.99294274616152e+228,
  2.69047270731805e+230,
  3.65904288195255e+232,
  5.01288874827499e+234,
  6.91778647261949e+236,
  9.61572319694109e+238,
  1.34620124757175e+241,
  1.89814375907617e+243,
  2.69536413788816e+245,
  3.85437071718007e+247,
  5.5502938327393e+249,
  8.04792605747199e+251,
  1.17499720439091e+254,
  1.72724589045464e+256,
  2.55632391787286e+258,
  3.80892263763057e+260,
  5.71338395644585e+262,
  8.62720977423323e+264,
  1.31133588568345e+267,
  2.00634390509568e+269,
  3.08976961384735e+271,
  4.78914290146339e+273,
  7.47106292628289e+275,
  1.17295687942641e+278,
  1.85327186949373e+280,
  2.94670227249504e+282,
  4.71472363599206e+284,
  7.59070505394721e+286,
  1.22969421873945e+289,
  2.0044015765453e+291,
  3.28721858553429e+293,
  5.42391066613159e+295,
  9.00369170577843e+297,
  1.503616514865e+300], dtype=np.float64)
    return fac_arr[n]

@nb.njit(nb.f8(nb.i8, nb.i8, nb.i8), cache=True,
         fastmath=True, nogil=True)
def deltacg(l1, l2, l):
    sfaccg = factorial((l1 + l2 + l) // 2 + 1)
    return np.sqrt(factorial((l1 + l2 - l) // 2) *
                   factorial((l1 - l2 + l) // 2) *
                   factorial((-l1 + l2 + l) // 2) / sfaccg)

@nb.njit(nb.c16(nb.c16, nb.c16, nb.i8, nb.i8, nb.i8),
            cache=True, fastmath=True, nogil=True)
def Wigner_D(Ra, Rb, twol, twomp, twom):

    ra, phia = cmath.polar(Ra)
    rb, phib = cmath.polar(Rb)

    epsilon = 10**(-15)
    if ra <= epsilon:
        if twomp != -twom or abs(twomp) > twol or abs(twom) > twol:
            return 0.0j

        else:
            if (twol - twom) % 4 == 0:
                return Rb**twom
            else:
                return -Rb**twom

    elif rb <= epsilon:
        if twomp != twom or abs(twomp) > twol or abs(twom) > twol:
            return 0.0j
        else:
            return Ra**twom

    elif (ra < rb):
        x = - ra*ra / rb / rb

        if (abs(twomp) > twol or abs(twom) > twol):
            return 0.0j
        else:
            Prefactor = cmath.rect(
                rb **(twol - (twom+twomp)/2)
                * ra ** ((twom+twomp)/2),
                phib * (twom - twomp)/2 + phia * (twom + twomp)/2)
            Prefactor *= np.sqrt(factorial(round(twol/2+twom/2))*
                                 factorial(round(twol/2-twom/2))*
                                 factorial(round(twol/2-twomp/2))*
                                 factorial(round(twol/2-twomp/2)))

            if Prefactor == 0.0j:
                return 0.0j
            else:
                l = twol/2
                mp = twomp/2
                m = twom/2
                kmax = round(min(l-mp, l-m))
                kmin = round(max(0, -mp-m))

                if ((twol - twom) %4 != 0):
                    Prefactor *= -1

                Sum = 1/factorial(kmax)/factorial(l-m-kmax)/factorial(mp+m+kmax)/factorial(l-mp-kmax)
                for k in range(kmax-1, kmin-1, -1):
                    Sum *= x
                    Sum += 1/factorial(k)/factorial(l-m-k)/factorial(mp+m+k)/factorial(l-mp-k)
                Sum *= x**(kmin)
                return Prefactor * Sum

    else:
        x = - rb*rb / (ra * ra)
        if (abs(twomp) > twol or abs(twom) > twol):
            return 0.0j

        else:
            Prefactor = cmath.rect(
                                   ra **(twol - twom/2 + twomp/2)
                                   * rb **(twom/2 - twomp/2),
                                   phia * (twom + twomp)/2 + phib * (twom - twomp)/2)
            Prefactor *= np.sqrt(factorial(round(twol/2+twom/2))*
                                 factorial(round(twol/2-twom/2))*
                                 factorial(round(twol/2-twomp/2))*
                                 factorial(round(twol/2-twomp/2)))

            if Prefactor == 0.0j:
                return 0.0j

            else:
                l = twol/2
                mp = twomp/2
                m = twom/2
                kmax = round(min(l + mp, l - m))
                kmin = round(max(0, mp-m))

                Sum = 1/factorial(kmax)/factorial(l+mp-kmax)/factorial(l-m-kmax)/factorial(-mp+m+kmax)
                for k in range(kmax-1, kmin-1, -1):
                    Sum *= x
                    Sum += 1/factorial(k)/factorial(l+mp-k)/factorial(l-m-k)/factorial(-mp+m+k)
                Sum *= x**(kmin)
                return Prefactor * Sum

@nb.njit(nb.c16(nb.c16, nb.c16, nb.i8, nb.i8, nb.i8, nb.c16[:], nb.c16[:], nb.c16[:]),
            cache=True, fastmath=True, nogil=True)
def Wigner_D_wDerivative(Ra, Rb, twol, twomp, twom, gradRa, gradRb, gradArr):

    ra, phia = cmath.polar(Ra)
    rb, phib = cmath.polar(Rb)

    epsilon = 10**(-15)
    if ra <= epsilon:
        if twomp != -twom or abs(twomp) > twol or abs(twom) > twol:
            return 0.0j

        else:
            if (twol - twom) % 4 == 0:
                D = Rb**twom
                gradArr += twom/Rb*D*gradRb
                return D
            else:
                D = -Rb**twom
                gradArr += twom/Rb*D*gradRb
                return D

    elif rb <= epsilon:
        if twomp != twom or abs(twomp) > twol or abs(twom) > twol:
            return 0.0j
        else:
            D = Ra**twom
            gradArr += twom/Ra*D*gradRa
            return D

    elif (ra < rb):
        x = - ra*ra / rb / rb

        if (abs(twomp) > twol or abs(twom) > twol):
            return 0.0j
        else:
            Prefactor = cmath.rect(
                rb **(twol - (twom+twomp)/2)
                * ra ** ((twom+twomp)/2),
                phib * (twom - twomp)/2 + phia * (twom + twomp)/2)
            Prefactor *= np.sqrt(factorial(round(twol/2+twom/2))*
                                 factorial(round(twol/2-twom/2))*
                                 factorial(round(twol/2-twomp/2))*
                                 factorial(round(twol/2-twomp/2)))

            if Prefactor == 0.0j:
                return 0.0j
            else:
                l = twol/2
                mp = twomp/2
                m = twom/2
                kmax = round(min(l-mp, l-m))
                kmin = round(max(0, -mp-m))

                if ((twol - twom) %4 != 0):
                    Prefactor *= -1

                Sum = 1/factorial(kmax)/factorial(l-m-kmax)/factorial(mp+m+kmax)/factorial(l-mp-kmax)
                dSum = 0
                for k in range(kmax-1, kmin-1, -1):
                    dSum = Sum + x*dSum
                    Sum *= x
                    Sum += 1/factorial(k)/factorial(l-m-k)/factorial(mp+m+k)/factorial(l-mp-k)

                for k in range(kmin-1, -1,-1):
                    dSum = Sum + x*dSum
                    Sum *= x

                D = Prefactor * Sum


                gradArr += D*((mp+m)*gradRa/Ra + (l-mp)*gradRb/Rb + (l-m)*np.conj(gradRb/Rb)) + \
                        Prefactor * dSum * x * (gradRa/Ra - gradRb/Rb + np.conj(gradRa/Ra) - np.conj(gradRb/Rb))
                return D

    else:
        x = - rb*rb / (ra * ra)
        if (abs(twomp) > twol or abs(twom) > twol):
            return 0.0j

        else:
            Prefactor = cmath.rect(
                                   ra **(twol - twom/2 + twomp/2)
                                   * rb **(twom/2 - twomp/2),
                                   phia * (twom + twomp)/2 + phib * (twom - twomp)/2)
            Prefactor *= np.sqrt(factorial(round(twol/2+twom/2))*
                                 factorial(round(twol/2-twom/2))*
                                 factorial(round(twol/2-twomp/2))*
                                 factorial(round(twol/2-twomp/2)))

            if Prefactor == 0.0j:
                return 0.0j

            else:
                l = twol/2
                mp = twomp/2
                m = twom/2
                kmax = round(min(l + mp, l - m))
                kmin = round(max(0, mp-m))

                Sum = 1/factorial(kmax)/factorial(l+mp-kmax)/factorial(l-m-kmax)/factorial(-mp+m+kmax)
                dSum = 0
                for k in range(kmax-1, kmin-1, -1):
                    dSum = Sum + x*dSum
                    Sum *= x
                    Sum += 1/factorial(k)/factorial(l+mp-k)/factorial(l-m-k)/factorial(-mp+m+k)

                for k in range(kmin-1, -1, -1):
                    dSum = Sum + x*dSum
                    Sum *= x

                D = Prefactor * Sum

                gradArr += D*((l+mp)*gradRa/Ra + (m-mp)*gradRb/Rb + (l-m)*np.conj(gradRa/Ra)) + \
                        Prefactor*dSum * x * (gradRb/Rb - gradRa/Ra + np.conj(gradRb/Rb) - np.conj(gradRa/Ra))
                return D

