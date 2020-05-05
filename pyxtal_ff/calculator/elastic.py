#!/usr/bin/env python
# -*- coding: utf-8 -*-
#    Copyright 1998-2017 by Paweł T. Jochym <pawel.jochym@ifj.edu.pl>
#
#    This file is part of Elastic.

#    Elastic is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    Elastic is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with Elastic.  If not, see <http://www.gnu.org/licenses/>.

'''

.. _elastic-mod:

Elastic Module
^^^^^^^^^^^^^^

Elastic is a module for calculation of :math:`C_{ij}` components of elastic
tensor from the strain-stress relation.

The strain components here are ordered in standard way which is different
to ordering in previous versions of the code (up to 4.0).
The ordering is: :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}`.

The general ordering of :math:`C_{ij}` components is (except for triclinic
symmetry and taking into account customary names of constants - e.g.
:math:`C_{16} \\rightarrow C_{14}`):

.. math::
   C_{11}, C_{22}, C_{33}, C_{12}, C_{13}, C_{23},
   C_{44}, C_{55}, C_{66}, C_{16}, C_{26}, C_{36}, C_{45}

The functions with the name of bravais lattices define the symmetry of the
:math:`C_{ij}` matrix. The matrix is N columns by 6 rows where the columns
corespond to independent elastic constants of the given crystal, while the rows
corespond to the canonical deformations of a crystal. The elements are the
second partial derivatives of the free energy formula for the crystal written
down as a quadratic form of the deformations with respect to elastic constant
and deformation.

*Note:*
The elements for deformations :math:`u_{xy}, u_{xz}, u_{yz}`
have to be divided by 2 to properly match the usual definition
of elastic constants.

See: [LL]_ L.D. Landau, E.M. Lifszyc, "Theory of elasticity"

There is some usefull summary also at:
`ScienceWorld <http://scienceworld.wolfram.com/physics/Elasticity.html>`_

---------

'''

from __future__ import print_function, division, absolute_import
import re
from ase.atoms import Atoms

import spglib as spg
from scipy.linalg import norm, lstsq
from scipy import optimize
from numpy.linalg import inv
from numpy import dot, diag, ones, zeros, reshape, linspace, array, mean
from math import acos, pi, cos, sin, sqrt
import ase.units as units

"""
# to add 
#1, stability criterion
#2, Young's modulus
#3, shear modulus
#4, bulk modulus
#5, Poisson ratio

def stability():


"""

def BMEOS(v, v0, b0, b0p):
    return (b0/b0p)*(pow(v0/v, b0p) - 1)


def ctg(x):
    return cos(x)/sin(x)


def csc(x):
    return 1/sin(x)


def regular(u):
    '''
    Equation matrix generation for the regular (cubic) lattice.
    The order of constants is as follows:

    .. math::
       C_{11}, C_{12}, C_{44}

    :param u: vector of deformations:
        [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    :returns: Symmetry defined stress-strain equation matrix
    '''
    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array(
               [[uxx,   uyy + uzz,      0],
                [uyy,   uxx + uzz,      0],
                [uzz,   uxx + uyy,      0],
                [0,             0,      2*uyz],
                [0,             0,      2*uxz],
                [0,             0,      2*uxy]])


def tetragonal(u):
    '''
    Equation matrix generation for the tetragonal lattice.
    The order of constants is as follows:

    .. math::
       C_{11}, C_{33}, C_{12}, C_{13}, C_{44}, C_{14}

    :param u: vector of deformations:
        [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    :returns: Symmetry defined stress-strain equation matrix
    '''

    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array(
                [[uxx,   0,    uyy,  uzz,      0,      0],
                 [uyy,   0,    uxx,  uzz,      0,      0],
                 [0,     uzz,  0,    uxx+uyy,  0,      0],
                 [0,     0,    0,    0,        2*uyz,  0],
                 [0,     0,    0,    0,        2*uxz,  0],
                 [0,     0,    0,    0,        0,  2*uxy]])


def orthorombic(u):
    '''
    Equation matrix generation for the orthorombic lattice.
    The order of constants is as follows:

    .. math::
       C_{11}, C_{22}, C_{33}, C_{12}, C_{13}, C_{23},
       C_{44}, C_{55}, C_{66}

    :param u: vector of deformations:
        [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    :returns: Symmetry defined stress-strain equation matrix
    '''

    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array(
                [[uxx,     0,    0,  uyy,  uzz,    0,     0,     0,     0],
                 [0,     uyy,    0,  uxx,    0,  uzz,     0,     0,     0],
                 [0,       0,  uzz,    0,  uxx,  uyy,     0,     0,     0],
                 [0,       0,    0,    0,    0,    0, 2*uyz,     0,     0],
                 [0,       0,    0,    0,    0,    0,     0, 2*uxz,     0],
                 [0,       0,    0,    0,    0,    0,     0,     0, 2*uxy]])


def trigonal(u):
    '''
    The matrix is constructed based on the approach from L&L
    using auxiliary coordinates: :math:`\\xi=x+iy`, :math:`\\eta=x-iy`.
    The components are calculated from free energy using formula
    introduced in :ref:`symmetry` with appropriate coordinate changes.
    The order of constants is as follows:

    .. math::
       C_{11}, C_{33}, C_{12}, C_{13}, C_{44}, C_{14}

    :param u: vector of deformations:
        [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    :returns: Symmetry defined stress-strain equation matrix
    '''

    # TODO: Not tested yet.
    # TODO: There is still some doubt about the :math:`C_{14}` constant.
    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array(
                [[   uxx,   0,    uyy,     uzz,     0,   2*uxz      ],
                 [   uyy,   0,    uxx,     uzz,     0,  -2*uxz      ],
                 [     0, uzz,      0, uxx+uyy,     0,   0          ],
                 [     0,   0,      0,       0, 2*uyz,  -4*uxy      ],
                 [     0,   0,      0,       0, 2*uxz,   2*(uxx-uyy)],
                 [ 2*uxy,   0, -2*uxy,       0,     0,  -4*uyz      ]])


def hexagonal(u):
    '''
    The matrix is constructed based on the approach from L&L
    using auxiliary coordinates: :math:`\\xi=x+iy`, :math:`\\eta=x-iy`.
    The components are calculated from free energy using formula
    introduced in :ref:`symmetry` with appropriate coordinate changes.
    The order of constants is as follows:

    .. math::
       C_{11}, C_{33}, C_{12}, C_{13}, C_{44}

    :param u: vector of deformations:
        [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    :returns: Symmetry defined stress-strain equation matrix
    '''

    # TODO: Still needs good verification
    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array(
                [[   uxx,   0,    uyy,     uzz,     0   ],
                 [   uyy,   0,    uxx,     uzz,     0   ],
                 [     0, uzz,      0, uxx+uyy,     0   ],
                 [     0,   0,      0,       0, 2*uyz   ],
                 [     0,   0,      0,       0, 2*uxz   ],
                 [ 2*uxy,   0, -2*uxy,       0,     0   ]])


def monoclinic(u):
    '''Monoclinic group,

    The ordering of constants is:

    .. math::
       C_{11}, C_{22}, C_{33}, C_{12}, C_{13}, C_{23},
       C_{44}, C_{55}, C_{66}, C_{16}, C_{26}, C_{36}, C_{45}

    :param u: vector of deformations:
        [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    :returns: Symmetry defined stress-strain equation matrix
    '''

    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array(
                [[uxx,  0,  0,uyy,uzz,  0,    0,    0,    0,uxy,  0,  0,  0],
                 [  0,uyy,  0,uxx,  0,uzz,    0,    0,    0,  0,uxy,  0,  0],
                 [  0,  0,uzz,  0,uxx,uyy,    0,    0,    0,  0,  0,uxy,  0],
                 [  0,  0,  0,  0,  0,  0,2*uyz,    0,    0,  0,  0,  0,uxz],
                 [  0,  0,  0,  0,  0,  0,    0,2*uxz,    0,  0,  0,  0,uyz],
                 [  0,  0,  0,  0,  0,  0,    0,    0,2*uxy,uxx,uyy,uzz,  0]])


def triclinic(u):
    '''Triclinic crystals.

    *Note*: This was never tested on the real case. Beware!

    The ordering of constants is:

    .. math::
       C_{11}, C_{22}, C_{33},
       C_{12}, C_{13}, C_{23},
       C_{44}, C_{55}, C_{66},
       C_{16}, C_{26}, C_{36}, C_{46}, C_{56},
       C_{14}, C_{15}, C_{25}, C_{45}

    :param u: vector of deformations:
        [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    :returns: Symmetry defined stress-strain equation matrix
    '''

    # Based on the monoclinic matrix and not tested on real case.
    # If you have test cases for this symmetry send them to the author.
    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array(
    [[uxx,  0,  0,uyy,uzz,  0,    0,    0,    0,uxy,  0,  0,  0,  0,uyz,uxz,  0,  0],
     [  0,uyy,  0,uxx,  0,uzz,    0,    0,    0,  0,uxy,  0,  0,  0,  0,  0,uxz,  0],
     [  0,  0,uzz,  0,uxx,uyy,    0,    0,    0,  0,  0,uxy,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,2*uyz,    0,    0,  0,  0,  0,uxy,  0,uxx,  0,  0,uxz],
     [  0,  0,  0,  0,  0,  0,    0,2*uxz,    0,  0,  0,  0,  0,uxy,  0,uxx,uyy,uyz],
     [  0,  0,  0,  0,  0,  0,    0,    0,2*uxy,uxx,uyy,uzz,uyz,uxz,  0,  0,  0,  0]])

def get_tensor_matrix(Cij, cryst):
    '''Give the 6*6 elastic matrix from the array

    :param Cij: array of elastic tensor
    :param cryst: ASE Atoms object

    :returns: Order of elastic constants in 6*6 matrix
    Need to test on other symmtry systems(QZ)
    '''

    lattyp, brav, sg_name, sg_nr = get_lattice_type(cryst)
    matrix = zeros((6,6))
    if lattyp == 7: #cubic
        matrix[0,0] = matrix[1,1] = matrix[2,2] = Cij[0]
        matrix[0,1] = matrix[1,0] = matrix[0,2] = matrix[2,0] = matrix[1,2] = matrix[2,1] = Cij[1]
        matrix[3,3] = matrix[4,4] = matrix[5,5] = Cij[2]
    elif lattyp == 6: #hexagonal
        matrix[0,0] = matrix[1,1] = Cij[0]
        matrix[2,2] = Cij[1]
        matrix[0,1] = matrix[1,0] = Cij[2]
        matrix[0,2] = matrix[2,0] = Cij[3]
        matrix[3,3] = matrix[4,4] = Cij[4]
    elif lattyp == 5: #trigonal
        raise NotImplementedError
    elif lattyp == 4: #tetragonal
        matrix[0,0] = matrix[1,1] = Cij[0]
        matrix[2,2] = Cij[1]
        matrix[0,1] = matrix[1,0] = matrix[0,2] = matrix[2,0] = Cij[2]
        matrix[1,2] = matrix[2,1] = Cij[3]
        matrix[3,3] = matrix[4,4] = Cij[4]
        matrix[5,5] = Cij[5]
    elif lattyp == 3: #orthorhombic
        matrix[0,0] = Cij[0]
        matrix[1,1] = Cij[1]
        matrix[2,2] = Cij[2]
        matrix[0,1] = matrix[1,0] = Cij[3]
        matrix[0,2] = matrix[2,0] = Cij[4]
        matrix[1,2] = matrix[2,1] = Cij[5]
        matrix[3,3] = Cij[6]
        matrix[4,4] = Cij[7]
        matrix[5,5] = Cij[8]
    elif lattyp == 2:
        raise NotImplementedError
    elif lattyp == 1:
        raise NotImplementedError
    
    return matrix


def get_cij_order(cryst):
    '''Give order of of elastic constants for the structure

    :param cryst: ASE Atoms object

    :returns: Order of elastic constants as a tuple of strings: C_ij
    '''

    orders = {
            1: ('C_11', 'C_22', 'C_33', 'C_12', 'C_13', 'C_23',
                'C_44', 'C_55', 'C_66', 'C_16', 'C_26', 'C_36',
                'C_46', 'C_56', 'C_14', 'C_15', 'C_25', 'C_45'),
            2: ('C_11', 'C_22', 'C_33', 'C_12', 'C_13', 'C_23',
                'C_44', 'C_55', 'C_66', 'C_16', 'C_26', 'C_36', 'C_45'),
            3: ('C_11', 'C_22', 'C_33', 'C_12', 'C_13', 'C_23', 'C_44',
                'C_55', 'C_66'),
            4: ('C_11', 'C_33', 'C_12', 'C_13', 'C_44', 'C_66'), #error in the original code
            5: ('C_11', 'C_33', 'C_12', 'C_13', 'C_44', 'C_14'), #C_66=1/2(C11-C22)
            6: ('C_11', 'C_33', 'C_12', 'C_13', 'C_44'),
            7: ('C_11', 'C_12', 'C_44'),
            }
    return orders[get_lattice_type(cryst)[0]]


def get_lattice_type(cryst):
    '''Find the symmetry of the crystal using spglib symmetry finder.

    Derive name of the space group and its number extracted from the result.
    Based on the group number identify also the lattice type and the Bravais
    lattice of the crystal. The lattice type numbers are
    (the numbering starts from 1):

    Triclinic (1), Monoclinic (2), Orthorombic (3),
    Tetragonal (4), Trigonal (5), Hexagonal (6), Cubic (7)

    :param cryst: ASE Atoms object

    :returns: tuple (lattice type number (1-7), lattice name, space group
                     name, space group number)
    '''

    # Table of lattice types and correcponding group numbers dividing
    # the ranges. See get_lattice_type method for precise definition.
    lattice_types = [
            [3,   "Triclinic"],
            [16,  "Monoclinic"],
            [75,  "Orthorombic"],
            [143, "Tetragonal"],
            [168, "Trigonal"],
            [195, "Hexagonal"],
            [231, "Cubic"]
        ]

    sg = spg.get_spacegroup(cryst)
    m = re.match(r'([A-Z].*\b)\s*\(([0-9]*)\)', sg)
    sg_name = m.group(1)
    sg_nr = int(m.group(2))

    for n, l in enumerate(lattice_types):
        if sg_nr < l[0]:
            bravais = l[1]
            lattype = n+1
            break

    return lattype, bravais, sg_name, sg_nr


def get_bulk_modulus(cryst):
    '''Calculate bulk modulus using the Birch-Murnaghan equation of state.

    The EOS must be previously calculated by get_BM_EOS routine.
    The returned bulk modulus is a :math:`B_0` coefficient of the B-M EOS.
    The units of the result are defined by ASE. To get the result in
    any particular units (e.g. GPa) you need to divide it by
    ase.units.<unit name>::

        get_bulk_modulus(cryst)/ase.units.GPa

    :param cryst: ASE Atoms object

    :returns: float, bulk modulus :math:`B_0` in ASE units.
    '''

    if getattr(cryst, 'bm_eos', None) is None:
        raise RuntimeError('Missing B-M EOS data.')
    cryst.bulk_modulus = cryst.bm_eos[1]
    return cryst.bulk_modulus


def get_pressure(s):
    '''Return *external* isotropic (hydrostatic) pressure in ASE units.

    If the pressure is positive the system is under external pressure.
    This is a convenience function to convert output of get_stress function
    into external pressure.

    :param cryst: stress tensor in Voight (vector) notation as returned by
        the get_stress() method.

    :returns: float, external hydrostatic pressure in ASE units.
    '''

    return -mean(s[:3])


def get_BM_EOS(cryst, systems):
    """Calculate Birch-Murnaghan Equation of State for the crystal.

    The B-M equation of state is defined by:

    .. math::
       P(V)= \\frac{B_0}{B'_0}\\left[
       \\left({\\frac{V}{V_0}}\\right)^{-B'_0} - 1
       \\right]

    It's coefficients are estimated using n single-point structures ganerated
    from the crystal (cryst) by the scan_volumes function between two relative
    volumes. The BM EOS is fitted to the computed points by
    least squares method. The returned value is a list of fitted
    parameters: :math:`V_0, B_0, B_0'` if the fit succeded.
    If the fitting fails the ``RuntimeError('Calculation failed')`` is raised.
    The data from the calculation and fit is stored in the bm_eos and pv
    members of cryst for future reference. You have to provide properly
    optimized structures in cryst and systems list.

    :param cryst: Atoms object, basic structure
    :param systems: A list of calculated structures

    :returns: tuple of EOS parameters :math:`V_0, B_0, B_0'`.
    """

    pvdat = array([[r.get_volume(),
                    get_pressure(r.get_stress()),
                    norm(r.get_cell()[:, 0]),
                    norm(r.get_cell()[:, 1]),
                    norm(r.get_cell()[:, 2])] for r in systems]).T

    # Estimate the initial guess assuming b0p=1
    # Limiting volumes
    v1 = min(pvdat[0])
    v2 = max(pvdat[0])

    # The pressure is falling with the growing volume
    p2 = min(pvdat[1])
    p1 = max(pvdat[1])
    b0 = (p1*v1-p2*v2)/(v2-v1)
    v0 = v1*(p1+b0)/b0

    # Initial guess
    p0 = [v0, b0, 1]

    # Fitting
    try :
        p1, succ = optimize.curve_fit(BMEOS, pvdat[0], pvdat[1], p0)
    except (ValueError, RuntimeError, optimize.OptimizeWarning) as ex:
        raise RuntimeError('Calculation failed')

    cryst.bm_eos = p1
    cryst.pv = pvdat
    return cryst.bm_eos


def get_elementary_deformations(cryst, n=5, d=2):
    '''Generate elementary deformations for elastic tensor calculation.

    The deformations are created based on the symmetry of the crystal and
    are limited to the non-equivalet axes of the crystal.

    :param cryst: Atoms object, basic structure
    :param n: integer, number of deformations per non-equivalent axis
    :param d: float, size of the maximum deformation in percent and degrees

    :returns: list of deformed structures
    '''
    # Deformation look-up table
    # Perhaps the number of deformations for trigonal
    # system could be reduced to [0,3] but better safe then sorry
    deform = {
        "Cubic": [[0, 5], regular],
        "Hexagonal": [[0, 2, 3, 5], hexagonal],
        "Trigonal": [[0, 1, 2, 3, 4, 5], trigonal],
        "Tetragonal": [[0, 2, 3, 5], tetragonal],
        "Orthorombic": [[0, 1, 2, 3, 4, 5], orthorombic],
        "Monoclinic": [[0, 1, 2, 3, 4, 5], monoclinic],
        "Triclinic": [[0, 1, 2, 3, 4, 5], triclinic]
    }

    lattyp, brav, sg_name, sg_nr = get_lattice_type(cryst)
    print('This is a {:s} system'.format(brav))
    # Decide which deformations should be used
    axis, symm = deform[brav]
    print('{:d} deformations were generated'.format(len(axis)))

    systems = []
    for a in axis:
        if a < 3:  # tetragonal deformation
            for dx in linspace(-d, d, n):
                systems.append(
                        get_cart_deformed_cell(cryst, axis=a, size=dx))
        elif a < 6:  # shear deformation (skip the zero angle)
            for dx in linspace(d/2.0, d, n):
                systems.append(
                        get_cart_deformed_cell(cryst, axis=a, size=dx))
    return systems


def get_elastic_tensor(cryst, systems):
    '''Calculate elastic tensor of the crystal.

    The elastic tensor is calculated from the stress-strain relation
    and derived by fitting this relation to the set of linear equations
    build from the symmetry of the crystal and strains and stresses
    of the set of elementary deformations of the unit cell.

    It is assumed that the crystal is converged and optimized
    under intended pressure/stress. The geometry and stress on the
    cryst is taken as the reference point. No additional optimization
    will be run. Structures in cryst and systems list must have calculated
    stresses. The function returns tuple of :math:`C_{ij}` elastic tensor,
    raw Birch coefficients :math:`B_{ij}` and fitting results: residuals,
    solution rank, singular values returned by numpy.linalg.lstsq.

    :param cryst: Atoms object, basic structure
    :param systems: list of Atoms object with calculated deformed structures

    :returns: tuple(:math:`C_{ij}` float vector,
                    tuple(:math:`B_{ij}` float vector,
                          residuals, solution rank, singular values)
    '''

    # Deformation look-up table
    # Perhaps the number of deformations for trigonal
    # system could be reduced to [0,3] but better safe then sorry
    deform = {
        "Cubic": [[0, 5], regular],
        "Hexagonal": [[0, 2, 3, 5], hexagonal],
        "Trigonal": [[0, 1, 2, 3, 4, 5], trigonal],
        "Tetragonal": [[0, 2, 3, 5], tetragonal],
        "Orthorombic": [[0, 1, 2, 3, 4, 5], orthorombic],
        "Monoclinic": [[0, 1, 2, 3, 4, 5], monoclinic],
        "Triclinic": [[0, 1, 2, 3, 4, 5], triclinic]
    }

    lattyp, brav, sg_name, sg_nr = get_lattice_type(cryst)
    # Decide which deformations should be used
    axis, symm = deform[brav]

    ul = []
    sl = []
    p = get_pressure(cryst.get_stress())
    for g in systems:
        ul.append(get_strain(g, refcell=cryst))
        # Remove the ambient pressure from the stress tensor
        sl.append(g.get_stress()-array([p, p, p, 0, 0, 0]))
        #print(g.get_stress()/units.GPa, get_strain(g, refcell=cryst))
    eqm = array([symm(u) for u in ul])
    # print(eqm[0].shape, eqm.shape)
    eqm = reshape(eqm, (eqm.shape[0]*eqm.shape[1], eqm.shape[2]))
    #print(eqm)
    slm = reshape(array(sl), (-1,))
    # print(eqm.shape, slm.shape)
    #print(slm)
    Bij = lstsq(eqm, slm)
    #print(Bij)
    # print(Bij[0] / units.GPa)
    # Calculate elastic constants from Birch coeff.
    # TODO: Check the sign of the pressure array in the B <=> C relation
    if (symm == orthorombic):
        Cij = Bij[0] - array([-p, -p, -p, p, p, p, -p, -p, -p])

    elif (symm == tetragonal):
        Cij = Bij[0] - array([-p, -p, p, p, -p, -p])

    elif (symm == regular):
        Cij = Bij[0] - array([-p, p, -p])

    elif (symm == trigonal):
        Cij = Bij[0] - array([-p, -p, p, p, -p, p])

    elif (symm == hexagonal):
        Cij = Bij[0] - array([-p, -p, p, p, -p])

    elif (symm == monoclinic):
        # TODO: verify this pressure array
        Cij = Bij[0] - array([-p, -p, -p, p, p, p, -p, -p, -p, p, p, p, p])
        
    elif (symm == triclinic):
        # TODO: verify this pressure array
        Cij = Bij[0] - array([-p, -p, -p, p, p, p, -p, -p, -p,
                              p, p, p, p, p, p, p, p, p])
    Cij /= units.GPa
    matrix = get_tensor_matrix(Cij, cryst)
    name = get_cij_order(cryst)
    return Cij, Bij, name, matrix


def scan_pressures(cryst, lo, hi, n=5, eos=None):
    '''
    Scan the pressure axis from lo to hi (inclusive)
    using B-M EOS as the volume predictor.
    Pressure (lo, hi) in GPa
    '''
    # Inverse B-M EOS to get volumes from pressures
    # This will work only in limited pressure range p>-B/B'.
    # Warning! Relative, the V0 prefactor is removed.
    def invbmeos(b, bp, x):
        return array([pow(b/(bp*xv+b), 1/(3*bp)) for xv in x])

    if eos is None:
        raise RuntimeError('Required EOS data missing')

    # Limit negative pressures to 90% of the singularity value.
    # Beyond this B-M EOS is bound to be wrong anyway.
    lo = max(lo, -0.9*eos[1]/eos[2])

    scale = (eos[0]/cryst.get_volume())*invbmeos(eos[1], eos[2],
                                                 linspace(lo, hi, num=n))
    # print(scale)
    uc = cryst.get_cell()
    systems = [Atoms(cryst) for s in scale]
    for n, s in enumerate(scale):
        systems[n].set_cell(s*uc, scale_atoms=True)

    return systems


def scan_volumes(cryst, lo=0.98, hi=1.02, n=5, scale_volumes=True):
    '''
    Provide set of crystals along volume axis from lo to hi (inclusive).
    No volume cell optimization is performed. Bounds are specified as
    fractions (1.10 = 10% increase). If scale_volumes==False the scalling
    is applied to lattice vectors instead of volumes.

    :param lo: lower bound of the V/V_0 in the scan
    :param hi: upper bound of the V/V_0 in the scan
    :param n: number of volume sample points
    :param scale_volumes: If True scale the unit cell volume or,
                            if False, scale the length of lattice axes.

    :returns: a list of deformed systems
    '''
    scale = linspace(lo, hi, num=n)
    if scale_volumes:
        scale **= (1.0/3.0)
    uc = cryst.get_cell()
    systems = [Atoms(cryst) for s in scale]
    for n, s in enumerate(scale):
        systems[n].set_cell(s*uc, scale_atoms=True)
    return systems


def get_vecang_cell(cryst, uc=None):
    '''
    Compute A,B,C, alpha,beta,gamma cell params
    from the unit cell matrix (uc) or cryst.
    Angles in radians.
    '''
    if uc is None:
        uc = cryst.get_cell()
    ucv = [uc[i, :]/norm(uc[i, :]) for i in range(3)]
    uca = [acos(dot(ucv[(i+1) % 3], ucv[(i+2) % 3])) for i in range(3)]
    return [norm(uc[i, :]) for i in range(3)] + uca


def get_deformed_cell(base_cryst, axis=0, size=1):
    '''
    Return the cell (with atoms) deformed along one
    cell parameter (0,1,2 = a,b,c ; 3,4,5 = alpha,beta,gamma) by
    size percent or size degrees (axis/angles).
    '''
    cryst = Atoms(base_cryst)
    uc = base_cryst.get_cell()
    if axis < 3:
        uc[axis, :] = (1+size/100.0)*uc[axis, :]
    else:
        (a, b, c, alp, bet, gam) = get_vecang_cell(cryst)
        d = array([0.0, 0.0, 0.0])
        d[axis-3] = pi*size/180
        (alp, bet, gam) = array((alp, bet, gam))+d
        t = 1 - (ctg(bet)*ctg(gam)-cos(alp)*csc(bet)*csc(gam))**2
        if t < 0.0:
            print('''
            The parameters (alpha,beta,gamma)=(%f,%f,%f) are probably
            incorrect and lead to imaginary coordinates.
            This range of parameters is unsupported by this program
            (and is, let me say, very strange for a crystal).
            Cennot continue, bye.''' % (alp, bet, gam))
            raise ValueError
        else:
            uc = [[a, 0.0, 0.0],
                  [b*cos(gam), b*sin(gam), 0],
                  [c*cos(bet),
                   c*(cos(alp)/sin(gam) - cos(bet)*ctg(gam)),
                   c*sin(bet)*sqrt(t)]]
    cryst.set_cell(uc, scale_atoms=True)
    # print(cryst.get_cell())
    # print(uc)
    return cryst


def get_cart_deformed_cell(base_cryst, axis=0, size=1):
    '''Return the cell deformed along one of the cartesian directions

    Creates new deformed structure. The deformation is based on the
    base structure and is performed along single axis. The axis is
    specified as follows: 0,1,2 = x,y,z ; sheers: 3,4,5 = yz, xz, xy.
    The size of the deformation is in percent and degrees, respectively.

    :param base_cryst: structure to be deformed
    :param axis: direction of deformation
    :param size: size of the deformation

    :returns: new, deformed structure
    '''
    cryst = Atoms(base_cryst)
    uc = base_cryst.get_cell()
    s = size/100.0
    L = diag(ones(3))
    if axis < 3:
        L[axis, axis] += s
    else:
        if axis == 3:
            L[1, 2] += s
        elif axis == 4:
            L[0, 2] += s
        else:
            L[0, 1] += s
    uc = dot(uc, L)
    cryst.set_cell(uc, scale_atoms=True)
    # print(cryst.get_cell())
    # print(uc)
    return cryst


def get_strain(cryst, refcell=None):
    '''Calculate strain tensor in the Voight notation

    Computes the strain tensor in the Voight notation as a conventional
    6-vector. The calculation is done with respect to the crystal
    geometry passed in refcell parameter.

    :param cryst: deformed structure
    :param refcell: reference, undeformed structure

    :returns: 6-vector of strain tensor in the Voight notation
    '''
    if refcell is None:
        refcell = cryst
    du = cryst.get_cell()-refcell.get_cell()
    m = refcell.get_cell()
    m = inv(m)
    u = dot(m, du)
    u = (u+u.T)/2
    return array([u[0, 0], u[1, 1], u[2, 2], u[2, 1], u[2, 0], u[1, 0]])


if __name__ == '__main__':
    from ase.spacegroup import crystal

    a = 4.194
    cryst = crystal(['Mg', 'O'],
                    [(0, 0, 0), (0.5, 0.5, 0.5)],
                    spacegroup=225,
                    cellpar=[a, a, a, 90, 90, 90])

    sl = scan_volumes(cryst)
    print('Volumes: ', end='')
    for c in sl:
        print('%.2f (%.1f%%)' % (c.get_volume(),
                                 100*c.get_volume()/cryst.get_volume()),
              end=' ')

    print()

    sl = get_elementary_deformations(cryst)
    print('Structures: ')
    print('   Vol             A       B       C          alph    bet     gam')
    for n, c in enumerate(sl):
        print('%.4f (%5.1f%%)' % (c.get_volume(),
                                  100*c.get_volume()/cryst.get_volume()),
              end='')
        print((3*' %7.4f' + '  ' + 3*' %7.2f') %
              tuple(c.get_cell_lengths_and_angles()))
