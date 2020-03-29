#!/usr/bin/env python
'''

Cell optimization class
'''

from ase import *
from ase.io import read,write
from ase import units
import numpy as np

class mushybox(Atoms):
    def __init__(self, atomsx, express=np.zeros((3,3)), fixstrain=np.ones((3,3)), frozen=[], rigid=[]):
        """box relaxation
        atomsx: an Atoms object
        express: external pressure, a 3*3 lower triangular matrix in the unit of GPa;
                 define positive values as compression
        fixstrain: 3*3 matrix as express. 0 fixes strain at the corresponding direction
        # QZ: add
        frozen: fix during the relaxation
        rigid: rigid body during the relaxation
        """
        self.atomsx = atomsx 
        self.express= express * units.GPa
        self.rigid = rigid
        self.frozen = frozen
        if express[0][1]**2+express[0][2]**2+express[1][2]**2 > 1e-5:
           express[0][1] = 0
           express[0][2] = 0
           express[1][2] = 0
           print("warning: xy, xz, yz components of the external pressure will be set to zero")
        self.fixstrain = fixstrain

        cell       = atomsx.get_cell()
        vol        = atomsx.get_volume()
        self.natom = atomsx.get_number_of_atoms()
        avglen     = (vol/self.natom)**(1.0/3.0)
        self.jacobian = avglen * self.natom**0.5
        Atoms.__init__(self,atomsx)

    def get_positions(self):
        r    = self.atomsx.get_positions()*0.0
        Rc   = np.vstack((r, self.atomsx.get_cell()*0.0))
        return Rc

    def set_positions(self,dr):
        rcell  = self.atomsx.get_cell()
        rcell += np.dot(rcell, dr[-3:]) / self.jacobian
        self.atomsx.set_cell(rcell, scale_atoms=True)
        ratom  = self.atomsx.get_positions() + dr[:-3]
        self.atomsx.set_positions(ratom)

    def __len__(self):
        return self.natom+3

    def get_forces(self,apply_constraint=True):
        f    = self.atomsx.get_forces(apply_constraint)
        #QZ: modify the force for the rigidbody and frozen set
        for i in self.frozen:
            f[i,:] = 0
    
        stt  = self.atomsx.get_stress()
        vol  = self.atomsx.get_volume()*(-1)
        st   = np.zeros((3,3))
        # following the order of get_stress in vasp.py
        # (the order of stress in ase are the same for all calculators)
        st[0][0] = stt[0] * vol  
        st[1][1] = stt[1] * vol
        st[2][2] = stt[2] * vol
        st[2][1] = stt[3] * vol
        st[2][0] = stt[4] * vol
        st[1][0] = stt[5] * vol
        st  -= self.express * (-1)*vol
        # apply constrain
        st *= self.fixstrain
        #print "original stress (no projecton applied):"
        #print st
        Fc   = np.vstack((f, st/self.jacobian))
        return Fc
    
    def get_potential_energy(self, force_consistent=False):
        return self.atomsx.get_potential_energy()

    def copy(self):
        """Return a copy."""
        import copy
        atomsy = self.atomsx.copy()
        atoms = self.__class__(atomsy, self.express)

        atoms.arrays = {}
        for name, a in self.arrays.items():
            atoms.arrays[name] = a.copy()
        #atoms.constraints = copy.deepcopy(self.constraints)
        #atoms.adsorbate_info = copy.deepcopy(self.adsorbate_info)
        return atoms

