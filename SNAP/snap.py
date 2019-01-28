# Running Bispectrum Calculations with LAMMPS

import subprocess

import numpy as np

from pymatgen import Lattice, Structure
from pymatgen.io.lammps.data import LammpsData


class bispectrum(object):
    """

    """
    def __init__(self, structures, rcutfac, twojmax, element_profile, rfac0=0.99363, rmin0=0, diagonal=3):
        self.exe = 'lmp_daily'
        self.pre_cmds = ['units metal',
                         'atom_style charge',
                         'box tilt large',
                         'read_data data.0',
                         'pair_style lj/cut 10',
                         'pair_coeff * * 1 1']
        self.compute_cmds = ['compute sna all sna/atom ',
                             'compute snad all snad/atom ',
                             'compute snav all snav/atom ',
                             'dump 1 all custom 1 dump.element element',
                             'dump 2 all custom 1 dump.sna c_sna[*]',
                             'dump 3 all custom 1 dump.snad c_snad[*]',
                             'dump 4 all custom 1 dump.snav c_snav[*]',
                             'dump_modify 1 element ']
        self.post_cmds = ['run 0']
        self.input_file = 'in.sna'


        self.structures = structures
        self.rcutfac = rcutfac
        self.twojmax = twojmax
        self.elements = element_profile.keys()
        self.ele = []
        self.Rs = []
        self.Ws = []
        for key, value in element_profile.items():
            self.ele.append(key)
            self.Rs.append(element_profile[key]['r'])
            self.Ws.append(element_profile[key]['w'])
        
        self.rfac0 = rfac0
        self.rmin0 = rmin0
        assert diagonal in range(4), 'Invalid diagonal style, must be 0, 1, 2, or 3.'
        self.diagonal = diagonal
        
        self.calculate()

    
    def calculate(self):
        data = self.get_lammps_data(self.structures, self.elements)
        data.write_file('data.0')
        self.get_lammps_input(self.input_file)
        p = subprocess.Popen([self.exe, '-in', self.input_file], stdout=subprocess.PIPE)
        stdout = p.communicate()[0]
        rc = p.returncode
        if rc != 0:
            raise RuntimeError("LAMMPS didn't work properly")
        

    def get_lammps_data(self, structure, elements):
        data = LammpsData.from_structure(structure, elements)
 
        return data

    def get_lammps_input(self, input_file):
        sna = f"{self.rcutfac} {self.rfac0} {self.twojmax} "
        for R in self.Rs:
            sna += f"{R} "
        for W in self.Ws:
            sna += f"{W} "
        sna += f"diagonal {self.diagonal} rmin0 {self.rmin0}"

        self.compute_cmds[0] += sna
        self.compute_cmds[1] += sna
        self.compute_cmds[2] += sna

        for el in self.ele:
            self.compute_cmds[-1] += f"{el} "
        
        self.CMDS = self.pre_cmds + self.compute_cmds + self.post_cmds

        with open(input_file, 'w') as f:
            for line in self.CMDS:
                f.write("%s\n" %line)


s = Structure.from_spacegroup(225, Lattice.cubic(5.69169),
                                      ['Na', 'Cl'],
                                      [[0, 0, 0], [0, 0, 0.5]])
profile = dict(Na=dict(r=0.3, w=0.9), Cl=dict(r=0.7, w=3.0))

L = bispectrum(s, 5.0, 3, profile)
