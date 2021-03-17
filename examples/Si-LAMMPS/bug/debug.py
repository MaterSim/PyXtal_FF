from random import random
from ase.build import bulk
from ase.io import read
from lammps import lammps

from pyxtal_ff import PyXtal_FF
from pyxtal_ff.calculator import PyXtalFFCalculator
from pyxtal_ff.calculator.lammpslib import LAMMPSlib

import warnings
warnings.simplefilter("ignore")

#des, folder = "sna", "Si-snap"
des, folder = "so3", "../Si-so3"
mliap  = folder + "/16-16-checkpoint.pth"


# initial silicon crystal
si = read('1.vasp', format='vasp')

# ase pyxtal_ff calculator
ff = PyXtal_FF(model={'system': ["Si"]}, logo=False)
ff.run(mode='predict', mliap=mliap)
calc_pff = PyXtalFFCalculator(ff=ff)

# check for single configuration
for calc in [calc_pff]:
    si.set_calculator(calc)
    print(calc)
    print("Energy: {:8.3f} eV".format(si.get_potential_energy()))
    print("Forces (eV/A)")
    print(si.get_forces())
    print("Stresses (GPa)")
    print(si.get_stress())

    print(calc_pff.desp['x'])
