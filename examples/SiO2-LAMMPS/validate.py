from random import random
from ase.io import read
from lammps import lammps

from pyxtal_ff import PyXtal_FF
from pyxtal_ff.calculator import PyXtalFFCalculator
from pyxtal_ff.calculator.lammpslib import LAMMPSlib

import warnings
warnings.simplefilter("ignore")

des, folder = "sna", "SiO2-snap"
mliap  = folder + "/30-30-checkpoint.pth"
lmpiap = folder + "/NN_weights.txt"
lmpdes = folder + "/DescriptorParam.txt"


# initial silicon crystal
si = read('lt_quartz.cif')
# set the ordering
si.set_tags([2]*3+[1]*6) 
si.positions[0,0] += (random() - 0.5)

# ase pyxtal_ff calculator
ff = PyXtal_FF(model={'system': ["Si", "O"]}, logo=False)
ff.run(mode='predict', mliap=mliap)
calc_pff = PyXtalFFCalculator(ff=ff)

# ase lammps calculatoor
lammps_name=''
comm=None
log_file='lammps.log'
cmd_args = ['-echo', 'log', '-log', log_file,
            '-screen', 'none', '-nocite']
lmp = lammps(lammps_name, cmd_args, comm)

parameters = ["mass * 1.0",
              "pair_style mliap model nn " + lmpiap + " descriptor " + des + " " + lmpdes,
              "pair_coeff * * O Si"
              ]

calc_lmp = LAMMPSlib(lmp=lmp, lmpcmds=parameters)

# check for single configuration
for calc in [calc_pff, calc_lmp]:
    si.set_calculator(calc)
    print(calc)
    print("Energy: {:8.3f} eV".format(si.get_potential_energy()))
    print("Forces (eV/A)")
    print(si.get_forces())
    print("Stresses (GPa)")
    print(si.get_stress())

