from random import random
import numpy as np 
from ase.build import bulk
from ase import units
from lammps import lammps

from pyxtal_ff import PyXtal_FF
from pyxtal_ff.calculator import PyXtalFFCalculator
from pyxtal_ff.calculator.lammpslib import LAMMPSlib

import warnings
warnings.simplefilter("ignore")

if True:
    des, folder = "sna", "Si-snap-zbl"
else:
    des, folder = "so3", "Si-so3"

mliap  = folder + "/12-12-checkpoint.pth"
lmpiap = folder + "/NN_weights.txt"
lmpdes = folder + "/DescriptorParam.txt"


# initial silicon crystal
si = bulk('Si', 'diamond', a=5.0, cubic=True)
si.positions[0,0] += (random() - 0.5)

# ase pyxtal_ff calculator
ff = PyXtal_FF(model={'system': ["Si"]}, logo=False)
ff.run(mode='predict', mliap=mliap)
calc_pff = PyXtalFFCalculator(ff=ff)

# ase lammps calculatoor
lammps_name=''
comm=None
log_file='lammps.log'
cmd_args = ['-echo', 'log', '-log', log_file,
            '-screen', 'none', '-nocite']
lmp = lammps(lammps_name, cmd_args, comm)


# the pair style command to appear in lammps
parameters = ["mass 1 28.0855",
              "pair_style hybrid/overlay &",
              "mliap model nn " + lmpiap + " descriptor " + des + " " + lmpdes + " &",
              "zbl 1.5 2.0",
              "pair_coeff 1 1 zbl 14.0 14.0",
              "pair_coeff * * mliap Si",
              ]

calc_lmp = LAMMPSlib(lmp=lmp, lmpcmds=parameters)

# check for single configuration
for i in range(10):
    si = bulk('Si', 'diamond', a=5.2, cubic=True)
    si.positions[0,0] += (random() - 0.5)
    eng = []
    force = []
    stress = []
    for j, calc in enumerate([calc_pff, calc_lmp]):
        si.set_calculator(calc)
        eng.append(si.get_potential_energy())
        force.append(si.get_forces())
        stress.append(si.get_stress())

    e_diff = eng[0]-eng[1]
    f_diff = np.linalg.norm((force[0] - force[1]).flatten())
    s_diff = np.linalg.norm((stress[0] - stress[1]).flatten())

    print("{:3d} {:8.3f} eV {:8.3f} GPa {:8.3f} {:8.3f} {:8.3f}".format(i, eng[0], -stress[0][0]/units.GPa, e_diff, f_diff, s_diff))
    if abs(e_diff) > 1e-2:
        break

