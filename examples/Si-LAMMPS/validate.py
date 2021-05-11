from random import random
import numpy as np 
from ase.build import bulk
from ase.io import read
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

mliap  = folder + "/16-16-checkpoint.pth"
lmpiap = folder + "/NN_weights.txt"
lmpdes = folder + "/DescriptorParam.txt"


# initial silicon crystal
si = bulk('Si', 'diamond', a=5.0, cubic=True)
si.positions[0,0] += (random() - 0.5)
#si.write('1.vasp', format='vasp', vasp5=True, direct=True)
#si = read('1.vasp', format='vasp')

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
parameters = ["mass 1 1.0",
              "pair_style hybrid/overlay &",
              "mliap model nn " + lmpiap + " descriptor " + des + " " + lmpdes + " &",
              "zbl 2.0 4.0",
              "pair_coeff 1 1 zbl 14.0 14.0",
              "pair_coeff * * mliap Si",
              ]

calc_lmp = LAMMPSlib(lmp=lmp, lmpcmds=parameters)

# check for single configuration
for i in range(100):
    si = bulk('Si', 'diamond', a=5.0, cubic=True)
    si.positions[0,0] += (random() - 0.5)
    eng = []
    force = []
    stress = []
    for calc in [calc_pff, calc_lmp]:
        si.set_calculator(calc)
        eng.append(si.get_potential_energy())
        force.append(si.get_forces())
        stress.append(si.get_stress())
        #print(calc)
        #print("Energy: {:8.3f} eV".format(si.get_potential_energy()))
        #print("Forces (eV/A)")
        #print(si.get_forces())
        #print("Stresses (GPa)")
        #print(si.get_stress())
    e_diff = eng[0]-eng[1]
    f_diff = np.linalg.norm((force[0] - force[1]).flatten())
    s_diff = np.linalg.norm((stress[0] - stress[1]).flatten())
    #print(stress[0])
    #print(stress[1])
    print("{:3d} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f}".format(i, eng[0], eng[1], e_diff, f_diff, s_diff))
    if abs(eng[0]-eng[1]) > 1e-2:
        break

