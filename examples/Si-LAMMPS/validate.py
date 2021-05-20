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
    mliap  = folder + "/12-12-checkpoint.pth"
    lmpiap = folder + "/NN_weights.txt"
    lmpdes = folder + "/DescriptorParam.txt"
    # the pair style command to appear in lammps
    parameters = ["mass 1 28.0855",
              "pair_style hybrid/overlay &",
              "mliap model nn " + lmpiap + " descriptor " + des + " " + lmpdes + " &",
              "zbl 1.0 2.0",
              "pair_coeff 1 1 zbl 14.0 14.0",
              "pair_coeff * * mliap Si",
              ]
else:
    des, folder = "so3", "Si-so3"
    mliap  = folder + "/12-12-checkpoint.pth"
    lmpiap = folder + "/NN_weights.txt"
    lmpdes = folder + "/DescriptorParam.txt"
    parameters = ["mass 1 28.0855",
              "pair_style mliap model nn " + lmpiap + " descriptor " + des + " " + lmpdes,
              "pair_coeff * * Si",
              ]

print("Testing", des, folder)

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



calc_lmp = LAMMPSlib(lmp=lmp, lmpcmds=parameters)

np.random.seed(0)
# check for single configuration
for i in range(10):
    a = 4.5 + np.random.random()
    si = bulk('Si', 'diamond', a=a, cubic=True)*2
    si.positions += 0.025*(np.random.random_sample([len(si),3])-0.5)
    eng = []
    force = []
    stress = []
    for j, calc in enumerate([calc_lmp, calc_pff]):
        si.set_calculator(calc)
        eng.append(si.get_potential_energy())
        force.append(si.get_forces())
        stress.append(si.get_stress())

    e_diff = eng[0]-eng[1]
    f_diff = np.linalg.norm((force[0] - force[1]).flatten())
    s_diff = np.linalg.norm((stress[0] - stress[1]).flatten())/units.GPa
    #calc.print_energy()
    print("{:3d} {:8.3f} eV {:8.3f} GPa {:8.3f} {:8.3f} {:8.3f}".format(i, eng[0], -stress[0][0]/units.GPa, e_diff, f_diff, s_diff))
    if abs(e_diff) > 1e-2 or f_diff > 1e-2 or s_diff > 1e-2:
        print("eng: ", eng[0], eng[1])
        print("Forces from LAMMPS and PyXtal_FF")
        for f1, f2 in zip(force[0], force[1]):
            print("{:8.3f} {:8.3f} {:8.3f} -> {:8.3f} {:8.3f} {:8.3f} -> {:8.3f} {:8.3f} {:8.3f}".format(*f1, *f2, *(f1-f2)))
        print("\n Breakdown of Pyxtal_FF")
        calc.print_all()

        print("\n Breakdown of LAMMPS")
        parameters = ["mass 1 28.0855",
              "pair_style mliap model nn " + lmpiap + " descriptor " + des + " " + lmpdes,
              "pair_coeff * * Si Si",
              ]

        calc_lmp = LAMMPSlib(lmp=lmp, lmpcmds=parameters)
        si.set_calculator(calc_lmp)
        eng1 = si.get_potential_energy()
        f1s = si.get_forces()
        s1 = -si.get_stress()/units.GPa
        
        if des == 'sna':
            parameters = ["mass 1 28.0855",
                  "pair_style zbl 1.0 2.0",
                  "pair_coeff 1 1 14.0 14.0",
                  ]

            calc_lmp = LAMMPSlib(lmp=lmp, lmpcmds=parameters)
            si.set_calculator(calc_lmp)
            eng2 = si.get_potential_energy()
            f2s = si.get_forces()
            s2 = -si.get_stress()/units.GPa
        else:
            eng2 = 0
            f2s = np.zeros([len(si), 3])
            s2 = np.zeros([6])
        print("energy_ml (eV):", eng1)
        print("energy_zbl(eV):", eng2)
        print("Forces")
        for f1, f2 in zip(f1s, f2s):
            print("{:8.3f} {:8.3f} {:8.3f} -> {:8.3f} {:8.3f} {:8.3f}".format(*f1, *f2))
        print("stress_ml (GPa, xx, yy, zz, xy, xz, yz):", s1[[0,1,2,5,4,3]])
        print("stress_zbl(GPa, xx, yy, zz, xy, xz, yz):", s2[[0,1,2,5,4,3]])
        break
