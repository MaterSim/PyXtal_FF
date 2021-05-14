"""
A script to check the stress calculation
"""

from random import random
import numpy as np 
from ase.build import bulk
from ase.db import connect
from ase import units

from pyxtal_ff import PyXtal_FF
from pyxtal_ff.calculator import PyXtalFFCalculator

import warnings
warnings.simplefilter("ignore")

des, folder = "sna", "Si-snap-zbl"
mliap  = folder + "/12-12-checkpoint.pth"
lmpiap = folder + "/NN_weights.txt"
lmpdes = folder + "/DescriptorParam.txt"

# ase pyxtal_ff calculator
ff = PyXtal_FF(model={'system': ["Si"]}, logo=False)
ff.run(mode='predict', mliap=mliap)
calc = PyXtalFFCalculator(ff=ff, style='lammps') #GPa, xx, yy, zz, xy, xz, yz

with connect(folder + '/ase.db') as db:
    print(len(db))
    maes = []
    for i, row in enumerate(db.select()):
        if "stress" in row.data and row.data["group"]=="Elastic":
            s = db.get_atoms(row.id)
            s.set_calculator(calc)
            e = s.get_potential_energy()
            s1 = row.data['stress'] # GPa 
            s2 = s.get_stress()
            mae = np.mean(np.abs(s1-s2))
            print("\nDFT Energy: {:6.3f} v.s. PFF energy {:6.3f} S_MAE: {:6.3f}".format(row.data['energy'], e, mae))
            print("DFT (GPa): {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f}".format(*s1))
            print("PFF (GPa): {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f}".format(*s2))
            calc.print_stresses()
            maes.append(mae)
    print("Final MAE (GPa)", np.mean(np.array(maes)))
