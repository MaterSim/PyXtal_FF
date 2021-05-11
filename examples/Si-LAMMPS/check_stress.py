from random import random
import numpy as np 
from ase.build import bulk
from ase.db import connect
from ase import units

from pyxtal_ff import PyXtal_FF
from pyxtal_ff.calculator import PyXtalFFCalculator

#import warnings
#warnings.simplefilter("ignore")

des, folder = "sna", "Si-snap-zbl"
mliap  = folder + "/16-16-checkpoint.pth"
lmpiap = folder + "/NN_weights.txt"
lmpdes = folder + "/DescriptorParam.txt"

# ase pyxtal_ff calculator
ff = PyXtal_FF(model={'system': ["Si"]}, logo=False)
ff.run(mode='predict', mliap=mliap)
calc = PyXtalFFCalculator(ff=ff)

with connect(folder + '/ase.db') as db:
    print(len(db))
    for i, row in enumerate(db.select()):
        if "stress" in row.data:
            s = db.get_atoms(row.id)
            s.set_calculator(calc)
            e = s.get_potential_energy()
            stress = [x*units.GPa for x in row.data['stress']] # GPa to eV/A^3
            print("\nDFT Energy: {:6.3f} v.s. PFF energy {:6.3f}".format(row.data['energy'], e))
            print("DFT: {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f}".format(*stress))
            print("PFF: {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f}".format(*s.get_stress()))
            calc.print_stresses()
