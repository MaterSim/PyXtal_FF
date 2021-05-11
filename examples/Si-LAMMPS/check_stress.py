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
            dftstress = row.data['stress'] # in GPa -> [xx,yy,zz,xy,xz,yz]
            pffstress = (-1*s.get_stress()/units.GPa)[[0,1,2,5,4,3]] # GPa to eV/A^3 -> [xx,yy,zz,yz,xz,xy] to [xx,yy,zz,xy,xz,yz]
            print("\nDFT Energy: {:6.3f} v.s. PFF energy {:6.3f}".format(row.data['energy'], e))
            print("DFT: {:6.3f} {:6.4f} {:6.4f} {:6.4f} {:6.4f} {:6.4f}".format(*dftstress))
            print("PFF: {:6.3f} {:6.4f} {:6.4f} {:6.4f} {:6.4f} {:6.4f}".format(*pffstress))
            #calc.print_stresses()
