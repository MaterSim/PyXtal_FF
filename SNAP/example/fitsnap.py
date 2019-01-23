# This is a tester

from ase.io import read, write
from ase import Atoms, Atom
from ase.calculators.lammpsrun import LAMMPS

data = read('../../POSCARs/POSCAR-NaCl')

calc = LAMMPS()
data.set_calculator(calc)

print(data.write_lammps_in())
