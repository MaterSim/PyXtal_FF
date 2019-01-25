# This script is to convert pymatgen data file to lammps dump files.

import json

from ase.io import read, write
from ase import Atoms, Atom
from ase.calculators.lammpsrun import LAMMPS

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor


with open("training/AIMD.json") as data_file:
    datas = json.load(data_file)


# Convert to Pymatgen obj structure type
    
structures = []
for i in range(len(datas)):
    struc = datas[i]["structure"]
    spes = []
    coord = []
    for site in struc["sites"]:
        spes.append(site["label"])
        coord.append(site["xyz"])
    Struc = Structure(struc["lattice"]["matrix"], spes, coord)
    
    structures.append(Struc)


# Convert to ASE atomic environment

ase_structures = []
for structure in structures:
    struc = AseAtomsAdaptor.get_atoms(structure)
    ase_structures.append(struc)


# Calculating potential energy

calc = LAMMPS()
for i, struc in enumerate(ase_structures):
    struc.set_calculator(calc)
    print(i+1, struc.get_potential_energy())
    calc.write_lammps_data(lammps_data='data.'+str(i))