import itertools

from element import Element
from pymatgen import Structure
from pymatgen.io.lammps.data import LammpsData

class zbl:
    """
    This class calculate the Ziegler-Biersack-Littmark (ZBL) 
    screened nuclear repulsion for describing high-energy collision 
    between atoms.


    """
    def __init__(self, inner, outer):
        self.pre_cmds = ['units metal',
                         'atom_style charge',
                         'box tilt large',
                         'read_data data.0',
                         f'pair_style zbl {inner} {outer}']
        
        self.post_cmds = ['run 0']

        self.input_file = 'in.zbl'

    def calculate(self, structures):

        for structure in structures:
            atom_types = structure.symbol_set
            no_atom_types = len(atom_types)
            
            n_atom_types = []
            for i in range(no_atom_types):
                n_atom_types.append(i+1)
            n_pair_types = list(itertools.combinations_with_replacement(n_atom_types,2))
            
            cmds = []
            for pair in n_pair_types:
                Z1 = Element(atom_types[pair[0]-1]).z
                Z2 = Element(atom_types[pair[1]-1]).z
                cmds.append(f"pair_coeff {pair[0]} {pair[1]} {Z1} {Z2}")
            
            # cmds.append(dump potential energy)

            self.CMDS = self.pre_cmds + cmds + self.post_cmds

            with open(self.input_file, 'w') as f:
                for line in self.CMDS:
                    f.write("%s \n" %line)

            
import json
from pymatgen import Structure
with open("C4_train.json") as f:
    datas = json.load(f)

structures = []
for data in datas[:1]:
    lattice = data['lattice']['data']
    species = data['elements']
    coords = data['coords']['data']
    s = Structure(lattice, species, coords)
    structures.append(s)

z = zbl(2, 3)
z.calculate(structures)
