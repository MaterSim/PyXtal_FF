from pymatgen import Structure
from pymatgen.io.lammps.data import LammpsData

class zbl:
    """
    This class calculate the Ziegler-Biersack-Littmark (ZBL) 
    screened nuclear repulsion for describing high-energy collision 
    between atoms.


    """
    def __init__(self, structures, inner, outer):
        pre_cmds = ['units metal',
                         'atom_style charge',
                         'box tilt large',
                         'read_data data.0',
                         f'pair_style zbl {inner} {outer}']
        
        post_cmds = ['run 0']

        for s in structures:


z = zbl(1, 2, 3)
