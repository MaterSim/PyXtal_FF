import os
import subprocess
import itertools
import numpy as np

from element import Element
from pymatgen import Structure
from pymatgen.io.lammps.data import LammpsData

class zbl:
    """
    This class calculate the Ziegler-Biersack-Littmark (ZBL) 
    screened nuclear repulsion for describing high-energy collision 
    between atoms.


    """
    def __init__(self, structure, inner, outer, force=True):
        self.exe = 'lmp_serial'
        self.pre_cmds = ['units metal',
                         'atom_style charge',
                         'box tilt large',
                         'read_data data.0',
                         f'pair_style zbl {inner} {outer}']
        
        self.post_cmds = ['run 0']

        self.input_file = 'in.zbl'

        self.structure = structure
        self.inner = inner
        self.outer = outer

        # Calculate the ZBL energy and/or forces
        self.calculate()
        
        # Get energy and forces
        self.energy = np.asarray([self.get_energy()])

        self.force = self.get_force()
        print(self.force)

        #os.remove("data.0")
        #os.remove("in.zbl")
        #os.remove("log.lammps")


    def calculate(self):
        """ Run LAMMPS to calculate the ZBL energy and force."""
        self.no_of_atoms = self.structure.num_sites
        self.atom_types = self.structure.symbol_set
        self.no_atom_types = len(self.atom_types)

        # Convert to dump file
        data = self.get_dump(self.structure, self.atom_types)
        data.write_file('data.0')

        # Initialize LAMMPS input
        self.get_lammps_input(self.input_file)

        # Run LAMMPS
        p = subprocess.Popen([self.exe, '-in', self.input_file],
                             stdout=subprocess.PIPE)
        stdout = p.communicate()[0]
        rc = p.returncode
        if rc != 0:
            error_msg = 'LAMMPS exited with return code %d' % rc
            msg = stdout.decode("utf-8").split('\n')[:-1]
            try:
                error = [i for i, m in enumerate(msg)
                        if m.startswith('ERROR')][0]
                error_msg += ', '.join([e for e in msg[error:]])
            except:
                error_msg += msg[-1]
            raise RuntimeError(error_msg)

    
    def get_lammps_input(self, input_file):
        """Create LAMMPS input file"""
        
        n_atom_types = []
        for i in range(self.no_atom_types):
            n_atom_types.append(i+1)
        n_pair_types = list(itertools.combinations_with_replacement(n_atom_types,2))
        
        cmds = []
        for pair in n_pair_types:
            Z1 = Element(self.atom_types[pair[0]-1]).z
            Z2 = Element(self.atom_types[pair[1]-1]).z
            cmds.append(f"pair_coeff {pair[0]} {pair[1]} {Z1} {Z2}")
        
        cmds.append("compute force all property/atom type fx fy fz")
        cmds.append("dump 1 all custom 1 dump.force c_force[1] c_force[2] c_force[3] c_force[4]")

        self.CMDS = self.pre_cmds + cmds + self.post_cmds

        with open(self.input_file, 'w') as f:
            for line in self.CMDS:
                f.write("%s \n" %line)


    def get_dump(self, structure, elements):
        """Convert Pymatgen structure object to LAMMPS dump file."""
        data = LammpsData.from_structure(structure, elements)
 
        return data


    def get_energy(self):
        """Obtain the ZBL energy."""
        lines = []
        temp = None
        quit = False
        
        with open("log.lammps", 'r') as f:
            for i, line in enumerate(f):
                if quit == False:
                    lines.append(line)
                    if temp == "Step Temp E_pair E_mol TotEng Press \n":
                        quit = True
                else:
                    break
                temp = line
        
        str_array = lines[-1].split()
        energy_per_atom = float(str_array[2])/self.no_of_atoms

        return energy_per_atom


    def get_force(self):
        """Obtain the ZBL force."""
        lines = []

        with open("dump.force", 'r') as f:
            for line in f:
                lines.append(line.split())

        lines = np.array(lines[-self.no_of_atoms:]).astype(float)
        
        atom_types = []
        fx, fy, fz = [], [], []
        for line in lines:
            if int(line[0]) <= self.no_atom_types:
                atom_types.append(int(line[0]))
                fx.append(line[1])
                fy.append(line[2])
                fz.append(line[3])
            else:
                raise ValueError(f"The no. of atom types should be no more than {self.atom_types}")
        
        print(max(atom_types))
        
        force = np.concatenate((fx, fy, fz))

        return force


            
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

z = zbl(structures[0], 2, 3)
