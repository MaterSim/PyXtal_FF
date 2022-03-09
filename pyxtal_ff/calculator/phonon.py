import numpy as np
from ase import Atoms
from phonopy.api_phonopy import Phonopy

class Phonon(object):
    """
    This module is to compute phonon from the trained Pyxtal_FF model.
    We use phonopy API to construct and handle force constant matrix

    Args:
    atoms: ase atoms object
    calculator: ase calculator object
    supercell_matrix:  3x3 matrix supercell
    primitive cell:  3x3 matrix primitive cell
    displacement_distance: displacement distance in Angstroms
    show_progress: Set true to display progress of calculation
    """

    def __init__(self,
                 atoms,
                 calculator,
                 supercell_matrix=np.identity(3),
                 primitive_matrix='auto',
                 displacement_distance=0.01,
                 show_progress=True,
                 symmetrize=True):

        self._structure = atoms
        self._calculator = calculator
        self._supercell_matrix = supercell_matrix
        self._primitive_matrix = primitive_matrix
        self._d = displacement_distance
        self._show_progress = show_progress
        self._symmetrize = symmetrize

        self._force_constants = None
        self._data_set = None

    def get_force_constants(self, include_data_set=False):
        """
        calculate the force constants with phonopy 

        Return: 
        ForceConstants type object containing force constants
        """

        self.phonon = Phonopy(self._structure,
                              self._supercell_matrix,
                              primitive_matrix = self._primitive_matrix,
                              is_symmetry = self._symmetrize)
        self.phonon.get_displacement_dataset()
        self.phonon.generate_displacements(distance=self._d) 
        cells_with_disp = self.phonon.get_supercells_with_displacements()
        data_set = self.phonon.get_displacement_dataset()
        # Compute forces for the displaced structures
        for i, cell in enumerate(cells_with_disp):
            if self._show_progress:
                print('displacement {} / {}'.format(i+1, len(cells_with_disp)))
            print(cell)
            forces = self.get_forces(cell)
            data_set['first_atoms'][i]['forces'] = forces

        self.phonon.set_displacement_dataset(data_set)
        self.phonon.produce_force_constants()
        self._force_constants = self.phonon.get_force_constants()
        self._data_set = data_set

        return self._force_constants

    def get_displaced_atoms(self):
        """
        Get the dispaced structures for further process

        Return: 
        A list of ASE atoms object
        """

        self.phonon = Phonopy(self._structure,
                              self._supercell_matrix,
                              primitive_matrix = self._primitive_matrix,
                              is_symmetry = self._symmetrize)
        self.phonon.get_displacement_dataset()
        self.phonon.generate_displacements(distance=self._d) 
        cells_with_disp = self.phonon.get_supercells_with_displacements()

        supercells = []
        numbers = self._structure.numbers
        for cell_with_disp in cells_with_disp:
            s_numbers = []
            mult = int(round(np.linalg.det(self._supercell_matrix)))
            atom_id = 0
            disps = cell_with_disp.get_positions()
            for id, disp in enumerate(disps):
                s_numbers.append(numbers[atom_id])
                if id%mult == (mult-1):
                    atom_id += 1
            
            cell = self._structure.cell.dot(self._supercell_matrix)
            supercell = Atoms(s_numbers, positions=disps, cell=cell, pbc=[1,1,1])
            #supercell.write('tmp.vasp', format='vasp', vasp5=True, direct=True)
            supercells.append(supercell)
        return supercells

    def get_forces(self, cell_with_disp):
        """
        Calculate the forces of a supercell using lammps
        Args:
        cell_with_disp: supercell from which determine the forces

        Return:
        forces: [Natoms_of_supercell, 3] array
        """

        numbers = self._structure.numbers
        s_numbers = []
        mult = int(round(np.linalg.det(self._supercell_matrix)))
        atom_id = 0
        disps = cell_with_disp.get_positions()
        for id, disp in enumerate(disps):
            s_numbers.append(numbers[atom_id])
            if id%mult == (mult-1):
                atom_id += 1
        
        cell = self._structure.cell.dot(self._supercell_matrix)
        supercell = Atoms(s_numbers, positions=disps, cell=cell, pbc=[1,1,1])
        supercell.set_calculator(self._calculator)
        return supercell.get_forces()

if  __name__ == "__main__":

    from optparse import OptionParser
    from ase.build import bulk
    from pyxtal_ff.calculator import PyXtalFFCalculator
    from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

    parser = OptionParser()
    parser.add_option("-f", "--file", dest="file",
                      help="pretrained file from pyxtal_ff, REQUIRED",
                      metavar="file")

    (options, args) = parser.parse_args()
    calc = PyXtalFFCalculator(mliap=options.file, logo=False)
    si = bulk('Si', 'diamond', a=5.459, cubic=True)
    supercell_matrix=np.diag([2,2,2])
    ph = Phonon(si, calc, supercell_matrix=supercell_matrix) 
    force_constants = ph.get_force_constants()
    ph.phonon.set_force_constants(force_constants)

    #ph.phonon.auto_band_structure()
    path = [[[0, 0, 0], [0.5, 0, 0.5], [0.375, 0.375, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]]]
    labels = ["$\\Gamma$", "X", "K", "$\\Gamma$", "L"]
    qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
    ph.phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)

    ph.phonon.set_mesh([10,10,10])
    ph.phonon.set_total_DOS()
    ph.phonon.plot_band_structure_and_dos().savefig('Si_phonon.png', dpi=300)
    print('The detected space group is {:s} with a tolerance of {:.2e}'.format(
          ph.phonon._symmetry.dataset['international'], ph.phonon._symprec))
